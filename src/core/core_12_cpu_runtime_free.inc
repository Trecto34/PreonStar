static void p36_cpu_runtime_free(p36_cpu_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        free(rt->full[il].k);
        free(rt->full[il].v);
        free(rt->recurrent[il].conv);
        free(rt->recurrent[il].state);
    }
    free(rt->hidden);
    free(rt->next_hidden);
    free(rt->work0);
    free(rt->work1);
    free(rt->work2);
    free(rt->work3);
    free(rt->work4);
    free(rt->work5);
    free(rt->scores);
    free(rt->batch_hidden);
    free(rt->batch_next_hidden);
    free(rt->batch_norm);
    free(rt->batch_qg);
    free(rt->batch_q);
    free(rt->batch_k);
    free(rt->batch_v);
    free(rt->batch_attn_out);
    free(rt->batch_recur_qkv);
    free(rt->batch_recur_z);
    free(rt->batch_recur_alpha);
    free(rt->batch_recur_beta);
    free(rt->batch_recur_proj);
    free(rt->batch_ffn_gate_logits);
    free(rt->batch_ffn_shared_gate);
    free(rt->batch_ffn_shared_up);
    free(rt->batch_ffn_shared_mid);
    free(rt->batch_ffn_shared_out);
    free(rt->batch_ffn_scalar);
    free(rt->batch_xq);
    free(rt->batch_had);
    free(rt);
}

#ifndef P36_NO_GPU
static bool p36_gpu_tensor_zero(p36_gpu_tensor *tensor) {
    void *p;
    if (!tensor) return false;
    p = p36_gpu_tensor_contents_named(tensor, "submit_wait_tensor_zero");
    if (!p) return false;
    memset(p, 0, (size_t)p36_gpu_tensor_bytes(tensor));
    return true;
}

static bool p36_gpu_tensor_all_finite(const p36_gpu_tensor *tensor, uint32_t n) {
    float *p = p36_gpu_tensor_contents_named((p36_gpu_tensor *)tensor, "submit_wait_all_finite");
    return p && p36_all_finite(p, n);
}

static bool p36_gpu_tensor_scale_host(p36_gpu_tensor *tensor, uint32_t n, float scale) {
    float *p = p36_gpu_tensor_contents_named(tensor, "submit_wait_scale_host");
    if (!p) return false;
    p36_scale_inplace(p, n, scale);
    return true;
}

static bool p36_gpu_embed_tokens(const p36_model *m, const p36_tensor *t,
                                 const int *tokens, uint32_t n_tok, p36_gpu_tensor *dst) {
    /* Decode takes the row straight off the device copy of the table: no
     * host dequant, and no host write into a staging buffer the GPU is
     * about to read.  Prefill batches and the hadamard-inverse variant
     * keep the host path. */
    if (!t->hadamard_inverse &&
        (t->type == P36_TENSOR_Q8_0 || t->type == P36_TENSOR_Q4_K)) {
        p36_gpu_embed_table_prepare(m->map, m->size, t->abs_offset, t->bytes);
    }
    if (n_tok == 1 && !t->hadamard_inverse &&
        tokens[0] >= 0 && tokens[0] < (int)P36_N_VOCAB &&
        p36_gpu_embed_row_tensor(dst, m->map, m->size, t->abs_offset, t->bytes,
                                 P36_N_EMBD, (uint32_t)t->type, NULL,
                                 (uint32_t)tokens[0])) {
        return true;
    }
    float *p = p36_gpu_tensor_contents_named(dst, "submit_wait_embed_tokens");
    if (!p) return false;
    for (uint32_t i = 0; i < n_tok; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)P36_N_VOCAB) return false;
        float *row = p + (uint64_t)i * P36_N_EMBD;
        if (!p36_tensor_row_to_float(m, t, (uint64_t)tokens[i], row, P36_N_EMBD)) {
            return false;
        }
        if (t->hadamard_inverse) p36_hadamard_inverse_host(row, P36_N_EMBD);
    }
    return true;
}

/* Build the n_tok conv windows in token order while sliding the persistent
 * conv history, exactly n_tok runs of p36_recurrent_conv_step(). */
static P36_MAYBE_UNUSED bool p36_gpu_recurrent_conv_step(p36_gpu_tensor *cache_conv,
                                        const p36_gpu_tensor *cur,
                                        p36_gpu_tensor *window,
                                        uint32_t n_tok) {
    const uint32_t hist = P36_N_SSM_CONV - 1u;
    const size_t row_bytes = (size_t)P36_N_SSM_CONV_DIM * sizeof(float);
    float *cachep = p36_gpu_tensor_contents_named(cache_conv, "submit_wait_conv_cache");
    float *windowp = p36_gpu_tensor_contents_named(window, "submit_wait_conv_window");
    float *curp = p36_gpu_tensor_contents_named((p36_gpu_tensor *)cur, "submit_wait_conv_cur");
    if (!cachep || !windowp || !curp) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        float *wt = windowp + (uint64_t)t * P36_N_SSM_CONV * P36_N_SSM_CONV_DIM;
        const float *ct = curp + (uint64_t)t * P36_N_SSM_CONV_DIM;
        for (uint32_t i = 0; i < hist; i++) {
            memcpy(wt + (uint64_t)i * P36_N_SSM_CONV_DIM,
                   cachep + (uint64_t)i * P36_N_SSM_CONV_DIM,
                   row_bytes);
        }
        memcpy(wt + (uint64_t)hist * P36_N_SSM_CONV_DIM, ct, row_bytes);
        if (hist > 1u) {
            memmove(cachep,
                    cachep + P36_N_SSM_CONV_DIM,
                    (size_t)(hist - 1u) * row_bytes);
        }
        memcpy(cachep + (uint64_t)(hist - 1u) * P36_N_SSM_CONV_DIM, ct, row_bytes);
    }
    return true;
}

static P36_MAYBE_UNUSED bool p36_gpu_extract_full_attn_q(p36_gpu_tensor *dst, const p36_gpu_tensor *qg, uint32_t n_tok) {
    const float *src = p36_gpu_tensor_contents_named((p36_gpu_tensor *)qg, "submit_wait_extract_q_src");
    float *out = p36_gpu_tensor_contents_named(dst, "submit_wait_extract_q_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < P36_N_HEAD; h++) {
            memcpy(out + ((uint64_t)t * P36_N_HEAD + h) * P36_N_HEAD_DIM,
                   src + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u,
                   (size_t)P36_N_HEAD_DIM * sizeof(float));
        }
    }
    return true;
}

static P36_MAYBE_UNUSED bool p36_gpu_extract_recurrent_v(p36_gpu_tensor *dst, const p36_gpu_tensor *conv, uint32_t n_tok) {
    const float *src = p36_gpu_tensor_contents_named((p36_gpu_tensor *)conv, "submit_wait_extract_v_src");
    float *out = p36_gpu_tensor_contents_named(dst, "submit_wait_extract_v_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        memcpy(out + (uint64_t)t * P36_N_SSM_INNER,
               src + (uint64_t)t * P36_N_SSM_CONV_DIM + (uint64_t)P36_N_SSM_QK * 2u,
               (size_t)P36_N_SSM_INNER * sizeof(float));
    }
    return true;
}

static bool p36_gpu_tensor_matmul_scaled(const p36_model *m,
                                         const p36_tensor *t,
                                         const p36_gpu_tensor *x,
                                         p36_gpu_tensor *out,
                                         uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t n_tok,
                                         float scale) {
    bool ok = false;
    if (!m || !t || !x || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case P36_TENSOR_F32:
        ok = p36_gpu_matmul_f32_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                              in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_F16:
        ok = p36_gpu_matmul_f16_tensor(out, m->map, m->size, t->abs_offset,
                                       in_dim, out_dim, x, n_tok) != 0;
        if (ok && scale != 1.0f) ok = p36_gpu_tensor_scale_host(out, n_tok * out_dim, scale);
        break;
    case P36_TENSOR_BF16:
        ok = p36_gpu_matmul_bf16_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                               in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_Q8_0:
        ok = p36_gpu_matmul_q8_0_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                               in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
        ok = p36_gpu_matmul_k_quant_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                  t->type, in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        ok = p36_gpu_matmul_iq_quant_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                   t->type, in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    default:
        return false;
    }
    return ok;
}

static bool p36_gpu_tensor_matmul_q8_scaled(const p36_model *m,
                                            const p36_tensor *t,
                                            const p36_gpu_tensor *xq,
                                            p36_gpu_tensor *out,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            uint32_t n_tok,
                                            float scale) {
    if (!m || !t || !xq || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
        return p36_gpu_matmul_k_quant_q8_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                        t->type, in_dim, out_dim, xq, n_tok, scale) != 0;
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return p36_gpu_matmul_iq_quant_q8_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                        t->type, in_dim, out_dim, xq, n_tok, scale) != 0;
    default:
        return false;
    }
}

static bool p36_gpu_tensor_matmul_q8_or_float_scaled(const p36_model *m,
                                                     const p36_tensor *t,
                                                     const p36_gpu_tensor *x,
                                                     const p36_gpu_tensor *xq,
                                                     p36_gpu_tensor *out,
                                                     uint32_t in_dim,
                                                     uint32_t out_dim,
                                                     uint32_t n_tok,
                                                     float scale) {
    if (!t) return false;
#if defined(P36_METAL)
    /* Dense Metal has native float-RHS IQ3/K-quant kernels.  Keep the original
     * activation instead of forcing a host-synchronized q8_K staging pass.
     * Vulkan retains its tuned q8_K dense path unchanged. */
    if (P36_MODEL_DENSE) {
        return p36_gpu_tensor_matmul_scaled(m, t, x, out,
                                             in_dim, out_dim, n_tok, scale);
    }
#endif
    if (t->type == P36_TENSOR_F32 || t->type == P36_TENSOR_F16 ||
        t->type == P36_TENSOR_BF16 || t->type == P36_TENSOR_Q8_0) {
        return p36_gpu_tensor_matmul_scaled(m, t, x, out, in_dim, out_dim, n_tok, scale);
    }
    return p36_gpu_tensor_matmul_q8_scaled(m, t, xq, out, in_dim, out_dim, n_tok, scale);
}

static bool p36_gpu_tensor_matmul_dense_q8_scaled(const p36_model *m,
                                                   const p36_tensor *t,
                                                   const p36_gpu_tensor *x,
                                                   const p36_gpu_tensor *xq,
                                                   p36_gpu_tensor *out,
                                                   uint32_t in_dim,
                                                   uint32_t out_dim,
                                                   uint32_t n_tok,
                                                   float scale) {
    if (!P36_MODEL_DENSE)
        return p36_gpu_tensor_matmul_scaled(m, t, x, out,
                                            in_dim, out_dim, n_tok, scale);
    return p36_gpu_tensor_matmul_q8_or_float_scaled(m, t, x, xq, out,
                                                     in_dim, out_dim, n_tok, scale);
}

static bool p36_vulkan_runtime_reset(p36_vulkan_runtime *rt) {
    uint64_t state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    uint64_t state_bytes;
    if (!rt) return false;
    rt->inp_q8_valid = false;
    state_bytes = state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float));
    if (!p36_gpu_tensor_zero(rt->hidden) || !p36_gpu_tensor_zero(rt->next_hidden)) return false;
    if (rt->last_h && !p36_gpu_tensor_zero(rt->last_h)) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (rt->recurrent[il].conv && !p36_gpu_tensor_zero(rt->recurrent[il].conv)) return false;
        if (rt->recurrent[il].state && !p36_gpu_tensor_zero(rt->recurrent[il].state)) return false;
        if (rt->recurrent[il].state && p36_gpu_tensor_bytes(rt->recurrent[il].state) != state_bytes) return false;
    }
    return true;
}

static void p36_vulkan_runtime_free(p36_vulkan_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_gpu_tensor_free(rt->full[il].k);
        p36_gpu_tensor_free(rt->full[il].v);
        p36_gpu_tensor_free(rt->recurrent[il].conv);
        p36_gpu_tensor_free(rt->recurrent[il].state);
        p36_gpu_tensor_free(rt->spec_recurrent[il].conv);
        p36_gpu_tensor_free(rt->spec_recurrent[il].state);
        p36_gpu_tensor_free(rt->spec_recur_qkv[il]);
        p36_gpu_tensor_free(rt->spec_recur_gb[il]);
    }
    p36_gpu_tensor_free(rt->hidden);
    p36_gpu_tensor_free(rt->next_hidden);
    p36_gpu_tensor_free(rt->embed_stage[0]);
    p36_gpu_tensor_free(rt->embed_stage[1]);
    p36_gpu_tensor_free(rt->norm);
    p36_gpu_tensor_free(rt->last_h);
    p36_gpu_tensor_free(rt->last_h_raw);
    p36_gpu_tensor_free(rt->inp_q8);
    p36_gpu_tensor_free(rt->had_q8);
    p36_gpu_tensor_free(rt->had_signs);
    p36_gpu_tensor_free(rt->attn_qg);
    p36_gpu_tensor_free(rt->attn_q);
    p36_gpu_tensor_free(rt->attn_k);
    p36_gpu_tensor_free(rt->attn_v);
    p36_gpu_tensor_free(rt->attn_out);
    p36_gpu_tensor_free(rt->recur_qkv);
    p36_gpu_tensor_free(rt->recur_window);
    p36_gpu_tensor_free(rt->recur_conv);
    p36_gpu_tensor_free(rt->recur_z);
    p36_gpu_tensor_free(rt->recur_alpha);
    p36_gpu_tensor_free(rt->recur_beta);
    p36_gpu_tensor_free(rt->recur_gb);
    p36_gpu_tensor_free(rt->recur_q);
    p36_gpu_tensor_free(rt->recur_k);
    p36_gpu_tensor_free(rt->recur_v);
    p36_gpu_tensor_free(rt->recur_proj);
    p36_gpu_tensor_free(rt->ffn_gate_logits);
    p36_gpu_tensor_free(rt->ffn_selected);
    p36_gpu_tensor_free(rt->ffn_weights);
    p36_gpu_tensor_free(rt->ffn_shared_gate);
    p36_gpu_tensor_free(rt->ffn_shared_up);
    p36_gpu_tensor_free(rt->ffn_shared_mid);
    p36_gpu_tensor_free(rt->ffn_shared_out);
    p36_gpu_tensor_free(rt->ffn_scalar);
    p36_gpu_tensor_free(rt->logits);
    p36_gpu_tensor_free(rt->top2[0]);
    p36_gpu_tensor_free(rt->top2[1]);
    p36_gpu_tensor_free(rt->topk8);
    p36_gpu_tensor_free(rt->minp_pack);
    p36_gpu_tensor_free(rt->scores);
    p36_gpu_tensor_free(rt->mtp_full.k);
    p36_gpu_tensor_free(rt->mtp_full.v);
    p36_gpu_tensor_free(rt->spec_logits);
    p36_gpu_tensor_free(rt->spec_tops);
    p36_gpu_tensor_free(rt->mtp_tok_embd);
    p36_gpu_tensor_free(rt->mtp_e_norm);
    p36_gpu_tensor_free(rt->mtp_h_norm);
    p36_gpu_tensor_free(rt->mtp_concat);
    p36_gpu_tensor_free(rt->mtp_cur);
    p36_gpu_tensor_free(rt->mtp_next);
    p36_gpu_tensor_free(rt->mtp_head);
    p36_gpu_tensor_free(rt->mtp_logits);
    free(rt);
}

static p36_vulkan_runtime *p36_vulkan_runtime_create(int ctx_size,
                                                     uint32_t prefill_cap,
                                                     bool enable_mtp,
                                                     bool enable_spec,
                                                     bool quality,
                                                     bool ssd_streaming,
                                                     p36_kv_cache_type cache_type_k,
                                                     p36_kv_cache_type cache_type_v) {
    p36_vulkan_runtime *rt;
    uint64_t state_dim;
    uint64_t hist_rows;
    uint32_t kv_cap;
    if (ctx_size <= 0 || !p36_gpu_init()) return NULL;
    uint32_t k_row_bytes = p36_kv_cache_row_bytes(cache_type_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    uint32_t v_row_bytes = p36_kv_cache_row_bytes(cache_type_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    if (!k_row_bytes || !v_row_bytes) return NULL;
    rt = xcalloc(1, sizeof(*rt));
    kv_cap = p36_kv_initial_cap(ctx_size);
    if (prefill_cap < 1) prefill_cap = 1;
    rt->prefill_cap = prefill_cap;
    rt->mtp_enabled = enable_mtp;
    /* Verify scratch is shared: prompt lookup speculates without an MTP head. */
    const bool spec_scratch = enable_mtp || enable_spec;
    {
        const char *env = getenv("P36_VK_RECURRENT_CONV_DECODE");
        rt->recur_conv_fused = !quality && !ssd_streaming &&
                               (!env || !env[0] || env[0] != '0');
    }
#ifndef P36_METAL
    {
        const char *env = getenv("P36_VK_RECURRENT_STATE_F16");
        rt->recur_state_f16 = !quality && !ssd_streaming &&
                              !getenv("P36_VK_STRICT_PARITY") &&
                              (!env || !env[0] || env[0] != '0');
    }
#endif
    state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    hist_rows = P36_N_SSM_CONV - 1u;
    /* Activation scratch carries prefill_cap token rows for batched prefill;
     * single-token decode just uses row 0. */
#define P36_GPU_ALLOC_F32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_SCRATCH_F32(field, n) do { \
        rt->field = quality ? \
            p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(float)) : \
            p36_gpu_tensor_alloc_scratch((uint64_t)(n) * prefill_cap * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_U32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(uint32_t)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_MTP_F32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
    P36_GPU_ALLOC_F32(hidden, P36_N_EMBD);
    P36_GPU_ALLOC_F32(next_hidden, P36_N_EMBD);
    P36_GPU_ALLOC_F32(embed_stage[0], P36_N_EMBD);
    P36_GPU_ALLOC_F32(embed_stage[1], P36_N_EMBD);
    P36_GPU_ALLOC_SCRATCH_F32(norm, P36_N_EMBD);
    rt->last_h = p36_gpu_tensor_alloc((uint64_t)P36_N_EMBD * sizeof(float));
    if (!rt->last_h) goto fail;
    rt->last_h_raw = p36_gpu_tensor_alloc((uint64_t)P36_N_EMBD * sizeof(float));
    if (!rt->last_h_raw) goto fail;
#ifdef P36_METAL
    P36_GPU_ALLOC_F32(recur_v, P36_N_SSM_INNER);
    {
        const uint64_t attn_kv_bytes =
            (uint64_t)P36_N_HEAD_KV *
            (P36_N_HEAD_DIM + P36_N_VALUE_DIM) *
            prefill_cap * sizeof(float);
        const uint64_t inp_q8_bytes =
            (uint64_t)((P36_N_EMBD + P36_QK_K - 1u) / P36_QK_K) *
            P36_VK_Q8_K_BYTES * prefill_cap;
        rt->inp_q8 = p36_gpu_tensor_view(
            rt->recur_v, attn_kv_bytes, inp_q8_bytes);
        if (!rt->inp_q8) goto fail;
    }
    /* Allocate attention partials before recurrent scratch: on Metal, large
     * contexts can reuse this owner on layers where attention is inactive. */
    if (!p36_gpu_attn_fused_enabled()) {
        uint64_t score_bytes = p36_metal_attention_scratch_bytes(
            (uint32_t)ctx_size, prefill_cap, cache_type_k, cache_type_v);
        rt->scores = p36_gpu_tensor_alloc(score_bytes);
        if (!rt->scores) goto fail;
    }
    P36_GPU_ALLOC_F32(attn_qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u);
    P36_GPU_ALLOC_F32(attn_q, P36_N_HEAD * P36_N_HEAD_DIM);
    rt->attn_k = p36_gpu_tensor_view(rt->recur_v, 0,
        (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM *
            prefill_cap * sizeof(float));
    rt->attn_v = p36_gpu_tensor_view(rt->recur_v,
        (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM *
            prefill_cap * sizeof(float),
        (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM *
            prefill_cap * sizeof(float));
    if (!rt->attn_k || !rt->attn_v) goto fail;
    P36_GPU_ALLOC_F32(attn_out, P36_N_SSM_INNER);
#else
    uint32_t q8_width = P36_MODEL_DENSE && P36_N_FF_SHARED > P36_N_EMBD ?
        P36_N_FF_SHARED : P36_N_EMBD;
    rt->inp_q8 = quality ?
        p36_gpu_tensor_alloc((uint64_t)((q8_width + P36_QK_K - 1u) / P36_QK_K) * P36_VK_Q8_K_BYTES * prefill_cap) :
        p36_gpu_tensor_alloc_scratch((uint64_t)((q8_width + P36_QK_K - 1u) / P36_QK_K) * P36_VK_Q8_K_BYTES * prefill_cap);
    if (!rt->inp_q8) goto fail;
    P36_GPU_ALLOC_SCRATCH_F32(attn_qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u);
    P36_GPU_ALLOC_SCRATCH_F32(attn_q, P36_N_HEAD * P36_N_HEAD_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_out, P36_N_SSM_INNER);
#endif
    if (g_p36_hadamard.enabled) {
        uint32_t had_width = P36_N_EMBD;
        uint64_t signs_total = 0;
        if (P36_N_FF_SHARED > had_width) had_width = P36_N_FF_SHARED;
        if (P36_N_SSM_INNER > had_width) had_width = P36_N_SSM_INNER;
        if (P36_N_SSM_CONV_DIM > had_width) had_width = P36_N_SSM_CONV_DIM;
        {
            uint64_t q8_bytes = (uint64_t)((had_width + P36_QK_K - 1u) / P36_QK_K) *
                                P36_VK_Q8_K_BYTES * prefill_cap;
            rt->had_q8 = quality ? p36_gpu_tensor_alloc(q8_bytes)
                                 : p36_gpu_tensor_alloc_scratch(q8_bytes);
            if (!rt->had_q8) goto fail;
        }
        for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
            signs_total += g_p36_hadamard.widths[i];
        }
        rt->had_signs = p36_gpu_tensor_alloc(signs_total * sizeof(float));
        if (!rt->had_signs) goto fail;
        {
            float *sp = (float *)p36_gpu_tensor_contents(rt->had_signs);
            uint64_t off = 0;
            if (!sp) goto fail;
            for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
                memcpy(sp + off, g_p36_hadamard.signs[i],
                       (size_t)g_p36_hadamard.widths[i] * sizeof(float));
                off += g_p36_hadamard.widths[i];
            }
        }
    }
    /* A layer is attention or recurrent, never both. These equal-sized
     * views remove 64 MiB of mutually exclusive 1024-row scratch. */
    rt->recur_qkv = p36_gpu_tensor_view(rt->attn_qg, 0,
        (uint64_t)P36_N_SSM_CONV_DIM * prefill_cap * sizeof(float));
    if (!rt->recur_qkv) goto fail;
    if (!rt->recur_conv_fused)
        P36_GPU_ALLOC_F32(recur_window, (uint64_t)P36_N_SSM_CONV * P36_N_SSM_CONV_DIM);
#ifdef P36_METAL
    {
        const uint64_t conv_bytes =
            (uint64_t)P36_N_SSM_CONV_DIM * prefill_cap * sizeof(float);
        const uint64_t alpha_bytes =
            (uint64_t)P36_N_SSM_DT_RANK * prefill_cap * sizeof(float);
        const uint64_t gb_bytes = alpha_bytes * 2u;
        const uint64_t aux_bytes = conv_bytes + alpha_bytes * 2u + gb_bytes;
        if (rt->scores && p36_gpu_tensor_bytes(rt->scores) >= aux_bytes) {
            uint64_t offset = 0;
            rt->recur_conv = p36_gpu_tensor_view(
                rt->scores, offset, conv_bytes);
            offset += conv_bytes;
            rt->recur_alpha = p36_gpu_tensor_view(
                rt->scores, offset, alpha_bytes);
            offset += alpha_bytes;
            rt->recur_beta = p36_gpu_tensor_view(
                rt->scores, offset, alpha_bytes);
            offset += alpha_bytes;
            rt->recur_gb = p36_gpu_tensor_view(
                rt->scores, offset, gb_bytes);
            if (!rt->recur_conv || !rt->recur_alpha ||
                !rt->recur_beta || !rt->recur_gb) goto fail;
        } else {
            P36_GPU_ALLOC_F32(recur_conv, P36_N_SSM_CONV_DIM);
            P36_GPU_ALLOC_F32(recur_alpha, P36_N_SSM_DT_RANK);
            P36_GPU_ALLOC_F32(recur_beta, P36_N_SSM_DT_RANK);
            P36_GPU_ALLOC_F32(recur_gb, P36_N_SSM_DT_RANK * 2u);
        }
    }
#else
    if (!P36_MODEL_DENSE)
        P36_GPU_ALLOC_F32(recur_conv, P36_N_SSM_CONV_DIM);
#endif
    rt->recur_z = p36_gpu_tensor_view(rt->attn_q, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_z) goto fail;
#ifndef P36_METAL
    if (!P36_MODEL_DENSE) {
        P36_GPU_ALLOC_F32(recur_alpha, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_beta, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_gb, P36_N_SSM_DT_RANK * 2u);
    }
#endif
#ifdef P36_METAL
    /* Recurrent QKV is dead once convolution has produced recur_conv.  The
     * following delta-net stage needs Q and K simultaneously, so split that
     * dead row into two views. Command ordering preserves the dependency. */
    rt->recur_q = p36_gpu_tensor_view(rt->attn_qg, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    rt->recur_k = p36_gpu_tensor_view(rt->attn_qg,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float),
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_q || !rt->recur_k) goto fail;
#else
    if (P36_MODEL_DENSE) {
        rt->recur_q = p36_gpu_tensor_view(rt->attn_qg, 0,
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
        rt->recur_k = p36_gpu_tensor_view(rt->attn_qg,
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float),
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
        if (!rt->recur_q || !rt->recur_k) goto fail;
    } else {
        P36_GPU_ALLOC_F32(recur_q, P36_N_SSM_INNER);
        P36_GPU_ALLOC_F32(recur_k, P36_N_SSM_INNER);
    }
#endif
#ifndef P36_METAL
    if (!P36_MODEL_DENSE)
        P36_GPU_ALLOC_F32(recur_v, P36_N_SSM_INNER);
#endif
    rt->recur_proj = p36_gpu_tensor_view(rt->attn_out, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_proj) goto fail;
#ifdef P36_METAL
    /* FFN starts after attention/recurrent output and the residual norm, so
     * their QGV storage is dead for the rest of the layer. Pack all FFN
     * control/shared tensors into non-overlapping views of that 32 MiB owner. */
    if (P36_MODEL_DENSE) {
        /* Dense gate/up need two full-width owners. Mid overwrites gate;
         * the routed controls, shared tail and scalar are never used.  Large
         * attention scratch is dead by this point in every layer, so reuse it
         * for one or both panels when large enough. */
        const uint64_t ffn_panel_bytes =
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float);
        if (rt->scores &&
            p36_gpu_tensor_bytes(rt->scores) >= 2u * ffn_panel_bytes) {
            rt->ffn_shared_gate = p36_gpu_tensor_view(
                rt->scores, 0, ffn_panel_bytes);
            rt->ffn_shared_up = p36_gpu_tensor_view(
                rt->scores, ffn_panel_bytes, ffn_panel_bytes);
            if (!rt->ffn_shared_gate || !rt->ffn_shared_up) goto fail;
        } else if (rt->scores &&
                   p36_gpu_tensor_bytes(rt->scores) >= ffn_panel_bytes) {
            rt->ffn_shared_gate = p36_gpu_tensor_view(
                rt->scores, 0, ffn_panel_bytes);
            P36_GPU_ALLOC_F32(ffn_shared_up, P36_N_FF_SHARED);
        } else {
            P36_GPU_ALLOC_F32(ffn_shared_gate, P36_N_FF_SHARED);
            P36_GPU_ALLOC_F32(ffn_shared_up, P36_N_FF_SHARED);
        }
        rt->ffn_shared_mid = p36_gpu_tensor_view(
            rt->ffn_shared_gate, 0,
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
        if (!rt->ffn_shared_mid) goto fail;
    } else {
        uint64_t ffn_alias_offset = 0;
#define P36_GPU_ALIAS_FFN(field, n, type) do { \
        uint64_t alias_bytes = (uint64_t)(n) * prefill_cap * sizeof(type); \
        rt->field = p36_gpu_tensor_view( \
            rt->attn_qg, ffn_alias_offset, alias_bytes); \
        if (!rt->field) goto fail; \
        ffn_alias_offset += alias_bytes; \
    } while (0)
    P36_GPU_ALIAS_FFN(ffn_gate_logits, P36_N_EXPERT, float);
    P36_GPU_ALIAS_FFN(ffn_selected, P36_N_EXPERT_USED, uint32_t);
    P36_GPU_ALIAS_FFN(ffn_weights, P36_N_EXPERT_USED, float);
    P36_GPU_ALIAS_FFN(ffn_shared_gate, P36_N_FF_SHARED, float);
    P36_GPU_ALIAS_FFN(ffn_shared_up, P36_N_FF_SHARED, float);
    rt->ffn_shared_mid = p36_gpu_tensor_view(rt->ffn_shared_gate, 0,
        (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
    if (!rt->ffn_shared_mid) goto fail;
    P36_GPU_ALIAS_FFN(ffn_shared_out, P36_N_EMBD, float);
    P36_GPU_ALIAS_FFN(ffn_scalar, 1, float);
#undef P36_GPU_ALIAS_FFN
    }
#else
    if (P36_MODEL_DENSE) {
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_gate, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_up, P36_N_FF_SHARED);
        {
            uint64_t offset = 0;
#define P36_GPU_ALIAS_DENSE_RECUR(field, n) do { \
            uint64_t alias_bytes = (uint64_t)(n) * prefill_cap * sizeof(float); \
            rt->field = p36_gpu_tensor_view(rt->ffn_shared_gate, offset, alias_bytes); \
            if (!rt->field) goto fail; \
            offset += alias_bytes; \
        } while (0)
            P36_GPU_ALIAS_DENSE_RECUR(recur_conv, P36_N_SSM_CONV_DIM);
            P36_GPU_ALIAS_DENSE_RECUR(recur_v, P36_N_SSM_INNER);
#undef P36_GPU_ALIAS_DENSE_RECUR
        }
        /* Own buffers, not aliases of the scratch root: beta, alpha and the
         * gates are independent writes, and on one root the hazard tracker
         * sees a write-after-write and drains the queue between them. */
        P36_GPU_ALLOC_F32(recur_alpha, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_beta, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_gb, P36_N_SSM_DT_RANK * 2u);
        rt->ffn_shared_mid = p36_gpu_tensor_view(
            rt->ffn_shared_gate, 0,
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
        if (!rt->ffn_shared_mid) goto fail;
    } else {
        P36_GPU_ALLOC_F32(ffn_gate_logits, P36_N_EXPERT);
        P36_GPU_ALLOC_U32(ffn_selected, P36_N_EXPERT_USED);
        P36_GPU_ALLOC_F32(ffn_weights, P36_N_EXPERT_USED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_gate, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_up, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_mid, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_out, P36_N_EMBD);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_scalar, 1);
    }
    /* The fused attention path never touches the scores scratch; skipping it
     * saves ctx_size * n_head * prefill_cap floats (128 MiB at ctx 32k). */
    if (!p36_gpu_attn_fused_enabled())
        P36_GPU_ALLOC_SCRATCH_F32(scores, (uint64_t)ctx_size * P36_N_HEAD);
#endif
    rt->logits = p36_gpu_tensor_alloc((uint64_t)P36_N_VOCAB * sizeof(float));
    rt->top2[0] = p36_gpu_tensor_alloc(2u * sizeof(int32_t));
    rt->top2[1] = p36_gpu_tensor_alloc(2u * sizeof(int32_t));
    rt->topk8 = p36_gpu_tensor_alloc(8u * (sizeof(int32_t) + sizeof(float)));
    rt->minp_pack = p36_gpu_tensor_alloc(P36_GPU_MINP_BYTES(P36_N_VOCAB));
    if (!rt->logits || !rt->top2[0] || !rt->top2[1] || !rt->topk8 || !rt->minp_pack) goto fail;
    if (enable_mtp) {
        rt->mtp_full.cap = kv_cap;
        rt->mtp_full.type_k = cache_type_k;
        rt->mtp_full.type_v = cache_type_v;
        rt->mtp_full.k_row_bytes = k_row_bytes;
        rt->mtp_full.v_row_bytes = v_row_bytes;
        rt->mtp_full.k = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * k_row_bytes);
        rt->mtp_full.v = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * v_row_bytes);
        if (!rt->mtp_full.k || !rt->mtp_full.v) goto fail;
        P36_GPU_ALLOC_MTP_F32(mtp_tok_embd, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_e_norm, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_h_norm, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_concat, P36_N_EMBD * 2u);
        P36_GPU_ALLOC_MTP_F32(mtp_cur, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_next, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_head, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_logits, P36_N_VOCAB);
    }
    if (spec_scratch) {
        rt->spec_logits = p36_gpu_tensor_alloc((uint64_t)P36_MTP_MAX_DRAFT * P36_N_VOCAB * sizeof(float));
        rt->spec_tops = p36_gpu_tensor_alloc((uint64_t)P36_MTP_MAX_DRAFT * 2u * sizeof(int32_t));
        if (!rt->spec_logits || !rt->spec_tops) goto fail;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            rt->full[il].cap = kv_cap;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * k_row_bytes);
            rt->full[il].v = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * v_row_bytes);
            if (!rt->full[il].k || !rt->full[il].v) goto fail;
        } else {
            rt->recurrent[il].conv = p36_gpu_tensor_alloc(hist_rows * P36_N_SSM_CONV_DIM * sizeof(float));
            rt->recurrent[il].state = p36_gpu_tensor_alloc(
                state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float)));
            if (!rt->recurrent[il].conv || !rt->recurrent[il].state) goto fail;
            if (spec_scratch) {
                rt->spec_recurrent[il].conv = p36_gpu_tensor_alloc(hist_rows * P36_N_SSM_CONV_DIM * sizeof(float));
                rt->spec_recurrent[il].state = p36_gpu_tensor_alloc(
                    state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float)));
                rt->spec_recur_qkv[il] = p36_gpu_tensor_alloc(
                    (uint64_t)P36_MTP_MAX_DRAFT * P36_N_SSM_CONV_DIM * sizeof(float));
                rt->spec_recur_gb[il] = p36_gpu_tensor_alloc(
                    (uint64_t)P36_MTP_MAX_DRAFT * P36_N_SSM_DT_RANK * 2u * sizeof(float));
                if (!rt->spec_recurrent[il].conv || !rt->spec_recurrent[il].state ||
                    !rt->spec_recur_qkv[il] || !rt->spec_recur_gb[il]) goto fail;
            }
        }
    }
    if (!p36_vulkan_runtime_reset(rt)) goto fail;
#undef P36_GPU_ALLOC_F32
#undef P36_GPU_ALLOC_SCRATCH_F32
#undef P36_GPU_ALLOC_U32
#undef P36_GPU_ALLOC_MTP_F32
    return rt;
fail:
#undef P36_GPU_ALLOC_F32
#undef P36_GPU_ALLOC_SCRATCH_F32
#undef P36_GPU_ALLOC_U32
#undef P36_GPU_ALLOC_MTP_F32
    p36_vulkan_runtime_free(rt);
    return NULL;
}

static bool p36_vulkan_cache_reserve(p36_vulkan_full_attn_cache *c,
                                     uint32_t cap, uint32_t rows) {
    p36_gpu_tensor *k, *v;
    if (!c || rows > c->cap || cap <= c->cap) return false;
    p36_gpu_trim_scratch();
    k = p36_gpu_tensor_alloc_uninitialized((uint64_t)cap * c->k_row_bytes);
    v = p36_gpu_tensor_alloc_uninitialized((uint64_t)cap * c->v_row_bytes);
    if (!k || !v) goto fail;
    if (rows &&
        (!p36_gpu_tensor_copy(k, 0, c->k, 0, (uint64_t)rows * c->k_row_bytes) ||
         !p36_gpu_tensor_copy(v, 0, c->v, 0, (uint64_t)rows * c->v_row_bytes))) goto fail;
    if (!p36_gpu_synchronize()) goto fail;
    p36_gpu_tensor_free(c->k);
    p36_gpu_tensor_free(c->v);
    c->k = k;
    c->v = v;
    c->cap = cap;
    return true;
fail:
    p36_gpu_tensor_free(k);
    p36_gpu_tensor_free(v);
    return false;
}

static bool p36_vulkan_runtime_reserve_kv(p36_vulkan_runtime *rt,
                                          uint32_t need, uint32_t limit,
                                          uint32_t rows) {
    if (!rt || need > limit || rows > need) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_vulkan_full_attn_cache *c;
        uint32_t cap;
        if (!p36_layer_is_full_attention(il)) continue;
        c = &rt->full[il];
        if (need <= c->cap) continue;
        cap = p36_kv_next_cap(c->cap, need, limit);
        if (!cap) return false;
        /* Tight heap: retry with the exact need instead of the 1.25x step. */
        if (!p36_vulkan_cache_reserve(c, cap, rows) &&
            (need == cap || !p36_vulkan_cache_reserve(c, need, rows))) return false;
    }
    if (rt->mtp_enabled && need > rt->mtp_full.cap) {
        uint32_t cap = p36_kv_next_cap(rt->mtp_full.cap, need, limit);
        if (!cap || !p36_vulkan_cache_reserve(&rt->mtp_full, cap, rows)) return false;
    }
    return true;
}
#endif

static bool p36_embed_token(const p36_engine *e, int token, float *out) {
    if (!e || !out || token < 0 || token >= (int)P36_N_VOCAB) return false;
    if (!p36_tensor_row_to_float(&e->model, e->weights.token_embd, (uint64_t)token, out, P36_N_EMBD)) {
        return false;
    }
    if (e->weights.token_embd->hadamard_inverse) p36_hadamard_inverse_host(out, P36_N_EMBD);
    return true;
}

/* Multi-section M-RoPE rotation, shared by the CPU reference and the Vulkan
 * scheduler (p36_vulkan_rope_heads): both engines must go through this one
 * compiled body, because a second copy of the rotation fuses differently
 * under -ffast-math and drifts by ulps. */
