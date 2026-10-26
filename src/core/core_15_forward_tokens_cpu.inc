static bool p36_forward_tokens_cpu(p36_session *s,
                                   const int *tokens,
                                   uint32_t n_tok,
                                   uint32_t pos0,
                                   bool compute_logits) {
    p36_engine *e;
    p36_cpu_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (p36_cpu_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        if (!p36_embed_token(e, tokens[t], rt->batch_hidden + (uint64_t)t * P36_N_EMBD)) {
            fprintf(stderr, "p36: embed failed for token=%d pos=%u\n", tokens[t], pos0 + t);
            return false;
        }
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        p36_rms_norm_rows(rt->batch_norm,
                          rt->batch_hidden,
                          (const float *)(e->model.map + l->attn_norm->abs_offset),
                          n_tok,
                          P36_N_EMBD,
                          P36_RMS_EPS,
                          e->n_threads);
        if (l->kind == P36_LAYER_FULL_ATTN) {
            if (!p36_forward_full_attn_batch(e, l, il, pos0, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "p36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        } else {
            if (!p36_forward_recurrent_batch(e, l, il, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "p36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        }
        p36_directional_steering_project_rows(rt->batch_next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_attn_scale);
        if (!p36_all_finite(rt->batch_next_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        p36_add_rows(rt->batch_next_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     P36_N_EMBD,
                     e->n_threads);
        p36_rms_norm_rows(rt->batch_norm,
                          rt->batch_next_hidden,
                          (const float *)(e->model.map + l->post_attention_norm->abs_offset),
                          n_tok,
                          P36_N_EMBD,
                          P36_RMS_EPS,
                          e->n_threads);
        if (!p36_forward_ffn_batch(e, l, rt->batch_norm, rt->batch_hidden, n_tok, rt)) {
            fprintf(stderr, "p36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        p36_directional_steering_project_rows(rt->batch_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_ffn_scale);
        if (!p36_all_finite(rt->batch_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        p36_add_rows(rt->batch_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     P36_N_EMBD,
                     e->n_threads);
        if (!p36_all_finite(rt->batch_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
    }
    p36_rms_norm_rows(rt->batch_norm,
                      rt->batch_hidden,
                      (const float *)(e->model.map + e->weights.output_norm->abs_offset),
                      n_tok,
                      P36_N_EMBD,
                      P36_RMS_EPS,
                      e->n_threads);
    if (!p36_all_finite(rt->batch_norm, n_tok * P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (compute_logits) {
        const float *last = rt->batch_norm + (uint64_t)(n_tok - 1u) * P36_N_EMBD;
        last = p36_hadamard_forward_rows(e->weights.output, last, rt->batch_had, P36_N_EMBD, 1u);
        if (!p36_tensor_matvec(e, e->weights.output, last, s->logits, rt->work1, P36_N_EMBD, P36_N_VOCAB)) {
            fprintf(stderr, "p36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        p36_scale_inplace(s->logits, P36_N_VOCAB,
                          p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!p36_all_finite(s->logits, P36_N_VOCAB)) {
            fprintf(stderr, "p36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return true;
}

#ifndef P36_NO_GPU
typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} p36_vulkan_ffn_tail_ctx;

static P36_MAYBE_UNUSED void p36_vulkan_ffn_tail_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vulkan_ffn_tail_ctx *ctx = (p36_vulkan_ffn_tail_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        float g = p36_sigmoidf(ctx->scalar[t]);
        const float *st = ctx->shared + t * P36_N_EMBD;
        float *ot = ctx->out + t * P36_N_EMBD;
        for (uint32_t j = 0; j < P36_N_EMBD; j++) ot[j] = fmaf(st[j], g, ot[j]);
    }
}

typedef struct {
    float *x;
    uint32_t n_head;
    uint32_t pos0;
    const uint32_t *positions;
} p36_vulkan_rope_ctx;

static void p36_vulkan_rope_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vulkan_rope_ctx *ctx = (p36_vulkan_rope_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        uint32_t pt = ctx->pos0 + (uint32_t)t;
        uint32_t ph = pt, pw = pt;
        if (ctx->positions) {
            pt = ctx->positions[t * 3u];
            ph = ctx->positions[t * 3u + 1u];
            pw = ctx->positions[t * 3u + 2u];
        }
        for (uint32_t h = 0; h < ctx->n_head; h++) {
            p36_apply_rope_axes(ctx->x + (t * ctx->n_head + h) * P36_N_HEAD_DIM,
                                P36_N_HEAD_DIM, pt, ph, pw);
        }
    }
}

static uint32_t p36_hadamard_total_signs(void) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        total += g_p36_hadamard.widths[i];
    }
    return total;
}

static int p36_hadamard_sign_offset(uint32_t width, uint32_t *offset) {
    uint32_t off = 0;
    if (!g_p36_hadamard.enabled) return 0;
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        if (g_p36_hadamard.widths[i] == width) {
            if (offset) *offset = off;
            return 1;
        }
        off += g_p36_hadamard.widths[i];
    }
    return 0;
}

/* Forward activation transform for a rotated weight: x' = H (s . x), quantized
 * directly to rt->had_q8 for the q8_K matmul path in a single fused dispatch.
 * rt->inp_q8 keeps the plain activation for the unrotated weights that share it. */
static bool p36_hadamard_prepare(p36_vulkan_runtime *rt,
                                 const p36_gpu_tensor *src,
                                 uint32_t width, uint32_t n_rows) {
    uint32_t offset = 0;
    if (!rt || !rt->had_q8 || !rt->had_signs) return false;
    if (!p36_hadamard_sign_offset(width, &offset)) return false;
    return p36_gpu_hadamard_prepare_tensor(rt->had_q8, src, rt->had_signs,
                                           width, n_rows, offset,
                                           p36_hadamard_total_signs(),
                                           g_p36_hadamard.inv_sqrt_block, 0);
}

/* Diagnostic: does the fused norm's Q8 output agree with a fresh quantization
 * of the same norm rows? A row-1 mismatch means the epilogue wrote the wrong
 * row; a match means the consumer selected the wrong row. */
static void p36_vk_layer0_q8_probe(p36_vulkan_runtime *rt, uint32_t n_tok) {
    const uint64_t row = (P36_N_EMBD / P36_QK_K) * P36_VK_Q8_K_BYTES;
    const uint64_t bytes = (uint64_t)n_tok * row;
    p36_gpu_tensor *tmp = p36_gpu_tensor_alloc(bytes);
    p36_gpu_tensor *ha = p36_gpu_tensor_alloc(bytes);
    p36_gpu_tensor *hb = p36_gpu_tensor_alloc(bytes);
    unsigned char *a = malloc((size_t)bytes);
    unsigned char *b = malloc((size_t)bytes);
    if (!tmp || !ha || !hb || !a || !b) {
        fprintf(stderr, "p36: q8 probe alloc failed rows=%u\n", n_tok);
        goto done;
    }
    if (!p36_gpu_quantize_q8_k_tensor(tmp, rt->norm, P36_N_EMBD, n_tok)) {
        fprintf(stderr, "p36: q8 probe quantize failed rows=%u\n", n_tok);
        goto done;
    }
    if (!p36_gpu_tensor_copy(ha, 0, tmp, 0, bytes) ||
        !p36_gpu_tensor_copy(hb, 0, rt->inp_q8, 0, bytes) ||
        !p36_gpu_tensor_read(ha, 0, a, bytes) ||
        !p36_gpu_tensor_read(hb, 0, b, bytes)) {
        fprintf(stderr, "p36: q8 probe read failed rows=%u\n", n_tok);
        goto done;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint8_t *ra = a + (uint64_t)t * row;
        const uint8_t *rb = b + (uint64_t)t * row;
        uint64_t i = 0;
        while (i < row && ra[i] == rb[i]) i++;
        fprintf(stderr, "p36: q8 probe row=%u %s at byte=%llu/%llu\n",
                t, i == row ? "match" : "MISMATCH",
                (unsigned long long)i, (unsigned long long)row);
    }
done:
    free(a); free(b);
    p36_gpu_tensor_free(tmp); p36_gpu_tensor_free(ha); p36_gpu_tensor_free(hb);
}

/* Diagnostic: pull one Vulkan tensor to disk so a batched forward and a
 * row-serialized forward can be diffed without guessing which kernel drifted.
 * Gated by P36_VK_DUMP_LAYER (or legacy P36_VK_DUMP_LAYER0); one file per
 * tensor under P36_VK_DUMP_DIR. P36_VK_DUMP_POS pins the token position. */
static bool p36_vk_dump_layer_hit(uint32_t il, uint32_t n_tok, uint32_t pos0) {
    const char *dl = getenv("P36_VK_DUMP_LAYER");
    const char *dp = getenv("P36_VK_DUMP_POS");
    uint32_t want;
    if (!(dl || getenv("P36_VK_DUMP_LAYER0"))) return false;
    if (n_tok < 2u || n_tok > 4u) return false;
    want = (dl && dl[0]) ? (uint32_t)strtoul(dl, NULL, 10) : 0u;
    if (il != want) return false;
    if (dp && dp[0] && pos0 != (uint32_t)strtoul(dp, NULL, 10)) return false;
    return true;
}

static void p36_vk_layer0_dump(const char *tag, const p36_gpu_tensor *t,
                               uint64_t bytes) {
    const char *dir = getenv("P36_VK_DUMP_DIR");
    char path[512];
    void *buf;
    FILE *f;
    if (!dir || !dir[0] || !t || bytes == 0) {
        fprintf(stderr, "p36: dump %s unavailable (bytes=%llu)\n", tag,
                (unsigned long long)bytes);
        return;
    }
    buf = malloc((size_t)bytes);
    if (!buf) return;
    {
        /* Scratch tensors (norm, qkv, conv) are device-local; bounce through
         * a host-visible allocation before reading. */
        p36_gpu_tensor *tmp = p36_gpu_tensor_alloc(bytes);
        int ok = tmp && p36_gpu_tensor_copy(tmp, 0, t, 0, bytes) &&
                 p36_gpu_tensor_read(tmp, 0, buf, bytes);
        p36_gpu_tensor_free(tmp);
        if (!ok) {
            fprintf(stderr, "p36: dump %s read failed (bytes=%llu tensor=%llu)\n", tag,
                    (unsigned long long)bytes,
                    (unsigned long long)p36_gpu_tensor_bytes(t));
            free(buf);
            return;
        }
    }
    snprintf(path, sizeof(path), "%s/%s.f32", dir, tag);
    f = fopen(path, "wb");
    if (f) {
        fwrite(buf, 1, (size_t)bytes, f);
        fclose(f);
    }
    free(buf);
}

/* Same, for a folded weight whose rotation axis kept the training V order
 * (prism.hadamard.gdn_v_grouped): the runtime activation is tiled, so permute
 * tiled -> grouped inside the fused kernel before the signs and the transform. */
static bool p36_hadamard_prepare_grouped(p36_vulkan_runtime *rt,
                                         const p36_gpu_tensor *src,
                                         uint32_t width, uint32_t n_rows) {
    uint32_t offset = 0;
    uint32_t n_v = P36_N_SSM_DT_RANK;
    uint32_t n_k = P36_N_SSM_GROUP;
    if (!rt || !rt->had_q8 || !rt->had_signs) return false;
    if (!n_k || !n_v || (n_v % n_k) != 0 || (width % n_v) != 0) return false;
    if (!p36_hadamard_sign_offset(width, &offset)) return false;
    return p36_gpu_hadamard_prepare_tensor(rt->had_q8, src, rt->had_signs,
                                           width, n_rows, offset,
                                           p36_hadamard_total_signs(),
                                           g_p36_hadamard.inv_sqrt_block, 1);
}

static bool p36_forward_ffn_vulkan_model(p36_vulkan_runtime *rt,
                                         const p36_model *m,
                                         const p36_layer_weights *l,
                                         float expert_weights_scale,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out,
                                         bool session_batch) {
    p36_gpu_moe_weight gate;
    p36_gpu_moe_weight up;
    p36_gpu_moe_weight down;
    const float *shared;
    const float *scalar;
    float *outp;
    if (!rt || !m || !l || !inp || !out) return false;
    if (P36_MODEL_DENSE) {
        const p36_gpu_tensor *ffn_in = inp;
        const p36_gpu_tensor *ffn_q8 = rt->inp_q8;
        bool gate_rot = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ||
                        p36_hadamard_rotate_enabled(l->ffn_up_shexp);
        if (gate_rot) {
            if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
            /* The fused prepare writes only had_q8; folded weights always take
             * the q8_K path, so ffn_in stays the f32 source and is unused.
             * ponytail: a folded f32/f16/q8_0 weight would need the prepare
             * kernel to also emit the f32 activation. */
            ffn_q8 = rt->had_q8;
        } else if (!rt->inp_q8_valid && !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
            return false;
        }
        rt->inp_q8_valid = false;
        bool pair_projected = false;
#ifdef P36_METAL
        const char *iq3_pair = getenv("P36_METAL_IQ3_PAIR");
        if (n_tok == 1u && (!iq3_pair || iq3_pair[0] != '0') &&
            l->ffn_gate_shexp->type == P36_TENSOR_IQ3_S &&
            l->ffn_up_shexp->type == P36_TENSOR_IQ3_S) {
            pair_projected = p36_gpu_matmul_iq3_s_pair_tensor(
                rt->ffn_shared_gate, rt->ffn_shared_up,
                m->map, m->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                P36_N_EMBD, P36_N_FF_SHARED, inp) != 0;
        } else if (n_tok == 1u && (!iq3_pair || iq3_pair[0] != '0') &&
                   l->ffn_gate_shexp->type == P36_TENSOR_IQ3_XXS &&
                   l->ffn_up_shexp->type == P36_TENSOR_IQ3_XXS) {
            pair_projected = p36_gpu_matmul_iq3_xxs_pair_tensor(
                rt->ffn_shared_gate, rt->ffn_shared_up,
                m->map, m->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                P36_N_EMBD, P36_N_FF_SHARED, inp) != 0;
        }
#endif
#ifndef P36_METAL
        /* Vulkan prefill pair: one dispatch decodes the gate and up tiles
         * against a single staged activation tile.  Small 2..4-row batches
         * (MTP verify) skip the 128-row tile and fall through to the
         * single-projection nx kernels, which stream each weight once for
         * the whole batch. */
        {
            const char *iq3_pair = getenv("P36_VK_DENSE_IQ3_PAIR");
            if (!pair_projected && n_tok > 4u &&
                (!iq3_pair || iq3_pair[0] != '0') &&
                l->ffn_gate_shexp->type == P36_TENSOR_IQ3_XXS &&
                l->ffn_up_shexp->type == P36_TENSOR_IQ3_XXS) {
                pair_projected = p36_gpu_matmul_iq3_xxs_pair_mmq_tensor(
                    rt->ffn_shared_gate, rt->ffn_shared_up,
                    m->map, m->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    P36_N_EMBD, P36_N_FF_SHARED, ffn_q8, n_tok, 1.0f) != 0;
            }
        }
#endif
        if (!pair_projected &&
            (!p36_gpu_tensor_matmul_q8_or_float_scaled(
                 m, l->ffn_gate_shexp, ffn_in, ffn_q8,
                 rt->ffn_shared_gate, P36_N_EMBD, P36_N_FF_SHARED,
                 n_tok, 1.0f) ||
             !p36_gpu_tensor_matmul_q8_or_float_scaled(
                 m, l->ffn_up_shexp, ffn_in, ffn_q8,
                 rt->ffn_shared_up, P36_N_EMBD, P36_N_FF_SHARED,
                 n_tok, 1.0f))) {
            return false;
        }
        bool fused_swiglu_q8 = false;
#ifndef P36_METAL
        fused_swiglu_q8 = p36_gpu_swiglu_q8_k_tensor(
            rt->ffn_shared_mid, rt->inp_q8, rt->ffn_shared_gate,
            rt->ffn_shared_up, P36_N_FF_SHARED, n_tok, 0.0f, 1.0f);
#endif
        if (!fused_swiglu_q8 &&
            !p36_gpu_swiglu_tensor(
                 rt->ffn_shared_mid, rt->ffn_shared_gate,
                 rt->ffn_shared_up, n_tok * P36_N_FF_SHARED, 0.0f, 1.0f)) {
            return false;
        }
        const p36_gpu_tensor *down_in = rt->ffn_shared_mid;
        const p36_gpu_tensor *down_q8 = rt->inp_q8;
        if (p36_hadamard_rotate_enabled(l->ffn_down_shexp)) {
            if (!p36_hadamard_prepare(rt, rt->ffn_shared_mid, P36_N_FF_SHARED, n_tok)) {
                return false;
            }
            down_q8 = rt->had_q8;
        } else if (!fused_swiglu_q8 &&
                   !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, rt->ffn_shared_mid,
                                                 P36_N_FF_SHARED, n_tok)) {
            return false;
        }
        if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
                m, l->ffn_down_shexp, down_in, down_q8,
                out, P36_N_FF_SHARED, P36_N_EMBD, n_tok, 1.0f)) {
            return false;
        }
        return true;
    }
    gate.offset = l->ffn_gate_exps->abs_offset;
    gate.type = l->ffn_gate_exps->type;
    gate.scales_offset = l->ffn_gate_exps_scale ? l->ffn_gate_exps_scale->abs_offset : 0;
    gate.has_scales = l->ffn_gate_exps_scale != NULL;
    up.offset = l->ffn_up_exps->abs_offset;
    up.type = l->ffn_up_exps->type;
    up.scales_offset = l->ffn_up_exps_scale ? l->ffn_up_exps_scale->abs_offset : 0;
    up.has_scales = l->ffn_up_exps_scale != NULL;
    down.offset = l->ffn_down_exps->abs_offset;
    down.type = l->ffn_down_exps->type;
    down.scales_offset = l->ffn_down_exps_scale ? l->ffn_down_exps_scale->abs_offset : 0;
    down.has_scales = l->ffn_down_exps_scale != NULL;
    /* The router and the shared-expert gate scalar run on the host from the
     * host-written norm, and the shared-expert gate/up dispatches record
     * before the routed pass so the routed block's one flush completes them
     * too. */
    if (!p36_gpu_tensor_matmul_scaled(
            m, l->ffn_gate_inp, inp, rt->ffn_gate_logits,
            P36_N_EMBD, P36_N_EXPERT, n_tok, 1.0f) ||
        !p36_gpu_router_topk_tensor(
            rt->ffn_selected, rt->ffn_weights, rt->ffn_gate_logits,
            P36_N_EXPERT, P36_N_EXPERT_USED, n_tok,
            expert_weights_scale) ||
        !p36_gpu_tensor_matmul_scaled(
            m, l->ffn_gate_inp_shexp, inp, rt->ffn_scalar,
            P36_N_EMBD, 1, n_tok, 1.0f)) {
        return false;
    }
    /* Fast routed pass, decode and prefill: f32 activations straight into
     * the fused IQ2_XXS gate/up and Q2_K down kernels, expert schedule
     * built and consumed on GPU. When it runs, the q8_K input staging is
     * only needed for k-quant shared-expert projections, which this model
     * does not have. */
    bool routed_done =
        p36_gpu_moe_ffn_f32_tensor(out, m->map, m->size,
                                          &gate, &up, &down,
                                          rt->ffn_selected, rt->ffn_weights,
                                          il, P36_N_EXPERT_USED, inp, n_tok,
                                          P36_N_EMBD, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT) != 0;
    /* Diagnostic only: why a layer left the fused f32 expert path. */
    if (!routed_done && getenv("P36_VK_MOE_ROUTE_DEBUG")) {
        fprintf(stderr, "p36: moe f32 route miss il=%u n_tok=%u gate=%u up=%u down=%u\n",
                il, n_tok, gate.type, up.type, down.type);
    }
    if (routed_done && (n_tok == 1 || session_batch) &&
        l->ffn_gate_shexp->type == P36_TENSOR_Q8_0 &&
        l->ffn_up_shexp->type == P36_TENSOR_Q8_0 &&
        l->ffn_down_shexp->type == P36_TENSOR_Q8_0) {
        const char *env = getenv("P36_VK_SHARED_FFN_DECODE");
        if (!env || !env[0] || env[0] != '0') {
            if (n_tok == 1 && p36_gpu_shared_ffn_decode_tensor(
                    out, rt->ffn_shared_mid, inp, rt->ffn_scalar,
                    m->map, m->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    l->ffn_down_shexp->abs_offset,
                    P36_N_EMBD, P36_N_FF_SHARED, P36_N_EMBD,
                    p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                    p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                    p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
                return true;
            }
            bool ok = true;
            for (uint32_t row = 0; session_batch && row < n_tok && ok; row++) {
                p36_gpu_tensor *row_out = p36_gpu_tensor_view(
                    out, (uint64_t)row * P36_N_EMBD * sizeof(float),
                    (uint64_t)P36_N_EMBD * sizeof(float));
                p36_gpu_tensor *row_mid = p36_gpu_tensor_view(
                    rt->ffn_shared_mid,
                    (uint64_t)row * P36_N_FF_SHARED * sizeof(float),
                    (uint64_t)P36_N_FF_SHARED * sizeof(float));
                p36_gpu_tensor *row_inp = p36_gpu_tensor_view(
                    inp, (uint64_t)row * P36_N_EMBD * sizeof(float),
                    (uint64_t)P36_N_EMBD * sizeof(float));
                p36_gpu_tensor *row_scalar = p36_gpu_tensor_view(
                    rt->ffn_scalar, (uint64_t)row * sizeof(float),
                    sizeof(float));
                ok = row_out && row_mid && row_inp && row_scalar &&
                    p36_gpu_shared_ffn_decode_tensor(
                        row_out, row_mid, row_inp, row_scalar,
                        m->map, m->size,
                        l->ffn_gate_shexp->abs_offset,
                        l->ffn_up_shexp->abs_offset,
                        l->ffn_down_shexp->abs_offset,
                        P36_N_EMBD, P36_N_FF_SHARED, P36_N_EMBD,
                        p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                        p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                        p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f));
                p36_gpu_tensor_free(row_scalar);
                p36_gpu_tensor_free(row_inp);
                p36_gpu_tensor_free(row_mid);
                p36_gpu_tensor_free(row_out);
            }
            if (session_batch && ok) return true;
        }
    }
    if ((!routed_done ||
         l->ffn_gate_shexp->type != P36_TENSOR_Q8_0 ||
         l->ffn_up_shexp->type != P36_TENSOR_Q8_0) &&
        !rt->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_gate_shexp, inp, rt->inp_q8, rt->ffn_shared_gate,
                                                  P36_N_EMBD, P36_N_FF_SHARED, n_tok,
                                                  p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f))) {
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_up_shexp, inp, rt->inp_q8, rt->ffn_shared_up,
                                                  P36_N_EMBD, P36_N_FF_SHARED, n_tok,
                                                  p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f))) {
        return false;
    }
    if (!routed_done &&
        !p36_gpu_moe_ffn_q8_tensor(out, m->map, m->size,
                                   &gate, &up, &down,
                                   rt->ffn_selected, rt->ffn_weights,
                                   il, P36_N_EXPERT_USED, rt->inp_q8, n_tok,
                                   P36_N_EMBD, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT)) {
        return false;
    }
    if (!p36_gpu_swiglu_tensor(rt->ffn_shared_mid, rt->ffn_shared_gate, rt->ffn_shared_up,
                               n_tok * P36_N_FF_SHARED, 0.0f, 1.0f)) {
        return false;
    }
    if (!p36_gpu_tensor_matmul_scaled(m, l->ffn_down_shexp, rt->ffn_shared_mid, rt->ffn_shared_out,
                                      P36_N_FF_SHARED, P36_N_EMBD, n_tok,
                                      p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }
    /* Shared-expert tail on the host: the CPU compiles "out += shared * g"
     * to one fma after the down projection's scale, so the fused rounding
     * must be reproduced with fmaf() on the unscaled-by-g down output. */
    (void)shared;
    (void)scalar;
    (void)outp;
    return p36_gpu_ffn_tail_tensor(out, rt->ffn_shared_out, rt->ffn_scalar, P36_N_EMBD, n_tok) != 0;
}

static bool p36_forward_ffn_vulkan(p36_session *s,
                                   const p36_layer_weights *l,
                                   uint32_t il,
                                   uint32_t n_tok,
                                   const p36_gpu_tensor *inp,
                                   p36_gpu_tensor *out) {
    p36_engine *e;
    if (!s) return false;
    e = s->engine;
    return p36_forward_ffn_vulkan_model((p36_vulkan_runtime *)s->runtime,
                                        &e->model,
                                        l,
                                        e->expert_weights_scale,
                                        il,
                                        n_tok,
                                        inp,
                                        out,
                                        false);
}

/* RoPE stays on the host even on the Vulkan path: the rotated values feed KV
 * storage, so they must match the CPU reference bit for bit, and only the
 * p36_apply_rope_one() body compiled in this unit guarantees that.
 * Token row t rotates at absolute position pos0 + t. */
static bool p36_vulkan_rope_heads(p36_gpu_tensor *x, uint32_t n_head,
                                  uint32_t pos0, uint32_t n_tok,
                                  const uint32_t *positions, bool quality) {
    const char *env = getenv("P36_VK_GPU_ROPE");
    bool use_gpu = !env || !env[0] || env[0] != '0';
    if (positions)
        return p36_gpu_rope_qwen_mrope_rows_tensor(x, n_head,
                                                    positions, n_tok) != 0;
    if (!positions && !quality && use_gpu)
        return p36_gpu_rope_qwen_rows_tensor(x, n_head, pos0, n_tok) != 0;
    p36_vulkan_rope_ctx ctx;
    float *xp;
    uint64_t bytes = (uint64_t)n_tok * n_head * P36_N_HEAD_DIM * sizeof(float);
    if (p36_gpu_tensor_bytes(x) < bytes) return false;
    xp = p36_gpu_tensor_contents_named(x, "submit_wait_host_rope");
    if (!xp) return false;
    ctx.x = xp;
    ctx.n_head = n_head;
    ctx.pos0 = pos0;
    ctx.positions = positions;
    if ((uint64_t)n_tok * n_head * P36_N_HEAD_DIM >= 4096u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vulkan_rope_rows, &ctx);
    else p36_vulkan_rope_rows(&ctx, 0, n_tok);
    return true;
}

static bool p36_vulkan_prepare_attn_heads(p36_gpu_tensor *dst,
                                          const p36_gpu_tensor *src,
                                          const p36_model *m,
                                          const p36_tensor *norm,
                                          uint32_t src_stride,
                                          uint32_t n_head,
                                          uint32_t pos0,
                                          uint32_t n_tok,
                                          const uint32_t *positions,
                                          bool quality) {
    const char *env = getenv("P36_VK_NORM_ROPE");
    bool fused = !env || !env[0] || env[0] != '0';
    if (!positions && !quality && fused) {
        return p36_gpu_rms_norm_rope_qwen_rows_tensor(dst, src,
                                                       m->map, m->size,
                                                       norm->abs_offset,
                                                       src_stride, n_head,
                                                       pos0, n_tok,
                                                       P36_RMS_EPS) != 0;
    }
    if (dst != src && !p36_gpu_extract_full_attn_q_tensor(dst, src, n_tok)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(dst, dst,
                                             m->map, m->size, norm->abs_offset,
                                             P36_N_HEAD_DIM, n_tok * n_head,
                                             P36_RMS_EPS)) {
        return false;
    }
    return p36_vulkan_rope_heads(dst, n_head, pos0, n_tok, positions, quality);
}

static bool p36_forward_full_attn_vulkan_model(p36_vulkan_runtime *rt,
                                               const p36_model *m,
                                               const p36_layer_weights *l,
                                               p36_vulkan_full_attn_cache *cache,
                                               bool quality,
                                               uint32_t il,
                                               uint32_t pos0,
                                               uint32_t rope_pos0,
                                               const uint32_t *positions,
                                               uint32_t n_tok,
                                               const p36_gpu_tensor *inp,
                                               p36_gpu_tensor *out) {
    if (!rt || !m || !l || !cache || !inp || !out) return false;
    const p36_gpu_tensor *qkv_in = inp;
    const p36_gpu_tensor *qkv_q8 = rt->inp_q8;
    if (p36_hadamard_rotate_enabled(l->attn_q) ||
        p36_hadamard_rotate_enabled(l->attn_k) ||
        p36_hadamard_rotate_enabled(l->attn_v)) {
        if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
        qkv_q8 = rt->had_q8;
    } else if ((l->attn_q->type != P36_TENSOR_Q8_0 || l->attn_k->type != P36_TENSOR_Q8_0 ||
                l->attn_v->type != P36_TENSOR_Q8_0) &&
               !rt->inp_q8_valid &&
               !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_q, qkv_in, qkv_q8, rt->attn_qg,
                                                  P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_q_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k, qkv_in, qkv_q8, rt->attn_k,
                                                  P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_k_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v, qkv_in, qkv_q8, rt->attn_v,
                                                  P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_v_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_vulkan_prepare_attn_heads(rt->attn_q, rt->attn_qg, m, l->attn_q_norm,
                                       P36_N_HEAD_DIM * 2u, P36_N_HEAD,
                                       rope_pos0, n_tok, positions, quality)) return false;
    {
        const char *env = getenv("P36_VK_NORM_ROPE_KV");
        bool fused_kv = !positions && rope_pos0 == pos0 && !quality &&
                        (!env || !env[0] || env[0] != '0') &&
                        cache->type_k == 0u && cache->type_v == 0u &&
                        cache->k_row_bytes == P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(uint16_t);
        if (fused_kv) {
            if (!p36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, rt->attn_k, rt->attn_v,
                    m->map, m->size, l->attn_k_norm->abs_offset,
                    P36_N_HEAD_DIM, P36_N_HEAD_KV,
                    pos0, n_tok, cache->cap, P36_RMS_EPS)) {
                return false;
            }
        } else if (!positions && rope_pos0 == pos0 && !quality &&
                   (!env || !env[0] || env[0] != '0') &&
                   cache->type_k == P36_KV_CACHE_Q8_0 &&
                   cache->type_v == P36_KV_CACHE_Q4_0 &&
                   (P36_N_HEAD_DIM % 32u) == 0u &&
                   (P36_N_VALUE_DIM % 32u) == 0u &&
                   p36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
                       cache->k, cache->v, rt->attn_k, rt->attn_v,
                       m->map, m->size, l->attn_k_norm->abs_offset,
                       P36_N_HEAD_DIM, P36_N_HEAD_KV,
                       pos0, n_tok, cache->cap, P36_RMS_EPS,
                       cache->k_row_bytes, cache->v_row_bytes)) {
            /* Fused quantized KV store succeeded */
        } else {
            if (!p36_vulkan_prepare_attn_heads(rt->attn_k, rt->attn_k, m, l->attn_k_norm,
                                               P36_N_HEAD_DIM, P36_N_HEAD_KV,
                                               rope_pos0, n_tok, positions,
                                               quality)) return false;
            if (!p36_gpu_attn_kv_store_tensor(cache->k, cache->v,
                                              rt->attn_k, rt->attn_v,
                                              pos0, n_tok, cache->cap,
                                              P36_N_HEAD_KV * P36_N_HEAD_DIM,
                                              P36_N_HEAD_KV * P36_N_VALUE_DIM,
                                              cache->type_k,
                                              cache->type_v,
                                              cache->k_row_bytes,
                                              cache->v_row_bytes)) return false;
        }
    }
    if (!p36_gpu_attn_decode_tensor(rt->attn_out,
                                    rt->attn_q,
                                    rt->attn_qg,
                                     cache->k,
                                     cache->v,
                                     rt->scores,
                                     m->map,
                                     m->size,
                                     l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                                    l->attn_sinks != NULL,
                                    pos0,
                                    n_tok,
                                    P36_N_HEAD,
                                    P36_N_HEAD_KV,
                                    P36_N_HEAD_DIM,
                                    cache->type_k,
                                    cache->type_v,
                                    cache->k_row_bytes,
                                    cache->v_row_bytes)) {
        return false;
    }
    if (p36_hadamard_rotate_enabled(l->attn_output)) {
        if (!p36_hadamard_prepare(rt, rt->attn_out, P36_N_SSM_INNER, n_tok) ||
            !p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_output, rt->attn_out, rt->had_q8, out,
                                                      P36_N_SSM_INNER, P36_N_EMBD, n_tok,
                                                      p36_tensor_scalar_or(m, l->attn_output_scale, 1.0f))) {
            fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos0);
            return false;
        }
    } else if (!p36_gpu_tensor_matmul_scaled(m, l->attn_output, rt->attn_out, out,
                                             P36_N_SSM_INNER, P36_N_EMBD, n_tok,
                                             p36_tensor_scalar_or(m, l->attn_output_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    return true;
}

static bool p36_forward_full_attn_vulkan(p36_session *s,
                                         const p36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t pos0,
                                         const uint32_t *positions,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    if (!s) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    return p36_forward_full_attn_vulkan_model(rt,
                                              &e->model,
                                              l,
                                              &rt->full[il],
                                              e->quality,
                                              il,
                                              pos0,
                                              (uint32_t)((int32_t)pos0 + s->rope_delta),
                                              positions,
                                              n_tok,
                                              inp,
                                              out);
}

static bool p36_forward_recurrent_vulkan(p36_session *s,
                                         const p36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    p36_vulkan_recurrent_cache *cache;
    if (!s || !l || !inp || !out) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    cache = &rt->recurrent[il];
    const char *q8_pair_env = getenv("P36_VK_Q8_PAIR");
    bool q8_pair = !getenv("P36_VK_STRICT_PARITY") &&
                   (!q8_pair_env || q8_pair_env[0] != '0');
    if ((l->attn_qkv->type != P36_TENSOR_Q8_0 || l->attn_gate->type != P36_TENSOR_Q8_0) &&
        !rt->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    const p36_gpu_tensor *qkv_in = inp;
    const p36_gpu_tensor *qkv_q8 = rt->inp_q8;
    if (p36_hadamard_rotate_enabled(l->attn_qkv) ||
        p36_hadamard_rotate_enabled(l->attn_gate)) {
        if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
        qkv_q8 = rt->had_q8;
    }
    bool pair_projected = q8_pair && n_tok == 1u &&
                          l->attn_qkv->type == P36_TENSOR_Q8_0 &&
                          l->attn_gate->type == P36_TENSOR_Q8_0 &&
                          p36_gpu_matmul_q8_0_pair_scaled_tensor(
                              rt->recur_qkv, rt->recur_z,
                              e->model.map, e->model.size,
                              l->attn_qkv->abs_offset, l->attn_gate->abs_offset,
                              P36_N_EMBD, P36_N_SSM_CONV_DIM, P36_N_SSM_INNER, inp,
                              p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f),
                              p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
    if (!pair_projected) {
        bool serial_qkv = n_tok > 1u && n_tok <= 4u &&
                          getenv("P36_VK_SERIAL_QKV") != NULL;
        if (serial_qkv) {
            /* Diagnostic parity probe: project one row at a time into that
             * row's slot, so the batch dump compares equal-width rows. */
            for (uint32_t t = 0; t < n_tok; t++) {
                p36_gpu_tensor *in = p36_gpu_tensor_view(
                    qkv_in, (uint64_t)t * P36_N_EMBD * sizeof(float),
                    P36_N_EMBD * sizeof(float));
                p36_gpu_tensor *q = p36_gpu_tensor_view(
                    rt->recur_qkv, (uint64_t)t * P36_N_SSM_CONV_DIM * sizeof(float),
                    P36_N_SSM_CONV_DIM * sizeof(float));
                p36_gpu_tensor *z = p36_gpu_tensor_view(
                    rt->recur_z, (uint64_t)t * P36_N_SSM_INNER * sizeof(float),
                    P36_N_SSM_INNER * sizeof(float));
                bool ok = in && q && z &&
                    p36_gpu_tensor_matmul_q8_or_float_scaled(
                        &e->model, l->attn_qkv, in, qkv_q8, q,
                        P36_N_EMBD, P36_N_SSM_CONV_DIM, 1,
                        p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f)) &&
                    p36_gpu_tensor_matmul_q8_or_float_scaled(
                        &e->model, l->attn_gate, in, qkv_q8, z,
                        P36_N_EMBD, P36_N_SSM_INNER, 1,
                        p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
                p36_gpu_tensor_free(in);
                p36_gpu_tensor_free(q);
                p36_gpu_tensor_free(z);
                if (!ok) {
                    fprintf(stderr, "p36: recurrent serial qkv failed at layer=%u row=%u\n",
                            il, t);
                    return false;
                }
            }
        } else {
            if (!p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_qkv, qkv_in, qkv_q8, rt->recur_qkv,
                                                          P36_N_EMBD, P36_N_SSM_CONV_DIM, n_tok,
                                                          p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f))) {
                fprintf(stderr, "p36: recurrent attn_qkv failed at layer=%u\n", il);
                return false;
            }
            /* The z gate dispatch records before the host conv step so the flush
             * the conv's qkv read triggers completes both projections. */
            if (!p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_gate, qkv_in, qkv_q8, rt->recur_z,
                                                          P36_N_EMBD, P36_N_SSM_INNER, n_tok,
                                                          p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f))) {
                fprintf(stderr, "p36: recurrent attn_gate failed at layer=%u\n", il);
                return false;
            }
        }
    }
    if (il == 0u && getenv("P36_VK_DUMP_QKV")) {
        static uint32_t qkv_call;
        char tag[48];
        p36_gpu_synchronize();
        snprintf(tag, sizeof(tag), "qkv_pre_%u", qkv_call++);
        p36_vk_layer0_dump(tag, rt->recur_qkv,
            (uint64_t)n_tok * P36_N_SSM_CONV_DIM * sizeof(float));
    }
    bool front=false;
#ifndef P36_METAL
    if (!e->quality && !e->ssd_streaming && n_tok==1u && rt->recur_conv_fused &&
        P36_N_SSM_STATE==128u && P36_N_SSM_CONV==4u &&
        l->ssm_alpha->type==P36_TENSOR_Q8_0 && l->ssm_beta->type==P36_TENSOR_Q8_0) {
        const uint64_t offsets[5]={l->ssm_conv1d->abs_offset,l->ssm_alpha->abs_offset,
            l->ssm_beta->abs_offset,l->ssm_dt->abs_offset,l->ssm_a->abs_offset};
        front=p36_gpu_gdn_front_tensor(cache->conv,rt->recur_qkv,inp,
            rt->recur_q,rt->recur_k,rt->recur_v,rt->recur_gb,
            e->model.map,e->model.size,offsets,P36_N_EMBD,P36_N_SSM_GROUP,P36_N_SSM_DT_RANK,
            p36_tensor_scalar_or(&e->model,l->ssm_alpha_scale,1.0f),
            p36_tensor_scalar_or(&e->model,l->ssm_beta_scale,1.0f),P36_RMS_EPS)!=0;
    }
#endif
    if (front) goto recurrent_update;
    {
        if (!rt->recur_conv_fused) {
            if (!p36_gpu_recurrent_conv_step_tensor(cache->conv, rt->recur_qkv,
                                                    rt->recur_window, n_tok)) return false;
            if (!p36_gpu_ssm_conv_silu_tensor(rt->recur_conv,
                                              rt->recur_window,
                                              e->model.map,
                                              e->model.size,
                                              l->ssm_conv1d->abs_offset,
                                              P36_N_SSM_CONV_DIM,
                                              P36_N_SSM_CONV,
                                              n_tok)) return false;
        } else if (!p36_gpu_recurrent_conv_silu_tensor(
                       cache->conv, rt->recur_qkv, rt->recur_conv,
                       e->model.map, e->model.size, l->ssm_conv1d->abs_offset,
                       P36_N_SSM_CONV_DIM, P36_N_SSM_CONV, n_tok)) {
            return false;
        }
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_conv, n_tok * P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    pair_projected = q8_pair && n_tok == 1u &&
                     l->ssm_beta->type == P36_TENSOR_Q8_0 &&
                     l->ssm_alpha->type == P36_TENSOR_Q8_0 &&
                     p36_gpu_matmul_q8_0_pair_scaled_tensor(
                         rt->recur_beta, rt->recur_alpha,
                         e->model.map, e->model.size,
                         l->ssm_beta->abs_offset, l->ssm_alpha->abs_offset,
                         P36_N_EMBD, P36_N_SSM_DT_RANK, P36_N_SSM_DT_RANK, inp,
                         p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f),
                         p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    /* Both IQ2_S (this file's shape): one shared Q8_K quantize instead of
     * the two the individual calls below would each do internally, plus
     * one dispatch instead of two -- ssm_alpha/beta are tiny (2048x32) so
     * dispatch/quantize overhead dominates their cost, not DRAM bandwidth
     * (see karpathy/evidence/raw/iq2m-reprofile-2026-09-21.md). */
    if (!pair_projected && n_tok == 1u &&
        l->ssm_beta->type == P36_TENSOR_IQ2_S &&
        l->ssm_alpha->type == P36_TENSOR_IQ2_S) {
        if (!rt->inp_q8_valid &&
            !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
            return false;
        }
        rt->inp_q8_valid = false;
        pair_projected = p36_gpu_matmul_iq2s_pair_scaled_tensor(
            rt->recur_beta, rt->recur_alpha,
            e->model.map, e->model.size,
            l->ssm_beta->abs_offset, l->ssm_alpha->abs_offset,
            P36_N_EMBD, P36_N_SSM_DT_RANK, P36_N_SSM_DT_RANK, rt->inp_q8,
            p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f),
            p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    }
    if (!pair_projected) {
        if (!p36_gpu_tensor_matmul_dense_q8_scaled(
                &e->model, l->ssm_beta, inp, rt->inp_q8, rt->recur_beta,
                P36_N_EMBD, P36_N_SSM_DT_RANK, n_tok,
                p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent ssm_beta failed at layer=%u\n", il);
            return false;
        }
        if (!p36_gpu_tensor_matmul_dense_q8_scaled(
                &e->model, l->ssm_alpha, inp, rt->inp_q8, rt->recur_alpha,
                P36_N_EMBD, P36_N_SSM_DT_RANK, n_tok,
                p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent ssm_alpha failed at layer=%u\n", il);
            return false;
        }
    }
    {
        const char *env = getenv("P36_VK_DELTA_QKV");
        bool fused = !e->quality && !e->ssd_streaming &&
                     (!env || !env[0] || env[0] != '0') &&
                     p36_gpu_delta_qkv_l2_norm_tensor(
                         rt->recur_q, rt->recur_k, rt->recur_v,
                         rt->recur_conv, P36_N_SSM_DT_RANK,
                         P36_N_SSM_GROUP, P36_N_SSM_STATE,
                         P36_N_SSM_CONV_DIM, n_tok, P36_RMS_EPS);
        if (!fused) {
            if (!p36_gpu_delta_qk_l2_norm_tensor(rt->recur_q,
                                                 rt->recur_k,
                                                 rt->recur_conv,
                                                 P36_N_SSM_DT_RANK,
                                                 P36_N_SSM_GROUP,
                                                 P36_N_SSM_STATE,
                                                 P36_N_SSM_CONV_DIM,
                                                 n_tok,
                                                 P36_RMS_EPS)) {
                return false;
            }
            if (!p36_gpu_extract_recurrent_v_tensor(rt->recur_v, rt->recur_conv, n_tok)) return false;
        }
    }
    if (!p36_gpu_delta_net_gates_tensor(
            rt->recur_gb, rt->recur_alpha, rt->recur_beta,
            e->model.map, e->model.size, l->ssm_dt->abs_offset,
            l->ssm_a->abs_offset, P36_N_SSM_DT_RANK, n_tok)) {
        return false;
    }
recurrent_update:
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_gb, n_tok * P36_N_SSM_DT_RANK * 2u)) {
        fprintf(stderr, "p36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_gpu_delta_net_decode_tensor(cache->state,
                                         rt->recur_q,
                                         rt->recur_k,
                                         rt->recur_v,
                                         rt->recur_gb,
                                         rt->recur_proj,
                                         P36_N_SSM_DT_RANK,
                                         P36_N_SSM_STATE,
                                         n_tok)) {
        return false;
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_proj, n_tok * P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    bool norm_gate_q8 = false;
#ifndef P36_METAL
    norm_gate_q8 = !e->quality &&
        p36_gpu_recurrent_norm_gate_q8_k_tensor(
            rt->recur_proj, rt->inp_q8, rt->recur_z,
            e->model.map, e->model.size, l->ssm_norm->abs_offset,
            n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS);
#endif
    bool norm_gate_fused = norm_gate_q8 || (!e->quality &&
        p36_gpu_recurrent_norm_gate_tensor(
            rt->recur_proj, rt->recur_z, e->model.map, e->model.size,
            l->ssm_norm->abs_offset, P36_N_SSM_STATE,
            n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS));
    if (!norm_gate_fused &&
        (!p36_gpu_rms_norm_weight_rows_tensor(
             rt->recur_proj, rt->recur_proj,
             e->model.map, e->model.size, l->ssm_norm->abs_offset,
             P36_N_SSM_STATE, n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS) ||
         !p36_gpu_swiglu_tensor(
             rt->recur_proj, rt->recur_z, rt->recur_proj,
             n_tok * P36_N_SSM_INNER, 0.0f, 1.0f))) {
        return false;
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_proj, n_tok * P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    float out_scale = p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f);
    if (p36_hadamard_rotate_enabled(l->ssm_out) &&
        !p36_hadamard_prepare_grouped(rt, rt->recur_proj, P36_N_SSM_INNER, n_tok)) {
        return false;
    }
    bool out_ok = p36_hadamard_rotate_enabled(l->ssm_out) ?
        p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->ssm_out, rt->recur_proj, rt->had_q8, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale) :
        norm_gate_q8 ?
        p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->ssm_out, rt->recur_proj, rt->inp_q8, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale) :
        p36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_out, rt->recur_proj, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale);
    if (!out_ok) {
        fprintf(stderr, "p36: recurrent ssm_out failed at layer=%u\n", il);
        return false;
    }
    return true;
}

/* Forward n_tok prompt tokens at positions pos0..pos0+n_tok-1 through the
 * Vulkan runtime, optionally leaving the last token's logits in s->logits.
 * Each token runs the exact arithmetic of the single-token path; batching
 * only widens the matmul dispatches and lets the host parity ops process
 * the whole chunk per GPU round-trip, which is where prefill time went. */
static bool p36_forward_tokens_vulkan_into(p36_session *s,
                                           const int *tokens,
                                           const float *embeddings,
                                           const uint32_t *positions,
                                           uint32_t n_tok,
                                           uint32_t pos0,
                                           p36_gpu_tensor *logits_out,
                                           bool logits_all_rows) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    p36_gpu_stream_expert_cache_note_tokens(n_tok);
    /* Embed into the host-only stage, then a queued device copy into the
     * GPU-written hidden tensor: mapping hidden here would wait out the
     * previous chunk's whole in-flight batch. */
    {
        p36_gpu_tensor *stage = rt->embed_stage[rt->embed_flip & 1u];
        rt->embed_flip ^= 1u;
        if (embeddings) {
            if (!p36_gpu_tensor_write(stage, 0, embeddings,
                    (uint64_t)n_tok * P36_N_EMBD * sizeof(*embeddings))) return false;
        } else if (!p36_gpu_embed_tokens(&e->model, e->weights.token_embd, tokens, n_tok, stage)) {
            fprintf(stderr, "p36: embed failed for %u tokens at pos=%u\n", n_tok, pos0);
            return false;
        }
        if (!p36_gpu_copy_f32_tensor(rt->hidden, stage, n_tok * P36_N_EMBD)) return false;
    }
    /* Every residual add feeds the next RMS norm (post-attention norm, the
     * next layer's attn norm, or the final output norm), so both adds fuse
     * with their norm; only layer 0's input norm stands alone. */
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm, rt->hidden,
                                             e->model.map, e->model.size,
                                             e->weights.layer[0].attn_norm->abs_offset,
                                             P36_N_EMBD, n_tok, P36_RMS_EPS)) {
        return false;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        if (l->kind == P36_LAYER_FULL_ATTN) {
            bool serial_attn = n_tok > 1u && n_tok <= 4u &&
                               getenv("P36_VK_SERIAL_ATTN") != NULL;
            if (serial_attn) {
                for (uint32_t t = 0; t < n_tok; t++) {
                    uint64_t off = (uint64_t)t * P36_N_EMBD * sizeof(float);
                    p36_gpu_tensor *in = p36_gpu_tensor_view(rt->norm, off,
                                                             P36_N_EMBD * sizeof(float));
                    p36_gpu_tensor *out = p36_gpu_tensor_view(rt->next_hidden, off,
                                                              P36_N_EMBD * sizeof(float));
                    bool row_ok = in && out && p36_forward_full_attn_vulkan(
                        s, l, il, pos0 + t, positions ? positions + t : NULL,
                        1, in, out);
                    p36_gpu_tensor_free(in);
                    p36_gpu_tensor_free(out);
                    if (!row_ok) {
                        fprintf(stderr, "p36: attention serial row failed at layer=%u row=%u\n",
                                il, t);
                        return false;
                    }
                    if (p36_vk_dump_layer_hit(il, n_tok, pos0)) {
                        char tag[48];
                        p36_gpu_synchronize();
                        snprintf(tag, sizeof(tag), "attn_qg_row%u", t);
                        p36_vk_layer0_dump(tag, rt->attn_qg,
                            (uint64_t)P36_N_HEAD * P36_N_HEAD_DIM * 2u * sizeof(float));
                        snprintf(tag, sizeof(tag), "attn_q_row%u", t);
                        p36_vk_layer0_dump(tag, rt->attn_q,
                            (uint64_t)P36_N_HEAD * P36_N_HEAD_DIM * sizeof(float));
                        snprintf(tag, sizeof(tag), "attn_out_row%u", t);
                        p36_vk_layer0_dump(tag, rt->attn_out,
                            (uint64_t)P36_N_SSM_INNER * sizeof(float));
                    }
                }
            } else if (!p36_forward_full_attn_vulkan(s, l, il, pos0, positions,
                                                      n_tok, rt->norm, rt->next_hidden)) {
                fprintf(stderr, "p36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
            /* attn_qg aliases FFN scratch for dense layers, so snapshot the
             * attention intermediates before the FFN overwrites them. */
            if (!serial_attn && p36_vk_dump_layer_hit(il, n_tok, pos0)) {
                p36_gpu_synchronize();
                p36_vk_layer0_dump("attn_qg", rt->attn_qg,
                                   (uint64_t)n_tok * P36_N_HEAD *
                                   P36_N_HEAD_DIM * 2u * sizeof(float));
                p36_vk_layer0_dump("attn_q", rt->attn_q,
                                   (uint64_t)n_tok * P36_N_HEAD *
                                   P36_N_HEAD_DIM * sizeof(float));
                p36_vk_layer0_dump("attn_out", rt->attn_out,
                                   (uint64_t)n_tok * P36_N_SSM_INNER * sizeof(float));
            }
        } else {
            bool serial_recur = n_tok > 1u && n_tok <= 4u &&
                                getenv("P36_VK_SERIAL_RECURRENT") != NULL;
            if (serial_recur) {
                /* Diagnostic parity probe: feed the recurrent block one row
                 * at a time so conv/scan state follows the exact serial
                 * order. Attention and FFN stay batched. */
                for (uint32_t t = 0; t < n_tok; t++) {
                    uint64_t off = (uint64_t)t * P36_N_EMBD * sizeof(float);
                    p36_gpu_tensor *in = p36_gpu_tensor_view(rt->norm, off,
                                                             P36_N_EMBD * sizeof(float));
                    p36_gpu_tensor *out = p36_gpu_tensor_view(rt->next_hidden, off,
                                                              P36_N_EMBD * sizeof(float));
                    bool row_ok = in && out &&
                        p36_forward_recurrent_vulkan(s, l, il, 1, in, out);
                    p36_gpu_tensor_free(in);
                    p36_gpu_tensor_free(out);
                    if (!row_ok) {
                        fprintf(stderr, "p36: recurrent serial row failed at layer=%u row=%u\n",
                                il, t);
                        return false;
                    }
                    if (p36_vk_dump_layer_hit(il, n_tok, pos0)) {
                        char tag[48];
                        p36_gpu_synchronize();
                        snprintf(tag, sizeof(tag), "recur_qkv_row%u", t);
                        p36_vk_layer0_dump(tag, rt->recur_qkv,
                            (uint64_t)P36_N_SSM_CONV_DIM * sizeof(float));
                        snprintf(tag, sizeof(tag), "recur_conv_row%u", t);
                        p36_vk_layer0_dump(tag, rt->recur_conv,
                            (uint64_t)P36_N_SSM_CONV_DIM * sizeof(float));
                        snprintf(tag, sizeof(tag), "recur_z_row%u", t);
                        p36_vk_layer0_dump(tag, rt->recur_z,
                            (uint64_t)P36_N_SSM_INNER * sizeof(float));
                    }
                }
            } else if (!p36_forward_recurrent_vulkan(s, l, il, n_tok,
                                                     rt->norm, rt->next_hidden)) {
                fprintf(stderr, "p36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
            if (!serial_recur && rt->recur_qkv && p36_vk_dump_layer_hit(il, n_tok, pos0)) {
                p36_gpu_synchronize();
                p36_vk_layer0_dump("recur_qkv_batch", rt->recur_qkv,
                    (uint64_t)n_tok * P36_N_SSM_CONV_DIM * sizeof(float));
                p36_vk_layer0_dump("recur_conv_batch", rt->recur_conv,
                    (uint64_t)n_tok * P36_N_SSM_CONV_DIM * sizeof(float));
                p36_vk_layer0_dump("recur_z_batch", rt->recur_z,
                    (uint64_t)n_tok * P36_N_SSM_INNER * sizeof(float));
            }
        }
        if (e->directional_steering_attn_scale != 0.0f &&
            !p36_gpu_directional_steering_project_tensor(
                rt->next_hidden,
                e->directional_steering_gpu,
                il,
                P36_N_EMBD,
                n_tok,
                e->directional_steering_attn_scale)) {
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->next_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        bool fused_norm = p36_gpu_add_rms_norm_q8_k_tensor(
            rt->norm, rt->inp_q8, rt->next_hidden, rt->next_hidden, rt->hidden,
            e->model.map, e->model.size, l->post_attention_norm->abs_offset,
            P36_N_EMBD, n_tok, P36_RMS_EPS);
        if (!fused_norm && !p36_gpu_add_rms_norm_tensor(
                rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                e->model.map, e->model.size, l->post_attention_norm->abs_offset,
                P36_N_EMBD, n_tok, P36_RMS_EPS)) {
            return false;
        }
        if (fused_norm) rt->inp_q8_valid = true;
        if (!p36_forward_ffn_vulkan(s, l, il, n_tok, rt->norm, rt->hidden)) {
            fprintf(stderr, "p36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        if (e->directional_steering_ffn_scale != 0.0f &&
            !p36_gpu_directional_steering_project_tensor(
                rt->hidden,
                e->directional_steering_gpu,
                il,
                P36_N_EMBD,
                n_tok,
                e->directional_steering_ffn_scale)) {
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        {
            const p36_tensor *next_norm = il + 1u < P36_N_LAYER ?
                e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            p36_gpu_tensor *target_q8 = (il + 1u < P36_N_LAYER) ? rt->inp_q8 : NULL;
            bool fused_next = target_q8 && p36_gpu_add_rms_norm_q8_k_tensor(
                rt->norm, target_q8, rt->hidden, rt->next_hidden, rt->hidden,
                e->model.map, e->model.size, next_norm->abs_offset,
                P36_N_EMBD, n_tok, P36_RMS_EPS);
            if (!fused_next && !p36_gpu_add_rms_norm_tensor(
                    rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size, next_norm->abs_offset,
                    P36_N_EMBD, n_tok, P36_RMS_EPS)) {
                return false;
            }
            if (fused_next) rt->inp_q8_valid = true;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        if (il == 0u && n_tok > 1u && getenv("P36_VK_Q8_CHECK")) {
            static bool q8_probed;
            if (!q8_probed) {
                q8_probed = true;
                fprintf(stderr, "p36: q8 probe at layer0 rows=%u kind=%u qkv=%u gate=%u\n",
                        n_tok, (unsigned)l->kind,
                        (unsigned)l->attn_qkv->type, (unsigned)l->attn_gate->type);
                p36_vk_layer0_q8_probe(rt, n_tok);
            }
        }
        {
            static bool dumped;
            if (p36_vk_dump_layer_hit(il, n_tok, pos0) && !dumped) {
                const uint64_t act = (uint64_t)n_tok * P36_N_EMBD * sizeof(float);
                const uint64_t ssm = (uint64_t)n_tok * P36_N_SSM_CONV_DIM * sizeof(float);
                const uint64_t q8 = p36_gpu_tensor_bytes(rt->inp_q8);
                dumped = true;
                p36_gpu_synchronize();
                p36_vk_layer0_dump("norm", rt->norm, act);
                p36_vk_layer0_dump("hidden", rt->hidden, act);
                p36_vk_layer0_dump("next_hidden", rt->next_hidden, act);
                p36_vk_layer0_dump("inp_q8", rt->inp_q8, q8);
                p36_vk_layer0_dump("recur_qkv", rt->recur_qkv, ssm);
                p36_vk_layer0_dump("recur_conv", rt->recur_conv, ssm);
                if (!p36_layer_is_full_attention(il)) {
                    p36_vk_layer0_dump("cache_conv", rt->recurrent[il].conv,
                                       p36_gpu_tensor_bytes(rt->recurrent[il].conv));
                    p36_vk_layer0_dump("cache_state", rt->recurrent[il].state,
                                       p36_gpu_tensor_bytes(rt->recurrent[il].state));
                } else {
                    p36_vk_layer0_dump("cache_k", rt->full[il].k,
                                       p36_gpu_tensor_bytes(rt->full[il].k));
                    p36_vk_layer0_dump("cache_v", rt->full[il].v,
                                       p36_gpu_tensor_bytes(rt->full[il].v));
                }
            }
        }
        if (getenv("P36_VK_SWEEP_LAYERS")) {
            const char *sp = getenv("P36_VK_SWEEP_POS");
            uint32_t want_pos = (sp && sp[0]) ? (uint32_t)strtoul(sp, NULL, 10) : 0u;
            if (pos0 == want_pos) {
                const uint64_t act = (uint64_t)n_tok * P36_N_EMBD * sizeof(float);
                char tag[32];
                p36_gpu_synchronize();
                snprintf(tag, sizeof(tag), "L%02u_hidden", il);
                p36_vk_layer0_dump(tag, rt->hidden, act);
                snprintf(tag, sizeof(tag), "L%02u_ffn", il);
                p36_vk_layer0_dump(tag, rt->next_hidden, act);
            }
        }
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->norm, n_tok * P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (logits_out) {
        const float out_scale = p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f);
        bool rot = p36_hadamard_rotate_enabled(e->weights.output);
        bool ok = true;
        p36_gpu_tensor *last = NULL;
        const p36_gpu_tensor *out_in = rt->norm;
        uint32_t rows = n_tok;
        if (!logits_all_rows) {
            last = p36_gpu_tensor_view(rt->norm,
                                       (uint64_t)(n_tok - 1u) * P36_N_EMBD * sizeof(float),
                                       (uint64_t)P36_N_EMBD * sizeof(float));
            out_in = last;
            rows = 1;
        }
        if (rot) {
            ok = out_in && p36_hadamard_prepare(rt, out_in, P36_N_EMBD, rows) &&
                 p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, e->weights.output,
                                                          out_in, rt->had_q8, logits_out,
                                                          P36_N_EMBD, P36_N_VOCAB, rows, out_scale);
        } else {
            ok = out_in && p36_gpu_tensor_matmul_scaled(&e->model, e->weights.output,
                                                        out_in, logits_out,
                                                        P36_N_EMBD, P36_N_VOCAB, rows, out_scale);
        }
        p36_gpu_tensor_free(last);
        if (!ok) {
            fprintf(stderr, "p36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(logits_out, P36_N_VOCAB * (logits_all_rows ? n_tok : 1u))) {
            fprintf(stderr, "p36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
    }
    return true;
}

static bool p36_vulkan_update_last_h(p36_vulkan_runtime *rt, uint32_t row) {
    if (!rt || !rt->last_h) return false;
    return p36_gpu_tensor_copy(rt->last_h,
                               0,
                               rt->norm,
                               (uint64_t)row * P36_N_EMBD * sizeof(float),
                               (uint64_t)P36_N_EMBD * sizeof(float)) != 0;
}

static p36_gpu_tensor *p36_vulkan_row(p36_gpu_tensor *t, uint32_t row,
                                      uint64_t width) {
    uint64_t bytes = width * sizeof(float);
    return p36_gpu_tensor_view(t, (uint64_t)row * bytes, bytes);
}

static bool p36_sessions_full_attn_vulkan(p36_decode_item *items, int count,
                                          const p36_layer_weights *l,
                                          uint32_t il,
                                          p36_gpu_tensor *inp,
                                          p36_gpu_tensor *out) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_q->type != P36_TENSOR_Q8_0 ||
         l->attn_k->type != P36_TENSOR_Q8_0 ||
         l->attn_v->type != P36_TENSOR_Q8_0) &&
        !batch->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, P36_N_EMBD, rows)) {
        return false;
    }
    batch->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_q, inp, batch->inp_q8, batch->attn_qg,
            P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u, rows,
            p36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_k, inp, batch->inp_q8, batch->attn_k,
            P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_v, inp, batch->inp_q8, batch->attn_v,
            P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        p36_vulkan_runtime *rt = s->runtime;
        p36_vulkan_full_attn_cache *cache = &rt->full[il];
        uint32_t pos = (uint32_t)s->checkpoint.len;
        p36_gpu_tensor *qg = p36_vulkan_row(batch->attn_qg, (uint32_t)i,
                P36_N_HEAD * P36_N_HEAD_DIM * 2u);
        p36_gpu_tensor *q = p36_vulkan_row(batch->attn_q, (uint32_t)i,
                P36_N_HEAD * P36_N_HEAD_DIM);
        p36_gpu_tensor *k = p36_vulkan_row(batch->attn_k, (uint32_t)i,
                P36_N_HEAD_KV * P36_N_HEAD_DIM);
        p36_gpu_tensor *v = p36_vulkan_row(batch->attn_v, (uint32_t)i,
                P36_N_HEAD_KV * P36_N_VALUE_DIM);
        p36_gpu_tensor *attn = p36_vulkan_row(batch->attn_out, (uint32_t)i,
                P36_N_SSM_INNER);
        bool ok = qg && q && k && v && attn &&
            p36_vulkan_prepare_attn_heads(q, qg, &e->model, l->attn_q_norm,
                    P36_N_HEAD_DIM * 2u, P36_N_HEAD, pos, 1, NULL, e->quality);
        bool fused_kv = !e->quality && cache->type_k == P36_KV_CACHE_F16 &&
                        cache->type_v == P36_KV_CACHE_F16 &&
                        cache->k_row_bytes == P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(uint16_t);
        const char *fused_env = getenv("P36_VK_NORM_ROPE_KV");
        fused_kv = fused_kv && (!fused_env || !fused_env[0] || fused_env[0] != '0');
        if (ok && fused_kv) {
            ok = p36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, k, v, e->model.map, e->model.size,
                    l->attn_k_norm->abs_offset, P36_N_HEAD_DIM,
                    P36_N_HEAD_KV, pos, 1, cache->cap, P36_RMS_EPS) != 0;
        } else if (ok) {
            ok = p36_vulkan_prepare_attn_heads(k, k, &e->model,
                    l->attn_k_norm, P36_N_HEAD_DIM, P36_N_HEAD_KV,
                    pos, 1, NULL, e->quality) &&
                 p36_gpu_attn_kv_store_tensor(
                    cache->k, cache->v, k, v, pos, 1, cache->cap,
                    P36_N_HEAD_KV * P36_N_HEAD_DIM,
                    P36_N_HEAD_KV * P36_N_VALUE_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        if (ok) {
            ok = p36_gpu_attn_decode_tensor(
                    attn, q, qg, cache->k, cache->v, rt->scores,
                    e->model.map, e->model.size,
                    l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                    l->attn_sinks != NULL, pos, 1,
                    P36_N_HEAD, P36_N_HEAD_KV, P36_N_HEAD_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        p36_gpu_tensor_free(attn);
        p36_gpu_tensor_free(v);
        p36_gpu_tensor_free(k);
        p36_gpu_tensor_free(q);
        p36_gpu_tensor_free(qg);
        if (!ok) return false;
    }

    return p36_gpu_tensor_matmul_scaled(
            &e->model, l->attn_output, batch->attn_out, out,
            P36_N_SSM_INNER, P36_N_EMBD, rows,
            p36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
}

static bool p36_sessions_recurrent_vulkan(p36_decode_item *items, int count,
                                          const p36_layer_weights *l,
                                          uint32_t il,
                                          p36_gpu_tensor *inp,
                                          p36_gpu_tensor *out) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_qkv->type != P36_TENSOR_Q8_0 ||
         l->attn_gate->type != P36_TENSOR_Q8_0) &&
        !batch->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, P36_N_EMBD, rows)) {
        return false;
    }
    batch->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_qkv, inp, batch->inp_q8, batch->recur_qkv,
            P36_N_EMBD, P36_N_SSM_CONV_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_gate, inp, batch->inp_q8, batch->recur_z,
            P36_N_EMBD, P36_N_SSM_INNER, rows,
            p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_dense_q8_scaled(
            &e->model, l->ssm_beta, inp, batch->inp_q8, batch->recur_beta,
            P36_N_EMBD, P36_N_SSM_DT_RANK, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_dense_q8_scaled(
            &e->model, l->ssm_alpha, inp, batch->inp_q8, batch->recur_alpha,
            P36_N_EMBD, P36_N_SSM_DT_RANK, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        p36_vulkan_runtime *rt = items[i].session->runtime;
        p36_vulkan_recurrent_cache *cache = &rt->recurrent[il];
        p36_gpu_tensor *qkv = p36_vulkan_row(batch->recur_qkv, (uint32_t)i,
                P36_N_SSM_CONV_DIM);
        p36_gpu_tensor *conv = p36_vulkan_row(batch->recur_conv, (uint32_t)i,
                P36_N_SSM_CONV_DIM);
        p36_gpu_tensor *q = p36_vulkan_row(batch->recur_q, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *k = p36_vulkan_row(batch->recur_k, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *v = p36_vulkan_row(batch->recur_v, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *gb = p36_vulkan_row(batch->recur_gb, (uint32_t)i,
                P36_N_SSM_DT_RANK * 2u);
        p36_gpu_tensor *alpha = p36_vulkan_row(batch->recur_alpha, (uint32_t)i,
                P36_N_SSM_DT_RANK);
        p36_gpu_tensor *beta = p36_vulkan_row(batch->recur_beta, (uint32_t)i,
                P36_N_SSM_DT_RANK);
        p36_gpu_tensor *proj = p36_vulkan_row(batch->recur_proj, (uint32_t)i,
                P36_N_SSM_INNER);
        bool ok = qkv && conv && q && k && v && gb && alpha && beta && proj;
        if (ok) {
            if (rt->recur_conv_fused) {
                ok = p36_gpu_recurrent_conv_silu_tensor(
                        cache->conv, qkv, conv, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, P36_N_SSM_CONV_DIM,
                        P36_N_SSM_CONV, 1) != 0;
            } else {
                ok = p36_gpu_recurrent_conv_step_tensor(
                        cache->conv, qkv, rt->recur_window, 1) != 0 &&
                     p36_gpu_ssm_conv_silu_tensor(
                        conv, rt->recur_window, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, P36_N_SSM_CONV_DIM,
                        P36_N_SSM_CONV, 1) != 0;
            }
        }
        if (ok && !p36_gpu_delta_qkv_l2_norm_tensor(
                    q, k, v, conv, P36_N_SSM_DT_RANK, P36_N_SSM_GROUP,
                    P36_N_SSM_STATE, P36_N_SSM_CONV_DIM, 1, P36_RMS_EPS)) {
            ok = p36_gpu_delta_qk_l2_norm_tensor(
                    q, k, conv, P36_N_SSM_DT_RANK, P36_N_SSM_GROUP,
                    P36_N_SSM_STATE, P36_N_SSM_CONV_DIM, 1,
                    P36_RMS_EPS) != 0 &&
                 p36_gpu_extract_recurrent_v_tensor(v, conv, 1) != 0;
        }
        if (ok) {
            ok = p36_gpu_delta_net_gates_tensor(
                    gb, alpha, beta, e->model.map, e->model.size,
                    l->ssm_dt->abs_offset, l->ssm_a->abs_offset,
                    P36_N_SSM_DT_RANK, 1) != 0 &&
                 p36_gpu_delta_net_decode_tensor(
                    cache->state, q, k, v, gb, proj,
                    P36_N_SSM_DT_RANK, P36_N_SSM_STATE, 1) != 0;
        }
        p36_gpu_tensor_free(proj);
        p36_gpu_tensor_free(beta);
        p36_gpu_tensor_free(alpha);
        p36_gpu_tensor_free(gb);
        p36_gpu_tensor_free(v);
        p36_gpu_tensor_free(k);
        p36_gpu_tensor_free(q);
        p36_gpu_tensor_free(conv);
        p36_gpu_tensor_free(qkv);
        if (!ok) return false;
    }

    if (!p36_gpu_rms_norm_weight_rows_tensor(
            batch->recur_proj, batch->recur_proj,
            e->model.map, e->model.size, l->ssm_norm->abs_offset,
            P36_N_SSM_STATE, rows * P36_N_SSM_DT_RANK, P36_RMS_EPS) ||
        !p36_gpu_swiglu_tensor(batch->recur_proj, batch->recur_z,
                               batch->recur_proj,
                               rows * P36_N_SSM_INNER, 0.0f, 1.0f)) {
        return false;
    }
    return p36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_out, batch->recur_proj, out,
            P36_N_SSM_INNER, P36_N_EMBD, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
}

static bool p36_sessions_eval_batch_vulkan(p36_decode_item *items, int count) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *rt = first->runtime;
    uint32_t rows = (uint32_t)count;
    int tokens[8];

    for (int i = 0; i < count; i++) tokens[i] = items[i].token;
    p36_gpu_stream_expert_cache_note_tokens(rows);
    if (!p36_gpu_embed_tokens(&e->model, e->weights.token_embd,
                              tokens, rows, rt->embed_stage[0]) ||
        !p36_gpu_copy_f32_tensor(rt->hidden, rt->embed_stage[0],
                                 rows * P36_N_EMBD) ||
        !p36_gpu_rms_norm_weight_rows_tensor(
                rt->norm, rt->hidden, e->model.map, e->model.size,
                e->weights.layer[0].attn_norm->abs_offset,
                P36_N_EMBD, rows, P36_RMS_EPS)) {
        return false;
    }

    p36_gpu_set_micro_batch(true);
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        bool ok = l->kind == P36_LAYER_FULL_ATTN
            ? p36_sessions_full_attn_vulkan(items, count, l, il,
                                            rt->norm, rt->next_hidden)
            : p36_sessions_recurrent_vulkan(items, count, l, il,
                                             rt->norm, rt->next_hidden);
        if (ok && e->directional_steering_attn_scale != 0.0f) {
            ok = p36_gpu_directional_steering_project_tensor(
                    rt->next_hidden, e->directional_steering_gpu, il,
                    P36_N_EMBD, rows,
                    e->directional_steering_attn_scale) != 0;
        }
        if (ok) {
            bool fused_norm = p36_gpu_add_rms_norm_q8_k_tensor(
                    rt->norm, rt->inp_q8, rt->next_hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size,
                    l->post_attention_norm->abs_offset,
                    P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            if (fused_norm) {
                rt->inp_q8_valid = true;
            } else {
                ok = p36_gpu_add_rms_norm_tensor(
                        rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                        e->model.map, e->model.size,
                        l->post_attention_norm->abs_offset,
                        P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            }
        }
        if (ok) {
            ok = p36_forward_ffn_vulkan_model(
                    rt, &e->model, l, e->expert_weights_scale, il,
                    rows, rt->norm, rt->hidden, true);
        }
        if (ok && e->directional_steering_ffn_scale != 0.0f) {
            ok = p36_gpu_directional_steering_project_tensor(
                    rt->hidden, e->directional_steering_gpu, il,
                    P36_N_EMBD, rows,
                    e->directional_steering_ffn_scale) != 0;
        }
        if (ok) {
            const p36_tensor *next_norm = il + 1u < P36_N_LAYER
                ? e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            p36_gpu_tensor *target_q8 = (il + 1u < P36_N_LAYER) ? rt->inp_q8 : NULL;
            bool fused_next = target_q8 && p36_gpu_add_rms_norm_q8_k_tensor(
                    rt->norm, target_q8, rt->hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size, next_norm->abs_offset,
                    P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            if (fused_next) {
                rt->inp_q8_valid = true;
            } else {
                ok = p36_gpu_add_rms_norm_tensor(
                        rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                        e->model.map, e->model.size, next_norm->abs_offset,
                        P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            }
        }
        if (!ok) {
            p36_gpu_set_micro_batch(false);
            return false;
        }
    }
    p36_gpu_set_micro_batch(false);

    if (e->session_batch_logits_cap < rows) {
        p36_gpu_tensor *logits = p36_gpu_tensor_alloc(
                (uint64_t)rows * P36_N_VOCAB * sizeof(float));
        if (!logits) return false;
        p36_gpu_tensor_free(e->session_batch_logits);
        e->session_batch_logits = logits;
        e->session_batch_logits_cap = rows;
    }
    if (!p36_gpu_tensor_matmul_scaled(
            &e->model, e->weights.output, rt->norm,
            e->session_batch_logits, P36_N_EMBD, P36_N_VOCAB, rows,
            p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        p36_vulkan_runtime *dst = items[i].session->runtime;
        if (!p36_gpu_tensor_copy(
                dst->last_h, 0, rt->norm,
                (uint64_t)i * P36_N_EMBD * sizeof(float),
                (uint64_t)P36_N_EMBD * sizeof(float)) ||
            !p36_gpu_tensor_read(
                e->session_batch_logits,
                (uint64_t)i * P36_N_VOCAB * sizeof(float),
                items[i].session->logits,
                (uint64_t)P36_N_VOCAB * sizeof(float))) {
            return false;
        }
        items[i].session->logits_host_valid = true;
        items[i].session->gpu_top2_valid = false;
    }
    return true;
}

/* P36_HOST_PROF: per-token host breakdown of the sampling handoff.  The
 * handoff is where the GPU sits idle with nothing recorded, so the question
 * is always which of these four phases owns the stall. */
static uint64_t p36_hprof_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool p36_forward_tokens_vulkan(p36_session *s, const int *tokens, uint32_t n_tok,
                                      uint32_t pos0, bool compute_logits) {
    static bool hprof_on, hprof_init;
    static uint64_t hprof_prev_exit;
    uint64_t h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0;
    p36_vulkan_runtime *rt;
    if (!hprof_init) {
        hprof_init = true;
        hprof_on = getenv("P36_HOST_PROF") != NULL;
    }
    if (hprof_on) h0 = p36_hprof_ns();
    if (!s) return false;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!p36_forward_tokens_vulkan_into(s,
                                        tokens,
                                        NULL,
                                        NULL,
                                        n_tok,
                                        pos0,
                                        compute_logits ? rt->logits : NULL,
                                        false)) {
        return false;
    }
    if (hprof_on) h1 = p36_hprof_ns();
    if (!p36_vulkan_update_last_h(rt, n_tok - 1u)) return false;
    if (hprof_on) h2 = p36_hprof_ns();
    if (compute_logits) {
        s->gpu_top2_valid = false;
        s->logits_host_valid = false;
        /* Alternate buffers per token: the reader can still be waiting on
         * this result while the next token's dispatch writes the other. */
        p36_gpu_tensor *top2 = rt->top2[rt->top2_flip & 1u];
        s->gpu_top2_slot = (uint8_t)(rt->top2_flip & 1u);
        rt->top2_flip ^= 1u;
        if (p36_gpu_top2_tensor(top2, rt->logits, P36_N_VOCAB)) {
            int32_t ids[2] = {-1, -1};
            if (hprof_on) h3 = p36_hprof_ns();
            if (!p36_gpu_tensor_read(top2, 0, ids, sizeof(ids)))
                return false;
            if (hprof_on) h4 = p36_hprof_ns();
            s->gpu_top2[0] = ids[0];
            s->gpu_top2[1] = ids[1];
            s->gpu_top2_valid = ids[0] >= 0 && ids[1] >= 0;
        } else {
            if (!p36_gpu_tensor_read(
                    rt->logits, 0, s->logits,
                    (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) {
                return false;
            }
            s->logits_host_valid = true;
        }
    }
    if (hprof_on) {
        uint64_t h5 = p36_hprof_ns();
        fprintf(stderr,
                "hprof caller_gap=%.3f forward=%.3f last_h=%.3f top2=%.3f read=%.3f ms\n",
                (double)(h0 - hprof_prev_exit) / 1e6,
                (double)(h1 - h0) / 1e6,
                (double)(h2 - h1) / 1e6,
                (double)(h3 - h2) / 1e6,
                (double)(h4 - h3) / 1e6);
        hprof_prev_exit = h5;
    }
    return true;
}

static bool p36_forward_embeddings_vulkan(p36_session *s, const int *tokens,
                                           const float *embeddings,
                                           const uint32_t *positions,
                                           uint32_t n_tok, uint32_t pos0,
                                           bool compute_logits) {
    p36_vulkan_runtime *rt = s ? s->runtime : NULL;
    if (!rt || !p36_forward_tokens_vulkan_into(
            s, tokens, embeddings, positions, n_tok, pos0,
            compute_logits ? rt->logits : NULL, false) ||
        !p36_vulkan_update_last_h(rt, n_tok - 1u)) return false;
    if (compute_logits) {
        s->gpu_top2_valid = false;
        s->logits_host_valid = false;
    }
    return true;
}

static bool p36_forward_token_vulkan(p36_session *s, int token, uint32_t pos, bool compute_logits) {
    return p36_forward_tokens_vulkan(s, &token, 1, pos, compute_logits);
}

static uint64_t p36_vulkan_recurrent_conv_bytes(void) {
    return (uint64_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float);
}

static bool p36_vulkan_spec_frontier_copy(p36_vulkan_recurrent_cache *dst,
                                          const p36_vulkan_recurrent_cache *src) {
    if (!dst || !src || !dst->conv || !src->conv || !dst->state || !src->state) return false;
    if (!p36_gpu_tensor_copy(dst->conv, 0, src->conv, 0, p36_vulkan_recurrent_conv_bytes())) return false;
    if (p36_gpu_tensor_bytes(dst->state) != p36_gpu_tensor_bytes(src->state) ||
        !p36_gpu_tensor_copy(dst->state, 0, src->state, 0,
                             p36_gpu_tensor_bytes(src->state))) return false;
    return true;
}

static bool p36_vulkan_spec_frontier_snapshot(p36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) continue;
        if (!p36_vulkan_spec_frontier_copy(&rt->spec_recurrent[il], &rt->recurrent[il])) return false;
    }
    return true;
}

static bool p36_vulkan_spec_frontier_restore(p36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) continue;
        if (!p36_vulkan_spec_frontier_copy(&rt->recurrent[il], &rt->spec_recurrent[il])) return false;
    }
    return true;
}

static bool p36_vulkan_verify_suffix_tops(p36_session *s,
                                          const int *tokens,
                                          uint32_t n_tok,
                                          uint32_t pos0,
                                          int *row_tops) {
    p36_vulkan_runtime *rt;
    if (!s || !tokens || n_tok == 0 || n_tok > P36_MTP_MAX_DRAFT) return false;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt || !rt->spec_logits || !rt->spec_tops) return false;
    const bool timed = getenv("P36_LOOKUP_TIME") != NULL;
    struct timespec t_start;
    if (timed) clock_gettime(CLOCK_MONOTONIC, &t_start);
    p36_gpu_set_micro_batch(true);
    {
        /* Project every row through the graph's own output path: it applies
         * the Hadamard rotation of the output weight when the loader pre-
         * rotated it, which a plain matmul on rt->norm would miss. */
        bool fwd = p36_forward_tokens_vulkan_into(s, tokens, NULL, NULL,
                                                  n_tok, pos0, rt->spec_logits, true);
        p36_gpu_set_micro_batch(false);
        if (!fwd) return false;
    }
    /* Top-2 of every row on the device, then one small read: pulling each
     * vocabulary row back on the host drains the queue once per draft and
     * costs more than the drafts can save.  The helper also leaves rt->logits
     * equal to the last row so later lazy host reads stay correct.
     * P36_VK_ARGMAX=0 keeps the plain per-row readback below. */
    if (p36_gpu_spec_verify_tops(rt->logits, rt->spec_tops, rt->spec_logits,
                                 n_tok, P36_N_VOCAB)) {
        int32_t tops[2 * P36_MTP_MAX_DRAFT];
        if (!p36_gpu_tensor_read(rt->spec_tops, 0, tops,
                                 (uint64_t)n_tok * 2u * sizeof(int32_t))) return false;
        if (timed) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            fprintf(stderr, "verify n=%u ms=%.3f\n", n_tok,
                    (now.tv_sec - t_start.tv_sec) * 1e3 +
                    (now.tv_nsec - t_start.tv_nsec) / 1e6);
        }
        for (uint32_t i = 0; i + 1u < n_tok; i++) row_tops[i] = tops[2u * i];
        s->gpu_top2[0] = tops[2u * (n_tok - 1u)];
        s->gpu_top2[1] = tops[2u * (n_tok - 1u) + 1u];
        s->gpu_top2_valid = s->gpu_top2[0] >= 0 && s->gpu_top2[1] >= 0;
        s->logits_host_valid = false;
        return true;
    }
    for (uint32_t i = 0; i + 1u < n_tok; i++) {
        if (!p36_gpu_tensor_read(rt->spec_logits,
                                 (uint64_t)i * P36_N_VOCAB * sizeof(*s->logits),
                                 s->logits,
                                 (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) return false;
        row_tops[i] = sample_argmax(s->logits, P36_N_VOCAB);
    }
    bool ok = p36_gpu_tensor_read(
        rt->spec_logits,
        (uint64_t)(n_tok - 1u) * P36_N_VOCAB * sizeof(*s->logits),
        s->logits, (uint64_t)P36_N_VOCAB * sizeof(*s->logits)) != 0;
    if (ok) {
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return ok;
}

static bool p36_mtp_concat(p36_gpu_tensor *out,
                           const p36_gpu_tensor *a,
                           const p36_gpu_tensor *b) {
    /* Device copies: mapping these GPU-written tensors on the host would
     * drain the whole in-flight draft pipeline. */
    return p36_gpu_tensor_copy(out, 0, a, 0, (uint64_t)P36_N_EMBD * sizeof(float)) &&
           p36_gpu_tensor_copy(out, (uint64_t)P36_N_EMBD * sizeof(float),
                               b, 0, (uint64_t)P36_N_EMBD * sizeof(float));
}

static bool p36_mtp_eval_vulkan(p36_session *s,
                                int token,
                                uint32_t pos,
                                const p36_gpu_tensor *h_in) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    p36_mtp_weights *mw;
    p36_layer_weights *l;
    if (!s || !h_in) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!e || !e->mtp_ready || !rt || !rt->mtp_enabled || !s->mtp_logits) return false;
    mw = &e->mtp_weights;
    l = &mw->block;

    if (!p36_gpu_embed_tokens(&e->model, mw->token_embd, &token, 1, rt->mtp_tok_embd)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_e_norm,
                                             rt->mtp_tok_embd,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->enorm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_h_norm,
                                             h_in,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->hnorm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_mtp_concat(rt->mtp_concat, rt->mtp_e_norm, rt->mtp_h_norm)) return false;
    if (!p36_gpu_tensor_matmul_scaled(&e->mtp_model,
                                      mw->eh_proj,
                                      rt->mtp_concat,
                                      rt->mtp_cur,
                                      P36_N_EMBD * 2u,
                                      P36_N_EMBD,
                                      1,
                                      1.0f)) return false;

    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->attn_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_forward_full_attn_vulkan_model(rt,
                                            &e->mtp_model,
                                            l,
                                            &rt->mtp_full,
                                            e->quality,
                                            P36_N_LAYER,
                                            pos,
                                            (uint32_t)((int32_t)pos + s->rope_delta),
                                            NULL,
                                            1,
                                            rt->norm,
                                            rt->mtp_next)) return false;
    if (!p36_gpu_add_tensor(rt->mtp_next, rt->mtp_next, rt->mtp_cur, P36_N_EMBD)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_next,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->post_attention_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_forward_ffn_vulkan_model(rt,
                                      &e->mtp_model,
                                      l,
                                      e->mtp_expert_weights_scale,
                                      P36_N_LAYER,
                                      1,
                                      rt->norm,
                                      rt->mtp_cur,
                                      false)) return false;
    if (!p36_gpu_add_tensor(rt->mtp_cur, rt->mtp_next, rt->mtp_cur, P36_N_EMBD)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_head,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->shared_head_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_gpu_tensor_matmul_scaled(&e->model,
                                      e->weights.output,
                                      rt->mtp_head,
                                      rt->mtp_logits,
                                      P36_N_EMBD,
                                      P36_N_VOCAB,
                                      1,
                                      p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) return false;
    if (e->quality && !p36_gpu_tensor_all_finite(rt->mtp_logits, P36_N_VOCAB)) return false;
    return p36_gpu_tensor_read(rt->mtp_logits,
                               0,
                               s->mtp_logits,
                               (uint64_t)P36_N_VOCAB * sizeof(*s->mtp_logits)) != 0;
}
#endif

static bool p36_forward_token_cpu(p36_session *s, int token, uint32_t pos, bool compute_logits) {
    p36_engine *e;
    p36_cpu_runtime *rt;
    if (!s || !s->runtime) return false;
    e = s->engine;
    rt = (p36_cpu_runtime *)s->runtime;
    if (!p36_embed_token(e, token, rt->hidden)) {
        fprintf(stderr, "p36: embed failed for token=%d pos=%u\n", token, pos);
        return false;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        p36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + l->attn_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
        if (l->kind == P36_LAYER_FULL_ATTN) {
            if (!p36_forward_full_attn(e, l, il, pos, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "p36: full attention block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        } else {
            if (!p36_forward_recurrent(e, l, il, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "p36: recurrent block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        }
        p36_directional_steering_project_rows(rt->next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_attn_scale);
        if (!p36_all_finite(rt->next_hidden, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at token=%d pos=%u layer=%u kind=%s\n",
                    token, pos, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        for (uint32_t i = 0; i < P36_N_EMBD; i++) rt->next_hidden[i] += rt->hidden[i];
        p36_ref_rms_norm(rt->work0, rt->next_hidden, (const float *)(e->model.map + l->post_attention_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
        if (!p36_forward_ffn(e, l, rt->work0, rt->work1, rt)) {
            fprintf(stderr, "p36: ffn block failed for token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        p36_directional_steering_project_rows(rt->work1,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_ffn_scale);
        if (!p36_all_finite(rt->work1, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        for (uint32_t i = 0; i < P36_N_EMBD; i++) rt->hidden[i] = rt->next_hidden[i] + rt->work1[i];
        if (!p36_all_finite(rt->hidden, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
    }
    p36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + e->weights.output_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
    if (!p36_all_finite(rt->work0, P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at token=%d pos=%u\n", token, pos);
        return false;
    }
    if (compute_logits) {
        const float *out_in = p36_hadamard_forward_rows(e->weights.output, rt->work0, rt->batch_had, P36_N_EMBD, 1u);
        if (!p36_tensor_matvec(e, e->weights.output, out_in, s->logits, rt->work1, P36_N_EMBD, P36_N_VOCAB)) {
            fprintf(stderr, "p36: output projection failed for token=%d pos=%u\n", token, pos);
            return false;
        }
        p36_scale_inplace(s->logits, P36_N_VOCAB,
                          p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!p36_all_finite(s->logits, P36_N_VOCAB)) {
            fprintf(stderr, "p36: non-finite logits at token=%d pos=%u\n", token, pos);
            return false;
        }
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return true;
}

