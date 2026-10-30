static void p36_hadamard_fwht_host(float *x, uint32_t n) {
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += 2u * len) {
            for (uint32_t j = 0; j < len; j++) {
                float a = x[i + j];
                float b = x[i + len + j];
                x[i + j] = a + b;
                x[i + len + j] = a - b;
            }
        }
    }
}

/* Inverse-after-lookup for a rotated embedding row: h = s . (H z). */
static void p36_hadamard_inverse_host(float *x, uint32_t width) {
    const float *signs;
    uint32_t block;
    if (!g_p36_hadamard.enabled || width % g_p36_hadamard.block_size != 0) return;
    signs = p36_hadamard_signs(width);
    if (!signs) return;
    block = g_p36_hadamard.block_size;
    for (uint32_t b = 0; b < width; b += block) {
        float *blk = x + b;
        p36_hadamard_fwht_host(blk, block);
        for (uint32_t i = 0; i < block; i++) {
            blk[i] *= g_p36_hadamard.inv_sqrt_block * signs[b + i];
        }
    }
}

/* Forward activation transform for a folded weight: x' = H (s . x) / sqrt(block).
 * Order matters -- signs and scale are applied before the butterfly, matching
 * vulkan/hadamard_prepare.comp. The inverse embedding above applies them after;
 * the two are not interchangeable. */
static void p36_hadamard_forward_host(float *x, uint32_t width) {
    const float *signs;
    uint32_t block;
    if (!g_p36_hadamard.enabled || width % g_p36_hadamard.block_size != 0) return;
    signs = p36_hadamard_signs(width);
    if (!signs) return;
    block = g_p36_hadamard.block_size;
    for (uint32_t b = 0; b < width; b += block) {
        float *blk = x + b;
        for (uint32_t i = 0; i < block; i++)
            blk[i] *= g_p36_hadamard.inv_sqrt_block * signs[b + i];
        p36_hadamard_fwht_host(blk, block);
    }
}

/* Forward transform for the folded GDN ssm_out weight: permute tiled -> grouped
 * first, then the signs+FWHT. Mirrors the is_grouped path of the Vulkan kernel
 * (width = n_ssm_inner, hd = n_ssm_state, nk = n_ssm_group, rep = dt_rank/group). */
static bool p36_hadamard_forward_grouped_host(float *x, uint32_t width) {
    uint32_t hd = P36_N_SSM_STATE, nk = P36_N_SSM_GROUP, rep;
    float *tmp;
    if (!g_p36_hadamard.enabled || hd == 0 || nk == 0 ||
        width % hd != 0 || (width / hd) % nk != 0) return false;
    rep = width / hd / nk;
    tmp = xmalloc((size_t)width * sizeof(float));
    for (uint32_t col = 0; col < width; col++) {
        uint32_t hd_idx = col % hd;
        uint32_t rep_idx = (col / hd) % rep;
        uint32_t nk_idx = col / (hd * rep);
        tmp[col] = x[hd_idx + hd * (nk_idx + nk * rep_idx)];
    }
    memcpy(x, tmp, (size_t)width * sizeof(float));
    free(tmp);
    p36_hadamard_forward_host(x, width);
    return true;
}

/* Copy n_rows activations of `in_dim` into `dst` applying the forward transform
 * required by `t`. Returns dst (or x when no transform applies). */
static const float *p36_hadamard_forward_rows(const p36_tensor *t, const float *x,
                                              float *dst, uint32_t in_dim, uint32_t n_rows) {
    if (!p36_hadamard_rotate_enabled(t)) return x;
    for (uint32_t r = 0; r < n_rows; r++) {
        float *d = dst + (uint64_t)r * in_dim;
        memcpy(d, x + (uint64_t)r * in_dim, (size_t)in_dim * sizeof(float));
        if (t->hadamard_grouped) {
            if (!p36_hadamard_forward_grouped_host(d, in_dim)) return x;
        } else {
            p36_hadamard_forward_host(d, in_dim);
        }
    }
    return dst;
}

static const char *p36_hadamard_str(const p36_str s, char *tmp, size_t tmp_size) {
    if (s.len + 1 > tmp_size) p36_die("prism.hadamard metadata name is too long");
    memcpy(tmp, s.ptr, s.len);
    tmp[s.len] = '\0';
    return tmp;
}

/* Parse the prism.hadamard.* GGUF block that marks weights stored in a
 * rotated (normalized Walsh-Hadamard + explicit sign) basis. The transform is
 * applied to the activation before the matmul, so the flag rides on the tensor
 * and the sign vectors are looked up by activation width. */
static void hadamard_parse(p36_model *m) {
    p36_str transform, axis, sign_mode;
    p36_array_ref widths_arr, values_arr, names_arr;
    p36_cursor c;
    uint32_t block_size = 0;
    uint64_t total = 0;
    char tmp[256];

    if (!model_find_kv(m, "prism.hadamard.weight_names")) return;

    if (!model_get_u32(m, "prism.hadamard.block_size", &block_size) || block_size == 0 ||
        (block_size & (block_size - 1u)) != 0) {
        p36_die("prism.hadamard.block_size is missing or not a power of two");
    }
    transform = required_string(m, "prism.hadamard.transform");
    if (!p36_streq(transform, "normalized-sylvester-walsh-hadamard")) {
        p36_die("unsupported prism.hadamard.transform");
    }
    axis = required_string(m, "prism.hadamard.axis");
    if (!p36_streq(axis, "input-last-dimension")) {
        p36_die("unsupported prism.hadamard.axis");
    }
    sign_mode = required_string(m, "prism.hadamard.sign_mode");
    if (!p36_streq(sign_mode, "explicit")) {
        p36_die("unsupported prism.hadamard.sign_mode");
    }

    g_p36_hadamard.enabled = true;
    g_p36_hadamard.block_size = block_size;
    g_p36_hadamard.inv_sqrt_block = 1.0f / sqrtf((float)block_size);

    if (!model_get_array(m, "prism.hadamard.sign_widths", &widths_arr) ||
        widths_arr.type != GGUF_VALUE_INT32 ||
        !model_get_array(m, "prism.hadamard.sign_values", &values_arr) ||
        values_arr.type != GGUF_VALUE_INT32) {
        p36_die("prism.hadamard explicit signs are missing");
    }
    if (widths_arr.len == 0 || widths_arr.len > P36_HADAMARD_MAX_WIDTHS) {
        p36_die("prism.hadamard.sign_widths has an unsupported length");
    }
    g_p36_hadamard.n_widths = (uint32_t)widths_arr.len;
    c = cursor_at(m, widths_arr.data_pos);
    for (uint64_t i = 0; i < widths_arr.len; i++) {
        int32_t w = 0;
        if (!cursor_read(&c, &w, sizeof(w))) p36_die(c.error);
        if (w <= 0 || (uint32_t)w % block_size != 0) p36_die("prism.hadamard sign width is invalid");
        g_p36_hadamard.widths[i] = (uint32_t)w;
        total += (uint64_t)w;
    }
    if (total != values_arr.len) p36_die("prism.hadamard sign_values length mismatch");
    c = cursor_at(m, values_arr.data_pos);
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        uint32_t w = g_p36_hadamard.widths[i];
        float *buf = (float *)xmalloc((size_t)w * sizeof(float));
        for (uint32_t j = 0; j < w; j++) {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) p36_die(c.error);
            if (v != 1 && v != -1) p36_die("prism.hadamard sign value must be +/-1");
            buf[j] = (float)v;
        }
        g_p36_hadamard.signs[i] = buf;
    }

    if (!model_get_array(m, "prism.hadamard.weight_names", &names_arr) ||
        names_arr.type != GGUF_VALUE_STRING) {
        p36_die("prism.hadamard.weight_names is missing");
    }
    c = cursor_at(m, names_arr.data_pos);
    for (uint64_t i = 0; i < names_arr.len; i++) {
        p36_str name;
        p36_tensor *t;
        if (!cursor_string(&c, &name)) p36_die(c.error);
        t = model_find_tensor(m, p36_hadamard_str(name, tmp, sizeof(tmp)));
        if (!t) {
            fprintf(stderr, "p36: prism.hadamard weight not found: %s\n", tmp);
            exit(1);
        }
        if ((t->dim[0] % block_size) != 0) {
            fprintf(stderr, "p36: prism.hadamard block size does not divide input dim of %s\n", tmp);
            exit(1);
        }
        t->hadamard_rotate = true;
        /* The GDN ssm_out projection kept the training V order
         * (prism.hadamard.gdn_v_grouped); its activation is permuted
         * tiled -> grouped before the transform, exactly as the Vulkan
         * hadamard_prepare is_grouped path does. */
        if (strstr(tmp, "ssm_out") != NULL) t->hadamard_grouped = true;
    }

    if (model_get_array(m, "prism.hadamard.inverse_weight_names", &names_arr) &&
        names_arr.type == GGUF_VALUE_STRING) {
        c = cursor_at(m, names_arr.data_pos);
        for (uint64_t i = 0; i < names_arr.len; i++) {
            p36_str name;
            p36_tensor *t;
            if (!cursor_string(&c, &name)) p36_die(c.error);
            p36_hadamard_str(name, tmp, sizeof(tmp));
            if (strcmp(tmp, "token_embd.weight") != 0) {
                fprintf(stderr, "p36: unsupported prism.hadamard inverse weight: %s\n", tmp);
                exit(1);
            }
            t = model_find_tensor(m, tmp);
            if (!t) {
                fprintf(stderr, "p36: prism.hadamard inverse weight not found: %s\n", tmp);
                exit(1);
            }
            t->hadamard_inverse = true;
        }
    }

    fprintf(stderr, "p36: prism.hadamard: block=%u widths=%u values=%" PRIu64 "\n",
            block_size, g_p36_hadamard.n_widths, total);
}

static void config_validate_model(const p36_model *m) {
    char key[128];
    const char *prefix;
    uint64_t ctx_train = 0;
    uint32_t block_count;
    uint32_t nextn_layers = 0;
    bool embedded_nextn_block;
    p36_str arch = required_string(m, "general.architecture");
    p36_str tok_model = required_string(m, "tokenizer.ggml.model");
    p36_str tok_pre = required_string(m, "tokenizer.ggml.pre");
    if (p36_streq(arch, P36_SHAPE_35B_A3B.arch)) {
        g_p36_shape = P36_SHAPE_35B_A3B;
    } else if (p36_streq(arch, P36_SHAPE_CLOVER_35B_A3B.arch)) {
        g_p36_shape = P36_SHAPE_CLOVER_35B_A3B;
    } else if (p36_streq(arch, P36_SHAPE_27B.arch)) {
        g_p36_shape = P36_SHAPE_27B;
    } else {
        fprintf(stderr, "p36: unsupported architecture: %.*s\n",
                (int)arch.len, arch.ptr);
        exit(1);
    }
    const p36_tensor *embd = model_find_tensor(m, "token_embd.weight");
    if (embd && embd->ndim == 2 && embd->dim[1] > 0) {
        g_p36_shape.n_vocab = (uint32_t)embd->dim[1];
    }
    prefix = g_p36_shape.arch;
    config_expect_string("general.architecture", arch, prefix);
    config_expect_string("tokenizer.ggml.model", tok_model, "gpt2");
    /* Some converters label the identical Qwen3.5 vocab/merges "qwen2"; the
     * engine's pre-tokenizer is the qwen35 one either way. */
    if (!p36_streq(tok_pre, "qwen2"))
        config_expect_string("tokenizer.ggml.pre", tok_pre, "qwen35");
#define P36_META_KEY(suffix) snprintf(key, sizeof(key), "%s.%s", prefix, suffix)
#define P36_EXPECT_U32(suffix, expected) do { \
    P36_META_KEY(suffix); \
    config_expect_u32(key, required_u32(m, key), expected); \
} while (0)
#define P36_EXPECT_F32(suffix, expected) do { \
    P36_META_KEY(suffix); \
    config_expect_f32(key, required_f32(m, key), expected); \
} while (0)
    P36_META_KEY("block_count");
    block_count = required_u32(m, key);
    P36_META_KEY("nextn_predict_layers");
    model_get_u32(m, key, &nextn_layers);
    embedded_nextn_block = nextn_layers == 1 && block_count == P36_N_LAYER + 1;
    if (block_count != P36_N_LAYER && !embedded_nextn_block) {
        P36_META_KEY("block_count");
        config_expect_u32(key, block_count, P36_N_LAYER);
    }
    P36_META_KEY("context_length");
    if (!model_get_u64_compat(m, key, &ctx_train)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    if (ctx_train != P36_CONTEXT_TRAIN) {
        fprintf(stderr, "p36: expected %s=%u, got %" PRIu64 "\n",
                key, P36_CONTEXT_TRAIN, ctx_train);
        exit(1);
    }
    P36_EXPECT_U32("embedding_length", P36_N_EMBD);
    P36_EXPECT_U32("attention.head_count", P36_N_HEAD);
    P36_EXPECT_U32("attention.head_count_kv", P36_N_HEAD_KV);
    P36_EXPECT_U32("attention.key_length", P36_N_HEAD_DIM);
    P36_EXPECT_U32("attention.value_length", P36_N_VALUE_DIM);
    P36_EXPECT_U32("rope.dimension_count", P36_N_ROT);
    P36_EXPECT_F32("rope.freq_base", 10000000.0f);
    if (g_p36_shape.variant != P36_VARIANT_CLOVER_35B_A3B) {
        P36_META_KEY("rope.dimension_sections");
        config_expect_u32_array(m, key, P36_ROPE_SECTIONS, 4);
    }
    P36_EXPECT_F32("attention.layer_norm_rms_epsilon", P36_RMS_EPS);
    if (P36_MODEL_DENSE) {
        P36_EXPECT_U32("feed_forward_length", P36_N_FF_SHARED);
    } else {
        P36_EXPECT_U32("expert_count", P36_N_EXPERT);
        P36_EXPECT_U32("expert_used_count", P36_N_EXPERT_USED);
        P36_EXPECT_U32("expert_feed_forward_length", P36_N_FF_EXP);
        P36_EXPECT_U32("expert_shared_feed_forward_length", P36_N_FF_SHARED);
    }
    P36_EXPECT_U32("ssm.conv_kernel", P36_N_SSM_CONV);
    P36_EXPECT_U32("ssm.state_size", P36_N_SSM_STATE);
    P36_EXPECT_U32("ssm.group_count", P36_N_SSM_GROUP);
    P36_EXPECT_U32("ssm.time_step_rank", P36_N_SSM_DT_RANK);
    P36_EXPECT_U32("ssm.inner_size", P36_N_SSM_INNER);
    P36_EXPECT_U32("full_attention_interval", P36_FULL_ATTENTION_INTERVAL);
#undef P36_EXPECT_F32
#undef P36_EXPECT_U32
#undef P36_META_KEY
    if (m->n_tensors != P36_TENSOR_COUNT &&
        !(embedded_nextn_block &&
          m->n_tensors == P36_TENSOR_COUNT + (P36_MODEL_DENSE ? 15u : 20u))) {
        fprintf(stderr, "p36: expected %u tensors, got %" PRIu64 "\n", P36_TENSOR_COUNT, m->n_tensors);
        exit(1);
    }
}

static void print_size(uint64_t bytes) {
    printf("%.2f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static void model_summary(const p36_model *m) {
    p36_str name = {0};
    p36_str arch = {0};
    uint64_t ctx_train = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;
    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    if (!model_get_u64_compat(m, "qwen35moe.context_length", &ctx_train) &&
        !model_get_u64_compat(m, "clover_moe.context_length", &ctx_train))
        model_get_u64_compat(m, "qwen35.context_length", &ctx_train);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }
    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n", m->version, m->n_kv, m->n_tensors);
    printf("layers: %u\n", P36_N_LAYER);
    printf("train context: %" PRIu64 "\n", ctx_train);
    printf("attention: heads=%u kv_heads=%u head_dim=%u full_interval=%u\n",
           P36_N_HEAD, P36_N_HEAD_KV, P36_N_HEAD_DIM, P36_FULL_ATTENTION_INTERVAL);
    if (P36_MODEL_DENSE) printf("ffn: dense, width=%u\n", P36_N_FF_SHARED);
    else printf("experts: count=%u used=%u ff=%u shared_ff=%u\n",
                P36_N_EXPERT, P36_N_EXPERT_USED, P36_N_FF_EXP, P36_N_FF_SHARED);
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);
}

enum { P36_KV_INITIAL_CAP = 32768 };

static uint32_t p36_kv_initial_cap(int ctx_size) {
    return (uint32_t)(ctx_size < P36_KV_INITIAL_CAP ? ctx_size : P36_KV_INITIAL_CAP);
}

static uint32_t p36_kv_next_cap(uint32_t cap, uint32_t need, uint32_t limit) {
    uint32_t next = cap;
    while (next < need) {
        if (next >= limit) return 0;
        uint32_t step = next / 4u;
        uint32_t grown = step <= limit - next ? next + step : limit;
        if (grown <= next) return 0;
        next = grown;
    }
    return next;
}

static p36_cpu_runtime *p36_cpu_runtime_create(int ctx_size,
                                               uint32_t prefill_cap,
                                               p36_kv_cache_type cache_type_k,
                                               p36_kv_cache_type cache_type_v) {
    p36_cpu_runtime *rt;
    uint32_t full_cap;
    uint64_t state_dim;
    size_t scratch_bytes;
    if (ctx_size <= 0) return NULL;
    if (prefill_cap < 1) prefill_cap = 1;
    uint32_t k_row_bytes = p36_kv_cache_row_bytes(cache_type_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    uint32_t v_row_bytes = p36_kv_cache_row_bytes(cache_type_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    uint32_t had_width;
    if (!k_row_bytes || !v_row_bytes) return NULL;
    had_width = P36_N_EMBD;
    if (P36_N_FF_SHARED > had_width) had_width = P36_N_FF_SHARED;
    if (P36_N_SSM_INNER > had_width) had_width = P36_N_SSM_INNER;
    rt = xcalloc(1, sizeof(*rt));
    full_cap = p36_kv_initial_cap(ctx_size);
    state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    scratch_bytes = (size_t)P36_CPU_SCRATCH_FLOATS * sizeof(float);
    rt->prefill_cap = prefill_cap;
    rt->hidden = xmalloc((size_t)P36_N_EMBD * sizeof(float));
    rt->next_hidden = xmalloc((size_t)P36_N_EMBD * sizeof(float));
    rt->work0 = xmalloc(scratch_bytes);
    rt->work1 = xmalloc(scratch_bytes);
    rt->work2 = xmalloc(scratch_bytes);
    rt->work3 = xmalloc(scratch_bytes);
    rt->work4 = xmalloc(scratch_bytes);
    rt->work5 = xmalloc(scratch_bytes);
    rt->scores = xmalloc((size_t)full_cap * sizeof(float));
    rt->batch_hidden = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_next_hidden = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_norm = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_qg = xmalloc((size_t)prefill_cap * P36_N_HEAD * P36_N_HEAD_DIM * 2u * sizeof(float));
    rt->batch_q = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_k = xmalloc((size_t)prefill_cap * P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(float));
    rt->batch_v = xmalloc((size_t)prefill_cap * P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(float));
    rt->batch_attn_out = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_qkv = xmalloc((size_t)prefill_cap * P36_N_SSM_CONV_DIM * sizeof(float));
    rt->batch_recur_z = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_alpha = xmalloc((size_t)prefill_cap * P36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_beta = xmalloc((size_t)prefill_cap * P36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_proj = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_ffn_gate_logits = xmalloc((size_t)prefill_cap * P36_N_EXPERT * sizeof(float));
    rt->batch_ffn_shared_gate = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_up = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_mid = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_out = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_ffn_scalar = xmalloc((size_t)prefill_cap * sizeof(float));
    rt->batch_xq = xmalloc((size_t)prefill_cap * P36_MAX_Q8_K_BYTES);
    rt->batch_had = xmalloc((size_t)prefill_cap * had_width * sizeof(float));
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            rt->full[il].cap = full_cap;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = xmalloc((size_t)full_cap * k_row_bytes);
            rt->full[il].v = xmalloc((size_t)full_cap * v_row_bytes);
        } else {
            rt->recurrent[il].conv = xmalloc_zeroed((size_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM, sizeof(float));
            rt->recurrent[il].state = xmalloc_zeroed((size_t)state_dim, sizeof(float));
        }
    }
    return rt;
}

static bool p36_cpu_runtime_reserve_kv(p36_cpu_runtime *rt, uint32_t need,
                                       uint32_t limit) {
    if (!rt || need > limit) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_full_attn_cache *c;
        uint32_t cap;
        uint8_t *k, *v;
        if (!p36_layer_is_full_attention(il)) continue;
        c = &rt->full[il];
        if (need <= c->cap) continue;
        cap = p36_kv_next_cap(c->cap, need, limit);
        if (!cap) return false;
        k = xmalloc((size_t)cap * c->k_row_bytes);
        v = xmalloc((size_t)cap * c->v_row_bytes);
        memcpy(k, c->k, (size_t)c->len * c->k_row_bytes);
        memcpy(v, c->v, (size_t)c->len * c->v_row_bytes);
        free(c->k);
        free(c->v);
        c->k = k;
        c->v = v;
        c->cap = cap;
    }
    return true;
}

static void p36_cpu_runtime_reset(p36_cpu_runtime *rt) {
    uint64_t state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    if (!rt) return;
    memset(rt->hidden, 0, (size_t)P36_N_EMBD * sizeof(float));
    memset(rt->next_hidden, 0, (size_t)P36_N_EMBD * sizeof(float));
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        rt->full[il].len = 0;
        if (rt->recurrent[il].conv) {
            memset(rt->recurrent[il].conv, 0, (size_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float));
        }
        if (rt->recurrent[il].state) {
            memset(rt->recurrent[il].state, 0, (size_t)state_dim * sizeof(float));
        }
    }
}

