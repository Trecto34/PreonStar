static void p36_vk_conv_silu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_conv_silu_ctx *ctx = (p36_vk_conv_silu_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *wt = ctx->window + tok * ctx->n_taps * ctx->conv_dim;
        float *ot = ctx->out + tok * ctx->conv_dim;
        for (uint32_t c = 0; c < ctx->conv_dim; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < ctx->n_taps; t++) {
                acc += (double)wt[(uint64_t)t * ctx->conv_dim + c] * (double)ctx->kernel[(uint64_t)c * ctx->n_taps + t];
            }
            {
                float a = (float)acc;
                ot[c] = a / (1.0f + expf(-a));
            }
        }
    }
}

int p36_gpu_ssm_conv_silu_tensor(p36_gpu_tensor *out,
                                 const p36_gpu_tensor *window,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t kernel_offset,
                                 uint32_t conv_dim,
                                 uint32_t n_taps,
                                 uint32_t n_tok) {
    uint64_t dim_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes = dim_bytes * n_taps;
    if (conv_dim == 0 || n_taps == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, dim_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(window, 0, kernel_bytes * n_tok)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        const float *kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
        if (!kernel) return 0;
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *kt = p36_vk_weight_get_unlocked(kernel, kernel_bytes);
        int ok = kt != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { window, kt, out };
            ok = p36_vk_run_unlocked("recurrent_conv_silu", &p36_vk.conv_silu, bindings, &push, sizeof(push),
                                     (conv_dim + 255u) / 256u, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_conv_silu_ctx ctx;
    const float *kernel;
    const float *win;
    float *op;
    kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
    win = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)window, "submit_wait_recur_conv_window");
    op = p36_vk_tensor_contents_labeled(out, "submit_wait_recur_conv_out");
    if (!kernel || !win || !op) return 0;

    ctx.out = op;
    ctx.window = win;
    ctx.kernel = kernel;
    ctx.conv_dim = conv_dim;
    ctx.n_taps = n_taps;
    if ((uint64_t)n_tok * conv_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_conv_silu_rows, &ctx);
    else p36_vk_conv_silu_rows(&ctx, 0, n_tok);
    return 1;
}

/* Delta-net q/k head assembly on the host: group gather plus the
 * p36_l2_norm() float accumulation, like the loop in p36_forward_recurrent().
 * conv_stride is the per-token float stride of the conv tensor (its rows
 * carry the v part after the q/k groups). */
typedef struct {
    float *q;
    float *k;
    const float *conv;
    uint32_t n_heads;
    uint32_t n_groups;
    uint32_t state_dim;
    uint32_t conv_stride;
    uint64_t head_floats;
    float eps;
} p36_vk_delta_qk_ctx;

static void p36_vk_delta_qk_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_delta_qk_ctx *ctx = (p36_vk_delta_qk_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *ct = ctx->conv + tok * ctx->conv_stride;
        float *qt = ctx->q + tok * ctx->head_floats;
        float *kt = ctx->k + tok * ctx->head_floats;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            uint32_t g = h % ctx->n_groups;
            memcpy(qt + (uint64_t)h * ctx->state_dim, ct + (uint64_t)g * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            memcpy(kt + (uint64_t)h * ctx->state_dim, ct + ((uint64_t)ctx->n_groups + g) * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            if (ctx->state_dim == 128u) {
                p36_vk_l2_norm_row_128(qt + (uint64_t)h * 128u, ctx->eps);
                p36_vk_l2_norm_row_128(kt + (uint64_t)h * 128u, ctx->eps);
            } else {
                p36_vk_l2_norm_row(qt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
                p36_vk_l2_norm_row(kt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
            }
        }
    }
}

int p36_gpu_delta_qk_l2_norm_tensor(p36_gpu_tensor *q_out,
                                    p36_gpu_tensor *k_out,
                                    const p36_gpu_tensor *conv,
                                    uint32_t n_heads,
                                    uint32_t n_groups,
                                    uint32_t state_dim,
                                    uint32_t conv_stride,
                                    uint32_t n_tok,
                                    float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_bytes;
    if (n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim) return 0;
    conv_bytes = ((uint64_t)conv_stride * (n_tok - 1u) + 2u * (uint64_t)n_groups * state_dim) * sizeof(float);
    if (!p36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(conv, 0, conv_bytes)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        struct {
            uint32_t n_heads;
            uint32_t n_groups;
            uint32_t state_dim;
            uint32_t conv_stride;
            uint32_t n_tok;
            float eps;
        } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
        const p36_gpu_tensor *bindings[3] = { conv, q_out, k_out };
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_run_unlocked("delta_qk_l2", &p36_vk.delta_qk, bindings, &push, sizeof(push), n_heads, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_delta_qk_ctx ctx;
    const float *cp;
    float *qp;
    float *kp;
    cp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)conv, "submit_wait_delta_qk_conv");
    qp = p36_vk_tensor_contents_labeled(q_out, "submit_wait_delta_qk_q");
    kp = p36_vk_tensor_contents_labeled(k_out, "submit_wait_delta_qk_k");
    if (!cp || !qp || !kp) return 0;

    ctx.q = qp;
    ctx.k = kp;
    ctx.conv = cp;
    ctx.n_heads = n_heads;
    ctx.n_groups = n_groups;
    ctx.state_dim = state_dim;
    ctx.conv_stride = conv_stride;
    ctx.head_floats = head_floats;
    ctx.eps = eps;
    if ((uint64_t)n_tok * n_heads * state_dim >= 16384u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_delta_qk_rows, &ctx);
    else p36_vk_delta_qk_rows(&ctx, 0, n_tok);
    return 1;
}

int p36_gpu_delta_qkv_l2_norm_tensor(p36_gpu_tensor *q_out,
                                     p36_gpu_tensor *k_out,
                                     p36_gpu_tensor *v_out,
                                     const p36_gpu_tensor *conv,
                                     uint32_t n_heads,
                                     uint32_t n_groups,
                                     uint32_t state_dim,
                                     uint32_t conv_stride,
                                     uint32_t n_tok,
                                     float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_floats;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent() ||
        n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim + head_floats) return 0;
    conv_floats = (uint64_t)conv_stride * (n_tok - 1u) +
                  2u * (uint64_t)n_groups * state_dim + head_floats;
    if (!p36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(v_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(conv, 0, conv_floats * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t n_groups;
        uint32_t state_dim;
        uint32_t conv_stride;
        uint32_t n_tok;
        float eps;
    } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
    const p36_gpu_tensor *bindings[4] = { conv, q_out, k_out, v_out };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("delta_qkv_l2", &p36_vk.delta_qkv, bindings,
                                 &push, sizeof(push), n_heads, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Host-side per-head gate prep (64 scalars): decay = expf(softplus(alpha +
 * dt_bias) * a) and beta = sigmoid(beta_raw), exactly the elementwise loop
 * before p36_delta_net_decode() plus its expf, so the shader never needs a
 * GPU exp that would round differently from libm. */
typedef struct {
    float *gb;
    const float *alpha;
    const float *beta;
    const float *dt;
    const float *a;
    uint32_t n_heads;
} p36_vk_delta_gates_ctx;

static void p36_vk_delta_gates_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_delta_gates_ctx *ctx = (p36_vk_delta_gates_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *at = ctx->alpha + tok * ctx->n_heads;
        const float *bt = ctx->beta + tok * ctx->n_heads;
        float *gt = ctx->gb + tok * 2u * ctx->n_heads;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            gt[h] = expf(p36_vk_softplus(at[h] + ctx->dt[h]) * ctx->a[h]);
            gt[ctx->n_heads + h] = p36_vk_sigmoid_stable(bt[h]);
        }
    }
}

int p36_gpu_delta_net_gates_tensor(p36_gpu_tensor *gb,
                                   const p36_gpu_tensor *alpha,
                                   const p36_gpu_tensor *beta,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t dt_bias_offset,
                                   uint64_t a_offset,
                                   uint32_t n_heads,
                                   uint32_t n_tok) {
    uint64_t bytes = (uint64_t)n_heads * sizeof(float);
    if (n_heads == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(gb, 0, 2u * bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(alpha, 0, bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(beta, 0, bytes * n_tok)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        const float *dt = (const float *)p36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
        const float *a = (const float *)p36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
        if (!dt || !a) return 0;
        struct {
            uint32_t n_heads;
            uint32_t n_tok;
        } push = { n_heads, n_tok };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *dt_t = p36_vk_weight_get_unlocked(dt, bytes);
        p36_gpu_tensor *a_t = dt_t ? p36_vk_weight_get_unlocked(a, bytes) : NULL;
        int ok = dt_t != NULL && a_t != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[5] = { alpha, beta, dt_t, a_t, gb };
            ok = p36_vk_run_unlocked("delta_gates", &p36_vk.delta_gates, bindings, &push, sizeof(push),
                                     (n_heads + 63u) / 64u, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_delta_gates_ctx ctx;
    const float *dt;
    const float *a;
    const float *alpha_host;
    const float *beta_host;
    float *gb_host;
    dt = (const float *)p36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
    a = (const float *)p36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
    alpha_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)alpha, "submit_wait_delta_gates_alpha");
    beta_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)beta, "submit_wait_delta_gates_beta");
    gb_host = p36_vk_tensor_contents_labeled(gb, "submit_wait_delta_gates_out");
    if (!dt || !a || !alpha_host || !beta_host || !gb_host) return 0;

    ctx.gb = gb_host;
    ctx.alpha = alpha_host;
    ctx.beta = beta_host;
    ctx.dt = dt;
    ctx.a = a;
    ctx.n_heads = n_heads;
    if ((uint64_t)n_tok * n_heads >= 128u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_delta_gates_rows, &ctx);
    else p36_vk_delta_gates_rows(&ctx, 0, n_tok);
    return 1;
}

/* n_tok sequential state updates in one dispatch: the kernel loops tokens
 * inside the workgroup, preserving the exact per-head chain order, so a
 * prefill batch costs one dispatch instead of n_tok barrier-separated
 * ones (per-dispatch overhead is what bounds this stack). */
int p36_gpu_delta_net_decode_tensor(p36_gpu_tensor *state,
                                    const p36_gpu_tensor *q,
                                    const p36_gpu_tensor *k,
                                    const p36_gpu_tensor *v,
                                    const p36_gpu_tensor *gb,
                                    p36_gpu_tensor *out,
                                    uint32_t n_heads,
                                    uint32_t state_dim,
                                    uint32_t n_tok) {
    uint64_t vec_bytes = (uint64_t)n_heads * state_dim * sizeof(float);
    uint64_t gb_bytes = 2u * (uint64_t)n_heads * sizeof(float);
    uint64_t state_bytes = vec_bytes * state_dim;
    bool state_f16 = state && state->bytes == state_bytes / 2u;
    if (n_heads == 0 || state_dim == 0 || state_dim > 128 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(state, 0, state_f16 ? state_bytes / 2u : state_bytes) ||
        !p36_gpu_tensor_range_ok(q, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(k, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(v, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(gb, 0, gb_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(out, 0, vec_bytes * n_tok)) {
        return 0;
    }

    struct {
        uint32_t n_heads;
        uint32_t state_dim;
        uint32_t n_tok;
    } push = { n_heads, state_dim, n_tok };
    const p36_gpu_tensor *bindings[6] = { state, q, k, v, gb, out };
    p36_vk_kernel *kernel = &p36_vk.delta_net;
    const char *name = "delta_net";
    uint32_t groups_y = 1;
    if (!p36_gpu_quality) {
        if (n_tok >= 32u && state_dim == 128u && p36_vk_use_delta_col_prefill()) {
            /* Four columns share each q/k vector, so a 128-token window stays
             * cache-friendly while avoiding an intermediate state round-trip. */
            int ok = 1;
            pthread_mutex_lock(&p36_vk_mu);
            for (uint32_t t0 = 0; ok && t0 < n_tok; t0 += 128u) {
                struct {
                    uint32_t n_heads;
                    uint32_t state_dim;
                    uint32_t tok0;
                    uint32_t tok_end;
                } cpush = { n_heads, state_dim, t0,
                            t0 + 128u < n_tok ? t0 + 128u : n_tok };
                ok = p36_vk_run_unlocked("delta_net_cols",
                                         state_f16 ? &p36_vk.delta_net_cols_f16 :
                                                     &p36_vk.delta_net_cols,
                                         bindings, &cpush, sizeof(cpush),
                                         n_heads, (state_dim + 3u) / 4u, 1);
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        } else if (state_dim == 128u && p36_vk_use_delta_decode()) {
            bool reg = state_f16 || p36_vk_use_delta_reg();
            kernel = state_f16 ? &p36_vk.delta_net_decode_reg_f16 :
                     reg ? &p36_vk.delta_net_decode_reg :
                           &p36_vk.delta_net_decode;
            name = "delta_net_decode";
            /* COLS for the register kernel: P36_VK_DELTA_COLS selects the
             * .spv variant's column count so the grid matches it.  The default
             * 32 must match -DP36_COLS in the Makefile rule; the env override
             * only works against a .spv built with the same value. */
            uint32_t reg_cols = 32u;
            if (reg) {
                const char *env = getenv("P36_VK_DELTA_COLS");
                if (env && env[0]) {
                    unsigned long v = strtoul(env, NULL, 10);
                    if (v == 8ul || v == 16ul || v == 32ul) reg_cols = (uint32_t)v;
                }
            }
            groups_y = state_dim / (reg ? reg_cols : 32u);
        } else if (p36_vk_use_delta_fast_prefill()) {
            kernel = &p36_vk.delta_net_fast;
            name = "delta_net_fast";
        }
    }
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked(name, kernel, bindings, &push, sizeof(push),
                                 n_heads, groups_y, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* KV cache append, the GPU twin of p36_full_attn_cache_store(): row t of
 * k/v lands at position pos0 + t.  Rows are host-converted when the fast
 * f16 GPU append path is not applicable. */
int p36_gpu_attn_kv_store_tensor(p36_gpu_tensor *k_cache,
                                 p36_gpu_tensor *v_cache,
                                 const p36_gpu_tensor *k,
                                 const p36_gpu_tensor *v,
                                 uint32_t pos0,
                                 uint32_t n_tok,
                                 uint32_t cap,
                                 uint32_t k_row,
                                 uint32_t v_row,
                                 uint32_t k_cache_type,
                                 uint32_t v_cache_type,
                                 uint32_t k_cache_row_bytes,
                                 uint32_t v_cache_row_bytes) {
    const float *k_host;
    const float *v_host;
    unsigned char *kc;
    unsigned char *vc;
    p36_vk_prof_scope prof;
    int ok = 0;
    uint32_t k_expect;
    uint32_t v_expect;
    if (cap == 0 || n_tok == 0 || pos0 >= cap || n_tok > cap - pos0 || k_row == 0 || v_row == 0) return 0;
    if (k_cache_type > 2u || v_cache_type > 2u || k_cache_row_bytes == 0 || v_cache_row_bytes == 0) return 0;
    k_expect = p36_vk_kv_cache_row_bytes(k_cache_type, k_row);
    v_expect = p36_vk_kv_cache_row_bytes(v_cache_type, v_row);
    if (k_cache_row_bytes != k_expect || v_cache_row_bytes != v_expect) return 0;
    if (!p36_gpu_tensor_range_ok(k, 0, (uint64_t)n_tok * k_row * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, (uint64_t)n_tok * v_row * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, (uint64_t)cap * k_cache_row_bytes) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, (uint64_t)cap * v_cache_row_bytes)) {
        return 0;
    }
    /* Fast decode appends KV on GPU; packHalf2x16 rounds to nearest-even,
     * matching P36's host half conversion. */
    if (k_cache_type == 0u && v_cache_type == 0u &&
        k_cache_row_bytes == k_row * sizeof(uint16_t) &&
        v_cache_row_bytes == v_row * sizeof(uint16_t) &&
        !p36_gpu_quality && (k_row % 2u) == 0u && (v_row % 2u) == 0u &&
        p36_vk_env_default_on("P36_VK_GPU_KV_STORE")) {
        struct {
            uint32_t k_row;
            uint32_t v_row;
            uint32_t pos0;
            uint32_t n_tok;
        } push = { k_row, v_row, pos0, n_tok };
        const p36_gpu_tensor *bindings[4] = { k_cache, v_cache, k, v };
        uint32_t row_max = k_row > v_row ? k_row : v_row;
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("attn_kv_store", &p36_vk.kv_store, bindings, &push, sizeof(push),
                                 (row_max / 2u + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    /* GPU quantized KV store: writes Q8_0/Q4_0 blocks directly on the GPU,
     * avoiding the host round-trip that costs ~200 pipeline flushes per run. */
    if (!p36_gpu_quality &&
        (k_row % 32u) == 0u && (v_row % 32u) == 0u &&
        k_row <= 1024u && v_row <= 1024u &&
        k_cache_type >= 1u && k_cache_type <= 2u &&
        v_cache_type >= 1u && v_cache_type <= 2u &&
        p36_vk_env_default_on("P36_VK_GPU_KV_STORE")) {
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } kpush =
            { k_row, k_cache_row_bytes, pos0, n_tok, k_cache_type };
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } vpush =
            { v_row, v_cache_row_bytes, pos0, n_tok, v_cache_type };
        const p36_gpu_tensor *kbind[2] = { k_cache, k };
        const p36_gpu_tensor *vbind[2] = { v_cache, v };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("kv_store_quant_k", &p36_vk.kv_store_quant,
                                 kbind, &kpush, sizeof(kpush), 1, n_tok, 1);
        if (ok) {
            ok = p36_vk_run_unlocked("kv_store_quant_v", &p36_vk.kv_store_quant,
                                     vbind, &vpush, sizeof(vpush), 1, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    p36_vk_prof_scope_begin(&prof, "attn_kv_store_host");
    k_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)k, "submit_wait_attn_kv_k");
    v_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)v, "submit_wait_attn_kv_v");
    kc = p36_vk_tensor_contents_labeled(k_cache, "submit_wait_attn_kv_k_cache");
    vc = p36_vk_tensor_contents_labeled(v_cache, "submit_wait_attn_kv_v_cache");
    if (!k_host || !v_host || !kc || !vc) goto done;

    for (uint32_t t = 0; t < n_tok; t++) {
        uint64_t pos = pos0 + t;
        const float *kt = k_host + (uint64_t)t * k_row;
        const float *vt = v_host + (uint64_t)t * v_row;
        unsigned char *kdst = kc + pos * k_cache_row_bytes;
        unsigned char *vdst = vc + pos * v_cache_row_bytes;
        if (k_cache_type == 0u) {
            uint16_t *h = (uint16_t *)kdst;
            for (uint32_t i = 0; i < k_row; i++) h[i] = p36_quant_f32_to_f16(kt[i]);
        } else if (k_cache_type == 1u) {
            p36_quant_q8_0(kt, kdst, (int64_t)k_row);
        } else {
            p36_quant_q4_0_kv(kt, kdst, (int64_t)k_row);
        }
        if (v_cache_type == 0u) {
            uint16_t *h = (uint16_t *)vdst;
            for (uint32_t i = 0; i < v_row; i++) h[i] = p36_quant_f32_to_f16(vt[i]);
        } else if (v_cache_type == 1u) {
            p36_quant_q8_0(vt, vdst, (int64_t)v_row);
        } else {
            p36_quant_q4_0_kv(vt, vdst, (int64_t)v_row);
        }
    }
    ok = 1;
done:
    p36_vk_prof_scope_end(&prof);
    return ok;
}

/* Full-attention decode: score pass then softmax/weighted-sum/gating pass,
 * mirroring the per-head loop in p36_forward_full_attn().  Causal batch:
 * query row t (absolute position pos0 + t) attends keys 0..pos0+t.  One
 * score dispatch covers the whole chunk (token on workgroup z), the softmax
 * runs on the host for every token after one flush, then one reduce
 * dispatch finishes it, so the cost is one GPU round-trip and two
 * dispatches per batch instead of per token.  scores is caller scratch of
 * n_tok * n_head * (pos0 + n_tok) floats and is clobbered. */
typedef struct {
    float *aux;
    float *scores;
    const float *qg;
    const float *sinks;
    uint64_t score_stride;
    uint64_t qg_stride;
    uint64_t aux_floats;
    uint32_t pos0;
    uint32_t n_head;
    uint32_t head_dim;
    bool has_sinks;
} p36_vk_attn_host_ctx;

