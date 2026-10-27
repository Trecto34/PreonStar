int p36_gpu_v_grouped_permute_tensor(p36_gpu_tensor *dst,
                                     const p36_gpu_tensor *src,
                                     uint32_t width,
                                     uint32_t hd,
                                     uint32_t nk,
                                     uint32_t rep,
                                     uint32_t n_rows) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    if (!dst || !src || width == 0 || n_rows == 0) return 0;
    if (hd == 0 || nk == 0 || rep == 0) return 0;
    if ((uint64_t)hd * nk * rep != width) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(dst, 0, act_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t hd;
        uint32_t nk;
        uint32_t rep;
        uint32_t n_rows;
    } push = { width, hd, nk, rep, n_rows };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("v_grouped_permute", &p36_vk.v_grouped_permute, bindings,
                                 &push, sizeof(push), (width + 255u) / 256u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_hadamard_prepare_tensor(p36_gpu_tensor *q8_dst,
                                    const p36_gpu_tensor *src,
                                    const p36_gpu_tensor *signs,
                                    uint32_t width,
                                    uint32_t n_rows,
                                    uint32_t sign_offset,
                                    uint32_t total_signs,
                                    float scale,
                                    uint32_t is_grouped) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    uint64_t sign_bytes = 0;
    uint64_t q8_bytes = 0;
    if (!q8_dst || !src || !signs || width == 0 || n_rows == 0) return 0;
    if ((width % 1024u) != 0 || width > UINT32_MAX || n_rows > UINT32_MAX) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes) ||
        !p36_u64_mul_ok(total_signs, sizeof(float), &sign_bytes)) {
        return 0;
    }
    if (sign_offset > total_signs || (uint64_t)sign_offset + width > total_signs) return 0;
    const uint64_t blocks = width / P36_VK_QK_K;
    if (!p36_u64_mul_ok(n_rows, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(signs, 0, sign_bytes) ||
        !p36_gpu_tensor_range_ok(q8_dst, 0, q8_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t sign_offset;
        float    scale;
        uint32_t n_rows;
        uint32_t blocks;
        uint32_t is_grouped;
    } push = { width, sign_offset, scale, n_rows, (uint32_t)blocks, is_grouped };
    const p36_gpu_tensor *bindings[3] = { src, signs, q8_dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("hadamard_prepare", &p36_vk.hadamard_prepare, bindings,
                                 &push, sizeof(push), width / 1024u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_add_tensor(p36_gpu_tensor *out,
                       const p36_gpu_tensor *a,
                       const p36_gpu_tensor *b,
                       uint32_t n) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) ||
        !p36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }

    struct {
        uint32_t n;
    } push = { n };

    const p36_gpu_tensor *bindings[3] = { a, b, out };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("residual_add",
                                 &p36_vk.add,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (n + 255u) / 256u,
                                 1,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_directional_steering_project_tensor(
        p36_gpu_tensor       *x,
        const p36_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    uint64_t x_elems;
    uint64_t dir_elems;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    if (!p36_u64_mul_ok(width, rows, &x_elems) ||
        !p36_u64_mul_ok((uint64_t)layer + 1u, width, &dir_elems) ||
        !p36_gpu_tensor_range_ok(x, 0, x_elems * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(directions, 0, dir_elems * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t layer;
        uint32_t width;
        uint32_t rows;
        float scale;
    } push = { layer, width, rows, scale };
    const p36_gpu_tensor *bindings[2] = { x, directions };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("directional_steering",
                                 &p36_vk.directional_steering,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 rows,
                                 1,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Fused residual add + weighted RMS norm: out_sum = a + b (the new residual)
 * and out_norm = rmsnorm(out_sum) * w in one dispatch.  Bit-identical to
 * p36_gpu_add_tensor() + p36_gpu_rms_norm_weight_rows_tensor(), which stays
 * the fallback for quality mode and P36_VK_ADD_RMS=0.  out_sum may alias a
 * or b. */
int p36_gpu_add_rms_norm_tensor(p36_gpu_tensor *out_norm,
                                p36_gpu_tensor *out_sum,
                                const p36_gpu_tensor *a,
                                const p36_gpu_tensor *b,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint32_t n,
                                uint32_t rows,
                                float eps) {
    if (n == 0 || rows == 0) return 0;
    if (p36_gpu_quality || !p36_vk_use_add_rms() || !p36_vk_use_gpu_rms() || n > 5120u) {
        return p36_gpu_add_tensor(out_sum, a, b, n * rows) &&
               p36_gpu_rms_norm_weight_rows_tensor(out_norm, out_sum,
                                                   model_map, model_size, weight_offset,
                                                   n, rows, eps);
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out_norm, 0, bytes) || !p36_gpu_tensor_range_ok(out_sum, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) || !p36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;

    struct {
        uint32_t n;
        float eps;
    } push = { n, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { a, b, wt, out_sum, out_norm };
        ok = p36_vk_run_unlocked("add_rms_norm", &p36_vk.add_rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static bool p36_vk_use_fused_rms_q8(void) {
    /* Add + RMS norm + q8_K quantization in one dispatch.  The fused shader
     * writes the same bytes the standalone quantizer did (md5 55f55566... on
     * the 64-token probe) and removes ~36 dispatches per decode token, most
     * of which were the norm's own pass.  It was parked behind an env switch
     * while it was unmeasured; unconditional now. */
    return true;
}

int p36_gpu_add_rms_norm_q8_k_rot_tensor(p36_gpu_tensor *out_norm,
                                         p36_gpu_tensor *out_q8,
                                         p36_gpu_tensor *out_had_q8,
                                         p36_gpu_tensor *out_sum,
                                         const p36_gpu_tensor *a,
                                         const p36_gpu_tensor *b,
                                         const p36_gpu_tensor *signs,
                                         uint32_t sign_offset,
                                         float rot_scale,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t weight_offset,
                                         uint32_t n,
                                         uint32_t rows,
                                         float eps) {
    if (!out_q8 || p36_gpu_quality || !p36_vk_use_add_rms() || !p36_vk_use_gpu_rms() ||
        !p36_vk_use_fused_rms_q8() ||
        (n & (P36_VK_QK_K - 1u)) != 0 || n > 5120u) {
        return 0;
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    uint32_t blocks = n / P36_VK_QK_K;
    uint64_t q8_bytes = (uint64_t)rows * blocks * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(out_norm, 0, bytes) || !p36_gpu_tensor_range_ok(out_sum, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) || !p36_gpu_tensor_range_ok(b, 0, bytes) ||
        !p36_gpu_tensor_range_ok(out_q8, 0, q8_bytes)) {
        return 0;
    }
    /* The rotated path needs both the sign table and somewhere to put the
     * Q8_K of the transformed row; without them the epilogue is skipped. */
    uint32_t rot = out_had_q8 && signs && (n & 1023u) == 0u ? 1u : 0u;
    if (rot && !p36_gpu_tensor_range_ok(out_had_q8, 0, q8_bytes)) return 0;
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;

    struct {
        uint32_t n;
        float eps;
        uint32_t blocks;
        uint32_t rot;
        uint32_t sign_offset;
        float inv_sqrt_block;
    } push = { n, eps, blocks, rot, sign_offset, rot_scale };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        /* rot == 0 still has to supply in-range descriptors for 6 and 7. */
        const p36_gpu_tensor *bindings[8] = { a, b, wt, out_sum, out_norm, out_q8,
                                              rot ? signs : out_norm,
                                              rot ? out_had_q8 : out_q8 };
        ok = p36_vk_run_unlocked("add_rms_norm_q8_k", &p36_vk.add_rms_norm_q8_k, bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_add_rms_norm_q8_k_tensor(p36_gpu_tensor *out_norm,
                                     p36_gpu_tensor *out_q8,
                                     p36_gpu_tensor *out_sum,
                                     const p36_gpu_tensor *a,
                                     const p36_gpu_tensor *b,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint32_t n,
                                     uint32_t rows,
                                     float eps) {
    return p36_gpu_add_rms_norm_q8_k_rot_tensor(out_norm, out_q8, NULL, out_sum, a, b, NULL, 0u,
                                                0.0f, model_map, model_size, weight_offset, n,
                                                rows, eps);
}







/* ===========================================================================
 * Qwen3.6 decode kernels.
 * ===========================================================================
 *
 * Faithful GPU ports of the CPU reference path in p36.c.  Each wrapper mirrors
 * one step of p36_forward_token_cpu(): the multi-section RoPE, the full
 * attention score/softmax/weighted-sum with sinks and output gating, the
 * delta-net recurrent decode, the SSM conv window, and the routed MoE with the
 * model's per-tensor/per-expert .scale factors.
 */

/* Codebook grids for the IQ expert quants, uploaded once and bound to the
 * moe_matvec kernel.  Layout must match the TAB_* offsets in the shader. */
static p36_gpu_tensor *p36_vk_iq_tables_unlocked(void) {
    if (p36_vk.iq_tables) return p36_vk.iq_tables;
    p36_gpu_tensor *t = p36_vk_tensor_alloc_kind_unlocked(P36_VK_TAB_TOTAL_BYTES, true, true);
    if (!t) return NULL;
    p36_vk_pack_iq_tables(t->data);
    p36_vk.iq_tables = t;
    return t;
}

static float p36_vk_softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

/* In-place row L2 norm on the host: p36_l2_norm()'s float accumulation. */
static void p36_vk_l2_norm_row(float *x, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

/* Fixed-size clone, same reasoning as p36_vk_rms_norm_row_128(). */
static void p36_vk_l2_norm_row_128(float *x, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < 128u; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < 128u; i++) x[i] *= scale;
    }
}

int p36_gpu_l2_norm_rows_tensor(p36_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    uint64_t elems = 0;
    uint64_t bytes = 0;
    float *xp;
    if (n == 0 || rows == 0) return 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(x, 0, bytes)) return 0;
    xp = p36_vk_tensor_contents_labeled(x, "submit_wait_l2_norm_x");
    if (!xp) return 0;
    for (uint32_t r = 0; r < rows; r++) {
        if (n == 128u) p36_vk_l2_norm_row_128(xp + (uint64_t)r * 128u, eps);
        else p36_vk_l2_norm_row(xp + (uint64_t)r * n, n, eps);
    }
    return 1;
}

int p36_gpu_recurrent_conv_step_tensor(p36_gpu_tensor *cache_conv,
                                       const p36_gpu_tensor *cur,
                                       p36_gpu_tensor *window,
                                       uint32_t n_tok) {
    const uint32_t hist = P36_VK_N_SSM_CONV - 1u;
    uint64_t row_bytes = (uint64_t)P36_VK_N_SSM_CONV_DIM * sizeof(float);
    if (n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(cache_conv, 0, hist * row_bytes) ||
        !p36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !p36_gpu_tensor_range_ok(window, 0, (uint64_t)n_tok * P36_VK_N_SSM_CONV * row_bytes)) {
        return 0;
    }
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent()) {
        float *cachep = p36_vk_tensor_contents_labeled(cache_conv, "submit_wait_recur_window_cache");
        float *windowp = p36_vk_tensor_contents_labeled(window, "submit_wait_recur_window_out");
        float *curp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)cur, "submit_wait_recur_window_cur");
        if (!cachep || !windowp || !curp) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float *wt = windowp + (uint64_t)t * P36_VK_N_SSM_CONV * P36_VK_N_SSM_CONV_DIM;
            const float *ct = curp + (uint64_t)t * P36_VK_N_SSM_CONV_DIM;
            for (uint32_t i = 0; i < hist; i++) memcpy(wt + (uint64_t)i * P36_VK_N_SSM_CONV_DIM, cachep + (uint64_t)i * P36_VK_N_SSM_CONV_DIM, (size_t)row_bytes);
            memcpy(wt + (uint64_t)hist * P36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
            if (hist > 1u) memmove(cachep, cachep + P36_VK_N_SSM_CONV_DIM, (size_t)(hist - 1u) * (size_t)row_bytes);
            memcpy(cachep + (uint64_t)(hist - 1u) * P36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
        }
        return 1;
    }
    struct {
        uint32_t conv_dim;
        uint32_t hist;
        uint32_t n_tok;
    } push = { P36_VK_N_SSM_CONV_DIM, hist, n_tok };
    const p36_gpu_tensor *bindings[3] = { cache_conv, cur, window };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("recurrent_window", &p36_vk.recur_window, bindings, &push, sizeof(push),
                                 (P36_VK_N_SSM_CONV_DIM + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_gdn_front_tensor(p36_gpu_tensor *history,
    const p36_gpu_tensor *cur, const p36_gpu_tensor *x,
    p36_gpu_tensor *q, p36_gpu_tensor *k, p36_gpu_tensor *v, p36_gpu_tensor *gb,
    const void *map, uint64_t size, const uint64_t offsets[5],
    uint32_t width, uint32_t groups, uint32_t heads,
    float alpha_scale, float beta_scale, float eps) {
    if (!p36_gpu_dense_model || !p36_vk_env_default_on("P36_VK_GDN_FRONT") ||
        p36_gpu_quality || !p36_vk_use_gpu_recurrent() ||
        !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        p36_vk.props.limits.maxPerStageDescriptorStorageBuffers<12 ||
        p36_vk.props.limits.maxDescriptorSetStorageBuffers<12 ||
        p36_vk.props.limits.maxComputeWorkGroupInvocations<384 ||
        p36_vk.props.limits.maxComputeWorkGroupSize[0]<384 ||
        groups!=16 || heads!=48 || p36_vk.subgroup_size!=64 || !width || width%32) return 0;
    uint64_t channels=(2u*(uint64_t)groups+heads)*128u;
    uint64_t bytes[5]={channels*16, (uint64_t)width/32*34*heads,
                      (uint64_t)width/32*34*heads, heads*4u, heads*4u};
    if (!p36_gpu_tensor_range_ok(history,0,channels*12) ||
        !p36_gpu_tensor_range_ok(cur,0,channels*4) ||
        !p36_gpu_tensor_range_ok(x,0,(uint64_t)width*4) ||
        !p36_gpu_tensor_range_ok(q,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(k,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(v,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(gb,0,(uint64_t)heads*8)) return 0;
    const void *w[5];
    for (unsigned i=0;i<5;i++) {
        w[i]=p36_gpu_weight_bytes(map,size,offsets[i],bytes[i]);
        if (!w[i]) return 0;
    }
    struct { uint32_t groups,heads,blocks; float alpha_scale,beta_scale,eps; }
        push={groups,heads,width/32,alpha_scale,beta_scale,eps};
    pthread_mutex_lock(&p36_vk_mu);
    const p36_gpu_tensor *b[12]={history,cur,NULL,x,NULL,NULL,NULL,NULL,q,k,v,gb};
    const unsigned slots[5]={2,4,5,6,7};
    int ok=1;
    for (unsigned i=0;i<5;i++) {
        b[slots[i]]=p36_vk_weight_get_unlocked(w[i],bytes[i]);
        if (!b[slots[i]]) { ok=0; break; }
    }
    if (ok) ok=p36_vk_run_unlocked("gdn_front",&p36_gdn_front,b,&push,sizeof(push),groups,1,1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_recurrent_conv_silu_tensor(p36_gpu_tensor *cache_conv,
                                       const p36_gpu_tensor *cur,
                                       p36_gpu_tensor *out,
                                       const void *model_map,
                                       uint64_t model_size,
                                       uint64_t kernel_offset,
                                       uint32_t conv_dim,
                                       uint32_t n_taps,
                                       uint32_t n_tok) {
    uint64_t row_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent() || conv_dim == 0 || n_taps != 4u || n_tok == 0 ||
        !p36_u64_mul_ok(row_bytes, n_taps, &kernel_bytes) ||
        !p36_gpu_tensor_range_ok(cache_conv, 0, (uint64_t)(n_taps - 1u) * row_bytes) ||
        !p36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * row_bytes)) {
        return 0;
    }
    const float *kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              kernel_offset, kernel_bytes);
    if (!kernel) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *kt = p36_vk_weight_get_unlocked(kernel, kernel_bytes);
    int ok = kt != NULL;
    if (ok) {
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        const p36_gpu_tensor *bindings[4] = { cache_conv, cur, kt, out };
        ok = p36_vk_run_unlocked("recurrent_conv_silu_decode", &p36_vk.recur_conv_silu_decode,
                                 bindings, &push, sizeof(push),
                                 (conv_dim + 255u) / 256u, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_copy_f32_tensor(p36_gpu_tensor *dst,
                            const p36_gpu_tensor *src,
                            uint32_t n_floats) {
    if (n_floats == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { n_floats, n_floats, 0u, n_floats, 1u };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("embed_stage_copy", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (n_floats + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}


/* Host-side mirror of the quantized embedding table, uploaded once.  It is
 * read on every decode token and would otherwise have to be dequantized by
 * the CPU, so it gets its own exact-size device allocation instead of
 * competing with the expert banks for arena blocks. */
static p36_gpu_tensor *p36_vk_embed_table;
static uint64_t p36_vk_embed_table_off;
static bool p36_vk_embed_table_failed;

static p36_gpu_tensor *p36_vk_embed_table_resident(const void *model_map,
                                                   uint64_t model_size,
                                                   uint64_t table_offset,
                                                   uint64_t table_bytes) {
    const unsigned char *source;
    uint64_t t0;
    if (table_bytes == 0) return NULL;
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_embed_table && p36_vk_embed_table_off != table_offset) {
        p36_vk_tensor_free_unlocked(p36_vk_embed_table);
        p36_vk_embed_table = NULL;
    }
    if (!p36_vk_embed_table && !p36_vk_embed_table_failed) {
        source = p36_gpu_weight_bytes(model_map, model_size, table_offset, table_bytes);
        p36_vk_embed_table = source ?
            p36_vk_tensor_alloc_kind_unlocked(table_bytes, true, false) : NULL;
        if (p36_vk_embed_table) {
            t0 = p36_vk_now_ns();
            {
                p36_vk_weight_copy_ctx c = { p36_vk_embed_table->data, source, table_bytes };
                p36_gpu_parallel_for_rows((table_bytes + ((1u << 20) - 1)) >> 20, 8,
                                          p36_vk_weight_copy_rows, &c);
            }
            p36_vk_embed_table_off = table_offset;
            fprintf(stderr, "p36: embedding table %.1f MiB resident on device (%.0f ms)\n",
                    (double)table_bytes / (1024.0 * 1024.0),
                    (double)(p36_vk_now_ns() - t0) / 1e6);
        } else {
            p36_vk_embed_table_failed = true;
            fprintf(stderr, "p36: embedding table does not fit on device, "
                            "keeping the host dequant path\n");
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return p36_vk_embed_table;
}

void p36_gpu_embed_table_prepare(const void *model_map, uint64_t model_size,
                                 uint64_t table_offset, uint64_t table_bytes) {
    (void)p36_vk_embed_table_resident(model_map, model_size, table_offset, table_bytes);
}
/* Decode-path embedding: read one quantized row off the device copy of the
 * table instead of dequantizing it on the host.  The id comes from a
 * device buffer when one is passed (the top-2 result of the previous
 * token), which is what lets the next token be recorded without the host
 * knowing it yet. */
int p36_gpu_embed_row_tensor(p36_gpu_tensor *dst,
                             const void *model_map,
                             uint64_t model_size,
                             uint64_t table_offset,
                             uint64_t table_bytes,
                             uint32_t n_embd,
                             uint32_t type,
                             const p36_gpu_tensor *ids,
                             uint32_t row) {
    uint64_t row_bytes;
    const unsigned char *source;
    p36_gpu_tensor *weights;
    int ok = 0;
    struct {
        uint32_t n_embd;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t row;
        uint32_t use_ids;
        uint32_t type;
    } push;
    if (type == 8u) row_bytes = (uint64_t)(n_embd / 32u) * 34u;        /* Q8_0 */
    else if (type == 12u) row_bytes = (uint64_t)(n_embd / 256u) * 144u; /* Q4_K */
    else return 0;
    if (n_embd == 0 || table_bytes == 0) return 0;
    push.n_embd = n_embd;
    push.blocks = (uint32_t)(n_embd / 32u);
    push.row_bytes = (uint32_t)row_bytes;
    push.row = row;
    push.use_ids = ids ? 1u : 0u;
    push.type = type;
    source = p36_gpu_weight_bytes(model_map, model_size, table_offset, table_bytes);
    if (!source) return 0;
    weights = p36_vk_embed_table_resident(model_map, model_size, table_offset, table_bytes);
    if (weights) {
        const p36_gpu_tensor *bindings[3] = { weights, ids ? ids : dst, dst };
        ok = p36_vk_run_unlocked("embed_row", &p36_vk.embed_row, bindings,
                                 &push, sizeof(push), (n_embd + 255u) / 256u, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}
int p36_gpu_extract_full_attn_q_tensor(p36_gpu_tensor *dst,
                                       const p36_gpu_tensor *qg,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { P36_VK_N_HEAD_DIM * 2u, P36_VK_N_HEAD_DIM, 0u, P36_VK_N_HEAD_DIM, n_tok * P36_VK_N_HEAD };
    const p36_gpu_tensor *bindings[2] = { qg, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_extract_q", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (P36_VK_N_HEAD_DIM + 255u) / 256u, n_tok * P36_VK_N_HEAD, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_extract_recurrent_v_tensor(p36_gpu_tensor *dst,
                                       const p36_gpu_tensor *conv,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent()) {
        const float *src = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)conv, "submit_wait_recur_extract_v_src");
        float *out = p36_vk_tensor_contents_labeled(dst, "submit_wait_recur_extract_v_out");
        if (!src || !out) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            memcpy(out + (uint64_t)t * 4096u,
                   src + (uint64_t)t * P36_VK_N_SSM_CONV_DIM + (uint64_t)P36_VK_N_SSM_QK * 2u,
                   (size_t)4096u * sizeof(float));
        }
        return 1;
    }
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { P36_VK_N_SSM_CONV_DIM, 4096u, P36_VK_N_SSM_QK * 2u, 4096u, n_tok };
    const p36_gpu_tensor *bindings[2] = { conv, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("recurrent_extract_v", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (4096u + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rope_qwen_rows_tensor(p36_gpu_tensor *x,
                                  uint32_t n_head,
                                  uint32_t pos0,
                                  uint32_t n_tok) {
    if (n_head == 0 || n_tok == 0) return 0;
    struct {
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t pos0;
        uint32_t n_tok;
    } push = { n_head, P36_VK_N_HEAD_DIM, pos0, n_tok };
    const p36_gpu_tensor *bindings[1] = { x };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_rope", &p36_vk.rope, bindings, &push, sizeof(push), n_head, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rope_qwen_mrope_rows_tensor(p36_gpu_tensor *x,
                                        uint32_t n_head,
                                        const uint32_t *positions,
                                        uint32_t n_tok) {
    if (n_head == 0 || n_tok == 0 || !positions) return 0;
    p36_gpu_tensor *gpos = p36_gpu_tensor_alloc((uint64_t)n_tok * 3u * sizeof(*positions));
    if (!gpos || !p36_gpu_tensor_write(gpos, 0, positions,
                                        (uint64_t)n_tok * 3u * sizeof(*positions))) {
        p36_gpu_tensor_free(gpos);
        return 0;
    }
    struct {
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t n_tok;
    } push = { n_head, P36_VK_N_HEAD_DIM, n_tok };
    const p36_gpu_tensor *bindings[2] = { x, gpos };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_mrope", &p36_vk.rope_mrope,
                                 bindings, &push, sizeof(push), n_head, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(gpos);
    return ok;
}

int p36_gpu_rms_norm_rope_qwen_rows_tensor(p36_gpu_tensor *dst,
                                           const p36_gpu_tensor *src,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t src_stride,
                                           uint32_t n_head,
                                           uint32_t pos0,
                                           uint32_t n_tok,
                                           float eps) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t src_floats;
    uint64_t dst_floats;
    if (src_stride < head_dim || rows == 0) return 0;
    src_floats = (rows - 1u) * src_stride + head_dim;
    dst_floats = rows * head_dim;
    if (!p36_gpu_tensor_range_ok(src, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(dst, 0, dst_floats * sizeof(float))) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        float eps;
        uint32_t pad;
    } push = { src_stride, n_head, pos0, n_tok, eps, 0u };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { src, wt, dst };
        ok = p36_vk_run_unlocked("norm_rope", &p36_vk.rms_norm_rope,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rms_norm_rope_qwen_kv_store_tensor(p36_gpu_tensor *k_cache,
                                                p36_gpu_tensor *v_cache,
                                                const p36_gpu_tensor *k,
                                                const p36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t cache_values = (uint64_t)cap * n_head * head_dim;
    uint64_t src_floats;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) {
        return 0;
    }
    src_floats = (rows - 1u) * src_stride + head_dim;
    if (!p36_gpu_tensor_range_ok(k, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, rows * head_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, cache_values * sizeof(uint16_t)) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, cache_values * sizeof(uint16_t))) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = p36_vk_run_unlocked("norm_rope_kv", &p36_vk.rms_norm_rope_kv,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Fused RMS norm + Qwen RoPE + quantized KV store (Q8_0 K / Q4_0 V).
 * Same semantics as p36_gpu_rms_norm_rope_qwen_kv_store_tensor but writes
 * quantized blocks directly on the GPU, eliminating the host round-trip. */
int p36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(p36_gpu_tensor *k_cache,
                                                p36_gpu_tensor *v_cache,
                                                const p36_gpu_tensor *k,
                                                const p36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps,
                                                uint32_t k_row_bytes,
                                                uint32_t v_row_bytes) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t src_floats;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) return 0;
    /* The kernel writes fixed Q8_0 K rows and Q4_0 V rows; reject any
     * other cache layout. */
    if (k_row_bytes != n_head * (head_dim / 32u) * 34u ||
        v_row_bytes != n_head * (head_dim / 32u) * 18u) return 0;
    src_floats = (rows - 1u) * src_stride + head_dim;
    if (!p36_gpu_tensor_range_ok(k, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, rows * head_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, (uint64_t)cap * k_row_bytes) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, (uint64_t)cap * v_row_bytes)) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
        uint32_t k_row_bytes;
        uint32_t v_row_bytes;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps, k_row_bytes, v_row_bytes };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = p36_vk_run_unlocked("norm_rope_kv_quant", &p36_vk.rms_norm_rope_kv_quant,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Short causal conv1d + SiLU on the host: fp64 tap accumulation like
 * p36_ssm_conv_apply() and p36_siluf()'s exact formula with libm expf.
 * Batched over n_tok windows of n_taps rows each. */
typedef struct {
    float *out;
    const float *window;
    const float *kernel;
    uint32_t conv_dim;
    uint32_t n_taps;
} p36_vk_conv_silu_ctx;

