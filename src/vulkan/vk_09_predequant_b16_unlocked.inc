static p36_gpu_tensor *p36_vk_predequant_b16_unlocked(const p36_gpu_tensor *q8,
                                                      uint32_t n_tok,
                                                      uint32_t blocks) {
    const uint64_t total = (uint64_t)n_tok * blocks;
    const uint64_t bytes = total * 512u;
    uint64_t root_off = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root((p36_gpu_tensor *)q8, &root_off);
    if (root && p36_vk_b16_last.b16 &&
        p36_vk_b16_last.root == root && p36_vk_b16_last.root_off == root_off &&
        p36_vk_b16_last.write_gen == root->write_gen &&
        p36_vk_b16_last.n_tok == n_tok && p36_vk_b16_last.blocks == blocks) {
        return p36_vk_b16_last.b16;
    }
    p36_gpu_tensor *b16 = NULL;
    for (uint32_t k = 0; k < P36_VK_B16_RING; k++) {
        uint32_t i = (p36_vk_b16_next + 1u + k) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t && p36_vk_b16_ring[i].bytes == bytes &&
            p36_vk_b16_ring[i].t != p36_vk_b16_last.b16) {
            b16 = p36_vk_b16_ring[i].t;
            p36_vk_b16_next = i;
            break;
        }
    }
    if (!b16) {
        uint32_t i = (p36_vk_b16_next + 1u) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t == p36_vk_b16_last.b16 && p36_vk_b16_ring[i].t)
            i = (i + 1u) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t) p36_vk_tensor_release_unlocked(p36_vk_b16_ring[i].t);
        p36_vk_b16_ring[i].t = p36_vk_tensor_alloc_scratch_unlocked(bytes);
        p36_vk_b16_ring[i].bytes = p36_vk_b16_ring[i].t ? bytes : 0;
        b16 = p36_vk_b16_ring[i].t;
        p36_vk_b16_next = i;
        if (!b16) return NULL;
    }
    struct { uint32_t total_blocks; } push = { (uint32_t)total };
    const p36_gpu_tensor *bindings[2] = { q8, b16 };
    if (!p36_vk_run_unlocked("predequant_b16", &p36_vk.predequant_b16,
                             bindings, &push, sizeof(push),
                             (uint32_t)((total * 64u + 255u) / 256u), 1, 1)) {
        return NULL;
    }
    p36_vk_b16_last.root = root;
    p36_vk_b16_last.root_off = root_off;
    p36_vk_b16_last.write_gen = root ? root->write_gen : 0;
    p36_vk_b16_last.n_tok = n_tok;
    p36_vk_b16_last.blocks = blocks;
    p36_vk_b16_last.b16 = b16;
    return b16;
}

static int p36_vk_matmul_kquant_mmq_unlocked(p36_gpu_tensor *out,
                                             const p36_gpu_tensor *weights,
                                             const p36_gpu_tensor *q8,
                                             uint32_t type,
                                             uint32_t out_dim,
                                             uint32_t n_tok,
                                             uint32_t blocks,
                                             float scale) {
    p36_vk_kernel *kernel = NULL;
    bool fast = p36_vk_use_mmq_fast();
    if (type == P36_VK_TENSOR_Q5_K) kernel = fast ? &p36_vk.matmul_q5k_mmq_fast : &p36_vk.matmul_q5k_mmq;
    else if (type == P36_VK_TENSOR_Q6_K) kernel = fast ? &p36_vk.matmul_q6k_mmq_fast : &p36_vk.matmul_q6k_mmq;
    else return 0;
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        float scale;
    } push = { out_dim, n_tok, blocks, scale };
    const p36_gpu_tensor *bindings[3] = { weights, q8, out };
    return p36_vk_run_unlocked("dense_kquant_mmq",
                               kernel,
                               bindings,
                               &push,
                               sizeof(push),
                               (out_dim + 7u) / 8u,
                               (n_tok + 3u) / 4u,
                               1);
}

int p36_gpu_matmul_k_quant_q8_scaled_tensor(p36_gpu_tensor *out,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t weight_offset,
                                            uint32_t weight_type,
                                            uint64_t in_dim,
                                            uint64_t out_dim,
                                            const p36_gpu_tensor *q8,
                                            uint64_t n_tok,
                                            float scale) {
    if (in_dim == 0 || (in_dim % 256u) != 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t block_bytes = p36_vk_kquant_row_bytes(weight_type);
    if (block_bytes == 0) return 0;
    const uint64_t blocks = in_dim / P36_VK_QK_K;
    const uint64_t row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX) return 0;

    uint64_t out_bytes = 0;
    uint64_t q8_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, blocks, &q8_bytes) || !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;

    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    /* Named for the dense-model launch scripts that first exercised it, but
     * the kernel itself only reads this one tensor's own dims/type/scale --
     * nothing here depends on the rest of the model being dense.  MoE models
     * carry K-quant non-expert tensors too (e.g. attn_qkv/output.weight on
     * Qwen3.8-35B-A3B-IQ2_M), and those hit the AVX2-style 8x8 generic
     * matmul_kquant fallback below unconditionally because this gate was
     * never widened past p36_gpu_dense_model. */
    if (!p36_gpu_quality && n_tok == 1u &&
        p36_vk_env_default_on("P36_VK_DENSE_KQUANT_DECODE")) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        int ok = weights != NULL;
        if (ok) {
            struct {
                uint32_t in_dim;
                uint32_t out_dim;
                uint32_t n_tok;
                uint32_t blocks;
                uint32_t row_bytes;
                uint32_t type;
                float scale;
            } push = {
                (uint32_t)in_dim, (uint32_t)out_dim, 1u,
                (uint32_t)blocks, (uint32_t)row_bytes, weight_type, scale,
            };
            const p36_gpu_tensor *bindings[3] = { weights, q8, out };
            const char *op = weight_type == P36_VK_TENSOR_Q2_K ? "dense_q2k_decode" :
                             (weight_type == P36_VK_TENSOR_Q3_K ? "dense_q3k_decode" :
                             (weight_type == P36_VK_TENSOR_Q4_K ? "dense_q4k_decode" :
                             (weight_type == P36_VK_TENSOR_Q5_K ? "dense_q5k_decode" :
                                                                 "dense_q6k_decode")));
            op = p36_vk_prof_iq3_shape(
                weight_type, 1u, (uint32_t)in_dim, (uint32_t)out_dim, op);
            bool q4_full = weight_type == P36_VK_TENSOR_Q4_K &&
                           out_dim % 4u == 0u;
            bool q5_full = weight_type == P36_VK_TENSOR_Q5_K &&
                           out_dim % 4u == 0u;
            bool q3_full = weight_type == P36_VK_TENSOR_Q3_K &&
                           out_dim % 4u == 0u &&
                           p36_vk_env_default_on("P36_VK_DENSE_Q3K_DECODE");
            bool q2_full = weight_type == P36_VK_TENSOR_Q2_K &&
                           out_dim % 4u == 0u &&
                           p36_vk_env_default_on("P36_VK_DENSE_Q2K_DECODE");
            p36_vk_kernel *kernel = q4_full ? &p36_vk.dense_q4k_decode :
                                   q5_full ? &p36_vk.dense_q5k_decode :
                                   q3_full ? &p36_vk.dense_q3k_decode :
                                   q2_full ? &p36_vk.dense_q2k_decode :
                                             &p36_vk.dense_kquant_decode;
            ok = p36_vk_run_unlocked(op, kernel,
                                     bindings, &push, sizeof(push),
                                     ((uint32_t)out_dim + 3u) / 4u, 1, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    /* Small multi-row verify on the K-quant trunk: stream each weight block
     * once and broadcast it over up to four activation rows, instead of paying
     * the 128-row MMQ tile for a 2..4 token verifier. */
#ifndef P36_VK_KQUANT_ROWS
#define P36_VK_KQUANT_ROWS 4u
#endif
    bool kquant_rows = weight_type == P36_VK_TENSOR_Q3_K ||
                       weight_type == P36_VK_TENSOR_Q4_K ||
                       weight_type == P36_VK_TENSOR_Q5_K ||
                       weight_type == P36_VK_TENSOR_Q6_K;
    if (!p36_gpu_quality && kquant_rows && n_tok > 1u && n_tok <= 4u &&
        out_dim % P36_VK_KQUANT_ROWS == 0u &&
        p36_vk_env_default_on("P36_VK_DENSE_KQUANT_ROWS")) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        int ok = weights != NULL;
        if (ok) {
            struct {
                uint32_t in_dim;
                uint32_t out_dim;
                uint32_t n_tok;
                uint32_t blocks;
                uint32_t row_bytes;
                uint32_t type;
                float scale;
            } push = {
                (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok,
                (uint32_t)blocks, (uint32_t)row_bytes, weight_type, scale,
            };
            const p36_gpu_tensor *bindings[3] = { weights, q8, out };
            const char *op =
                weight_type == P36_VK_TENSOR_Q3_K ? "dense_q3k_rows" :
                weight_type == P36_VK_TENSOR_Q4_K ? "dense_q4k_rows" :
                weight_type == P36_VK_TENSOR_Q5_K ? "dense_q5k_rows" :
                                                    "dense_q6k_rows";
            /* The 4-row build computes all four token slots whatever n_tok
             * is, so a 2-row verify (the usual MTP draft-1 cycle) paid for
             * four; the 2-row build does only the live rows, same math. */
            bool two = n_tok == 2u;
            p36_vk_kernel *kernel =
                weight_type == P36_VK_TENSOR_Q3_K ? (two ? &p36_vk.dense_q3k_rows2 : &p36_vk.dense_q3k_rows) :
                weight_type == P36_VK_TENSOR_Q4_K ? (two ? &p36_vk.dense_q4k_rows2 : &p36_vk.dense_q4k_rows) :
                weight_type == P36_VK_TENSOR_Q5_K ? (two ? &p36_vk.dense_q5k_rows2 : &p36_vk.dense_q5k_rows) :
                                                    (two ? &p36_vk.dense_q6k_rows2 : &p36_vk.dense_q6k_rows);
            ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                     ((uint32_t)out_dim + P36_VK_KQUANT_ROWS - 1u) /
                                         P36_VK_KQUANT_ROWS, 1, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    /* Same widening as the decode branch above: prefill-side K-quant trunk
     * tensors are just as generic on a MoE model. */
    if (!p36_gpu_quality && n_tok > 1u &&
        p36_vk.have_f16 && p36_vk.subgroup_size_control &&
        p36_vk_env_default_on("P36_VK_DENSE_KQUANT_MMQ")) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        int ok = weights != NULL;
        if (ok) {
            struct {
                uint32_t in_dim;
                uint32_t out_dim;
                uint32_t n_tok;
                uint32_t blocks;
                uint32_t row_bytes;
                uint32_t type;
                float scale;
            } push = {
                (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok,
                (uint32_t)blocks, (uint32_t)row_bytes, weight_type, scale,
            };
            p36_gpu_tensor *b16 = p36_vk_predequant_b16_unlocked(
                q8, (uint32_t)n_tok, (uint32_t)blocks);
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, b16 };
            /* One pipeline, branch on push-constant type: label the profiler
             * row by type so per-arm changes are not lost in the shared row. */
            const char *mmq_op =
                weight_type == P36_VK_TENSOR_Q2_K ? "dense_kquant_mmq_q2k" :
                weight_type == P36_VK_TENSOR_Q3_K ? "dense_kquant_mmq_q3k" :
                weight_type == P36_VK_TENSOR_Q4_K ? "dense_kquant_mmq_q4k" :
                weight_type == P36_VK_TENSOR_Q5_K ? "dense_kquant_mmq_q5k" :
                weight_type == P36_VK_TENSOR_Q6_K ? "dense_kquant_mmq_q6k" :
                                                    "dense_kquant_mmq";
            const char *op = p36_vk_prof_iq3_shape(
                weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                (uint32_t)out_dim, mmq_op);
            ok = b16 != NULL &&
                 (n_tok <= 64u ?
                  p36_vk_run_unlocked(op, &p36_vk.dense_kquant_mmq_bn64,
                                      bindings, &push, sizeof(push),
                                      ((uint32_t)out_dim + 31u) / 32u, 1, 1) :
                  p36_vk_run_unlocked(op, &p36_vk.dense_kquant_mmq,
                                      bindings, &push, sizeof(push),
                                      ((uint32_t)out_dim + 31u) / 32u,
                                      ((uint32_t)n_tok + 127u) / 128u, 1));
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (!p36_gpu_quality && p36_vk_use_mmq() &&
        (weight_type == P36_VK_TENSOR_Q5_K || weight_type == P36_VK_TENSOR_Q6_K) &&
        (n_tok % 4u) == 0u && n_tok >= 4u) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *packed = p36_vk_packed_weight_get_unlocked(source, weight_bytes, weight_type,
                                                                   (uint32_t)in_dim, (uint32_t)out_dim);
        int ok = packed && p36_vk_matmul_kquant_mmq_unlocked(out, packed, q8, (uint32_t)weight_type,
                                                             (uint32_t)out_dim, (uint32_t)n_tok,
                                                             (uint32_t)blocks, scale);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
    int ok = weights != NULL;

    if (ok) {
        struct {
            uint32_t in_dim;
            uint32_t out_dim;
            uint32_t n_tok;
            uint32_t blocks;
            uint32_t row_bytes;
            uint32_t type;
            float scale;
        } mpush = {
            (uint32_t)in_dim,
            (uint32_t)out_dim,
            (uint32_t)n_tok,
            (uint32_t)blocks,
            (uint32_t)row_bytes,
            weight_type,
            scale,
        };
        const p36_gpu_tensor *mbindings[3] = { weights, q8, out };
        /* 8 rows x 8 AVX2 lanes per workgroup. */
        ok = p36_vk_run_unlocked("dense_kquant",
                                 &p36_vk.matmul_kquant,
                                 mbindings,
                                 &mpush,
                                 sizeof(mpush),
                                 ((uint32_t)out_dim + 7u) / 8u,
                                 ((uint32_t)n_tok + P36_VK_MATMUL_TOK_TILE - 1u) / P36_VK_MATMUL_TOK_TILE,
                                 1);
    }

    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_k_quant_scaled_tensor(p36_gpu_tensor *out,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t weight_offset,
                                         uint32_t weight_type,
                                         uint64_t in_dim,
                                         uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok,
                                         float scale) {
    p36_gpu_tensor *q8 = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!q8) return 0;
    int ok = p36_gpu_matmul_k_quant_q8_scaled_tensor(out, model_map, model_size, weight_offset,
                                                     weight_type, in_dim, out_dim, q8, n_tok, scale);
    p36_gpu_tensor_free(q8);
    return ok;
}

int p36_gpu_matmul_k_quant_tensor(p36_gpu_tensor *out,
                                  const void *model_map,
                                  uint64_t model_size,
                                  uint64_t weight_offset,
                                  uint32_t weight_type,
                                  uint64_t in_dim,
                                  uint64_t out_dim,
                                  const p36_gpu_tensor *x,
                                  uint64_t n_tok) {
    return p36_gpu_matmul_k_quant_scaled_tensor(out, model_map, model_size, weight_offset,
                                                weight_type, in_dim, out_dim, x, n_tok, 1.0f);
}

/* Row-wise RMS norm on the host, the exact float-accumulation numerics of
 * p36_ref_rms_norm() (the mode the CPU parity gates were tuned for): scalar
 * square sum, 1/sqrtf scale, optional f32 weight. */
static void p36_vk_rms_norm_row(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
    {
        float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

/* Fixed-size clone of the CPU engine's 128-wide norm: gcc compiles that
 * constant-size copy to an 8-lane ymm square-sum reduced in hsum pair
 * order; the explicit lanes reproduce those roundings exactly. */
static void p36_vk_rms_norm_row_128(float *dst, const float *src, const float *weight, float eps) {
    float lanes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float ss;
    for (uint32_t i = 0; i < 128u; i += 8u) {
        for (uint32_t k = 0; k < 8u; k++) lanes[k] += src[i + k] * src[i + k];
    }
    ss = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
       + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    {
        float scale = 1.0f / sqrtf(ss / 128.0f + eps);
        for (uint32_t i = 0; i < 128u; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

typedef struct {
    float *out;
    const float *x;
    const float *weight;
    uint32_t n;
    float eps;
} p36_vk_rms_norm_ctx;

static void p36_vk_rms_norm_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_rms_norm_ctx *ctx = (p36_vk_rms_norm_ctx *)opaque;

    for (uint64_t r = row0; r < row1; r++) {
        if (ctx->n == 128u) p36_vk_rms_norm_row_128(ctx->out + r * 128u, ctx->x + r * 128u, ctx->weight, ctx->eps);
        else p36_vk_rms_norm_row(ctx->out + r * ctx->n, ctx->x + r * ctx->n, ctx->weight, ctx->n, ctx->eps);
    }
}

static int p36_vk_rms_norm_tensor(p36_gpu_tensor *out,
                                  const p36_gpu_tensor *x,
                                  const float *weight,
                                  uint32_t n,
                                  uint32_t rows,
                                  float eps) {
    if (n == 0 || rows == 0) return 0;
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) || !p36_gpu_tensor_range_ok(x, 0, bytes)) return 0;

    /* Fast mode always norms on GPU: with f32-B dense matvecs the host no
     * longer reads activations afterwards, so a host norm here would be the
     * only remaining decode flush cause. */
    if (!p36_gpu_quality && p36_vk_use_gpu_rms()) {
        struct {
            uint32_t n;
            uint32_t has_weight;
            float eps;
        } push = { n, weight ? 1u : 0u, eps };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *wt = weight ? p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float)) : (p36_gpu_tensor *)x;
        int ok = wt != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { x, wt, out };
            ok = p36_vk_run_unlocked("norm", &p36_vk.rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_rms_norm_ctx ctx;
    const float *xp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_rms_norm_x");
    float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_rms_norm_out");
    if (!xp || !op) return 0;
    ctx.out = op;
    ctx.x = xp;
    ctx.weight = weight;
    ctx.n = n;
    ctx.eps = eps;
    if (elems >= 8192u) p36_gpu_parallel_for_rows(rows, 2, p36_vk_rms_norm_rows, &ctx);
    else p36_vk_rms_norm_rows(&ctx, 0, rows);
    return 1;
}

int p36_gpu_rms_norm_plain_rows_tensor(p36_gpu_tensor *out,
                                       const p36_gpu_tensor *x,
                                       uint32_t n,
                                       uint32_t rows,
                                       float eps) {
    return p36_vk_rms_norm_tensor(out, x, NULL, n, rows, eps);
}

int p36_gpu_rms_norm_weight_rows_tensor(p36_gpu_tensor *out,
                                         const p36_gpu_tensor *x,
                                         const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint32_t n,
                                        uint32_t rows,
                                        float eps) {
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;
    return p36_vk_rms_norm_tensor(out, x, weight, n, rows, eps);
}

/* SwiGLU on the host: p36_siluf()'s exact x / (1 + expf(-x)) with the same
 * scalar libm expf the CPU engine calls. */
typedef struct {
    float *out;
    const float *gate;
    const float *up;
    float clamp;
    float weight;
} p36_vk_swiglu_ctx;

static void p36_vk_swiglu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_swiglu_ctx *ctx = (p36_vk_swiglu_ctx *)opaque;

    for (uint64_t i = row0; i < row1; i++) {
        float g = ctx->gate[i];
        float u = ctx->up[i];
        if (ctx->clamp > 1.0e-6f) {
            if (g > ctx->clamp) g = ctx->clamp;
            if (u > ctx->clamp) u = ctx->clamp;
            if (u < -ctx->clamp) u = -ctx->clamp;
        }
        ctx->out[i] = g / (1.0f + expf(-g)) * u * ctx->weight;
    }
}

int p36_gpu_swiglu_tensor(p36_gpu_tensor *out,
                          const p36_gpu_tensor *gate,
                          const p36_gpu_tensor *up,
                          uint32_t n,
                          float clamp,
                          float weight) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !p36_gpu_tensor_range_ok(up, 0, bytes)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_swiglu()) {
        struct {
            uint32_t n;
            float clamp;
            float weight;
        } push = { n, clamp, weight };
        const p36_gpu_tensor *bindings[3] = { gate, up, out };
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_run_unlocked("swiglu", &p36_vk.swiglu, bindings, &push, sizeof(push), (n + 255u) / 256u, 1, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_swiglu_ctx ctx;
    const float *gp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)gate, "submit_wait_swiglu_gate");
    const float *upp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)up, "submit_wait_swiglu_up");
    float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_swiglu_out");
    if (!gp || !upp || !op) return 0;

    ctx.out = op;
    ctx.gate = gp;
    ctx.up = upp;
    ctx.clamp = clamp;
    ctx.weight = weight;
    if (n >= 16384u) p36_gpu_parallel_for_rows(n, 1024, p36_vk_swiglu_rows, &ctx);
    else p36_vk_swiglu_rows(&ctx, 0, n);
    return 1;
}

int p36_gpu_swiglu_q8_k_tensor(p36_gpu_tensor *out,
                                p36_gpu_tensor *q8,
                                const p36_gpu_tensor *gate,
                                const p36_gpu_tensor *up,
                                uint32_t in_dim,
                                uint32_t n_tok,
                                float clamp,
                                float weight) {
    if (p36_gpu_quality || !p36_vk_env_default_on("P36_VK_SWIGLU_Q8") ||
        !in_dim || (in_dim & 255u) || !n_tok) return 0;
    uint64_t elems = (uint64_t)in_dim * n_tok;
    uint64_t bytes = elems * sizeof(float);
    uint32_t blocks = in_dim / 256u;
    uint64_t q8_bytes = (uint64_t)n_tok * blocks * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !p36_gpu_tensor_range_ok(up, 0, bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
        float clamp;
        float weight;
    } push = { in_dim, blocks, n_tok, clamp, weight };
    const p36_gpu_tensor *bindings[4] = { gate, up, out, q8 };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("swiglu_q8_k", &p36_vk.swiglu_q8_k,
                                 bindings, &push, sizeof(push),
                                 blocks, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_ffn_tail_tensor(p36_gpu_tensor *out,
                            const p36_gpu_tensor *shared,
                            const p36_gpu_tensor *scalar,
                            uint32_t out_dim,
                            uint32_t n_tok) {
    if (out_dim == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(shared, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(scalar, 0, (uint64_t)n_tok * sizeof(float))) {
        return 0;
    }
    if (p36_gpu_quality || !p36_vk_use_gpu_ffn_tail()) {
        const float *sp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)shared, "submit_wait_ffn_tail_shared");
        const float *gp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)scalar, "submit_wait_ffn_tail_scalar");
        float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_ffn_tail_out");
        if (!sp || !gp || !op) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float g = p36_vk_sigmoid_stable(gp[t]);
            if (!isfinite(g)) return 0;
            for (uint32_t j = 0; j < out_dim; j++) {
                uint64_t idx = (uint64_t)t * out_dim + j;
                op[idx] = fmaf(sp[idx], g, op[idx]);
            }
        }
        return 1;
    }
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
    } push = { out_dim, n_tok };
    const p36_gpu_tensor *bindings[3] = { out, shared, scalar };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("moe_tail", &p36_vk.ffn_tail, bindings, &push, sizeof(push), (out_dim + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_fwht_tensor(p36_gpu_tensor *dst,
                        const p36_gpu_tensor *src,
                        uint32_t width,
                        uint32_t n_rows,
                        float scale) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    if (!dst || !src || width == 0 || n_rows == 0) return 0;
    if ((width % 1024u) != 0 || width > UINT32_MAX) return 0;
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
        float    scale;
    } push = { width, scale };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("fwht", &p36_vk.fwht, bindings, &push, sizeof(push),
                                 width / 1024u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_signs_mul_tensor(p36_gpu_tensor *dst,
                             const p36_gpu_tensor *src,
                             const p36_gpu_tensor *signs,
                             uint32_t width,
                             uint32_t n_rows,
                             uint32_t sign_offset,
                             uint32_t total_signs) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    uint64_t sign_bytes = 0;
    if (!dst || !src || !signs || width == 0 || n_rows == 0) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes) ||
        !p36_u64_mul_ok(total_signs, sizeof(float), &sign_bytes)) {
        return 0;
    }
    if (sign_offset > total_signs || (uint64_t)sign_offset + width > total_signs) return 0;
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(dst, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(signs, 0, sign_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t sign_offset;
    } push = { width, sign_offset };
    const p36_gpu_tensor *bindings[3] = { src, signs, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("signs_mul", &p36_vk.signs_mul, bindings, &push,
                                 sizeof(push), (width + 255u) / 256u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

