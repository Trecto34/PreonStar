static const char *p36_vk_q8_0_quant_wait_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * P36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "submit_wait_q8_0_q_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "submit_wait_q8_0_q_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "submit_wait_q8_0_q_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "submit_wait_q8_0_q_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "submit_wait_q8_0_q_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "submit_wait_q8_0_q_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "submit_wait_q8_0_q_vocabx2048";
        return "submit_wait_q8_0_q_other";
    }
    return "submit_wait_q8_0_quant_x";
}

static int p36_vk_matmul_q8_0_gpu(p36_gpu_tensor *out,
                                  const unsigned char *w,
                                  uint64_t weight_bytes,
                                  const unsigned char *xq,
                                  uint64_t xq_bytes,
                                  uint64_t row_bytes,
                                  uint64_t out_dim,
                                  uint64_t n_tok,
                                  uint64_t blocks,
                                  float scale) {
    if (n_tok > 1u && !p36_vk_have_subgroups(true)) return 0;
    p36_gpu_tensor *xq_tensor = p36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;
    unsigned char *xq_host = p36_vk_tensor_contents_labeled(xq_tensor, "submit_wait_q8_0_stage_xq");
    if (!xq_host) {
        p36_gpu_tensor_free(xq_tensor);
        return 0;
    }
    memcpy(xq_host, xq, (size_t)xq_bytes);

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, xq_tensor, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            p36_vk_kernel *kernel = &p36_vk.matmul_q8_0_decode;
            if (blocks == 64u && row_bytes == 64u * 34u) kernel = &p36_vk.matmul_q8_0_decode_b64;
            ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(xq_tensor);
    return ok;
}

static bool p36_vk_use_q8_mm_f16(void) {
    return p36_vk.have_f16 && p36_vk.have_storage16 &&
           p36_vk_env_default_on("P36_VK_Q8_MM_F16");
}

static bool p36_vk_use_q8_mm_f16_out32(void) {
    return p36_vk_env_default_on("P36_VK_Q8_MM_F16_OUT32");
}

/* MTP verify/replay window: micro batches there must be bit-identical to
 * the decode matvec, not to the prefill GEMM, so speculative rows equal
 * plain greedy decode. Session-sync small extends must keep the GEMM or
 * warm resume would stop matching the chunked cold rebuild. */
static bool p36_vk_micro_batch;

void p36_gpu_set_micro_batch(bool enabled) {
    p36_vk_micro_batch = enabled;
}

/* Prefill GEMM against raw f32 activations: 64x64 shared-memory tiles,
 * weights dequantized once per k-block for all tokens. Default is the
 * packed-f16 kernel; P36_VK_Q8_MM_F16=0 keeps the f32 tiles. */
static int p36_vk_matmul_q8_0_mm(p36_gpu_tensor *out,
                                 const unsigned char *w,
                                 uint64_t weight_bytes,
                                 const p36_gpu_tensor *x,
                                 uint64_t row_bytes,
                                 uint64_t out_dim,
                                 uint64_t n_tok,
                                 uint64_t blocks,
                                 float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        int out32 = n_tok >= 32u && out_dim == 32u && p36_vk_use_q8_mm_f16() && p36_vk_use_q8_mm_f16_out32();
        int f16 = n_tok >= 32u && out_dim >= 64u && p36_vk_use_q8_mm_f16();
        p36_vk_kernel *kernel = out32 ? &p36_vk.matmul_q8_0_mm_f16_out32 :
                                f16 ? &p36_vk.matmul_q8_0_mm_f16 :
                                      &p36_vk.matmul_q8_0_mm;
        uint32_t gx = out32 ? 1u :
                      f16 ? (uint32_t)((out_dim + 127u) / 128u) :
                            (uint32_t)((out_dim + 63u) / 64u);
        uint32_t gy = (uint32_t)((n_tok + 63u) / 64u);
        ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Dense decode matvec against raw f32 activations: no activation
 * quantization, so no host read of x and no q8 staging upload. Numerics
 * differ from the CPU path (which dots q8-quantized activations), so this
 * stays out of --quality. */
static int p36_vk_matmul_q8_0_f32b(p36_gpu_tensor *out,
                                   const unsigned char *w,
                                   uint64_t weight_bytes,
                                   const p36_gpu_tensor *x,
                                   uint64_t row_bytes,
                                   uint64_t out_dim,
                                   uint64_t blocks,
                                   float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, 1u, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, 1) : "dense_q8_0";
        ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_f32b, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_q8_0_pair_scaled_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_a_dim,
                                           uint64_t out_b_dim,
                                           const p36_gpu_tensor *x,
                                           float scale_a,
                                           float scale_b) {
    uint64_t blocks, row_bytes, weight_a_bytes, weight_b_bytes;
    uint64_t in_bytes, out_a_bytes, out_b_bytes;
    if (p36_gpu_quality || !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        in_dim == 0 || (in_dim % P36_VK_QK8_0) != 0 ||
        out_b_dim == 0 || out_b_dim > out_a_dim ||
        in_dim > UINT32_MAX || out_a_dim > UINT32_MAX || out_b_dim > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_0), &row_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_a_dim, &weight_a_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_b_dim, &weight_b_bytes) ||
        !p36_u64_mul_ok(in_dim, sizeof(float), &in_bytes) ||
        !p36_u64_mul_ok(out_a_dim, sizeof(float), &out_a_bytes) ||
        !p36_u64_mul_ok(out_b_dim, sizeof(float), &out_b_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(x, 0, in_bytes) ||
        !p36_gpu_tensor_range_ok(out_a, 0, out_a_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_b_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_a_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_b_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_a;
        uint32_t out_b;
        uint32_t blocks;
        float scale_a;
        float scale_b;
    } push = { (uint32_t)out_a_dim, (uint32_t)out_b_dim, (uint32_t)blocks,
               scale_a, scale_b };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights_a = p36_vk_weight_get_unlocked(wa, weight_a_bytes);
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_b_bytes) : NULL;
    int ok = weights_a != NULL && weights_b != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { weights_a, weights_b, x, out_a, out_b };
        ok = p36_vk_run_unlocked("dense_q8_0_d_pair", &p36_vk.matmul_q8_0_f32b_pair,
                                 bindings, &push, sizeof(push),
                                 (uint32_t)out_a_dim, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Micro-batch decode matvec: 2..8 tokens accumulated in registers per
 * weight load. Weight traffic equals one matvec and each token's
 * accumulation order equals the single-token kernel, so MTP verify rows
 * are bit-identical to plain decode of the same tokens. */
static int p36_vk_matmul_q8_0_f32b_nx(p36_gpu_tensor *out,
                                      const unsigned char *w,
                                      uint64_t weight_bytes,
                                      const p36_gpu_tensor *x,
                                      uint64_t row_bytes,
                                      uint64_t out_dim,
                                      uint64_t n_tok,
                                      uint64_t blocks,
                                      float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_f32b_nx, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static int p36_vk_matmul_q8_0_gpu_quant(p36_gpu_tensor *out,
                                        const unsigned char *w,
                                        uint64_t weight_bytes,
                                        const p36_gpu_tensor *x,
                                        uint64_t row_bytes,
                                        uint64_t out_dim,
                                        uint64_t n_tok,
                                        uint64_t blocks,
                                        float scale) {
    uint64_t xq_bytes;
    if (!p36_u64_mul_ok(n_tok, blocks, &xq_bytes) ||
        !p36_u64_mul_ok(xq_bytes, 36u, &xq_bytes)) return 0;
    p36_gpu_tensor *xq_tensor = p36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } qpush = { (uint32_t)(blocks * P36_VK_QK8_0), (uint32_t)blocks, (uint32_t)n_tok };

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } mpush = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *qbindings[2] = { x, xq_tensor };
        ok = p36_vk_run_unlocked("q8_0_quant", &p36_vk.quantize_q8_0, qbindings, &qpush, sizeof(qpush),
                                 (uint32_t)blocks, (uint32_t)n_tok, 1);
    }
    if (ok) {
        const p36_gpu_tensor *mbindings[3] = { weights, xq_tensor, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_decode_p36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_p36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(xq_tensor);
    return ok;
}

int p36_gpu_matmul_q8_0_scaled_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const p36_gpu_tensor *x,
                                      uint64_t n_tok,
                                      float scale) {
    uint64_t blocks;
    uint64_t row_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t xq_bytes = 0;
    unsigned char *xq;

    if (in_dim == 0 || n_tok == 0 || (in_dim % P36_VK_QK8_0) != 0 || in_dim > INT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_0), &row_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, n_tok, &xq_bytes) || xq_bytes > SIZE_MAX) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    const unsigned char *w = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    if (!p36_gpu_quality && n_tok == 1 && p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_f32b(out, w, weight_bytes, x, row_bytes, out_dim, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && p36_vk_micro_batch && n_tok >= 2 && n_tok <= 8 &&
        p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_f32b_nx(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && n_tok >= 2 && p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_mm(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && n_tok > 1u && p36_vk_env_default_on("P36_VK_Q8_0_GPU_QUANT") && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_gpu_quant(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x,
                                                         p36_vk.prof_ops ? p36_vk_q8_0_quant_wait_name(out_dim, blocks, n_tok) :
                                                                          "submit_wait_q8_0_quant_x");
    if (!x_host) return 0;

    xq = malloc((size_t)xq_bytes);
    if (!xq) return 0;

    p36_vk_quantize_q8_0_ctx qctx = { xq, x_host, in_dim, row_bytes };
    if (n_tok * in_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 4, p36_vk_quantize_q8_0_rows, &qctx);
    else p36_vk_quantize_q8_0_rows(&qctx, 0, n_tok);

    if (!p36_gpu_quality && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_gpu(out, w, weight_bytes, xq, xq_bytes, row_bytes, out_dim, n_tok, blocks, scale)) {
        free(xq);
        return 1;
    }

    float *out_host = p36_vk_tensor_contents_labeled(out, "submit_wait_q8_0_host_out");
    if (!out_host) {
        free(xq);
        return 0;
    }
    p36_vk_matmul_q8_0_ctx mctx = { w, xq, out_host, in_dim, out_dim, n_tok, row_bytes, scale };
    if (out_dim >= 64u) p36_gpu_parallel_for_rows(out_dim, 64, p36_vk_matmul_q8_0_rows, &mctx);
    else p36_vk_matmul_q8_0_rows(&mctx, 0, out_dim);
    free(xq);
    return 1;
}

int p36_gpu_shared_ffn_decode_tensor(p36_gpu_tensor *out,
                                      p36_gpu_tensor *mid,
                                      const p36_gpu_tensor *x,
                                      const p36_gpu_tensor *scalar,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      float gate_scale,
                                      float up_scale,
                                      float down_scale) {
    uint64_t in_blocks, mid_blocks, gate_bytes, down_bytes;
    if (p36_gpu_quality || !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        !out || !mid || !x || !scalar || in_dim == 0 || mid_dim == 0 || out_dim == 0 ||
        in_dim % P36_VK_QK8_0 != 0 || mid_dim % P36_VK_QK8_0 != 0) {
        return 0;
    }
    in_blocks = in_dim / P36_VK_QK8_0;
    mid_blocks = mid_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(in_blocks * sizeof(p36_vk_block_q8_0), mid_dim, &gate_bytes) ||
        !p36_u64_mul_ok(mid_blocks * sizeof(p36_vk_block_q8_0), out_dim, &down_bytes) ||
        !p36_gpu_tensor_range_ok(x, 0, (uint64_t)in_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(mid, 0, (uint64_t)mid_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(scalar, 0, sizeof(float)) ||
        !p36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * sizeof(float))) {
        return 0;
    }
    const unsigned char *gate = p36_gpu_weight_bytes(model_map, model_size, gate_offset, gate_bytes);
    const unsigned char *up = p36_gpu_weight_bytes(model_map, model_size, up_offset, gate_bytes);
    const unsigned char *down = p36_gpu_weight_bytes(model_map, model_size, down_offset, down_bytes);
    if (!gate || !up || !down) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *gate_tensor = p36_vk_weight_get_unlocked(gate, gate_bytes);
    p36_gpu_tensor *up_tensor = p36_vk_weight_get_unlocked(up, gate_bytes);
    p36_gpu_tensor *down_tensor = p36_vk_weight_get_unlocked(down, down_bytes);
    int ok = gate_tensor && up_tensor && down_tensor;
    if (ok) {
        struct {
            uint32_t mid_dim;
            uint32_t blocks;
            float gate_scale;
            float up_scale;
        } push = { mid_dim, (uint32_t)in_blocks, gate_scale, up_scale };
        const p36_gpu_tensor *bindings[4] = { gate_tensor, up_tensor, x, mid };
        ok = p36_vk_run_unlocked("shared_gate_up_decode", &p36_vk.shared_gate_up_decode,
                                 bindings, &push, sizeof(push), mid_dim, 1, 1);
    }
    if (ok) {
        struct {
            uint32_t out_dim;
            uint32_t blocks;
            float scale;
        } push = { out_dim, (uint32_t)mid_blocks, down_scale };
        const p36_gpu_tensor *bindings[4] = { down_tensor, mid, scalar, out };
        ok = p36_vk_run_unlocked("shared_down_tail_decode", &p36_vk.shared_down_tail_decode,
                                 bindings, &push, sizeof(push), out_dim, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static uint64_t p36_vk_kquant_row_bytes(uint32_t type) {
    switch (type) {
    case P36_VK_TENSOR_Q2_K: return 84;
    case P36_VK_TENSOR_Q3_K: return 110;
    case P36_VK_TENSOR_Q4_K: return 144;
    case P36_VK_TENSOR_Q5_K: return 176;
    case P36_VK_TENSOR_Q6_K: return 210;
    default: return 0;
    }
}

static void p36_vk_pack_q5_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            p36_vk_block_q5_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                uint8_t sc = 0;
                uint8_t mn = 0;
                if (row >= out_dim) continue;
                const p36_vk_block_q5_K_src *blk = (const p36_vk_block_q5_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                packed.dmin[r] = blk->dmin;
                for (uint32_t g = 0; g < 8u; g++) {
                    p36_vk_q5_scale_min(blk->scales, g, &sc, &mn);
                    packed.scales[g * 8u + r] = sc;
                    packed.mins[g * 8u + r] = mn;
                }
                for (uint32_t i = 0; i < P36_VK_QK_K / 8u; i++) packed.qh[i * 8u + r] = blk->qh[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 2u; i++) packed.qs[i * 8u + r] = blk->qs[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static void p36_vk_pack_q6_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            p36_vk_block_q6_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                if (row >= out_dim) continue;
                const p36_vk_block_q6_K_src *blk = (const p36_vk_block_q6_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                for (uint32_t i = 0; i < P36_VK_QK_K / 16u; i++) packed.scales[i * 8u + r] = blk->scales[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 2u; i++) packed.ql[i * 8u + r] = blk->ql[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 4u; i++) packed.qh[i * 8u + r] = blk->qh[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static p36_gpu_tensor *p36_vk_packed_weight_get_unlocked(const unsigned char *source,
                                                         uint64_t src_bytes,
                                                         uint32_t type,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim) {
    for (p36_vk_packed_weight *w = p36_vk.packed_weights; w; w = w->next) {
        if (w->source == source && w->src_bytes == src_bytes && w->type == type &&
            w->in_dim == in_dim && w->out_dim == out_dim) {
            return w->tensor;
        }
    }

    const uint32_t blocks = in_dim / P36_VK_QK_K;
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    const uint64_t pack_block_bytes = p36_vk_kquant_mmq_block_bytes(type);
    const uint64_t src_row_bytes = p36_vk_kquant_row_bytes(type) * (uint64_t)blocks;
    const uint64_t packed_bytes = (uint64_t)row_tiles * blocks * pack_block_bytes;
    if (pack_block_bytes == 0 || blocks == 0 || row_tiles == 0) return NULL;

    p36_gpu_tensor *tensor = p36_vk_arena_alloc_unlocked(packed_bytes);
    if (!tensor) return NULL;
    uint64_t offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
    if (!root) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    if (type == P36_VK_TENSOR_Q5_K) {
        p36_vk_pack_q5_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else if (type == P36_VK_TENSOR_Q6_K) {
        p36_vk_pack_q6_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    p36_vk_packed_weight *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    entry->source = source;
    entry->src_bytes = src_bytes;
    entry->type = type;
    entry->in_dim = in_dim;
    entry->out_dim = out_dim;
    entry->tensor = tensor;
    entry->next = p36_vk.packed_weights;
    p36_vk.packed_weights = entry;
    return tensor;
}

/* Expand a Q8_K activation panel once into packed f16 pairs (natural k
 * order, value = f16(d * q), the exact rounding the MMQ tile staging
 * applied).  Every 32-row output tile of a dense MMQ dispatch used to
 * re-dequantize the same panel; with the panel shared they stage the
 * token tile with plain vector copies.
 *
 * Panels live in a small persistent ring instead of the scratch pool:
 * a fresh scratch panel per matmul call sat on the retire list across
 * the whole command-buffer ring and pushed the 2k/4k peak up by half a
 * gigabyte.  Reusing a ring slot while an old batch still reads it is
 * safe because the write-after-read hazard inserts a barrier.  The
 * one-entry cache lets gate/up (same activation, same q8 write_gen)
 * share one panel and one expansion dispatch. */
enum { P36_VK_B16_RING = 6 };
static struct { p36_gpu_tensor *t; uint64_t bytes; } p36_vk_b16_ring[P36_VK_B16_RING];
static uint32_t p36_vk_b16_next;
static struct {
    const p36_gpu_tensor *root;
    uint64_t root_off;
    uint64_t write_gen;
    uint32_t n_tok;
    uint32_t blocks;
    p36_gpu_tensor *b16;
} p36_vk_b16_last;

static void p36_vk_b16_ring_destroy_unlocked(void) {
    for (uint32_t i = 0; i < P36_VK_B16_RING; i++) {
        if (p36_vk_b16_ring[i].t) p36_vk_tensor_release_unlocked(p36_vk_b16_ring[i].t);
        p36_vk_b16_ring[i].t = NULL;
        p36_vk_b16_ring[i].bytes = 0;
    }
    memset(&p36_vk_b16_last, 0, sizeof(p36_vk_b16_last));
    p36_vk_b16_next = 0;
}

