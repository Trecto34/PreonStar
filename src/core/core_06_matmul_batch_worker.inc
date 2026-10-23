static void p36_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_matmul_batch_ctx *ctx = (p36_matmul_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case P36_TENSOR_F32: {
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case P36_TENSOR_F16: {
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)p36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !p36_tensor_row_ptr(m, ctx->t, row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!p36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void p36_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_matmul_pair_batch_ctx *ctx = (p36_matmul_pair_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !p36_tensor_row_ptr(m, ctx->t0, row, &src0, NULL) ||
            !p36_tensor_row_ptr(m, ctx->t1, row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!p36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void p36_expert_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_expert_matmul_batch_ctx *ctx = (p36_expert_matmul_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case P36_TENSOR_F32: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case P36_TENSOR_F16: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)p36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !p36_tensor_expert_row_ptr(m, ctx->t, ctx->expert, (uint32_t)row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!p36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void p36_expert_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_expert_matmul_pair_batch_ctx *ctx = (p36_expert_matmul_pair_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !p36_tensor_expert_row_ptr(m, ctx->t0, ctx->expert, (uint32_t)row, &src0, NULL) ||
            !p36_tensor_expert_row_ptr(m, ctx->t1, ctx->expert, (uint32_t)row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!p36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void p36_vector_dot_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vector_dot_batch_ctx *ctx = (p36_vector_dot_batch_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        const float *xr = ctx->x + row * ctx->n;
        double acc = 0.0;
        for (uint32_t i = 0; i < ctx->n; i++) acc += (double)ctx->w[i] * (double)xr[i];
        ctx->out[row] = (float)acc;
    }
}

static void p36_scale_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_scale_rows_ctx *ctx = (p36_scale_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *xr = ctx->x + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) xr[i] *= ctx->scale;
    }
}

static void p36_add_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_add_rows_ctx *ctx = (p36_add_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *dr = ctx->dst + row * ctx->cols;
        const float *ar = ctx->a + row * ctx->cols;
        const float *br = ctx->b + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) dr[i] = ar[i] + br[i];
    }
}

static void p36_rms_norm_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_rms_norm_rows_ctx *ctx = (p36_rms_norm_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        p36_ref_rms_norm(ctx->dst + row * ctx->cols,
                         ctx->src + row * ctx->cols,
                         ctx->weight,
                         ctx->cols,
                         ctx->eps);
    }
}

static void p36_swiglu_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_swiglu_rows_ctx *ctx = (p36_swiglu_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *orow = ctx->out + row * ctx->cols;
        const float *grow = ctx->gate + row * ctx->cols;
        const float *urow = ctx->up + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++)
            orow[i] = grow[i] / (1.0f + expf(-grow[i])) * urow[i];
    }
}

/* Contract-aware whole-model matmul: emulates the dense MMQ kernel's f16
 * staging/accumulation (see p36_mmq_contract.h). Used only when
 * e->cpu_mmq_contract is set, i.e. the parity reference. The activation f16
 * values are precomputed once per token and the weight decode is shared across
 * tokens, which is what makes the scalar emulation affordable. */
typedef struct {
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;      /* NULL for a single projection */
    const float *act;
    float *out0;
    float *out1;
    uint32_t n_tok, in_dim, out_dim, q8k_blocks;
    uint32_t blk_bytes0, blk_per_q8k0, lane_shift0, lane_mask0;
    uint32_t blk_bytes1, blk_per_q8k1, lane_shift1, lane_mask1;
    float scale0, scale1;
} p36_contract_matmul_ctx;

static void p36_contract_matmul_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_contract_matmul_ctx *ctx = (p36_contract_matmul_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    float *sums = malloc((size_t)ctx->n_tok * sizeof(float));
    if (!sums) return;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0 = NULL;
        const uint8_t *src1 = NULL;
        if (!p36_tensor_row_ptr(m, ctx->t0, row, &src0, NULL)) continue;
        if (ctx->t1 && !p36_tensor_row_ptr(m, ctx->t1, row, &src1, NULL)) continue;
        p36_contract_mmq_q2_dot_multi(
            src0, ctx->act, sums, ctx->n_tok, ctx->in_dim, ctx->q8k_blocks,
            ctx->blk_bytes0, ctx->blk_per_q8k0, ctx->lane_shift0,
            ctx->lane_mask0, ctx->scale0, ctx->out0 + row, ctx->out_dim);
        if (ctx->t1) {
            p36_contract_mmq_q2_dot_multi(
                src1, ctx->act, sums, ctx->n_tok, ctx->in_dim, ctx->q8k_blocks,
                ctx->blk_bytes1, ctx->blk_per_q8k1, ctx->lane_shift1,
                ctx->lane_mask1, ctx->scale1, ctx->out1 + row, ctx->out_dim);
        }
    }
    free(sums);
}

static bool p36_contract_matmul_batch(const p36_engine *e,
                                      const p36_tensor *t0,
                                      const p36_tensor *t1,
                                      const uint8_t *xq,
                                      float *out0,
                                      float *out1,
                                      uint32_t n_tok,
                                      uint32_t in_dim,
                                      uint32_t out_dim,
                                      float scale0,
                                      float scale1) {
    p36_contract_matmul_ctx ctx;
    uint32_t bb0, bq0, ls0, lm0, bb1 = 0, bq1 = 0, ls1 = 0, lm1 = 0;
    uint32_t q8k_blocks, row_bytes;
    float *act;
    if (!e || !t0 || !xq || n_tok == 0 || in_dim == 0 || out_dim == 0 ||
        in_dim % P36_QK_K != 0) return false;
    if (!p36_contract_q2_geometry(t0->type, &bb0, &bq0, &ls0, &lm0)) return false;
    if (t1 && !p36_contract_q2_geometry(t1->type, &bb1, &bq1, &ls1, &lm1)) return false;
    q8k_blocks = in_dim / P36_QK_K;
    row_bytes = q8k_blocks * P36_Q8_K_BYTES;
    act = malloc((size_t)n_tok * in_dim * sizeof(float));
    if (!act) return false;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        p36_contract_mmq_q2_act(xq + (uint64_t)tok * row_bytes,
                                P36_Q8_K_BYTES, 4u, q8k_blocks,
                                act + (uint64_t)tok * in_dim);
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.act = act;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.q8k_blocks = q8k_blocks;
    ctx.blk_bytes0 = bb0; ctx.blk_per_q8k0 = bq0;
    ctx.lane_shift0 = ls0; ctx.lane_mask0 = lm0;
    ctx.blk_bytes1 = bb1; ctx.blk_per_q8k1 = bq1;
    ctx.lane_shift1 = ls1; ctx.lane_mask1 = lm1;
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    p36_parallel_for_rows(out_dim, 1, e->n_threads, p36_contract_matmul_worker, &ctx);
    free(act);
    return true;
}

static bool p36_tensor_matmul_batch_prequant(const p36_engine *e,
                                             const p36_tensor *t,
                                             const float *x,
                                             const uint8_t *xq,
                                             float *out,
                                             uint32_t n_tok,
                                             uint32_t in_dim,
                                             uint32_t out_dim,
                                             float scale) {
    p36_matmul_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 2 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    if (e->cpu_mmq_contract && n_tok > 1u && xq) {
        uint32_t bb, bq, ls, lm;
        if (p36_contract_q2_geometry(t->type, &bb, &bq, &ls, &lm))
            return p36_contract_matmul_batch(e, t, NULL, xq, out, NULL,
                                             n_tok, in_dim, out_dim, scale, 0.0f);
    }
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_matmul_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_matmul_pair_batch_prequant(const p36_engine *e,
                                                  const p36_tensor *t0,
                                                  const p36_tensor *t1,
                                                  const float *x,
                                                  const uint8_t *xq,
                                                  float *out0,
                                                  float *out1,
                                                  uint32_t n_tok,
                                                  uint32_t in_dim,
                                                  uint32_t out_dim,
                                                  float scale0,
                                                  float scale1) {
    p36_matmul_pair_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!p36_tensor_pair_can_fuse(t0, t1, 2, in_dim, out_dim)) return false;
    if (e->cpu_mmq_contract && n_tok > 1u && xq) {
        uint32_t bb, bq, ls, lm;
        if (p36_contract_q2_geometry(t0->type, &bb, &bq, &ls, &lm) &&
            p36_contract_q2_geometry(t1->type, &bb, &bq, &ls, &lm))
            return p36_contract_matmul_batch(e, t0, t1, xq, out0, out1,
                                             n_tok, in_dim, out_dim, scale0, scale1);
    }
    kind = p36_activation_quant_kind_for_type(t0->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_matmul_pair_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_expert_matmul_batch_prequant(const p36_engine *e,
                                                    const p36_tensor *t,
                                                    uint32_t expert,
                                                    const float *x,
                                                    const uint8_t *xq,
                                                    float *out,
                                                    uint32_t n_tok,
                                                    uint32_t in_dim,
                                                    uint32_t out_dim,
                                                    float scale) {
    p36_expert_matmul_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 3 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_expert_matmul_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_expert_matmul_pair_batch_prequant(const p36_engine *e,
                                                         const p36_tensor *t0,
                                                         const p36_tensor *t1,
                                                         uint32_t expert,
                                                         const float *x,
                                                         const uint8_t *xq,
                                                         float *out0,
                                                         float *out1,
                                                         uint32_t n_tok,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim,
                                                         float scale0,
                                                         float scale1) {
    p36_expert_matmul_pair_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!p36_tensor_pair_can_fuse(t0, t1, 3, in_dim, out_dim)) return false;
    kind = p36_activation_quant_kind_for_type(t0->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_expert_matmul_pair_batch_worker, &ctx);
    return true;
}

static void p36_tensor_vector_dot_batch(float *out,
                                        const float *w,
                                        const float *x,
                                        uint32_t n_tok,
                                        uint32_t n,
                                        uint32_t n_threads) {
    p36_vector_dot_batch_ctx ctx;
    if (!out || !w || !x || n_tok == 0) return;
    ctx.out = out;
    ctx.w = w;
    ctx.x = x;
    ctx.n_tok = n_tok;
    ctx.n = n;
    p36_parallel_for_rows(n_tok, 2, n_threads, p36_vector_dot_batch_worker, &ctx);
}

static P36_MAYBE_UNUSED void p36_scale_rows(float *x, uint32_t rows, uint32_t cols, float scale, uint32_t n_threads) {
    p36_scale_rows_ctx ctx;
    if (!x || scale == 1.0f || rows == 0) return;
    ctx.x = x;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.scale = scale;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_scale_rows_worker, &ctx);
}

static void p36_add_rows(float *dst,
                         const float *a,
                         const float *b,
                         uint32_t rows,
                         uint32_t cols,
                         uint32_t n_threads) {
    p36_add_rows_ctx ctx;
    if (!dst || !a || !b || rows == 0) return;
    ctx.dst = dst;
    ctx.a = a;
    ctx.b = b;
    ctx.rows = rows;
    ctx.cols = cols;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_add_rows_worker, &ctx);
}

static void p36_directional_steering_project_rows(float *x,
                                                   const float *dirs,
                                                   uint32_t layer,
                                                   uint32_t rows,
                                                   float scale) {
    if (!x || !dirs || rows == 0 || scale == 0.0f) return;
    const float *dir = dirs + (uint64_t)layer * P36_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * P36_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) dot += xr[i] * dir[i];
        float coeff = scale * dot;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) xr[i] -= coeff * dir[i];
    }
}

