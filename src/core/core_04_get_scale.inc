p36_think_mode p36_think_mode_for_context(p36_think_mode mode, int ctx_size) {
    if (mode == P36_THINK_MAX && ctx_size < (int)P36_THINK_MAX_MIN_CONTEXT) {
        return P36_THINK_HIGH;
    }
    return mode;
}

static uint64_t p36_metal_attention_scratch_bytes(
        uint32_t ctx, uint32_t cap,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
#ifdef P36_METAL
    uint64_t bytes;
    const char *gqa_env = getenv("P36_METAL_ATTN_GQA");
    const char *split_env = getenv("P36_METAL_ATTN_SPLIT");
    const bool gqa = !gqa_env || !gqa_env[0] || gqa_env[0] != '0';
    const bool split = !split_env || !split_env[0] || split_env[0] != '0';
    const bool compact = cache_type_k == P36_KV_CACHE_Q8_0 &&
                         cache_type_v == P36_KV_CACHE_Q4_0;

    if (!compact) return (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);

    /* Up to 1024 keys Metal uses the dense parallel kernel.  Above that,
     * normal multi-token prefill uses one partial per 4096-key GQA group;
     * a one-token tail/decode uses split-K partials.  Size for each path so
     * compacting this buffer cannot change kernel selection or numerics. */
    const uint32_t prefix = ctx < 1024u ? ctx : 1024u;
    const uint32_t prefix_rows = cap < prefix ? cap : prefix;
    bytes = (uint64_t)prefix_rows * P36_N_HEAD * prefix * sizeof(float);
    if (gqa) {
        uint64_t grouped = (uint64_t)cap * P36_N_HEAD *
            ((ctx + 4095u) / 4096u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (grouped > bytes) bytes = grouped;
    } else if (split) {
        uint64_t grouped = (uint64_t)cap * P36_N_HEAD *
            ((ctx + 511u) / 512u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (grouped > bytes) bytes = grouped;
    } else {
        uint64_t dense = (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);
        if (dense > bytes) bytes = dense;
    }
    if (split) {
        uint64_t tail = (uint64_t)P36_N_HEAD *
            ((ctx + 511u) / 512u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (tail > bytes) bytes = tail;
    } else {
        uint64_t dense_rows = (uint64_t)P36_N_HEAD * ctx * sizeof(float);
        if (dense_rows > bytes) bytes = dense_rows;
    }
    return bytes;
#else
    (void)cache_type_k;
    (void)cache_type_v;
    return (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);
#endif
}

static uint64_t p36_vulkan_scratch_bytes(
        p36_backend backend, uint32_t ctx, uint32_t cap,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
    uint64_t bytes = 0;
    uint64_t metal_attention_bytes = backend == P36_BACKEND_METAL ?
        p36_metal_attention_scratch_bytes(
            ctx, cap, cache_type_k, cache_type_v) : 0;
    bytes += (uint64_t)cap * P36_N_EMBD * 5u * sizeof(float);
    bytes += (uint64_t)P36_N_EMBD * sizeof(float);
    if (backend != P36_BACKEND_METAL) {
        bytes += (uint64_t)cap *
                 ((P36_N_EMBD + P36_QK_K - 1u) / P36_QK_K) *
                 P36_VK_Q8_K_BYTES;
    }
    bytes += (uint64_t)cap *
             (P36_N_HEAD * P36_N_HEAD_DIM * 4u +
              (backend == P36_BACKEND_METAL ? 0u : P36_N_HEAD_KV) *
                  (P36_N_HEAD_DIM + P36_N_VALUE_DIM)) *
             sizeof(float);
    bytes += (uint64_t)cap * P36_N_SSM_INNER *
             (backend == P36_BACKEND_METAL ? 1u :
              P36_MODEL_DENSE ? 0u : 3u) * sizeof(float);
    {
        uint64_t recurrent_aux_bytes = (uint64_t)cap *
            (P36_N_SSM_CONV_DIM + P36_N_SSM_DT_RANK * 4u) * sizeof(float);
        if ((!P36_MODEL_DENSE && backend != P36_BACKEND_METAL) ||
            (backend == P36_BACKEND_METAL &&
             metal_attention_bytes < recurrent_aux_bytes)) {
            bytes += recurrent_aux_bytes;
        }
    }
    if (backend == P36_BACKEND_METAL && P36_MODEL_DENSE) {
        const uint64_t dense_ffn_panel_bytes =
            (uint64_t)cap * P36_N_FF_SHARED * sizeof(float);
        const uint64_t reusable_panels = metal_attention_bytes /
            dense_ffn_panel_bytes > 2u ? 2u :
            metal_attention_bytes / dense_ffn_panel_bytes;
        /* Attention/recurrent scratch is dead before dense FFN starts.  Reuse
         * it for as many complete FFN panels as it can hold. */
        bytes += (2u - reusable_panels) * dense_ffn_panel_bytes;
    } else if (backend != P36_BACKEND_METAL) {
        bytes += (uint64_t)cap *
                 (P36_N_FF_SHARED * 2u +
                  (P36_MODEL_DENSE ? 0u :
                   P36_N_EXPERT + P36_N_EXPERT_USED * 2u + P36_N_EMBD + 1u)) *
                 sizeof(float);
    }
    bytes += (uint64_t)P36_N_VOCAB * sizeof(float);
    if (backend == P36_BACKEND_METAL) {
        bytes += metal_attention_bytes;
    } else {
        bytes += (uint64_t)cap * P36_N_HEAD *
                 ((ctx + 8191u) / 8192u) *
                 (P36_N_HEAD_DIM + 2u) * sizeof(float);
    }
    return bytes;
}

p36_context_memory p36_context_memory_estimate_configured(
        p36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
    p36_context_memory m = {0};
    uint32_t cpu_prefill_cap = p36_default_cpu_prefill_cap();
#ifndef P36_NO_GPU
    uint32_t gpu_prefill_cap = p36_default_gpu_prefill_cap();
#endif
    if (ctx_size <= 0) return m;
#ifndef P36_NO_GPU
    m.prefill_cap = p36_backend_uses_graph(backend) ? gpu_prefill_cap : cpu_prefill_cap;
#else
    m.prefill_cap = cpu_prefill_cap;
#endif
    if (prefill_chunk != 0) m.prefill_cap = prefill_chunk;
#ifndef P36_NO_GPU
    /* Size scratch for the chunk that will actually run.  The graph path
     * clamps the chunk for watchdog safety, so estimating against the
     * unclamped request overshoots by ~206 KB per token of the difference --
     * a 4096 request against a 1024 cap reserves an extra 634 MB that is
     * never touched, on a board with ~15 GiB of unified memory. */
    if (p36_backend_uses_graph(backend)) {
        m.prefill_cap = p36_engine_clamp_prefill_cap(m.prefill_cap);
    }
#endif
    m.raw_cap = p36_backend_uses_graph(backend) ? (uint32_t)ctx_size : 0;
    m.comp_cap = 0;
    if (p36_backend_uses_graph(backend)) {
        uint64_t full_layers = P36_N_LAYER / P36_FULL_ATTENTION_INTERVAL;
        uint64_t recurrent_layers = P36_N_LAYER - full_layers;
        uint64_t k_row = p36_kv_cache_row_bytes(cache_type_k,
                                                P36_N_HEAD_KV * P36_N_HEAD_DIM);
        uint64_t v_row = p36_kv_cache_row_bytes(cache_type_v,
                                                P36_N_HEAD_KV * P36_N_VALUE_DIM);
        m.raw_bytes = full_layers * (uint64_t)ctx_size * (k_row + v_row);
        m.compressed_bytes = recurrent_layers *
                             ((uint64_t)(P36_N_SSM_CONV - 1u) *
                                  P36_N_SSM_CONV_DIM * sizeof(float) +
                              (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE *
                                  P36_N_SSM_DT_RANK * sizeof(float));
        m.scratch_bytes = p36_vulkan_scratch_bytes(
            backend, (uint32_t)ctx_size, m.prefill_cap,
            cache_type_k, cache_type_v);
    } else {
        m.raw_bytes = (uint64_t)P36_N_LAYER * P36_N_HEAD_KV *
                      P36_N_HEAD_DIM * (uint64_t)ctx_size * sizeof(float);
        m.compressed_bytes = (uint64_t)P36_N_LAYER * P36_N_SSM_STATE *
                             P36_N_SSM_GROUP * sizeof(float);
        m.scratch_bytes = p36_cpu_prefill_scratch_bytes(m.prefill_cap);
    }
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

p36_context_memory p36_context_memory_estimate(p36_backend backend, int ctx_size) {
    return p36_context_memory_estimate_configured(
            backend, ctx_size, 0,
            p36_default_kv_cache_type_k(backend, false),
            p36_default_kv_cache_type_v(backend, false));
}

p36_context_memory p36_context_memory_estimate_with_prefill(p36_backend backend,
                                                            int ctx_size,
                                                            uint32_t prefill_chunk) {
    return p36_context_memory_estimate_configured(
            backend, ctx_size, prefill_chunk,
            p36_default_kv_cache_type_k(backend, false),
            p36_default_kv_cache_type_v(backend, false));
}

void p36_tokens_push(p36_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void p36_tokens_free(p36_tokens *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void p36_tokens_copy(p36_tokens *dst, const p36_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) p36_tokens_push(dst, src->v[i]);
}

bool p36_tokens_starts_with(const p36_tokens *tokens, const p36_tokens *prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

static float p36_f16_to_f32(uint16_t h) {
    return p36_quant_f16_to_f32(h);
}

static uint16_t p36_f32_to_f16(float f) {
    return p36_quant_f32_to_f16(f);
}

float p36_quant_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    float out;

    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t p36_quant_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    uint32_t w;
    uint32_t out;
    float base;

    memcpy(&w, &f, sizeof(w));
    base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    {
        const uint32_t shl1 = w + w;
        const uint32_t sign = w & UINT32_C(0x80000000);
        uint32_t bias = shl1 & UINT32_C(0xff000000);
        float bias_f;
        if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
        bias = (bias >> 1) + UINT32_C(0x07800000);
        memcpy(&bias_f, &bias, sizeof(bias_f));
        base += bias_f;
        memcpy(&out, &base, sizeof(out));
        out = ((out >> 13) & UINT32_C(0x00007c00)) +
              (out & UINT32_C(0x00000fff));
        return (uint16_t)((sign >> 16) |
                          (shl1 > UINT32_C(0xff000000) ? UINT16_C(0x7e00) : out));
    }
}

/* Compiled without fast-math: -ffast-math substitutes rcpss approximations
 * for the scale divisions (off by 1 ulp on some inputs, varying across
 * CPUs), which breaks bit-identity with the GPU quantization kernels. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")
#endif
void p36_quant_q8_0(const float *x, void *out, int64_t n) {
    p36_block_q8_0 *y = out;
    int64_t blocks = n / P36_QK8_0;
    if (!x || !out || n < 0 || n % P36_QK8_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        float amax = 0.0f;
        for (int j = 0; j < (int)P36_QK8_0; j++) {
            float a = fabsf(x[b * P36_QK8_0 + j]);
            if (a > amax) amax = a;
        }
        /* Double-rounded divisions: -ffast-math -march=native substitutes
         * rcpss approximations for f32 division, which is off by 1 ulp on
         * some inputs and varies across CPUs. Going through double keeps
         * the result correctly rounded, so the GPU twin can match bit for
         * bit. */
        float d = (float)((double)amax / 127.0);
        float inv = d ? (float)(1.0 / (double)d) : 0.0f;
        y[b].d = p36_quant_f32_to_f16(d);
        for (int j = 0; j < (int)P36_QK8_0; j++)
            y[b].qs[j] = (int8_t)nearbyintf(x[b * P36_QK8_0 + j] * inv);
    }
}

static void p36_quant_q4_0_impl(const float *x, void *out, int64_t n,
                               bool refine) {
    p36_block_q4_0 *y = out;
    int64_t blocks = n / P36_QK4_0;
    if (!x || !out || n < 0 || n % P36_QK4_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        const float *row = x + b * P36_QK4_0;
        float max = 0.0f;
        float amax = 0.0f;
        for (uint32_t i = 0; i < P36_QK4_0; i++) {
            float a = fabsf(row[i]);
            if (a > amax) {
                amax = a;
                max = row[i];
            }
        }
        float d = max / -8.0f;
        float inv = d ? (float)(1.0 / (double)d) : 0.0f;
        /* KV values favor reconstruction error over canonical Q4_0 bytes. */
        if (refine) {
            double num = 0.0;
            int den = 0;
            for (uint32_t i = 0; i < P36_QK4_0; i++) {
                int q = (int)(row[i] * inv + 8.5f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                q -= 8;
                num += (double)row[i] * (double)q;
                den += q * q;
            }
            if (den) {
                float fit = (float)(num / (double)den);
                d = d + (fit - d) * 0.25f;
                inv = d ? (float)(1.0 / (double)d) : 0.0f;
            }
        }
        y[b].d = p36_quant_f32_to_f16(d);
        for (uint32_t i = 0; i < P36_QK4_0 / 2u; i++) {
            int q0 = (int8_t)(row[i] * inv + 8.5f);
            int q1 = (int8_t)(row[i + P36_QK4_0 / 2u] * inv + 8.5f);
            if (refine && q0 < 0) q0 = 0;
            if (refine && q1 < 0) q1 = 0;
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            y[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void p36_quant_q4_0(const float *x, void *out, int64_t n) {
    p36_quant_q4_0_impl(x, out, n, false);
}

void p36_quant_q4_0_kv(const float *x, void *out, int64_t n) {
    p36_quant_q4_0_impl(x, out, n, true);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void p36_quant_q8_k(const float *x, void *out, int64_t n) {
    p36_block_q8_k *y = out;
    int64_t blocks = n / P36_QK_K;
    if (!x || !out || n < 0 || n % P36_QK_K) return;
    for (int64_t b = 0; b < blocks; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *row = x + b * P36_QK_K;
        for (int j = 0; j < (int)P36_QK_K; j++) {
            float a = fabsf(row[j]);
            if (a > amax) {
                amax = a;
                max = row[j];
            }
        }
        if (amax == 0.0f) {
            memset(&y[b], 0, sizeof(y[b]));
            continue;
        }
        float inv = -127.0f / max;
        for (int j = 0; j < (int)P36_QK_K; j++) {
            int v = (int)lrintf(inv * row[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < (int)P36_QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / inv;
    }
}

float p36_quant_dot_q8_0(const void *a, const void *b, int n) {
    const p36_block_q8_0 *x = a;
    const p36_block_q8_0 *y = b;
    float sum = 0.0f;
    if (!a || !b || n < 0 || n % P36_QK8_0) return 0.0f;
    for (int block = 0; block < n / (int)P36_QK8_0; block++) {
        int isum = 0;
        for (int i = 0; i < (int)P36_QK8_0; i++) isum += x[block].qs[i] * y[block].qs[i];
        sum += p36_f16_to_f32(x[block].d) * p36_f16_to_f32(y[block].d) * (float)isum;
    }
    return sum;
}

static float p36_ref_rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

static void p36_ref_rms_norm(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float scale;
    if (p36_float_norm_accum_mode) {
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
        scale = p36_ref_rsqrtf(ss / (float)n + eps);
    } else {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)src[i] * (double)src[i];
        scale = p36_ref_rsqrtf((float)(ss / (double)n) + eps);
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = src[i] * scale;
        dst[i] = weight ? v * weight[i] : v;
    }
}

static P36_MAYBE_UNUSED void p36_ref_matmul_f32(float *out, const float *w, const float *x, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t row = 0; row < out_dim; row++) {
        const float *wr = w + (uint64_t)row * in_dim;
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)wr[i] * (double)x[i];
        out[row] = (float)acc;
    }
}

static float p36_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float p36_siluf(float x) {
    return x / (1.0f + expf(-x));
}

static float p36_softplusf(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

static void p36_l2_norm(float *x, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    if (ss <= 0.0) return;
    {
        float scale = 1.0f / fmaxf(sqrtf((float)ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

static void p36_softmax_inplace(float *x, uint32_t n) {
    float maxv = -INFINITY;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) if (x[i] > maxv) maxv = x[i];
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum == 0.0) return;
    {
        float inv = (float)(1.0 / sum);
        for (uint32_t i = 0; i < n; i++) x[i] *= inv;
    }
}

static bool p36_all_finite(const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

static void p36_topk_indices(const float *x, uint32_t n, uint32_t k, uint32_t *idx, float *val) {
    for (uint32_t i = 0; i < k; i++) {
        idx[i] = 0;
        val[i] = -INFINITY;
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        for (uint32_t j = 0; j < k; j++) {
            if (v > val[j]) {
                for (uint32_t t = k - 1; t > j; t--) {
                    idx[t] = idx[t - 1];
                    val[t] = val[t - 1];
                }
                idx[j] = i;
                val[j] = v;
                break;
            }
        }
    }
}

