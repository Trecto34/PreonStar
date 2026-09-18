#ifndef Q36_MMQ_CONTRACT_H
#define Q36_MMQ_CONTRACT_H

/* CPU emulation of the dense MMQ prefill kernel's intentionally-float16
 * arithmetic contract (vulkan/dense_extra_mmq.comp).
 *
 * The production kernel rounds every operand to binary16, keeps the activation
 * scale in binary16, and accumulates the running sum in binary16 with one
 * rounding per v_mul_f16/v_fma_f16. A float32 CPU reference therefore cannot
 * match it, and the difference is by design, not a kernel defect. This header
 * is the single implementation of that contract, shared by
 * test_mmq_q2_family_ref (the targeted oracle) and the CPU parity reference, so
 * there is only one place that can be wrong.
 *
 * The accumulation order below matches the kernel's q8_K / 8-slice / 16-element
 * loop and has been validated bit-exact against the GPU by the
 * --dense-quant-model-rows oracle.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* IEEE-754 binary16 -> binary32, from the format definition. */
static inline float q36_contract_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    const uint32_t man = (uint32_t)h & 0x3ffu;
    double v;
    if (exp == 0u) v = ldexp((double)man, -24);                  /* subnormal */
    else if (exp == 31u) v = man ? (double)NAN : (double)INFINITY;
    else v = ldexp((double)(man | 0x400u), (int)exp - 25);
    return (float)(sign ? -v : v);
}

/* Round to IEEE-754 binary16 (nearest, ties to even) and widen back to float.
 * The hardware _Float16 convert is used when the compiler provides it (every
 * finite input was verified bit-identical to the integer/double reference
 * below, including subnormals and the 65504/65520/65536 boundaries); the
 * software path is kept for portability. */
#if defined(__FLT16_MANT_DIG__) && !defined(Q36_MMQ_CONTRACT_SOFT_HALF)
static inline float q36_contract_round_half(double v) {
    return (float)(_Float16)v;
}
#else
static inline float q36_contract_round_half(double v) {
    double a, step, r;
    int e;
    if (isnan(v)) return (float)v;
    a = fabs(v);
    if (a >= 65520.0) return (float)(v < 0.0 ? -INFINITY : INFINITY);
    if (a < 6.103515625e-05) {           /* binary16 subnormal: ulp = 2^-24 */
        r = ldexp(nearbyint(ldexp(a, 24)), -24);
    } else {
        (void)frexp(a, &e);              /* a = m * 2^e, m in [0.5, 1) */
        step = ldexp(1.0, e - 11);       /* binary16 keeps 11 significand bits */
        r = nearbyint(a / step) * step;
        if (r >= 65536.0) return (float)(v < 0.0 ? -INFINITY : INFINITY);
    }
    return (float)(v < 0.0 ? -r : r);
}
#endif

/* One f16 multiply and one f16 fused multiply-add, each with a single rounding
 * as v_mul_f16 / v_fma_f16 do. Inputs are exact binary16 values, so the product
 * and sum are exact in double and only the final rounding is inexact. */
static inline float q36_contract_mul_half(float a, float b) {
    return q36_contract_round_half((double)a * (double)b);
}

static inline float q36_contract_fma_half(float a, float b, float c) {
    return q36_contract_round_half((double)a * (double)b + (double)c);
}

/* Production-equivalent f16 MMQ dot for the Q2_0 family (types 42 / 142).
 *
 * `row_ptr`  packed weight row. `xq` is a contiguous array of q8_K blocks with
 *            stride `xq_block_bytes`; the f32 block scale is the first field
 *            and the int8 quants start at `qs_offset` (the production and test
 *            q8_K layouts differ in both, so they are parameters).
 * `blk_bytes`, `blk_per_q8k`, `lane_shift`, `lane_mask` are the compile-time
 *            Q2 geometry (18/4/1/1 for Q2_0, 34/2/2/3 for PQ2_0).
 * `pc_scale` the per-tensor matmul scale, applied inside the kernel before the
 *            weight scale is rounded to f16 -- callers must not re-apply it. */
static inline float q36_contract_mmq_q2_dot(const uint8_t *row_ptr,
                                            const uint8_t *xq,
                                            uint32_t xq_block_bytes,
                                            uint32_t qs_offset,
                                            uint32_t q8k_blocks,
                                            uint32_t blk_bytes,
                                            uint32_t blk_per_q8k,
                                            uint32_t lane_shift,
                                            uint32_t lane_mask,
                                            float pc_scale) {
    float sum = 0.0f;
    uint32_t qb, slice, e, c;
    for (qb = 0; qb < q8k_blocks; qb++) {
        const uint8_t *xb = xq + (uint64_t)qb * xq_block_bytes;
        const int8_t *qs = (const int8_t *)(xb + qs_offset);
        float d_raw = 0.0f;
        float scale_in;
        memcpy(&d_raw, xb, sizeof(d_raw));
        scale_in = q36_contract_round_half((double)d_raw);
        for (slice = 0; slice < 8u; slice++) {
            const uint32_t qblock = blk_per_q8k * qb + (slice >> lane_shift);
            const uint8_t *blk = row_ptr + (uint64_t)qblock * blk_bytes;
            const uint16_t dbits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
            const float d = q36_contract_round_half(
                (double)q36_contract_half_to_float(dbits) * (double)pc_scale);
            const uint8_t *codes = blk + 2 + (slice & lane_mask) * 8u;
            for (e = 0; e < 16u; e++) {
                const uint32_t kw = e >> 3;
                const uint32_t p = (e & 7u) >> 1;
                const uint32_t q = e & 1u;
                const uint32_t bits = codes[kw * 4u + p];
                float a[2], bv[2];
                for (c = 0; c < 2u; c++) {
                    const uint32_t sub = 2u * q + c;
                    const uint32_t code = (bits >> (sub * 2u)) & 3u;
                    const int8_t qv =
                        qs[slice * 32u + kw * 16u + p * 4u + sub];
                    a[c] = q36_contract_mul_half(
                        q36_contract_round_half((double)(int)code - 1.0), d);
                    bv[c] = q36_contract_mul_half(
                        q36_contract_round_half((double)(int)qv), scale_in);
                }
                sum = q36_contract_fma_half(a[1], bv[1], sum);
                sum = q36_contract_fma_half(a[0], bv[0], sum);
            }
        }
    }
    return sum;
}

/* Precompute the f16-rounded activation values for one token row:
 * act[qb*256 + j] = round_half(round_half(qs[j]) * scale_in). */
static inline void q36_contract_mmq_q2_act(const uint8_t *xq,
                                           uint32_t xq_block_bytes,
                                           uint32_t qs_offset,
                                           uint32_t q8k_blocks,
                                           float *act) {
    uint32_t qb, j;
    for (qb = 0; qb < q8k_blocks; qb++) {
        const uint8_t *xb = xq + (uint64_t)qb * xq_block_bytes;
        const int8_t *qs = (const int8_t *)(xb + qs_offset);
        float d_raw = 0.0f;
        float scale_in;
        memcpy(&d_raw, xb, sizeof(d_raw));
        scale_in = q36_contract_round_half((double)d_raw);
        for (j = 0; j < 256u; j++) {
            act[(uint64_t)qb * 256u + j] =
                q36_contract_mul_half(q36_contract_round_half((double)qs[j]),
                                      scale_in);
        }
    }
}

/* Multi-token q2 dot: the weight decode (a[]) is shared across tokens and the
 * per-token activation values are the precomputed f16 values `act` (in_dim per
 * token). Per-token accumulation order is identical to
 * q36_contract_mmq_q2_dot, so a 1-token call with a freshly precomputed act is
 * bit-identical to the single-token routine. `sums` is caller scratch of n_tok
 * floats. */
static inline void q36_contract_mmq_q2_dot_multi(const uint8_t *row_ptr,
                                                 const float *act,
                                                 float *sums,
                                                 uint32_t n_tok,
                                                 uint32_t in_dim,
                                                 uint32_t q8k_blocks,
                                                 uint32_t blk_bytes,
                                                 uint32_t blk_per_q8k,
                                                 uint32_t lane_shift,
                                                 uint32_t lane_mask,
                                                 float pc_scale,
                                                 float *out,
                                                 uint32_t out_stride) {
    uint32_t qb, slice, e, t;
    for (t = 0; t < n_tok; t++) sums[t] = 0.0f;
    for (qb = 0; qb < q8k_blocks; qb++) {
        for (slice = 0; slice < 8u; slice++) {
            const uint32_t qblock = blk_per_q8k * qb + (slice >> lane_shift);
            const uint8_t *blk = row_ptr + (uint64_t)qblock * blk_bytes;
            const uint16_t dbits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
            const float d = q36_contract_round_half(
                (double)q36_contract_half_to_float(dbits) * (double)pc_scale);
            const uint8_t *codes = blk + 2 + (slice & lane_mask) * 8u;
            for (e = 0; e < 16u; e++) {
                const uint32_t kw = e >> 3;
                const uint32_t p = (e & 7u) >> 1;
                const uint32_t q = e & 1u;
                const uint32_t bits = codes[kw * 4u + p];
                const uint32_t base = (uint64_t)qb * 256u + slice * 32u +
                                      kw * 16u + p * 4u + 2u * q;
                const float a0 = q36_contract_mul_half(
                    q36_contract_round_half(
                        (double)(int)((bits >> (4u * q)) & 3u) - 1.0), d);
                const float a1 = q36_contract_mul_half(
                    q36_contract_round_half(
                        (double)(int)((bits >> (4u * q + 2u)) & 3u) - 1.0), d);
                for (t = 0; t < n_tok; t++) {
                    const float *arow = act + (uint64_t)t * in_dim;
                    sums[t] = q36_contract_fma_half(a1, arow[base + 1u], sums[t]);
                    sums[t] = q36_contract_fma_half(a0, arow[base + 0u], sums[t]);
                }
            }
        }
    }
    for (t = 0; t < n_tok; t++)
        out[(uint64_t)t * out_stride] = sums[t];
}

/* Geometry for a Q2_0-family type. Returns 0 for anything else. */
static inline uint32_t q36_contract_q2_geometry(uint32_t type,
                                                uint32_t *blk_bytes,
                                                uint32_t *blk_per_q8k,
                                                uint32_t *lane_shift,
                                                uint32_t *lane_mask) {
    if (type == 142u) {          /* PQ2_0: 128 weights / 34 B */
        *blk_bytes = 34u; *blk_per_q8k = 2u; *lane_shift = 2u; *lane_mask = 3u;
        return 1u;
    }
    if (type == 42u) {           /* Q2_0: 64 weights / 18 B */
        *blk_bytes = 18u; *blk_per_q8k = 4u; *lane_shift = 1u; *lane_mask = 1u;
        return 1u;
    }
    return 0u;
}

#endif /* Q36_MMQ_CONTRACT_H */
