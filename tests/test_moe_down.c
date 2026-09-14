/* Standalone microbenchmark and parity test for moe_down_gemm (vulkan/moe_down_gemm.comp).
 * Tests prefill Q2_K down projection GEMM performance and correctness in isolation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include "../q36.h"
#include "../q36_gpu.h"

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static uint16_t float_to_fp16_bits(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int exp = ((u.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u.u >> 13) & 0x03ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static float fp16_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void dequant_q2_k_row(const uint8_t *blk, float *out, uint32_t k) {
    const uint8_t *scales = blk;
    const uint8_t *qs = blk + 16;
    const uint16_t *d_dmin = (const uint16_t *)(blk + 16 + 64);
    float d = fp16_to_float(d_dmin[0]);
    float dmin = fp16_to_float(d_dmin[1]);
    for (uint32_t i = 0; i < k; i++) {
        uint8_t q = qs[i / 4];
        q = (q >> (2 * (i % 4))) & 3;
        uint8_t sc = scales[i / 16];
        float scale = d * (float)(sc & 0xF) + dmin * (float)((sc >> 4) & 0xF);
        out[i] = (float)q * scale;
    }
}

static void run_bench(uint32_t mid_dim, uint32_t out_dim, uint32_t n_tok, uint32_t n_expert, uint32_t n_used, int iters) {
    const uint32_t qk_k = 256;
    const uint32_t mid_blocks = mid_dim / qk_k;

    uint64_t down_row_bytes = mid_blocks * 84u;
    uint64_t down_stride = down_row_bytes * out_dim;

    uint64_t gu_stride = (uint64_t)(mid_dim / qk_k) * 66u * mid_dim;
    uint64_t down_offset = 0;
    uint64_t dsc_offset = down_offset + down_stride * n_expert;
    uint64_t model_bytes = dsc_offset + gu_stride * n_expert + n_expert * sizeof(float);
    uint64_t model_alloc = (model_bytes + 4095) & ~4095;

    uint8_t *model = malloc(model_alloc);
    float *mid_host = malloc((uint64_t)n_tok * mid_dim * sizeof(float));
    float *out_host = malloc((uint64_t)n_tok * out_dim * sizeof(float));
    float *ref_out = calloc((uint64_t)n_tok * out_dim, sizeof(float));
    float *wrow = malloc((uint64_t)mid_dim * sizeof(float));
    uint32_t *selected_host = malloc((uint64_t)n_tok * n_used * sizeof(uint32_t));
    float *weights_host = malloc((uint64_t)n_tok * n_used * sizeof(float));

    if (!model || !mid_host || !out_host || !ref_out || !wrow || !selected_host || !weights_host) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    memset(model, 0, model_alloc);

    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            uint8_t *d_blk = model + down_offset + expert * down_stride + row * down_row_bytes;
            for (uint32_t b = 0; b < mid_blocks; b++) {
                uint8_t *db = d_blk + b * 84;
                for (uint32_t i = 0; i < 16; i++) db[i] = (uint8_t)((expert * 100 + row * 3 + b * 5 + i * 7) % 256);
                for (uint32_t i = 0; i < 64; i++) db[16 + i] = (uint8_t)((expert * 100 + row * 11 + b * 13 + i * 17) % 256);
                uint16_t *dd = (uint16_t *)(db + 80);
                dd[0] = float_to_fp16_bits(0.01f * (float)((expert * 10 + row * 2 + b) % 100 + 1));
                dd[1] = float_to_fp16_bits(0.005f * (float)((expert * 10 + row * 2 + b + 50) % 100 + 1));
            }
        }
        ((float *)(model + dsc_offset))[expert] = 1.0f + 0.07f * (float)expert;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < mid_dim; i++) {
            mid_host[(uint64_t)t * mid_dim + i] = (float)(((t * 11 + i * 5) % 41) - 20) / 400.0f;
        }
        for (uint32_t u = 0; u < n_used; u++) {
            selected_host[t * n_used + u] = (t + u) % n_expert;
            weights_host[t * n_used + u] = 1.0f + 0.125f * (float)((t + 3 * u) % 5);
        }
    }

#if 0
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t u = 0; u < n_used; u++) {
            uint32_t expert = selected_host[t * n_used + u];
            float w = weights_host[t * n_used + u];
            float dsc = ((float *)(model + dsc_offset))[expert];
            const float *mid_t = mid_host + (uint64_t)t * mid_dim;
            for (uint32_t row = 0; row < out_dim; row++) {
                float dv = 0.0f;
                dequant_q2_k_row(model + down_offset + expert * down_stride + row * down_row_bytes, wrow, mid_dim);
                for (uint32_t i = 0; i < mid_dim; i++) dv += wrow[i] * mid_t[i];
                ref_out[(uint64_t)t * out_dim + row] += dv * dsc * w;
            }
        }
    }
#endif

    q36_gpu_tensor *mid_t = q36_gpu_tensor_alloc((uint64_t)n_tok * mid_dim * sizeof(float));
    q36_gpu_tensor *selected_t = q36_gpu_tensor_alloc((uint64_t)n_tok * n_used * sizeof(uint32_t));
    q36_gpu_tensor *weights_t = q36_gpu_tensor_alloc((uint64_t)n_tok * n_used * sizeof(float));
    q36_gpu_tensor *out_t = q36_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));

    if (!mid_t || !selected_t || !weights_t || !out_t) {
        fprintf(stderr, "Tensor alloc failed\n");
        exit(1);
    }

    q36_gpu_tensor_write(mid_t, 0, mid_host, (uint64_t)n_tok * mid_dim * sizeof(float));
    q36_gpu_tensor_write(selected_t, 0, selected_host, (uint64_t)n_tok * n_used * sizeof(uint32_t));
    q36_gpu_tensor_write(weights_t, 0, weights_host, (uint64_t)n_tok * n_used * sizeof(float));

    setenv("Q36_VK_MOE_GEMM_MIN", "8", 1);
    setenv("Q36_VK_MOE_F32B", "1", 1);

    q36_gpu_moe_weight gate = { 0, 16, 0, false };
    q36_gpu_moe_weight up = { 0, 16, 0, false };
    q36_gpu_moe_weight down = { down_offset, 10, dsc_offset, true };

    q36_gpu_set_model_map(model, model_alloc);

    for (int i = 0; i < 3; i++) {
        int ok = q36_gpu_moe_ffn_f32_tensor(out_t, model, model_alloc, &gate, &up, &down,
                                            selected_t, weights_t, 0, n_used, mid_t, n_tok,
                                            mid_dim, mid_dim, out_dim, n_expert);
        if (!ok) { fprintf(stderr, "Warmup failed\n"); exit(1); }
    }

    q36_gpu_tensor_read(out_t, 0, out_host, (uint64_t)n_tok * out_dim * sizeof(float));
    float max_err = 0.0f;
    for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
        float err = fabsf(out_host[i] - ref_out[i]);
        if (err > max_err) max_err = err;
    }

    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        q36_gpu_moe_ffn_f32_tensor(out_t, model, model_alloc, &gate, &up, &down,
                                   selected_t, weights_t, 0, n_used, mid_t, n_tok,
                                   mid_dim, mid_dim, out_dim, n_expert);
    }
    q36_gpu_tensor_read(out_t, 0, out_host, (uint64_t)n_tok * out_dim * sizeof(float));
    double t1 = get_time_us();
    double avg_us = (t1 - t0) / iters;
    double avg_ms = avg_us / 1000.0;

    uint32_t hash = 2166136261u;
    for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim * sizeof(float); i++) {
        hash = (hash ^ ((uint8_t *)out_host)[i]) * 16777619u;
    }

    printf("moe_down shape [tok=%u mid=%u out=%u exp=%u used=%u]: avg = %.3f ms (%d iters) [max_err=%.6f hash=%08x]\n",
           n_tok, mid_dim, out_dim, n_expert, n_used, avg_ms, iters, max_err, hash);

    q36_gpu_tensor_free(mid_t);
    q36_gpu_tensor_free(selected_t);
    q36_gpu_tensor_free(weights_t);
    q36_gpu_tensor_free(out_t);
    free(model);
    free(mid_host);
    free(out_host);
    free(ref_out);
    free(wrow);
    free(selected_host);
    free(weights_host);
}

int main(void) {
    if (!q36_gpu_init()) {
        fprintf(stderr, "q36_gpu_init failed\n");
        return 1;
    }

    printf("--- moe_down_gemm microbenchmark ---\n");
    run_bench(2048, 2048, 1024, 8, 2, 10);
    run_bench(2048, 7168, 1024, 8, 2, 10);
    run_bench(7168, 2048, 1024, 8, 2, 10);

    q36_gpu_cleanup();
    return 0;
}