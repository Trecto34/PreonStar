/* Standalone microbenchmark and parity test for matmul_q8_0_mm_f16 (vulkan/matmul_q8_0_mm_f16.comp).
 * Tests prefill Q8_0 GEMM performance and correctness in isolation without loading full model.
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

static void run_bench(uint32_t out_dim, uint32_t in_dim, uint32_t n_tok, int iters) {
    uint32_t blocks = in_dim / 32u;
    uint32_t row_bytes = blocks * 34u;
    uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    uint64_t in_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    uint8_t *w_buf = malloc(weight_bytes);
    float *x_host = malloc(in_bytes);
    float *out_host = malloc(out_bytes);
    float *ref_out = malloc(out_bytes);

    if (!w_buf || !x_host || !out_host || !ref_out) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    /* Fill weights: vary d per block and varied int8 qs */
    for (uint32_t r = 0; r < out_dim; r++) {
        uint8_t *row = w_buf + r * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = row + b * 34u;
            float d_f32 = (float)(((r + b) % 15) + 1) * 0.0009765625f;
            uint16_t d_fp16 = float_to_fp16_bits(d_f32);
            memcpy(blk, &d_fp16, 2);
            for (int i = 0; i < 32; i++) {
                int8_t q = (int8_t)(((r * 3 + b * 7 + i * 5) % 25) - 12);
                blk[2 + i] = (uint8_t)q;
            }
        }
    }

    /* Fill activations x */
    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) {
        x_host[i] = (float)(((int)(i % 17) - 8)) * 0.125f;
    }

    /* Compute CPU reference for first token / subset to check correctness */
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            float sum = 0.0f;
            const uint8_t *row = w_buf + r * row_bytes;
            const float *x_row = x_host + t * in_dim;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = row + b * 34u;
                float d_f32 = (float)(((r + b) % 15) + 1) * 0.0009765625f;
                for (int i = 0; i < 32; i++) {
                    int8_t q = (int8_t)blk[2 + i];
                    float w_val = (float)q * d_f32;
                    sum += w_val * x_row[b * 32 + i];
                }
            }
            ref_out[t * out_dim + r] = sum;
        }
    }

    q36_gpu_tensor *x_t = q36_gpu_tensor_alloc(in_bytes);
    q36_gpu_tensor *out_t = q36_gpu_tensor_alloc(out_bytes);
    q36_gpu_tensor_write(x_t, 0, x_host, in_bytes);

    q36_gpu_set_quality(false);

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        int ok = q36_gpu_matmul_q8_0_scaled_tensor(
            out_t, w_buf, weight_bytes, 0,
            in_dim, out_dim, x_t, n_tok, 1.0f);
        if (!ok) {
            fprintf(stderr, "q36_gpu_matmul_q8_0_scaled_tensor failed\n");
            exit(1);
        }
    }

    /* Read back & check parity */
    q36_gpu_tensor_read(out_t, 0, out_host, out_bytes);
    float max_err = 0.0f;
    for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
        float err = fabsf(out_host[i] - ref_out[i]);
        if (err > max_err) max_err = err;
    }

    /* Measure latency */
    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        q36_gpu_matmul_q8_0_scaled_tensor(
            out_t, w_buf, weight_bytes, 0,
            in_dim, out_dim, x_t, n_tok, 1.0f);
    }
    q36_gpu_tensor_read(out_t, 0, out_host, out_bytes);
    double t1 = get_time_us();
    double avg_us = (t1 - t0) / iters;
    double avg_ms = avg_us / 1000.0;

    uint32_t hash = 2166136261u;
    for (uint64_t i = 0; i < out_bytes; i++) {
        hash = (hash ^ ((uint8_t *)out_host)[i]) * 16777619u;
    }

    printf("shape [%u x %u x %u]: avg = %.3f ms (%d iters) [max_err=%.6f hash=%08x]\n",
           out_dim, in_dim, n_tok, avg_ms, iters, max_err, hash);

    q36_gpu_tensor_free(x_t);
    q36_gpu_tensor_free(out_t);
    free(w_buf);
    free(x_host);
    free(out_host);
    free(ref_out);
}

int main(void) {
    if (!q36_gpu_init()) {
        fprintf(stderr, "q36_gpu_init failed\n");
        return 1;
    }

    printf("--- matmul_q8_0_mm_f16 microbenchmark ---\n");
    /* Qwen 3.6 35B matrix dimensions for prefill (n_tok = 1024) */
    run_bench(2048, 2048, 1024, 20);
    run_bench(7168, 2048, 1024, 20);
    run_bench(2048, 7168, 1024, 20);

    q36_gpu_cleanup();
    return 0;
}
