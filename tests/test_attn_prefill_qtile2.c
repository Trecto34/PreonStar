/* Isolated microbenchmark for attn_prefill_qtile2.
 * Tests GQA-tiled prefill attention with Q8_0 K / Q4_0 V at context 2048.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "../q36.h"
#include "../q36_gpu.h"

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static uint32_t kv_row_bytes(uint32_t type, uint32_t n) {
    uint32_t blocks = (n + 31u) / 32u;
    if (type == 0) return n * 2u;
    if (type == 1) return blocks * 34u;
    if (type == 2) return blocks * 18u;
    return 0;
}

static void run_bench(uint32_t n_tok, uint32_t pos0, int iters) {
    const uint32_t n_head = 16, n_head_kv = 2, head_dim = 256;
    const uint32_t kv_max = pos0 + n_tok;
    const uint32_t cache_row = n_head_kv * head_dim;
    const uint32_t k_type = 1;
    const uint32_t v_type = 2;
    const uint32_t k_row_bytes = kv_row_bytes(k_type, cache_row);
    const uint32_t v_row_bytes = kv_row_bytes(v_type, cache_row);
    const uint64_t q_floats = (uint64_t)n_tok * n_head * head_dim;
    const uint64_t qg_floats = q_floats * 2u;
    const uint64_t out_floats = q_floats;
    const uint64_t sinks_alloc = 4096;

    void *sinks_raw = NULL;
    if (posix_memalign(&sinks_raw, (size_t)getpagesize(), (size_t)sinks_alloc) != 0) return;
    float *sinks = (float *)sinks_raw;
    memset(sinks, 0, (size_t)sinks_alloc);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.5f;

    float *q_host = malloc(q_floats * sizeof(float));
    float *qg_host = malloc(qg_floats * sizeof(float));
    uint8_t *kc_host = calloc(kv_max, k_row_bytes);
    uint8_t *vc_host = calloc(kv_max, v_row_bytes);
    float *out_host = malloc(out_floats * sizeof(float));

    for (uint64_t i = 0; i < q_floats; i++) q_host[i] = 0.1f * ((int)(i % 17) - 8);
    for (uint64_t i = 0; i < qg_floats; i++) qg_host[i] = 0.05f * ((int)(i % 19) - 9);

    /* Fill K/V cache with valid quantized data */
    uint16_t d_fp16 = 0x2e66; /* ~0.05f in fp16 */
    for (uint32_t k = 0; k < kv_max; k++) {
        uint8_t *k_row = kc_host + k * k_row_bytes;
        for (uint32_t b = 0; b < cache_row / 32u; b++) {
            uint8_t *blk = k_row + b * 34u;
            memcpy(blk, &d_fp16, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(((k * 13 + b * 7 + i * 3) % 25) - 12);
        }
        uint8_t *v_row = vc_host + k * v_row_bytes;
        for (uint32_t b = 0; b < cache_row / 32u; b++) {
            uint8_t *blk = v_row + b * 18u;
            memcpy(blk, &d_fp16, 2);
            for (int i = 0; i < 16; i++) blk[2 + i] = (uint8_t)((k * 11 + b * 5 + i * 7) % 256);
        }
    }

    q36_gpu_tensor *q = q36_gpu_tensor_alloc(q_floats * sizeof(float));
    q36_gpu_tensor *qg = q36_gpu_tensor_alloc(qg_floats * sizeof(float));
    q36_gpu_tensor *kc = q36_gpu_tensor_alloc((uint64_t)kv_max * k_row_bytes);
    q36_gpu_tensor *vc = q36_gpu_tensor_alloc((uint64_t)kv_max * v_row_bytes);
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(out_floats * sizeof(float));

    q36_gpu_tensor_write(q, 0, q_host, q_floats * sizeof(float));
    q36_gpu_tensor_write(qg, 0, qg_host, qg_floats * sizeof(float));
    q36_gpu_tensor_write(kc, 0, kc_host, (uint64_t)kv_max * k_row_bytes);
    q36_gpu_tensor_write(vc, 0, vc_host, (uint64_t)kv_max * v_row_bytes);
    q36_gpu_set_model_map(sinks_raw, sinks_alloc);

    q36_gpu_set_quality(false);

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        int ok = q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL,
                                             sinks_raw, sinks_alloc, 0, true,
                                             pos0, n_tok, n_head, n_head_kv,
                                             head_dim, k_type, v_type,
                                             k_row_bytes, v_row_bytes);
        if (!ok) {
            fprintf(stderr, "Warmup failed at n_tok=%u pos0=%u\n", n_tok, pos0);
            exit(1);
        }
    }

    q36_gpu_tensor_read(out, 0, out_host, out_floats * sizeof(float));

    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL,
                                    sinks_raw, sinks_alloc, 0, true,
                                    pos0, n_tok, n_head, n_head_kv,
                                    head_dim, k_type, v_type,
                                    k_row_bytes, v_row_bytes);
    }
    q36_gpu_tensor_read(out, 0, out_host, out_floats * sizeof(float));
    double t1 = get_time_us();
    double avg_us = (t1 - t0) / iters;

    uint32_t hash = 2166136261u;
    for (uint64_t i = 0; i < out_floats * sizeof(float); i++) {
        hash = (hash ^ ((uint8_t *)out_host)[i]) * 16777619u;
    }

    printf("n_tok=%u pos0=%u (ctx ~%u): avg = %.2f us (%d iters) [hash=%08x] sample=[%.4f, %.4f, %.4f, %.4f]\n",
           n_tok, pos0, pos0 + n_tok, avg_us, iters, hash,
           out_host[0], out_host[1], out_host[2], out_host[3]);

    q36_gpu_tensor_free(q);
    q36_gpu_tensor_free(qg);
    q36_gpu_tensor_free(kc);
    q36_gpu_tensor_free(vc);
    q36_gpu_tensor_free(out);
    free(q_host);
    free(qg_host);
    free(kc_host);
    free(vc_host);
    free(out_host);
    free(sinks_raw);
}

int main(void) {
    if (!q36_gpu_init()) {
        fprintf(stderr, "q36_gpu_init failed\n");
        return 1;
    }

    printf("--- attn_prefill_qtile2 microbenchmark ---\n");
    run_bench(7, 1024, 10);      /* odd tokens */
    run_bench(16, 1024, 10);     /* small chunk */
    run_bench(1024, 1024, 10);   /* ctx 2048 */
    run_bench(1024, 3072, 10);   /* ctx 4096 */

    q36_gpu_cleanup();
    return 0;
}
