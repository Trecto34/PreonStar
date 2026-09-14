/* Isolated microbenchmark for rms_norm_rope_kv_qwen_quant.
 * Tests fused RMS norm + RoPE + quantized KV store (Q8_0 K / Q4_0 V)
 * during prefill (n_tok = 1024) and decode (n_tok = 1).
 */
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

static void run_bench(uint32_t n_tok, int iters) {
    const uint32_t n_head = 2, head_dim = 256;
    const uint32_t pos0 = 0;
    const uint32_t cap = 4096;
    const float eps = 1e-6f;
    const uint32_t k_row_bytes = n_head * (head_dim / 32u) * 34u; /* 544 */
    const uint32_t v_row_bytes = n_head * (head_dim / 32u) * 18u; /* 288 */
    const uint64_t rows = (uint64_t)n_head * n_tok;
    const uint64_t k_floats = rows * head_dim;
    const uint64_t v_floats = rows * head_dim;

    void *w_raw = NULL;
    if (posix_memalign(&w_raw, (size_t)getpagesize(), head_dim * sizeof(float)) != 0) return;
    float *w = (float *)w_raw;
    for (uint32_t i = 0; i < head_dim; i++) w[i] = 1.0f + 0.01f * (float)i;

    float *k_host = malloc(k_floats * sizeof(float));
    float *v_host = malloc(v_floats * sizeof(float));
    for (uint64_t i = 0; i < k_floats; i++) k_host[i] = 0.05f * ((int)(i % 23) - 11);
    for (uint64_t i = 0; i < v_floats; i++) v_host[i] = 0.03f * ((int)(i % 19) - 9);

    q36_gpu_tensor *k_t = q36_gpu_tensor_alloc(k_floats * sizeof(float));
    q36_gpu_tensor *v_t = q36_gpu_tensor_alloc(v_floats * sizeof(float));
    q36_gpu_tensor *kc = q36_gpu_tensor_alloc((uint64_t)cap * k_row_bytes);
    q36_gpu_tensor *vc = q36_gpu_tensor_alloc((uint64_t)cap * v_row_bytes);

    q36_gpu_tensor_write(k_t, 0, k_host, k_floats * sizeof(float));
    q36_gpu_tensor_write(v_t, 0, v_host, v_floats * sizeof(float));
    q36_gpu_set_quality(false);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        int ok = q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
            kc, vc, k_t, v_t,
            w_raw, head_dim * sizeof(float), 0,
            head_dim, n_head, pos0, n_tok, cap, eps,
            k_row_bytes, v_row_bytes);
        if (!ok) {
            fprintf(stderr, "q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor failed at n_tok=%u\n", n_tok);
            exit(1);
        }
    }

    uint8_t *kc_host = malloc((uint64_t)cap * k_row_bytes);
    q36_gpu_tensor_read(kc, 0, kc_host, (uint64_t)cap * k_row_bytes);

    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
            kc, vc, k_t, v_t,
            w_raw, head_dim * sizeof(float), 0,
            head_dim, n_head, pos0, n_tok, cap, eps,
            k_row_bytes, v_row_bytes);
    }
    q36_gpu_tensor_read(kc, 0, kc_host, (uint64_t)cap * k_row_bytes);
    uint8_t *vc_host = malloc((uint64_t)cap * v_row_bytes);
    q36_gpu_tensor_read(vc, 0, vc_host, (uint64_t)cap * v_row_bytes);
    double t1 = get_time_us();
    double avg_us = (t1 - t0) / iters;

    uint32_t khash = 2166136261u;
    for (uint64_t i = 0; i < (uint64_t)n_tok * k_row_bytes; i++)
        khash = (khash ^ kc_host[i]) * 16777619u;
    uint32_t vhash = 2166136261u;
    for (uint64_t i = 0; i < (uint64_t)n_tok * v_row_bytes; i++)
        vhash = (vhash ^ vc_host[i]) * 16777619u;

    printf("n_tok=%u (rows=%lu): avg latency = %.2f us (%d iters) [khash=%08x vhash=%08x]\n",
           n_tok, (unsigned long)rows, avg_us, iters, khash, vhash);

    free(kc_host);
    free(vc_host);

    q36_gpu_tensor_free(k_t);
    q36_gpu_tensor_free(v_t);
    q36_gpu_tensor_free(kc);
    q36_gpu_tensor_free(vc);
    free(k_host);
    free(v_host);
    free(w_raw);
}

int main(void) {
    if (!q36_gpu_init()) {
        fprintf(stderr, "q36_gpu_init failed\n");
        return 1;
    }

    printf("Benchmarking rms_norm_rope_kv_qwen_quant (n_head=2, head_dim=256):\n");
    run_bench(1, 100);    /* decode */
    run_bench(64, 50);    /* small batch / chunk */
    run_bench(512, 50);   /* medium prefill */
    run_bench(1024, 50);  /* 1k prefill chunk */

    q36_gpu_cleanup();
    return 0;
}
