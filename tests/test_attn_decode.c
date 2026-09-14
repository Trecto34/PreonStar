/* Isolated microbenchmark for attn_decode_split & attn_decode_fused.
 * Tests decode attention step with Q8_0 K and Q4_0 V at context 2048 and 4096.
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

static uint32_t kv_row_bytes(uint32_t type, uint32_t n) {
    uint32_t blocks = (n + 31u) / 32u;
    if (type == Q36_KV_CACHE_F16) return n * 2u;
    if (type == Q36_KV_CACHE_Q8_0) return blocks * 34u;
    if (type == Q36_KV_CACHE_Q4_0) return blocks * 18u;
    return 0;
}

static void run_bench(uint32_t pos0, int iters) {
    const uint32_t n_head = 16, n_head_kv = 2, head_dim = 256;
    const uint32_t n_tok = 1;
    const uint32_t kv_max = pos0 + n_tok;
    const uint32_t cache_row = n_head_kv * head_dim;
    const uint32_t k_type = Q36_KV_CACHE_Q8_0;
    const uint32_t v_type = Q36_KV_CACHE_Q4_0;
    const uint32_t k_row_bytes = kv_row_bytes(k_type, cache_row);
    const uint32_t v_row_bytes = kv_row_bytes(v_type, cache_row);
    const uint64_t q_floats = (uint64_t)n_tok * n_head * head_dim;
    const uint64_t qg_floats = q_floats * 2u;
    const uint64_t out_floats = q_floats;
    const uint64_t sinks_alloc = 4096;

    void *sinks_raw = NULL;
    if (posix_memalign(&sinks_raw, (size_t)getpagesize(), (size_t)sinks_alloc) != 0) return;
    float *sinks = sinks_raw;
    memset(sinks, 0, (size_t)sinks_alloc);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.5f;

    float *q_host = malloc(q_floats * sizeof(float));
    float *qg_host = malloc(qg_floats * sizeof(float));
    uint8_t *kc_host = calloc(kv_max, k_row_bytes);
    uint8_t *vc_host = calloc(kv_max, v_row_bytes);
    float *out_host = malloc(out_floats * sizeof(float));

    for (uint64_t i = 0; i < q_floats; i++) q_host[i] = 0.1f * ((int)(i % 17) - 8);
    for (uint64_t i = 0; i < qg_floats; i++) qg_host[i] = 0.05f * ((int)(i % 19) - 9);

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
    for (int i = 0; i < 5; i++) {
        q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL,
                                   sinks_raw, sinks_alloc, 0, true,
                                   pos0, n_tok, n_head, n_head_kv,
                                   head_dim, k_type, v_type,
                                   k_row_bytes, v_row_bytes);
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

    printf("pos0=%u (ctx ~%u): avg latency = %.2f us (%d iters)\n",
           pos0, pos0 + 1, avg_us, iters);

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

    printf("Benchmarking attn_decode (Q8_0 K / Q4_0 V, n_head=16, n_head_kv=2, head_dim=256):\n");
    run_bench(256, 50);   /* Fused (1 span) */
    run_bench(1024, 50);  /* Split (2 spans) */
    run_bench(2048, 50);  /* Split (4 spans) */
    run_bench(4096, 50);  /* Split (8 spans) */

    q36_gpu_cleanup();
    return 0;
}
