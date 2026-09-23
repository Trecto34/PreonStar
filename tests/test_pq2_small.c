/* Parity + timing for the Q2_0/PQ2_0 small-batch matmul (dense_extra_small_q2.comp).
 * For each shape and n_tok, the batched call must reproduce, bit for bit, n_tok
 * separate one-token calls (dense_extra_decode_pq2_0) on the same q8_K rows.
 * Random PQ2_0 weights; activations quantized on the GPU.  Batches above
 * Q36_VK_Q2_SMALL_MAX (default 16) still take the f16-accumulating mmq tile and
 * are timed only.  Exit status is nonzero on any checked mismatch. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../q36.h"
#include "../q36_gpu.h"

#define PQ2_0_TYPE 142u
#define PQ2_BLK 34u
#define PQ2_QK 128u

static uint32_t rng = 1234567u;
static uint32_t urand(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static float frand(void) { return (float)(urand() >> 8) / 8388608.0f - 1.0f; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* Batches the small kernel serves (<= Q36_VK_Q2_SMALL_MAX, default 16).  Larger
 * ones go to the f16-accumulating mmq tile and are only timed, not checked. */
static uint32_t small_max = 16;

static int run_case(uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, int iters) {
    const uint64_t row_bytes = (uint64_t)in_dim / PQ2_QK * PQ2_BLK;
    const uint64_t wbytes = row_bytes * out_dim;
    const uint64_t q8_tok_bytes = (uint64_t)in_dim / 256u * 296u;
    unsigned char *w = NULL;
    if (posix_memalign((void **)&w, (size_t)getpagesize(), (wbytes + 4095) & ~4095ull)) return 1;
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t b = 0; b < in_dim / PQ2_QK; b++) {
            unsigned char *blk = w + r * row_bytes + b * PQ2_BLK;
            uint16_t d = (uint16_t)(0x2000u + (urand() & 0x3ffu)); /* ~0.008..0.016 */
            memcpy(blk, &d, 2);
            for (uint32_t i = 2; i < PQ2_BLK; i++) blk[i] = (unsigned char)urand();
        }
    }
    float *x = malloc((size_t)n_tok * in_dim * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = frand();

    q36_gpu_tensor *xt = q36_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    q36_gpu_tensor *q8 = q36_gpu_tensor_alloc((uint64_t)n_tok * q8_tok_bytes);
    q36_gpu_tensor *yb = q36_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    q36_gpu_tensor *x1 = q36_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    q36_gpu_tensor *q81 = q36_gpu_tensor_alloc(q8_tok_bytes);
    q36_gpu_tensor *y1 = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    float *ob = malloc((size_t)n_tok * out_dim * sizeof(float));
    float *o1 = malloc((size_t)n_tok * out_dim * sizeof(float));
    int bad = 0;
    if (!xt || !q8 || !yb || !x1 || !q81 || !y1) { fprintf(stderr, "alloc failed\n"); return 1; }

    q36_gpu_tensor_write(xt, 0, x, (uint64_t)n_tok * in_dim * sizeof(float));
    bad |= !q36_gpu_quantize_q8_k_tensor(q8, xt, in_dim, n_tok);
    bad |= !q36_gpu_matmul_iq_quant_q8_scaled_tensor(yb, w, wbytes, 0, PQ2_0_TYPE, in_dim, out_dim,
                                                     q8, n_tok, 1.0f);
    bad |= !q36_gpu_synchronize();
    q36_gpu_tensor_read(yb, 0, ob, (uint64_t)n_tok * out_dim * sizeof(float));

    for (uint32_t t = 0; t < n_tok; t++) {
        q36_gpu_tensor_write(x1, 0, x + (uint64_t)t * in_dim, (uint64_t)in_dim * sizeof(float));
        bad |= !q36_gpu_quantize_q8_k_tensor(q81, x1, in_dim, 1);
        bad |= !q36_gpu_matmul_iq_quant_q8_scaled_tensor(y1, w, wbytes, 0, PQ2_0_TYPE, in_dim, out_dim,
                                                         q81, 1, 1.0f);
        bad |= !q36_gpu_synchronize();
        q36_gpu_tensor_read(y1, 0, o1 + (uint64_t)t * out_dim, (uint64_t)out_dim * sizeof(float));
    }
    if (bad) { fprintf(stderr, "gpu call failed\n"); return 1; }

    uint64_t mism = 0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
        if (memcmp(&ob[i], &o1[i], 4) != 0) mism++;
        float d = ob[i] - o1[i];
        if (d < 0) d = -d;
        if (d > max_abs) max_abs = d;
    }

    double t0 = now_ms();
    for (int it = 0; it < iters; it++)
        q36_gpu_matmul_iq_quant_q8_scaled_tensor(yb, w, wbytes, 0, PQ2_0_TYPE, in_dim, out_dim, q8, n_tok, 1.0f);
    q36_gpu_synchronize();
    double batched = (now_ms() - t0) / iters;
    t0 = now_ms();
    for (int it = 0; it < iters; it++)
        q36_gpu_matmul_iq_quant_q8_scaled_tensor(y1, w, wbytes, 0, PQ2_0_TYPE, in_dim, out_dim, q81, 1, 1.0f);
    q36_gpu_synchronize();
    double single = (now_ms() - t0) / iters;

    bool checked = n_tok <= small_max;
    printf("%-4s in=%5u out=%5u n_tok=%3u  mismatches=%llu max_abs=%.3g  batched %.3f ms  one-token %.3f ms  ratio %.2fx\n",
           !checked ? "mmq" : mism ? "FAIL" : "PASS", in_dim, out_dim, n_tok, (unsigned long long)mism, max_abs,
           batched, single, batched / single);

    q36_gpu_tensor_free(xt); q36_gpu_tensor_free(q8); q36_gpu_tensor_free(yb);
    q36_gpu_tensor_free(x1); q36_gpu_tensor_free(q81); q36_gpu_tensor_free(y1);
    free(x); free(ob); free(o1);
    /* w stays allocated: the weight cache is keyed by host pointer. */
    return checked && mism != 0;
}

int main(void) {
    const char *env = getenv("Q36_VK_Q2_SMALL_MAX");
    if (env && env[0]) small_max = (uint32_t)strtoul(env, NULL, 10);
    if (!q36_gpu_init()) { fprintf(stderr, "q36_gpu_init failed\n"); return 1; }
    q36_gpu_set_quality(false);
    int bad = 0;
    const uint32_t toks[] = { 1, 2, 3, 4, 7, 8, 9, 13, 16, 24, 32, 48, 64 };
    const uint32_t shapes[][2] = { { 5120, 17408 }, { 17408, 5120 }, { 5120, 1001 } };
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++)
        for (size_t t = 0; t < sizeof toks / sizeof *toks; t++)
            bad |= run_case(shapes[s][0], shapes[s][1], toks[t], 10);
    printf("%s\n", bad ? "FAILED" : "ALL PASS");
    return bad;
}
