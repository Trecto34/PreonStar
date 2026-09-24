/* Parity + cost for the two-row IQ3_XXS decode kernel
 * (vulkan/dense_iq3_xxs_decode_nx.comp, n_tok == 2) against two separate
 * one-token dispatches of dense_iq3_xxs_decode_r4.
 *
 * The gate is bit-exactness: the nx kernel unpacks each weight word once and
 * then walks, per row, the same FMA chain in the same order as the r4 kernel, so
 * row t of the n_tok == 2 dispatch must equal the n_tok == 1 dispatch of the same
 * activation row, bit for bit.  The real-weight path is exercised through
 * q36_gpu_matmul_iq_quant_q8_scaled_tensor with a host-backed model map.
 *
 * Weights are random bytes: grid entries are byte LUT indices, the sign and
 * scale fields are read as-is, and the row scale is a positive f16.  The q8_k
 * activation rows are written by hand in the layout add_rms_norm_q8_k.comp
 * produces (word 0 = f32 block scale, words 2..65 = 256 int8).
 *
 * Exit status is nonzero if any output bit differs.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "../q36.h"
#include "../q36_gpu.h"

#define IQ3_XXS 18u /* Q36_VK_TENSOR_IQ3_XXS, q36_vulkan.c:33 (not exported in q36_gpu.h) */
#define Q8K_WORDS 74u
#define BLOCK_BYTES 98u
#define QK_K 256u

static uint32_t rng = 987654321u;
static uint32_t rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

static uint16_t f16_pos(float f) { /* positive normal f16 */
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((((u >> 23) & 0xffu) - 112u) << 10 | ((u >> 13) & 0x3ffu));
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void make_weights(uint8_t *w, uint32_t out_dim, uint32_t blocks) {
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + ((size_t)r * blocks + b) * BLOCK_BYTES;
            uint16_t d = f16_pos(0.01f + 0.001f * (float)(rnd() & 63u));
            memcpy(blk, &d, 2);
            for (uint32_t i = 2; i < BLOCK_BYTES; i++) blk[i] = (uint8_t)rnd();
        }
    }
}

static void make_q8_row(uint8_t *row, uint32_t blocks) {
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t *w = (uint32_t *)(row + (size_t)b * Q8K_WORDS * 4u);
        float d = 0.013f + 0.001f * (float)(rnd() & 7u);
        memcpy(&w[0], &d, 4);
        w[1] = 0;
        for (uint32_t i = 0; i < 64u; i++) w[2 + i] = rnd() * 2654435761u;
        for (uint32_t i = 0; i < 8u; i++) w[66 + i] = 0;
    }
}

int main(int argc, char **argv) {
    /* Real shapes are out_dim 5120..17408 x in_dim 17408 (68 IQ3_XXS blocks a
     * row); the default is the attn_qkv shape. */
    const uint32_t out_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 5120u;
    const uint32_t in_dim = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 0) : 17408u;
    const uint32_t n_tok = 2u;
    const uint32_t blocks = in_dim / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * BLOCK_BYTES;
    const uint64_t weight_bytes = row_bytes * out_dim;
    const float scale = 1.0f;
    const int iters = argc > 1 ? atoi(argv[1]) : 200;

    uint8_t *wmap;
    if (posix_memalign((void **)&wmap, (size_t)getpagesize(), weight_bytes)) return 1;
    make_weights(wmap, out_dim, blocks);

    if (!q36_gpu_init()) { fprintf(stderr, "q36_gpu_init failed\n"); return 1; }
    q36_gpu_set_quality(false);
    q36_gpu_set_model_map(wmap, weight_bytes);

    /* q8 rows: one tensor per arm so both arms read the same bytes. */
    const uint64_t row_words = (uint64_t)blocks * Q8K_WORDS * 4u;
    q36_gpu_tensor *q8_1 = q36_gpu_tensor_alloc(row_words);
    q36_gpu_tensor *q8_2 = q36_gpu_tensor_alloc(row_words);
    q36_gpu_tensor *q8_12 = q36_gpu_tensor_alloc(row_words * 2u);
    q36_gpu_tensor *oa = q36_gpu_tensor_alloc(out_dim * 4u);
    q36_gpu_tensor *ob = q36_gpu_tensor_alloc(out_dim * 4u);
    q36_gpu_tensor *o2 = q36_gpu_tensor_alloc(2u * out_dim * 4u);
    q36_gpu_tensor *o2m = q36_gpu_tensor_alloc(2u * out_dim * 4u); /* MMQ tile, nx off */
    if (!q8_1 || !q8_2 || !q8_12 || !oa || !ob || !o2 || !o2m) { fprintf(stderr, "alloc failed\n"); return 1; }

    uint8_t *rows = malloc(row_words * 2u);
    if (!rows) return 1;
    make_q8_row(rows, blocks);
    make_q8_row(rows + row_words, blocks);
    q36_gpu_tensor_write(q8_1, 0, rows, row_words);
    q36_gpu_tensor_write(q8_2, 0, rows + row_words, row_words);
    q36_gpu_tensor_write(q8_12, 0, rows, row_words * 2u);

    int ok = 1;
    double ms[3] = { 0, 0, 0 };
    for (int arm = 0; arm < 3; arm++) {
        /* Arm 2 is the shipped n_tok == 2 path (the 128-row MMQ tile); it is not
         * the parity reference, only the source of the logit ULP difference that
         * explains why MTP acceptance counts move when nx takes over the
         * two-row verify: MMQ accumulates in f16 (P3), nx in f32. */
        if (arm == 2) setenv("Q36_VK_DENSE_IQ3_NX", "0", 1);
        double best = 1e30;
        for (int it = 0; it < iters; it++) {
            double t0 = now_ms();
            if (arm == 0) {
                ok &= q36_gpu_matmul_iq_quant_q8_scaled_tensor(oa, wmap, weight_bytes, 0, IQ3_XXS,
                                                               in_dim, out_dim, q8_1, 1, scale);
                ok &= q36_gpu_matmul_iq_quant_q8_scaled_tensor(ob, wmap, weight_bytes, 0, IQ3_XXS,
                                                               in_dim, out_dim, q8_2, 1, scale);
            } else if (arm == 1) {
                ok &= q36_gpu_matmul_iq_quant_q8_scaled_tensor(o2, wmap, weight_bytes, 0, IQ3_XXS,
                                                               in_dim, out_dim, q8_12, n_tok, scale);
            } else {
                ok &= q36_gpu_matmul_iq_quant_q8_scaled_tensor(o2m, wmap, weight_bytes, 0, IQ3_XXS,
                                                               in_dim, out_dim, q8_12, n_tok, scale);
            }
            /* Dispatches are recorded into an eager ring, so without a sync the
             * host time is pure record time. */
            ok &= q36_gpu_synchronize() != 0;
            if (it == 0) continue; /* warmup */
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        ms[arm] = best;
        if (!ok) { fprintf(stderr, "dispatch failed (arm %d)\n", arm); return 1; }
    }
    unsetenv("Q36_VK_DENSE_IQ3_NX");

    float *ra = calloc(out_dim, 4), *rb = calloc(out_dim, 4), *r2 = calloc(2u * out_dim, 4);
    float *r2m = calloc(2u * out_dim, 4);
    q36_gpu_tensor_read(oa, 0, ra, out_dim * 4u);
    q36_gpu_tensor_read(ob, 0, rb, out_dim * 4u);
    q36_gpu_tensor_read(o2, 0, r2, 2u * out_dim * 4u);
    q36_gpu_tensor_read(o2m, 0, r2m, 2u * out_dim * 4u);

    uint64_t mism = 0;
    double max_abs = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *ref = t ? rb : ra;
        for (uint32_t i = 0; i < out_dim; i++) {
            double d = fabs((double)ref[i] - (double)r2[t * out_dim + i]);
            if (!(d <= max_abs)) max_abs = d;
            if (memcmp(&ref[i], &r2[t * out_dim + i], 4) != 0) mism++;
        }
    }
    /* nx vs the shipped MMQ 2-row path: expected to differ in the last bits, not
     * to be equal.  Measured so the MTP acceptance drift has a named cause. */
    double mmq_max_abs = 0;
    uint32_t mmq_bit_diff = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < out_dim; i++) {
            double d = fabs((double)r2m[t * out_dim + i] - (double)r2[t * out_dim + i]);
            if (!(d <= mmq_max_abs)) mmq_max_abs = d;
            if (memcmp(&r2m[t * out_dim + i], &r2[t * out_dim + i], 4) != 0) mmq_bit_diff++;
        }
    }
    uint32_t mmq_argmax_moved = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t am = 0, an = 0;
        for (uint32_t i = 1; i < out_dim; i++) {
            if (r2m[t * out_dim + i] > r2m[t * out_dim + am]) am = i;
            if (r2[t * out_dim + i] > r2[t * out_dim + an]) an = i;
        }
        if (am != an) mmq_argmax_moved++;
    }
    printf("iq3_xxs nx2: out_dim=%u in_dim=%u blocks=%u iters=%d  mismatches=%llu max_abs=%.3g  %s\n",
           out_dim, in_dim, blocks, iters, (unsigned long long)mism, max_abs, mism ? "FAIL" : "ok");
    printf("   two one-token dispatches %8.3f ms   one two-token dispatch %8.3f ms   (%.2fx, %.3f vs %.3f ms/token)\n",
           ms[0], ms[1], ms[0] / ms[1], ms[0] / 2.0, ms[1] / 2.0);
    printf("   nx vs MMQ 2-row tile   %8.3f ms   bit-diff %u/%u  max_abs=%.3g  argmax moved %u/%u rows\n",
           ms[2], mmq_bit_diff, 2u * out_dim, mmq_max_abs, mmq_argmax_moved, n_tok);
    q36_gpu_cleanup();
    return mism != 0;
}
