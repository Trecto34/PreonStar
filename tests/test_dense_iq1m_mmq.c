/* IQ1_M prefill (the IQ1_M arm of vulkan/dense_extra_mmq.comp, n_tok > 1)
 * against n_tok one-token dispatches of dense_iq1_m (f32 accumulation).
 *
 * Not bit-exact by design: the mmq tile stages weights and activations as f16
 * and accumulates in f16 (P3), like every other dense mmq type.  The gate is
 * the RMS relative error, judged against the same number for IQ1_S.
 *
 * scale 0.37 catches a missing or doubled pc.scale; n_tok > 8 reaches mmq.
 * Usage: test_dense_iq1m_mmq [out_dim] [in_dim] [n_tok] [s]
 * A trailing "s" runs the same check on IQ1_S, whose mmq arm already ships:
 * that is the f16-staging floor this IQ1_M number is judged against.
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
#include "../q36.h"
#include "../q36_gpu.h"
#include "../q36_cli_args.h"

#define IQ1_M 29u /* Q36_VK_TENSOR_IQ1_M, q36_vulkan.c:39 */
#define Q8K_WORDS 74u
#define QK_K 256u

static uint32_t rng = 123456789u;
static uint32_t rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

static uint16_t f16_pos(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((((u >> 23) & 0xffu) - 112u) << 10 | ((u >> 13) & 0x3ffu));
}

int main(int argc, char **argv) {
    int cli_rc = q36_cli_reject_option_arguments(argc, argv, "test_dense_iq1m_mmq");
    if (cli_rc) return cli_rc;
    /* Default is the Swift-1.5 blk.13 ffn_gate shape, 256-token prefill chunk. */
    const uint32_t out_dim = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 0) : 17408u;
    const uint32_t in_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 5120u;
    const uint32_t n_tok = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 0) : 256u;
    const int iq1s = argc > 4 && argv[4][0] == 's';
    const uint32_t type = iq1s ? 19u : IQ1_M, BLOCK_BYTES = iq1s ? 50u : 56u;
    const uint32_t blocks = in_dim / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * BLOCK_BYTES;
    const uint64_t weight_bytes = row_bytes * out_dim;
    const uint64_t row_words = (uint64_t)blocks * Q8K_WORDS * 4u;

    uint8_t *w;
    if (n_tok < 9u || posix_memalign((void **)&w, (size_t)getpagesize(), weight_bytes)) return 1;
    for (uint64_t i = 0; i < weight_bytes; i += BLOCK_BYTES) {
        if (iq1s) { /* f16 d, then 32 qs + 8 qh words, all random */
            uint16_t d = f16_pos(0.01f + 0.001f * (float)(rnd() & 63u));
            memcpy(w + i, &d, 2);
            for (uint32_t j = 2; j < 50u; j++) w[i + j] = (uint8_t)rnd();
            continue;
        }
        for (uint32_t j = 0; j < 48u; j++) w[i + j] = (uint8_t)rnd();
        /* Four scale words: low 12 bits = 3-bit sub-scales, top nibbles = f16 d. */
        uint16_t d = f16_pos(0.01f + 0.001f * (float)(rnd() & 63u)), sc[4];
        for (uint32_t k = 0; k < 4u; k++)
            sc[k] = (uint16_t)((rnd() & 0x0fffu) | (((d >> (4u * k)) & 15u) << 12));
        memcpy(w + i + 48u, sc, 8);
    }

    if (!q36_gpu_init()) { fprintf(stderr, "q36_gpu_init failed\n"); return 1; }
    q36_gpu_set_quality(false);
    q36_gpu_set_model_map(w, weight_bytes);

    uint8_t *rows = malloc(row_words * n_tok);
    if (!rows) return 1;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t *q = (uint32_t *)(rows + t * row_words + (size_t)b * Q8K_WORDS * 4u);
            float qd = 0.013f + 0.001f * (float)(rnd() & 7u);
            memcpy(&q[0], &qd, 4);
            q[1] = 0;
            for (uint32_t i = 0; i < 64u; i++) q[2 + i] = rnd() * 2654435761u;
            for (uint32_t i = 0; i < 8u; i++) q[66 + i] = 0;
        }
    }
    q36_gpu_tensor *q8_1 = q36_gpu_tensor_alloc(row_words);
    q36_gpu_tensor *q8_n = q36_gpu_tensor_alloc(row_words * n_tok);
    q36_gpu_tensor *o1 = q36_gpu_tensor_alloc(out_dim * 4u);
    q36_gpu_tensor *on = q36_gpu_tensor_alloc((uint64_t)n_tok * out_dim * 4u);
    if (!q8_1 || !q8_n || !o1 || !on) { fprintf(stderr, "alloc failed\n"); return 1; }
    q36_gpu_tensor_write(q8_n, 0, rows, row_words * n_tok);

    int ok = q36_gpu_matmul_iq_quant_q8_scaled_tensor(on, w, weight_bytes, 0, type, in_dim,
                                                      out_dim, q8_n, n_tok, 0.37f);
    ok &= q36_gpu_synchronize() != 0;
    float *rn = calloc((size_t)n_tok * out_dim, 4), *ref = calloc(out_dim, 4);
    q36_gpu_tensor_read(on, 0, rn, (uint64_t)n_tok * out_dim * 4u);

    double max_abs = 0, max_ref = 0, se = 0, sr = 0;
    for (uint32_t t = 0; ok && t < n_tok; t++) {
        ok &= q36_gpu_tensor_write(q8_1, 0, rows + t * row_words, row_words);
        ok &= q36_gpu_matmul_iq_quant_q8_scaled_tensor(o1, w, weight_bytes, 0, type, in_dim,
                                                       out_dim, q8_1, 1, 0.37f);
        ok &= q36_gpu_synchronize() != 0;
        q36_gpu_tensor_read(o1, 0, ref, out_dim * 4u);
        for (uint32_t i = 0; i < out_dim; i++) {
            double d = fabs((double)ref[i] - (double)rn[(size_t)t * out_dim + i]);
            if (!(d <= max_abs)) max_abs = d; /* also catches NaN */
            if (fabs(ref[i]) > max_ref) max_ref = fabs(ref[i]);
            se += d * d;
            sr += (double)ref[i] * ref[i];
        }
    }
    if (!ok) { fprintf(stderr, "dispatch failed\n"); return 1; }
    double rel = max_abs / max_ref, rms = sqrt(se / sr);
    /* IQ1_S (shipped) measures rms_rel 0.0104 on these random inputs; 2x that
     * is the gate.  A wrong index, scale or delta bit is uncorrelated noise. */
    int pass = rms < 0.02;
    printf("%s mmq n_tok=%u out_dim=%u in_dim=%u: max_abs=%.3g max_ref=%.3g rel=%.3g rms_rel=%.3g %s\n",
           iq1s ? "iq1_s" : "iq1_m", n_tok, out_dim, in_dim, max_abs, max_ref, rel, rms, pass ? "ok" : "FAIL");
    q36_gpu_cleanup();
    return !pass;
}
