/* Bounded decode-attention parity / D2 reproducer.
 *
 * The D2 experiment (--attn-span 256 at ctx 512) changed generated tokens at
 * step 93.  This harness isolates the attention kernels: one synthetic decode
 * step per position (n_tok=1, f16 K/V caches), run through
 * q36_gpu_attn_decode_tensor at one span setting.  Inputs are a fixed-seed
 * PRNG, identical across span settings, so two runs can be compared exactly.
 * Each run also computes a float64 CPU reference from the same inputs.
 *
 * span is cached process-wide by q36_vk_attn_span(), so run the binary once
 * per span and compare the dumped outputs (see attn_parity_cmp.py).
 *
 * Build (from repo root, after `make`):
 *   cc -O2 -o logs/bc250-sustained-20260918/attn_parity \
 *      logs/bc250-sustained-20260918/attn_parity.c \
 *      q36_gpu_core.o q36_vulkan.o q36_ssd.o q36_prompt_prefix.o \
 *      q36_image.o -lm -pthread -ldl -lvulkan
 * Run:
 *   ./logs/bc250-sustained-20260918/attn_parity 24 4 256 512 /tmp/attn-512.bin
 *   ./logs/bc250-sustained-20260918/attn_parity 24 4 256 256 /tmp/attn-256.bin
 */
#define _GNU_SOURCE
#include "q36_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t POS0S[] = { 511u, 512u, 767u, 768u, 1023u, 1024u, 1279u, 1280u };
#define NPOS (sizeof(POS0S) / sizeof(POS0S[0]))

static uint32_t rng = 0x9e3779b9u;
static uint32_t next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static float frnd(void) { return (float)((int32_t)(next_u32() % 2001u) - 1000) / 1000.0f; }

static float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u, e = (uint32_t)(h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t f;
    if (e == 0u) f = s << 31;
    else if (e == 31u) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e - 15u + 127u) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}
static uint16_t rnd_half(void) {
    uint32_t s = next_u32() & 1u;
    uint32_t e = 11u + (next_u32() % 8u);
    uint32_t m = next_u32() & 0x3ffu;
    return (uint16_t)((s << 15) | (e << 10) | m);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s n_head n_head_kv head_dim span out.bin\n", argv[0]); return 2; }
    uint32_t n_head = (uint32_t)atoi(argv[1]), n_head_kv = (uint32_t)atoi(argv[2]);
    uint32_t head_dim = (uint32_t)atoi(argv[3]);
    const char *span = argv[4];
    const char *outpath = argv[5];
    setenv("Q36_VK_ATTN_SPAN", span, 1);

    if (!q36_gpu_init()) { fprintf(stderr, "gpu init failed\n"); return 2; }
    size_t nq = (size_t)n_head * head_dim;
    uint32_t k_row = n_head_kv * head_dim * 2u, v_row = k_row;
    FILE *of = fopen(outpath, "wb");
    if (!of) { perror(outpath); return 2; }

    for (size_t pi = 0; pi < NPOS; pi++) {
        uint32_t pos0 = POS0S[pi], kv_max = pos0 + 1u;
        q36_gpu_tensor *q  = q36_gpu_tensor_alloc(nq * sizeof(float));
        q36_gpu_tensor *qg = q36_gpu_tensor_alloc(2u * nq * sizeof(float));
        q36_gpu_tensor *kc = q36_gpu_tensor_alloc((uint64_t)kv_max * k_row);
        q36_gpu_tensor *vc = q36_gpu_tensor_alloc((uint64_t)kv_max * v_row);
        q36_gpu_tensor *sc = q36_gpu_tensor_alloc((uint64_t)n_head * kv_max * sizeof(float));
        q36_gpu_tensor *oa = q36_gpu_tensor_alloc(nq * sizeof(float));
        if (!q || !qg || !kc || !vc || !sc || !oa) { fprintf(stderr, "alloc failed\n"); return 2; }

        float *hq = malloc(nq * sizeof(float));
        float *hqg = malloc(2u * nq * sizeof(float));
        uint16_t *hk = malloc((size_t)kv_max * k_row);
        uint16_t *hv = malloc((size_t)kv_max * v_row);
        for (size_t i = 0; i < nq; i++) { hq[i] = frnd(); hqg[i] = frnd(); hqg[nq + i] = frnd(); }
        for (size_t i = 0; i < (size_t)kv_max * k_row / 2u; i++) hk[i] = rnd_half();
        for (size_t i = 0; i < (size_t)kv_max * v_row / 2u; i++) hv[i] = rnd_half();
        q36_gpu_tensor_write(q, 0, hq, nq * sizeof(float));
        q36_gpu_tensor_write(qg, 0, hqg, 2u * nq * sizeof(float));
        q36_gpu_tensor_write(kc, 0, hk, (uint64_t)kv_max * k_row);
        q36_gpu_tensor_write(vc, 0, hv, (uint64_t)kv_max * v_row);

        if (!q36_gpu_attn_decode_tensor(oa, q, qg, kc, vc, sc, NULL, 0, 0, false,
                                        pos0, 1, n_head, n_head_kv, head_dim,
                                        0, 0, k_row, v_row)) {
            fprintf(stderr, "pos0=%u attn_decode failed\n", pos0); return 2;
        }
        float *out = malloc(nq * sizeof(float));
        q36_gpu_tensor_read(oa, 0, out, nq * sizeof(float));
        fwrite(out, sizeof(float), nq, of);

        /* float64 CPU reference from the same f16 inputs */
        uint32_t ratio = n_head / n_head_kv;
        double maxerr = 0.0;
        for (uint32_t h = 0; h < n_head; h++) {
            uint32_t kvh = h / ratio;
            double m = -INFINITY;
            double *sc_ref = malloc((size_t)kv_max * sizeof(double));
            for (uint32_t t = 0; t < kv_max; t++) {
                double s = 0.0;
                for (uint32_t i = 0; i < head_dim; i++)
                    s += (double)hq[h * head_dim + i] *
                         half_to_float(hk[(size_t)t * k_row / 2u + kvh * head_dim + i]);
                s /= sqrt((double)head_dim);
                sc_ref[t] = s; if (s > m) m = s;
            }
            double l = 0.0;
            double *acc = calloc(head_dim, sizeof(double));
            for (uint32_t t = 0; t < kv_max; t++) {
                double p = exp(sc_ref[t] - m);
                l += p;
                for (uint32_t i = 0; i < head_dim; i++)
                    acc[i] += p * half_to_float(hv[(size_t)t * v_row / 2u + kvh * head_dim + i]);
            }
            for (uint32_t i = 0; i < head_dim; i++) {
                /* qg rows are [head][2*head_dim]; gate is the second half. */
                double g = hqg[(size_t)h * 2u * head_dim + head_dim + i];
                double ref = (acc[i] / l) * (1.0 / (1.0 + exp(-g)));
                double e = fabs(ref - (double)out[h * head_dim + i]);
                if (e > maxerr) maxerr = e;
            }
            free(sc_ref); free(acc);
        }
        printf("span=%s pos0=%-5u n_kv=%-5u maxabs_gpu_ref=%.3e\n", span, pos0, kv_max, maxerr);
        free(hq); free(hqg); free(hk); free(hv); free(out);
        q36_gpu_tensor_free(q); q36_gpu_tensor_free(qg); q36_gpu_tensor_free(kc);
        q36_gpu_tensor_free(vc); q36_gpu_tensor_free(sc); q36_gpu_tensor_free(oa);
    }
    fclose(of);
    q36_gpu_cleanup();
    return 0;
}
