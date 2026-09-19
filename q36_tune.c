/*
 * q36_tune.c – minimal offline kernel-variant auto-tuner for the
 * dense_extra_mmq (PTQ1_0) GEMM kernel.
 *
 * Design follows Splash’s LinearTuning.cpp:
 *   1. Enumerate legal tile / subgroup / LDS variants.
 *   2. Compile each variant (glslc –D flags) to a .spv.
 *   3. Generate a deterministic test case (fixed PRNG seed) and obtain a
 *      CPU reference result using the existing dequant + naive GEMM path.
 *   4. For every variant that passes byte‑exact correctness:
 *        – run the kernel 32×, take the median elapsed time.
 *   5. A variant is promoted only if it beats the current baseline by ≥5 %.
 *
 * The file is *compile‑only* at this stage.  The GPU timing loop is guarded
 * by ENABLE_GPU_TUNING so that the first real GPU run requires an explicit
 * review gate.
 */

#include "q36.h"
#include "q36_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Variant description – mirrors the compile‑time constants used in     */
/* vulkan/dense_extra_mmq.comp (BM, BN, BK, STRIDE, local_size_x).    */
/* ------------------------------------------------------------------ */
struct tune_variant {
    const char *name;          /* short tag, used in filenames        */
    uint32_t    BM;            /* rows per work‑group (output dim)    */
    uint32_t    BN;            /* cols per work‑group (token dim)     */
    uint32_t    BK;            /* K‑tile size                         */
    uint32_t    local_size_x;  /* work‑group size (must be multiple of 32) */
};

/* ------------------------------------------------------------------ */
/* Plausible variants for the PTQ1_0 kernel.  The baseline (v0) is the */
/* currently shipped configuration (BM=32, BN=128, BK=32, 128 threads). */
/* ------------------------------------------------------------------ */
static const struct tune_variant variants[] = {
    { "v0_baseline",   32, 128, 32, 128 },
    { "v1_bm16",       16, 128, 32, 128 },
    { "v2_bn64",       32,  64, 32, 128 },
    { "v3_bk16",       32, 128, 16, 128 },
    { "v4_bm64",       64, 128, 32, 256 },
    { "v5_bn256",      32, 256, 32, 256 },
};
static const size_t n_variants = sizeof(variants) / sizeof(variants[0]);

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Compile a single variant with glslc.  Returns 0 on success. */
static int compile_variant(const struct tune_variant *v, char *out_path, size_t out_sz)
{
    snprintf(out_path, out_sz,
             "vulkan/dense_extra_mmq_%s.spv", v->name);

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "glslc -D BM=%u -D BN=%u -D BK=%u "
             "-D LOCAL_SIZE_X=%u "
             "-o %s vulkan/dense_extra_mmq.comp",
             v->BM, v->BN, v->BK, v->local_size_x, out_path);

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "tune: glslc failed for %s (rc=%d)\n", v->name, rc);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* CPU reference: dequantise the IQ2_XXS blocks with the public        */
/* dequant routine and run a naïve GEMM.  This mirrors the path used   */
/* by q36 when --dense-quant-model-rows is exercised.                 */
/* ------------------------------------------------------------------ */
static int cpu_reference_iq2_xxs(const q36_block_iq2_xxs *weights,
                                 const q36_block_q8_0 *q8,
                                 uint32_t out_dim, uint32_t n_tok,
                                 uint32_t blocks, float scale,
                                 float *out)          /* out[n_tok * out_dim] */
{
    /* Dequantise weight matrix to float[out_dim][blocks*QK_K] */
    size_t k = blocks * Q36_QK_K;
    float *w_f = malloc((size_t)out_dim * k * sizeof(float));
    float *q_f = malloc((size_t)n_tok * k * sizeof(float));
    if (!w_f || !q_f) { free(w_f); free(q_f); return -1; }

    for (uint32_t r = 0; r < out_dim; ++r)
        q36_dequantize_row_iq2_xxs(&weights[r * blocks], w_f + (size_t)r * k, k);
    for (uint32_t t = 0; t < n_tok; ++t)
        for (uint32_t b = 0; b < blocks; ++b)
            q36_quant_q8_0_to_f32(&q8[t * blocks + b], q_f + (size_t)t * k + b * Q36_QK_K);

    /* Naïve GEMM: out[t][r] = scale * sum_k w_f[r][k] * q_f[t][k] */
    for (uint32_t t = 0; t < n_tok; ++t)
        for (uint32_t r = 0; r < out_dim; ++r) {
            double acc = 0.0;
            for (size_t kk = 0; kk < k; ++kk)
                acc += (double)w_f[(size_t)r * k + kk] *
                       (double)q_f[(size_t)t * k + kk];
            out[(size_t)t * out_dim + r] = (float)(acc * scale);
        }

    free(w_f);
    free(q_f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Byte‑exact comparison                                               */
/* ------------------------------------------------------------------ */
static int compare_exact(const float *gpu, const float *cpu, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (gpu[i] != cpu[i])   /* bit‑wise equality required */
            return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* GPU dispatch – uses the internal Vulkan helper to load a one‑off    */
/* kernel from a .spv file.  The real q36_vulkan.c does not expose a   */
/* public “load arbitrary spv” API, so this function is a stub that    */
/* will be wired up after reviewer sign‑off.                           */
/* ------------------------------------------------------------------ */
#ifdef ENABLE_GPU_TUNING
static int gpu_run_variant(const struct tune_variant *v,
                           const char *spv_path,
                           const q36_block_iq2_xxs *weights,
                           const q36_block_q8_0 *q8,
                           uint32_t out_dim, uint32_t n_tok,
                           uint32_t blocks, float scale,
                           float *out)
{
    (void)v; (void)spv_path; (void)weights; (void)q8;
    (void)out_dim; (void)n_tok; (void)blocks; (void)scale; (void)out;
    /* TODO: create temporary q36_vk_kernel from spv_path,
       bind the same buffers as q36_gpu_matmul_iq_quant_q8_scaled_tensor,
       dispatch with work‑group counts derived from out_dim / BM,
       n_tok / BN, record elapsed time. */
    return -1;   /* not wired yet */
}
#endif

/* ------------------------------------------------------------------ */
/* Median of an array of uint64_t (nanoseconds)                        */
/* ------------------------------------------------------------------ */
static uint64_t median_u64(uint64_t *a, size_t n)
{
    /* simple in‑place quick‑select for median */
    size_t l = 0, r = n - 1, k = n / 2;
    while (l < r) {
        uint64_t pivot = a[(l + r) >> 1];
        size_t i = l, j = r;
        while (i <= j) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j) {
                uint64_t t = a[i]; a[i] = a[j]; a[j] = t;
                ++i; --j;
            }
        }
        if (k <= j) r = j;
        else if (k >= i) l = i;
        else break;
    }
    return a[k];
}

/* ------------------------------------------------------------------ */
/* Main tuning entry point                                            */
/* ------------------------------------------------------------------ */
int q36_tune_dense_extra_mmq(void)
{
    /* Fixed test problem – small enough to run quickly, deterministic */
    const uint32_t out_dim = 256;
    const uint32_t n_tok   = 128;
    const uint32_t blocks  = 32;            /* in_dim = blocks * 256 = 8192 */
    const float    scale   = 0.0123f;

    /* Allocate host buffers */
    size_t w_sz = (size_t)out_dim * blocks * sizeof(q36_block_iq2_xxs);
    size_t q_sz = (size_t)n_tok   * blocks * sizeof(q36_block_q8_0);
    size_t o_sz = (size_t)n_tok   * out_dim * sizeof(float);

    q36_block_iq2_xxs *h_weights = malloc(w_sz);
    q36_block_q8_0    *h_q8      = malloc(q_sz);
    float             *h_cpu_ref = malloc(o_sz);
    float             *h_gpu_out = malloc(o_sz);
    if (!h_weights || !h_q8 || !h_cpu_ref || !h_gpu_out) {
        fprintf(stderr, "tune: host OOM\n");
        return -1;
    }

    /* Deterministic PRNG (xorshift64*) */
    uint64_t rng = 0x9e3779b97f4a7c15ull;
    auto_rand64 = ^uint64_t(void) { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * 0x2545F4914F6CDD1Dull; };
    for (size_t i = 0; i < out_dim * blocks; ++i)
        for (int k = 0; k < 8; ++k) ((uint8_t*)&h_weights[i])[k] = (uint8_t)auto_rand64();
    for (size_t i = 0; i < n_tok * blocks; ++i)
        for (int k = 0; k < 16; ++k) ((uint8_t*)&h_q8[i])[k] = (uint8_t)auto_rand64();

    /* 1️⃣ CPU reference */
    if (cpu_reference_iq2_xxs(h_weights, h_q8, out_dim, n_tok, blocks, scale, h_cpu_ref) != 0) {
        fprintf(stderr, "tune: CPU reference failed\n");
        return -1;
    }
    printf("tune: CPU reference OK\n");

    /* 2️⃣ Iterate variants */
    const struct tune_variant *best = &variants[0];
    uint64_t best_median = UINT64_MAX;

    for (size_t vi = 0; vi < n_variants; ++vi) {
        const struct tune_variant *v = &variants[vi];
        char spv_path[256];
        if (compile_variant(v, spv_path, sizeof(spv_path)) != 0)
            continue;

        /* ---- GPU correctness + timing (guarded) ---- */
#ifdef ENABLE_GPU_TUNING
        uint64_t times[32];
        int ok = 1;
        for (int run = 0; run < 32; ++run) {
            uint64_t t0 = now_ns();
            int rc = gpu_run_variant(v, spv_path, h_weights, h_q8,
                                     out_dim, n_tok, blocks, scale, h_gpu_out);
            uint64_t t1 = now_ns();
            if (rc != 0) { ok = 0; break; }
            if (!compare_exact(h_gpu_out, h_cpu_ref, (size_t)n_tok * out_dim)) {
                fprintf(stderr, "tune: variant %s mismatch on run %d\n", v->name, run);
                ok = 0; break;
            }
            times[run] = t1 - t0;
        }
        if (!ok) {
            printf("tune: %s – correctness failure\n", v->name);
            continue;
        }
        uint64_t med = median_u64(times, 32);
        double speedup = (double)best_median / (double)med;
        printf("tune: %s median=%llu ns  speedup=%.2fx\n",
               v->name, (unsigned long long)med, speedup);
        if (speedup >= 1.05) {          /* ≥5 % faster */
            best = v;
            best_median = med;
        }
#else
        /* GPU path disabled – just report that we would test it */
        printf("tune: %s – compiled to %s (GPU timing disabled)\n", v->name, spv_path);
#endif
    }

    printf("tune: selected variant = %s\n", best->name);
    /* Emit a tiny static choice table for the runtime to consume later */
    printf("static const struct { const char *name; uint32_t BM,BN,BK,local; } q36_tune_choice = { \"%s\", %u,%u,%u,%u };\n",
           best->name, best->BM, best->BN, best->BK, best->local_size_x);

    free(h_weights);
    free(h_q8);
    free(h_cpu_ref);
    free(h_gpu_out);
    return 0;
}