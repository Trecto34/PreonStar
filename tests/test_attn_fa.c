/* Parity + timing check for attn_prefill_fa_gqa{6,8} against attn_prefill_qtile2{_gqa6}.
 * Dense Qwen3.8 (24 q / 4 kv heads) and 35B-A3B MoE (16 q / 2 kv heads) shapes,
 * head_dim 256, Q8_0 K, Q4_0 V.  Both kernels run through the real q36_gpu_attn_decode_tensor
 * dispatch; Q36_VK_ATTN_FA selects which one (read per call).
 * Exit status is nonzero if any case exceeds the tolerance. */
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

static uint32_t rng = 12345u;
static float frand(void) { /* uniform [-1, 1) */
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 8) / 8388608.0f - 1.0f;
}

static uint16_t f16_bits(float f) { /* positive normal range only */
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((((u >> 23) & 0xff) - 112) << 10 | ((u >> 13) & 0x3ff));
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* One sinks buffer for the whole run: the weight cache is keyed by host pointer. */
static float *sinks;
static const size_t sinks_alloc = 4096;

static int run_case(uint32_t n_head, uint32_t n_head_kv, uint32_t n_tok, uint32_t pos0,
                    bool has_sinks, int iters) {
    const uint32_t hd = 256;
    const uint32_t kv_max = pos0 + n_tok, cache_row = n_head_kv * hd;
    const uint32_t k_rb = cache_row / 32u * 34u, v_rb = cache_row / 32u * 18u;
    const uint64_t qf = (uint64_t)n_tok * n_head * hd;

    float *qh = malloc(qf * 4), *qgh = malloc(qf * 8);
    uint8_t *kh = malloc((size_t)kv_max * k_rb), *vh = malloc((size_t)kv_max * v_rb);
    float *oa = malloc(qf * 4), *ob = malloc(qf * 4);
    /* q scaled so post-1/16 scores span a few units: softmax is peaked, not flat. */
    const char *qs_env = getenv("Q36_TEST_QSCALE"); /* stress peaked softmax */
    const float qs = qs_env ? (float)atof(qs_env) : 1.5f;
    for (uint64_t i = 0; i < qf; i++) qh[i] = qs * frand();
    for (uint64_t i = 0; i < 2 * qf; i++) qgh[i] = 2.0f * frand();
    for (uint32_t k = 0; k < kv_max; k++) {
        for (uint32_t b = 0; b < cache_row / 32u; b++) {
            uint8_t *kb = kh + (size_t)k * k_rb + b * 34u;
            uint16_t d = f16_bits(0.02f + 0.03f * (frand() + 1.0f));
            memcpy(kb, &d, 2);
            for (int i = 0; i < 32; i++) kb[2 + i] = (uint8_t)(int8_t)(127.0f * frand());
            uint8_t *vb = vh + (size_t)k * v_rb + b * 18u;
            d = f16_bits(0.05f + 0.05f * (frand() + 1.0f));
            memcpy(vb, &d, 2);
            for (int i = 0; i < 16; i++) vb[2 + i] = (uint8_t)(128.0f * frand() + 128.0f);
        }
    }
    q36_gpu_tensor *q = q36_gpu_tensor_alloc(qf * 4), *qg = q36_gpu_tensor_alloc(qf * 8);
    q36_gpu_tensor *kc = q36_gpu_tensor_alloc((uint64_t)kv_max * k_rb);
    q36_gpu_tensor *vc = q36_gpu_tensor_alloc((uint64_t)kv_max * v_rb);
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(qf * 4);
    q36_gpu_tensor_write(q, 0, qh, qf * 4);
    q36_gpu_tensor_write(qg, 0, qgh, qf * 8);
    q36_gpu_tensor_write(kc, 0, kh, (uint64_t)kv_max * k_rb);
    q36_gpu_tensor_write(vc, 0, vh, (uint64_t)kv_max * v_rb);
    q36_gpu_set_model_map(sinks, sinks_alloc);

    double ms[2];
    for (int arm = 0; arm < 2; arm++) {
        setenv("Q36_VK_ATTN_FA", arm ? "1" : "0", 1);
        double best = 1e30;
        for (int it = 0; it < iters + 1; it++) {
            double t0 = now_ms();
            if (!q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL, sinks, sinks_alloc, 0, has_sinks,
                                            pos0, n_tok, n_head, n_head_kv, hd, 1, 2, k_rb, v_rb)) {
                fprintf(stderr, "dispatch failed n_tok=%u pos0=%u arm=%d\n", n_tok, pos0, arm);
                exit(1);
            }
            q36_gpu_tensor_read(out, 0, arm ? ob : oa, qf * 4);
            double dt = now_ms() - t0;
            if (it && dt < best) best = dt; /* it 0 is warmup */
        }
        ms[arm] = best;
    }
    double max_abs = 0, max_ref = 0;
    for (uint64_t i = 0; i < qf; i++) {
        double d = fabs((double)oa[i] - ob[i]);
        if (!(d <= max_abs)) max_abs = d; /* NaN propagates as failure */
        if (fabs(oa[i]) > max_ref) max_ref = fabs(oa[i]);
    }
    int bad = !(max_abs <= 1e-4 * (max_ref > 1 ? max_ref : 1));
    printf("gqa%u n_tok=%4u pos0=%5u sinks=%d  qtile2 %8.2f ms  fa %8.2f ms  (%.2fx)  max_abs=%.3g max_ref=%.3g %s\n",
           n_head / n_head_kv, n_tok, pos0, has_sinks, ms[0], ms[1], ms[0] / ms[1], max_abs, max_ref, bad ? "FAIL" : "ok");
    q36_gpu_tensor_free(q); q36_gpu_tensor_free(qg); q36_gpu_tensor_free(kc);
    q36_gpu_tensor_free(vc); q36_gpu_tensor_free(out);
    free(qh); free(qgh); free(kh); free(vh); free(oa); free(ob);
    return bad;
}

int main(void) {
    /* Any of these off would route both arms to the same kernel. */
    unsetenv("Q36_VK_ATTN_QTILE2");
    unsetenv("Q36_VK_ATTN_QTILE");
    unsetenv("Q36_VK_ATTN_FUSED");
    if (posix_memalign((void **)&sinks, (size_t)getpagesize(), sinks_alloc)) return 1;
    for (uint32_t h = 0; h < 24; h++) sinks[h] = 2.0f * frand();
    if (!q36_gpu_init()) { fprintf(stderr, "q36_gpu_init failed\n"); return 1; }
    q36_gpu_set_quality(false);
    int bad = 0;
    for (int g = 0; g < 2; g++) {
        /* dense Qwen3.8 24/4, then 35B-A3B MoE 16/2 */
        uint32_t nh = g ? 16 : 24, nkv = g ? 2 : 4;
        bad |= run_case(nh, nkv, 2, 0, false, 3);        /* smallest prefill */
        bad |= run_case(nh, nkv, 7, 1024, true, 3);      /* ragged token tile */
        bad |= run_case(nh, nkv, 256, 0, false, 3);      /* first chunk */
        bad |= run_case(nh, nkv, 256, 3968, true, 3);    /* tile straddles the 4096-key group edge */
        bad |= run_case(nh, nkv, 256, 7936, false, 3);   /* ctx 8192, eager-sliced path */
        bad |= run_case(nh, nkv, 256, 16128, true, 2);   /* ctx 16384 */
        bad |= run_case(nh, nkv, 1024, 7936, true, 2);   /* several attn_chunk slices: tok_base != 0 */
        bad |= run_case(nh, nkv, 2, 8190, false, 5);    /* 2-token step at long ctx (MTP verify / tiny chunks) */
        bad |= run_case(nh, nkv, 16, 8176, false, 5);   /* 16-token chunk at long ctx */
    }
    q36_gpu_cleanup();
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
