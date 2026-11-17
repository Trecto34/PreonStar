/* Parity + timing check for attn_decode_fa_{f16_,}gqa{6,8} against attn_decode_split.
 * Dense Qwen3.8 (24 q / 4 kv heads) and 35B-A3B MoE (16 q / 2 kv heads) shapes,
 * head_dim 256, Q8_0 K, Q4_0 V, n_tok 1 (decode) and 4 (MTP verify).
 * Both kernels run through p36_gpu_attn_decode_tensor; P36_VK_ATTN_DECODE_FA_MIN
 * selects split (0) vs decode_fa (64) per arm; P36_VK_ATTN_DECODE_FA_F16 selects
 * f32 (0) vs f16 (1, default) LDS staging at init.
 * Exit status is nonzero if any case exceeds the 2e-3 tolerance. */
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
#include "../p36.h"
#include "../p36_gpu.h"
#include "../p36_cli_args.h"

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

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static float *sinks;
static const size_t sinks_alloc = 4096;

static int run_case(uint32_t n_head, uint32_t n_head_kv, uint32_t n_tok, uint32_t pos0,
                    bool has_sinks, int iters) {
    const uint32_t hd = 256;
    const uint32_t kv_max = pos0 + n_tok, cache_row = n_head_kv * hd;
    const uint32_t k_rb = cache_row / 32u * 34u, v_rb = cache_row / 32u * 18u;
    const uint64_t qf = (uint64_t)n_tok * n_head * hd;
    const int batch_calls = 10; /* 10 full-attention layers per MoE decode token */

    float *qh = malloc(qf * 4), *qgh = malloc(qf * 8);
    uint8_t *kh = malloc((size_t)kv_max * k_rb), *vh = malloc((size_t)kv_max * v_rb);
    float *oa = malloc(qf * 4), *ob = malloc(qf * 4);
    const char *qs_env = getenv("P36_TEST_QSCALE");
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
    p36_gpu_tensor *q = p36_gpu_tensor_alloc(qf * 4), *qg = p36_gpu_tensor_alloc(qf * 8);
    p36_gpu_tensor *kc = p36_gpu_tensor_alloc((uint64_t)kv_max * k_rb);
    p36_gpu_tensor *vc = p36_gpu_tensor_alloc((uint64_t)kv_max * v_rb);
    p36_gpu_tensor *out = p36_gpu_tensor_alloc(qf * 4);
    p36_gpu_tensor_write(q, 0, qh, qf * 4);
    p36_gpu_tensor_write(qg, 0, qgh, qf * 8);
    p36_gpu_tensor_write(kc, 0, kh, (uint64_t)kv_max * k_rb);
    p36_gpu_tensor_write(vc, 0, vh, (uint64_t)kv_max * v_rb);
    p36_gpu_set_model_map(sinks, sinks_alloc);

    double us[2];
    for (int arm = 0; arm < 2; arm++) {
        setenv("P36_VK_ATTN_DECODE_FA_MIN", arm ? "64" : "0", 1);
        /* Warmup + parity readback */
        if (!p36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL, sinks, sinks_alloc, 0, has_sinks,
                                        pos0, n_tok, n_head, n_head_kv, hd, 1, 2, k_rb, v_rb)) {
            fprintf(stderr, "dispatch failed n_tok=%u pos0=%u arm=%d\n", n_tok, pos0, arm);
            exit(1);
        }
        p36_gpu_tensor_read(out, 0, arm ? ob : oa, qf * 4);

        double best = 1e30;
        for (int it = 0; it < iters; it++) {
            double t0 = now_us();
            for (int c = 0; c < batch_calls; c++) {
                if (!p36_gpu_attn_decode_tensor(out, q, qg, kc, vc, NULL, sinks, sinks_alloc, 0, has_sinks,
                                                pos0, n_tok, n_head, n_head_kv, hd, 1, 2, k_rb, v_rb)) {
                    fprintf(stderr, "dispatch failed n_tok=%u pos0=%u arm=%d\n", n_tok, pos0, arm);
                    exit(1);
                }
            }
            p36_gpu_tensor_read(out, 0, arm ? ob : oa, qf * 4);
            double dt = (now_us() - t0) / batch_calls;
            if (dt < best) best = dt;
        }
        us[arm] = best;
    }
    double max_abs = 0, max_ref = 0;
    for (uint64_t i = 0; i < qf; i++) {
        double d = fabs((double)oa[i] - ob[i]);
        if (!(d <= max_abs)) max_abs = d;
        if (fabs(oa[i]) > max_ref) max_ref = fabs(oa[i]);
    }
    int bad = !(max_abs <= 2e-3 * (max_ref > 1 ? max_ref : 1));
    printf("gqa%u n_tok=%u pos0=%5u sinks=%d  split %7.1f us  fa %7.1f us  (%.2fx)  max_abs=%.3g max_ref=%.3g %s\n",
           n_head / n_head_kv, n_tok, pos0, has_sinks, us[0], us[1], us[0] / us[1], max_abs, max_ref, bad ? "FAIL" : "ok");
    p36_gpu_tensor_free(q); p36_gpu_tensor_free(qg); p36_gpu_tensor_free(kc);
    p36_gpu_tensor_free(vc); p36_gpu_tensor_free(out);
    free(qh); free(qgh); free(kh); free(vh); free(oa); free(ob);
    return bad;
}

int main(int argc, char **argv) {
    int cli_rc = p36_cli_require_no_arguments(argc, argv, "test_attn_decode");
    if (cli_rc) return cli_rc;
    unsetenv("P36_VK_ATTN_FUSED");
    unsetenv("P36_VK_ATTN_SPLITK");
    unsetenv("P36_VK_ATTN_ROWS_DECODE");
    if (posix_memalign((void **)&sinks, (size_t)getpagesize(), sinks_alloc)) return 1;
    for (uint32_t h = 0; h < 24; h++) sinks[h] = 2.0f * frand();
    if (!p36_gpu_init()) { fprintf(stderr, "p36_gpu_init failed\n"); return 1; }
    p36_gpu_set_quality(false);
    int bad = 0;
    for (int g = 0; g < 2; g++) {
        /* dense Qwen3.8 24/4, then 35B-A3B MoE 16/2 */
        uint32_t nh = g ? 16 : 24, nkv = g ? 2 : 4;
        bad |= run_case(nh, nkv, 1, 1023, true, 5);    /* short ctx (below default 4096 threshold) */
        bad |= run_case(nh, nkv, 1, 4095, false, 5);   /* ctx 4096 threshold */
        bad |= run_case(nh, nkv, 1, 4110, true, 5);    /* ragged partial-group tail */
        bad |= run_case(nh, nkv, 1, 8191, false, 5);   /* ctx 8192 */
        bad |= run_case(nh, nkv, 1, 16383, true, 5);   /* ctx 16384 */
        bad |= run_case(nh, nkv, 1, 32767, true, 5);   /* ctx 32768 */
        bad |= run_case(nh, nkv, 4, 8188, false, 5);   /* 4-token MTP verify at 8k */
        bad |= run_case(nh, nkv, 4, 16380, true, 5);   /* 4-token MTP verify at 16k */
    }
    p36_gpu_cleanup();
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
