#define _GNU_SOURCE
#include "q36_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 0x63a198efu;
static uint32_t next_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int check(uint32_t width, uint32_t rows, uint32_t grouped, int fixture) {
    const uint32_t offset = 17;
    const uint32_t signs_n = offset + width;
    const size_t in_n = (size_t)width * rows;
    const size_t out_bytes = in_n / 256u * 296u;
    float *x = calloc(in_n, sizeof(*x));
    float *signs = calloc(signs_n, sizeof(*signs));
    unsigned char *a = malloc(out_bytes), *b = malloc(out_bytes);
    if (!x || !signs || !a || !b) return 0;
    for (size_t i = 0; i < in_n; i++) {
        switch (fixture) {
        case 0: x[i] = 0.0f; break;
        case 1: x[i] = (i % 4u == 0u ? -1.0f : 1.0f) * (float)(i % 17u) / 8.0f; break;
        case 2: x[i] = i % 256u == 0u ? -127.0f :
                         (float)((int)(i % 9u) - 4) + (i % 2u ? 0.5f : -0.5f); break;
        default: {
            int32_t r = (int32_t)(next_u32() % 200001u) - 100000;
            x[i] = (float)r * (i % 3u ? 0.00001f : 0.01f);
        } break;
        }
    }
    for (uint32_t i = 0; i < signs_n; i++) signs[i] = i % 3u ? 1.0f : -1.0f;
    q36_gpu_tensor *src = q36_gpu_tensor_alloc(in_n * sizeof(float));
    q36_gpu_tensor *sgn = q36_gpu_tensor_alloc(signs_n * sizeof(float));
    q36_gpu_tensor *out_a = q36_gpu_tensor_alloc(out_bytes);
    q36_gpu_tensor *out_b = q36_gpu_tensor_alloc(out_bytes);
    int ok = src && sgn && out_a && out_b &&
             q36_gpu_tensor_write(src, 0, x, in_n * sizeof(float)) &&
             q36_gpu_tensor_write(sgn, 0, signs, signs_n * sizeof(float));
    if (ok) {
        unsetenv("Q36_VK_HADAMARD_SUBGROUP");
        ok = q36_gpu_hadamard_prepare_tensor(out_a, src, sgn,
                width, rows, offset, signs_n, 0.03125f, grouped);
    }
    if (ok) {
        setenv("Q36_VK_HADAMARD_SUBGROUP", "1", 1);
        ok = q36_gpu_hadamard_prepare_tensor(out_b, src, sgn,
                width, rows, offset, signs_n, 0.03125f, grouped);
    }
    if (ok) ok = q36_gpu_tensor_read(out_a, 0, a, out_bytes) &&
                 q36_gpu_tensor_read(out_b, 0, b, out_bytes);
    if (ok && memcmp(a, b, out_bytes)) {
        for (size_t i = 0; i < out_bytes; i++) if (a[i] != b[i]) {
            fprintf(stderr, "mismatch width=%u rows=%u grouped=%u fixture=%d byte=%zu %02x/%02x\n",
                    width, rows, grouped, fixture, i, a[i], b[i]);
            break;
        }
        ok = 0;
    }
    printf("width=%u rows=%u grouped=%u fixture=%d bytes=%zu %s\n",
           width, rows, grouped, fixture, out_bytes, ok ? "exact" : "FAIL");
    q36_gpu_tensor_free(src); q36_gpu_tensor_free(sgn);
    q36_gpu_tensor_free(out_a); q36_gpu_tensor_free(out_b);
    free(x); free(signs); free(a); free(b);
    return ok;
}

int main(void) {
    const uint32_t widths[] = {1024, 5120, 6144, 17408};
    int ok = q36_gpu_init();
    for (size_t i = 0; ok && i < sizeof(widths)/sizeof(widths[0]); i++) {
        for (int fixture = 0; ok && fixture < 4; fixture++) {
            ok = check(widths[i], 1, widths[i] == 6144, fixture);
            if (ok) ok = check(widths[i], 3, widths[i] == 6144, fixture);
        }
    }
    q36_gpu_cleanup();
    return ok ? 0 : 1;
}
