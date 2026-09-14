/* Self-check for q36_gpu_topk8_tensor (vulkan/topk8.comp): the exact top-8
 * F32 selection used by q36_session_sample's GPU fast path for top_k<=8.
 * No model load needed — just q36_gpu_init() plus synthetic tensors.
 * Verifies against a CPU reference over random arrays and a set of edge
 * cases (ties, negatives, small counts, all-equal, sizes that don't divide
 * the 256-wide grid evenly). See AGENT.md: "Prefer short Vulkan smoke tests
 * for build verification." */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../q36_gpu.h"

static int better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

/* Mirrors topk8.comp's insert_topk exactly, so this is a genuine reference,
 * not a restatement of the same code under test. */
static void cpu_topk8(const float *x, int n, int ids[8], float vals[8]) {
    int m = 0;
    for (int k = 0; k < 8; k++) { vals[k] = -INFINITY; ids[k] = -1; }
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (m == 8 && !better(v, i, vals[7], ids[7])) continue;
        int j = m < 8 ? m++ : 7;
        while (j > 0 && better(v, i, vals[j - 1], ids[j - 1])) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = i;
    }
}

static int run_case(const char *name, const float *x, uint32_t n) {
    q36_gpu_tensor *logits = q36_gpu_tensor_alloc((uint64_t)n * sizeof(float));
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(8u * (sizeof(int32_t) + sizeof(float)));
    if (!logits || !out) {
        fprintf(stderr, "%s: tensor alloc failed\n", name);
        return 1;
    }
    if (!q36_gpu_tensor_write(logits, 0, x, (uint64_t)n * sizeof(float))) {
        fprintf(stderr, "%s: tensor write failed\n", name);
        return 1;
    }
    if (!q36_gpu_topk8_tensor(out, logits, n)) {
        fprintf(stderr, "%s: q36_gpu_topk8_tensor reported unsupported/failed\n", name);
        q36_gpu_tensor_free(logits);
        q36_gpu_tensor_free(out);
        return 1;
    }
    struct { int32_t ids[8]; float vals[8]; } gpu;
    if (!q36_gpu_tensor_read(out, 0, &gpu, sizeof(gpu))) {
        fprintf(stderr, "%s: readback failed\n", name);
        return 1;
    }
    q36_gpu_tensor_free(logits);
    q36_gpu_tensor_free(out);

    int ref_ids[8];
    float ref_vals[8];
    cpu_topk8(x, (int)n, ref_ids, ref_vals);

    int fail = 0;
    for (int k = 0; k < 8; k++) {
        if (gpu.ids[k] != ref_ids[k] || gpu.vals[k] != ref_vals[k]) {
            fprintf(stderr, "%s: slot %d mismatch: gpu (id=%d v=%.9g) ref (id=%d v=%.9g)\n",
                    name, k, gpu.ids[k], gpu.vals[k], ref_ids[k], ref_vals[k]);
            fail = 1;
        }
    }
    if (!fail) printf("%s: OK\n", name);
    return fail;
}

int main(void) {
    if (!q36_gpu_init()) {
        fprintf(stderr, "q36_gpu_init failed (no Vulkan device?) — skipping\n");
        return 0;
    }

    int failures = 0;
    uint64_t seed = 12345;

    /* Random arrays at vocab-like and off-grid sizes. */
    uint32_t sizes[] = { 8, 9, 100, 255, 256, 257, 1000, 4096, 65535, 65536, 248320, 300000 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        uint32_t n = sizes[s];
        float *x = malloc((size_t)n * sizeof(float));
        for (uint32_t i = 0; i < n; i++) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            x[i] = (float)((int64_t)(seed >> 40) % 20000 - 10000) / 100.0f;
        }
        char name[64];
        snprintf(name, sizeof(name), "random n=%u", n);
        failures += run_case(name, x, n);
        free(x);
    }

    /* All-equal: every slot ties, must resolve to the 8 lowest indices. */
    {
        uint32_t n = 5000;
        float *x = malloc((size_t)n * sizeof(float));
        for (uint32_t i = 0; i < n; i++) x[i] = 3.0f;
        failures += run_case("all-equal", x, n);
        free(x);
    }

    /* Ties at the top only, distinguishing which duplicate should win. */
    {
        uint32_t n = 1000;
        float *x = malloc((size_t)n * sizeof(float));
        for (uint32_t i = 0; i < n; i++) x[i] = -1.0f;
        for (uint32_t i = 0; i < 20; i++) x[i * 37 % n] = 9.0f; /* 20 ties for the top */
        failures += run_case("top-ties", x, n);
        free(x);
    }

    /* -inf entries mixed in (masked-out tokens). */
    {
        uint32_t n = 2000;
        float *x = malloc((size_t)n * sizeof(float));
        seed = 999;
        for (uint32_t i = 0; i < n; i++) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            x[i] = (i % 3 == 0) ? -INFINITY : (float)((int64_t)(seed >> 40) % 1000) / 10.0f;
        }
        failures += run_case("with -inf", x, n);
        free(x);
    }

    /* Exactly 8 elements (n == K, no sentinels expected). */
    {
        float x[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        failures += run_case("n==K", x, 8);
    }

    /* Fewer than 8 elements: sentinels must fill the rest and never be picked. */
    {
        float x[3] = { 5, -5, 0 };
        failures += run_case("n<K", x, 3);
    }

    q36_gpu_cleanup();

    if (failures) {
        fprintf(stderr, "%d case(s) FAILED\n", failures);
        return 1;
    }
    printf("all cases OK\n");
    return 0;
}
