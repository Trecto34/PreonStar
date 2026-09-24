#define _POSIX_C_SOURCE 200809L
/* P9a: what fraction of q36_sample_full_vocab's host cost can a GPU min-p
 * prefilter actually remove?  The filter may only *find* survivors; expf, the
 * sum and the draw must stay on the CPU to stay bit-identical, so the removal
 * is the full-vocab scan (pass 1 max, pass 2 scaled/store/branch, pass 3 draw)
 * and not the survivor work.  Emulates the GPU pack as a ready-made
 * (idx, scaled) list and times the two paths on the same logits. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define N 151936u

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static uint64_t rs = 1234567u;
static float rnd(void) {
    rs = rs * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rs >> 40) & 0xffffffu) / 16777216.0f;
}

static float lg[N], scratch[N];
static uint32_t surv_i[N];
static float surv_s[N];
static int nsurv;

static void run(const char *tag, float temperature, float min_p, int reps) {
    /* pass 1: max + best (the GPU top2 already gives max for free) */
    float mx = -INFINITY;
    int best = 0;
    for (unsigned i = 0; i < N; i++) {
        if (!isfinite(lg[i])) continue;
        if (lg[i] > mx) { mx = lg[i]; best = (int)i; }
    }

    /* the reference: q36_sample_full_vocab's two remaining loops over n_vocab */
    float reject_scaled = logf(min_p);
    for (int i = 0; i < 8 && isfinite(reject_scaled); i++) {
        reject_scaled = nextafterf(reject_scaled, -3.4e38f);
        if (expf(reject_scaled) < min_p) break;
    }
    double t = now_ms();
    uint64_t rcsum = 0;
    for (int r = 0; r < reps; r++) {
        float sum = 0;
        for (unsigned i = 0; i < N; i++) {
            const float v = lg[i];
            scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - mx) / temperature;
            if (scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_p) continue;
            scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) { rcsum += 1; continue; }
        float rr = rnd() * sum;
        for (unsigned i = 0; i < N; i++) {
            const float p = scratch[i];
            if (p < 0.0f) continue;
            rr -= p;
            if (rr <= 0.0f) { rcsum += (uint64_t)i; break; }
        }
    }
    double ms_ref = (now_ms() - t) / reps;

    /* the pack path: same max, same reject test, but only the survivor list */
    t = now_ms();
    uint64_t pksum = 0;
    uint64_t nexpf = 0;
    for (int r = 0; r < reps; r++) {
        float sum = 0;
        for (int k = 0; k < nsurv; k++) {
            const float scaled = surv_s[k];
            if (scaled <= reject_scaled) continue;   /* re-tested, as the CPU must */
            const float p = expf(scaled);
            nexpf++;
            if (p < min_p) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) { pksum += 1; continue; }
        float rr = rnd() * sum;
        for (int k = 0; k < nsurv; k++) {
            const float scaled = surv_s[k];
            if (scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_p) continue;
            rr -= p;
            if (rr <= 0.0f) { pksum += surv_i[k]; break; }
        }
    }
    double ms_pack = (now_ms() - t) / reps;

    /* the pack cost the GPU pays: scan + compact (proxy: the CPU prefilter scan) */
    t = now_ms();
    for (int r = 0; r < reps; r++) {
        int n = 0;
        for (unsigned i = 0; i < N; i++)
            if (lg[i] > mx + temperature * (logf(min_p) - 1e-3f)) n++;
        nsurv = n;
    }
    double ms_scan = (now_ms() - t) / reps;

    printf("%s temp=%.2f max=%.3f best=%d survivors=%d expf=%llu\n",
           tag, temperature, mx, best, nsurv, (unsigned long long)(nexpf / reps));
    printf("   full-vocab 2 loops  %7.3f ms\n", ms_ref);
    printf("   survivor-only loops %7.3f ms\n", ms_pack);
    printf("   removable           %7.3f ms  (%.0f%%)\n",
           ms_ref - ms_pack, 100.0 * (ms_ref - ms_pack) / ms_ref);
    printf("   CPU prefilter scan  %7.3f ms  (this moves to the GPU)\n", ms_scan);
}

int main(int argc, char **argv) {
    int reps = 200;
    int have = 0;
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "rb");
        if (!fp) { fprintf(stderr, "no %s\n", argv[1]); return 1; }
        if (fread(lg, sizeof(float), N, fp) != N) { fprintf(stderr, "short\n"); return 1; }
        fclose(fp);
        have = 1;
    }
    if (!have) {
        for (unsigned i = 0; i < N; i++) lg[i] = rnd() * 9.0f - 4.5f;
        lg[17] += 6.0f; lg[90000] += 5.0f;
        printf("(synthetic logits)\n");
    }

    /* build the survivor list once per setting, as the GPU pack would */
    for (int s = 0; s < 2; s++) {
        const float temperature = s ? 0.8f : 1.0f;
        const float min_p = 0.05f;
        float mx = -INFINITY;
        for (unsigned i = 0; i < N; i++) if (lg[i] > mx) mx = lg[i];
        const float cut = mx + temperature * (logf(min_p) - 1e-3f);
        nsurv = 0;
        for (unsigned i = 0; i < N; i++)
            if (lg[i] > cut) { surv_i[nsurv] = i; surv_s[nsurv] = (lg[i] - mx) / temperature; nsurv++; }
        run(have ? "file" : "synthetic", temperature, min_p, reps);
    }
    return 0;
}
