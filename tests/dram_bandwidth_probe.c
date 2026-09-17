/* bw.c - measurable DRAM bandwidth on BC-250 unified memory.
 * The GPU shares the same GDDR6 as the CPU, so a CPU streaming test is a
 * fair proxy for the achievable ceiling the kernels can pull against.
 * Reports read-only bandwidth (matvec-like) and copy bandwidth (read+write).
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NTH 8
#define REPS 20
#define GB (1024.0 * 1024.0 * 1024.0)

static size_t NELEM;
static float *A, *B;
static double read_best[NTH], copy_best[NTH];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

typedef struct { int id; } arg_t;

/* Read-only streaming: closest analogue to a matvec reading weights. */
static void *reader(void *p) {
    arg_t *a = (arg_t *)p;
    size_t per = NELEM / NTH;
    size_t off = (size_t)a->id * per;
    size_t n = (a->id == NTH - 1) ? NELEM - off : per;
    double best = 0.0;
    float acc = 0.f;
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += A[off + i];     s1 += A[off + i + 1];
            s2 += A[off + i + 2]; s3 += A[off + i + 3];
        }
        for (; i < n; i++) s0 += A[off + i];
        acc += (s0 + s1) + (s2 + s3);
        double dt = now() - t0;
        double gbs = ((double)n * sizeof(float)) / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    read_best[a->id] = best;
    if (acc == 12345.678f) printf("");   /* keep acc live */
    return NULL;
}

static void *copier(void *p) {
    arg_t *a = (arg_t *)p;
    size_t per = NELEM / NTH;
    size_t off = (size_t)a->id * per;
    size_t n = (a->id == NTH - 1) ? NELEM - off : per;
    double best = 0.0;
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        memcpy(B + off, A + off, n * sizeof(float));
        double dt = now() - t0;
        /* copy moves 2x bytes: one read + one write */
        double gbs = (2.0 * (double)n * sizeof(float)) / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    copy_best[a->id] = best;
    return NULL;
}

static double sum_best(double *v) {
    double s = 0.0;
    for (int i = 0; i < NTH; i++) s += v[i];
    return s;
}

int main(int argc, char **argv) {
    double gb = (argc > 1) ? atof(argv[1]) : 2.0;
    NELEM = (size_t)(gb * GB / sizeof(float));

    if (posix_memalign((void **)&A, 4096, NELEM * sizeof(float)) != 0) { perror("A"); return 1; }
    if (posix_memalign((void **)&B, 4096, NELEM * sizeof(float)) != 0) { perror("B"); return 1; }

    for (size_t i = 0; i < NELEM; i++) A[i] = (float)(i & 1023) * 0.5f;
    memset(B, 0, NELEM * sizeof(float));

    printf("buffer: %.2f GiB per array (A+B = %.2f GiB), threads=%d, reps=%d\n",
           (double)NELEM * sizeof(float) / GB, 2.0 * (double)NELEM * sizeof(float) / GB, NTH, REPS);

    pthread_t th[NTH];
    arg_t args[NTH];
    for (int i = 0; i < NTH; i++) args[i].id = i;

    /* warm the pages */
    for (int i = 0; i < NTH; i++) pthread_create(&th[i], NULL, copier, &args[i]);
    for (int i = 0; i < NTH; i++) pthread_join(th[i], NULL);

    for (int i = 0; i < NTH; i++) pthread_create(&th[i], NULL, reader, &args[i]);
    for (int i = 0; i < NTH; i++) pthread_join(th[i], NULL);
    printf("READ  (sum of %d threads, best rep): %8.1f GB/s\n", NTH, sum_best(read_best));

    for (int i = 0; i < NTH; i++) pthread_create(&th[i], NULL, copier, &args[i]);
    for (int i = 0; i < NTH; i++) pthread_join(th[i], NULL);
    printf("COPY  (read+write, sum of %d threads): %8.1f GB/s\n", NTH, sum_best(copy_best));

    return 0;
}
