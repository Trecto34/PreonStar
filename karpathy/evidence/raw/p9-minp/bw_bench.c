#define _POSIX_C_SOURCE 200809L
/* BC-250 single-core streaming read rate over the 608 KB logits vector, to see
 * whether the 0.61 ms of q36_sample_full_vocab's two full-vocab loops is the
 * scan itself or the per-element branches. */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define N 151936u
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static float a[N] __attribute__((aligned(64)));
static float b[N] __attribute__((aligned(64)));
int main(void){
    FILE *fp=fopen("/tmp/p36/logits.f32","rb");
    if(fp){ if(fread(a,sizeof(float),N,fp)!=N) return 1; fclose(fp); } else return 1;
    const int reps=400;
    double t0,t1; volatile float sink=0; volatile uint64_t isink=0;

    t0=now_ms();
    for(int r=0;r<reps;r++){ float s=0; for(unsigned i=0;i<N;i++) s+=a[i]; sink=s; }
    t1=now_ms(); double ms_sum=(t1-t0)/reps;

    t0=now_ms();
    for(int r=0;r<reps;r++){ float mx=-INFINITY; for(unsigned i=0;i<N;i++){ float v=a[i]; if(v>mx) mx=v; } sink=mx; }
    t1=now_ms(); double ms_max=(t1-t0)/reps;

    t0=now_ms();
    for(int r=0;r<reps;r++){ for(unsigned i=0;i<N;i++) b[i]=-1.0f; sink=b[7]; }
    t1=now_ms(); double ms_store=(t1-t0)/reps;

    t0=now_ms();
    for(int r=0;r<reps;r++){ float s=0; for(unsigned i=0;i<N;i++) s+=b[i]; sink=s; }
    t1=now_ms(); double ms_read=(t1-t0)/reps;

    printf("sum   %7.3f ms  %.1f GB/s\n", ms_sum, N*4.0/ms_sum/1e6);
    printf("max   %7.3f ms  %.1f GB/s\n", ms_max, N*4.0/ms_max/1e6);
    printf("store %7.3f ms  %.1f GB/s\n", ms_store, N*4.0/ms_store/1e6);
    printf("read  %7.3f ms  %.1f GB/s\n", ms_read, N*4.0/ms_read/1e6);
    return 0;
}
