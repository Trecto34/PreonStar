#include "q36_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t seed = 0x7254a19fu;
static uint32_t random_u32(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}

static int fixture(uint32_t width, uint32_t rows, int mode) {
    const uint32_t sign_offset=7, sign_count=width+sign_offset;
    const size_t elems=(size_t)width*rows;
    const size_t q8_bytes=elems/256u*296u;
    float *gate=malloc(elems*4), *up=malloc(elems*4), *signs=malloc(sign_count*4);
    unsigned char *a=malloc(q8_bytes), *b=malloc(q8_bytes);
    if (!gate || !up || !signs || !a || !b) return 0;
    for (size_t i=0;i<elems;i++) {
        if(mode==0) { gate[i]=0.0f; up[i]=0.0f; }
        else if(mode==1) {
            gate[i]=(float)((int)(i%17u)-8)/4.0f;
            up[i]=(float)((int)(i%5u)-2)/2.0f;
        } else if(mode==2) {
            gate[i]=i%256u==0u ? -127.0f : (float)((int)(i%9u)-4)*0.5f;
            up[i]=i%2u ? -1.0f : 1.0f;
        } else {
            gate[i]=(float)((int)(random_u32()%10001u)-5000)*0.001f;
            up[i]=(float)((int)(random_u32()%10001u)-5000)*0.00001f;
        }
    }
    for(uint32_t i=0;i<sign_count;i++) signs[i]=i%3u ? 1.0f : -1.0f;
    q36_gpu_tensor *g=q36_gpu_tensor_alloc(elems*4),
        *u=q36_gpu_tensor_alloc(elems*4),
        *s=q36_gpu_tensor_alloc(sign_count*4),
        *mid=q36_gpu_tensor_alloc(elems*4),
        *plain=q36_gpu_tensor_alloc(q8_bytes),
        *ref=q36_gpu_tensor_alloc(q8_bytes),
        *fused=q36_gpu_tensor_alloc(q8_bytes);
    int ok=g&&u&&s&&mid&&plain&&ref&&fused &&
        q36_gpu_tensor_write(g,0,gate,elems*4) &&
        q36_gpu_tensor_write(u,0,up,elems*4) &&
        q36_gpu_tensor_write(s,0,signs,sign_count*4) &&
        q36_gpu_swiglu_q8_k_tensor(mid,plain,g,u,width,rows,0.0f,1.0f) &&
        q36_gpu_hadamard_prepare_tensor(ref,mid,s,width,rows,sign_offset,sign_count,0.03125f,0) &&
        q36_gpu_swiglu_hadamard_prepare_tensor(fused,g,u,s,width,rows,sign_offset,sign_count,0.03125f) &&
        q36_gpu_tensor_read(ref,0,a,q8_bytes) &&
        q36_gpu_tensor_read(fused,0,b,q8_bytes);
    if(ok && memcmp(a,b,q8_bytes)) {
        for(size_t i=0;i<q8_bytes;i++) if(a[i]!=b[i]) {
            fprintf(stderr,"diff width=%u rows=%u mode=%d byte=%zu %02x/%02x\n",
                width,rows,mode,i,a[i],b[i]); break;
        }
        ok=0;
    }
    printf("width=%u rows=%u mode=%d bytes=%zu %s\n",width,rows,mode,q8_bytes,ok?"exact":"FAIL");
    q36_gpu_tensor_free(g);q36_gpu_tensor_free(u);q36_gpu_tensor_free(s);
    q36_gpu_tensor_free(mid);q36_gpu_tensor_free(plain);
    q36_gpu_tensor_free(ref);q36_gpu_tensor_free(fused);
    free(gate);free(up);free(signs);free(a);free(b);
    return ok;
}

int main(void) {
    const uint32_t widths[]={1024,5120,17408};
    int ok=q36_gpu_init();
    for(size_t i=0;ok&&i<sizeof(widths)/sizeof(widths[0]);i++)
        for(int mode=0;ok&&mode<4;mode++) {
            ok=fixture(widths[i],1,mode);
            if(ok) ok=fixture(widths[i],3,mode);
        }
    q36_gpu_cleanup();
    return ok?0:1;
}
