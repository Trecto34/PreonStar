#ifndef P36_QUANT_H
#define P36_QUANT_H

#include <stdbool.h>
#include <stdint.h>

#ifndef P36_QUANT_LINKAGE
#define P36_QUANT_LINKAGE
#endif

P36_QUANT_LINKAGE float p36_quant_f16_to_f32(uint16_t h);
P36_QUANT_LINKAGE uint16_t p36_quant_f32_to_f16(float f);
P36_QUANT_LINKAGE void p36_quant_q8_0(const float *x, void *out, int64_t n);
P36_QUANT_LINKAGE void p36_quant_q4_0(const float *x, void *out, int64_t n);
P36_QUANT_LINKAGE void p36_quant_q4_0_kv(const float *x, void *out, int64_t n);
P36_QUANT_LINKAGE void p36_quant_q8_k(const float *x, void *out, int64_t n);
P36_QUANT_LINKAGE float p36_quant_dot_q8_0(const void *a, const void *b, int n);
P36_QUANT_LINKAGE bool p36_quant_dequantize(
        uint32_t type, const void *src, float *out, uint32_t n);
P36_QUANT_LINKAGE void p36_quant_pack_iq1_grid(void *dst);

#endif
