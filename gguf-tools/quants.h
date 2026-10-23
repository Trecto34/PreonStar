#ifndef P36_QUANTS_H
#define P36_QUANTS_H

/*
 * Narrow quantization API used by the P36 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the P36
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define P36Q_MAX_DIMS 4

typedef enum {
    P36Q_TYPE_F32     = 0,
    P36Q_TYPE_F16     = 1,
    P36Q_TYPE_Q4_0    = 2,
    P36Q_TYPE_Q4_1    = 3,
    P36Q_TYPE_Q5_0    = 6,
    P36Q_TYPE_Q5_1    = 7,
    P36Q_TYPE_Q8_0    = 8,
    P36Q_TYPE_Q8_1    = 9,
    P36Q_TYPE_Q2_K    = 10,
    P36Q_TYPE_Q3_K    = 11,
    P36Q_TYPE_Q4_K    = 12,
    P36Q_TYPE_Q5_K    = 13,
    P36Q_TYPE_Q6_K    = 14,
    P36Q_TYPE_Q8_K    = 15,
    P36Q_TYPE_IQ2_XXS = 16,
    P36Q_TYPE_IQ2_XS  = 17,
    P36Q_TYPE_IQ3_XXS = 18,
    P36Q_TYPE_IQ1_S   = 19,
    P36Q_TYPE_IQ4_NL  = 20,
    P36Q_TYPE_IQ3_S   = 21,
    P36Q_TYPE_IQ2_S   = 22,
    P36Q_TYPE_IQ4_XS  = 23,
    P36Q_TYPE_I8      = 24,
    P36Q_TYPE_I16     = 25,
    P36Q_TYPE_I32     = 26,
    P36Q_TYPE_I64     = 27,
    P36Q_TYPE_F64     = 28,
    P36Q_TYPE_IQ1_M   = 29,
    P36Q_TYPE_BF16    = 30,
    P36Q_TYPE_TQ1_0   = 34,
    P36Q_TYPE_TQ2_0   = 35,
    P36Q_TYPE_MXFP4   = 39,
    P36Q_TYPE_NVFP4   = 40,
    P36Q_TYPE_Q1_0    = 41,
    P36Q_TYPE_COUNT   = 42,
} p36q_type;

static inline size_t p36q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *p36q_type_name(p36q_type type);
bool p36q_can_quantize(p36q_type type);
int64_t p36q_block_size(p36q_type type);
size_t p36q_row_size(p36q_type type, int64_t ne);
bool p36q_requires_imatrix(p36q_type type);
void p36q_quantize_init(p36q_type type);
size_t p36q_quantize_chunk(p36q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float p36q_f16_to_f32(uint16_t bits);
float p36q_bf16_to_f32(uint16_t bits);
void p36q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void p36q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
