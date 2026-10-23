#ifndef P36_IMAGE_H
#define P36_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#define P36_IMAGE_MAX_ENCODED_BYTES (64u * 1024u * 1024u)
#define P36_IMAGE_MAX_DIMENSION 16384u
#define P36_IMAGE_MAX_PIXELS (64u * 1024u * 1024u)

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t *rgb;
    uint8_t fingerprint[32];
} p36_image;

typedef struct {
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t patch_count;
    uint32_t image_token_count;
    float *patches;
} p36_image_patches;

int p36_image_decode_memory(p36_image *out, const uint8_t *encoded,
                            size_t encoded_len, char *error, size_t error_cap);
int p36_image_decode_file(p36_image *out, const char *path,
                          char *error, size_t error_cap);
void p36_image_free(p36_image *image);

int p36_image_preprocess_qwen3vl(p36_image_patches *out,
                                 const p36_image *image,
                                 uint32_t max_patches,
                                 char *error, size_t error_cap);
void p36_image_patches_free(p36_image_patches *patches);

#endif
