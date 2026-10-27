/* Extract the Qwen3.6 MTP (nextn) block from a GGUF into a standalone
 * support file, requantizing block quants the engine cannot dispatch.
 *
 * The MTP block lives at layer == block_count (blk.40 for a 41-block model) and
 * is only ever read by the speculative head, so it can be repacked on its own:
 * a 0.3-0.7 GB support file instead of a second copy of the target model.
 * Routing experts are stored Q4_0 in the APEX-style files; the engine's MoE
 * kernels do not dispatch Q4_0, so dequantize those two tensors and re-emit
 * them as Q4_K with the engine's own quantizer.
 *
 * usage: mtp-head-extract --in MODEL.gguf --out HEAD.gguf [--layer N]
 */
#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_ALIGNMENT 32

enum {
    GGUF_TYPE_UINT8 = 0, GGUF_TYPE_INT8 = 1, GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3, GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL = 7, GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9, GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

#define GGML_Q4_0 2
#define GGML_Q4_K 12

typedef struct { const uint8_t *base; size_t size, pos; } cursor;
typedef struct { char *name; uint32_t ndim, type; uint64_t dim[4], offset, elements; } tensor_info;

static void die(const char *msg) { fprintf(stderr, "mtp-head-extract: %s\n", msg); exit(1); }

static void need(cursor *c, size_t n) { if (c->size - c->pos < n) die("truncated GGUF"); }

static uint64_t rd(cursor *c, size_t n) {
    uint64_t v = 0;
    need(c, n);
    memcpy(&v, c->base + c->pos, n);
    c->pos += n;
    return v;
}

static char *rd_str(cursor *c, uint64_t *len) {
    uint64_t n = rd(c, 8);
    need(c, (size_t)n);
    char *s = malloc((size_t)n + 1);
    if (!s) die("out of memory");
    memcpy(s, c->base + c->pos, (size_t)n);
    s[n] = 0;
    c->pos += (size_t)n;
    if (len) *len = n;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
    default: return 0;
    }
}

static void skip_value(cursor *c, uint32_t t) {
    if (t == GGUF_TYPE_STRING) { uint64_t n = rd(c, 8); need(c, (size_t)n); c->pos += (size_t)n; return; }
    if (t == GGUF_TYPE_ARRAY) {
        uint32_t it = (uint32_t)rd(c, 4);
        uint64_t n = rd(c, 8);
        for (uint64_t i = 0; i < n; i++) skip_value(c, it);
        return;
    }
    need(c, scalar_size(t));
    c->pos += scalar_size(t);
}

static bool kv_u32(cursor *c, uint32_t t, uint32_t *out) {
    size_t at = c->pos;
    if (t != GGUF_TYPE_UINT32) return false;
    memcpy(out, c->base + at, 4);
    return true;
}

static size_t type_row_size(uint32_t type, uint64_t ne0) {
    /* (block elements, block bytes) for every quant the engine can read. */
    static const struct { uint32_t elems, bytes; } blk[] = {
        [0]  = {1, 4},      /* f32 */
        [1]  = {1, 2},      /* f16 */
        [2]  = {32, 18},    /* q4_0 */
        [3]  = {32, 20},    /* q4_1 */
        [8]  = {32, 34},    /* q8_0 */
        [10] = {256, 84},   /* q2_k */
        [11] = {256, 110},  /* q3_k */
        [12] = {256, 144},  /* q4_k */
        [13] = {256, 176},  /* q5_k */
        [14] = {256, 210},  /* q6_k */
        [15] = {256, 292},  /* q8_k */
        [16] = {256, 66},   /* iq2_xxs */
        [17] = {256, 74},   /* iq2_xs */
        [18] = {256, 98},   /* iq3_xxs */
        [19] = {256, 50},   /* iq1_s */
        [20] = {32, 18},    /* iq4_nl */
        [21] = {256, 110},  /* iq3_s */
        [22] = {256, 82},   /* iq2_s */
        [23] = {256, 136},  /* iq4_xs */
        [29] = {256, 56},   /* iq1_m */
        [30] = {1, 2},      /* bf16 */
    };
    if (type >= sizeof(blk) / sizeof(blk[0]) || blk[type].elems == 0) return 0;
    if (ne0 % blk[type].elems != 0) return 0;
    return (size_t)(ne0 / blk[type].elems) * blk[type].bytes;
}

/* Q4_0 block: f16 scale then 32 4-bit values (low nibble = i, high = i+16). */
static void dequant_q4_0(const uint8_t *src, float *dst, uint64_t ne) {
    uint64_t blocks = ne / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t dbits;
        memcpy(&dbits, src, 2);
        float d = p36q_f16_to_f32(dbits);
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 16; j++) {
            dst[j] = d * (float)((qs[j] & 0x0F) - 8);
            dst[j + 16] = d * (float)((qs[j] >> 4) - 8);
        }
        src += 18;
        dst += 32;
    }
}

static void put32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4); *p += 4; }
static void put64(uint8_t **p, uint64_t v) { memcpy(*p, &v, 8); *p += 8; }
static void put_str(uint8_t **p, const char *s, uint64_t n) {
    put64(p, n);
    memcpy(*p, s, (size_t)n);
    *p += n;
}

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL;
    p36q_type expert_type = P36Q_TYPE_Q4_K;
    uint32_t expert_ggml = GGML_Q4_K;
    long layer = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--expert-quant") && i + 1 < argc) {
            const char *q = argv[++i];
            if (!strcmp(q, "q4_k")) { expert_type = P36Q_TYPE_Q4_K; expert_ggml = GGML_Q4_K; }
            else if (!strcmp(q, "q8_0")) { expert_type = P36Q_TYPE_Q8_0; expert_ggml = 8; }
            else die("--expert-quant must be q4_k or q8_0");
        }
        else die("usage: mtp-head-extract --in MODEL.gguf --out HEAD.gguf [--layer N]");
    }
    if (!in_path || !out_path) die("usage: mtp-head-extract --in MODEL.gguf --out HEAD.gguf [--layer N]");

    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("cannot open input");
    struct stat st;
    if (fstat(fd, &st) != 0) die("cannot stat input");
    uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("cannot map input");

    cursor c = {map, (size_t)st.st_size, 0};
    if (rd(&c, 4) != 0x46554747u) die("not a GGUF file"); /* 'GGUF' */
    uint32_t version = (uint32_t)rd(&c, 4);
    if (version != 3) die("expected GGUF v3");
    uint64_t n_tensors = rd(&c, 8), n_kv = rd(&c, 8);

    size_t kv_start = c.pos;
    uint32_t block_count = 0, nextn = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = rd_str(&c, NULL);
        uint32_t t = (uint32_t)rd(&c, 4);
        if (strstr(key, "block_count")) (void)kv_u32(&c, t, &block_count);
        if (strstr(key, "nextn_predict_layers")) (void)kv_u32(&c, t, &nextn);
        skip_value(&c, t);
        free(key);
    }
    size_t kv_end = c.pos;

    tensor_info *ti = calloc((size_t)n_tensors, sizeof(*ti));
    if (!ti) die("out of memory");
    for (uint64_t i = 0; i < n_tensors; i++) {
        ti[i].name = rd_str(&c, NULL);
        ti[i].ndim = (uint32_t)rd(&c, 4);
        if (ti[i].ndim == 0 || ti[i].ndim > 4) die("unsupported tensor rank");
        ti[i].elements = 1;
        for (uint32_t d = 0; d < ti[i].ndim; d++) {
            ti[i].dim[d] = rd(&c, 8);
            ti[i].elements *= ti[i].dim[d];
        }
        ti[i].type = (uint32_t)rd(&c, 4);
        ti[i].offset = rd(&c, 8);
    }
    size_t info_end = c.pos;
    size_t data_start = (info_end + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
    /* The head block sits right after the target layer stack. */
    if (layer < 0) layer = (long)(block_count > nextn ? block_count - nextn : block_count);
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%ld.", layer);

    /* Select the head block; the run stores output data per selected tensor. */
    typedef struct { const tensor_info *t; uint32_t out_type; size_t out_bytes; } item;
    item *items = calloc((size_t)n_tensors, sizeof(*items));
    if (!items) die("out of memory");
    size_t n_sel = 0, data_bytes = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &ti[i];
        if (strncmp(t->name, prefix, strlen(prefix)) != 0) continue;
        items[n_sel].t = t;
        items[n_sel].out_type = t->type;
        items[n_sel].out_bytes = 0;
        n_sel++;
    }
    if (n_sel == 0) die("no nextn block tensors found for that layer");

    /* Compute output sizes; requantize Q4_0 routing experts to Q4_K so the
     * MoE kernels can dispatch them. */
    for (size_t i = 0; i < n_sel; i++) {
        const tensor_info *t = items[i].t;
        bool expert = strstr(t->name, "_exps.") != NULL;
        if (t->type == 30 || t->type == 1) { /* bf16/f16 -> f32: small head vectors */
            items[i].out_type = 0;
        } else if (expert && t->type == GGML_Q4_0) {
                    if (t->dim[0] % p36q_block_size(expert_type) != 0)
                die("expert row is not a multiple of the target quant block");
            items[i].out_type = expert_ggml;
        }
        size_t row = type_row_size(items[i].out_type, t->dim[0]);
        if (row == 0) die("unsupported tensor type in head block");
        items[i].out_bytes = row * (size_t)(t->elements / t->dim[0]);
        data_bytes = (data_bytes + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
        data_bytes += items[i].out_bytes;
    }
    data_bytes = (data_bytes + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;

    /* Header: same KV bytes, only the selected tensor infos. */
    size_t hdr = kv_end + 8 + 8; /* magic/version + counts */
    hdr = 4 + 4 + 8 + 8 + (kv_end - kv_start);
    size_t info_bytes = 0;
    for (size_t i = 0; i < n_sel; i++) {
        const tensor_info *t = items[i].t;
        info_bytes += 8 + strlen(t->name) + 4 + 8 * t->ndim + 4 + 8;
    }
    /* GGUF keys are padded to the alignment before tensor data. */
    size_t info_end_out = hdr + info_bytes;
    size_t out_data = (info_end_out + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
    size_t total = out_data + data_bytes;

    uint8_t *buf = calloc(1, total);
    if (!buf) die("out of memory");
    uint8_t *p = buf;
    put32(&p, 0x46554747u);
    put32(&p, 3);
    put64(&p, n_sel);
    put64(&p, n_kv);
    memcpy(p, map + kv_start, kv_end - kv_start);
    p += kv_end - kv_start;

    size_t off = 0;
    for (size_t i = 0; i < n_sel; i++) {
        const tensor_info *t = items[i].t;
        put_str(&p, t->name, strlen(t->name));
        put32(&p, t->ndim);
        for (uint32_t d = 0; d < t->ndim; d++) put64(&p, t->dim[d]);
        put32(&p, items[i].out_type);
        put64(&p, off);
        off += items[i].out_bytes;
        off = (off + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
    }

    p36q_quantize_init(expert_type);
    size_t woff = 0;
    for (size_t i = 0; i < n_sel; i++) {
        const tensor_info *t = items[i].t;
        const uint8_t *src = map + data_start + t->offset;
        size_t rows = (size_t)(t->elements / t->dim[0]);
        uint8_t *dst = buf + out_data + woff;
        if (items[i].out_type == 0 && t->type != 0) {
            float *f = malloc((size_t)t->elements * sizeof(float));
            if (!f) die("out of memory");
            if (t->type == 30) {
                for (uint64_t e = 0; e < t->elements; e++) {
                    uint16_t bits;
                    memcpy(&bits, src + 2 * e, 2);
                    f[e] = p36q_bf16_to_f32(bits);
                }
            } else {
                for (uint64_t e = 0; e < t->elements; e++) {
                    uint16_t bits;
                    memcpy(&bits, src + 2 * e, 2);
                    f[e] = p36q_f16_to_f32(bits);
                }
            }
            memcpy(dst, f, (size_t)t->elements * sizeof(float));
            free(f);
        } else if (items[i].out_type != t->type) {
            float *f = malloc((size_t)t->elements * sizeof(float));
            if (!f) die("out of memory");
            dequant_q4_0(src, f, t->elements);
            p36q_quantize_chunk(expert_type, f, dst, 0, (int64_t)rows,
                                (int64_t)t->dim[0], NULL);
            free(f);
            fprintf(stderr, "mtp-head-extract: %s q4_0 -> %s (%zu bytes)\n",
                    t->name, p36q_type_name(expert_type), items[i].out_bytes);
        } else {
            memcpy(dst, src, items[i].out_bytes);
        }
        woff += items[i].out_bytes;
        woff = (woff + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
    }

    int ofd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) die("cannot create output");
    if (write(ofd, buf, total) != (ssize_t)total) die("short write");
    close(ofd);
    fprintf(stderr, "mtp-head-extract: wrote %s (%zu tensors, %.2f MB)\n",
            out_path, n_sel, (double)total / (1024.0 * 1024.0));
    return 0;
}
