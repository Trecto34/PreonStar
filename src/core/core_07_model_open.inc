static void p36_rms_norm_rows(float *dst,
                              const float *src,
                              const float *weight,
                              uint32_t rows,
                              uint32_t cols,
                              float eps,
                              uint32_t n_threads) {
    p36_rms_norm_rows_ctx ctx;
    if (!dst || !src || rows == 0) return;
    ctx.dst = dst;
    ctx.src = src;
    ctx.weight = weight;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.eps = eps;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_rms_norm_rows_worker, &ctx);
}

static void p36_swiglu_rows(float *out,
                            const float *gate,
                            const float *up,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t n_threads) {
    p36_swiglu_rows_ctx ctx;
    if (!out || !gate || !up || rows == 0) return;
    ctx.out = out;
    ctx.gate = gate;
    ctx.up = up;
    ctx.rows = rows;
    ctx.cols = cols;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_swiglu_rows_worker, &ctx);
}

static bool p36_tensor_matvec_prequant(const p36_engine *e,
                                       const p36_tensor *t,
                                       const float *x,
                                       const uint8_t *xq,
                                       float *out,
                                       float *rowbuf,
                                       uint32_t in_dim,
                                       uint32_t out_dim) {
    (void)rowbuf;
    return p36_tensor_matmul_batch_prequant(e, t, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool p36_tensor_matvec_pair_prequant(const p36_engine *e,
                                            const p36_tensor *t0,
                                            const p36_tensor *t1,
                                            const float *x,
                                            const uint8_t *xq,
                                            float *out0,
                                            float *out1,
                                            float *rowbuf,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            float scale0,
                                            float scale1) {
    (void)rowbuf;
    return p36_tensor_matmul_pair_batch_prequant(e, t0, t1, x, xq, out0, out1, 1, in_dim, out_dim, scale0, scale1);
}

static bool p36_tensor_matvec(const p36_engine *e, const p36_tensor *t, const float *x, float *out,
                              float *rowbuf, uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    p36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE &&
        p36_activation_quant_valid_dim(kind, in_dim) &&
        p36_activation_quant_row_bytes(kind, in_dim) <= P36_MAX_Q8_K_BYTES &&
        p36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return p36_tensor_matvec_prequant(e, t, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static bool p36_tensor_expert_matvec_prequant(const p36_engine *e,
                                              const p36_tensor *t,
                                              uint32_t expert,
                                              const float *x,
                                              const uint8_t *xq,
                                              float *out,
                                              float *rowbuf,
                                              uint32_t in_dim,
                                              uint32_t out_dim) {
    (void)rowbuf;
    return p36_tensor_expert_matmul_batch_prequant(e, t, expert, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool p36_tensor_expert_matvec_pair_prequant(const p36_engine *e,
                                                   const p36_tensor *t0,
                                                   const p36_tensor *t1,
                                                   uint32_t expert,
                                                   const float *x,
                                                   const uint8_t *xq,
                                                   float *out0,
                                                   float *out1,
                                                   float *rowbuf,
                                                   uint32_t in_dim,
                                                   uint32_t out_dim,
                                                   float scale0,
                                                   float scale1) {
    (void)rowbuf;
    return p36_tensor_expert_matmul_pair_batch_prequant(e, t0, t1, expert,
                                                         x, xq, out0, out1,
                                                         1, in_dim, out_dim,
                                                         scale0, scale1);
}

static bool p36_tensor_expert_matvec(const p36_engine *e, const p36_tensor *t, uint32_t expert,
                                     const float *x, float *out, float *rowbuf,
                                     uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    p36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE &&
        p36_activation_quant_valid_dim(kind, in_dim) &&
        p36_activation_quant_row_bytes(kind, in_dim) <= P36_MAX_Q8_K_BYTES &&
        p36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return p36_tensor_expert_matvec_prequant(e, t, expert, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static void cursor_error(p36_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(p36_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(p36_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(p36_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(p36_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(p36_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(p36_cursor *c, p36_str *s) {
    uint64_t len = 0;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    if (rem == 0) return value;
    if (value > UINT64_MAX - (alignment - rem)) p36_die("GGUF alignment overflow");
    return value + alignment - rem;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(p36_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    {
        uint64_t scalar = scalar_value_size(type);
        if (scalar != 0) return cursor_skip(c, scalar);
    }

    if (type == GGUF_VALUE_STRING) {
        p36_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t len = 0;
        uint64_t item_size = 0;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;
        item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = (uint32_t)(sizeof(gguf_types) / sizeof(gguf_types[0]));
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    uint64_t blocks = 0;
    if (!info || info->block_elems == 0) return false;
    blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static p36_cursor cursor_at(const p36_model *m, uint64_t pos) {
    p36_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.error[0] = '\0';
    return c;
}

static p36_kv *model_find_kv(const p36_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (p36_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const p36_model *m, const char *key, p36_str *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const p36_model *m, const char *key, uint32_t *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64_compat(const p36_model *m, const char *key, uint64_t *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32_compat(const p36_model *m, const char *key, float *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) return cursor_read(&c, out, sizeof(*out));
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const p36_model *m, const char *key, bool *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    uint8_t v = 0;
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

static bool model_get_array(const p36_model *m, const char *key, p36_array_ref *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(p36_model *m) {
    if (!m) return;
    if (m->owned) {
        free(m->kv);
        free(m->tensors);
        if (m->map) munmap((void *)m->map, (size_t)m->map_len);
        if (m->fd >= 0) close(m->fd);
    }
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void parse_metadata(p36_model *m, p36_cursor *c) {
    uint64_t remaining = c->size - c->pos;
    if (m->n_kv > remaining / 12 ||
        m->n_kv > SIZE_MAX / sizeof(m->kv[0]))
        p36_die("GGUF metadata count exceeds file size");
    m->kv = xcalloc((size_t)m->n_kv, sizeof(m->kv[0]));
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        p36_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) p36_die(c->error);
        if (!cursor_u32(c, &kv->type)) p36_die(c->error);
        kv->value_pos = c->pos;
        if (p36_streq(kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
            p36_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t align = 0;
            if (cursor_u32(&tmp, &align) && align != 0) m->alignment = align;
        }
        if (!skip_value(c, kv->type, 0)) p36_die(c->error);
    }
}

/* True for a tensor of the appended NextN/MTP block (blk.<block_count-1>.*).
 * Some third-party GGUFs store that block in types the loader has no decoder
 * for (e.g. Q4_0); it is only read when the MTP head is bound, so such tensors
 * are kept as zero-byte placeholders instead of rejecting the whole file. */
static bool tensor_in_appended_nextn_block(const p36_model *m, p36_str name) {
    p36_str arch;
    char key[96], prefix[24];
    uint32_t block_count = 0, nextn = 0;
    int len;
    if (!model_get_string(m, "general.architecture", &arch) || arch.len > 64) return false;
    snprintf(key, sizeof(key), "%.*s.block_count", (int)arch.len, arch.ptr);
    if (!model_get_u32(m, key, &block_count) || block_count == 0) return false;
    snprintf(key, sizeof(key), "%.*s.nextn_predict_layers", (int)arch.len, arch.ptr);
    if (!model_get_u32(m, key, &nextn) || nextn != 1) return false;
    len = snprintf(prefix, sizeof(prefix), "blk.%u.", block_count - 1);
    return len > 0 && (uint64_t)len <= name.len && memcmp(name.ptr, prefix, (size_t)len) == 0;
}

/* Re-encode Q4_0 tensors as Q8_0 into an anonymous arena right after the file
 * in the reserved address range, so downstream code keeps addressing weights as
 * map + abs_offset.  Lossless: Q4_0 value (nibble - 8) is in [-8, 7] and keeps
 * the block's f16 scale, so the dequantized weights are bit-identical. */
static void model_convert_q4_0(p36_model *m) {
    const uint64_t arena_off = align_up(m->file_size, 1u << 20);
    uint64_t used = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            if (used == 0) return;
            uint64_t len = align_up(used, 1u << 20);
            if (arena_off + len > m->map_len ||
                mmap((void *)(m->map + arena_off), (size_t)len, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED)
                p36_die("cannot map the Q4_0 conversion arena");
            used = 0;
        }
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            p36_tensor *t = &m->tensors[i];
            if (!t->from_q4_0) continue;
            uint64_t src = m->tensor_data_pos + t->rel_offset;
            uint64_t q4_bytes = t->elements / P36_QK8_0 * P36_Q4_0_BYTES;
            if (src > m->file_size || q4_bytes > m->file_size - src)
                p36_die("tensor points outside GGUF file");
            uint64_t dst = align_up(used, 32);
            used = dst + t->bytes;
            if (pass == 0) continue;
            const uint8_t *q4 = m->map + src;
            p36_block_q8_0 *q8 = (p36_block_q8_0 *)(void *)((uint8_t *)m->map + arena_off + dst);
            for (uint64_t b = 0; b < t->elements / P36_QK8_0; b++, q4 += P36_Q4_0_BYTES) {
                memcpy(&q8[b].d, q4, sizeof(q8[b].d));
                for (uint32_t j = 0; j < 16; j++) {
                    q8[b].qs[j] = (int8_t)((q4[2 + j] & 15u) - 8);
                    q8[b].qs[j + 16] = (int8_t)((q4[2 + j] >> 4) - 8);
                }
            }
            t->abs_offset = arena_off + dst;
            if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
        }
    }
    mprotect((void *)(m->map + arena_off), (size_t)align_up(used, 1u << 20), PROT_READ);
    m->size = arena_off + used;
}

static void parse_tensors(p36_model *m, p36_cursor *c) {
    uint64_t remaining = c->size - c->pos;
    if (m->n_tensors > remaining / 32 ||
        m->n_tensors > SIZE_MAX / sizeof(m->tensors[0]))
        p36_die("GGUF tensor count exceeds file size");
    m->tensors = xcalloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        p36_tensor *t = &m->tensors[i];
        if (!cursor_string(c, &t->name)) p36_die(c->error);
        if (!cursor_u32(c, &t->ndim)) p36_die(c->error);
        if (t->ndim == 0 || t->ndim > P36_MAX_DIMS) p36_die("tensor has an unsupported number of dimensions");
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) p36_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) p36_die("tensor element count overflow");
            t->elements *= t->dim[d];
        }
        if (!cursor_u32(c, &t->type)) p36_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) p36_die(c->error);
        if (t->type == P36_TENSOR_Q4_0 && !tensor_in_appended_nextn_block(m, t->name)) {
            /* Q4_0 -> Q8_0 is exact (q-8 fits int8, same f16 scale), so the
             * whole Q8_0 kernel set serves these tensors; see
             * model_convert_q4_0. */
            if ((t->elements % P36_QK8_0) != 0) p36_die("q4_0 tensor row is not a multiple of 32");
            t->from_q4_0 = true;
            t->type = P36_TENSOR_Q8_0;
            t->bytes = t->elements / P36_QK8_0 * sizeof(p36_block_q8_0);
            continue;
        }
        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            if (tensor_in_appended_nextn_block(m, t->name)) {
                t->bytes = 0;
                continue;
            }
            fprintf(stderr, "p36: unsupported tensor type %u for %.*s\n", t->type, (int)t->name.len, t->name.ptr);
            exit(1);
        }
    }
    m->tensor_data_pos = align_up(c->pos, m->alignment);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        p36_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) p36_die("tensor offset overflow");
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->from_q4_0) continue;   /* placed by model_convert_q4_0 */
        if (t->bytes != 0 && (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            p36_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
    }
    model_convert_q4_0(m);
}

