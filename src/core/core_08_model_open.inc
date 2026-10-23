static void model_open(p36_model *m, const char *path, bool graph_mapping) {
    struct stat st;
    int fd;
    void *map;
    p36_cursor c;
    uint32_t magic = 0;
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    fd = open(path, O_RDONLY);
    if (fd == -1) p36_die_errno("cannot open model", path);
    if (fstat(fd, &st) == -1) p36_die_errno("cannot stat model", path);
    if (st.st_size < 32) p36_die("model file is too small to be GGUF");
    if ((uintmax_t)st.st_size > SIZE_MAX) p36_die("model file is too large to map");
    /* Reserve twice the file in address space (never committed) so a Q4_0
     * conversion arena can follow the file mapping; see model_convert_q4_0. */
    uint64_t reserve = align_up((uint64_t)st.st_size, 1u << 20) + 2u * (uint64_t)st.st_size;
    map = mmap(NULL, (size_t)reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map == MAP_FAILED) p36_die_errno("cannot reserve model address range", path);
    if (mmap(map, (size_t)st.st_size, PROT_READ, (graph_mapping ? MAP_SHARED : MAP_PRIVATE) | MAP_FIXED, fd, 0) == MAP_FAILED)
        p36_die_errno("cannot mmap model", path);
    m->fd = fd;
    m->map = map;
    m->map_len = reserve;
    m->size = m->file_size = (uint64_t)st.st_size;
    m->owned = true;
    c = cursor_at(m, 0);
    if (!cursor_u32(&c, &magic)) p36_die(c.error);
    if (magic != P36_GGUF_MAGIC) p36_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) p36_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) p36_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) p36_die(c.error);
    if (m->version != 3) p36_die("only GGUF v3 is supported");
    parse_metadata(m, &c);
    parse_tensors(m, &c);
}

static p36_tensor *model_find_tensor(const p36_model *m, const char *name) {
    size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len && memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static uint32_t required_u32(const p36_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const p36_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static p36_str required_string(const p36_model *m, const char *key) {
    p36_str s = {0};
    if (!model_get_string(m, key, &s)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return s;
}

static p36_tensor *required_tensor(const p36_model *m, const char *name) {
    p36_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "p36: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static p36_tensor *tensor_by_namef(const p36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) p36_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static p36_tensor *required_tensorf(const p36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) p36_die("tensor name is too long");
    return required_tensor(m, name);
}

typedef struct {
    const p36_model *model;
    const p36_tensor *weight;
    const float *input;
    float *output;
    uint32_t rows;
    uint32_t in_dim;
    uint32_t out_dim;
    bool add;
} p36_vision_mm_ctx;

static bool p36_vision_weight_type_ok(const p36_tensor *t, uint32_t in_dim) {
    if (!t) return false;
    if (t->type == P36_TENSOR_F16 || t->type == P36_TENSOR_F32) return true;
    return t->type == P36_TENSOR_Q8_0 && in_dim % P36_QK8_0 == 0;
}

/* Dequantize output row `o` of a vision weight into dst[in_dim]. */
static void p36_vision_weight_row_f32(const p36_model *m, const p36_tensor *t,
                                      uint32_t in_dim, uint64_t o, float *dst) {
    const uint8_t *base = m->map + t->abs_offset;
    if (t->type == P36_TENSOR_F32) {
        memcpy(dst, (const float *)base + o * in_dim, (size_t)in_dim * sizeof(float));
    } else if (t->type == P36_TENSOR_F16) {
        const uint16_t *wr = (const uint16_t *)base + o * in_dim;
        for (uint32_t i = 0; i < in_dim; i++) dst[i] = p36_f16_to_f32(wr[i]);
    } else {
        const uint32_t nb = in_dim / P36_QK8_0;
        const p36_block_q8_0 *b = (const p36_block_q8_0 *)base + o * nb;
        for (uint32_t k = 0; k < nb; k++) {
            const float d = p36_f16_to_f32(b[k].d);
            for (uint32_t j = 0; j < P36_QK8_0; j++)
                dst[k * P36_QK8_0 + j] = d * (float)b[k].qs[j];
        }
    }
}

static void p36_vision_mm_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_mm_ctx *c = opaque;
    if (c->weight->type != P36_TENSOR_F16) {
        float *wrow = malloc((size_t)c->in_dim * sizeof(float));
        if (!wrow) p36_die("out of memory");
        for (uint64_t o = row0; o < row1; o++) {
            p36_vision_weight_row_f32(c->model, c->weight, c->in_dim, o, wrow);
            for (uint32_t r = 0; r < c->rows; r++) {
                const float *x = c->input + (uint64_t)r * c->in_dim;
                double sum = 0.0;
                for (uint32_t i = 0; i < c->in_dim; i++)
                    sum += (double)wrow[i] * x[i];
                float *dst = c->output + (uint64_t)r * c->out_dim + o;
                if (c->add) *dst += (float)sum;
                else *dst = (float)sum;
            }
        }
        free(wrow);
        return;
    }
    const uint16_t *weight = (const uint16_t *)(c->model->map + c->weight->abs_offset);
    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *wr = weight + o * c->in_dim;
        for (uint32_t r = 0; r < c->rows; r++) {
            const float *x = c->input + (uint64_t)r * c->in_dim;
            double sum = 0.0;
            for (uint32_t i = 0; i < c->in_dim; i++)
                sum += (double)p36_f16_to_f32(wr[i]) * x[i];
            float *dst = c->output + (uint64_t)r * c->out_dim + o;
            if (c->add) *dst += (float)sum;
            else *dst = (float)sum;
        }
    }
}

#if !defined(P36_NO_GPU) && !defined(P36_METAL)
typedef struct {
    const p36_model *model;
    const p36_tensor *weight;
    uint32_t in_dim;
    uint16_t *dst;
} p36_vision_fill_ctx;

static void p36_vision_fill_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_fill_ctx *c = opaque;
    const uint8_t *base = c->model->map + c->weight->abs_offset;
    for (uint64_t o = row0; o < row1; o++) {
        uint16_t *out = c->dst + o * c->in_dim;
        if (c->weight->type == P36_TENSOR_F32) {
            const float *src = (const float *)base + o * c->in_dim;
            for (uint32_t i = 0; i < c->in_dim; i++) out[i] = p36_f32_to_f16(src[i]);
        } else {
            const uint32_t nb = c->in_dim / P36_QK8_0;
            const p36_block_q8_0 *b = (const p36_block_q8_0 *)base + o * nb;
            for (uint32_t k = 0; k < nb; k++) {
                const float d = p36_f16_to_f32(b[k].d);
                for (uint32_t j = 0; j < P36_QK8_0; j++)
                    out[k * P36_QK8_0 + j] = p36_f32_to_f16(d * (float)b[k].qs[j]);
            }
        }
    }
}

typedef struct {
    p36_vision_fill_ctx fill;
    uint32_t out_dim;
    uint32_t n_threads;
} p36_vision_fill_job;

static int p36_vision_fill_cb(void *opaque, uint16_t *dst) {
    p36_vision_fill_job *j = opaque;
    j->fill.dst = dst;
    p36_parallel_for_rows(j->out_dim, 16, j->n_threads, p36_vision_fill_rows, &j->fill);
    return 1;
}
#endif

static bool p36_vision_mm(p36_engine *e, const p36_tensor *weight,
                          const float *input, float *output,
                          uint32_t rows, uint32_t in_dim, uint32_t out_dim,
                          bool add) {
    p36_model *m = &e->vision_model;
    if (!p36_vision_weight_type_ok(weight, in_dim) ||
        weight->elements != (uint64_t)in_dim * out_dim ||
        (weight->ndim == 2 &&
         (weight->dim[0] != in_dim || weight->dim[1] != out_dim)))
        return false;
    if (e->backend == P36_BACKEND_VULKAN) {
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
        uint64_t in_bytes = (uint64_t)rows * in_dim * sizeof(float);
        uint64_t out_bytes = (uint64_t)rows * out_dim * sizeof(float);
        p36_gpu_tensor *gx = p36_gpu_tensor_alloc_uninitialized(in_bytes);
        p36_gpu_tensor *gy = p36_gpu_tensor_alloc_uninitialized(out_bytes);
        float *tmp = add ? malloc((size_t)out_bytes) : output;
        p36_vision_fill_job fj = {{m, weight, in_dim, NULL}, out_dim, e->n_threads};
        int ok = gx && gy && tmp && p36_gpu_tensor_write(gx, 0, input, in_bytes) &&
            (weight->type == P36_TENSOR_F16
                 ? p36_gpu_vision_matmul_f16_disk(gy, m->fd, weight->abs_offset,
                                                  in_dim, out_dim, gx, rows)
                 : p36_gpu_vision_matmul_f16_fill(gy, p36_vision_fill_cb, &fj,
                                                  in_dim, out_dim, gx, rows)) &&
            p36_gpu_tensor_read(gy, 0, tmp, out_bytes);
        if (ok && add)
            for (uint64_t i = 0, n = (uint64_t)rows * out_dim; i < n; i++)
                output[i] += tmp[i];
        if (add) free(tmp);
        p36_gpu_tensor_free(gx);
        p36_gpu_tensor_free(gy);
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(m->fd, (off_t)weight->abs_offset, (off_t)weight->bytes,
                      POSIX_FADV_DONTNEED);
#endif
        if (ok) return true;
        return false;
#endif
    }
    p36_vision_mm_ctx c = {m, weight, input, output, rows, in_dim, out_dim, add};
    p36_parallel_for_rows(out_dim, 4, e->n_threads, p36_vision_mm_worker, &c);
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(m->fd, (off_t)weight->abs_offset, (off_t)weight->bytes,
                  POSIX_FADV_DONTNEED);
#endif
    return true;
}

static const float *p36_vision_f32(const p36_model *m, const p36_tensor *t,
                                   uint32_t n) {
    if (!t || t->type != P36_TENSOR_F32 || t->elements != n) return NULL;
    return (const float *)(m->map + t->abs_offset);
}

static void p36_vision_layer_norm(float *out, const float *x,
                                  const float *weight, const float *bias,
                                  uint32_t rows, uint32_t width) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x + (uint64_t)r * width;
        float *yr = out + (uint64_t)r * width;
        double mean = 0.0, variance = 0.0;
        for (uint32_t i = 0; i < width; i++) mean += xr[i];
        mean /= width;
        for (uint32_t i = 0; i < width; i++) {
            double d = xr[i] - mean;
            variance += d * d;
        }
        float inv = 1.0f / sqrtf((float)(variance / width) + 1.0e-6f);
        for (uint32_t i = 0; i < width; i++)
            yr[i] = (xr[i] - (float)mean) * inv * weight[i] + bias[i];
    }
}

static void p36_vision_bias_residual(float *x, const float *bias,
                                     const float *residual,
                                     uint32_t rows, uint32_t width) {
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++)
        x[i] += bias[i % width] + (residual ? residual[i] : 0.0f);
}

static void p36_vision_gelu_bias(float *x, const float *bias,
                                 uint32_t rows, uint32_t width) {
    const float k = 0.7071067811865475f;
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++) {
        float v = x[i] + bias[i % width];
        x[i] = 0.5f * v * (1.0f + erff(v * k));
    }
}

typedef struct {
    const float *qkv;
    float *out;
    uint32_t rows;
} p36_vision_attn_ctx;

static void p36_vision_attention_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_attn_ctx *c = opaque;
    const float scale = 1.0f / sqrtf(72.0f);
    for (uint64_t job = row0; job < row1; job++) {
        uint32_t row = (uint32_t)(job / 16u), head = (uint32_t)(job % 16u);
        const float *q = c->qkv + (uint64_t)row * 3456u + head * 72u;
        float *dst = c->out + (uint64_t)row * 1152u + head * 72u;
        float max_score = -INFINITY, denom = 0.0f;
        memset(dst, 0, 72u * sizeof(*dst));
        for (uint32_t kr = 0; kr < c->rows; kr++) {
            const float *k = c->qkv + (uint64_t)kr * 3456u + 1152u + head * 72u;
            const float *v = c->qkv + (uint64_t)kr * 3456u + 2304u + head * 72u;
            float score = 0.0f;
            for (uint32_t d = 0; d < 72u; d++) score += q[d] * k[d];
            score *= scale;
            float next = fmaxf(max_score, score);
            float old_scale = isfinite(max_score) ? expf(max_score - next) : 0.0f;
            float new_scale = expf(score - next);
            denom = denom * old_scale + new_scale;
            for (uint32_t d = 0; d < 72u; d++)
                dst[d] = dst[d] * old_scale + new_scale * v[d];
            max_score = next;
        }
        for (uint32_t d = 0; d < 72u; d++) dst[d] /= denom;
    }
}

static bool p36_vision_attention(p36_engine *e, const float *qkv,
                                 float *out, uint32_t rows) {
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
    if (e->backend == P36_BACKEND_VULKAN) {
        uint64_t qkv_bytes = (uint64_t)rows * 3456u * sizeof(float);
        uint64_t out_bytes = (uint64_t)rows * 1152u * sizeof(float);
        p36_gpu_tensor *gqkv = p36_gpu_tensor_alloc_uninitialized(qkv_bytes);
        p36_gpu_tensor *gout = p36_gpu_tensor_alloc_uninitialized(out_bytes);
        int ok = gqkv && gout && p36_gpu_tensor_write(gqkv, 0, qkv, qkv_bytes) &&
            p36_gpu_vision_attention(gout, gqkv, rows) &&
            p36_gpu_tensor_read(gout, 0, out, out_bytes);
        p36_gpu_tensor_free(gqkv);
        p36_gpu_tensor_free(gout);
        return ok != 0;
    }
#endif
    p36_vision_attn_ctx ac = {qkv, out, rows};
    p36_parallel_for_rows((uint64_t)rows * 16u, 8, e->n_threads,
                          p36_vision_attention_worker, &ac);
    return true;
}

static void p36_vision_rope(float *qkv, uint32_t rows, uint32_t grid_width) {
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t group = r / 4u, within = r & 3u;
        uint32_t y = (group / (grid_width / 2u)) * 2u + within / 2u;
        uint32_t x = (group % (grid_width / 2u)) * 2u + within % 2u;
        for (uint32_t part = 0; part < 2u; part++) {
            for (uint32_t h = 0; h < 16u; h++) {
                float *v = qkv + (uint64_t)r * 3456u + part * 1152u + h * 72u;
                for (uint32_t p = 0; p < 36u; p++) {
                    uint32_t pos = p < 18u ? y : x;
                    uint32_t freq = p % 18u;
                    float angle = pos * powf(10000.0f, -(float)freq / 18.0f);
                    float cs = cosf(angle), sn = sinf(angle);
                    float a = v[p], b = v[p + 36u];
                    v[p] = a * cs - b * sn;
                    v[p + 36u] = b * cs + a * sn;
                }
            }
        }
    }
}

static p36_tensor *p36_vision_tensorf(const p36_model *m, uint32_t layer,
                                      const char *suffix) {
    char name[96];
    int n = snprintf(name, sizeof(name), "v.blk.%u.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return NULL;
    return model_find_tensor(m, name);
}

static int p36_vision_encode_image(p36_engine *e, const p36_image *image,
                                   p36_vision_embedding *out,
                                   char *err, size_t errlen) {
    p36_model *m = &e->vision_model;
    p36_image_patches patches = {0};
    float *x = NULL, *norm = NULL, *qkv = NULL, *work = NULL, *wide = NULL;
    int ok = 0;
    if (!e->vision_ready || !p36_image_preprocess_qwen3vl(
            &patches, image, 1024u, err, errlen)) return 0;
    uint32_t rows = patches.patch_count;
    uint32_t merged = rows / 4u;
    uint64_t base_values = (uint64_t)rows * 1152u;
    x = malloc(base_values * sizeof(*x));
    norm = malloc(base_values * sizeof(*norm));
    work = malloc(base_values * sizeof(*work));
    qkv = malloc((uint64_t)rows * 3456u * sizeof(*qkv));
    wide = malloc((uint64_t)rows * 4304u * sizeof(*wide));
    if (!x || !norm || !work || !qkv || !wide) {
        if (err && errlen) snprintf(err, errlen, "unable to allocate vision workspace");
        goto done;
    }

    p36_tensor *patch0 = model_find_tensor(m, "v.patch_embd.weight");
    p36_tensor *patch1 = model_find_tensor(m, "v.patch_embd.weight.1");
    p36_tensor *patch_bias_t = model_find_tensor(m, "v.patch_embd.bias");
    p36_tensor *position_t = model_find_tensor(m, "v.position_embd.weight");
    const float *patch_bias = p36_vision_f32(m, patch_bias_t, 1152u);
    const float *position = p36_vision_f32(m, position_t, 1152u * 2304u);
    if (!patch_bias || !position ||
        !p36_vision_mm(e, patch0, patches.patches, x, rows, 768u, 1152u, false) ||
        !p36_vision_mm(e, patch1, patches.patches, x, rows, 768u, 1152u, true)) {
        if (err && errlen) snprintf(err, errlen, "invalid Qwen3-VL patch tensors");
        goto done;
    }
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t group = r / 4u, within = r & 3u;
        uint32_t gy = (group / (patches.grid_width / 2u)) * 2u + within / 2u;
        uint32_t gx = (group % (patches.grid_width / 2u)) * 2u + within % 2u;
        float sy = patches.grid_height == 1 ? 0.0f : gy * 47.0f / (patches.grid_height - 1u);
        float sx = patches.grid_width == 1 ? 0.0f : gx * 47.0f / (patches.grid_width - 1u);
        uint32_t y0 = (uint32_t)floorf(sy), x0 = (uint32_t)floorf(sx);
        uint32_t y1 = y0 < 47u ? y0 + 1u : y0, x1 = x0 < 47u ? x0 + 1u : x0;
        float fy = sy - y0, fx = sx - x0;
        for (uint32_t d = 0; d < 1152u; d++) {
            float a = position[((uint64_t)y0 * 48u + x0) * 1152u + d];
            float b = position[((uint64_t)y0 * 48u + x1) * 1152u + d];
            float c = position[((uint64_t)y1 * 48u + x0) * 1152u + d];
            float z = position[((uint64_t)y1 * 48u + x1) * 1152u + d];
            x[(uint64_t)r * 1152u + d] += patch_bias[d] +
                (a + (b - a) * fx) * (1.0f - fy) + (c + (z - c) * fx) * fy;
        }
    }

    for (uint32_t il = 0; il < 27u; il++) {
        p36_tensor *ln1w = p36_vision_tensorf(m, il, "ln1.weight");
        p36_tensor *ln1b = p36_vision_tensorf(m, il, "ln1.bias");
        p36_tensor *qkvw = p36_vision_tensorf(m, il, "attn_qkv.weight");
        p36_tensor *qkvb = p36_vision_tensorf(m, il, "attn_qkv.bias");
        p36_tensor *outw = p36_vision_tensorf(m, il, "attn_out.weight");
        p36_tensor *outb = p36_vision_tensorf(m, il, "attn_out.bias");
        p36_tensor *ln2w = p36_vision_tensorf(m, il, "ln2.weight");
        p36_tensor *ln2b = p36_vision_tensorf(m, il, "ln2.bias");
        p36_tensor *upw = p36_vision_tensorf(m, il, "ffn_up.weight");
        p36_tensor *upb = p36_vision_tensorf(m, il, "ffn_up.bias");
        p36_tensor *downw = p36_vision_tensorf(m, il, "ffn_down.weight");
        p36_tensor *downb = p36_vision_tensorf(m, il, "ffn_down.bias");
        const float *ln1wp = p36_vision_f32(m, ln1w, 1152u);
        const float *ln1bp = p36_vision_f32(m, ln1b, 1152u);
        const float *qkvbp = p36_vision_f32(m, qkvb, 3456u);
        const float *outbp = p36_vision_f32(m, outb, 1152u);
        const float *ln2wp = p36_vision_f32(m, ln2w, 1152u);
        const float *ln2bp = p36_vision_f32(m, ln2b, 1152u);
        const float *upbp = p36_vision_f32(m, upb, 4304u);
        const float *downbp = p36_vision_f32(m, downb, 1152u);
        if (!ln1wp || !ln1bp || !qkvbp || !outbp || !ln2wp || !ln2bp ||
            !upbp || !downbp) {
            if (err && errlen) snprintf(err, errlen, "invalid vision block %u", il);
            goto done;
        }
        p36_vision_layer_norm(norm, x, ln1wp, ln1bp, rows, 1152u);
        if (!p36_vision_mm(e, qkvw, norm, qkv, rows, 1152u, 3456u, false)) goto bad_mm;
        p36_vision_bias_residual(qkv, qkvbp, NULL, rows, 3456u);
        p36_vision_rope(qkv, rows, patches.grid_width);
        if (!p36_vision_attention(e, qkv, work, rows)) goto bad_mm;
        if (!p36_vision_mm(e, outw, work, norm, rows, 1152u, 1152u, false)) goto bad_mm;
        p36_vision_bias_residual(norm, outbp, x, rows, 1152u);
        memcpy(x, norm, base_values * sizeof(*x));
        p36_vision_layer_norm(norm, x, ln2wp, ln2bp, rows, 1152u);
        if (!p36_vision_mm(e, upw, norm, wide, rows, 1152u, 4304u, false)) goto bad_mm;
        p36_vision_gelu_bias(wide, upbp, rows, 4304u);
        if (!p36_vision_mm(e, downw, wide, norm, rows, 4304u, 1152u, false)) goto bad_mm;
        p36_vision_bias_residual(norm, downbp, x, rows, 1152u);
        memcpy(x, norm, base_values * sizeof(*x));
        continue;
bad_mm:
        if (err && errlen) snprintf(err, errlen, "invalid vision matmul in block %u", il);
        goto done;
    }

    {
        const float *postw = p36_vision_f32(m, model_find_tensor(m, "v.post_ln.weight"), 1152u);
        const float *postb = p36_vision_f32(m, model_find_tensor(m, "v.post_ln.bias"), 1152u);
        p36_tensor *mm0 = model_find_tensor(m, "mm.0.weight");
        const float *mm0b = p36_vision_f32(m, model_find_tensor(m, "mm.0.bias"), 4608u);
        p36_tensor *mm2 = model_find_tensor(m, "mm.2.weight");
        const float *mm2b = p36_vision_f32(m, model_find_tensor(m, "mm.2.bias"), P36_N_EMBD);
        float *merged_in = qkv;
        if (!postw || !postb || !mm0b || !mm2b) {
            if (err && errlen) snprintf(err, errlen, "invalid vision merger tensors");
            goto done;
        }
        p36_vision_layer_norm(norm, x, postw, postb, rows, 1152u);
        memcpy(merged_in, norm, base_values * sizeof(*norm));
        if (!p36_vision_mm(e, mm0, merged_in, wide, merged, 4608u, 4608u, false)) goto merger_bad;
        p36_vision_gelu_bias(wide, mm0b, merged, 4608u);
        out->data = malloc((uint64_t)merged * P36_N_EMBD * sizeof(*out->data));
        if (!out->data) {
            if (err && errlen) snprintf(err, errlen, "unable to allocate vision embedding");
            goto done;
        }
        if (!p36_vision_mm(e, mm2, wide, out->data, merged, 4608u, P36_N_EMBD, false)) goto merger_bad;
        p36_vision_bias_residual(out->data, mm2b, NULL, merged, P36_N_EMBD);
        out->token_count = merged;
        out->grid_width = patches.grid_width / 2u;
        out->grid_height = patches.grid_height / 2u;
        out->width = image->width;
        out->height = image->height;
        memcpy(out->fingerprint, image->fingerprint, sizeof(out->fingerprint));
        ok = 1;
        goto done;
merger_bad:
        free(out->data);
        memset(out, 0, sizeof(*out));
        if (err && errlen) snprintf(err, errlen, "invalid vision merger matmul");
    }
done:
    free(x); free(norm); free(qkv); free(work); free(wide);
    p36_image_patches_free(&patches);
#ifdef MADV_DONTNEED
    madvise((void *)m->map, (size_t)m->file_size, MADV_DONTNEED);
#endif
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(m->fd, 0, (off_t)m->file_size, POSIX_FADV_DONTNEED);
#endif
    return ok;
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "p36: expected %s=%u, got %u\n", name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "p36: expected %s=%.9g, got %.9g\n", name, (double)expected, (double)got);
    exit(1);
}

static void config_expect_string(const char *name, p36_str got, const char *expected) {
    if (p36_streq(got, expected)) return;
    fprintf(stderr, "p36: expected %s=%s\n", name, expected);
    exit(1);
}

static void config_expect_u32_array(const p36_model *m, const char *key, const uint32_t *expected, uint64_t n) {
    p36_array_ref arr;
    p36_cursor c;
    if (!model_get_array(m, key, &arr) || (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "p36: required metadata array is missing: %s\n", key);
        exit(1);
    }
    if (arr.len != n) {
        fprintf(stderr, "p36: expected %s length=%" PRIu64 ", got %" PRIu64 "\n", key, n, arr.len);
        exit(1);
    }
    c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) p36_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) p36_die(c.error);
            if (v < 0) p36_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }
        if (got != expected[i]) {
            fprintf(stderr, "p36: unexpected %s[%" PRIu64 "]=%u, expected %u\n", key, i, got, expected[i]);
            exit(1);
        }
    }
}

static bool p36_layer_is_full_attention(uint32_t il) {
    return ((il + 1u) % P36_FULL_ATTENTION_INTERVAL) == 0;
}

static bool tensor_type_is_routed_gate_up(uint32_t type) {
    return type == P36_TENSOR_Q4_K || type == P36_TENSOR_Q3_K ||
           type == P36_TENSOR_IQ2_XXS || type == P36_TENSOR_IQ2_S || type == P36_TENSOR_IQ3_XXS ||
           type == P36_TENSOR_IQ3_S ||
           type == P36_TENSOR_Q8_0;
}

static bool tensor_type_is_routed_down(uint32_t type) {
    return type == P36_TENSOR_Q2_K || type == P36_TENSOR_Q3_K ||
           type == P36_TENSOR_Q4_K ||
           type == P36_TENSOR_Q5_K || type == P36_TENSOR_Q6_K ||
           type == P36_TENSOR_IQ2_S || type == P36_TENSOR_IQ3_S ||
           type == P36_TENSOR_IQ3_XXS ||
           type == P36_TENSOR_Q8_0;
}

static void tensor_expect_layout(const p36_tensor *t, uint32_t type, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
        exit(1);
    }
}

static void tensor_expect_plain_layout(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != P36_TENSOR_F32 && t->type != P36_TENSOR_F16) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected f32 or f16\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q8_0(const p36_tensor *t, uint32_t type,
                                         uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != P36_TENSOR_Q8_0) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q6_k_or_q8_0(const p36_tensor *t, uint32_t type,
                                                uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != P36_TENSOR_Q6_K && t->type != P36_TENSOR_Q8_0) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s, q6_k, or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_optional_plain(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (t) tensor_expect_plain_layout(t, ndim, d0, d1, d2);
}

static bool tensor_type_is_cpu_matrix(uint32_t type) {
    switch (type) {
    case P36_TENSOR_F32:
    case P36_TENSOR_F16:
    case P36_TENSOR_BF16:
    case P36_TENSOR_Q8_0:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return true;
    default:
        return false;
    }
}

static void tensor_expect_cpu_matrix(const p36_tensor *t,
                                     uint64_t in_dim, uint64_t out_dim) {
    if (!t) p36_die("internal error: missing matrix tensor while validating layout");
    if (!tensor_type_is_cpu_matrix(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has unsupported CPU matrix type %s\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, 2, in_dim, out_dim, 0);
}

static void tensor_expect_moe_matrix(const p36_tensor *t, bool mixed_quants,
                                     uint32_t baseline_type, bool allow_q6_k,
                                     uint64_t in_dim, uint64_t out_dim) {
    if (mixed_quants) {
        tensor_expect_cpu_matrix(t, in_dim, out_dim);
    } else if (allow_q6_k) {
        tensor_expect_layout_or_q6_k_or_q8_0(
            t, baseline_type, 2, in_dim, out_dim, 0);
    } else {
        tensor_expect_layout_or_q8_0(t, baseline_type, 2, in_dim, out_dim, 0);
    }
}

static void tensor_expect_routed_gate_up(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_gate_up(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected a routed gate/up expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static void tensor_expect_routed_down(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_down(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected a routed down expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static int p36_quant_bits_from_type(uint32_t type) {
    switch (type) {
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
        return 2;
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_Q3_K:
        return 3;
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ4_NL:
        return 4;
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ1_M:
        return 1;
    case P36_TENSOR_Q4_K:
        return 4;
    case P36_TENSOR_Q5_K:
        return 5;
    case P36_TENSOR_Q6_K:
        return 6;
    case P36_TENSOR_Q8_0:
        return 8;
    case P36_TENSOR_BF16:
        return 16;
    default:
        return 0;
    }
}

static void weights_validate_layout(const p36_weights *w, bool mixed_moe_quants) {
    if (!w) p36_die("internal error: missing weights while validating layout");
    if (P36_MODEL_DENSE) {
        tensor_expect_cpu_matrix(w->token_embd, P36_N_EMBD, P36_N_VOCAB);
        tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_cpu_matrix(w->output, P36_N_EMBD, P36_N_VOCAB);
        tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            const p36_layer_weights *l = &w->layer[il];
            tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
            tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
            if (l->kind == P36_LAYER_FULL_ATTN) {
                tensor_expect_cpu_matrix(l->attn_q, P36_N_EMBD, P36_N_SSM_INNER * 2u);
                tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
                tensor_expect_cpu_matrix(l->attn_k, P36_N_EMBD,
                                         (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM);
                tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
                tensor_expect_cpu_matrix(l->attn_v, P36_N_EMBD,
                                         (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM);
                tensor_expect_cpu_matrix(l->attn_output, P36_N_SSM_INNER, P36_N_EMBD);
                tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
            } else {
                tensor_expect_cpu_matrix(l->attn_gate, P36_N_EMBD, P36_N_SSM_INNER);
                tensor_expect_cpu_matrix(l->attn_qkv, P36_N_EMBD,
                                         P36_N_SSM_CONV_DIM);
                tensor_expect_layout(l->ssm_a, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
                tensor_expect_cpu_matrix(l->ssm_alpha, P36_N_EMBD, P36_N_SSM_DT_RANK);
                tensor_expect_cpu_matrix(l->ssm_beta, P36_N_EMBD, P36_N_SSM_DT_RANK);
                tensor_expect_layout(l->ssm_conv1d, P36_TENSOR_F32, 2,
                                     P36_N_SSM_CONV, P36_N_SSM_CONV_DIM, 0);
                tensor_expect_layout(l->ssm_dt, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
                tensor_expect_layout(l->ssm_norm, P36_TENSOR_F32, 1, P36_N_SSM_STATE, 0, 0);
                tensor_expect_cpu_matrix(l->ssm_out, P36_N_SSM_INNER, P36_N_EMBD);
            }
            tensor_expect_cpu_matrix(l->ffn_gate_shexp, P36_N_EMBD, P36_N_FF_SHARED);
            tensor_expect_cpu_matrix(l->ffn_up_shexp, P36_N_EMBD, P36_N_FF_SHARED);
            tensor_expect_cpu_matrix(l->ffn_down_shexp, P36_N_FF_SHARED, P36_N_EMBD);
        }
        return;
    }
    /* Upstream IQ2_M uses a mixed quant recipe for the 40-layer trunk.  All
     * of these matrix types are handled by the generic CPU/Vulkan dispatch. */
    tensor_expect_moe_matrix(w->token_embd, mixed_moe_quants,
                             P36_TENSOR_Q4_K, false, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_moe_matrix(w->output, mixed_moe_quants,
                             P36_TENSOR_Q4_K, true, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);

    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &w->layer[il];
        tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);

        if (l->kind == P36_LAYER_FULL_ATTN) {
            tensor_expect_moe_matrix(l->attn_q, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER * 2u);
            tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
            tensor_expect_moe_matrix(l->attn_k, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false, P36_N_EMBD,
                                     (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM);
            tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
            tensor_expect_moe_matrix(l->attn_v, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false, P36_N_EMBD,
                                     (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM);
            tensor_expect_moe_matrix(l->attn_output, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_SSM_INNER, P36_N_EMBD);
            tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
            tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);
        } else {
            tensor_expect_moe_matrix(l->attn_gate, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER);
            tensor_expect_moe_matrix(l->attn_qkv, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER * 2u);
            tensor_expect_layout(l->ssm_a, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_moe_matrix(l->ssm_alpha, mixed_moe_quants,
                                     P36_TENSOR_F32, false,
                                     P36_N_EMBD, P36_N_SSM_DT_RANK);
            tensor_expect_moe_matrix(l->ssm_beta, mixed_moe_quants,
                                     P36_TENSOR_F32, false,
                                     P36_N_EMBD, P36_N_SSM_DT_RANK);
            tensor_expect_layout(l->ssm_conv1d, P36_TENSOR_F32, 2, P36_N_SSM_CONV, P36_N_SSM_INNER * 2u, 0);
            tensor_expect_layout(l->ssm_dt, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_layout(l->ssm_norm, P36_TENSOR_F32, 1, P36_N_SSM_STATE, 0, 0);
            tensor_expect_moe_matrix(l->ssm_out, mixed_moe_quants,
                                     P36_TENSOR_Q6_K, false,
                                     P36_N_SSM_INNER, P36_N_EMBD);
            tensor_expect_optional_plain(l->attn_gate_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_qkv_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_alpha_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_beta_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_out_scale, 1, 1, 0, 0);
        }

        tensor_expect_layout_or_q8_0(l->ffn_gate_inp, P36_TENSOR_F32, 2, P36_N_EMBD, P36_N_EXPERT, 0);
        tensor_expect_layout(l->ffn_gate_inp_shexp, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_gate_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q5_K, false,
                                 P36_N_EMBD, P36_N_FF_SHARED);
        tensor_expect_routed_gate_up(l->ffn_up_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_up_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q5_K, false,
                                 P36_N_EMBD, P36_N_FF_SHARED);
        tensor_expect_routed_down(l->ffn_down_exps, 3, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_down_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q6_K, false,
                                 P36_N_FF_SHARED, P36_N_EMBD);
        tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
    }
}

