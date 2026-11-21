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

#if !defined(P36_NO_GPU) && !defined(P36_METAL)
static bool p36_vision_ensure_resident_gpu(p36_engine *e) {
    if (e->vision_gpu_buf && e->vision_gpu_offsets) return true;
    p36_model *m = &e->vision_model;
    if (!m->n_tensors || !m->map) return false;
    uint64_t *offs = calloc((size_t)m->n_tensors, sizeof(*offs));
    if (!offs) return false;
    uint64_t total = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        offs[i] = total;
        total += (m->tensors[i].bytes + 255u) & ~255ull;
    }
    p36_gpu_tensor *buf = p36_gpu_tensor_alloc_uninitialized(total);
    if (!buf) {
        free(offs);
        return false;
    }
    int ok = 1;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const p36_tensor *t = &m->tensors[i];
        if (!p36_gpu_tensor_write(buf, offs[i], m->map + t->abs_offset, t->bytes)) {
            ok = 0;
            break;
        }
    }
    if (!ok) {
        p36_gpu_tensor_free(buf);
        free(offs);
        return false;
    }
    e->vision_gpu_buf = buf;
    e->vision_gpu_offsets = offs;
    return true;
}

static p36_gpu_tensor *p36_vision_resident_view(p36_engine *e, const p36_tensor *t) {
    if (!e || !e->vision_gpu_buf || !e->vision_gpu_offsets || !t) return NULL;
    uint64_t idx = (uint64_t)(t - e->vision_model.tensors);
    if (idx >= e->vision_model.n_tensors) return NULL;
    return p36_gpu_tensor_view(e->vision_gpu_buf, e->vision_gpu_offsets[idx], t->bytes);
}

static bool p36_vision_mm_gpu(p36_engine *e, const p36_tensor *weight,
                              const p36_gpu_tensor *gx, p36_gpu_tensor *gy,
                              uint32_t rows, uint32_t in_dim, uint32_t out_dim) {
    p36_model *m = &e->vision_model;
    if (!p36_vision_weight_type_ok(weight, in_dim) ||
        weight->elements != (uint64_t)in_dim * out_dim ||
        (weight->ndim == 2 &&
         (weight->dim[0] != in_dim || weight->dim[1] != out_dim)))
        return false;
    if (weight->type == P36_TENSOR_Q8_0) {
        if (e->vision_gpu_buf && e->vision_gpu_offsets) {
            p36_gpu_tensor *wv = p36_vision_resident_view(e, weight);
            if (!wv) return false;
            int ok = p36_gpu_vision_matmul_q8_0_tensor(gy, wv, in_dim, out_dim, gx, rows);
            p36_gpu_tensor_free(wv);
            return ok != 0;
        }
        return p36_gpu_vision_matmul_q8_0_mem(gy, m->map + weight->abs_offset,
                                              weight->bytes, in_dim, out_dim,
                                              gx, rows) != 0;
    }
    if (weight->type == P36_TENSOR_F16) {
        return p36_gpu_vision_matmul_f16_disk(gy, m->fd, weight->abs_offset,
                                              in_dim, out_dim, gx, rows) != 0;
    }
    p36_vision_fill_job fj = {{m, weight, in_dim, NULL}, out_dim, e->n_threads};
    return p36_gpu_vision_matmul_f16_fill(gy, p36_vision_fill_cb, &fj,
                                          in_dim, out_dim, gx, rows) != 0;
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
        int ok = gx && gy && tmp && p36_gpu_tensor_write(gx, 0, input, in_bytes) &&
            p36_vision_mm_gpu(e, weight, gx, gy, rows, in_dim, out_dim) &&
            p36_gpu_tensor_read(gy, 0, tmp, out_bytes);
        if (ok && add)
            for (uint64_t i = 0, n = (uint64_t)rows * out_dim; i < n; i++)
                output[i] += tmp[i];
        if (add) free(tmp);
        p36_gpu_tensor_free(gx);
        p36_gpu_tensor_free(gy);
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

static void p36_vision_layer_norm_eps(float *out, const float *x,
                                      const float *weight, const float *bias,
                                      uint32_t rows, uint32_t width, float eps) {
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
        float inv = 1.0f / sqrtf((float)(variance / width) + eps);
        for (uint32_t i = 0; i < width; i++)
            yr[i] = (xr[i] - (float)mean) * inv * weight[i] + bias[i];
    }
}

static void p36_vision_layer_norm(float *out, const float *x,
                                  const float *weight, const float *bias,
                                  uint32_t rows, uint32_t width) {
    p36_vision_layer_norm_eps(out, x, weight, bias, rows, width, 1.0e-6f);
}

static void p36_vision_bias_residual(float *x, const float *bias,
                                     const float *residual,
                                     uint32_t rows, uint32_t width) {
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++)
        x[i] += (bias ? bias[i % width] : 0.0f) + (residual ? residual[i] : 0.0f);
}

static void p36_vision_layerscale_residual(float *dst, const float *x,
                                           const float *bias,
                                           const float *scale,
                                           const float *residual,
                                           uint32_t rows, uint32_t width) {
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++) {
        uint32_t d = (uint32_t)(i % width);
        float v = x[i] + (bias ? bias[d] : 0.0f);
        if (scale) v *= scale[d];
        dst[i] = v + (residual ? residual[i] : 0.0f);
    }
}

static void p36_vision_gelu_bias(float *x, const float *bias,
                                 uint32_t rows, uint32_t width) {
    const float k = 0.7071067811865475f;
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++) {
        float v = x[i] + (bias ? bias[i % width] : 0.0f);
        x[i] = 0.5f * v * (1.0f + erff(v * k));
    }
}

typedef struct {
    const float *qkv;
    float *out;
    uint32_t rows;
    uint32_t n_heads;
    uint32_t head_dim;
} p36_vision_attn_ctx;

static void p36_vision_attention_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_attn_ctx *c = opaque;
    const uint32_t n_heads = c->n_heads;
    const uint32_t head_dim = c->head_dim;
    const uint32_t embd = n_heads * head_dim;
    const uint32_t stride = embd * 3u;
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint64_t job = row0; job < row1; job++) {
        uint32_t row = (uint32_t)(job / n_heads), head = (uint32_t)(job % n_heads);
        const float *q = c->qkv + (uint64_t)row * stride + head * head_dim;
        float *dst = c->out + (uint64_t)row * embd + head * head_dim;
        float max_score = -INFINITY, denom = 0.0f;
        memset(dst, 0, (size_t)head_dim * sizeof(*dst));
        for (uint32_t kr = 0; kr < c->rows; kr++) {
            const float *k = c->qkv + (uint64_t)kr * stride + embd + head * head_dim;
            const float *v = c->qkv + (uint64_t)kr * stride + 2u * embd + head * head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) score += q[d] * k[d];
            score *= scale;
            float next = fmaxf(max_score, score);
            float old_scale = isfinite(max_score) ? expf(max_score - next) : 0.0f;
            float new_scale = expf(score - next);
            denom = denom * old_scale + new_scale;
            for (uint32_t d = 0; d < head_dim; d++)
                dst[d] = dst[d] * old_scale + new_scale * v[d];
            max_score = next;
        }
        for (uint32_t d = 0; d < head_dim; d++) dst[d] /= denom;
    }
}

static bool p36_vision_attention_ex(p36_engine *e, const float *qkv,
                                    float *out, uint32_t rows,
                                    uint32_t n_heads, uint32_t head_dim) {
    const uint32_t embd = n_heads * head_dim;
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
    if (e->backend == P36_BACKEND_VULKAN) {
        uint64_t qkv_bytes = (uint64_t)rows * (embd * 3u) * sizeof(float);
        uint64_t out_bytes = (uint64_t)rows * embd * sizeof(float);
        p36_gpu_tensor *gqkv = p36_gpu_tensor_alloc_uninitialized(qkv_bytes);
        p36_gpu_tensor *gout = p36_gpu_tensor_alloc_uninitialized(out_bytes);
        int ok = gqkv && gout && p36_gpu_tensor_write(gqkv, 0, qkv, qkv_bytes) &&
            p36_gpu_vision_attention_ex(gout, gqkv, rows, n_heads, head_dim) &&
            p36_gpu_tensor_read(gout, 0, out, out_bytes);
        p36_gpu_tensor_free(gqkv);
        p36_gpu_tensor_free(gout);
        return ok != 0;
    }
#endif
    p36_vision_attn_ctx ac = {qkv, out, rows, n_heads, head_dim};
    p36_parallel_for_rows((uint64_t)rows * n_heads, 8, e->n_threads,
                          p36_vision_attention_worker, &ac);
    return true;
}

static bool p36_vision_attention(p36_engine *e, const float *qkv,
                                 float *out, uint32_t rows) {
    return p36_vision_attention_ex(e, qkv, out, rows, 16u, 72u);
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

static void p36_vision_rope_inkling2v(float *qkv, uint32_t rows,
                                      uint32_t grid_w, uint32_t n_heads,
                                      uint32_t head_dim) {
    const uint32_t embd = n_heads * head_dim;
    const uint32_t stride = embd * 3u;
    const uint32_t half = head_dim / 2u;
    const uint32_t pairs = half / 2u;
    for (uint32_t r = 1u; r < rows; r++) { /* row 0 is CLS at (0, 0) -> identity */
        uint32_t idx = r - 1u;
        float pos_w = (float)(idx % grid_w + 1u);
        float pos_h = (float)(idx / grid_w + 1u);
        for (uint32_t part = 0; part < 2u; part++) { /* 0=Q, 1=K */
            for (uint32_t h = 0; h < n_heads; h++) {
                float *v = qkv + (uint64_t)r * stride + part * embd + h * head_dim;
                for (uint32_t i = 0; i < pairs; i++) {
                    float inv_freq = powf(10000.0f, -2.0f * (float)i / (float)half);
                    float aw = pos_w * inv_freq;
                    float cw = cosf(aw), sw = sinf(aw);
                    float x0 = v[2u * i], x1 = v[2u * i + 1u];
                    v[2u * i] = x0 * cw - x1 * sw;
                    v[2u * i + 1u] = x0 * sw + x1 * cw;

                    float ah = pos_h * inv_freq;
                    float ch = cosf(ah), sh = sinf(ah);
                    float y0 = v[half + 2u * i], y1 = v[half + 2u * i + 1u];
                    v[half + 2u * i] = y0 * ch - y1 * sh;
                    v[half + 2u * i + 1u] = y0 * sh + y1 * ch;
                }
            }
        }
    }
}

static int p36_vision_encode_image_inkling2v(p36_engine *e,
                                             const p36_image *image,
                                             p36_vision_embedding *out,
                                             char *err, size_t errlen) {
    p36_model *m = &e->vision_model;
    const uint32_t patch_size = e->vision_patch_size;
    const uint32_t merge = e->vision_spatial_merge;
    const uint32_t D = e->vision_n_embd;
    const uint32_t FF = e->vision_n_ff;
    const uint32_t n_layer = e->vision_n_layer;
    const uint32_t n_head = e->vision_n_head;
    const uint32_t head_dim = D / n_head;
    const float eps = e->vision_ln_eps;
    const uint32_t patch_in_dim = 3u * patch_size * patch_size;
    const uint32_t merged_dim = D * merge * merge;

    p36_image_patches patches = {0};
    float *x = NULL, *norm = NULL, *qkv = NULL, *proj_tmp = NULL;
    float *work = NULL, *wide = NULL, *merged_in = NULL;
    int ok = 0;

    if (!e->vision_ready || !p36_image_preprocess_inkling2v(
            &patches, image, patch_size, merge, 4096u, err, errlen)) return 0;

    const uint32_t n_patches = patches.patch_count;
    const uint32_t rows = n_patches + 1u; /* CLS + patches */
    const uint32_t merged = patches.image_token_count;
    const uint64_t base_values = (uint64_t)rows * D;

    x = malloc(base_values * sizeof(*x));
    uint64_t wide_count = (uint64_t)rows * FF;
    if ((uint64_t)merged * merged_dim > wide_count)
        wide_count = (uint64_t)merged * merged_dim;
    bool vk_path = false;
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
    vk_path = (e->backend == P36_BACKEND_VULKAN);
#endif
    if (!vk_path) {
        norm = malloc(base_values * sizeof(*norm));
        work = malloc(base_values * sizeof(*work));
        proj_tmp = malloc(base_values * sizeof(*proj_tmp));
        qkv = malloc((uint64_t)rows * (3u * D) * sizeof(*qkv));
        wide = malloc(wide_count * sizeof(*wide));
        merged_in = malloc((uint64_t)merged * merged_dim * sizeof(*merged_in));
    }
    if (!x || (!vk_path && (!norm || !work || !proj_tmp || !qkv || !wide || !merged_in))) {
        if (err && errlen) snprintf(err, errlen, "unable to allocate Inkling-2 vision workspace");
        goto done;
    }

    /* 1. Patch embedding + bilinear resized position embedding + CLS token + pre_ln */
    {
        p36_tensor *patch_w = model_find_tensor(m, "v.patch_embd.weight");
        p36_tensor *cls_t = model_find_tensor(m, "v.class_embd");
        p36_tensor *cls_pos_t = model_find_tensor(m, "v.class_position_embd.weight");
        p36_tensor *pos_t = model_find_tensor(m, "v.position_embd.weight");
        p36_tensor *pre_ln_w_t = model_find_tensor(m, "v.pre_ln.weight");
        p36_tensor *pre_ln_b_t = model_find_tensor(m, "v.pre_ln.bias");
        const float *cls = p36_vision_f32(m, cls_t, D);
        const float *cls_pos = p36_vision_f32(m, cls_pos_t, D);
        const float *pre_ln_w = p36_vision_f32(m, pre_ln_w_t, D);
        const float *pre_ln_b = p36_vision_f32(m, pre_ln_b_t, D);
        if (!patch_w || !cls || !cls_pos || !pos_t || pos_t->type != P36_TENSOR_F32 ||
            (pos_t->elements % D) != 0 || !pre_ln_w || !pre_ln_b) {
            if (err && errlen) snprintf(err, errlen, "invalid Inkling-2 vision embedding tensors");
            goto done;
        }
        uint32_t pos_n = (uint32_t)(pos_t->elements / D);
        uint32_t n_per_side = (uint32_t)lround(sqrt((double)pos_n));
        if (n_per_side == 0 || n_per_side * n_per_side != pos_n) {
            if (err && errlen) snprintf(err, errlen, "invalid Inkling-2 position embedding grid");
            goto done;
        }
        const float *position = (const float *)(m->map + pos_t->abs_offset);

        /* Row 0: CLS + class_position_embd */
        for (uint32_t d = 0; d < D; d++) x[d] = cls[d] + cls_pos[d];

        /* Rows 1..n_patches: patch_embd(patches) */
        if (!p36_vision_mm(e, patch_w, patches.patches, x + D, n_patches, patch_in_dim, D, false)) {
            if (err && errlen) snprintf(err, errlen, "Inkling-2 patch embedding matmul failed");
            goto done;
        }

        /* Add learned position embeddings (bilinear resize if grid != n_per_side x n_per_side) */
        for (uint32_t py = 0; py < patches.grid_height; py++) {
            float sy = patches.grid_height == n_per_side
                ? (float)py
                : ((float)py + 0.5f) * (float)n_per_side / (float)patches.grid_height - 0.5f;
            if (sy < 0.0f) sy = 0.0f;
            if (sy > (float)(n_per_side - 1u)) sy = (float)(n_per_side - 1u);
            uint32_t y0 = (uint32_t)floorf(sy);
            uint32_t y1 = y0 + 1u < n_per_side ? y0 + 1u : y0;
            float fy = sy - (float)y0;
            for (uint32_t px = 0; px < patches.grid_width; px++) {
                float sx = patches.grid_width == n_per_side
                    ? (float)px
                    : ((float)px + 0.5f) * (float)n_per_side / (float)patches.grid_width - 0.5f;
                if (sx < 0.0f) sx = 0.0f;
                if (sx > (float)(n_per_side - 1u)) sx = (float)(n_per_side - 1u);
                uint32_t x0 = (uint32_t)floorf(sx);
                uint32_t x1 = x0 + 1u < n_per_side ? x0 + 1u : x0;
                float fx = sx - (float)x0;
                float *dst = x + (uint64_t)(1u + py * patches.grid_width + px) * D;
                const float *p00 = position + ((uint64_t)y0 * n_per_side + x0) * D;
                const float *p01 = position + ((uint64_t)y0 * n_per_side + x1) * D;
                const float *p10 = position + ((uint64_t)y1 * n_per_side + x0) * D;
                const float *p11 = position + ((uint64_t)y1 * n_per_side + x1) * D;
                for (uint32_t d = 0; d < D; d++) {
                    float top = p00[d] + (p01[d] - p00[d]) * fx;
                    float bot = p10[d] + (p11[d] - p10[d]) * fx;
                    dst[d] += top + (bot - top) * fy;
                }
            }
        }

        /* Pre-LayerNorm (on GPU when Vulkan, on CPU otherwise) */
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
        if (e->backend != P36_BACKEND_VULKAN)
#endif
        {
            p36_vision_layer_norm_eps(norm, x, pre_ln_w, pre_ln_b, rows, D, eps);
            memcpy(x, norm, base_values * sizeof(*x));
        }
    }

    /* 2. ViT blocks + GPU spatial merge + projector (single batched Vulkan pass) */
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
    if (e->backend == P36_BACKEND_VULKAN) {
        bool resident = p36_vision_ensure_resident_gpu(e);
        const bool use_hd64 = (head_dim == 64u && p36_gpu_vision_has_hd64());
        const uint32_t rope_op = use_hd64 ? 5u : 1u;

        p36_gpu_tensor *gx = p36_gpu_tensor_alloc_uninitialized(base_values * sizeof(float));
        p36_gpu_tensor *gnorm = p36_gpu_tensor_alloc_uninitialized(base_values * sizeof(float));
        p36_gpu_tensor *gproj = p36_gpu_tensor_alloc_uninitialized(base_values * sizeof(float));
        p36_gpu_tensor *gwork = p36_gpu_tensor_alloc_uninitialized(base_values * sizeof(float));
        p36_gpu_tensor *gqkv = p36_gpu_tensor_alloc_uninitialized((uint64_t)rows * (3u * D) * sizeof(float));
        p36_gpu_tensor *gwide = p36_gpu_tensor_alloc_uninitialized(wide_count * sizeof(float));
        p36_gpu_tensor *gmerged = p36_gpu_tensor_alloc_uninitialized((uint64_t)merged * merged_dim * sizeof(float));
        p36_gpu_tensor *gout = p36_gpu_tensor_alloc_uninitialized((uint64_t)merged * P36_N_EMBD * sizeof(float));
        p36_gpu_tensor *g1d = NULL;
        uint8_t *host_1d = NULL;
        uint64_t d_stride = (((uint64_t)D * sizeof(float)) + 255u) & ~255ull;
        uint64_t ff_stride = (((uint64_t)FF * sizeof(float)) + 255u) & ~255ull;
        uint64_t layer_1d_stride = 11u * d_stride + ff_stride;
        int gpu_ok = (gx && gnorm && gproj && gwork && gqkv && gwide && gmerged && gout);

        /* Fallback 1D upload buffer only if full resident upload didn't fit in VRAM. */
        if (gpu_ok && !resident) {
            uint64_t total_1d_bytes = 2u * d_stride + (uint64_t)n_layer * layer_1d_stride;
            host_1d = calloc(1, (size_t)total_1d_bytes);
            g1d = p36_gpu_tensor_alloc_uninitialized(total_1d_bytes);
            const float *pre_ln_w = p36_vision_f32(m, model_find_tensor(m, "v.pre_ln.weight"), D);
            const float *pre_ln_b = p36_vision_f32(m, model_find_tensor(m, "v.pre_ln.bias"), D);
            if (!host_1d || !g1d || !pre_ln_w || !pre_ln_b) {
                gpu_ok = 0;
            } else {
                memcpy(host_1d + 0u * d_stride, pre_ln_w, (size_t)D * sizeof(float));
                memcpy(host_1d + 1u * d_stride, pre_ln_b, (size_t)D * sizeof(float));
                for (uint32_t il = 0; il < n_layer && gpu_ok; il++) {
                    const float *v11[11] = {
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ln1.weight"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ln1.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_q.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_k.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_v.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_out.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ls1.weight"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ln2.weight"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ln2.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ffn_down.bias"), D),
                        p36_vision_f32(m, p36_vision_tensorf(m, il, "ls2.weight"), D),
                    };
                    const float *upb = p36_vision_f32(m, p36_vision_tensorf(m, il, "ffn_up.bias"), FF);
                    for (int k = 0; k < 11; k++) if (!v11[k]) gpu_ok = 0;
                    if (!upb || !gpu_ok) { gpu_ok = 0; break; }
                    uint64_t lbase = 2u * d_stride + (uint64_t)il * layer_1d_stride;
                    for (int k = 0; k < 11; k++)
                        memcpy(host_1d + lbase + (uint64_t)k * d_stride, v11[k], (size_t)D * sizeof(float));
                    memcpy(host_1d + lbase + 11u * d_stride, upb, (size_t)FF * sizeof(float));
                }
                if (gpu_ok) gpu_ok = p36_gpu_tensor_write(g1d, 0, host_1d, total_1d_bytes);
            }
        }

        if (gpu_ok) {
            gpu_ok = p36_gpu_tensor_write(gnorm, 0, x, base_values * sizeof(float));
        }

        /* Batch the entire ViT + spatial merge + projector into 1 Vulkan command stream. */
        bool batched = (gpu_ok && p36_gpu_begin_commands() != 0);

        if (gpu_ok) {
            p36_gpu_tensor *v_pre_w = resident
                ? p36_vision_resident_view(e, model_find_tensor(m, "v.pre_ln.weight"))
                : p36_gpu_tensor_view(g1d, 0u, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_pre_b = resident
                ? p36_vision_resident_view(e, model_find_tensor(m, "v.pre_ln.bias"))
                : p36_gpu_tensor_view(g1d, d_stride, (uint64_t)D * sizeof(float));
            if (v_pre_w && v_pre_b) {
                gpu_ok = p36_gpu_vision_op(0u, gx, gnorm, v_pre_w, v_pre_b, rows, D, 0u, 0u, 0u, eps, rows);
            } else {
                gpu_ok = 0;
            }
            p36_gpu_tensor_free(v_pre_w);
            p36_gpu_tensor_free(v_pre_b);
        }

        const uint32_t gx_rope = (uint32_t)(((uint64_t)rows * (D / 2u) + 255u) / 256u);
        const uint32_t gx_d = (uint32_t)((base_values + 255u) / 256u);
        const uint32_t gx_ff = (uint32_t)(((uint64_t)rows * FF + 255u) / 256u);

        for (uint32_t il = 0; gpu_ok && il < n_layer; il++) {
            uint64_t lbase = 2u * d_stride + (uint64_t)il * layer_1d_stride;
            p36_gpu_tensor *v_ln1w  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ln1.weight"))      : p36_gpu_tensor_view(g1d, lbase + 0u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_ln1b  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ln1.bias"))        : p36_gpu_tensor_view(g1d, lbase + 1u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_qb    = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "attn_q.bias"))     : p36_gpu_tensor_view(g1d, lbase + 2u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_kb    = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "attn_k.bias"))     : p36_gpu_tensor_view(g1d, lbase + 3u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_vb    = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "attn_v.bias"))     : p36_gpu_tensor_view(g1d, lbase + 4u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_outb  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "attn_out.bias"))   : p36_gpu_tensor_view(g1d, lbase + 5u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_ls1w  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ls1.weight"))      : p36_gpu_tensor_view(g1d, lbase + 6u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_ln2w  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ln2.weight"))      : p36_gpu_tensor_view(g1d, lbase + 7u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_ln2b  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ln2.bias"))        : p36_gpu_tensor_view(g1d, lbase + 8u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_downb = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ffn_down.bias"))   : p36_gpu_tensor_view(g1d, lbase + 9u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_ls2w  = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ls2.weight"))      : p36_gpu_tensor_view(g1d, lbase + 10u * d_stride, (uint64_t)D * sizeof(float));
            p36_gpu_tensor *v_upb   = resident ? p36_vision_resident_view(e, p36_vision_tensorf(m, il, "ffn_up.bias"))     : p36_gpu_tensor_view(g1d, lbase + 11u * d_stride, (uint64_t)FF * sizeof(float));

            p36_tensor *qw    = p36_vision_tensorf(m, il, "attn_q.weight");
            p36_tensor *kw    = p36_vision_tensorf(m, il, "attn_k.weight");
            p36_tensor *vw    = p36_vision_tensorf(m, il, "attn_v.weight");
            p36_tensor *outw  = p36_vision_tensorf(m, il, "attn_out.weight");
            p36_tensor *upw   = p36_vision_tensorf(m, il, "ffn_up.weight");
            p36_tensor *downw = p36_vision_tensorf(m, il, "ffn_down.weight");

            gpu_ok = v_ln1w && v_ln1b && v_qb && v_kb && v_vb && v_outb &&
                     v_ls1w && v_ln2w && v_ln2b && v_downb && v_ls2w && v_upb &&
                     p36_gpu_vision_op(0u, gnorm, gx, v_ln1w, v_ln1b, rows, D, 0u, 0u, 0u, eps, rows) &&
                     p36_vision_mm_gpu(e, qw, gnorm, gproj, rows, D, D) &&
                     p36_gpu_vision_op(rope_op, gqkv, gproj, v_qb, NULL, rows, D, 0u, patches.grid_width, head_dim, 0.0f, gx_rope) &&
                     p36_vision_mm_gpu(e, kw, gnorm, gproj, rows, D, D) &&
                     p36_gpu_vision_op(rope_op, gqkv, gproj, v_kb, NULL, rows, D, 1u, patches.grid_width, head_dim, 0.0f, gx_rope) &&
                     p36_vision_mm_gpu(e, vw, gnorm, gproj, rows, D, D) &&
                     p36_gpu_vision_op(rope_op, gqkv, gproj, v_vb, NULL, rows, D, 2u, patches.grid_width, head_dim, 0.0f, gx_rope) &&
                     p36_gpu_vision_attention_ex(gwork, gqkv, rows, n_head, head_dim) &&
                     p36_vision_mm_gpu(e, outw, gwork, gnorm, rows, D, D) &&
                     p36_gpu_vision_op(2u, gx, gnorm, v_outb, v_ls1w, rows, D, 0u, 0u, 0u, 0.0f, gx_d) &&
                     p36_gpu_vision_op(0u, gnorm, gx, v_ln2w, v_ln2b, rows, D, 0u, 0u, 0u, eps, rows) &&
                     p36_vision_mm_gpu(e, upw, gnorm, gwide, rows, D, FF) &&
                     p36_gpu_vision_op(3u, gwide, NULL, v_upb, NULL, rows, FF, 1u, 0u, 0u, 0.0f, gx_ff) &&
                     p36_vision_mm_gpu(e, downw, gwide, gnorm, rows, FF, D) &&
                     p36_gpu_vision_op(2u, gx, gnorm, v_downb, v_ls2w, rows, D, 0u, 0u, 0u, 0.0f, gx_d);

            p36_gpu_tensor_free(v_ln1w); p36_gpu_tensor_free(v_ln1b);
            p36_gpu_tensor_free(v_qb);   p36_gpu_tensor_free(v_kb);
            p36_gpu_tensor_free(v_vb);   p36_gpu_tensor_free(v_outb);
            p36_gpu_tensor_free(v_ls1w); p36_gpu_tensor_free(v_ln2w);
            p36_gpu_tensor_free(v_ln2b); p36_gpu_tensor_free(v_downb);
            p36_gpu_tensor_free(v_ls2w); p36_gpu_tensor_free(v_upb);
        }

        if (gpu_ok) {
            p36_tensor *mm1 = model_find_tensor(m, "mm.1.weight");
            p36_tensor *mm2 = model_find_tensor(m, "mm.2.weight");
            const float *mm1b = p36_vision_f32(m, model_find_tensor(m, "mm.1.bias"), merged_dim);
            const float *mm2b = p36_vision_f32(m, model_find_tensor(m, "mm.2.bias"), P36_N_EMBD);
            uint32_t gx_m = (uint32_t)(((uint64_t)merged * merged_dim + 255u) / 256u);
            gpu_ok = mm1 && mm2 && !mm1b &&
                     p36_gpu_vision_op(4u, gmerged, gx, NULL, NULL, merged, merged_dim,
                                       patches.grid_width, merge, D, 0.0f, gx_m) &&
                     p36_vision_mm_gpu(e, mm1, gmerged, gwide, merged, merged_dim, merged_dim) &&
                     p36_gpu_vision_op(3u, gwide, NULL, NULL, NULL, merged, merged_dim,
                                       0u, 0u, 0u, 0.0f, gx_m) &&
                     p36_vision_mm_gpu(e, mm2, gwide, gout, merged, merged_dim, P36_N_EMBD);
            if (batched) {
                if (!p36_gpu_end_commands()) gpu_ok = 0;
                batched = false;
            }
            if (gpu_ok) {
                out->data = malloc((uint64_t)merged * P36_N_EMBD * sizeof(*out->data));
                if (out->data &&
                    p36_gpu_tensor_read(gout, 0, out->data, (uint64_t)merged * P36_N_EMBD * sizeof(float))) {
                    if (mm2b) p36_vision_bias_residual(out->data, mm2b, NULL, merged, P36_N_EMBD);
                    out->token_count = merged;
                    out->grid_width = patches.grid_width / merge;
                    out->grid_height = patches.grid_height / merge;
                    out->width = image->width;
                    out->height = image->height;
                    memcpy(out->fingerprint, image->fingerprint, sizeof(out->fingerprint));
                    ok = 1;
                } else {
                    free(out->data);
                    memset(out, 0, sizeof(*out));
                    if (err && errlen) snprintf(err, errlen, "Inkling-2 GPU vision readback failed");
                }
            } else if (err && errlen) {
                snprintf(err, errlen, "Inkling-2 GPU vision projector failed");
            }
        } else {
            if (batched) p36_gpu_end_commands();
            if (err && errlen) snprintf(err, errlen, "Inkling-2 GPU vision encoder failed");
        }

        free(host_1d);
        p36_gpu_tensor_free(g1d);   p36_gpu_tensor_free(gx);
        p36_gpu_tensor_free(gnorm); p36_gpu_tensor_free(gproj);
        p36_gpu_tensor_free(gwork); p36_gpu_tensor_free(gqkv);
        p36_gpu_tensor_free(gwide); p36_gpu_tensor_free(gmerged);
        p36_gpu_tensor_free(gout);
        goto done;
    }
#endif

    /* CPU fallback: 23 ViT layers + projector */
    for (uint32_t il = 0; il < n_layer; il++) {
        const float *ln1w = p36_vision_f32(m, p36_vision_tensorf(m, il, "ln1.weight"), D);
        const float *ln1b = p36_vision_f32(m, p36_vision_tensorf(m, il, "ln1.bias"), D);
        p36_tensor *qw = p36_vision_tensorf(m, il, "attn_q.weight");
        const float *qb = p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_q.bias"), D);
        p36_tensor *kw = p36_vision_tensorf(m, il, "attn_k.weight");
        const float *kb = p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_k.bias"), D);
        p36_tensor *vw = p36_vision_tensorf(m, il, "attn_v.weight");
        const float *vb = p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_v.bias"), D);
        p36_tensor *outw = p36_vision_tensorf(m, il, "attn_out.weight");
        const float *outb = p36_vision_f32(m, p36_vision_tensorf(m, il, "attn_out.bias"), D);
        const float *ls1w = p36_vision_f32(m, p36_vision_tensorf(m, il, "ls1.weight"), D);
        const float *ln2w = p36_vision_f32(m, p36_vision_tensorf(m, il, "ln2.weight"), D);
        const float *ln2b = p36_vision_f32(m, p36_vision_tensorf(m, il, "ln2.bias"), D);
        p36_tensor *upw = p36_vision_tensorf(m, il, "ffn_up.weight");
        const float *upb = p36_vision_f32(m, p36_vision_tensorf(m, il, "ffn_up.bias"), FF);
        p36_tensor *downw = p36_vision_tensorf(m, il, "ffn_down.weight");
        const float *downb = p36_vision_f32(m, p36_vision_tensorf(m, il, "ffn_down.bias"), D);
        const float *ls2w = p36_vision_f32(m, p36_vision_tensorf(m, il, "ls2.weight"), D);
        if (!ln1w || !ln1b || !qb || !kb || !vb || !outb || !ls1w ||
            !ln2w || !ln2b || !upb || !downb || !ls2w) {
            if (err && errlen) snprintf(err, errlen, "invalid Inkling-2 vision block %u", il);
            goto done;
        }

        p36_vision_layer_norm_eps(norm, x, ln1w, ln1b, rows, D, eps);

        /* Q -> qkv[:, 0..D) */
        if (!p36_vision_mm(e, qw, norm, proj_tmp, rows, D, D, false)) goto bad_mm;
        for (uint32_t r = 0; r < rows; r++) {
            const float *src = proj_tmp + (uint64_t)r * D;
            float *dst = qkv + (uint64_t)r * (3u * D);
            for (uint32_t d = 0; d < D; d++) dst[d] = src[d] + qb[d];
        }
        /* K -> qkv[:, D..2D) */
        if (!p36_vision_mm(e, kw, norm, proj_tmp, rows, D, D, false)) goto bad_mm;
        for (uint32_t r = 0; r < rows; r++) {
            const float *src = proj_tmp + (uint64_t)r * D;
            float *dst = qkv + (uint64_t)r * (3u * D) + D;
            for (uint32_t d = 0; d < D; d++) dst[d] = src[d] + kb[d];
        }
        /* V -> qkv[:, 2D..3D) */
        if (!p36_vision_mm(e, vw, norm, proj_tmp, rows, D, D, false)) goto bad_mm;
        for (uint32_t r = 0; r < rows; r++) {
            const float *src = proj_tmp + (uint64_t)r * D;
            float *dst = qkv + (uint64_t)r * (3u * D) + 2u * D;
            for (uint32_t d = 0; d < D; d++) dst[d] = src[d] + vb[d];
        }

        p36_vision_rope_inkling2v(qkv, rows, patches.grid_width, n_head, head_dim);
        if (!p36_vision_attention_ex(e, qkv, work, rows, n_head, head_dim)) goto bad_mm;
        if (!p36_vision_mm(e, outw, work, norm, rows, D, D, false)) goto bad_mm;
        p36_vision_layerscale_residual(x, norm, outb, ls1w, x, rows, D);

        p36_vision_layer_norm_eps(norm, x, ln2w, ln2b, rows, D, eps);
        if (!p36_vision_mm(e, upw, norm, wide, rows, D, FF, false)) goto bad_mm;
        p36_vision_gelu_bias(wide, upb, rows, FF);
        if (!p36_vision_mm(e, downw, wide, norm, rows, FF, D, false)) goto bad_mm;
        p36_vision_layerscale_residual(x, norm, downb, ls2w, x, rows, D);
        continue;
bad_mm:
        if (err && errlen) snprintf(err, errlen, "Inkling-2 vision matmul failed in block %u", il);
        goto done;
    }

    /* 3. Drop CLS token, spatial merge (merge x merge), and 2-layer GELU_ERF projector */
    {
        const float *patch_out = x + D; /* skip CLS row 0 */
        const uint32_t gw_m = patches.grid_width / merge;
        const uint32_t gh_m = patches.grid_height / merge;
        for (uint32_t by = 0; by < gh_m; by++) {
            for (uint32_t bx = 0; bx < gw_m; bx++) {
                float *dst = merged_in + (uint64_t)(by * gw_m + bx) * merged_dim;
                uint32_t off = 0;
                for (uint32_t my = 0; my < merge; my++) {
                    for (uint32_t mx = 0; mx < merge; mx++) {
                        uint32_t py = by * merge + my;
                        uint32_t px = bx * merge + mx;
                        memcpy(dst + off,
                               patch_out + (uint64_t)(py * patches.grid_width + px) * D,
                               (size_t)D * sizeof(float));
                        off += D;
                    }
                }
            }
        }

        p36_tensor *mm1 = model_find_tensor(m, "mm.1.weight");
        p36_tensor *mm1b_t = model_find_tensor(m, "mm.1.bias");
        const float *mm1b = mm1b_t ? p36_vision_f32(m, mm1b_t, merged_dim) : NULL;
        p36_tensor *mm2 = model_find_tensor(m, "mm.2.weight");
        p36_tensor *mm2b_t = model_find_tensor(m, "mm.2.bias");
        const float *mm2b = mm2b_t ? p36_vision_f32(m, mm2b_t, P36_N_EMBD) : NULL;
        if (!mm1 || !mm2) {
            if (err && errlen) snprintf(err, errlen, "missing Inkling-2 projector tensors");
            goto done;
        }
        if (!p36_vision_mm(e, mm1, merged_in, wide, merged, merged_dim, merged_dim, false)) goto proj_bad;
        p36_vision_gelu_bias(wide, mm1b, merged, merged_dim);

        out->data = malloc((uint64_t)merged * P36_N_EMBD * sizeof(*out->data));
        if (!out->data) {
            if (err && errlen) snprintf(err, errlen, "unable to allocate vision embedding");
            goto done;
        }
        if (!p36_vision_mm(e, mm2, wide, out->data, merged, merged_dim, P36_N_EMBD, false)) goto proj_bad;
        if (mm2b) p36_vision_bias_residual(out->data, mm2b, NULL, merged, P36_N_EMBD);
        out->token_count = merged;
        out->grid_width = gw_m;
        out->grid_height = gh_m;
        out->width = image->width;
        out->height = image->height;
        memcpy(out->fingerprint, image->fingerprint, sizeof(out->fingerprint));
        ok = 1;
        goto done;
proj_bad:
        free(out->data);
        memset(out, 0, sizeof(*out));
        if (err && errlen) snprintf(err, errlen, "Inkling-2 projector matmul failed");
    }
done:
    free(x); free(norm); free(qkv); free(proj_tmp); free(work); free(wide); free(merged_in);
    p36_image_patches_free(&patches);
    return ok;
}

static int p36_vision_encode_image(p36_engine *e, const p36_image *image,
                                   p36_vision_embedding *out,
                                   char *err, size_t errlen) {
    if (e && e->vision_kind == 2u) {
        return p36_vision_encode_image_inkling2v(e, image, out, err, errlen);
    }
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
    case P36_TENSOR_NVFP4:
        return 4;
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ1_M:
        return 1;
    case P36_TENSOR_Q4_K:
        return 4;
    case P36_TENSOR_Q5_0:
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
            if (l->ssm_conv1d_right) {
                tensor_expect_layout(l->ssm_conv1d_right, P36_TENSOR_F32, 2,
                                     P36_N_SSM_CONV - 1u, P36_N_SSM_INNER * 2u, 0);
            }
            if (l->ssm_bd_logit_scale) {
                tensor_expect_layout(l->ssm_bd_logit_scale, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            }
            if (l->ssm_bd_history_bias) {
                tensor_expect_layout(l->ssm_bd_history_bias, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            }
            if (l->ssm_bd_history_weight) {
                tensor_expect_layout(l->ssm_bd_history_weight, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            }
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

