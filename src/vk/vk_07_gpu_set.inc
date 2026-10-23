void p36_gpu_set_dense_model(bool dense) {
    p36_gpu_dense_model = dense;
    /* Prewarm regardless of model class: MoE trunk tensors reach the same
     * K-quant decode/mmq kernels (see p36_gpu_matmul_k_quant_q8_scaled_tensor)
     * and some MoE files carry non-expert IQ3_S tensors too, so skipping this
     * for dense == false just moves the multi-second RADV pipeline-compile
     * stall onto the first live decode/prefill token instead of model open. */
    p36_vk_prepare_dense_kernels();
}

void p36_gpu_set_ssd_streaming(bool enabled) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.enabled != enabled) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.enabled = enabled;
    }
    p36_vk_moe_bank_cache = !enabled;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_full_layers(uint32_t layers) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream.full_layers = layers;
    p36_vk_moe_bank_cache = !p36_vk_stream.enabled || layers != 0;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.budget != experts) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.budget = experts;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.configured_expert_bytes != bytes) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.configured_expert_bytes = bytes;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}
void p36_gpu_set_streaming_expert_cache_layout(
        uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    uint64_t total = 0;
    if (gate_bytes <= UINT64_MAX - up_bytes &&
        gate_bytes + up_bytes <= UINT64_MAX - down_bytes)
        total = gate_bytes + up_bytes + down_bytes;
    p36_gpu_set_streaming_expert_cache_expert_bytes(total);
}

uint64_t p36_gpu_recommended_working_set_size(void) {
    if (!p36_vk.ready) return 0;
    return p36_vk_memory_total();
}

uint32_t p36_gpu_stream_expert_cache_configured_count(void) {
    return p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
}

uint32_t p36_gpu_stream_expert_cache_current_count(void) {
    pthread_mutex_lock(&p36_vk_mu);
    uint32_t count = 0;
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) count += p36_vk_stream.entry[i].valid ? 1u : 0u;
    pthread_mutex_unlock(&p36_vk_mu);
    return count;
}

void p36_gpu_stream_expert_cache_reset_route_hotness(void) {
    pthread_mutex_lock(&p36_vk_mu);
    memset(p36_vk_stream.route_hotness, 0, sizeof(p36_vk_stream.route_hotness));
    p36_vk_stream.hotness_decay_token = p36_vk_stream.route_tokens;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_stream_expert_cache_note_tokens(uint32_t n_tokens) {
    if (!p36_vk_stream.enabled || n_tokens == 0) return;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream.phase = n_tokens == 1u ? 1u : 0u;
    if (p36_vk_stream.route_tokens <= UINT64_MAX - n_tokens) {
        p36_vk_stream.route_tokens += n_tokens;
    } else {
        p36_vk_stream.route_tokens = UINT64_MAX;
    }
    p36_vk_stream_decay_hotness_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_stream_expert_cache_release_resident(void) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream_cache_release_unlocked(true);
    pthread_mutex_unlock(&p36_vk_mu);
}

uint32_t p36_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gate_expert_bytes, uint64_t up_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || up_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint64_t total = gate_expert_bytes + up_expert_bytes;
    if (total < gate_expert_bytes || total > UINT64_MAX - down_expert_bytes) return 0;
    total += down_expert_bytes;
    if (p36_vk_stream.configured_expert_bytes != 0 && p36_vk_stream.configured_expert_bytes != total) return 0;
    return p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
}

int p36_gpu_stream_expert_cache_seed_selected(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return p36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, n_selected, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_begin_selected_load(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return p36_gpu_stream_expert_cache_seed_selected(table, selected_ids, n_selected);
}

int p36_gpu_stream_expert_cache_prepare_selected_batch(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected) {
    uint64_t n = (uint64_t)n_tokens * n_selected;
    if (n > UINT32_MAX) return 0;
    return p36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_load_layer(const p36_gpu_stream_expert_table *table) {
    if (!table || table->n_total_expert == 0) return 0;
    uint32_t n = table->n_total_expert;
    uint32_t budget = p36_gpu_stream_expert_cache_configured_count();
    if (budget != 0 && n > budget) n = budget;
    int32_t *ids = malloc((size_t)n * sizeof(ids[0]));
    if (!ids) return 0;
    for (uint32_t i = 0; i < n; i++) ids[i] = (int32_t)i;
    int ok = p36_vk_stream_cache_prepare_i32(table, ids, NULL, n, NULL) ? 1 : 0;
    free(ids);
    return ok;
}

int p36_gpu_stream_expert_cache_seed_from_layer_selected(const p36_gpu_stream_expert_table *table, const p36_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected) {
    if (!selected || n_selected == 0) return 0;
    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    uint64_t n = (uint64_t)n_seed_tokens * n_selected;
    if (n > UINT32_MAX || !p36_gpu_tensor_range_ok(selected, 0, n * sizeof(int32_t))) return 0;
    const int32_t *ids = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_stream_seed_selected");
    if (!ids) return 0;
    return p36_vk_stream_cache_prepare_i32(table, ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_release_layer_cache(void) {
    return 1;
}

int p36_gpu_stream_expert_cache_seed_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts) {
    return p36_vk_stream_cache_prepare_i32(table, expert_ids, expert_priorities, n_experts, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_bias_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts) {
    if (!table || !expert_ids || table->layer >= P36_VK_STREAM_MAX_LAYERS) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    for (uint32_t i = 0; i < n_experts; i++) {
        int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert ||
            (uint32_t)expert >= P36_VK_STREAM_MAX_EXPERTS) {
            pthread_mutex_unlock(&p36_vk_mu);
            return 0;
        }
        uint32_t priority = expert_priorities ? expert_priorities[i] : 1u;
        uint32_t *bias = &p36_vk_stream.route_bias[table->layer][expert];
        if (priority > *bias) {
            *bias = priority;
            p36_vk_stream.has_route_bias = true;
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

void p36_gpu_print_memory_report(const char *label) {
    uint64_t arena_reserved = 0, arena_used = 0, arena_max_slack = 0;
    uint32_t arena_blocks = 0;
    for (p36_vk_arena_block *b = p36_vk.arena; b; b = b->next) {
        arena_reserved += b->base->alloc_bytes;
        arena_used += b->used;
        if (b->base->alloc_bytes - b->used > arena_max_slack)
            arena_max_slack = b->base->alloc_bytes - b->used;
        arena_blocks++;
    }
    fprintf(stderr,
            "p36: Vulkan memory%s%s live=%" PRIu64 " peak=%" PRIu64
            " arena=%" PRIu64 "/%" PRIu64 " arena_blocks=%u arena_max_slack=%" PRIu64
            " scratch_private=%" PRIu64 " scratch_private_peak=%" PRIu64
            " scratch_private_limit=%" PRIu64
            " fd=%d quality=%s ssd_streaming=%s cache=%u/%u requested=%u allocation_failures=%u lookup=%s lookup_steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 " device=%s api=%u.%u.%u\n",
            label ? " " : "",
            label ? label : "",
            p36_gpu_live_bytes,
            p36_gpu_peak_bytes,
            arena_used,
            arena_reserved,
            arena_blocks,
            arena_max_slack,
            p36_vk_private_bytes,
            p36_vk_private_peak,
            (uint64_t)P36_VK_PRIVATE_LIMIT,
            p36_gpu_model_fd,
            p36_gpu_quality ? "high" : "fast",
            p36_vk_stream.enabled ? "on" : "off",
            p36_gpu_stream_expert_cache_current_count(),
            p36_vk_stream.cap ? p36_vk_stream.cap : p36_gpu_stream_expert_cache_configured_count(),
            p36_vk_stream.budget,
            p36_vk_stream.allocation_failures,
            p36_vk_stream.indexed ? "index" : "scan",
            p36_vk_stream.lookup_steps,
            p36_vk_stream.hits,
            p36_vk_stream.misses,
            p36_vk_stream.loads,
            p36_vk_stream.evictions,
            p36_vk.props.deviceName[0] ? p36_vk.props.deviceName : "uninitialized",
            VK_VERSION_MAJOR(p36_vk.api_version),
            VK_VERSION_MINOR(p36_vk.api_version),
            VK_VERSION_PATCH(p36_vk.api_version));
}

void p36_gpu_prof_reset(void) {
    pthread_mutex_lock(&p36_vk_mu);
    (void)p36_vk_flush_reason_unlocked("submit_wait_prof_reset");
    p36_vk_prof_reset();
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_prof_report(const char *label) {
    pthread_mutex_lock(&p36_vk_mu);
    (void)p36_vk_flush_reason_unlocked("submit_wait_prof_report");
    pthread_mutex_unlock(&p36_vk_mu);
    if (label && label[0]) fprintf(stderr, "p36: Vulkan prof label=%s\n", label);
    p36_vk_prof_report();
}

/* The f32 dense path carries two dispatches with very different costs and
 * reported them under one "dense_f32" profiler row, which hid what the time
 * was actually spent on: the MoE router gate [n_embd, n_expert] runs
 * n_expert workgroups, while the shared-expert gate [n_embd, 1] runs a
 * single workgroup and pays a full pipeline round-trip for 8 KB.  Tagging by
 * shape mirrors p36_vk_q8_0_op_name(); it is diagnostic naming only and
 * changes no dispatch.  Names are literals, so no allocation or thread
 * safety concerns, and the strings only materialise when profiling is on. */
static const char *p36_vk_f32_op_name(uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, bool fast) {
    if (n_tok == 1) {
        if (out_dim == 1) return fast ? "dense_f32f_d_1xN" : "dense_f32_d_1xN";
        if (in_dim == 2048 && out_dim == 256) return fast ? "dense_f32f_d_256x2048" : "dense_f32_d_256x2048";
        return fast ? "dense_f32f_d_other" : "dense_f32_d_other";
    }
    if (out_dim == 1) return fast ? "dense_f32f_p_1xN" : "dense_f32_p_1xN";
    if (in_dim == 2048 && out_dim == 256) return fast ? "dense_f32f_p_256x2048" : "dense_f32_p_256x2048";
    return fast ? "dense_f32f_p_other" : "dense_f32_p_other";
}

static int p36_vk_matmul_dense(p36_vk_kernel *kernel,
                               p36_gpu_tensor *out,
                               const void *model_map,
                               uint64_t model_size,
                               uint64_t weight_offset,
                               uint64_t weight_bytes,
                               uint64_t in_dim,
                               uint64_t out_dim,
                               const p36_gpu_tensor *x,
                               uint64_t n_tok,
                               float scale) {
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
    if (!weights) {
        pthread_mutex_unlock(&p36_vk_mu);
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
        float scale;
    } push = { (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, scale };
    const p36_gpu_tensor *bindings[3] = { weights, x, out };
    /* The bf16 shader assigns one cooperative workgroup to each row.  The
     * legacy f16/f32 shaders index rows by global id, so their trailing
     * groups simply exit. */
    const char *op = "dense_f16";
    if (kernel != &p36_vk.matmul_f16 && kernel != &p36_vk.matmul_bf16) {
        op = p36_vk.prof_ops
                 ? p36_vk_f32_op_name(in_dim, out_dim, n_tok,
                                      kernel == &p36_vk.matmul_f32_fast ||
                                      kernel == &p36_vk.matmul_f32_fast_w256)
                 : "dense_f32";
    }
    int ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                 (uint32_t)out_dim, (uint32_t)n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}


int p36_gpu_matmul_f16_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const p36_gpu_tensor *x,
                              uint64_t n_tok) {
    uint64_t elems = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    return p36_vk_matmul_dense(&p36_vk.matmul_f16, out, model_map, model_size, weight_offset,
                               elems * sizeof(uint16_t), in_dim, out_dim, x, n_tok, 1.0f);
}

int p36_gpu_matmul_bf16_scaled_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const p36_gpu_tensor *x,
                                      uint64_t n_tok,
                                      float scale) {
    uint64_t elems = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    return p36_vk_matmul_dense(&p36_vk.matmul_bf16, out, model_map, model_size, weight_offset,
                               elems * sizeof(uint16_t), in_dim, out_dim, x, n_tok, scale);
}

int p36_gpu_vision_stream_init(uint64_t max_weight_bytes) {
    if (p36_vk_vision_weight && p36_vk_vision_weight->bytes >= max_weight_bytes)
        return 1;
    p36_gpu_tensor_free(p36_vk_vision_weight);
    p36_vk_vision_weight = p36_gpu_tensor_alloc_uninitialized(max_weight_bytes);
    return p36_vk_vision_weight != NULL;
}

static unsigned char *p36_vk_vision_weight_begin(uint64_t in_dim,
                                                 uint64_t out_dim,
                                                 uint64_t n_tok) {
    uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    if (!p36_vk_vision_weight || weight_bytes > p36_vk_vision_weight->bytes ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX)
        return NULL;
    return p36_gpu_tensor_contents_named(
        p36_vk_vision_weight, "submit_wait_vision_weight_reuse");
}

static int p36_vk_vision_matmul_dispatch(p36_gpu_tensor *out,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok);

int p36_gpu_vision_matmul_f16_fill(p36_gpu_tensor *out,
                                   p36_gpu_vision_fill_fn fill, void *ctx,
                                   uint64_t in_dim, uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok) {
    if (!fill) return 0;
    unsigned char *dst = p36_vk_vision_weight_begin(in_dim, out_dim, n_tok);
    if (!dst) return 0;
    if (!fill(ctx, (uint16_t *)dst)) return 0;
    return p36_vk_vision_matmul_dispatch(out, in_dim, out_dim, x, n_tok);
}

int p36_gpu_vision_matmul_f16_disk(p36_gpu_tensor *out, int fd,
                                   uint64_t offset, uint64_t in_dim,
                                   uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok) {
    uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    unsigned char *dst = p36_vk_vision_weight_begin(in_dim, out_dim, n_tok);
    if (!dst) return 0;
    uint64_t done = 0;
    while (done < weight_bytes) {
        ssize_t n = pread(fd, dst + done, (size_t)(weight_bytes - done),
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (uint64_t)n;
    }
    return p36_vk_vision_matmul_dispatch(out, in_dim, out_dim, x, n_tok);
}

static int p36_vk_vision_matmul_dispatch(p36_gpu_tensor *out,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok) {
    pthread_mutex_lock(&p36_vk_mu);
    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
        float scale;
    } push = {(uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, 1.0f};
    const p36_gpu_tensor *bindings[3] = {p36_vk_vision_weight, x, out};
    int ok = p36_vk_run_unlocked("vision_f16", &p36_vk.vision_matmul_f16,
                                 bindings, &push, sizeof(push),
                                 ((uint32_t)out_dim + 15u) / 16u,
                                 ((uint32_t)n_tok + 15u) / 16u, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_vision_attention(p36_gpu_tensor *out,
                             const p36_gpu_tensor *qkv,
                             uint32_t rows) {
    if (!out || !qkv || !rows) return 0;
    struct { uint32_t rows; } push = {rows};
    const p36_gpu_tensor *bindings[2] = {qkv, out};
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("vision_attention", &p36_vk.vision_attention,
                                 bindings, &push, sizeof(push), rows, 16u, 1u);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* The f32 tensors in this model are the small control projections (expert
 * router logits, delta-net alpha/beta, the shared-expert gate scalar):
 * out_dim is 1..256, far too little parallelism to feed the GPU, and the
 * CPU-exact serial fp64 chain is latency-bound there.  Running the same
 * loop as p36_tensor_matvec() on the host is faster than a dispatch and,
 * because the inputs are host-produced norms, costs no batch flush at
 * all -- the whole router stays off the GPU timeline. */
typedef struct {
    const float *w;
    const float *x;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    float scale;
} p36_vk_matmul_f32_ctx;

static void p36_vk_matmul_f32_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_matmul_f32_ctx *ctx = (p36_vk_matmul_f32_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *xt = ctx->x + tok * ctx->in_dim;
        float *ot = ctx->out + tok * ctx->out_dim;
        uint64_t row = 0;
        /* Four independent row chains per pass: each keeps the exact
         * sequential rounding of p36_tensor_matvec(), the interleave only
         * hides the fp64 add latency. */
        for (; row + 4 <= ctx->out_dim; row += 4) {
            const float *w0 = ctx->w + row * ctx->in_dim;
            const float *w1 = w0 + ctx->in_dim;
            const float *w2 = w1 + ctx->in_dim;
            const float *w3 = w2 + ctx->in_dim;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) {
                double xv = (double)xt[i];
                a0 += (double)w0[i] * xv;
                a1 += (double)w1[i] * xv;
                a2 += (double)w2[i] * xv;
                a3 += (double)w3[i] * xv;
            }
            ot[row] = (float)a0 * ctx->scale;
            ot[row + 1] = (float)a1 * ctx->scale;
            ot[row + 2] = (float)a2 * ctx->scale;
            ot[row + 3] = (float)a3 * ctx->scale;
        }
        for (; row < ctx->out_dim; row++) {
            const float *wr = ctx->w + row * ctx->in_dim;
            double acc = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xt[i];
            ot[row] = (float)acc * ctx->scale;
        }
    }
}

int p36_gpu_matmul_f32_scaled_tensor(p36_gpu_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const p36_gpu_tensor *x,
                                     uint64_t n_tok,
                                     float scale) {
    uint64_t elems = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    /* Fast mode keeps these small control matvecs on GPU: a host fallback
     * here would force a flush now that nothing else reads activations back
     * during decode. The vec4 f32 kernel is the normal path; --quality or
     * P36_VK_F32_FAST=0 retains the fp64 accumulation for exact routing. */
    /* The 256-thread variant is much faster but changes the router gate's
     * summation order, so it is opt-in until an eval clears it. */
    p36_vk_kernel *fast_kernel = &p36_vk.matmul_f32_fast;
    if (getenv("P36_VK_F32_FAST_WIDE") && getenv("P36_VK_F32_FAST_WIDE")[0] == '1')
        fast_kernel = &p36_vk.matmul_f32_fast_w256;
    if (!p36_gpu_quality &&
        p36_vk_matmul_dense(p36_vk_use_f32_fast() && (in_dim & 3u) == 0u
                                ? fast_kernel : &p36_vk.matmul_f32,
                            out, model_map, model_size, weight_offset,
                            elems * sizeof(float), in_dim, out_dim, x, n_tok, scale)) {
        return 1;
    }

    const float *w = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                         elems * sizeof(float));
    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_matmul_f32_x");
    float *out_host = p36_vk_tensor_contents_labeled(out, "submit_wait_matmul_f32_out");
    p36_vk_matmul_f32_ctx ctx;
    if (!w || !x_host || !out_host) return 0;

    ctx.w = w;
    ctx.x = x_host;
    ctx.out = out_host;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.scale = scale;
    if (n_tok > 1) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_matmul_f32_rows, &ctx);
    else p36_vk_matmul_f32_rows(&ctx, 0, n_tok);
    return 1;
}

int p36_gpu_matmul_f32_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const p36_gpu_tensor *x,
                              uint64_t n_tok) {
    return p36_gpu_matmul_f32_scaled_tensor(out, model_map, model_size, weight_offset,
                                            in_dim, out_dim, x, n_tok, 1.0f);
}

typedef struct {
    unsigned char *xq;
    const float *x;
    uint64_t in_dim;
    uint64_t row_bytes;
} p36_vk_quantize_q8_0_ctx;

static void p36_vk_quantize_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_quantize_q8_0_ctx *ctx = (p36_vk_quantize_q8_0_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        p36_quant_q8_0(ctx->x + t * ctx->in_dim,
                       ctx->xq + t * ctx->row_bytes,
                       (int64_t)ctx->in_dim);
    }
}

typedef struct {
    const unsigned char *w;
    const unsigned char *xq;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t n_tok;
    uint64_t row_bytes;
    float scale;
} p36_vk_matmul_q8_0_ctx;

static void p36_vk_matmul_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_matmul_q8_0_ctx *ctx = (p36_vk_matmul_q8_0_ctx *)opaque;

    for (uint64_t row = row0; row < row1; row++) {
        const void *wr = ctx->w + row * ctx->row_bytes;
        for (uint64_t tok = 0; tok < ctx->n_tok; tok++) {
            float acc = p36_quant_dot_q8_0(wr,
                                           ctx->xq + tok * ctx->row_bytes,
                                           (int)ctx->in_dim);
            ctx->out[tok * ctx->out_dim + row] = acc * ctx->scale;
        }
    }
}

static const char *p36_vk_q8_0_op_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * P36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_d_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_d_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_d_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_d_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_d_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_d_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_d_vocabx2048";
        return "dense_q8_0_d_other";
    }
    if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_p_2048x4096";
    if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_p_2048x512";
    if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_p_32x2048";
    if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_p_4096x2048";
    if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_p_512x2048";
    if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_p_8192x2048";
    if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_p_vocabx2048";
    return "dense_q8_0_p_other";
}

