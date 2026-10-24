static void p36_vk_prof_reset(void) {
    p36_vk_prof_dispatches = 0;
    p36_vk_prof_flushes = 0;
    p36_vk_prof_submit_ns = 0;
    p36_vk_prof_record_ns = 0;
    p36_vk_prof_query_count = 0;
    p36_vk_prof_kernel_n = 0;
    p36_vk_prof_op_n = 0;
}

static int p36_vk_stream_profile_cmp(const void *a, const void *b) {
    const p36_vk_stream_profile_entry *ea = a;
    const p36_vk_stream_profile_entry *eb = b;
    if (ea->count < eb->count) return 1;
    if (ea->count > eb->count) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

static void p36_vk_stream_profile_init(void) {
    const char *path = getenv("P36_VK_STREAMING_EXPERT_PROFILE");
    const char *hotlist = getenv("P36_VK_STREAMING_EXPERT_HOTLIST_OUT");
    if (hotlist && hotlist[0]) path = hotlist;
    if (!path || !path[0] || p36_vk_stream_prof.active) return;
    memset(&p36_vk_stream_prof, 0, sizeof(p36_vk_stream_prof));
    p36_vk_stream_prof.active = true;
    p36_vk_stream_prof.path = path;
    atexit(p36_vk_stream_profile_report);
    fprintf(stderr, "p36: Vulkan streaming expert profiler active: %s\n", path);
}

static void p36_vk_stream_profile_record_unlocked(const p36_gpu_stream_expert_table *t,
                                                  const uint32_t *selected,
                                                  uint64_t n_selected) {
    if (!p36_vk_stream_prof.active || !t || !selected) return;
    if (t->layer >= P36_VK_STREAM_PROFILE_LAYERS) return;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= P36_VK_STREAM_PROFILE_EXPERTS) continue;
        p36_vk_stream_prof.hist[t->layer][expert]++;
        p36_vk_stream_prof.total++;
    }
}

static void p36_vk_stream_profile_report(void) {
    p36_vk_stream_profile *p = &p36_vk_stream_prof;
    p36_vk_stream_profile_entry entries[P36_VK_STREAM_PROFILE_LAYERS *
                                        P36_VK_STREAM_PROFILE_EXPERTS];
    FILE *fp;
    uint32_t n = 0;
    if (!p->active || p->reported || !p->path || !p->path[0]) return;
    p->reported = true;
    for (uint32_t il = 0; il < P36_VK_STREAM_PROFILE_LAYERS; il++) {
        for (uint32_t expert = 0; expert < P36_VK_STREAM_PROFILE_EXPERTS; expert++) {
            entries[n++] = (p36_vk_stream_profile_entry) {
                .layer = il,
                .expert = expert,
                .count = p->hist[il][expert],
            };
        }
    }
    qsort(entries, n, sizeof(entries[0]), p36_vk_stream_profile_cmp);
    fp = fopen(p->path, "wb");
    if (!fp) {
        fprintf(stderr, "p36: failed to write streaming expert profile %s: %s\n",
                p->path, strerror(errno));
        return;
    }
    fprintf(fp,
            "# p36 expert hotlist v1\n"
            "# layers %u\n"
            "# experts %u\n"
            "# selections %" PRIu64 "\n"
            "# columns: layer expert hits weight\n",
            P36_VK_STREAM_PROFILE_LAYERS,
            P36_VK_STREAM_PROFILE_EXPERTS,
            p->total);
    for (uint32_t i = 0; i < n; i++) {
        fprintf(fp, "%u %u %" PRIu64 " 0\n",
                entries[i].layer,
                entries[i].expert,
                entries[i].count);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "p36: failed to close streaming expert profile %s: %s\n",
                p->path, strerror(errno));
    } else {
        fprintf(stderr, "p36: wrote Vulkan streaming expert profile %s (%" PRIu64 " selections)\n",
                p->path, p->total);
    }
}

static const char *p36_vk_prof_op_name(const char *name) {
    return (name && name[0]) ? name : "other";
}

static const char *p36_vk_prof_iq3_shape(uint32_t type, uint32_t n_tok,
                                         uint32_t in_dim, uint32_t out_dim,
                                         const char *fallback) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("P36_VK_PROF_SHAPE") != NULL;
    if (!enabled) return fallback;
    for (uint32_t i = 0; i < p36_vk_prof_shape_n; i++) {
        p36_vk_prof_shape *s = &p36_vk_prof_shapes[i];
        if (s->type == type && s->n_tok == n_tok &&
            s->in_dim == in_dim && s->out_dim == out_dim) return s->name;
    }
    if (p36_vk_prof_shape_n == P36_VK_PROF_SHAPE_CAP) return fallback;
    p36_vk_prof_shape *s = &p36_vk_prof_shapes[p36_vk_prof_shape_n++];
    s->type = type;
    s->n_tok = n_tok;
    s->in_dim = in_dim;
    s->out_dim = out_dim;
    const char *kind = type == P36_VK_TENSOR_IQ3_S ? "iq3_s" :
                       type == P36_VK_TENSOR_IQ3_XXS ? "iq3_xxs" :
                       type == P36_VK_TENSOR_IQ2_XXS ? "iq2_xxs" :
                       type == P36_VK_TENSOR_IQ2_XS ? "iq2_xs" :
                       type == P36_VK_TENSOR_IQ2_S ? "iq2_s" :
                       type == P36_VK_TENSOR_IQ1_S ? "iq1_s" :
                       type == P36_VK_TENSOR_IQ4_NL ? "iq4_nl" :
                       type == P36_VK_TENSOR_IQ4_XS ? "iq4_xs" :
                       type == P36_VK_TENSOR_Q2_K ? "q2k" :
                       type == P36_VK_TENSOR_Q3_K ? "q3k" :
                       type == P36_VK_TENSOR_Q4_K ? "q4k" :
                       type == P36_VK_TENSOR_Q5_K ? "q5k" :
                       type == P36_VK_TENSOR_Q6_K ? "q6k" : "iq";
    snprintf(s->name, sizeof(s->name), "dense_%s_%s_%ux%u_n%u",
             kind,
             n_tok == 1 ? "decode" : "mmq", in_dim, out_dim, n_tok);
    return s->name;
}

static p36_vk_prof_op_row *p36_vk_prof_op_get(const char *name) {
    name = p36_vk_prof_op_name(name);
    for (uint32_t i = 0; i < p36_vk_prof_op_n; i++) {
        if (p36_vk_prof_op_rows[i].name == name || !strcmp(p36_vk_prof_op_rows[i].name, name)) {
            return &p36_vk_prof_op_rows[i];
        }
    }
    if (p36_vk_prof_op_n == P36_VK_PROF_OP_CAP) return NULL;
    memset(&p36_vk_prof_op_rows[p36_vk_prof_op_n], 0, sizeof(p36_vk_prof_op_rows[p36_vk_prof_op_n]));
    p36_vk_prof_op_rows[p36_vk_prof_op_n].name = name;
    return &p36_vk_prof_op_rows[p36_vk_prof_op_n++];
}

static void p36_vk_prof_op_add(const char *name,
                               uint64_t dispatches,
                               uint64_t flushes,
                               uint64_t groups,
                               uint64_t record_ns,
                               uint64_t submit_ns,
                               uint64_t gpu_ns,
                               uint64_t wall_ns) {
    if (!p36_vk.prof_ops) return;
    p36_vk_prof_op_row *r = p36_vk_prof_op_get(name);
    if (!r) return;
    r->dispatches += dispatches;
    r->flushes += flushes;
    r->groups += groups;
    r->record_ns += record_ns;
    r->submit_ns += submit_ns;
    r->gpu_ns += gpu_ns;
    r->wall_ns += wall_ns;
}

/* Barrier ownership: the barrier belongs to the dispatch that needs it, so the
 * op rows show which chain is serializing the token. */
static void p36_vk_prof_op_barrier(const char *name) {
    if (!p36_vk.prof_ops) return;
    p36_vk_prof_op_row *r = p36_vk_prof_op_get(name);
    if (r) r->barriers++;
}

static void p36_vk_prof_scope_begin(p36_vk_prof_scope *s, const char *name) {
    if (!s) return;
    s->active = p36_vk.prof_ops;
    s->name = p36_vk_prof_op_name(name);
    s->t0 = s->active ? p36_vk_now_ns() : 0;
}

static void p36_vk_prof_scope_end(p36_vk_prof_scope *s) {
    if (!s || !s->active) return;
    p36_vk_prof_op_add(s->name, 0, 0, 0, 0, 0, 0, p36_vk_now_ns() - s->t0);
    s->active = false;
}

static void p36_vk_prof_kernel_add(const char *path, uint64_t groups, uint64_t gpu_ns) {
    for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) {
        if (p36_vk_prof_kernel_rows[i].path == path || !strcmp(p36_vk_prof_kernel_rows[i].path, path)) {
            p36_vk_prof_kernel_rows[i].dispatches++;
            p36_vk_prof_kernel_rows[i].groups += groups;
            p36_vk_prof_kernel_rows[i].gpu_ns += gpu_ns;
            return;
        }
    }
    if (p36_vk_prof_kernel_n == P36_VK_PROF_KERNEL_CAP) return;
    p36_vk_prof_kernel_rows[p36_vk_prof_kernel_n++] = (p36_vk_prof_kernel_row){
        .path = path,
        .dispatches = 1,
        .groups = groups,
        .gpu_ns = gpu_ns,
    };
}

static void p36_vk_prof_kernel_collect_unlocked(void) {
    if (!p36_vk.prof_kernel || !p36_vk.query_pool || p36_vk_prof_query_count == 0) return;

    uint64_t ticks[P36_VK_PROF_QUERY_CAP];
    VkResult rc = vkGetQueryPoolResults(p36_vk.device,
                                        p36_vk.query_pool,
                                        0,
                                        p36_vk_prof_query_count,
                                        sizeof(ticks[0]) * p36_vk_prof_query_count,
                                        ticks,
                                        sizeof(ticks[0]),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (rc != VK_SUCCESS) {
        p36_vk_prof_query_count = 0;
        return;
    }

    uint64_t mask = UINT64_MAX;
    if (p36_vk.timestamp_valid_bits > 0 && p36_vk.timestamp_valid_bits < 64) {
        mask = (1ull << p36_vk.timestamp_valid_bits) - 1ull;
    }
    if (!p36_vk_prof_timeline_tried) {
        p36_vk_prof_timeline_tried = true;
        const char *tp = getenv("P36_VK_PROF_TIMELINE");
        if (tp && tp[0]) {
            p36_vk_prof_timeline_fp = fopen(tp, "wb");
            if (p36_vk_prof_timeline_fp) {
                fprintf(p36_vk_prof_timeline_fp,
                        "# p36 vulkan dispatch timeline v1\n"
                        "# tick_period_ns=%.6f valid_bits=%u\n"
                        "# columns: seq flush path groups start_tick end_tick dur_ns host_ns\n",
                        (double)p36_vk.props.limits.timestampPeriod,
                        p36_vk.timestamp_valid_bits);
            }
        }
    }
    for (uint32_t q = 0; q + 1 < p36_vk_prof_query_count; q += 2) {
        uint64_t a = ticks[q] & mask;
        uint64_t b = ticks[q + 1u] & mask;
        uint64_t dt = (b >= a) ? (b - a) : ((mask - a) + b + 1u);
        uint64_t ns = (uint64_t)((double)dt * (double)p36_vk.props.limits.timestampPeriod);
        p36_vk_prof_kernel_add(p36_vk_prof_query_path[q / 2u], p36_vk_prof_query_groups[q / 2u], ns);
        p36_vk_prof_op_add(p36_vk_prof_query_op[q / 2u], 0, 0, 0, 0, 0, ns, 0);
        if (p36_vk_prof_timeline_fp) {
            fprintf(p36_vk_prof_timeline_fp,
                    "%" PRIu64 " %" PRIu64 " %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    p36_vk_prof_timeline_seq++,
                    p36_vk_prof_flushes,
                    p36_vk_prof_query_path[q / 2u],
                    p36_vk_prof_query_groups[q / 2u],
                    a,
                    b,
                    ns,
                    p36_vk_prof_query_host_ns[q / 2u]);
        }
    }
    if (p36_vk_prof_timeline_fp) fflush(p36_vk_prof_timeline_fp);
    p36_vk_prof_query_count = 0;
}

static void *p36_vk_host_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&p36_vk_host.mu);
        while (seen_generation == p36_vk_host.generation && !p36_vk_host.shutdown) {
            pthread_cond_wait(&p36_vk_host.work_cv, &p36_vk_host.mu);
        }
        if (p36_vk_host.shutdown) {
            pthread_mutex_unlock(&p36_vk_host.mu);
            return NULL;
        }

        seen_generation = p36_vk_host.generation;
        p36_gpu_parallel_fn fn = p36_vk_host.fn;
        void *ctx = p36_vk_host.ctx;
        uint64_t n_rows = p36_vk_host.n_rows;
        uint32_t n_threads = p36_vk_host.n_threads;
        pthread_mutex_unlock(&p36_vk_host.mu);

        {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                p36_vk_parallel_depth++;
                fn(ctx, row0, row1);
                p36_vk_parallel_depth--;
            }
        }

        pthread_mutex_lock(&p36_vk_host.mu);
        p36_vk_host.done++;
        if (p36_vk_host.done == p36_vk_host.n_workers) pthread_cond_signal(&p36_vk_host.done_cv);
        pthread_mutex_unlock(&p36_vk_host.mu);
    }
}

static void p36_vk_host_threads_init(void) {
    pthread_mutex_lock(&p36_vk_host_init_mu);
    if (p36_vk_host.initialized) {
        pthread_mutex_unlock(&p36_vk_host_init_mu);
        return;
    }

    uint32_t n_threads = 12;
    long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("P36_THREADS");
    if (online_cpus > 0) n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (n_threads > P36_VK_MAX_THREADS) n_threads = P36_VK_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&p36_vk_host.mu, NULL);
    pthread_mutex_init(&p36_vk_host.submit_mu, NULL);
    pthread_cond_init(&p36_vk_host.work_cv, NULL);
    pthread_cond_init(&p36_vk_host.done_cv, NULL);
    p36_vk_host.n_threads = 1;
    p36_vk_host.n_workers = 0;
    p36_vk_host.generation = 0;
    p36_vk_host.done = 0;
    p36_vk_host.shutdown = false;
    p36_vk_host.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&p36_vk_host.threads[i], NULL,
                           p36_vk_host_worker_main, (void *)(uintptr_t)i) != 0) {
            fprintf(stderr, "p36: Vulkan host pool started %u/%u threads\n", i, n_threads);
            break;
        }
        p36_vk_host.n_threads = i + 1u;
        p36_vk_host.n_workers = i;
    }
    pthread_mutex_unlock(&p36_vk_host_init_mu);
}

static void p36_vk_host_threads_shutdown(void) {
    pthread_mutex_lock(&p36_vk_host_init_mu);
    if (!p36_vk_host.initialized) {
        pthread_mutex_unlock(&p36_vk_host_init_mu);
        return;
    }

    pthread_mutex_lock(&p36_vk_host.submit_mu);
    pthread_mutex_lock(&p36_vk_host.mu);
    p36_vk_host.shutdown = true;
    p36_vk_host.generation++;
    pthread_cond_broadcast(&p36_vk_host.work_cv);
    pthread_mutex_unlock(&p36_vk_host.mu);
    for (uint32_t i = 1; i < p36_vk_host.n_threads; i++) pthread_join(p36_vk_host.threads[i], NULL);
    pthread_mutex_unlock(&p36_vk_host.submit_mu);

    pthread_cond_destroy(&p36_vk_host.done_cv);
    pthread_cond_destroy(&p36_vk_host.work_cv);
    pthread_mutex_destroy(&p36_vk_host.submit_mu);
    pthread_mutex_destroy(&p36_vk_host.mu);
    memset(&p36_vk_host, 0, sizeof(p36_vk_host));
    pthread_mutex_unlock(&p36_vk_host_init_mu);
}

void p36_gpu_parallel_for_rows(uint64_t n_rows,
                               uint64_t min_parallel_rows,
                               p36_gpu_parallel_fn fn,
                               void *ctx) {
    uint64_t rows_per_thread;
    uint64_t main_row1;

    if (!fn || n_rows == 0) return;
    p36_vk_host_threads_init();
    if (p36_vk_parallel_depth > 0 || p36_vk_host.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&p36_vk_host.submit_mu);
    pthread_mutex_lock(&p36_vk_host.mu);
    p36_vk_host.fn = fn;
    p36_vk_host.ctx = ctx;
    p36_vk_host.n_rows = n_rows;
    p36_vk_host.done = 0;
    p36_vk_host.generation++;
    pthread_cond_broadcast(&p36_vk_host.work_cv);
    rows_per_thread = (n_rows + p36_vk_host.n_threads - 1u) / p36_vk_host.n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&p36_vk_host.mu);

    if (main_row1 > 0) {
        p36_vk_parallel_depth++;
        fn(ctx, 0, main_row1);
        p36_vk_parallel_depth--;
    }

    pthread_mutex_lock(&p36_vk_host.mu);
    while (p36_vk_host.done < p36_vk_host.n_workers) {
        pthread_cond_wait(&p36_vk_host.done_cv, &p36_vk_host.mu);
    }
    pthread_mutex_unlock(&p36_vk_host.mu);
    pthread_mutex_unlock(&p36_vk_host.submit_mu);
}

/* Root tensors written by the open batch.  p36_gpu_tensor_contents() only
 * needs to submit when the host touches one of these (or the set
 * overflowed); reads of host-produced tensors (router selection, q8
 * staging, swiglu outputs) keep recording.  This is safe because the
 * schedulers never host-write a tensor that recorded-but-unsubmitted
 * commands read: activations flow forward, so a host write always precedes
 * the recording of its GPU consumers. */
/* A 512-token prefill chunk writes more distinct roots than this, so
 * batch_dirty_all trips once per chunk.  That used to drain the pipeline
 * on the next chunk's embed-stage map (~190ms/chunk); now contents()
 * consults the per-root gpu_written flag, so host-only tensors skip the
 * flush even under dirty_all, and scratch memory is bounded by the
 * seq-based retirement sweep on each eager submit rather than by the
 * accidental per-chunk drain. */
enum { P36_VK_WRITTEN_CAP = 64 };
static p36_gpu_tensor *p36_vk_batch_written[P36_VK_WRITTEN_CAP];
static uint32_t p36_vk_batch_written_n;
static bool p36_vk_batch_dirty_all;

/* Hazard sets since the last pipeline barrier in the open batch.  A barrier
 * is emitted only when a dispatch touches a tensor written since then, or
 * writes one that was read (RAW/WAW/WAR); independent dispatches overlap on
 * the GPU instead of draining the pipeline ~1300 times per decoded token.
 * Weight-arena roots never enter the sets: they are uploaded from the host
 * before use and no shader writes them. */
enum { P36_VK_HAZARD_CAP = 48 };
static p36_gpu_tensor *p36_vk_hazard_written[P36_VK_HAZARD_CAP];
static p36_gpu_tensor *p36_vk_hazard_read[P36_VK_HAZARD_CAP];
static uint32_t p36_vk_hazard_written_n;
static uint32_t p36_vk_hazard_read_n;
static bool p36_vk_hazard_all;

static bool p36_vk_hazard_in(p36_gpu_tensor **set, uint32_t n, const p36_gpu_tensor *root) {
    for (uint32_t i = 0; i < n; i++) {
        if (set[i] == root) return true;
    }
    return false;
}

static void p36_vk_hazard_add(p36_gpu_tensor **set, uint32_t *n, p36_gpu_tensor *root) {
    if (root->weight_only || p36_vk_hazard_in(set, *n, root)) return;
    if (*n == P36_VK_HAZARD_CAP) p36_vk_hazard_all = true;
    else set[(*n)++] = root;
}

static void p36_vk_hazard_reset(void) {
    p36_vk_hazard_written_n = 0;
    p36_vk_hazard_read_n = 0;
    p36_vk_hazard_all = false;
}

/* GPU work is outstanding while a batch records or an eager submit's fence
 * has not been waited on. Frees and host reads must respect both. */
static bool p36_vk_gpu_busy_unlocked(void);
static const char *p36_vk_flush_reason = "submit_wait_other";
static __thread const char *p36_vk_tensor_contents_reason = "submit_wait_tensor_contents";

static int p36_vk_flush_reason_unlocked(const char *reason);
static void *p36_vk_tensor_contents_labeled(p36_gpu_tensor *tensor, const char *reason);
static int p36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out);
static void p36_vk_note_written_tensor(p36_gpu_tensor *tensor);

static uint64_t p36_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static bool p36_u64_mul_ok(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool p36_gpu_tensor_range_ok(const p36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
static const unsigned char *p36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes);
static p36_gpu_tensor *p36_vk_quantize_q8_k_alloc_tensor(const p36_gpu_tensor *x,
                                                         uint64_t in_dim,
                                                         uint64_t n_tok);
static int p36_vk_run_unlocked(
        const char                 *op,
        p36_vk_kernel              *kernel,
        const p36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z);

typedef struct {
    float d;
    int8_t qs[P36_VK_QK_K];
    int16_t bsums[P36_VK_QK_K / 16];
} p36_vk_block_q8_K_packed;

enum { P36_MAX_VK_Q8_BLOCKS = 32 };

typedef struct {
    p36_vk_block_q8_K *dst;
    const float *x;
    uint64_t in_dim;
    uint64_t blocks;
} p36_vk_quantize_q8_k_ctx;

static void p36_vk_quantize_q8_k_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_quantize_q8_k_ctx *ctx = (p36_vk_quantize_q8_k_ctx *)opaque;
    p36_vk_block_q8_K_packed tmp[P36_MAX_VK_Q8_BLOCKS];

    for (uint64_t t = row0; t < row1; t++) {
        p36_vk_block_q8_K *qt = ctx->dst + t * ctx->blocks;
        p36_quant_q8_k(ctx->x + t * ctx->in_dim, tmp, (int64_t)ctx->in_dim);
        for (uint64_t b = 0; b < ctx->blocks; b++) {
            qt[b].d = tmp[b].d;
            qt[b].dmin = 0.0f;
            memcpy(qt[b].qs, tmp[b].qs, sizeof(qt[b].qs));
            memcpy(qt[b].bsums, tmp[b].bsums, sizeof(qt[b].bsums));
        }
    }
}

static void p36_vk_quantize_q8_k_host(p36_vk_block_q8_K *dst,
                                      const float *x,
                                      uint64_t n_tok,
                                      uint64_t in_dim) {
    const uint64_t blocks = in_dim / P36_VK_QK_K;
    p36_vk_quantize_q8_k_ctx ctx;

    if (blocks > P36_MAX_VK_Q8_BLOCKS) abort();
    ctx.dst = dst;
    ctx.x = x;
    ctx.in_dim = in_dim;
    ctx.blocks = blocks;
    if (n_tok * in_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 4, p36_vk_quantize_q8_k_rows, &ctx);
    else p36_vk_quantize_q8_k_rows(&ctx, 0, n_tok);
}

static void p36_vk_q5_scale_min(const uint8_t *src, uint32_t j, uint8_t *scale, uint8_t *mn) {
    const uint32_t s0 = src[j];
    if (j < 4u) {
        *scale = (uint8_t)(s0 & 63u);
        *mn = (uint8_t)(src[4u + j] & 63u);
    } else {
        const uint32_t sj4 = src[j + 4u];
        const uint32_t sjm4 = src[j - 4u];
        *scale = (uint8_t)((sj4 & 15u) | ((sjm4 >> 6) << 4));
        *mn = (uint8_t)((sj4 >> 4) | ((s0 >> 6) << 4));
    }
}

static uint64_t p36_vk_kquant_mmq_block_bytes(uint32_t type) {
    switch (type) {
    case P36_VK_TENSOR_Q5_K: return sizeof(p36_vk_block_q5_Kx8);
    case P36_VK_TENSOR_Q6_K: return sizeof(p36_vk_block_q6_Kx8);
    default: return 0;
    }
}

static bool p36_vk_use_mmq(void) {
    const char *env = getenv("P36_VK_MMQ");
    return env && env[0] && env[0] != '0';
}

static bool p36_vk_use_mmq_fast(void) {
    const char *env = getenv("P36_VK_MMQ_FAST");
    return env && env[0] && env[0] != '0';
}

/* Largest batch the Q2_0/PQ2_0 small-batch kernel takes before falling back
 * to the 128-token mmq tile.  0 disables it. */
static uint32_t p36_vk_q2_small_max(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("P36_VK_Q2_SMALL_MAX");
        long v = env && env[0] ? strtol(env, NULL, 10) : 16;
        cached = v < 0 ? 0 : (v > 4096 ? 4096 : (int)v);
    }
    return (uint32_t)cached;
}

static bool p36_vk_env_default_on(const char *name) {
    const char *env = getenv(name);
    return !env || !env[0] || env[0] != '0';
}

static bool p36_vk_have_subgroups(bool clustered) {
    return p36_vk.subgroup_size >= 32u && p36_vk.subgroup_arithmetic &&
           (!clustered || p36_vk.subgroup_clustered);
}

static bool p36_vk_use_gpu_attn_post(void) {
    return p36_vk_env_default_on("P36_VK_GPU_ATTN_POST");
}

/* Prefill batches at or above this token count route routed-MoE work
 * through the GEMM kernels; below it the matvec path keeps
 * decode-identical numerics for chunk tails and speculative windows. */
static uint32_t p36_vk_moe_gemm_min(void) {
    const char *env = getenv("P36_VK_MOE_GEMM_MIN");
    long v = env ? strtol(env, NULL, 10) : 0;
    return v > 0 ? (uint32_t)v : 128u;
}

static bool p36_vk_use_moe_down_sum_decode(void) {
    return p36_vk.have_storage16 && p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_MOE_DOWN_SUM_DECODE");
}

/* Sampled once: the runtime skips the scores scratch allocation when the
 * fused attention path is on, so the choice must not change mid-process. */
static bool p36_vk_use_attn_qtile(void) {
    static int cached = -1;
    if (cached < 0) cached = p36_vk_env_default_on("P36_VK_ATTN_QTILE") ? 1 : 0;
    return cached != 0 && p36_vk_have_subgroups(true);
}

