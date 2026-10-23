static uint32_t p36_q8_0_row_bytes(uint32_t n) {
    return ((n + P36_QK8_0 - 1u) / P36_QK8_0) * P36_Q8_0_BYTES;
}

static uint32_t p36_q4_0_row_bytes(uint32_t n) {
    return ((n + P36_QK4_0 - 1u) / P36_QK4_0) * P36_Q4_0_BYTES;
}

/* Two opt-in GPU fast paths measure faster but change numerics, so they stay
 * off unless asked for.  Every front end exposes the same two flags and both
 * resolve to the environment variables the Vulkan backend already reads, so
 * there is exactly one implementation of the behaviour.
 *
 *   --f32-fast-wide   256-thread matmul_f32_fast.  Measured +6.1% decode and
 *                     +17% prefill on a short benchmark, but it reassociates
 *                     the MoE router gate, which can change which experts a
 *                     token selects.  That is not a theoretical risk: on one
 *                     BC-250, long-context chat with tool calling degenerated
 *                     into unterminated generation with this flag on.
 *   --attn-span N     split-K span in keys (default 512).  128 measured +2.2%
 *                     decode at ctx 2048; it only regroups an ordered sum, but
 *                     the span count is context/span, so both the rounding and
 *                     attn_combine's own cost grow with context while the
 *                     saving does not.  Not validated for long context.
 *
 * Neither is bit-exact, so neither is on by default, and neither is passed by
 * the shipped launcher scripts. */
void p36_set_gpu_fast_path_env(const char *name, const char *value) {
    if (!name || !value) return;
    setenv(name, value, 1);
}

const char *p36_kv_cache_type_name(p36_kv_cache_type type) {
    switch (type) {
    case P36_KV_CACHE_F16: return "f16";
    case P36_KV_CACHE_Q8_0: return "q8_0";
    case P36_KV_CACHE_Q4_0: return "q4_0";
    default: return "unknown";
    }
}

p36_kv_cache_type p36_default_kv_cache_type_k(p36_backend backend, bool ssd_streaming) {
    return (backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL) &&
           !ssd_streaming ?
           P36_KV_CACHE_Q8_0 : P36_KV_CACHE_F16;
}

p36_kv_cache_type p36_default_kv_cache_type_v(p36_backend backend, bool ssd_streaming) {
    return (backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL) &&
           !ssd_streaming ?
           P36_KV_CACHE_Q4_0 : P36_KV_CACHE_F16;
}

bool p36_parse_kv_cache_type(const char *s, p36_kv_cache_type *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "f16")) {
        *out = P36_KV_CACHE_F16;
        return true;
    }
    if (!strcmp(s, "q8_0") || !strcmp(s, "q8_o")) {
        *out = P36_KV_CACHE_Q8_0;
        return true;
    }
    if (!strcmp(s, "q4_0") || !strcmp(s, "q4_o")) {
        *out = P36_KV_CACHE_Q4_0;
        return true;
    }
    return false;
}

static uint32_t p36_kv_cache_row_bytes(p36_kv_cache_type type, uint32_t n) {
    switch (type) {
    case P36_KV_CACHE_F16: return n * (uint32_t)sizeof(uint16_t);
    case P36_KV_CACHE_Q8_0: return p36_q8_0_row_bytes(n);
    case P36_KV_CACHE_Q4_0: return p36_q4_0_row_bytes(n);
    default: return 0;
    }
}

static float p36_kv_cache_at(const uint8_t *row, p36_kv_cache_type type, uint32_t i) {
    if (type == P36_KV_CACHE_F16) {
        const uint16_t *h = (const uint16_t *)row;
        return p36_f16_to_f32(h[i]);
    }
    if (type == P36_KV_CACHE_Q8_0) {
        const p36_block_q8_0 *b = (const p36_block_q8_0 *)row;
        const p36_block_q8_0 *x = b + i / P36_QK8_0;
        return p36_f16_to_f32(x->d) * (float)x->qs[i % P36_QK8_0];
    }
    if (type == P36_KV_CACHE_Q4_0) {
        const p36_block_q4_0 *b = (const p36_block_q4_0 *)row;
        const p36_block_q4_0 *x = b + i / P36_QK4_0;
        uint32_t j = i % P36_QK4_0;
        uint8_t q = x->qs[j % (P36_QK4_0 / 2u)];
        q = j < P36_QK4_0 / 2u ? q & 0x0fu : q >> 4;
        return p36_f16_to_f32(x->d) * (float)((int)q - 8);
    }
    return 0.0f;
}

static void p36_kv_cache_store_row(uint8_t *dst, p36_kv_cache_type type, const float *src, uint32_t n) {
    if (type == P36_KV_CACHE_F16) {
        uint16_t *h = (uint16_t *)dst;
        for (uint32_t i = 0; i < n; i++) h[i] = p36_f32_to_f16(src[i]);
    } else if (type == P36_KV_CACHE_Q8_0) {
        p36_quant_q8_0(src, dst, (int64_t)n);
    } else if (type == P36_KV_CACHE_Q4_0) {
        p36_quant_q4_0_kv(src, dst, (int64_t)n);
    }
}

static uint64_t p36_cpu_prefill_scratch_bytes(uint32_t cap) {
    uint64_t total = 0;
    total += (uint64_t)P36_N_EMBD * 2u * sizeof(float);
    total += (uint64_t)P36_CPU_SCRATCH_FLOATS * 6u * sizeof(float);
    total += (uint64_t)cap * P36_N_EMBD * 3u * sizeof(float);
    total += (uint64_t)cap * (P36_N_HEAD * P36_N_HEAD_DIM * 2u) * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_INNER * 3u * sizeof(float);
    total += (uint64_t)cap * P36_N_HEAD_KV * (P36_N_HEAD_DIM + P36_N_VALUE_DIM) * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_CONV_DIM * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_DT_RANK * 2u * sizeof(float);
    total += (uint64_t)cap * P36_N_EXPERT * sizeof(float);
    total += (uint64_t)cap * P36_N_FF_SHARED * 4u * sizeof(float);
    total += (uint64_t)cap * sizeof(float);
    total += (uint64_t)cap * P36_MAX_Q8_K_BYTES;
    return total;
}

static void p36_cpu_threads_shutdown(void) {
    pthread_mutex_lock(&p36_cpu_workers_init_mu);
    if (!p36_cpu_workers.initialized) {
        pthread_mutex_unlock(&p36_cpu_workers_init_mu);
        return;
    }
    pthread_mutex_lock(&p36_cpu_workers.submit_mu);
    pthread_mutex_lock(&p36_cpu_workers.mu);
    p36_cpu_workers.shutdown = true;
    p36_cpu_workers.generation++;
    pthread_cond_broadcast(&p36_cpu_workers.work_cv);
    pthread_mutex_unlock(&p36_cpu_workers.mu);
    for (uint32_t i = 1; i < p36_cpu_workers.n_threads; i++) {
        pthread_join(p36_cpu_workers.threads[i], NULL);
    }
    pthread_mutex_unlock(&p36_cpu_workers.submit_mu);
    pthread_cond_destroy(&p36_cpu_workers.done_cv);
    pthread_cond_destroy(&p36_cpu_workers.work_cv);
    pthread_mutex_destroy(&p36_cpu_workers.submit_mu);
    pthread_mutex_destroy(&p36_cpu_workers.mu);
    memset(&p36_cpu_workers, 0, sizeof(p36_cpu_workers));
    pthread_mutex_unlock(&p36_cpu_workers_init_mu);
}

static void *p36_cpu_worker_main(void *arg) {
    uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&p36_cpu_workers.mu);
        while (seen_generation == p36_cpu_workers.generation && !p36_cpu_workers.shutdown) {
            pthread_cond_wait(&p36_cpu_workers.work_cv, &p36_cpu_workers.mu);
        }
        if (p36_cpu_workers.shutdown) {
            pthread_mutex_unlock(&p36_cpu_workers.mu);
            return NULL;
        }
        seen_generation = p36_cpu_workers.generation;
        p36_parallel_fn fn = p36_cpu_workers.fn;
        void *ctx = p36_cpu_workers.ctx;
        uint64_t n_rows = p36_cpu_workers.n_rows;
        uint32_t n_threads = p36_cpu_workers.active_threads;
        pthread_mutex_unlock(&p36_cpu_workers.mu);

        if (fn && n_threads > 0) {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                p36_cpu_parallel_depth++;
                fn(ctx, row0, row1);
                p36_cpu_parallel_depth--;
            }
        }

        pthread_mutex_lock(&p36_cpu_workers.mu);
        p36_cpu_workers.done++;
        if (p36_cpu_workers.done == p36_cpu_workers.n_workers) {
            pthread_cond_signal(&p36_cpu_workers.done_cv);
        }
        pthread_mutex_unlock(&p36_cpu_workers.mu);
    }
}

static void p36_cpu_threads_reserve(uint32_t n_threads) {
    pthread_mutex_lock(&p36_cpu_workers_init_mu);
    if (!p36_cpu_workers.initialized) {
        pthread_mutex_init(&p36_cpu_workers.mu, NULL);
        pthread_mutex_init(&p36_cpu_workers.submit_mu, NULL);
        pthread_cond_init(&p36_cpu_workers.work_cv, NULL);
        pthread_cond_init(&p36_cpu_workers.done_cv, NULL);
        p36_cpu_workers.n_threads = 1;
        p36_cpu_workers.n_workers = 0;
        p36_cpu_workers.initialized = true;
        if (!p36_cpu_workers_atexit) {
            atexit(p36_cpu_threads_shutdown);
            p36_cpu_workers_atexit = true;
        }
    }
    if (n_threads > P36_CPU_MAX_THREADS) n_threads = P36_CPU_MAX_THREADS;
    while (p36_cpu_workers.n_threads < n_threads) {
        uint32_t tid = p36_cpu_workers.n_threads;
        if (pthread_create(&p36_cpu_workers.threads[tid], NULL, p36_cpu_worker_main,
                           (void *)(uintptr_t)tid) != 0) {
            break;
        }
        p36_cpu_workers.n_threads++;
        p36_cpu_workers.n_workers = p36_cpu_workers.n_threads - 1u;
    }
    pthread_mutex_unlock(&p36_cpu_workers_init_mu);
}

static void p36_parallel_for_rows(uint64_t n_rows,
                                  uint64_t min_parallel_rows,
                                  uint32_t requested_threads,
                                  p36_parallel_fn fn,
                                  void *ctx) {
    uint32_t n_threads;
    uint64_t rows_per_thread;
    uint64_t main_row1;
    if (!fn || n_rows == 0) return;
    if (requested_threads <= 1 || n_rows < min_parallel_rows || p36_cpu_parallel_depth > 0) {
        fn(ctx, 0, n_rows);
        return;
    }
    p36_cpu_threads_reserve(requested_threads);
    n_threads = p36_cpu_workers.n_threads;
    if (n_threads > requested_threads) n_threads = requested_threads;
    if (n_threads <= 1) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&p36_cpu_workers.submit_mu);
    pthread_mutex_lock(&p36_cpu_workers.mu);
    p36_cpu_workers.fn = fn;
    p36_cpu_workers.ctx = ctx;
    p36_cpu_workers.n_rows = n_rows;
    p36_cpu_workers.active_threads = n_threads;
    p36_cpu_workers.done = 0;
    p36_cpu_workers.generation++;
    pthread_cond_broadcast(&p36_cpu_workers.work_cv);
    rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&p36_cpu_workers.mu);

    p36_cpu_parallel_depth++;
    fn(ctx, 0, main_row1);
    p36_cpu_parallel_depth--;

    pthread_mutex_lock(&p36_cpu_workers.mu);
    while (p36_cpu_workers.done < p36_cpu_workers.n_workers) {
        pthread_cond_wait(&p36_cpu_workers.done_cv, &p36_cpu_workers.mu);
    }
    pthread_mutex_unlock(&p36_cpu_workers.mu);
    pthread_mutex_unlock(&p36_cpu_workers.submit_mu);
}

static bool p36_backend_uses_graph(p36_backend backend) {
    return backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL;
}

static bool p36_backend_supports_ssd_streaming(p36_backend backend) {
    return p36_backend_uses_graph(backend);
}


static bool p36_engine_uses_vulkan_runtime(const p36_engine *e) {
    return e && p36_backend_uses_graph(e->backend);
}

static void p36_die(const char *msg) {
    fprintf(stderr, "p36: %s\n", msg);
    exit(1);
}

static void p36_die_errno(const char *what, const char *path) {
    fprintf(stderr, "p36: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    return xcalloc(n, size);
}

static char *p36_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = xmalloc(n);
    memcpy(out, s, n);
    return out;
}

static double p36_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static bool p36_str_eq(p36_str a, p36_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static bool p36_streq(p36_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool p36_str_contains(p36_str s, const char *needle) {
    size_t n = strlen(needle);
    if (!n || n > s.len) return false;
    for (uint64_t i = 0; i <= s.len - n; i++) {
        if (!memcmp(s.ptr + i, needle, n)) return true;
    }
    return false;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static const char *p36_log_color_code(p36_log_type type) {
    switch (type) {
    case P36_LOG_PREFILL:
    case P36_LOG_TIMING:
        return "\x1b[36m";
    case P36_LOG_GENERATION:
    case P36_LOG_OK:
        return "\x1b[32m";
    case P36_LOG_KVCACHE:
        return "\x1b[33m";
    case P36_LOG_TOOL:
        return "\x1b[90m";
    case P36_LOG_WARNING:
        return "\x1b[38;5;208m";
    case P36_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool p36_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

void p36_log(FILE *fp, p36_log_type type, const char *fmt, ...) {
    va_list ap;
    const bool colorize = type != P36_LOG_DEFAULT && p36_log_is_tty(fp);
    va_start(ap, fmt);
    if (colorize) fputs(p36_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
    va_end(ap);
}

const char *p36_backend_name(p36_backend backend) {
    switch (backend) {
    case P36_BACKEND_VULKAN: return "vulkan";
    case P36_BACKEND_METAL: return "metal";
    case P36_BACKEND_CPU: return "cpu";
    default: return "unknown";
    }
}

bool p36_think_mode_enabled(p36_think_mode mode) {
    return mode != P36_THINK_NONE;
}

const char *p36_think_mode_name(p36_think_mode mode) {
    switch (mode) {
    case P36_THINK_NONE: return "none";
    case P36_THINK_HIGH: return "high";
    case P36_THINK_MAX: return "max";
    case P36_THINK_LOW: return "low";
    case P36_THINK_MEDIUM: return "medium";
    case P36_THINK_XHIGH: return "xhigh";
    default: return "unknown";
    }
}

const char *p36_qwen38_effort_instruction(p36_think_mode mode) {
    if (mode == P36_THINK_LOW)
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    if (mode == P36_THINK_MEDIUM || mode == P36_THINK_NONE) return NULL;
    return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
}

const char *p36_jinja_effort_name(p36_think_mode mode) {
    switch (mode) {
    case P36_THINK_LOW: return "low";
    case P36_THINK_MEDIUM: return "medium";
    case P36_THINK_HIGH: return "high";
    case P36_THINK_XHIGH:
    case P36_THINK_MAX: return "xhigh";
    case P36_THINK_NONE:
    default: return NULL;
    }
}

/* --- Runtime Jinja2 Chat Template Evaluator --- */
typedef struct jinja_arena_block {
    struct jinja_arena_block *next;
    size_t used;
    size_t cap;
    uint8_t data[];
} jinja_arena_block;

typedef struct {
    jinja_arena_block *head;
} jinja_arena;

static void *ja_alloc(jinja_arena *a, size_t size) {
    size = (size + 7u) & ~7u;
    if (!a->head || a->head->used + size > a->head->cap) {
        size_t cap = size > 65536u ? size : 65536u;
        jinja_arena_block *b = (jinja_arena_block *)calloc(1, sizeof(*b) + cap);
        if (!b) abort();
        b->cap = cap;
        b->next = a->head;
        a->head = b;
    }
    void *p = a->head->data + a->head->used;
    a->head->used += size;
    return p;
}

static char *ja_strndup(jinja_arena *a, const char *s, size_t n) {
    char *p = (char *)ja_alloc(a, n + 1);
    if (n && s) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *ja_strdup(jinja_arena *a, const char *s) {
    if (!s) return ja_strndup(a, "", 0);
    return ja_strndup(a, s, strlen(s));
}

static void ja_free(jinja_arena *a) {
    jinja_arena_block *b = a->head;
    while (b) {
        jinja_arena_block *next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} jbuf;

static void jbuf_append(jbuf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 256;
        while (ncap < b->len + n + 1) ncap *= 2;
        b->ptr = (char *)realloc(b->ptr, ncap);
        if (!b->ptr) abort();
        b->cap = ncap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void jbuf_puts(jbuf *b, const char *s) {
    if (s) jbuf_append(b, s, strlen(s));
}

static void jbuf_putc(jbuf *b, char c) {
    jbuf_append(b, &c, 1);
}

typedef enum {
    JVAL_UNDEFINED = 0,
    JVAL_NONE,
    JVAL_BOOL,
    JVAL_INT,
    JVAL_STR,
    JVAL_LIST,
    JVAL_MAP,
} jval_kind;

typedef struct jval jval;
typedef struct jmap jmap;
typedef struct jlist jlist;

struct jval {
    jval_kind kind;
    union {
        bool b;
        int64_t i;
        struct {
            const char *ptr;
            size_t len;
        } s;
        jlist *list;
        jmap *map;
    } u;
};

struct jlist {
    jval *items;
    size_t len;
    size_t cap;
};

typedef struct {
    const char *key;
    jval val;
} jkv;

struct jmap {
    jkv *entries;
    size_t len;
    size_t cap;
    const char *raw_json;
};

static jval jval_undef(void) { return (jval){.kind = JVAL_UNDEFINED}; }
static jval jval_none(void) { return (jval){.kind = JVAL_NONE}; }
static jval jval_bool(bool b) { return (jval){.kind = JVAL_BOOL, .u.b = b}; }
static jval jval_int(int64_t i) { return (jval){.kind = JVAL_INT, .u.i = i}; }
static jval jval_str_n(jinja_arena *a, const char *s, size_t n) {
    return (jval){.kind = JVAL_STR, .u.s = {ja_strndup(a, s, n), n}};
}
static jval jval_str(jinja_arena *a, const char *s) {
    if (!s) s = "";
    return jval_str_n(a, s, strlen(s));
}
static jval jval_list_new(jinja_arena *a, size_t cap) {
    jlist *l = (jlist *)ja_alloc(a, sizeof(*l));
    l->cap = cap ? cap : 4;
    l->items = (jval *)ja_alloc(a, l->cap * sizeof(jval));
    l->len = 0;
    return (jval){.kind = JVAL_LIST, .u.list = l};
}
static void jlist_push(jinja_arena *a, jlist *l, jval v) {
    if (l->len == l->cap) {
        size_t ncap = l->cap * 2;
        jval *nitems = (jval *)ja_alloc(a, ncap * sizeof(jval));
        memcpy(nitems, l->items, l->len * sizeof(jval));
        l->items = nitems;
        l->cap = ncap;
    }
    l->items[l->len++] = v;
}
static jval jval_map_new(jinja_arena *a, size_t cap) {
    jmap *m = (jmap *)ja_alloc(a, sizeof(*m));
    m->cap = cap ? cap : 8;
    m->entries = (jkv *)ja_alloc(a, m->cap * sizeof(jkv));
    m->len = 0;
    m->raw_json = NULL;
    return (jval){.kind = JVAL_MAP, .u.map = m};
}
static void jmap_set(jinja_arena *a, jmap *m, const char *key, jval v) {
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) {
            m->entries[i].val = v;
            return;
        }
    }
    if (m->len == m->cap) {
        size_t ncap = m->cap * 2;
        jkv *nentries = (jkv *)ja_alloc(a, ncap * sizeof(jkv));
        memcpy(nentries, m->entries, m->len * sizeof(jkv));
        m->entries = nentries;
        m->cap = ncap;
    }
    m->entries[m->len].key = ja_strdup(a, key);
    m->entries[m->len].val = v;
    m->len++;
}
static jval jmap_get(const jmap *m, const char *key) {
    if (!m || !key) return jval_undef();
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) return m->entries[i].val;
    }
    return jval_undef();
}
static bool jmap_has(const jmap *m, const char *key) {
    if (!m || !key) return false;
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) {
            return m->entries[i].val.kind != JVAL_UNDEFINED;
        }
    }
    return false;
}

static bool jval_truthy(jval v) {
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return false;
    case JVAL_BOOL: return v.u.b;
    case JVAL_INT: return v.u.i != 0;
    case JVAL_STR: return v.u.s.len > 0;
    case JVAL_LIST: return v.u.list && v.u.list->len > 0;
    case JVAL_MAP: return v.u.map && (v.u.map->len > 0 || (v.u.map->raw_json && v.u.map->raw_json[0]));
    }
    return false;
}

static bool jval_eq(jval a, jval b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return true;
    case JVAL_BOOL: return a.u.b == b.u.b;
    case JVAL_INT: return a.u.i == b.u.i;
    case JVAL_STR:
        return a.u.s.len == b.u.s.len &&
               (!a.u.s.len || !memcmp(a.u.s.ptr, b.u.s.ptr, a.u.s.len));
    case JVAL_LIST:
        if (a.u.list->len != b.u.list->len) return false;
        for (size_t i = 0; i < a.u.list->len; i++)
            if (!jval_eq(a.u.list->items[i], b.u.list->items[i])) return false;
        return true;
    case JVAL_MAP:
        return a.u.map == b.u.map;
    }
    return false;
}

static jval jval_to_str(jinja_arena *a, jval v) {
    char num[64];
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return jval_str(a, "");
    case JVAL_BOOL: return jval_str(a, v.u.b ? "True" : "False");
    case JVAL_INT:
        snprintf(num, sizeof(num), "%lld", (long long)v.u.i);
        return jval_str(a, num);
    case JVAL_STR: return v;
    default: return jval_str(a, "");
    }
}

static void jbuf_json_escape(jbuf *b, const char *s, size_t len) {
    jbuf_putc(b, '"');
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': jbuf_puts(b, "\\\""); break;
        case '\\': jbuf_puts(b, "\\\\"); break;
        case '\b': jbuf_puts(b, "\\b"); break;
        case '\f': jbuf_puts(b, "\\f"); break;
        case '\n': jbuf_puts(b, "\\n"); break;
        case '\r': jbuf_puts(b, "\\r"); break;
        case '\t': jbuf_puts(b, "\\t"); break;
        default:
            if (c < 0x20) {
                char hex[8];
                snprintf(hex, sizeof(hex), "\\u%04x", c);
                jbuf_puts(b, hex);
            } else {
                jbuf_putc(b, (char)c);
            }
        }
    }
    jbuf_putc(b, '"');
}

static void jval_write_json(jbuf *b, jval v) {
    char num[64];
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE:
        jbuf_puts(b, "null");
        break;
    case JVAL_BOOL:
        jbuf_puts(b, v.u.b ? "true" : "false");
        break;
    case JVAL_INT:
        snprintf(num, sizeof(num), "%lld", (long long)v.u.i);
        jbuf_puts(b, num);
        break;
    case JVAL_STR:
        jbuf_json_escape(b, v.u.s.ptr, v.u.s.len);
        break;
    case JVAL_LIST:
        jbuf_putc(b, '[');
        for (size_t i = 0; i < v.u.list->len; i++) {
            if (i) jbuf_puts(b, ", ");
            jval_write_json(b, v.u.list->items[i]);
        }
        jbuf_putc(b, ']');
        break;
    case JVAL_MAP:
        if (v.u.map->raw_json) {
            jbuf_puts(b, v.u.map->raw_json);
            break;
        }
        jbuf_putc(b, '{');
        for (size_t i = 0; i < v.u.map->len; i++) {
            if (i) jbuf_puts(b, ", ");
            jbuf_json_escape(b, v.u.map->entries[i].key, strlen(v.u.map->entries[i].key));
            jbuf_puts(b, ": ");
            jval_write_json(b, v.u.map->entries[i].val);
        }
        jbuf_putc(b, '}');
        break;
    }
}

typedef enum {
    JTOK_TEXT,
    JTOK_EXPR,
    JTOK_STMT,
} jtok_kind;

typedef struct {
    jtok_kind kind;
    const char *ptr;
    size_t len;
} jtok;

typedef enum {
    JSTMT_TEXT,
    JSTMT_EXPR,
    JSTMT_SET,
    JSTMT_IF,
    JSTMT_FOR,
    JSTMT_MACRO,
    JSTMT_BREAK,
    JSTMT_CONTINUE,
} jstmt_kind;

typedef struct jstmt jstmt;

typedef struct {
    jstmt *items;
    size_t len;
    size_t cap;
} jstmt_list;

typedef struct {
    const char *cond_ptr;
    size_t cond_len;
    jstmt_list body;
} jif_branch;

typedef struct {
    const char *name;
    const char *def_ptr;
    size_t def_len;
} jmacro_param;

struct jstmt {
    jstmt_kind kind;
    union {
        struct {
            const char *ptr;
            size_t len;
        } text;
        struct {
            const char *ptr;
            size_t len;
        } expr;
        struct {
            const char *var;
            const char *attr;
            const char *expr_ptr;
            size_t expr_len;
        } set;
        struct {
            jif_branch *branches;
            size_t branch_count;
        } if_stmt;
        struct {
            const char *var1;
            const char *var2;
            const char *iter_ptr;
            size_t iter_len;
            const char *cond_ptr;
            size_t cond_len;
            jstmt_list body;
            jstmt_list else_body;
        } for_stmt;
        struct {
            const char *name;
            jmacro_param *params;
            size_t param_count;
            jstmt_list body;
        } macro;
    } u;
};

static bool j_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

static const char *find_tag_end(const char *p, const char *end, char close_c) {
    char quote = 0;
    while (p + 1 < end) {
        char c = *p;
        if (quote) {
            if (c == '\\' && p + 1 < end) {
                p += 2;
                continue;
            }
            if (c == quote) quote = 0;
            p++;
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            p++;
            continue;
        }
        if (c == close_c && p[1] == '}') return p;
        p++;
    }
    return NULL;
}

static bool j_lex(jinja_arena *a, const char *src, jtok **out_toks, size_t *out_count,
                  char *err, size_t errlen) {
    size_t cap = 64, len = 0;
    jtok *toks = (jtok *)ja_alloc(a, cap * sizeof(jtok));
    const char *end = src + strlen(src);
    const char *p = src;

    while (p < end) {
        const char *tag = p;
        while (tag + 1 < end) {
            if (tag[0] == '{' && (tag[1] == '%' || tag[1] == '{' || tag[1] == '#')) break;
            tag++;
        }
        if (tag + 1 >= end) tag = end;

        size_t tlen = (size_t)(tag - p);
        if (tag < end) {
            char kind_c = tag[1];
            bool strip_left = (tag + 2 < end && tag[2] == '-');
            if (strip_left) {
                while (tlen > 0 && j_is_space(p[tlen - 1])) tlen--;
            } else if (kind_c == '%' || kind_c == '#') {
                size_t line_start = tlen;
                while (line_start > 0 && p[line_start - 1] != '\n') line_start--;
                bool all_ws = true;
                for (size_t i = line_start; i < tlen; i++) {
                    if (p[i] != ' ' && p[i] != '\t') {
                        all_ws = false;
                        break;
                    }
                }
                if (all_ws) tlen = line_start;
            }
        }
        if (tlen > 0) {
            if (len == cap) {
                size_t ncap = cap * 2;
                jtok *nt = (jtok *)ja_alloc(a, ncap * sizeof(jtok));
                memcpy(nt, toks, len * sizeof(jtok));
                toks = nt;
                cap = ncap;
            }
            toks[len++] = (jtok){JTOK_TEXT, p, tlen};
        }
        if (tag >= end) break;

        char kind_c = tag[1];
        bool strip_left = (tag + 2 < end && tag[2] == '-');
        const char *body_start = tag + 2 + (strip_left ? 1 : 0);
        const char *close = NULL;
        if (kind_c == '#') {
            const char *q = body_start;
            while (q + 1 < end) {
                if (q[0] == '#' && q[1] == '}') {
                    close = q;
                    break;
                }
                q++;
            }
        } else {
            close = find_tag_end(body_start, end, kind_c == '{' ? '}' : '%');
        }
        if (!close) {
            if (err && errlen) snprintf(err, errlen, "unclosed Jinja tag");
            return false;
        }
        bool strip_right = (close > body_start && close[-1] == '-');
        const char *body_end = close - (strip_right ? 1 : 0);
        while (body_start < body_end && j_is_space(*body_start)) body_start++;
        while (body_end > body_start && j_is_space(body_end[-1])) body_end--;

        if (kind_c != '#') {
            if (len == cap) {
                size_t ncap = cap * 2;
                jtok *nt = (jtok *)ja_alloc(a, ncap * sizeof(jtok));
                memcpy(nt, toks, len * sizeof(jtok));
                toks = nt;
                cap = ncap;
            }
            toks[len++] = (jtok){
                kind_c == '{' ? JTOK_EXPR : JTOK_STMT,
                body_start,
                (size_t)(body_end - body_start),
            };
        }

        p = close + 2;
        if (strip_right) {
            while (p < end && j_is_space(*p)) p++;
        } else if (kind_c == '%' || kind_c == '#') {
            if (p < end && *p == '\r') p++;
            if (p < end && *p == '\n') p++;
        }
    }

    *out_toks = toks;
    *out_count = len;
    return true;
}

static bool j_is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static bool j_is_ident_char(char c) {
    return j_is_ident_start(c) || (c >= '0' && c <= '9');
}

