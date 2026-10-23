static bool anthropic_sse_stream_update(int fd, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == ANTH_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        const char *tool = r->has_tools ?
            find_any_tool_start(raw + st->emit_pos) : NULL;
        const bool tool_before_close = tool && (!close || tool < close);
        const bool complete_tool =
            tool_before_close && complete_tool_text(tool);
        size_t limit;
        if (tool_before_close) {
            limit = trim_tool_separator_ws(raw, st->emit_pos,
                                           (size_t)(tool - raw));
        } else if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_THINKING)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_THINKING,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_thinking = true;
            st->emit_pos = limit;
        }

        if (tool_before_close) {
            if (complete_tool) {
                if (!anthropic_sse_close_block_live(fd, id, st)) return false;
                st->emit_pos = (size_t)(tool - raw);
                st->mode = ANTH_STREAM_SUPPRESS;
            }
            return true;
        }

        if (close || final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            if (close) {
                st->emit_pos = (size_t)(close - raw) + strlen("</think>");
                st->mode = ANTH_STREAM_TEXT;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
                return true;
            }
        } else {
            return true;
        }
    }

    if (st->mode == ANTH_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_text = true;
            st->emit_pos = limit;
        }

        if (tool) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->emit_pos = (size_t)(tool - raw);
            st->mode = ANTH_STREAM_SUPPRESS;
        } else if (final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->mode = ANTH_STREAM_SUPPRESS;
        }
    }
    return true;
}

static bool anthropic_sse_tool_blocks_live(int fd, const request *r, const char *id,
                                           anthropic_stream *st,
                                           const tool_calls *calls) {
    (void)r;
    if (!calls) return true;

    buf b = {0};
    for (int i = 0; i < calls->len; i++, st->next_index++) {
        const tool_call *tc = &calls->v[i];
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id, i);
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"tool_use\",\"id\":",
                   st->next_index);
        json_escape(&b, tc->id && tc->id[0] ? tc->id : idbuf);
        buf_puts(&b, ",\"name\":");
        json_escape(&b, tc->name ? tc->name : "");
        buf_puts(&b, ",\"input\":{}}}");
        bool ok = sse_event(fd, "content_block_start", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                   st->next_index);
        append_json_object_string(&b, tc->arguments);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
        if (!ok) return false;
    }
    return true;
}

static bool anthropic_sse_stop_live(int fd, const char *finish,
                                    int completion_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    buf_printf(&b, "%d}}", completion_tokens);
    bool ok = sse_event(fd, "message_delta", b.ptr);
    buf_free(&b);
    if (ok) ok = sse_event(fd, "message_stop", "{\"type\":\"message_stop\"}");
    return ok;
}

static bool anthropic_sse_finish_live(int fd, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const tool_calls *calls,
                                      const char *finish, int completion_tokens) {
    if (!anthropic_sse_stream_update(fd, r, id, st, raw, raw_len, true)) return false;

    if (st->sent_thinking && !st->sent_text && (!calls || calls->len == 0)) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
    }

    if (!anthropic_sse_tool_blocks_live(fd, r, id, st, calls)) return false;
    return anthropic_sse_stop_live(fd, finish, completion_tokens);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void server_log(p36_log_type type, const char *fmt, ...) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char ts[16];
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tm);

    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);

    fprintf(stderr, "%s ", ts);
    if (n < 0) {
        p36_log(stderr, type, "%s", fmt);
    } else {
        char *line = xmalloc((size_t)n + 1);
        vsnprintf(line, (size_t)n + 1, fmt, ap);
        p36_log(stderr, type, "%s", line);
        free(line);
    }
    va_end(ap);
    fputc('\n', stderr);
}

typedef struct job job;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence.  The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} kv_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
    bool system_checkpoint;   /* --kv-system-checkpoint: store the system+tools prefix once */
} kv_cache_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    kv_cache_options opt;
    char ident[41];        /* model/tokenizer/state fingerprint folded into every key; empty = legacy */
    int continued_last_store_tokens;
    kv_entry *entry;
    int len;
    int cap;
} kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *qwen_tool;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    bool empty_think;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

#define SERVER_IMAGE_CACHE_ENTRIES 32
#define SERVER_IMAGE_CACHE_BYTES (32u * 1024u * 1024u)

typedef struct {
    uint8_t *encoded;
    size_t encoded_len;
    p36_vision_embedding embedding;
    size_t data_bytes;
    uint64_t used;
} server_image_cache_entry;

typedef struct {
    server_image_cache_entry entries[SERVER_IMAGE_CACHE_ENTRIES];
    size_t bytes;
    uint64_t clock;
} server_image_cache;

static void server_image_cache_remove(server_image_cache *cache,
                                      server_image_cache_entry *entry) {
    cache->bytes -= entry->encoded_len + entry->data_bytes;
    free(entry->encoded);
    p36_vision_embedding_free(&entry->embedding);
    memset(entry, 0, sizeof(*entry));
}

static void server_image_cache_clear(server_image_cache *cache) {
    for (size_t i = 0; i < SERVER_IMAGE_CACHE_ENTRIES; i++)
        server_image_cache_remove(cache, &cache->entries[i]);
    cache->clock = 0;
}

static bool server_image_cache_get(server_image_cache *cache,
                                   const server_image_input *input,
                                   p36_vision_embedding *out) {
    for (size_t i = 0; i < SERVER_IMAGE_CACHE_ENTRIES; i++) {
        server_image_cache_entry *entry = &cache->entries[i];
        if (!entry->encoded || entry->encoded_len != input->encoded_len ||
            memcmp(entry->encoded, input->encoded, input->encoded_len)) continue;
        float *data = malloc(entry->data_bytes);
        if (!data) return false;
        memcpy(data, entry->embedding.data, entry->data_bytes);
        *out = entry->embedding;
        out->data = data;
        entry->used = ++cache->clock;
        return true;
    }
    return false;
}

/* Cache only successful encodes. Exact byte keys avoid trusting a client hash;
 * callers own a copy, so each request owns its image data. */
static void server_image_cache_put(server_image_cache *cache,
                                   const server_image_input *input,
                                   const p36_vision_embedding *embedding,
                                   uint32_t dim, size_t budget) {
    if (!embedding->data || !dim || !embedding->token_count ||
        !input->encoded_len ||
        embedding->token_count > budget / sizeof(float) / dim) return;
    size_t bytes = (size_t)embedding->token_count * dim * sizeof(float);
    if (input->encoded_len > budget - bytes) return;
    size_t total = input->encoded_len + bytes;
    server_image_cache_entry *dest;
    for (;;) {
        dest = &cache->entries[0];
        for (size_t i = 1; i < SERVER_IMAGE_CACHE_ENTRIES; i++) {
            if (cache->entries[i].used < dest->used) dest = &cache->entries[i];
        }
        if (!dest->encoded && cache->bytes <= budget - total) break;
        if (!dest->encoded) {
            for (size_t i = 0; i < SERVER_IMAGE_CACHE_ENTRIES; i++) {
                server_image_cache_entry *entry = &cache->entries[i];
                if (entry->encoded && (!dest->encoded || entry->used < dest->used))
                    dest = entry;
            }
        }
        server_image_cache_remove(cache, dest);
    }
    uint8_t *encoded = malloc(input->encoded_len);
    float *data = malloc((size_t)bytes);
    if (!encoded || !data) {
        free(encoded);
        free(data);
        return;
    }
    memcpy(encoded, input->encoded, input->encoded_len);
    memcpy(data, embedding->data, (size_t)bytes);
    *dest = (server_image_cache_entry) {
        .encoded = encoded, .encoded_len = input->encoded_len,
        .embedding = *embedding, .data_bytes = (size_t)bytes,
        .used = ++cache->clock,
    };
    dest->embedding.data = data;
    cache->bytes += total;
}


struct server {
    server_image_cache image_cache;
    p36_engine *engine;
    /* slots[0] remains the legacy session when batching is disabled. */
    p36_session *session;
    server_slot *slots;
    int slot_count;
    bool batched_mode;
    int mixed_prefill_quantum;
    pthread_t *slot_threads;
    pthread_t decode_thread;
    int default_tokens;
    float default_presence_penalty;
    float default_frequency_penalty;
    bool default_presence_penalty_set;
    bool default_frequency_penalty_set;
    float default_temperature;
    int default_top_k;
    float default_top_p;
    float default_min_p;
    float default_repeat_penalty;
    bool default_temperature_set;
    bool default_top_k_set;
    bool default_top_p_set;
    bool default_min_p_set;
    bool default_repeat_penalty_set;
    kv_disk_cache kv;
    tool_memory tool_mem;
    bool disable_exact_tool_replay;
    pthread_mutex_t tool_mu;
    pthread_mutex_t kv_mu;
    pthread_mutex_t inference_mu;
    pthread_mutex_t model_mu;
    pthread_cond_t model_cv;
    bool model_busy;
    bool model_stopping;
    int decode_pending;
    int active_generations;
    int last_prefill_slot;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
    /* --models-dir catalog.  One model is resident at a time; a request that
     * names another catalog entry swaps it in under swap_lock (write side),
     * while every other request holds the read side from parse to completion
     * because parsing already tokenizes with s->engine. */
    server_model *models;
    int model_count;
    int model_current;
    int model_initial;
    pthread_rwlock_t swap_lock;
    p36_engine_options engine_opt;
    int ctx_size;
    char *kv_root;
    uint64_t kv_budget_mb;
    bool kv_reject_different_quant;
    kv_cache_options kv_opt;
};

static void request_apply_default_penalties(request *r, server *s) {
    if (!s) return;
    if (s->default_presence_penalty_set) r->presence_penalty = s->default_presence_penalty;
    if (s->default_frequency_penalty_set) r->frequency_penalty = s->default_frequency_penalty;
    /* The *_set flags also stop request_apply_model_sampling_defaults from
     * replacing these with the model's defaults; client JSON parsed later wins. */
    if (s->default_temperature_set) { r->temperature = s->default_temperature; r->temperature_set = true; }
    if (s->default_top_k_set) { r->top_k = s->default_top_k; r->top_k_set = true; }
    if (s->default_top_p_set) { r->top_p = s->default_top_p; r->top_p_set = true; }
    if (s->default_min_p_set) { r->min_p = s->default_min_p; r->min_p_set = true; }
    if (s->default_repeat_penalty_set) r->repeat_penalty = s->default_repeat_penalty;
}

static void server_inference_lock(server *s) {
    pthread_mutex_lock(&s->inference_mu);
}

static void server_inference_unlock(server *s) {
    pthread_mutex_unlock(&s->inference_mu);
}

static bool server_encode_image(server *s, const server_image_input *input,
                                p36_vision_embedding *out,
                                char *err, size_t errlen) {
    if (server_image_cache_get(&s->image_cache, input, out)) {
        server_log(P36_LOG_KVCACHE, "p36-server: vision embedding cache hit");
        return true;
    }
    if (!p36_engine_vision_encode_memory(s->engine, input->encoded,
                                         input->encoded_len, out, err, errlen))
        return false;
    server_image_cache_put(&s->image_cache, input, out,
                            (uint32_t)p36_engine_embd_dim(s->engine),
                            SERVER_IMAGE_CACHE_BYTES);
    server_log(P36_LOG_KVCACHE, "p36-server: vision embedding encoded; cache=%zu bytes",
               s->image_cache.bytes);
    return true;
}

struct server_slot {
    server *srv;
    int id;
    p36_session *session;
    char **pending_ids;
    size_t pending_count;
    int pending_pos;
    api_style pending_api;
    char *pending_tools;
    int continued_last_store_tokens;
    job *assigned;
    bool busy;
    bool prefill_waiting;
    bool decode_pending;
    bool decode_in_flight;
    bool decode_done;
    int decode_token;
    int decode_rc;
    char decode_err[160];
};

static void slot_pending_clear(server_slot *slot) {
    for (size_t i = 0; i < slot->pending_count; i++) free(slot->pending_ids[i]);
    free(slot->pending_ids);
    free(slot->pending_tools);
    slot->pending_ids = NULL;
    slot->pending_tools = NULL;
    slot->pending_count = 0;
    slot->pending_pos = 0;
}

static bool slot_pending_matches(const server_slot *slot, const request *r) {
    if (!slot->pending_count || slot->pending_count != r->continuation_count ||
        slot->pending_api != r->api || slot->pending_pos != p36_session_pos(slot->session) ||
        strcmp(slot->pending_tools, r->tool_schema_key ? r->tool_schema_key : "")) return false;
    for (size_t i = 0; i < r->continuation_count; i++) {
        bool found = false;
        for (size_t k = 0; k < i; k++)
            if (!strcmp(r->continuation_ids[i], r->continuation_ids[k])) return false;
        for (size_t k = 0; k < slot->pending_count; k++)
            if (!strcmp(r->continuation_ids[i], slot->pending_ids[k])) found = true;
        if (!found) return false;
    }
    return true;
}

/* Jobs are stack-owned by the client thread.  The worker signals completion
 * after the response has been written, so request data and the socket remain
 * valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

/* =========================================================================
 * Tool Call Text Memory.
 * =========================================================================
 *
 * The model speaks native Qwen tool-call text, while OpenAI and Anthropic clients round-trip tool
 * calls as JSON.  Re-rendering that JSON is not always the same byte sequence:
 * clients may preserve, sort, or rebuild object keys differently.  Tool call
 * ids are the bridge between both worlds.  For every generated tool call we
 * remember the exact <tool_call> block sampled by the model under a random id.  When
 * the client later sends the same id back in conversation history, we replay
 * the sampled tool text verbatim and keep the KV cache aligned with the live model
 * state.
 */

#define P36_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define P36_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)

static int tool_memory_max_entries(const tool_memory *m) {
    return m && m->max_entries > 0 ? m->max_entries : P36_TOOL_MEMORY_DEFAULT_MAX_IDS;
}

static size_t tool_memory_max_bytes(const tool_memory *m) {
    return m && m->max_bytes > 0 ? m->max_bytes : P36_TOOL_MEMORY_MAX_BYTES;
}

static void tool_memory_init_locked(tool_memory *m) {
    if (m->by_id && m->by_block) return;
    m->by_id = raxNew();
    m->by_block = raxNew();
    if (!m->by_id || !m->by_block) die("out of memory");
}

