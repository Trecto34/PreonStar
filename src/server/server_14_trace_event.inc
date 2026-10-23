static void trace_event(server *s, uint64_t id, const char *fmt, ...) {
    if (!s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fputs("\n\n--- trace: ", s->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s->trace, fmt, ap);
    va_end(ap);
    fputs(" ---\n\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_finish(
        server *s,
        uint64_t id,
        const request *r,
        const char *final_finish,
        int completion,
        bool saw_tool_start,
        bool saw_tool_end,
        const char *parsed_content,
        const char *parsed_reasoning,
        const tool_calls *parsed_calls,
        double elapsed) {
    if (!s->trace || !id) return;

    pthread_mutex_lock(&s->trace_mu);
    fprintf(s->trace,
            "\n\n--- parsed message ---\nfinish: %s\ngenerated_tokens: %d\nqwen_tool_start: %d\nqwen_tool_end: %d\nelapsed_sec: %.3f\n",
            final_finish,
            completion,
            saw_tool_start ? 1 : 0,
            saw_tool_end ? 1 : 0,
            elapsed);
    if (r->kind == REQ_CHAT) {
        if (parsed_reasoning && parsed_reasoning[0]) {
            fputs("\nreasoning:\n", s->trace);
            fputs(parsed_reasoning, s->trace);
            fputc('\n', s->trace);
        }
        if (parsed_content && parsed_content[0]) {
            fputs("\ncontent:\n", s->trace);
            fputs(parsed_content, s->trace);
            fputc('\n', s->trace);
        }
        for (int i = 0; i < parsed_calls->len; i++) {
            const tool_call *tc = &parsed_calls->v[i];
            fprintf(s->trace, "\ntool_call[%d]:\nid: %s\nname: %s\narguments:\n%s\n",
                    i,
                    tc->id ? tc->id : "",
                    tc->name ? tc->name : "",
                    tc->arguments ? tc->arguments : "");
        }
    }
    fprintf(s->trace, "\n===== end request %llu =====\n", (unsigned long long)id);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

typedef struct {
    server *srv;
    server_slot *slot;
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    bool has_tools;
    double t0;
    double last_t;
    int last_current;
    bool seen;
} server_prefill_progress;

static void request_ctx_span(char *buf, size_t len, int cached, int prompt) {
    int suffix = prompt - cached;
    if (suffix < 0) suffix = 0;
    snprintf(buf, len, "%d..%d:%d", cached, prompt, suffix);
}

static void log_flags(char *buf, size_t len, bool tools, bool thinking,
                      bool qwen_tool_start, bool qwen_tool_end) {
    size_t used = 0;
    buf[0] = '\0';
#define ADD_FLAG(name) do { \
    int n = snprintf(buf + used, used < len ? len - used : 0, "%s%s", used ? " " : "", name); \
    if (n > 0) used += (size_t)n; \
} while (0)
    if (tools) ADD_FLAG("TOOLS");
    if (thinking) ADD_FLAG("THINKING");
    if (qwen_tool_start) ADD_FLAG("QWEN_TOOL_START");
    if (qwen_tool_end) ADD_FLAG("QWEN_TOOL_END");
#undef ADD_FLAG
}

static void log_decode_progress(req_kind kind, const char *ctx, int completion,
                                bool tools, bool thinking,
                                bool qwen_tool_start, bool qwen_tool_end,
                                double decode_t0,
                                double *last_t, int *last_completion) {
    const double now = now_sec();
    const double elapsed = now - decode_t0;
    const double interval_s = now - *last_t;
    const int interval_tokens = completion - *last_completion;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)completion / elapsed : 0.0;
    char flags[80];
    log_flags(flags, sizeof(flags), tools, thinking, qwen_tool_start, qwen_tool_end);
    server_log(P36_LOG_GENERATION,
               "p36-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs",
               kind == REQ_CHAT ? "chat" : "completion",
               ctx,
               completion,
               flags[0] ? " " : "",
               flags,
               chunk_tps,
               avg_tps,
               elapsed);
    *last_t = now;
    *last_completion = completion;
}

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

static bool thinking_tail_ends_with(const thinking_state *st, const char *s) {
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}

static void thinking_state_feed(thinking_state *st, const char *p, size_t len) {
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (thinking_tail_ends_with(st, "<think>")) st->inside = true;
        else if (thinking_tail_ends_with(st, "</think>")) st->inside = false;
    }
}

static thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        thinking_state_feed(&st, r->prompt_text, strlen(r->prompt_text));
    } else if (r && p36_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}

static bool complete_tool_call_inside_thinking(const char *text, size_t len,
                                               size_t *scan_from) {
    if (!text || !scan_from) return false;
    if (*scan_from > len) *scan_from = len;
    const char *start = find_any_tool_start(text + *scan_from);
    if (!start) {
        const size_t hold = 80;
        *scan_from = len > hold ? len - hold : 0;
        return false;
    }
    *scan_from = (size_t)(start - text);
    return complete_tool_text(start);
}

static int server_eval_token(server *s, server_slot *slot, int token,
                             char *err, size_t errlen);

static bool should_canonicalize_thinking_checkpoint(const request *r,
                                                    const thinking_state *thinking,
                                                    const char *finish) {
    if (!r || r->kind != REQ_CHAT) return false;
    if (!r->kat_coder && !p36_think_mode_enabled(r->think_mode)) {
        /* When history keeps the empty think block, the live graph already
         * matches what the next turn will render.  Canonicalizing would strip
         * that block and re-prefill the whole context for nothing. */
        if (r->preserve_thinking) return false;
        return !finish || (!strcmp(finish, "stop") &&
                           (!thinking || !thinking->inside));
    }
    if (r->has_tools && !r->kat_coder) return false;
    if (r->prompt_preserves_reasoning) return false;
    if (!p36_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length") ||
                   !strcmp(finish, "action_budget"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}

static void log_tool_calls_summary(const char *ctx, const tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
    }
    server_log(P36_LOG_TOOL,
               "p36-server: tool calls ctx=%s n=%d names=[%s]",
               ctx,
               calls->len,
               names.ptr ? names.ptr : "");
    buf_free(&names);
}

static int server_next_prefill_slot_locked(const server *s) {
    if (!s || s->slot_count <= 0) return -1;
    for (int n = 1; n <= s->slot_count; n++) {
        int id = (s->last_prefill_slot + n) % s->slot_count;
        if (s->slots[id].prefill_waiting) return id;
    }
    return -1;
}

static bool server_prefill_enter(server *s, server_slot *slot) {
    if (!s || !slot) return false;
    if (!s->batched_mode) {
        pthread_mutex_lock(&s->inference_mu);
        return true;
    }
    pthread_mutex_lock(&s->model_mu);
    slot->prefill_waiting = true;
    pthread_cond_broadcast(&s->model_cv);
    while (!g_stop_requested &&
           (s->model_busy || s->decode_pending > 0 ||
            server_next_prefill_slot_locked(s) != slot->id)) {
        pthread_cond_wait(&s->model_cv, &s->model_mu);
    }
    if (g_stop_requested) {
        slot->prefill_waiting = false;
        pthread_mutex_unlock(&s->model_mu);
        return false;
    }
    slot->prefill_waiting = false;
    s->last_prefill_slot = slot->id;
    s->model_busy = true;
    pthread_mutex_unlock(&s->model_mu);
    pthread_mutex_lock(&s->inference_mu);
    return true;
}

static void server_prefill_leave(server *s) {
    pthread_mutex_unlock(&s->inference_mu);
    if (!s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    s->model_busy = false;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static int server_prefill_quantum_default(bool generation_active) {
    int quantum = generation_active ? 128 : 2048;
    const char *env = getenv(generation_active
            ? "P36_SERVER_MIXED_PREFILL_QUANTUM"
            : "P36_SERVER_PREFILL_QUANTUM");
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && *end == '\0' && v > 0 && v <= INT_MAX) quantum = (int)v;
    }
    return quantum;
}

static int server_prefill_quantum_for(const server *s, bool generation_active) {
    if (generation_active && s && s->mixed_prefill_quantum > 0)
        return s->mixed_prefill_quantum;
    return server_prefill_quantum_default(generation_active);
}

static int server_prefill_quantum(server *s) {
    pthread_mutex_lock(&s->model_mu);
    bool generation_active = s->active_generations > 0;
    pthread_mutex_unlock(&s->model_mu);
    return server_prefill_quantum_for(s, generation_active);
}

static int server_session_sync(server *s, server_slot *slot,
                               const p36_tokens *prompt,
                               char *err, size_t errlen) {
    if (!s || !slot || !prompt) return 1;
    if (!s->batched_mode) {
        if (!server_prefill_enter(s, slot)) return P36_SESSION_SYNC_INTERRUPTED;
        int rc = p36_session_sync(slot->session, prompt, err, errlen);
        server_prefill_leave(s);
        return rc;
    }

    pthread_mutex_lock(&s->inference_mu);
    int live = p36_session_pos(slot->session);
    int common = p36_session_common_prefix(slot->session, prompt);
    pthread_mutex_unlock(&s->inference_mu);
    int done = common == live && prompt->len >= live ? live : 0;
    bool called = false;
    while (!g_stop_requested && (!called || done < prompt->len)) {
        int target = done + server_prefill_quantum(s);
        if (target > prompt->len || target < done) target = prompt->len;
        p36_tokens prefix = *prompt;
        prefix.len = target;
        if (!server_prefill_enter(s, slot)) return P36_SESSION_SYNC_INTERRUPTED;
        int rc = p36_session_sync(slot->session, &prefix, err, errlen);
        if (rc == 0) done = p36_session_pos(slot->session);
        server_prefill_leave(s);
        called = true;
        if (rc != 0) return rc;
        if (done >= prompt->len) return 0;
        if (done < target) {
            if (err && errlen) snprintf(err, errlen, "prefill made no progress");
            return 1;
        }
    }
    return g_stop_requested ? P36_SESSION_SYNC_INTERRUPTED : 0;
}

/* Yield between text chunks, but never cut a visual token block in half. */
static int server_session_sync_multimodal(server *s, server_slot *slot,
                                          const p36_tokens *prompt,
                                          const p36_vision_span *images, size_t count,
                                          char *err, size_t errlen) {
    if (!count) return server_session_sync(s, slot, prompt, err, errlen);
    int live = p36_session_pos(slot->session);
    int common = p36_session_common_prefix(slot->session, prompt);
    int done = common == live && prompt->len >= live &&
               p36_session_vision_prefix_matches(slot->session, images, count) ? live : 0;
    bool called = false;
    while (!g_stop_requested && (!called || done < prompt->len)) {
        int quantum = s->batched_mode ? server_prefill_quantum(s) : prompt->len;
        int target = done > prompt->len - quantum ? prompt->len : done + quantum;
        for (size_t i = 0; i < count; i++) {
            int start = (int)images[i].token_start - 1;
            int end = (int)(images[i].token_start + images[i].embedding.token_count + 1);
            if (target > start && target < end) target = end;
        }
        size_t n = 0;
        while (n < count && (uint64_t)images[n].token_start +
               images[n].embedding.token_count < (uint64_t)target) n++;
        p36_tokens prefix = *prompt;
        prefix.len = target;
        if (!server_prefill_enter(s, slot)) return P36_SESSION_SYNC_INTERRUPTED;
        int rc = n ? p36_session_sync_vision(slot->session, &prefix, images, n, err, errlen) :
                     p36_session_sync(slot->session, &prefix, err, errlen);
        if (!rc) done = p36_session_pos(slot->session);
        server_prefill_leave(s);
        called = true;
        if (rc) return rc;
        if (done < target) {
            if (err && errlen) snprintf(err, errlen, "multimodal prefill made no progress");
            return 1;
        }
    }
    return g_stop_requested ? P36_SESSION_SYNC_INTERRUPTED : 0;
}

static void server_generation_enter(server *s) {
    if (!s || !s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    s->active_generations++;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static void server_generation_leave(server *s) {
    if (!s || !s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    if (s->active_generations > 0) s->active_generations--;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static int server_eval_token(server *s, server_slot *slot, int token,
                             char *err, size_t errlen) {
    if (!s || !slot) return 1;
    if (!s->batched_mode) {
        pthread_mutex_lock(&s->inference_mu);
        int rc = p36_session_eval(slot->session, token, err, errlen);
        pthread_mutex_unlock(&s->inference_mu);
        return rc;
    }
    pthread_mutex_lock(&s->model_mu);
    if (slot->decode_pending || slot->decode_in_flight) {
        pthread_mutex_unlock(&s->model_mu);
        if (err && errlen) snprintf(err, errlen, "session already has a decode in flight");
        return 1;
    }
    slot->decode_token = token;
    slot->decode_rc = 1;
    slot->decode_err[0] = '\0';
    slot->decode_done = false;
    slot->decode_pending = true;
    s->decode_pending++;
    pthread_cond_broadcast(&s->model_cv);
    while (!slot->decode_done && !g_stop_requested) {
        pthread_cond_wait(&s->model_cv, &s->model_mu);
    }
    int rc = slot->decode_done ? slot->decode_rc : 1;
    if (rc != 0 && err && errlen) {
        snprintf(err, errlen, "%s",
                 slot->decode_err[0] ? slot->decode_err : "decode interrupted");
    }
    slot->decode_done = false;
    pthread_mutex_unlock(&s->model_mu);
    return rc;
}

static long server_decode_coalesce_us(void) {
    long us = 2000;
    const char *env = getenv("P36_SERVER_DECODE_COALESCE_US");
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && *end == '\0' && v >= 0 && v <= 100000) us = v;
    }
    return us;
}

static void server_cancel_decode_waiters_locked(server *s) {
    if (!s) return;
    for (int i = 0; i < s->slot_count; i++) {
        server_slot *slot = &s->slots[i];
        if (!slot->decode_pending) continue;
        slot->decode_pending = false;
        slot->decode_done = true;
        slot->decode_rc = 1;
        snprintf(slot->decode_err, sizeof(slot->decode_err),
                 "decode cancelled");
        if (s->decode_pending > 0) s->decode_pending--;
    }
    pthread_cond_broadcast(&s->model_cv);
}

static void timespec_add_us(struct timespec *ts, long us) {
    if (!ts || us <= 0) return;
    ts->tv_nsec += (us % 1000000L) * 1000L;
    ts->tv_sec += us / 1000000L + ts->tv_nsec / 1000000000L;
    ts->tv_nsec %= 1000000000L;
}

static void *decode_worker_main(void *arg) {
    server *s = arg;
    p36_decode_item *items = xmalloc((size_t)s->slot_count * sizeof(*items));
    server_slot **members = xmalloc((size_t)s->slot_count * sizeof(*members));
    const long coalesce_us = server_decode_coalesce_us();
    pthread_mutex_lock(&s->model_mu);
    for (;;) {
        while (s->decode_pending == 0 && !s->model_stopping)
            pthread_cond_wait(&s->model_cv, &s->model_mu);
        if (s->decode_pending == 0 && s->model_stopping) break;
        if (coalesce_us > 0 && s->decode_pending < s->slot_count) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            timespec_add_us(&deadline, coalesce_us);
            while (!s->model_stopping && s->decode_pending < s->slot_count &&
                   s->decode_pending < s->active_generations) {
                if (pthread_cond_timedwait(&s->model_cv, &s->model_mu,
                                           &deadline) == ETIMEDOUT) break;
            }
        }
        while (s->model_busy) pthread_cond_wait(&s->model_cv, &s->model_mu);
        int count = 0;
        for (int i = 0; i < s->slot_count; i++) {
            server_slot *slot = &s->slots[i];
            if (!slot->decode_pending) continue;
            slot->decode_pending = false;
            slot->decode_in_flight = true;
            s->decode_pending--;
            members[count] = slot;
            items[count].session = slot->session;
            items[count].token = slot->decode_token;
            count++;
        }
        if (count == 0) continue;
        s->model_busy = true;
        pthread_mutex_unlock(&s->model_mu);
        char batch_err[160] = {0};
        double t0 = now_sec();
        pthread_mutex_lock(&s->inference_mu);
        int rc = p36_sessions_eval_batch(items, count, batch_err, sizeof(batch_err));
        pthread_mutex_unlock(&s->inference_mu);
        if (getenv("P36_SERVER_BATCH_LOG")) {
            server_log(P36_LOG_DEFAULT,
                       "p36-server: decode batch count=%d elapsed=%.3f ms status=%s",
                       count, (now_sec() - t0) * 1000.0,
                       rc == 0 ? "ok" : "error");
        }
        pthread_mutex_lock(&s->model_mu);
        s->model_busy = false;
        for (int i = 0; i < count; i++) {
            server_slot *slot = members[i];
            slot->decode_in_flight = false;
            slot->decode_rc = rc;
            if (rc != 0) snprintf(slot->decode_err, sizeof(slot->decode_err),
                                  "%s", batch_err[0] ? batch_err : "batched decode failed");
            slot->decode_done = true;
        }
        pthread_cond_broadcast(&s->model_cv);
    }
    pthread_mutex_unlock(&s->model_mu);
    free(members);
    free(items);
    return NULL;
}

static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = ud;
    if (!p || !event || strcmp(event, "prefill_chunk")) return;

    double now = now_sec();
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && p->slot && current > p->cached_tokens)
            kv_cache_maybe_store_continued(p->srv, p->slot);
        return;
    }
    int display_start = p->cached_tokens;
    if (display_start < 0 || display_start > p->prompt_tokens) display_start = 0;
    if (current < display_start) display_start = 0; /* rebuild after failure: show absolute */
    int display_total = p->prompt_tokens - display_start;
    if (display_total <= 0) {
        display_start = 0;
        display_total = p->prompt_tokens > total ? p->prompt_tokens : total;
    }
    int display_current = current - display_start;
    if (display_current < 0) display_current = 0;
    if (display_current > display_total) display_current = display_total;
    double pct = display_total > 0 ? 100.0 * (double)display_current / (double)display_total : 100.0;
    double avg_tps = elapsed > 0.0 ? (double)display_current / elapsed : 0.0;
    int interval_tokens = p->seen ? current - p->last_current : 0;
    if (interval_tokens < 0) interval_tokens = 0;
    double interval_s = p->seen ? now - p->last_t : 0.0;
    double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    p->last_current = current;
    p->last_t = now;
    p->seen = true;
    char flags[64];
    log_flags(flags, sizeof(flags), p->has_tools, false, false, false);
    server_log(P36_LOG_PREFILL,
               "p36-server: %s ctx=%s%s%s prefill chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->kind == REQ_CHAT ? "chat" : "completion",
               p->ctx,
               flags[0] ? " " : "",
               flags,
               display_current,
               display_total,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    if (p->srv && p->slot && current > p->cached_tokens)
        kv_cache_maybe_store_continued(p->srv, p->slot);
}

static char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    buf suffix = {0};
    if (r && r->kat_coder) {
        if (p36_think_mode_enabled(r->think_mode)) {
            buf_puts_trimmed(&suffix, reasoning);
            buf_puts(&suffix, "\n</think>\n\n");
        }
        buf_puts_trimmed(&suffix, content);
        append_kat_tool_calls_text(&suffix, calls, &r->tool_orders,
                                   text_has_trimmed_content(content));
        buf_puts(&suffix, "<|im_end|>\n");
        return buf_take(&suffix);
    }
    if (r && r->chat_template) {
        if (p36_think_mode_enabled(r->think_mode)) {
            buf_puts_trimmed(&suffix, reasoning);
            buf_puts(&suffix, "\n</think>\n\n");
        }
        buf_puts_trimmed(&suffix, content);
        append_json_tool_calls_text(&suffix, calls, text_has_trimmed_content(content));
        buf_puts(&suffix, "<|im_end|>\n");
        return buf_take(&suffix);
    }
    if (r && p36_think_mode_enabled(r->think_mode)) {
        buf_puts_trimmed(&suffix, reasoning);
        buf_puts(&suffix, "\n</think>\n\n");
    }
    buf_puts_trimmed(&suffix, content);
    append_qwen_tool_calls_text(&suffix, calls, r ? &r->tool_orders : NULL,
                                text_has_trimmed_content(content));
    buf_puts(&suffix, "<|im_end|>\n");
    return buf_take(&suffix);
}

/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the conversation prefix is logically the same.  Rewrite the finished turn to
 * the exact toolless history form that render_chat_prompt_text() will produce:
 *
 *   prompt-without-final-<think> + visible-content + <|im_end|>\n
 *
 * This is the same policy the prompt renderer already applies.  The only goal
 * here is to make the live checkpoint match that policy before the next turn. */
static void canonicalize_thinking_checkpoint(server *s, server_slot *slot,
                                             const job *j, const char *ctx,
                                             uint64_t trace_id, const char *content) {
    for (size_t i = 0; i < j->req.image_count; i++)
        if (!j->req.images[i].embedding.data) return;

    if (!j->req.prompt_text) return;

    size_t pt_len = strlen(j->req.prompt_text);
    const char *think_tag = p36_think_mode_enabled(j->req.think_mode) ?
        "<think>\n" : "<think>\n\n</think>\n\n";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(j->req.prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return;
    }

    buf stable_prefix = {0};
    buf_append(&stable_prefix, j->req.prompt_text, pt_len - tag_len);

    buf rendered = {0};
    buf_puts(&rendered, stable_prefix.ptr ? stable_prefix.ptr : "");
    buf_puts_trimmed(&rendered, content);
    buf_puts(&rendered, "<|im_end|>\n");

    p36_tokens stable = {0};
    p36_tokens canonical = {0};
    p36_tokenize_rendered_chat(s->engine, stable_prefix.ptr ? stable_prefix.ptr : "", &stable);
    p36_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = p36_session_pos(slot->session);
    const int common = p36_session_common_prefix(slot->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    if (common < stable.len) {
        trace_event(s, trace_id,
                    "thinking checkpoint canonicalization skipped: common=%d stable=%d live=%d canonical=%d",
                    common, stable.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    if (j->req.image_count) {
        if (server_session_sync_multimodal(s, slot, &canonical, j->req.images,
                                            j->req.image_count, err, sizeof(err)) != 0)
            trace_event(s, trace_id, "vision checkpoint rebuild failed: %s", err);
        goto done;
    }
    p36_session_rewrite_result rr =
        p36_session_rewrite_from_common(slot->session, &canonical, common,
                                        err, sizeof(err));
    if (rr == P36_SESSION_REWRITE_OK) {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: thinking checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "thinking checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == P36_SESSION_REWRITE_REBUILD_NEEDED) {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: thinking checkpoint canonicalization needs rebuild ctx=%s common=%d live=%d canonical=%d reason=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        char *path = NULL;
        p36_tokens effective = {0};
        int loaded = server_kv_try_load_text(s, slot, rendered.ptr ? rendered.ptr : "",
                                            false, &effective, &path);
        if (loaded == 0) p36_session_invalidate(slot->session);

        char sync_err[160] = {0};
        const p36_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        if (server_session_sync(s, slot, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            if (loaded > 0) {
                server_log(P36_LOG_KVCACHE,
                           "p36-server: thinking checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d via=disk cached=%d",
                           ctx, common, live_len, canonical.len, loaded);
                trace_event(s, trace_id,
                            "thinking checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(P36_LOG_KVCACHE,
                           "p36-server: thinking checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d via=rebuild",
                           ctx, common, live_len, canonical.len);
                trace_event(s, trace_id,
                            "thinking checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            server_log(P36_LOG_KVCACHE,
                       "p36-server: thinking checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                       ctx, common, live_len, canonical.len, sync_err);
            trace_event(s, trace_id, "thinking checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        p36_tokens_free(&effective);
        free(path);
    } else {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: thinking checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "thinking checkpoint canonicalization failed: %s", err);
    }

done:
    p36_tokens_free(&stable);
    p36_tokens_free(&canonical);
    buf_free(&stable_prefix);
    buf_free(&rendered);
}

/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact tool block remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid Qwen tool-call text from JSON, and this function
 * either rewrites the short suffix in place or reloads an older disk checkpoint
 * before replay. */
static void canonicalize_tool_checkpoint(server *s, server_slot *slot,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    for (size_t i = 0; i < j->req.image_count; i++)
        if (!j->req.images[i].embedding.data) return;

    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    size_t prefix_len = strlen(j->req.prompt_text);
    const char *empty_think = "<think>\n\n</think>\n\n";
    size_t empty_len = strlen(empty_think);
    bool strip_empty = !calls->replay_empty_think && !j->req.kat_coder &&
                       !p36_think_mode_enabled(j->req.think_mode) &&
                       prefix_len >= empty_len &&
                       !memcmp(j->req.prompt_text + prefix_len - empty_len,
                               empty_think, empty_len);
    if (strip_empty) prefix_len -= empty_len;
    buf_append(&rendered, j->req.prompt_text, prefix_len);
    buf_puts(&rendered, suffix_text);

    p36_tokens canonical = {0};
    p36_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = p36_session_pos(slot->session);
    const int common = p36_session_common_prefix(slot->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, p36_session_tokens(slot->session), &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    int stable_len = j->req.prompt.len;
    if (strip_empty) {
        p36_tokens stable = {0};
        char *prefix = xstrndup(j->req.prompt_text, prefix_len);
        p36_tokenize_rendered_chat(s->engine, prefix, &stable);
        stable_len = stable.len;
        free(prefix);
        p36_tokens_free(&stable);
    }
    if (common < stable_len) {
        trace_event(s, trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    if (j->req.image_count) {
        if (server_session_sync_multimodal(s, slot, &canonical, j->req.images,
                                            j->req.image_count, err, sizeof(err)) != 0)
            trace_event(s, trace_id, "vision checkpoint rebuild failed: %s", err);
        goto done;
    }
    p36_session_rewrite_result rr =
        p36_session_rewrite_from_common(slot->session, &canonical, common,
                                        err, sizeof(err));
    if (rr == P36_SESSION_REWRITE_OK) {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == P36_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated QWEN_TOOL suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        server_log(P36_LOG_KVCACHE,
                   "p36-server: tool checkpoint canonicalization needs rebuild ctx=%s common=%d live=%d canonical=%d reason=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        char *path = NULL;
        p36_tokens effective = {0};
        int loaded = server_kv_try_load_text(s, slot, rendered.ptr ? rendered.ptr : "",
                                            false, &effective, &path);
        if (loaded == 0) p36_session_invalidate(slot->session);

        char sync_err[160] = {0};
        const p36_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        if (server_session_sync(s, slot, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            if (loaded > 0) {
                server_log(P36_LOG_KVCACHE,
                           "p36-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d via=disk cached=%d",
                           ctx, common, live_len, canonical.len, loaded);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(P36_LOG_KVCACHE,
                           "p36-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d via=rebuild",
                           ctx, common, live_len, canonical.len);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            server_log(P36_LOG_KVCACHE,
                       "p36-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                       ctx, common, live_len, canonical.len, sync_err);
            trace_event(s, trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        p36_tokens_free(&effective);
        free(path);
    } else {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    p36_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}

/* Exact pending call IDs bind a tool-only request to one live image context.
 * Historical spans carry identities only; they cannot rebuild a cold graph. */
static bool request_continue_vision(server *s, server_slot *slot, request *r) {
    if (!slot_pending_matches(slot, r)) return false;
    size_t old_count = 0;
    const p36_vision_span *old = p36_session_vision_spans(slot->session, &old_count);
    if (!old_count || old_count + r->image_count > 16) return false;
    const char *suffix = strstr(r->prompt_text, "<|im_start|>user\n<tool_response>\n");
    if (!suffix) return false;
    const p36_tokens *live = p36_session_tokens(slot->session);
    p36_tokens next = {0};
    p36_tokens_copy(&next, live);
    p36_tokenize_rendered_chat(s->engine, suffix, &next);
    int delta = next.len - r->prompt.len;
    size_t count = old_count + r->image_count;
    p36_vision_span *images = xmalloc(count * sizeof(*images));
    memcpy(images, old, old_count * sizeof(*images));
    for (size_t i = 0; i < r->image_count; i++) {
        images[old_count + i] = r->images[i];
        images[old_count + i].token_start = (uint32_t)((int)r->images[i].token_start + delta);
    }
    if (!p36_session_vision_prefix_matches(slot->session, images, count)) {
        free(images);
        p36_tokens_free(&next);
        return false;
    }
    free(r->images);
    r->images = images;
    r->image_count = count;
    free(r->prompt_text);
    r->prompt_text = render_tokens_text(s->engine, &next, NULL);
    p36_tokens_free(&r->prompt);
    r->prompt = next;
    return true;
}

static void slot_remember_vision_tools(server *s, server_slot *slot,
                                       const request *r, const tool_calls *calls) {
    if (!r->image_count || !calls->len || calls->len > 16) return;
    p36_tokens closed = {0};
    const p36_tokens *live = p36_session_tokens(slot->session);
    if (!live || !p36_session_vision_prefix_matches(slot->session, r->images, r->image_count)) return;
    p36_tokens_copy(&closed, live);
    char *text = render_tokens_text(s->engine, live, NULL);
    size_t len = strlen(text), end_len = strlen("<|im_end|>\n");
    if (len < end_len || strcmp(text + len - end_len, "<|im_end|>\n"))
        p36_tokenize_rendered_chat(s->engine, "<|im_end|>\n", &closed);
    free(text);
    char err[160] = {0};
    int rc = server_session_sync_multimodal(s, slot, &closed, r->images,
                                           r->image_count, err, sizeof(err));
    p36_tokens_free(&closed);
    if (rc) return;
    slot->pending_ids = xmalloc((size_t)calls->len * sizeof(char *));
    for (int i = 0; i < calls->len; i++)
        slot->pending_ids[slot->pending_count++] = xstrdup(calls->v[i].id);
    slot->pending_pos = p36_session_pos(slot->session);
    slot->pending_api = r->api;
    slot->pending_tools = xstrdup(r->tool_schema_key ? r->tool_schema_key : "");
}

/* Token count of the leading "<|im_start|>system ... <|im_end|>\n" turn, or 0 when
 * the prompt does not begin with a system turn that is followed by more text.
 * Tools are rendered inside this turn, so it is the stable system+tools prefix.
 * Every delimiter is a special token, so the prefix tokenizes identically on
 * its own and inside any longer prompt. */
static int system_boundary_tokens(const p36_tokens *t, int im_start, int system_word,
                                  int im_end, int newline) {
    if (!t || t->len < 6 || t->v[0] != im_start || t->v[1] != system_word) return 0;
    for (int i = 2; i + 2 < t->len; i++) {
        if (t->v[i] == im_end) return t->v[i + 1] == newline ? i + 2 : 0;
        if (t->v[i] == im_start) return 0;
    }
    return 0;
}

static int request_system_boundary(server *s, const request *r) {
    p36_tokens head = {0}, tail = {0};
    p36_tokenize_rendered_chat(s->engine, "<|im_start|>system\n", &head);
    p36_tokenize_rendered_chat(s->engine, "<|im_end|>\n", &tail);
    int n = 0;
    if (head.len == 3 && tail.len == 2)
        n = system_boundary_tokens(&r->prompt, head.v[0], head.v[1], tail.v[0], tail.v[1]);
    p36_tokens_free(&head);
    p36_tokens_free(&tail);
    /* Land on a prefill-chunk multiple.  A cold prompt prefills in fixed chunks
     * from position 0; a checkpoint inside a chunk would split that schedule
     * differently and the restored state would differ in the last bits (greedy
     * replies then diverge).  On a multiple, restored == cold exactly. */
    const uint32_t chunk = p36_engine_prefill_chunk(s->engine);
    if (n > 0 && chunk > 1) n -= n % (int)chunk;
    return n;
}

/* Execute one request on the worker-owned session.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then the disk text-prefix index, then a cold prefill.  On text-prefix hits we
 * build a fresh effective prompt from the checkpoint's exact token history plus
 * a newly tokenized string suffix; the canonical full-prompt tokens are not
 * sliced because BPE may merge across the byte boundary.  Cold prompt caching is
 * handled before generation: if the stable checkpoint is shorter than the full
 * prompt, we prefill to that boundary, store it, and immediately continue to the
 * real prompt.  The live graph therefore always moves forward. */
