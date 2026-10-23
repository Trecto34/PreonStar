static const char *context_length_error_param(const request *r) {
    if (!r) return "prompt";
    if (r->api == API_RESPONSES) return "input";
    return r->kind == REQ_COMPLETION ? "prompt" : "messages";
}

static bool request_exceeds_context(const request *r, int ctx_size) {
    /* Session sync needs one free slot for generation. */
    return r && r->prompt.len >= ctx_size;
}

static bool http_error_context_length_exceeded(int fd, const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size) {
    buf b = {0};
    char msg[160];
    snprintf(msg, sizeof(msg),
             "Prompt has %d tokens, but the configured context size is %d tokens",
             n_prompt_tokens, ctx_size);

    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
        json_escape(&b, msg);
        buf_printf(&b, ",\"n_prompt_tokens\":%d,\"n_ctx\":%d}}\n",
                   n_prompt_tokens, ctx_size);
    } else {
        buf_puts(&b, "{\"error\":{\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"type\":\"invalid_request_error\",\"param\":");
        json_escape(&b, context_length_error_param(r));
        buf_printf(&b,
                   ",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":%d,\"n_ctx\":%d}}\n",
                   n_prompt_tokens, ctx_size);
    }
    bool ok = http_response(fd, 400, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Streaming is a translation state machine over the raw P36 text.  The model
 * may produce <think> and QWEN_TOOL tool blocks; clients should receive those as
 * protocol-native reasoning/tool deltas, never as visible assistant text. */
static bool sse_headers(int fd) {
    buf h = {0};
    buf_puts(&h,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n");
    if (g_enable_cors) {
        buf_puts(&h,
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n");
    }
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    buf_free(&h);
    return ok;
}

static bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":");
        if (text) {
            buf_puts(&b, "{\"content\":");
            json_escape(&b, text);
            buf_putc(&b, '}');
        } else {
            buf_puts(&b, finish ? "{}" : "{\"role\":\"assistant\"}");
        }
        buf_puts(&b, ",\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

/* OpenAI-standard reasoning_tokens plus p36 timings, appended inside "usage". */
static void append_usage_ext(buf *b, const request *r, int completion_tokens) {
    if (r->kind != REQ_CHAT) return;
    int content = completion_tokens - r->reasoning_tokens;
    buf_printf(b,
        ",\"completion_tokens_details\":{\"reasoning_tokens\":%d,\"content_tokens\":%d}"
        ",\"timings\":{\"prefill_ms\":%.1f,\"decode_ms\":%.1f}",
        r->reasoning_tokens, content < 0 ? 0 : content,
        r->prefill_s * 1000.0, r->decode_s * 1000.0);
    if (r->sys_ckpt.enabled)
        buf_printf(b,
            ",\"system_checkpoint\":{\"hit\":%s,\"tokens\":%d,\"restore_ms\":%.1f,\"build_ms\":%.1f}",
            r->sys_ckpt.hit ? "true" : "false", r->sys_ckpt.tokens,
            r->sys_ckpt.restore_ms, r->sys_ckpt.build_ms);
}

static bool sse_usage_chunk(int fd, const request *r, const char *id,
                            int prompt_tokens, int completion_tokens) {
    if (!r->stream_include_usage) return true;

    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    }
    buf_printf(&b,
               "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}",
               prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, r->cached_tokens);
    append_usage_ext(&b, r, completion_tokens);
    buf_puts(&b, "}}\n\n");

    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens) {
    return sse_usage_chunk(fd, r, id, prompt_tokens, completion_tokens) &&
           send_all(fd, "data: [DONE]\n\n", 14);
}

static bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens) {
    if (!sse_chunk(fd, r, id, NULL, NULL)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (reasoning && reasoning[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
        json_escape(&b, reasoning);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (content && content[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        json_escape(&b, content);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (calls && calls->len) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    if (r->incomplete_tool_call) buf_puts(&b, ",\"incomplete_tool_call\":true");
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}

typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    OPENAI_TOOL_BETWEEN_INVOKES,
    OPENAI_TOOL_BETWEEN_PARAMS,
    OPENAI_TOOL_PARAM_VALUE,
    OPENAI_TOOL_BETWEEN_CALLS,
    OPENAI_TOOL_DONE,
    OPENAI_TOOL_ERROR,
} openai_tool_stream_state;

typedef struct {
    openai_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    char *param_name;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_reasoning;
    bool sent_content;
    openai_tool_stream tool;
} openai_stream;

static void openai_stream_start(const request *r, openai_stream *st) {
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = p36_think_mode_enabled(r->think_mode) ? OPENAI_STREAM_THINKING : OPENAI_STREAM_TEXT;
}

static void openai_tool_stream_free(openai_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    free(ts->param_name);
    ts->ids = NULL;
    ts->param_name = NULL;
    ts->ids_cap = 0;
}

static void openai_stream_free(openai_stream *st) {
    if (!st) return;
    openai_tool_stream_free(&st->tool);
}

static bool openai_tool_stream_has_id(const openai_tool_stream *ts,
                                      const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}

static const char *openai_tool_stream_id(server *s, openai_tool_stream *ts,
                                         int index) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), API_OPENAI);
            if (!openai_tool_stream_has_id(ts, id, index) &&
                !tool_memory_has_id(s, id)) break;
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}

static size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);

static bool sse_chat_delta_n(int fd, const request *r, const char *id,
                             const char *field, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&b, field);
    buf_putc(&b, ':');
    json_escape_n(&b, text, len);
    buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

/* OpenAI clients can consume function.arguments as a stream of JSON text
 * fragments.  P36 generates XML-ish QWEN_TOOL instead, so this parser switches to a
 * hidden tool mode at <...tool_calls>, emits the tool header once the invoke tag
 * is complete, then translates each parameter body into argument deltas while
 * holding only tiny tails for partial closing tags, UTF-8, and QWEN_TOOL entities. */
static bool sse_chat_tool_call_start_delta(int fd, const request *r, const char *id,
                                           int index, const char *tool_id,
                                           const char *name) {
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"id\":");
    json_escape(&b, tool_id ? tool_id : "");
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, name ? name : "");
    buf_puts(&b, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool sse_chat_tool_call_args_delta_n(int fd, const request *r, const char *id,
                                            int index, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"function\":{\"arguments\":");
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}

static bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}

static bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b) {
    return raw_partial_lit(raw, raw_len, pos, a) || raw_partial_lit(raw, raw_len, pos, b);
}

static const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) {
        if (!memcmp(s + i, lit, m)) return s + i;
    }
    return NULL;
}

typedef enum {
    QWEN_TOOL_DECODE_OUTSIDE,
    QWEN_TOOL_DECODE_STRUCTURAL,
    QWEN_TOOL_DECODE_PAYLOAD,
} qwen_tool_decode_state;

typedef enum {
    QWEN_TOOL_TRACK_SEARCH,
    QWEN_TOOL_TRACK_STRUCTURAL,
    QWEN_TOOL_TRACK_PAYLOAD,
    QWEN_TOOL_TRACK_DONE,
} qwen_tool_track_mode;

static const qwen_tool_syntax qwen_tool_syntaxes[] = {
    {
        P36_TOOL_CALLS_START, P36_TOOL_CALLS_END,
        P36_INVOKE_START, P36_INVOKE_END,
        P36_PARAM_START, P36_PARAM_END,
    },
};

typedef struct {
    qwen_tool_track_mode mode;
    qwen_tool_decode_state decode;
    const qwen_tool_syntax *syn;
    size_t pos;
} qwen_tool_decode_tracker;


static size_t qwen_tool_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < sizeof(qwen_tool_syntaxes) / sizeof(qwen_tool_syntaxes[0]); i++) {
        size_t n = strlen(qwen_tool_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}

static bool qwen_tool_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const qwen_tool_syntax **syn_out) {
    const char *best = NULL;
    const qwen_tool_syntax *best_syn = NULL;
    for (size_t i = 0; i < sizeof(qwen_tool_syntaxes) / sizeof(qwen_tool_syntaxes[0]); i++) {
        const char *p = find_lit_bounded(raw, raw_len, qwen_tool_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) {
            best = p;
            best_syn = &qwen_tool_syntaxes[i];
        }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->tool_calls_start);
    *syn_out = best_syn;
    return true;
}

static bool qwen_tool_find_tool_start_from(const char *raw, size_t raw_len,
                                      size_t start,
                                      size_t *pos_out,
                                      const qwen_tool_syntax **syn_out) {
    if (start > raw_len) return false;
    size_t rel = 0;
    if (!qwen_tool_find_tool_start(raw + start, raw_len - start, &rel, syn_out)) {
        return false;
    }
    *pos_out = start + rel;
    return true;
}

#ifdef P36_SERVER_TEST
static bool raw_suffix_partial_lit(const char *raw, size_t raw_len,
                                   const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || raw_len == 0 || lit_len == 0) return false;
    size_t max = raw_len < lit_len ? raw_len : lit_len - 1;
    for (size_t n = min_len; n <= max; n++) {
        if (!memcmp(raw + raw_len - n, lit, n)) return true;
    }
    return false;
}

/* Slow reference recognizer used by tests. */
static qwen_tool_decode_state qwen_tool_decode_state_for_text(const char *raw, size_t raw_len) {
    if (!raw || raw_len == 0) return QWEN_TOOL_DECODE_OUTSIDE;

    size_t pos = 0;
    const qwen_tool_syntax *syn = NULL;
    if (!qwen_tool_find_tool_start(raw, raw_len, &pos, &syn)) {
        return QWEN_TOOL_DECODE_OUTSIDE;
    }

    for (;;) {
        while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
        if (pos >= raw_len) return QWEN_TOOL_DECODE_STRUCTURAL;

        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_end)) {
            return QWEN_TOOL_DECODE_OUTSIDE;
        }
        if (raw[pos] == '{') {
            const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->tool_calls_end);
            if (!end) return QWEN_TOOL_DECODE_STRUCTURAL;
            pos = (size_t)(end - raw);
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_end)) {
            pos += strlen(syn->invoke_end);
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_start)) {
            const char *tag_end = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end) return QWEN_TOOL_DECODE_STRUCTURAL;
            pos = (size_t)(tag_end - raw) + 1;
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->param_start)) {
            const char *tag_end_ptr = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end_ptr) return QWEN_TOOL_DECODE_STRUCTURAL;
            size_t tag_end = (size_t)(tag_end_ptr - raw) + 1;
            pos = tag_end;
            const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
            if (!end) {
                if (raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2))
                    return QWEN_TOOL_DECODE_STRUCTURAL;
                return QWEN_TOOL_DECODE_PAYLOAD;
            }
            pos = (size_t)(end - raw) + strlen(syn->param_end);
            continue;
        }

        for (size_t i = 0; i < sizeof(qwen_tool_syntaxes) / sizeof(qwen_tool_syntaxes[0]); i++) {
            if (raw_partial_lit(raw, raw_len, pos, qwen_tool_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, qwen_tool_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, qwen_tool_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, qwen_tool_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, qwen_tool_syntaxes[i].param_end))
            {
                return QWEN_TOOL_DECODE_STRUCTURAL;
            }
        }
        return QWEN_TOOL_DECODE_STRUCTURAL;
    }
}
#endif

static bool qwen_tool_decode_state_is_tool(qwen_tool_decode_state state) {
    return state != QWEN_TOOL_DECODE_OUTSIDE;
}

static bool qwen_tool_decode_state_uses_payload_sampling(qwen_tool_decode_state state) {
    return state == QWEN_TOOL_DECODE_PAYLOAD;
}

static void qwen_tool_decode_tracker_init(qwen_tool_decode_tracker *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->mode = QWEN_TOOL_TRACK_SEARCH;
    dt->decode = QWEN_TOOL_DECODE_OUTSIDE;
}

/* Track where generation is inside a QWEN_TOOL tool call.  This is intentionally a
 * forgiving recognizer, not a validator: malformed QWEN_TOOL still gets parsed later
 * by the normal tool-call parser.  Here we only need enough state to decide
 * whether the next token belongs to protocol syntax or arbitrary payload. */
