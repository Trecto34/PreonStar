static bool final_response(int fd, const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, text ? text : "");
        if (reasoning && reasoning[0]) {
            buf_puts(&b, ",\"reasoning_content\":");
            json_escape(&b, reasoning);
        }
        if (calls && calls->len) {
            buf_puts(&b, ",\"tool_calls\":");
            append_tool_calls_json(&b, calls, id, &r->tool_orders);
        }
        buf_puts(&b, "},\"finish_reason\":");
        json_escape(&b, finish);
        if (r->incomplete_tool_call) buf_puts(&b, ",\"incomplete_tool_call\":true");
        buf_puts(&b, "}],\"usage\":");
    } else {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text);
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    }
    buf_printf(&b, "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}",
               prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, r->cached_tokens);
    append_usage_ext(&b, r, completion_tokens);
    buf_puts(&b, "}}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static const char *responses_status(const char *finish) {
    if (finish && !strcmp(finish, "in_progress")) return "in_progress";
    if (finish && !strcmp(finish, "error")) return "failed";
    if (finish && !strcmp(finish, "length")) return "incomplete";
    return "completed";
}

enum { RESP_REASONING, RESP_MESSAGE, RESP_TOOL };

static void append_responses_item(buf *b, const char *id, int kind, int call_index,
                                  const char *text, const tool_call *call,
                                  const char *finish) {
    const char *status = responses_status(finish);
    if (kind == RESP_TOOL) {
        buf_printf(b, "{\"id\":\"fc_%s_%d\",\"type\":\"function_call\",\"status\":", id, call_index);
        json_escape(b, status);
        buf_puts(b, ",\"call_id\":");
        json_escape(b, call->id ? call->id : "");
        buf_puts(b, ",\"name\":");
        json_escape(b, call->name ? call->name : "");
        buf_puts(b, ",\"arguments\":");
        json_escape(b, text ? text : "");
    } else {
        bool reasoning = kind == RESP_REASONING;
        buf_printf(b, "{\"id\":\"%s_%s\",\"type\":\"%s\",\"status\":",
                   reasoning ? "rs" : "msg", id, reasoning ? "reasoning" : "message");
        json_escape(b, status);
        if (!reasoning) buf_puts(b, ",\"role\":\"assistant\"");
        buf_printf(b, ",\"%s\":[", reasoning ? "summary" : "content");
        if (text) {
            buf_printf(b, "{\"type\":\"%s\",\"text\":", reasoning ? "summary_text" : "output_text");
            json_escape(b, text);
            if (!reasoning) buf_puts(b, ",\"annotations\":[]");
            buf_putc(b, '}');
        }
        buf_putc(b, ']');
    }
    buf_putc(b, '}');
}

static void append_responses_output(buf *b, const char *id, const char *text,
                                    const char *reasoning, const tool_calls *calls,
                                    const char *finish) {
    int output_index = 0;
    buf_putc(b, '[');
    if (reasoning && reasoning[0]) {
        append_responses_item(b, id, RESP_REASONING, 0, reasoning, NULL, finish);
        output_index++;
    }
    if (text && text[0]) {
        if (output_index++) buf_putc(b, ',');
        append_responses_item(b, id, RESP_MESSAGE, 0, text, NULL, finish);
    }
    for (int i = 0; calls && i < calls->len; i++) {
        if (output_index++) buf_putc(b, ',');
        append_responses_item(b, id, RESP_TOOL, i,
            calls->v[i].arguments ? calls->v[i].arguments : "{}", &calls->v[i], finish);
    }
    buf_putc(b, ']');
}

static void append_responses_object(buf *b, const request *r, const char *id,
                                    const char *text, const char *reasoning,
                                    const tool_calls *calls, const char *finish,
                                    int prompt_tokens, int completion_tokens) {
    const char *status = responses_status(finish);
    buf_puts(b, "{\"id\":");
    json_escape(b, id);
    buf_puts(b, ",\"object\":\"response\",\"created_at\":");
    buf_printf(b, "%ld", (long)time(NULL));
    buf_puts(b, ",\"status\":");
    json_escape(b, status);
    buf_puts(b, ",\"model\":");
    json_escape(b, r->model);
    buf_puts(b, ",\"output\":");
    append_responses_output(b, id, text, reasoning, calls, finish);
    if (!strcmp(status, "incomplete")) {
        buf_puts(b, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    } else {
        buf_puts(b, ",\"incomplete_details\":null");
    }
    buf_printf(b,
        ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,\"total_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d}}}",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, r->cached_tokens);
}

static bool responses_final_response(int fd, const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    append_responses_object(&b, r, id, text, reasoning, calls, finish,
                            prompt_tokens, completion_tokens);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_event(int fd, const char *type, const char *json) {
    buf b = {0};
    buf_puts(&b, "event: ");
    buf_puts(&b, type);
    buf_puts(&b, "\ndata: ");
    buf_puts(&b, json);
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool responses_sse_item(int fd, const char *id, int output_index,
                                int kind, int call_index, const char *text,
                                const tool_call *call, const char *finish, bool done) {
    buf b = {0};
    const char *event = done ? "response.output_item.done" : "response.output_item.added";
    buf_printf(&b, "{\"type\":\"%s\",\"output_index\":%d,\"item\":", event, output_index);
    append_responses_item(&b, id, kind, call_index, done ? text : NULL,
                           call, done ? finish : "in_progress");
    buf_putc(&b, '}');
    bool ok = responses_sse_event(fd, event, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_text(int fd, const char *id, int index, const char *text,
                                bool reasoning, const char *finish) {
    int kind = reasoning ? RESP_REASONING : RESP_MESSAGE;
    if (!responses_sse_item(fd, id, index, kind, 0, NULL, NULL, finish, false)) return false;
    const char *prefix = reasoning ? "rs" : "msg";
    const char *part = reasoning ? "reasoning_summary_part" : "content_part";
    const char *field = reasoning ? "summary_index" : "content_index";
    const char *type = reasoning ? "summary_text" : "output_text";
    const char *delta = reasoning ? "reasoning_summary_text" : "output_text";
    for (int stage = 0; stage < 4; stage++) {
        bool part_event = stage == 0 || stage == 3;
        char event[80];
        snprintf(event, sizeof(event), "response.%s.%s", part_event ? part : delta,
                 stage == 0 ? "added" : stage == 1 ? "delta" : "done");
        buf b = {0};
        buf_printf(&b, "{\"type\":\"%s\",\"item_id\":\"%s_%s\",\"output_index\":%d,\"%s\":0,",
                   event, prefix, id, index, field);
        if (part_event) buf_printf(&b, "\"part\":{\"type\":\"%s\",\"text\":", type);
        else buf_printf(&b, "\"%s\":", stage == 1 ? "delta" : "text");
        json_escape(&b, stage == 0 ? "" : text);
        if (part_event) {
            if (!reasoning) buf_puts(&b, ",\"annotations\":[]");
            buf_putc(&b, '}');
        }
        buf_putc(&b, '}');
        bool ok = responses_sse_event(fd, event, b.ptr);
        buf_free(&b);
        if (!ok) return false;
    }
    return responses_sse_item(fd, id, index, kind, 0, text, NULL, finish, true);
}

static bool responses_sse_finish(int fd, const request *r, const char *id,
                                 const char *text, const char *reasoning,
                                 const tool_calls *calls, const char *finish,
                                 int prompt_tokens, int completion_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"response.created\",\"response\":");
    append_responses_object(&b, r, id, "", NULL, NULL, "in_progress", 0, 0);
    buf_putc(&b, '}');
    bool ok = responses_sse_event(fd, "response.created", b.ptr);
    buf_free(&b);
    int output_index = 0;

    if (ok && reasoning && reasoning[0])
        ok = responses_sse_text(fd, id, output_index++, reasoning, true, finish);
    if (ok && text && text[0])
        ok = responses_sse_text(fd, id, output_index++, text, false, finish);
    for (int i = 0; ok && calls && i < calls->len; i++, output_index++) {
        const tool_call *tc = &calls->v[i];
        ok = responses_sse_item(fd, id, output_index, RESP_TOOL, i, NULL, tc, finish, false);
        if (!ok) break;
        buf_printf(&b,
            "{\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_%s_%d\",\"output_index\":%d,\"arguments\":",
            id, i, output_index);
        json_escape(&b, tc->arguments ? tc->arguments : "{}");
        buf_putc(&b, '}');
        ok = responses_sse_event(fd, "response.function_call_arguments.done", b.ptr);
        buf_free(&b);
        if (ok) ok = responses_sse_item(fd, id, output_index, RESP_TOOL, i,
            tc->arguments ? tc->arguments : "{}", tc, finish, true);
    }
    if (ok) {
        buf_puts(&b, "{\"type\":\"response.completed\",\"response\":");
        append_responses_object(&b, r, id, text, reasoning, calls, finish,
                                prompt_tokens, completion_tokens);
        buf_putc(&b, '}');
        ok = responses_sse_event(fd, "response.completed", b.ptr);
        buf_free(&b);
    }
    return ok;
}

static const char *anthropic_stop_reason(const char *finish) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    return "end_turn";
}

static void append_anthropic_tool_use(buf *b, const tool_call *tc, const char *id_prefix, int i,
                                      const tool_schema_orders *orders) {
    (void)orders;
    char idbuf[128];
    snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id_prefix, i);
    buf_puts(b, "{\"type\":\"tool_use\",\"id\":");
    json_escape(b, tc->id && tc->id[0] ? tc->id : idbuf);
    buf_puts(b, ",\"name\":");
    json_escape(b, tc->name ? tc->name : "");
    buf_puts(b, ",\"input\":");
    append_json_object_or_empty(b, tc->arguments);
    buf_putc(b, '}');
}

static void append_anthropic_thinking(buf *b, const char *reasoning, const char *signature) {
    buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
    json_escape(b, reasoning ? reasoning : "");
    buf_puts(b, ",\"signature\":");
    json_escape(b, signature ? signature : "");
    buf_putc(b, '}');
}

static void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix,
                                     const tool_schema_orders *orders) {
    buf_putc(b, '[');
    bool wrote = false;
    bool wrote_after_thinking = false;
    if (reasoning && reasoning[0]) {
        append_anthropic_thinking(b, reasoning, id_prefix);
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":");
        json_escape(b, text);
        buf_putc(b, '}');
        wrote = true;
        wrote_after_thinking = true;
    }
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(b, ',');
            append_anthropic_tool_use(b, &calls->v[i], id_prefix, i, orders);
            wrote = true;
            wrote_after_thinking = true;
        }
    }
    if (!wrote || ((reasoning && reasoning[0]) && !wrote_after_thinking)) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":\"\"}");
    }
    buf_putc(b, ']');
}

static bool anthropic_final_response(int fd, const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"content\":");
    append_anthropic_content(&b, text, reasoning, calls, id, &r->tool_orders);
    buf_puts(&b, ",\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null,\"usage\":");
    buf_printf(&b, "{\"input_tokens\":%d,\"output_tokens\":%d,\"cache_read_input_tokens\":%d}}\n",
               prompt_tokens - r->cached_tokens, completion_tokens, r->cached_tokens);
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool sse_event(int fd, const char *event, const char *data) {
    buf b = {0};
    buf_puts(&b, "event: ");
    buf_puts(&b, event);
    buf_puts(&b, "\ndata: ");
    buf_puts(&b, data);
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
} anthropic_block_type;

typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_thinking;
    bool sent_text;
} anthropic_stream;

static bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st) {
    buf b = {0};
    json_escape(&b, r->model);
    char *model_json = buf_take(&b);

    buf_printf(&b,
        "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":%s,\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":0,\"cache_read_input_tokens\":%d}}}",
        id, model_json, prompt_tokens - r->cached_tokens, r->cached_tokens);
    bool ok = sse_event(fd, "message_start", b.ptr);
    buf_free(&b);
    free(model_json);

    memset(st, 0, sizeof(*st));
    st->active = ok;
    st->mode = p36_think_mode_enabled(r->think_mode) ? ANTH_STREAM_THINKING : ANTH_STREAM_TEXT;
    return ok;
}

static bool anthropic_sse_open_block(int fd, anthropic_stream *st,
                                     anthropic_block_type type) {
    if (st->open_block == type) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\","
                   "\"signature\":\"\"}}",
                   st->next_index);
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
                   st->next_index);
    }
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    buf_free(&b);
    if (ok) st->open_block = type;
    return ok;
}

static bool anthropic_sse_delta_live(int fd, const anthropic_stream *st,
                                     anthropic_block_type type,
                                     const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"text_delta\",\"text\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    }
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    buf_free(&b);
    return ok;
}

static bool anthropic_sse_close_block_live(int fd, const char *id,
                                           anthropic_stream *st) {
    if (st->open_block == ANTH_BLOCK_NONE) return true;

    buf b = {0};
    bool ok = true;
    if (st->open_block == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"signature_delta\",\"signature\":",
                   st->next_index);
        json_escape(&b, id);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        st->open_block = ANTH_BLOCK_NONE;
        st->next_index++;
    }
    return ok;
}

static size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final) {
    if (raw_len <= start) return raw_len;

    size_t limit = raw_len;
    if (has_tools) {
        const char *tool = find_any_tool_start(raw + start);
        if (tool) {
            limit = trim_tool_separator_ws(raw, start, (size_t)(tool - raw));
            return utf8_stream_safe_len(raw, start, limit, true);
        }

        if (!final) {
            /* Tool calls are hidden from the API client and returned as
             * structured tool_use/tool_calls blocks.  The whitespace just
             * before the QWEN_TOOL marker is syntax too: if we stream it as
             * assistant text, the next client request sends it back and our
             * renderer adds the canonical "\n\n" separator again.  Hold
             * trailing whitespace until a following non-whitespace byte proves
             * it is ordinary text, or until a tool marker proves it should be
             * dropped. */
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;

            /* Also hold a partial '<...tool_calls...' marker that may be split
             * across generated tokens. */
            const size_t max_marker = 80;
            size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') {
                    size_t marker = i - 1;
                    if (marker < limit) limit = marker;
                    break;
                }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}

