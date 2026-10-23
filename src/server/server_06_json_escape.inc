static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

static void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}

static void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
}

static const char *find_any_tool_start(const char *s) {
    return s ? strstr(s, P36_TOOL_CALLS_START) : NULL;
}

static const char *find_any_tool_end(const char *s) {
    return s ? strstr(s, P36_TOOL_CALLS_END) : NULL;
}


static size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}

static char *qwen_tool_tag_value(const char *tag, const char *prefix) {
    size_t prefix_len = strlen(prefix);
    size_t tag_len = strlen(tag);
    if (tag_len <= prefix_len || strncmp(tag, prefix, prefix_len) != 0 ||
        tag[tag_len - 1] != '>') return NULL;
    size_t len = tag_len - prefix_len - 1;
    while (len && isspace((unsigned char)tag[prefix_len + len - 1])) len--;
    return len ? xstrndup(tag + prefix_len, len) : NULL;
}

typedef struct qwen_tool_syntax {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} qwen_tool_syntax;

static bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit);
static const char *find_lit_bounded(const char *s, size_t n, const char *lit);
static bool qwen_tool_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const qwen_tool_syntax **syn_out);

static const char *find_tool_structural_text(const char *s, const char *needle, bool last);

static void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out) {
    char *s = xstrndup(text ? text : "", n);
    char *body = s;
    if (!strncmp(body, "<think>", 7)) body += 7;

    char *think_end = (char *)find_tool_structural_text(body, "</think>", false);
    if (think_end) {
        *think_end = '\0';
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup(think_end + 8);
    } else {
        *reasoning_out = NULL;
        *content_out = xstrdup(s);
    }
    free(s);
}

static void unterminated_reasoning(const char *text,
                                   char **content_out,
                                   char **reasoning_out) {
    const char *body = text ? text : "";
    if (!strncmp(body, "<think>", 7)) body += 7;
    *reasoning_out = xstrdup(body);
    *content_out = xstrdup("");
}

static void unterminated_reasoning_before_tool(const char *text,
                                               size_t prefix_len,
                                               char **content_out,
                                               char **reasoning_out) {
    const char *body = text ? text : "";
    size_t body_len = strlen(body);
    if (prefix_len > body_len) prefix_len = body_len;
    if (prefix_len >= 7 && !strncmp(body, "<think>", 7)) {
        body += 7;
        prefix_len -= 7;
    }
    *reasoning_out = xstrndup(body, prefix_len);
    *content_out = xstrdup("");
}

static size_t raw_skip_ascii_ws(const char *raw, size_t raw_len, size_t pos) {
    while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
    return pos;
}

/* Parameter bodies are literal data, even when they contain thinking or tool tags. */
static const char *find_tool_structural_text(const char *s, const char *needle,
                                            bool last) {
    const char *found = NULL;
    if (!s) return NULL;
    while (*s) {
        if (!strncmp(s, needle, strlen(needle))) {
            found = s;
            if (!last) return found;
        }
        if (!strncmp(s, P36_PARAM_START, strlen(P36_PARAM_START))) {
            const char *tag = strchr(s, '>');
            const char *end = tag ? strstr(tag + 1, P36_PARAM_END) : NULL;
            if (!end) return found;
            s = end + strlen(P36_PARAM_END);
        } else if (!strncmp(s, P36_TOOL_CALLS_START, strlen(P36_TOOL_CALLS_START))) {
            const char *p = s + strlen(P36_TOOL_CALLS_START);
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == '{' && json_skip_value(&p)) {
                s = p;
            } else {
                s++;
            }
        } else s++;
    }
    return found;
}


static bool qwen_tool_read_tag(const char *raw, size_t raw_len, size_t pos,
                          char **tag_out, size_t *after_out) {
    if (!raw || pos >= raw_len) return false;
    const char *tag_end = memchr(raw + pos, '>', raw_len - pos);
    if (!tag_end) return false;
    *tag_out = xstrndup(raw + pos, (size_t)(tag_end - (raw + pos) + 1));
    *after_out = (size_t)(tag_end - raw) + 1;
    return true;
}

static void qwen_tool_append_json_value(buf *out, const char *value, size_t len) {
    if (len && value[0] == '\n') {
        value++;
        len--;
    }
    if (len && value[len - 1] == '\n') len--;
    char *raw = xstrndup(value, len);
    p36_tool_text_unescape(raw, P36_PARAM_END);
    if (json_raw_value_is_complete(raw)) {
        char *minified = json_minify_raw_value(raw);
        buf_puts(out, minified);
        free(minified);
    } else {
        json_escape(out, raw);
    }
    free(raw);
}

static bool parse_json_tool_call(const char **p, tool_call *tc) {
    if (!parse_function_call(p, tc)) {
        tool_call_free(tc);
        return false;
    }
    if (!tc->name || !tc->name[0]) {
        tool_call_free(tc);
        return false;
    }
    if (!tc->arguments || !tc->arguments[0]) {
        free(tc->arguments);
        tc->arguments = xstrdup("{}");
    } else {
        char *minified = json_minify_raw_value(tc->arguments);
        free(tc->arguments);
        tc->arguments = minified;
    }
    return true;
}

static bool parse_generated_message_ex(const char *text, bool thinking,
                                       char **content_out,
                                       char **reasoning_out, tool_calls *calls) {
    text = text ? text : "";
    size_t raw_len = strlen(text);
    const char *tool_search = text;
    bool recovered_unclosed_tool = false;
    if (thinking) {
        const char *think_end = find_tool_structural_text(text, "</think>", true);
        if (!think_end) {
            const char *candidate = find_any_tool_start(text);
            if (!candidate || !find_any_tool_end(candidate)) {
                unterminated_reasoning(text, content_out, reasoning_out);
                return true;
            }
            tool_search = candidate;
            recovered_unclosed_tool = true;
        } else {
            tool_search = think_end + strlen("</think>");
        }
    }
    const char *start = strstr(tool_search, P36_TOOL_CALLS_START);
    if (!start) {
        split_reasoning_content(text, raw_len, content_out, reasoning_out);
        return true;
    }

    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(start - text));
    size_t pos = (size_t)(start - text);
    size_t raw_end = pos;
    while (pos < raw_len && raw_full_lit(text, raw_len, pos, P36_TOOL_CALLS_START)) {
        pos += strlen(P36_TOOL_CALLS_START);
        pos = raw_skip_ascii_ws(text, raw_len, pos);
        if (pos < raw_len && text[pos] == '{') {
            const char *p = text + pos;
            tool_call tc = {0};
            if (!parse_json_tool_call(&p, &tc)) return false;
            pos = raw_skip_ascii_ws(text, raw_len, (size_t)(p - text));
            if (!raw_full_lit(text, raw_len, pos, P36_TOOL_CALLS_END)) {
                tool_call_free(&tc);
                return false;
            }
            pos += strlen(P36_TOOL_CALLS_END);
            raw_end = pos;
            tool_calls_push(calls, tc);
            pos = raw_skip_ascii_ws(text, raw_len, pos);
            continue;
        }
        if (!raw_full_lit(text, raw_len, pos, P36_INVOKE_START)) return false;

        char *tag = NULL;
        size_t after = 0;
        if (!qwen_tool_read_tag(text, raw_len, pos, &tag, &after)) return false;
        tool_call tc = {0};
        tc.name = qwen_tool_tag_value(tag, P36_INVOKE_START);
        free(tag);
        if (!tc.name) return false;
        pos = after;

        buf args = {0};
        buf_putc(&args, '{');
        bool wrote = false;
        for (;;) {
            pos = raw_skip_ascii_ws(text, raw_len, pos);
            if (raw_full_lit(text, raw_len, pos, P36_INVOKE_END)) break;
            if (!raw_full_lit(text, raw_len, pos, P36_PARAM_START)) goto bad_call;
            tag = NULL;
            if (!qwen_tool_read_tag(text, raw_len, pos, &tag, &after)) goto bad_call;
            char *name = qwen_tool_tag_value(tag, P36_PARAM_START);
            free(tag);
            if (!name) goto bad_call;
            const char *end = find_lit_bounded(text + after, raw_len - after,
                                               P36_PARAM_END);
            if (!end) {
                free(name);
                goto bad_call;
            }
            if (wrote) buf_putc(&args, ',');
            json_escape(&args, name);
            buf_putc(&args, ':');
            qwen_tool_append_json_value(&args, text + after,
                                        (size_t)(end - (text + after)));
            free(name);
            wrote = true;
            pos = (size_t)(end - text) + strlen(P36_PARAM_END);
        }
        buf_putc(&args, '}');
        tc.arguments = buf_take(&args);
        pos += strlen(P36_INVOKE_END);
        pos = raw_skip_ascii_ws(text, raw_len, pos);
        if (!raw_full_lit(text, raw_len, pos, P36_TOOL_CALLS_END)) goto bad_call;
        pos += strlen(P36_TOOL_CALLS_END);
        raw_end = pos;
        tool_calls_push(calls, tc);
        pos = raw_skip_ascii_ws(text, raw_len, pos);
        continue;

bad_call:
        buf_free(&args);
        tool_call_free(&tc);
        return false;
    }

    if (calls->len == 0) return false;
    if (pos < raw_len) return false;
    calls->raw_tool_text = xstrndup(start, raw_end - (size_t)(start - text));
    if (recovered_unclosed_tool) {
        unterminated_reasoning_before_tool(text, content_len,
                                           content_out, reasoning_out);
    } else {
        split_reasoning_content(text, content_len, content_out, reasoning_out);
    }
    return true;
}

static bool parse_generated_message(const char *text, char **content_out,
                                    char **reasoning_out, tool_calls *calls) {
    return parse_generated_message_ex(text, false, content_out, reasoning_out, calls);
}

static const char *tool_parse_failure_recovery_finish(const char *finish) {
    /* Once Qwen parsing failed there is no executable tool call to report.
     * Preserve a true length stop, because callers can distinguish truncation
     * from a completed turn.  Every other non-error tool-parse failure becomes
     * a normal assistant stop with the raw model text returned as content. */
    if (finish && !strcmp(finish, "length")) return "length";
    return "stop";
}

/* Drop a trailing partial delimiter ("</thi", "<tool_ca") left by a cut-off. */
static void strip_partial_suffix(char *s, const char *marker) {
    if (!s) return;
    size_t n = strlen(s), m = strlen(marker);
    for (size_t k = m - 1; k >= 2; k--) {
        if (n >= k && !strncmp(s + n - k, marker, k)) { s[n - k] = '\0'; return; }
    }
}

static void blank_if_whitespace(char *s) {
    if (!s) return;
    for (char *p = s; *p; p++) if (!isspace((unsigned char)*p)) return;
    *s = '\0';
}

/* The generation ended inside, or just after, a tool call.  Keep every complete
 * call, discard the unfinished tail and any prose after the last complete call,
 * and split the text before the first call into reasoning and content.  Returns
 * false when this is not a truncation (no tool start, or a closed but malformed
 * call, which the caller returns as text for the model to see). */
static bool recover_truncated_tool(const char *text, bool thinking,
                                   char **content_out, char **reasoning_out,
                                   tool_calls *calls, bool *tail_was_partial) {
    const char *think_end = thinking ? find_tool_structural_text(text, "</think>", true) : NULL;
    if (thinking && !think_end) return false;
    const char *first = find_any_tool_start(think_end ? think_end + 8 : text);
    if (!first) return false;

    size_t best_end = 0;
    for (const char *e = find_any_tool_end(first); e; e = find_any_tool_end(e + 1)) {
        size_t end = (size_t)(e - text) + strlen(P36_TOOL_CALLS_END);
        char *sub = xstrndup(first, end - (size_t)(first - text));
        tool_calls tmp = {0};
        char *c = NULL, *r = NULL;
        bool ok = parse_generated_message_ex(sub, false, &c, &r, &tmp) && tmp.len > 0;
        free(sub); free(c); free(r);
        if (ok) { best_end = end; tool_calls_free(calls); *calls = tmp; }
        else tool_calls_free(&tmp);
    }
    if (!best_end && find_any_tool_end(first)) return false;   /* closed but malformed */

    const char *tail = text + best_end;
    if (best_end) {
        while (*tail && isspace((unsigned char)*tail)) tail++;
        if (!*tail) { tool_calls_free(calls); return false; }  /* complete: normal path */
    }
    *tail_was_partial = !best_end || find_any_tool_start(tail) != NULL;
    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(first - text));
    split_reasoning_content(text, content_len, content_out, reasoning_out);
    return true;
}

static bool parse_generated_message_for_response(const char *text,
                                                 bool thinking,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out,
                                                 bool *incomplete_tool_out) {
    if (recovered_out) *recovered_out = false;
    if (incomplete_tool_out) *incomplete_tool_out = false;
    const char *fin = finish_io && *finish_io ? *finish_io : "stop";
    const bool cut = !strcmp(fin, "length") || !strcmp(fin, "action_budget");

    bool parsed_ok = parse_generated_message_ex(text ? text : "", thinking,
                                                content_out, reasoning_out, calls);
    if (!parsed_ok && has_tools && strcmp(fin, "error") != 0) {
        free(*content_out); free(*reasoning_out);
        *content_out = NULL; *reasoning_out = NULL;
        tool_calls_free(calls);
        bool partial = false;
        if (recover_truncated_tool(text ? text : "", thinking, content_out, reasoning_out,
                                   calls, &partial)) {
            if (incomplete_tool_out) *incomplete_tool_out = partial;
            parsed_ok = true;
        } else {
            *content_out = NULL; *reasoning_out = NULL;
        }
    }
    if (parsed_ok) {
        if (cut) {
            if (has_tools) strip_partial_suffix(*content_out, P36_TOOL_CALLS_START);
            strip_partial_suffix(*reasoning_out, "</think>");
        }
        blank_if_whitespace(*content_out);
        return true;
    }

    free(*content_out);
    free(*reasoning_out);
    *content_out = xstrdup(text ? text : "");
    *reasoning_out = NULL;
    tool_calls_free(calls);

    /* A malformed tool block is model output, not a server failure.  Returning
     * the sampled text as a regular assistant message keeps the API response
     * valid and gives the caller enough transcript context to retry or recover
     * on a later turn, instead of terminating the whole session on
     * finish_reason="error". */
    const char *finish = fin;
    if (has_tools && saw_tool_start && strcmp(finish, "error") != 0) {
        if (finish_io) *finish_io = tool_parse_failure_recovery_finish(finish);
        if (err && errlen) snprintf(err, errlen, "invalid tool call");
        if (recovered_out) *recovered_out = true;
    }
    return false;
}

static void append_json_object_string(buf *b, const char *json) {
    buf tmp = {0};
    append_json_object_or_empty(&tmp, json);
    json_escape(b, tmp.ptr ? tmp.ptr : "{}");
    buf_free(&tmp);
}

static void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

static void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"index\":");
        buf_printf(b, "%d", i);
        buf_puts(b, ",\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

static bool http_response(int fd, int code, const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" :
                         code == 400 ? "Bad Request" :
                         code == 404 ? "Not Found" :
                         code == 500 ? "Internal Server Error" : "Error";
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n",
        code, reason, type, strlen(body));
    if (g_enable_cors) {
        buf_puts(&h,
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n");
    }
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len) && send_all(fd, body, strlen(body));
    buf_free(&h);
    return ok;
}

static bool http_error(int fd, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, msg);
    buf_puts(&b, ",\"type\":\"invalid_request_error\"}}\n");
    bool ok = http_response(fd, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

