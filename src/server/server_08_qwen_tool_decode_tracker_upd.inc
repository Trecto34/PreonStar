static void qwen_tool_decode_tracker_update(qwen_tool_decode_tracker *dt,
                                       const char *raw, size_t raw_len) {
    if (!dt || !raw) return;

    for (;;) {
        if (dt->mode == QWEN_TOOL_TRACK_DONE) {
            size_t next = 0;
            const qwen_tool_syntax *syn = NULL;
            if (!qwen_tool_find_tool_start_from(raw, raw_len, dt->pos, &next, &syn)) return;
            dt->pos = next;
            dt->mode = QWEN_TOOL_TRACK_SEARCH;
            continue;
        }

        if (dt->mode == QWEN_TOOL_TRACK_SEARCH) {
            size_t pos = 0;
            const qwen_tool_syntax *syn = NULL;
            if (!qwen_tool_find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn)) {
                size_t hold = qwen_tool_max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = QWEN_TOOL_DECODE_OUTSIDE;
                return;
            }
            dt->syn = syn;
            dt->pos = pos;
            dt->mode = QWEN_TOOL_TRACK_STRUCTURAL;
            dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
        }

        if (dt->mode == QWEN_TOOL_TRACK_PAYLOAD) {
            while (dt->pos < raw_len) {
                if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->pos += strlen(dt->syn->param_end);
                    dt->mode = QWEN_TOOL_TRACK_STRUCTURAL;
                    dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                    goto structural;
                }
                if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->decode = raw_len - dt->pos >= 2 ?
                        QWEN_TOOL_DECODE_STRUCTURAL : QWEN_TOOL_DECODE_PAYLOAD;
                    return;
                }
                dt->pos++;
            }
            dt->decode = QWEN_TOOL_DECODE_PAYLOAD;
            return;
        }

structural:
        while (dt->mode == QWEN_TOOL_TRACK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) {
                dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                return;
            }

            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end)) {
                dt->mode = QWEN_TOOL_TRACK_DONE;
                dt->pos += strlen(dt->syn->tool_calls_end);
                dt->decode = QWEN_TOOL_DECODE_OUTSIDE;
                return;
            }
            if (raw[dt->pos] == '{') {
                const char *end = find_lit_bounded(raw + dt->pos, raw_len - dt->pos,
                                                   dt->syn->tool_calls_end);
                if (!end) {
                    dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos = (size_t)(end - raw);
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_end)) {
                dt->pos += strlen(dt->syn->invoke_end);
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_start)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos = (size_t)(tag_end - raw) + 1;
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_start)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                    return;
                }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                dt->pos = tag_after;
                dt->mode = QWEN_TOOL_TRACK_PAYLOAD;
                dt->decode = QWEN_TOOL_DECODE_PAYLOAD;
                break;
            }

            if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_end))
            {
                dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
                return;
            }

            dt->decode = QWEN_TOOL_DECODE_STRUCTURAL;
            return;
        }
    }
}

static void observe_tool_markers(const qwen_tool_decode_tracker *tracker,
                                 const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end) {
    if (tracker->syn) *saw_start = true;
    if (tracker->mode == QWEN_TOOL_TRACK_DONE) *saw_end = true;
    else if (!*saw_start && scan && find_any_tool_end(scan) && orphan_end)
        *orphan_end = true;
}

static bool complete_tool_text(const char *text) {
    if (!text || !find_any_tool_end(text)) return false;
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    bool complete = parse_generated_message_ex(text, false,
        &content, &reasoning, &calls) && calls.len > 0;
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    return complete;
}

static bool openai_tool_emit_args_fragment(int fd, const request *r, const char *id,
                                           openai_tool_stream *ts,
                                           const char *text, size_t len) {
    return sse_chat_tool_call_args_delta_n(fd, r, id, ts->index, text, len);
}

static bool openai_tool_emit_string_value(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf frag = {0};
    json_escape_fragment_n(&frag, text, len);
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}

static bool openai_tool_emit_param_prefix(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}

static bool openai_tool_stream_init(openai_tool_stream *ts, const char *raw,
                                    size_t raw_len, size_t pos) {
    openai_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = OPENAI_TOOL_BETWEEN_INVOKES;
    ts->parse_pos = pos;
    if (raw_full_lit(raw, raw_len, pos, P36_TOOL_CALLS_START)) {
        ts->parse_pos += strlen(P36_TOOL_CALLS_START);
        ts->tool_calls_end = P36_TOOL_CALLS_END;
        ts->invoke_start = P36_INVOKE_START;
        ts->invoke_end = P36_INVOKE_END;
        ts->param_start = P36_PARAM_START;
        ts->param_end = P36_PARAM_END;
    } else {
        ts->active = false;
        ts->state = OPENAI_TOOL_ERROR;
        return false;
    }
    return true;
}

static bool openai_tool_stream_fail(openai_tool_stream *ts) {
    ts->active = false;
    ts->state = OPENAI_TOOL_ERROR;
    return true;
}

static bool openai_tool_start_invoke(int fd, server *s, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t raw_len) {
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = qwen_tool_tag_value(tag, ts->invoke_start);
    free(tag);
    if (!name) return openai_tool_stream_fail(ts);

    const char *tool_id = openai_tool_stream_id(s, ts, ts->index);
    bool ok = sse_chat_tool_call_start_delta(fd, r, id, ts->index, tool_id, name) &&
              openai_tool_emit_args_fragment(fd, r, id, ts, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = OPENAI_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_start_param(int fd, const request *r, const char *id,
                                    openai_tool_stream *ts,
                                    const char *raw, size_t raw_len) {
    (void)fd;
    (void)r;
    (void)id;
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = qwen_tool_tag_value(tag, ts->param_start);
    free(tag);
    if (!name) {
        free(name);
        return openai_tool_stream_fail(ts);
    }
    free(ts->param_name);
    ts->param_name = name;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = OPENAI_TOOL_PARAM_VALUE;
    return true;
}

static bool openai_tool_finish_param(int fd, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t value_end) {
    const char *value = raw + ts->parse_pos;
    size_t len = value_end - ts->parse_pos;
    if (len && value[0] == '\n') {
        value++;
        len--;
    }
    if (len && value[len - 1] == '\n') len--;
    char *plain = xstrndup(value, len);
    p36_tool_text_unescape(plain, P36_PARAM_END);
    bool json_value = json_raw_value_is_complete(plain);
    if (!openai_tool_emit_param_prefix(fd, r, id, ts, ts->param_name,
                                       !json_value)) {
        free(plain);
        return false;
    }
    bool ok;
    if (json_value) {
        char *minified = json_minify_raw_value(plain);
        ok = openai_tool_emit_args_fragment(fd, r, id, ts,
                                            minified, strlen(minified));
        free(minified);
    } else {
        ok = openai_tool_emit_string_value(fd, r, id, ts, plain, strlen(plain)) &&
             openai_tool_emit_args_fragment(fd, r, id, ts, "\"", 1);
    }
    free(plain);
    if (!ok) return false;
    free(ts->param_name);
    ts->param_name = NULL;
    ts->parse_pos = value_end + strlen(ts->param_end);
    ts->state = OPENAI_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_stream_update(int fd, server *s, const request *r, const char *id,
                                      openai_tool_stream *ts,
                                      const char *raw, size_t raw_len) {
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == OPENAI_TOOL_BETWEEN_CALLS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos]))
                ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_partial_lit(raw, raw_len, ts->parse_pos, P36_TOOL_CALLS_START))
                return true;
            if (!raw_full_lit(raw, raw_len, ts->parse_pos, P36_TOOL_CALLS_START))
                return openai_tool_stream_fail(ts);
            ts->parse_pos += strlen(P36_TOOL_CALLS_START);
            ts->state = OPENAI_TOOL_BETWEEN_INVOKES;
            continue;
        }

        if (ts->state == OPENAI_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->tool_calls_end)) {
                ts->parse_pos += strlen(ts->tool_calls_end);
                ts->state = OPENAI_TOOL_BETWEEN_CALLS;
                continue;
            }
            if (raw[ts->parse_pos] == '{') {
                const char *end = find_lit_bounded(raw + ts->parse_pos,
                                                   raw_len - ts->parse_pos,
                                                   ts->tool_calls_end);
                if (!end) return true;
                char *json_body = xstrndup(raw + ts->parse_pos,
                                           (size_t)(end - (raw + ts->parse_pos)));
                const char *p = json_body;
                tool_call tc = {0};
                bool valid = parse_json_tool_call(&p, &tc);
                if (valid) {
                    json_ws(&p);
                    if (*p != '\0') valid = false;
                }
                free(json_body);
                if (!valid) {
                    tool_call_free(&tc);
                    return openai_tool_stream_fail(ts);
                }
                const char *tool_id = openai_tool_stream_id(s, ts, ts->index);
                bool ok = sse_chat_tool_call_start_delta(fd, r, id, ts->index, tool_id, tc.name) &&
                          openai_tool_emit_args_fragment(fd, r, id, ts, tc.arguments, strlen(tc.arguments));
                tool_call_free(&tc);
                if (!ok) return false;
                ts->emitted_any = true;
                ts->index++;
                ts->parse_pos = (size_t)(end - raw) + strlen(ts->tool_calls_end);
                ts->state = OPENAI_TOOL_BETWEEN_CALLS;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->tool_calls_end, ts->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                openai_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_invoke(fd, s, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == OPENAI_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_end)) {
                if (ts->args_open &&
                    !openai_tool_emit_args_fragment(fd, r, id, ts, "}", 1)) return false;
                ts->args_open = false;
                ts->parse_pos += strlen(ts->invoke_end);
                ts->index++;
                ts->state = OPENAI_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->invoke_end, ts->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->param_start)) {
                size_t before_pos = ts->parse_pos;
                openai_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_param(fd, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == OPENAI_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos,
                                               raw_len - ts->parse_pos,
                                               ts->param_end);
            if (end) {
                if (!openai_tool_finish_param(fd, r, id, ts, raw,
                                              (size_t)(end - raw))) return false;
                continue;
            }
            return true;
        }

        return true;
    }
    return true;
}

static bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == OPENAI_STREAM_THINKING) {
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
            if (!sse_chat_delta_n(fd, r, id, "reasoning_content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_reasoning = true;
            st->emit_pos = limit;
        }

        if (tool_before_close) {
            if (complete_tool) {
                st->emit_pos = (size_t)(tool - raw);
                st->mode = OPENAI_STREAM_SUPPRESS;
            }
            return true;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = OPENAI_STREAM_TEXT;
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == OPENAI_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_content = true;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            if (openai_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = OPENAI_STREAM_TOOL;
            } else {
                st->mode = OPENAI_STREAM_SUPPRESS;
            }
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
        }
    }

    if (st->mode == OPENAI_STREAM_TOOL) {
        if (!openai_tool_stream_update(fd, s, r, id, &st->tool, raw, raw_len)) return false;
        if (final && st->tool.state == OPENAI_TOOL_BETWEEN_CALLS) {
            st->tool.active = false;
            st->tool.state = OPENAI_TOOL_DONE;
        }
        if (!st->tool.active) st->mode = OPENAI_STREAM_SUPPRESS;
    }
    return true;
}

static bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens) {
    if (!openai_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (calls && calls->len && !st->tool.emitted_any) {
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

static bool request_uses_openai_live_stream(const request *r) {
    return r->stream && r->api == API_OPENAI && r->kind == REQ_CHAT;
}

static bool request_uses_structured_stream(const request *r) {
    return r->stream && (r->api == API_ANTHROPIC ||
                         r->api == API_RESPONSES ||
                         request_uses_openai_live_stream(r));
}

