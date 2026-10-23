static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    if (!strcmp(name, "google_search")) return "google ";
    if (!strcmp(name, "visit_page")) return "visit ";
    return NULL;
}

static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    renderer_color(sr->renderer, !strcmp(v->tool_name, "bash") ?
                                "\x1b[1;36m" : "\x1b[1;37m");
    const char *prefix = agent_tool_viz_prefix(v->tool_name);
    if (prefix) {
        agent_tool_viz_puts(sr, prefix);
    } else {
        agent_tool_viz_puts(sr, v->tool_name);
        agent_tool_viz_puts(sr, " ");
    }
    renderer_color(sr->renderer, "\x1b[0m");
}

static void agent_tool_viz_append(char *dst, size_t cap, char c) {
    size_t len = strlen(dst);
    if (len + 1 >= cap) return;
    dst[len] = c;
    dst[len + 1] = '\0';
}

static void agent_tool_viz_read_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (!strcmp(v->param_name, "path")) {
        agent_tool_viz_append(v->read_path, sizeof(v->read_path), c);
        if (v->read_prefix_rendered) agent_tool_viz_write(sr, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}

static void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, "<unknown>");
    }
    renderer_color(sr->renderer, "\x1b[33m");
    bool whole = agent_parse_bool_default(v->read_whole, false);
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start);
        agent_tool_viz_puts(sr, ":EOF");
    } else {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : "500");
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}

static bool agent_tool_viz_param_is_code_body(agent_tool_visualizer *v) {
    if (!strcmp(v->tool_name, "write") &&
        v->param_kind == AGENT_TOOL_PARAM_CONTENT)
        return true;
    if (!strcmp(v->tool_name, "edit") &&
        (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
         v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW ||
         v->param_kind == AGENT_TOOL_PARAM_CONTENT))
        return true;
    return false;
}

static const char *agent_tool_viz_diff_prefix(agent_tool_param_kind kind,
                                              const char **color) {
    if (color) *color = NULL;
    const char *prefix = NULL;
    if (kind == AGENT_TOOL_PARAM_DIFF_OLD) {
        prefix = "- ";
        if (color) *color = "\x1b[31m";
    } else if (kind == AGENT_TOOL_PARAM_DIFF_NEW) {
        prefix = "+ ";
        if (color) *color = "\x1b[32m";
    }
    return prefix;
}

static void agent_tool_viz_code_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    const char *color = NULL;
    const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
    if (!prefix) return;
    renderer_color(sr->renderer, color);
    renderer_write(sr->renderer, prefix, strlen(prefix));
    renderer_color(sr->renderer, "\x1b[0m");
    sr->renderer->wrote_visible_output = true;
    sr->renderer->last_output_newline = false;
    v->last_output_newline = false;
    v->at_line_start = false;
}

static void agent_tool_viz_code_begin(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    const agent_syntax *syntax = agent_syntax_for_path(v->tool_path);
    renderer_code_stream_begin(sr->renderer, syntax);
    renderer_code_stream_set_upto_marker(sr->renderer,
        !strcmp(v->tool_name, "edit") &&
        v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD);
    v->code_param_active = true;
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        const char *color = NULL;
        const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
        /* Diff prefixes are terminal UI, not code.  Keep them outside the
         * syntax buffer so a later row repaint preserves their red/green color
         * while highlighting only the actual edited line. */
        renderer_code_stream_set_prefix(sr->renderer, prefix, color);
        agent_tool_viz_code_prefix(sr);
    }
}

static void agent_tool_viz_code_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->code_param_active) return;
    renderer_code_end(sr->renderer);
    v->code_param_active = false;
    v->at_line_start = true;
    v->last_output_newline = sr->renderer->last_output_newline;
}

static void agent_tool_viz_code_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    agent_tool_viz_code_prefix(sr);
    renderer_code_byte(sr->renderer, c);
    v->last_output_newline = c == '\n';
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->parser->current.name)
        agent_tool_viz_tool(sr, sr->parser->current.name);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->at_line_start = true;
        agent_tool_viz_code_begin(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        if (strcmp(v->tool_name, "write")) {
            renderer_color(sr->renderer, "\x1b[1;37m");
            agent_tool_viz_puts(sr, v->param_name);
            agent_tool_viz_puts(sr, ":\n");
        }
        v->at_line_start = true;
        if (agent_tool_viz_param_is_code_body(v)) {
            agent_tool_viz_code_begin(sr);
        } else {
            renderer_color(sr->renderer, "\x1b[34m");
        }
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    } else {
        renderer_color(sr->renderer, agent_tool_param_color(AGENT_TOOL_PARAM_BASH_COMMAND));
        return;
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    if (v->code_param_active) agent_tool_viz_code_end(sr);
    if (!v->read_style) renderer_color(sr->renderer, "\x1b[0m");
    v->param_active = false;
    v->param_name[0] = '\0';
}

static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_PATH) {
        agent_tool_viz_append(v->tool_path, sizeof(v->tool_path), c);
    }
    if (v->code_param_active) {
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_BASH_COMMAND) {
        agent_tool_viz_write(sr, &c, 1);
        v->at_line_start = c == '\n';
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_code_begin(sr);
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, &c, 1);
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_restore_param_color(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active || !v->param_active || v->read_style) return;
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

/* Stream one Qwen tool parameter byte into the visualizer.  The visualizer must not
 * wait for the whole parameter: large write/edit contents should show progress
 * as the model emits them, while still detecting the closing parameter tag. */
static void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len == sizeof(v->param_end_tail)) {
            size_t keep = v->param_end_len;
            v->param_end_len = 0;
            for (size_t i = 0; i < keep; i++)
                agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
            if (c != '<') {
                agent_tool_viz_param_raw_byte(sr, c);
                return;
            }
        }
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_qwen_tool_parameter_close_tail(v->param_end_tail, v->param_end_len, &complete)) {
            if (complete) agent_tool_viz_param_end(sr);
            return;
        }
        size_t keep = v->param_end_len;
        v->param_end_len = 0;
        for (size_t i = 0; i < keep; i++)
            agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
        return;
    }
    agent_tool_viz_param_raw_byte(sr, c);
}

static void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[90m");
        agent_tool_viz_puts(sr, status);
        renderer_color(sr->renderer, "\x1b[0m");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    v->active = false;
}

static void agent_tool_viz_dump_invalid_qwen_tool(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;

    /* The normal path hides Qwen tool and paints a friendly semantic projection.  If
     * parsing fails, show the exact bytes we rejected so the next fix is based
     * on evidence instead of guessing from the projection. */
    if (v->param_active) {
        v->param_active = false;
        v->param_end_len = 0;
        v->param_name[0] = '\0';
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    renderer_color(sr->renderer, "\x1b[1;31m");
    if (sr->parser->raw && sr->parser->raw_len) {
        agent_tool_viz_write(sr, sr->parser->raw, sr->parser->raw_len);
    } else {
        agent_tool_viz_puts(sr, "<empty Qwen tool>");
    }
    renderer_color(sr->renderer, "\x1b[0m");
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
}

static void agent_stream_finish_ignored_qwen_tool(agent_stream_renderer *sr, const char *detail) {
    const char *msg =
        detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    sr->qwen_tool_in_think = true;
    sr->qwen_tool_in_think_reported = true;
    agent_trace(sr->renderer->worker, "qwen_tool ignored inside thinking: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[tool call ignored: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
    agent_qwen_tool_parser_reset(sr->parser);
    sr->qwen_tool_active = false;
    sr->qwen_tool_ignored = false;
}

static void agent_stream_malformed_qwen_tool(agent_stream_renderer *sr,
                                        const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "Qwen tool markup outside a valid tool_call block";
    if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) return;
    agent_qwen_tool_set_error(sr->parser, msg);
    agent_trace(sr->renderer->worker, "malformed qwen_tool in assistant output: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[invalid tool call: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
}

/* Mirror parser progress into the terminal visualizer.  Parser state is the
 * source of truth; this function only decides what the user should see. */
static void agent_stream_tool_events(agent_stream_renderer *sr) {
    agent_qwen_tool_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && p->current.name)
        agent_tool_viz_tool(sr, p->current.name);
    if (!v->tool_announced && !p->current.name &&
        p->state == AGENT_QWEN_TOOL_DONE && p->calls.len > 0)
    {
        for (int ci = 0; ci < p->calls.len; ci++) {
            const agent_tool_call *call = &p->calls.v[ci];
            agent_tool_viz_tool(sr, call->name);
            for (int ai = 0; ai < call->argc; ai++) {
                agent_tool_viz_param_begin(sr, call->args[ai].name);
                const char *val = call->args[ai].value ? call->args[ai].value : "";
                for (size_t k = 0; val[k]; k++)
                    agent_tool_viz_param_raw_byte(sr, val[k]);
                agent_tool_viz_param_end(sr);
            }
        }
    }
    if (v->tool_announced && !p->current.name && !v->param_active) {
        agent_tool_viz_render_read(sr);
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->read_style = false;
        v->read_prefix_rendered = false;
        v->read_line_rendered = false;
        v->read_path[0] = '\0';
        v->read_start[0] = '\0';
        v->read_max[0] = '\0';
        v->read_whole[0] = '\0';
        v->tool_announced = false;
    }
    if (!v->param_active && p->state == AGENT_QWEN_TOOL_PARAM_VALUE && p->param_name)
        agent_tool_viz_param_begin(sr, p->param_name);
}

static void agent_stream_preflight_closed_param(agent_stream_renderer *sr) {
    if (!sr || sr->replay || sr->qwen_tool_ignored || sr->tool_preflight_error)
        return;
    agent_qwen_tool_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!p || !v->param_active || strcmp(v->param_name, "old") != 0)
        return;
    if (!p->current.name || strcmp(p->current.name, "edit") != 0)
        return;

    char err[256] = {0};
    if (agent_preflight_edit_old(sr->renderer->worker, &p->current,
                                 err, sizeof(err)))
        return;

    sr->tool_preflight_error = true;
    snprintf(sr->tool_preflight_error_msg, sizeof(sr->tool_preflight_error_msg),
             "edit old selector failed before new was generated: %.200s",
             err[0] ? err : "old text is not a unique match");
    agent_trace(sr->renderer->worker, "edit old preflight failed: %s",
                sr->tool_preflight_error_msg);
}

static void agent_stream_feed_qwen_tool_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->qwen_tool_ignored && sr->viz.param_active;
    agent_qwen_tool_feed(sr->parser, &c, 1);
    if (!sr->qwen_tool_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
        if (was_param && sr->parser->state != AGENT_QWEN_TOOL_PARAM_VALUE &&
            sr->viz.param_active)
        {
            agent_stream_preflight_closed_param(sr);
            agent_tool_viz_param_end(sr);
        }
    }
    if (sr->parser->state == AGENT_QWEN_TOOL_DONE) {
        if (sr->qwen_tool_ignored) {
            agent_stream_finish_ignored_qwen_tool(
                sr, "tool calling is not allowed inside <think></think>");
        } else {
            agent_trace(sr->renderer->worker, "qwen_tool done calls=%d",
                        sr->parser->calls.len);
            agent_tool_viz_finish(sr, NULL);
            sr->qwen_tool_active = false;
        }
    } else if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) {
        if (sr->qwen_tool_ignored) {
            agent_stream_finish_ignored_qwen_tool(
                sr, "malformed tool call inside <think></think>");
        } else {
            char status[220];
            snprintf(status, sizeof(status), "[invalid tool call: %s]\n",
                     sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_trace(sr->renderer->worker, "qwen_tool error %s",
                        sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_tool_viz_dump_invalid_qwen_tool(sr);
            agent_tool_viz_finish(sr, status);
            sr->qwen_tool_active = false;
        }
    }
}

/* Start a Qwen3.6 tool block once its complete opening marker is buffered. */
static void agent_stream_start_qwen_tool(agent_stream_renderer *sr, bool ignored) {
    sr->qwen_tool_active = true;
    sr->qwen_tool_ignored = ignored;
    if (ignored) sr->qwen_tool_in_think = true;
    sr->qwen_tool_start_len = 0;
    sr->post_think_gap = false;
    agent_trace(sr->renderer->worker, "qwen_tool start detected%s",
                ignored ? " inside thinking" : "");
    agent_qwen_tool_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}

static void agent_stream_note_plain_qwen_tool_byte(agent_stream_renderer *sr, char c);

static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->qwen_tool_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->qwen_tool_start_len; i++) {
        renderer_write_char(sr->renderer, sr->qwen_tool_start_tail[i]);
        agent_stream_note_plain_qwen_tool_byte(sr, sr->qwen_tool_start_tail[i]);
        if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) break;
    }
    sr->qwen_tool_start_len = 0;
}

static bool agent_stream_qwen_tool_start_match(const char *tail, size_t len,
                                          bool *complete) {
    static const char canonical[] = "<tool_call>";
    *complete = false;
    size_t canonical_len = sizeof(canonical) - 1;
    if (len <= canonical_len && memcmp(canonical, tail, len) == 0) {
        *complete = len == canonical_len;
        return true;
    }
    return false;
}

static bool agent_tail_matches(const char *tail, size_t len,
                               const char *needle, size_t needle_len) {
    return len >= needle_len &&
           memcmp(tail + len - needle_len, needle, needle_len) == 0;
}

/* Detect tool-call markup not currently owned by the executable parser. Inside
 * <think> the marker means "tool call attempted too early", while in normal
 * assistant output it becomes a retryable tool error. */
static bool agent_qwen_tool_marker_detector_feed(agent_qwen_tool_marker_detector *d,
                                            char c) {
    if (d->len == sizeof(d->tail)) {
        memmove(d->tail, d->tail + 1, sizeof(d->tail) - 1);
        d->len--;
    }
    d->tail[d->len++] = c;

    static const char *markers[] = {
        "<function=", "<parameter=", "</parameter>",
        "</function>", "</tool_call>"
    };
    for (size_t i = 0; i < sizeof(markers) / sizeof(markers[0]); i++) {
        if (agent_tail_matches(d->tail, d->len, markers[i], strlen(markers[i])))
            return true;
    }
    return false;
}

static void agent_stream_note_thinking_qwen_tool_byte(agent_stream_renderer *sr,
                                                 char c) {
    if (!sr->in_think || sr->qwen_tool_in_think) return;
    if (agent_qwen_tool_marker_detector_feed(&sr->think_qwen_tool, c))
        sr->qwen_tool_in_think = true;
}

static void agent_stream_note_plain_qwen_tool_byte(agent_stream_renderer *sr,
                                              char c) {
    if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) return;
    if (sr->qwen_tool_active || sr->in_think || sr->qwen_tool_in_think) return;
    if (agent_qwen_tool_marker_detector_feed(&sr->plain_qwen_tool, c)) {
        agent_stream_malformed_qwen_tool(
            sr, "Qwen tool markup outside a valid tool_call block");
    }
}

/* Route ordinary assistant bytes either to normal markdown rendering or into
 * the Qwen tool detector. The opening marker may be split across tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char start[] = "<tool_call>";
    if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) return;
    agent_stream_note_thinking_qwen_tool_byte(sr, c);

    /* Qwen may emit blank lines after </think> before prose or a tool call. They are
     * visual gap between the hidden thinking phase and the real answer, and
     * printing them makes tool calls appear after odd empty lines.  We only
     * suppress whitespace in this very narrow post-thinking window; once the
     * first non-space byte arrives, normal rendering resumes. */
    if (sr->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
    {
        return;
    }

    if (sr->qwen_tool_start_len || c == start[0]) {
        if (sr->qwen_tool_start_len < sizeof(sr->qwen_tool_start_tail))
            sr->qwen_tool_start_tail[sr->qwen_tool_start_len++] = c;
        bool complete = false;
        if (agent_stream_qwen_tool_start_match(sr->qwen_tool_start_tail, sr->qwen_tool_start_len,
                                               &complete))
        {
            if (complete) {
                agent_stream_start_qwen_tool(sr, sr->in_think);
            }
            return;
        }
        if (sr->qwen_tool_start_len > 1 &&
            sr->qwen_tool_start_tail[sr->qwen_tool_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->qwen_tool_start_len - 1;
            for (size_t i = 0; i < flush; i++) {
                renderer_write_char(sr->renderer, sr->qwen_tool_start_tail[i]);
                agent_stream_note_plain_qwen_tool_byte(sr, sr->qwen_tool_start_tail[i]);
                if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) break;
            }
            if (sr->parser->state == AGENT_QWEN_TOOL_ERROR) {
                sr->qwen_tool_start_len = 0;
                return;
            }
            sr->qwen_tool_start_tail[0] = start[0];
            sr->qwen_tool_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    renderer_write_char(sr->renderer, c);
    agent_stream_note_plain_qwen_tool_byte(sr, c);
}

/* This is the single streaming display state machine for assistant output.  It
 * hides raw Qwen tool as soon as the tool_calls marker is complete, lets the Qwen tool
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten. */
static void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = sr->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (sr->pending_len) memcpy(buf, sr->pending, sr->pending_len);
    if (len) memcpy(buf + sr->pending_len, text, len);
    sr->pending_len = 0;

    /* The UI may reset terminal attributes while redrawing the editable prompt
     * between generated chunks.  If a Qwen tool parameter is still streaming, make
     * each new token fragment self-contained by restoring the active parameter
     * color before visible bytes are projected.  This keeps the prompt normal
     * without sacrificing long write/edit content coloring. */
    if (len) agent_tool_viz_restore_param_color(sr);
    if (len && !sr->qwen_tool_active) renderer_restore_text_attrs(sr->renderer);

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!sr->qwen_tool_active && bytes_has_prefix(cur, rem, think_open)) {
            agent_stream_flush_start_tail(sr);
            renderer_hint_end(sr->renderer);
            sr->post_think_gap = false;
            sr->in_think = true;
            sr->renderer->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!sr->qwen_tool_active && bytes_has_prefix(cur, rem, think_close)) {
            agent_stream_flush_start_tail(sr);
            renderer_hint_end(sr->renderer);
            sr->in_think = false;
            sr->renderer->in_think = false;
            renderer_reset_color(sr->renderer);
            if (!sr->renderer->last_output_newline)
                renderer_write(sr->renderer, "\n", 1);
            renderer_write(sr->renderer, "\n", 1);
            sr->renderer->last_output_newline = true;
            sr->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !sr->qwen_tool_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(sr->pending)) {
                memcpy(sr->pending, cur, rem);
                sr->pending_len = rem;
            }
            break;
        }

        if (sr->qwen_tool_active) {
            agent_stream_feed_qwen_tool_byte(sr, cur[0]);
        } else if (sr->in_think) {
            /* Tool calls are executable only after thinking has closed.  Still
             * route thinking bytes through the Qwen tool start detector so an
             * accidental in-think tool stanza can be suppressed cleanly instead
             * of being shown as raw markup or, worse, executed. */
            agent_stream_normal_byte(sr, cur[0]);
        } else {
            agent_stream_normal_byte(sr, cur[0]);
        }
        i++;
    }
    free(buf);

    if (finish) {
        agent_stream_flush_start_tail(sr);
        sr->post_think_gap = false;
        if (sr->qwen_tool_active) {
            if (sr->qwen_tool_ignored) {
                agent_stream_finish_ignored_qwen_tool(
                    sr, "tool calling is not allowed inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, sr->tool_preflight_error ?
                                      "[tool call stopped: edit old selector failed]\n" :
                                      "[tool call interrupted]\n");
                sr->qwen_tool_active = false;
            }
        }
        if (sr->qwen_tool_in_think && !sr->qwen_tool_in_think_reported) {
            agent_stream_finish_ignored_qwen_tool(
                sr, "tool calling is not allowed inside <think></think>");
        }
    }
}

/* ============================================================================
 * Worker Progress And Generic Buffers
 * ============================================================================
 */

static void worker_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    agent_worker *w = ud;
    if (!w || !event) return;
    if (strcmp(event, "prefill_chunk") && strcmp(event, "prefill_display")) return;
    worker_apply_pending_power(w);
    pthread_mutex_lock(&w->mu);
    int done = current - w->progress_base;
    if (done < 0) done = 0;
    if (done > w->status.prefill_total) done = w->status.prefill_total;
    w->status.prefill_done = done;
    double elapsed = now_sec() - w->progress_started_at;
    w->status.prefill_tps =
        done > 0 && elapsed > 0.0 ? (double)done / elapsed : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
}

/* Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending. */
static void worker_clear_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}

static bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}

static bool worker_cancel_session_cb(void *ud) {
    return worker_should_interrupt(ud);
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    size_t limit;
    bool truncated;
} agent_buf;

#define AGENT_TOOL_MAX_BYTES (128 * 1024)

static void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = b->limit ? b->limit : SIZE_MAX - 1;
    if (n > max - b->len) {
        n = max > b->len ? max - b->len : 0;
        while (n && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

static char *agent_buf_take(agent_buf *b) {
    if (b->truncated) {
        b->truncated = false;
        b->limit = 0;
        agent_buf_puts(b, "\n[Output truncated at the tool byte limit. Narrow the request.]\n");
    }
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static char *agent_tool_response_add_warning(char *content, int *failures) {
    if (agent_tool_response_is_error(content)) (*failures)++;
    else *failures = 0;
    if (*failures == 0) return content;

    agent_buf out = {0};
    agent_buf_puts(&out, content);
    if (*failures >= 2) {
        char count[32];
        snprintf(count, sizeof(count), "%d", *failures);
        agent_buf_puts(&out,
            "\n\n⚠️ SYSTEM WARNING: ");
        agent_buf_puts(&out, count);
        agent_buf_puts(&out,
            " consecutive tool errors detected. Your previous approach is "
            "incorrect. You MUST use a fundamentally "
            "different approach or corrected arguments.");
    } else {
        agent_buf_puts(&out,
            "\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. "
            "Diagnose the failure and retry with completely corrected arguments.");
    }
    free(content);
    return agent_buf_take(&out);
}

static bool agent_tokens_equal(const p36_tokens *a, const p36_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}

static bool agent_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static char *agent_default_cache_dir(void) {
    const char *cache = getenv("P36_AGENT_CACHE_DIR");
    if (cache && cache[0]) return xstrdup(cache);
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    agent_buf b = {0};
    agent_buf_puts(&b, home);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') agent_buf_puts(&b, "/");
    agent_buf_puts(&b, ".p36/kvcache");
    return agent_buf_take(&b);
}

static char *agent_kv_path_for_sha(const char *dir, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return p36_kvstore_path_join(dir, name);
}

static void agent_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* Agent session IDs are intentionally independent from the rendered transcript:
 * once a session has a title and creation time, resaving it keeps the same file
 * name while the transcript and KV payload evolve. */
static void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]) {
    size_t title_len = title ? strlen(title) : 0;
    agent_buf b = {0};
    agent_buf_append(&b, title ? title : "", title_len);
    uint8_t ts[8];
    agent_le_put64(ts, created_at);
    agent_buf_append(&b, (const char *)ts, sizeof(ts));
    p36_kvstore_sha1_bytes_hex(b.ptr ? b.ptr : "", b.len, sha_out);
    free(b.ptr);
}

static void agent_worker_clear_session_identity(agent_worker *w) {
    w->session_sha[0] = '\0';
    free(w->session_title);
    w->session_title = NULL;
    w->session_created_at = 0;
    free(w->legacy_session_path_to_delete);
    w->legacy_session_path_to_delete = NULL;
}

typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

