static char *render_qwen_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                          const tool_schema_orders *tool_orders,
                                          p36_think_mode think_mode,
                                          bool preserve_thinking,
                                          const char *reasoning_instruction) {
    bool thinking = p36_think_mode_enabled(think_mode);
    int last_query = last_chat_query(msgs);
    int failures = 0;

    buf out = {0};
    if (think_mode == P36_THINK_MAX) buf_puts(&out, p36_think_max_prefix());
    int system_head = 0;
    char *first_content = leading_system_content(msgs, &system_head, &thinking);
    bool has_tools = tool_schemas && tool_schemas[0];
    bool has_content = text_has_trimmed_content(first_content);
    if (has_tools || has_content || reasoning_instruction) {
        buf_puts(&out, "<|im_start|>system\n");
        if (reasoning_instruction) {
            buf_puts(&out, reasoning_instruction);
            if (has_tools || has_content) buf_puts(&out, "\n\n");
        }
        if (has_tools) {
            append_tools_prompt_text(&out, tool_schemas);
            if (has_content) buf_puts(&out, "\n\n");
        }
        if (has_content) buf_puts(&out, first_content);
        buf_puts(&out, "<|im_end|>\n");
    }
    free(first_content);

    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            if (i < system_head) continue;
            char *content = chat_content_text(m->content, true, &thinking);
            buf_puts(&out, "<|im_start|>system\n");
            buf_puts(&out, content);
            buf_puts(&out, "<|im_end|>\n");
            free(content);
        } else if (!strcmp(m->role, "user")) {
            char *content = chat_content_text(m->content, true, &thinking);
            failures = 0;
            buf_puts(&out, "<|im_start|>user\n");
            buf_puts(&out, content);
            buf_puts(&out, "<|im_end|>\n");
            free(content);
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            bool first = i == 0 || (strcmp(msgs->v[i - 1].role, "tool") &&
                                    strcmp(msgs->v[i - 1].role, "function"));
            bool last = i + 1 == msgs->len ||
                        (strcmp(msgs->v[i + 1].role, "tool") &&
                         strcmp(msgs->v[i + 1].role, "function"));
            char *content = chat_content_text(m->content, false, NULL);
            failures = tool_response_is_error(content) ? failures + 1 : 0;
            if (first) buf_puts(&out, "<|im_start|>user");
            buf_puts(&out, "\n<tool_response>\n");
            buf_puts(&out, content);
            append_tool_error_warning(&out, failures);
            buf_puts(&out, "\n</tool_response>");
            if (last) buf_puts(&out, "<|im_end|>\n");
            free(content);
        } else if (!strcmp(m->role, "assistant")) {
            char *reasoning = NULL;
            char *content = assistant_message_parts(m, &reasoning);
            bool keep_reasoning = (preserve_thinking || i > last_query) &&
                                  text_has_trimmed_content(reasoning);
            bool content_nonempty = text_has_trimmed_content(content);
            buf_puts(&out, "<|im_start|>assistant\n");
            if (keep_reasoning) {
                buf_puts(&out, "<think>\n");
                buf_puts(&out, reasoning);
                buf_puts(&out, "\n</think>\n\n");
            }
            /* With thinking off the generation prompt writes a closed empty
             * think block.  When reasoning is preserved, history must keep that
             * same block: it is in the live KV graph, and dropping it makes the
             * next prompt stop being a token/byte prefix, which forces a
             * full-context re-prefill in canonicalize_thinking_checkpoint(). */
            if (!keep_reasoning && (m->calls.replay_empty_think ||
                                    (!thinking && preserve_thinking &&
                                     !text_has_trimmed_content(reasoning))))
                buf_puts(&out, "<think>\n\n</think>\n\n");
            buf_puts(&out, content);
            append_qwen_tool_calls_text(&out, &m->calls, tool_orders,
                                        content_nonempty);
            buf_puts(&out, "<|im_end|>\n");
            free(reasoning);
            free(content);
        } else {
            char *content = chat_content_text(m->content, false, NULL);
            buf_puts(&out, "<|im_start|>user\n[");
            buf_puts(&out, m->role);
            buf_puts(&out, "]: ");
            buf_puts(&out, content);
            buf_puts(&out, "<|im_end|>\n");
            free(content);
        }
    }

    buf_puts(&out, "<|im_start|>assistant\n");
    if (!thinking) buf_puts(&out, "<think>\n\n</think>\n\n");
    else buf_puts(&out, "<think>\n");
    return buf_take(&out);
}

static void append_json_tool_calls_text(buf *b, const tool_calls *calls,
                                        bool content_nonempty) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        if (content_nonempty) buf_puts(b, "\n");
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i || content_nonempty) buf_putc(b, '\n');
        buf_puts(b, "<tool_call>\n{\"name\": ");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ", \"arguments\": ");
        append_json_object_or_empty(b, tc->arguments);
        buf_puts(b, "}\n</tool_call>");
    }
}

/* Convert chat messages into the template's message list.  Owned strings stay
 * in c->owned; c->thinking carries any in-band /think switch the messages set. */
static void jinja_conv_build(jinja_conv *c, const chat_msgs *msgs,
                             const char *tool_schemas, p36_think_mode think_mode,
                             bool preserve_thinking) {
    memset(c, 0, sizeof(*c));
    bool thinking = p36_think_mode_enabled(think_mode);
    int last_query = last_chat_query(msgs);
    int failures = 0;

    int system_head = 0;
    char *first_content = leading_system_content(msgs, &system_head, &thinking);
    bool has_first_sys = text_has_trimmed_content(first_content);

    c->cap = (size_t)msgs->len + 2;
    c->v = calloc(c->cap, sizeof(*c->v));
    c->owned = calloc((size_t)msgs->len * 3 + 2, sizeof(*c->owned));
    if (!c->v || !c->owned) die("out of memory");
    c->tools = tool_schemas ? xstrdup(tool_schemas) : NULL;

    if (has_first_sys) {
        c->v[c->len].role = "system";
        c->v[c->len].content = first_content;
        c->owned[c->n_owned++] = first_content;
        c->len++;
    } else {
        free(first_content);
    }

    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        p36_jinja_message *jm = &c->v[c->len];
        if (role_is_system(m->role)) {
            if (i < system_head) continue;
            char *content = chat_content_text(m->content, true, &thinking);
            c->owned[c->n_owned++] = content;
            jm->role = "system";
            jm->content = content;
        } else if (!strcmp(m->role, "user")) {
            char *content = chat_content_text(m->content, true, &thinking);
            c->owned[c->n_owned++] = content;
            failures = 0;
            jm->role = "user";
            jm->content = content;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            char *raw_content = chat_content_text(m->content, false, NULL);
            failures = tool_response_is_error(raw_content) ? failures + 1 : 0;
            buf tb = {0};
            buf_puts(&tb, raw_content);
            append_tool_error_warning(&tb, failures);
            free(raw_content);
            char *content = buf_take(&tb);
            c->owned[c->n_owned++] = content;
            jm->role = "tool";
            jm->content = content;
        } else if (!strcmp(m->role, "assistant")) {
            char *reasoning = NULL;
            char *content = assistant_message_parts(m, &reasoning);
            c->owned[c->n_owned++] = content;
            c->owned[c->n_owned++] = reasoning;
            bool keep_reasoning = (preserve_thinking || i > last_query) &&
                                  text_has_trimmed_content(reasoning);
            jm->role = "assistant";
            jm->content = content;
            jm->reasoning_content = keep_reasoning ? reasoning : NULL;
            if (m->calls.len > 0) {
                p36_jinja_tool_call *tcs = calloc((size_t)m->calls.len, sizeof(*tcs));
                if (!tcs) die("out of memory");
                for (int tc_i = 0; tc_i < m->calls.len; tc_i++) {
                    tcs[tc_i].name = m->calls.v[tc_i].name ? m->calls.v[tc_i].name : "";
                    tcs[tc_i].arguments_json = m->calls.v[tc_i].arguments && m->calls.v[tc_i].arguments[0]
                        ? m->calls.v[tc_i].arguments : "{}";
                }
                jm->tool_calls = tcs;
                jm->tool_call_count = (size_t)m->calls.len;
            }
        } else {
            char *raw_content = chat_content_text(m->content, false, NULL);
            buf ub = {0};
            buf_printf(&ub, "[%s]: %s", m->role, raw_content);
            free(raw_content);
            char *content = buf_take(&ub);
            c->owned[c->n_owned++] = content;
            jm->role = "user";
            jm->content = content;
        }
        c->len++;
    }
    c->thinking = thinking;
}

static char *jinja_conv_render(const char *chat_template, const jinja_conv *c,
                               const p36_jinja_message *msgs, size_t n_msgs,
                               bool add_generation_prompt, p36_think_mode think_mode,
                               bool preserve_thinking, const p36_jinja_kwarg *kwargs,
                               size_t kwarg_count, char *jerr, size_t jerrlen) {
    p36_jinja_render_params jp = {
        .messages = msgs,
        .message_count = n_msgs,
        .tools_json_lines = c->tools,
        .add_generation_prompt = add_generation_prompt,
        .enable_thinking_set = true,
        .enable_thinking = c->thinking,
        .preserve_thinking_set = true,
        .preserve_thinking = preserve_thinking,
        .reasoning_effort = c->thinking ? p36_jinja_effort_name(think_mode) : NULL,
        .extra_kwargs = kwargs,
        .extra_kwarg_count = kwarg_count,
    };
    return p36_jinja_render(chat_template, &jp, jerr, jerrlen);
}

/* keep (optional) receives the converted conversation on success. */
static char *render_jinja_chat_prompt_text_keep(const char *chat_template,
                                                const chat_msgs *msgs,
                                                const char *tool_schemas,
                                                p36_think_mode think_mode,
                                                bool preserve_thinking,
                                                const p36_jinja_kwarg *kwargs,
                                                size_t kwarg_count,
                                                jinja_conv *keep,
                                                char *err, size_t errlen) {
    jinja_conv c;
    jinja_conv_build(&c, msgs, tool_schemas, think_mode, preserve_thinking);
    char jerr[256] = {0};
    char *rendered = jinja_conv_render(chat_template, &c, c.v, c.len, true, think_mode,
                                       preserve_thinking, kwargs, kwarg_count,
                                       jerr, sizeof(jerr));
    if (rendered && keep) *keep = c;
    else jinja_conv_free(&c);

    if (!rendered && kwarg_count) {
        /* The client sent template kwargs, so a template error (often a
         * raise_exception validating them) must reach it: falling back to the
         * built-in prompt would silently drop the switch it asked for. */
        if (err && errlen) snprintf(err, errlen, "chat template error: %s", jerr[0] ? jerr : "render failed");
        return NULL;
    }
    if (!rendered) {
        return render_qwen_chat_prompt_text(msgs, tool_schemas, NULL, think_mode,
                                            preserve_thinking, NULL);
    }
    return rendered;
}

static __attribute__((unused)) char *render_jinja_chat_prompt_text(
                                           const char *chat_template,
                                           const chat_msgs *msgs,
                                           const char *tool_schemas,
                                           p36_think_mode think_mode,
                                           bool preserve_thinking,
                                           const p36_jinja_kwarg *kwargs,
                                           size_t kwarg_count,
                                           char *err, size_t errlen) {
    return render_jinja_chat_prompt_text_keep(chat_template, msgs, tool_schemas, think_mode,
                                              preserve_thinking, kwargs, kwarg_count,
                                              NULL, err, errlen);
}

static const char *inkling2_effort_level_str(p36_think_mode think_mode, bool thinking) {
    if (!thinking || think_mode == P36_THINK_NONE) return "0";
    switch (think_mode) {
    case P36_THINK_LOW: return "0.2";
    case P36_THINK_MEDIUM: return "0.7";
    case P36_THINK_HIGH: return "0.9";
    case P36_THINK_XHIGH:
    case P36_THINK_MAX: return "0.99";
    default: return "0.9";
    }
}

static void append_inkling2_one_tool_schema(buf *out, const char *line, size_t len) {
    char *copy = xstrndup(line, len);
    const char *p = copy;
    char *name = NULL;
    char *desc = NULL;
    char *params = NULL;
    json_ws(&p);
    if (*p == '{') {
        p++;
        json_ws(&p);
        while (*p && *p != '}') {
            char *key = NULL;
            if (!json_string(&p, &key)) { free(key); break; }
            json_ws(&p);
            if (*p != ':') { free(key); break; }
            p++;
            if (!strcmp(key, "name")) {
                if (!json_string_replace(&p, &name)) { free(key); break; }
            } else if (!strcmp(key, "description")) {
                if (!json_string_replace(&p, &desc)) { free(key); break; }
            } else if (!strcmp(key, "parameters") || !strcmp(key, "input_schema")) {
                if (!json_raw_value_replace(&p, &params)) { free(key); break; }
            } else if (!json_skip_value(&p)) {
                free(key);
                break;
            }
            free(key);
            json_ws(&p);
            if (*p == ',') p++;
            json_ws(&p);
        }
    }
    if (name) {
        buf_putc(out, '{');
        bool wrote = false;
        if (desc) {
            buf_puts(out, "\"description\":");
            json_escape(out, desc);
            wrote = true;
        }
        if (wrote) buf_putc(out, ',');
        buf_puts(out, "\"name\":");
        json_escape(out, name);
        if (params && params[0]) {
            char *minified = json_minify(params);
            buf_puts(out, ",\"parameters\":");
            buf_puts(out, minified ? minified : params);
            free(minified);
        }
        buf_puts(out, ",\"type\":\"function\"}");
    } else {
        buf_append(out, line, len);
    }
    free(name);
    free(desc);
    free(params);
    free(copy);
}

static void append_inkling2_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b, "<|message_system|>tool_declare<|content_xml|>[");
    const char *p = tool_schemas;
    bool first = true;
    while (*p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len > 0) {
            if (!first) buf_putc(b, ',');
            append_inkling2_one_tool_schema(b, p, len);
            first = false;
        }
        if (!nl) break;
        p = nl + 1;
    }
    buf_puts(b, "]<|end_message|>");
}

static void append_inkling2_tool_calls_text(buf *b, const tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text &&
        !strncmp(calls->raw_tool_text, "<|message_model|>", 17)) {
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const char *name = tc->name ? tc->name : "";
        buf_puts(b, "<|message_model|>");
        buf_puts(b, name);
        buf_puts(b, "<|content_invoke_tool_json|>{\"name\":");
        json_escape(b, name);
        buf_puts(b, ",\"args\":");
        append_json_object_or_empty(b, tc->arguments);
        buf_puts(b, "}<|end_message|>");
    }
}

static const char *find_inkling2_tool_name_by_id(const chat_msgs *msgs, int upto,
                                                 const char *tool_call_id) {
    if (!msgs || !tool_call_id || !tool_call_id[0]) return "unknown";
    if (upto > msgs->len) upto = msgs->len;
    for (int i = upto - 1; i >= 0; i--) {
        const chat_msg *m = &msgs->v[i];
        if (strcmp(m->role, "assistant")) continue;
        for (int k = 0; k < m->calls.len; k++) {
            const tool_call *tc = &m->calls.v[k];
            if (tc->id && !strcmp(tc->id, tool_call_id) &&
                tc->name && tc->name[0]) {
                return tc->name;
            }
        }
    }
    return "unknown";
}

static char *render_inkling2_chat_prompt_text(const chat_msgs *msgs,
                                              const char *tool_schemas,
                                              p36_think_mode think_mode,
                                              bool preserve_thinking) {
    bool thinking = p36_think_mode_enabled(think_mode);
    apply_chat_thinking_controls(msgs, &thinking);
    const char *effort = inkling2_effort_level_str(think_mode, thinking);
    int last_query = last_chat_query(msgs);
    int failures = 0;
    bool effort_emitted = false;

    buf out = {0};
    if (tool_schemas && tool_schemas[0]) {
        append_inkling2_tools_prompt_text(&out, tool_schemas);
    }

    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            char *content = chat_content_text(m->content, true, &thinking);
            if (text_has_trimmed_content(content)) {
                buf_puts(&out, "<|message_system|><|content_text|>");
                buf_puts(&out, content);
                buf_puts(&out, "<|end_message|>");
            }
            free(content);
            continue;
        }
        if (!effort_emitted) {
            buf_puts(&out, "<|message_system|><|content_text|>Thinking effort level: ");
            buf_puts(&out, effort);
            buf_puts(&out, "<|end_message|>");
            effort_emitted = true;
        }
        if (!strcmp(m->role, "user")) {
            if (m->result_count > 0 && !m->other_content) {
                const char *cursor = m->content ? m->content : "";
                for (size_t k = 0; k < m->result_count; k++) {
                    const char *open = strstr(cursor, "<tool_response>\n");
                    const char *close = open ? strstr(open + 16, "\n</tool_response>") : NULL;
                    char *raw = (open && close)
                        ? xstrndup(open + 16, (size_t)(close - (open + 16)))
                        : chat_content_text(cursor, false, NULL);
                    if (open && close) cursor = close + 17;
                    failures = tool_response_is_error(raw) ? failures + 1 : 0;
                    const char *fn_name = find_inkling2_tool_name_by_id(msgs, i, m->result_ids[k]);
                    buf_puts(&out, "<|message_tool|>");
                    buf_puts(&out, fn_name);
                    buf_puts(&out, "<|content_text|>");
                    buf_puts(&out, raw);
                    append_tool_error_warning(&out, failures);
                    buf_puts(&out, "<|end_message|>");
                    free(raw);
                }
            } else {
                char *content = chat_content_text(m->content, true, &thinking);
                failures = 0;
                if (m->images.len > 0) {
                    /* Inkling-2 gives each image its own <|message_user|><|content_image|>...<|end_message|>
                     * block; request_tokenize_multimodal_prompt replaces each marker with
                     * <|content_image|> + N embeddings + <|end_message|>. */
                    const char *cur = content;
                    bool emitted_any = false;
                    for (size_t img_i = 0; img_i < m->images.len; img_i++) {
                        const char *mk = strstr(cur, m->images.v[img_i].marker);
                        if (!mk) continue;
                        size_t pre_len = (size_t)(mk - cur);
                        if (pre_len > 0) {
                            buf_puts(&out, "<|message_user|><|content_text|>");
                            buf_append(&out, cur, pre_len);
                            buf_puts(&out, "<|end_message|>");
                            emitted_any = true;
                        }
                        buf_puts(&out, "<|message_user|>");
                        buf_puts(&out, m->images.v[img_i].marker);
                        emitted_any = true;
                        cur = mk + strlen(m->images.v[img_i].marker);
                    }
                    if (*cur || !emitted_any) {
                        buf_puts(&out, "<|message_user|><|content_text|>");
                        buf_puts(&out, cur);
                        buf_puts(&out, "<|end_message|>");
                    }
                } else {
                    buf_puts(&out, "<|message_user|><|content_text|>");
                    buf_puts(&out, content);
                    buf_puts(&out, "<|end_message|>");
                }
                free(content);
            }
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            char *content = chat_content_text(m->content, false, NULL);
            failures = tool_response_is_error(content) ? failures + 1 : 0;
            const char *fn_name = find_inkling2_tool_name_by_id(msgs, i, m->tool_call_id);
            if (m->images.len > 0) {
                /* Inkling-2 gives each image its own <|message_tool|>{fn}<|content_image|>...<|end_message|>
                 * block; request_tokenize_multimodal_prompt replaces each marker with
                 * <|content_image|> + N embeddings + <|end_message|>. */
                const char *cur = content;
                bool emitted_any = false;
                for (size_t img_i = 0; img_i < m->images.len; img_i++) {
                    const char *mk = strstr(cur, m->images.v[img_i].marker);
                    if (!mk) continue;
                    size_t pre_len = (size_t)(mk - cur);
                    if (pre_len > 0) {
                        buf_puts(&out, "<|message_tool|>");
                        buf_puts(&out, fn_name);
                        buf_puts(&out, "<|content_text|>");
                        buf_append(&out, cur, pre_len);
                        buf_puts(&out, "<|end_message|>");
                        emitted_any = true;
                    }
                    buf_puts(&out, "<|message_tool|>");
                    buf_puts(&out, fn_name);
                    buf_puts(&out, m->images.v[img_i].marker);
                    emitted_any = true;
                    cur = mk + strlen(m->images.v[img_i].marker);
                }
                if (*cur || !emitted_any) {
                    buf_puts(&out, "<|message_tool|>");
                    buf_puts(&out, fn_name);
                    buf_puts(&out, "<|content_text|>");
                    buf_puts(&out, cur);
                    append_tool_error_warning(&out, failures);
                    buf_puts(&out, "<|end_message|>");
                }
            } else {
                buf_puts(&out, "<|message_tool|>");
                buf_puts(&out, fn_name);
                buf_puts(&out, "<|content_text|>");
                buf_puts(&out, content);
                append_tool_error_warning(&out, failures);
                buf_puts(&out, "<|end_message|>");
            }
            free(content);
        } else if (!strcmp(m->role, "assistant")) {
            char *reasoning = NULL;
            char *content = assistant_message_parts(m, &reasoning);
            bool keep_reasoning = (preserve_thinking || i > last_query) &&
                                  text_has_trimmed_content(reasoning);
            bool content_nonempty = text_has_trimmed_content(content);
            if (keep_reasoning) {
                buf_puts(&out, "<|message_model|><|content_thinking|>");
                buf_puts(&out, reasoning);
                buf_puts(&out, "<|end_message|>");
            }
            if (content_nonempty || (!keep_reasoning && m->calls.len == 0)) {
                buf_puts(&out, "<|message_model|><|content_text|>");
                buf_puts(&out, content);
                buf_puts(&out, "<|end_message|>");
            }
            append_inkling2_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<|content_model_end_sampling|>");
            free(reasoning);
            free(content);
        } else {
            char *content = chat_content_text(m->content, false, NULL);
            buf_puts(&out, "<|message_user|><|content_text|>[");
            buf_puts(&out, m->role);
            buf_puts(&out, "]: ");
            buf_puts(&out, content);
            buf_puts(&out, "<|end_message|>");
            free(content);
        }
    }
    if (!effort_emitted) {
        buf_puts(&out, "<|message_system|><|content_text|>Thinking effort level: ");
        buf_puts(&out, effort);
        buf_puts(&out, "<|end_message|>");
    }
    return buf_take(&out);
}

static char *render_chat_prompt_text_profile(const chat_msgs *msgs,
                                             const char *tool_schemas,
                                             const tool_schema_orders *tool_orders,
                                             p36_think_mode think_mode,
                                             bool kat_coder,
                                             bool preserve_thinking) {
    if (kat_coder) return render_kat_chat_prompt_text(msgs, tool_schemas, tool_orders,
                                                      think_mode, preserve_thinking);
    return render_qwen_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode,
                                        preserve_thinking, NULL);
}

static char *render_chat_prompt_text_model(const chat_msgs *msgs,
                                           const char *tool_schemas,
                                           const tool_schema_orders *tool_orders,
                                           p36_think_mode think_mode,
                                           bool kat_coder, bool qwen38,
                                           bool preserve_thinking) {
    if (!qwen38) return render_chat_prompt_text_profile(
        msgs, tool_schemas, tool_orders, think_mode, kat_coder, preserve_thinking);
    return render_qwen_chat_prompt_text(
        msgs, tool_schemas, tool_orders, think_mode, preserve_thinking,
        p36_qwen38_effort_instruction(think_mode));
}

/* NULL (with err set) only when a --chat-template render fails on a request
 * that supplied extra chat_template_kwargs. */
static char *render_request_chat_prompt_text(request *r,
                                             const chat_msgs *msgs,
                                             const char *tool_schemas,
                                             char *err, size_t errlen) {
    if (r->chat_template) {
        jinja_conv_free(&r->tmpl_conv);
        return render_jinja_chat_prompt_text_keep(r->chat_template, msgs, tool_schemas,
                                                  r->think_mode, r->preserve_thinking,
                                                  r->tmpl_kwargs, r->tmpl_kwarg_count,
                                                  &r->tmpl_conv, err, errlen);
    }
    if (r->inkling2) {
        return render_inkling2_chat_prompt_text(msgs, tool_schemas,
                                                r->think_mode, r->preserve_thinking);
    }
    return render_chat_prompt_text_model(msgs, tool_schemas, &r->tool_orders,
                                         r->think_mode, r->kat_coder, r->qwen38,
                                         r->preserve_thinking);
}

#ifdef P36_SERVER_TEST
static char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     p36_think_mode think_mode) {
    return render_qwen_chat_prompt_text(msgs, tool_schemas, tool_orders, think_mode, true, NULL);
}
#endif

static bool chat_prompt_preserves_reasoning(const chat_msgs *msgs,
                                            bool preserve_thinking) {
    if (preserve_thinking) return true;
    int last_query = last_chat_query(msgs);
    for (int i = last_query + 1; i < msgs->len; i++) {
        if (!strcmp(msgs->v[i].role, "assistant")) return true;
    }
    return false;
}

/* The API parsers are intentionally selective JSON parsers: they keep only
 * fields that affect model semantics, rendering, streaming, or cache keys, and
 * skip extension fields.  The output is always a rendered P36 chat/completion
 * prompt plus the small amount of protocol state needed to translate the reply. */
static char *render_tokens_text(p36_engine *engine, const p36_tokens *tokens, size_t *out_len);

static void request_note_tool_continuation(request *r, const chat_msgs *msgs,
                                           const char *schemas) {
    r->tool_schema_key = xstrdup(schemas ? schemas : "");
    size_t count = 0;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!strcmp(m->role, "tool") && m->tool_call_id && m->tool_call_id[0]) count++;
        else if (r->api == API_ANTHROPIC && !strcmp(m->role, "user") &&
                 m->result_count && !m->other_content) count += m->result_count;
        else return;
    }
    if (!count || count > 16) return;
    r->continuation_ids = xmalloc(count * sizeof(char *));
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!strcmp(m->role, "tool"))
            r->continuation_ids[r->continuation_count++] = xstrdup(m->tool_call_id);
        else for (size_t k = 0; k < m->result_count; k++)
            r->continuation_ids[r->continuation_count++] = xstrdup(m->result_ids[k]);
    }
}

static bool request_tokenize_multimodal_prompt(p36_engine *e, server *s,
                                               request *r,
                                               const chat_msgs *msgs,
                                               char *err, size_t errlen) {
    size_t count = 0;
    for (int i = 0; msgs && i < msgs->len; i++) count += msgs->v[i].images.len;
    if (count == 0) {
        p36_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
        return true;
    }
    if (count > 16) {
        snprintf(err, errlen, "too many images; at most 16 are allowed");
        return false;
    }
    if (!e || !s || !p36_engine_has_vision(e)) {
        snprintf(err, errlen, "image input requires starting p36-server with --vision FILE");
        return false;
    }

    server_image_input **inputs = xmalloc(count * sizeof(inputs[0]));
    p36_vision_embedding *embeddings = xmalloc(count * sizeof(embeddings[0]));
    memset(embeddings, 0, count * sizeof(embeddings[0]));
    size_t next = 0;
    for (int i = 0; i < msgs->len; i++) {
        for (size_t j = 0; j < msgs->v[i].images.len; j++)
            inputs[next++] = &msgs->v[i].images.v[j];
    }

    bool ok = true;
    server_inference_lock(s);
    for (size_t i = 0; i < count; i++) {
        if (!server_encode_image(s, inputs[i], &embeddings[i], err, errlen)) {
            ok = false;
            break;
        }
    }
    server_inference_unlock(s);
    if (!ok) goto done;

    r->images = xmalloc(count * sizeof(r->images[0]));
    r->image_markers = xmalloc(count * sizeof(r->image_markers[0]));
    memset(r->images, 0, count * sizeof(r->images[0]));
    const char *cursor = r->prompt_text;
    for (size_t i = 0; i < count; i++) {
        const char *marker = strstr(cursor, inputs[i]->marker);
        if (!marker) {
            snprintf(err, errlen, "image marker was lost while rendering the request");
            ok = false;
            break;
        }
        char *prefix = xstrndup(cursor, (size_t)(marker - cursor));
        p36_tokenize_rendered_chat(e, prefix, &r->prompt);
        free(prefix);
        if (!p36_prompt_append_vision(e, &r->prompt, &r->images[i],
                                      &embeddings[i], err, errlen)) {
            ok = false;
            break;
        }
        r->image_count++;
        memcpy(r->image_markers[i], inputs[i]->marker, sizeof(r->image_markers[i]));
        cursor = marker + strlen(inputs[i]->marker);
    }
    if (ok) {
        p36_tokenize_rendered_chat(e, cursor, &r->prompt);
        char *rendered = render_tokens_text(e, &r->prompt, NULL);
        free(r->prompt_text);
        r->prompt_text = rendered;
    }

done:
    for (size_t i = 0; i < count; i++)
        p36_vision_embedding_free(&embeddings[i]);
    free(embeddings);
    free(inputs);
    if (!ok) {
        p36_tokens_free(&r->prompt);
        for (size_t i = 0; i < r->image_count; i++)
            p36_vision_embedding_free(&r->images[i].embedding);
        free(r->images);
        free(r->image_markers);
        r->image_markers = NULL;
        r->images = NULL;
        r->image_count = 0;
    }
    return ok;
}

