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

static char *render_jinja_chat_prompt_text(const char *chat_template,
                                           const chat_msgs *msgs,
                                           const char *tool_schemas,
                                           p36_think_mode think_mode,
                                           bool preserve_thinking) {
    bool thinking = p36_think_mode_enabled(think_mode);
    int last_query = last_chat_query(msgs);
    int failures = 0;

    int system_head = 0;
    char *first_content = leading_system_content(msgs, &system_head, &thinking);
    bool has_first_sys = text_has_trimmed_content(first_content);

    size_t max_msgs = (size_t)msgs->len + 1;
    p36_jinja_message *jmsgs = calloc(max_msgs, sizeof(*jmsgs));
    char **owned_strs = calloc(max_msgs * 3 + 1, sizeof(*owned_strs));
    if (!jmsgs || !owned_strs) die("out of memory");
    size_t n_owned = 0;
    size_t n_jmsgs = 0;

    if (has_first_sys) {
        jmsgs[n_jmsgs].role = "system";
        jmsgs[n_jmsgs].content = first_content;
        owned_strs[n_owned++] = first_content;
        n_jmsgs++;
    } else {
        free(first_content);
    }

    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            if (i < system_head) continue;
            char *content = chat_content_text(m->content, true, &thinking);
            owned_strs[n_owned++] = content;
            jmsgs[n_jmsgs].role = "system";
            jmsgs[n_jmsgs].content = content;
            n_jmsgs++;
        } else if (!strcmp(m->role, "user")) {
            char *content = chat_content_text(m->content, true, &thinking);
            owned_strs[n_owned++] = content;
            failures = 0;
            jmsgs[n_jmsgs].role = "user";
            jmsgs[n_jmsgs].content = content;
            n_jmsgs++;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            char *raw_content = chat_content_text(m->content, false, NULL);
            failures = tool_response_is_error(raw_content) ? failures + 1 : 0;
            buf tb = {0};
            buf_puts(&tb, raw_content);
            append_tool_error_warning(&tb, failures);
            free(raw_content);
            char *content = buf_take(&tb);
            owned_strs[n_owned++] = content;
            jmsgs[n_jmsgs].role = "tool";
            jmsgs[n_jmsgs].content = content;
            n_jmsgs++;
        } else if (!strcmp(m->role, "assistant")) {
            char *reasoning = NULL;
            char *content = assistant_message_parts(m, &reasoning);
            owned_strs[n_owned++] = content;
            owned_strs[n_owned++] = reasoning;
            bool keep_reasoning = (preserve_thinking || i > last_query) &&
                                  text_has_trimmed_content(reasoning);
            jmsgs[n_jmsgs].role = "assistant";
            jmsgs[n_jmsgs].content = content;
            jmsgs[n_jmsgs].reasoning_content = keep_reasoning ? reasoning : NULL;
            if (m->calls.len > 0) {
                p36_jinja_tool_call *tcs = calloc((size_t)m->calls.len, sizeof(*tcs));
                if (!tcs) die("out of memory");
                for (int tc_i = 0; tc_i < m->calls.len; tc_i++) {
                    tcs[tc_i].name = m->calls.v[tc_i].name ? m->calls.v[tc_i].name : "";
                    tcs[tc_i].arguments_json = m->calls.v[tc_i].arguments && m->calls.v[tc_i].arguments[0]
                        ? m->calls.v[tc_i].arguments : "{}";
                }
                jmsgs[n_jmsgs].tool_calls = tcs;
                jmsgs[n_jmsgs].tool_call_count = (size_t)m->calls.len;
            }
            n_jmsgs++;
        } else {
            char *raw_content = chat_content_text(m->content, false, NULL);
            buf ub = {0};
            buf_printf(&ub, "[%s]: %s", m->role, raw_content);
            free(raw_content);
            char *content = buf_take(&ub);
            owned_strs[n_owned++] = content;
            jmsgs[n_jmsgs].role = "user";
            jmsgs[n_jmsgs].content = content;
            n_jmsgs++;
        }
    }

    p36_jinja_render_params jp = {
        .messages = jmsgs,
        .message_count = n_jmsgs,
        .tools_json_lines = tool_schemas,
        .add_generation_prompt = true,
        .enable_thinking_set = true,
        .enable_thinking = thinking,
        .preserve_thinking_set = true,
        .preserve_thinking = preserve_thinking,
        .reasoning_effort = thinking ? p36_jinja_effort_name(think_mode) : NULL,
    };
    char *rendered = p36_jinja_render(chat_template, &jp, NULL, 0);

    for (size_t i = 0; i < n_jmsgs; i++) free((void *)jmsgs[i].tool_calls);
    free(jmsgs);
    for (size_t i = 0; i < n_owned; i++) free(owned_strs[i]);
    free(owned_strs);

    if (!rendered) {
        return render_qwen_chat_prompt_text(msgs, tool_schemas, NULL, think_mode,
                                            preserve_thinking, NULL);
    }
    return rendered;
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

static char *render_request_chat_prompt_text(const request *r,
                                             const chat_msgs *msgs,
                                             const char *tool_schemas) {
    if (r->chat_template) {
        return render_jinja_chat_prompt_text(r->chat_template, msgs, tool_schemas,
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

