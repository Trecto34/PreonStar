static bool parse_chat_request(p36_engine *e, server *s, const char *body, int def_tokens,
                               int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    request_set_model_profile(r, e);
    request_apply_default_penalties(r, s);
    const char *p = body;
    bool got_messages = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    p36_think_mode reasoning_effort = P36_THINK_HIGH;
    chat_msgs msgs = {0};
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            if (!parse_messages(&p, &msgs)) {
                free(key);
                goto bad;
            }
            got_messages = true;
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                if (!json_string(&p, &choice)) {
                    free(key);
                    goto bad;
                }
                tool_choice_none = !strcmp(choice, "none");
                free(choice);
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            if (!json_string_replace(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens") || !strcmp(key, "max_completion_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking_budget")) {
            if (!json_int(&p, &r->thinking_budget)) {
                free(key);
                goto bad;
            }
            r->thinking_budget_hard = r->thinking_budget > 0;
        } else if (!strcmp(key, "action_budget")) {
            if (!json_int(&p, &r->action_budget)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stop_after_tool_call")) {
            if (!json_bool(&p, &r->stop_after_tool_call)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
            r->temperature_set = true;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
            r->top_p_set = true;
        } else if (!strcmp(key, "min_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->min_p = (float)v;
            r->min_p_set = true;
        } else if (!strcmp(key, "presence_penalty")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->presence_penalty = (float)v;
        } else if (!strcmp(key, "repeat_penalty") || !strcmp(key, "repetition_penalty")) {
            if (!json_repeat_penalty(&p, &r->repeat_penalty)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "frequency_penalty")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->frequency_penalty = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
            r->top_k_set = true;
        } else if (!strcmp(key, "seed")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "ignore_eos")) {
            if (!json_bool(&p, &r->ignore_eos)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream_options")) {
            if (!parse_stream_options(&p, &r->stream_include_usage)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "chat_template_kwargs")) {
            if (!parse_chat_template_kwargs_ex(&p, &thinking_enabled, &got_thinking,
                                                &r->preserve_thinking, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled,
                                              &r->thinking_budget)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (!request_validate_ignore_eos(r, err, errlen)) {
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->has_tools = tool_schemas && tool_schemas[0] && !tool_choice_none;
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    apply_chat_thinking_controls(&msgs, &thinking_enabled);
    r->think_mode = p36_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled,
            think_mode_from_budget(r, reasoning_effort)), ctx_size);
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_prompt_preserves_reasoning(&msgs, r->preserve_thinking);
    r->prompt_text = render_request_chat_prompt_text(
        r, &msgs, active_tool_schemas);
    request_note_tool_continuation(r, &msgs, active_tool_schemas);
    if (!request_tokenize_multimodal_prompt(e, s, r, &msgs, err, errlen)) goto image_error;
    chat_msgs_free(&msgs);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
image_error:
    chat_msgs_free(&msgs);
    free(tool_schemas);
    request_free(r);
    return false;
}

static bool parse_responses_string_or_json(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    return json_raw_value(p, out);
}

static bool parse_responses_string_or_json_replace(const char **p, char **out) {
    char *value = NULL;
    if (!parse_responses_string_or_json(p, &value)) return false;
    free(*out);
    *out = value;
    return true;
}

static bool parse_responses_input_item(const char **p, chat_msgs *msgs) {
    chat_msg msg = {0};
    tool_call call = {0};
    char *type = NULL;
    chat_msg output = {0};
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "type")) {
            if (!json_string_replace(p, &type)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "role")) {
            if (!json_string_replace(p, &msg.role)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "content")) {
            free(msg.content);
            msg.content = NULL;
            server_image_inputs_free(&msg.images);
            if (!parse_openai_content(p, &msg)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "output")) {
            chat_msg_free(&output);
            if (!parse_openai_content(p, &output)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "call_id")) {
            if (!json_string_replace(p, &call.id)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "name")) {
            if (!json_string_replace(p, &call.name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "arguments")) {
            if (!parse_responses_string_or_json_replace(p, &call.arguments)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;

    if (type && !strcmp(type, "reasoning")) {
        if (msg.images.len || output.images.len) goto bad;
        goto done;
    }
    if (type && !strcmp(type, "function_call")) {
        if (msg.images.len) goto bad;
        free(msg.role);
        free(msg.content);
        msg.role = xstrdup("assistant");
        msg.content = xstrdup("");
        if (!call.name) call.name = xstrdup("");
        if (!call.arguments) call.arguments = xstrdup("{}");
        tool_calls *target = &msg.calls;
        if (msgs->len && !strcmp(msgs->v[msgs->len - 1].role, "assistant"))
            target = &msgs->v[msgs->len - 1].calls;
        tool_calls_push(target, call);
        memset(&call, 0, sizeof(call));
        if (target != &msg.calls) goto done;
    } else if (type && (!strcmp(type, "function_call_output") || !strcmp(type, "custom_tool_call_output"))) {
        free(msg.role);
        msg.role = xstrdup("tool");
        msg.tool_call_id = call.id ? xstrdup(call.id) : NULL;
        free(msg.content);
        server_image_inputs_free(&msg.images);
        msg.content = output.content;
        msg.images = output.images;
        memset(&output, 0, sizeof(output));
        if (!msg.content) msg.content = xstrdup("");
    } else {
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
    }
    if (msg.images.len && strcmp(msg.role, "user") && strcmp(msg.role, "tool")) goto bad;
    chat_msgs_push(msgs, msg);
    memset(&msg, 0, sizeof(msg));

done:
    free(type);
    chat_msg_free(&output);
    tool_call_free(&call);
    chat_msg_free(&msg);
    return true;
bad:
    free(type);
    chat_msg_free(&output);
    tool_call_free(&call);
    chat_msg_free(&msg);
    return false;
}

static bool parse_responses_input(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p == '"') {
        chat_msg msg = {.role = xstrdup("user")};
        if (!json_string(p, &msg.content)) {
            chat_msg_free(&msg);
            return false;
        }
        chat_msgs_push(msgs, msg);
        return true;
    }
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            chat_msg msg = {.role = xstrdup("user")};
            if (!json_string(p, &msg.content)) {
                chat_msg_free(&msg);
                return false;
            }
            chat_msgs_push(msgs, msg);
        } else if (!parse_responses_input_item(p, msgs)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_responses_reasoning(const char **p, p36_think_mode *effort) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        bool ok = !strcmp(key, "effort") ?
            parse_reasoning_effort_value(p, effort) : json_skip_value(p);
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_responses_request(p36_engine *e, server *s, const char *body,
                                    int def_tokens, int ctx_size, request *r,
                                    char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    request_set_model_profile(r, e);
    request_apply_default_penalties(r, s);
    r->api = API_RESPONSES;
    const char *p = body;
    bool got_input = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    p36_think_mode reasoning_effort = P36_THINK_HIGH;
    chat_msgs msgs = {0};
    char *instructions = NULL;
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "input")) {
            chat_msgs_free(&msgs);
            if (!parse_responses_input(&p, &msgs)) {
                free(key);
                goto bad;
            }
            got_input = true;
        } else if (!strcmp(key, "instructions")) {
            if (!json_content_replace(&p, &instructions)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                if (!json_string(&p, &choice)) {
                    free(key);
                    goto bad;
                }
                tool_choice_none = !strcmp(choice, "none");
                free(choice);
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            if (!json_string_replace(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_output_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
            r->temperature_set = true;
        } else if (!strcmp(key, "top_p")) {
            double v;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
            r->top_p_set = true;
        } else if (!strcmp(key, "presence_penalty")) {
            double v;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->presence_penalty = (float)v;
        } else if (!strcmp(key, "repeat_penalty") || !strcmp(key, "repetition_penalty")) {
            if (!json_repeat_penalty(&p, &r->repeat_penalty)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "frequency_penalty")) {
            double v;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->frequency_penalty = (float)v;
        } else if (!strcmp(key, "ignore_eos")) {
            if (!json_bool(&p, &r->ignore_eos)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "chat_template_kwargs")) {
            if (!parse_chat_template_kwargs_ex(&p, &thinking_enabled, &got_thinking,
                                                &r->preserve_thinking, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning")) {
            if (!parse_responses_reasoning(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}' || !got_input) goto bad;

    if (instructions && instructions[0]) {
        chat_msg system = {.role = xstrdup("system"), .content = instructions};
        instructions = NULL;
        if (msgs.len == msgs.cap) {
            msgs.cap = msgs.cap ? msgs.cap * 2 : 8;
            msgs.v = xrealloc(msgs.v, (size_t)msgs.cap * sizeof(msgs.v[0]));
        }
        memmove(msgs.v + 1, msgs.v, (size_t)msgs.len * sizeof(msgs.v[0]));
        msgs.v[0] = system;
        msgs.len++;
    }
    if (!request_validate_ignore_eos(r, err, errlen)) {
        chat_msgs_free(&msgs);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->has_tools = tool_schemas && tool_schemas[0] && !tool_choice_none;
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    apply_chat_thinking_controls(&msgs, &thinking_enabled);
    r->think_mode = p36_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled,
            think_mode_from_budget(r, reasoning_effort)), ctx_size);
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    const char *active_tools = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_prompt_preserves_reasoning(&msgs, r->preserve_thinking);
    r->prompt_text = render_request_chat_prompt_text(
        r, &msgs, active_tools);
    request_note_tool_continuation(r, &msgs, active_tools);
    if (!request_tokenize_multimodal_prompt(e, s, r, &msgs, err, errlen)) goto image_error;
    chat_msgs_free(&msgs);
    free(instructions);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(instructions);
    free(tool_schemas);
    snprintf(err, errlen, got_input ? "invalid JSON request" : "missing input");
    request_free(r);
    return false;
image_error:
    chat_msgs_free(&msgs);
    free(instructions);
    free(tool_schemas);
    request_free(r);
    return false;
}

