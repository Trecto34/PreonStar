static bool parse_anthropic_request(p36_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    request_set_model_profile(r, e);
    request_apply_default_penalties(r, s);
    r->api = API_ANTHROPIC;
    const char *p = body;
    bool got_messages = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    p36_think_mode reasoning_effort = P36_THINK_HIGH;
    chat_msgs msgs = {0};
    char *system = NULL;
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
            if (!parse_anthropic_messages(&p, &msgs)) {
                free(key);
                goto bad;
            }
            got_messages = true;
        } else if (!strcmp(key, "system")) {
            char *value = NULL;
            if (!parse_anthropic_system(&p, &value)) {
                free(key);
                goto bad;
            }
            free(system);
            system = value;
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '{') {
                p++;
                json_ws(&p);
                while (*p && *p != '}') {
                    char *ckey = NULL;
                    if (!json_string(&p, &ckey)) {
                        free(key);
                        goto bad;
                    }
                    json_ws(&p);
                    if (*p != ':') {
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    p++;
                    if (!strcmp(ckey, "type")) {
                        char *choice = NULL;
                        if (!json_string(&p, &choice)) {
                            free(ckey);
                            free(key);
                            goto bad;
                        }
                        tool_choice_none = !strcmp(choice, "none");
                        free(choice);
                    } else if (!json_skip_value(&p)) {
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    free(ckey);
                    json_ws(&p);
                    if (*p == ',') p++;
                    json_ws(&p);
                }
                if (*p != '}') {
                    free(key);
                    goto bad;
                }
                p++;
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            if (!json_string_replace(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
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
        } else if (!strcmp(key, "stop_sequences")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled,
                                              &r->thinking_budget)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "chat_template_kwargs")) {
            if (!parse_chat_template_kwargs_ex(&p, &thinking_enabled, &got_thinking,
                                                &r->preserve_thinking, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "output_config")) {
            if (!parse_output_config_effort(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
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
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (system && system[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = system;
        system = NULL;
        if (msgs.len == msgs.cap) {
            msgs.cap = msgs.cap ? msgs.cap * 2 : 8;
            msgs.v = xrealloc(msgs.v, (size_t)msgs.cap * sizeof(msgs.v[0]));
        }
        memmove(msgs.v + 1, msgs.v, (size_t)msgs.len * sizeof(msgs.v[0]));
        msgs.v[0] = msg;
        msgs.len++;
    }
    if (!request_validate_ignore_eos(r, err, errlen)) {
        chat_msgs_free(&msgs);
        free(system);
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
    free(system);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
image_error:
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    request_free(r);
    return false;
}

static bool parse_prompt(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    if (**p == '"') {
        if (!json_string(p, out)) return false;
    } else {
        *out = xstrdup("");
        if (**p && **p != ']' && !json_skip_value(p)) return false;
    }
    while (**p && **p != ']') {
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            if (!json_skip_value(p)) return false;
        } else {
            break;
        }
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_completion_request(p36_engine *e, server *s, const char *body, int def_tokens,
                                     int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_COMPLETION, def_tokens);
    request_set_model_profile(r, e);
    request_apply_default_penalties(r, s);
    const char *p = body;
    char *prompt = NULL;
    bool got_thinking = false;
    bool thinking_enabled = true;
    p36_think_mode reasoning_effort = P36_THINK_HIGH;

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
        if (!strcmp(key, "prompt")) {
            char *value = NULL;
            if (!parse_prompt(&p, &value)) {
                free(key);
                goto bad;
            }
            free(prompt);
            prompt = value;
        } else if (!strcmp(key, "model")) {
            if (!json_string_replace(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
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
    if (!prompt) {
        snprintf(err, errlen, "missing prompt");
        request_free(r);
        return false;
    }
    if (!request_validate_ignore_eos(r, err, errlen)) {
        free(prompt);
        request_free(r);
        return false;
    }
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;

    chat_msgs msgs = {0};
    chat_msg system = {
        .role = xstrdup("system"),
        .content = xstrdup("You are a helpful assistant"),
    };
    chat_msg user = {.role = xstrdup("user"), .content = prompt};
    prompt = NULL;
    chat_msgs_push(&msgs, system);
    chat_msgs_push(&msgs, user);
    apply_chat_thinking_controls(&msgs, &thinking_enabled);
    r->think_mode = p36_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled,
            think_mode_from_budget(r, reasoning_effort)), ctx_size);
    r->prompt_text = render_request_chat_prompt_text(r, &msgs, NULL);
    p36_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    return true;
bad:
    free(prompt);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static bool send_all(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + P36_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
#ifdef P36_SERVER_TEST
        ssize_t w = write(fd, s, n);
#else
        ssize_t w = send(fd, s, n, 0);
#endif
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + P36_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

