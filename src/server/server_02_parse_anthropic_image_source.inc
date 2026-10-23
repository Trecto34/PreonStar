static bool parse_anthropic_image_source(const char **p,
                                         char **source_type,
                                         char **media_type,
                                         char **data) {
    json_ws(p);
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
        bool ok;
        if (!strcmp(key, "type"))
            ok = json_string_replace(p, source_type);
        else if (!strcmp(key, "media_type"))
            ok = json_string_replace(p, media_type);
        else if (!strcmp(key, "data"))
            ok = json_string_replace(p, data);
        else
            ok = json_skip_value(p);
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

/* Anthropic content is block-structured, while the engine consumes one compact
 * chat_msg per role.  Parsing collapses text/thinking into strings, converts
 * assistant tool_use blocks to tool_calls, and keeps tool_result blocks as
 * escaped text because P36 sees tool results in its chat template. */
static bool parse_anthropic_content_mode(const char **p, chat_msg *msg, bool allow_tools);

static bool parse_anthropic_content_block(const char **p, chat_msg *msg, bool allow_tools) {
    if (**p != '{') return false;
    (*p)++;
    char *type = NULL;
    char *text = NULL;
    char *thinking = NULL;
    char *id = NULL;
    char *name = NULL;
    char *input = NULL;
    char *tool_result = NULL;
    chat_msg result = {0};
    char *source_type = NULL;
    char *media_type = NULL;
    char *image_data = NULL;

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
        } else if (!strcmp(key, "text")) {
            if (!json_content_replace(p, &text)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!json_content_replace(p, &thinking)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "id") || !strcmp(key, "tool_use_id")) {
            if (!json_string_replace(p, &id)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "name")) {
            if (!json_string_replace(p, &name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "input")) {
            if (!json_raw_value_replace(p, &input)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "content")) {
            if (!json_raw_value_replace(p, &tool_result)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "source")) {
            if (!parse_anthropic_image_source(p, &source_type, &media_type, &image_data)) {
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

    if (!allow_tools && (!type || (strcmp(type, "text") && strcmp(type, "image")))) goto bad;
    if (!type || strcmp(type, "tool_result")) msg->other_content = true;
    if (type && !strcmp(type, "tool_use")) {
        tool_call tc = {0};
        tc.id = id ? xstrdup(id) : NULL;
        tc.name = name ? xstrdup(name) : xstrdup("");
        tc.arguments = input ? xstrdup(input) : xstrdup("{}");
        tool_calls_push(&msg->calls, tc);
    } else if (type && !strcmp(type, "tool_result")) {
        if (!id || !id[0]) msg->other_content = true;
        else {
            msg->result_ids = xrealloc(msg->result_ids,
                                      (msg->result_count + 1) * sizeof(char *));
            msg->result_ids[msg->result_count++] = xstrdup(id);
        }
        const char *nested = tool_result ? tool_result : "\"\"";
        if (!parse_anthropic_content_mode(&nested, &result, false)) goto bad;
        buf b = {0};
        buf_puts(&b, msg->content ? msg->content : "");
        buf_puts(&b, "<tool_response>\n");
        append_xml_text_escaped(&b, result.content);
        for (size_t i = 0; i < result.images.len; i++) {
            if (msg->images.len == msg->images.cap) {
                msg->images.cap = msg->images.cap ? msg->images.cap * 2 : 2;
                msg->images.v = xrealloc(msg->images.v, msg->images.cap * sizeof(msg->images.v[0]));
            }
            msg->images.v[msg->images.len++] = result.images.v[i];
            memset(&result.images.v[i], 0, sizeof(result.images.v[i]));
        }
        buf_puts(&b, "\n</tool_response>");
        free(msg->content);
        msg->content = buf_take(&b);
    } else if (type && !strcmp(type, "image")) {
        char marker[SERVER_IMAGE_MARKER_BYTES];
        if (!source_type || strcmp(source_type, "base64") ||
            !server_image_inputs_push_base64(&msg->images, media_type, image_data, marker)) goto bad;
        append_owned_text(&msg->content, marker);
    } else {
        if (text) {
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            append_anthropic_block_content(&b, text);
            free(msg->content);
            msg->content = buf_take(&b);
        }
        if (thinking) {
            buf b = {0};
            buf_puts(&b, msg->reasoning ? msg->reasoning : "");
            append_anthropic_block_content(&b, thinking);
            free(msg->reasoning);
            msg->reasoning = buf_take(&b);
        }
    }

    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    chat_msg_free(&result);
    free(source_type);
    free(media_type);
    free(image_data);
    return true;
bad:
    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    chat_msg_free(&result);
    free(source_type);
    free(media_type);
    free(image_data);
    return false;
}

static bool parse_anthropic_content_mode(const char **p, chat_msg *msg, bool allow_tools) {
    json_ws(p);
    if (**p == '"') {
        msg->other_content = true;
        return json_string(p, &msg->content);
    }
    if (json_lit(p, "null")) {
        msg->content = xstrdup("");
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            msg->other_content = true;
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            buf_puts(&b, s);
            free(msg->content);
            msg->content = buf_take(&b);
            free(s);
        } else if (**p == '{') {
            if (!parse_anthropic_content_block(p, msg, allow_tools)) return false;
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    if (!msg->content) msg->content = xstrdup("");
    return true;
}

static bool parse_anthropic_content(const char **p, chat_msg *msg) {
    return parse_anthropic_content_mode(p, msg, true);
}

static bool parse_anthropic_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                if (!json_string_replace(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                chat_msg parsed = {0};
                parsed.role = msg.role ? xstrdup(msg.role) : NULL;
                if (!parse_anthropic_content(p, &parsed)) {
                    chat_msg_free(&parsed);
                    free(key);
                    goto fail;
                }
                free(msg.content);
                free(msg.reasoning);
                tool_calls_free(&msg.calls);
                server_image_inputs_free(&msg.images);
                for (size_t i = 0; i < msg.result_count; i++) free(msg.result_ids[i]);
                free(msg.result_ids);
                msg.result_ids = parsed.result_ids;
                msg.result_count = parsed.result_count;
                msg.other_content = parsed.other_content;
                parsed.result_ids = NULL;
                parsed.result_count = 0;
                msg.images = parsed.images;
                memset(&parsed.images, 0, sizeof(parsed.images));
                msg.content = parsed.content;
                msg.reasoning = parsed.reasoning;
                msg.calls = parsed.calls;
                parsed.content = NULL;
                parsed.reasoning = NULL;
                memset(&parsed.calls, 0, sizeof(parsed.calls));
                chat_msg_free(&parsed);
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        if ((msg.images.len && strcmp(msg.role, "user")) ||
            (msg.calls.len && strcmp(msg.role, "assistant"))) goto fail;
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool anthropic_system_part_is_private(const char *s) {
    return s && !strncmp(s, "x-anthropic-", 12);
}

static void append_anthropic_system_part(buf *b, const char *s) {
    if (!s || !s[0] || anthropic_system_part_is_private(s)) return;
    if (b->len && b->ptr[b->len - 1] != '\n') buf_putc(b, '\n');
    buf_puts(b, s);
}

static bool parse_anthropic_system_object(const char **p, buf *out) {
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
        if (!strcmp(key, "text")) {
            char *text = NULL;
            if (!json_string(p, &text)) {
                free(key);
                return false;
            }
            append_anthropic_system_part(out, text);
            free(text);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_anthropic_system(const char **p, char **out) {
    json_ws(p);
    buf b = {0};
    if (**p == '"') {
        char *text = NULL;
        if (!json_string(p, &text)) return false;
        append_anthropic_system_part(&b, text);
        free(text);
        *out = buf_take(&b);
        return true;
    }
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *text = NULL;
            if (!json_string(p, &text)) goto bad;
            append_anthropic_system_part(&b, text);
            free(text);
        } else if (**p == '{') {
            if (!parse_anthropic_system_object(p, &b)) goto bad;
        } else if (!json_skip_value(p)) {
            goto bad;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = buf_take(&b);
    return true;
bad:
    buf_free(&b);
    return false;
}

static void append_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
    buf_puts(b, tool_schemas);
    buf_puts(b,
        "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.\n"
        "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.\n"
        "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.\n"
        "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user without any tool call.\n"
        "</IMPORTANT>");
}

static void json_escape(buf *b, const char *s);

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

static void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}

static void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        args->v = xrealloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
    }
    args->v[args->len++] = arg;
}

static int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key)) return i;
    }
    return -1;
}

static bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }

        json_arg arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        key = value = NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}

static void append_json_arg_pair(buf *b, const json_arg *arg) {
    json_escape(b, arg->key);
    buf_puts(b, ":");
    if (arg->is_string) json_escape(b, arg->value);
    else buf_puts(b, arg->value);
}

static void append_json_object_or_empty(buf *b, const char *json) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) {
        buf_puts(b, "{}");
        return;
    }
    buf_putc(b, '{');
    bool wrote = false;
    for (int i = 0; i < args.len; i++) {
        if (wrote) buf_putc(b, ',');
        append_json_arg_pair(b, &args.v[i]);
        wrote = true;
    }
    buf_putc(b, '}');
    json_args_free(&args);
}

static void append_qwen_parameter(buf *b, const json_arg *arg) {
    buf_puts(b, "<parameter=");
    buf_puts(b, arg->key ? arg->key : "");
    buf_puts(b, ">\n");
    for (const char *p = arg->value ? arg->value : ""; *p; p++) {
        if (p36_tool_text_needs_escape(p, P36_PARAM_END))
            buf_puts(b, *p == '<' ? "&lt;" : "&amp;");
        else buf_putc(b, *p);
    }
    buf_puts(b, "\n</parameter>\n");
}

static void append_qwen_arguments(buf *b, const char *json,
                                  const tool_schema_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) {
        if (json && json[0]) buf_puts(b, json);
        return;
    }
    if (order) {
        for (int i = 0; i < order->len; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_qwen_parameter(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (!args.v[i].used) append_qwen_parameter(b, &args.v[i]);
    }
    json_args_free(&args);
}

static void append_qwen_tool_calls_text(buf *b, const tool_calls *calls,
                                        const tool_schema_orders *orders,
                                        bool content_nonempty) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        if (content_nonempty) buf_puts(b, "\n\n");
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
        if (i) buf_putc(b, '\n');
        else if (content_nonempty) buf_puts(b, "\n\n");
        buf_puts(b, "<tool_call>\n<function=");
        buf_puts(b, tc->name ? tc->name : "");
        buf_puts(b, ">\n");
        append_qwen_arguments(b, tc->arguments, order);
        buf_puts(b, "</function>\n</tool_call>");
    }
}

static bool role_is_system(const char *role) {
    return !strcmp(role, "system") || !strcmp(role, "developer");
}

static void buf_puts_trimmed(buf *b, const char *s) {
    const char *start = s ? s : "";
    const char *end = start + strlen(start);
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    buf_append(b, start, (size_t)(end - start));
}

static bool text_has_trimmed_content(const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (!isspace((unsigned char)*s)) return true;
    }
    return false;
}

static char *chat_content_text(const char *s, bool controls, bool *thinking) {
    static const char think_off[] = "<|think_off|>";
    static const char think_on[] = "<|think_on|>";
    const char *remove = NULL;
    size_t remove_len = 0;

    s = s ? s : "";
    if (controls && strstr(s, think_off)) {
        *thinking = false;
        remove = think_off;
        remove_len = sizeof(think_off) - 1;
    } else if (controls && strstr(s, think_on)) {
        *thinking = true;
        remove = think_on;
        remove_len = sizeof(think_on) - 1;
    }

    buf out = {0};
    while (remove) {
        const char *p = strstr(s, remove);
        if (!p) break;
        buf_append(&out, s, (size_t)(p - s));
        s = p + remove_len;
    }
    buf_puts(&out, s);

    char *raw = buf_take(&out);
    const char *start = raw;
    const char *end = raw + strlen(raw);
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    char *trimmed = xstrndup(start, (size_t)(end - start));
    free(raw);
    return trimmed;
}

static bool ascii_contains_ci(const char *s, const char *needle) {
    size_t n = strlen(needle);
    if (!n) return true;
    for (; *s; s++) {
        size_t i = 0;
        while (i < n && s[i] &&
               tolower((unsigned char)s[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == n) return true;
    }
    return false;
}

static bool tool_response_is_error(const char *content) {
    char head[121];
    content = content ? content : "";
    size_t len = strlen(content);
    size_t head_len = len < sizeof(head) - 1 ? len : sizeof(head) - 1;
    for (size_t i = 0; i < head_len; i++)
        head[i] = (char)tolower((unsigned char)content[i]);
    head[head_len] = '\0';

    bool code = ascii_contains_ci(content, "throw new ") ||
                ascii_contains_ci(content, "throw error") ||
                ascii_contains_ci(content, "console.error") ||
                ascii_contains_ci(content, "logger.error") ||
                ascii_contains_ci(content, "logging.error") ||
                strstr(head, "import ") || strstr(head, "def ") ||
                strstr(head, "function ");
    if (code) return false;

    bool exit_zero = strstr(head, "exit code: 0") ||
                     strstr(head, "process exited with code 0");
    bool error_ok = strstr(head, "\"error\": null") ||
                    strstr(head, "\"error\":null") ||
                    strstr(head, "\"error\": false") ||
                    strstr(head, "\"error\":false") ||
                    strstr(head, "\"error\": \"\"") ||
                    strstr(head, "\"error\":\"\"");
    bool strong = (strstr(head, "\"error\":") && !error_ok) ||
                  strstr(head, "\"status\": \"error\"") ||
                  strstr(head, "\"status\":\"error\"") ||
                  strstr(head, "traceback (most recent call last):") ||
                  strstr(head, "command not found") ||
                  strstr(head, "invalid syntax") || strstr(head, "fatal:") ||
                  ((strstr(head, "exit code: ") ||
                    strstr(head, "process exited with code")) && !exit_zero) ||
                  !strncmp(head, "exception:", 10) ||
                  !strncmp(head, "failed to ", 10);
    bool weak = strstr(head, "error:") || strstr(head, "err!");
    bool weak_suppressed = strstr(head, "$ ") || strstr(head, "took ") || len >= 600;
    return strong || (weak && !weak_suppressed);
}

static void append_tool_error_warning(buf *out, int failures) {
    if (failures >= 2) {
        buf_printf(out,
            "\n\n⚠️ SYSTEM WARNING: %d consecutive tool errors detected. "
            "Your previous approach is incorrect. You MUST use a fundamentally "
            "different approach or corrected arguments.", failures);
    } else if (failures == 1) {
        buf_puts(out,
            "\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. "
            "Diagnose the failure and retry with completely corrected arguments.");
    }
}

static int last_chat_query(const chat_msgs *msgs) {
    static const char open[] = "<tool_response>";
    static const char close[] = "</tool_response>";
    for (int i = msgs->len - 1; i >= 0; i--) {
        const chat_msg *m = &msgs->v[i];
        if (strcmp(m->role, "user")) continue;
        const char *start = m->content ? m->content : "";
        const char *end = start + strlen(start);
        while (start < end && isspace((unsigned char)*start)) start++;
        while (end > start && isspace((unsigned char)end[-1])) end--;
        bool tool_result = (size_t)(end - start) >= sizeof(open) + sizeof(close) - 2 &&
                           !memcmp(start, open, sizeof(open) - 1) &&
                           !memcmp(end - (sizeof(close) - 1), close, sizeof(close) - 1);
        if (!tool_result) return i;
    }
    return -1;
}

static void apply_chat_thinking_controls(const chat_msgs *msgs, bool *thinking) {
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role) && strcmp(m->role, "user")) continue;
        const char *content = m->content ? m->content : "";
        if (strstr(content, "<|think_off|>")) *thinking = false;
        else if (strstr(content, "<|think_on|>")) *thinking = true;
    }
}

static char *leading_system_content(const chat_msgs *msgs, int *count,
                                    bool *thinking) {
    buf out = {0};
    *count = 0;
    while (*count < msgs->len && role_is_system(msgs->v[*count].role)) {
        char *part = chat_content_text(msgs->v[*count].content, true, thinking);
        if (text_has_trimmed_content(part)) {
            if (out.len) buf_puts(&out, "\n\n");
            buf_puts(&out, part);
        }
        free(part);
        (*count)++;
    }
    return buf_take(&out);
}

static char *assistant_message_parts(const chat_msg *m, char **reasoning_out) {
    static const char *ends[] = {
        "\n</think>", "\n</thinking>", "\n</ think>", "\n</think >",
    };
    const char *content = m->content ? m->content : "";
    if (text_has_trimmed_content(m->reasoning)) {
        *reasoning_out = chat_content_text(m->reasoning, false, NULL);
        const char *end = NULL;
        size_t end_len = 0;
        if (!strncmp(content, "<think>", 7)) {
            end = strstr(content, "</think>");
            end_len = 8;
        } else if (!strncmp(content, "<thinking>", 10)) {
            end = strstr(content, "</thinking>");
            end_len = 11;
        } else if (!strncmp(content, "</think>", 8)) {
            end = content;
            end_len = 8;
        } else if (!strncmp(content, "</thinking>", 11)) {
            end = content;
            end_len = 11;
        }
        if (end) content = end + end_len;
        return chat_content_text(content, false, NULL);
    }

    const char *end = NULL;
    const char *tag = NULL;
    if (!strncmp(content, "</think>", 8)) {
        end = content;
        tag = "</think>";
    } else if (!strncmp(content, "</thinking>", 11)) {
        end = content;
        tag = "</thinking>";
    } else if (!strncmp(content, "<think>", 7) && strstr(content, "</think>")) {
        end = strstr(content, "</think>");
        tag = "</think>";
    } else if (!strncmp(content, "<thinking>", 10) && strstr(content, "</thinking>")) {
        end = strstr(content, "</thinking>");
        tag = "</thinking>";
    }
    for (size_t i = 0; i < sizeof(ends) / sizeof(ends[0]); i++) {
        const char *p = strstr(content, ends[i]);
        if (p && (!end || p < end)) {
            end = p;
            tag = ends[i];
        }
    }
    if (!end) {
        *reasoning_out = xstrdup("");
        return chat_content_text(content, false, NULL);
    }

    const char *start = content;
    const char *open = strstr(content, strstr(tag, "thinking") ? "<thinking>" : "<think>");
    if (open && open < end) start = open + (strstr(tag, "thinking") ? 10 : 7);
    *reasoning_out = xstrndup(start, (size_t)(end - start));
    char *trimmed_reasoning = chat_content_text(*reasoning_out, false, NULL);
    free(*reasoning_out);
    *reasoning_out = trimmed_reasoning;
    return chat_content_text(end + strlen(tag), false, NULL);
}

static void append_kat_tools_prompt_text(buf *b, const char *tool_schemas) {
    append_tools_prompt_text(b, tool_schemas);
}

static void append_kat_tool_calls_text(buf *b, const tool_calls *calls,
                                       const tool_schema_orders *orders,
                                       bool content_nonempty) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        if (content_nonempty) buf_puts(b, "\n\n");
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
        if (i) buf_putc(b, '\n');
        else if (content_nonempty) buf_puts(b, "\n\n");
        buf_puts(b, "<tool_call>\n<function=");
        buf_puts(b, tc->name ? tc->name : "");
        buf_puts(b, ">\n");
        append_qwen_arguments(b, tc->arguments, order);
        buf_puts(b, "</function>\n</tool_call>");
    }
}

static char *render_kat_chat_prompt_text(const chat_msgs *msgs,
                                         const char *tool_schemas,
                                         const tool_schema_orders *tool_orders,
                                         p36_think_mode think_mode,
                                         bool preserve_thinking) {
    bool thinking = p36_think_mode_enabled(think_mode);
    int last_query = last_chat_query(msgs);
    int failures = 0;

    buf out = {0};
    int system_head = 0;
    char *first_content = leading_system_content(msgs, &system_head, &thinking);
    if (tool_schemas && tool_schemas[0]) {
        buf_puts(&out, "<|im_start|>system\n");
        append_kat_tools_prompt_text(&out, tool_schemas);
        if (text_has_trimmed_content(first_content)) {
            buf_puts(&out, "\n\n");
            buf_puts(&out, first_content);
        }
        buf_puts(&out, "<|im_end|>\n");
    } else if (text_has_trimmed_content(first_content)) {
        buf_puts(&out, "<|im_start|>system\n");
        buf_puts(&out, first_content);
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
            buf_puts(&out, content);
            append_kat_tool_calls_text(&out, &m->calls, tool_orders,
                                       content_nonempty);
            buf_puts(&out, "<|im_end|>\n");
            free(reasoning);
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
    else {
        buf_puts(&out, "<think>\n");
        if (think_mode == P36_THINK_MAX) buf_puts(&out, p36_think_max_prefix());
    }
    return buf_take(&out);
}

