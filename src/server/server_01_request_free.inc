static void request_free(request *r) {
    p36_tokens_free(&r->prompt);
    for (size_t i = 0; i < r->image_count; i++)
        p36_vision_embedding_free(&r->images[i].embedding);
    free(r->images);
    free(r->image_markers);
    free(r->model);
    for (int i = 0; i < r->stops.len; i++) free(r->stops.v[i]);
    free(r->stops.v);
    free(r->raw_body);
    free(r->prompt_text);
    for (size_t i = 0; i < r->continuation_count; i++) free(r->continuation_ids[i]);
    free(r->continuation_ids);
    free(r->tool_schema_key);
    tool_schema_orders_free(&r->tool_orders);
    memset(r, 0, sizeof(*r));
}

static void request_sampling(const request *r, float *temperature, int *top_k,
                             float *top_p, float *min_p) {
    *temperature = r->temperature;
    *top_k = r->top_k;
    *top_p = r->top_p;
    *min_p = r->min_p;
}

static bool g_force_nothink;  /* --nothink / --no-think */

static p36_think_mode think_mode_from_enabled(bool enabled, p36_think_mode effort) {
    if (g_force_nothink || !enabled || effort == P36_THINK_NONE) return P36_THINK_NONE;
    return effort;
}

static p36_think_mode think_mode_from_budget(const request *r, p36_think_mode effort) {
    if (r->qwen38 && r->thinking_budget > 0 &&
        effort != P36_THINK_NONE && effort != P36_THINK_MAX)
        return p36_qwen38_mode_for_budget(r->thinking_budget);
    return effort;
}

static bool parse_reasoning_effort_name(const char *s, p36_think_mode *out) {
    if (!s || !*s) return false;
    if (!strcmp(s, "none") || !strcmp(s, "off")) {
        *out = P36_THINK_NONE;
        return true;
    }
    if (!strcmp(s, "max")) {
        *out = P36_THINK_MAX;
        return true;
    }
    if (!strcmp(s, "low") || !strcmp(s, "minimal")) *out = P36_THINK_LOW;
    else if (!strcmp(s, "medium")) *out = P36_THINK_MEDIUM;
    else if (!strcmp(s, "high")) *out = P36_THINK_HIGH;
    else if (!strcmp(s, "xhigh")) *out = P36_THINK_XHIGH;
    else {
        unsigned level = 0;
        for (const char *p = s; *p; p++) {
            if (*p < '0' || *p > '9') return false;
            level = level * 10u + (unsigned)(*p - '0');
            if (level > 100) return false;
        }
        *out = !level ? P36_THINK_NONE : level <= 33 ? P36_THINK_LOW :
               level <= 66 ? P36_THINK_MEDIUM : P36_THINK_XHIGH;
    }
    return true;
}

static bool parse_reasoning_effort_value(const char **p, p36_think_mode *out) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    char *effort = NULL;
    if (!json_string(p, &effort)) return false;
    bool ok = parse_reasoning_effort_name(effort, out);
    free(effort);
    return ok;
}

static bool parse_thinking_control_value(const char **p, bool *thinking_enabled,
                                         int *thinking_budget) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p == 't' || **p == 'f') return json_bool(p, thinking_enabled);
    if (**p != '{') return json_skip_value(p);
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
        if (!strcmp(key, "type")) {
            char *type = NULL;
            if (!json_string(p, &type)) {
                free(key);
                return false;
            }
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!strcmp(key, "budget_tokens")) {
            double value;
            if (!json_number(p, &value) || value < 0 || value > INT_MAX ||
                value != (int)value) {
                free(key);
                return false;
            }
            *thinking_budget = (int)value;
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

static bool parse_chat_template_kwargs_ex(const char **p, bool *thinking_enabled,
                                          bool *got_thinking, bool *preserve_thinking,
                                          p36_think_mode *reasoning_effort) {
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
        if (!strcmp(key, "enable_thinking")) {
            if (!json_bool(p, thinking_enabled)) {
                free(key);
                return false;
            }
            *got_thinking = true;
        } else if (!strcmp(key, "preserve_thinking")) {
            if (!json_bool(p, preserve_thinking)) {
                free(key);
                return false;
            }
        } else if (!strcmp(key, "reasoning_effort") && reasoning_effort) {
            if (!parse_reasoning_effort_value(p, reasoning_effort)) {
                free(key);
                return false;
            }
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

#ifdef P36_SERVER_TEST
static bool parse_chat_template_kwargs(const char **p, bool *thinking_enabled,
                                       bool *got_thinking, bool *preserve_thinking) {
    return parse_chat_template_kwargs_ex(p, thinking_enabled, got_thinking,
                                         preserve_thinking, NULL);
}
#endif

static bool parse_output_config_effort(const char **p, p36_think_mode *effort) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
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
        if (!strcmp(key, "effort")) {
            if (!parse_reasoning_effort_value(p, effort)) {
                free(key);
                return false;
            }
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

static bool model_alias_disables_thinking(const char *model) {
    return model && (!strcmp(model, "qwen3.6-35b-a3b-nothink") ||
                     !strcmp(model, "kat-coder-v2.5-dev-nothink"));
}

static bool model_alias_enables_thinking(const char *model) {
    return model && (!strcmp(model, "qwen3.6-35b-a3b-think") ||
                     !strcmp(model, "kat-coder-v2.5-dev-think"));
}

static void stop_list_clear(stop_list *stops) {
    for (int i = 0; i < stops->len; i++) free(stops->v[i]);
    stops->len = 0;
    stops->max_len = 0;
}

static void stop_list_push(stop_list *stops, char *s) {
    if (!s || !s[0]) {
        free(s);
        return;
    }
    if (stops->len == stops->cap) {
        stops->cap = stops->cap ? stops->cap * 2 : 4;
        stops->v = xrealloc(stops->v, (size_t)stops->cap * sizeof(stops->v[0]));
    }
    size_t n = strlen(s);
    if (n > stops->max_len) stops->max_len = n;
    stops->v[stops->len++] = s;
}

static bool parse_stop(const char **p, stop_list *out) {
    json_ws(p);
    stop_list_clear(out);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        stop_list_push(out, s);
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            stop_list_push(out, s);
        } else if (!json_skip_value(p)) {
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

static bool stop_list_find_from(const stop_list *stops, const char *text,
                                size_t from, size_t *pos, size_t *len) {
    if (!stops->len || !text) return false;
    bool found = false;
    size_t best_pos = 0, best_len = 0;
    for (int i = 0; i < stops->len; i++) {
        const char *p = strstr(text + from, stops->v[i]);
        if (!p) continue;
        size_t ppos = (size_t)(p - text);
        size_t plen = strlen(stops->v[i]);
        if (!found || ppos < best_pos) {
            found = true;
            best_pos = ppos;
            best_len = plen;
        }
    }
    if (!found) return false;
    *pos = best_pos;
    *len = best_len;
    return true;
}

static size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len) {
    /* Streaming cannot emit the last max_stop_len-1 bytes yet: a stop sequence
     * may start there and finish in the next token.  The final flush releases
     * this small tail once generation ends without a stop hit. */
    if (!stops->len || stops->max_len <= 1) return text_len;
    const size_t hold = stops->max_len - 1;
    return text_len > hold ? text_len - hold : 0;
}

static int utf8_expected_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

/* Tokenizers can split a multi-byte UTF-8 character across two tokens.  If an
 * SSE delta ends at that boundary, some clients replace the incomplete byte
 * sequence with U+FFFD and later send the corrupted text back, destroying KV
 * cache prefix matches.  Hold only the trailing incomplete character; the next
 * generated token will complete it. */
static size_t utf8_stream_safe_len(const char *s, size_t start,
                                   size_t limit, bool final) {
    if (final || !s || limit <= start) return limit;

    size_t p = limit;
    int cont = 0;
    while (p > start && cont < 4 &&
           (((unsigned char)s[p - 1] & 0xc0) == 0x80))
    {
        p--;
        cont++;
    }

    if (p == limit) {
        return utf8_expected_len((unsigned char)s[limit - 1]) > 1 ?
               limit - 1 : limit;
    }
    if (p == start && (((unsigned char)s[p] & 0xc0) == 0x80)) return start;

    size_t lead = p - 1;
    int need = utf8_expected_len((unsigned char)s[lead]);
    return (limit - lead) < (size_t)need ? lead : limit;
}

static bool parse_stream_options(const char **p, bool *include_usage) {
    json_ws(p);
    if (**p != '{') return json_skip_value(p);
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
        if (!strcmp(key, "include_usage")) {
            if (!json_bool(p, include_usage)) {
                free(key);
                return false;
            }
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

static bool parse_function_call(const char **p, tool_call *tc) {
    json_ws(p);
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
        if (!strcmp(key, "name")) {
            if (!json_string_replace(p, &tc->name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "arguments")) {
            json_ws(p);
            if (**p == '"') {
                if (!json_string_replace(p, &tc->arguments)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_raw_value_replace(p, &tc->arguments)) {
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
    return true;
bad:
    return false;
}

static bool parse_tool_calls_value(const char **p, tool_calls *calls) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        tool_call tc = {0};
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
            if (!strcmp(key, "id")) {
                if (!json_string_replace(p, &tc.id)) {
                    free(key);
                    goto bad;
                }
            } else if (!strcmp(key, "function")) {
                if (!parse_function_call(p, &tc)) {
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
        if (tc.name && tc.arguments) {
            tool_calls_push(calls, tc);
            memset(&tc, 0, sizeof(tc));
        }
        tool_call_free(&tc);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
bad:
        tool_call_free(&tc);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static void append_raw_json_line(buf *b, const char *json) {
    if (!json || !json[0]) return;
    if (b->len) buf_putc(b, '\n');
    buf_puts(b, json);
}

static char *openai_function_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        if (!strcmp(key, "function")) {
            free(key);
            if (!json_raw_value(&p, &value)) return NULL;
            return value;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}

static bool parse_schema_properties(const char *json, tool_schema_order *order) {
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return false;
        }
        p++;
        if (!strcmp(key, "properties")) {
            free(key);
            json_ws(&p);
            if (*p != '{') return false;
            p++;
            json_ws(&p);
            while (*p && *p != '}') {
                char *prop = NULL;
                if (!json_string(&p, &prop)) return false;
                json_ws(&p);
                if (*p != ':') {
                    free(prop);
                    return false;
                }
                p++;
                tool_schema_order_prop_push(order, prop);
                if (!json_skip_value(&p)) return false;
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != '}') return false;
            p++;
        } else {
            free(key);
            if (!json_skip_value(&p)) return false;
        }
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return *p == '}';
}

static void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json) {
    if (!orders || !json) return;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return;
    p++;
    tool_schema_order order = {0};
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "name")) {
            if (!json_string_replace(&p, &order.name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "input_schema") || !strcmp(key, "parameters")) {
            char *schema = NULL;
            if (!json_raw_value(&p, &schema)) {
                free(key);
                goto done;
            }
            parse_schema_properties(schema, &order);
            free(schema);
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (order.name && order.len > 0) {
        tool_schema_orders_push(orders, order);
        memset(&order, 0, sizeof(order));
    }
done:
    tool_schema_order_free(&order);
}

static char *json_minify(const char *in) {
    if (!in) return NULL;
    size_t len = strlen(in);
    char *out = xmalloc(len + 1);
    char *dst = out;
    const char *src = in;
    bool in_string = false;
    bool escaped = false;
    while (*src) {
        char c = *src++;
        if (in_string) {
            *dst++ = c;
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else {
            if (c == '"') {
                in_string = true;
                *dst++ = c;
            } else if ((unsigned char)c > ' ') {
                *dst++ = c;
            }
        }
    }
    *dst = '\0';
    return out;
}

static char *tool_schema_extract_name(const char *json) {
    if (!json) return NULL;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        if (!strcmp(key, "name")) {
            free(key);
            char *name = NULL;
            if (json_string(&p, &name)) return name;
            return NULL;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}

typedef struct {
    char *name;
    char *schema;
} parsed_tool_entry;

static int compare_tool_entries(const void *a, const void *b) {
    const parsed_tool_entry *ta = a;
    const parsed_tool_entry *tb = b;
    if (!ta->name && !tb->name) return 0;
    if (!ta->name) return 1;
    if (!tb->name) return -1;
    return strcmp(ta->name, tb->name);
}

/* OpenAI wraps tools as {"type":"function","function":{...}}. Anthropic sends
 * the function schema directly as {"name":...,"input_schema":...}. The P36
 * prompt wants one raw function schema per line, so unwrap OpenAI tools, minify
 * each schema into compact single-line JSON, and sort deterministically by tool
 * name to maximize server-side KV cache prefix reuse across turns. */
static bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders) {
    json_ws(p);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') return false;
    (*p)++;

    parsed_tool_entry *entries = NULL;
    int entry_count = 0, entry_cap = 0;
    buf schemas = {0};

    json_ws(p);
    while (**p && **p != ']') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) goto bad;
        char *function = openai_function_schema_from_tool(raw);
        const char *schema = function ? function : raw;
        char *minified = json_minify(schema);
        char *name = tool_schema_extract_name(minified ? minified : schema);

        tool_schema_orders_add_json(orders, minified ? minified : schema);

        if (entry_count == entry_cap) {
            entry_cap = entry_cap ? entry_cap * 2 : 8;
            entries = xrealloc(entries, (size_t)entry_cap * sizeof(entries[0]));
        }
        entries[entry_count].name = name ? name : xstrdup("");
        entries[entry_count].schema = minified ? minified : xstrdup(schema);
        entry_count++;

        free(function);
        free(raw);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;

    if (entry_count > 1) {
        qsort(entries, (size_t)entry_count, sizeof(entries[0]), compare_tool_entries);
    }

    for (int i = 0; i < entry_count; i++) {
        append_raw_json_line(&schemas, entries[i].schema);
        free(entries[i].name);
        free(entries[i].schema);
    }
    free(entries);

    *out = buf_take(&schemas);
    return true;

bad:
    for (int i = 0; i < entry_count; i++) {
        free(entries[i].name);
        free(entries[i].schema);
    }
    free(entries);
    buf_free(&schemas);
    return false;
}

static bool parse_image_url_value(const char **p, char **url) {
    json_ws(p);
    if (**p == '"') return json_string(p, url);
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
        bool ok = true;
        if (!strcmp(key, "url")) {
            char *next = NULL;
            ok = json_string(p, &next);
            if (ok) {
                free(*url);
                *url = next;
            }
        } else {
            ok = json_skip_value(p);
        }
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return *url != NULL;
}

static bool parse_openai_content_object(const char **p, chat_msg *msg) {
    if (**p != '{') return false;
    (*p)++;
    char *type = NULL;
    char *text = NULL;
    char *image_url = NULL;
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
        bool ok = true;
        if (!strcmp(key, "type"))
            ok = json_string_replace(p, &type);
        else if (!strcmp(key, "text"))
            ok = json_content_replace(p, &text);
        else if (!strcmp(key, "image_url")) {
            free(image_url);
            image_url = NULL;
            ok = parse_image_url_value(p, &image_url);
        } else
            ok = json_skip_value(p);
        free(key);
        if (!ok) goto bad;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;

    if (type && (!strcmp(type, "image_url") || !strcmp(type, "input_image"))) {
        char marker[SERVER_IMAGE_MARKER_BYTES];
        if (!server_image_inputs_push_data_uri(&msg->images, image_url, marker))
            goto bad;
        append_owned_text(&msg->content, marker);
    } else if (text) {
        append_owned_text(&msg->content, text);
    }
    free(type);
    free(text);
    free(image_url);
    return true;
bad:
    free(type);
    free(text);
    free(image_url);
    return false;
}

static bool parse_openai_content(const char **p, chat_msg *msg) {
    json_ws(p);
    if (**p == '"') {
        char *text = NULL;
        if (!json_string(p, &text)) return false;
        append_owned_text(&msg->content, text);
        free(text);
        return true;
    }
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *text = NULL;
            if (!json_string(p, &text)) return false;
            append_owned_text(&msg->content, text);
            free(text);
        } else if (**p == '{') {
            if (!parse_openai_content_object(p, msg)) return false;
        } else if (!json_skip_value(p)) {
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

static bool parse_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        int reasoning_rank = 0;
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
                free(msg.content);
                msg.content = NULL;
                server_image_inputs_free(&msg.images);
                if (!parse_openai_content(p, &msg)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "reasoning_content") ||
                       !strcmp(key, "thinking") ||
                       !strcmp(key, "reasoning")) {
                char *reasoning = NULL;
                int rank = !strcmp(key, "reasoning_content") ? 3 :
                           !strcmp(key, "thinking") ? 2 : 1;
                if (!json_content_replace(p, &reasoning)) {
                    free(key);
                    goto fail;
                }
                if (rank > reasoning_rank) {
                    free(msg.reasoning);
                    msg.reasoning = reasoning;
                    reasoning_rank = rank;
                } else {
                    free(reasoning);
                }
            } else if (!strcmp(key, "tool_call_id")) {
                if (!json_string_replace(p, &msg.tool_call_id)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "tool_calls")) {
                tool_calls_free(&msg.calls);
                if (!parse_tool_calls_value(p, &msg.calls)) {
                    free(key);
                    goto fail;
                }
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
        if (msg.images.len && strcmp(msg.role, "user") && strcmp(msg.role, "tool")) goto fail;
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

static void append_xml_text_escaped(buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '&') buf_puts(b, "&amp;");
        else if (*s == '<') buf_puts(b, "&lt;");
        else if (*s == '>') buf_puts(b, "&gt;");
        else buf_putc(b, *s);
    }
}

static bool append_anthropic_block_content(buf *dst, const char *text) {
    if (!text || !text[0]) return true;
    buf_puts(dst, text);
    return true;
}

