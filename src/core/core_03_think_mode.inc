static jval j_eval_cmp(jctx *ctx, const char **p, const char *end, bool active) {
    jval left = j_eval_add(ctx, p, end, active);
    for (;;) {
        j_skip_ws(p, end);
        const char *save = *p;
        if (j_match_kw(p, end, "is")) {
            bool neg = j_match_kw(p, end, "not");
            const char *tname = j_read_ident(ctx->arena, p, end);
            if (!tname) { *p = save; break; }
            bool res = false;
            if (active) {
                if (!strcmp(tname, "defined")) res = (left.kind != JVAL_UNDEFINED);
                else if (!strcmp(tname, "undefined")) res = (left.kind == JVAL_UNDEFINED);
                else if (!strcmp(tname, "none") || !strcmp(tname, "null")) res = (left.kind == JVAL_NONE);
                else if (!strcmp(tname, "string")) res = (left.kind == JVAL_STR);
                else if (!strcmp(tname, "mapping")) res = (left.kind == JVAL_MAP);
                else if (!strcmp(tname, "iterable") || !strcmp(tname, "sequence"))
                    res = (left.kind == JVAL_LIST || left.kind == JVAL_MAP || left.kind == JVAL_STR);
                else if (!strcmp(tname, "true") || !strcmp(tname, "True"))
                    res = (left.kind == JVAL_BOOL && left.u.b);
                else if (!strcmp(tname, "false") || !strcmp(tname, "False"))
                    res = (left.kind == JVAL_BOOL && !left.u.b);
                else if (!strcmp(tname, "boolean")) res = (left.kind == JVAL_BOOL);
                else if (!strcmp(tname, "integer") || !strcmp(tname, "number")) res = (left.kind == JVAL_INT);
                if (neg) res = !res;
            }
            left = jval_bool(res);
            continue;
        }
        bool not_in = false;
        if (j_match_kw(p, end, "not")) {
            if (j_match_kw(p, end, "in")) not_in = true;
            else { *p = save; break; }
        }
        if (not_in || j_match_kw(p, end, "in")) {
            jval right = j_eval_add(ctx, p, end, active);
            if (active) {
                bool res = jval_in(left, right);
                left = jval_bool(not_in ? !res : res);
            }
            continue;
        }
        if (*p + 1 < end && ((*p)[0] == '=' || (*p)[0] == '!' || (*p)[0] == '<' || (*p)[0] == '>') && (*p)[1] == '=') {
            char op = (*p)[0];
            *p += 2;
            jval right = j_eval_add(ctx, p, end, active);
            if (active) {
                if (op == '=') left = jval_bool(jval_eq(left, right));
                else if (op == '!') left = jval_bool(!jval_eq(left, right));
                else if (op == '<' && left.kind == JVAL_INT && right.kind == JVAL_INT)
                    left = jval_bool(left.u.i <= right.u.i);
                else if (op == '>' && left.kind == JVAL_INT && right.kind == JVAL_INT)
                    left = jval_bool(left.u.i >= right.u.i);
                else left = jval_bool(false);
            }
            continue;
        }
        if (*p < end && (**p == '<' || **p == '>')) {
            char op = *(*p)++;
            jval right = j_eval_add(ctx, p, end, active);
            if (active && left.kind == JVAL_INT && right.kind == JVAL_INT) {
                left = jval_bool(op == '<' ? left.u.i < right.u.i : left.u.i > right.u.i);
            } else if (active) {
                left = jval_bool(false);
            }
            continue;
        }
        break;
    }
    return left;
}

static jval j_eval_not(jctx *ctx, const char **p, const char *end, bool active) {
    if (j_match_kw(p, end, "not")) {
        jval v = j_eval_not(ctx, p, end, active);
        return active ? jval_bool(!jval_truthy(v)) : jval_undef();
    }
    return j_eval_cmp(ctx, p, end, active);
}

static jval j_eval_and(jctx *ctx, const char **p, const char *end, bool active) {
    jval left = j_eval_not(ctx, p, end, active);
    while (j_match_kw(p, end, "and")) {
        bool rhs_active = active && jval_truthy(left);
        jval right = j_eval_not(ctx, p, end, rhs_active);
        if (active) left = rhs_active ? right : jval_bool(false);
    }
    return left;
}

static jval j_eval_or(jctx *ctx, const char **p, const char *end, bool active) {
    jval left = j_eval_and(ctx, p, end, active);
    while (j_match_kw(p, end, "or")) {
        bool rhs_active = active && !jval_truthy(left);
        jval right = j_eval_and(ctx, p, end, rhs_active);
        if (active && rhs_active) left = right;
    }
    return left;
}

static jval j_eval_expr(jctx *ctx, const char **p, const char *end, bool active) {
    jval val = j_eval_or(ctx, p, end, active);
    const char *save = *p;
    if (j_match_kw(p, end, "if")) {
        jval cond = j_eval_or(ctx, p, end, active);
        bool take_true = active && jval_truthy(cond);
        jval else_val = jval_str(ctx->arena, "");
        if (j_match_kw(p, end, "else")) {
            else_val = j_eval_expr(ctx, p, end, active && !take_true);
        }
        if (active) return take_true ? val : else_val;
        return jval_undef();
    }
    *p = save;
    return val;
}

static bool j_exec_block(jctx *ctx, const jstmt_list *block, jbuf *out) {
    for (size_t i = 0; i < block->len; i++) {
        if (ctx->failed || ctx->loop_break || ctx->loop_continue) return !ctx->failed;
        const jstmt *s = &block->items[i];
        switch (s->kind) {
        case JSTMT_TEXT:
            jbuf_append(out, s->u.text.ptr, s->u.text.len);
            break;
        case JSTMT_EXPR: {
            const char *p = s->u.expr.ptr;
            jval v = j_eval_expr(ctx, &p, s->u.expr.ptr + s->u.expr.len, true);
            if (ctx->failed) return false;
            jval sv = jval_to_str(ctx->arena, v);
            jbuf_append(out, sv.u.s.ptr, sv.u.s.len);
            break;
        }
        case JSTMT_SET: {
            const char *p = s->u.set.expr_ptr;
            jval v = j_eval_expr(ctx, &p, s->u.set.expr_ptr + s->u.set.expr_len, true);
            if (ctx->failed) return false;
            if (s->u.set.attr) {
                jval target = jctx_lookup(ctx, s->u.set.var);
                if (target.kind == JVAL_MAP) {
                    jmap_set(ctx->arena, target.u.map, s->u.set.attr, v);
                }
            } else {
                jmap_set(ctx->arena, ctx->cur_scope->vars, s->u.set.var, v);
            }
            break;
        }
        case JSTMT_IF: {
            for (size_t bi = 0; bi < s->u.if_stmt.branch_count; bi++) {
                const jif_branch *br = &s->u.if_stmt.branches[bi];
                bool take = true;
                if (br->cond_ptr) {
                    const char *p = br->cond_ptr;
                    jval cv = j_eval_expr(ctx, &p, br->cond_ptr + br->cond_len, true);
                    if (ctx->failed) return false;
                    take = jval_truthy(cv);
                }
                if (take) {
                    if (!j_exec_block(ctx, &br->body, out)) return false;
                    break;
                }
            }
            break;
        }
        case JSTMT_FOR: {
            const char *p = s->u.for_stmt.iter_ptr;
            jval iter = j_eval_expr(ctx, &p, s->u.for_stmt.iter_ptr + s->u.for_stmt.iter_len, true);
            if (ctx->failed) return false;
            if (iter.kind != JVAL_LIST || iter.u.list->len == 0) {
                if (!j_exec_block(ctx, &s->u.for_stmt.else_body, out)) return false;
                break;
            }
            jlist *items = iter.u.list;
            jscope loop_scope = {
                .parent = ctx->cur_scope,
                .vars = jval_map_new(ctx->arena, 8).u.map,
            };
            jscope *prev_scope = ctx->cur_scope;
            ctx->cur_scope = &loop_scope;
            if (s->u.for_stmt.cond_ptr) {
                jval filtered = jval_list_new(ctx->arena, items->len);
                for (size_t idx = 0; idx < items->len; idx++) {
                    jmap_set(ctx->arena, loop_scope.vars, s->u.for_stmt.var1, items->items[idx]);
                    const char *cp = s->u.for_stmt.cond_ptr;
                    jval cv = j_eval_expr(ctx, &cp, s->u.for_stmt.cond_ptr + s->u.for_stmt.cond_len, true);
                    if (ctx->failed) { ctx->cur_scope = prev_scope; return false; }
                    if (jval_truthy(cv)) jlist_push(ctx->arena, filtered.u.list, items->items[idx]);
                }
                items = filtered.u.list;
            }
            if (items->len == 0) {
                ctx->cur_scope = prev_scope;
                if (!j_exec_block(ctx, &s->u.for_stmt.else_body, out)) return false;
                break;
            }
            jval loop_obj = jval_map_new(ctx->arena, 8);
            jmap_set(ctx->arena, loop_scope.vars, "loop", loop_obj);
            for (size_t idx = 0; idx < items->len; idx++) {
                jmap_set(ctx->arena, loop_scope.vars, s->u.for_stmt.var1, items->items[idx]);
                jmap_set(ctx->arena, loop_obj.u.map, "index0", jval_int((int64_t)idx));
                jmap_set(ctx->arena, loop_obj.u.map, "index", jval_int((int64_t)(idx + 1)));
                jmap_set(ctx->arena, loop_obj.u.map, "revindex0", jval_int((int64_t)(items->len - 1 - idx)));
                jmap_set(ctx->arena, loop_obj.u.map, "revindex", jval_int((int64_t)(items->len - idx)));
                jmap_set(ctx->arena, loop_obj.u.map, "first", jval_bool(idx == 0));
                jmap_set(ctx->arena, loop_obj.u.map, "last", jval_bool(idx + 1 == items->len));
                jmap_set(ctx->arena, loop_obj.u.map, "length", jval_int((int64_t)items->len));
                jmap_set(ctx->arena, loop_obj.u.map, "previtem",
                         idx > 0 ? items->items[idx - 1] : jval_undef());
                jmap_set(ctx->arena, loop_obj.u.map, "nextitem",
                         idx + 1 < items->len ? items->items[idx + 1] : jval_undef());
                if (!j_exec_block(ctx, &s->u.for_stmt.body, out)) {
                    ctx->cur_scope = prev_scope;
                    return false;
                }
                if (ctx->loop_continue) ctx->loop_continue = false;
                if (ctx->loop_break) {
                    ctx->loop_break = false;
                    break;
                }
            }
            ctx->cur_scope = prev_scope;
            break;
        }
        case JSTMT_MACRO:
            if (ctx->macro_count == ctx->macro_cap) {
                size_t ncap = ctx->macro_cap ? ctx->macro_cap * 2 : 8;
                const jstmt **nm = (const jstmt **)ja_alloc(ctx->arena, ncap * sizeof(*nm));
                if (ctx->macro_count) memcpy(nm, ctx->macros, ctx->macro_count * sizeof(*nm));
                ctx->macros = nm;
                ctx->macro_cap = ncap;
            }
            ctx->macros[ctx->macro_count++] = s;
            break;
        case JSTMT_BREAK:
            ctx->loop_break = true;
            return true;
        case JSTMT_CONTINUE:
            ctx->loop_continue = true;
            return true;
        }
    }
    return !ctx->failed;
}

static bool p36_jinja_validate(const char *template_src, char *err, size_t errlen) {
    jinja_arena arena = {0};
    jtok *toks = NULL;
    size_t tok_count = 0;
    if (!j_lex(&arena, template_src, &toks, &tok_count, err, errlen)) {
        ja_free(&arena);
        return false;
    }
    size_t idx = 0;
    jstmt_list root_block = {0};
    const char *stop = NULL;
    bool ok = j_parse_block(&arena, toks, tok_count, &idx, &root_block, &stop, err, errlen) && !stop;
    if (!ok && stop && err && errlen && !err[0])
        snprintf(err, errlen, "unexpected block end: %s", stop);
    ja_free(&arena);
    return ok;
}

char *p36_jinja_render(const char *template_src,
                       const p36_jinja_render_params *params,
                       char *err, size_t errlen) {
    if (!template_src || !params) {
        if (err && errlen) snprintf(err, errlen, "null template or params");
        return NULL;
    }
    jinja_arena arena = {0};
    jtok *toks = NULL;
    size_t tok_count = 0;
    if (!j_lex(&arena, template_src, &toks, &tok_count, err, errlen)) {
        ja_free(&arena);
        return NULL;
    }
    size_t idx = 0;
    jstmt_list root_block = {0};
    const char *stop = NULL;
    if (!j_parse_block(&arena, toks, tok_count, &idx, &root_block, &stop, err, errlen) || stop) {
        if (stop && err && errlen && !err[0])
            snprintf(err, errlen, "unexpected block end: %s", stop);
        ja_free(&arena);
        return NULL;
    }

    jscope root_scope = {
        .parent = NULL,
        .vars = jval_map_new(&arena, 16).u.map,
    };

    jval msgs_val = jval_list_new(&arena, params->message_count ? params->message_count : 4);
    for (size_t i = 0; i < params->message_count; i++) {
        const p36_jinja_message *m = &params->messages[i];
        jval mv = jval_map_new(&arena, 6);
        jmap_set(&arena, mv.u.map, "role", jval_str(&arena, m->role ? m->role : "user"));
        if (m->parts && m->image_count > 0) {
            jval clist = jval_list_new(&arena, m->image_count * 2 + 1);
            for (size_t k = 0; k <= m->image_count; k++) {
                const char *txt = m->parts[k];
                if (txt && txt[0]) {
                    jval item = jval_map_new(&arena, 2);
                    jmap_set(&arena, item.u.map, "type", jval_str(&arena, "text"));
                    jmap_set(&arena, item.u.map, "text", jval_str(&arena, txt));
                    jlist_push(&arena, clist.u.list, item);
                }
                if (k < m->image_count) {
                    jval img = jval_map_new(&arena, 2);
                    jmap_set(&arena, img.u.map, "type", jval_str(&arena, "image"));
                    jmap_set(&arena, img.u.map, "image", jval_bool(true));
                    jlist_push(&arena, clist.u.list, img);
                }
            }
            jmap_set(&arena, mv.u.map, "content", clist);
        } else {
            jmap_set(&arena, mv.u.map, "content", jval_str(&arena, m->content ? m->content : ""));
        }
        if (m->reasoning_content && m->reasoning_content[0]) {
            jmap_set(&arena, mv.u.map, "reasoning_content", jval_str(&arena, m->reasoning_content));
        }
        if (m->tool_calls && m->tool_call_count > 0) {
            jval tclist = jval_list_new(&arena, m->tool_call_count);
            for (size_t k = 0; k < m->tool_call_count; k++) {
                const p36_jinja_tool_call *tc = &m->tool_calls[k];
                jval tcv = jval_map_new(&arena, 4);
                jval fnv = jval_map_new(&arena, 4);
                jval arg_map = jval_map_new(&arena, 2);
                arg_map.u.map->raw_json = ja_strdup(&arena, tc->arguments_json && tc->arguments_json[0] ? tc->arguments_json : "{}");
                jmap_set(&arena, fnv.u.map, "name", jval_str(&arena, tc->name ? tc->name : ""));
                jmap_set(&arena, fnv.u.map, "arguments", arg_map);
                jmap_set(&arena, tcv.u.map, "name", jval_str(&arena, tc->name ? tc->name : ""));
                jmap_set(&arena, tcv.u.map, "arguments", arg_map);
                jmap_set(&arena, tcv.u.map, "function", fnv);
                jlist_push(&arena, tclist.u.list, tcv);
            }
            jmap_set(&arena, mv.u.map, "tool_calls", tclist);
        }
        jlist_push(&arena, msgs_val.u.list, mv);
    }
    jmap_set(&arena, root_scope.vars, "messages", msgs_val);

    if (params->tools_json_lines && params->tools_json_lines[0]) {
        jval tlist = jval_list_new(&arena, 8);
        const char *p = params->tools_json_lines;
        while (*p) {
            while (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') p++;
            if (!*p) break;
            if (*p == '{') {
                const char *start = p;
                int depth = 0;
                bool in_str = false;
                while (*p) {
                    if (in_str) {
                        if (*p == '\\' && p[1]) p++;
                        else if (*p == '"') in_str = false;
                    } else {
                        if (*p == '"') in_str = true;
                        else if (*p == '{') depth++;
                        else if (*p == '}') {
                            depth--;
                            if (depth == 0) {
                                p++;
                                break;
                            }
                        }
                    }
                    p++;
                }
                size_t len = (size_t)(p - start);
                if (len > 0) {
                    jval tv = jval_map_new(&arena, 2);
                    tv.u.map->raw_json = ja_strndup(&arena, start, len);
                    jlist_push(&arena, tlist.u.list, tv);
                }
            } else {
                const char *nl = strchr(p, '\n');
                size_t len = nl ? (size_t)(nl - p) : strlen(p);
                while (len > 0 && (p[len - 1] == '\r' || p[len - 1] == ' ' || p[len - 1] == '\t')) len--;
                if (len > 0) {
                    jval tv = jval_map_new(&arena, 2);
                    tv.u.map->raw_json = ja_strndup(&arena, p, len);
                    jlist_push(&arena, tlist.u.list, tv);
                }
                if (!nl) break;
                p = nl + 1;
            }
        }
        if (tlist.u.list->len > 0) {
            jmap_set(&arena, root_scope.vars, "tools", tlist);
        }
    }

    jmap_set(&arena, root_scope.vars, "add_generation_prompt", jval_bool(params->add_generation_prompt));
    if (params->enable_thinking_set)
        jmap_set(&arena, root_scope.vars, "enable_thinking", jval_bool(params->enable_thinking));
    if (params->preserve_thinking_set)
        jmap_set(&arena, root_scope.vars, "preserve_thinking", jval_bool(params->preserve_thinking));
    if (params->reasoning_effort && params->reasoning_effort[0])
        jmap_set(&arena, root_scope.vars, "reasoning_effort", jval_str(&arena, params->reasoning_effort));

    jctx ctx = {
        .arena = &arena,
        .root_scope = &root_scope,
        .cur_scope = &root_scope,
        .err = err,
        .errlen = errlen,
    };
    jbuf out = {0};
    if (!j_exec_block(&ctx, &root_block, &out)) {
        free(out.ptr);
        ja_free(&arena);
        return NULL;
    }
    if (!out.ptr) {
        out.ptr = (char *)calloc(1, 1);
    }
    ja_free(&arena);
    return out.ptr;
}

char *p36_jinja_render_system_block(const char *template_src,
                                    const char *system_text,
                                    const char *tools_json_lines,
                                    p36_think_mode think_mode,
                                    bool thinking_enabled) {
    if (!template_src) return NULL;
    static const char sentinel[] = "__P36_USER_SPLIT__";
    p36_jinja_message msgs[2];
    size_t count = 0;
    if (system_text && system_text[0]) {
        msgs[count++] = (p36_jinja_message){.role = "system", .content = system_text};
    }
    msgs[count++] = (p36_jinja_message){.role = "user", .content = sentinel};
    p36_jinja_render_params params = {
        .messages = msgs,
        .message_count = count,
        .tools_json_lines = tools_json_lines,
        .add_generation_prompt = false,
        .enable_thinking_set = true,
        .enable_thinking = thinking_enabled,
        .reasoning_effort = thinking_enabled ? p36_jinja_effort_name(think_mode) : NULL,
    };
    char *rendered = p36_jinja_render(template_src, &params, NULL, 0);
    if (!rendered) return NULL;
    char *split = strstr(rendered, "<|im_start|>user\n__P36_USER_SPLIT__");
    if (!split) split = strstr(rendered, sentinel);
    if (split) *split = '\0';
    return rendered;
}

p36_think_mode p36_qwen38_mode_for_budget(int budget) {
    if (budget <= 8000) return P36_THINK_LOW;
    if (budget <= 16000) return P36_THINK_MEDIUM;
    if (budget <= 24000) return P36_THINK_HIGH;
    return P36_THINK_XHIGH;
}

int p36_think_close_rank_limit(int think_tokens, int start_tokens) {
    if (think_tokens < start_tokens || start_tokens <= 0) return 0;
    int64_t elapsed = (int64_t)think_tokens - start_tokens;
    if (start_tokens < 8000) {
        if (elapsed * 100 >= (int64_t)start_tokens * 98) return 64;
        if (elapsed * 100 >= (int64_t)start_tokens * 95) return 32;
        if (elapsed * 100 >= (int64_t)start_tokens * 90) return 16;
        if (elapsed * 100 >= (int64_t)start_tokens * 80) return 8;
        if (elapsed >= 4096) return 5;
        if (elapsed >= 2048) return 4;
        if (elapsed >= 1024) return 3;
        if (elapsed >= 512) return 2;
        return 1;
    }
    int64_t window = start_tokens / 2;
    if (window > 8192) window = 8192;
    if (elapsed * 4 >= window * 5) return 256;
    if (elapsed * 8 >= window * 9) return 128;
    if (elapsed >= window) return 64;
    if (elapsed * 16 >= window * 15) return 32;
    if (elapsed * 8 >= window * 7) return 16;
    if (elapsed * 4 >= window * 3) return 8;
    int64_t rank5 = window * 3 / 4;
    if (rank5 > 4096) rank5 = 4096;
    if (elapsed >= rank5) return 5;
    int64_t rank4 = window / 2;
    if (rank4 > 2048) rank4 = 2048;
    if (elapsed >= rank4) return 4;
    int64_t rank3 = window / 4;
    if (rank3 > 1024) rank3 = 1024;
    if (elapsed >= rank3) return 3;
    int64_t rank2 = window / 8;
    if (rank2 > 512) rank2 = 512;
    if (elapsed >= rank2) return 2;
    return 1;
}

const char *p36_think_max_prefix(void) {
    return P36_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t p36_think_max_min_context(void) {
    return P36_THINK_MAX_MIN_CONTEXT;
}

