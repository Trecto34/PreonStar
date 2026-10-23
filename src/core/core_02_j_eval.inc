static void j_skip_ws(const char **p, const char *end) {
    while (*p < end && j_is_space(**p)) (*p)++;
}

static bool j_match_kw(const char **p, const char *end, const char *kw) {
    j_skip_ws(p, end);
    size_t n = strlen(kw);
    if ((size_t)(end - *p) < n) return false;
    if (memcmp(*p, kw, n) != 0) return false;
    if ((size_t)(end - *p) > n && j_is_ident_char((*p)[n])) return false;
    *p += n;
    return true;
}

static bool j_peek_kw(const char *p, const char *end, const char *kw) {
    return j_match_kw(&p, end, kw);
}

static const char *j_read_ident(jinja_arena *a, const char **p, const char *end) {
    j_skip_ws(p, end);
    if (*p >= end || !j_is_ident_start(**p)) return NULL;
    const char *start = *p;
    while (*p < end && j_is_ident_char(**p)) (*p)++;
    return ja_strndup(a, start, (size_t)(*p - start));
}

static void jstmt_list_push(jinja_arena *a, jstmt_list *list, jstmt s) {
    if (list->len == list->cap) {
        size_t ncap = list->cap ? list->cap * 2 : 8;
        jstmt *nitems = (jstmt *)ja_alloc(a, ncap * sizeof(jstmt));
        if (list->len) memcpy(nitems, list->items, list->len * sizeof(jstmt));
        list->items = nitems;
        list->cap = ncap;
    }
    list->items[list->len++] = s;
}

static bool j_parse_block(jinja_arena *a, const jtok *toks, size_t count, size_t *idx,
                          jstmt_list *out, const char **stop_kw, char *err, size_t errlen);

static bool j_parse_if(jinja_arena *a, const jtok *toks, size_t count, size_t *idx,
                       jstmt *out, char *err, size_t errlen) {
    const jtok *t = &toks[*idx];
    const char *p = t->ptr, *end = t->ptr + t->len;
    j_match_kw(&p, end, "if");
    j_skip_ws(&p, end);
    (*idx)++;

    size_t bcap = 4, blen = 0;
    jif_branch *branches = (jif_branch *)ja_alloc(a, bcap * sizeof(jif_branch));
    branches[blen].cond_ptr = p;
    branches[blen].cond_len = (size_t)(end - p);
    branches[blen].body = (jstmt_list){0};

    const char *stop = NULL;
    if (!j_parse_block(a, toks, count, idx, &branches[blen].body, &stop, err, errlen))
        return false;
    blen++;

    while (stop && !strcmp(stop, "elif")) {
        const jtok *et = &toks[*idx];
        const char *ep = et->ptr, *eend = et->ptr + et->len;
        j_match_kw(&ep, eend, "elif");
        j_skip_ws(&ep, eend);
        (*idx)++;
        if (blen == bcap) {
            size_t ncap = bcap * 2;
            jif_branch *nb = (jif_branch *)ja_alloc(a, ncap * sizeof(jif_branch));
            memcpy(nb, branches, blen * sizeof(jif_branch));
            branches = nb;
            bcap = ncap;
        }
        branches[blen].cond_ptr = ep;
        branches[blen].cond_len = (size_t)(eend - ep);
        branches[blen].body = (jstmt_list){0};
        stop = NULL;
        if (!j_parse_block(a, toks, count, idx, &branches[blen].body, &stop, err, errlen))
            return false;
        blen++;
    }

    if (stop && !strcmp(stop, "else")) {
        (*idx)++;
        if (blen == bcap) {
            size_t ncap = bcap * 2;
            jif_branch *nb = (jif_branch *)ja_alloc(a, ncap * sizeof(jif_branch));
            memcpy(nb, branches, blen * sizeof(jif_branch));
            branches = nb;
            bcap = ncap;
        }
        branches[blen].cond_ptr = NULL;
        branches[blen].cond_len = 0;
        branches[blen].body = (jstmt_list){0};
        stop = NULL;
        if (!j_parse_block(a, toks, count, idx, &branches[blen].body, &stop, err, errlen))
            return false;
        blen++;
    }

    if (!stop || strcmp(stop, "endif")) {
        if (err && errlen) snprintf(err, errlen, "expected endif");
        return false;
    }
    (*idx)++;
    out->kind = JSTMT_IF;
    out->u.if_stmt.branches = branches;
    out->u.if_stmt.branch_count = blen;
    return true;
}

static const char *find_for_if(const char *p, const char *end) {
    int depth = 0;
    char quote = 0;
    for (const char *q = p; q < end; q++) {
        char c = *q;
        if (quote) {
            if (c == '\\' && q + 1 < end) { q++; continue; }
            if (c == quote) quote = 0;
            continue;
        }
        if (c == '\'' || c == '"') { quote = c; continue; }
        if (c == '(' || c == '[' || c == '{') { depth++; continue; }
        if (c == ')' || c == ']' || c == '}') { if (depth > 0) depth--; continue; }
        if (depth == 0 && (q == p || !j_is_ident_char(q[-1]))) {
            const char *tmp = q;
            if (j_match_kw(&tmp, end, "if")) return q;
        }
    }
    return NULL;
}

static bool j_parse_for(jinja_arena *a, const jtok *toks, size_t count, size_t *idx,
                        jstmt *out, char *err, size_t errlen) {
    const jtok *t = &toks[*idx];
    const char *p = t->ptr, *end = t->ptr + t->len;
    j_match_kw(&p, end, "for");
    const char *var1 = j_read_ident(a, &p, end);
    const char *var2 = NULL;
    j_skip_ws(&p, end);
    if (p < end && *p == ',') {
        p++;
        var2 = j_read_ident(a, &p, end);
    }
    if (!var1 || !j_match_kw(&p, end, "in")) {
        if (err && errlen) snprintf(err, errlen, "invalid for loop syntax");
        return false;
    }
    j_skip_ws(&p, end);
    const char *if_pos = find_for_if(p, end);
    const char *iter_end = if_pos ? if_pos : end;
    const char *cond_ptr = NULL;
    size_t cond_len = 0;
    if (if_pos) {
        const char *cp = if_pos;
        j_match_kw(&cp, end, "if");
        j_skip_ws(&cp, end);
        cond_ptr = cp;
        cond_len = (size_t)(end - cp);
    }
    (*idx)++;

    memset(out, 0, sizeof(*out));
    out->kind = JSTMT_FOR;
    out->u.for_stmt.var1 = var1;
    out->u.for_stmt.var2 = var2;
    out->u.for_stmt.iter_ptr = p;
    out->u.for_stmt.iter_len = (size_t)(iter_end - p);
    out->u.for_stmt.cond_ptr = cond_ptr;
    out->u.for_stmt.cond_len = cond_len;

    const char *stop = NULL;
    if (!j_parse_block(a, toks, count, idx, &out->u.for_stmt.body, &stop, err, errlen))
        return false;
    if (stop && !strcmp(stop, "else")) {
        (*idx)++;
        stop = NULL;
        if (!j_parse_block(a, toks, count, idx, &out->u.for_stmt.else_body, &stop, err, errlen))
            return false;
    }
    if (!stop || strcmp(stop, "endfor")) {
        if (err && errlen) snprintf(err, errlen, "expected endfor");
        return false;
    }
    (*idx)++;
    return true;
}

static bool j_parse_macro(jinja_arena *a, const jtok *toks, size_t count, size_t *idx,
                          jstmt *out, char *err, size_t errlen) {
    const jtok *t = &toks[*idx];
    const char *p = t->ptr, *end = t->ptr + t->len;
    j_match_kw(&p, end, "macro");
    const char *name = j_read_ident(a, &p, end);
    j_skip_ws(&p, end);
    if (!name || p >= end || *p != '(') {
        if (err && errlen) snprintf(err, errlen, "invalid macro header");
        return false;
    }
    p++;
    size_t pcap = 4, plen = 0;
    jmacro_param *params = (jmacro_param *)ja_alloc(a, pcap * sizeof(jmacro_param));
    j_skip_ws(&p, end);
    while (p < end && *p != ')') {
        const char *pname = j_read_ident(a, &p, end);
        if (!pname) {
            if (err && errlen) snprintf(err, errlen, "invalid macro param");
            return false;
        }
        j_skip_ws(&p, end);
        const char *def_ptr = NULL;
        size_t def_len = 0;
        if (p < end && *p == '=') {
            p++;
            j_skip_ws(&p, end);
            const char *dstart = p;
            int depth = 0;
            char quote = 0;
            while (p < end) {
                char c = *p;
                if (quote) {
                    if (c == '\\' && p + 1 < end) { p += 2; continue; }
                    if (c == quote) quote = 0;
                    p++;
                    continue;
                }
                if (c == '\'' || c == '"') { quote = c; p++; continue; }
                if (c == '(' || c == '[' || c == '{') { depth++; p++; continue; }
                if (c == ')' || c == ']' || c == '}') {
                    if (depth == 0 && c == ')') break;
                    if (depth > 0) depth--;
                    p++;
                    continue;
                }
                if (depth == 0 && c == ',') break;
                p++;
            }
            def_ptr = dstart;
            def_len = (size_t)(p - dstart);
        }
        if (plen == pcap) {
            size_t ncap = pcap * 2;
            jmacro_param *np = (jmacro_param *)ja_alloc(a, ncap * sizeof(jmacro_param));
            memcpy(np, params, plen * sizeof(jmacro_param));
            params = np;
            pcap = ncap;
        }
        params[plen++] = (jmacro_param){pname, def_ptr, def_len};
        j_skip_ws(&p, end);
        if (p < end && *p == ',') p++;
        j_skip_ws(&p, end);
    }
    if (p >= end || *p != ')') {
        if (err && errlen) snprintf(err, errlen, "unclosed macro param list");
        return false;
    }
    (*idx)++;

    memset(out, 0, sizeof(*out));
    out->kind = JSTMT_MACRO;
    out->u.macro.name = name;
    out->u.macro.params = params;
    out->u.macro.param_count = plen;

    const char *stop = NULL;
    if (!j_parse_block(a, toks, count, idx, &out->u.macro.body, &stop, err, errlen))
        return false;
    if (!stop || strcmp(stop, "endmacro")) {
        if (err && errlen) snprintf(err, errlen, "expected endmacro");
        return false;
    }
    (*idx)++;
    return true;
}

static bool j_parse_set(jinja_arena *a, const jtok *t, jstmt *out, char *err, size_t errlen) {
    const char *p = t->ptr, *end = t->ptr + t->len;
    j_match_kw(&p, end, "set");
    const char *var = j_read_ident(a, &p, end);
    const char *attr = NULL;
    j_skip_ws(&p, end);
    if (p < end && *p == '.') {
        p++;
        attr = j_read_ident(a, &p, end);
    }
    j_skip_ws(&p, end);
    if (!var || p >= end || *p != '=') {
        if (err && errlen) snprintf(err, errlen, "invalid set statement");
        return false;
    }
    p++;
    j_skip_ws(&p, end);
    out->kind = JSTMT_SET;
    out->u.set.var = var;
    out->u.set.attr = attr;
    out->u.set.expr_ptr = p;
    out->u.set.expr_len = (size_t)(end - p);
    return true;
}

static bool j_parse_block(jinja_arena *a, const jtok *toks, size_t count, size_t *idx,
                          jstmt_list *out, const char **stop_kw, char *err, size_t errlen) {
    while (*idx < count) {
        const jtok *t = &toks[*idx];
        if (t->kind == JTOK_TEXT) {
            jstmt s = {.kind = JSTMT_TEXT, .u.text = {t->ptr, t->len}};
            jstmt_list_push(a, out, s);
            (*idx)++;
            continue;
        }
        if (t->kind == JTOK_EXPR) {
            jstmt s = {.kind = JSTMT_EXPR, .u.expr = {t->ptr, t->len}};
            jstmt_list_push(a, out, s);
            (*idx)++;
            continue;
        }
        const char *p = t->ptr, *end = t->ptr + t->len;
        if (j_peek_kw(p, end, "endif")) { if (stop_kw) *stop_kw = "endif"; return true; }
        if (j_peek_kw(p, end, "elif")) { if (stop_kw) *stop_kw = "elif"; return true; }
        if (j_peek_kw(p, end, "else")) { if (stop_kw) *stop_kw = "else"; return true; }
        if (j_peek_kw(p, end, "endfor")) { if (stop_kw) *stop_kw = "endfor"; return true; }
        if (j_peek_kw(p, end, "endmacro")) { if (stop_kw) *stop_kw = "endmacro"; return true; }

        jstmt s = {0};
        if (j_peek_kw(p, end, "if")) {
            if (!j_parse_if(a, toks, count, idx, &s, err, errlen)) return false;
            jstmt_list_push(a, out, s);
        } else if (j_peek_kw(p, end, "for")) {
            if (!j_parse_for(a, toks, count, idx, &s, err, errlen)) return false;
            jstmt_list_push(a, out, s);
        } else if (j_peek_kw(p, end, "macro")) {
            if (!j_parse_macro(a, toks, count, idx, &s, err, errlen)) return false;
            jstmt_list_push(a, out, s);
        } else if (j_peek_kw(p, end, "set")) {
            if (!j_parse_set(a, t, &s, err, errlen)) return false;
            jstmt_list_push(a, out, s);
            (*idx)++;
        } else if (j_peek_kw(p, end, "break")) {
            s.kind = JSTMT_BREAK;
            jstmt_list_push(a, out, s);
            (*idx)++;
        } else if (j_peek_kw(p, end, "continue")) {
            s.kind = JSTMT_CONTINUE;
            jstmt_list_push(a, out, s);
            (*idx)++;
        } else {
            if (err && errlen)
                snprintf(err, errlen, "unsupported statement: %.*s", (int)t->len, t->ptr);
            return false;
        }
    }
    if (stop_kw) *stop_kw = NULL;
    return true;
}

typedef struct jscope {
    struct jscope *parent;
    jmap *vars;
} jscope;

typedef struct {
    jinja_arena *arena;
    jscope *root_scope;
    jscope *cur_scope;
    const jstmt **macros;
    size_t macro_count;
    size_t macro_cap;
    bool failed;
    char *err;
    size_t errlen;
    bool loop_break;
    bool loop_continue;
} jctx;

static void jctx_fail(jctx *ctx, const char *msg) {
    if (ctx->failed) return;
    ctx->failed = true;
    if (ctx->err && ctx->errlen) snprintf(ctx->err, ctx->errlen, "%s", msg ? msg : "Jinja error");
}

static jval jctx_lookup(jctx *ctx, const char *name) {
    for (jscope *s = ctx->cur_scope; s; s = s->parent) {
        if (jmap_has(s->vars, name)) return jmap_get(s->vars, name);
    }
    return jval_undef();
}

static jval j_eval_expr(jctx *ctx, const char **p, const char *end, bool active);
static bool j_exec_block(jctx *ctx, const jstmt_list *block, jbuf *out);

static jval j_parse_string_lit(jctx *ctx, const char **p, const char *end) {
    char quote = **p;
    (*p)++;
    jbuf b = {0};
    while (*p < end && **p != quote) {
        char c = *(*p)++;
        if (c == '\\' && *p < end) {
            char esc = *(*p)++;
            switch (esc) {
            case 'n': jbuf_putc(&b, '\n'); break;
            case 'r': jbuf_putc(&b, '\r'); break;
            case 't': jbuf_putc(&b, '\t'); break;
            case '\\': jbuf_putc(&b, '\\'); break;
            case '\'': jbuf_putc(&b, '\''); break;
            case '"': jbuf_putc(&b, '"'); break;
            default: jbuf_putc(&b, esc); break;
            }
        } else {
            jbuf_putc(&b, c);
        }
    }
    if (*p < end && **p == quote) (*p)++;
    jval v = jval_str_n(ctx->arena, b.ptr ? b.ptr : "", b.len);
    free(b.ptr);
    return v;
}

typedef struct {
    const char *kw;
    jval val;
} jcall_arg;

static bool j_parse_call_args(jctx *ctx, const char **p, const char *end, bool active,
                              jcall_arg **out_args, size_t *out_count) {
    size_t cap = 4, len = 0;
    jcall_arg *args = (jcall_arg *)ja_alloc(ctx->arena, cap * sizeof(jcall_arg));
    j_skip_ws(p, end);
    while (*p < end && **p != ')') {
        const char *save = *p;
        const char *ident = j_read_ident(ctx->arena, p, end);
        j_skip_ws(p, end);
        const char *kw = NULL;
        if (ident && *p < end && **p == '=' && (*p + 1 >= end || (*p)[1] != '=')) {
            (*p)++;
            kw = ident;
        } else {
            *p = save;
        }
        jval val = j_eval_expr(ctx, p, end, active);
        if (ctx->failed) return false;
        if (len == cap) {
            size_t ncap = cap * 2;
            jcall_arg *na = (jcall_arg *)ja_alloc(ctx->arena, ncap * sizeof(jcall_arg));
            memcpy(na, args, len * sizeof(jcall_arg));
            args = na;
            cap = ncap;
        }
        args[len++] = (jcall_arg){kw, val};
        j_skip_ws(p, end);
        if (*p < end && **p == ',') (*p)++;
        j_skip_ws(p, end);
    }
    if (*p < end && **p == ')') (*p)++;
    *out_args = args;
    *out_count = len;
    return true;
}

static bool char_in_set(char c, const char *chars, size_t clen) {
    if (!chars) return j_is_space(c);
    for (size_t i = 0; i < clen; i++) {
        if (c == chars[i]) return true;
    }
    return false;
}

static jval j_str_strip_impl(jinja_arena *a, jval s, const char *chars, size_t clen,
                             bool left, bool right) {
    if (s.kind != JVAL_STR) return jval_str(a, "");
    const char *p = s.u.s.ptr;
    size_t start = 0, end = s.u.s.len;
    if (left) {
        while (start < end && char_in_set(p[start], chars, clen)) start++;
    }
    if (right) {
        while (end > start && char_in_set(p[end - 1], chars, clen)) end--;
    }
    return jval_str_n(a, p + start, end - start);
}

static jval j_call_str_method(jctx *ctx, jval obj, const char *method,
                              const jcall_arg *args, size_t argc) {
    const char *s = obj.u.s.ptr;
    size_t slen = obj.u.s.len;
    if (!strcmp(method, "strip") || !strcmp(method, "lstrip") || !strcmp(method, "rstrip")) {
        const char *chars = NULL;
        size_t clen = 0;
        if (argc > 0 && args[0].val.kind == JVAL_STR) {
            chars = args[0].val.u.s.ptr;
            clen = args[0].val.u.s.len;
        }
        bool left = strcmp(method, "rstrip") != 0;
        bool right = strcmp(method, "lstrip") != 0;
        return j_str_strip_impl(ctx->arena, obj, chars, clen, left, right);
    }
    if (!strcmp(method, "startswith")) {
        if (argc < 1 || args[0].val.kind != JVAL_STR) return jval_bool(false);
        size_t plen = args[0].val.u.s.len;
        return jval_bool(slen >= plen && !memcmp(s, args[0].val.u.s.ptr, plen));
    }
    if (!strcmp(method, "endswith")) {
        if (argc < 1 || args[0].val.kind != JVAL_STR) return jval_bool(false);
        size_t plen = args[0].val.u.s.len;
        return jval_bool(slen >= plen && !memcmp(s + slen - plen, args[0].val.u.s.ptr, plen));
    }
    if (!strcmp(method, "split")) {
        jval list = jval_list_new(ctx->arena, 4);
        const char *sep = (argc >= 1 && args[0].val.kind == JVAL_STR) ? args[0].val.u.s.ptr : NULL;
        size_t sep_len = (argc >= 1 && args[0].val.kind == JVAL_STR) ? args[0].val.u.s.len : 0;
        int64_t maxsplit = (argc >= 2 && args[1].val.kind == JVAL_INT) ? args[1].val.u.i : -1;
        if (!sep || sep_len == 0) {
            jlist_push(ctx->arena, list.u.list, obj);
            return list;
        }
        size_t pos = 0;
        int64_t splits = 0;
        while (pos <= slen) {
            if (maxsplit >= 0 && splits >= maxsplit) {
                jlist_push(ctx->arena, list.u.list, jval_str_n(ctx->arena, s + pos, slen - pos));
                break;
            }
            const char *found = NULL;
            if (pos + sep_len <= slen) {
                for (size_t i = pos; i + sep_len <= slen; i++) {
                    if (!memcmp(s + i, sep, sep_len)) {
                        found = s + i;
                        break;
                    }
                }
            }
            if (!found) {
                jlist_push(ctx->arena, list.u.list, jval_str_n(ctx->arena, s + pos, slen - pos));
                break;
            }
            size_t part_len = (size_t)(found - (s + pos));
            jlist_push(ctx->arena, list.u.list, jval_str_n(ctx->arena, s + pos, part_len));
            pos = (size_t)(found - s) + sep_len;
            splits++;
        }
        return list;
    }
    if (!strcmp(method, "lower")) {
        char *dst = ja_strndup(ctx->arena, s, slen);
        for (size_t i = 0; i < slen; i++) dst[i] = (char)tolower((unsigned char)dst[i]);
        return (jval){.kind = JVAL_STR, .u.s = {dst, slen}};
    }
    if (!strcmp(method, "upper")) {
        char *dst = ja_strndup(ctx->arena, s, slen);
        for (size_t i = 0; i < slen; i++) dst[i] = (char)toupper((unsigned char)dst[i]);
        return (jval){.kind = JVAL_STR, .u.s = {dst, slen}};
    }
    return jval_undef();
}

static jval j_call_function(jctx *ctx, const char *name, const jcall_arg *args, size_t argc) {
    if (!strcmp(name, "namespace")) {
        jval m = jval_map_new(ctx->arena, argc ? argc : 4);
        for (size_t i = 0; i < argc; i++) {
            if (args[i].kw) jmap_set(ctx->arena, m.u.map, args[i].kw, args[i].val);
        }
        return m;
    }
    if (!strcmp(name, "raise_exception")) {
        const char *msg = (argc > 0 && args[0].val.kind == JVAL_STR) ?
            args[0].val.u.s.ptr : "Jinja template exception";
        jctx_fail(ctx, msg);
        return jval_undef();
    }
    if (!strcmp(name, "range")) {
        int64_t start = 0, stop = 0, step = 1;
        if (argc == 1 && args[0].val.kind == JVAL_INT) stop = args[0].val.u.i;
        else if (argc >= 2 && args[0].val.kind == JVAL_INT && args[1].val.kind == JVAL_INT) {
            start = args[0].val.u.i;
            stop = args[1].val.u.i;
            if (argc >= 3 && args[2].val.kind == JVAL_INT && args[2].val.u.i != 0)
                step = args[2].val.u.i;
        }
        jval list = jval_list_new(ctx->arena, 8);
        if (step > 0) {
            for (int64_t v = start; v < stop; v += step)
                jlist_push(ctx->arena, list.u.list, jval_int(v));
        } else {
            for (int64_t v = start; v > stop; v += step)
                jlist_push(ctx->arena, list.u.list, jval_int(v));
        }
        return list;
    }
    for (size_t i = 0; i < ctx->macro_count; i++) {
        const jstmt *m = ctx->macros[i];
        if (!strcmp(m->u.macro.name, name)) {
            jscope macro_scope = {
                .parent = ctx->root_scope,
                .vars = jval_map_new(ctx->arena, 8).u.map,
            };
            for (size_t pi = 0; pi < m->u.macro.param_count; pi++) {
                const jmacro_param *mp = &m->u.macro.params[pi];
                jval val = jval_undef();
                bool found = false;
                for (size_t ai = 0; ai < argc; ai++) {
                    if (args[ai].kw && !strcmp(args[ai].kw, mp->name)) {
                        val = args[ai].val;
                        found = true;
                        break;
                    }
                }
                if (!found && pi < argc && !args[pi].kw) {
                    val = args[pi].val;
                    found = true;
                }
                if (!found && mp->def_ptr) {
                    const char *dp = mp->def_ptr;
                    val = j_eval_expr(ctx, &dp, mp->def_ptr + mp->def_len, true);
                }
                jmap_set(ctx->arena, macro_scope.vars, mp->name, val);
            }
            jscope *prev = ctx->cur_scope;
            ctx->cur_scope = &macro_scope;
            jbuf mbuf = {0};
            j_exec_block(ctx, &m->u.macro.body, &mbuf);
            ctx->cur_scope = prev;
            jval res = jval_str_n(ctx->arena, mbuf.ptr ? mbuf.ptr : "", mbuf.len);
            free(mbuf.ptr);
            return res;
        }
    }
    return jval_undef();
}

static jval j_eval_primary(jctx *ctx, const char **p, const char *end, bool active) {
    j_skip_ws(p, end);
    if (*p >= end) return jval_undef();
    char c = **p;
    if (c == '(') {
        (*p)++;
        jval v = j_eval_expr(ctx, p, end, active);
        j_skip_ws(p, end);
        if (*p < end && **p == ')') (*p)++;
        return v;
    }
    if (c == '[') {
        (*p)++;
        jval list = active ? jval_list_new(ctx->arena, 4) : jval_undef();
        j_skip_ws(p, end);
        while (*p < end && **p != ']') {
            jval item = j_eval_expr(ctx, p, end, active);
            if (active) jlist_push(ctx->arena, list.u.list, item);
            j_skip_ws(p, end);
            if (*p < end && **p == ',') (*p)++;
            j_skip_ws(p, end);
        }
        if (*p < end && **p == ']') (*p)++;
        return list;
    }
    if (c == '\'' || c == '"') {
        return j_parse_string_lit(ctx, p, end);
    }
    if (c >= '0' && c <= '9') {
        int64_t val = 0;
        while (*p < end && **p >= '0' && **p <= '9') {
            val = val * 10 + (**p - '0');
            (*p)++;
        }
        return jval_int(val);
    }
    const char *ident = j_read_ident(ctx->arena, p, end);
    if (!ident) return jval_undef();
    if (!strcmp(ident, "true") || !strcmp(ident, "True")) return jval_bool(true);
    if (!strcmp(ident, "false") || !strcmp(ident, "False")) return jval_bool(false);
    if (!strcmp(ident, "none") || !strcmp(ident, "None") || !strcmp(ident, "null")) return jval_none();

    j_skip_ws(p, end);
    if (*p < end && **p == '(') {
        (*p)++;
        jcall_arg *args = NULL;
        size_t argc = 0;
        if (!j_parse_call_args(ctx, p, end, active, &args, &argc)) return jval_undef();
        if (!active) return jval_undef();
        return j_call_function(ctx, ident, args, argc);
    }
    if (!active) return jval_undef();
    return jctx_lookup(ctx, ident);
}

static jval j_eval_subscript(jctx *ctx, jval obj, const char **p, const char *end, bool active) {
    j_skip_ws(p, end);
    const char *scan = *p;
    int depth = 0;
    char quote = 0;
    const char *colon1 = NULL, *colon2 = NULL;
    const char *close_bracket = NULL;
    while (scan < end) {
        char c = *scan;
        if (quote) {
            if (c == '\\' && scan + 1 < end) { scan += 2; continue; }
            if (c == quote) quote = 0;
            scan++;
            continue;
        }
        if (c == '\'' || c == '"') { quote = c; scan++; continue; }
        if (c == '(' || c == '[' || c == '{') { depth++; scan++; continue; }
        if (c == ')' || c == ']' || c == '}') {
            if (depth == 0 && c == ']') { close_bracket = scan; break; }
            if (depth > 0) depth--;
            scan++;
            continue;
        }
        if (depth == 0 && c == ':') {
            if (!colon1) colon1 = scan;
            else if (!colon2) colon2 = scan;
        }
        scan++;
    }
    if (!close_bracket) return jval_undef();

    if (!colon1) {
        jval idx = j_eval_expr(ctx, p, close_bracket, active);
        *p = close_bracket + 1;
        if (!active) return jval_undef();
        if (obj.kind == JVAL_LIST && idx.kind == JVAL_INT) {
            int64_t i = idx.u.i;
            int64_t len = (int64_t)obj.u.list->len;
            if (i < 0) i += len;
            if (i >= 0 && i < len) return obj.u.list->items[i];
            return jval_undef();
        }
        if (obj.kind == JVAL_MAP && idx.kind == JVAL_STR) {
            return jmap_get(obj.u.map, idx.u.s.ptr);
        }
        return jval_undef();
    }

    const char *s1 = *p, *e1 = colon1;
    const char *s2 = colon1 + 1, *e2 = colon2 ? colon2 : close_bracket;
    const char *s3 = colon2 ? colon2 + 1 : close_bracket, *e3 = close_bracket;
    *p = close_bracket + 1;

    j_skip_ws(&s1, e1);
    j_skip_ws(&s2, e2);
    j_skip_ws(&s3, e3);
    bool has_start = (s1 < e1);
    bool has_stop = (s2 < e2);
    bool has_step = (s3 < e3);

    jval vstart = has_start ? j_eval_expr(ctx, &s1, e1, active) : jval_undef();
    jval vstop = has_stop ? j_eval_expr(ctx, &s2, e2, active) : jval_undef();
    jval vstep = has_step ? j_eval_expr(ctx, &s3, e3, active) : jval_int(1);
    if (!active || obj.kind != JVAL_LIST) return jval_undef();

    int64_t step = (vstep.kind == JVAL_INT && vstep.u.i != 0) ? vstep.u.i : 1;
    int64_t len = (int64_t)obj.u.list->len;
    int64_t start, stop;
    if (step > 0) {
        start = (vstart.kind == JVAL_INT) ? (vstart.u.i < 0 ? vstart.u.i + len : vstart.u.i) : 0;
        stop = (vstop.kind == JVAL_INT) ? (vstop.u.i < 0 ? vstop.u.i + len : vstop.u.i) : len;
        if (start < 0) start = 0;
        if (stop > len) stop = len;
        jval res = jval_list_new(ctx->arena, (size_t)(stop > start ? stop - start : 1));
        for (int64_t i = start; i < stop; i += step)
            jlist_push(ctx->arena, res.u.list, obj.u.list->items[i]);
        return res;
    } else {
        start = (vstart.kind == JVAL_INT) ? (vstart.u.i < 0 ? vstart.u.i + len : vstart.u.i) : (len - 1);
        stop = (vstop.kind == JVAL_INT) ? (vstop.u.i < 0 ? vstop.u.i + len : vstop.u.i) : -1;
        if (start >= len) start = len - 1;
        if (stop < -1) stop = -1;
        jval res = jval_list_new(ctx->arena, (size_t)len);
        for (int64_t i = start; i > stop; i += step)
            jlist_push(ctx->arena, res.u.list, obj.u.list->items[i]);
        return res;
    }
}

static jval j_eval_postfix(jctx *ctx, const char **p, const char *end, bool active) {
    jval v = j_eval_primary(ctx, p, end, active);
    for (;;) {
        j_skip_ws(p, end);
        if (*p < end && **p == '.') {
            (*p)++;
            const char *attr = j_read_ident(ctx->arena, p, end);
            if (!attr) break;
            j_skip_ws(p, end);
            if (*p < end && **p == '(') {
                (*p)++;
                jcall_arg *args = NULL;
                size_t argc = 0;
                if (!j_parse_call_args(ctx, p, end, active, &args, &argc)) return jval_undef();
                if (!active) {
                    v = jval_undef();
                } else if (v.kind == JVAL_STR) {
                    v = j_call_str_method(ctx, v, attr, args, argc);
                } else if (v.kind == JVAL_MAP && !strcmp(attr, "get")) {
                    if (argc >= 1 && args[0].val.kind == JVAL_STR && jmap_has(v.u.map, args[0].val.u.s.ptr))
                        v = jmap_get(v.u.map, args[0].val.u.s.ptr);
                    else if (argc >= 2)
                        v = args[1].val;
                    else
                        v = jval_none();
                } else {
                    v = jval_undef();
                }
            } else {
                if (!active) v = jval_undef();
                else if (v.kind == JVAL_MAP) v = jmap_get(v.u.map, attr);
                else v = jval_undef();
            }
            continue;
        }
        if (*p < end && **p == '[') {
            (*p)++;
            v = j_eval_subscript(ctx, v, p, end, active);
            continue;
        }
        break;
    }
    return v;
}

static jval j_eval_filter(jctx *ctx, const char **p, const char *end, bool active) {
    jval v = j_eval_postfix(ctx, p, end, active);
    for (;;) {
        j_skip_ws(p, end);
        if (*p >= end || **p != '|' || (*p + 1 < end && (*p)[1] == '|')) break;
        (*p)++;
        const char *fname = j_read_ident(ctx->arena, p, end);
        if (!fname) break;
        jcall_arg *args = NULL;
        size_t argc = 0;
        j_skip_ws(p, end);
        if (*p < end && **p == '(') {
            (*p)++;
            if (!j_parse_call_args(ctx, p, end, active, &args, &argc)) return jval_undef();
        }
        if (!active) continue;
        if (!strcmp(fname, "trim") || !strcmp(fname, "strip")) {
            v = j_str_strip_impl(ctx->arena, jval_to_str(ctx->arena, v), NULL, 0, true, true);
        } else if (!strcmp(fname, "string")) {
            v = jval_to_str(ctx->arena, v);
        } else if (!strcmp(fname, "length") || !strcmp(fname, "count")) {
            if (v.kind == JVAL_LIST) v = jval_int((int64_t)v.u.list->len);
            else if (v.kind == JVAL_STR) v = jval_int((int64_t)v.u.s.len);
            else if (v.kind == JVAL_MAP) v = jval_int((int64_t)v.u.map->len);
            else v = jval_int(0);
        } else if (!strcmp(fname, "tojson")) {
            jbuf b = {0};
            jval_write_json(&b, v);
            v = jval_str_n(ctx->arena, b.ptr ? b.ptr : "", b.len);
            free(b.ptr);
        } else if (!strcmp(fname, "lower")) {
            v = j_call_str_method(ctx, jval_to_str(ctx->arena, v), "lower", NULL, 0);
        } else if (!strcmp(fname, "upper")) {
            v = j_call_str_method(ctx, jval_to_str(ctx->arena, v), "upper", NULL, 0);
        } else if (!strcmp(fname, "first")) {
            if (v.kind == JVAL_LIST && v.u.list->len > 0) v = v.u.list->items[0];
            else v = jval_undef();
        } else if (!strcmp(fname, "last")) {
            if (v.kind == JVAL_LIST && v.u.list->len > 0) v = v.u.list->items[v.u.list->len - 1];
            else v = jval_undef();
        } else if (!strcmp(fname, "default")) {
            bool use_def = (v.kind == JVAL_UNDEFINED);
            if (argc >= 2 && args[1].val.kind == JVAL_BOOL && args[1].val.u.b)
                use_def = !jval_truthy(v);
            if (use_def && argc >= 1) v = args[0].val;
        }
    }
    return v;
}

static jval j_eval_unary(jctx *ctx, const char **p, const char *end, bool active) {
    j_skip_ws(p, end);
    if (*p < end && **p == '-') {
        (*p)++;
        jval v = j_eval_unary(ctx, p, end, active);
        if (active && v.kind == JVAL_INT) return jval_int(-v.u.i);
        return jval_int(0);
    }
    if (*p < end && **p == '+') {
        (*p)++;
        return j_eval_unary(ctx, p, end, active);
    }
    return j_eval_filter(ctx, p, end, active);
}

static jval j_eval_mul(jctx *ctx, const char **p, const char *end, bool active) {
    jval left = j_eval_unary(ctx, p, end, active);
    for (;;) {
        j_skip_ws(p, end);
        if (*p < end && (**p == '*' || **p == '/' || **p == '%')) {
            char op = *(*p)++;
            if (op == '/' && *p < end && **p == '/') (*p)++;
            jval right = j_eval_unary(ctx, p, end, active);
            if (active && left.kind == JVAL_INT && right.kind == JVAL_INT) {
                if (op == '*') left = jval_int(left.u.i * right.u.i);
                else if (op == '/' && right.u.i != 0) left = jval_int(left.u.i / right.u.i);
                else if (op == '%' && right.u.i != 0) left = jval_int(left.u.i % right.u.i);
            }
            continue;
        }
        break;
    }
    return left;
}

static jval j_eval_add(jctx *ctx, const char **p, const char *end, bool active) {
    jval left = j_eval_mul(ctx, p, end, active);
    for (;;) {
        j_skip_ws(p, end);
        if (*p < end && (**p == '+' || **p == '-' || **p == '~')) {
            char op = *(*p)++;
            jval right = j_eval_mul(ctx, p, end, active);
            if (!active) continue;
            if (op == '~' || (op == '+' && (left.kind == JVAL_STR || right.kind == JVAL_STR))) {
                jval ls = jval_to_str(ctx->arena, left);
                jval rs = jval_to_str(ctx->arena, right);
                size_t n = ls.u.s.len + rs.u.s.len;
                char *buf = (char *)ja_alloc(ctx->arena, n + 1);
                if (ls.u.s.len) memcpy(buf, ls.u.s.ptr, ls.u.s.len);
                if (rs.u.s.len) memcpy(buf + ls.u.s.len, rs.u.s.ptr, rs.u.s.len);
                buf[n] = '\0';
                left = (jval){.kind = JVAL_STR, .u.s = {buf, n}};
            } else if (left.kind == JVAL_INT && right.kind == JVAL_INT) {
                left = jval_int(op == '+' ? left.u.i + right.u.i : left.u.i - right.u.i);
            }
            continue;
        }
        break;
    }
    return left;
}

static bool jval_in(jval needle, jval haystack) {
    if (haystack.kind == JVAL_LIST) {
        for (size_t i = 0; i < haystack.u.list->len; i++) {
            if (jval_eq(needle, haystack.u.list->items[i])) return true;
        }
        return false;
    }
    if (haystack.kind == JVAL_MAP && needle.kind == JVAL_STR) {
        return jmap_has(haystack.u.map, needle.u.s.ptr);
    }
    if (haystack.kind == JVAL_STR && needle.kind == JVAL_STR) {
        if (needle.u.s.len == 0) return true;
        if (needle.u.s.len > haystack.u.s.len) return false;
        for (size_t i = 0; i + needle.u.s.len <= haystack.u.s.len; i++) {
            if (!memcmp(haystack.u.s.ptr + i, needle.u.s.ptr, needle.u.s.len)) return true;
        }
        return false;
    }
    return false;
}

