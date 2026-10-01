#include "q36_remote.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* ---------------------------------------------------------------- strings */

typedef struct { char *p; size_t len, cap; } rbuf;

static void rb_add(rbuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 256;
        while (cap < b->len + n + 1) cap *= 2;
        b->p = realloc(b->p, cap);
        if (!b->p) abort();
        b->cap = cap;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = 0;
}

static char *dupn(const char *s, size_t n) {
    char *o = malloc(n + 1);
    if (!o) abort();
    memcpy(o, s, n);
    o[n] = 0;
    return o;
}

static void append_str(char **dst, const char *s, size_t n) {
    size_t old = *dst ? strlen(*dst) : 0;
    char *o = realloc(*dst, old + n + 1);
    if (!o) abort();
    memcpy(o + old, s, n);
    o[old + n] = 0;
    *dst = o;
}

void q36_remote_json_quote(char **out, size_t *len, size_t *cap, const char *s) {
    rbuf b = {*out, *len, *cap};
    rb_add(&b, "\"", 1);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        char tmp[8];
        switch (c) {
        case '"': rb_add(&b, "\\\"", 2); break;
        case '\\': rb_add(&b, "\\\\", 2); break;
        case '\n': rb_add(&b, "\\n", 2); break;
        case '\r': rb_add(&b, "\\r", 2); break;
        case '\t': rb_add(&b, "\\t", 2); break;
        default:
            if (c < 0x20) { snprintf(tmp, sizeof(tmp), "\\u%04x", c); rb_add(&b, tmp, 6); }
            else rb_add(&b, (const char *)&c, 1);
        }
    }
    rb_add(&b, "\"", 1);
    *out = b.p; *len = b.len; *cap = b.cap;
}

/* ------------------------------------------------------------------- JSON */

struct q36_jv {
    char type;              /* s n b 0 a o */
    char *s;
    double n;
    bool b;
    struct q36_jv **kid;
    char **key;
    int nkid, cap;
};

static void jv_push(q36_jv *p, char *key, q36_jv *v) {
    if (p->nkid == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 4;
        p->kid = realloc(p->kid, (size_t)p->cap * sizeof(*p->kid));
        p->key = realloc(p->key, (size_t)p->cap * sizeof(*p->key));
        if (!p->kid || !p->key) abort();
    }
    p->kid[p->nkid] = v;
    p->key[p->nkid++] = key;
}

void q36_jv_free(q36_jv *v) {
    if (!v) return;
    for (int i = 0; i < v->nkid; i++) { q36_jv_free(v->kid[i]); free(v->key[i]); }
    free(v->kid); free(v->key); free(v->s); free(v);
}

static void jws(const char **p) { while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++; }

static void utf8_put(rbuf *b, unsigned cp) {
    char t[4];
    int n = 0;
    if (cp < 0x80) t[n++] = (char)cp;
    else if (cp < 0x800) { t[n++] = (char)(0xC0 | cp >> 6); t[n++] = (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { t[n++] = (char)(0xE0 | cp >> 12); t[n++] = (char)(0x80 | ((cp >> 6) & 63)); t[n++] = (char)(0x80 | (cp & 63)); }
    else { t[n++] = (char)(0xF0 | cp >> 18); t[n++] = (char)(0x80 | ((cp >> 12) & 63)); t[n++] = (char)(0x80 | ((cp >> 6) & 63)); t[n++] = (char)(0x80 | (cp & 63)); }
    rb_add(b, t, (size_t)n);
}

static bool hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        if (!isxdigit((unsigned char)p[i])) return false;
        v = v * 16 + (unsigned)(isdigit((unsigned char)p[i]) ? p[i] - '0' : (tolower(p[i]) - 'a' + 10));
    }
    *out = v;
    return true;
}

static char *jparse_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    rbuf b = {0};
    rb_add(&b, "", 0);
    while (*p && *p != '"') {
        if (*p != '\\') { rb_add(&b, p, 1); p++; continue; }
        p++;
        switch (*p) {
        case 'n': rb_add(&b, "\n", 1); break;
        case 'r': rb_add(&b, "\r", 1); break;
        case 't': rb_add(&b, "\t", 1); break;
        case 'b': rb_add(&b, "\b", 1); break;
        case 'f': rb_add(&b, "\f", 1); break;
        case 'u': {
            unsigned cp;
            if (!hex4(p + 1, &cp)) { free(b.p); return NULL; }
            p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u') {
                unsigned lo;
                if (hex4(p + 3, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        case 0: free(b.p); return NULL;
        default: rb_add(&b, p, 1);
        }
        p++;
    }
    if (*p != '"') { free(b.p); return NULL; }
    *pp = p + 1;
    return b.p;
}

static q36_jv *jparse(const char **pp, int depth) {
    if (depth > 64) return NULL;
    jws(pp);
    const char *p = *pp;
    q36_jv *v = calloc(1, sizeof(*v));
    if (!v) abort();
    if (*p == '"') {
        v->type = 's';
        v->s = jparse_string(pp);
        if (!v->s) goto fail;
        return v;
    }
    if (*p == '{' || *p == '[') {
        char close = *p == '{' ? '}' : ']';
        v->type = *p == '{' ? 'o' : 'a';
        p++;
        jws(&p);
        if (*p == close) { *pp = p + 1; return v; }
        for (;;) {
            jws(&p);
            char *key = NULL;
            if (v->type == 'o') {
                key = jparse_string(&p);
                if (!key) goto fail;
                jws(&p);
                if (*p++ != ':') { free(key); goto fail; }
            }
            q36_jv *kid = jparse(&p, depth + 1);
            if (!kid) { free(key); goto fail; }
            jv_push(v, key ? key : dupn("", 0), kid);
            jws(&p);
            if (*p == ',') { p++; continue; }
            if (*p == close) { *pp = p + 1; return v; }
            goto fail;
        }
    }
    if (!strncmp(p, "true", 4)) { v->type = 'b'; v->b = true; *pp = p + 4; return v; }
    if (!strncmp(p, "false", 5)) { v->type = 'b'; *pp = p + 5; return v; }
    if (!strncmp(p, "null", 4)) { v->type = '0'; *pp = p + 4; return v; }
    char *end;
    double d = strtod(p, &end);
    if (end == p) goto fail;
    v->type = 'n';
    v->n = d;
    *pp = end;
    return v;
fail:
    q36_jv_free(v);
    return NULL;
}

q36_jv *q36_jv_parse(const char *text) {
    if (!text) return NULL;
    const char *p = text;
    q36_jv *v = jparse(&p, 0);
    if (!v) return NULL;
    jws(&p);
    if (*p) { q36_jv_free(v); return NULL; }
    return v;
}

const q36_jv *q36_jv_get(const q36_jv *o, const char *key) {
    if (!o || o->type != 'o') return NULL;
    for (int i = 0; i < o->nkid; i++)
        if (!strcmp(o->key[i], key)) return o->kid[i];
    return NULL;
}

const q36_jv *q36_jv_at(const q36_jv *a, int i) {
    return a && (a->type == 'a' || a->type == 'o') && i >= 0 && i < a->nkid ? a->kid[i] : NULL;
}

int q36_jv_count(const q36_jv *v) { return v ? v->nkid : 0; }
const char *q36_jv_key(const q36_jv *o, int i) { return o && i >= 0 && i < o->nkid ? o->key[i] : NULL; }

const char *q36_jv_text(const q36_jv *v, char *scratch, size_t n) {
    if (!v) return NULL;
    switch (v->type) {
    case 's': return v->s;
    case 'n':
        if (v->n == floor(v->n) && fabs(v->n) < 1e15) snprintf(scratch, n, "%lld", (long long)v->n);
        else snprintf(scratch, n, "%.17g", v->n);
        return scratch;
    case 'b': return v->b ? "true" : "false";
    case '0': return "null";
    default: return NULL;
    }
}

static double jnum(const q36_jv *v, double def) { return v && v->type == 'n' ? v->n : def; }

/* ------------------------------------------------------------------- HTTP */

typedef struct {
    char host[256], port[16], prefix[256];
} remote_url;

static bool parse_url(const char *url, remote_url *u, char *err, size_t en) {
    memset(u, 0, sizeof(*u));
    if (!url || strncmp(url, "http://", 7)) {
        snprintf(err, en, "server URL must start with http:// (got %s)", url ? url : "(null)");
        return false;
    }
    const char *h = url + 7, *slash = strchr(h, '/');
    size_t hl = slash ? (size_t)(slash - h) : strlen(h);
    const char *colon = memchr(h, ':', hl);
    size_t nl = colon ? (size_t)(colon - h) : hl;
    if (!nl || nl >= sizeof(u->host)) { snprintf(err, en, "bad server host"); return false; }
    memcpy(u->host, h, nl);
    snprintf(u->port, sizeof(u->port), "%.*s", colon ? (int)(hl - nl - 1) : 2, colon ? colon + 1 : "80");
    if (slash) {
        snprintf(u->prefix, sizeof(u->prefix), "%s", slash);
        size_t l = strlen(u->prefix);
        while (l && u->prefix[l - 1] == '/') u->prefix[--l] = 0;
    }
    return true;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef void (*sink_fn)(void *ud, const char *data, size_t n);

typedef struct {
    int status;
    bool chunked;
    bool sse;
    /* chunk decoder */
    int cstate;            /* 0 size line, 1 data, 2 CRLF */
    size_t cleft;
    char csize[24];
    int csz_len;
} http_state;

static void body_feed(http_state *st, const char *d, size_t n, sink_fn sink, void *ud) {
    if (!st->chunked) { sink(ud, d, n); return; }
    for (size_t i = 0; i < n;) {
        if (st->cstate == 0) {
            char c = d[i++];
            if (c == '\n') {
                st->csize[st->csz_len] = 0;
                st->cleft = strtoul(st->csize, NULL, 16);
                st->csz_len = 0;
                st->cstate = st->cleft ? 1 : 2;
            } else if (c != '\r' && st->csz_len < (int)sizeof(st->csize) - 1) {
                st->csize[st->csz_len++] = c;
            }
        } else if (st->cstate == 1) {
            size_t take = n - i < st->cleft ? n - i : st->cleft;
            sink(ud, d + i, take);
            i += take;
            st->cleft -= take;
            if (!st->cleft) st->cstate = 2;
        } else {
            if (d[i++] == '\n') st->cstate = 0;
        }
    }
}

static int tcp_connect(const remote_url *u, char *err, size_t en) {
    struct addrinfo hints = {.ai_socktype = SOCK_STREAM}, *res = NULL;
    int g = getaddrinfo(u->host, u->port, &hints, &res);
    if (g) { snprintf(err, en, "cannot resolve %s: %s", u->host, gai_strerror(g)); return -1; }
    int fd = -1;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) snprintf(err, en, "cannot connect to %s:%s: %s", u->host, u->port, strerror(errno));
    return fd;
}

/* One request/response.  Decoded body bytes go to sink; non-2xx bodies go to
 * errbody (truncated).  Returns HTTP status, or -1 with err set. */
static int http_do(const q36_remote_cfg *cfg, const char *method, const char *path,
                   const char *body, sink_fn sink, void *ud, q36_remote_cancel_fn cancelled,
                   void *cancel_ud, char *errbody, size_t errbody_len, char *err, size_t en) {
    remote_url u;
    if (!parse_url(cfg->base_url, &u, err, en)) return -1;
    int fd = tcp_connect(&u, err, en);
    if (fd < 0) return -1;
    rbuf req = {0};
    char line[600];
    snprintf(line, sizeof(line), "%s %s%s HTTP/1.1\r\nHost: %s:%s\r\nConnection: close\r\nAccept: */*\r\n",
             method, u.prefix, path, u.host, u.port);
    rb_add(&req, line, strlen(line));
    if (cfg->api_key && cfg->api_key[0]) {
        snprintf(line, sizeof(line), "Authorization: Bearer %s\r\n", cfg->api_key);
        rb_add(&req, line, strlen(line));
    }
    if (body) {
        snprintf(line, sizeof(line), "Content-Type: application/json\r\nContent-Length: %zu\r\n", strlen(body));
        rb_add(&req, line, strlen(line));
    }
    rb_add(&req, "\r\n", 2);
    if (body) rb_add(&req, body, strlen(body));
    for (size_t off = 0; off < req.len;) {
        ssize_t w = send(fd, req.p + off, req.len - off, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            snprintf(err, en, "send failed: %s", strerror(errno));
            free(req.p); close(fd); return -1;
        }
        off += (size_t)w;
    }
    free(req.p);

    http_state st = {0};
    rbuf hdr = {0};
    bool in_body = false;
    size_t errlen = 0;
    int idle_limit = cfg->timeout_s > 0 ? cfg->timeout_s : 600;
    double last = now_s();
    char buf[8192];
    int rc = -1;
    for (;;) {
        if (cancelled && cancelled(cancel_ud)) { snprintf(err, en, "interrupted"); break; }
        struct pollfd pf = {.fd = fd, .events = POLLIN};
        int pr = poll(&pf, 1, 250);
        if (pr < 0 && errno == EINTR) continue;
        if (pr == 0) {
            if (now_s() - last > idle_limit) { snprintf(err, en, "server timed out after %ds", idle_limit); break; }
            continue;
        }
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n < 0) { if (errno == EINTR) continue; snprintf(err, en, "recv failed: %s", strerror(errno)); break; }
        if (n == 0) {
            if (!in_body) snprintf(err, en, "server closed the connection before replying");
            else rc = st.status;
            break;
        }
        last = now_s();
        const char *data = buf;
        size_t dn = (size_t)n;
        if (!in_body) {
            rb_add(&hdr, buf, dn);
            char *end = strstr(hdr.p, "\r\n\r\n");
            if (!end) continue;
            *end = 0;
            st.status = atoi(hdr.p + (strncmp(hdr.p, "HTTP/1.", 7) == 0 ? 9 : 0));
            for (char *l = strchr(hdr.p, '\n'); l; l = strchr(l + 1, '\n')) {
                if (!strncasecmp(l + 1, "transfer-encoding:", 18) && strcasestr(l + 1, "chunked")) st.chunked = true;
            }
            in_body = true;
            data = end + 4;
            dn = (size_t)(hdr.p + hdr.len - data);
        }
        if (st.status >= 200 && st.status < 300) {
            body_feed(&st, data, dn, sink, ud);
        } else {
            size_t take = errbody_len - 1 - errlen < dn ? errbody_len - 1 - errlen : dn;
            if (errbody && errbody_len) { memcpy(errbody + errlen, data, take); errlen += take; errbody[errlen] = 0; }
        }
    }
    free(hdr.p);
    close(fd);
    return rc;
}

/* ------------------------------------------------------------ chat client */

typedef struct {
    q36_remote_result *res;
    q36_remote_delta_fn on_delta;
    void *ud;
    rbuf line;
    rbuf whole;     /* non-SSE JSON body */
    bool done;
} chat_sink;

static q36_remote_call *call_slot(q36_remote_result *r, int idx) {
    if (idx < 0 || idx > 63) idx = 0;
    if (idx >= r->ncalls) {
        r->calls = realloc(r->calls, (size_t)(idx + 1) * sizeof(*r->calls));
        if (!r->calls) abort();
        memset(r->calls + r->ncalls, 0, (size_t)(idx + 1 - r->ncalls) * sizeof(*r->calls));
        r->ncalls = idx + 1;
    }
    return &r->calls[idx];
}

static void take_usage(q36_remote_result *r, const q36_jv *u) {
    if (!u) return;
    r->prompt_tokens = (int)jnum(q36_jv_get(u, "prompt_tokens"), r->prompt_tokens);
    r->completion_tokens = (int)jnum(q36_jv_get(u, "completion_tokens"), r->completion_tokens);
    r->cached_tokens = (int)jnum(q36_jv_get(q36_jv_get(u, "prompt_tokens_details"), "cached_tokens"), r->cached_tokens);
    r->reasoning_tokens = (int)jnum(q36_jv_get(q36_jv_get(u, "completion_tokens_details"), "reasoning_tokens"), r->reasoning_tokens);
    const q36_jv *t = q36_jv_get(u, "timings");
    r->prefill_ms = jnum(q36_jv_get(t, "prefill_ms"), r->prefill_ms);
    r->decode_ms = jnum(q36_jv_get(t, "decode_ms"), r->decode_ms);
}

static void put_text(chat_sink *c, int kind, const q36_jv *v) {
    if (!v || v->type != 's' || !v->s[0]) return;
    char **dst = kind ? &c->res->reasoning : &c->res->content;
    append_str(dst, v->s, strlen(v->s));
    if (c->on_delta) c->on_delta(c->ud, kind, v->s, strlen(v->s));
}

/* One decoded chat object: a stream chunk (delta) or a full message. */
static void take_chunk(chat_sink *c, const q36_jv *j) {
    q36_remote_result *r = c->res;
    const q36_jv *e = q36_jv_get(j, "error");
    if (e && !r->err[0]) {
        const q36_jv *m = q36_jv_get(e, "message");
        snprintf(r->err, sizeof(r->err), "server error: %s", m && m->type == 's' ? m->s : "unknown");
    }
    take_usage(r, q36_jv_get(j, "usage"));
    const q36_jv *ch = q36_jv_at(q36_jv_get(j, "choices"), 0);
    if (!ch) return;
    const q36_jv *fr = q36_jv_get(ch, "finish_reason");
    if (fr && fr->type == 's') snprintf(r->finish, sizeof(r->finish), "%s", fr->s);
    const q36_jv *inc = q36_jv_get(ch, "incomplete_tool_call");
    if (inc && inc->type == 'b' && inc->b) r->incomplete_tool = true;
    const q36_jv *d = q36_jv_get(ch, "delta");
    if (!d) d = q36_jv_get(ch, "message");
    if (!d) return;
    put_text(c, 0, q36_jv_get(d, "content"));
    const q36_jv *rs = q36_jv_get(d, "reasoning_content");
    put_text(c, 1, rs ? rs : q36_jv_get(d, "reasoning"));
    const q36_jv *tcs = q36_jv_get(d, "tool_calls");
    for (int i = 0; i < q36_jv_count(tcs); i++) {
        const q36_jv *tc = q36_jv_at(tcs, i);
        q36_remote_call *slot = call_slot(r, (int)jnum(q36_jv_get(tc, "index"), i));
        const q36_jv *id = q36_jv_get(tc, "id");
        if (id && id->type == 's') { free(slot->id); slot->id = strdup(id->s); }
        const q36_jv *fn = q36_jv_get(tc, "function");
        const q36_jv *nm = q36_jv_get(fn, "name");
        if (nm && nm->type == 's' && nm->s[0]) append_str(&slot->name, nm->s, strlen(nm->s));
        const q36_jv *ar = q36_jv_get(fn, "arguments");
        if (ar && ar->type == 's') append_str(&slot->arguments, ar->s, strlen(ar->s));
    }
}

static void sse_line(chat_sink *c, const char *l) {
    if (strncmp(l, "data:", 5)) return;
    l += 5;
    while (*l == ' ') l++;
    if (!strcmp(l, "[DONE]")) { c->done = true; return; }
    q36_jv *j = q36_jv_parse(l);
    if (j) take_chunk(c, j);
    q36_jv_free(j);
}

static void chat_sink_fn(void *ud, const char *d, size_t n) {
    chat_sink *c = ud;
    rb_add(&c->whole, d, n);
    for (size_t i = 0; i < n; i++) {
        if (d[i] == '\n') {
            if (c->line.len && c->line.p[c->line.len - 1] == '\r') c->line.len--;
            if (c->line.p) { c->line.p[c->line.len] = 0; sse_line(c, c->line.p); }
            c->line.len = 0;
        } else {
            rb_add(&c->line, d + i, 1);
        }
    }
}

int q36_remote_chat(const q36_remote_cfg *cfg, const char *request_json,
                    q36_remote_delta_fn on_delta, q36_remote_cancel_fn cancelled,
                    void *ud, q36_remote_result *res) {
    memset(res, 0, sizeof(*res));
    chat_sink c = {.res = res, .on_delta = on_delta, .ud = ud};
    char errbody[1024] = {0}, err[300] = {0};
    int st = http_do(cfg, "POST", "/v1/chat/completions", request_json, chat_sink_fn, &c,
                     cancelled, ud, errbody, sizeof(errbody), err, sizeof(err));
    int rc = 0;
    if (st < 0) {
        snprintf(res->err, sizeof(res->err), "%s", err);
        rc = -1;
    } else if (st < 200 || st >= 300) {
        snprintf(res->err, sizeof(res->err), "server returned HTTP %d: %.200s", st, errbody);
        rc = -1;
    } else {
        /* A server that ignored stream:true replies with one JSON object. */
        if (c.line.len) { c.line.p[c.line.len] = 0; sse_line(&c, c.line.p); }
        if (!res->finish[0] && c.whole.p && c.whole.p[0] == '{') {
            q36_jv *j = q36_jv_parse(c.whole.p);
            if (j) take_chunk(&c, j);
            q36_jv_free(j);
        }
        if (res->err[0]) rc = -1;
        else if (!res->finish[0]) { snprintf(res->err, sizeof(res->err), "stream ended without a finish_reason"); rc = -1; }
    }
    free(c.line.p);
    free(c.whole.p);
    return rc;
}

void q36_remote_result_free(q36_remote_result *r) {
    free(r->content);
    free(r->reasoning);
    for (int i = 0; i < r->ncalls; i++) { free(r->calls[i].id); free(r->calls[i].name); free(r->calls[i].arguments); }
    free(r->calls);
    memset(r, 0, sizeof(*r));
}

typedef struct { rbuf body; } models_sink;
static void models_sink_fn(void *ud, const char *d, size_t n) { rb_add(&((models_sink *)ud)->body, d, n); }

int q36_remote_models(const q36_remote_cfg *cfg, char **model_id, int *context_length,
                      char *err, size_t err_len) {
    models_sink ms = {{0}};
    char errbody[256] = {0};
    int st = http_do(cfg, "GET", "/v1/models", NULL, models_sink_fn, &ms, NULL, NULL,
                     errbody, sizeof(errbody), err, err_len);
    int rc = -1;
    if (st >= 200 && st < 300 && ms.body.p) {
        q36_jv *j = q36_jv_parse(ms.body.p);
        const q36_jv *m = q36_jv_at(q36_jv_get(j, "data"), 0);
        if (m) {
            const q36_jv *id = q36_jv_get(m, "id");
            if (model_id) *model_id = id && id->type == 's' ? strdup(id->s) : NULL;
            if (context_length) *context_length = (int)jnum(q36_jv_get(m, "context_length"), 0);
            rc = 0;
        } else {
            snprintf(err, err_len, "server lists no models");
        }
        q36_jv_free(j);
    } else if (st >= 0 && !err[0]) {
        snprintf(err, err_len, "GET /v1/models returned HTTP %d", st);
    }
    free(ms.body.p);
    return rc;
}
