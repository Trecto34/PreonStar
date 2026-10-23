#include "p36.h"
#include "p36_tool_text.h"
#include "rax.h"
#include "p36_cli_args.h"

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * Vulkan worker.  The worker owns the p36_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;
static bool g_enable_cors = false;

#define P36_SERVER_IO_TIMEOUT_SEC 10
#define P36_SERVER_SEND_STALL_TIMEOUT_MS 2000

#define P36_TOOL_CALLS_START "<tool_call>"
#define P36_TOOL_CALLS_END "</tool_call>"
#define P36_INVOKE_START "<function="
#define P36_INVOKE_END "</function>"
#define P36_PARAM_START "<parameter="
#define P36_PARAM_END "</parameter>"

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static void die(const char *msg) {
    fprintf(stderr, "p36-server: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

#include "p36_loopguard.h"

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static bool random_bytes(void *dst, size_t len) {
    unsigned char *p = dst;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            return false;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    close(fd);
    return true;
}

static void responses_random_id(char *dst, size_t dstlen, const char *prefix) {
    unsigned char bytes[12];
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    static uint64_t fallback_ctr;
    if (pos >= dstlen) return;
    if (!random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint32_t b = (uint32_t)(++fallback_ctr ^ (uint64_t)(uintptr_t)dst);
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(b));
    }
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 1 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        if (pos + 1 < dstlen) dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) {
            cap = need;
            break;
        }
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}

static void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    *out = NULL;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p || !isfinite(v)) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (!isfinite(v)) return false;
    if (v < 0) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) {
        *out = true;
        return true;
    }
    if (json_lit(p, "false")) {
        *out = false;
        return true;
    }
    return false;
}

/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

static char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;

    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            buf_putc(&b, (char)c);
        }
    }
    return buf_take(&b);
}

static bool json_raw_value_is_complete(const char *json) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (!json_skip_value(&p)) return false;
    json_ws(&p);
    return *p == '\0';
}

#define SERVER_IMAGE_MARKER_BYTES 64

typedef struct {
    char marker[SERVER_IMAGE_MARKER_BYTES];
    uint8_t *encoded;
    size_t encoded_len;
} server_image_input;

typedef struct {
    server_image_input *v;
    size_t len;
    size_t cap;
} server_image_inputs;

static void server_image_inputs_free(server_image_inputs *images) {
    if (!images) return;
    for (size_t i = 0; i < images->len; i++) free(images->v[i].encoded);
    free(images->v);
    memset(images, 0, sizeof(*images));
}

static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return 26 + c - 'a';
    if (c >= '0' && c <= '9') return 52 + c - '0';
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool server_decode_base64(const char *src, uint8_t **out, size_t *out_len) {
    const size_t n = src ? strlen(src) : 0;
    if (n == 0 || (n & 3u) != 0 || n > 64u * 1024u * 1024u)
        return false;
    size_t cap = (n / 4u) * 3u;
    uint8_t *decoded = xmalloc(cap ? cap : 1);
    size_t used = 0;
    for (size_t i = 0; i < n; i += 4) {
        int a = base64_value((unsigned char)src[i]);
        int b = base64_value((unsigned char)src[i + 1]);
        bool pad2 = src[i + 2] == '=';
        bool pad3 = src[i + 3] == '=';
        int c = pad2 ? 0 : base64_value((unsigned char)src[i + 2]);
        int d = pad3 ? 0 : base64_value((unsigned char)src[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0 ||
            (pad2 && !pad3) || ((pad2 || pad3) && i + 4 != n)) {
            free(decoded);
            return false;
        }
        uint32_t bits = ((uint32_t)a << 18) | ((uint32_t)b << 12) |
                        ((uint32_t)c << 6) | (uint32_t)d;
        decoded[used++] = (uint8_t)(bits >> 16);
        if (!pad2) decoded[used++] = (uint8_t)(bits >> 8);
        if (!pad3) decoded[used++] = (uint8_t)bits;
    }
    *out = decoded;
    *out_len = used;
    return true;
}

static bool server_image_media_type(const char *media_type) {
    return media_type &&
           (!strcasecmp(media_type, "image/png") ||
            !strcasecmp(media_type, "image/jpeg") ||
            !strcasecmp(media_type, "image/jpg"));
}

static bool server_image_inputs_push_base64(server_image_inputs *images,
                                            const char *media_type,
                                            const char *base64,
                                            char marker[SERVER_IMAGE_MARKER_BYTES]) {
    if (!images || images->len >= 16 || !server_image_media_type(media_type)) return false;
    server_image_input image = {0};
    if (!server_decode_base64(base64, &image.encoded, &image.encoded_len))
        return false;
    unsigned char nonce[12];
    if (!random_bytes(nonce, sizeof(nonce))) {
        uint64_t fallback = (uint64_t)time(NULL) ^
                            ((uint64_t)getpid() << 32) ^
                            (uint64_t)(uintptr_t)images;
        memcpy(nonce, &fallback, sizeof(fallback));
        memset(nonce + sizeof(fallback), 0, sizeof(nonce) - sizeof(fallback));
    }
    static const char hex[] = "0123456789abcdef";
    size_t pos = (size_t)snprintf(image.marker, sizeof(image.marker),
                                  "\036" "P36_IMAGE_");
    for (size_t i = 0; i < sizeof(nonce) && pos + 2 < sizeof(image.marker); i++) {
        image.marker[pos++] = hex[nonce[i] >> 4];
        image.marker[pos++] = hex[nonce[i] & 15];
    }
    image.marker[pos++] = '\x1f';
    image.marker[pos] = '\0';
    if (images->len == images->cap) {
        size_t cap = images->cap ? images->cap * 2 : 2;
        images->v = xrealloc(images->v, cap * sizeof(images->v[0]));
        images->cap = cap;
    }
    images->v[images->len++] = image;
    snprintf(marker, SERVER_IMAGE_MARKER_BYTES, "%s", image.marker);
    return true;
}

static bool server_image_inputs_push_data_uri(
        server_image_inputs *images, const char *uri,
        char marker[SERVER_IMAGE_MARKER_BYTES]) {
    static const char png[] = "data:image/png;base64,";
    static const char jpeg[] = "data:image/jpeg;base64,";
    static const char jpg[] = "data:image/jpg;base64,";
    if (!uri) return false;
    if (!strncmp(uri, png, sizeof(png) - 1))
        return server_image_inputs_push_base64(
            images, "image/png", uri + sizeof(png) - 1, marker);
    if (!strncmp(uri, jpeg, sizeof(jpeg) - 1))
        return server_image_inputs_push_base64(
            images, "image/jpeg", uri + sizeof(jpeg) - 1, marker);
    if (!strncmp(uri, jpg, sizeof(jpg) - 1))
        return server_image_inputs_push_base64(
            images, "image/jpg", uri + sizeof(jpg) - 1, marker);
    return false;
}

static void append_owned_text(char **dst, const char *text) {
    buf b = {0};
    buf_puts(&b, *dst ? *dst : "");
    buf_puts(&b, text ? text : "");
    free(*dst);
    *dst = buf_take(&b);
}

static bool json_content(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
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
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
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
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    if (!json_string(p, &s)) {
                        free(key);
                        goto fail;
                    }
                    buf_puts(&b, s);
                    free(s);
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
        } else if (!json_skip_value(p)) {
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_string_replace(const char **p, char **dst) {
    char *tmp = NULL;
    if (!json_string(p, &tmp)) return false;
    free(*dst);
    *dst = tmp;
    return true;
}

static bool json_raw_value_replace(const char **p, char **dst) {
    char *tmp = NULL;
    if (!json_raw_value(p, &tmp)) return false;
    free(*dst);
    *dst = tmp;
    return true;
}

static bool json_content_replace(const char **p, char **dst) {
    char *tmp = NULL;
    if (!json_content(p, &tmp)) return false;
    free(*dst);
    *dst = tmp;
    return true;
}

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

static void random_tool_id(char *dst, size_t dstlen, api_style api) {
    static uint64_t fallback_ctr;
    unsigned char bytes[16];
    const char *prefix = api == API_ANTHROPIC ? "toolu_" : "call_";
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    if (pos >= dstlen) return;

    if (!random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = ++fallback_ctr ^ (uint64_t)(uintptr_t)dst;
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(b));
    }

    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}

typedef struct server server;

/* One --models-dir entry: the API id is the file name without ".gguf". */
typedef struct {
    char *id;
    char *path;
} server_model;
static void server_inference_lock(server *s);
static void server_inference_unlock(server *s);
static bool server_encode_image(server *s, const server_image_input *input,
                                p36_vision_embedding *out, char *err, size_t errlen);
typedef struct server_slot server_slot;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_tool_text;
    bool replay_empty_think;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char **prop;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    server_image_inputs images;
    char *reasoning;
    char *tool_call_id;
    char **result_ids;
    size_t result_count;
    bool other_content;
    tool_calls calls;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

static void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats);
static bool tool_memory_has_id(server *s, const char *id);
static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

typedef struct {
    req_kind kind;
    api_style api;
    p36_tokens prompt;
    p36_vision_span *images;
    size_t image_count;
    char (*image_markers)[SERVER_IMAGE_MARKER_BYTES];
    char *model;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    char **continuation_ids;
    size_t continuation_count;
    char *tool_schema_key;
    tool_schema_orders tool_orders;
    int max_tokens;
    int cached_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    float presence_penalty;
    float frequency_penalty;
    float repeat_penalty;        /* multiplicative; 1.0 = off */
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    bool top_k_set;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    bool ignore_eos;
    p36_think_mode think_mode;
    int thinking_budget;
    /* Agent-control extensions (chat only, all off by default). */
    struct { bool enabled, hit; int tokens; double restore_ms, build_ms; } sys_ckpt;
    bool incomplete_tool_call;   /* a truncated tool call was discarded (never returned as text) */
    bool thinking_budget_hard;   /* top-level thinking_budget: force </think> at the cap */
    int action_budget;           /* post-think content tokens before the leak check; 0 = off */
    bool stop_after_tool_call;   /* end decode at the first complete tool call */
    int reasoning_tokens;        /* filled after decode, reported in usage */
    double prefill_s, decode_s;
    bool has_tools;
    bool kat_coder;
    bool qwen38;
    const char *chat_template;
    bool preserve_thinking;
    bool prompt_preserves_reasoning;
    tool_replay_stats tool_replay;
} request;

static void tool_call_free(tool_call *tc) {
    free(tc->id);
    free(tc->name);
    free(tc->arguments);
    memset(tc, 0, sizeof(*tc));
}

static void tool_calls_free(tool_calls *calls) {
    for (int i = 0; i < calls->len; i++) tool_call_free(&calls->v[i]);
    free(calls->raw_tool_text);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void tool_calls_push(tool_calls *calls, tool_call tc) {
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = tc;
}

static void chat_msg_free(chat_msg *m) {
    server_image_inputs_free(&m->images);
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (size_t i = 0; i < m->result_count; i++) free(m->result_ids[i]);
    free(m->result_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

static void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}

static void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 8;
        msgs->v = xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}

static void tool_schema_order_free(tool_schema_order *o) {
    free(o->name);
    for (int i = 0; i < o->len; i++) free(o->prop[i]);
    free(o->prop);
    memset(o, 0, sizeof(*o));
}

static void tool_schema_orders_free(tool_schema_orders *orders) {
    for (int i = 0; i < orders->len; i++) tool_schema_order_free(&orders->v[i]);
    free(orders->v);
    memset(orders, 0, sizeof(*orders));
}

static void tool_schema_order_prop_push(tool_schema_order *o, char *prop) {
    if (o->len == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 8;
        o->prop = xrealloc(o->prop, (size_t)o->cap * sizeof(o->prop[0]));
    }
    o->prop[o->len++] = prop;
}

static int tool_schema_orders_find_index(const tool_schema_orders *orders, const char *name) {
    if (!orders || !name) return -1;
    for (int i = 0; i < orders->len; i++) {
        if (orders->v[i].name && !strcmp(orders->v[i].name, name)) return i;
    }
    return -1;
}

static void tool_schema_orders_push(tool_schema_orders *orders, tool_schema_order order) {
    int idx = tool_schema_orders_find_index(orders, order.name);
    if (idx >= 0) {
        tool_schema_order_free(&orders->v[idx]);
        orders->v[idx] = order;
        return;
    }
    if (orders->len == orders->cap) {
        orders->cap = orders->cap ? orders->cap * 2 : 8;
        orders->v = xrealloc(orders->v, (size_t)orders->cap * sizeof(orders->v[0]));
    }
    orders->v[orders->len++] = order;
}

static const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name) {
    int idx = tool_schema_orders_find_index(orders, name);
    return idx >= 0 ? &orders->v[idx] : NULL;
}

static void request_init(request *r, req_kind kind, int max_tokens) {
    memset(r, 0, sizeof(*r));
    r->kind = kind;
    r->api = API_OPENAI;
    r->model = xstrdup("qwen3.6-35b-a3b");
    r->max_tokens = max_tokens;
    r->top_k = 0;
    r->temperature = P36_DEFAULT_TEMPERATURE;
    r->top_p = P36_DEFAULT_TOP_P;
    r->min_p = P36_DEFAULT_MIN_P;
    r->repeat_penalty = 1.0f;
    r->think_mode = P36_THINK_HIGH;
    r->preserve_thinking = true;
}

/* Defined after struct server is complete; dereferences s->default_*_penalty*. */
static void request_apply_default_penalties(request *r, server *s);

static void request_set_model_profile(request *r, p36_engine *e) {
    free(r->model);
    r->model = xstrdup(p36_engine_model_name(e));
    r->kat_coder = p36_engine_is_kat_coder(e);
    r->qwen38 = p36_engine_is_qwen38(e);
    r->chat_template = p36_engine_chat_template(e);
    if (r->kat_coder) r->presence_penalty = 1.5f;
}

static void request_apply_model_sampling_defaults(p36_engine *engine, request *r) {
    float temperature, top_p, min_p;
    int top_k;
    p36_engine_sampling_defaults(engine, &temperature, &top_k, &top_p, &min_p);
    if (!r->temperature_set) r->temperature = temperature;
    if (!r->top_k_set) r->top_k = top_k;
    if (!r->top_p_set) r->top_p = top_p;
    if (!r->min_p_set) r->min_p = min_p;
}

/* repeat_penalty must be positive; out-of-range values are clamped to the
 * 0.5..2.0 band the CLI accepts, so one extreme client value cannot wreck decoding. */
static bool json_repeat_penalty(const char **p, float *out) {
    double v = 0.0;
    if (!json_number(p, &v) || v <= 0.0) return false;
    if (v < 0.5) v = 0.5;
    if (v > 2.0) v = 2.0;
    *out = (float)v;
    return true;
}

static bool request_validate_ignore_eos(const request *r, char *err, size_t errlen) {
    if (!r->ignore_eos || (r->temperature_set && r->temperature == 0.0f)) return true;
    snprintf(err, errlen, "ignore_eos requires an explicit temperature of 0");
    return false;
}

