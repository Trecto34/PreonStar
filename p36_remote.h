#ifndef P36_REMOTE_H
#define P36_REMOTE_H

/* Minimal client for a p36-server (or any OpenAI-compatible) /v1/chat/completions
 * endpoint: plain-HTTP, streaming SSE, structured tool calls.  The server owns
 * the chat template, tokenizer and reasoning split; the client only moves JSON. */

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *base_url;   /* http://host:port[/prefix] */
    const char *api_key;    /* optional Bearer token */
    int timeout_s;          /* max idle seconds between server bytes */
} p36_remote_cfg;

typedef struct {
    char *id;
    char *name;
    char *arguments;        /* JSON object text */
} p36_remote_call;

typedef struct {
    char *content;
    char *reasoning;
    p36_remote_call *calls;
    int ncalls;
    char finish[40];
    bool incomplete_tool;   /* server discarded a truncated tool call */
    int prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens;
    double prefill_ms, decode_ms;
    char err[300];
} p36_remote_result;

/* kind: 0 = content, 1 = reasoning. */
typedef void (*p36_remote_delta_fn)(void *ud, int kind, const char *s, size_t n);
typedef bool (*p36_remote_cancel_fn)(void *ud);

/* POST request_json (stream must be true).  Returns 0 on success; on failure
 * res->err is set.  res must be freed with p36_remote_result_free either way. */
int p36_remote_chat(const p36_remote_cfg *cfg, const char *request_json,
                    p36_remote_delta_fn on_delta, p36_remote_cancel_fn cancelled,
                    void *ud, p36_remote_result *res);
void p36_remote_result_free(p36_remote_result *res);

/* GET /v1/models: first model's id (malloc'd, may be NULL) and context_length (0 if unknown). */
int p36_remote_models(const p36_remote_cfg *cfg, char **model_id, int *context_length,
                      char *err, size_t err_len);

/* Append s to *out as a JSON string literal (with quotes). */
void p36_remote_json_quote(char **out, size_t *len, size_t *cap, const char *s);

/* Tiny JSON reader, exposed for converting tool-call arguments. */
typedef struct p36_jv p36_jv;
p36_jv *p36_jv_parse(const char *text);
void p36_jv_free(p36_jv *v);
const p36_jv *p36_jv_get(const p36_jv *obj, const char *key);
const p36_jv *p36_jv_at(const p36_jv *arr, int i);
int p36_jv_count(const p36_jv *v);
const char *p36_jv_key(const p36_jv *obj, int i);
/* Scalar as text (strings verbatim; numbers/bools/null spelled out); NULL for containers. */
const char *p36_jv_text(const p36_jv *v, char *scratch, size_t scratch_len);

#endif
