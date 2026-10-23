#ifndef P36_H
#define P36_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "p36_ssd.h"

#define P36_DEFAULT_MODEL_FILE "Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf"
#define P36_DEFAULT_MODEL_PATH "gguf/" P36_DEFAULT_MODEL_FILE
#define P36_DEFAULT_VISION_PATH "gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf"
#define P36_DENSE_VISION_PATH "gguf/Qwen3.8-27B-mmproj-F16.gguf"
#define P36_CONTEXT_MAX 262144
#define P36_CONTEXT_ALLOC_MAX (P36_CONTEXT_MAX + 1)

typedef enum {
    P36_BACKEND_VULKAN,
    P36_BACKEND_METAL,
    P36_BACKEND_CPU,
} p36_backend;

typedef enum {
    P36_KV_CACHE_F16,
    P36_KV_CACHE_Q8_0,
    P36_KV_CACHE_Q4_0,
} p36_kv_cache_type;

typedef enum {
    P36_THINK_NONE,
    P36_THINK_HIGH,
    P36_THINK_MAX,
    P36_THINK_LOW,
    P36_THINK_MEDIUM,
    P36_THINK_XHIGH,
} p36_think_mode;

typedef enum {
    P36_LOG_DEFAULT,
    P36_LOG_PREFILL,
    P36_LOG_GENERATION,
    P36_LOG_KVCACHE,
    P36_LOG_TOOL,
    P36_LOG_WARNING,
    P36_LOG_TIMING,
    P36_LOG_OK,
    P36_LOG_ERROR,
} p36_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} p36_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} p36_token_score;

/* ds4-style sampling: untouched temperature, no fixed-mass or fixed-count
 * truncation, min-p as the only filter.  The floor scales with the top
 * token, so confident steps prune hard while flat (quant-noise) steps drop
 * the junk tail that fixed top-p/top-k let through on the Q2 expert mix. */
#define P36_DEFAULT_TEMPERATURE 1.0f
#define P36_DEFAULT_TOP_P 1.0f
#define P36_DEFAULT_MIN_P 0.05f

typedef struct p36_engine p36_engine;
typedef struct p36_session p36_session;

typedef struct {
    p36_session *session;
    int token;
} p36_decode_item;

typedef void (*p36_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*p36_session_cancel_fn)(void *ud);

#define P36_SESSION_SYNC_INTERRUPTED 2

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *vision_path;
    const char *chat_template;
    p36_backend backend;
    int n_threads;
    int context_size;   /* Planned context per session; zero uses 4096. */
    int session_count;  /* Planned simultaneous sessions; zero uses one. */
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint32_t ssd_streaming_full_layers;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    p36_kv_cache_type cache_type_k;
    p36_kv_cache_type cache_type_v;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
} p36_engine_options;

typedef struct {
    float *data;
    uint32_t token_count;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t width;
    uint32_t height;
    uint8_t fingerprint[32];
} p36_vision_embedding;

typedef struct {
    uint32_t token_start;
    p36_vision_embedding embedding;
} p36_vision_span;

typedef void (*p36_token_emit_fn)(void *ud, int token);
typedef void (*p36_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} p36_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} p36_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} p36_session_payload_file;

int p36_engine_open(p36_engine **out, const p36_engine_options *opt);
void p36_engine_close(p36_engine *e);
void p36_engine_summary(p36_engine *e);
int p36_inspect_model(const char *model_path);
int p36_engine_embd_dim(p36_engine *e);
int p36_chat_append_multimodal_message(p36_engine *e, p36_tokens *tokens,
                                       const char *role, const char **parts,
                                       p36_vision_embedding *images, size_t image_count,
                                       p36_vision_span *spans, char *err, size_t errlen);
int p36_engine_vocab_size(p36_engine *e);
int p36_engine_power(p36_engine *e);
int p36_engine_set_power(p36_engine *e, int power_percent);
const char *p36_engine_model_name(p36_engine *e);
int p36_engine_model_id(p36_engine *e);
/* Effective prefill chunk (tokens per forward pass): chunk boundaries decide the
 * floating-point schedule, so cache checkpoints must land on multiples of it. */
uint32_t p36_engine_prefill_chunk(p36_engine *e);
typedef struct {
    const char *name;
    const char *arguments_json;
} p36_jinja_tool_call;

typedef struct {
    const char *role;
    const char *content;
    const char *const *parts;
    size_t image_count;
    const char *reasoning_content;
    const p36_jinja_tool_call *tool_calls;
    size_t tool_call_count;
} p36_jinja_message;

typedef struct {
    const p36_jinja_message *messages;
    size_t message_count;
    const char *tools_json_lines;
    bool add_generation_prompt;
    bool enable_thinking_set;
    bool enable_thinking;
    bool preserve_thinking_set;
    bool preserve_thinking;
    const char *reasoning_effort;
} p36_jinja_render_params;

bool p36_engine_is_qwen38(p36_engine *e);
bool p36_engine_is_kat_coder(p36_engine *e);
const char *p36_engine_chat_template(p36_engine *e);
const char *p36_qwen38_effort_instruction(p36_think_mode mode);
const char *p36_jinja_effort_name(p36_think_mode mode);
char *p36_jinja_render(const char *template_src,
                       const p36_jinja_render_params *params,
                       char *err, size_t errlen);
char *p36_jinja_render_system_block(const char *template_src,
                                    const char *system_text,
                                    const char *tools_json_lines,
                                    p36_think_mode think_mode,
                                    bool thinking_enabled);
p36_think_mode p36_qwen38_mode_for_budget(int budget);
int p36_think_close_rank_limit(int think_tokens, int start_tokens);
bool p36_engine_has_vision(p36_engine *e);
int p36_engine_vision_encode_file(p36_engine *e, const char *path,
                                  p36_vision_embedding *out,
                                  char *err, size_t errlen);
int p36_engine_vision_encode_memory(p36_engine *e,
                                    const uint8_t *data, size_t len,
                                    p36_vision_embedding *out,
                                    char *err, size_t errlen);
void p36_vision_embedding_free(p36_vision_embedding *embedding);
int p36_prompt_append_vision(p36_engine *e, p36_tokens *prompt,
                             p36_vision_span *span,
                             p36_vision_embedding *embedding,
                             char *err, size_t errlen);
void p36_engine_sampling_defaults(p36_engine *e, float *temperature,
                                  int *top_k, float *top_p, float *min_p);
const char *p36_backend_name(p36_backend backend);
const char *p36_kv_cache_type_name(p36_kv_cache_type type);
bool p36_parse_kv_cache_type(const char *s, p36_kv_cache_type *out);
p36_kv_cache_type p36_default_kv_cache_type_k(p36_backend backend, bool ssd_streaming);
p36_kv_cache_type p36_default_kv_cache_type_v(p36_backend backend, bool ssd_streaming);
bool p36_think_mode_enabled(p36_think_mode mode);
const char *p36_think_mode_name(p36_think_mode mode);
const char *p36_think_max_prefix(void);
uint32_t p36_think_max_min_context(void);
p36_think_mode p36_think_mode_for_context(p36_think_mode mode, int ctx_size);
p36_context_memory p36_context_memory_estimate(p36_backend backend, int ctx_size);
p36_context_memory p36_context_memory_estimate_with_prefill(
        p36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
/* Sets an opt-in GPU fast-path switch (see p36.c).  Front ends only. */
void p36_set_gpu_fast_path_env(const char *name, const char *value);

p36_context_memory p36_context_memory_estimate_configured(
        p36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v);
bool p36_log_is_tty(FILE *fp);
void p36_log(FILE *fp, p36_log_type type, const char *fmt, ...);
int p36_engine_generate_argmax(p36_engine *e, const p36_tokens *prompt,
                               int n_predict, int ctx_size,
                               p36_token_emit_fn emit,
                               p36_generation_done_fn done,
                               void *emit_ud,
                               p36_session_progress_fn progress,
                               void *progress_ud);
void p36_engine_dump_tokens(p36_engine *e, const p36_tokens *tokens);
int p36_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int p36_engine_head_test(p36_engine *e, const p36_tokens *prompt);
int p36_engine_first_token_test(p36_engine *e, const p36_tokens *prompt);
int p36_engine_vulkan_graph_test(p36_engine *e, const p36_tokens *prompt);
int p36_engine_vulkan_graph_full_test(p36_engine *e, const p36_tokens *prompt);
int p36_engine_vulkan_graph_prompt_test(p36_engine *e, const p36_tokens *prompt, int ctx_size);

void p36_tokens_push(p36_tokens *tv, int token);
void p36_tokens_free(p36_tokens *tv);
void p36_tokens_copy(p36_tokens *dst, const p36_tokens *src);
bool p36_tokens_starts_with(const p36_tokens *tokens, const p36_tokens *prefix);

void p36_tokenize_text(p36_engine *e, const char *text, p36_tokens *out);
void p36_tokenize_rendered_chat(p36_engine *e, const char *text, p36_tokens *out);
void p36_chat_apply_thinking_control(const char *content, bool *thinking);
void p36_chat_begin(p36_engine *e, p36_tokens *tokens);
void p36_encode_chat_prompt(
        p36_engine *e,
        const char *system,
        const char *prompt,
        p36_think_mode think_mode,
        p36_tokens *out);
void p36_chat_append_max_effort_prefix(p36_engine *e, p36_tokens *tokens);
void p36_chat_append_message(p36_engine *e, p36_tokens *tokens, const char *role, const char *content);
int p36_chat_append_vision_message(p36_engine *e, p36_tokens *tokens,
                                    const char *role, const char *content,
                                    p36_vision_span *span,
                                    p36_vision_embedding *embedding,
                                    char *err, size_t errlen);
void p36_chat_append_assistant_prefix(p36_engine *e, p36_tokens *tokens, p36_think_mode think_mode);

char *p36_token_text(p36_engine *e, int token, size_t *len);
int p36_token_eos(p36_engine *e);

int p36_session_create(p36_session **out, p36_engine *e, int ctx_size);
void p36_session_free(p36_session *s);
int p36_session_power(p36_session *s);
int p36_session_set_power(p36_session *s, int power_percent);
float p36_session_directional_steering_ffn(p36_session *s);
int p36_session_set_directional_steering_ffn(p36_session *s, float scale);
void p36_session_set_progress(p36_session *s, p36_session_progress_fn fn, void *ud);
void p36_session_set_display_progress(p36_session *s, p36_session_progress_fn fn, void *ud);
void p36_session_set_cancel(p36_session *s, p36_session_cancel_fn fn, void *ud);

typedef enum {
    P36_SESSION_REWRITE_ERROR = -1,
    P36_SESSION_REWRITE_OK = 0,
    P36_SESSION_REWRITE_REBUILD_NEEDED = 1,
} p36_session_rewrite_result;

int p36_session_sync(p36_session *s, const p36_tokens *prompt, char *err, size_t errlen);
int p36_session_sync_vision(p36_session *s, const p36_tokens *prompt,
                            const p36_vision_span *images, size_t image_count,
                            char *err, size_t errlen);
bool p36_session_vision_prefix_matches(const p36_session *s,
                                       const p36_vision_span *images, size_t count);
int p36_session_argmax_penalized_excluding(p36_session *s, int excluded_id,
                                           const int *tokens, int n_tokens,
                                           float presence, float frequency,
                                           float repeat_penalty);
bool p36_session_has_vision_state(const p36_session *s);
/* Identity metadata only; embedding data is NULL. Valid until the next sync. */
const p36_vision_span *p36_session_vision_spans(const p36_session *s, size_t *count);
bool p36_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
p36_session_rewrite_result p36_session_rewrite_from_common(
        p36_session *s, const p36_tokens *prompt, int common,
        char *err, size_t errlen);
int p36_session_common_prefix(p36_session *s, const p36_tokens *prompt);
int p36_session_argmax(p36_session *s);
int p36_session_argmax_excluding(p36_session *s, int excluded_id);
int p36_session_sample(p36_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int p36_session_sample_penalized(p36_session *s,
                                 float temperature, int top_k,
                                 float top_p, float min_p,
                                 const int *tokens, int n_tokens,
                                 float presence_penalty,
                                 float frequency_penalty,
                                 float repeat_penalty,
                                 uint64_t *rng);
#ifdef P36_TEST_HOOKS
int p36_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);
int p36_test_sample_logits_penalized(float *logits, uint32_t n_vocab,
                                     float temperature, int top_k,
                                     float top_p, float min_p,
                                     const int *tokens, int n_tokens,
                                     float presence_penalty,
                                     float frequency_penalty,
                                     float repeat_penalty,
                                     uint64_t *rng, float *prob_scratch);
uint32_t p36_test_kv_next_cap(uint32_t cap, uint32_t need, uint32_t limit);
#endif
bool p36_session_in_think(p36_session *s);
int p36_session_eos_to_think_close(p36_session *s, int token);
int p36_session_token_rank(p36_session *s, int token, int max_rank);
int p36_session_top_logprobs(p36_session *s, p36_token_score *out, int k);
int p36_session_eval(p36_session *s, int token, char *err, size_t errlen);
int p36_sessions_eval_batch(p36_decode_item *items, int count,
                            char *err, size_t errlen);
int p36_sessions_eval_batch_with_prefill(
        p36_decode_item *items, int count,
        p36_session *prefill_session, const p36_tokens *prefill_prompt,
        char *err, size_t errlen);
int p36_session_eval_speculative_argmax(p36_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
const float *p36_session_logits(p36_session *s, int *n_vocab);
void p36_session_invalidate(p36_session *s);
void p36_session_rewind(p36_session *s, int pos);
int p36_session_pos(p36_session *s);
int p36_session_ctx(p36_session *s);
int p36_engine_routed_quant_bits(p36_engine *e);
bool p36_engine_has_mtp(p36_engine *e);
int p36_engine_mtp_draft_tokens(p36_engine *e);
const p36_tokens *p36_session_tokens(p36_session *s);
int p36_engine_debug_tensor_row(p36_engine *e, const char *tensor_name, uint64_t row, float *dst, uint32_t n);
int p36_engine_debug_first_tensor_of_type(p36_engine *e, uint32_t type, char *name, size_t name_cap, uint32_t *n);
uint32_t p36_engine_debug_count_tensors_of_type(p36_engine *e, uint32_t type);
int p36_engine_debug_tensor_of_type_at(p36_engine *e, uint32_t type, uint32_t ordinal,
                                       char *name, size_t name_cap,
                                       uint32_t *in_dim, uint32_t *out_dim);
int p36_engine_debug_tensor_row_packed(p36_engine *e, const char *tensor_name, uint64_t row,
                                       void *dst, uint64_t dst_cap,
                                       uint64_t *row_bytes, uint32_t *type, uint32_t *n);

uint64_t p36_session_payload_bytes(p36_session *s);
int p36_session_stage_payload(p36_session *s, p36_session_payload_file *out,
                              char *err, size_t errlen);
int p36_session_write_staged_payload(const p36_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void p36_session_payload_file_free(p36_session_payload_file *payload);
int p36_session_save_payload(p36_session *s, FILE *fp, char *err, size_t errlen);
int p36_session_load_payload(p36_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int p36_session_save_snapshot(p36_session *s, p36_session_snapshot *snap, char *err, size_t errlen);
int p36_session_load_snapshot(p36_session *s, const p36_session_snapshot *snap, char *err, size_t errlen);
void p36_session_snapshot_free(p36_session_snapshot *snap);

int p36_qwen35_n_layer(void);
int p36_qwen35_n_embd(void);
int p36_qwen35_n_vocab(void);
int p36_engine_prefill_layers(p36_engine *e,
                              const p36_tokens *prompt,
                              int ctx_size,
                              float *layer_out,
                              float *logits,
                              char *err,
                              size_t errlen);
int p36_compare_prefill_layers(p36_engine *p36_e,
                               p36_engine *other_e,
                               const p36_tokens *prompt,
                               int ctx_size,
                               float *p36_layer_out,
                               float *other_layer_out,
                               float *p36_logits,
                               float *other_logits,
                               char *err,
                               size_t errlen);

#endif
