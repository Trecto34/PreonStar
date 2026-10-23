/* =========================================================================
 * p36.c - Qwen 3.6 bootstrap.
 * =========================================================================
 *
 * This file is the one active p36 engine implementation.
 * It owns:
 * - GGUF loading and model validation
 * - Qwen 3.6 tensor binding
 * - tokenizer and chat rendering
 * - CPU inference/reference helpers
 * - session and runtime scaffolding
 *
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "p36.h"
#include "p36_image.h"
#include "p36_gpu.h"
#include "p36_iq_tables.h"
#include "p36_mmq_contract.h"
#include "p36_quant.h"
#include "p36_ssd.h"

#define P36_GGUF_MAGIC 0x46554747u
#define P36_MAX_DIMS 8
#define P36_NEG_INF (-1.0e30f)
#define P36_RMS_EPS (1.0e-6f)
/* DS4 enables Think Max at 3/8 of its full context window. */
#define P36_THINK_MAX_MIN_CONTEXT 98304u
#define P36_QK8_0 32u
#define P36_Q8_0_BYTES 34u
#define P36_QK4_0 32u
#define P36_Q4_0_BYTES 18u
#define P36_QK2_0 64u
#define P36_Q2_0_BYTES 18u
/* PQ2_0 (Prism type 142) is the same 2-bit codec as Q2_0 with one f16
 * scale per 128 weights instead of per 64: 34 B/128 = 2.125 bpw against
 * Q2_0's 36 B/128 = 2.25.  Same field order, so the dequant differs only
 * in the group size. */
#define P36_QK_PQ2_0 128u
#define P36_PQ2_0_BYTES 34u

/* PTQ1_0 (Prism type 143) is ternary (trits), 128 weights per block,
 * 28 bytes/block (qs[24], qh[2], d). Scale is LAST, not first like PQ2_0. */
#define P36_QK_PTQ1_0 128u
#define P36_PTQ1_0_BYTES 28u

#define P36_QK_K 256u
#define P36_K_SCALE_SIZE 12u
#define P36_Q8_K_BYTES 292u
#define P36_VK_Q8_K_BYTES 296u
#define P36_MAX_Q8_K_BYTES 19856u

#if defined(__GNUC__) || defined(__clang__)
#define P36_MAYBE_UNUSED __attribute__((unused))
#else
#define P36_MAYBE_UNUSED
#endif

static float p36_f16_to_f32(uint16_t h);
static uint16_t p36_f32_to_f16(float f);

static const char P36_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

enum {
    P36_MAX_LAYER = 64,
    P36_MTP_MAX_DRAFT = 16,
    P36_CPU_MAX_THREADS = 32,
    P36_CPU_PREFILL_CHUNK_DEFAULT = 16,
};

typedef enum {
    P36_VARIANT_35B_A3B,
    P36_VARIANT_27B,
} p36_variant;

typedef struct {
    const char *name;
    const char *arch;
    p36_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_tensor;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    uint32_t n_ff_shared;
    uint32_t n_ssm_conv;
    uint32_t n_ssm_state;
    uint32_t n_ssm_group;
    uint32_t n_ssm_dt_rank;
    uint32_t n_ssm_inner;
    uint32_t full_attention_interval;
    bool dense;
} p36_shape;

static const p36_shape P36_SHAPE_35B_A3B = {
    .name = "Qwen 3.6 35B A3B",
    .arch = "qwen35moe",
    .variant = P36_VARIANT_35B_A3B,
    .n_layer = 40,
    .n_embd = 2048,
    .n_vocab = 248320,
    .n_tensor = 733,
    .n_head = 16,
    .n_head_kv = 2,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_ff_exp = 512,
    .n_ff_shared = 512,
    .n_ssm_conv = 4,
    .n_ssm_state = 128,
    .n_ssm_group = 16,
    .n_ssm_dt_rank = 32,
    .n_ssm_inner = 4096,
    .full_attention_interval = 4,
};

static const p36_shape P36_SHAPE_27B = {
    .name = "Qwen 27B dense",
    .arch = "qwen35",
    .variant = P36_VARIANT_27B,
    .n_layer = 64,
    .n_embd = 5120,
    .n_vocab = 248320,
    .n_tensor = 851,
    .n_head = 24,
    .n_head_kv = 4,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_expert = 1,
    .n_expert_used = 1,
    .n_ff_exp = 17408,
    .n_ff_shared = 17408,
    .n_ssm_conv = 4,
    .n_ssm_state = 128,
    .n_ssm_group = 16,
    .n_ssm_dt_rank = 48,
    .n_ssm_inner = 6144,
    .full_attention_interval = 4,
    .dense = true,
};

static p36_shape g_p36_shape = {
    .name = "Qwen 3.6 35B A3B",
    .arch = "qwen35moe",
    .variant = P36_VARIANT_35B_A3B,
    .n_layer = 40,
    .n_embd = 2048,
    .n_vocab = 248320,
    .n_tensor = 733,
    .n_head = 16,
    .n_head_kv = 2,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_ff_exp = 512,
    .n_ff_shared = 512,
    .n_ssm_conv = 4,
    .n_ssm_state = 128,
    .n_ssm_group = 16,
    .n_ssm_dt_rank = 32,
    .n_ssm_inner = 4096,
    .full_attention_interval = 4,
};

#define P36_N_LAYER                 (g_p36_shape.n_layer)
#define P36_N_EMBD                  (g_p36_shape.n_embd)
#define P36_N_VOCAB                 (g_p36_shape.n_vocab)
#define P36_TENSOR_COUNT            (g_p36_shape.n_tensor)
#define P36_CONTEXT_TRAIN           P36_CONTEXT_MAX
#define P36_N_HEAD                  (g_p36_shape.n_head)
#define P36_N_HEAD_KV               (g_p36_shape.n_head_kv)
#define P36_N_HEAD_DIM              (g_p36_shape.n_head_dim)
#define P36_N_VALUE_DIM             (g_p36_shape.n_value_dim)
#define P36_N_ROT                   (g_p36_shape.n_rot)
#define P36_N_EXPERT                (g_p36_shape.n_expert)
#define P36_N_EXPERT_USED           (g_p36_shape.n_expert_used)
#define P36_N_FF_EXP                (g_p36_shape.n_ff_exp)
#define P36_N_FF_SHARED             (g_p36_shape.n_ff_shared)
#define P36_N_SSM_CONV              (g_p36_shape.n_ssm_conv)
#define P36_N_SSM_STATE             (g_p36_shape.n_ssm_state)
#define P36_N_SSM_GROUP             (g_p36_shape.n_ssm_group)
#define P36_N_SSM_DT_RANK           (g_p36_shape.n_ssm_dt_rank)
#define P36_N_SSM_INNER             (g_p36_shape.n_ssm_inner)
#define P36_FULL_ATTENTION_INTERVAL (g_p36_shape.full_attention_interval)
#define P36_MODEL_DENSE             (g_p36_shape.dense)
#define P36_N_SSM_QK                (P36_N_SSM_GROUP * P36_N_SSM_STATE)
#define P36_N_SSM_CONV_DIM          (P36_N_SSM_QK * 2u + P36_N_SSM_INNER)
#define P36_CPU_SCRATCH_FLOATS      (P36_N_SSM_CONV * P36_N_SSM_CONV_DIM)

#ifndef P36_NO_GPU
#include "p36_streaming_hotlist.inc"
#endif

static volatile int p36_float_norm_accum_mode = 1;

static const uint32_t P36_ROPE_SECTIONS[4] = {11, 11, 10, 0};

typedef struct {
    const char *ptr;
    uint64_t len;
} p36_str;

typedef p36_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} p36_cursor;

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32", 1, 4},
    [1]  = {"f16", 1, 2},
    [8]  = {"q8_0", 32, 34},
    [10] = {"q2_k", 256, 84},
    [11] = {"q3_k", 256, 110},
    [12] = {"q4_k", 256, 144},
    [13] = {"q5_k", 256, 176},
    [14] = {"q6_k", 256, 210},
    [15] = {"q8_k", 256, 292},
    [16] = {"iq2_xxs", 256, 66},
    [17] = {"iq2_xs", 256, 74},
    [18] = {"iq3_xxs", 256, 98},
    [19] = {"iq1_s", 256, 50},
    [20] = {"iq4_nl", 32, 18},
    [21] = {"iq3_s", 256, 110},
    [22] = {"iq2_s", 256, 82},
    [23] = {"iq4_xs", 256, 136},
    [26] = {"i32", 1, 4},
    [29] = {"iq1_m", 256, 56},
    [30] = {"bf16", 1, 2},
    [42] = {"q2_0", 64, 18},
    [142] = {"pq2_0", 128, 34},
    [143] = {"ptq1_0", 128, 28},
};

enum {
    P36_TENSOR_F32 = 0,
    P36_TENSOR_F16 = 1,
    P36_TENSOR_Q4_0 = 2,
    P36_TENSOR_Q8_0 = 8,
    P36_TENSOR_Q2_K = 10,
    P36_TENSOR_Q3_K = 11,
    P36_TENSOR_Q4_K = 12,
    P36_TENSOR_Q5_K = 13,
    P36_TENSOR_Q6_K = 14,
    P36_TENSOR_IQ2_XXS = 16,
    P36_TENSOR_IQ2_XS = 17,
    P36_TENSOR_IQ3_XXS = 18,
    P36_TENSOR_IQ1_S = 19,
    P36_TENSOR_IQ4_NL = 20,
    P36_TENSOR_IQ3_S = 21,
    P36_TENSOR_IQ2_S = 22,
    P36_TENSOR_IQ4_XS = 23,
    P36_TENSOR_I32 = 26,
    P36_TENSOR_IQ1_M = 29,
    P36_TENSOR_BF16 = 30,
    P36_TENSOR_Q2_0 = 42,
    P36_TENSOR_PQ2_0 = 142,
    P36_TENSOR_PTQ1_0 = 143,
};

typedef struct {
    p36_str key;
    uint32_t type;
    uint64_t value_pos;
} p36_kv;

typedef struct {
    p36_str name;
    uint32_t ndim;
    uint64_t dim[P36_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
    bool hadamard_rotate;
    bool hadamard_inverse;
    bool hadamard_grouped;   /* GDN ssm_out: activation needs a tiled->grouped permute */
    bool from_q4_0;          /* file stores Q4_0; loader re-encodes it as Q8_0 in the arena */
} p36_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;          /* file bytes plus the converted-tensor arena, if any */
    uint64_t file_size;     /* bytes actually backed by the GGUF file */
    uint64_t map_len;       /* reserved address range starting at map */
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;
    p36_kv *kv;
    p36_tensor *tensors;
    bool owned;
} p36_model;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} p36_array_ref;

typedef struct {
    uint16_t d;
    int8_t qs[P36_QK8_0];
} p36_block_q8_0;

typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK4_0 / 2u];
} p36_block_q4_0;

typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK2_0 / 4u];
} p36_block_q2_0;

typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK_PQ2_0 / 4u];
} p36_block_pq2_0;

typedef struct {
    uint8_t qs[24];
    uint8_t qh[2];
    uint16_t d;
} p36_block_ptq1_0;

typedef struct {
    float d;
    int8_t qs[P36_QK_K];
    int16_t bsums[P36_QK_K / 16];
} p36_block_q8_k;

typedef struct {
    uint8_t scales[P36_QK_K / 16];
    uint8_t qs[P36_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} p36_block_q2_k;

typedef struct {
    uint8_t hmask[P36_QK_K / 8];
    uint8_t qs[P36_QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
} p36_block_q3_k;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[P36_K_SCALE_SIZE];
    uint8_t qs[P36_QK_K / 2];
} p36_block_q4_k;

typedef struct {
    uint16_t d;
    uint16_t qs[P36_QK_K / 8];
} p36_block_iq2_xxs;

typedef struct {
    uint16_t d;
    uint16_t qs[P36_QK_K / 8];
    uint8_t scales[P36_QK_K / 32];
} p36_block_iq2_xs;

typedef struct {
    uint16_t d;
    uint8_t qs[3 * P36_QK_K / 8];
} p36_block_iq3_xxs;

typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK_K / 4];
    uint8_t qh[P36_QK_K / 32];
    uint8_t scales[P36_QK_K / 32];
} p36_block_iq2_s;

#define P36_IQ3S_N_SCALE (P36_QK_K / 64)
typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK_K / 4];
    uint8_t qh[P36_QK_K / 32];
    uint8_t signs[P36_QK_K / 8];
    uint8_t scales[P36_IQ3S_N_SCALE];
} p36_block_iq3_s;

typedef struct {
    uint16_t d;
    uint8_t qs[P36_QK_K / 8];
    uint16_t qh[P36_QK_K / 32];
} p36_block_iq1_s;

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} p36_block_iq4_nl;

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[P36_QK_K / 64];
    uint8_t qs[P36_QK_K / 2];
} p36_block_iq4_xs;

typedef struct {
    uint8_t qs[P36_QK_K / 8];
    uint8_t qh[P36_QK_K / 16];
    uint8_t scales[P36_QK_K / 32];
} p36_block_iq1_m;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[P36_K_SCALE_SIZE];
    uint8_t qh[P36_QK_K / 8];
    uint8_t qs[P36_QK_K / 2];
} p36_block_q5_k;

typedef struct {
    uint8_t ql[P36_QK_K / 2];
    uint8_t qh[P36_QK_K / 4];
    int8_t scales[P36_QK_K / 16];
    uint16_t d;
} p36_block_q6_k;

typedef struct {
    p36_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
} str_i32_table;

typedef struct {
    p36_str text;
    int id;
} p36_special_token;

typedef struct p36_vocab {
    p36_str *token;
    p36_special_token *special;
    int n_vocab;
    int n_special;
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
    int think_start_id;
    int think_end_id;
    int vision_start_id;
    int vision_end_id;
    int image_pad_id;
    int video_pad_id;
    bool add_bos;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
} p36_vocab;

typedef enum {
    P36_LAYER_RECURRENT,
    P36_LAYER_FULL_ATTN,
} p36_layer_kind;

typedef struct {
    p36_layer_kind kind;
    p36_tensor *attn_norm;
    p36_tensor *post_attention_norm;

    p36_tensor *attn_gate;
    p36_tensor *attn_qkv;
    p36_tensor *ssm_a;
    p36_tensor *ssm_alpha;
    p36_tensor *ssm_beta;
    p36_tensor *ssm_conv1d;
    p36_tensor *ssm_dt;
    p36_tensor *ssm_norm;
    p36_tensor *ssm_out;
    p36_tensor *attn_gate_scale;
    p36_tensor *attn_qkv_scale;
    p36_tensor *ssm_alpha_scale;
    p36_tensor *ssm_beta_scale;
    p36_tensor *ssm_out_scale;

    p36_tensor *attn_q;
    p36_tensor *attn_q_norm;
    p36_tensor *attn_k;
    p36_tensor *attn_k_norm;
    p36_tensor *attn_v;
    p36_tensor *attn_output;
    p36_tensor *attn_sinks;
    p36_tensor *attn_q_scale;
    p36_tensor *attn_k_scale;
    p36_tensor *attn_v_scale;
    p36_tensor *attn_output_scale;

    p36_tensor *ffn_gate_inp;
    p36_tensor *ffn_gate_inp_shexp;
    p36_tensor *ffn_gate_exps;
    p36_tensor *ffn_gate_shexp;
    p36_tensor *ffn_up_exps;
    p36_tensor *ffn_up_shexp;
    p36_tensor *ffn_down_exps;
    p36_tensor *ffn_down_shexp;
    p36_tensor *ffn_gate_exps_scale;
    p36_tensor *ffn_gate_shexp_scale;
    p36_tensor *ffn_up_exps_scale;
    p36_tensor *ffn_up_shexp_scale;
    p36_tensor *ffn_down_exps_scale;
    p36_tensor *ffn_down_shexp_scale;
} p36_layer_weights;

typedef struct {
    p36_tensor *token_embd;
    p36_tensor *output_norm;
    p36_tensor *output;
    p36_tensor *output_scale;
    p36_layer_weights layer[P36_MAX_LAYER];
} p36_weights;

typedef struct {
    p36_tensor *token_embd;
    p36_tensor *output_norm;
    p36_tensor *output;
    p36_tensor *output_scale;
    p36_tensor *eh_proj;
    p36_tensor *enorm;
    p36_tensor *hnorm;
    p36_tensor *shared_head_norm;
    p36_layer_weights block;
} p36_mtp_weights;

struct p36_engine {
    p36_model model;
    p36_model mtp_model;
    p36_model vision_model;
    p36_vocab vocab;
    p36_weights weights;
    p36_mtp_weights mtp_weights;
    p36_backend backend;
    uint32_t n_threads;
    uint32_t cpu_prefill_cap;
    uint32_t prefill_cap_override;
    int context_size;
    uint32_t planned_sessions, live_sessions;
    uint64_t session_bytes, context_hint_bytes;
    uint64_t streaming_model_limit;
    int power_percent;
    int mtp_draft_tokens;
    float mtp_margin;
    float mtp_expert_weights_scale;
    p36_kv_cache_type cache_type_k;
    p36_kv_cache_type cache_type_v;
    char *directional_steering_file;
    float *directional_steering_dirs;
#ifndef P36_NO_GPU
    p36_gpu_tensor *directional_steering_gpu;
    p36_gpu_tensor *session_batch_logits;
    uint32_t session_batch_logits_cap;
#endif
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    int routed_quant_bits;
    float expert_weights_scale;
    uint32_t ssd_streaming_cache_experts;
    uint32_t ssd_streaming_full_layers;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    p36_ssd_memory_lock simulated_memory;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool mtp_ready;
    bool vision_ready;
    bool kat_coder;
    char *chat_template;
    bool cpu_mmq_contract;   /* CPU reference emulates the f16 dense MMQ contract */
    p36_variant variant;
};

struct p36_session {
    p36_engine *engine;
    int ctx_size;
    p36_tokens checkpoint;
    bool checkpoint_valid;
    bool vision_state;
    p36_vision_span *vision_spans;
    size_t vision_count;
    int32_t rope_delta;
    uint64_t reserved_context_bytes;
    float *logits;
    float *sample_probs;
    float *mtp_logits;
    bool logits_host_valid;
    bool gpu_top2_valid;
    int gpu_top2[2];
    void *runtime;
    p36_session_progress_fn progress;
    void *progress_ud;
    p36_session_progress_fn display_progress;
    void *display_progress_ud;
    p36_session_cancel_fn cancel;
    void *cancel_ud;
    int mtp_draft_token;
    bool mtp_draft_valid;
    float mtp_draft_margin;
    int mtp_backoff;
    int mtp_backoff_len;
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tokens;
} p36_payload_header;

typedef struct {
    uint32_t len;
    uint32_t cap;
    p36_kv_cache_type type_k;
    p36_kv_cache_type type_v;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    uint8_t *k;
    uint8_t *v;
} p36_full_attn_cache;

typedef struct {
    float *conv;
    float *state;
} p36_recurrent_cache;

typedef struct {
    p36_full_attn_cache full[P36_MAX_LAYER];
    p36_recurrent_cache recurrent[P36_MAX_LAYER];
    uint32_t prefill_cap;
    float *hidden;
    float *next_hidden;
    float *work0;
    float *work1;
    float *work2;
    float *work3;
    float *work4;
    float *work5;
    float *scores;
    float *batch_hidden;
    float *batch_next_hidden;
    float *batch_norm;
    float *batch_qg;
    float *batch_q;
    float *batch_k;
    float *batch_v;
    float *batch_attn_out;
    float *batch_recur_qkv;
    float *batch_recur_z;
    float *batch_recur_alpha;
    float *batch_recur_beta;
    float *batch_recur_proj;
    float *batch_ffn_gate_logits;
    float *batch_ffn_shared_gate;
    float *batch_ffn_shared_up;
    float *batch_ffn_shared_mid;
    float *batch_ffn_shared_out;
    float *batch_ffn_scalar;
    uint8_t *batch_xq;
    float *batch_had;        /* forward-Hadamard activation scratch, widest projection */
} p36_cpu_runtime;

typedef void (*p36_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

typedef struct {
    pthread_mutex_t mu;
    pthread_mutex_t submit_mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    pthread_t threads[P36_CPU_MAX_THREADS];
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t active_threads;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    p36_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} p36_cpu_pool;

static p36_cpu_pool p36_cpu_workers;
static pthread_mutex_t p36_cpu_workers_init_mu = PTHREAD_MUTEX_INITIALIZER;
static bool p36_cpu_workers_atexit;
static __thread int p36_cpu_parallel_depth;

#ifndef P36_NO_GPU
/* Prefill batch width: the Vulkan scratch tensors hold this many token rows,
 * so p36_session_sync() can walk a prompt in chunks.  1024 keeps the routed
 * MoE GEMM tiles ~full (8192 slots, 32 per expert on average); the 8192-slot
 * schedule cap and the attention-partials scratch (which scales with
 * chunk x context spans) are why this does not go higher.  Chunk width was
 * never numerics-invariant once the prefill GEMM kernels landed; Vulkan SSD
 * streaming keeps its own 512 default. */
enum { P36_GPU_PREFILL_CHUNK_DEFAULT = 1024 };

typedef struct {
    uint32_t cap;
    p36_kv_cache_type type_k;
    p36_kv_cache_type type_v;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    p36_gpu_tensor *k;
    p36_gpu_tensor *v;
} p36_vulkan_full_attn_cache;

typedef struct {
    p36_gpu_tensor *conv;
    p36_gpu_tensor *state;
} p36_vulkan_recurrent_cache;

typedef struct {
    uint32_t prefill_cap;
    bool recur_conv_fused;
    bool recur_state_f16;
    p36_vulkan_full_attn_cache full[P36_MAX_LAYER];
    p36_vulkan_recurrent_cache recurrent[P36_MAX_LAYER];
    p36_gpu_tensor *hidden;
    p36_gpu_tensor *next_hidden;
    /* Host-written embed staging, ping-ponged across prefill chunks: the
     * CPU fills the next chunk's rows while the GPU still runs the current
     * one, instead of mapping the GPU-written hidden tensor (which drains
     * the whole in-flight batch).  Two buffers suffice: the submit ring
     * bounds record-ahead to 4 batches, far less than one chunk. */
    p36_gpu_tensor *embed_stage[2];
    uint32_t embed_flip;
    p36_gpu_tensor *norm;
    p36_gpu_tensor *last_h;
    p36_gpu_tensor *inp_q8;
    bool inp_q8_valid;
    p36_gpu_tensor *had_q8;     /* q8_K of the transformed activation */
    p36_gpu_tensor *had_signs;  /* concatenated per-width sign vectors */
    p36_gpu_tensor *attn_qg;
    p36_gpu_tensor *attn_q;
    p36_gpu_tensor *attn_k;
    p36_gpu_tensor *attn_v;
    p36_gpu_tensor *attn_out;
    p36_gpu_tensor *recur_qkv;
    p36_gpu_tensor *recur_window;
    p36_gpu_tensor *recur_conv;
    p36_gpu_tensor *recur_z;
    p36_gpu_tensor *recur_alpha;
    p36_gpu_tensor *recur_beta;
    p36_gpu_tensor *recur_gb;
    p36_gpu_tensor *recur_q;
    p36_gpu_tensor *recur_k;
    p36_gpu_tensor *recur_v;
    p36_gpu_tensor *recur_proj;
    p36_gpu_tensor *ffn_gate_logits;
    p36_gpu_tensor *ffn_selected;
    p36_gpu_tensor *ffn_weights;
    p36_gpu_tensor *ffn_shared_gate;
    p36_gpu_tensor *ffn_shared_up;
    p36_gpu_tensor *ffn_shared_mid;
    p36_gpu_tensor *ffn_shared_out;
    p36_gpu_tensor *ffn_scalar;
    p36_gpu_tensor *logits;
    p36_gpu_tensor *top2;
    p36_gpu_tensor *topk8;
    p36_gpu_tensor *minp_pack;
    p36_gpu_tensor *scores;
    p36_vulkan_full_attn_cache mtp_full;
    p36_vulkan_recurrent_cache spec_recurrent[P36_MAX_LAYER];
    p36_gpu_tensor *spec_logits;
    p36_gpu_tensor *mtp_tok_embd;
    p36_gpu_tensor *mtp_e_norm;
    p36_gpu_tensor *mtp_h_norm;
    p36_gpu_tensor *mtp_concat;
    p36_gpu_tensor *mtp_cur;
    p36_gpu_tensor *mtp_next;
    p36_gpu_tensor *mtp_head;
    p36_gpu_tensor *mtp_logits;
    bool mtp_enabled;
} p36_vulkan_runtime;
#endif

enum {
    P36_PAYLOAD_MAGIC = 0x51563336u,
    P36_PAYLOAD_VERSION_TOKEN_ONLY = 1,
    P36_PAYLOAD_VERSION = 2,
    P36_PAYLOAD_VERSION_TYPED_KV = 3,
    P36_PAYLOAD_U32_FIELDS = 14,
    P36_PAYLOAD_U32_FIELDS_TYPED_KV = 16,
};

static void p36_session_reset_runtime(p36_session *s);
static int sample_argmax(const float *logits, uint32_t n_vocab);
#ifndef P36_NO_GPU
static int sample_argmax_margin(const float *logits, uint32_t n_vocab, float *margin);

static uint64_t p36_mtp_stat_calls;
static uint64_t p36_mtp_stat_drafted;
static uint64_t p36_mtp_stat_accepted;
static uint64_t p36_mtp_stat_full;
static bool p36_mtp_stat_registered;

static void p36_mtp_stats_report(void) {
    if (!p36_mtp_stat_calls) return;
    fprintf(stderr,
            "p36: MTP stats calls=%" PRIu64 " drafted=%" PRIu64 " accepted=%" PRIu64 " full=%" PRIu64 " accept=%.1f%%\n",
            p36_mtp_stat_calls,
            p36_mtp_stat_drafted,
            p36_mtp_stat_accepted,
            p36_mtp_stat_full,
            p36_mtp_stat_drafted ? 100.0 * (double)p36_mtp_stat_accepted / (double)p36_mtp_stat_drafted : 0.0);
}

static void p36_mtp_stats_add(int drafted, int accepted, bool full) {
    if (!getenv("P36_MTP_STATS")) return;
    if (!p36_mtp_stat_registered) {
        atexit(p36_mtp_stats_report);
        p36_mtp_stat_registered = true;
    }
    p36_mtp_stat_calls++;
    if (drafted > 0) p36_mtp_stat_drafted += (uint64_t)drafted;
    if (accepted > 0) p36_mtp_stat_accepted += (uint64_t)accepted;
    if (full) p36_mtp_stat_full++;
}
#endif

static uint32_t p36_online_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 1;
    if ((unsigned long)n > P36_CPU_MAX_THREADS) return P36_CPU_MAX_THREADS;
    return (uint32_t)n;
}

static uint32_t p36_parse_env_u32(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && v > 0) {
            if ((unsigned long)v > UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return fallback;
}

static uint32_t p36_resolve_thread_count(int requested) {
    uint32_t n = p36_online_cpus();
    if (requested > 0) n = (uint32_t)requested;
    else n = p36_parse_env_u32("P36_THREADS", n);
    if (n < 1) n = 1;
    if (n > P36_CPU_MAX_THREADS) n = P36_CPU_MAX_THREADS;
    return n;
}

static uint32_t p36_default_cpu_prefill_cap(void) {
    uint32_t cap = p36_parse_env_u32("P36_CPU_PREFILL_CHUNK", P36_CPU_PREFILL_CHUNK_DEFAULT);
    if (cap < 1) cap = 1;
    return cap;
}

#ifndef P36_NO_GPU
static uint32_t p36_default_gpu_prefill_cap(void) {
    /* Keep the established environment name for compatibility; this graph
     * default is shared by Vulkan and Metal. */
    uint32_t cap = p36_parse_env_u32(
        "P36_VK_PREFILL_CHUNK", P36_GPU_PREFILL_CHUNK_DEFAULT);
    if (cap < 1) cap = 1;
    return cap;
}

/* SSD streaming prefill is IO-bound: each chunk re-streams close to the
 * whole routed expert set per layer, so expert bytes per token scale as
 * 1/chunk.  Measured on UD-Q4_K_M (ctx512, thermal-gated A/B): chunk 512
 * doubles prefill throughput over the resident-path default of 64.  An
 * explicit --prefill-chunk or P36_VK_PREFILL_CHUNK still wins. */
/* GFX1013 runs every dispatch on the gfx ring under amdgpu's 10 s TDR, and
 * a timeout there destroys the context outright.  Two independent costs
 * scale with the prefill chunk and both are unbounded without a cap:
 *
 *   - attention work per dispatch.  Shapes that miss the specialised
 *     attn_prefill_qtile2 shader (it requires head_dim 256 and GQA exactly
 *     8) fall back to attn_prefill_qtile, which has been measured to cross
 *     the watchdog at ctx >= 12288 with a 1024-token chunk.
 *   - scratch memory, which grows ~206 KB per chunk token: a 4096 chunk
 *     costs +634 MB, a 6x violation of the SPEEDUP.md 100 MB budget.
 *
 * A 4096 chunk was also measured at 202 t/s prefill against 616 t/s at
 * 1024, so the absolute cap costs nothing anyone wants.  The GQA test is a
 * secondary guard: the 35B hybrid is GQA 8 and still crashed at 4096, so
 * the absolute cap is what actually prevents the observed failure.
 * P36_VK_PREFILL_CHUNK_UNSAFE=1 restores the unclamped value. */
static uint32_t p36_engine_clamp_prefill_cap(uint32_t cap) {
    static int warned;
    if (!p36_gpu_device_is_gfx1013()) return cap;
    if (getenv("P36_VK_PREFILL_CHUNK_UNSAFE")) return cap;
    uint32_t limit = (P36_N_HEAD_KV && (P36_N_HEAD / P36_N_HEAD_KV) == 8u) ? 1024u : 256u;
    if (cap <= limit) return cap;
    if (!warned) {
        warned = 1;
        fprintf(stderr,
                "p36: clamping prefill chunk %u -> %u on GFX1013 (GQA %u); a larger chunk "
                "can exceed the 10 s amdgpu watchdog and destroy the GPU context. "
                "Set P36_VK_PREFILL_CHUNK_UNSAFE=1 to override.\n",
                cap, limit, P36_N_HEAD_KV ? P36_N_HEAD / P36_N_HEAD_KV : 0u);
    }
    return limit;
}

static uint32_t p36_engine_gpu_prefill_cap(const p36_engine *e) {
#ifdef P36_METAL
    if (e->ssd_streaming) {
        uint32_t cap = e->prefill_cap_override ?
            e->prefill_cap_override : p36_default_gpu_prefill_cap();
        uint32_t cache_tokens =
            e->ssd_streaming_cache_experts / P36_N_EXPERT_USED;
        if (cache_tokens < 1) cache_tokens = 1;
        if (cap > cache_tokens) cap = cache_tokens;
        return cap;
    }
#endif
    uint32_t cap;
    if (e->prefill_cap_override) cap = e->prefill_cap_override;
    else if (e->ssd_streaming && !getenv("P36_VK_PREFILL_CHUNK")) cap = 512;
    else cap = p36_default_gpu_prefill_cap();
    return p36_engine_clamp_prefill_cap(cap);
}
#endif

static uint32_t p36_q8k_row_bytes(uint32_t n) {
    return ((n + P36_QK_K - 1u) / P36_QK_K) * P36_Q8_K_BYTES;
}

static uint32_t p36_q8_0_row_bytes(uint32_t n) {
    return ((n + P36_QK8_0 - 1u) / P36_QK8_0) * P36_Q8_0_BYTES;
}

static uint32_t p36_q4_0_row_bytes(uint32_t n) {
    return ((n + P36_QK4_0 - 1u) / P36_QK4_0) * P36_Q4_0_BYTES;
}

/* Two opt-in GPU fast paths measure faster but change numerics, so they stay
 * off unless asked for.  Every front end exposes the same two flags and both
 * resolve to the environment variables the Vulkan backend already reads, so
 * there is exactly one implementation of the behaviour.
 *
 *   --f32-fast-wide   256-thread matmul_f32_fast.  Measured +6.1% decode and
 *                     +17% prefill on a short benchmark, but it reassociates
 *                     the MoE router gate, which can change which experts a
 *                     token selects.  That is not a theoretical risk: on one
 *                     BC-250, long-context chat with tool calling degenerated
 *                     into unterminated generation with this flag on.
 *   --attn-span N     split-K span in keys (default 512).  128 measured +2.2%
 *                     decode at ctx 2048; it only regroups an ordered sum, but
 *                     the span count is context/span, so both the rounding and
 *                     attn_combine's own cost grow with context while the
 *                     saving does not.  Not validated for long context.
 *
 * Neither is bit-exact, so neither is on by default, and neither is passed by
 * the shipped launcher scripts. */
void p36_set_gpu_fast_path_env(const char *name, const char *value) {
    if (!name || !value) return;
    setenv(name, value, 1);
}

const char *p36_kv_cache_type_name(p36_kv_cache_type type) {
    switch (type) {
    case P36_KV_CACHE_F16: return "f16";
    case P36_KV_CACHE_Q8_0: return "q8_0";
    case P36_KV_CACHE_Q4_0: return "q4_0";
    default: return "unknown";
    }
}

p36_kv_cache_type p36_default_kv_cache_type_k(p36_backend backend, bool ssd_streaming) {
    return (backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL) &&
           !ssd_streaming ?
           P36_KV_CACHE_Q8_0 : P36_KV_CACHE_F16;
}

p36_kv_cache_type p36_default_kv_cache_type_v(p36_backend backend, bool ssd_streaming) {
    return (backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL) &&
           !ssd_streaming ?
           P36_KV_CACHE_Q4_0 : P36_KV_CACHE_F16;
}

bool p36_parse_kv_cache_type(const char *s, p36_kv_cache_type *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "f16")) {
        *out = P36_KV_CACHE_F16;
        return true;
    }
    if (!strcmp(s, "q8_0") || !strcmp(s, "q8_o")) {
        *out = P36_KV_CACHE_Q8_0;
        return true;
    }
    if (!strcmp(s, "q4_0") || !strcmp(s, "q4_o")) {
        *out = P36_KV_CACHE_Q4_0;
        return true;
    }
    return false;
}

static uint32_t p36_kv_cache_row_bytes(p36_kv_cache_type type, uint32_t n) {
    switch (type) {
    case P36_KV_CACHE_F16: return n * (uint32_t)sizeof(uint16_t);
    case P36_KV_CACHE_Q8_0: return p36_q8_0_row_bytes(n);
    case P36_KV_CACHE_Q4_0: return p36_q4_0_row_bytes(n);
    default: return 0;
    }
}

static float p36_kv_cache_at(const uint8_t *row, p36_kv_cache_type type, uint32_t i) {
    if (type == P36_KV_CACHE_F16) {
        const uint16_t *h = (const uint16_t *)row;
        return p36_f16_to_f32(h[i]);
    }
    if (type == P36_KV_CACHE_Q8_0) {
        const p36_block_q8_0 *b = (const p36_block_q8_0 *)row;
        const p36_block_q8_0 *x = b + i / P36_QK8_0;
        return p36_f16_to_f32(x->d) * (float)x->qs[i % P36_QK8_0];
    }
    if (type == P36_KV_CACHE_Q4_0) {
        const p36_block_q4_0 *b = (const p36_block_q4_0 *)row;
        const p36_block_q4_0 *x = b + i / P36_QK4_0;
        uint32_t j = i % P36_QK4_0;
        uint8_t q = x->qs[j % (P36_QK4_0 / 2u)];
        q = j < P36_QK4_0 / 2u ? q & 0x0fu : q >> 4;
        return p36_f16_to_f32(x->d) * (float)((int)q - 8);
    }
    return 0.0f;
}

static void p36_kv_cache_store_row(uint8_t *dst, p36_kv_cache_type type, const float *src, uint32_t n) {
    if (type == P36_KV_CACHE_F16) {
        uint16_t *h = (uint16_t *)dst;
        for (uint32_t i = 0; i < n; i++) h[i] = p36_f32_to_f16(src[i]);
    } else if (type == P36_KV_CACHE_Q8_0) {
        p36_quant_q8_0(src, dst, (int64_t)n);
    } else if (type == P36_KV_CACHE_Q4_0) {
        p36_quant_q4_0_kv(src, dst, (int64_t)n);
    }
}

static uint64_t p36_cpu_prefill_scratch_bytes(uint32_t cap) {
    uint64_t total = 0;
    total += (uint64_t)P36_N_EMBD * 2u * sizeof(float);
    total += (uint64_t)P36_CPU_SCRATCH_FLOATS * 6u * sizeof(float);
    total += (uint64_t)cap * P36_N_EMBD * 3u * sizeof(float);
    total += (uint64_t)cap * (P36_N_HEAD * P36_N_HEAD_DIM * 2u) * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_INNER * 3u * sizeof(float);
    total += (uint64_t)cap * P36_N_HEAD_KV * (P36_N_HEAD_DIM + P36_N_VALUE_DIM) * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_CONV_DIM * sizeof(float);
    total += (uint64_t)cap * P36_N_SSM_DT_RANK * 2u * sizeof(float);
    total += (uint64_t)cap * P36_N_EXPERT * sizeof(float);
    total += (uint64_t)cap * P36_N_FF_SHARED * 4u * sizeof(float);
    total += (uint64_t)cap * sizeof(float);
    total += (uint64_t)cap * P36_MAX_Q8_K_BYTES;
    return total;
}

static void p36_cpu_threads_shutdown(void) {
    pthread_mutex_lock(&p36_cpu_workers_init_mu);
    if (!p36_cpu_workers.initialized) {
        pthread_mutex_unlock(&p36_cpu_workers_init_mu);
        return;
    }
    pthread_mutex_lock(&p36_cpu_workers.submit_mu);
    pthread_mutex_lock(&p36_cpu_workers.mu);
    p36_cpu_workers.shutdown = true;
    p36_cpu_workers.generation++;
    pthread_cond_broadcast(&p36_cpu_workers.work_cv);
    pthread_mutex_unlock(&p36_cpu_workers.mu);
    for (uint32_t i = 1; i < p36_cpu_workers.n_threads; i++) {
        pthread_join(p36_cpu_workers.threads[i], NULL);
    }
    pthread_mutex_unlock(&p36_cpu_workers.submit_mu);
    pthread_cond_destroy(&p36_cpu_workers.done_cv);
    pthread_cond_destroy(&p36_cpu_workers.work_cv);
    pthread_mutex_destroy(&p36_cpu_workers.submit_mu);
    pthread_mutex_destroy(&p36_cpu_workers.mu);
    memset(&p36_cpu_workers, 0, sizeof(p36_cpu_workers));
    pthread_mutex_unlock(&p36_cpu_workers_init_mu);
}

static void *p36_cpu_worker_main(void *arg) {
    uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&p36_cpu_workers.mu);
        while (seen_generation == p36_cpu_workers.generation && !p36_cpu_workers.shutdown) {
            pthread_cond_wait(&p36_cpu_workers.work_cv, &p36_cpu_workers.mu);
        }
        if (p36_cpu_workers.shutdown) {
            pthread_mutex_unlock(&p36_cpu_workers.mu);
            return NULL;
        }
        seen_generation = p36_cpu_workers.generation;
        p36_parallel_fn fn = p36_cpu_workers.fn;
        void *ctx = p36_cpu_workers.ctx;
        uint64_t n_rows = p36_cpu_workers.n_rows;
        uint32_t n_threads = p36_cpu_workers.active_threads;
        pthread_mutex_unlock(&p36_cpu_workers.mu);

        if (fn && n_threads > 0) {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                p36_cpu_parallel_depth++;
                fn(ctx, row0, row1);
                p36_cpu_parallel_depth--;
            }
        }

        pthread_mutex_lock(&p36_cpu_workers.mu);
        p36_cpu_workers.done++;
        if (p36_cpu_workers.done == p36_cpu_workers.n_workers) {
            pthread_cond_signal(&p36_cpu_workers.done_cv);
        }
        pthread_mutex_unlock(&p36_cpu_workers.mu);
    }
}

static void p36_cpu_threads_reserve(uint32_t n_threads) {
    pthread_mutex_lock(&p36_cpu_workers_init_mu);
    if (!p36_cpu_workers.initialized) {
        pthread_mutex_init(&p36_cpu_workers.mu, NULL);
        pthread_mutex_init(&p36_cpu_workers.submit_mu, NULL);
        pthread_cond_init(&p36_cpu_workers.work_cv, NULL);
        pthread_cond_init(&p36_cpu_workers.done_cv, NULL);
        p36_cpu_workers.n_threads = 1;
        p36_cpu_workers.n_workers = 0;
        p36_cpu_workers.initialized = true;
        if (!p36_cpu_workers_atexit) {
            atexit(p36_cpu_threads_shutdown);
            p36_cpu_workers_atexit = true;
        }
    }
    if (n_threads > P36_CPU_MAX_THREADS) n_threads = P36_CPU_MAX_THREADS;
    while (p36_cpu_workers.n_threads < n_threads) {
        uint32_t tid = p36_cpu_workers.n_threads;
        if (pthread_create(&p36_cpu_workers.threads[tid], NULL, p36_cpu_worker_main,
                           (void *)(uintptr_t)tid) != 0) {
            break;
        }
        p36_cpu_workers.n_threads++;
        p36_cpu_workers.n_workers = p36_cpu_workers.n_threads - 1u;
    }
    pthread_mutex_unlock(&p36_cpu_workers_init_mu);
}

static void p36_parallel_for_rows(uint64_t n_rows,
                                  uint64_t min_parallel_rows,
                                  uint32_t requested_threads,
                                  p36_parallel_fn fn,
                                  void *ctx) {
    uint32_t n_threads;
    uint64_t rows_per_thread;
    uint64_t main_row1;
    if (!fn || n_rows == 0) return;
    if (requested_threads <= 1 || n_rows < min_parallel_rows || p36_cpu_parallel_depth > 0) {
        fn(ctx, 0, n_rows);
        return;
    }
    p36_cpu_threads_reserve(requested_threads);
    n_threads = p36_cpu_workers.n_threads;
    if (n_threads > requested_threads) n_threads = requested_threads;
    if (n_threads <= 1) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&p36_cpu_workers.submit_mu);
    pthread_mutex_lock(&p36_cpu_workers.mu);
    p36_cpu_workers.fn = fn;
    p36_cpu_workers.ctx = ctx;
    p36_cpu_workers.n_rows = n_rows;
    p36_cpu_workers.active_threads = n_threads;
    p36_cpu_workers.done = 0;
    p36_cpu_workers.generation++;
    pthread_cond_broadcast(&p36_cpu_workers.work_cv);
    rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&p36_cpu_workers.mu);

    p36_cpu_parallel_depth++;
    fn(ctx, 0, main_row1);
    p36_cpu_parallel_depth--;

    pthread_mutex_lock(&p36_cpu_workers.mu);
    while (p36_cpu_workers.done < p36_cpu_workers.n_workers) {
        pthread_cond_wait(&p36_cpu_workers.done_cv, &p36_cpu_workers.mu);
    }
    pthread_mutex_unlock(&p36_cpu_workers.mu);
    pthread_mutex_unlock(&p36_cpu_workers.submit_mu);
}

static bool p36_backend_uses_graph(p36_backend backend) {
    return backend == P36_BACKEND_VULKAN || backend == P36_BACKEND_METAL;
}

static bool p36_backend_supports_ssd_streaming(p36_backend backend) {
    return p36_backend_uses_graph(backend);
}


static bool p36_engine_uses_vulkan_runtime(const p36_engine *e) {
    return e && p36_backend_uses_graph(e->backend);
}

static void p36_die(const char *msg) {
    fprintf(stderr, "p36: %s\n", msg);
    exit(1);
}

static void p36_die_errno(const char *what, const char *path) {
    fprintf(stderr, "p36: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) p36_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    return xcalloc(n, size);
}

static char *p36_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = xmalloc(n);
    memcpy(out, s, n);
    return out;
}

static double p36_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static bool p36_str_eq(p36_str a, p36_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static bool p36_streq(p36_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool p36_str_contains(p36_str s, const char *needle) {
    size_t n = strlen(needle);
    if (!n || n > s.len) return false;
    for (uint64_t i = 0; i <= s.len - n; i++) {
        if (!memcmp(s.ptr + i, needle, n)) return true;
    }
    return false;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static const char *p36_log_color_code(p36_log_type type) {
    switch (type) {
    case P36_LOG_PREFILL:
    case P36_LOG_TIMING:
        return "\x1b[36m";
    case P36_LOG_GENERATION:
    case P36_LOG_OK:
        return "\x1b[32m";
    case P36_LOG_KVCACHE:
        return "\x1b[33m";
    case P36_LOG_TOOL:
        return "\x1b[90m";
    case P36_LOG_WARNING:
        return "\x1b[38;5;208m";
    case P36_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool p36_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

void p36_log(FILE *fp, p36_log_type type, const char *fmt, ...) {
    va_list ap;
    const bool colorize = type != P36_LOG_DEFAULT && p36_log_is_tty(fp);
    va_start(ap, fmt);
    if (colorize) fputs(p36_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
    va_end(ap);
}

const char *p36_backend_name(p36_backend backend) {
    switch (backend) {
    case P36_BACKEND_VULKAN: return "vulkan";
    case P36_BACKEND_METAL: return "metal";
    case P36_BACKEND_CPU: return "cpu";
    default: return "unknown";
    }
}

bool p36_think_mode_enabled(p36_think_mode mode) {
    return mode != P36_THINK_NONE;
}

const char *p36_think_mode_name(p36_think_mode mode) {
    switch (mode) {
    case P36_THINK_NONE: return "none";
    case P36_THINK_HIGH: return "high";
    case P36_THINK_MAX: return "max";
    case P36_THINK_LOW: return "low";
    case P36_THINK_MEDIUM: return "medium";
    case P36_THINK_XHIGH: return "xhigh";
    default: return "unknown";
    }
}

const char *p36_qwen38_effort_instruction(p36_think_mode mode) {
    if (mode == P36_THINK_LOW)
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    if (mode == P36_THINK_MEDIUM || mode == P36_THINK_NONE) return NULL;
    return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
}

const char *p36_jinja_effort_name(p36_think_mode mode) {
    switch (mode) {
    case P36_THINK_LOW: return "low";
    case P36_THINK_MEDIUM: return "medium";
    case P36_THINK_HIGH: return "high";
    case P36_THINK_XHIGH:
    case P36_THINK_MAX: return "xhigh";
    case P36_THINK_NONE:
    default: return NULL;
    }
}

/* --- Runtime Jinja2 Chat Template Evaluator --- */
typedef struct jinja_arena_block {
    struct jinja_arena_block *next;
    size_t used;
    size_t cap;
    uint8_t data[];
} jinja_arena_block;

typedef struct {
    jinja_arena_block *head;
} jinja_arena;

static void *ja_alloc(jinja_arena *a, size_t size) {
    size = (size + 7u) & ~7u;
    if (!a->head || a->head->used + size > a->head->cap) {
        size_t cap = size > 65536u ? size : 65536u;
        jinja_arena_block *b = (jinja_arena_block *)calloc(1, sizeof(*b) + cap);
        if (!b) abort();
        b->cap = cap;
        b->next = a->head;
        a->head = b;
    }
    void *p = a->head->data + a->head->used;
    a->head->used += size;
    return p;
}

static char *ja_strndup(jinja_arena *a, const char *s, size_t n) {
    char *p = (char *)ja_alloc(a, n + 1);
    if (n && s) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *ja_strdup(jinja_arena *a, const char *s) {
    if (!s) return ja_strndup(a, "", 0);
    return ja_strndup(a, s, strlen(s));
}

static void ja_free(jinja_arena *a) {
    jinja_arena_block *b = a->head;
    while (b) {
        jinja_arena_block *next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} jbuf;

static void jbuf_append(jbuf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 256;
        while (ncap < b->len + n + 1) ncap *= 2;
        b->ptr = (char *)realloc(b->ptr, ncap);
        if (!b->ptr) abort();
        b->cap = ncap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void jbuf_puts(jbuf *b, const char *s) {
    if (s) jbuf_append(b, s, strlen(s));
}

static void jbuf_putc(jbuf *b, char c) {
    jbuf_append(b, &c, 1);
}

typedef enum {
    JVAL_UNDEFINED = 0,
    JVAL_NONE,
    JVAL_BOOL,
    JVAL_INT,
    JVAL_STR,
    JVAL_LIST,
    JVAL_MAP,
} jval_kind;

typedef struct jval jval;
typedef struct jmap jmap;
typedef struct jlist jlist;

struct jval {
    jval_kind kind;
    union {
        bool b;
        int64_t i;
        struct {
            const char *ptr;
            size_t len;
        } s;
        jlist *list;
        jmap *map;
    } u;
};

struct jlist {
    jval *items;
    size_t len;
    size_t cap;
};

typedef struct {
    const char *key;
    jval val;
} jkv;

struct jmap {
    jkv *entries;
    size_t len;
    size_t cap;
    const char *raw_json;
};

static jval jval_undef(void) { return (jval){.kind = JVAL_UNDEFINED}; }
static jval jval_none(void) { return (jval){.kind = JVAL_NONE}; }
static jval jval_bool(bool b) { return (jval){.kind = JVAL_BOOL, .u.b = b}; }
static jval jval_int(int64_t i) { return (jval){.kind = JVAL_INT, .u.i = i}; }
static jval jval_str_n(jinja_arena *a, const char *s, size_t n) {
    return (jval){.kind = JVAL_STR, .u.s = {ja_strndup(a, s, n), n}};
}
static jval jval_str(jinja_arena *a, const char *s) {
    if (!s) s = "";
    return jval_str_n(a, s, strlen(s));
}
static jval jval_list_new(jinja_arena *a, size_t cap) {
    jlist *l = (jlist *)ja_alloc(a, sizeof(*l));
    l->cap = cap ? cap : 4;
    l->items = (jval *)ja_alloc(a, l->cap * sizeof(jval));
    l->len = 0;
    return (jval){.kind = JVAL_LIST, .u.list = l};
}
static void jlist_push(jinja_arena *a, jlist *l, jval v) {
    if (l->len == l->cap) {
        size_t ncap = l->cap * 2;
        jval *nitems = (jval *)ja_alloc(a, ncap * sizeof(jval));
        memcpy(nitems, l->items, l->len * sizeof(jval));
        l->items = nitems;
        l->cap = ncap;
    }
    l->items[l->len++] = v;
}
static jval jval_map_new(jinja_arena *a, size_t cap) {
    jmap *m = (jmap *)ja_alloc(a, sizeof(*m));
    m->cap = cap ? cap : 8;
    m->entries = (jkv *)ja_alloc(a, m->cap * sizeof(jkv));
    m->len = 0;
    m->raw_json = NULL;
    return (jval){.kind = JVAL_MAP, .u.map = m};
}
static void jmap_set(jinja_arena *a, jmap *m, const char *key, jval v) {
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) {
            m->entries[i].val = v;
            return;
        }
    }
    if (m->len == m->cap) {
        size_t ncap = m->cap * 2;
        jkv *nentries = (jkv *)ja_alloc(a, ncap * sizeof(jkv));
        memcpy(nentries, m->entries, m->len * sizeof(jkv));
        m->entries = nentries;
        m->cap = ncap;
    }
    m->entries[m->len].key = ja_strdup(a, key);
    m->entries[m->len].val = v;
    m->len++;
}
static jval jmap_get(const jmap *m, const char *key) {
    if (!m || !key) return jval_undef();
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) return m->entries[i].val;
    }
    return jval_undef();
}
static bool jmap_has(const jmap *m, const char *key) {
    if (!m || !key) return false;
    for (size_t i = 0; i < m->len; i++) {
        if (!strcmp(m->entries[i].key, key)) {
            return m->entries[i].val.kind != JVAL_UNDEFINED;
        }
    }
    return false;
}

static bool jval_truthy(jval v) {
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return false;
    case JVAL_BOOL: return v.u.b;
    case JVAL_INT: return v.u.i != 0;
    case JVAL_STR: return v.u.s.len > 0;
    case JVAL_LIST: return v.u.list && v.u.list->len > 0;
    case JVAL_MAP: return v.u.map && (v.u.map->len > 0 || (v.u.map->raw_json && v.u.map->raw_json[0]));
    }
    return false;
}

static bool jval_eq(jval a, jval b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return true;
    case JVAL_BOOL: return a.u.b == b.u.b;
    case JVAL_INT: return a.u.i == b.u.i;
    case JVAL_STR:
        return a.u.s.len == b.u.s.len &&
               (!a.u.s.len || !memcmp(a.u.s.ptr, b.u.s.ptr, a.u.s.len));
    case JVAL_LIST:
        if (a.u.list->len != b.u.list->len) return false;
        for (size_t i = 0; i < a.u.list->len; i++)
            if (!jval_eq(a.u.list->items[i], b.u.list->items[i])) return false;
        return true;
    case JVAL_MAP:
        return a.u.map == b.u.map;
    }
    return false;
}

static jval jval_to_str(jinja_arena *a, jval v) {
    char num[64];
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE: return jval_str(a, "");
    case JVAL_BOOL: return jval_str(a, v.u.b ? "True" : "False");
    case JVAL_INT:
        snprintf(num, sizeof(num), "%lld", (long long)v.u.i);
        return jval_str(a, num);
    case JVAL_STR: return v;
    default: return jval_str(a, "");
    }
}

static void jbuf_json_escape(jbuf *b, const char *s, size_t len) {
    jbuf_putc(b, '"');
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': jbuf_puts(b, "\\\""); break;
        case '\\': jbuf_puts(b, "\\\\"); break;
        case '\b': jbuf_puts(b, "\\b"); break;
        case '\f': jbuf_puts(b, "\\f"); break;
        case '\n': jbuf_puts(b, "\\n"); break;
        case '\r': jbuf_puts(b, "\\r"); break;
        case '\t': jbuf_puts(b, "\\t"); break;
        default:
            if (c < 0x20) {
                char hex[8];
                snprintf(hex, sizeof(hex), "\\u%04x", c);
                jbuf_puts(b, hex);
            } else {
                jbuf_putc(b, (char)c);
            }
        }
    }
    jbuf_putc(b, '"');
}

static void jval_write_json(jbuf *b, jval v) {
    char num[64];
    switch (v.kind) {
    case JVAL_UNDEFINED:
    case JVAL_NONE:
        jbuf_puts(b, "null");
        break;
    case JVAL_BOOL:
        jbuf_puts(b, v.u.b ? "true" : "false");
        break;
    case JVAL_INT:
        snprintf(num, sizeof(num), "%lld", (long long)v.u.i);
        jbuf_puts(b, num);
        break;
    case JVAL_STR:
        jbuf_json_escape(b, v.u.s.ptr, v.u.s.len);
        break;
    case JVAL_LIST:
        jbuf_putc(b, '[');
        for (size_t i = 0; i < v.u.list->len; i++) {
            if (i) jbuf_puts(b, ", ");
            jval_write_json(b, v.u.list->items[i]);
        }
        jbuf_putc(b, ']');
        break;
    case JVAL_MAP:
        if (v.u.map->raw_json) {
            jbuf_puts(b, v.u.map->raw_json);
            break;
        }
        jbuf_putc(b, '{');
        for (size_t i = 0; i < v.u.map->len; i++) {
            if (i) jbuf_puts(b, ", ");
            jbuf_json_escape(b, v.u.map->entries[i].key, strlen(v.u.map->entries[i].key));
            jbuf_puts(b, ": ");
            jval_write_json(b, v.u.map->entries[i].val);
        }
        jbuf_putc(b, '}');
        break;
    }
}

typedef enum {
    JTOK_TEXT,
    JTOK_EXPR,
    JTOK_STMT,
} jtok_kind;

typedef struct {
    jtok_kind kind;
    const char *ptr;
    size_t len;
} jtok;

typedef enum {
    JSTMT_TEXT,
    JSTMT_EXPR,
    JSTMT_SET,
    JSTMT_IF,
    JSTMT_FOR,
    JSTMT_MACRO,
    JSTMT_BREAK,
    JSTMT_CONTINUE,
} jstmt_kind;

typedef struct jstmt jstmt;

typedef struct {
    jstmt *items;
    size_t len;
    size_t cap;
} jstmt_list;

typedef struct {
    const char *cond_ptr;
    size_t cond_len;
    jstmt_list body;
} jif_branch;

typedef struct {
    const char *name;
    const char *def_ptr;
    size_t def_len;
} jmacro_param;

struct jstmt {
    jstmt_kind kind;
    union {
        struct {
            const char *ptr;
            size_t len;
        } text;
        struct {
            const char *ptr;
            size_t len;
        } expr;
        struct {
            const char *var;
            const char *attr;
            const char *expr_ptr;
            size_t expr_len;
        } set;
        struct {
            jif_branch *branches;
            size_t branch_count;
        } if_stmt;
        struct {
            const char *var1;
            const char *var2;
            const char *iter_ptr;
            size_t iter_len;
            const char *cond_ptr;
            size_t cond_len;
            jstmt_list body;
            jstmt_list else_body;
        } for_stmt;
        struct {
            const char *name;
            jmacro_param *params;
            size_t param_count;
            jstmt_list body;
        } macro;
    } u;
};

static bool j_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

static const char *find_tag_end(const char *p, const char *end, char close_c) {
    char quote = 0;
    while (p + 1 < end) {
        char c = *p;
        if (quote) {
            if (c == '\\' && p + 1 < end) {
                p += 2;
                continue;
            }
            if (c == quote) quote = 0;
            p++;
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            p++;
            continue;
        }
        if (c == close_c && p[1] == '}') return p;
        p++;
    }
    return NULL;
}

static bool j_lex(jinja_arena *a, const char *src, jtok **out_toks, size_t *out_count,
                  char *err, size_t errlen) {
    size_t cap = 64, len = 0;
    jtok *toks = (jtok *)ja_alloc(a, cap * sizeof(jtok));
    const char *end = src + strlen(src);
    const char *p = src;

    while (p < end) {
        const char *tag = p;
        while (tag + 1 < end) {
            if (tag[0] == '{' && (tag[1] == '%' || tag[1] == '{' || tag[1] == '#')) break;
            tag++;
        }
        if (tag + 1 >= end) tag = end;

        size_t tlen = (size_t)(tag - p);
        if (tag < end) {
            char kind_c = tag[1];
            bool strip_left = (tag + 2 < end && tag[2] == '-');
            if (strip_left) {
                while (tlen > 0 && j_is_space(p[tlen - 1])) tlen--;
            } else if (kind_c == '%' || kind_c == '#') {
                size_t line_start = tlen;
                while (line_start > 0 && p[line_start - 1] != '\n') line_start--;
                bool all_ws = true;
                for (size_t i = line_start; i < tlen; i++) {
                    if (p[i] != ' ' && p[i] != '\t') {
                        all_ws = false;
                        break;
                    }
                }
                if (all_ws) tlen = line_start;
            }
        }
        if (tlen > 0) {
            if (len == cap) {
                size_t ncap = cap * 2;
                jtok *nt = (jtok *)ja_alloc(a, ncap * sizeof(jtok));
                memcpy(nt, toks, len * sizeof(jtok));
                toks = nt;
                cap = ncap;
            }
            toks[len++] = (jtok){JTOK_TEXT, p, tlen};
        }
        if (tag >= end) break;

        char kind_c = tag[1];
        bool strip_left = (tag + 2 < end && tag[2] == '-');
        const char *body_start = tag + 2 + (strip_left ? 1 : 0);
        const char *close = NULL;
        if (kind_c == '#') {
            const char *q = body_start;
            while (q + 1 < end) {
                if (q[0] == '#' && q[1] == '}') {
                    close = q;
                    break;
                }
                q++;
            }
        } else {
            close = find_tag_end(body_start, end, kind_c == '{' ? '}' : '%');
        }
        if (!close) {
            if (err && errlen) snprintf(err, errlen, "unclosed Jinja tag");
            return false;
        }
        bool strip_right = (close > body_start && close[-1] == '-');
        const char *body_end = close - (strip_right ? 1 : 0);
        while (body_start < body_end && j_is_space(*body_start)) body_start++;
        while (body_end > body_start && j_is_space(body_end[-1])) body_end--;

        if (kind_c != '#') {
            if (len == cap) {
                size_t ncap = cap * 2;
                jtok *nt = (jtok *)ja_alloc(a, ncap * sizeof(jtok));
                memcpy(nt, toks, len * sizeof(jtok));
                toks = nt;
                cap = ncap;
            }
            toks[len++] = (jtok){
                kind_c == '{' ? JTOK_EXPR : JTOK_STMT,
                body_start,
                (size_t)(body_end - body_start),
            };
        }

        p = close + 2;
        if (strip_right) {
            while (p < end && j_is_space(*p)) p++;
        } else if (kind_c == '%' || kind_c == '#') {
            if (p < end && *p == '\r') p++;
            if (p < end && *p == '\n') p++;
        }
    }

    *out_toks = toks;
    *out_count = len;
    return true;
}

static bool j_is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static bool j_is_ident_char(char c) {
    return j_is_ident_start(c) || (c >= '0' && c <= '9');
}

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

p36_think_mode p36_think_mode_for_context(p36_think_mode mode, int ctx_size) {
    if (mode == P36_THINK_MAX && ctx_size < (int)P36_THINK_MAX_MIN_CONTEXT) {
        return P36_THINK_HIGH;
    }
    return mode;
}

static uint64_t p36_metal_attention_scratch_bytes(
        uint32_t ctx, uint32_t cap,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
#ifdef P36_METAL
    uint64_t bytes;
    const char *gqa_env = getenv("P36_METAL_ATTN_GQA");
    const char *split_env = getenv("P36_METAL_ATTN_SPLIT");
    const bool gqa = !gqa_env || !gqa_env[0] || gqa_env[0] != '0';
    const bool split = !split_env || !split_env[0] || split_env[0] != '0';
    const bool compact = cache_type_k == P36_KV_CACHE_Q8_0 &&
                         cache_type_v == P36_KV_CACHE_Q4_0;

    if (!compact) return (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);

    /* Up to 1024 keys Metal uses the dense parallel kernel.  Above that,
     * normal multi-token prefill uses one partial per 4096-key GQA group;
     * a one-token tail/decode uses split-K partials.  Size for each path so
     * compacting this buffer cannot change kernel selection or numerics. */
    const uint32_t prefix = ctx < 1024u ? ctx : 1024u;
    const uint32_t prefix_rows = cap < prefix ? cap : prefix;
    bytes = (uint64_t)prefix_rows * P36_N_HEAD * prefix * sizeof(float);
    if (gqa) {
        uint64_t grouped = (uint64_t)cap * P36_N_HEAD *
            ((ctx + 4095u) / 4096u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (grouped > bytes) bytes = grouped;
    } else if (split) {
        uint64_t grouped = (uint64_t)cap * P36_N_HEAD *
            ((ctx + 511u) / 512u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (grouped > bytes) bytes = grouped;
    } else {
        uint64_t dense = (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);
        if (dense > bytes) bytes = dense;
    }
    if (split) {
        uint64_t tail = (uint64_t)P36_N_HEAD *
            ((ctx + 511u) / 512u) * (P36_N_HEAD_DIM + 2u) * sizeof(float);
        if (tail > bytes) bytes = tail;
    } else {
        uint64_t dense_rows = (uint64_t)P36_N_HEAD * ctx * sizeof(float);
        if (dense_rows > bytes) bytes = dense_rows;
    }
    return bytes;
#else
    (void)cache_type_k;
    (void)cache_type_v;
    return (uint64_t)cap * P36_N_HEAD * ctx * sizeof(float);
#endif
}

static uint64_t p36_vulkan_scratch_bytes(
        p36_backend backend, uint32_t ctx, uint32_t cap,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
    uint64_t bytes = 0;
    uint64_t metal_attention_bytes = backend == P36_BACKEND_METAL ?
        p36_metal_attention_scratch_bytes(
            ctx, cap, cache_type_k, cache_type_v) : 0;
    bytes += (uint64_t)cap * P36_N_EMBD * 5u * sizeof(float);
    bytes += (uint64_t)P36_N_EMBD * sizeof(float);
    if (backend != P36_BACKEND_METAL) {
        bytes += (uint64_t)cap *
                 ((P36_N_EMBD + P36_QK_K - 1u) / P36_QK_K) *
                 P36_VK_Q8_K_BYTES;
    }
    bytes += (uint64_t)cap *
             (P36_N_HEAD * P36_N_HEAD_DIM * 4u +
              (backend == P36_BACKEND_METAL ? 0u : P36_N_HEAD_KV) *
                  (P36_N_HEAD_DIM + P36_N_VALUE_DIM)) *
             sizeof(float);
    bytes += (uint64_t)cap * P36_N_SSM_INNER *
             (backend == P36_BACKEND_METAL ? 1u :
              P36_MODEL_DENSE ? 0u : 3u) * sizeof(float);
    {
        uint64_t recurrent_aux_bytes = (uint64_t)cap *
            (P36_N_SSM_CONV_DIM + P36_N_SSM_DT_RANK * 4u) * sizeof(float);
        if ((!P36_MODEL_DENSE && backend != P36_BACKEND_METAL) ||
            (backend == P36_BACKEND_METAL &&
             metal_attention_bytes < recurrent_aux_bytes)) {
            bytes += recurrent_aux_bytes;
        }
    }
    if (backend == P36_BACKEND_METAL && P36_MODEL_DENSE) {
        const uint64_t dense_ffn_panel_bytes =
            (uint64_t)cap * P36_N_FF_SHARED * sizeof(float);
        const uint64_t reusable_panels = metal_attention_bytes /
            dense_ffn_panel_bytes > 2u ? 2u :
            metal_attention_bytes / dense_ffn_panel_bytes;
        /* Attention/recurrent scratch is dead before dense FFN starts.  Reuse
         * it for as many complete FFN panels as it can hold. */
        bytes += (2u - reusable_panels) * dense_ffn_panel_bytes;
    } else if (backend != P36_BACKEND_METAL) {
        bytes += (uint64_t)cap *
                 (P36_N_FF_SHARED * 2u +
                  (P36_MODEL_DENSE ? 0u :
                   P36_N_EXPERT + P36_N_EXPERT_USED * 2u + P36_N_EMBD + 1u)) *
                 sizeof(float);
    }
    bytes += (uint64_t)P36_N_VOCAB * sizeof(float);
    if (backend == P36_BACKEND_METAL) {
        bytes += metal_attention_bytes;
    } else {
        bytes += (uint64_t)cap * P36_N_HEAD *
                 ((ctx + 8191u) / 8192u) *
                 (P36_N_HEAD_DIM + 2u) * sizeof(float);
    }
    return bytes;
}

p36_context_memory p36_context_memory_estimate_configured(
        p36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        p36_kv_cache_type cache_type_k,
        p36_kv_cache_type cache_type_v) {
    p36_context_memory m = {0};
    uint32_t cpu_prefill_cap = p36_default_cpu_prefill_cap();
#ifndef P36_NO_GPU
    uint32_t gpu_prefill_cap = p36_default_gpu_prefill_cap();
#endif
    if (ctx_size <= 0) return m;
#ifndef P36_NO_GPU
    m.prefill_cap = p36_backend_uses_graph(backend) ? gpu_prefill_cap : cpu_prefill_cap;
#else
    m.prefill_cap = cpu_prefill_cap;
#endif
    if (prefill_chunk != 0) m.prefill_cap = prefill_chunk;
#ifndef P36_NO_GPU
    /* Size scratch for the chunk that will actually run.  The graph path
     * clamps the chunk for watchdog safety, so estimating against the
     * unclamped request overshoots by ~206 KB per token of the difference --
     * a 4096 request against a 1024 cap reserves an extra 634 MB that is
     * never touched, on a board with ~15 GiB of unified memory. */
    if (p36_backend_uses_graph(backend)) {
        m.prefill_cap = p36_engine_clamp_prefill_cap(m.prefill_cap);
    }
#endif
    m.raw_cap = p36_backend_uses_graph(backend) ? (uint32_t)ctx_size : 0;
    m.comp_cap = 0;
    if (p36_backend_uses_graph(backend)) {
        uint64_t full_layers = P36_N_LAYER / P36_FULL_ATTENTION_INTERVAL;
        uint64_t recurrent_layers = P36_N_LAYER - full_layers;
        uint64_t k_row = p36_kv_cache_row_bytes(cache_type_k,
                                                P36_N_HEAD_KV * P36_N_HEAD_DIM);
        uint64_t v_row = p36_kv_cache_row_bytes(cache_type_v,
                                                P36_N_HEAD_KV * P36_N_VALUE_DIM);
        m.raw_bytes = full_layers * (uint64_t)ctx_size * (k_row + v_row);
        m.compressed_bytes = recurrent_layers *
                             ((uint64_t)(P36_N_SSM_CONV - 1u) *
                                  P36_N_SSM_CONV_DIM * sizeof(float) +
                              (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE *
                                  P36_N_SSM_DT_RANK * sizeof(float));
        m.scratch_bytes = p36_vulkan_scratch_bytes(
            backend, (uint32_t)ctx_size, m.prefill_cap,
            cache_type_k, cache_type_v);
    } else {
        m.raw_bytes = (uint64_t)P36_N_LAYER * P36_N_HEAD_KV *
                      P36_N_HEAD_DIM * (uint64_t)ctx_size * sizeof(float);
        m.compressed_bytes = (uint64_t)P36_N_LAYER * P36_N_SSM_STATE *
                             P36_N_SSM_GROUP * sizeof(float);
        m.scratch_bytes = p36_cpu_prefill_scratch_bytes(m.prefill_cap);
    }
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

p36_context_memory p36_context_memory_estimate(p36_backend backend, int ctx_size) {
    return p36_context_memory_estimate_configured(
            backend, ctx_size, 0,
            p36_default_kv_cache_type_k(backend, false),
            p36_default_kv_cache_type_v(backend, false));
}

p36_context_memory p36_context_memory_estimate_with_prefill(p36_backend backend,
                                                            int ctx_size,
                                                            uint32_t prefill_chunk) {
    return p36_context_memory_estimate_configured(
            backend, ctx_size, prefill_chunk,
            p36_default_kv_cache_type_k(backend, false),
            p36_default_kv_cache_type_v(backend, false));
}

void p36_tokens_push(p36_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void p36_tokens_free(p36_tokens *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void p36_tokens_copy(p36_tokens *dst, const p36_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) p36_tokens_push(dst, src->v[i]);
}

bool p36_tokens_starts_with(const p36_tokens *tokens, const p36_tokens *prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

static float p36_f16_to_f32(uint16_t h) {
    return p36_quant_f16_to_f32(h);
}

static uint16_t p36_f32_to_f16(float f) {
    return p36_quant_f32_to_f16(f);
}

float p36_quant_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    float out;

    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t p36_quant_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    uint32_t w;
    uint32_t out;
    float base;

    memcpy(&w, &f, sizeof(w));
    base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    {
        const uint32_t shl1 = w + w;
        const uint32_t sign = w & UINT32_C(0x80000000);
        uint32_t bias = shl1 & UINT32_C(0xff000000);
        float bias_f;
        if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
        bias = (bias >> 1) + UINT32_C(0x07800000);
        memcpy(&bias_f, &bias, sizeof(bias_f));
        base += bias_f;
        memcpy(&out, &base, sizeof(out));
        out = ((out >> 13) & UINT32_C(0x00007c00)) +
              (out & UINT32_C(0x00000fff));
        return (uint16_t)((sign >> 16) |
                          (shl1 > UINT32_C(0xff000000) ? UINT16_C(0x7e00) : out));
    }
}

/* Compiled without fast-math: -ffast-math substitutes rcpss approximations
 * for the scale divisions (off by 1 ulp on some inputs, varying across
 * CPUs), which breaks bit-identity with the GPU quantization kernels. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")
#endif
void p36_quant_q8_0(const float *x, void *out, int64_t n) {
    p36_block_q8_0 *y = out;
    int64_t blocks = n / P36_QK8_0;
    if (!x || !out || n < 0 || n % P36_QK8_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        float amax = 0.0f;
        for (int j = 0; j < (int)P36_QK8_0; j++) {
            float a = fabsf(x[b * P36_QK8_0 + j]);
            if (a > amax) amax = a;
        }
        /* Double-rounded divisions: -ffast-math -march=native substitutes
         * rcpss approximations for f32 division, which is off by 1 ulp on
         * some inputs and varies across CPUs. Going through double keeps
         * the result correctly rounded, so the GPU twin can match bit for
         * bit. */
        float d = (float)((double)amax / 127.0);
        float inv = d ? (float)(1.0 / (double)d) : 0.0f;
        y[b].d = p36_quant_f32_to_f16(d);
        for (int j = 0; j < (int)P36_QK8_0; j++)
            y[b].qs[j] = (int8_t)nearbyintf(x[b * P36_QK8_0 + j] * inv);
    }
}

static void p36_quant_q4_0_impl(const float *x, void *out, int64_t n,
                               bool refine) {
    p36_block_q4_0 *y = out;
    int64_t blocks = n / P36_QK4_0;
    if (!x || !out || n < 0 || n % P36_QK4_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        const float *row = x + b * P36_QK4_0;
        float max = 0.0f;
        float amax = 0.0f;
        for (uint32_t i = 0; i < P36_QK4_0; i++) {
            float a = fabsf(row[i]);
            if (a > amax) {
                amax = a;
                max = row[i];
            }
        }
        float d = max / -8.0f;
        float inv = d ? (float)(1.0 / (double)d) : 0.0f;
        /* KV values favor reconstruction error over canonical Q4_0 bytes. */
        if (refine) {
            double num = 0.0;
            int den = 0;
            for (uint32_t i = 0; i < P36_QK4_0; i++) {
                int q = (int)(row[i] * inv + 8.5f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                q -= 8;
                num += (double)row[i] * (double)q;
                den += q * q;
            }
            if (den) {
                float fit = (float)(num / (double)den);
                d = d + (fit - d) * 0.25f;
                inv = d ? (float)(1.0 / (double)d) : 0.0f;
            }
        }
        y[b].d = p36_quant_f32_to_f16(d);
        for (uint32_t i = 0; i < P36_QK4_0 / 2u; i++) {
            int q0 = (int8_t)(row[i] * inv + 8.5f);
            int q1 = (int8_t)(row[i + P36_QK4_0 / 2u] * inv + 8.5f);
            if (refine && q0 < 0) q0 = 0;
            if (refine && q1 < 0) q1 = 0;
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            y[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void p36_quant_q4_0(const float *x, void *out, int64_t n) {
    p36_quant_q4_0_impl(x, out, n, false);
}

void p36_quant_q4_0_kv(const float *x, void *out, int64_t n) {
    p36_quant_q4_0_impl(x, out, n, true);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

void p36_quant_q8_k(const float *x, void *out, int64_t n) {
    p36_block_q8_k *y = out;
    int64_t blocks = n / P36_QK_K;
    if (!x || !out || n < 0 || n % P36_QK_K) return;
    for (int64_t b = 0; b < blocks; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *row = x + b * P36_QK_K;
        for (int j = 0; j < (int)P36_QK_K; j++) {
            float a = fabsf(row[j]);
            if (a > amax) {
                amax = a;
                max = row[j];
            }
        }
        if (amax == 0.0f) {
            memset(&y[b], 0, sizeof(y[b]));
            continue;
        }
        float inv = -127.0f / max;
        for (int j = 0; j < (int)P36_QK_K; j++) {
            int v = (int)lrintf(inv * row[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < (int)P36_QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / inv;
    }
}

float p36_quant_dot_q8_0(const void *a, const void *b, int n) {
    const p36_block_q8_0 *x = a;
    const p36_block_q8_0 *y = b;
    float sum = 0.0f;
    if (!a || !b || n < 0 || n % P36_QK8_0) return 0.0f;
    for (int block = 0; block < n / (int)P36_QK8_0; block++) {
        int isum = 0;
        for (int i = 0; i < (int)P36_QK8_0; i++) isum += x[block].qs[i] * y[block].qs[i];
        sum += p36_f16_to_f32(x[block].d) * p36_f16_to_f32(y[block].d) * (float)isum;
    }
    return sum;
}

static float p36_ref_rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

static void p36_ref_rms_norm(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float scale;
    if (p36_float_norm_accum_mode) {
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
        scale = p36_ref_rsqrtf(ss / (float)n + eps);
    } else {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)src[i] * (double)src[i];
        scale = p36_ref_rsqrtf((float)(ss / (double)n) + eps);
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = src[i] * scale;
        dst[i] = weight ? v * weight[i] : v;
    }
}

static P36_MAYBE_UNUSED void p36_ref_matmul_f32(float *out, const float *w, const float *x, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t row = 0; row < out_dim; row++) {
        const float *wr = w + (uint64_t)row * in_dim;
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)wr[i] * (double)x[i];
        out[row] = (float)acc;
    }
}

static float p36_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float p36_siluf(float x) {
    return x / (1.0f + expf(-x));
}

static float p36_softplusf(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

static void p36_l2_norm(float *x, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    if (ss <= 0.0) return;
    {
        float scale = 1.0f / fmaxf(sqrtf((float)ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

static void p36_softmax_inplace(float *x, uint32_t n) {
    float maxv = -INFINITY;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) if (x[i] > maxv) maxv = x[i];
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum == 0.0) return;
    {
        float inv = (float)(1.0 / sum);
        for (uint32_t i = 0; i < n; i++) x[i] *= inv;
    }
}

static bool p36_all_finite(const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

static void p36_topk_indices(const float *x, uint32_t n, uint32_t k, uint32_t *idx, float *val) {
    for (uint32_t i = 0; i < k; i++) {
        idx[i] = 0;
        val[i] = -INFINITY;
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        for (uint32_t j = 0; j < k; j++) {
            if (v > val[j]) {
                for (uint32_t t = k - 1; t > j; t--) {
                    idx[t] = idx[t - 1];
                    val[t] = val[t - 1];
                }
                idx[j] = i;
                val[j] = v;
                break;
            }
        }
    }
}

static void p36_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
static void p36_dequantize_row_q8_0(const p36_block_q8_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_bf16(const uint16_t *x, float *y, uint64_t k);
static void p36_dequantize_row_q2_0(const p36_block_q2_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_pq2_0(const p36_block_pq2_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_ptq1_0(const p36_block_ptq1_0 *x, float *y, uint64_t k);


static void p36_dequantize_row_q2_k(const p36_block_q2_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q3_k(const p36_block_q3_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q4_k(const p36_block_q4_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q5_k(const p36_block_q5_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q6_k(const p36_block_q6_k *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_xxs(const p36_block_iq2_xxs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_xs(const p36_block_iq2_xs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq3_xxs(const p36_block_iq3_xxs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq1_s(const p36_block_iq1_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq4_nl(const p36_block_iq4_nl *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_s(const p36_block_iq2_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq3_s(const p36_block_iq3_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq4_xs(const p36_block_iq4_xs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq1_m(const p36_block_iq1_m *x, float *y, uint64_t k);

static const uint8_t *p36_iq2xxs_grid_at(uint32_t idx) {
    return p36_iq2xxs_grid + (size_t)idx * 8u;
}

static const uint8_t *p36_iq2xs_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq2xs_grid[idx];
}

static const uint8_t *p36_iq2s_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq2s_grid[idx];
}

static const uint8_t *p36_iq3s_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq3s_grid[idx];
}

static const uint8_t *p36_iq3xxs_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq3xxs_grid[idx];
}

static bool p36_dequantize_row_from_ptr(uint32_t type, const uint8_t *src, float *dst, uint32_t n) {
    switch (type) {
    case P36_TENSOR_F32:
        memcpy(dst, src, (size_t)n * sizeof(*dst));
        return true;
    case P36_TENSOR_F16: {
        const uint16_t *h = (const uint16_t *)src;
        for (uint32_t i = 0; i < n; i++) dst[i] = p36_f16_to_f32(h[i]);
        return true;
    }
    case P36_TENSOR_BF16:
        p36_dequantize_row_bf16((const uint16_t *)src, dst, n);
        return true;
    case P36_TENSOR_Q8_0:
        p36_dequantize_row_q8_0((const p36_block_q8_0 *)src, dst, n);
        return true;
    case P36_TENSOR_Q2_0:
        p36_dequantize_row_q2_0((const p36_block_q2_0 *)src, dst, n);
        return true;
    case P36_TENSOR_PQ2_0:
        p36_dequantize_row_pq2_0((const p36_block_pq2_0 *)src, dst, n);
        return true;
    case P36_TENSOR_PTQ1_0:
        p36_dequantize_row_ptq1_0((const p36_block_ptq1_0 *)src, dst, n);
        return true;
    case P36_TENSOR_Q2_K:
        p36_dequantize_row_q2_k((const p36_block_q2_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q3_K:
        p36_dequantize_row_q3_k((const p36_block_q3_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q4_K:
        p36_dequantize_row_q4_k((const p36_block_q4_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q5_K:
        p36_dequantize_row_q5_k((const p36_block_q5_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q6_K:
        p36_dequantize_row_q6_k((const p36_block_q6_k *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_XXS:
        p36_dequantize_row_iq2_xxs((const p36_block_iq2_xxs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_XS:
        p36_dequantize_row_iq2_xs((const p36_block_iq2_xs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ3_XXS:
        p36_dequantize_row_iq3_xxs((const p36_block_iq3_xxs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ1_S:
        p36_dequantize_row_iq1_s((const p36_block_iq1_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ4_NL:
        p36_dequantize_row_iq4_nl((const p36_block_iq4_nl *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_S:
        p36_dequantize_row_iq2_s((const p36_block_iq2_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ3_S:
        p36_dequantize_row_iq3_s((const p36_block_iq3_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ4_XS:
        p36_dequantize_row_iq4_xs((const p36_block_iq4_xs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ1_M:
        p36_dequantize_row_iq1_m((const p36_block_iq1_m *)src, dst, n);
        return true;
    default:
        return false;
    }
}

bool p36_quant_dequantize(uint32_t type, const void *src, float *out, uint32_t n) {
    return src && out && p36_dequantize_row_from_ptr(type, src, out, n);
}

void p36_quant_pack_iq1_grid(void *dst) {
    memcpy(dst, p36_iq1s_grid, sizeof(p36_iq1s_grid));
}

static void p36_dequantize_row_q8_0(const p36_block_q8_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK8_0) != 0) p36_die("q8_0 row size mismatch");
    nb = k / P36_QK8_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK8_0; j++) *y++ = d * (float)x[i].qs[j];
    }
}

static void p36_dequantize_row_bf16(const uint16_t *x, float *y, uint64_t k) {
    for (uint64_t i = 0; i < k; i++) {
        uint32_t bits = (uint32_t)x[i] << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        y[i] = f;
    }
}

static void p36_dequantize_row_q2_0(const p36_block_q2_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK2_0) != 0) p36_die("q2_0 row size mismatch");
    nb = k / P36_QK2_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK2_0; j++) {
            uint8_t q = (uint8_t)((x[i].qs[j / 4u] >> ((j % 4u) * 2u)) & 0x03u);
            *y++ = (float)((int)q - 1) * d;
        }
    }
}

static void p36_dequantize_row_pq2_0(const p36_block_pq2_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_PQ2_0) != 0) p36_die("pq2_0 row size mismatch");
    nb = k / P36_QK_PQ2_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK_PQ2_0; j++) {
            uint8_t q = (uint8_t)((x[i].qs[j / 4u] >> ((j % 4u) * 2u)) & 0x03u);
            *y++ = (float)((int)q - 1) * d;
        }
    }
}

static void p36_dequantize_row_ptq1_0(const p36_block_ptq1_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_PTQ1_0) != 0) p36_die("ptq1_0 row size mismatch");
    nb = k / P36_QK_PTQ1_0;
    const uint32_t pow3[5] = {1,3,9,27,81};
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t p = 0; p < P36_QK_PTQ1_0; p++) {
            uint8_t byte;
            uint8_t n;
            if (p < 80) {
                byte = x[i].qs[p % 16];
                n = p / 16;
            } else if (p < 120) {
                uint32_t l = p - 80;
                byte = x[i].qs[16 + (l % 8)];
                n = l / 8;
            } else {
                uint32_t l = p - 120;
                byte = x[i].qh[l % 2];
                n = l / 2;
            }
            uint32_t qq = (byte * pow3[n]) & 0xffu;
            uint32_t xi = (qq * 3u) >> 8u;
            *y++ = ((float)xi - 1.0f) * d;
        }
    }
}

static void p36_dequantize_row_q2_k(const p36_block_q2_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q2_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (float)(sc & 0x0f);
                float ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3u) - ml;

                sc = x[i].scales[is++];
                dl = d * (float)(sc & 0x0f);
                ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

static void p36_dequantize_row_q3_k(const p36_block_q3_k *x, float *y, uint64_t k) {
    const uint32_t mask2 = 0x03030303u;
    const uint32_t mask4 = 0x0f0f0f0fu;
    if ((k % P36_QK_K) != 0) p36_die("q3_k row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        uint32_t aux[4] = {0};
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & mask4) | (((tmp >> 4) & mask2) << 4);
        aux[3] = ((aux[1] >> 4) & mask4) | (((tmp >> 6) & mask2) << 4);
        aux[0] = (aux[0] & mask4) | (((tmp >> 0) & mask2) << 4);
        aux[1] = (aux[1] & mask4) | (((tmp >> 2) & mask2) << 4);

        const int8_t *scales = (const int8_t *)aux;
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        float d = p36_f16_to_f32(x[i].d);
        uint8_t m = 1;
        int is = 0;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l] >> shift) & 3u) - ((hm[l] & m) ? 0 : 4));
                dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3u) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void p36_dequantize_row_q4_k(const p36_block_q4_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q4_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t j = 0; j < P36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            p36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            p36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void p36_dequantize_row_q5_k(const p36_block_q5_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q5_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint32_t j = 0; j < P36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            p36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            p36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0f) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void p36_dequantize_row_q6_k(const p36_block_q6_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q6_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void p36_dequantize_row_iq2_xxs(const p36_block_iq2_xxs *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq2_xxs row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ++ib32) {
            uint32_t aux32[2];
            const uint8_t *aux8;
            memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            aux8 = (const uint8_t *)aux32;
            {
                const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *grid = p36_iq2xxs_grid_at(aux8[l]);
                    const uint8_t signs = p36_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; ++j) {
                        y[j] = db * (float)grid[j] * ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
            }
        }
    }
}

static void p36_dequantize_row_iq2_xs(const p36_block_iq2_xs *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq2_xs row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            float db[2] = {
                d * (0.5f + (float)(x[i].scales[ib] & 15u)) * 0.25f,
                d * (0.5f + (float)(x[i].scales[ib] >> 4)) * 0.25f,
            };
            for (uint32_t l = 0; l < 4; l++) {
                uint16_t qs = x[i].qs[4 * ib + l];
                const uint8_t *grid = p36_iq2xs_grid_at(qs & 511u);
                uint8_t signs = p36_ksigns_iq2xs[qs >> 9];
                for (uint32_t j = 0; j < 8; j++)
                    y[j] = db[l / 2] * (float)grid[j] *
                        ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                y += 8;
            }
        }
    }
}

static void p36_dequantize_row_iq2_s(const p36_block_iq2_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq2_s row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + P36_QK_K / 8;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ++ib32) {
            float db0 = d * (0.5f + (float)(x[i].scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (float)(x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = l < 2 ? db0 : db1;
                const uint8_t *grid = p36_iq2s_grid_at((uint32_t)(qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (float)grid[j] * ((signs[l] & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void p36_dequantize_row_iq3_xxs(const p36_block_iq3_xxs *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq3_xxs row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_and_signs = qs + P36_QK_K / 4;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ib32++) {
            uint32_t aux;
            memcpy(&aux, scales_and_signs + 4 * ib32, sizeof(aux));
            const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const uint8_t signs = p36_ksigns_iq2xs[(aux >> (7 * l)) & 127u];
                const uint8_t *grid1 = p36_iq3xxs_grid_at(qs[2 * l]);
                const uint8_t *grid2 = p36_iq3xxs_grid_at(qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j] = db * (float)grid1[j] *
                        ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4] = db * (float)grid2[j] *
                        ((signs & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void p36_dequantize_row_iq3_s(const p36_block_iq3_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq3_s row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ib32 += 2) {
            const float db1 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] & 0x0f));
            const float db2 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * (float)grid1[j] * ((signs[l] & p36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db1 * (float)grid2[j] * ((signs[l] & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * (float)grid1[j] * ((signs[l] & p36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db2 * (float)grid2[j] * ((signs[l] & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static void p36_dequantize_row_iq4_xs(const p36_block_iq4_xs *x, float *y, uint64_t k) {
    static const int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    if ((k % P36_QK_K) != 0) p36_die("iq4_xs row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        const uint8_t *qs = x[b].qs;
        const float d = p36_f16_to_f32(x[b].d);
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            int ls = ((x[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 15) |
                     (((x[b].scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            for (uint32_t j = 0; j < 16; j++) {
                y[j] = dl * values[qs[j] & 15];
                y[j + 16] = dl * values[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

static void p36_dequantize_row_iq1_s(const p36_block_iq1_s *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq1_s row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        float d = p36_f16_to_f32(x[b].d);
        const uint8_t *qs = x[b].qs;
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            uint16_t qh = x[b].qh[ib];
            float dl = d * (float)(2u * ((qh >> 12) & 7u) + 1u);
            float delta = qh & 0x8000u ? -0.125f : 0.125f;
            for (uint32_t l = 0; l < 4; l++) {
                uint32_t idx = qs[l] | (((qh >> (3u * l)) & 7u) << 8);
                const int8_t *grid = (const int8_t *)&p36_iq1s_grid[idx];
                for (uint32_t j = 0; j < 8; j++) y[j] = dl * ((float)grid[j] + delta);
                y += 8;
            }
            qs += 4;
        }
    }
}

static void p36_dequantize_row_iq4_nl(const p36_block_iq4_nl *x, float *y, uint64_t k) {
    static const int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    if ((k % 32u) != 0) p36_die("iq4_nl row size mismatch");
    for (uint64_t b = 0; b < k / 32u; b++) {
        float d = p36_f16_to_f32(x[b].d);
        for (uint32_t j = 0; j < 16; j++) {
            y[j] = d * (float)values[x[b].qs[j] & 15u];
            y[j + 16] = d * (float)values[x[b].qs[j] >> 4];
        }
        y += 32;
    }
}

static void p36_dequantize_row_iq1_m(const p36_block_iq1_m *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq1_m row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        const uint16_t *sc = (const uint16_t *)x[b].scales;
        uint16_t dh = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        float d = p36_f16_to_f32(dh);
        const uint8_t *qs = x[b].qs;
        const uint8_t *qh = x[b].qh;
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            float dl[2] = {
                d * (float)(2 * ((sc[ib / 2] >> (6 * (ib & 1))) & 7) + 1),
                d * (float)(2 * ((sc[ib / 2] >> (6 * (ib & 1) + 3)) & 7) + 1),
            };
            for (uint32_t l = 0; l < 4; l++) {
                uint32_t hi = l & 1 ? qh[l / 2] << 4 : qh[l / 2] << 8;
                const int8_t *grid = (const int8_t *)&p36_iq1s_grid[qs[l] | (hi & 0x700)];
                float delta = qh[l / 2] & (l & 1 ? 0x80 : 0x08) ? -0.125f : 0.125f;
                for (uint32_t j = 0; j < 8; j++) y[j] = dl[l / 2] * ((float)grid[j] + delta);
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

static bool p36_tensor_row_to_float(const p36_model *m, const p36_tensor *t, uint64_t row, float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t row_count;
    uint64_t row_bytes;
    if (!m || !t || !dst || t->ndim == 0 || t->dim[0] != (uint64_t)n) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &row_bytes)) return false;
    src = m->map + t->abs_offset + row * row_bytes;
    return p36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool p36_tensor_row_ptr(const p36_model *m, const p36_tensor *t, uint64_t row, const uint8_t **src, uint64_t *row_bytes) {
    uint64_t row_count;
    uint64_t got_row_bytes;
    if (!m || !t || !src || t->ndim == 0) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return false;
    *src = m->map + t->abs_offset + row * got_row_bytes;
    if (row_bytes) *row_bytes = got_row_bytes;
    return true;
}

static bool p36_tensor_expert_row_ptr(const p36_model *m, const p36_tensor *t, uint32_t expert, uint32_t row,
                                      const uint8_t **src, uint64_t *row_bytes) {
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !src || t->ndim != 3) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    return p36_tensor_row_ptr(m, t, flat_row, src, row_bytes);
}

static P36_MAYBE_UNUSED bool p36_tensor_expert_row_to_float(const p36_model *m, const p36_tensor *t, uint32_t expert, uint32_t row,
                                           float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !dst || t->ndim != 3 || t->dim[0] != (uint64_t)n) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    if (!p36_tensor_row_ptr(m, t, flat_row, &src, NULL)) return false;
    return p36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool p36_tensor_get_plain(const p36_model *m, const p36_tensor *t, float *dst, uint32_t n) {
    if (!m || !t || !dst || t->ndim != 1 || t->dim[0] != (uint64_t)n) return false;
    return p36_dequantize_row_from_ptr(t->type, m->map + t->abs_offset, dst, n);
}

static float p36_tensor_scalar_or(const p36_model *m, const p36_tensor *t, float fallback) {
    float v = fallback;
    if (!t) return fallback;
    if (!p36_tensor_get_plain(m, t, &v, 1)) return fallback;
    return v;
}

static float p36_tensor_index_or(const p36_model *m, const p36_tensor *t, uint32_t idx, float fallback) {
    if (!m || !t || t->ndim != 1 || idx >= t->dim[0]) return fallback;
    if (t->type == P36_TENSOR_F32) {
        const float *src = (const float *)(m->map + t->abs_offset);
        return src[idx];
    }
    if (t->type == P36_TENSOR_F16) {
        const uint16_t *src = (const uint16_t *)(m->map + t->abs_offset);
        return p36_f16_to_f32(src[idx]);
    }
    return fallback;
}

static void p36_scale_inplace(float *x, uint32_t n, float scale) {
    if (!x || scale == 1.0f) return;
    for (uint32_t i = 0; i < n; i++) x[i] *= scale;
}

static bool p36_tensor_type_supports_q8k_dot(uint32_t type) {
    switch (type) {
    case P36_TENSOR_BF16:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return true;
    default:
        return false;
    }
}

static bool p36_tensor_type_supports_q8_0_dot(uint32_t type) {
    return type == P36_TENSOR_Q8_0;
}

static bool p36_tensor_type_supports_prequant_dot(uint32_t type) {
    return p36_tensor_type_supports_q8k_dot(type) || p36_tensor_type_supports_q8_0_dot(type);
}

static bool p36_tensor_pair_can_fuse(const p36_tensor *t0,
                                     const p36_tensor *t1,
                                     uint32_t ndim,
                                     uint32_t in_dim,
                                     uint32_t out_dim) {
    if (!t0 || !t1 || t0->type != t1->type || t0->ndim != ndim || t1->ndim != ndim) return false;
    if (t0->dim[0] != in_dim || t1->dim[0] != in_dim || t0->dim[1] != out_dim || t1->dim[1] != out_dim) return false;
    if (ndim == 3 && t0->dim[2] != t1->dim[2]) return false;
    return p36_tensor_type_supports_prequant_dot(t0->type);
}

static bool p36_tensor_row_dot_q8k(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    const p36_block_q8_k *q = (const p36_block_q8_k *)xq;
    float weights[P36_QK_K];
    double acc = 0.0;
    uint32_t block_bytes;
    if (!row || !xq || !out || (n % P36_QK_K) != 0) return false;
    if (type >= sizeof(gguf_types) / sizeof(gguf_types[0]) || !gguf_types[type].block_bytes) return false;
    block_bytes = gguf_types[type].block_bytes *
                  (P36_QK_K / gguf_types[type].block_elems);
    for (uint32_t block = 0; block < n / P36_QK_K; block++) {
        if (!p36_dequantize_row_from_ptr(type, row + (uint64_t)block * block_bytes,
                                         weights, P36_QK_K)) return false;
        for (uint32_t i = 0; i < P36_QK_K; i++)
            acc += (double)weights[i] * (double)(q[block].d * (float)q[block].qs[i]);
    }
    *out = (float)acc;
    return true;
}

static bool p36_tensor_row_dot_q8k_pair(uint32_t type,
                                        const uint8_t *row0,
                                        const uint8_t *row1,
                                        const uint8_t *xq,
                                        uint32_t n,
                                        float *out0,
                                        float *out1) {
    if (!row0 || !row1 || !xq || !out0 || !out1 || (n % P36_QK_K) != 0) return false;
    return p36_tensor_row_dot_q8k(type, row0, xq, n, out0) &&
           p36_tensor_row_dot_q8k(type, row1, xq, n, out1);
}

static bool p36_tensor_row_dot_prequant(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    float acc = 0.0f;
    if (type == P36_TENSOR_Q8_0) {
        if (!row || !xq || !out || (n % P36_QK8_0) != 0) return false;
        acc = p36_quant_dot_q8_0(row, xq, (int)n);
        *out = acc;
        return true;
    }
    return p36_tensor_row_dot_q8k(type, row, xq, n, out);
}

static bool p36_tensor_row_dot_prequant_pair(uint32_t type,
                                             const uint8_t *row0,
                                             const uint8_t *row1,
                                             const uint8_t *xq,
                                             uint32_t n,
                                             float *out0,
                                             float *out1) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (type == P36_TENSOR_Q8_0) {
        if (!row0 || !row1 || !xq || !out0 || !out1 || (n % P36_QK8_0) != 0) return false;
        acc0 = p36_quant_dot_q8_0(row0, xq, (int)n);
        acc1 = p36_quant_dot_q8_0(row1, xq, (int)n);
        *out0 = acc0;
        *out1 = acc1;
        return true;
    }
    return p36_tensor_row_dot_q8k_pair(type, row0, row1, xq, n, out0, out1);
}

typedef enum {
    P36_ACTIVATION_QUANT_NONE,
    P36_ACTIVATION_QUANT_Q8_K,
    P36_ACTIVATION_QUANT_Q8_0,
} p36_activation_quant_kind;

static p36_activation_quant_kind p36_activation_quant_kind_for_type(uint32_t type) {
    if (p36_tensor_type_supports_q8_0_dot(type)) return P36_ACTIVATION_QUANT_Q8_0;
    if (p36_tensor_type_supports_q8k_dot(type)) return P36_ACTIVATION_QUANT_Q8_K;
    return P36_ACTIVATION_QUANT_NONE;
}

static bool p36_activation_quant_valid_dim(p36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case P36_ACTIVATION_QUANT_NONE:
        return true;
    case P36_ACTIVATION_QUANT_Q8_K:
        return (n % P36_QK_K) == 0;
    case P36_ACTIVATION_QUANT_Q8_0:
        return (n % P36_QK8_0) == 0;
    default:
        return false;
    }
}

static uint32_t p36_activation_quant_row_bytes(p36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case P36_ACTIVATION_QUANT_Q8_K:
        return p36_q8k_row_bytes(n);
    case P36_ACTIVATION_QUANT_Q8_0:
        return p36_q8_0_row_bytes(n);
    default:
        return 0;
    }
}

static bool p36_quantize_activation_row(p36_activation_quant_kind kind, const float *x, uint8_t *xq, uint32_t n) {
    if (kind == P36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || !p36_activation_quant_valid_dim(kind, n)) return false;
    if (kind == P36_ACTIVATION_QUANT_Q8_0) p36_quant_q8_0(x, xq, (int64_t)n);
    else p36_quant_q8_k(x, xq, (int64_t)n);
    return true;
}

typedef struct {
    const float *x;
    uint8_t *xq;
    p36_activation_quant_kind kind;
    uint32_t in_dim;
    uint32_t row_bytes;
} p36_quantize_activation_batch_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_tensor *t;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} p36_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} p36_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_tensor *t;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} p36_expert_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} p36_expert_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const float *w;
    const float *x;
    uint32_t n_tok;
    uint32_t n;
} p36_vector_dot_batch_ctx;

typedef struct {
    float *x;
    uint32_t rows;
    uint32_t cols;
    float scale;
} p36_scale_rows_ctx;

typedef struct {
    float *dst;
    const float *a;
    const float *b;
    uint32_t rows;
    uint32_t cols;
} p36_add_rows_ctx;

typedef struct {
    float *dst;
    const float *src;
    const float *weight;
    uint32_t rows;
    uint32_t cols;
    float eps;
} p36_rms_norm_rows_ctx;

typedef struct {
    float *out;
    const float *gate;
    const float *up;
    uint32_t rows;
    uint32_t cols;
} p36_swiglu_rows_ctx;

static void p36_quantize_activation_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_quantize_activation_batch_ctx *ctx = (p36_quantize_activation_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        p36_quantize_activation_row(ctx->kind,
                                    ctx->x + t * ctx->in_dim,
                                    ctx->xq + t * ctx->row_bytes,
                                    ctx->in_dim);
    }
}

static bool p36_quantize_activation_batch(p36_activation_quant_kind kind,
                                          const float *x,
                                          uint8_t *xq,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                          uint32_t n_threads) {
    p36_quantize_activation_batch_ctx ctx;
    if (kind == P36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || n_tok == 0 || !p36_activation_quant_valid_dim(kind, in_dim)) return false;
    ctx.x = x;
    ctx.xq = xq;
    ctx.kind = kind;
    ctx.in_dim = in_dim;
    ctx.row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    p36_parallel_for_rows(n_tok, 2, n_threads, p36_quantize_activation_batch_worker, &ctx);
    return true;
}

static bool p36_quantize_activation_batch_for_type(uint32_t type,
                                                   const float *x,
                                                   uint8_t *xq,
                                                   uint32_t n_tok,
                                                   uint32_t in_dim,
                                                   uint32_t n_threads) {
    return p36_quantize_activation_batch(p36_activation_quant_kind_for_type(type), x, xq, n_tok, in_dim, n_threads);
}

static void p36_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_matmul_batch_ctx *ctx = (p36_matmul_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case P36_TENSOR_F32: {
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case P36_TENSOR_F16: {
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)p36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !p36_tensor_row_ptr(m, ctx->t, row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!p36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void p36_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_matmul_pair_batch_ctx *ctx = (p36_matmul_pair_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !p36_tensor_row_ptr(m, ctx->t0, row, &src0, NULL) ||
            !p36_tensor_row_ptr(m, ctx->t1, row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!p36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void p36_expert_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_expert_matmul_batch_ctx *ctx = (p36_expert_matmul_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case P36_TENSOR_F32: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case P36_TENSOR_F16: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)p36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !p36_tensor_expert_row_ptr(m, ctx->t, ctx->expert, (uint32_t)row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!p36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void p36_expert_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_expert_matmul_pair_batch_ctx *ctx = (p36_expert_matmul_pair_batch_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !p36_tensor_expert_row_ptr(m, ctx->t0, ctx->expert, (uint32_t)row, &src0, NULL) ||
            !p36_tensor_expert_row_ptr(m, ctx->t1, ctx->expert, (uint32_t)row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!p36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void p36_vector_dot_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vector_dot_batch_ctx *ctx = (p36_vector_dot_batch_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        const float *xr = ctx->x + row * ctx->n;
        double acc = 0.0;
        for (uint32_t i = 0; i < ctx->n; i++) acc += (double)ctx->w[i] * (double)xr[i];
        ctx->out[row] = (float)acc;
    }
}

static void p36_scale_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_scale_rows_ctx *ctx = (p36_scale_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *xr = ctx->x + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) xr[i] *= ctx->scale;
    }
}

static void p36_add_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_add_rows_ctx *ctx = (p36_add_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *dr = ctx->dst + row * ctx->cols;
        const float *ar = ctx->a + row * ctx->cols;
        const float *br = ctx->b + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) dr[i] = ar[i] + br[i];
    }
}

static void p36_rms_norm_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_rms_norm_rows_ctx *ctx = (p36_rms_norm_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        p36_ref_rms_norm(ctx->dst + row * ctx->cols,
                         ctx->src + row * ctx->cols,
                         ctx->weight,
                         ctx->cols,
                         ctx->eps);
    }
}

static void p36_swiglu_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_swiglu_rows_ctx *ctx = (p36_swiglu_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *orow = ctx->out + row * ctx->cols;
        const float *grow = ctx->gate + row * ctx->cols;
        const float *urow = ctx->up + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++)
            orow[i] = grow[i] / (1.0f + expf(-grow[i])) * urow[i];
    }
}

/* Contract-aware whole-model matmul: emulates the dense MMQ kernel's f16
 * staging/accumulation (see p36_mmq_contract.h). Used only when
 * e->cpu_mmq_contract is set, i.e. the parity reference. The activation f16
 * values are precomputed once per token and the weight decode is shared across
 * tokens, which is what makes the scalar emulation affordable. */
typedef struct {
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;      /* NULL for a single projection */
    const float *act;
    float *out0;
    float *out1;
    uint32_t n_tok, in_dim, out_dim, q8k_blocks;
    uint32_t blk_bytes0, blk_per_q8k0, lane_shift0, lane_mask0;
    uint32_t blk_bytes1, blk_per_q8k1, lane_shift1, lane_mask1;
    float scale0, scale1;
} p36_contract_matmul_ctx;

static void p36_contract_matmul_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_contract_matmul_ctx *ctx = (p36_contract_matmul_ctx *)opaque;
    const p36_model *m = &ctx->e->model;
    float *sums = malloc((size_t)ctx->n_tok * sizeof(float));
    if (!sums) return;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0 = NULL;
        const uint8_t *src1 = NULL;
        if (!p36_tensor_row_ptr(m, ctx->t0, row, &src0, NULL)) continue;
        if (ctx->t1 && !p36_tensor_row_ptr(m, ctx->t1, row, &src1, NULL)) continue;
        p36_contract_mmq_q2_dot_multi(
            src0, ctx->act, sums, ctx->n_tok, ctx->in_dim, ctx->q8k_blocks,
            ctx->blk_bytes0, ctx->blk_per_q8k0, ctx->lane_shift0,
            ctx->lane_mask0, ctx->scale0, ctx->out0 + row, ctx->out_dim);
        if (ctx->t1) {
            p36_contract_mmq_q2_dot_multi(
                src1, ctx->act, sums, ctx->n_tok, ctx->in_dim, ctx->q8k_blocks,
                ctx->blk_bytes1, ctx->blk_per_q8k1, ctx->lane_shift1,
                ctx->lane_mask1, ctx->scale1, ctx->out1 + row, ctx->out_dim);
        }
    }
    free(sums);
}

static bool p36_contract_matmul_batch(const p36_engine *e,
                                      const p36_tensor *t0,
                                      const p36_tensor *t1,
                                      const uint8_t *xq,
                                      float *out0,
                                      float *out1,
                                      uint32_t n_tok,
                                      uint32_t in_dim,
                                      uint32_t out_dim,
                                      float scale0,
                                      float scale1) {
    p36_contract_matmul_ctx ctx;
    uint32_t bb0, bq0, ls0, lm0, bb1 = 0, bq1 = 0, ls1 = 0, lm1 = 0;
    uint32_t q8k_blocks, row_bytes;
    float *act;
    if (!e || !t0 || !xq || n_tok == 0 || in_dim == 0 || out_dim == 0 ||
        in_dim % P36_QK_K != 0) return false;
    if (!p36_contract_q2_geometry(t0->type, &bb0, &bq0, &ls0, &lm0)) return false;
    if (t1 && !p36_contract_q2_geometry(t1->type, &bb1, &bq1, &ls1, &lm1)) return false;
    q8k_blocks = in_dim / P36_QK_K;
    row_bytes = q8k_blocks * P36_Q8_K_BYTES;
    act = malloc((size_t)n_tok * in_dim * sizeof(float));
    if (!act) return false;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        p36_contract_mmq_q2_act(xq + (uint64_t)tok * row_bytes,
                                P36_Q8_K_BYTES, 4u, q8k_blocks,
                                act + (uint64_t)tok * in_dim);
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.act = act;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.q8k_blocks = q8k_blocks;
    ctx.blk_bytes0 = bb0; ctx.blk_per_q8k0 = bq0;
    ctx.lane_shift0 = ls0; ctx.lane_mask0 = lm0;
    ctx.blk_bytes1 = bb1; ctx.blk_per_q8k1 = bq1;
    ctx.lane_shift1 = ls1; ctx.lane_mask1 = lm1;
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    p36_parallel_for_rows(out_dim, 1, e->n_threads, p36_contract_matmul_worker, &ctx);
    free(act);
    return true;
}

static bool p36_tensor_matmul_batch_prequant(const p36_engine *e,
                                             const p36_tensor *t,
                                             const float *x,
                                             const uint8_t *xq,
                                             float *out,
                                             uint32_t n_tok,
                                             uint32_t in_dim,
                                             uint32_t out_dim,
                                             float scale) {
    p36_matmul_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 2 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    if (e->cpu_mmq_contract && n_tok > 1u && xq) {
        uint32_t bb, bq, ls, lm;
        if (p36_contract_q2_geometry(t->type, &bb, &bq, &ls, &lm))
            return p36_contract_matmul_batch(e, t, NULL, xq, out, NULL,
                                             n_tok, in_dim, out_dim, scale, 0.0f);
    }
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_matmul_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_matmul_pair_batch_prequant(const p36_engine *e,
                                                  const p36_tensor *t0,
                                                  const p36_tensor *t1,
                                                  const float *x,
                                                  const uint8_t *xq,
                                                  float *out0,
                                                  float *out1,
                                                  uint32_t n_tok,
                                                  uint32_t in_dim,
                                                  uint32_t out_dim,
                                                  float scale0,
                                                  float scale1) {
    p36_matmul_pair_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!p36_tensor_pair_can_fuse(t0, t1, 2, in_dim, out_dim)) return false;
    if (e->cpu_mmq_contract && n_tok > 1u && xq) {
        uint32_t bb, bq, ls, lm;
        if (p36_contract_q2_geometry(t0->type, &bb, &bq, &ls, &lm) &&
            p36_contract_q2_geometry(t1->type, &bb, &bq, &ls, &lm))
            return p36_contract_matmul_batch(e, t0, t1, xq, out0, out1,
                                             n_tok, in_dim, out_dim, scale0, scale1);
    }
    kind = p36_activation_quant_kind_for_type(t0->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_matmul_pair_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_expert_matmul_batch_prequant(const p36_engine *e,
                                                    const p36_tensor *t,
                                                    uint32_t expert,
                                                    const float *x,
                                                    const uint8_t *xq,
                                                    float *out,
                                                    uint32_t n_tok,
                                                    uint32_t in_dim,
                                                    uint32_t out_dim,
                                                    float scale) {
    p36_expert_matmul_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 3 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_expert_matmul_batch_worker, &ctx);
    return true;
}

static bool p36_tensor_expert_matmul_pair_batch_prequant(const p36_engine *e,
                                                         const p36_tensor *t0,
                                                         const p36_tensor *t1,
                                                         uint32_t expert,
                                                         const float *x,
                                                         const uint8_t *xq,
                                                         float *out0,
                                                         float *out1,
                                                         uint32_t n_tok,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim,
                                                         float scale0,
                                                         float scale1) {
    p36_expert_matmul_pair_batch_ctx ctx;
    p36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!p36_tensor_pair_can_fuse(t0, t1, 3, in_dim, out_dim)) return false;
    kind = p36_activation_quant_kind_for_type(t0->type);
    if (kind != P36_ACTIVATION_QUANT_NONE && (!xq || !p36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    p36_parallel_for_rows(out_dim, min_rows, e->n_threads, p36_expert_matmul_pair_batch_worker, &ctx);
    return true;
}

static void p36_tensor_vector_dot_batch(float *out,
                                        const float *w,
                                        const float *x,
                                        uint32_t n_tok,
                                        uint32_t n,
                                        uint32_t n_threads) {
    p36_vector_dot_batch_ctx ctx;
    if (!out || !w || !x || n_tok == 0) return;
    ctx.out = out;
    ctx.w = w;
    ctx.x = x;
    ctx.n_tok = n_tok;
    ctx.n = n;
    p36_parallel_for_rows(n_tok, 2, n_threads, p36_vector_dot_batch_worker, &ctx);
}

static P36_MAYBE_UNUSED void p36_scale_rows(float *x, uint32_t rows, uint32_t cols, float scale, uint32_t n_threads) {
    p36_scale_rows_ctx ctx;
    if (!x || scale == 1.0f || rows == 0) return;
    ctx.x = x;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.scale = scale;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_scale_rows_worker, &ctx);
}

static void p36_add_rows(float *dst,
                         const float *a,
                         const float *b,
                         uint32_t rows,
                         uint32_t cols,
                         uint32_t n_threads) {
    p36_add_rows_ctx ctx;
    if (!dst || !a || !b || rows == 0) return;
    ctx.dst = dst;
    ctx.a = a;
    ctx.b = b;
    ctx.rows = rows;
    ctx.cols = cols;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_add_rows_worker, &ctx);
}

static void p36_directional_steering_project_rows(float *x,
                                                   const float *dirs,
                                                   uint32_t layer,
                                                   uint32_t rows,
                                                   float scale) {
    if (!x || !dirs || rows == 0 || scale == 0.0f) return;
    const float *dir = dirs + (uint64_t)layer * P36_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * P36_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) dot += xr[i] * dir[i];
        float coeff = scale * dot;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) xr[i] -= coeff * dir[i];
    }
}

static void p36_rms_norm_rows(float *dst,
                              const float *src,
                              const float *weight,
                              uint32_t rows,
                              uint32_t cols,
                              float eps,
                              uint32_t n_threads) {
    p36_rms_norm_rows_ctx ctx;
    if (!dst || !src || rows == 0) return;
    ctx.dst = dst;
    ctx.src = src;
    ctx.weight = weight;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.eps = eps;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_rms_norm_rows_worker, &ctx);
}

static void p36_swiglu_rows(float *out,
                            const float *gate,
                            const float *up,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t n_threads) {
    p36_swiglu_rows_ctx ctx;
    if (!out || !gate || !up || rows == 0) return;
    ctx.out = out;
    ctx.gate = gate;
    ctx.up = up;
    ctx.rows = rows;
    ctx.cols = cols;
    p36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, p36_swiglu_rows_worker, &ctx);
}

static bool p36_tensor_matvec_prequant(const p36_engine *e,
                                       const p36_tensor *t,
                                       const float *x,
                                       const uint8_t *xq,
                                       float *out,
                                       float *rowbuf,
                                       uint32_t in_dim,
                                       uint32_t out_dim) {
    (void)rowbuf;
    return p36_tensor_matmul_batch_prequant(e, t, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool p36_tensor_matvec_pair_prequant(const p36_engine *e,
                                            const p36_tensor *t0,
                                            const p36_tensor *t1,
                                            const float *x,
                                            const uint8_t *xq,
                                            float *out0,
                                            float *out1,
                                            float *rowbuf,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            float scale0,
                                            float scale1) {
    (void)rowbuf;
    return p36_tensor_matmul_pair_batch_prequant(e, t0, t1, x, xq, out0, out1, 1, in_dim, out_dim, scale0, scale1);
}

static bool p36_tensor_matvec(const p36_engine *e, const p36_tensor *t, const float *x, float *out,
                              float *rowbuf, uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    p36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE &&
        p36_activation_quant_valid_dim(kind, in_dim) &&
        p36_activation_quant_row_bytes(kind, in_dim) <= P36_MAX_Q8_K_BYTES &&
        p36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return p36_tensor_matvec_prequant(e, t, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static bool p36_tensor_expert_matvec_prequant(const p36_engine *e,
                                              const p36_tensor *t,
                                              uint32_t expert,
                                              const float *x,
                                              const uint8_t *xq,
                                              float *out,
                                              float *rowbuf,
                                              uint32_t in_dim,
                                              uint32_t out_dim) {
    (void)rowbuf;
    return p36_tensor_expert_matmul_batch_prequant(e, t, expert, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool p36_tensor_expert_matvec_pair_prequant(const p36_engine *e,
                                                   const p36_tensor *t0,
                                                   const p36_tensor *t1,
                                                   uint32_t expert,
                                                   const float *x,
                                                   const uint8_t *xq,
                                                   float *out0,
                                                   float *out1,
                                                   float *rowbuf,
                                                   uint32_t in_dim,
                                                   uint32_t out_dim,
                                                   float scale0,
                                                   float scale1) {
    (void)rowbuf;
    return p36_tensor_expert_matmul_pair_batch_prequant(e, t0, t1, expert,
                                                         x, xq, out0, out1,
                                                         1, in_dim, out_dim,
                                                         scale0, scale1);
}

static bool p36_tensor_expert_matvec(const p36_engine *e, const p36_tensor *t, uint32_t expert,
                                     const float *x, float *out, float *rowbuf,
                                     uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    p36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = p36_activation_quant_kind_for_type(t->type);
    if (kind != P36_ACTIVATION_QUANT_NONE &&
        p36_activation_quant_valid_dim(kind, in_dim) &&
        p36_activation_quant_row_bytes(kind, in_dim) <= P36_MAX_Q8_K_BYTES &&
        p36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return p36_tensor_expert_matvec_prequant(e, t, expert, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static void cursor_error(p36_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(p36_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(p36_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(p36_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(p36_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(p36_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(p36_cursor *c, p36_str *s) {
    uint64_t len = 0;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    if (rem == 0) return value;
    if (value > UINT64_MAX - (alignment - rem)) p36_die("GGUF alignment overflow");
    return value + alignment - rem;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(p36_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    {
        uint64_t scalar = scalar_value_size(type);
        if (scalar != 0) return cursor_skip(c, scalar);
    }

    if (type == GGUF_VALUE_STRING) {
        p36_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t len = 0;
        uint64_t item_size = 0;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;
        item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = (uint32_t)(sizeof(gguf_types) / sizeof(gguf_types[0]));
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    uint64_t blocks = 0;
    if (!info || info->block_elems == 0) return false;
    blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static p36_cursor cursor_at(const p36_model *m, uint64_t pos) {
    p36_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.error[0] = '\0';
    return c;
}

static p36_kv *model_find_kv(const p36_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (p36_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const p36_model *m, const char *key, p36_str *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const p36_model *m, const char *key, uint32_t *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64_compat(const p36_model *m, const char *key, uint64_t *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32_compat(const p36_model *m, const char *key, float *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) return cursor_read(&c, out, sizeof(*out));
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const p36_model *m, const char *key, bool *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    uint8_t v = 0;
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

static bool model_get_array(const p36_model *m, const char *key, p36_array_ref *out) {
    p36_kv *kv = model_find_kv(m, key);
    p36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(p36_model *m) {
    if (!m) return;
    if (m->owned) {
        free(m->kv);
        free(m->tensors);
        if (m->map) munmap((void *)m->map, (size_t)m->map_len);
        if (m->fd >= 0) close(m->fd);
    }
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void parse_metadata(p36_model *m, p36_cursor *c) {
    uint64_t remaining = c->size - c->pos;
    if (m->n_kv > remaining / 12 ||
        m->n_kv > SIZE_MAX / sizeof(m->kv[0]))
        p36_die("GGUF metadata count exceeds file size");
    m->kv = xcalloc((size_t)m->n_kv, sizeof(m->kv[0]));
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        p36_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) p36_die(c->error);
        if (!cursor_u32(c, &kv->type)) p36_die(c->error);
        kv->value_pos = c->pos;
        if (p36_streq(kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
            p36_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t align = 0;
            if (cursor_u32(&tmp, &align) && align != 0) m->alignment = align;
        }
        if (!skip_value(c, kv->type, 0)) p36_die(c->error);
    }
}

/* True for a tensor of the appended NextN/MTP block (blk.<block_count-1>.*).
 * Some third-party GGUFs store that block in types the loader has no decoder
 * for (e.g. Q4_0); it is only read when the MTP head is bound, so such tensors
 * are kept as zero-byte placeholders instead of rejecting the whole file. */
static bool tensor_in_appended_nextn_block(const p36_model *m, p36_str name) {
    p36_str arch;
    char key[96], prefix[24];
    uint32_t block_count = 0, nextn = 0;
    int len;
    if (!model_get_string(m, "general.architecture", &arch) || arch.len > 64) return false;
    snprintf(key, sizeof(key), "%.*s.block_count", (int)arch.len, arch.ptr);
    if (!model_get_u32(m, key, &block_count) || block_count == 0) return false;
    snprintf(key, sizeof(key), "%.*s.nextn_predict_layers", (int)arch.len, arch.ptr);
    if (!model_get_u32(m, key, &nextn) || nextn != 1) return false;
    len = snprintf(prefix, sizeof(prefix), "blk.%u.", block_count - 1);
    return len > 0 && (uint64_t)len <= name.len && memcmp(name.ptr, prefix, (size_t)len) == 0;
}

/* Re-encode Q4_0 tensors as Q8_0 into an anonymous arena right after the file
 * in the reserved address range, so downstream code keeps addressing weights as
 * map + abs_offset.  Lossless: Q4_0 value (nibble - 8) is in [-8, 7] and keeps
 * the block's f16 scale, so the dequantized weights are bit-identical. */
static void model_convert_q4_0(p36_model *m) {
    const uint64_t arena_off = align_up(m->file_size, 1u << 20);
    uint64_t used = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            if (used == 0) return;
            uint64_t len = align_up(used, 1u << 20);
            if (arena_off + len > m->map_len ||
                mmap((void *)(m->map + arena_off), (size_t)len, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED)
                p36_die("cannot map the Q4_0 conversion arena");
            used = 0;
        }
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            p36_tensor *t = &m->tensors[i];
            if (!t->from_q4_0) continue;
            uint64_t src = m->tensor_data_pos + t->rel_offset;
            uint64_t q4_bytes = t->elements / P36_QK8_0 * P36_Q4_0_BYTES;
            if (src > m->file_size || q4_bytes > m->file_size - src)
                p36_die("tensor points outside GGUF file");
            uint64_t dst = align_up(used, 32);
            used = dst + t->bytes;
            if (pass == 0) continue;
            const uint8_t *q4 = m->map + src;
            p36_block_q8_0 *q8 = (p36_block_q8_0 *)(void *)((uint8_t *)m->map + arena_off + dst);
            for (uint64_t b = 0; b < t->elements / P36_QK8_0; b++, q4 += P36_Q4_0_BYTES) {
                memcpy(&q8[b].d, q4, sizeof(q8[b].d));
                for (uint32_t j = 0; j < 16; j++) {
                    q8[b].qs[j] = (int8_t)((q4[2 + j] & 15u) - 8);
                    q8[b].qs[j + 16] = (int8_t)((q4[2 + j] >> 4) - 8);
                }
            }
            t->abs_offset = arena_off + dst;
            if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
        }
    }
    mprotect((void *)(m->map + arena_off), (size_t)align_up(used, 1u << 20), PROT_READ);
    m->size = arena_off + used;
}

static void parse_tensors(p36_model *m, p36_cursor *c) {
    uint64_t remaining = c->size - c->pos;
    if (m->n_tensors > remaining / 32 ||
        m->n_tensors > SIZE_MAX / sizeof(m->tensors[0]))
        p36_die("GGUF tensor count exceeds file size");
    m->tensors = xcalloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        p36_tensor *t = &m->tensors[i];
        if (!cursor_string(c, &t->name)) p36_die(c->error);
        if (!cursor_u32(c, &t->ndim)) p36_die(c->error);
        if (t->ndim == 0 || t->ndim > P36_MAX_DIMS) p36_die("tensor has an unsupported number of dimensions");
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) p36_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) p36_die("tensor element count overflow");
            t->elements *= t->dim[d];
        }
        if (!cursor_u32(c, &t->type)) p36_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) p36_die(c->error);
        if (t->type == P36_TENSOR_Q4_0 && !tensor_in_appended_nextn_block(m, t->name)) {
            /* Q4_0 -> Q8_0 is exact (q-8 fits int8, same f16 scale), so the
             * whole Q8_0 kernel set serves these tensors; see
             * model_convert_q4_0. */
            if ((t->elements % P36_QK8_0) != 0) p36_die("q4_0 tensor row is not a multiple of 32");
            t->from_q4_0 = true;
            t->type = P36_TENSOR_Q8_0;
            t->bytes = t->elements / P36_QK8_0 * sizeof(p36_block_q8_0);
            continue;
        }
        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            if (tensor_in_appended_nextn_block(m, t->name)) {
                t->bytes = 0;
                continue;
            }
            fprintf(stderr, "p36: unsupported tensor type %u for %.*s\n", t->type, (int)t->name.len, t->name.ptr);
            exit(1);
        }
    }
    m->tensor_data_pos = align_up(c->pos, m->alignment);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        p36_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) p36_die("tensor offset overflow");
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->from_q4_0) continue;   /* placed by model_convert_q4_0 */
        if (t->bytes != 0 && (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            p36_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
    }
    model_convert_q4_0(m);
}

static void model_open(p36_model *m, const char *path, bool graph_mapping) {
    struct stat st;
    int fd;
    void *map;
    p36_cursor c;
    uint32_t magic = 0;
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    fd = open(path, O_RDONLY);
    if (fd == -1) p36_die_errno("cannot open model", path);
    if (fstat(fd, &st) == -1) p36_die_errno("cannot stat model", path);
    if (st.st_size < 32) p36_die("model file is too small to be GGUF");
    if ((uintmax_t)st.st_size > SIZE_MAX) p36_die("model file is too large to map");
    /* Reserve twice the file in address space (never committed) so a Q4_0
     * conversion arena can follow the file mapping; see model_convert_q4_0. */
    uint64_t reserve = align_up((uint64_t)st.st_size, 1u << 20) + 2u * (uint64_t)st.st_size;
    map = mmap(NULL, (size_t)reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (map == MAP_FAILED) p36_die_errno("cannot reserve model address range", path);
    if (mmap(map, (size_t)st.st_size, PROT_READ, (graph_mapping ? MAP_SHARED : MAP_PRIVATE) | MAP_FIXED, fd, 0) == MAP_FAILED)
        p36_die_errno("cannot mmap model", path);
    m->fd = fd;
    m->map = map;
    m->map_len = reserve;
    m->size = m->file_size = (uint64_t)st.st_size;
    m->owned = true;
    c = cursor_at(m, 0);
    if (!cursor_u32(&c, &magic)) p36_die(c.error);
    if (magic != P36_GGUF_MAGIC) p36_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) p36_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) p36_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) p36_die(c.error);
    if (m->version != 3) p36_die("only GGUF v3 is supported");
    parse_metadata(m, &c);
    parse_tensors(m, &c);
}

static p36_tensor *model_find_tensor(const p36_model *m, const char *name) {
    size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len && memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static uint32_t required_u32(const p36_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const p36_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static p36_str required_string(const p36_model *m, const char *key) {
    p36_str s = {0};
    if (!model_get_string(m, key, &s)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return s;
}

static p36_tensor *required_tensor(const p36_model *m, const char *name) {
    p36_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "p36: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static p36_tensor *tensor_by_namef(const p36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) p36_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static p36_tensor *required_tensorf(const p36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) p36_die("tensor name is too long");
    return required_tensor(m, name);
}

typedef struct {
    const p36_model *model;
    const p36_tensor *weight;
    const float *input;
    float *output;
    uint32_t rows;
    uint32_t in_dim;
    uint32_t out_dim;
    bool add;
} p36_vision_mm_ctx;

static bool p36_vision_weight_type_ok(const p36_tensor *t, uint32_t in_dim) {
    if (!t) return false;
    if (t->type == P36_TENSOR_F16 || t->type == P36_TENSOR_F32) return true;
    return t->type == P36_TENSOR_Q8_0 && in_dim % P36_QK8_0 == 0;
}

/* Dequantize output row `o` of a vision weight into dst[in_dim]. */
static void p36_vision_weight_row_f32(const p36_model *m, const p36_tensor *t,
                                      uint32_t in_dim, uint64_t o, float *dst) {
    const uint8_t *base = m->map + t->abs_offset;
    if (t->type == P36_TENSOR_F32) {
        memcpy(dst, (const float *)base + o * in_dim, (size_t)in_dim * sizeof(float));
    } else if (t->type == P36_TENSOR_F16) {
        const uint16_t *wr = (const uint16_t *)base + o * in_dim;
        for (uint32_t i = 0; i < in_dim; i++) dst[i] = p36_f16_to_f32(wr[i]);
    } else {
        const uint32_t nb = in_dim / P36_QK8_0;
        const p36_block_q8_0 *b = (const p36_block_q8_0 *)base + o * nb;
        for (uint32_t k = 0; k < nb; k++) {
            const float d = p36_f16_to_f32(b[k].d);
            for (uint32_t j = 0; j < P36_QK8_0; j++)
                dst[k * P36_QK8_0 + j] = d * (float)b[k].qs[j];
        }
    }
}

static void p36_vision_mm_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_mm_ctx *c = opaque;
    if (c->weight->type != P36_TENSOR_F16) {
        float *wrow = malloc((size_t)c->in_dim * sizeof(float));
        if (!wrow) p36_die("out of memory");
        for (uint64_t o = row0; o < row1; o++) {
            p36_vision_weight_row_f32(c->model, c->weight, c->in_dim, o, wrow);
            for (uint32_t r = 0; r < c->rows; r++) {
                const float *x = c->input + (uint64_t)r * c->in_dim;
                double sum = 0.0;
                for (uint32_t i = 0; i < c->in_dim; i++)
                    sum += (double)wrow[i] * x[i];
                float *dst = c->output + (uint64_t)r * c->out_dim + o;
                if (c->add) *dst += (float)sum;
                else *dst = (float)sum;
            }
        }
        free(wrow);
        return;
    }
    const uint16_t *weight = (const uint16_t *)(c->model->map + c->weight->abs_offset);
    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *wr = weight + o * c->in_dim;
        for (uint32_t r = 0; r < c->rows; r++) {
            const float *x = c->input + (uint64_t)r * c->in_dim;
            double sum = 0.0;
            for (uint32_t i = 0; i < c->in_dim; i++)
                sum += (double)p36_f16_to_f32(wr[i]) * x[i];
            float *dst = c->output + (uint64_t)r * c->out_dim + o;
            if (c->add) *dst += (float)sum;
            else *dst = (float)sum;
        }
    }
}

#if !defined(P36_NO_GPU) && !defined(P36_METAL)
typedef struct {
    const p36_model *model;
    const p36_tensor *weight;
    uint32_t in_dim;
    uint16_t *dst;
} p36_vision_fill_ctx;

static void p36_vision_fill_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_fill_ctx *c = opaque;
    const uint8_t *base = c->model->map + c->weight->abs_offset;
    for (uint64_t o = row0; o < row1; o++) {
        uint16_t *out = c->dst + o * c->in_dim;
        if (c->weight->type == P36_TENSOR_F32) {
            const float *src = (const float *)base + o * c->in_dim;
            for (uint32_t i = 0; i < c->in_dim; i++) out[i] = p36_f32_to_f16(src[i]);
        } else {
            const uint32_t nb = c->in_dim / P36_QK8_0;
            const p36_block_q8_0 *b = (const p36_block_q8_0 *)base + o * nb;
            for (uint32_t k = 0; k < nb; k++) {
                const float d = p36_f16_to_f32(b[k].d);
                for (uint32_t j = 0; j < P36_QK8_0; j++)
                    out[k * P36_QK8_0 + j] = p36_f32_to_f16(d * (float)b[k].qs[j]);
            }
        }
    }
}

typedef struct {
    p36_vision_fill_ctx fill;
    uint32_t out_dim;
    uint32_t n_threads;
} p36_vision_fill_job;

static int p36_vision_fill_cb(void *opaque, uint16_t *dst) {
    p36_vision_fill_job *j = opaque;
    j->fill.dst = dst;
    p36_parallel_for_rows(j->out_dim, 16, j->n_threads, p36_vision_fill_rows, &j->fill);
    return 1;
}
#endif

static bool p36_vision_mm(p36_engine *e, const p36_tensor *weight,
                          const float *input, float *output,
                          uint32_t rows, uint32_t in_dim, uint32_t out_dim,
                          bool add) {
    p36_model *m = &e->vision_model;
    if (!p36_vision_weight_type_ok(weight, in_dim) ||
        weight->elements != (uint64_t)in_dim * out_dim ||
        (weight->ndim == 2 &&
         (weight->dim[0] != in_dim || weight->dim[1] != out_dim)))
        return false;
    if (e->backend == P36_BACKEND_VULKAN) {
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
        uint64_t in_bytes = (uint64_t)rows * in_dim * sizeof(float);
        uint64_t out_bytes = (uint64_t)rows * out_dim * sizeof(float);
        p36_gpu_tensor *gx = p36_gpu_tensor_alloc_uninitialized(in_bytes);
        p36_gpu_tensor *gy = p36_gpu_tensor_alloc_uninitialized(out_bytes);
        float *tmp = add ? malloc((size_t)out_bytes) : output;
        p36_vision_fill_job fj = {{m, weight, in_dim, NULL}, out_dim, e->n_threads};
        int ok = gx && gy && tmp && p36_gpu_tensor_write(gx, 0, input, in_bytes) &&
            (weight->type == P36_TENSOR_F16
                 ? p36_gpu_vision_matmul_f16_disk(gy, m->fd, weight->abs_offset,
                                                  in_dim, out_dim, gx, rows)
                 : p36_gpu_vision_matmul_f16_fill(gy, p36_vision_fill_cb, &fj,
                                                  in_dim, out_dim, gx, rows)) &&
            p36_gpu_tensor_read(gy, 0, tmp, out_bytes);
        if (ok && add)
            for (uint64_t i = 0, n = (uint64_t)rows * out_dim; i < n; i++)
                output[i] += tmp[i];
        if (add) free(tmp);
        p36_gpu_tensor_free(gx);
        p36_gpu_tensor_free(gy);
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(m->fd, (off_t)weight->abs_offset, (off_t)weight->bytes,
                      POSIX_FADV_DONTNEED);
#endif
        if (ok) return true;
        return false;
#endif
    }
    p36_vision_mm_ctx c = {m, weight, input, output, rows, in_dim, out_dim, add};
    p36_parallel_for_rows(out_dim, 4, e->n_threads, p36_vision_mm_worker, &c);
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(m->fd, (off_t)weight->abs_offset, (off_t)weight->bytes,
                  POSIX_FADV_DONTNEED);
#endif
    return true;
}

static const float *p36_vision_f32(const p36_model *m, const p36_tensor *t,
                                   uint32_t n) {
    if (!t || t->type != P36_TENSOR_F32 || t->elements != n) return NULL;
    return (const float *)(m->map + t->abs_offset);
}

static void p36_vision_layer_norm(float *out, const float *x,
                                  const float *weight, const float *bias,
                                  uint32_t rows, uint32_t width) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x + (uint64_t)r * width;
        float *yr = out + (uint64_t)r * width;
        double mean = 0.0, variance = 0.0;
        for (uint32_t i = 0; i < width; i++) mean += xr[i];
        mean /= width;
        for (uint32_t i = 0; i < width; i++) {
            double d = xr[i] - mean;
            variance += d * d;
        }
        float inv = 1.0f / sqrtf((float)(variance / width) + 1.0e-6f);
        for (uint32_t i = 0; i < width; i++)
            yr[i] = (xr[i] - (float)mean) * inv * weight[i] + bias[i];
    }
}

static void p36_vision_bias_residual(float *x, const float *bias,
                                     const float *residual,
                                     uint32_t rows, uint32_t width) {
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++)
        x[i] += bias[i % width] + (residual ? residual[i] : 0.0f);
}

static void p36_vision_gelu_bias(float *x, const float *bias,
                                 uint32_t rows, uint32_t width) {
    const float k = 0.7071067811865475f;
    for (uint64_t i = 0, n = (uint64_t)rows * width; i < n; i++) {
        float v = x[i] + bias[i % width];
        x[i] = 0.5f * v * (1.0f + erff(v * k));
    }
}

typedef struct {
    const float *qkv;
    float *out;
    uint32_t rows;
} p36_vision_attn_ctx;

static void p36_vision_attention_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vision_attn_ctx *c = opaque;
    const float scale = 1.0f / sqrtf(72.0f);
    for (uint64_t job = row0; job < row1; job++) {
        uint32_t row = (uint32_t)(job / 16u), head = (uint32_t)(job % 16u);
        const float *q = c->qkv + (uint64_t)row * 3456u + head * 72u;
        float *dst = c->out + (uint64_t)row * 1152u + head * 72u;
        float max_score = -INFINITY, denom = 0.0f;
        memset(dst, 0, 72u * sizeof(*dst));
        for (uint32_t kr = 0; kr < c->rows; kr++) {
            const float *k = c->qkv + (uint64_t)kr * 3456u + 1152u + head * 72u;
            const float *v = c->qkv + (uint64_t)kr * 3456u + 2304u + head * 72u;
            float score = 0.0f;
            for (uint32_t d = 0; d < 72u; d++) score += q[d] * k[d];
            score *= scale;
            float next = fmaxf(max_score, score);
            float old_scale = isfinite(max_score) ? expf(max_score - next) : 0.0f;
            float new_scale = expf(score - next);
            denom = denom * old_scale + new_scale;
            for (uint32_t d = 0; d < 72u; d++)
                dst[d] = dst[d] * old_scale + new_scale * v[d];
            max_score = next;
        }
        for (uint32_t d = 0; d < 72u; d++) dst[d] /= denom;
    }
}

static bool p36_vision_attention(p36_engine *e, const float *qkv,
                                 float *out, uint32_t rows) {
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
    if (e->backend == P36_BACKEND_VULKAN) {
        uint64_t qkv_bytes = (uint64_t)rows * 3456u * sizeof(float);
        uint64_t out_bytes = (uint64_t)rows * 1152u * sizeof(float);
        p36_gpu_tensor *gqkv = p36_gpu_tensor_alloc_uninitialized(qkv_bytes);
        p36_gpu_tensor *gout = p36_gpu_tensor_alloc_uninitialized(out_bytes);
        int ok = gqkv && gout && p36_gpu_tensor_write(gqkv, 0, qkv, qkv_bytes) &&
            p36_gpu_vision_attention(gout, gqkv, rows) &&
            p36_gpu_tensor_read(gout, 0, out, out_bytes);
        p36_gpu_tensor_free(gqkv);
        p36_gpu_tensor_free(gout);
        return ok != 0;
    }
#endif
    p36_vision_attn_ctx ac = {qkv, out, rows};
    p36_parallel_for_rows((uint64_t)rows * 16u, 8, e->n_threads,
                          p36_vision_attention_worker, &ac);
    return true;
}

static void p36_vision_rope(float *qkv, uint32_t rows, uint32_t grid_width) {
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t group = r / 4u, within = r & 3u;
        uint32_t y = (group / (grid_width / 2u)) * 2u + within / 2u;
        uint32_t x = (group % (grid_width / 2u)) * 2u + within % 2u;
        for (uint32_t part = 0; part < 2u; part++) {
            for (uint32_t h = 0; h < 16u; h++) {
                float *v = qkv + (uint64_t)r * 3456u + part * 1152u + h * 72u;
                for (uint32_t p = 0; p < 36u; p++) {
                    uint32_t pos = p < 18u ? y : x;
                    uint32_t freq = p % 18u;
                    float angle = pos * powf(10000.0f, -(float)freq / 18.0f);
                    float cs = cosf(angle), sn = sinf(angle);
                    float a = v[p], b = v[p + 36u];
                    v[p] = a * cs - b * sn;
                    v[p + 36u] = b * cs + a * sn;
                }
            }
        }
    }
}

static p36_tensor *p36_vision_tensorf(const p36_model *m, uint32_t layer,
                                      const char *suffix) {
    char name[96];
    int n = snprintf(name, sizeof(name), "v.blk.%u.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return NULL;
    return model_find_tensor(m, name);
}

static int p36_vision_encode_image(p36_engine *e, const p36_image *image,
                                   p36_vision_embedding *out,
                                   char *err, size_t errlen) {
    p36_model *m = &e->vision_model;
    p36_image_patches patches = {0};
    float *x = NULL, *norm = NULL, *qkv = NULL, *work = NULL, *wide = NULL;
    int ok = 0;
    if (!e->vision_ready || !p36_image_preprocess_qwen3vl(
            &patches, image, 1024u, err, errlen)) return 0;
    uint32_t rows = patches.patch_count;
    uint32_t merged = rows / 4u;
    uint64_t base_values = (uint64_t)rows * 1152u;
    x = malloc(base_values * sizeof(*x));
    norm = malloc(base_values * sizeof(*norm));
    work = malloc(base_values * sizeof(*work));
    qkv = malloc((uint64_t)rows * 3456u * sizeof(*qkv));
    wide = malloc((uint64_t)rows * 4304u * sizeof(*wide));
    if (!x || !norm || !work || !qkv || !wide) {
        if (err && errlen) snprintf(err, errlen, "unable to allocate vision workspace");
        goto done;
    }

    p36_tensor *patch0 = model_find_tensor(m, "v.patch_embd.weight");
    p36_tensor *patch1 = model_find_tensor(m, "v.patch_embd.weight.1");
    p36_tensor *patch_bias_t = model_find_tensor(m, "v.patch_embd.bias");
    p36_tensor *position_t = model_find_tensor(m, "v.position_embd.weight");
    const float *patch_bias = p36_vision_f32(m, patch_bias_t, 1152u);
    const float *position = p36_vision_f32(m, position_t, 1152u * 2304u);
    if (!patch_bias || !position ||
        !p36_vision_mm(e, patch0, patches.patches, x, rows, 768u, 1152u, false) ||
        !p36_vision_mm(e, patch1, patches.patches, x, rows, 768u, 1152u, true)) {
        if (err && errlen) snprintf(err, errlen, "invalid Qwen3-VL patch tensors");
        goto done;
    }
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t group = r / 4u, within = r & 3u;
        uint32_t gy = (group / (patches.grid_width / 2u)) * 2u + within / 2u;
        uint32_t gx = (group % (patches.grid_width / 2u)) * 2u + within % 2u;
        float sy = patches.grid_height == 1 ? 0.0f : gy * 47.0f / (patches.grid_height - 1u);
        float sx = patches.grid_width == 1 ? 0.0f : gx * 47.0f / (patches.grid_width - 1u);
        uint32_t y0 = (uint32_t)floorf(sy), x0 = (uint32_t)floorf(sx);
        uint32_t y1 = y0 < 47u ? y0 + 1u : y0, x1 = x0 < 47u ? x0 + 1u : x0;
        float fy = sy - y0, fx = sx - x0;
        for (uint32_t d = 0; d < 1152u; d++) {
            float a = position[((uint64_t)y0 * 48u + x0) * 1152u + d];
            float b = position[((uint64_t)y0 * 48u + x1) * 1152u + d];
            float c = position[((uint64_t)y1 * 48u + x0) * 1152u + d];
            float z = position[((uint64_t)y1 * 48u + x1) * 1152u + d];
            x[(uint64_t)r * 1152u + d] += patch_bias[d] +
                (a + (b - a) * fx) * (1.0f - fy) + (c + (z - c) * fx) * fy;
        }
    }

    for (uint32_t il = 0; il < 27u; il++) {
        p36_tensor *ln1w = p36_vision_tensorf(m, il, "ln1.weight");
        p36_tensor *ln1b = p36_vision_tensorf(m, il, "ln1.bias");
        p36_tensor *qkvw = p36_vision_tensorf(m, il, "attn_qkv.weight");
        p36_tensor *qkvb = p36_vision_tensorf(m, il, "attn_qkv.bias");
        p36_tensor *outw = p36_vision_tensorf(m, il, "attn_out.weight");
        p36_tensor *outb = p36_vision_tensorf(m, il, "attn_out.bias");
        p36_tensor *ln2w = p36_vision_tensorf(m, il, "ln2.weight");
        p36_tensor *ln2b = p36_vision_tensorf(m, il, "ln2.bias");
        p36_tensor *upw = p36_vision_tensorf(m, il, "ffn_up.weight");
        p36_tensor *upb = p36_vision_tensorf(m, il, "ffn_up.bias");
        p36_tensor *downw = p36_vision_tensorf(m, il, "ffn_down.weight");
        p36_tensor *downb = p36_vision_tensorf(m, il, "ffn_down.bias");
        const float *ln1wp = p36_vision_f32(m, ln1w, 1152u);
        const float *ln1bp = p36_vision_f32(m, ln1b, 1152u);
        const float *qkvbp = p36_vision_f32(m, qkvb, 3456u);
        const float *outbp = p36_vision_f32(m, outb, 1152u);
        const float *ln2wp = p36_vision_f32(m, ln2w, 1152u);
        const float *ln2bp = p36_vision_f32(m, ln2b, 1152u);
        const float *upbp = p36_vision_f32(m, upb, 4304u);
        const float *downbp = p36_vision_f32(m, downb, 1152u);
        if (!ln1wp || !ln1bp || !qkvbp || !outbp || !ln2wp || !ln2bp ||
            !upbp || !downbp) {
            if (err && errlen) snprintf(err, errlen, "invalid vision block %u", il);
            goto done;
        }
        p36_vision_layer_norm(norm, x, ln1wp, ln1bp, rows, 1152u);
        if (!p36_vision_mm(e, qkvw, norm, qkv, rows, 1152u, 3456u, false)) goto bad_mm;
        p36_vision_bias_residual(qkv, qkvbp, NULL, rows, 3456u);
        p36_vision_rope(qkv, rows, patches.grid_width);
        if (!p36_vision_attention(e, qkv, work, rows)) goto bad_mm;
        if (!p36_vision_mm(e, outw, work, norm, rows, 1152u, 1152u, false)) goto bad_mm;
        p36_vision_bias_residual(norm, outbp, x, rows, 1152u);
        memcpy(x, norm, base_values * sizeof(*x));
        p36_vision_layer_norm(norm, x, ln2wp, ln2bp, rows, 1152u);
        if (!p36_vision_mm(e, upw, norm, wide, rows, 1152u, 4304u, false)) goto bad_mm;
        p36_vision_gelu_bias(wide, upbp, rows, 4304u);
        if (!p36_vision_mm(e, downw, wide, norm, rows, 4304u, 1152u, false)) goto bad_mm;
        p36_vision_bias_residual(norm, downbp, x, rows, 1152u);
        memcpy(x, norm, base_values * sizeof(*x));
        continue;
bad_mm:
        if (err && errlen) snprintf(err, errlen, "invalid vision matmul in block %u", il);
        goto done;
    }

    {
        const float *postw = p36_vision_f32(m, model_find_tensor(m, "v.post_ln.weight"), 1152u);
        const float *postb = p36_vision_f32(m, model_find_tensor(m, "v.post_ln.bias"), 1152u);
        p36_tensor *mm0 = model_find_tensor(m, "mm.0.weight");
        const float *mm0b = p36_vision_f32(m, model_find_tensor(m, "mm.0.bias"), 4608u);
        p36_tensor *mm2 = model_find_tensor(m, "mm.2.weight");
        const float *mm2b = p36_vision_f32(m, model_find_tensor(m, "mm.2.bias"), P36_N_EMBD);
        float *merged_in = qkv;
        if (!postw || !postb || !mm0b || !mm2b) {
            if (err && errlen) snprintf(err, errlen, "invalid vision merger tensors");
            goto done;
        }
        p36_vision_layer_norm(norm, x, postw, postb, rows, 1152u);
        memcpy(merged_in, norm, base_values * sizeof(*norm));
        if (!p36_vision_mm(e, mm0, merged_in, wide, merged, 4608u, 4608u, false)) goto merger_bad;
        p36_vision_gelu_bias(wide, mm0b, merged, 4608u);
        out->data = malloc((uint64_t)merged * P36_N_EMBD * sizeof(*out->data));
        if (!out->data) {
            if (err && errlen) snprintf(err, errlen, "unable to allocate vision embedding");
            goto done;
        }
        if (!p36_vision_mm(e, mm2, wide, out->data, merged, 4608u, P36_N_EMBD, false)) goto merger_bad;
        p36_vision_bias_residual(out->data, mm2b, NULL, merged, P36_N_EMBD);
        out->token_count = merged;
        out->grid_width = patches.grid_width / 2u;
        out->grid_height = patches.grid_height / 2u;
        out->width = image->width;
        out->height = image->height;
        memcpy(out->fingerprint, image->fingerprint, sizeof(out->fingerprint));
        ok = 1;
        goto done;
merger_bad:
        free(out->data);
        memset(out, 0, sizeof(*out));
        if (err && errlen) snprintf(err, errlen, "invalid vision merger matmul");
    }
done:
    free(x); free(norm); free(qkv); free(work); free(wide);
    p36_image_patches_free(&patches);
#ifdef MADV_DONTNEED
    madvise((void *)m->map, (size_t)m->file_size, MADV_DONTNEED);
#endif
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(m->fd, 0, (off_t)m->file_size, POSIX_FADV_DONTNEED);
#endif
    return ok;
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "p36: expected %s=%u, got %u\n", name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "p36: expected %s=%.9g, got %.9g\n", name, (double)expected, (double)got);
    exit(1);
}

static void config_expect_string(const char *name, p36_str got, const char *expected) {
    if (p36_streq(got, expected)) return;
    fprintf(stderr, "p36: expected %s=%s\n", name, expected);
    exit(1);
}

static void config_expect_u32_array(const p36_model *m, const char *key, const uint32_t *expected, uint64_t n) {
    p36_array_ref arr;
    p36_cursor c;
    if (!model_get_array(m, key, &arr) || (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "p36: required metadata array is missing: %s\n", key);
        exit(1);
    }
    if (arr.len != n) {
        fprintf(stderr, "p36: expected %s length=%" PRIu64 ", got %" PRIu64 "\n", key, n, arr.len);
        exit(1);
    }
    c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) p36_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) p36_die(c.error);
            if (v < 0) p36_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }
        if (got != expected[i]) {
            fprintf(stderr, "p36: unexpected %s[%" PRIu64 "]=%u, expected %u\n", key, i, got, expected[i]);
            exit(1);
        }
    }
}

static bool p36_layer_is_full_attention(uint32_t il) {
    return ((il + 1u) % P36_FULL_ATTENTION_INTERVAL) == 0;
}

static bool tensor_type_is_routed_gate_up(uint32_t type) {
    return type == P36_TENSOR_Q4_K || type == P36_TENSOR_Q3_K ||
           type == P36_TENSOR_IQ2_XXS || type == P36_TENSOR_IQ2_S || type == P36_TENSOR_IQ3_XXS ||
           type == P36_TENSOR_IQ3_S ||
           type == P36_TENSOR_Q8_0;
}

static bool tensor_type_is_routed_down(uint32_t type) {
    return type == P36_TENSOR_Q2_K || type == P36_TENSOR_Q3_K ||
           type == P36_TENSOR_Q4_K ||
           type == P36_TENSOR_Q5_K || type == P36_TENSOR_Q6_K ||
           type == P36_TENSOR_IQ2_S || type == P36_TENSOR_IQ3_S ||
           type == P36_TENSOR_IQ3_XXS ||
           type == P36_TENSOR_Q8_0;
}

static void tensor_expect_layout(const p36_tensor *t, uint32_t type, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
        exit(1);
    }
}

static void tensor_expect_plain_layout(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != P36_TENSOR_F32 && t->type != P36_TENSOR_F16) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected f32 or f16\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q8_0(const p36_tensor *t, uint32_t type,
                                         uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != P36_TENSOR_Q8_0) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q6_k_or_q8_0(const p36_tensor *t, uint32_t type,
                                                uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) p36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != P36_TENSOR_Q6_K && t->type != P36_TENSOR_Q8_0) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected %s, q6_k, or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_optional_plain(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (t) tensor_expect_plain_layout(t, ndim, d0, d1, d2);
}

static bool tensor_type_is_cpu_matrix(uint32_t type) {
    switch (type) {
    case P36_TENSOR_F32:
    case P36_TENSOR_F16:
    case P36_TENSOR_BF16:
    case P36_TENSOR_Q8_0:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return true;
    default:
        return false;
    }
}

static void tensor_expect_cpu_matrix(const p36_tensor *t,
                                     uint64_t in_dim, uint64_t out_dim) {
    if (!t) p36_die("internal error: missing matrix tensor while validating layout");
    if (!tensor_type_is_cpu_matrix(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has unsupported CPU matrix type %s\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, 2, in_dim, out_dim, 0);
}

static void tensor_expect_moe_matrix(const p36_tensor *t, bool mixed_quants,
                                     uint32_t baseline_type, bool allow_q6_k,
                                     uint64_t in_dim, uint64_t out_dim) {
    if (mixed_quants) {
        tensor_expect_cpu_matrix(t, in_dim, out_dim);
    } else if (allow_q6_k) {
        tensor_expect_layout_or_q6_k_or_q8_0(
            t, baseline_type, 2, in_dim, out_dim, 0);
    } else {
        tensor_expect_layout_or_q8_0(t, baseline_type, 2, in_dim, out_dim, 0);
    }
}

static void tensor_expect_routed_gate_up(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_gate_up(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected a routed gate/up expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static void tensor_expect_routed_down(const p36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) p36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_down(t->type)) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected a routed down expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "p36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "p36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static int p36_quant_bits_from_type(uint32_t type) {
    switch (type) {
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
        return 2;
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_Q3_K:
        return 3;
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ4_NL:
        return 4;
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ1_M:
        return 1;
    case P36_TENSOR_Q4_K:
        return 4;
    case P36_TENSOR_Q5_K:
        return 5;
    case P36_TENSOR_Q6_K:
        return 6;
    case P36_TENSOR_Q8_0:
        return 8;
    case P36_TENSOR_BF16:
        return 16;
    default:
        return 0;
    }
}

static void weights_validate_layout(const p36_weights *w, bool mixed_moe_quants) {
    if (!w) p36_die("internal error: missing weights while validating layout");
    if (P36_MODEL_DENSE) {
        tensor_expect_cpu_matrix(w->token_embd, P36_N_EMBD, P36_N_VOCAB);
        tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_cpu_matrix(w->output, P36_N_EMBD, P36_N_VOCAB);
        tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            const p36_layer_weights *l = &w->layer[il];
            tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
            tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
            if (l->kind == P36_LAYER_FULL_ATTN) {
                tensor_expect_cpu_matrix(l->attn_q, P36_N_EMBD, P36_N_SSM_INNER * 2u);
                tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
                tensor_expect_cpu_matrix(l->attn_k, P36_N_EMBD,
                                         (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM);
                tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
                tensor_expect_cpu_matrix(l->attn_v, P36_N_EMBD,
                                         (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM);
                tensor_expect_cpu_matrix(l->attn_output, P36_N_SSM_INNER, P36_N_EMBD);
                tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
            } else {
                tensor_expect_cpu_matrix(l->attn_gate, P36_N_EMBD, P36_N_SSM_INNER);
                tensor_expect_cpu_matrix(l->attn_qkv, P36_N_EMBD,
                                         P36_N_SSM_CONV_DIM);
                tensor_expect_layout(l->ssm_a, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
                tensor_expect_cpu_matrix(l->ssm_alpha, P36_N_EMBD, P36_N_SSM_DT_RANK);
                tensor_expect_cpu_matrix(l->ssm_beta, P36_N_EMBD, P36_N_SSM_DT_RANK);
                tensor_expect_layout(l->ssm_conv1d, P36_TENSOR_F32, 2,
                                     P36_N_SSM_CONV, P36_N_SSM_CONV_DIM, 0);
                tensor_expect_layout(l->ssm_dt, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
                tensor_expect_layout(l->ssm_norm, P36_TENSOR_F32, 1, P36_N_SSM_STATE, 0, 0);
                tensor_expect_cpu_matrix(l->ssm_out, P36_N_SSM_INNER, P36_N_EMBD);
            }
            tensor_expect_cpu_matrix(l->ffn_gate_shexp, P36_N_EMBD, P36_N_FF_SHARED);
            tensor_expect_cpu_matrix(l->ffn_up_shexp, P36_N_EMBD, P36_N_FF_SHARED);
            tensor_expect_cpu_matrix(l->ffn_down_shexp, P36_N_FF_SHARED, P36_N_EMBD);
        }
        return;
    }
    /* Upstream IQ2_M uses a mixed quant recipe for the 40-layer trunk.  All
     * of these matrix types are handled by the generic CPU/Vulkan dispatch. */
    tensor_expect_moe_matrix(w->token_embd, mixed_moe_quants,
                             P36_TENSOR_Q4_K, false, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_moe_matrix(w->output, mixed_moe_quants,
                             P36_TENSOR_Q4_K, true, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);

    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &w->layer[il];
        tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);

        if (l->kind == P36_LAYER_FULL_ATTN) {
            tensor_expect_moe_matrix(l->attn_q, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER * 2u);
            tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
            tensor_expect_moe_matrix(l->attn_k, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false, P36_N_EMBD,
                                     (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM);
            tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
            tensor_expect_moe_matrix(l->attn_v, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false, P36_N_EMBD,
                                     (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM);
            tensor_expect_moe_matrix(l->attn_output, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_SSM_INNER, P36_N_EMBD);
            tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
            tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);
        } else {
            tensor_expect_moe_matrix(l->attn_gate, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER);
            tensor_expect_moe_matrix(l->attn_qkv, mixed_moe_quants,
                                     P36_TENSOR_Q5_K, false,
                                     P36_N_EMBD, P36_N_SSM_INNER * 2u);
            tensor_expect_layout(l->ssm_a, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_moe_matrix(l->ssm_alpha, mixed_moe_quants,
                                     P36_TENSOR_F32, false,
                                     P36_N_EMBD, P36_N_SSM_DT_RANK);
            tensor_expect_moe_matrix(l->ssm_beta, mixed_moe_quants,
                                     P36_TENSOR_F32, false,
                                     P36_N_EMBD, P36_N_SSM_DT_RANK);
            tensor_expect_layout(l->ssm_conv1d, P36_TENSOR_F32, 2, P36_N_SSM_CONV, P36_N_SSM_INNER * 2u, 0);
            tensor_expect_layout(l->ssm_dt, P36_TENSOR_F32, 1, P36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_layout(l->ssm_norm, P36_TENSOR_F32, 1, P36_N_SSM_STATE, 0, 0);
            tensor_expect_moe_matrix(l->ssm_out, mixed_moe_quants,
                                     P36_TENSOR_Q6_K, false,
                                     P36_N_SSM_INNER, P36_N_EMBD);
            tensor_expect_optional_plain(l->attn_gate_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_qkv_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_alpha_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_beta_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_out_scale, 1, 1, 0, 0);
        }

        tensor_expect_layout_or_q8_0(l->ffn_gate_inp, P36_TENSOR_F32, 2, P36_N_EMBD, P36_N_EXPERT, 0);
        tensor_expect_layout(l->ffn_gate_inp_shexp, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
        tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_gate_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q5_K, false,
                                 P36_N_EMBD, P36_N_FF_SHARED);
        tensor_expect_routed_gate_up(l->ffn_up_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_up_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q5_K, false,
                                 P36_N_EMBD, P36_N_FF_SHARED);
        tensor_expect_routed_down(l->ffn_down_exps, 3, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT);
        tensor_expect_moe_matrix(l->ffn_down_shexp, mixed_moe_quants,
                                 P36_TENSOR_Q6_K, false,
                                 P36_N_FF_SHARED, P36_N_EMBD);
        tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, P36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
    }
}

static void weights_bind(p36_weights *w, const p36_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_scale = model_find_tensor(m, "output.scale");

    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_layer_weights *l = &w->layer[il];
        l->kind = p36_layer_is_full_attention(il) ? P36_LAYER_FULL_ATTN : P36_LAYER_RECURRENT;
        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->post_attention_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
        if (l->kind == P36_LAYER_FULL_ATTN) {
            l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
            l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
            l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
            l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
            l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
            l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
            l->attn_sinks = tensor_by_namef(m, "blk.%u.attn_sinks.weight", il);
            l->attn_q_scale = tensor_by_namef(m, "blk.%u.attn_q.scale", il);
            l->attn_k_scale = tensor_by_namef(m, "blk.%u.attn_k.scale", il);
            l->attn_v_scale = tensor_by_namef(m, "blk.%u.attn_v.scale", il);
            l->attn_output_scale = tensor_by_namef(m, "blk.%u.attn_output.scale", il);
        } else {
            l->attn_gate = required_tensorf(m, "blk.%u.attn_gate.weight", il);
            l->attn_qkv = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
            l->ssm_a = required_tensorf(m, "blk.%u.ssm_a", il);
            l->ssm_alpha = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
            l->ssm_beta = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
            l->ssm_conv1d = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
            l->ssm_dt = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
            l->ssm_norm = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
            l->ssm_out = required_tensorf(m, "blk.%u.ssm_out.weight", il);
            l->attn_gate_scale = tensor_by_namef(m, "blk.%u.attn_gate.scale", il);
            l->attn_qkv_scale = tensor_by_namef(m, "blk.%u.attn_qkv.scale", il);
            l->ssm_alpha_scale = tensor_by_namef(m, "blk.%u.ssm_alpha.scale", il);
            l->ssm_beta_scale = tensor_by_namef(m, "blk.%u.ssm_beta.scale", il);
            l->ssm_out_scale = tensor_by_namef(m, "blk.%u.ssm_out.scale", il);
        }
        if (P36_MODEL_DENSE) {
            l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
            l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up.weight", il);
            l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down.weight", il);
        } else {
            l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
            l->ffn_gate_inp_shexp = required_tensorf(m, "blk.%u.ffn_gate_inp_shexp.weight", il);
            l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
            l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
            l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
            l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
            l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
            l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
            l->ffn_gate_exps_scale = tensor_by_namef(m, "blk.%u.ffn_gate_exps.scale", il);
            l->ffn_gate_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_gate_shexp.scale", il);
            l->ffn_up_exps_scale = tensor_by_namef(m, "blk.%u.ffn_up_exps.scale", il);
            l->ffn_up_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_up_shexp.scale", il);
            l->ffn_down_exps_scale = tensor_by_namef(m, "blk.%u.ffn_down_exps.scale", il);
            l->ffn_down_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_down_shexp.scale", il);
        }
    }
    /* All non-routed matrix types admitted by tensor_expect_cpu_matrix have
     * CPU and generic GPU dispatches.  Quant recipes such as APEX vary those
     * tensors without carrying the 20-tensor embedded NextN block, so tensor
     * count is not a valid proxy for a mixed recipe. */
    weights_validate_layout(w, !P36_MODEL_DENSE);
}

#ifndef P36_NO_GPU
static void mtp_weights_validate_layout(const p36_mtp_weights *w) {
    const p36_layer_weights *l;
    if (!w) p36_die("internal error: missing MTP weights while validating layout");
    l = &w->block;
    tensor_expect_layout(w->token_embd, P36_TENSOR_Q8_0, 2, P36_N_EMBD, P36_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(w->output, P36_TENSOR_Q8_0, 2, P36_N_EMBD, P36_N_VOCAB, 0);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);
    tensor_expect_layout(w->eh_proj, P36_TENSOR_Q8_0, 2, P36_N_EMBD * 2u, P36_N_EMBD, 0);
    tensor_expect_layout(w->enorm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(w->shared_head_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);

    tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q, P36_TENSOR_Q8_0, 2, P36_N_EMBD, P36_N_SSM_INNER * 2u, 0);
    tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_k, P36_TENSOR_Q8_0, 2, P36_N_EMBD, (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_v, P36_TENSOR_Q8_0, 2, P36_N_EMBD, (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM, 0);
    tensor_expect_layout(l->attn_output, P36_TENSOR_Q8_0, 2, P36_N_SSM_INNER, P36_N_EMBD, 0);
    tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
    tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);

    tensor_expect_layout_or_q8_0(l->ffn_gate_inp, P36_TENSOR_F32, 2, P36_N_EMBD, P36_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_gate_inp_shexp, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_gate_shexp, P36_TENSOR_Q5_K, 2, P36_N_EMBD, P36_N_FF_SHARED, 0);
    tensor_expect_routed_gate_up(l->ffn_up_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_up_shexp, P36_TENSOR_Q5_K, 2, P36_N_EMBD, P36_N_FF_SHARED, 0);
    tensor_expect_routed_down(l->ffn_down_exps, 3, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_down_shexp, P36_TENSOR_Q6_K, 2, P36_N_FF_SHARED, P36_N_EMBD, 0);
    tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
}

static void mtp_weights_bind(p36_mtp_weights *w, const p36_model *m) {
    const uint32_t il = P36_N_LAYER;
    p36_layer_weights *l;
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_scale = model_find_tensor(m, "output.scale");
    w->eh_proj = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
    w->enorm = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
    w->hnorm = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
    w->shared_head_norm = required_tensorf(m, "blk.%u.nextn.shared_head_norm.weight", il);

    l = &w->block;
    l->kind = P36_LAYER_FULL_ATTN;
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->post_attention_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
    l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
    l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
    l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
    l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->attn_sinks = tensor_by_namef(m, "blk.%u.attn_sinks.weight", il);
    l->attn_q_scale = tensor_by_namef(m, "blk.%u.attn_q.scale", il);
    l->attn_k_scale = tensor_by_namef(m, "blk.%u.attn_k.scale", il);
    l->attn_v_scale = tensor_by_namef(m, "blk.%u.attn_v.scale", il);
    l->attn_output_scale = tensor_by_namef(m, "blk.%u.attn_output.scale", il);
    l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_gate_inp_shexp = required_tensorf(m, "blk.%u.ffn_gate_inp_shexp.weight", il);
    l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
    l->ffn_gate_exps_scale = tensor_by_namef(m, "blk.%u.ffn_gate_exps.scale", il);
    l->ffn_gate_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_gate_shexp.scale", il);
    l->ffn_up_exps_scale = tensor_by_namef(m, "blk.%u.ffn_up_exps.scale", il);
    l->ffn_up_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_up_shexp.scale", il);
    l->ffn_down_exps_scale = tensor_by_namef(m, "blk.%u.ffn_down_exps.scale", il);
    l->ffn_down_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_down_shexp.scale", il);
    mtp_weights_validate_layout(w);
}

static void mtp_weights_validate_dense_layout(const p36_mtp_weights *w) {
    const p36_layer_weights *l;
    if (!w) p36_die("internal error: missing MTP weights while validating dense layout");
    l = &w->block;
    tensor_expect_cpu_matrix(w->token_embd, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_layout(w->output_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_cpu_matrix(w->output, P36_N_EMBD, P36_N_VOCAB);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);
    tensor_expect_cpu_matrix(w->eh_proj, P36_N_EMBD * 2u, P36_N_EMBD);
    tensor_expect_layout(w->enorm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(w->shared_head_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);

    tensor_expect_layout(l->attn_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_layout(l->post_attention_norm, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_cpu_matrix(l->attn_q, P36_N_EMBD, P36_N_SSM_INNER * 2u);
    tensor_expect_layout(l->attn_q_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
    tensor_expect_cpu_matrix(l->attn_k, P36_N_EMBD, (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM);
    tensor_expect_layout(l->attn_k_norm, P36_TENSOR_F32, 1, P36_N_HEAD_DIM, 0, 0);
    tensor_expect_cpu_matrix(l->attn_v, P36_N_EMBD, (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM);
    tensor_expect_cpu_matrix(l->attn_output, P36_N_SSM_INNER, P36_N_EMBD);
    tensor_expect_optional_plain(l->attn_sinks, 1, P36_N_HEAD, 0, 0);
    tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);

    tensor_expect_cpu_matrix(l->ffn_gate_shexp, P36_N_EMBD, P36_N_FF_SHARED);
    tensor_expect_cpu_matrix(l->ffn_up_shexp, P36_N_EMBD, P36_N_FF_SHARED);
    tensor_expect_cpu_matrix(l->ffn_down_shexp, P36_N_FF_SHARED, P36_N_EMBD);
    tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
}

static void mtp_weights_bind_dense(p36_mtp_weights *w, const p36_model *m) {
    const uint32_t il = P36_N_LAYER;
    p36_layer_weights *l;
    memset(w, 0, sizeof(*w));
    w->token_embd = model_find_tensor(m, "token_embd.weight");
    w->output_norm = model_find_tensor(m, "output_norm.weight");
    w->output = model_find_tensor(m, "output.weight");
    w->output_scale = model_find_tensor(m, "output.scale");
    w->eh_proj = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
    w->enorm = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
    w->hnorm = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
    w->shared_head_norm = required_tensorf(m, "blk.%u.nextn.shared_head_norm.weight", il);

    l = &w->block;
    l->kind = P36_LAYER_FULL_ATTN;
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->post_attention_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
    l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
    l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
    l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
    l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->attn_sinks = tensor_by_namef(m, "blk.%u.attn_sinks.weight", il);
    l->attn_q_scale = tensor_by_namef(m, "blk.%u.attn_q.scale", il);
    l->attn_k_scale = tensor_by_namef(m, "blk.%u.attn_k.scale", il);
    l->attn_v_scale = tensor_by_namef(m, "blk.%u.attn_v.scale", il);
    l->attn_output_scale = tensor_by_namef(m, "blk.%u.attn_output.scale", il);
    l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
    l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up.weight", il);
    l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down.weight", il);
    l->ffn_gate_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_gate.scale", il);
    l->ffn_up_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_up.scale", il);
    l->ffn_down_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_down.scale", il);
    mtp_weights_validate_dense_layout(w);
}
#endif

#ifndef P36_NO_GPU
static bool p36_streaming_tensor_expert_bytes(const p36_tensor *t, uint64_t *bytes_out) {
    uint64_t row_bytes = 0;
    uint64_t expert_bytes = 0;
    if (bytes_out) *bytes_out = 0;
    if (!t || !bytes_out || t->ndim < 3 || t->dim[2] == 0) return false;
    if (!tensor_nbytes(t->type, t->dim[0], &row_bytes) || row_bytes == 0) return false;
    if (t->dim[1] > UINT64_MAX / row_bytes) return false;
    expert_bytes = t->dim[1] * row_bytes;
    if (expert_bytes == 0) return false;
    *bytes_out = expert_bytes;
    return true;
}

static bool p36_streaming_layer_routed_expert_bytes(const p36_layer_weights *layer,
                                                    uint64_t *gate_bytes,
                                                    uint64_t *up_bytes,
                                                    uint64_t *down_bytes,
                                                    uint64_t *total_bytes) {
    uint64_t gate = 0, up = 0, down = 0, total = 0;
    if (gate_bytes) *gate_bytes = 0;
    if (up_bytes) *up_bytes = 0;
    if (down_bytes) *down_bytes = 0;
    if (total_bytes) *total_bytes = 0;
    if (!layer ||
        !p36_streaming_tensor_expert_bytes(layer->ffn_gate_exps, &gate) ||
        !p36_streaming_tensor_expert_bytes(layer->ffn_up_exps, &up) ||
        !p36_streaming_tensor_expert_bytes(layer->ffn_down_exps, &down)) {
        return false;
    }
    if (gate > UINT64_MAX - up) return false;
    total = gate + up;
    if (total > UINT64_MAX - down) return false;
    total += down;
    if (gate_bytes) *gate_bytes = gate;
    if (up_bytes) *up_bytes = up;
    if (down_bytes) *down_bytes = down;
    if (total_bytes) *total_bytes = total;
    return total != 0;
}

static bool p36_streaming_routed_expert_layout(
        const p36_weights *weights,
        uint64_t *gate_bytes_out, uint64_t *up_bytes_out,
        uint64_t *down_bytes_out, uint64_t *per_expert_bytes_out) {
    uint64_t max_gate = 0, max_up = 0, max_down = 0, total = 0;
    if (gate_bytes_out) *gate_bytes_out = 0;
    if (up_bytes_out) *up_bytes_out = 0;
    if (down_bytes_out) *down_bytes_out = 0;
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!weights || !per_expert_bytes_out) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        uint64_t gate = 0, up = 0, down = 0;
        if (p36_streaming_layer_routed_expert_bytes(
                &weights->layer[il], &gate, &up, &down, NULL)) {
#ifdef P36_METAL
            if (gate > max_gate) max_gate = gate;
            if (up > max_up) max_up = up;
            if (down > max_down) max_down = down;
#else
            /* Preserve the Vulkan single-size-class cache ABI. */
            max_gate = gate;
            max_up = up;
            max_down = down;
            break;
#endif
        }
    }
    if (max_gate > UINT64_MAX - max_up) return false;
    total = max_gate + max_up;
    if (total > UINT64_MAX - max_down) return false;
    total += max_down;
    if (gate_bytes_out) *gate_bytes_out = max_gate;
    if (up_bytes_out) *up_bytes_out = max_up;
    if (down_bytes_out) *down_bytes_out = max_down;
    *per_expert_bytes_out = total;
    return total != 0;
}

static bool p36_streaming_routed_expert_bytes(const p36_weights *weights,
                                              uint64_t *per_expert_bytes_out) {
    return p36_streaming_routed_expert_layout(
        weights, NULL, NULL, NULL, per_expert_bytes_out);
}

static bool p36_weights_streaming_layer_experts_uniform(
        const p36_weights *weights, uint32_t il) {
#ifdef P36_METAL
    (void)weights;
    (void)il;
    /* Metal uses component-wise padded expert slots, so every routed
     * precision class is cache-compatible. */
    return true;
#else
    uint64_t base = 0;
    uint64_t bytes = 0;
    if (!weights || il >= P36_N_LAYER) return true;
    for (uint32_t base_il = 0; base_il < P36_N_LAYER; base_il++) {
        if (p36_streaming_layer_routed_expert_bytes(
                &weights->layer[base_il], NULL, NULL, NULL, &base))
            break;
    }
    if (base == 0 ||
        !p36_streaming_layer_routed_expert_bytes(
            &weights->layer[il], NULL, NULL, NULL, &bytes))
        return true;
    return bytes == base;
#endif
}

static bool p36_streaming_full_layer_budget(const p36_weights *weights,
                                            uint32_t layers,
                                            uint32_t *slots_out,
                                            uint64_t *bytes_out) {
    uint64_t base = 0, bytes = 0;
    if (slots_out) *slots_out = 0;
    if (bytes_out) *bytes_out = 0;
    if (!weights || !slots_out || !bytes_out || layers > P36_N_LAYER ||
        !p36_streaming_routed_expert_bytes(weights, &base) || base == 0) {
        return false;
    }
    for (uint32_t il = 0; il < layers; il++) {
        uint64_t per_expert = 0;
        if (!p36_streaming_layer_routed_expert_bytes(&weights->layer[il],
                                                      NULL, NULL, NULL,
                                                      &per_expert) ||
            per_expert > UINT64_MAX / P36_N_EXPERT) {
            return false;
        }
        uint64_t layer_bytes = per_expert * P36_N_EXPERT;
        if (bytes > UINT64_MAX - layer_bytes) return false;
        bytes += layer_bytes;
    }
    uint64_t slots = bytes / base + (bytes % base != 0);
    if (slots > UINT32_MAX) return false;
    *slots_out = (uint32_t)slots;
    *bytes_out = bytes;
    return true;
}
#endif


#ifndef P36_NO_GPU
static bool p36_weights_streaming_non_routed_bytes(const p36_weights *weights,
                                                   uint64_t *bytes_out) {
    uint64_t bytes = 0;
    if (bytes_out) *bytes_out = 0;
    if (!weights || !bytes_out) return false;
    const p36_tensor *skip[P36_N_LAYER * 3u];
    uint32_t n_skip = 0;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &weights->layer[il];
        skip[n_skip++] = l->ffn_gate_exps;
        skip[n_skip++] = l->ffn_up_exps;
        skip[n_skip++] = l->ffn_down_exps;
    }
    const p36_tensor *all[4 + P36_N_LAYER * 40u];
    uint32_t n = 0;
    all[n++] = weights->token_embd;
    all[n++] = weights->output_norm;
    all[n++] = weights->output;
    all[n++] = weights->output_scale;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &weights->layer[il];
        all[n++] = l->attn_norm;
        all[n++] = l->post_attention_norm;
        all[n++] = l->attn_q;
        all[n++] = l->attn_q_norm;
        all[n++] = l->attn_k;
        all[n++] = l->attn_k_norm;
        all[n++] = l->attn_v;
        all[n++] = l->attn_output;
        all[n++] = l->attn_sinks;
        all[n++] = l->attn_q_scale;
        all[n++] = l->attn_k_scale;
        all[n++] = l->attn_v_scale;
        all[n++] = l->attn_output_scale;
        all[n++] = l->attn_gate;
        all[n++] = l->attn_qkv;
        all[n++] = l->ssm_a;
        all[n++] = l->ssm_alpha;
        all[n++] = l->ssm_beta;
        all[n++] = l->ssm_conv1d;
        all[n++] = l->ssm_dt;
        all[n++] = l->ssm_norm;
        all[n++] = l->ssm_out;
        all[n++] = l->attn_gate_scale;
        all[n++] = l->attn_qkv_scale;
        all[n++] = l->ssm_alpha_scale;
        all[n++] = l->ssm_beta_scale;
        all[n++] = l->ssm_out_scale;
        all[n++] = l->ffn_gate_inp;
        all[n++] = l->ffn_gate_inp_shexp;
        all[n++] = l->ffn_gate_shexp;
        all[n++] = l->ffn_up_shexp;
        all[n++] = l->ffn_down_shexp;
        all[n++] = l->ffn_gate_exps_scale;
        all[n++] = l->ffn_gate_shexp_scale;
        all[n++] = l->ffn_up_exps_scale;
        all[n++] = l->ffn_up_shexp_scale;
        all[n++] = l->ffn_down_exps_scale;
        all[n++] = l->ffn_down_shexp_scale;
    }
    for (uint32_t i = 0; i < n; i++) {
        const p36_tensor *t = all[i];
        bool routed = false;
        if (!t) continue;
        for (uint32_t j = 0; j < n_skip; j++) {
            if (t == skip[j]) {
                routed = true;
                break;
            }
        }
        if (routed) continue;
        if (bytes > UINT64_MAX - t->bytes) return false;
        bytes += t->bytes;
    }
    *bytes_out = bytes;
    return true;
}
#endif

#ifndef P36_NO_GPU
static p36_gpu_stream_expert_table p36_stream_expert_table_make(const p36_model *model,
                                                                const p36_layer_weights *layer,
                                                                uint32_t il) {
    p36_gpu_stream_expert_table table;
    uint64_t gate = 0, up = 0, down = 0;
    memset(&table, 0, sizeof(table));
    if (!model || !layer ||
        !p36_streaming_layer_routed_expert_bytes(layer, &gate, &up, &down, NULL)) {
        return table;
    }
    table.model_map = model->map;
    table.model_size = model->size;
    table.layer = il;
    table.n_total_expert = P36_N_EXPERT;
    table.gate_offset = layer->ffn_gate_exps->abs_offset;
    table.up_offset = layer->ffn_up_exps->abs_offset;
    table.down_offset = layer->ffn_down_exps->abs_offset;
    table.gate_scales_offset = layer->ffn_gate_exps_scale ? layer->ffn_gate_exps_scale->abs_offset : 0;
    table.up_scales_offset = layer->ffn_up_exps_scale ? layer->ffn_up_exps_scale->abs_offset : 0;
    table.down_scales_offset = layer->ffn_down_exps_scale ? layer->ffn_down_exps_scale->abs_offset : 0;
    table.gate_expert_bytes = gate;
    table.up_expert_bytes = up;
    table.down_expert_bytes = down;
    table.gate_type = layer->ffn_gate_exps->type;
    table.up_type = layer->ffn_up_exps->type;
    table.down_type = layer->ffn_down_exps->type;
    table.has_gate_scales = layer->ffn_gate_exps_scale != NULL;
    table.has_up_scales = layer->ffn_up_exps_scale != NULL;
    table.has_down_scales = layer->ffn_down_exps_scale != NULL;
    return table;
}
#endif

#ifndef P36_NO_GPU
static uint64_t p36_streaming_model_limit(p36_engine *e, uint64_t contexts) {
    uint64_t recommended = p36_gpu_recommended_working_set_size();
    /* The remaining 20% belongs to the OS and driver. Reserve graph/KV state
     * explicitly, plus a bounded staging margin independent of context size. */
    uint64_t budget = recommended / 5 * 4;
    uint64_t reserve = 256ull * 1024 * 1024;
    if (e->mtp_ready) reserve += e->mtp_model.size;
    if (contexts >= budget || reserve >= budget - contexts) return 0;
    return budget - contexts - reserve;
}

static uint64_t p36_streaming_context_bytes(p36_engine *e, int ctx) {
    p36_context_memory m = p36_context_memory_estimate_configured(
        e->backend, ctx, p36_engine_gpu_prefill_cap(e), e->cache_type_k, e->cache_type_v);
    return m.total_bytes;
}
#endif

#ifndef P36_NO_GPU
static bool p36_engine_configure_streaming_auto_cache(p36_engine *e) {
    if (!e || !e->ssd_streaming || P36_MODEL_DENSE) return true;
    uint64_t recommended = p36_gpu_recommended_working_set_size();
    uint64_t fixed = 0, per_expert = 0;
    if (!recommended || !p36_weights_streaming_non_routed_bytes(&e->weights, &fixed) ||
        !p36_streaming_routed_expert_bytes(&e->weights, &per_expert)) return false;
    e->context_hint_bytes = p36_streaming_context_bytes(e, e->context_size);
    if (e->context_hint_bytes > UINT64_MAX / e->planned_sessions) return false;
    uint64_t contexts = e->context_hint_bytes * e->planned_sessions;
    e->streaming_model_limit = p36_streaming_model_limit(e, contexts);
    if (fixed >= e->streaming_model_limit ||
        per_expert > (e->streaming_model_limit - fixed) / P36_N_EXPERT_USED) {
        fprintf(stderr, "p36: SSD streaming context and static weights leave no expert-cache room\n");
        return false;
    }
    p36_ssd_cache_plan plan;
    if (!p36_ssd_auto_cache_plan(recommended, 80, e->streaming_model_limit, fixed,
                                 per_expert, (uint64_t)P36_N_LAYER * P36_N_EXPERT, &plan)) return false;
    uint64_t requested = e->ssd_streaming_cache_bytes ?
        e->ssd_streaming_cache_bytes / per_expert : e->ssd_streaming_cache_experts;
    uint32_t fitted = plan.cache_experts;
    if (requested && requested < fitted) fitted = (uint32_t)requested;
    if (!fitted || (e->ssd_streaming_cache_bytes && !requested)) {
        fprintf(stderr, "p36: SSD streaming cache target is smaller than one expert\n");
        return false;
    }
    e->ssd_streaming_cache_experts = fitted;
    if (e->ssd_streaming_cache_bytes) e->ssd_streaming_cache_bytes = (uint64_t)fitted * per_expert;
    fprintf(stderr, "p36: SSD streaming budget: working set %.2f GiB, contexts %.2f GiB "
                    "(%u x %d), model limit %.2f GiB, static %.2f GiB, cache %u experts (%.2f GiB)\n",
            recommended / 1073741824.0, contexts / 1073741824.0,
            e->planned_sessions, e->context_size, e->streaming_model_limit / 1073741824.0,
            fixed / 1073741824.0, fitted, fitted * per_expert / 1073741824.0);
    if (requested > fitted)
        fprintf(stderr, "p36: SSD cache target reduced from %" PRIu64 " to %u experts for runtime headroom\n", requested, fitted);
    return true;
}
#endif

#ifndef P36_NO_GPU
static bool p36_streaming_reserve_session(p36_engine *e, int ctx, uint64_t *bytes_out) {
    *bytes_out = 0;
    if (!e->ssd_streaming || P36_MODEL_DENSE) return true;
    uint64_t bytes = p36_streaming_context_bytes(e, ctx);
    if (bytes > UINT64_MAX - e->session_bytes) return false;
    uint64_t contexts = e->session_bytes + bytes;
    uint32_t remaining = e->planned_sessions > e->live_sessions + 1 ?
        e->planned_sessions - e->live_sessions - 1 : 0;
    if (remaining && e->context_hint_bytes > (UINT64_MAX - contexts) / remaining) return false;
    contexts += e->context_hint_bytes * remaining;
    uint64_t limit = p36_streaming_model_limit(e, contexts);
    uint64_t fixed = 0, resident = 0, per_expert = 0;
    uint32_t slots = 0;
    if (!p36_weights_streaming_non_routed_bytes(&e->weights, &fixed) ||
        !p36_streaming_routed_expert_bytes(&e->weights, &per_expert) ||
        (e->ssd_streaming_full_layers && !p36_streaming_full_layer_budget(
            &e->weights, e->ssd_streaming_full_layers, &slots, &resident))) return false;
    if (fixed >= limit || resident >= limit - fixed) return false;
    uint64_t fit = (limit - fixed - resident) / per_expert;
    if (fit < P36_N_EXPERT_USED) return false;
    if (fit < e->ssd_streaming_cache_experts) {
        fprintf(stderr, "p36: SSD cache reduced %u -> %" PRIu64 " experts for %u live sessions\n",
                e->ssd_streaming_cache_experts, fit, e->live_sessions + 1);
        e->ssd_streaming_cache_experts = (uint32_t)fit;
        p36_gpu_set_streaming_expert_cache_budget((uint32_t)fit);
    }
    *bytes_out = bytes;
    return true;
}
#endif

#ifndef P36_NO_GPU
static bool p36_streaming_hotlist_add(uint32_t layer,
                                      uint32_t expert,
                                      uint32_t priority,
                                      int32_t experts[P36_N_LAYER][P36_N_EXPERT],
                                      uint32_t priorities[P36_N_LAYER][P36_N_EXPERT],
                                      uint32_t counts[P36_N_LAYER],
                                      bool seen[P36_N_LAYER][P36_N_EXPERT],
                                      uint32_t *loaded) {
    if (layer >= P36_N_LAYER || expert >= P36_N_EXPERT) return true;
    if (seen[layer][expert]) return true;
    if (counts[layer] >= P36_N_EXPERT) return false;
    seen[layer][expert] = true;
    experts[layer][counts[layer]] = (int32_t)expert;
    priorities[layer][counts[layer]] = priority ? priority : 1u;
    counts[layer]++;
    (*loaded)++;
    return true;
}

static bool p36_streaming_hotlist_load_file(const char *path,
                                            uint32_t max_entries,
                                            int32_t experts[P36_N_LAYER][P36_N_EXPERT],
                                            uint32_t priorities[P36_N_LAYER][P36_N_EXPERT],
                                            uint32_t counts[P36_N_LAYER],
                                            bool seen[P36_N_LAYER][P36_N_EXPERT],
                                            uint32_t *loaded_out) {
    FILE *fp;
    char line[256];
    uint64_t lineno = 0;
    uint32_t loaded = 0;
    if (!path || !path[0] || max_entries == 0 || !loaded_out) return false;
    fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "p36: failed to open streaming expert hotlist %s: %s\n",
                path, strerror(errno));
        return false;
    }
    while (loaded < max_entries && fgets(line, sizeof(line), fp)) {
        char *p = line;
        char *end;
        unsigned long layer, expert;
        unsigned long long hits;
        lineno++;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        errno = 0;
        layer = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        expert = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        hits = strtoull(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        if (hits == 0) continue;
        if (!p36_streaming_hotlist_add((uint32_t)layer,
                                       (uint32_t)expert,
                                       hits > UINT32_MAX ? UINT32_MAX : (uint32_t)hits,
                                       experts,
                                       priorities,
                                       counts,
                                       seen,
                                       &loaded)) {
            goto bad;
        }
    }
    if (ferror(fp)) {
        fprintf(stderr, "p36: failed to read streaming expert hotlist %s: %s\n",
                path, strerror(errno));
        fclose(fp);
        return false;
    }
    fclose(fp);
    *loaded_out = loaded;
    return true;

bad:
    fprintf(stderr, "p36: invalid streaming expert hotlist line %" PRIu64 " in %s\n",
            lineno, path);
    fclose(fp);
    return false;
}

static bool p36_streaming_hotlist_load_default(uint32_t max_entries,
                                               int32_t experts[P36_N_LAYER][P36_N_EXPERT],
                                               uint32_t priorities[P36_N_LAYER][P36_N_EXPERT],
                                               uint32_t counts[P36_N_LAYER],
                                               bool seen[P36_N_LAYER][P36_N_EXPERT],
                                               uint32_t *loaded_out) {
    uint32_t loaded = 0;
    if (max_entries == 0 || !loaded_out) return false;
    for (uint32_t i = 0; i < p36_default_streaming_hotlist_count && loaded < max_entries; i++) {
        if (!p36_streaming_hotlist_add(p36_default_streaming_hotlist[i][0],
                                       p36_default_streaming_hotlist[i][1],
                                       p36_default_streaming_hotlist[i][2],
                                       experts,
                                       priorities,
                                       counts,
                                       seen,
                                       &loaded)) {
            return false;
        }
    }
    *loaded_out = loaded;
    return true;
}

static bool p36_engine_seed_streaming_expert_cache_blind(p36_engine *e,
                                                         uint32_t preload,
                                                         uint32_t *seeded_out) {
    uint32_t remaining = preload;
    uint32_t seeded = 0;
    for (uint32_t il = 0; il < P36_N_LAYER && remaining != 0; il++) {
        const p36_layer_weights *layer = &e->weights.layer[il];
        if (!p36_weights_streaming_layer_experts_uniform(&e->weights, il)) continue;
        p36_gpu_stream_expert_table table = p36_stream_expert_table_make(&e->model, layer, il);
        if (!table.model_map) continue;
        uint32_t n = remaining > P36_N_EXPERT ? P36_N_EXPERT : remaining;
        int32_t ids[P36_N_EXPERT];
        uint32_t prio[P36_N_EXPERT];
        for (uint32_t i = 0; i < n; i++) {
            ids[i] = (int32_t)i;
            prio[i] = n - i;
        }
        if (!p36_gpu_stream_expert_cache_seed_experts(&table, ids, prio, n)) {
            fprintf(stderr, "p36: SSD streaming failed to preload experts for layer %u\n", il);
            return false;
        }
        seeded += n;
        remaining -= n;
    }
    *seeded_out = seeded;
    return true;
}
#endif

#ifndef P36_NO_GPU
static bool p36_engine_seed_streaming_expert_cache(p36_engine *e) {
    if (!e || !e->ssd_streaming) return true;
    uint32_t budget = p36_gpu_stream_expert_cache_configured_count();
    uint32_t preload = e->ssd_streaming_preload_experts;
    const bool automatic = preload == 0;
    const bool bias_only = automatic || e->ssd_streaming_cold;
    uint32_t seeded = 0;
    uint32_t biased = 0;
    uint32_t loaded = 0;
    int32_t experts[P36_N_LAYER][P36_N_EXPERT];
    uint32_t priorities[P36_N_LAYER][P36_N_EXPERT];
    uint32_t counts[P36_N_LAYER];
    bool seen[P36_N_LAYER][P36_N_EXPERT];
    const char *path = getenv("P36_VK_STREAMING_EXPERT_HOTLIST");
    const bool disable_default = getenv("P36_VK_DISABLE_STREAMING_EXPERT_HOTLIST") != NULL;
    double t0 = p36_now_sec();

    if (automatic && (!path || !path[0]) &&
        (disable_default || p36_default_streaming_hotlist_count == 0)) {
        return true;
    }
    if (automatic) preload = budget < 4096u ? budget : 4096u;
    if (preload == 0) return true;
    if (budget != 0 && preload > budget) preload = budget;

    memset(experts, 0, sizeof(experts));
    memset(priorities, 0, sizeof(priorities));
    memset(counts, 0, sizeof(counts));
    memset(seen, 0, sizeof(seen));

    if (path && path[0]) {
        if (!p36_streaming_hotlist_load_file(path, preload, experts, priorities, counts, seen, &loaded)) {
            return false;
        }
    } else if (!disable_default) {
        if (!p36_streaming_hotlist_load_default(preload, experts, priorities, counts, seen, &loaded)) {
            return false;
        }
    }

    if (loaded == 0) {
        if (automatic || e->ssd_streaming_cold) return true;
        if (!p36_engine_seed_streaming_expert_cache_blind(e, preload, &seeded)) return false;
    } else {
        if (bias_only) {
            for (uint32_t il = 0; il < P36_N_LAYER; il++) {
                const p36_layer_weights *layer = &e->weights.layer[il];
                if (!layer->ffn_gate_exps || !layer->ffn_up_exps ||
                    !layer->ffn_down_exps) continue;
                if (!p36_weights_streaming_layer_experts_uniform(
                        &e->weights, il)) {
                    fprintf(stderr,
                            "p36: SSD streaming hotlist bias disabled for mixed expert sizes\n");
                    return true;
                }
            }
        }
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            uint32_t n = counts[il];
            if (n == 0) continue;
            const p36_layer_weights *layer = &e->weights.layer[il];
            if (!p36_weights_streaming_layer_experts_uniform(&e->weights, il)) continue;
            p36_gpu_stream_expert_table table = p36_stream_expert_table_make(&e->model, layer, il);
            if (!table.model_map) continue;
            int ok = bias_only ?
                p36_gpu_stream_expert_cache_bias_experts(
                    &table, experts[il], priorities[il], n) :
                p36_gpu_stream_expert_cache_seed_experts(
                    &table, experts[il], priorities[il], n);
            if (!ok) {
                fprintf(stderr, "p36: SSD streaming failed to apply hotlist experts for layer %u\n", il);
                return false;
            }
            if (bias_only) biased += n;
            else seeded += n;
        }
    }
    if (biased != 0) {
        fprintf(stderr,
                "p36: SSD streaming biased eviction with %u hotlist experts without preload\n",
                biased);
    }
    if (seeded != 0) {
        fprintf(stderr,
                "p36: SSD streaming preloaded %u expert slots%s in %.2fs\n",
                seeded,
                loaded ? " from hotlist" : "",
                p36_now_sec() - t0);
    }
    return true;
}
#endif

#ifndef P36_NO_GPU
static bool p36_vulkan_prewarm_enabled(void) {
    const char *env = getenv("P36_VK_PREWARM");
    return !env || !env[0] || env[0] != '0';
}

static int p36_weights_tensor_routed_layer(const p36_weights *w, const p36_tensor *t) {
    if (!w || !t) return -1;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &w->layer[il];
        if (t == l->ffn_gate_exps || t == l->ffn_up_exps || t == l->ffn_down_exps)
            return (int)il;
    }
    return -1;
}

static bool p36_tensor_is_disabled_embedded_mtp(const p36_engine *e,
                                                const p36_tensor *t) {
    char prefix[32];
    int len;
    if (!e || e->mtp_ready || !t) return false;
    len = snprintf(prefix, sizeof(prefix), "blk.%u.", P36_N_LAYER);
    return len > 0 && (uint64_t)len <= t->name.len &&
           memcmp(t->name.ptr, prefix, (size_t)len) == 0;
}

#ifdef P36_METAL
static bool p36_metal_q5_output_cache_candidate(const p36_engine *e,
                                                const p36_tensor *t) {
    const char *enabled = getenv("P36_METAL_Q5_OUTPUT_CACHE");
    return e && !e->ssd_streaming &&
           (!enabled || enabled[0] != '0') &&
           P36_MODEL_DENSE && t && t == e->weights.output &&
           t->type == P36_TENSOR_Q5_K;
}

#endif

static bool p36_vulkan_prewarm_skip_tensor(const p36_engine *e, const p36_tensor *t) {
    /* The main graph executes only the trunk; an unbound appended NextN block
     * must not consume the resident model arena. */
    if (t == e->weights.token_embd || t->bytes == 0 ||
        p36_tensor_is_disabled_embedded_mtp(e, t)) return true;
#ifdef P36_METAL
    if (p36_metal_q5_output_cache_candidate(e, t)) return true;
#endif
    int il = p36_weights_tensor_routed_layer(&e->weights, t);
    return e->ssd_streaming && il >= 0 && (uint32_t)il >= e->ssd_streaming_full_layers;
}

#ifdef P36_METAL
static void p36_streaming_lock_static_weights(p36_engine *e) {
    if (!e->ssd_streaming || getenv("P36_METAL_DISABLE_STREAMING_STATIC_LOCK")) return;
    uint64_t fixed = 0, per_expert = 0, resident = 0;
    uint32_t slots = 0;
    if (!p36_weights_streaming_non_routed_bytes(&e->weights, &fixed) ||
        !p36_streaming_routed_expert_bytes(&e->weights, &per_expert) ||
        (e->ssd_streaming_full_layers && !p36_streaming_full_layer_budget(
            &e->weights, e->ssd_streaming_full_layers, &slots, &resident))) return;
    uint64_t budget = e->streaming_model_limit;
    uint64_t experts = (uint64_t)e->ssd_streaming_cache_experts * per_expert;
    if (fixed > budget || resident > budget - fixed || experts > budget - fixed - resident) return;
    uint64_t pin_budget = budget - resident - experts;
    uint64_t locked = 0;
    size_t page = (size_t)getpagesize();
    bool can_lock = true;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const p36_tensor *t = &e->model.tensors[i];
        if (!t->bytes || p36_tensor_is_disabled_embedded_mtp(e, t) ||
            p36_weights_tensor_routed_layer(&e->weights, t) >= 0) continue;
        uint64_t off = t->abs_offset / page * page, end = t->abs_offset + t->bytes;
        if (end > e->model.size) continue;
        while (can_lock && off < end) {
            size_t len = end - off > 256ull * 1024 * 1024 ? 256u * 1024 * 1024 : (size_t)(end - off);
            uint64_t charged = ((uint64_t)len + page - 1) / page * page;
            if (charged > pin_budget - locked) break;
            if (mlock(e->model.map + off, len)) can_lock = false;
            else locked += charged;
            off += len;
        }
    }
    fprintf(stderr, "p36: Metal SSD static weights locked %.2f GiB within runtime budget\n", locked / 1073741824.0);
}
#endif

static bool p36_vulkan_set_model_spans(const p36_engine *e) {
    uint64_t *offsets = malloc(e->model.n_tensors * sizeof(*offsets));
    uint64_t *sizes = malloc(e->model.n_tensors * sizeof(*sizes));
    uint64_t max_bytes = 0;
    uint32_t count = 0;
    bool ok;

    if (!offsets || !sizes) {
        free(sizes);
        free(offsets);
        return false;
    }
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const p36_tensor *t = &e->model.tensors[i];
        if (p36_vulkan_prewarm_skip_tensor(e, t)) continue;
        offsets[count] = t->abs_offset;
        sizes[count] = t->bytes;
        if (t->bytes > max_bytes) max_bytes = t->bytes;
        count++;
    }
    ok = p36_gpu_set_model_map_spans(e->model.map, e->model.size,
                                      offsets, sizes, count, max_bytes) != 0;
    free(sizes);
    free(offsets);
    return ok;
}

/* Prewarm reader: fills the page cache with pread() a bounded distance ahead
 * of the staging memcpy, so SSD reads overlap the copy into GPU memory.  The
 * bound keeps read-ahead pages from being evicted again before staging on
 * unified-memory boxes where the GPU heap and the page cache share RAM. */
typedef struct {
    const p36_engine *e;
    uint64_t staged;            /* written by the stager, read by the reader */
} p36_prewarm_reader;

static void *p36_prewarm_reader_main(void *arg) {
    p36_prewarm_reader *r = arg;
    const p36_engine *e = r->e;
    const uint64_t window = 512ull * 1024 * 1024;
    const size_t chunk = 4u * 1024 * 1024;
    char *buf = malloc(chunk);
    uint64_t read_bytes = 0;
    if (!buf) return NULL;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const p36_tensor *t = &e->model.tensors[i];
        if (p36_vulkan_prewarm_skip_tensor(e, t)) continue;
        for (uint64_t off = 0; off < t->bytes; off += chunk) {
            uint64_t staged;
            while ((staged = __atomic_load_n(&r->staged, __ATOMIC_RELAXED)) < read_bytes &&
                   read_bytes - staged > window) {
                usleep(2000);
            }
            size_t n = t->bytes - off < chunk ? (size_t)(t->bytes - off) : chunk;
            if (pread(e->model.fd, buf, n, (off_t)(t->abs_offset + off)) < 0) break;
            read_bytes += n;
        }
    }
    free(buf);
    return NULL;
}

static void p36_vulkan_prewarm_weights(const p36_engine *e) {
    if (!e || !p36_vulkan_prewarm_enabled()) return;
    p36_prewarm_reader reader = { .e = e, .staged = 0 };
    pthread_t reader_thread;
    bool threaded = e->model.fd >= 0 &&
                    pthread_create(&reader_thread, NULL, p36_prewarm_reader_main, &reader) == 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const p36_tensor *t = &e->model.tensors[i];
        if (p36_vulkan_prewarm_skip_tensor(e, t)) continue;
        (void)p36_gpu_cache_model_range(e->model.map, e->model.size, t->abs_offset, t->bytes, NULL);
        __atomic_add_fetch(&reader.staged, t->bytes, __ATOMIC_RELAXED);
    }
    if (threaded) {
        __atomic_store_n(&reader.staged, UINT64_MAX, __ATOMIC_RELAXED);
        pthread_join(reader_thread, NULL);
    }
#ifdef P36_METAL
    if (p36_metal_q5_output_cache_candidate(e, e->weights.output) &&
        !p36_gpu_prepare_q5_k_output_cache(
            e->model.map, e->model.size, e->weights.output->abs_offset,
            P36_N_EMBD, P36_N_VOCAB)) {
        fprintf(stderr,
                "p36: Metal Q5_K output cache unavailable; using GGUF weights\n");
    }
#endif
    /* The MTP support model is fetched through the same whole-tensor cache
     * keys at draft time; without a prewarm the first drafts pay its whole
     * upload (~1.2s spread over the first replies). */
    if (e->mtp_ready) {
        for (uint64_t i = 0; i < e->mtp_model.n_tensors; i++) {
            const p36_tensor *t = &e->mtp_model.tensors[i];
            if (t == e->mtp_weights.token_embd || t->bytes == 0) continue;
            (void)p36_gpu_cache_model_range(e->mtp_model.map, e->mtp_model.size, t->abs_offset, t->bytes, NULL);
        }
    }
}
#endif

#define P36_HADAMARD_MAX_WIDTHS 16

typedef struct {
    bool enabled;
    uint32_t block_size;
    float inv_sqrt_block;
    uint32_t n_widths;
    uint32_t widths[P36_HADAMARD_MAX_WIDTHS];
    float *signs[P36_HADAMARD_MAX_WIDTHS];
} p36_hadamard;

static p36_hadamard g_p36_hadamard;

static int p36_hadamard_width_index(uint32_t width) {
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        if (g_p36_hadamard.widths[i] == width) return (int)i;
    }
    return -1;
}

static const float *p36_hadamard_signs(uint32_t width) {
    int i = p36_hadamard_width_index(width);
    return i < 0 ? NULL : g_p36_hadamard.signs[i];
}

static P36_MAYBE_UNUSED bool p36_hadamard_rotate_enabled(const p36_tensor *t) {
    return g_p36_hadamard.enabled && t && t->hadamard_rotate;
}

/* Unnormalized in-place Sylvester Walsh-Hadamard butterfly of length n. */
static void p36_hadamard_fwht_host(float *x, uint32_t n) {
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += 2u * len) {
            for (uint32_t j = 0; j < len; j++) {
                float a = x[i + j];
                float b = x[i + len + j];
                x[i + j] = a + b;
                x[i + len + j] = a - b;
            }
        }
    }
}

/* Inverse-after-lookup for a rotated embedding row: h = s . (H z). */
static void p36_hadamard_inverse_host(float *x, uint32_t width) {
    const float *signs;
    uint32_t block;
    if (!g_p36_hadamard.enabled || width % g_p36_hadamard.block_size != 0) return;
    signs = p36_hadamard_signs(width);
    if (!signs) return;
    block = g_p36_hadamard.block_size;
    for (uint32_t b = 0; b < width; b += block) {
        float *blk = x + b;
        p36_hadamard_fwht_host(blk, block);
        for (uint32_t i = 0; i < block; i++) {
            blk[i] *= g_p36_hadamard.inv_sqrt_block * signs[b + i];
        }
    }
}

/* Forward activation transform for a folded weight: x' = H (s . x) / sqrt(block).
 * Order matters -- signs and scale are applied before the butterfly, matching
 * vulkan/hadamard_prepare.comp. The inverse embedding above applies them after;
 * the two are not interchangeable. */
static void p36_hadamard_forward_host(float *x, uint32_t width) {
    const float *signs;
    uint32_t block;
    if (!g_p36_hadamard.enabled || width % g_p36_hadamard.block_size != 0) return;
    signs = p36_hadamard_signs(width);
    if (!signs) return;
    block = g_p36_hadamard.block_size;
    for (uint32_t b = 0; b < width; b += block) {
        float *blk = x + b;
        for (uint32_t i = 0; i < block; i++)
            blk[i] *= g_p36_hadamard.inv_sqrt_block * signs[b + i];
        p36_hadamard_fwht_host(blk, block);
    }
}

/* Forward transform for the folded GDN ssm_out weight: permute tiled -> grouped
 * first, then the signs+FWHT. Mirrors the is_grouped path of the Vulkan kernel
 * (width = n_ssm_inner, hd = n_ssm_state, nk = n_ssm_group, rep = dt_rank/group). */
static bool p36_hadamard_forward_grouped_host(float *x, uint32_t width) {
    uint32_t hd = P36_N_SSM_STATE, nk = P36_N_SSM_GROUP, rep;
    float *tmp;
    if (!g_p36_hadamard.enabled || hd == 0 || nk == 0 ||
        width % hd != 0 || (width / hd) % nk != 0) return false;
    rep = width / hd / nk;
    tmp = xmalloc((size_t)width * sizeof(float));
    for (uint32_t col = 0; col < width; col++) {
        uint32_t hd_idx = col % hd;
        uint32_t rep_idx = (col / hd) % rep;
        uint32_t nk_idx = col / (hd * rep);
        tmp[col] = x[hd_idx + hd * (nk_idx + nk * rep_idx)];
    }
    memcpy(x, tmp, (size_t)width * sizeof(float));
    free(tmp);
    p36_hadamard_forward_host(x, width);
    return true;
}

/* Copy n_rows activations of `in_dim` into `dst` applying the forward transform
 * required by `t`. Returns dst (or x when no transform applies). */
static const float *p36_hadamard_forward_rows(const p36_tensor *t, const float *x,
                                              float *dst, uint32_t in_dim, uint32_t n_rows) {
    if (!p36_hadamard_rotate_enabled(t)) return x;
    for (uint32_t r = 0; r < n_rows; r++) {
        float *d = dst + (uint64_t)r * in_dim;
        memcpy(d, x + (uint64_t)r * in_dim, (size_t)in_dim * sizeof(float));
        if (t->hadamard_grouped) {
            if (!p36_hadamard_forward_grouped_host(d, in_dim)) return x;
        } else {
            p36_hadamard_forward_host(d, in_dim);
        }
    }
    return dst;
}

static const char *p36_hadamard_str(const p36_str s, char *tmp, size_t tmp_size) {
    if (s.len + 1 > tmp_size) p36_die("prism.hadamard metadata name is too long");
    memcpy(tmp, s.ptr, s.len);
    tmp[s.len] = '\0';
    return tmp;
}

/* Parse the prism.hadamard.* GGUF block that marks weights stored in a
 * rotated (normalized Walsh-Hadamard + explicit sign) basis. The transform is
 * applied to the activation before the matmul, so the flag rides on the tensor
 * and the sign vectors are looked up by activation width. */
static void hadamard_parse(p36_model *m) {
    p36_str transform, axis, sign_mode;
    p36_array_ref widths_arr, values_arr, names_arr;
    p36_cursor c;
    uint32_t block_size = 0;
    uint64_t total = 0;
    char tmp[256];

    if (!model_find_kv(m, "prism.hadamard.weight_names")) return;

    if (!model_get_u32(m, "prism.hadamard.block_size", &block_size) || block_size == 0 ||
        (block_size & (block_size - 1u)) != 0) {
        p36_die("prism.hadamard.block_size is missing or not a power of two");
    }
    transform = required_string(m, "prism.hadamard.transform");
    if (!p36_streq(transform, "normalized-sylvester-walsh-hadamard")) {
        p36_die("unsupported prism.hadamard.transform");
    }
    axis = required_string(m, "prism.hadamard.axis");
    if (!p36_streq(axis, "input-last-dimension")) {
        p36_die("unsupported prism.hadamard.axis");
    }
    sign_mode = required_string(m, "prism.hadamard.sign_mode");
    if (!p36_streq(sign_mode, "explicit")) {
        p36_die("unsupported prism.hadamard.sign_mode");
    }

    g_p36_hadamard.enabled = true;
    g_p36_hadamard.block_size = block_size;
    g_p36_hadamard.inv_sqrt_block = 1.0f / sqrtf((float)block_size);

    if (!model_get_array(m, "prism.hadamard.sign_widths", &widths_arr) ||
        widths_arr.type != GGUF_VALUE_INT32 ||
        !model_get_array(m, "prism.hadamard.sign_values", &values_arr) ||
        values_arr.type != GGUF_VALUE_INT32) {
        p36_die("prism.hadamard explicit signs are missing");
    }
    if (widths_arr.len == 0 || widths_arr.len > P36_HADAMARD_MAX_WIDTHS) {
        p36_die("prism.hadamard.sign_widths has an unsupported length");
    }
    g_p36_hadamard.n_widths = (uint32_t)widths_arr.len;
    c = cursor_at(m, widths_arr.data_pos);
    for (uint64_t i = 0; i < widths_arr.len; i++) {
        int32_t w = 0;
        if (!cursor_read(&c, &w, sizeof(w))) p36_die(c.error);
        if (w <= 0 || (uint32_t)w % block_size != 0) p36_die("prism.hadamard sign width is invalid");
        g_p36_hadamard.widths[i] = (uint32_t)w;
        total += (uint64_t)w;
    }
    if (total != values_arr.len) p36_die("prism.hadamard sign_values length mismatch");
    c = cursor_at(m, values_arr.data_pos);
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        uint32_t w = g_p36_hadamard.widths[i];
        float *buf = (float *)xmalloc((size_t)w * sizeof(float));
        for (uint32_t j = 0; j < w; j++) {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) p36_die(c.error);
            if (v != 1 && v != -1) p36_die("prism.hadamard sign value must be +/-1");
            buf[j] = (float)v;
        }
        g_p36_hadamard.signs[i] = buf;
    }

    if (!model_get_array(m, "prism.hadamard.weight_names", &names_arr) ||
        names_arr.type != GGUF_VALUE_STRING) {
        p36_die("prism.hadamard.weight_names is missing");
    }
    c = cursor_at(m, names_arr.data_pos);
    for (uint64_t i = 0; i < names_arr.len; i++) {
        p36_str name;
        p36_tensor *t;
        if (!cursor_string(&c, &name)) p36_die(c.error);
        t = model_find_tensor(m, p36_hadamard_str(name, tmp, sizeof(tmp)));
        if (!t) {
            fprintf(stderr, "p36: prism.hadamard weight not found: %s\n", tmp);
            exit(1);
        }
        if ((t->dim[0] % block_size) != 0) {
            fprintf(stderr, "p36: prism.hadamard block size does not divide input dim of %s\n", tmp);
            exit(1);
        }
        t->hadamard_rotate = true;
        /* The GDN ssm_out projection kept the training V order
         * (prism.hadamard.gdn_v_grouped); its activation is permuted
         * tiled -> grouped before the transform, exactly as the Vulkan
         * hadamard_prepare is_grouped path does. */
        if (strstr(tmp, "ssm_out") != NULL) t->hadamard_grouped = true;
    }

    if (model_get_array(m, "prism.hadamard.inverse_weight_names", &names_arr) &&
        names_arr.type == GGUF_VALUE_STRING) {
        c = cursor_at(m, names_arr.data_pos);
        for (uint64_t i = 0; i < names_arr.len; i++) {
            p36_str name;
            p36_tensor *t;
            if (!cursor_string(&c, &name)) p36_die(c.error);
            p36_hadamard_str(name, tmp, sizeof(tmp));
            if (strcmp(tmp, "token_embd.weight") != 0) {
                fprintf(stderr, "p36: unsupported prism.hadamard inverse weight: %s\n", tmp);
                exit(1);
            }
            t = model_find_tensor(m, tmp);
            if (!t) {
                fprintf(stderr, "p36: prism.hadamard inverse weight not found: %s\n", tmp);
                exit(1);
            }
            t->hadamard_inverse = true;
        }
    }

    fprintf(stderr, "p36: prism.hadamard: block=%u widths=%u values=%" PRIu64 "\n",
            block_size, g_p36_hadamard.n_widths, total);
}

static void config_validate_model(const p36_model *m) {
    char key[128];
    const char *prefix;
    uint64_t ctx_train = 0;
    uint32_t block_count;
    uint32_t nextn_layers = 0;
    bool embedded_nextn_block;
    p36_str arch = required_string(m, "general.architecture");
    p36_str tok_model = required_string(m, "tokenizer.ggml.model");
    p36_str tok_pre = required_string(m, "tokenizer.ggml.pre");
    if (p36_streq(arch, P36_SHAPE_35B_A3B.arch)) {
        g_p36_shape = P36_SHAPE_35B_A3B;
    } else if (p36_streq(arch, P36_SHAPE_27B.arch)) {
        g_p36_shape = P36_SHAPE_27B;
    } else {
        fprintf(stderr, "p36: unsupported architecture: %.*s\n",
                (int)arch.len, arch.ptr);
        exit(1);
    }
    const p36_tensor *embd = model_find_tensor(m, "token_embd.weight");
    if (embd && embd->ndim == 2 && embd->dim[1] > 0) {
        g_p36_shape.n_vocab = (uint32_t)embd->dim[1];
    }
    prefix = g_p36_shape.arch;
    config_expect_string("general.architecture", arch, prefix);
    config_expect_string("tokenizer.ggml.model", tok_model, "gpt2");
    /* Some converters label the identical Qwen3.5 vocab/merges "qwen2"; the
     * engine's pre-tokenizer is the qwen35 one either way. */
    if (!p36_streq(tok_pre, "qwen2"))
        config_expect_string("tokenizer.ggml.pre", tok_pre, "qwen35");
#define P36_META_KEY(suffix) snprintf(key, sizeof(key), "%s.%s", prefix, suffix)
#define P36_EXPECT_U32(suffix, expected) do { \
    P36_META_KEY(suffix); \
    config_expect_u32(key, required_u32(m, key), expected); \
} while (0)
#define P36_EXPECT_F32(suffix, expected) do { \
    P36_META_KEY(suffix); \
    config_expect_f32(key, required_f32(m, key), expected); \
} while (0)
    P36_META_KEY("block_count");
    block_count = required_u32(m, key);
    P36_META_KEY("nextn_predict_layers");
    model_get_u32(m, key, &nextn_layers);
    embedded_nextn_block = nextn_layers == 1 && block_count == P36_N_LAYER + 1;
    if (block_count != P36_N_LAYER && !embedded_nextn_block) {
        P36_META_KEY("block_count");
        config_expect_u32(key, block_count, P36_N_LAYER);
    }
    P36_META_KEY("context_length");
    if (!model_get_u64_compat(m, key, &ctx_train)) {
        fprintf(stderr, "p36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    if (ctx_train != P36_CONTEXT_TRAIN) {
        fprintf(stderr, "p36: expected %s=%u, got %" PRIu64 "\n",
                key, P36_CONTEXT_TRAIN, ctx_train);
        exit(1);
    }
    P36_EXPECT_U32("embedding_length", P36_N_EMBD);
    P36_EXPECT_U32("attention.head_count", P36_N_HEAD);
    P36_EXPECT_U32("attention.head_count_kv", P36_N_HEAD_KV);
    P36_EXPECT_U32("attention.key_length", P36_N_HEAD_DIM);
    P36_EXPECT_U32("attention.value_length", P36_N_VALUE_DIM);
    P36_EXPECT_U32("rope.dimension_count", P36_N_ROT);
    P36_EXPECT_F32("rope.freq_base", 10000000.0f);
    P36_META_KEY("rope.dimension_sections");
    config_expect_u32_array(m, key, P36_ROPE_SECTIONS, 4);
    P36_EXPECT_F32("attention.layer_norm_rms_epsilon", P36_RMS_EPS);
    if (P36_MODEL_DENSE) {
        P36_EXPECT_U32("feed_forward_length", P36_N_FF_SHARED);
    } else {
        P36_EXPECT_U32("expert_count", P36_N_EXPERT);
        P36_EXPECT_U32("expert_used_count", P36_N_EXPERT_USED);
        P36_EXPECT_U32("expert_feed_forward_length", P36_N_FF_EXP);
        P36_EXPECT_U32("expert_shared_feed_forward_length", P36_N_FF_SHARED);
    }
    P36_EXPECT_U32("ssm.conv_kernel", P36_N_SSM_CONV);
    P36_EXPECT_U32("ssm.state_size", P36_N_SSM_STATE);
    P36_EXPECT_U32("ssm.group_count", P36_N_SSM_GROUP);
    P36_EXPECT_U32("ssm.time_step_rank", P36_N_SSM_DT_RANK);
    P36_EXPECT_U32("ssm.inner_size", P36_N_SSM_INNER);
    P36_EXPECT_U32("full_attention_interval", P36_FULL_ATTENTION_INTERVAL);
#undef P36_EXPECT_F32
#undef P36_EXPECT_U32
#undef P36_META_KEY
    if (m->n_tensors != P36_TENSOR_COUNT &&
        !(embedded_nextn_block &&
          m->n_tensors == P36_TENSOR_COUNT + (P36_MODEL_DENSE ? 15u : 20u))) {
        fprintf(stderr, "p36: expected %u tensors, got %" PRIu64 "\n", P36_TENSOR_COUNT, m->n_tensors);
        exit(1);
    }
}

static void print_size(uint64_t bytes) {
    printf("%.2f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static void model_summary(const p36_model *m) {
    p36_str name = {0};
    p36_str arch = {0};
    uint64_t ctx_train = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;
    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    if (!model_get_u64_compat(m, "qwen35moe.context_length", &ctx_train))
        model_get_u64_compat(m, "qwen35.context_length", &ctx_train);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }
    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n", m->version, m->n_kv, m->n_tensors);
    printf("layers: %u\n", P36_N_LAYER);
    printf("train context: %" PRIu64 "\n", ctx_train);
    printf("attention: heads=%u kv_heads=%u head_dim=%u full_interval=%u\n",
           P36_N_HEAD, P36_N_HEAD_KV, P36_N_HEAD_DIM, P36_FULL_ATTENTION_INTERVAL);
    if (P36_MODEL_DENSE) printf("ffn: dense, width=%u\n", P36_N_FF_SHARED);
    else printf("experts: count=%u used=%u ff=%u shared_ff=%u\n",
                P36_N_EXPERT, P36_N_EXPERT_USED, P36_N_FF_EXP, P36_N_FF_SHARED);
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);
}

enum { P36_KV_INITIAL_CAP = 32768 };

static uint32_t p36_kv_initial_cap(int ctx_size) {
    return (uint32_t)(ctx_size < P36_KV_INITIAL_CAP ? ctx_size : P36_KV_INITIAL_CAP);
}

static uint32_t p36_kv_next_cap(uint32_t cap, uint32_t need, uint32_t limit) {
    uint32_t next = cap;
    while (next < need) {
        if (next >= limit) return 0;
        uint32_t step = next / 4u;
        uint32_t grown = step <= limit - next ? next + step : limit;
        if (grown <= next) return 0;
        next = grown;
    }
    return next;
}

static p36_cpu_runtime *p36_cpu_runtime_create(int ctx_size,
                                               uint32_t prefill_cap,
                                               p36_kv_cache_type cache_type_k,
                                               p36_kv_cache_type cache_type_v) {
    p36_cpu_runtime *rt;
    uint32_t full_cap;
    uint64_t state_dim;
    size_t scratch_bytes;
    if (ctx_size <= 0) return NULL;
    if (prefill_cap < 1) prefill_cap = 1;
    uint32_t k_row_bytes = p36_kv_cache_row_bytes(cache_type_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    uint32_t v_row_bytes = p36_kv_cache_row_bytes(cache_type_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    uint32_t had_width;
    if (!k_row_bytes || !v_row_bytes) return NULL;
    had_width = P36_N_EMBD;
    if (P36_N_FF_SHARED > had_width) had_width = P36_N_FF_SHARED;
    if (P36_N_SSM_INNER > had_width) had_width = P36_N_SSM_INNER;
    rt = xcalloc(1, sizeof(*rt));
    full_cap = p36_kv_initial_cap(ctx_size);
    state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    scratch_bytes = (size_t)P36_CPU_SCRATCH_FLOATS * sizeof(float);
    rt->prefill_cap = prefill_cap;
    rt->hidden = xmalloc((size_t)P36_N_EMBD * sizeof(float));
    rt->next_hidden = xmalloc((size_t)P36_N_EMBD * sizeof(float));
    rt->work0 = xmalloc(scratch_bytes);
    rt->work1 = xmalloc(scratch_bytes);
    rt->work2 = xmalloc(scratch_bytes);
    rt->work3 = xmalloc(scratch_bytes);
    rt->work4 = xmalloc(scratch_bytes);
    rt->work5 = xmalloc(scratch_bytes);
    rt->scores = xmalloc((size_t)full_cap * sizeof(float));
    rt->batch_hidden = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_next_hidden = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_norm = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_qg = xmalloc((size_t)prefill_cap * P36_N_HEAD * P36_N_HEAD_DIM * 2u * sizeof(float));
    rt->batch_q = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_k = xmalloc((size_t)prefill_cap * P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(float));
    rt->batch_v = xmalloc((size_t)prefill_cap * P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(float));
    rt->batch_attn_out = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_qkv = xmalloc((size_t)prefill_cap * P36_N_SSM_CONV_DIM * sizeof(float));
    rt->batch_recur_z = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_alpha = xmalloc((size_t)prefill_cap * P36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_beta = xmalloc((size_t)prefill_cap * P36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_proj = xmalloc((size_t)prefill_cap * P36_N_SSM_INNER * sizeof(float));
    rt->batch_ffn_gate_logits = xmalloc((size_t)prefill_cap * P36_N_EXPERT * sizeof(float));
    rt->batch_ffn_shared_gate = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_up = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_mid = xmalloc((size_t)prefill_cap * P36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_out = xmalloc((size_t)prefill_cap * P36_N_EMBD * sizeof(float));
    rt->batch_ffn_scalar = xmalloc((size_t)prefill_cap * sizeof(float));
    rt->batch_xq = xmalloc((size_t)prefill_cap * P36_MAX_Q8_K_BYTES);
    rt->batch_had = xmalloc((size_t)prefill_cap * had_width * sizeof(float));
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            rt->full[il].cap = full_cap;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = xmalloc((size_t)full_cap * k_row_bytes);
            rt->full[il].v = xmalloc((size_t)full_cap * v_row_bytes);
        } else {
            rt->recurrent[il].conv = xmalloc_zeroed((size_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM, sizeof(float));
            rt->recurrent[il].state = xmalloc_zeroed((size_t)state_dim, sizeof(float));
        }
    }
    return rt;
}

static bool p36_cpu_runtime_reserve_kv(p36_cpu_runtime *rt, uint32_t need,
                                       uint32_t limit) {
    if (!rt || need > limit) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_full_attn_cache *c;
        uint32_t cap;
        uint8_t *k, *v;
        if (!p36_layer_is_full_attention(il)) continue;
        c = &rt->full[il];
        if (need <= c->cap) continue;
        cap = p36_kv_next_cap(c->cap, need, limit);
        if (!cap) return false;
        k = xmalloc((size_t)cap * c->k_row_bytes);
        v = xmalloc((size_t)cap * c->v_row_bytes);
        memcpy(k, c->k, (size_t)c->len * c->k_row_bytes);
        memcpy(v, c->v, (size_t)c->len * c->v_row_bytes);
        free(c->k);
        free(c->v);
        c->k = k;
        c->v = v;
        c->cap = cap;
    }
    return true;
}

static void p36_cpu_runtime_reset(p36_cpu_runtime *rt) {
    uint64_t state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    if (!rt) return;
    memset(rt->hidden, 0, (size_t)P36_N_EMBD * sizeof(float));
    memset(rt->next_hidden, 0, (size_t)P36_N_EMBD * sizeof(float));
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        rt->full[il].len = 0;
        if (rt->recurrent[il].conv) {
            memset(rt->recurrent[il].conv, 0, (size_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float));
        }
        if (rt->recurrent[il].state) {
            memset(rt->recurrent[il].state, 0, (size_t)state_dim * sizeof(float));
        }
    }
}

static void p36_cpu_runtime_free(p36_cpu_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        free(rt->full[il].k);
        free(rt->full[il].v);
        free(rt->recurrent[il].conv);
        free(rt->recurrent[il].state);
    }
    free(rt->hidden);
    free(rt->next_hidden);
    free(rt->work0);
    free(rt->work1);
    free(rt->work2);
    free(rt->work3);
    free(rt->work4);
    free(rt->work5);
    free(rt->scores);
    free(rt->batch_hidden);
    free(rt->batch_next_hidden);
    free(rt->batch_norm);
    free(rt->batch_qg);
    free(rt->batch_q);
    free(rt->batch_k);
    free(rt->batch_v);
    free(rt->batch_attn_out);
    free(rt->batch_recur_qkv);
    free(rt->batch_recur_z);
    free(rt->batch_recur_alpha);
    free(rt->batch_recur_beta);
    free(rt->batch_recur_proj);
    free(rt->batch_ffn_gate_logits);
    free(rt->batch_ffn_shared_gate);
    free(rt->batch_ffn_shared_up);
    free(rt->batch_ffn_shared_mid);
    free(rt->batch_ffn_shared_out);
    free(rt->batch_ffn_scalar);
    free(rt->batch_xq);
    free(rt->batch_had);
    free(rt);
}

#ifndef P36_NO_GPU
static bool p36_gpu_tensor_zero(p36_gpu_tensor *tensor) {
    void *p;
    if (!tensor) return false;
    p = p36_gpu_tensor_contents_named(tensor, "submit_wait_tensor_zero");
    if (!p) return false;
    memset(p, 0, (size_t)p36_gpu_tensor_bytes(tensor));
    return true;
}

static bool p36_gpu_tensor_all_finite(const p36_gpu_tensor *tensor, uint32_t n) {
    float *p = p36_gpu_tensor_contents_named((p36_gpu_tensor *)tensor, "submit_wait_all_finite");
    return p && p36_all_finite(p, n);
}

static bool p36_gpu_tensor_scale_host(p36_gpu_tensor *tensor, uint32_t n, float scale) {
    float *p = p36_gpu_tensor_contents_named(tensor, "submit_wait_scale_host");
    if (!p) return false;
    p36_scale_inplace(p, n, scale);
    return true;
}

static bool p36_gpu_embed_tokens(const p36_model *m, const p36_tensor *t,
                                 const int *tokens, uint32_t n_tok, p36_gpu_tensor *dst) {
    float *p = p36_gpu_tensor_contents_named(dst, "submit_wait_embed_tokens");
    if (!p) return false;
    for (uint32_t i = 0; i < n_tok; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)P36_N_VOCAB) return false;
        float *row = p + (uint64_t)i * P36_N_EMBD;
        if (!p36_tensor_row_to_float(m, t, (uint64_t)tokens[i], row, P36_N_EMBD)) {
            return false;
        }
        if (t->hadamard_inverse) p36_hadamard_inverse_host(row, P36_N_EMBD);
    }
    return true;
}

/* Build the n_tok conv windows in token order while sliding the persistent
 * conv history, exactly n_tok runs of p36_recurrent_conv_step(). */
static P36_MAYBE_UNUSED bool p36_gpu_recurrent_conv_step(p36_gpu_tensor *cache_conv,
                                        const p36_gpu_tensor *cur,
                                        p36_gpu_tensor *window,
                                        uint32_t n_tok) {
    const uint32_t hist = P36_N_SSM_CONV - 1u;
    const size_t row_bytes = (size_t)P36_N_SSM_CONV_DIM * sizeof(float);
    float *cachep = p36_gpu_tensor_contents_named(cache_conv, "submit_wait_conv_cache");
    float *windowp = p36_gpu_tensor_contents_named(window, "submit_wait_conv_window");
    float *curp = p36_gpu_tensor_contents_named((p36_gpu_tensor *)cur, "submit_wait_conv_cur");
    if (!cachep || !windowp || !curp) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        float *wt = windowp + (uint64_t)t * P36_N_SSM_CONV * P36_N_SSM_CONV_DIM;
        const float *ct = curp + (uint64_t)t * P36_N_SSM_CONV_DIM;
        for (uint32_t i = 0; i < hist; i++) {
            memcpy(wt + (uint64_t)i * P36_N_SSM_CONV_DIM,
                   cachep + (uint64_t)i * P36_N_SSM_CONV_DIM,
                   row_bytes);
        }
        memcpy(wt + (uint64_t)hist * P36_N_SSM_CONV_DIM, ct, row_bytes);
        if (hist > 1u) {
            memmove(cachep,
                    cachep + P36_N_SSM_CONV_DIM,
                    (size_t)(hist - 1u) * row_bytes);
        }
        memcpy(cachep + (uint64_t)(hist - 1u) * P36_N_SSM_CONV_DIM, ct, row_bytes);
    }
    return true;
}

static P36_MAYBE_UNUSED bool p36_gpu_extract_full_attn_q(p36_gpu_tensor *dst, const p36_gpu_tensor *qg, uint32_t n_tok) {
    const float *src = p36_gpu_tensor_contents_named((p36_gpu_tensor *)qg, "submit_wait_extract_q_src");
    float *out = p36_gpu_tensor_contents_named(dst, "submit_wait_extract_q_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < P36_N_HEAD; h++) {
            memcpy(out + ((uint64_t)t * P36_N_HEAD + h) * P36_N_HEAD_DIM,
                   src + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u,
                   (size_t)P36_N_HEAD_DIM * sizeof(float));
        }
    }
    return true;
}

static P36_MAYBE_UNUSED bool p36_gpu_extract_recurrent_v(p36_gpu_tensor *dst, const p36_gpu_tensor *conv, uint32_t n_tok) {
    const float *src = p36_gpu_tensor_contents_named((p36_gpu_tensor *)conv, "submit_wait_extract_v_src");
    float *out = p36_gpu_tensor_contents_named(dst, "submit_wait_extract_v_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        memcpy(out + (uint64_t)t * P36_N_SSM_INNER,
               src + (uint64_t)t * P36_N_SSM_CONV_DIM + (uint64_t)P36_N_SSM_QK * 2u,
               (size_t)P36_N_SSM_INNER * sizeof(float));
    }
    return true;
}

static bool p36_gpu_tensor_matmul_scaled(const p36_model *m,
                                         const p36_tensor *t,
                                         const p36_gpu_tensor *x,
                                         p36_gpu_tensor *out,
                                         uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t n_tok,
                                         float scale) {
    bool ok = false;
    if (!m || !t || !x || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case P36_TENSOR_F32:
        ok = p36_gpu_matmul_f32_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                              in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_F16:
        ok = p36_gpu_matmul_f16_tensor(out, m->map, m->size, t->abs_offset,
                                       in_dim, out_dim, x, n_tok) != 0;
        if (ok && scale != 1.0f) ok = p36_gpu_tensor_scale_host(out, n_tok * out_dim, scale);
        break;
    case P36_TENSOR_BF16:
        ok = p36_gpu_matmul_bf16_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                               in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_Q8_0:
        ok = p36_gpu_matmul_q8_0_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                               in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
        ok = p36_gpu_matmul_k_quant_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                  t->type, in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        ok = p36_gpu_matmul_iq_quant_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                   t->type, in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    default:
        return false;
    }
    return ok;
}

static bool p36_gpu_tensor_matmul_q8_scaled(const p36_model *m,
                                            const p36_tensor *t,
                                            const p36_gpu_tensor *xq,
                                            p36_gpu_tensor *out,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            uint32_t n_tok,
                                            float scale) {
    if (!m || !t || !xq || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
        return p36_gpu_matmul_k_quant_q8_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                        t->type, in_dim, out_dim, xq, n_tok, scale) != 0;
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return p36_gpu_matmul_iq_quant_q8_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                        t->type, in_dim, out_dim, xq, n_tok, scale) != 0;
    default:
        return false;
    }
}

static bool p36_gpu_tensor_matmul_q8_or_float_scaled(const p36_model *m,
                                                     const p36_tensor *t,
                                                     const p36_gpu_tensor *x,
                                                     const p36_gpu_tensor *xq,
                                                     p36_gpu_tensor *out,
                                                     uint32_t in_dim,
                                                     uint32_t out_dim,
                                                     uint32_t n_tok,
                                                     float scale) {
    if (!t) return false;
#if defined(P36_METAL)
    /* Dense Metal has native float-RHS IQ3/K-quant kernels.  Keep the original
     * activation instead of forcing a host-synchronized q8_K staging pass.
     * Vulkan retains its tuned q8_K dense path unchanged. */
    if (P36_MODEL_DENSE) {
        return p36_gpu_tensor_matmul_scaled(m, t, x, out,
                                             in_dim, out_dim, n_tok, scale);
    }
#endif
    if (t->type == P36_TENSOR_F32 || t->type == P36_TENSOR_F16 ||
        t->type == P36_TENSOR_BF16 || t->type == P36_TENSOR_Q8_0) {
        return p36_gpu_tensor_matmul_scaled(m, t, x, out, in_dim, out_dim, n_tok, scale);
    }
    return p36_gpu_tensor_matmul_q8_scaled(m, t, xq, out, in_dim, out_dim, n_tok, scale);
}

static bool p36_gpu_tensor_matmul_dense_q8_scaled(const p36_model *m,
                                                   const p36_tensor *t,
                                                   const p36_gpu_tensor *x,
                                                   const p36_gpu_tensor *xq,
                                                   p36_gpu_tensor *out,
                                                   uint32_t in_dim,
                                                   uint32_t out_dim,
                                                   uint32_t n_tok,
                                                   float scale) {
    if (!P36_MODEL_DENSE)
        return p36_gpu_tensor_matmul_scaled(m, t, x, out,
                                            in_dim, out_dim, n_tok, scale);
    return p36_gpu_tensor_matmul_q8_or_float_scaled(m, t, x, xq, out,
                                                     in_dim, out_dim, n_tok, scale);
}

static bool p36_vulkan_runtime_reset(p36_vulkan_runtime *rt) {
    uint64_t state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    uint64_t state_bytes;
    if (!rt) return false;
    rt->inp_q8_valid = false;
    state_bytes = state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float));
    if (!p36_gpu_tensor_zero(rt->hidden) || !p36_gpu_tensor_zero(rt->next_hidden)) return false;
    if (rt->last_h && !p36_gpu_tensor_zero(rt->last_h)) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (rt->recurrent[il].conv && !p36_gpu_tensor_zero(rt->recurrent[il].conv)) return false;
        if (rt->recurrent[il].state && !p36_gpu_tensor_zero(rt->recurrent[il].state)) return false;
        if (rt->recurrent[il].state && p36_gpu_tensor_bytes(rt->recurrent[il].state) != state_bytes) return false;
    }
    return true;
}

static void p36_vulkan_runtime_free(p36_vulkan_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_gpu_tensor_free(rt->full[il].k);
        p36_gpu_tensor_free(rt->full[il].v);
        p36_gpu_tensor_free(rt->recurrent[il].conv);
        p36_gpu_tensor_free(rt->recurrent[il].state);
        p36_gpu_tensor_free(rt->spec_recurrent[il].conv);
        p36_gpu_tensor_free(rt->spec_recurrent[il].state);
    }
    p36_gpu_tensor_free(rt->hidden);
    p36_gpu_tensor_free(rt->next_hidden);
    p36_gpu_tensor_free(rt->embed_stage[0]);
    p36_gpu_tensor_free(rt->embed_stage[1]);
    p36_gpu_tensor_free(rt->norm);
    p36_gpu_tensor_free(rt->last_h);
    p36_gpu_tensor_free(rt->inp_q8);
    p36_gpu_tensor_free(rt->had_q8);
    p36_gpu_tensor_free(rt->had_signs);
    p36_gpu_tensor_free(rt->attn_qg);
    p36_gpu_tensor_free(rt->attn_q);
    p36_gpu_tensor_free(rt->attn_k);
    p36_gpu_tensor_free(rt->attn_v);
    p36_gpu_tensor_free(rt->attn_out);
    p36_gpu_tensor_free(rt->recur_qkv);
    p36_gpu_tensor_free(rt->recur_window);
    p36_gpu_tensor_free(rt->recur_conv);
    p36_gpu_tensor_free(rt->recur_z);
    p36_gpu_tensor_free(rt->recur_alpha);
    p36_gpu_tensor_free(rt->recur_beta);
    p36_gpu_tensor_free(rt->recur_gb);
    p36_gpu_tensor_free(rt->recur_q);
    p36_gpu_tensor_free(rt->recur_k);
    p36_gpu_tensor_free(rt->recur_v);
    p36_gpu_tensor_free(rt->recur_proj);
    p36_gpu_tensor_free(rt->ffn_gate_logits);
    p36_gpu_tensor_free(rt->ffn_selected);
    p36_gpu_tensor_free(rt->ffn_weights);
    p36_gpu_tensor_free(rt->ffn_shared_gate);
    p36_gpu_tensor_free(rt->ffn_shared_up);
    p36_gpu_tensor_free(rt->ffn_shared_mid);
    p36_gpu_tensor_free(rt->ffn_shared_out);
    p36_gpu_tensor_free(rt->ffn_scalar);
    p36_gpu_tensor_free(rt->logits);
    p36_gpu_tensor_free(rt->top2);
    p36_gpu_tensor_free(rt->topk8);
    p36_gpu_tensor_free(rt->minp_pack);
    p36_gpu_tensor_free(rt->scores);
    p36_gpu_tensor_free(rt->mtp_full.k);
    p36_gpu_tensor_free(rt->mtp_full.v);
    p36_gpu_tensor_free(rt->spec_logits);
    p36_gpu_tensor_free(rt->mtp_tok_embd);
    p36_gpu_tensor_free(rt->mtp_e_norm);
    p36_gpu_tensor_free(rt->mtp_h_norm);
    p36_gpu_tensor_free(rt->mtp_concat);
    p36_gpu_tensor_free(rt->mtp_cur);
    p36_gpu_tensor_free(rt->mtp_next);
    p36_gpu_tensor_free(rt->mtp_head);
    p36_gpu_tensor_free(rt->mtp_logits);
    free(rt);
}

static p36_vulkan_runtime *p36_vulkan_runtime_create(int ctx_size,
                                                     uint32_t prefill_cap,
                                                     bool enable_mtp,
                                                     bool quality,
                                                     bool ssd_streaming,
                                                     p36_kv_cache_type cache_type_k,
                                                     p36_kv_cache_type cache_type_v) {
    p36_vulkan_runtime *rt;
    uint64_t state_dim;
    uint64_t hist_rows;
    uint32_t kv_cap;
    if (ctx_size <= 0 || !p36_gpu_init()) return NULL;
    uint32_t k_row_bytes = p36_kv_cache_row_bytes(cache_type_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    uint32_t v_row_bytes = p36_kv_cache_row_bytes(cache_type_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    if (!k_row_bytes || !v_row_bytes) return NULL;
    rt = xcalloc(1, sizeof(*rt));
    kv_cap = p36_kv_initial_cap(ctx_size);
    if (prefill_cap < 1) prefill_cap = 1;
    rt->prefill_cap = prefill_cap;
    rt->mtp_enabled = enable_mtp;
    {
        const char *env = getenv("P36_VK_RECURRENT_CONV_DECODE");
        rt->recur_conv_fused = !quality && !ssd_streaming &&
                               (!env || !env[0] || env[0] != '0');
    }
#ifndef P36_METAL
    {
        const char *env = getenv("P36_VK_RECURRENT_STATE_F16");
        rt->recur_state_f16 = !quality && !ssd_streaming &&
                              (!env || !env[0] || env[0] != '0');
    }
#endif
    state_dim = (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK;
    hist_rows = P36_N_SSM_CONV - 1u;
    /* Activation scratch carries prefill_cap token rows for batched prefill;
     * single-token decode just uses row 0. */
#define P36_GPU_ALLOC_F32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_SCRATCH_F32(field, n) do { \
        rt->field = quality ? \
            p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(float)) : \
            p36_gpu_tensor_alloc_scratch((uint64_t)(n) * prefill_cap * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_U32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(uint32_t)); \
        if (!rt->field) goto fail; \
    } while (0)
#define P36_GPU_ALLOC_MTP_F32(field, n) do { \
        rt->field = p36_gpu_tensor_alloc((uint64_t)(n) * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
    P36_GPU_ALLOC_F32(hidden, P36_N_EMBD);
    P36_GPU_ALLOC_F32(next_hidden, P36_N_EMBD);
    P36_GPU_ALLOC_F32(embed_stage[0], P36_N_EMBD);
    P36_GPU_ALLOC_F32(embed_stage[1], P36_N_EMBD);
    P36_GPU_ALLOC_SCRATCH_F32(norm, P36_N_EMBD);
    rt->last_h = p36_gpu_tensor_alloc((uint64_t)P36_N_EMBD * sizeof(float));
    if (!rt->last_h) goto fail;
#ifdef P36_METAL
    P36_GPU_ALLOC_F32(recur_v, P36_N_SSM_INNER);
    {
        const uint64_t attn_kv_bytes =
            (uint64_t)P36_N_HEAD_KV *
            (P36_N_HEAD_DIM + P36_N_VALUE_DIM) *
            prefill_cap * sizeof(float);
        const uint64_t inp_q8_bytes =
            (uint64_t)((P36_N_EMBD + P36_QK_K - 1u) / P36_QK_K) *
            P36_VK_Q8_K_BYTES * prefill_cap;
        rt->inp_q8 = p36_gpu_tensor_view(
            rt->recur_v, attn_kv_bytes, inp_q8_bytes);
        if (!rt->inp_q8) goto fail;
    }
    /* Allocate attention partials before recurrent scratch: on Metal, large
     * contexts can reuse this owner on layers where attention is inactive. */
    if (!p36_gpu_attn_fused_enabled()) {
        uint64_t score_bytes = p36_metal_attention_scratch_bytes(
            (uint32_t)ctx_size, prefill_cap, cache_type_k, cache_type_v);
        rt->scores = p36_gpu_tensor_alloc(score_bytes);
        if (!rt->scores) goto fail;
    }
    P36_GPU_ALLOC_F32(attn_qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u);
    P36_GPU_ALLOC_F32(attn_q, P36_N_HEAD * P36_N_HEAD_DIM);
    rt->attn_k = p36_gpu_tensor_view(rt->recur_v, 0,
        (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM *
            prefill_cap * sizeof(float));
    rt->attn_v = p36_gpu_tensor_view(rt->recur_v,
        (uint64_t)P36_N_HEAD_KV * P36_N_HEAD_DIM *
            prefill_cap * sizeof(float),
        (uint64_t)P36_N_HEAD_KV * P36_N_VALUE_DIM *
            prefill_cap * sizeof(float));
    if (!rt->attn_k || !rt->attn_v) goto fail;
    P36_GPU_ALLOC_F32(attn_out, P36_N_SSM_INNER);
#else
    uint32_t q8_width = P36_MODEL_DENSE && P36_N_FF_SHARED > P36_N_EMBD ?
        P36_N_FF_SHARED : P36_N_EMBD;
    rt->inp_q8 = quality ?
        p36_gpu_tensor_alloc((uint64_t)((q8_width + P36_QK_K - 1u) / P36_QK_K) * P36_VK_Q8_K_BYTES * prefill_cap) :
        p36_gpu_tensor_alloc_scratch((uint64_t)((q8_width + P36_QK_K - 1u) / P36_QK_K) * P36_VK_Q8_K_BYTES * prefill_cap);
    if (!rt->inp_q8) goto fail;
    P36_GPU_ALLOC_SCRATCH_F32(attn_qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u);
    P36_GPU_ALLOC_SCRATCH_F32(attn_q, P36_N_HEAD * P36_N_HEAD_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    P36_GPU_ALLOC_SCRATCH_F32(attn_out, P36_N_SSM_INNER);
#endif
    if (g_p36_hadamard.enabled) {
        uint32_t had_width = P36_N_EMBD;
        uint64_t signs_total = 0;
        if (P36_N_FF_SHARED > had_width) had_width = P36_N_FF_SHARED;
        if (P36_N_SSM_INNER > had_width) had_width = P36_N_SSM_INNER;
        if (P36_N_SSM_CONV_DIM > had_width) had_width = P36_N_SSM_CONV_DIM;
        {
            uint64_t q8_bytes = (uint64_t)((had_width + P36_QK_K - 1u) / P36_QK_K) *
                                P36_VK_Q8_K_BYTES * prefill_cap;
            rt->had_q8 = quality ? p36_gpu_tensor_alloc(q8_bytes)
                                 : p36_gpu_tensor_alloc_scratch(q8_bytes);
            if (!rt->had_q8) goto fail;
        }
        for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
            signs_total += g_p36_hadamard.widths[i];
        }
        rt->had_signs = p36_gpu_tensor_alloc(signs_total * sizeof(float));
        if (!rt->had_signs) goto fail;
        {
            float *sp = (float *)p36_gpu_tensor_contents(rt->had_signs);
            uint64_t off = 0;
            if (!sp) goto fail;
            for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
                memcpy(sp + off, g_p36_hadamard.signs[i],
                       (size_t)g_p36_hadamard.widths[i] * sizeof(float));
                off += g_p36_hadamard.widths[i];
            }
        }
    }
    /* A layer is attention or recurrent, never both. These equal-sized
     * views remove 64 MiB of mutually exclusive 1024-row scratch. */
    rt->recur_qkv = p36_gpu_tensor_view(rt->attn_qg, 0,
        (uint64_t)P36_N_SSM_CONV_DIM * prefill_cap * sizeof(float));
    if (!rt->recur_qkv) goto fail;
    if (!rt->recur_conv_fused)
        P36_GPU_ALLOC_F32(recur_window, (uint64_t)P36_N_SSM_CONV * P36_N_SSM_CONV_DIM);
#ifdef P36_METAL
    {
        const uint64_t conv_bytes =
            (uint64_t)P36_N_SSM_CONV_DIM * prefill_cap * sizeof(float);
        const uint64_t alpha_bytes =
            (uint64_t)P36_N_SSM_DT_RANK * prefill_cap * sizeof(float);
        const uint64_t gb_bytes = alpha_bytes * 2u;
        const uint64_t aux_bytes = conv_bytes + alpha_bytes * 2u + gb_bytes;
        if (rt->scores && p36_gpu_tensor_bytes(rt->scores) >= aux_bytes) {
            uint64_t offset = 0;
            rt->recur_conv = p36_gpu_tensor_view(
                rt->scores, offset, conv_bytes);
            offset += conv_bytes;
            rt->recur_alpha = p36_gpu_tensor_view(
                rt->scores, offset, alpha_bytes);
            offset += alpha_bytes;
            rt->recur_beta = p36_gpu_tensor_view(
                rt->scores, offset, alpha_bytes);
            offset += alpha_bytes;
            rt->recur_gb = p36_gpu_tensor_view(
                rt->scores, offset, gb_bytes);
            if (!rt->recur_conv || !rt->recur_alpha ||
                !rt->recur_beta || !rt->recur_gb) goto fail;
        } else {
            P36_GPU_ALLOC_F32(recur_conv, P36_N_SSM_CONV_DIM);
            P36_GPU_ALLOC_F32(recur_alpha, P36_N_SSM_DT_RANK);
            P36_GPU_ALLOC_F32(recur_beta, P36_N_SSM_DT_RANK);
            P36_GPU_ALLOC_F32(recur_gb, P36_N_SSM_DT_RANK * 2u);
        }
    }
#else
    if (!P36_MODEL_DENSE)
        P36_GPU_ALLOC_F32(recur_conv, P36_N_SSM_CONV_DIM);
#endif
    rt->recur_z = p36_gpu_tensor_view(rt->attn_q, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_z) goto fail;
#ifndef P36_METAL
    if (!P36_MODEL_DENSE) {
        P36_GPU_ALLOC_F32(recur_alpha, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_beta, P36_N_SSM_DT_RANK);
        P36_GPU_ALLOC_F32(recur_gb, P36_N_SSM_DT_RANK * 2u);
    }
#endif
#ifdef P36_METAL
    /* Recurrent QKV is dead once convolution has produced recur_conv.  The
     * following delta-net stage needs Q and K simultaneously, so split that
     * dead row into two views. Command ordering preserves the dependency. */
    rt->recur_q = p36_gpu_tensor_view(rt->attn_qg, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    rt->recur_k = p36_gpu_tensor_view(rt->attn_qg,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float),
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_q || !rt->recur_k) goto fail;
#else
    if (P36_MODEL_DENSE) {
        rt->recur_q = p36_gpu_tensor_view(rt->attn_qg, 0,
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
        rt->recur_k = p36_gpu_tensor_view(rt->attn_qg,
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float),
            (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
        if (!rt->recur_q || !rt->recur_k) goto fail;
    } else {
        P36_GPU_ALLOC_F32(recur_q, P36_N_SSM_INNER);
        P36_GPU_ALLOC_F32(recur_k, P36_N_SSM_INNER);
    }
#endif
#ifndef P36_METAL
    if (!P36_MODEL_DENSE)
        P36_GPU_ALLOC_F32(recur_v, P36_N_SSM_INNER);
#endif
    rt->recur_proj = p36_gpu_tensor_view(rt->attn_out, 0,
        (uint64_t)P36_N_SSM_INNER * prefill_cap * sizeof(float));
    if (!rt->recur_proj) goto fail;
#ifdef P36_METAL
    /* FFN starts after attention/recurrent output and the residual norm, so
     * their QGV storage is dead for the rest of the layer. Pack all FFN
     * control/shared tensors into non-overlapping views of that 32 MiB owner. */
    if (P36_MODEL_DENSE) {
        /* Dense gate/up need two full-width owners. Mid overwrites gate;
         * the routed controls, shared tail and scalar are never used.  Large
         * attention scratch is dead by this point in every layer, so reuse it
         * for one or both panels when large enough. */
        const uint64_t ffn_panel_bytes =
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float);
        if (rt->scores &&
            p36_gpu_tensor_bytes(rt->scores) >= 2u * ffn_panel_bytes) {
            rt->ffn_shared_gate = p36_gpu_tensor_view(
                rt->scores, 0, ffn_panel_bytes);
            rt->ffn_shared_up = p36_gpu_tensor_view(
                rt->scores, ffn_panel_bytes, ffn_panel_bytes);
            if (!rt->ffn_shared_gate || !rt->ffn_shared_up) goto fail;
        } else if (rt->scores &&
                   p36_gpu_tensor_bytes(rt->scores) >= ffn_panel_bytes) {
            rt->ffn_shared_gate = p36_gpu_tensor_view(
                rt->scores, 0, ffn_panel_bytes);
            P36_GPU_ALLOC_F32(ffn_shared_up, P36_N_FF_SHARED);
        } else {
            P36_GPU_ALLOC_F32(ffn_shared_gate, P36_N_FF_SHARED);
            P36_GPU_ALLOC_F32(ffn_shared_up, P36_N_FF_SHARED);
        }
        rt->ffn_shared_mid = p36_gpu_tensor_view(
            rt->ffn_shared_gate, 0,
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
        if (!rt->ffn_shared_mid) goto fail;
    } else {
        uint64_t ffn_alias_offset = 0;
#define P36_GPU_ALIAS_FFN(field, n, type) do { \
        uint64_t alias_bytes = (uint64_t)(n) * prefill_cap * sizeof(type); \
        rt->field = p36_gpu_tensor_view( \
            rt->attn_qg, ffn_alias_offset, alias_bytes); \
        if (!rt->field) goto fail; \
        ffn_alias_offset += alias_bytes; \
    } while (0)
    P36_GPU_ALIAS_FFN(ffn_gate_logits, P36_N_EXPERT, float);
    P36_GPU_ALIAS_FFN(ffn_selected, P36_N_EXPERT_USED, uint32_t);
    P36_GPU_ALIAS_FFN(ffn_weights, P36_N_EXPERT_USED, float);
    P36_GPU_ALIAS_FFN(ffn_shared_gate, P36_N_FF_SHARED, float);
    P36_GPU_ALIAS_FFN(ffn_shared_up, P36_N_FF_SHARED, float);
    rt->ffn_shared_mid = p36_gpu_tensor_view(rt->ffn_shared_gate, 0,
        (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
    if (!rt->ffn_shared_mid) goto fail;
    P36_GPU_ALIAS_FFN(ffn_shared_out, P36_N_EMBD, float);
    P36_GPU_ALIAS_FFN(ffn_scalar, 1, float);
#undef P36_GPU_ALIAS_FFN
    }
#else
    if (P36_MODEL_DENSE) {
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_gate, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_up, P36_N_FF_SHARED);
        {
            uint64_t offset = 0;
#define P36_GPU_ALIAS_DENSE_RECUR(field, n) do { \
            uint64_t alias_bytes = (uint64_t)(n) * prefill_cap * sizeof(float); \
            rt->field = p36_gpu_tensor_view(rt->ffn_shared_gate, offset, alias_bytes); \
            if (!rt->field) goto fail; \
            offset += alias_bytes; \
        } while (0)
            P36_GPU_ALIAS_DENSE_RECUR(recur_conv, P36_N_SSM_CONV_DIM);
            P36_GPU_ALIAS_DENSE_RECUR(recur_alpha, P36_N_SSM_DT_RANK);
            P36_GPU_ALIAS_DENSE_RECUR(recur_beta, P36_N_SSM_DT_RANK);
            P36_GPU_ALIAS_DENSE_RECUR(recur_gb, P36_N_SSM_DT_RANK * 2u);
            P36_GPU_ALIAS_DENSE_RECUR(recur_v, P36_N_SSM_INNER);
#undef P36_GPU_ALIAS_DENSE_RECUR
        }
        rt->ffn_shared_mid = p36_gpu_tensor_view(
            rt->ffn_shared_gate, 0,
            (uint64_t)P36_N_FF_SHARED * prefill_cap * sizeof(float));
        if (!rt->ffn_shared_mid) goto fail;
    } else {
        P36_GPU_ALLOC_F32(ffn_gate_logits, P36_N_EXPERT);
        P36_GPU_ALLOC_U32(ffn_selected, P36_N_EXPERT_USED);
        P36_GPU_ALLOC_F32(ffn_weights, P36_N_EXPERT_USED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_gate, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_up, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_mid, P36_N_FF_SHARED);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_shared_out, P36_N_EMBD);
        P36_GPU_ALLOC_SCRATCH_F32(ffn_scalar, 1);
    }
    /* The fused attention path never touches the scores scratch; skipping it
     * saves ctx_size * n_head * prefill_cap floats (128 MiB at ctx 32k). */
    if (!p36_gpu_attn_fused_enabled())
        P36_GPU_ALLOC_SCRATCH_F32(scores, (uint64_t)ctx_size * P36_N_HEAD);
#endif
    rt->logits = p36_gpu_tensor_alloc((uint64_t)P36_N_VOCAB * sizeof(float));
    rt->top2 = p36_gpu_tensor_alloc(2u * sizeof(int32_t));
    rt->topk8 = p36_gpu_tensor_alloc(8u * (sizeof(int32_t) + sizeof(float)));
    rt->minp_pack = p36_gpu_tensor_alloc(P36_GPU_MINP_BYTES(P36_N_VOCAB));
    if (!rt->logits || !rt->top2 || !rt->topk8 || !rt->minp_pack) goto fail;
    if (enable_mtp) {
        rt->mtp_full.cap = kv_cap;
        rt->mtp_full.type_k = cache_type_k;
        rt->mtp_full.type_v = cache_type_v;
        rt->mtp_full.k_row_bytes = k_row_bytes;
        rt->mtp_full.v_row_bytes = v_row_bytes;
        rt->mtp_full.k = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * k_row_bytes);
        rt->mtp_full.v = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * v_row_bytes);
        if (!rt->mtp_full.k || !rt->mtp_full.v) goto fail;
        rt->spec_logits = p36_gpu_tensor_alloc((uint64_t)P36_MTP_MAX_DRAFT * P36_N_VOCAB * sizeof(float));
        if (!rt->spec_logits) goto fail;
        P36_GPU_ALLOC_MTP_F32(mtp_tok_embd, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_e_norm, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_h_norm, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_concat, P36_N_EMBD * 2u);
        P36_GPU_ALLOC_MTP_F32(mtp_cur, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_next, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_head, P36_N_EMBD);
        P36_GPU_ALLOC_MTP_F32(mtp_logits, P36_N_VOCAB);
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            rt->full[il].cap = kv_cap;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * k_row_bytes);
            rt->full[il].v = p36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * v_row_bytes);
            if (!rt->full[il].k || !rt->full[il].v) goto fail;
        } else {
            rt->recurrent[il].conv = p36_gpu_tensor_alloc(hist_rows * P36_N_SSM_CONV_DIM * sizeof(float));
            rt->recurrent[il].state = p36_gpu_tensor_alloc(
                state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float)));
            if (!rt->recurrent[il].conv || !rt->recurrent[il].state) goto fail;
            if (enable_mtp) {
                rt->spec_recurrent[il].conv = p36_gpu_tensor_alloc(hist_rows * P36_N_SSM_CONV_DIM * sizeof(float));
                rt->spec_recurrent[il].state = p36_gpu_tensor_alloc(
                    state_dim * (rt->recur_state_f16 ? sizeof(uint16_t) : sizeof(float)));
                if (!rt->spec_recurrent[il].conv || !rt->spec_recurrent[il].state) goto fail;
            }
        }
    }
    if (!p36_vulkan_runtime_reset(rt)) goto fail;
#undef P36_GPU_ALLOC_F32
#undef P36_GPU_ALLOC_SCRATCH_F32
#undef P36_GPU_ALLOC_U32
#undef P36_GPU_ALLOC_MTP_F32
    return rt;
fail:
#undef P36_GPU_ALLOC_F32
#undef P36_GPU_ALLOC_SCRATCH_F32
#undef P36_GPU_ALLOC_U32
#undef P36_GPU_ALLOC_MTP_F32
    p36_vulkan_runtime_free(rt);
    return NULL;
}

static bool p36_vulkan_cache_reserve(p36_vulkan_full_attn_cache *c,
                                     uint32_t cap, uint32_t rows) {
    p36_gpu_tensor *k, *v;
    if (!c || rows > c->cap || cap <= c->cap) return false;
    p36_gpu_trim_scratch();
    k = p36_gpu_tensor_alloc_uninitialized((uint64_t)cap * c->k_row_bytes);
    v = p36_gpu_tensor_alloc_uninitialized((uint64_t)cap * c->v_row_bytes);
    if (!k || !v) goto fail;
    if (rows &&
        (!p36_gpu_tensor_copy(k, 0, c->k, 0, (uint64_t)rows * c->k_row_bytes) ||
         !p36_gpu_tensor_copy(v, 0, c->v, 0, (uint64_t)rows * c->v_row_bytes))) goto fail;
    if (!p36_gpu_synchronize()) goto fail;
    p36_gpu_tensor_free(c->k);
    p36_gpu_tensor_free(c->v);
    c->k = k;
    c->v = v;
    c->cap = cap;
    return true;
fail:
    p36_gpu_tensor_free(k);
    p36_gpu_tensor_free(v);
    return false;
}

static bool p36_vulkan_runtime_reserve_kv(p36_vulkan_runtime *rt,
                                          uint32_t need, uint32_t limit,
                                          uint32_t rows) {
    if (!rt || need > limit || rows > need) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        p36_vulkan_full_attn_cache *c;
        uint32_t cap;
        if (!p36_layer_is_full_attention(il)) continue;
        c = &rt->full[il];
        if (need <= c->cap) continue;
        cap = p36_kv_next_cap(c->cap, need, limit);
        if (!cap) return false;
        /* Tight heap: retry with the exact need instead of the 1.25x step. */
        if (!p36_vulkan_cache_reserve(c, cap, rows) &&
            (need == cap || !p36_vulkan_cache_reserve(c, need, rows))) return false;
    }
    if (rt->mtp_enabled && need > rt->mtp_full.cap) {
        uint32_t cap = p36_kv_next_cap(rt->mtp_full.cap, need, limit);
        if (!cap || !p36_vulkan_cache_reserve(&rt->mtp_full, cap, rows)) return false;
    }
    return true;
}
#endif

static bool p36_embed_token(const p36_engine *e, int token, float *out) {
    if (!e || !out || token < 0 || token >= (int)P36_N_VOCAB) return false;
    if (!p36_tensor_row_to_float(&e->model, e->weights.token_embd, (uint64_t)token, out, P36_N_EMBD)) {
        return false;
    }
    if (e->weights.token_embd->hadamard_inverse) p36_hadamard_inverse_host(out, P36_N_EMBD);
    return true;
}

/* Multi-section M-RoPE rotation, shared by the CPU reference and the Vulkan
 * scheduler (p36_vulkan_rope_heads): both engines must go through this one
 * compiled body, because a second copy of the rotation fuses differently
 * under -ffast-math and drifts by ulps. */
static void p36_apply_rope_axes(float *x, uint32_t head_dim,
                                uint32_t t, uint32_t h, uint32_t w) {
    const uint32_t n_rot = P36_N_ROT;
    const uint32_t half = n_rot / 2u;
    const float base = 10000000.0f;
    const uint32_t sect_dims = P36_ROPE_SECTIONS[0] + P36_ROPE_SECTIONS[1] + P36_ROPE_SECTIONS[2] + P36_ROPE_SECTIONS[3];
    const float theta_scale = powf(base, -2.0f / (float)n_rot);
    float theta[4] = {(float)t, (float)h, (float)w, 0.0f};
    if (head_dim < n_rot) return;
    if (sect_dims == 0 || sect_dims > half) return;
    for (uint32_t pair = 0; pair < half; pair++) {
        uint32_t axis = 3;
        uint32_t sector = pair % sect_dims;
        if (sector % 3u == 1u && sector < 3u * P36_ROPE_SECTIONS[1]) axis = 1;
        else if (sector % 3u == 2u && sector < 3u * P36_ROPE_SECTIONS[2]) axis = 2;
        else if (sector % 3u == 0u && sector < 3u * P36_ROPE_SECTIONS[0]) axis = 0;
        {
            float c = cosf(theta[axis]);
            float s = sinf(theta[axis]);
            float x0 = x[pair];
            float x1 = x[pair + half];
            x[pair] = x0 * c - x1 * s;
            x[pair + half] = x0 * s + x1 * c;
        }
        for (uint32_t i = 0; i < 4; i++) theta[i] *= theta_scale;
    }
}

static void p36_apply_rope_one(float *x, uint32_t head_dim, uint32_t pos) {
    p36_apply_rope_axes(x, head_dim, pos, pos, pos);
}

static void p36_full_attn_cache_store(p36_full_attn_cache *cache, uint32_t pos, const float *k, const float *v) {
    uint8_t *dstk;
    uint8_t *dstv;
    if (!cache || pos >= cache->cap) return;
    dstk = cache->k + (uint64_t)pos * cache->k_row_bytes;
    dstv = cache->v + (uint64_t)pos * cache->v_row_bytes;
    p36_kv_cache_store_row(dstk, cache->type_k, k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    p36_kv_cache_store_row(dstv, cache->type_v, v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    if (cache->len <= pos) cache->len = pos + 1u;
}

static void p36_recurrent_conv_step(p36_recurrent_cache *cache, const float *cur, float *window) {
    uint32_t hist = P36_N_SSM_CONV - 1u;
    for (uint32_t i = 0; i < hist; i++) {
        memcpy(window + (uint64_t)i * P36_N_SSM_CONV_DIM,
               cache->conv + (uint64_t)i * P36_N_SSM_CONV_DIM,
               (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(window + (uint64_t)hist * P36_N_SSM_CONV_DIM, cur, (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
    if (hist > 1u) {
        memmove(cache->conv,
                cache->conv + P36_N_SSM_CONV_DIM,
                (size_t)(hist - 1u) * P36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(cache->conv + (uint64_t)(hist - 1u) * P36_N_SSM_CONV_DIM, cur, (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
}

static void p36_ssm_conv_apply(const float *window, const float *kernel, float *out) {
    for (uint32_t c = 0; c < P36_N_SSM_CONV_DIM; c++) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < P36_N_SSM_CONV; t++) {
            acc += window[(uint64_t)t * P36_N_SSM_CONV_DIM + c] * kernel[(uint64_t)c * P36_N_SSM_CONV + t];
        }
        out[c] = acc;
    }
}

static void p36_delta_net_decode(float *state, const float *q, const float *k, const float *v,
                                 const float *g, const float *b, float *out) {
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        float *s = state + (uint64_t)h * P36_N_SSM_STATE * P36_N_SSM_STATE;
        const float *qh = q + (uint64_t)h * P36_N_SSM_STATE;
        const float *kh = k + (uint64_t)h * P36_N_SSM_STATE;
        const float *vh = v + (uint64_t)h * P36_N_SSM_STATE;
        float decay = expf(g[h]);
        float beta = b[h];
        float sk[P36_N_SSM_STATE];
        float d[P36_N_SSM_STATE];
        for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) {
            for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) s[(uint64_t)i * P36_N_SSM_STATE + j] *= decay;
        }
        for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * P36_N_SSM_STATE + j] * (double)kh[i];
            sk[j] = (float)acc;
            d[j] = (vh[j] - sk[j]) * beta;
        }
        for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) {
            float ki = kh[i];
            for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) s[(uint64_t)i * P36_N_SSM_STATE + j] += ki * d[j];
        }
        for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * P36_N_SSM_STATE + j] * (double)qh[i];
            out[(uint64_t)h * P36_N_SSM_STATE + j] = (float)(acc / sqrt((double)P36_N_SSM_STATE));
        }
    }
}

static bool p36_forward_ffn(const p36_engine *e, const p36_layer_weights *l, const float *inp, float *out,
                            p36_cpu_runtime *rt) {
    float gate_logits[P36_N_EXPERT];
    uint32_t top_idx[P36_N_EXPERT_USED];
    float top_val[P36_N_EXPERT_USED];
    float top_w[P36_N_EXPERT_USED];
    uint8_t inpq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq;
    p36_activation_quant_kind inpq_kind;
    float *gate = rt->work2;
    float *up = rt->work3;
    float *mid = rt->work4;
    float *rowbuf = rt->work5;
    float shared_gate;
    if (P36_MODEL_DENSE) {
        const p36_tensor *rot_gate = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ? l->ffn_gate_shexp :
                                     (p36_hadamard_rotate_enabled(l->ffn_up_shexp) ? l->ffn_up_shexp : NULL);
        const float *ffn_in = rot_gate ? p36_hadamard_forward_rows(rot_gate, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
        const float *down_in;
        inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (!p36_quantize_activation_row(inpq_kind, ffn_in, inpq, P36_N_EMBD)) return false;
        xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
        if (!p36_tensor_matvec_pair_prequant(e,
                                             l->ffn_gate_shexp,
                                             l->ffn_up_shexp,
                                             ffn_in, xq, gate, up, rowbuf,
                                             P36_N_EMBD, P36_N_FF_SHARED,
                                             1.0f, 1.0f)) {
            if (!p36_tensor_matvec_prequant(
                    e, l->ffn_gate_shexp, ffn_in, xq, gate, rowbuf,
                    P36_N_EMBD, P36_N_FF_SHARED) ||
                !p36_tensor_matvec_prequant(
                    e, l->ffn_up_shexp, ffn_in, xq, up, rowbuf,
                    P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
        }
        for (uint32_t j = 0; j < P36_N_FF_SHARED; j++)
            mid[j] = p36_siluf(gate[j]) * up[j];
        down_in = p36_hadamard_forward_rows(l->ffn_down_shexp, mid, rt->batch_had, P36_N_FF_SHARED, 1u);
        return p36_tensor_matvec(e, l->ffn_down_shexp, down_in, out, rowbuf,
                                 P36_N_FF_SHARED, P36_N_EMBD);
    }
    if (!p36_tensor_matvec(e, l->ffn_gate_inp, inp, gate_logits, rowbuf, P36_N_EMBD, P36_N_EXPERT)) return false;
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!p36_quantize_activation_row(inpq_kind, inp, inpq, P36_N_EMBD)) return false;
    xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
    p36_softmax_inplace(gate_logits, P36_N_EXPERT);
    p36_topk_indices(gate_logits, P36_N_EXPERT, P36_N_EXPERT_USED, top_idx, top_val);
    {
        double sum = 0.0;
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) sum += top_val[i];
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * e->expert_weights_scale;
    }
    memset(out, 0, (size_t)P36_N_EMBD * sizeof(float));
    for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) {
        float gate_scale = p36_tensor_index_or(&e->model, l->ffn_gate_exps_scale, top_idx[i], 1.0f);
        float up_scale = p36_tensor_index_or(&e->model, l->ffn_up_exps_scale, top_idx[i], 1.0f);
        if (!p36_tensor_expert_matvec_pair_prequant(e,
                                                    l->ffn_gate_exps,
                                                    l->ffn_up_exps,
                                                    top_idx[i],
                                                    inp,
                                                    xq,
                                                    gate,
                                                    up,
                                                    rowbuf,
                                                    P36_N_EMBD,
                                                    P36_N_FF_EXP,
                                                    gate_scale,
                                                    up_scale)) {
            if (!p36_tensor_expert_matvec_prequant(e, l->ffn_gate_exps, top_idx[i], inp, xq,
                                                   gate, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                return false;
            }
            p36_scale_inplace(gate, P36_N_FF_EXP, gate_scale);
            if (!p36_tensor_expert_matvec_prequant(e, l->ffn_up_exps, top_idx[i], inp, xq,
                                                   up, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                return false;
            }
            p36_scale_inplace(up, P36_N_FF_EXP, up_scale);
        }
        for (uint32_t j = 0; j < P36_N_FF_EXP; j++) mid[j] = p36_siluf(gate[j]) * up[j];
        if (!p36_tensor_expert_matvec(e, l->ffn_down_exps, top_idx[i], mid, gate, rowbuf, P36_N_FF_EXP, P36_N_EMBD)) return false;
        p36_scale_inplace(gate, P36_N_EMBD,
                          p36_tensor_index_or(&e->model, l->ffn_down_exps_scale, top_idx[i], 1.0f));
        for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += gate[j] * top_w[i];
    }
    {
        float gate_scale = p36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = p36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        p36_activation_quant_kind shared_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!p36_quantize_activation_row(shared_kind, inp, inpq, P36_N_EMBD)) return false;
            xq = shared_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
            inpq_kind = shared_kind;
        }
        if (!p36_tensor_matvec_pair_prequant(e,
                                             l->ffn_gate_shexp,
                                             l->ffn_up_shexp,
                                             inp,
                                             xq,
                                             gate,
                                             up,
                                             rowbuf,
                                             P36_N_EMBD,
                                             P36_N_FF_SHARED,
                                             gate_scale,
                                             up_scale)) {
            if (!p36_tensor_matvec_prequant(e, l->ffn_gate_shexp, inp, xq,
                                            gate, rowbuf, P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
            p36_scale_inplace(gate, P36_N_FF_SHARED, gate_scale);
            if (!p36_tensor_matvec_prequant(e, l->ffn_up_shexp, inp, xq,
                                            up, rowbuf, P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
            p36_scale_inplace(up, P36_N_FF_SHARED, up_scale);
        }
    }
    for (uint32_t j = 0; j < P36_N_FF_SHARED; j++)
        mid[j] = gate[j] / (1.0f + expf(-gate[j])) * up[j];
    if (!p36_tensor_matvec(e, l->ffn_down_shexp, mid, gate, rowbuf, P36_N_FF_SHARED, P36_N_EMBD)) return false;
    p36_scale_inplace(gate, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f));
    if (!p36_tensor_get_plain(&e->model, l->ffn_gate_inp_shexp, rowbuf, P36_N_EMBD)) return false;
    {
        double acc = 0.0;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) acc += (double)rowbuf[i] * (double)inp[i];
        up[0] = (float)acc;
    }
    shared_gate = p36_sigmoidf(up[0]);
    for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += gate[j] * shared_gate;
    return true;
}

static bool p36_forward_full_attn(const p36_engine *e, const p36_layer_weights *l, uint32_t il, uint32_t pos,
                                  const float *inp, float *out, p36_cpu_runtime *rt) {
    float *qg = rt->work5;
    float *q = rt->work0;
    float *k = rt->work1;
    float *v = rt->work2;
    float *acc = rt->work3;
    float sinks[P36_N_HEAD];
    p36_full_attn_cache *cache = &rt->full[il];
    bool have_sinks = false;
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_q) ? l->attn_q :
                                (p36_hadamard_rotate_enabled(l->attn_k) ? l->attn_k :
                                (p36_hadamard_rotate_enabled(l->attn_v) ? l->attn_v : NULL));
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
    if (!p36_tensor_matvec(e, l->attn_q, qkv_in, qg, rt->work4, P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u)) {
        fprintf(stderr, "p36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u,
                      p36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_k, qkv_in, k, rt->work4, P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM)) {
        fprintf(stderr, "p36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(k, P36_N_HEAD_KV * P36_N_HEAD_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_v, qkv_in, v, rt->work4, P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM)) {
        fprintf(stderr, "p36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(v, P36_N_HEAD_KV * P36_N_VALUE_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f));
    for (uint32_t h = 0; h < P36_N_HEAD; h++) {
        p36_ref_rms_norm(q + (uint64_t)h * P36_N_HEAD_DIM,
                         qg + (uint64_t)h * P36_N_HEAD_DIM * 2u,
                         (const float *)(e->model.map + l->attn_q_norm->abs_offset),
                         P36_N_HEAD_DIM,
                         P36_RMS_EPS);
        p36_apply_rope_one(q + (uint64_t)h * P36_N_HEAD_DIM, P36_N_HEAD_DIM, pos);
    }
    for (uint32_t h = 0; h < P36_N_HEAD_KV; h++) {
        p36_ref_rms_norm(k + (uint64_t)h * P36_N_HEAD_DIM,
                         k + (uint64_t)h * P36_N_HEAD_DIM,
                         (const float *)(e->model.map + l->attn_k_norm->abs_offset),
                         P36_N_HEAD_DIM,
                         P36_RMS_EPS);
        p36_apply_rope_one(k + (uint64_t)h * P36_N_HEAD_DIM, P36_N_HEAD_DIM, pos);
    }
    if (l->attn_sinks) have_sinks = p36_tensor_get_plain(&e->model, l->attn_sinks, sinks, P36_N_HEAD);
    p36_full_attn_cache_store(cache, pos, k, v);
    for (uint32_t h = 0; h < P36_N_HEAD; h++) {
        uint32_t kvh = h / (P36_N_HEAD / P36_N_HEAD_KV);
        float *head_out = acc + (uint64_t)h * P36_N_VALUE_DIM;
        float max_score = have_sinks ? sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)P36_N_VALUE_DIM * sizeof(float));
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *kc = cache->k + (uint64_t)t * cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(cache->type_k, P36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t i = 0; i < P36_N_HEAD_DIM; i++) dot += (double)q[(uint64_t)h * P36_N_HEAD_DIM + i] * (double)p36_kv_cache_at(kc, cache->type_k, i);
            rt->scores[t] = (float)(dot / sqrt((double)P36_N_HEAD_DIM));
            if (rt->scores[t] > max_score) max_score = rt->scores[t];
        }
        if (have_sinks) denom = exp((double)sinks[h] - (double)max_score);
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *vc = cache->v + (uint64_t)t * cache->v_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(cache->type_v, P36_N_VALUE_DIM);
            float w = expf(rt->scores[t] - max_score);
            denom += w;
            for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) head_out[i] += w * p36_kv_cache_at(vc, cache->type_v, i);
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) head_out[i] *= inv;
        }
        for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) {
            uint64_t gate_off = (uint64_t)h * P36_N_HEAD_DIM * 2u + P36_N_HEAD_DIM + i;
            head_out[i] *= p36_sigmoidf(qg[gate_off]);
        }
    }
    {
        const float *ao_in = p36_hadamard_forward_rows(l->attn_output, acc, rt->batch_had, P36_N_SSM_INNER, 1u);
        if (!p36_tensor_matvec(e, l->attn_output, ao_in, out, rt->work4, P36_N_SSM_INNER, P36_N_EMBD)) {
            fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos);
            return false;
        }
    }
    p36_scale_inplace(out, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
    return true;
}

static bool p36_forward_recurrent(const p36_engine *e, const p36_layer_weights *l, uint32_t il,
                                  const float *inp, float *out, p36_cpu_runtime *rt) {
    p36_recurrent_cache *cache = &rt->recurrent[il];
    float *qkv = rt->work5;
    float *window = rt->work4;
    float *conv = rt->work3;
    float *z = rt->work2;
    float *beta = rt->work1;
    float *q = rt->work5;
    float *k = rt->work4;
    float *v = rt->work0;
    float *proj = rt->work3;
    float *gate = rt->work1 + P36_N_SSM_DT_RANK;
    const float *kernel = (const float *)(e->model.map + l->ssm_conv1d->abs_offset);
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_qkv) ? l->attn_qkv :
                                (p36_hadamard_rotate_enabled(l->attn_gate) ? l->attn_gate : NULL);
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
    if (!p36_tensor_matvec(e, l->attn_qkv, qkv_in, qkv, z, P36_N_EMBD, P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent attn_qkv failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(qkv, P36_N_SSM_CONV_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f));
    p36_recurrent_conv_step(cache, qkv, window);
    p36_ssm_conv_apply(window, kernel, conv);
    for (uint32_t i = 0; i < P36_N_SSM_CONV_DIM; i++) conv[i] = p36_siluf(conv[i]);
    if (!p36_all_finite(conv, P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_tensor_matvec(e, l->ssm_beta, inp, beta, z, P36_N_EMBD, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent ssm_beta failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(beta, P36_N_SSM_DT_RANK,
                      p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->ssm_alpha, inp, gate, z, P36_N_EMBD, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent ssm_alpha failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(gate, P36_N_SSM_DT_RANK,
                      p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_gate, qkv_in, z, window, P36_N_EMBD, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent attn_gate failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(z, P36_N_SSM_INNER,
                      p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        uint32_t gh = h % P36_N_SSM_GROUP;
        memcpy(q + (uint64_t)h * P36_N_SSM_STATE,
               conv + (uint64_t)gh * P36_N_SSM_STATE,
               (size_t)P36_N_SSM_STATE * sizeof(float));
        memcpy(k + (uint64_t)h * P36_N_SSM_STATE,
               conv + (uint64_t)P36_N_SSM_QK + (uint64_t)gh * P36_N_SSM_STATE,
               (size_t)P36_N_SSM_STATE * sizeof(float));
    }
    memcpy(v, conv + (uint64_t)P36_N_SSM_QK * 2u,
           (size_t)P36_N_SSM_DT_RANK * P36_N_SSM_STATE * sizeof(float));
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        p36_l2_norm(q + (uint64_t)h * P36_N_SSM_STATE, P36_N_SSM_STATE, P36_RMS_EPS);
        p36_l2_norm(k + (uint64_t)h * P36_N_SSM_STATE, P36_N_SSM_STATE, P36_RMS_EPS);
        beta[h] = p36_sigmoidf(beta[h]);
        gate[h] = p36_softplusf(gate[h] + ((const float *)(e->model.map + l->ssm_dt->abs_offset))[h]) * ((const float *)(e->model.map + l->ssm_a->abs_offset))[h];
    }
    if (!p36_all_finite(beta, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent beta non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_all_finite(gate, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    p36_delta_net_decode(cache->state, q, k, v, gate, beta, proj);
    if (!p36_all_finite(proj, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        p36_ref_rms_norm(proj + (uint64_t)h * P36_N_SSM_STATE,
                         proj + (uint64_t)h * P36_N_SSM_STATE,
                         (const float *)(e->model.map + l->ssm_norm->abs_offset),
                         P36_N_SSM_STATE,
                         P36_RMS_EPS);
    }
    for (uint32_t i = 0; i < P36_N_SSM_INNER; i++) proj[i] *= p36_siluf(z[i]);
    if (!p36_all_finite(proj, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    {
        const float *ssm_in = p36_hadamard_forward_rows(l->ssm_out, proj, rt->batch_had, P36_N_SSM_INNER, 1u);
        if (!p36_tensor_matvec(e, l->ssm_out, ssm_in, out, q, P36_N_SSM_INNER, P36_N_EMBD)) {
            fprintf(stderr, "p36: recurrent ssm_out failed at layer=%u\n", il);
            return false;
        }
    }
    p36_scale_inplace(out, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
    return true;
}

typedef struct {
    float *q;
    const float *qg;
    const float *weight;
    uint32_t pos0;
} p36_full_attn_q_rows_ctx;

typedef struct {
    float *k;
    const float *weight;
    uint32_t pos0;
} p36_full_attn_k_rows_ctx;

typedef struct {
    float *out;
    const float *q;
    const float *qg;
    const p36_full_attn_cache *cache;
    const float *sinks;
    uint32_t pos0;
    bool have_sinks;
} p36_full_attn_scores_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_layer_weights *l;
    const float *inp;
    float *gate_logits;
    const uint8_t *xq;
    uint32_t xq_row_bytes;
} p36_ffn_routed_batch_ctx;

typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} p36_ffn_combine_batch_ctx;

static void p36_full_attn_q_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_q_rows_ctx *ctx = (p36_full_attn_q_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD);
        const float *src = ctx->qg + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u;
        float *dst = ctx->q + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        p36_ref_rms_norm(dst, src, ctx->weight, P36_N_HEAD_DIM, P36_RMS_EPS);
        p36_apply_rope_one(dst, P36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void p36_full_attn_k_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_k_rows_ctx *ctx = (p36_full_attn_k_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD_KV);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD_KV);
        float *dst = ctx->k + (uint64_t)t * P36_N_HEAD_KV * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        p36_ref_rms_norm(dst, dst, ctx->weight, P36_N_HEAD_DIM, P36_RMS_EPS);
        p36_apply_rope_one(dst, P36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void p36_full_attn_scores_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_scores_ctx *ctx = (p36_full_attn_scores_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD);
        uint32_t kvh = h / (P36_N_HEAD / P36_N_HEAD_KV);
        uint32_t pos = ctx->pos0 + t;
        const float *qh = ctx->q + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        const float *gate = ctx->qg + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u + P36_N_HEAD_DIM;
        float *head_out = ctx->out + ((uint64_t)t * P36_N_HEAD + h) * P36_N_VALUE_DIM;
        float max_score = ctx->have_sinks ? ctx->sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)P36_N_VALUE_DIM * sizeof(float));
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_k, P36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < P36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)p36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float score = (float)(dot / sqrt((double)P36_N_HEAD_DIM));
                if (score > max_score) max_score = score;
            }
        }
        if (ctx->have_sinks) denom = exp((double)ctx->sinks[h] - (double)max_score);
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_k, P36_N_HEAD_DIM);
            const uint8_t *vc = ctx->cache->v + (uint64_t)i * ctx->cache->v_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_v, P36_N_VALUE_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < P36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)p36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float w = expf((float)(dot / sqrt((double)P36_N_HEAD_DIM)) - max_score);
                denom += w;
                for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] += w * p36_kv_cache_at(vc, ctx->cache->type_v, j);
            }
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] *= inv;
        }
        for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] *= p36_sigmoidf(gate[j]);
    }
}

static void p36_ffn_routed_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_ffn_routed_batch_ctx *ctx = (p36_ffn_routed_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float top_val[P36_N_EXPERT_USED];
        float top_w[P36_N_EXPERT_USED];
        uint32_t top_idx[P36_N_EXPERT_USED];
        float gate[P36_N_FF_EXP];
        float up[P36_N_FF_EXP];
        float mid[P36_N_FF_EXP];
        float down[P36_N_EMBD];
        float rowbuf[P36_N_SSM_INNER];
        float *logits = ctx->gate_logits + t * P36_N_EXPERT;
        float *out = ctx->out + t * P36_N_EMBD;
        const float *inp = ctx->inp + t * P36_N_EMBD;
        const uint8_t *inpq = ctx->xq ? ctx->xq + (uint64_t)t * ctx->xq_row_bytes : NULL;
        p36_softmax_inplace(logits, P36_N_EXPERT);
        p36_topk_indices(logits, P36_N_EXPERT, P36_N_EXPERT_USED, top_idx, top_val);
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) sum += top_val[i];
            if (sum < 6.103515625e-5) sum = 6.103515625e-5;
            for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * ctx->e->expert_weights_scale;
        }
        memset(out, 0, (size_t)P36_N_EMBD * sizeof(float));
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) {
            float gate_scale = p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_gate_exps_scale, top_idx[i], 1.0f);
            float up_scale = p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_up_exps_scale, top_idx[i], 1.0f);
            if (!p36_tensor_expert_matvec_pair_prequant(ctx->e,
                                                        ctx->l->ffn_gate_exps,
                                                        ctx->l->ffn_up_exps,
                                                        top_idx[i],
                                                        inp,
                                                        inpq,
                                                        gate,
                                                        up,
                                                        rowbuf,
                                                        P36_N_EMBD,
                                                        P36_N_FF_EXP,
                                                        gate_scale,
                                                        up_scale)) {
                if (!p36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_gate_exps, top_idx[i], inp, inpq,
                                                       gate, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                    continue;
                }
                p36_scale_inplace(gate, P36_N_FF_EXP, gate_scale);
                if (!p36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_up_exps, top_idx[i], inp, inpq,
                                                       up, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                    continue;
                }
                p36_scale_inplace(up, P36_N_FF_EXP, up_scale);
            }
            for (uint32_t j = 0; j < P36_N_FF_EXP; j++) mid[j] = p36_siluf(gate[j]) * up[j];
            if (!p36_tensor_expert_matvec(ctx->e, ctx->l->ffn_down_exps, top_idx[i], mid, down, rowbuf, P36_N_FF_EXP, P36_N_EMBD)) continue;
            p36_scale_inplace(down, P36_N_EMBD,
                              p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_down_exps_scale, top_idx[i], 1.0f));
            for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += down[j] * top_w[i];
        }
    }
}

static void p36_ffn_combine_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_ffn_combine_batch_ctx *ctx = (p36_ffn_combine_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float g = p36_sigmoidf(ctx->scalar[t]);
        float *out = ctx->out + t * P36_N_EMBD;
        const float *shared = ctx->shared + t * P36_N_EMBD;
        for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += shared[j] * g;
    }
}

static bool p36_forward_ffn_batch(const p36_engine *e,
                                  const p36_layer_weights *l,
                                  const float *inp,
                                  float *out,
                                  uint32_t n_tok,
                                  p36_cpu_runtime *rt) {
    p36_ffn_routed_batch_ctx routed_ctx;
    p36_ffn_combine_batch_ctx combine_ctx;
    const float *shared_gate_inp;
    const uint8_t *gate_inp_xq = NULL;
    p36_activation_quant_kind inpq_kind;
    if (P36_MODEL_DENSE) {
        const p36_tensor *rot_gate = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ? l->ffn_gate_shexp :
                                     (p36_hadamard_rotate_enabled(l->ffn_up_shexp) ? l->ffn_up_shexp : NULL);
        const float *ffn_in = rot_gate ? p36_hadamard_forward_rows(rot_gate, inp, rt->batch_had, P36_N_EMBD, n_tok) : inp;
        const float *down_in;
        inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (!p36_quantize_activation_batch(inpq_kind, ffn_in, rt->batch_xq,
                                           n_tok, P36_N_EMBD, e->n_threads)) return false;
        gate_inp_xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : rt->batch_xq;
        if (!p36_tensor_matmul_pair_batch_prequant(
                e, l->ffn_gate_shexp, l->ffn_up_shexp,
                ffn_in, gate_inp_xq,
                rt->batch_ffn_shared_gate, rt->batch_ffn_shared_up,
                n_tok, P36_N_EMBD, P36_N_FF_SHARED, 1.0f, 1.0f)) {
            if (!p36_tensor_matmul_batch_prequant(
                    e, l->ffn_gate_shexp, ffn_in, gate_inp_xq,
                    rt->batch_ffn_shared_gate, n_tok,
                    P36_N_EMBD, P36_N_FF_SHARED, 1.0f) ||
                !p36_tensor_matmul_batch_prequant(
                    e, l->ffn_up_shexp, ffn_in, gate_inp_xq,
                    rt->batch_ffn_shared_up, n_tok,
                    P36_N_EMBD, P36_N_FF_SHARED, 1.0f)) {
                return false;
            }
        }
        p36_swiglu_rows(rt->batch_ffn_shared_mid,
                        rt->batch_ffn_shared_gate,
                        rt->batch_ffn_shared_up,
                        n_tok, P36_N_FF_SHARED, e->n_threads);
        down_in = p36_hadamard_forward_rows(l->ffn_down_shexp, rt->batch_ffn_shared_mid,
                                            rt->batch_had, P36_N_FF_SHARED, n_tok);
        if (!p36_quantize_activation_batch_for_type(
                l->ffn_down_shexp->type, down_in,
                rt->batch_xq, n_tok, P36_N_FF_SHARED, e->n_threads)) {
            return false;
        }
        return p36_tensor_matmul_batch_prequant(
                e, l->ffn_down_shexp, down_in,
                rt->batch_xq, out, n_tok,
                P36_N_FF_SHARED, P36_N_EMBD, 1.0f);
    }
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_inp->type);
    if (inpq_kind != P36_ACTIVATION_QUANT_NONE) {
        if (!p36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
        gate_inp_xq = rt->batch_xq;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ffn_gate_inp, inp, gate_inp_xq,
                                          rt->batch_ffn_gate_logits,
                                          n_tok, P36_N_EMBD, P36_N_EXPERT, 1.0f)) return false;
    routed_ctx.out = out;
    routed_ctx.e = e;
    routed_ctx.l = l;
    routed_ctx.inp = inp;
    routed_ctx.gate_logits = rt->batch_ffn_gate_logits;
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!p36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    routed_ctx.xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : rt->batch_xq;
    routed_ctx.xq_row_bytes = p36_activation_quant_row_bytes(inpq_kind, P36_N_EMBD);
    p36_parallel_for_rows(n_tok, 2, e->n_threads, p36_ffn_routed_batch_worker, &routed_ctx);

    {
        float gate_scale = p36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = p36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        p36_activation_quant_kind shared_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!p36_quantize_activation_batch(shared_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
            inpq_kind = shared_kind;
        }
        if (!p36_tensor_matmul_pair_batch_prequant(e,
                                                   l->ffn_gate_shexp,
                                                   l->ffn_up_shexp,
                                                   inp,
                                                   rt->batch_xq,
                                                   rt->batch_ffn_shared_gate,
                                                   rt->batch_ffn_shared_up,
                                                   n_tok,
                                                   P36_N_EMBD,
                                                   P36_N_FF_SHARED,
                                                   gate_scale,
                                                   up_scale)) {
            if (!p36_tensor_matmul_batch_prequant(e, l->ffn_gate_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_gate,
                                                  n_tok, P36_N_EMBD, P36_N_FF_SHARED,
                                                  gate_scale)) {
                return false;
            }
            if (!p36_tensor_matmul_batch_prequant(e, l->ffn_up_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_up,
                                                  n_tok, P36_N_EMBD, P36_N_FF_SHARED,
                                                  up_scale)) {
                return false;
            }
        }
    }
    p36_swiglu_rows(rt->batch_ffn_shared_mid,
                    rt->batch_ffn_shared_gate,
                    rt->batch_ffn_shared_up,
                    n_tok,
                    P36_N_FF_SHARED,
                    e->n_threads);

    if (!p36_quantize_activation_batch_for_type(l->ffn_down_shexp->type, rt->batch_ffn_shared_mid,
                                               rt->batch_xq, n_tok, P36_N_FF_SHARED, e->n_threads)) {
        return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ffn_down_shexp, rt->batch_ffn_shared_mid, rt->batch_xq,
                                          rt->batch_ffn_shared_out,
                                          n_tok, P36_N_FF_SHARED, P36_N_EMBD,
                                          p36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }

    shared_gate_inp = (const float *)(e->model.map + l->ffn_gate_inp_shexp->abs_offset);
    p36_tensor_vector_dot_batch(rt->batch_ffn_scalar, shared_gate_inp, inp, n_tok, P36_N_EMBD, e->n_threads);
    combine_ctx.out = out;
    combine_ctx.shared = rt->batch_ffn_shared_out;
    combine_ctx.scalar = rt->batch_ffn_scalar;
    p36_parallel_for_rows(n_tok, 2, e->n_threads, p36_ffn_combine_batch_worker, &combine_ctx);
    return true;
}

static bool p36_forward_full_attn_batch(const p36_engine *e,
                                        const p36_layer_weights *l,
                                        uint32_t il,
                                        uint32_t pos0,
                                        const float *inp,
                                        float *out,
                                        uint32_t n_tok,
                                        p36_cpu_runtime *rt) {
    p36_full_attn_q_rows_ctx qctx;
    p36_full_attn_k_rows_ctx kctx;
    p36_full_attn_scores_ctx sctx;
    p36_full_attn_cache *cache = &rt->full[il];
    float sinks[P36_N_HEAD];
    bool have_sinks = false;
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_q) ? l->attn_q :
                                (p36_hadamard_rotate_enabled(l->attn_k) ? l->attn_k :
                                (p36_hadamard_rotate_enabled(l->attn_v) ? l->attn_v : NULL));
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, n_tok) : inp;
    p36_activation_quant_kind xq_kind = p36_activation_quant_kind_for_type(l->attn_q->type);
    if (!p36_quantize_activation_batch(xq_kind, qkv_in, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    if (!p36_tensor_matmul_batch_prequant(e, l->attn_q, qkv_in, rt->batch_xq,
                                          rt->batch_qg,
                                          n_tok, P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u,
                                          p36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f))) {
        return false;
    }
    if (p36_activation_quant_kind_for_type(l->attn_k->type) != xq_kind) {
        xq_kind = p36_activation_quant_kind_for_type(l->attn_k->type);
        if (!p36_quantize_activation_batch(xq_kind, qkv_in, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->attn_k, qkv_in, rt->batch_xq,
                                          rt->batch_k,
                                          n_tok, P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM,
                                          p36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f))) {
        return false;
    }
    if (p36_activation_quant_kind_for_type(l->attn_v->type) != xq_kind) {
        xq_kind = p36_activation_quant_kind_for_type(l->attn_v->type);
        if (!p36_quantize_activation_batch(xq_kind, qkv_in, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->attn_v, qkv_in, rt->batch_xq,
                                          rt->batch_v,
                                          n_tok, P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM,
                                          p36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f))) {
        return false;
    }
    qctx.q = rt->batch_q;
    qctx.qg = rt->batch_qg;
    qctx.weight = (const float *)(e->model.map + l->attn_q_norm->abs_offset);
    qctx.pos0 = pos0;
    p36_parallel_for_rows((uint64_t)n_tok * P36_N_HEAD, 1, e->n_threads, p36_full_attn_q_rows_worker, &qctx);
    kctx.k = rt->batch_k;
    kctx.weight = (const float *)(e->model.map + l->attn_k_norm->abs_offset);
    kctx.pos0 = pos0;
    p36_parallel_for_rows((uint64_t)n_tok * P36_N_HEAD_KV, 1, e->n_threads, p36_full_attn_k_rows_worker, &kctx);
    if (l->attn_sinks) have_sinks = p36_tensor_get_plain(&e->model, l->attn_sinks, sinks, P36_N_HEAD);
    for (uint32_t t = 0; t < n_tok; t++) {
        p36_full_attn_cache_store(cache,
                                  pos0 + t,
                                  rt->batch_k + (uint64_t)t * P36_N_HEAD_KV * P36_N_HEAD_DIM,
                                  rt->batch_v + (uint64_t)t * P36_N_HEAD_KV * P36_N_VALUE_DIM);
    }
    sctx.out = rt->batch_attn_out;
    sctx.q = rt->batch_q;
    sctx.qg = rt->batch_qg;
    sctx.cache = cache;
    sctx.sinks = sinks;
    sctx.pos0 = pos0;
    sctx.have_sinks = have_sinks;
    p36_parallel_for_rows((uint64_t)n_tok * P36_N_HEAD, 1, e->n_threads, p36_full_attn_scores_worker, &sctx);
    {
        const float *ao_in = p36_hadamard_forward_rows(l->attn_output, rt->batch_attn_out,
                                                       rt->batch_had, P36_N_SSM_INNER, n_tok);
        if (!p36_quantize_activation_batch_for_type(l->attn_output->type, ao_in,
                                                   rt->batch_xq, n_tok, P36_N_SSM_INNER, e->n_threads)) {
            return false;
        }
        if (!p36_tensor_matmul_batch_prequant(e, l->attn_output, ao_in, rt->batch_xq,
                                              out,
                                              n_tok, P36_N_SSM_INNER, P36_N_EMBD,
                                              p36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f))) {
            return false;
        }
    }
    return true;
}

static bool p36_forward_recurrent_batch(const p36_engine *e,
                                        const p36_layer_weights *l,
                                        uint32_t il,
                                        const float *inp,
                                        float *out,
                                        uint32_t n_tok,
                                        p36_cpu_runtime *rt) {
    p36_recurrent_cache *cache = &rt->recurrent[il];
    const float *kernel = (const float *)(e->model.map + l->ssm_conv1d->abs_offset);
    const float *dt_bias = (const float *)(e->model.map + l->ssm_dt->abs_offset);
    const float *a = (const float *)(e->model.map + l->ssm_a->abs_offset);
    const float *ssm_norm = (const float *)(e->model.map + l->ssm_norm->abs_offset);
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_qkv) ? l->attn_qkv :
                                (p36_hadamard_rotate_enabled(l->attn_gate) ? l->attn_gate : NULL);
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, n_tok) : inp;
    p36_activation_quant_kind xq_kind = p36_activation_quant_kind_for_type(l->attn_qkv->type);
    if (!p36_quantize_activation_batch(xq_kind, qkv_in, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    if (!p36_tensor_matmul_batch_prequant(e, l->attn_qkv, qkv_in, rt->batch_xq,
                                          rt->batch_recur_qkv,
                                          n_tok, P36_N_EMBD, P36_N_SSM_CONV_DIM,
                                          p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f))) {
        return false;
    }
    if (p36_activation_quant_kind_for_type(l->attn_gate->type) != xq_kind) {
        xq_kind = p36_activation_quant_kind_for_type(l->attn_gate->type);
        if (!p36_quantize_activation_batch(xq_kind, qkv_in, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->attn_gate, qkv_in, rt->batch_xq,
                                          rt->batch_recur_z,
                                          n_tok, P36_N_EMBD, P36_N_SSM_INNER,
                                          p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f))) {
        return false;
    }
    if (p36_activation_quant_kind_for_type(l->ssm_beta->type) != xq_kind) {
        xq_kind = p36_activation_quant_kind_for_type(l->ssm_beta->type);
        if (!p36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ssm_beta, inp, rt->batch_xq,
                                          rt->batch_recur_beta,
                                          n_tok, P36_N_EMBD, P36_N_SSM_DT_RANK,
                                          p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f))) {
        return false;
    }
    if (p36_activation_quant_kind_for_type(l->ssm_alpha->type) != xq_kind) {
        xq_kind = p36_activation_quant_kind_for_type(l->ssm_alpha->type);
        if (!p36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ssm_alpha, inp, rt->batch_xq,
                                          rt->batch_recur_alpha,
                                          n_tok, P36_N_EMBD, P36_N_SSM_DT_RANK,
                                          p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        return false;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        float *qkv = rt->batch_recur_qkv + (uint64_t)t * P36_N_SSM_CONV_DIM;
        float *z = rt->batch_recur_z + (uint64_t)t * P36_N_SSM_INNER;
        float *beta = rt->batch_recur_beta + (uint64_t)t * P36_N_SSM_DT_RANK;
        float *gate = rt->batch_recur_alpha + (uint64_t)t * P36_N_SSM_DT_RANK;
        float *window = rt->work4;
        float *conv = rt->work3;
        float *q = rt->work5;
        float *k = rt->work4;
        float *v = rt->work0;
        float *proj = rt->batch_recur_proj + (uint64_t)t * P36_N_SSM_INNER;
        p36_recurrent_conv_step(cache, qkv, window);
        p36_ssm_conv_apply(window, kernel, conv);
        for (uint32_t i = 0; i < P36_N_SSM_CONV_DIM; i++) conv[i] = p36_siluf(conv[i]);
        if (!p36_all_finite(conv, P36_N_SSM_CONV_DIM)) return false;
        for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
            uint32_t gh = h % P36_N_SSM_GROUP;
            memcpy(q + (uint64_t)h * P36_N_SSM_STATE,
                   conv + (uint64_t)gh * P36_N_SSM_STATE,
                   (size_t)P36_N_SSM_STATE * sizeof(float));
            memcpy(k + (uint64_t)h * P36_N_SSM_STATE,
                   conv + (uint64_t)P36_N_SSM_QK + (uint64_t)gh * P36_N_SSM_STATE,
                   (size_t)P36_N_SSM_STATE * sizeof(float));
        }
        memcpy(v, conv + (uint64_t)P36_N_SSM_QK * 2u,
               (size_t)P36_N_SSM_DT_RANK * P36_N_SSM_STATE * sizeof(float));
        for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
            p36_l2_norm(q + (uint64_t)h * P36_N_SSM_STATE, P36_N_SSM_STATE, P36_RMS_EPS);
            p36_l2_norm(k + (uint64_t)h * P36_N_SSM_STATE, P36_N_SSM_STATE, P36_RMS_EPS);
            beta[h] = p36_sigmoidf(beta[h]);
            gate[h] = p36_softplusf(gate[h] + dt_bias[h]) * a[h];
        }
        if (!p36_all_finite(beta, P36_N_SSM_DT_RANK)) return false;
        if (!p36_all_finite(gate, P36_N_SSM_DT_RANK)) return false;
        p36_delta_net_decode(cache->state, q, k, v, gate, beta, proj);
        if (!p36_all_finite(proj, P36_N_SSM_INNER)) return false;
        for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
            p36_ref_rms_norm(proj + (uint64_t)h * P36_N_SSM_STATE,
                             proj + (uint64_t)h * P36_N_SSM_STATE,
                             ssm_norm,
                             P36_N_SSM_STATE,
                             P36_RMS_EPS);
        }
        for (uint32_t i = 0; i < P36_N_SSM_INNER; i++) proj[i] *= p36_siluf(z[i]);
        if (!p36_all_finite(proj, P36_N_SSM_INNER)) return false;
    }
    {
        const float *ssm_in = p36_hadamard_forward_rows(l->ssm_out, rt->batch_recur_proj,
                                                        rt->batch_had, P36_N_SSM_INNER, n_tok);
        if (!p36_quantize_activation_batch_for_type(l->ssm_out->type, ssm_in,
                                                   rt->batch_xq, n_tok, P36_N_SSM_INNER, e->n_threads)) {
            return false;
        }
        if (!p36_tensor_matmul_batch_prequant(e, l->ssm_out, ssm_in, rt->batch_xq,
                                              out,
                                              n_tok, P36_N_SSM_INNER, P36_N_EMBD,
                                              p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f))) {
            return false;
        }
    }
    return true;
}

static bool p36_forward_tokens_cpu(p36_session *s,
                                   const int *tokens,
                                   uint32_t n_tok,
                                   uint32_t pos0,
                                   bool compute_logits) {
    p36_engine *e;
    p36_cpu_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (p36_cpu_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        if (!p36_embed_token(e, tokens[t], rt->batch_hidden + (uint64_t)t * P36_N_EMBD)) {
            fprintf(stderr, "p36: embed failed for token=%d pos=%u\n", tokens[t], pos0 + t);
            return false;
        }
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        p36_rms_norm_rows(rt->batch_norm,
                          rt->batch_hidden,
                          (const float *)(e->model.map + l->attn_norm->abs_offset),
                          n_tok,
                          P36_N_EMBD,
                          P36_RMS_EPS,
                          e->n_threads);
        if (l->kind == P36_LAYER_FULL_ATTN) {
            if (!p36_forward_full_attn_batch(e, l, il, pos0, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "p36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        } else {
            if (!p36_forward_recurrent_batch(e, l, il, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "p36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        }
        p36_directional_steering_project_rows(rt->batch_next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_attn_scale);
        if (!p36_all_finite(rt->batch_next_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        p36_add_rows(rt->batch_next_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     P36_N_EMBD,
                     e->n_threads);
        p36_rms_norm_rows(rt->batch_norm,
                          rt->batch_next_hidden,
                          (const float *)(e->model.map + l->post_attention_norm->abs_offset),
                          n_tok,
                          P36_N_EMBD,
                          P36_RMS_EPS,
                          e->n_threads);
        if (!p36_forward_ffn_batch(e, l, rt->batch_norm, rt->batch_hidden, n_tok, rt)) {
            fprintf(stderr, "p36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        p36_directional_steering_project_rows(rt->batch_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_ffn_scale);
        if (!p36_all_finite(rt->batch_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        p36_add_rows(rt->batch_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     P36_N_EMBD,
                     e->n_threads);
        if (!p36_all_finite(rt->batch_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
    }
    p36_rms_norm_rows(rt->batch_norm,
                      rt->batch_hidden,
                      (const float *)(e->model.map + e->weights.output_norm->abs_offset),
                      n_tok,
                      P36_N_EMBD,
                      P36_RMS_EPS,
                      e->n_threads);
    if (!p36_all_finite(rt->batch_norm, n_tok * P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (compute_logits) {
        const float *last = rt->batch_norm + (uint64_t)(n_tok - 1u) * P36_N_EMBD;
        last = p36_hadamard_forward_rows(e->weights.output, last, rt->batch_had, P36_N_EMBD, 1u);
        if (!p36_tensor_matvec(e, e->weights.output, last, s->logits, rt->work1, P36_N_EMBD, P36_N_VOCAB)) {
            fprintf(stderr, "p36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        p36_scale_inplace(s->logits, P36_N_VOCAB,
                          p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!p36_all_finite(s->logits, P36_N_VOCAB)) {
            fprintf(stderr, "p36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return true;
}

#ifndef P36_NO_GPU
typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} p36_vulkan_ffn_tail_ctx;

static P36_MAYBE_UNUSED void p36_vulkan_ffn_tail_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vulkan_ffn_tail_ctx *ctx = (p36_vulkan_ffn_tail_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        float g = p36_sigmoidf(ctx->scalar[t]);
        const float *st = ctx->shared + t * P36_N_EMBD;
        float *ot = ctx->out + t * P36_N_EMBD;
        for (uint32_t j = 0; j < P36_N_EMBD; j++) ot[j] = fmaf(st[j], g, ot[j]);
    }
}

typedef struct {
    float *x;
    uint32_t n_head;
    uint32_t pos0;
    const uint32_t *positions;
} p36_vulkan_rope_ctx;

static void p36_vulkan_rope_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vulkan_rope_ctx *ctx = (p36_vulkan_rope_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        uint32_t pt = ctx->pos0 + (uint32_t)t;
        uint32_t ph = pt, pw = pt;
        if (ctx->positions) {
            pt = ctx->positions[t * 3u];
            ph = ctx->positions[t * 3u + 1u];
            pw = ctx->positions[t * 3u + 2u];
        }
        for (uint32_t h = 0; h < ctx->n_head; h++) {
            p36_apply_rope_axes(ctx->x + (t * ctx->n_head + h) * P36_N_HEAD_DIM,
                                P36_N_HEAD_DIM, pt, ph, pw);
        }
    }
}

static uint32_t p36_hadamard_total_signs(void) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        total += g_p36_hadamard.widths[i];
    }
    return total;
}

static int p36_hadamard_sign_offset(uint32_t width, uint32_t *offset) {
    uint32_t off = 0;
    if (!g_p36_hadamard.enabled) return 0;
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        if (g_p36_hadamard.widths[i] == width) {
            if (offset) *offset = off;
            return 1;
        }
        off += g_p36_hadamard.widths[i];
    }
    return 0;
}

/* Forward activation transform for a rotated weight: x' = H (s . x), quantized
 * directly to rt->had_q8 for the q8_K matmul path in a single fused dispatch.
 * rt->inp_q8 keeps the plain activation for the unrotated weights that share it. */
static bool p36_hadamard_prepare(p36_vulkan_runtime *rt,
                                 const p36_gpu_tensor *src,
                                 uint32_t width, uint32_t n_rows) {
    uint32_t offset = 0;
    if (!rt || !rt->had_q8 || !rt->had_signs) return false;
    if (!p36_hadamard_sign_offset(width, &offset)) return false;
    return p36_gpu_hadamard_prepare_tensor(rt->had_q8, src, rt->had_signs,
                                           width, n_rows, offset,
                                           p36_hadamard_total_signs(),
                                           g_p36_hadamard.inv_sqrt_block, 0);
}

/* Same, for a folded weight whose rotation axis kept the training V order
 * (prism.hadamard.gdn_v_grouped): the runtime activation is tiled, so permute
 * tiled -> grouped inside the fused kernel before the signs and the transform. */
static bool p36_hadamard_prepare_grouped(p36_vulkan_runtime *rt,
                                         const p36_gpu_tensor *src,
                                         uint32_t width, uint32_t n_rows) {
    uint32_t offset = 0;
    uint32_t n_v = P36_N_SSM_DT_RANK;
    uint32_t n_k = P36_N_SSM_GROUP;
    if (!rt || !rt->had_q8 || !rt->had_signs) return false;
    if (!n_k || !n_v || (n_v % n_k) != 0 || (width % n_v) != 0) return false;
    if (!p36_hadamard_sign_offset(width, &offset)) return false;
    return p36_gpu_hadamard_prepare_tensor(rt->had_q8, src, rt->had_signs,
                                           width, n_rows, offset,
                                           p36_hadamard_total_signs(),
                                           g_p36_hadamard.inv_sqrt_block, 1);
}

static bool p36_forward_ffn_vulkan_model(p36_vulkan_runtime *rt,
                                         const p36_model *m,
                                         const p36_layer_weights *l,
                                         float expert_weights_scale,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out,
                                         bool session_batch) {
    p36_gpu_moe_weight gate;
    p36_gpu_moe_weight up;
    p36_gpu_moe_weight down;
    const float *shared;
    const float *scalar;
    float *outp;
    if (!rt || !m || !l || !inp || !out) return false;
    if (P36_MODEL_DENSE) {
        const p36_gpu_tensor *ffn_in = inp;
        const p36_gpu_tensor *ffn_q8 = rt->inp_q8;
        bool gate_rot = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ||
                        p36_hadamard_rotate_enabled(l->ffn_up_shexp);
        if (gate_rot) {
            if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
            /* The fused prepare writes only had_q8; folded weights always take
             * the q8_K path, so ffn_in stays the f32 source and is unused.
             * ponytail: a folded f32/f16/q8_0 weight would need the prepare
             * kernel to also emit the f32 activation. */
            ffn_q8 = rt->had_q8;
        } else if (!rt->inp_q8_valid && !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
            return false;
        }
        rt->inp_q8_valid = false;
        bool pair_projected = false;
#ifdef P36_METAL
        const char *iq3_pair = getenv("P36_METAL_IQ3_PAIR");
        if (n_tok == 1u && (!iq3_pair || iq3_pair[0] != '0') &&
            l->ffn_gate_shexp->type == P36_TENSOR_IQ3_S &&
            l->ffn_up_shexp->type == P36_TENSOR_IQ3_S) {
            pair_projected = p36_gpu_matmul_iq3_s_pair_tensor(
                rt->ffn_shared_gate, rt->ffn_shared_up,
                m->map, m->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                P36_N_EMBD, P36_N_FF_SHARED, inp) != 0;
        } else if (n_tok == 1u && (!iq3_pair || iq3_pair[0] != '0') &&
                   l->ffn_gate_shexp->type == P36_TENSOR_IQ3_XXS &&
                   l->ffn_up_shexp->type == P36_TENSOR_IQ3_XXS) {
            pair_projected = p36_gpu_matmul_iq3_xxs_pair_tensor(
                rt->ffn_shared_gate, rt->ffn_shared_up,
                m->map, m->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                P36_N_EMBD, P36_N_FF_SHARED, inp) != 0;
        }
#endif
#ifndef P36_METAL
        /* Vulkan prefill pair: one dispatch decodes the gate and up tiles
         * against a single staged activation tile.  n_tok == 1 keeps the
         * per-row decode kernels, and any refusal falls through to the two
         * single-projection dispatches below. */
        {
            const char *iq3_pair = getenv("P36_VK_DENSE_IQ3_PAIR");
            if (!pair_projected && n_tok > 1u &&
                (!iq3_pair || iq3_pair[0] != '0') &&
                l->ffn_gate_shexp->type == P36_TENSOR_IQ3_XXS &&
                l->ffn_up_shexp->type == P36_TENSOR_IQ3_XXS) {
                pair_projected = p36_gpu_matmul_iq3_xxs_pair_mmq_tensor(
                    rt->ffn_shared_gate, rt->ffn_shared_up,
                    m->map, m->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    P36_N_EMBD, P36_N_FF_SHARED, ffn_q8, n_tok, 1.0f) != 0;
            }
        }
#endif
        if (!pair_projected &&
            (!p36_gpu_tensor_matmul_q8_or_float_scaled(
                 m, l->ffn_gate_shexp, ffn_in, ffn_q8,
                 rt->ffn_shared_gate, P36_N_EMBD, P36_N_FF_SHARED,
                 n_tok, 1.0f) ||
             !p36_gpu_tensor_matmul_q8_or_float_scaled(
                 m, l->ffn_up_shexp, ffn_in, ffn_q8,
                 rt->ffn_shared_up, P36_N_EMBD, P36_N_FF_SHARED,
                 n_tok, 1.0f))) {
            return false;
        }
        bool fused_swiglu_q8 = false;
#ifndef P36_METAL
        fused_swiglu_q8 = p36_gpu_swiglu_q8_k_tensor(
            rt->ffn_shared_mid, rt->inp_q8, rt->ffn_shared_gate,
            rt->ffn_shared_up, P36_N_FF_SHARED, n_tok, 0.0f, 1.0f);
#endif
        if (!fused_swiglu_q8 &&
            !p36_gpu_swiglu_tensor(
                 rt->ffn_shared_mid, rt->ffn_shared_gate,
                 rt->ffn_shared_up, n_tok * P36_N_FF_SHARED, 0.0f, 1.0f)) {
            return false;
        }
        const p36_gpu_tensor *down_in = rt->ffn_shared_mid;
        const p36_gpu_tensor *down_q8 = rt->inp_q8;
        if (p36_hadamard_rotate_enabled(l->ffn_down_shexp)) {
            if (!p36_hadamard_prepare(rt, rt->ffn_shared_mid, P36_N_FF_SHARED, n_tok)) {
                return false;
            }
            down_q8 = rt->had_q8;
        } else if (!fused_swiglu_q8 &&
                   !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, rt->ffn_shared_mid,
                                                 P36_N_FF_SHARED, n_tok)) {
            return false;
        }
        if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
                m, l->ffn_down_shexp, down_in, down_q8,
                out, P36_N_FF_SHARED, P36_N_EMBD, n_tok, 1.0f)) {
            return false;
        }
        return true;
    }
    gate.offset = l->ffn_gate_exps->abs_offset;
    gate.type = l->ffn_gate_exps->type;
    gate.scales_offset = l->ffn_gate_exps_scale ? l->ffn_gate_exps_scale->abs_offset : 0;
    gate.has_scales = l->ffn_gate_exps_scale != NULL;
    up.offset = l->ffn_up_exps->abs_offset;
    up.type = l->ffn_up_exps->type;
    up.scales_offset = l->ffn_up_exps_scale ? l->ffn_up_exps_scale->abs_offset : 0;
    up.has_scales = l->ffn_up_exps_scale != NULL;
    down.offset = l->ffn_down_exps->abs_offset;
    down.type = l->ffn_down_exps->type;
    down.scales_offset = l->ffn_down_exps_scale ? l->ffn_down_exps_scale->abs_offset : 0;
    down.has_scales = l->ffn_down_exps_scale != NULL;
    /* The router and the shared-expert gate scalar run on the host from the
     * host-written norm, and the shared-expert gate/up dispatches record
     * before the routed pass so the routed block's one flush completes them
     * too. */
    if (!p36_gpu_tensor_matmul_scaled(
            m, l->ffn_gate_inp, inp, rt->ffn_gate_logits,
            P36_N_EMBD, P36_N_EXPERT, n_tok, 1.0f) ||
        !p36_gpu_router_topk_tensor(
            rt->ffn_selected, rt->ffn_weights, rt->ffn_gate_logits,
            P36_N_EXPERT, P36_N_EXPERT_USED, n_tok,
            expert_weights_scale) ||
        !p36_gpu_tensor_matmul_scaled(
            m, l->ffn_gate_inp_shexp, inp, rt->ffn_scalar,
            P36_N_EMBD, 1, n_tok, 1.0f)) {
        return false;
    }
    /* Fast routed pass, decode and prefill: f32 activations straight into
     * the fused IQ2_XXS gate/up and Q2_K down kernels, expert schedule
     * built and consumed on GPU. When it runs, the q8_K input staging is
     * only needed for k-quant shared-expert projections, which this model
     * does not have. */
    bool routed_done =
        p36_gpu_moe_ffn_f32_tensor(out, m->map, m->size,
                                          &gate, &up, &down,
                                          rt->ffn_selected, rt->ffn_weights,
                                          il, P36_N_EXPERT_USED, inp, n_tok,
                                          P36_N_EMBD, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT) != 0;
    /* Diagnostic only: why a layer left the fused f32 expert path. */
    if (!routed_done && getenv("P36_VK_MOE_ROUTE_DEBUG")) {
        fprintf(stderr, "p36: moe f32 route miss il=%u n_tok=%u gate=%u up=%u down=%u\n",
                il, n_tok, gate.type, up.type, down.type);
    }
    if (routed_done && (n_tok == 1 || session_batch) &&
        l->ffn_gate_shexp->type == P36_TENSOR_Q8_0 &&
        l->ffn_up_shexp->type == P36_TENSOR_Q8_0 &&
        l->ffn_down_shexp->type == P36_TENSOR_Q8_0) {
        const char *env = getenv("P36_VK_SHARED_FFN_DECODE");
        if (!env || !env[0] || env[0] != '0') {
            if (n_tok == 1 && p36_gpu_shared_ffn_decode_tensor(
                    out, rt->ffn_shared_mid, inp, rt->ffn_scalar,
                    m->map, m->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    l->ffn_down_shexp->abs_offset,
                    P36_N_EMBD, P36_N_FF_SHARED, P36_N_EMBD,
                    p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                    p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                    p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
                return true;
            }
            bool ok = true;
            for (uint32_t row = 0; session_batch && row < n_tok && ok; row++) {
                p36_gpu_tensor *row_out = p36_gpu_tensor_view(
                    out, (uint64_t)row * P36_N_EMBD * sizeof(float),
                    (uint64_t)P36_N_EMBD * sizeof(float));
                p36_gpu_tensor *row_mid = p36_gpu_tensor_view(
                    rt->ffn_shared_mid,
                    (uint64_t)row * P36_N_FF_SHARED * sizeof(float),
                    (uint64_t)P36_N_FF_SHARED * sizeof(float));
                p36_gpu_tensor *row_inp = p36_gpu_tensor_view(
                    inp, (uint64_t)row * P36_N_EMBD * sizeof(float),
                    (uint64_t)P36_N_EMBD * sizeof(float));
                p36_gpu_tensor *row_scalar = p36_gpu_tensor_view(
                    rt->ffn_scalar, (uint64_t)row * sizeof(float),
                    sizeof(float));
                ok = row_out && row_mid && row_inp && row_scalar &&
                    p36_gpu_shared_ffn_decode_tensor(
                        row_out, row_mid, row_inp, row_scalar,
                        m->map, m->size,
                        l->ffn_gate_shexp->abs_offset,
                        l->ffn_up_shexp->abs_offset,
                        l->ffn_down_shexp->abs_offset,
                        P36_N_EMBD, P36_N_FF_SHARED, P36_N_EMBD,
                        p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                        p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                        p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f));
                p36_gpu_tensor_free(row_scalar);
                p36_gpu_tensor_free(row_inp);
                p36_gpu_tensor_free(row_mid);
                p36_gpu_tensor_free(row_out);
            }
            if (session_batch && ok) return true;
        }
    }
    if ((!routed_done ||
         l->ffn_gate_shexp->type != P36_TENSOR_Q8_0 ||
         l->ffn_up_shexp->type != P36_TENSOR_Q8_0) &&
        !rt->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_gate_shexp, inp, rt->inp_q8, rt->ffn_shared_gate,
                                                  P36_N_EMBD, P36_N_FF_SHARED, n_tok,
                                                  p36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f))) {
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_up_shexp, inp, rt->inp_q8, rt->ffn_shared_up,
                                                  P36_N_EMBD, P36_N_FF_SHARED, n_tok,
                                                  p36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f))) {
        return false;
    }
    if (!routed_done &&
        !p36_gpu_moe_ffn_q8_tensor(out, m->map, m->size,
                                   &gate, &up, &down,
                                   rt->ffn_selected, rt->ffn_weights,
                                   il, P36_N_EXPERT_USED, rt->inp_q8, n_tok,
                                   P36_N_EMBD, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT)) {
        return false;
    }
    if (!p36_gpu_swiglu_tensor(rt->ffn_shared_mid, rt->ffn_shared_gate, rt->ffn_shared_up,
                               n_tok * P36_N_FF_SHARED, 0.0f, 1.0f)) {
        return false;
    }
    if (!p36_gpu_tensor_matmul_scaled(m, l->ffn_down_shexp, rt->ffn_shared_mid, rt->ffn_shared_out,
                                      P36_N_FF_SHARED, P36_N_EMBD, n_tok,
                                      p36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }
    /* Shared-expert tail on the host: the CPU compiles "out += shared * g"
     * to one fma after the down projection's scale, so the fused rounding
     * must be reproduced with fmaf() on the unscaled-by-g down output. */
    (void)shared;
    (void)scalar;
    (void)outp;
    return p36_gpu_ffn_tail_tensor(out, rt->ffn_shared_out, rt->ffn_scalar, P36_N_EMBD, n_tok) != 0;
}

static bool p36_forward_ffn_vulkan(p36_session *s,
                                   const p36_layer_weights *l,
                                   uint32_t il,
                                   uint32_t n_tok,
                                   const p36_gpu_tensor *inp,
                                   p36_gpu_tensor *out) {
    p36_engine *e;
    if (!s) return false;
    e = s->engine;
    return p36_forward_ffn_vulkan_model((p36_vulkan_runtime *)s->runtime,
                                        &e->model,
                                        l,
                                        e->expert_weights_scale,
                                        il,
                                        n_tok,
                                        inp,
                                        out,
                                        false);
}

/* RoPE stays on the host even on the Vulkan path: the rotated values feed KV
 * storage, so they must match the CPU reference bit for bit, and only the
 * p36_apply_rope_one() body compiled in this unit guarantees that.
 * Token row t rotates at absolute position pos0 + t. */
static bool p36_vulkan_rope_heads(p36_gpu_tensor *x, uint32_t n_head,
                                  uint32_t pos0, uint32_t n_tok,
                                  const uint32_t *positions, bool quality) {
    const char *env = getenv("P36_VK_GPU_ROPE");
    bool use_gpu = !env || !env[0] || env[0] != '0';
    if (positions)
        return p36_gpu_rope_qwen_mrope_rows_tensor(x, n_head,
                                                    positions, n_tok) != 0;
    if (!positions && !quality && use_gpu)
        return p36_gpu_rope_qwen_rows_tensor(x, n_head, pos0, n_tok) != 0;
    p36_vulkan_rope_ctx ctx;
    float *xp;
    uint64_t bytes = (uint64_t)n_tok * n_head * P36_N_HEAD_DIM * sizeof(float);
    if (p36_gpu_tensor_bytes(x) < bytes) return false;
    xp = p36_gpu_tensor_contents_named(x, "submit_wait_host_rope");
    if (!xp) return false;
    ctx.x = xp;
    ctx.n_head = n_head;
    ctx.pos0 = pos0;
    ctx.positions = positions;
    if ((uint64_t)n_tok * n_head * P36_N_HEAD_DIM >= 4096u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vulkan_rope_rows, &ctx);
    else p36_vulkan_rope_rows(&ctx, 0, n_tok);
    return true;
}

static bool p36_vulkan_prepare_attn_heads(p36_gpu_tensor *dst,
                                          const p36_gpu_tensor *src,
                                          const p36_model *m,
                                          const p36_tensor *norm,
                                          uint32_t src_stride,
                                          uint32_t n_head,
                                          uint32_t pos0,
                                          uint32_t n_tok,
                                          const uint32_t *positions,
                                          bool quality) {
    const char *env = getenv("P36_VK_NORM_ROPE");
    bool fused = !env || !env[0] || env[0] != '0';
    if (!positions && !quality && fused) {
        return p36_gpu_rms_norm_rope_qwen_rows_tensor(dst, src,
                                                       m->map, m->size,
                                                       norm->abs_offset,
                                                       src_stride, n_head,
                                                       pos0, n_tok,
                                                       P36_RMS_EPS) != 0;
    }
    if (dst != src && !p36_gpu_extract_full_attn_q_tensor(dst, src, n_tok)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(dst, dst,
                                             m->map, m->size, norm->abs_offset,
                                             P36_N_HEAD_DIM, n_tok * n_head,
                                             P36_RMS_EPS)) {
        return false;
    }
    return p36_vulkan_rope_heads(dst, n_head, pos0, n_tok, positions, quality);
}

static bool p36_forward_full_attn_vulkan_model(p36_vulkan_runtime *rt,
                                               const p36_model *m,
                                               const p36_layer_weights *l,
                                               p36_vulkan_full_attn_cache *cache,
                                               bool quality,
                                               uint32_t il,
                                               uint32_t pos0,
                                               uint32_t rope_pos0,
                                               const uint32_t *positions,
                                               uint32_t n_tok,
                                               const p36_gpu_tensor *inp,
                                               p36_gpu_tensor *out) {
    if (!rt || !m || !l || !cache || !inp || !out) return false;
    const p36_gpu_tensor *qkv_in = inp;
    const p36_gpu_tensor *qkv_q8 = rt->inp_q8;
    if (p36_hadamard_rotate_enabled(l->attn_q) ||
        p36_hadamard_rotate_enabled(l->attn_k) ||
        p36_hadamard_rotate_enabled(l->attn_v)) {
        if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
        qkv_q8 = rt->had_q8;
    } else if ((l->attn_q->type != P36_TENSOR_Q8_0 || l->attn_k->type != P36_TENSOR_Q8_0 ||
                l->attn_v->type != P36_TENSOR_Q8_0) &&
               !rt->inp_q8_valid &&
               !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_q, qkv_in, qkv_q8, rt->attn_qg,
                                                  P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_q_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k, qkv_in, qkv_q8, rt->attn_k,
                                                  P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_k_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v, qkv_in, qkv_q8, rt->attn_v,
                                                  P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM, n_tok,
                                                  p36_tensor_scalar_or(m, l->attn_v_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!p36_vulkan_prepare_attn_heads(rt->attn_q, rt->attn_qg, m, l->attn_q_norm,
                                       P36_N_HEAD_DIM * 2u, P36_N_HEAD,
                                       rope_pos0, n_tok, positions, quality)) return false;
    {
        const char *env = getenv("P36_VK_NORM_ROPE_KV");
        bool fused_kv = !positions && rope_pos0 == pos0 && !quality &&
                        (!env || !env[0] || env[0] != '0') &&
                        cache->type_k == 0u && cache->type_v == 0u &&
                        cache->k_row_bytes == P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(uint16_t);
        if (fused_kv) {
            if (!p36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, rt->attn_k, rt->attn_v,
                    m->map, m->size, l->attn_k_norm->abs_offset,
                    P36_N_HEAD_DIM, P36_N_HEAD_KV,
                    pos0, n_tok, cache->cap, P36_RMS_EPS)) {
                return false;
            }
        } else if (!positions && rope_pos0 == pos0 && !quality &&
                   (!env || !env[0] || env[0] != '0') &&
                   cache->type_k == P36_KV_CACHE_Q8_0 &&
                   cache->type_v == P36_KV_CACHE_Q4_0 &&
                   (P36_N_HEAD_DIM % 32u) == 0u &&
                   (P36_N_VALUE_DIM % 32u) == 0u &&
                   p36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
                       cache->k, cache->v, rt->attn_k, rt->attn_v,
                       m->map, m->size, l->attn_k_norm->abs_offset,
                       P36_N_HEAD_DIM, P36_N_HEAD_KV,
                       pos0, n_tok, cache->cap, P36_RMS_EPS,
                       cache->k_row_bytes, cache->v_row_bytes)) {
            /* Fused quantized KV store succeeded */
        } else {
            if (!p36_vulkan_prepare_attn_heads(rt->attn_k, rt->attn_k, m, l->attn_k_norm,
                                               P36_N_HEAD_DIM, P36_N_HEAD_KV,
                                               rope_pos0, n_tok, positions,
                                               quality)) return false;
            if (!p36_gpu_attn_kv_store_tensor(cache->k, cache->v,
                                              rt->attn_k, rt->attn_v,
                                              pos0, n_tok, cache->cap,
                                              P36_N_HEAD_KV * P36_N_HEAD_DIM,
                                              P36_N_HEAD_KV * P36_N_VALUE_DIM,
                                              cache->type_k,
                                              cache->type_v,
                                              cache->k_row_bytes,
                                              cache->v_row_bytes)) return false;
        }
    }
    if (!p36_gpu_attn_decode_tensor(rt->attn_out,
                                    rt->attn_q,
                                    rt->attn_qg,
                                     cache->k,
                                     cache->v,
                                     rt->scores,
                                     m->map,
                                     m->size,
                                     l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                                    l->attn_sinks != NULL,
                                    pos0,
                                    n_tok,
                                    P36_N_HEAD,
                                    P36_N_HEAD_KV,
                                    P36_N_HEAD_DIM,
                                    cache->type_k,
                                    cache->type_v,
                                    cache->k_row_bytes,
                                    cache->v_row_bytes)) {
        return false;
    }
    if (p36_hadamard_rotate_enabled(l->attn_output)) {
        if (!p36_hadamard_prepare(rt, rt->attn_out, P36_N_SSM_INNER, n_tok) ||
            !p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_output, rt->attn_out, rt->had_q8, out,
                                                      P36_N_SSM_INNER, P36_N_EMBD, n_tok,
                                                      p36_tensor_scalar_or(m, l->attn_output_scale, 1.0f))) {
            fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos0);
            return false;
        }
    } else if (!p36_gpu_tensor_matmul_scaled(m, l->attn_output, rt->attn_out, out,
                                             P36_N_SSM_INNER, P36_N_EMBD, n_tok,
                                             p36_tensor_scalar_or(m, l->attn_output_scale, 1.0f))) {
        fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    return true;
}

static bool p36_forward_full_attn_vulkan(p36_session *s,
                                         const p36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t pos0,
                                         const uint32_t *positions,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    if (!s) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    return p36_forward_full_attn_vulkan_model(rt,
                                              &e->model,
                                              l,
                                              &rt->full[il],
                                              e->quality,
                                              il,
                                              pos0,
                                              (uint32_t)((int32_t)pos0 + s->rope_delta),
                                              positions,
                                              n_tok,
                                              inp,
                                              out);
}

static bool p36_forward_recurrent_vulkan(p36_session *s,
                                         const p36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const p36_gpu_tensor *inp,
                                         p36_gpu_tensor *out) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    p36_vulkan_recurrent_cache *cache;
    if (!s || !l || !inp || !out) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    cache = &rt->recurrent[il];
    if ((l->attn_qkv->type != P36_TENSOR_Q8_0 || l->attn_gate->type != P36_TENSOR_Q8_0) &&
        !rt->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
        return false;
    }
    rt->inp_q8_valid = false;
    const p36_gpu_tensor *qkv_in = inp;
    const p36_gpu_tensor *qkv_q8 = rt->inp_q8;
    if (p36_hadamard_rotate_enabled(l->attn_qkv) ||
        p36_hadamard_rotate_enabled(l->attn_gate)) {
        if (!p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)) return false;
        qkv_q8 = rt->had_q8;
    }
    bool pair_projected = n_tok == 1u &&
                          l->attn_qkv->type == P36_TENSOR_Q8_0 &&
                          l->attn_gate->type == P36_TENSOR_Q8_0 &&
                          p36_gpu_matmul_q8_0_pair_scaled_tensor(
                              rt->recur_qkv, rt->recur_z,
                              e->model.map, e->model.size,
                              l->attn_qkv->abs_offset, l->attn_gate->abs_offset,
                              P36_N_EMBD, P36_N_SSM_CONV_DIM, P36_N_SSM_INNER, inp,
                              p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f),
                              p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
    if (!pair_projected) {
        if (!p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_qkv, qkv_in, qkv_q8, rt->recur_qkv,
                                                      P36_N_EMBD, P36_N_SSM_CONV_DIM, n_tok,
                                                      p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent attn_qkv failed at layer=%u\n", il);
            return false;
        }
        /* The z gate dispatch records before the host conv step so the flush
         * the conv's qkv read triggers completes both projections. */
        if (!p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_gate, qkv_in, qkv_q8, rt->recur_z,
                                                      P36_N_EMBD, P36_N_SSM_INNER, n_tok,
                                                      p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent attn_gate failed at layer=%u\n", il);
            return false;
        }
    }
    bool front=false;
#ifndef P36_METAL
    if (!e->quality && !e->ssd_streaming && n_tok==1u && rt->recur_conv_fused &&
        P36_N_SSM_STATE==128u && P36_N_SSM_CONV==4u &&
        l->ssm_alpha->type==P36_TENSOR_Q8_0 && l->ssm_beta->type==P36_TENSOR_Q8_0) {
        const uint64_t offsets[5]={l->ssm_conv1d->abs_offset,l->ssm_alpha->abs_offset,
            l->ssm_beta->abs_offset,l->ssm_dt->abs_offset,l->ssm_a->abs_offset};
        front=p36_gpu_gdn_front_tensor(cache->conv,rt->recur_qkv,inp,
            rt->recur_q,rt->recur_k,rt->recur_v,rt->recur_gb,
            e->model.map,e->model.size,offsets,P36_N_EMBD,P36_N_SSM_GROUP,P36_N_SSM_DT_RANK,
            p36_tensor_scalar_or(&e->model,l->ssm_alpha_scale,1.0f),
            p36_tensor_scalar_or(&e->model,l->ssm_beta_scale,1.0f),P36_RMS_EPS)!=0;
    }
#endif
    if (front) goto recurrent_update;
    {
        if (!rt->recur_conv_fused) {
            if (!p36_gpu_recurrent_conv_step_tensor(cache->conv, rt->recur_qkv,
                                                    rt->recur_window, n_tok)) return false;
            if (!p36_gpu_ssm_conv_silu_tensor(rt->recur_conv,
                                              rt->recur_window,
                                              e->model.map,
                                              e->model.size,
                                              l->ssm_conv1d->abs_offset,
                                              P36_N_SSM_CONV_DIM,
                                              P36_N_SSM_CONV,
                                              n_tok)) return false;
        } else if (!p36_gpu_recurrent_conv_silu_tensor(
                       cache->conv, rt->recur_qkv, rt->recur_conv,
                       e->model.map, e->model.size, l->ssm_conv1d->abs_offset,
                       P36_N_SSM_CONV_DIM, P36_N_SSM_CONV, n_tok)) {
            return false;
        }
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_conv, n_tok * P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    pair_projected = n_tok == 1u &&
                     l->ssm_beta->type == P36_TENSOR_Q8_0 &&
                     l->ssm_alpha->type == P36_TENSOR_Q8_0 &&
                     p36_gpu_matmul_q8_0_pair_scaled_tensor(
                         rt->recur_beta, rt->recur_alpha,
                         e->model.map, e->model.size,
                         l->ssm_beta->abs_offset, l->ssm_alpha->abs_offset,
                         P36_N_EMBD, P36_N_SSM_DT_RANK, P36_N_SSM_DT_RANK, inp,
                         p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f),
                         p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    /* Both IQ2_S (this file's shape): one shared Q8_K quantize instead of
     * the two the individual calls below would each do internally, plus
     * one dispatch instead of two -- ssm_alpha/beta are tiny (2048x32) so
     * dispatch/quantize overhead dominates their cost, not DRAM bandwidth
     * (see karpathy/evidence/raw/iq2m-reprofile-2026-09-21.md). */
    if (!pair_projected && n_tok == 1u &&
        l->ssm_beta->type == P36_TENSOR_IQ2_S &&
        l->ssm_alpha->type == P36_TENSOR_IQ2_S) {
        if (!rt->inp_q8_valid &&
            !p36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, P36_N_EMBD, n_tok)) {
            return false;
        }
        rt->inp_q8_valid = false;
        pair_projected = p36_gpu_matmul_iq2s_pair_scaled_tensor(
            rt->recur_beta, rt->recur_alpha,
            e->model.map, e->model.size,
            l->ssm_beta->abs_offset, l->ssm_alpha->abs_offset,
            P36_N_EMBD, P36_N_SSM_DT_RANK, P36_N_SSM_DT_RANK, rt->inp_q8,
            p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f),
            p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    }
    if (!pair_projected) {
        if (!p36_gpu_tensor_matmul_dense_q8_scaled(
                &e->model, l->ssm_beta, inp, rt->inp_q8, rt->recur_beta,
                P36_N_EMBD, P36_N_SSM_DT_RANK, n_tok,
                p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent ssm_beta failed at layer=%u\n", il);
            return false;
        }
        if (!p36_gpu_tensor_matmul_dense_q8_scaled(
                &e->model, l->ssm_alpha, inp, rt->inp_q8, rt->recur_alpha,
                P36_N_EMBD, P36_N_SSM_DT_RANK, n_tok,
                p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
            fprintf(stderr, "p36: recurrent ssm_alpha failed at layer=%u\n", il);
            return false;
        }
    }
    {
        const char *env = getenv("P36_VK_DELTA_QKV");
        bool fused = !e->quality && !e->ssd_streaming &&
                     (!env || !env[0] || env[0] != '0') &&
                     p36_gpu_delta_qkv_l2_norm_tensor(
                         rt->recur_q, rt->recur_k, rt->recur_v,
                         rt->recur_conv, P36_N_SSM_DT_RANK,
                         P36_N_SSM_GROUP, P36_N_SSM_STATE,
                         P36_N_SSM_CONV_DIM, n_tok, P36_RMS_EPS);
        if (!fused) {
            if (!p36_gpu_delta_qk_l2_norm_tensor(rt->recur_q,
                                                 rt->recur_k,
                                                 rt->recur_conv,
                                                 P36_N_SSM_DT_RANK,
                                                 P36_N_SSM_GROUP,
                                                 P36_N_SSM_STATE,
                                                 P36_N_SSM_CONV_DIM,
                                                 n_tok,
                                                 P36_RMS_EPS)) {
                return false;
            }
            if (!p36_gpu_extract_recurrent_v_tensor(rt->recur_v, rt->recur_conv, n_tok)) return false;
        }
    }
    if (!p36_gpu_delta_net_gates_tensor(
            rt->recur_gb, rt->recur_alpha, rt->recur_beta,
            e->model.map, e->model.size, l->ssm_dt->abs_offset,
            l->ssm_a->abs_offset, P36_N_SSM_DT_RANK, n_tok)) {
        return false;
    }
recurrent_update:
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_gb, n_tok * P36_N_SSM_DT_RANK * 2u)) {
        fprintf(stderr, "p36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_gpu_delta_net_decode_tensor(cache->state,
                                         rt->recur_q,
                                         rt->recur_k,
                                         rt->recur_v,
                                         rt->recur_gb,
                                         rt->recur_proj,
                                         P36_N_SSM_DT_RANK,
                                         P36_N_SSM_STATE,
                                         n_tok)) {
        return false;
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_proj, n_tok * P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    bool norm_gate_q8 = false;
#ifndef P36_METAL
    norm_gate_q8 = !e->quality &&
        p36_gpu_recurrent_norm_gate_q8_k_tensor(
            rt->recur_proj, rt->inp_q8, rt->recur_z,
            e->model.map, e->model.size, l->ssm_norm->abs_offset,
            n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS);
#endif
    bool norm_gate_fused = norm_gate_q8 || (!e->quality &&
        p36_gpu_recurrent_norm_gate_tensor(
            rt->recur_proj, rt->recur_z, e->model.map, e->model.size,
            l->ssm_norm->abs_offset, P36_N_SSM_STATE,
            n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS));
    if (!norm_gate_fused &&
        (!p36_gpu_rms_norm_weight_rows_tensor(
             rt->recur_proj, rt->recur_proj,
             e->model.map, e->model.size, l->ssm_norm->abs_offset,
             P36_N_SSM_STATE, n_tok * P36_N_SSM_DT_RANK, P36_RMS_EPS) ||
         !p36_gpu_swiglu_tensor(
             rt->recur_proj, rt->recur_z, rt->recur_proj,
             n_tok * P36_N_SSM_INNER, 0.0f, 1.0f))) {
        return false;
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->recur_proj, n_tok * P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    float out_scale = p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f);
    if (p36_hadamard_rotate_enabled(l->ssm_out) &&
        !p36_hadamard_prepare_grouped(rt, rt->recur_proj, P36_N_SSM_INNER, n_tok)) {
        return false;
    }
    bool out_ok = p36_hadamard_rotate_enabled(l->ssm_out) ?
        p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->ssm_out, rt->recur_proj, rt->had_q8, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale) :
        norm_gate_q8 ?
        p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->ssm_out, rt->recur_proj, rt->inp_q8, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale) :
        p36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_out, rt->recur_proj, out,
            P36_N_SSM_INNER, P36_N_EMBD, n_tok, out_scale);
    if (!out_ok) {
        fprintf(stderr, "p36: recurrent ssm_out failed at layer=%u\n", il);
        return false;
    }
    return true;
}

/* Forward n_tok prompt tokens at positions pos0..pos0+n_tok-1 through the
 * Vulkan runtime, optionally leaving the last token's logits in s->logits.
 * Each token runs the exact arithmetic of the single-token path; batching
 * only widens the matmul dispatches and lets the host parity ops process
 * the whole chunk per GPU round-trip, which is where prefill time went. */
static bool p36_forward_tokens_vulkan_into(p36_session *s,
                                           const int *tokens,
                                           const float *embeddings,
                                           const uint32_t *positions,
                                           uint32_t n_tok,
                                           uint32_t pos0,
                                           p36_gpu_tensor *logits_out,
                                           bool logits_all_rows) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    p36_gpu_stream_expert_cache_note_tokens(n_tok);
    /* Embed into the host-only stage, then a queued device copy into the
     * GPU-written hidden tensor: mapping hidden here would wait out the
     * previous chunk's whole in-flight batch. */
    {
        p36_gpu_tensor *stage = rt->embed_stage[rt->embed_flip & 1u];
        rt->embed_flip ^= 1u;
        if (embeddings) {
            if (!p36_gpu_tensor_write(stage, 0, embeddings,
                    (uint64_t)n_tok * P36_N_EMBD * sizeof(*embeddings))) return false;
        } else if (!p36_gpu_embed_tokens(&e->model, e->weights.token_embd, tokens, n_tok, stage)) {
            fprintf(stderr, "p36: embed failed for %u tokens at pos=%u\n", n_tok, pos0);
            return false;
        }
        if (!p36_gpu_copy_f32_tensor(rt->hidden, stage, n_tok * P36_N_EMBD)) return false;
    }
    /* Every residual add feeds the next RMS norm (post-attention norm, the
     * next layer's attn norm, or the final output norm), so both adds fuse
     * with their norm; only layer 0's input norm stands alone. */
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm, rt->hidden,
                                             e->model.map, e->model.size,
                                             e->weights.layer[0].attn_norm->abs_offset,
                                             P36_N_EMBD, n_tok, P36_RMS_EPS)) {
        return false;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        if (l->kind == P36_LAYER_FULL_ATTN) {
            if (!p36_forward_full_attn_vulkan(s, l, il, pos0, positions,
                                               n_tok, rt->norm, rt->next_hidden)) {
                fprintf(stderr, "p36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        } else {
            if (!p36_forward_recurrent_vulkan(s, l, il, n_tok, rt->norm, rt->next_hidden)) {
                fprintf(stderr, "p36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        }
        if (e->directional_steering_attn_scale != 0.0f &&
            !p36_gpu_directional_steering_project_tensor(
                rt->next_hidden,
                e->directional_steering_gpu,
                il,
                P36_N_EMBD,
                n_tok,
                e->directional_steering_attn_scale)) {
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->next_hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        bool fused_norm = p36_gpu_add_rms_norm_q8_k_tensor(
            rt->norm, rt->inp_q8, rt->next_hidden, rt->next_hidden, rt->hidden,
            e->model.map, e->model.size, l->post_attention_norm->abs_offset,
            P36_N_EMBD, n_tok, P36_RMS_EPS);
        if (!fused_norm && !p36_gpu_add_rms_norm_tensor(
                rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                e->model.map, e->model.size, l->post_attention_norm->abs_offset,
                P36_N_EMBD, n_tok, P36_RMS_EPS)) {
            return false;
        }
        if (fused_norm) rt->inp_q8_valid = true;
        if (!p36_forward_ffn_vulkan(s, l, il, n_tok, rt->norm, rt->hidden)) {
            fprintf(stderr, "p36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        if (e->directional_steering_ffn_scale != 0.0f &&
            !p36_gpu_directional_steering_project_tensor(
                rt->hidden,
                e->directional_steering_gpu,
                il,
                P36_N_EMBD,
                n_tok,
                e->directional_steering_ffn_scale)) {
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        {
            const p36_tensor *next_norm = il + 1u < P36_N_LAYER ?
                e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            p36_gpu_tensor *target_q8 = (il + 1u < P36_N_LAYER) ? rt->inp_q8 : NULL;
            bool fused_next = target_q8 && p36_gpu_add_rms_norm_q8_k_tensor(
                rt->norm, target_q8, rt->hidden, rt->next_hidden, rt->hidden,
                e->model.map, e->model.size, next_norm->abs_offset,
                P36_N_EMBD, n_tok, P36_RMS_EPS);
            if (!fused_next && !p36_gpu_add_rms_norm_tensor(
                    rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size, next_norm->abs_offset,
                    P36_N_EMBD, n_tok, P36_RMS_EPS)) {
                return false;
            }
            if (fused_next) rt->inp_q8_valid = true;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(rt->hidden, n_tok * P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
    }
    if (e->quality && !p36_gpu_tensor_all_finite(rt->norm, n_tok * P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (logits_out) {
        const float out_scale = p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f);
        bool rot = p36_hadamard_rotate_enabled(e->weights.output);
        bool ok = true;
        p36_gpu_tensor *last = NULL;
        const p36_gpu_tensor *out_in = rt->norm;
        uint32_t rows = n_tok;
        if (!logits_all_rows) {
            last = p36_gpu_tensor_view(rt->norm,
                                       (uint64_t)(n_tok - 1u) * P36_N_EMBD * sizeof(float),
                                       (uint64_t)P36_N_EMBD * sizeof(float));
            out_in = last;
            rows = 1;
        }
        if (rot) {
            ok = out_in && p36_hadamard_prepare(rt, out_in, P36_N_EMBD, rows) &&
                 p36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, e->weights.output,
                                                          out_in, rt->had_q8, logits_out,
                                                          P36_N_EMBD, P36_N_VOCAB, rows, out_scale);
        } else {
            ok = out_in && p36_gpu_tensor_matmul_scaled(&e->model, e->weights.output,
                                                        out_in, logits_out,
                                                        P36_N_EMBD, P36_N_VOCAB, rows, out_scale);
        }
        p36_gpu_tensor_free(last);
        if (!ok) {
            fprintf(stderr, "p36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        if (e->quality && !p36_gpu_tensor_all_finite(logits_out, P36_N_VOCAB * (logits_all_rows ? n_tok : 1u))) {
            fprintf(stderr, "p36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
    }
    return true;
}

static bool p36_vulkan_update_last_h(p36_vulkan_runtime *rt, uint32_t row) {
    if (!rt || !rt->last_h) return false;
    return p36_gpu_tensor_copy(rt->last_h,
                               0,
                               rt->norm,
                               (uint64_t)row * P36_N_EMBD * sizeof(float),
                               (uint64_t)P36_N_EMBD * sizeof(float)) != 0;
}

static p36_gpu_tensor *p36_vulkan_row(p36_gpu_tensor *t, uint32_t row,
                                      uint64_t width) {
    uint64_t bytes = width * sizeof(float);
    return p36_gpu_tensor_view(t, (uint64_t)row * bytes, bytes);
}

static bool p36_sessions_full_attn_vulkan(p36_decode_item *items, int count,
                                          const p36_layer_weights *l,
                                          uint32_t il,
                                          p36_gpu_tensor *inp,
                                          p36_gpu_tensor *out) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_q->type != P36_TENSOR_Q8_0 ||
         l->attn_k->type != P36_TENSOR_Q8_0 ||
         l->attn_v->type != P36_TENSOR_Q8_0) &&
        !batch->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, P36_N_EMBD, rows)) {
        return false;
    }
    batch->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_q, inp, batch->inp_q8, batch->attn_qg,
            P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u, rows,
            p36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_k, inp, batch->inp_q8, batch->attn_k,
            P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_v, inp, batch->inp_q8, batch->attn_v,
            P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        p36_vulkan_runtime *rt = s->runtime;
        p36_vulkan_full_attn_cache *cache = &rt->full[il];
        uint32_t pos = (uint32_t)s->checkpoint.len;
        p36_gpu_tensor *qg = p36_vulkan_row(batch->attn_qg, (uint32_t)i,
                P36_N_HEAD * P36_N_HEAD_DIM * 2u);
        p36_gpu_tensor *q = p36_vulkan_row(batch->attn_q, (uint32_t)i,
                P36_N_HEAD * P36_N_HEAD_DIM);
        p36_gpu_tensor *k = p36_vulkan_row(batch->attn_k, (uint32_t)i,
                P36_N_HEAD_KV * P36_N_HEAD_DIM);
        p36_gpu_tensor *v = p36_vulkan_row(batch->attn_v, (uint32_t)i,
                P36_N_HEAD_KV * P36_N_VALUE_DIM);
        p36_gpu_tensor *attn = p36_vulkan_row(batch->attn_out, (uint32_t)i,
                P36_N_SSM_INNER);
        bool ok = qg && q && k && v && attn &&
            p36_vulkan_prepare_attn_heads(q, qg, &e->model, l->attn_q_norm,
                    P36_N_HEAD_DIM * 2u, P36_N_HEAD, pos, 1, NULL, e->quality);
        bool fused_kv = !e->quality && cache->type_k == P36_KV_CACHE_F16 &&
                        cache->type_v == P36_KV_CACHE_F16 &&
                        cache->k_row_bytes == P36_N_HEAD_KV * P36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == P36_N_HEAD_KV * P36_N_VALUE_DIM * sizeof(uint16_t);
        const char *fused_env = getenv("P36_VK_NORM_ROPE_KV");
        fused_kv = fused_kv && (!fused_env || !fused_env[0] || fused_env[0] != '0');
        if (ok && fused_kv) {
            ok = p36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, k, v, e->model.map, e->model.size,
                    l->attn_k_norm->abs_offset, P36_N_HEAD_DIM,
                    P36_N_HEAD_KV, pos, 1, cache->cap, P36_RMS_EPS) != 0;
        } else if (ok) {
            ok = p36_vulkan_prepare_attn_heads(k, k, &e->model,
                    l->attn_k_norm, P36_N_HEAD_DIM, P36_N_HEAD_KV,
                    pos, 1, NULL, e->quality) &&
                 p36_gpu_attn_kv_store_tensor(
                    cache->k, cache->v, k, v, pos, 1, cache->cap,
                    P36_N_HEAD_KV * P36_N_HEAD_DIM,
                    P36_N_HEAD_KV * P36_N_VALUE_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        if (ok) {
            ok = p36_gpu_attn_decode_tensor(
                    attn, q, qg, cache->k, cache->v, rt->scores,
                    e->model.map, e->model.size,
                    l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                    l->attn_sinks != NULL, pos, 1,
                    P36_N_HEAD, P36_N_HEAD_KV, P36_N_HEAD_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        p36_gpu_tensor_free(attn);
        p36_gpu_tensor_free(v);
        p36_gpu_tensor_free(k);
        p36_gpu_tensor_free(q);
        p36_gpu_tensor_free(qg);
        if (!ok) return false;
    }

    return p36_gpu_tensor_matmul_scaled(
            &e->model, l->attn_output, batch->attn_out, out,
            P36_N_SSM_INNER, P36_N_EMBD, rows,
            p36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
}

static bool p36_sessions_recurrent_vulkan(p36_decode_item *items, int count,
                                          const p36_layer_weights *l,
                                          uint32_t il,
                                          p36_gpu_tensor *inp,
                                          p36_gpu_tensor *out) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_qkv->type != P36_TENSOR_Q8_0 ||
         l->attn_gate->type != P36_TENSOR_Q8_0) &&
        !batch->inp_q8_valid &&
        !p36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, P36_N_EMBD, rows)) {
        return false;
    }
    batch->inp_q8_valid = false;
    if (!p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_qkv, inp, batch->inp_q8, batch->recur_qkv,
            P36_N_EMBD, P36_N_SSM_CONV_DIM, rows,
            p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_gate, inp, batch->inp_q8, batch->recur_z,
            P36_N_EMBD, P36_N_SSM_INNER, rows,
            p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_dense_q8_scaled(
            &e->model, l->ssm_beta, inp, batch->inp_q8, batch->recur_beta,
            P36_N_EMBD, P36_N_SSM_DT_RANK, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f)) ||
        !p36_gpu_tensor_matmul_dense_q8_scaled(
            &e->model, l->ssm_alpha, inp, batch->inp_q8, batch->recur_alpha,
            P36_N_EMBD, P36_N_SSM_DT_RANK, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        p36_vulkan_runtime *rt = items[i].session->runtime;
        p36_vulkan_recurrent_cache *cache = &rt->recurrent[il];
        p36_gpu_tensor *qkv = p36_vulkan_row(batch->recur_qkv, (uint32_t)i,
                P36_N_SSM_CONV_DIM);
        p36_gpu_tensor *conv = p36_vulkan_row(batch->recur_conv, (uint32_t)i,
                P36_N_SSM_CONV_DIM);
        p36_gpu_tensor *q = p36_vulkan_row(batch->recur_q, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *k = p36_vulkan_row(batch->recur_k, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *v = p36_vulkan_row(batch->recur_v, (uint32_t)i,
                P36_N_SSM_INNER);
        p36_gpu_tensor *gb = p36_vulkan_row(batch->recur_gb, (uint32_t)i,
                P36_N_SSM_DT_RANK * 2u);
        p36_gpu_tensor *alpha = p36_vulkan_row(batch->recur_alpha, (uint32_t)i,
                P36_N_SSM_DT_RANK);
        p36_gpu_tensor *beta = p36_vulkan_row(batch->recur_beta, (uint32_t)i,
                P36_N_SSM_DT_RANK);
        p36_gpu_tensor *proj = p36_vulkan_row(batch->recur_proj, (uint32_t)i,
                P36_N_SSM_INNER);
        bool ok = qkv && conv && q && k && v && gb && alpha && beta && proj;
        if (ok) {
            if (rt->recur_conv_fused) {
                ok = p36_gpu_recurrent_conv_silu_tensor(
                        cache->conv, qkv, conv, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, P36_N_SSM_CONV_DIM,
                        P36_N_SSM_CONV, 1) != 0;
            } else {
                ok = p36_gpu_recurrent_conv_step_tensor(
                        cache->conv, qkv, rt->recur_window, 1) != 0 &&
                     p36_gpu_ssm_conv_silu_tensor(
                        conv, rt->recur_window, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, P36_N_SSM_CONV_DIM,
                        P36_N_SSM_CONV, 1) != 0;
            }
        }
        if (ok && !p36_gpu_delta_qkv_l2_norm_tensor(
                    q, k, v, conv, P36_N_SSM_DT_RANK, P36_N_SSM_GROUP,
                    P36_N_SSM_STATE, P36_N_SSM_CONV_DIM, 1, P36_RMS_EPS)) {
            ok = p36_gpu_delta_qk_l2_norm_tensor(
                    q, k, conv, P36_N_SSM_DT_RANK, P36_N_SSM_GROUP,
                    P36_N_SSM_STATE, P36_N_SSM_CONV_DIM, 1,
                    P36_RMS_EPS) != 0 &&
                 p36_gpu_extract_recurrent_v_tensor(v, conv, 1) != 0;
        }
        if (ok) {
            ok = p36_gpu_delta_net_gates_tensor(
                    gb, alpha, beta, e->model.map, e->model.size,
                    l->ssm_dt->abs_offset, l->ssm_a->abs_offset,
                    P36_N_SSM_DT_RANK, 1) != 0 &&
                 p36_gpu_delta_net_decode_tensor(
                    cache->state, q, k, v, gb, proj,
                    P36_N_SSM_DT_RANK, P36_N_SSM_STATE, 1) != 0;
        }
        p36_gpu_tensor_free(proj);
        p36_gpu_tensor_free(beta);
        p36_gpu_tensor_free(alpha);
        p36_gpu_tensor_free(gb);
        p36_gpu_tensor_free(v);
        p36_gpu_tensor_free(k);
        p36_gpu_tensor_free(q);
        p36_gpu_tensor_free(conv);
        p36_gpu_tensor_free(qkv);
        if (!ok) return false;
    }

    if (!p36_gpu_rms_norm_weight_rows_tensor(
            batch->recur_proj, batch->recur_proj,
            e->model.map, e->model.size, l->ssm_norm->abs_offset,
            P36_N_SSM_STATE, rows * P36_N_SSM_DT_RANK, P36_RMS_EPS) ||
        !p36_gpu_swiglu_tensor(batch->recur_proj, batch->recur_z,
                               batch->recur_proj,
                               rows * P36_N_SSM_INNER, 0.0f, 1.0f)) {
        return false;
    }
    return p36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_out, batch->recur_proj, out,
            P36_N_SSM_INNER, P36_N_EMBD, rows,
            p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
}

static bool p36_sessions_eval_batch_vulkan(p36_decode_item *items, int count) {
    p36_session *first = items[0].session;
    p36_engine *e = first->engine;
    p36_vulkan_runtime *rt = first->runtime;
    uint32_t rows = (uint32_t)count;
    int tokens[8];

    for (int i = 0; i < count; i++) tokens[i] = items[i].token;
    p36_gpu_stream_expert_cache_note_tokens(rows);
    if (!p36_gpu_embed_tokens(&e->model, e->weights.token_embd,
                              tokens, rows, rt->embed_stage[0]) ||
        !p36_gpu_copy_f32_tensor(rt->hidden, rt->embed_stage[0],
                                 rows * P36_N_EMBD) ||
        !p36_gpu_rms_norm_weight_rows_tensor(
                rt->norm, rt->hidden, e->model.map, e->model.size,
                e->weights.layer[0].attn_norm->abs_offset,
                P36_N_EMBD, rows, P36_RMS_EPS)) {
        return false;
    }

    p36_gpu_set_micro_batch(true);
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        bool ok = l->kind == P36_LAYER_FULL_ATTN
            ? p36_sessions_full_attn_vulkan(items, count, l, il,
                                            rt->norm, rt->next_hidden)
            : p36_sessions_recurrent_vulkan(items, count, l, il,
                                             rt->norm, rt->next_hidden);
        if (ok && e->directional_steering_attn_scale != 0.0f) {
            ok = p36_gpu_directional_steering_project_tensor(
                    rt->next_hidden, e->directional_steering_gpu, il,
                    P36_N_EMBD, rows,
                    e->directional_steering_attn_scale) != 0;
        }
        if (ok) {
            bool fused_norm = p36_gpu_add_rms_norm_q8_k_tensor(
                    rt->norm, rt->inp_q8, rt->next_hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size,
                    l->post_attention_norm->abs_offset,
                    P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            if (fused_norm) {
                rt->inp_q8_valid = true;
            } else {
                ok = p36_gpu_add_rms_norm_tensor(
                        rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                        e->model.map, e->model.size,
                        l->post_attention_norm->abs_offset,
                        P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            }
        }
        if (ok) {
            ok = p36_forward_ffn_vulkan_model(
                    rt, &e->model, l, e->expert_weights_scale, il,
                    rows, rt->norm, rt->hidden, true);
        }
        if (ok && e->directional_steering_ffn_scale != 0.0f) {
            ok = p36_gpu_directional_steering_project_tensor(
                    rt->hidden, e->directional_steering_gpu, il,
                    P36_N_EMBD, rows,
                    e->directional_steering_ffn_scale) != 0;
        }
        if (ok) {
            const p36_tensor *next_norm = il + 1u < P36_N_LAYER
                ? e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            p36_gpu_tensor *target_q8 = (il + 1u < P36_N_LAYER) ? rt->inp_q8 : NULL;
            bool fused_next = target_q8 && p36_gpu_add_rms_norm_q8_k_tensor(
                    rt->norm, target_q8, rt->hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size, next_norm->abs_offset,
                    P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            if (fused_next) {
                rt->inp_q8_valid = true;
            } else {
                ok = p36_gpu_add_rms_norm_tensor(
                        rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                        e->model.map, e->model.size, next_norm->abs_offset,
                        P36_N_EMBD, rows, P36_RMS_EPS) != 0;
            }
        }
        if (!ok) {
            p36_gpu_set_micro_batch(false);
            return false;
        }
    }
    p36_gpu_set_micro_batch(false);

    if (e->session_batch_logits_cap < rows) {
        p36_gpu_tensor *logits = p36_gpu_tensor_alloc(
                (uint64_t)rows * P36_N_VOCAB * sizeof(float));
        if (!logits) return false;
        p36_gpu_tensor_free(e->session_batch_logits);
        e->session_batch_logits = logits;
        e->session_batch_logits_cap = rows;
    }
    if (!p36_gpu_tensor_matmul_scaled(
            &e->model, e->weights.output, rt->norm,
            e->session_batch_logits, P36_N_EMBD, P36_N_VOCAB, rows,
            p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        p36_vulkan_runtime *dst = items[i].session->runtime;
        if (!p36_gpu_tensor_copy(
                dst->last_h, 0, rt->norm,
                (uint64_t)i * P36_N_EMBD * sizeof(float),
                (uint64_t)P36_N_EMBD * sizeof(float)) ||
            !p36_gpu_tensor_read(
                e->session_batch_logits,
                (uint64_t)i * P36_N_VOCAB * sizeof(float),
                items[i].session->logits,
                (uint64_t)P36_N_VOCAB * sizeof(float))) {
            return false;
        }
        items[i].session->logits_host_valid = true;
        items[i].session->gpu_top2_valid = false;
    }
    return true;
}

static bool p36_forward_tokens_vulkan(p36_session *s, const int *tokens, uint32_t n_tok,
                                      uint32_t pos0, bool compute_logits) {
    p36_vulkan_runtime *rt;
    if (!s) return false;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!p36_forward_tokens_vulkan_into(s,
                                        tokens,
                                        NULL,
                                        NULL,
                                        n_tok,
                                        pos0,
                                        compute_logits ? rt->logits : NULL,
                                        false)) {
        return false;
    }
    if (!p36_vulkan_update_last_h(rt, n_tok - 1u)) return false;
    if (compute_logits) {
        s->gpu_top2_valid = false;
        s->logits_host_valid = false;
        if (p36_gpu_top2_tensor(rt->top2, rt->logits, P36_N_VOCAB)) {
            int32_t ids[2] = {-1, -1};
            if (!p36_gpu_tensor_read(rt->top2, 0, ids, sizeof(ids)))
                return false;
            s->gpu_top2[0] = ids[0];
            s->gpu_top2[1] = ids[1];
            s->gpu_top2_valid = ids[0] >= 0 && ids[1] >= 0;
        } else {
            if (!p36_gpu_tensor_read(
                    rt->logits, 0, s->logits,
                    (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) {
                return false;
            }
            s->logits_host_valid = true;
        }
    }
    return true;
}

static bool p36_forward_embeddings_vulkan(p36_session *s, const int *tokens,
                                           const float *embeddings,
                                           const uint32_t *positions,
                                           uint32_t n_tok, uint32_t pos0,
                                           bool compute_logits) {
    p36_vulkan_runtime *rt = s ? s->runtime : NULL;
    if (!rt || !p36_forward_tokens_vulkan_into(
            s, tokens, embeddings, positions, n_tok, pos0,
            compute_logits ? rt->logits : NULL, false) ||
        !p36_vulkan_update_last_h(rt, n_tok - 1u)) return false;
    if (compute_logits) {
        s->gpu_top2_valid = false;
        s->logits_host_valid = false;
    }
    return true;
}

static bool p36_forward_token_vulkan(p36_session *s, int token, uint32_t pos, bool compute_logits) {
    return p36_forward_tokens_vulkan(s, &token, 1, pos, compute_logits);
}

static uint64_t p36_vulkan_recurrent_conv_bytes(void) {
    return (uint64_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float);
}

static bool p36_vulkan_spec_frontier_copy(p36_vulkan_recurrent_cache *dst,
                                          const p36_vulkan_recurrent_cache *src) {
    if (!dst || !src || !dst->conv || !src->conv || !dst->state || !src->state) return false;
    if (!p36_gpu_tensor_copy(dst->conv, 0, src->conv, 0, p36_vulkan_recurrent_conv_bytes())) return false;
    if (p36_gpu_tensor_bytes(dst->state) != p36_gpu_tensor_bytes(src->state) ||
        !p36_gpu_tensor_copy(dst->state, 0, src->state, 0,
                             p36_gpu_tensor_bytes(src->state))) return false;
    return true;
}

static bool p36_vulkan_spec_frontier_snapshot(p36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) continue;
        if (!p36_vulkan_spec_frontier_copy(&rt->spec_recurrent[il], &rt->recurrent[il])) return false;
    }
    return true;
}

static bool p36_vulkan_spec_frontier_restore(p36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) continue;
        if (!p36_vulkan_spec_frontier_copy(&rt->recurrent[il], &rt->spec_recurrent[il])) return false;
    }
    return true;
}

static bool p36_vulkan_verify_suffix_tops(p36_session *s,
                                          const int *tokens,
                                          uint32_t n_tok,
                                          uint32_t pos0,
                                          int *row_tops) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    if (!s || !tokens || n_tok == 0 || n_tok > P36_MTP_MAX_DRAFT) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt || !rt->spec_logits) return false;
    p36_gpu_set_micro_batch(true);
    {
        bool fwd = p36_forward_tokens_vulkan_into(s, tokens, NULL, NULL,
                                                  n_tok, pos0, NULL, false) &&
                   p36_gpu_tensor_matmul_scaled(&e->model,
                                                e->weights.output,
                                                rt->norm,
                                                rt->spec_logits,
                                                P36_N_EMBD,
                                                P36_N_VOCAB,
                                                n_tok,
                                                p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        p36_gpu_set_micro_batch(false);
        if (!fwd) return false;
    }
    for (uint32_t i = 0; i + 1u < n_tok; i++) {
        if (!p36_gpu_tensor_read(rt->spec_logits,
                                 (uint64_t)i * P36_N_VOCAB * sizeof(*s->logits),
                                 s->logits,
                                 (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) return false;
        row_tops[i] = sample_argmax(s->logits, P36_N_VOCAB);
    }
    bool ok = p36_gpu_tensor_read(
        rt->spec_logits,
        (uint64_t)(n_tok - 1u) * P36_N_VOCAB * sizeof(*s->logits),
        s->logits, (uint64_t)P36_N_VOCAB * sizeof(*s->logits)) != 0;
    if (ok) {
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return ok;
}

static bool p36_mtp_concat(p36_gpu_tensor *out,
                           const p36_gpu_tensor *a,
                           const p36_gpu_tensor *b) {
    /* Device copies: mapping these GPU-written tensors on the host would
     * drain the whole in-flight draft pipeline. */
    return p36_gpu_tensor_copy(out, 0, a, 0, (uint64_t)P36_N_EMBD * sizeof(float)) &&
           p36_gpu_tensor_copy(out, (uint64_t)P36_N_EMBD * sizeof(float),
                               b, 0, (uint64_t)P36_N_EMBD * sizeof(float));
}

static bool p36_mtp_eval_vulkan(p36_session *s,
                                int token,
                                uint32_t pos,
                                const p36_gpu_tensor *h_in) {
    p36_engine *e;
    p36_vulkan_runtime *rt;
    p36_mtp_weights *mw;
    p36_layer_weights *l;
    if (!s || !h_in) return false;
    e = s->engine;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!e || !e->mtp_ready || !rt || !rt->mtp_enabled || !s->mtp_logits) return false;
    mw = &e->mtp_weights;
    l = &mw->block;

    if (!p36_gpu_embed_tokens(&e->mtp_model, mw->token_embd, &token, 1, rt->mtp_tok_embd)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_e_norm,
                                             rt->mtp_tok_embd,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->enorm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_h_norm,
                                             h_in,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->hnorm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_mtp_concat(rt->mtp_concat, rt->mtp_e_norm, rt->mtp_h_norm)) return false;
    if (!p36_gpu_tensor_matmul_scaled(&e->mtp_model,
                                      mw->eh_proj,
                                      rt->mtp_concat,
                                      rt->mtp_cur,
                                      P36_N_EMBD * 2u,
                                      P36_N_EMBD,
                                      1,
                                      1.0f)) return false;

    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->attn_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_forward_full_attn_vulkan_model(rt,
                                            &e->mtp_model,
                                            l,
                                            &rt->mtp_full,
                                            e->quality,
                                            P36_N_LAYER,
                                            pos,
                                            (uint32_t)((int32_t)pos + s->rope_delta),
                                            NULL,
                                            1,
                                            rt->norm,
                                            rt->mtp_next)) return false;
    if (!p36_gpu_add_tensor(rt->mtp_next, rt->mtp_next, rt->mtp_cur, P36_N_EMBD)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_next,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->post_attention_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_forward_ffn_vulkan_model(rt,
                                      &e->mtp_model,
                                      l,
                                      e->mtp_expert_weights_scale,
                                      P36_N_LAYER,
                                      1,
                                      rt->norm,
                                      rt->mtp_cur,
                                      false)) return false;
    if (!p36_gpu_add_tensor(rt->mtp_cur, rt->mtp_next, rt->mtp_cur, P36_N_EMBD)) return false;
    if (!p36_gpu_rms_norm_weight_rows_tensor(rt->mtp_head,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->shared_head_norm->abs_offset,
                                             P36_N_EMBD,
                                             1,
                                             P36_RMS_EPS)) return false;
    if (!p36_gpu_tensor_matmul_scaled(&e->model,
                                      e->weights.output,
                                      rt->mtp_head,
                                      rt->mtp_logits,
                                      P36_N_EMBD,
                                      P36_N_VOCAB,
                                      1,
                                      p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) return false;
    if (e->quality && !p36_gpu_tensor_all_finite(rt->mtp_logits, P36_N_VOCAB)) return false;
    return p36_gpu_tensor_read(rt->mtp_logits,
                               0,
                               s->mtp_logits,
                               (uint64_t)P36_N_VOCAB * sizeof(*s->mtp_logits)) != 0;
}
#endif

static bool p36_forward_token_cpu(p36_session *s, int token, uint32_t pos, bool compute_logits) {
    p36_engine *e;
    p36_cpu_runtime *rt;
    if (!s || !s->runtime) return false;
    e = s->engine;
    rt = (p36_cpu_runtime *)s->runtime;
    if (!p36_embed_token(e, token, rt->hidden)) {
        fprintf(stderr, "p36: embed failed for token=%d pos=%u\n", token, pos);
        return false;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        const p36_layer_weights *l = &e->weights.layer[il];
        p36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + l->attn_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
        if (l->kind == P36_LAYER_FULL_ATTN) {
            if (!p36_forward_full_attn(e, l, il, pos, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "p36: full attention block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        } else {
            if (!p36_forward_recurrent(e, l, il, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "p36: recurrent block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        }
        p36_directional_steering_project_rows(rt->next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_attn_scale);
        if (!p36_all_finite(rt->next_hidden, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite attn output at token=%d pos=%u layer=%u kind=%s\n",
                    token, pos, il, l->kind == P36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        for (uint32_t i = 0; i < P36_N_EMBD; i++) rt->next_hidden[i] += rt->hidden[i];
        p36_ref_rms_norm(rt->work0, rt->next_hidden, (const float *)(e->model.map + l->post_attention_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
        if (!p36_forward_ffn(e, l, rt->work0, rt->work1, rt)) {
            fprintf(stderr, "p36: ffn block failed for token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        p36_directional_steering_project_rows(rt->work1,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_ffn_scale);
        if (!p36_all_finite(rt->work1, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite ffn output at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        for (uint32_t i = 0; i < P36_N_EMBD; i++) rt->hidden[i] = rt->next_hidden[i] + rt->work1[i];
        if (!p36_all_finite(rt->hidden, P36_N_EMBD)) {
            fprintf(stderr, "p36: non-finite hidden state at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
    }
    p36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + e->weights.output_norm->abs_offset), P36_N_EMBD, P36_RMS_EPS);
    if (!p36_all_finite(rt->work0, P36_N_EMBD)) {
        fprintf(stderr, "p36: non-finite output norm at token=%d pos=%u\n", token, pos);
        return false;
    }
    if (compute_logits) {
        const float *out_in = p36_hadamard_forward_rows(e->weights.output, rt->work0, rt->batch_had, P36_N_EMBD, 1u);
        if (!p36_tensor_matvec(e, e->weights.output, out_in, s->logits, rt->work1, P36_N_EMBD, P36_N_VOCAB)) {
            fprintf(stderr, "p36: output projection failed for token=%d pos=%u\n", token, pos);
            return false;
        }
        p36_scale_inplace(s->logits, P36_N_VOCAB,
                          p36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!p36_all_finite(s->logits, P36_N_VOCAB)) {
            fprintf(stderr, "p36: non-finite logits at token=%d pos=%u\n", token, pos);
            return false;
        }
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
    }
    return true;
}

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, p36_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (p36_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    uint64_t mask;
    uint64_t i;
    if (t->cap == 0) return false;
    mask = t->cap - 1;
    i = hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        p36_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    {
        uint32_t n = 0;
        for (uint32_t x = 0; x < 256; x++) {
            if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
            if (x == b) return 256 + n;
            n++;
        }
    }
    return b;
}

static char *byte_encode(p36_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const p36_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    int rank = -1;
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const p36_vocab *vocab, p36_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);
    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        {
            owned_str merged;
            merged.len = sym[best_i].len + sym[best_i + 1].len;
            merged.ptr = xmalloc((size_t)merged.len);
            memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
            memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
            free(sym[best_i].ptr);
            free(sym[best_i + 1].ptr);
            sym[best_i] = merged;
            for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
            n_sym--;
        }
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            p36_tokens_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    p36_tokens_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }
    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool qwen_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    if ((uint8_t)s[pos] < 128) return ascii_alpha((uint8_t)s[pos]);
    return true;
}

static bool qwen_punct_symbol_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c;
    (void)len;
    c = (uint8_t)s[pos];
    return c < 128 && !ascii_alpha(c) && !ascii_digit(c) && !ascii_space(c);
}

static bool qwen_match_suffix(const char *s, uint64_t len, uint64_t pos, uint64_t *out_pos) {
    static const char *suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (pos >= len || s[pos] != '\'') return false;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        size_t n = strlen(suffixes[i]);
        if (pos + n > len) continue;
        if (strncasecmp(s + pos, suffixes[i], n) == 0) {
            *out_pos = pos + n;
            return true;
        }
    }
    return false;
}

static uint64_t qwen_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && qwen_letter_like_at(s, len, pos)) pos = next_utf8_char(s, len, pos);
    return pos;
}

static void bpe_tokenize_text(const p36_vocab *vocab, const char *text, token_vec *out) {
    uint64_t len = strlen(text);
    uint64_t pos = 0;
    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (c == ' ' && pos + 1 < len) {
            uint64_t next = pos + 1;
            uint8_t nc = (uint8_t)text[next];
            if (ascii_digit(nc)) {
                int ndigits = 0;
                pos++;
                while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                    pos++;
                    ndigits++;
                }
            } else if (qwen_letter_like_at(text, len, next) ||
                       (!ascii_newline(nc) && !ascii_digit(nc) && !qwen_letter_like_at(text, len, next) &&
                        !ascii_space(nc) && next + 1 < len && qwen_letter_like_at(text, len, next + 1))) {
                uint64_t after = 0;
                pos++;
                if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                pos = qwen_consume_letters(text, len, pos);
                if (qwen_match_suffix(text, len, pos, &after)) pos = after;
            } else if (qwen_punct_symbol_at(text, len, next)) {
                pos++;
                while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
            }
        } else if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (qwen_letter_like_at(text, len, pos) ||
                   (!ascii_newline(c) && !ascii_digit(c) && !qwen_letter_like_at(text, len, pos) &&
                    !ascii_space(c) && pos + 1 < len && qwen_letter_like_at(text, len, pos + 1))) {
            uint64_t after = 0;
            if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            pos = qwen_consume_letters(text, len, pos);
            if (qwen_match_suffix(text, len, pos, &after)) pos = after;
        } else if ((c == ' ' && pos + 1 < len && qwen_punct_symbol_at(text, len, pos + 1)) || qwen_punct_symbol_at(text, len, pos)) {
            if (c == ' ') pos++;
            while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (qwen_letter_like_at(text, len, p) || qwen_punct_symbol_at(text, len, p))) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (p36_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const p36_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "p36: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

static void vocab_load(p36_vocab *vocab, const p36_model *model) {
    p36_array_ref tokens;
    p36_array_ref token_types;
    p36_array_ref merges;
    p36_cursor c;
    p36_cursor types;
    int special_cap = 0;
    memset(vocab, 0, sizeof(*vocab));
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) || tokens.type != GGUF_VALUE_STRING || tokens.len > INT32_MAX) {
        p36_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.token_type", &token_types) ||
        token_types.type != GGUF_VALUE_INT32 || token_types.len != tokens.len) {
        p36_die("GGUF tokenizer token types are missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) || merges.type != GGUF_VALUE_STRING) {
        p36_die("GGUF tokenizer merge table is missing or invalid");
    }
    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);
    c = cursor_at(model, tokens.data_pos);
    types = cursor_at(model, token_types.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        uint32_t type;
        if (!cursor_string(&c, &vocab->token[i])) p36_die(c.error);
        if (!cursor_u32(&types, &type)) p36_die(types.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
        if (type == 3 || type == 4) {
            if (vocab->n_special == special_cap) {
                special_cap = special_cap ? special_cap * 2 : 32;
                vocab->special = xrealloc(vocab->special,
                    (size_t)special_cap * sizeof(vocab->special[0]));
            }
            vocab->special[vocab->n_special++] =
                (p36_special_token){.text = vocab->token[i], .id = i};
        }
    }
    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        p36_str merge;
        if (!cursor_string(&c, &merge)) p36_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }
    vocab->bos_id = (int)required_u32(model, "tokenizer.ggml.bos_token_id");
    vocab->eos_id = (int)required_u32(model, "tokenizer.ggml.eos_token_id");
    vocab->add_bos = model_get_bool(model, "tokenizer.ggml.add_bos_token", &vocab->add_bos) ? vocab->add_bos : false;
    vocab->im_start_id = vocab_lookup(vocab, "<|im_start|>");
    vocab->im_end_id = vocab_lookup(vocab, "<|im_end|>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->vision_start_id = vocab_lookup(vocab, "<|vision_start|>");
    vocab->vision_end_id = vocab_lookup(vocab, "<|vision_end|>");
    vocab->image_pad_id = vocab_lookup(vocab, "<|image_pad|>");
    vocab->video_pad_id = vocab_lookup(vocab, "<|video_pad|>");
}

static void vocab_free(p36_vocab *vocab) {
    free(vocab->token);
    free(vocab->special);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

void p36_tokenize_text(p36_engine *e, const char *text, p36_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const p36_vocab *vocab, const char *p,
                             size_t remain, int *token, size_t *len) {
    for (int i = 0; i < vocab->n_special; i++) {
        const p36_special_token *special = &vocab->special[i];
        if (special->text.len <= remain &&
            !memcmp(p, special->text.ptr, special->text.len)) {
            *token = special->id;
            *len = (size_t)special->text.len;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const p36_vocab *vocab, const char *p, size_t n, token_vec *out) {
    char *tmp;
    if (!n) return;
    tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const p36_vocab *vocab, const char *text, token_vec *out) {
    const char *span;
    const char *p;
    const char *end;
    if (!text) text = "";
    span = text;
    p = text;
    end = text + strlen(text);
    while (p < end) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, (size_t)(end - p), &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            p36_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(end - span), out);
}

void p36_tokenize_rendered_chat(p36_engine *e, const char *text, p36_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void p36_chat_apply_thinking_control(const char *content, bool *thinking) {
    if (!content || !thinking) return;
    if (strstr(content, "<|think_off|>")) *thinking = false;
    else if (strstr(content, "<|think_on|>")) *thinking = true;
}

static char *p36_chat_content_text(const char *text, bool controls) {
    static const char think_off[] = "<|think_off|>";
    static const char think_on[] = "<|think_on|>";
    const char *remove = NULL;
    size_t remove_len = 0;
    const char *start = text ? text : "";
    const char *end = start + strlen(start);

    if (controls && strstr(start, think_off)) {
        remove = think_off;
        remove_len = sizeof(think_off) - 1;
    } else if (controls && strstr(start, think_on)) {
        remove = think_on;
        remove_len = sizeof(think_on) - 1;
    }

    char *plain = xmalloc((size_t)(end - start) + 1);
    size_t used = 0;
    while (remove) {
        const char *p = strstr(start, remove);
        if (!p) break;
        size_t n = (size_t)(p - start);
        memcpy(plain + used, start, n);
        used += n;
        start = p + remove_len;
    }
    size_t tail = (size_t)(end - start);
    memcpy(plain + used, start, tail + 1);
    start = plain;
    end = plain + used + tail;
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    char *trimmed = xmalloc((size_t)(end - start) + 1);
    memcpy(trimmed, start, (size_t)(end - start));
    trimmed[end - start] = '\0';
    free(plain);
    return trimmed;
}

void p36_chat_begin(p36_engine *e, p36_tokens *tokens) {
    if (e->vocab.add_bos && e->vocab.bos_id >= 0)
        p36_tokens_push(tokens, e->vocab.bos_id);
}

static void chat_append_open_role(p36_engine *e, token_vec *tokens, const char *role) {
    p36_tokenize_rendered_chat(e, "<|im_start|>", tokens);
    bpe_tokenize_text(&e->vocab, role, tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

static void chat_append_close_role(p36_engine *e, token_vec *tokens) {
    p36_tokenize_rendered_chat(e, "<|im_end|>", tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

void p36_chat_append_max_effort_prefix(p36_engine *e, p36_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, P36_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void p36_chat_append_message(p36_engine *e, p36_tokens *tokens, const char *role, const char *content) {
    if (!role) role = "user";
    if (!content) content = "";
    if (!strcmp(role, "developer")) role = "system";
    if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        char *text = p36_chat_content_text(content, false);
        chat_append_open_role(e, tokens, "user");
        p36_tokenize_rendered_chat(e, "<tool_response>\n", tokens);
        p36_tokenize_rendered_chat(e, text, tokens);
        p36_tokenize_rendered_chat(e, "\n</tool_response>", tokens);
        chat_append_close_role(e, tokens);
        free(text);
        return;
    }
    bool controls = !strcmp(role, "system") || !strcmp(role, "user");
    char *text = p36_chat_content_text(content, controls);
    if (!strcmp(role, "system") && !text[0]) {
        free(text);
        return;
    }
    chat_append_open_role(e, tokens, role);
    p36_tokenize_rendered_chat(e, text, tokens);
    chat_append_close_role(e, tokens);
    free(text);
}

int p36_chat_append_multimodal_message(p36_engine *e, p36_tokens *tokens,
                                       const char *role, const char **parts,
                                       p36_vision_embedding *images, size_t image_count,
                                       p36_vision_span *spans, char *err, size_t errlen) {
    if (!e || !tokens || !role || !parts || (image_count && (!images || !spans)))
        goto invalid;
    for (size_t i = 0; i < image_count; i++) {
        if (!images[i].data || !images[i].token_count ||
            (uint64_t)images[i].grid_width * images[i].grid_height != images[i].token_count ||
            e->vocab.vision_start_id < 0 || e->vocab.vision_end_id < 0 || e->vocab.image_pad_id < 0)
            goto invalid;
    }
    if (!image_count) {
        p36_chat_append_message(e, tokens, role, parts[0]);
        return 1;
    }
    bool tool = !strcmp(role, "tool");
    chat_append_open_role(e, tokens, tool ? "user" : role);
    if (tool) p36_tokenize_rendered_chat(e, "<tool_response>\n", tokens);
    for (size_t i = 0; i <= image_count; i++) {
        if (parts[i]) p36_tokenize_rendered_chat(e, parts[i], tokens);
        if (i < image_count) p36_prompt_append_vision(e, tokens, &spans[i], &images[i], err, errlen);
    }
    if (tool) p36_tokenize_rendered_chat(e, "\n</tool_response>", tokens);
    chat_append_close_role(e, tokens);
    return 1;
invalid:
    if (err && errlen) snprintf(err, errlen, "invalid image observation");
    return 0;
}

int p36_chat_append_vision_message(p36_engine *e, p36_tokens *tokens,
                                    const char *role, const char *content,
                                    p36_vision_span *span,
                                    p36_vision_embedding *embedding,
                                    char *err, size_t errlen) {
    if (!e || !tokens || !span || !embedding) return 0;
    char *text = p36_chat_content_text(content, true);
    chat_append_open_role(e, tokens, role ? role : "user");
    if (text[0]) p36_tokenize_rendered_chat(e, text, tokens);
    free(text);
    if (!p36_prompt_append_vision(e, tokens, span, embedding, err, errlen))
        return 0;
    chat_append_close_role(e, tokens);
    return 1;
}

void p36_chat_append_assistant_prefix(p36_engine *e, p36_tokens *tokens, p36_think_mode think_mode) {
    chat_append_open_role(e, tokens, "assistant");
    if (think_mode == P36_THINK_NONE) {
        p36_tokenize_rendered_chat(e, "<think>\n\n</think>\n\n", tokens);
        return;
    }
    p36_tokenize_rendered_chat(e, "<think>\n", tokens);
    if (think_mode == P36_THINK_MAX && (!e || !e->chat_template))
        p36_chat_append_max_effort_prefix(e, tokens);
}

void p36_encode_chat_prompt(p36_engine *e, const char *system, const char *prompt, p36_think_mode think_mode, p36_tokens *out) {
    bool thinking = p36_think_mode_enabled(think_mode);
    p36_chat_apply_thinking_control(system, &thinking);
    p36_chat_apply_thinking_control(prompt, &thinking);
    if (!thinking) think_mode = P36_THINK_NONE;
    else if (think_mode == P36_THINK_NONE) think_mode = P36_THINK_HIGH;
    if (e && e->chat_template) {
        p36_jinja_message msgs[2] = {0};
        size_t n_msgs = 0;
        if (system && system[0]) {
            char *sys_trimmed = p36_chat_content_text(system, true);
            if (sys_trimmed && sys_trimmed[0]) {
                msgs[n_msgs].role = "system";
                msgs[n_msgs].content = sys_trimmed;
                n_msgs++;
            } else {
                free(sys_trimmed);
            }
        }
        char *usr_trimmed = p36_chat_content_text(prompt ? prompt : "", true);
        msgs[n_msgs].role = "user";
        msgs[n_msgs].content = usr_trimmed ? usr_trimmed : "";
        n_msgs++;
        p36_jinja_render_params jp = {
            .messages = msgs,
            .message_count = n_msgs,
            .tools_json_lines = NULL,
            .add_generation_prompt = true,
            .enable_thinking_set = true,
            .enable_thinking = think_mode != P36_THINK_NONE,
            .reasoning_effort = think_mode != P36_THINK_NONE ? p36_jinja_effort_name(think_mode) : NULL,
        };
        char *rendered = p36_jinja_render(e->chat_template, &jp, NULL, 0);
        for (size_t i = 0; i < n_msgs; i++) free((void *)msgs[i].content);
        if (rendered) {
            out->len = 0;
            p36_tokenize_rendered_chat(e, rendered, out);
            free(rendered);
            return;
        }
    }
    p36_chat_begin(e, out);
    if (system && system[0]) {
        p36_chat_append_message(e, out, "system", system);
    }
    p36_chat_append_message(e, out, "user", prompt ? prompt : "");
    p36_chat_append_assistant_prefix(e, out, think_mode);
}

static void dump_tokens_fp(FILE *fp, const p36_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");
    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
    {
        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) continue;
            if (cp == 256 + n) return (int)b;
            n++;
        }
    }
    return -1;
}

static bool vocab_token_is_literal_special(p36_str s) {
    return (s.len >= 2 && s.ptr[0] == '<' && s.ptr[1] == '|') ||
           p36_streq(s, "<think>") ||
           p36_streq(s, "</think>");
}

char *p36_token_text(p36_engine *e, int token, size_t *len) {
    p36_str s;
    char *out;
    size_t n = 0;
    uint64_t pos = 0;
    if (!e || token < 0 || token >= e->vocab.n_vocab) {
        out = xmalloc(1);
        out[0] = '\0';
        if (len) *len = 0;
        return out;
    }
    s = e->vocab.token[token];
    out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int p36_token_eos(p36_engine *e) {
    return e ? e->vocab.eos_id : -1;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = P36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

#ifndef P36_NO_GPU
static int sample_argmax_margin(const float *logits, uint32_t n_vocab, float *margin) {
    int best = 0;
    float best_v = P36_NEG_INF;
    float second_v = P36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            second_v = best_v;
            best_v = v;
            best = (int)i;
        } else if (v > second_v) {
            second_v = v;
        }
    }
    if (margin) *margin = best_v - second_v;
    return best;
}
#endif

typedef struct {
    int id;
    float logit;
    float prob;
} p36_sample_candidate;

static uint64_t p36_sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dull;
}

static float p36_sample_rng_f32(uint64_t *state) {
    const uint64_t x = p36_sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int p36_sample_candidate_cmp_desc(const void *a, const void *b) {
    const p36_sample_candidate *ca = (const p36_sample_candidate *)a;
    const p36_sample_candidate *cb = (const p36_sample_candidate *)b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int p36_sample_full_vocab(const float *logits, uint32_t n_vocab,
                                 float temperature, float top_p, float min_p,
                                 uint64_t *rng, float *prob_scratch) {
    float max_logit = P36_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        if (min_rel > 1.0f) return best;

        /* Reject only values proven to fail the ordinary expf comparison.
         * The small retreat covers logf/expf boundary rounding. */
        float reject_scaled = P36_NEG_INF;
        bool have_reject_scaled = false;
        if (min_rel > 0.0f && isfinite(min_rel)) {
            float cutoff = logf(min_rel);
            for (int i = 0; i < 8 && isfinite(cutoff); i++) {
                cutoff = nextafterf(cutoff, -FLT_MAX);
                if (expf(cutoff) < min_rel) {
                    reject_scaled = cutoff;
                    have_reject_scaled = true;
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - max_logit) / temperature;
            if (have_reject_scaled && scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_rel) continue;
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = p36_sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    {
        uint32_t n = 0;
        float sum = 0.0f;
        p36_sample_candidate *cand = NULL;
        if (min_p > 0.0f && min_p <= 1.0f) {
            /* Normalization cancels in the min-p comparison. Keep the full
             * softmax sum, but sort only candidates that can survive. */
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                prob_scratch[i] = -1.0f;
                if (!isfinite(v)) continue;
                const float p = expf((v - max_logit) / temperature);
                prob_scratch[i] = p;
                sum += p;
            }
            if (sum <= 0.0f || !isfinite(sum)) return best;

            const float min_prob = (1.0f / sum) * min_p;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float p = prob_scratch[i];
                if (p < 0.0f || p / sum < min_prob) continue;
                n++;
            }
            if (n == 0) return best;
            cand = xmalloc((size_t)n * sizeof(cand[0]));
            uint32_t out = 0;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float p = prob_scratch[i];
                if (p < 0.0f || p / sum < min_prob) continue;
                cand[out++] = (p36_sample_candidate){
                    .id = (int)i, .logit = logits[i], .prob = p
                };
            }
        } else {
            cand = xmalloc((size_t)finite * sizeof(cand[0]));
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                if (!isfinite(v)) continue;
                const float p = expf((v - max_logit) / temperature);
                cand[n++] = (p36_sample_candidate){
                    .id = (int)i, .logit = v, .prob = p
                };
                sum += p;
            }
        }
        if (sum <= 0.0f || !isfinite(sum)) {
            free(cand);
            return best;
        }

        qsort(cand, (size_t)n, sizeof(cand[0]), p36_sample_candidate_cmp_desc);
        {
            const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
            float filtered_sum = 0.0f;
            uint32_t filtered = 0;
            for (uint32_t i = 0; i < n; i++) {
                const float p = cand[i].prob / sum;
                if (i > 0 && p < min_prob) break;
                filtered_sum += cand[i].prob;
                filtered++;
                if (filtered_sum / sum >= top_p) break;
            }
            if (filtered == 0) {
                free(cand);
                return best;
            }
            {
                float r = p36_sample_rng_f32(rng) * filtered_sum;
                for (uint32_t i = 0; i < filtered; i++) {
                    r -= cand[i].prob;
                    if (r <= 0.0f) {
                        const int id = cand[i].id;
                        free(cand);
                        return id;
                    }
                }
                {
                    const int id = cand[filtered - 1].id;
                    free(cand);
                    return id;
                }
            }
        }
    }
}

/* Shared tail of top-k sampling: given candidate ids/vals already sorted
 * descending by value (length n, n<=1024), apply the temperature softmax,
 * the min-p relative floor and top-p cumulative-mass cutoff, then draw one
 * weighted sample. Used both by the CPU top-k scan below and by the GPU
 * top-8 fast path in p36_session_sample, so the two stay identical past
 * candidate selection. */
static int p36_sample_from_sorted_topk(const int *ids, const float *vals, int n,
                                       float temperature, float top_p, float min_p,
                                       uint64_t *rng) {
    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = p36_sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

static int p36_sample_top_p_min_p(const float *logits, uint32_t n_vocab,
                                  float temperature, int top_k,
                                  float top_p, float min_p,
                                  uint64_t *rng, float *prob_scratch) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        const bool owned_scratch = prob_scratch == NULL;
        if (owned_scratch) {
            prob_scratch = xmalloc((size_t)n_vocab * sizeof(prob_scratch[0]));
        }
        const int token = p36_sample_full_vocab(logits, n_vocab, temperature,
                                                top_p, min_p, rng, prob_scratch);
        if (owned_scratch) free(prob_scratch);
        return token;
    }
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    {
        int ids[1024];
        float vals[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            {
                int j = n < top_k ? n++ : n - 1;
                while (j > 0 && vals[j - 1] < v) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                    j--;
                }
                vals[j] = v;
                ids[j] = (int)i;
            }
        }
        if (n == 0) return sample_argmax(logits, n_vocab);
        return p36_sample_from_sorted_topk(ids, vals, n, temperature, top_p, min_p, rng);
    }
}

static int p36_sample_logits_penalized(float *logits, uint32_t n_vocab,
                                       float temperature, int top_k,
                                       float top_p, float min_p,
                                       const int *tokens, int n_tokens,
                                       float presence_penalty,
                                       float frequency_penalty,
                                       float repeat_penalty,
                                       uint64_t *rng, float *scratch) {
    if (!tokens || n_tokens <= 0 ||
        (presence_penalty == 0.0f && frequency_penalty == 0.0f &&
         repeat_penalty == 1.0f)) {
        return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                      top_p, min_p, rng, scratch);
    }

    /* Distinct seen ids plus their original logits, so the logits can be
     * restored exactly after sampling (no float add-back drift). */
    struct { int id; float logit; } *saved =
        malloc((size_t)n_tokens * sizeof(*saved));
    if (!saved) {
        return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                      top_p, min_p, rng, scratch);
    }
    int n_saved = 0;
    memset(scratch, 0, (size_t)n_vocab * sizeof(scratch[0]));
    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        if (token < 0 || (uint32_t)token >= n_vocab) continue;
        if (scratch[token] != 0.0f) continue;
        scratch[token] = 1.0f;
        saved[n_saved].id = token;
        saved[n_saved].logit = logits[token];
        n_saved++;
    }
    /* llama.cpp-style multiplicative penalty: positive logits shrink, negative grow. */
    for (int k = 0; k < n_saved; k++) {
        float *l = &logits[saved[k].id];
        if (repeat_penalty != 1.0f) *l = *l > 0.0f ? *l / repeat_penalty : *l * repeat_penalty;
    }
    /* Scratch counts occurrences for frequency/presence; first hit applies presence. */
    memset(scratch, 0, (size_t)n_vocab * sizeof(scratch[0]));
    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        if (token < 0 || (uint32_t)token >= n_vocab) continue;
        if (scratch[token] == 0.0f) logits[token] -= presence_penalty;
        scratch[token] += 1.0f;
        logits[token] -= frequency_penalty;
    }
    int sampled = p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                         top_p, min_p, rng, scratch);
    for (int k = 0; k < n_saved; k++) logits[saved[k].id] = saved[k].logit;
    free(saved);
    return sampled;
}

#ifdef P36_TEST_HOOKS
int p36_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch) {
    if (!logits || !rng || n_vocab == 0) return -1;
    return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                  top_p, min_p, rng, prob_scratch);
}

int p36_test_sample_logits_penalized(float *logits, uint32_t n_vocab,
                                     float temperature, int top_k,
                                     float top_p, float min_p,
                                     const int *tokens, int n_tokens,
                                     float presence_penalty,
                                     float frequency_penalty,
                                     float repeat_penalty,
                                     uint64_t *rng, float *prob_scratch) {
    if (!logits || !rng || !prob_scratch || n_vocab == 0) return -1;
    return p36_sample_logits_penalized(
        logits, n_vocab, temperature, top_k, top_p, min_p,
        tokens, n_tokens, presence_penalty, frequency_penalty,
        repeat_penalty, rng, prob_scratch);
}

uint32_t p36_test_kv_next_cap(uint32_t cap, uint32_t need, uint32_t limit) {
    return p36_kv_next_cap(cap, need, limit);
}
#endif

static bool p36_load_directional_steering(p36_engine *e) {
    uint64_t count;
    uint64_t bytes;
    struct stat st;
    FILE *fp;
    if (!e || (e->directional_steering_attn_scale == 0.0f &&
               e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }
    if (!e->directional_steering_file || !e->directional_steering_file[0]) {
        fprintf(stderr, "p36: directional steering needs --dir-steering-file\n");
        return false;
    }
    count = (uint64_t)P36_N_LAYER * P36_N_EMBD;
    bytes = count * sizeof(float);
    fp = fopen(e->directional_steering_file, "rb");
    if (!fp) {
        fprintf(stderr, "p36: failed to open directional steering file %s: %s\n",
                e->directional_steering_file, strerror(errno));
        return false;
    }
    if (fstat(fileno(fp), &st) != 0 || st.st_size < 0 || (uint64_t)st.st_size != bytes) {
        fprintf(stderr,
                "p36: directional steering file must contain exactly %u x %u f32 values (%" PRIu64 " bytes)\n",
                P36_N_LAYER, P36_N_EMBD, bytes);
        fclose(fp);
        return false;
    }
    e->directional_steering_dirs = xmalloc((size_t)bytes);
    if (fread(e->directional_steering_dirs, sizeof(float), (size_t)count, fp) != count ||
        fclose(fp) != 0) {
        fprintf(stderr, "p36: failed to read directional steering vectors from %s\n",
                e->directional_steering_file);
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        return false;
    }
#ifndef P36_NO_GPU
    if (p36_backend_uses_graph(e->backend)) {
        e->directional_steering_gpu = p36_gpu_tensor_alloc(bytes);
        if (!e->directional_steering_gpu ||
            !p36_gpu_tensor_write(e->directional_steering_gpu, 0,
                                  e->directional_steering_dirs, bytes)) {
            fprintf(stderr, "p36: failed to upload directional steering vectors\n");
            return false;
        }
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
    }
#endif
    fprintf(stderr, "p36: directional steering enabled: %s attn=%g ffn=%g\n",
            e->directional_steering_file,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

int p36_engine_open(p36_engine **out, const p36_engine_options *opt) {
    p36_engine *e;
    if (!out || !opt || !opt->model_path || !opt->model_path[0]) return 1;
#ifdef P36_NO_GPU
    if (p36_backend_uses_graph(opt->backend)) return 1;
#else
#if defined(P36_METAL) && !defined(P36_METAL_TEST_COMPAT)
    if (opt->backend == P36_BACKEND_VULKAN) {
        fprintf(stderr, "p36: Vulkan was requested from a Metal binary; use --metal or rebuild with 'make gpu'\n");
        return 1;
    }
#elif !defined(P36_METAL)
    if (opt->backend == P36_BACKEND_METAL) {
        fprintf(stderr, "p36: Metal was requested from a Vulkan binary; use --vulkan or build with 'make metal'\n");
        return 1;
    }
#endif
    if (p36_backend_uses_graph(opt->backend) && !p36_gpu_init()) return 1;
#endif
    e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->vision_model.fd = -1;
#if defined(P36_METAL) && defined(P36_METAL_TEST_COMPAT)
    /* The GPU ABI is shared by both native runtimes.  Normalize legacy
     * Vulkan-named test callers to the backend linked into this binary. */
    e->backend = p36_backend_uses_graph(opt->backend)
        ? P36_BACKEND_METAL : opt->backend;
#else
    e->backend = opt->backend;
#endif
    e->n_threads = p36_resolve_thread_count(opt->n_threads);
    e->cpu_prefill_cap = opt->prefill_chunk ? opt->prefill_chunk : p36_default_cpu_prefill_cap();
    e->prefill_cap_override = opt->prefill_chunk;
    e->context_size = opt->context_size > 0 ? opt->context_size : 4096;
    e->planned_sessions = opt->session_count > 0 ? (uint32_t)opt->session_count : 1;
    e->cache_type_k = opt->cache_type_k;
    e->cache_type_v = opt->cache_type_v;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    e->quality = opt->quality;
    {
        const char *mmq_contract = getenv("P36_CPU_MMQ_CONTRACT");
        e->cpu_mmq_contract = mmq_contract && mmq_contract[0] && mmq_contract[0] != '0';
    }
    e->ssd_streaming = opt->ssd_streaming;
    e->ssd_streaming_cold = opt->ssd_streaming_cold;
    e->ssd_streaming_full_layers_set = opt->ssd_streaming_full_layers_set;
    e->ssd_streaming_cache_experts = opt->ssd_streaming_cache_experts;
    e->ssd_streaming_full_layers = opt->ssd_streaming_full_layers;
    e->ssd_streaming_cache_bytes = opt->ssd_streaming_cache_bytes;
    e->ssd_streaming_preload_experts = opt->ssd_streaming_preload_experts;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > P36_MTP_MAX_DRAFT) e->mtp_draft_tokens = P36_MTP_MAX_DRAFT;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 0.0f;
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = p36_strdup(opt->directional_steering_file);
    }
    e->directional_steering_attn_scale = opt->directional_steering_attn;
    e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    if (opt->simulate_used_memory_bytes != 0 &&
        !p36_ssd_memory_lock_acquire(&e->simulated_memory,
                                     opt->simulate_used_memory_bytes)) {
        p36_engine_close(e);
        return 1;
    }
    model_open(&e->model, opt->model_path, p36_backend_uses_graph(opt->backend));
    {
        p36_str name = {0};
        if (model_get_string(&e->model, "general.name", &name)) {
            e->kat_coder = p36_str_contains(name, "KAT Coder") ||
                           p36_str_contains(name, "KAT-Coder");
        }
    }
    {
        const char *tmpl_opt = opt->chat_template;
        if (!tmpl_opt || !tmpl_opt[0]) tmpl_opt = getenv("P36_CHAT_TEMPLATE");
        if (tmpl_opt && tmpl_opt[0] &&
            strcasecmp(tmpl_opt, "off") != 0 &&
            strcasecmp(tmpl_opt, "none") != 0 &&
            strcasecmp(tmpl_opt, "false") != 0 &&
            strcasecmp(tmpl_opt, "0") != 0 &&
            strcasecmp(tmpl_opt, "default") != 0) {
            FILE *fp = fopen(tmpl_opt, "rb");
            if (!fp) {
                fprintf(stderr, "p36: failed to open chat template %s: %s\n",
                        tmpl_opt, strerror(errno));
                p36_engine_close(e);
                return 1;
            }
            if (fseek(fp, 0, SEEK_END) != 0) {
                fclose(fp);
                fprintf(stderr, "p36: failed to seek chat template %s\n", tmpl_opt);
                p36_engine_close(e);
                return 1;
            }
            long fsize = ftell(fp);
            if (fsize <= 0 || fsize > 1024 * 1024 || fseek(fp, 0, SEEK_SET) != 0) {
                fclose(fp);
                fprintf(stderr, "p36: invalid chat template size in %s\n", tmpl_opt);
                p36_engine_close(e);
                return 1;
            }
            char *buf = xmalloc((size_t)fsize + 1);
            size_t nread = fread(buf, 1, (size_t)fsize, fp);
            fclose(fp);
            buf[nread] = '\0';
            char err[256] = {0};
            if (!p36_jinja_validate(buf, err, sizeof(err))) {
                free(buf);
                fprintf(stderr, "p36: invalid chat template in %s: %s\n",
                        tmpl_opt, err[0] ? err : "validation failed");
                p36_engine_close(e);
                return 1;
            }
            e->chat_template = buf;
        }
    }
    config_validate_model(&e->model);
    hadamard_parse(&e->model);
    e->variant = g_p36_shape.variant;
    if (!p36_load_directional_steering(e)) {
        p36_engine_close(e);
        return 1;
    }
    if (opt->vision_path && opt->vision_path[0]) {
        model_open(&e->vision_model, opt->vision_path, false);
        p36_str arch = required_string(&e->vision_model, "general.architecture");
        p36_str projector = required_string(&e->vision_model, "clip.projector_type");
        p36_tensor *projection = model_find_tensor(&e->vision_model, "mm.2.weight");
        if (!p36_streq(arch, "clip") || !p36_streq(projector, "qwen3vl_merger") ||
            !projection ||
            (projection->type != P36_TENSOR_F16 && projection->type != P36_TENSOR_F32 &&
             projection->type != P36_TENSOR_Q8_0) ||
            projection->dim[0] != 4608u || projection->dim[1] != P36_N_EMBD) {
            fprintf(stderr, "p36: vision sidecar does not match the language model\n");
            p36_engine_close(e);
            return 1;
        }
#ifdef MADV_RANDOM
        madvise((void *)e->vision_model.map, (size_t)e->vision_model.size, MADV_RANDOM);
#endif
#ifdef MADV_DONTNEED
        madvise((void *)e->vision_model.map, (size_t)e->vision_model.size, MADV_DONTNEED);
#endif
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
        /* Non-F16 matrices are expanded to F16 in the stream buffer. */
        uint64_t vision_stream_bytes = e->vision_model.max_tensor_bytes;
        for (uint64_t i = 0; i < e->vision_model.n_tensors; i++) {
            const p36_tensor *vt = &e->vision_model.tensors[i];
            if (vt->ndim < 2) continue;
            uint64_t f16_bytes = vt->elements * sizeof(uint16_t);
            if (f16_bytes > vision_stream_bytes) vision_stream_bytes = f16_bytes;
        }
        if (e->backend == P36_BACKEND_VULKAN &&
            !p36_gpu_vision_stream_init(vision_stream_bytes)) {
            fprintf(stderr, "p36: unable to allocate streamed vision weight buffer\n");
            p36_engine_close(e);
            return 1;
        }
#endif
        e->vision_ready = true;
        fprintf(stderr, "p36: vision sidecar mapped from disk: %s\n", opt->vision_path);
    }
    if (P36_MODEL_DENSE && e->ssd_streaming) {
        fprintf(stderr, "p36: Qwen3.6 27B dense does not use expert SSD streaming\n");
        p36_engine_close(e);
        return 1;
    }
    vocab_load(&e->vocab, &e->model);
    weights_bind(&e->weights, &e->model);
    if (P36_MODEL_DENSE && getenv("P36_DUMP_DENSE_TYPES")) {
        uint64_t iq4_xs_tensors = 0;
        uint64_t iq4_xs_bytes = 0;
        uint64_t iq4_xs_blocks = 0;
        for (uint32_t il = 0; il < P36_N_LAYER; ++il) {
            const p36_layer_weights *l = &e->weights.layer[il];
            if (l->kind == P36_LAYER_FULL_ATTN) {
                fprintf(stderr,
                        "p36: dense layer %u kind=attention q=%s k=%s v=%s o=%s ffn_gate=%s ffn_up=%s ffn_down=%s\n",
                        il, tensor_type_name(l->attn_q->type),
                        tensor_type_name(l->attn_k->type),
                        tensor_type_name(l->attn_v->type),
                        tensor_type_name(l->attn_output->type),
                        tensor_type_name(l->ffn_gate_shexp->type),
                        tensor_type_name(l->ffn_up_shexp->type),
                        tensor_type_name(l->ffn_down_shexp->type));
            } else {
                fprintf(stderr,
                        "p36: dense layer %u kind=recurrent qkv=%s z=%s beta=%s alpha=%s o=%s ffn_gate=%s ffn_up=%s ffn_down=%s\n",
                        il, tensor_type_name(l->attn_qkv->type),
                        tensor_type_name(l->attn_gate->type),
                        tensor_type_name(l->ssm_beta->type),
                        tensor_type_name(l->ssm_alpha->type),
                        tensor_type_name(l->ssm_out->type),
                        tensor_type_name(l->ffn_gate_shexp->type),
                        tensor_type_name(l->ffn_up_shexp->type),
                        tensor_type_name(l->ffn_down_shexp->type));
            }
        }
        for (uint64_t it = 0; it < e->model.n_tensors; ++it) {
            const p36_tensor *t = &e->model.tensors[it];
            if (t->type != P36_TENSOR_IQ4_XS) continue;
            iq4_xs_tensors++;
            iq4_xs_bytes += t->bytes;
            iq4_xs_blocks += t->elements / P36_QK_K;
        }
        fprintf(stderr,
                "p36: dense IQ4_XS tensors=%" PRIu64 " bytes=%" PRIu64
                " blocks=%" PRIu64 " exact-scale-expanded-bytes=%" PRIu64
                " delta=%" PRIu64 "\n",
                iq4_xs_tensors, iq4_xs_bytes, iq4_xs_blocks,
                iq4_xs_bytes + iq4_xs_blocks * 2u,
                iq4_xs_blocks * 2u);
    }
    if (e->ssd_streaming && !p36_backend_supports_ssd_streaming(e->backend)) {
        fprintf(stderr,
                "p36: --ssd-streaming requires a Metal or Vulkan GPU graph backend\n");
        p36_engine_close(e);
        return 1;
    }
    e->routed_quant_bits = p36_quant_bits_from_type(
        P36_MODEL_DENSE ? e->weights.layer[0].ffn_gate_shexp->type :
                          e->weights.layer[0].ffn_gate_exps->type);
    e->expert_weights_scale = 1.0f;
    model_get_f32_compat(&e->model, "qwen35moe.expert_weights_scale", &e->expert_weights_scale);
    e->mtp_expert_weights_scale = e->expert_weights_scale;
    if (opt->mtp_path && opt->mtp_path[0]) {
        if (P36_MODEL_DENSE) {
            fprintf(stderr, "p36: Qwen3.6 27B dense does not support a MoE MTP model\n");
            p36_engine_close(e);
            return 1;
        }
        if (!p36_backend_uses_graph(opt->backend)) {
            fprintf(stderr, "p36: --mtp requires a GPU graph backend\n");
            p36_engine_close(e);
            return 1;
        }
#ifdef P36_NO_GPU
        fprintf(stderr, "p36: --mtp is not available in CPU-only builds\n");
        p36_engine_close(e);
        return 1;
#else
        model_open(&e->mtp_model, opt->mtp_path, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
        model_get_f32_compat(&e->mtp_model, "qwen35moe.expert_weights_scale", &e->mtp_expert_weights_scale);
        e->mtp_ready = true;
        fprintf(stderr, "p36: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
#endif
    } else if (P36_MODEL_DENSE && p36_backend_uses_graph(opt->backend) &&
               tensor_by_namef(&e->model, "blk.%u.nextn.eh_proj.weight", P36_N_LAYER) &&
               tensor_by_namef(&e->model, "blk.%u.nextn.eh_proj.weight", P36_N_LAYER)->bytes != 0) {
#ifndef P36_NO_GPU
        e->mtp_model = e->model;
        e->mtp_model.owned = false;
        mtp_weights_bind_dense(&e->mtp_weights, &e->model);
        e->mtp_ready = true;
        fprintf(stderr, "p36: in-file MTP head bound (blk.%u nextn), draft=%d\n",
                P36_N_LAYER,
                e->mtp_draft_tokens);
#endif
    }
#ifndef P36_NO_GPU
    if (p36_backend_uses_graph(opt->backend)) {
        p36_gpu_set_quality(opt->quality);
        p36_gpu_set_dense_model(P36_MODEL_DENSE);
        p36_gpu_set_ssd_streaming(e->ssd_streaming);
        if (!p36_engine_configure_streaming_auto_cache(e)) {
            p36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming_full_layers > P36_N_LAYER) {
            fprintf(stderr, "p36: --ssd-streaming-full-layers must be between 0 and %u\n",
                    P36_N_LAYER);
            p36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming && e->ssd_streaming_full_layers != 0) {
            uint32_t reserve = 0;
            uint64_t resident_bytes = 0;
            if (!p36_streaming_full_layer_budget(&e->weights,
                                                  e->ssd_streaming_full_layers,
                                                  &reserve,
                                                  &resident_bytes)) {
                fprintf(stderr, "p36: SSD streaming full-layer byte accounting failed\n");
                p36_engine_close(e);
                return 1;
            }
            if ((uint64_t)reserve + P36_N_EXPERT > e->ssd_streaming_cache_experts) {
                uint32_t old = e->ssd_streaming_full_layers;
                e->ssd_streaming_full_layers = e->ssd_streaming_cache_experts > P36_N_EXPERT ?
                    (e->ssd_streaming_cache_experts - P36_N_EXPERT) / P36_N_EXPERT : 0;
                if (!p36_streaming_full_layer_budget(&e->weights, e->ssd_streaming_full_layers,
                                                      &reserve, &resident_bytes)) {
                    p36_engine_close(e);
                    return 1;
                }
                fprintf(stderr, "p36: SSD full layers reduced %u -> %u for runtime headroom\n",
                        old, e->ssd_streaming_full_layers);
            }
            e->ssd_streaming_cache_experts -= (uint32_t)reserve;
            fprintf(stderr,
                    "p36: SSD streaming keeps %u full routed layers resident "
                    "(%.2f GiB); dynamic cache %u experts\n",
                    e->ssd_streaming_full_layers,
                    (double)resident_bytes / 1073741824.0,
                    e->ssd_streaming_cache_experts);
        } else if (e->ssd_streaming && e->ssd_streaming_full_layers_set) {
            fprintf(stderr, "p36: SSD streaming full resident layers disabled\n");
        }
        p36_gpu_set_streaming_full_layers(e->ssd_streaming ?
                                          e->ssd_streaming_full_layers : 0);
        p36_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
#ifdef P36_METAL
        p36_streaming_lock_static_weights(e);
#endif
        if (e->ssd_streaming) {
            uint64_t slab_gate_bytes = 0, slab_up_bytes = 0;
            uint64_t slab_down_bytes = 0, slab_expert_bytes = 0;
            if (p36_streaming_routed_expert_layout(
                    &e->weights, &slab_gate_bytes, &slab_up_bytes,
                    &slab_down_bytes, &slab_expert_bytes)) {
                p36_gpu_set_streaming_expert_cache_layout(
                    slab_gate_bytes, slab_up_bytes, slab_down_bytes);
                uint32_t routed = 0;
                for (uint32_t il = 0; il < P36_N_LAYER; il++) {
                    const p36_layer_weights *l = &e->weights.layer[il];
                    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) continue;
                    routed++;
                }
                uint32_t cache_floor = 2u * routed * P36_N_EXPERT_USED;
                if (e->ssd_streaming_cache_experts < cache_floor) {
                    fprintf(stderr,
                            "p36: WARNING: SSD streaming cache has %u experts; %u are recommended to retain two decode routes per cacheable layer\n",
                            e->ssd_streaming_cache_experts, cache_floor);
                }
            }
        }
        if (!p36_vulkan_set_model_spans(e) ||
            !p36_gpu_set_model_fd(e->model.fd)) {
            p36_engine_close(e);
            return 1;
        }
#ifdef P36_METAL
        if (e->ssd_streaming && e->ssd_streaming_full_layers != 0) {
            for (uint32_t il = 0;
                 il < e->ssd_streaming_full_layers; il++) {
                p36_gpu_stream_expert_table table =
                    p36_stream_expert_table_make(
                        &e->model, &e->weights.layer[il], il);
                if (!p36_gpu_stream_expert_cache_load_layer(&table)) {
                    fprintf(stderr,
                            "p36: Metal failed to load full resident routed layer %u\n",
                            il);
                    p36_engine_close(e);
                    return 1;
                }
            }
        }
#endif
        p36_vulkan_prewarm_weights(e);
        if (!p36_gpu_finish_model_cache()) {
            p36_engine_close(e);
            return 1;
        }
        if (!p36_engine_seed_streaming_expert_cache(e)) {
            p36_engine_close(e);
            return 1;
        }
    }
#endif
    *out = e;
    return 0;
}

void p36_engine_close(p36_engine *e) {
    if (!e) return;
#ifndef P36_NO_GPU
    p36_gpu_tensor_free(e->directional_steering_gpu);
    p36_gpu_tensor_free(e->session_batch_logits);
    if (p36_backend_uses_graph(e->backend)) p36_gpu_cleanup();
#endif
    free(e->directional_steering_dirs);
    vocab_free(&e->vocab);
    model_close(&e->vision_model);
    model_close(&e->mtp_model);
    model_close(&e->model);
    free(e->directional_steering_file);
    free(e->chat_template);
    p36_ssd_memory_lock_release(&e->simulated_memory);
    free(e);
}

void p36_engine_summary(p36_engine *e) {
    if (!e) return;
    model_summary(&e->model);
}

bool p36_engine_has_vision(p36_engine *e) {
    return e && e->vision_ready;
}

void p36_vision_embedding_free(p36_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->data);
    memset(embedding, 0, sizeof(*embedding));
}

static int p36_engine_vision_encode(p36_engine *e, p36_image *image,
                                    p36_vision_embedding *out,
                                    char *err, size_t errlen) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!e || !e->vision_ready) {
        if (err && errlen) snprintf(err, errlen, "vision is not enabled; pass --vision FILE");
        return 0;
    }
    if (!p36_engine_uses_vulkan_runtime(e)) {
        if (err && errlen) snprintf(err, errlen, "vision requires the Vulkan backend");
        return 0;
    }
    return p36_vision_encode_image(e, image, out, err, errlen);
}

int p36_engine_vision_encode_file(p36_engine *e, const char *path,
                                  p36_vision_embedding *out,
                                  char *err, size_t errlen) {
    p36_image image = {0};
    if (!p36_image_decode_file(&image, path, err, errlen)) return 0;
    int ok = p36_engine_vision_encode(e, &image, out, err, errlen);
    p36_image_free(&image);
    return ok;
}

int p36_engine_vision_encode_memory(p36_engine *e,
                                    const uint8_t *data, size_t len,
                                    p36_vision_embedding *out,
                                    char *err, size_t errlen) {
    p36_image image = {0};
    if (!p36_image_decode_memory(&image, data, len, err, errlen)) return 0;
    int ok = p36_engine_vision_encode(e, &image, out, err, errlen);
    p36_image_free(&image);
    return ok;
}

int p36_prompt_append_vision(p36_engine *e, p36_tokens *prompt,
                             p36_vision_span *span,
                             p36_vision_embedding *embedding,
                             char *err, size_t errlen) {
    if (!e || !prompt || !span || !embedding || !embedding->data ||
        !embedding->token_count || e->vocab.vision_start_id < 0 ||
        e->vocab.vision_end_id < 0 || e->vocab.image_pad_id < 0) {
        if (err && errlen) snprintf(err, errlen, "invalid vision prompt");
        return 0;
    }
    p36_tokens_push(prompt, e->vocab.vision_start_id);
    span->token_start = (uint32_t)prompt->len;
    for (uint32_t i = 0; i < embedding->token_count; i++)
        p36_tokens_push(prompt, e->vocab.image_pad_id);
    p36_tokens_push(prompt, e->vocab.vision_end_id);
    span->embedding = *embedding;
    memset(embedding, 0, sizeof(*embedding));
    return 1;
}

int p36_engine_embd_dim(p36_engine *e) {
    return e ? (int)P36_N_EMBD : 0;
}

int p36_engine_vocab_size(p36_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

int p36_engine_power(p36_engine *e) {
    return e && e->power_percent > 0 ? e->power_percent : 100;
}

int p36_engine_set_power(p36_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *p36_engine_model_name(p36_engine *e) {
    if (e && e->kat_coder) return "kat-coder-v2.5-dev";
    if (e && e->variant == P36_VARIANT_27B) return "qwen3.6-27b";
    return "qwen3.6-35b-a3b";
}

int p36_engine_model_id(p36_engine *e) {
    if (e && e->kat_coder) return 2;
    return e && e->variant == P36_VARIANT_27B ? 3 : 1;
}

uint32_t p36_engine_prefill_chunk(p36_engine *e) {
    if (!e) return 0;
#ifndef P36_NO_GPU
    if (p36_engine_uses_vulkan_runtime(e)) return p36_engine_gpu_prefill_cap(e);
#endif
    return e->cpu_prefill_cap;
}

bool p36_engine_is_qwen38(p36_engine *e) {
    return e && e->variant == P36_VARIANT_27B;
}

bool p36_engine_is_kat_coder(p36_engine *e) {
    return e && e->kat_coder;
}

const char *p36_engine_chat_template(p36_engine *e) {
    return e ? e->chat_template : NULL;
}

void p36_engine_sampling_defaults(p36_engine *e, float *temperature,
                                  int *top_k, float *top_p, float *min_p) {
    if (!temperature || !top_k || !top_p || !min_p) return;
    *temperature = P36_DEFAULT_TEMPERATURE;
    *top_k = 0;
    *top_p = P36_DEFAULT_TOP_P;
    *min_p = P36_DEFAULT_MIN_P;
    if (!e || !e->kat_coder) return;
    *temperature = 0.7f;
    *top_k = 20;
    *top_p = 0.8f;
    *min_p = 0.05f;
}

int p36_inspect_model(const char *model_path) {
    p36_engine *e = NULL;
    p36_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = P36_BACKEND_CPU;
    if (p36_engine_open(&e, &opt) != 0) return 1;
    p36_engine_summary(e);
    p36_engine_close(e);
    return 0;
}

void p36_engine_dump_tokens(p36_engine *e, const p36_tokens *tokens) {
    if (!e || !tokens) return;
    dump_tokens_fp(stdout, &e->vocab, tokens);
}

int p36_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    p36_engine *e = NULL;
    p36_engine_options opt;
    p36_tokens tokens = {0};
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = P36_BACKEND_CPU;
    if (!fp) fp = stdout;
    if (p36_engine_open(&e, &opt) != 0) return 1;
    p36_tokenize_text(e, text ? text : "", &tokens);
    dump_tokens_fp(fp, &e->vocab, &tokens);
    p36_tokens_free(&tokens);
    p36_engine_close(e);
    return 0;
}

static int p36_unsupported_runtime(char *err, size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "Qwen inference/session path is not wired on the new baseline yet");
    return 1;
}

int p36_engine_generate_argmax(p36_engine *e, const p36_tokens *prompt,
                               int n_predict, int ctx_size,
                               p36_token_emit_fn emit,
                               p36_generation_done_fn done,
                               void *emit_ud,
                               p36_session_progress_fn progress,
                               void *progress_ud) {
    p36_session *s = NULL;
    char err[160] = {0};
    int eos;
    int generated = 0;
    double t_prefill0;
    double t_prefill1;
    double t_decode0;
    double t_decode1;
    int rc = 1;
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return 1;
    if (p36_session_create(&s, e, ctx_size) != 0 || !s) return 1;
    p36_session_set_progress(s, progress, progress_ud);
    t_prefill0 = p36_now_sec();
    if (p36_session_sync(s, prompt, err, sizeof(err)) != 0) goto done;
    t_prefill1 = p36_now_sec();
    eos = p36_token_eos(e);
    t_decode0 = p36_now_sec();
    int room = p36_session_ctx(s) - p36_session_pos(s);
    int max_tokens = room <= 1 ? 0 : n_predict;
    if (max_tokens > room - 1) max_tokens = room - 1;
    for (int i = 0; i < max_tokens; i++) {
        int token = p36_session_argmax(s);
        if (token < 0) goto done;
        token = p36_session_eos_to_think_close(s, token);
        if (token == eos) break;
        if (emit) emit(emit_ud, token);
        if (p36_session_eval(s, token, err, sizeof(err)) != 0) goto done;
        generated++;
    }
    t_decode1 = p36_now_sec();
    if (done) done(emit_ud);
    p36_log(stderr,
            P36_LOG_TIMING,
            "p36: prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)generated / (t_decode1 - t_decode0) : 0.0);
    rc = 0;
done:
    p36_session_free(s);
    return rc;
}

int p36_engine_head_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_first_token_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_full_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_prompt_test(p36_engine *e, const p36_tokens *prompt, int ctx_size) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    return 1;
}

int p36_engine_debug_tensor_row(p36_engine *e, const char *tensor_name, uint64_t row, float *dst, uint32_t n) {
    p36_tensor *t;
    if (!e || !tensor_name || !dst) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t) return 1;
    return p36_tensor_row_to_float(&e->model, t, row, dst, n) ? 0 : 1;
}

int p36_engine_debug_first_tensor_of_type(p36_engine *e, uint32_t type, char *name, size_t name_cap, uint32_t *n) {
    if (!e || !name || name_cap == 0) return 1;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        if (t->name.len + 1 > name_cap) return 1;
        memcpy(name, t->name.ptr, t->name.len);
        name[t->name.len] = '\0';
        if (n) *n = (uint32_t)t->dim[0];
        return 0;
    }
    return 1;
}

int p36_engine_debug_tensor_of_type_at(p36_engine *e, uint32_t type, uint32_t ordinal,
                                       char *name, size_t name_cap,
                                       uint32_t *in_dim, uint32_t *out_dim) {
    uint32_t seen = 0;
    if (!e || !name || name_cap == 0) return 1;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        if (seen++ != ordinal) continue;
        if (t->name.len + 1 > name_cap) return 1;
        memcpy(name, t->name.ptr, t->name.len);
        name[t->name.len] = '\0';
        if (in_dim) *in_dim = (uint32_t)t->dim[0];
        if (out_dim) *out_dim = t->ndim > 1 ? (uint32_t)t->dim[1] : 1u;
        return 0;
    }
    return 1;
}

uint32_t p36_engine_debug_count_tensors_of_type(p36_engine *e, uint32_t type) {
    uint32_t n = 0;
    if (!e) return 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        n++;
    }
    return n;
}

int p36_engine_debug_tensor_row_packed(p36_engine *e, const char *tensor_name, uint64_t row,
                                       void *dst, uint64_t dst_cap,
                                       uint64_t *row_bytes, uint32_t *type, uint32_t *n) {
    p36_tensor *t;
    uint64_t got_row_bytes;
    uint64_t row_count;
    const uint8_t *src;
    if (!e || !tensor_name) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t || t->ndim == 0) return 1;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return 1;
    if (row_bytes) *row_bytes = got_row_bytes;
    if (type) *type = t->type;
    if (n) *n = (uint32_t)t->dim[0];
    if (!dst) return 0;
    if (dst_cap < got_row_bytes) return 1;
    src = e->model.map + t->abs_offset + row * got_row_bytes;
    memcpy(dst, src, (size_t)got_row_bytes);
    return 0;
}

int p36_session_create(p36_session **out, p36_engine *e, int ctx_size) {
    p36_session *s;
    if (!out || !e || ctx_size <= 0 || ctx_size > P36_CONTEXT_ALLOC_MAX) return 1;
    uint64_t reserved = 0;
#ifndef P36_NO_GPU
    if (!p36_streaming_reserve_session(e, ctx_size, &reserved)) {
        fprintf(stderr, "p36: requested context leaves insufficient SSD streaming memory\n");
        return 1;
    }
#endif
    s = xcalloc(1, sizeof(*s));
    s->reserved_context_bytes = reserved;
    s->engine = e;
    s->ctx_size = ctx_size;
    s->logits = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->logits));
    s->sample_probs = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->sample_probs));
    memset(s->logits, 0, (size_t)P36_N_VOCAB * sizeof(*s->logits));
    s->logits_host_valid = true;
    s->gpu_top2_valid = false;
    if (p36_engine_uses_vulkan_runtime(e)) {
#ifdef P36_NO_GPU
        s->runtime = NULL;
#else
        s->runtime = p36_vulkan_runtime_create(ctx_size,
                                               p36_engine_gpu_prefill_cap(e),
                                               e->mtp_ready,
                                               e->quality,
                                               e->ssd_streaming,
                                               e->cache_type_k,
                                               e->cache_type_v);
#endif
    } else {
        s->runtime = p36_cpu_runtime_create(ctx_size,
                                            e->cpu_prefill_cap,
                                            e->cache_type_k,
                                            e->cache_type_v);
    }
    if (!s->runtime) {
        free(s->logits);
        free(s->sample_probs);
        free(s->mtp_logits);
        free(s->vision_spans);
        free(s);
        return 1;
    }
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->mtp_logits));
        memset(s->mtp_logits, 0, (size_t)P36_N_VOCAB * sizeof(*s->mtp_logits));
        s->mtp_draft_token = -1;
    }
    p36_session_reset_runtime(s);
    e->session_bytes += reserved;
    e->live_sessions++;
    *out = s;
    return 0;
}

void p36_session_free(p36_session *s) {
    if (!s) return;
    s->engine->session_bytes -= s->reserved_context_bytes;
    s->engine->live_sessions--;
    p36_tokens_free(&s->checkpoint);
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef P36_NO_GPU
        p36_vulkan_runtime_free((p36_vulkan_runtime *)s->runtime);
#endif
    } else {
        p36_cpu_runtime_free((p36_cpu_runtime *)s->runtime);
    }
    free(s->logits);
    free(s->sample_probs);
    free(s->mtp_logits);
    free(s->vision_spans);
    free(s);
}

void p36_session_set_progress(p36_session *s, p36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

int p36_session_power(p36_session *s) {
    return s ? p36_engine_power(s->engine) : 100;
}

static bool p36_session_reserve_kv(p36_session *s, uint32_t need,
                                   uint32_t rows) {
    if (!s || need > (uint32_t)s->ctx_size) return false;
#ifdef P36_NO_GPU
    (void)rows;
#endif
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifdef P36_NO_GPU
        return false;
#else
        return p36_vulkan_runtime_reserve_kv(
            (p36_vulkan_runtime *)s->runtime, need,
            (uint32_t)s->ctx_size, rows);
#endif
    }
    return p36_cpu_runtime_reserve_kv(
        (p36_cpu_runtime *)s->runtime, need, (uint32_t)s->ctx_size);
}

int p36_session_set_power(p36_session *s, int power_percent) {
    return s ? p36_engine_set_power(s->engine, power_percent) : 1;
}

float p36_session_directional_steering_ffn(p36_session *s) {
    return s && s->engine ? s->engine->directional_steering_ffn_scale : 0.0f;
}

int p36_session_set_directional_steering_ffn(p36_session *s, float scale) {
    if (!s || !s->engine || !isfinite(scale) || scale < -100.0f || scale > 100.0f)
        return 1;
    bool loaded = s->engine->directional_steering_dirs != NULL;
#ifndef P36_NO_GPU
    loaded = loaded || s->engine->directional_steering_gpu != NULL;
#endif
    if (scale != 0.0f && !loaded) {
        fprintf(stderr, "p36: live FFN steering needs a vector loaded at startup\n");
        return 1;
    }
    s->engine->directional_steering_ffn_scale = scale;
    s->mtp_draft_valid = false;
    return 0;
}

void p36_session_set_display_progress(p36_session *s, p36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void p36_session_set_cancel(p36_session *s, p36_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

/* Evaluate prompt tokens [start, len) on top of the current checkpoint
 * (start must equal s->checkpoint.len), leaving the last token's logits in
 * s->logits.  The Vulkan runtime walks the range in prefill-sized batches;
 * the CPU path uses smaller layer-major chunks that keep recurrent state
 * updates ordered while batching projections and FFN work. */
static int p36_session_prefill_range(p36_session *s, const p36_tokens *prompt, int start,
                                     char *err, size_t errlen) {
    int i = start;
#ifndef P36_NO_GPU
    if (start == 0 && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_gpu_stream_expert_cache_reset_route_hotness();
    }
#endif
    while (i < prompt->len) {
        if (s->cancel && s->cancel(s->cancel_ud)) {
            if (err && errlen) snprintf(err, errlen, "interrupted");
            return P36_SESSION_SYNC_INTERRUPTED;
        }
        int n = prompt->len - i;
        bool compute_logits;
        bool ok;
#ifndef P36_NO_GPU
        if (p36_engine_uses_vulkan_runtime(s->engine)) {
            p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
        } else
#endif
        {
            p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
        }
        if (!p36_session_reserve_kv(s, (uint32_t)(i + n),
                                    (uint32_t)s->checkpoint.len)) {
            if (err && errlen) snprintf(err, errlen, "failed to grow KV cache");
            return 1;
        }
        compute_logits = (i + n) == prompt->len;
#ifndef P36_NO_GPU
        if (p36_engine_uses_vulkan_runtime(s->engine)) {
            ok = p36_forward_tokens_vulkan(s, prompt->v + i, (uint32_t)n,
                                           (uint32_t)i, compute_logits);
        } else
#endif
        {
            p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
            ok = !p36_engine_uses_vulkan_runtime(s->engine) &&
                 (n == 1 && rt && rt->prefill_cap == 1
                  ? p36_forward_token_cpu(s, prompt->v[i], (uint32_t)i, compute_logits)
                  : p36_forward_tokens_cpu(s, prompt->v + i, (uint32_t)n, (uint32_t)i, compute_logits));
        }
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "%s prefill failed at token %d",
                                         p36_backend_name(s->engine->backend), i);
            s->checkpoint_valid = false;
            return 1;
        }
        for (int j = 0; j < n; j++) p36_tokens_push(&s->checkpoint, prompt->v[i + j]);
        s->checkpoint_valid = true;
        i += n;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_chunk", i, prompt->len);
    }
    return 0;
}

/* Image pads alone cannot identify the image that conditioned the live state. */
const p36_vision_span *p36_session_vision_spans(const p36_session *s, size_t *count) {
    if (count) *count = s && s->checkpoint_valid ? s->vision_count : 0;
    return s && s->checkpoint_valid ? s->vision_spans : NULL;
}

bool p36_session_vision_prefix_matches(const p36_session *s,
                                       const p36_vision_span *images, size_t count) {
    if (!s || !s->checkpoint_valid || (count && !images)) return false;
    if (count < s->vision_count) return false;
    for (size_t i = 0; i < s->vision_count; i++) {
        const p36_vision_span *a = &s->vision_spans[i], *b = &images[i];
        if (a->token_start != b->token_start ||
            a->embedding.token_count != b->embedding.token_count ||
            a->embedding.grid_width != b->embedding.grid_width ||
            a->embedding.grid_height != b->embedding.grid_height ||
            memcmp(a->embedding.fingerprint, b->embedding.fingerprint, 32)) return false;
    }
    if (s->vision_state && !s->vision_count) return false;
    for (size_t i = s->vision_count; i < count; i++) {
        if (!images[i].token_start || images[i].token_start - 1u < (uint32_t)s->checkpoint.len)
            return false;
    }
    return true;
}

int p36_session_sync(p36_session *s, const p36_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt) {
        if (err && errlen) snprintf(err, errlen, "invalid sync arguments");
        return 1;
    }
    if (prompt->len <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (!s->vision_state && s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        p36_tokens_starts_with(prompt, &s->checkpoint))
    {
        return p36_session_prefill_range(s, prompt, s->checkpoint.len, err, errlen);
    }
    p36_session_reset_runtime(s);
    s->vision_state = false;
    free(s->vision_spans);
    s->vision_spans = NULL;
    s->vision_count = 0;
    s->rope_delta = 0;
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    return p36_session_prefill_range(s, prompt, 0, err, errlen);
}

int p36_session_sync_vision(p36_session *s, const p36_tokens *prompt,
                            const p36_vision_span *images, size_t image_count,
                            char *err, size_t errlen) {
    if (!s || !prompt || !images || image_count == 0 ||
        !p36_engine_has_vision(s->engine)) {
        if (err && errlen) snprintf(err, errlen, "invalid vision sync arguments");
        return 1;
    }
    if (!p36_engine_uses_vulkan_runtime(s->engine)) {
        if (err && errlen) snprintf(err, errlen, "vision prompts require the Vulkan backend");
        return 1;
    }
    if (prompt->len <= 0 || prompt->len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "vision prompt exceeds context");
        return 1;
    }
    bool reuse = p36_session_vision_prefix_matches(s, images, image_count) &&
                 prompt->len >= s->checkpoint.len &&
                 p36_tokens_starts_with(prompt, &s->checkpoint);
    uint64_t previous_end = 0;
    for (size_t k = 0; k < image_count; k++) {
        const p36_vision_span *span = &images[k];
        uint64_t end = (uint64_t)span->token_start + span->embedding.token_count;
        if ((!span->embedding.data && !(reuse && k < s->vision_count)) ||
            !span->embedding.token_count ||
            !span->embedding.grid_width || !span->embedding.grid_height ||
            (uint64_t)span->embedding.grid_width * span->embedding.grid_height !=
                span->embedding.token_count ||
            span->token_start == 0 || span->token_start - 1u < previous_end ||
            end >= (uint64_t)prompt->len ||
            prompt->v[span->token_start - 1] != s->engine->vocab.vision_start_id ||
            prompt->v[end] != s->engine->vocab.vision_end_id) {
            if (err && errlen) snprintf(err, errlen, "malformed vision span");
            return 1;
        }
        for (uint64_t i = span->token_start; i < end; i++) {
            if (prompt->v[i] != s->engine->vocab.image_pad_id) {
                if (err && errlen) snprintf(err, errlen, "vision span does not cover image tokens");
                return 1;
            }
        }
        previous_end = end + 1;
    }

    int pos = 0;
    size_t first_image = 0;
    if (reuse) {
        pos = s->checkpoint.len;
        first_image = s->vision_count;
    } else {
        p36_session_invalidate(s);
    }
    p36_vision_span *identities = realloc(s->vision_spans, image_count * sizeof(*identities));
    if (!identities) {
        if (err && errlen) snprintf(err, errlen, "unable to allocate image identities");
        p36_session_invalidate(s);
        return 1;
    }
    s->vision_spans = identities;
    s->vision_state = true;
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
#ifndef P36_NO_GPU
    if (!reuse) p36_gpu_stream_expert_cache_reset_route_hotness();
    for (size_t k = first_image; k < image_count; k++) {
        const p36_vision_span *span = &images[k];
        p36_tokens prefix = *prompt;
        prefix.len = (int)span->token_start;
        int rc = p36_session_prefill_range(s, &prefix, pos, err, errlen);
        if (rc != 0) return rc;
        pos = prefix.len;
        uint32_t base = (uint32_t)((int32_t)pos + s->rope_delta);
        uint32_t *positions = malloc((uint64_t)span->embedding.token_count *
                                     3u * sizeof(*positions));
        if (!positions) {
            if (err && errlen) snprintf(err, errlen,
                                        "unable to allocate vision positions");
            return 1;
        }
        for (uint32_t i = 0; i < span->embedding.token_count; i++) {
            positions[i * 3u] = base;
            positions[i * 3u + 1u] = base + i / span->embedding.grid_width;
            positions[i * 3u + 2u] = base + i % span->embedding.grid_width;
        }
        uint32_t done = 0;
        p36_vulkan_runtime *rt = s->runtime;
        while (done < span->embedding.token_count) {
            if (s->cancel && s->cancel(s->cancel_ud)) {
                if (err && errlen) snprintf(err, errlen, "interrupted");
                free(positions);
                return P36_SESSION_SYNC_INTERRUPTED;
            }
            uint32_t n = span->embedding.token_count - done;
            if (n > rt->prefill_cap) n = rt->prefill_cap;
            if (!p36_session_reserve_kv(s, (uint32_t)pos + n,
                                        (uint32_t)s->checkpoint.len) ||
                !p36_forward_embeddings_vulkan(
                    s, prompt->v + pos,
                    span->embedding.data + (uint64_t)done * P36_N_EMBD,
                    positions + (uint64_t)done * 3u,
                    n, (uint32_t)pos,
                    pos + (int)n == prompt->len)) {
                if (err && errlen) snprintf(err, errlen,
                                             "vision prefill failed at token %d", pos);
                s->checkpoint_valid = false;
                free(positions);
                return 1;
            }
            for (uint32_t i = 0; i < n; i++)
                p36_tokens_push(&s->checkpoint, prompt->v[pos + (int)i]);
            s->checkpoint_valid = true;
            pos += (int)n;
            done += n;
            if (s->progress) s->progress(s->progress_ud, "prefill_chunk", pos, prompt->len);
            if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_chunk", pos, prompt->len);
        }
        free(positions);
        uint32_t extent = span->embedding.grid_width > span->embedding.grid_height ?
                          span->embedding.grid_width : span->embedding.grid_height;
        s->rope_delta = (int32_t)(base + extent) - pos;
        s->vision_spans[k] = *span;
        s->vision_spans[k].embedding.data = NULL;
        s->vision_count = k + 1;
    }
    return p36_session_prefill_range(s, prompt, pos, err, errlen);
#else
    (void)previous_end;
    (void)pos;
    (void)first_image;
    return 1;
#endif
}

bool p36_session_has_vision_state(const p36_session *s) {
    return s && s->vision_state;
}

bool p36_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

p36_session_rewrite_result p36_session_rewrite_from_common(p36_session *s, const p36_tokens *prompt, int common,
                                                           char *err, size_t errlen) {
    if (!s || !prompt || common < 0 || common > prompt->len) {
        if (err && errlen) snprintf(err, errlen, "invalid rewrite arguments");
        return P36_SESSION_REWRITE_ERROR;
    }
    if (p36_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        return p36_session_sync(s, prompt, err, errlen) == 0 ? P36_SESSION_REWRITE_OK : P36_SESSION_REWRITE_ERROR;
    }
    if (common == s->checkpoint.len && common == prompt->len) return P36_SESSION_REWRITE_OK;
    s->checkpoint.len = common;
    s->checkpoint_valid = true;
    if (p36_session_prefill_range(s, prompt, common, err, errlen) != 0) {
        return P36_SESSION_REWRITE_ERROR;
    }
    return P36_SESSION_REWRITE_OK;
}

int p36_session_common_prefix(p36_session *s, const p36_tokens *prompt) {
    int n;
    int i;
    if (!s || !prompt || !s->checkpoint_valid) return 0;
    n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

static bool p36_session_ensure_logits_host(p36_session *s) {
    if (!s || !s->logits) return false;
    if (s->logits_host_valid) return true;
#ifndef P36_NO_GPU
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (rt && p36_gpu_tensor_read(
                rt->logits, 0, s->logits,
                (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) {
            s->logits_host_valid = true;
            return true;
        }
    }
#endif
    return false;
}

int p36_session_argmax(p36_session *s) {
    if (!s || !s->logits) return -1;
    if (s->gpu_top2_valid) return s->gpu_top2[0];
    if (!p36_session_ensure_logits_host(s)) return -1;
    return sample_argmax(s->logits, P36_N_VOCAB);
}

int p36_session_argmax_excluding(p36_session *s, int excluded_id) {
    int best = -1;
    float best_logit = P36_NEG_INF;
    if (!s || !s->logits) return -1;
    if (s->gpu_top2_valid) {
        return s->gpu_top2[0] == excluded_id
            ? s->gpu_top2[1] : s->gpu_top2[0];
    }
    if (!p36_session_ensure_logits_host(s)) return -1;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v;
        if ((int)i == excluded_id) continue;
        v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

/* GPU min-p prefilter path for the default sampler: the same arithmetic as
 * p36_sample_full_vocab's `top_p >= 1.0f` branch, except that the max and the
 * `scaled > reject_scaled` cut are done on the GPU (vulkan/logits_minp_pack.comp)
 * and the host walks only the survivors. The expf, the sum and the draw stay
 * here so the token stream is bit-identical to the scalar path. Returns the
 * token, or -1 when the packed list cannot reproduce that arithmetic exactly
 * (a block over its cap, no survivor, a non-Vulkan backend), in which case the
 * caller falls back to the scalar path. */
static int p36_session_sample_packed(p36_session *s, float temperature,
                                     int top_k, float top_p, float min_p,
                                     uint64_t *rng) {
#ifndef P36_NO_GPU
    p36_vulkan_runtime *rt;
    const uint32_t nb = P36_GPU_MINP_NB(P36_N_VOCAB);
    const uint32_t cap = P36_GPU_MINP_CAP;
    const uint32_t words = P36_GPU_MINP_WORDS(P36_N_VOCAB);
    const uint32_t cnt0 = P36_GPU_MINP_HDR_WORDS + nb;
    const uint32_t idx0 = cnt0 + nb;
    const uint32_t scl0 = idx0 + nb * cap;
    uint32_t *w;
    float reject_scaled = P36_NEG_INF;
    float sum = 0.0f, best_p = -1.0f, r;
    int best = -1, n = 0;
    bool have_reject = false;

    if (top_k != 0 || !(top_p >= 1.0f)) return -1;
    if (!(min_p > 0.0f) || !(min_p <= 1.0f) || !isfinite(min_p)) return -1;
    if (!p36_engine_uses_vulkan_runtime(s->engine)) return -1;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt || !rt->logits || !rt->minp_pack) return -1;

    /* Same retreated cutoff p36_sample_full_vocab derives. */
    {
        float cutoff = logf(min_p);
        for (int i = 0; i < 8 && isfinite(cutoff); i++) {
            cutoff = nextafterf(cutoff, -FLT_MAX);
            if (expf(cutoff) < min_p) {
                reject_scaled = cutoff;
                have_reject = true;
                break;
            }
        }
    }
    if (!have_reject) return -1;
    if (!p36_gpu_logits_minp_pack(rt->minp_pack, rt->logits, P36_N_VOCAB,
                                  temperature, reject_scaled)) return -1;

    w = xmalloc((size_t)words * sizeof(w[0]));
    if (!p36_gpu_tensor_read(rt->minp_pack, 0, w, (uint64_t)words * sizeof(w[0]))) {
        free(w);
        return -1;
    }
    for (uint32_t b = 0; b < nb; b++) {
        if (w[cnt0 + b] > cap) { free(w); return -1; }
    }
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t n_b = w[cnt0 + b];
        for (uint32_t k = 0; k < n_b; k++) {
            const uint32_t slot = b * cap + k;
            const uint32_t id = w[idx0 + slot];
            float scaled, p;
            memcpy(&scaled, &w[scl0 + slot], sizeof(scaled));
            p = -1.0f;
            if (scaled > reject_scaled) {
                p = expf(scaled);
                if (p < min_p) p = -1.0f;
            }
            s->sample_probs[id] = p;
            if (p >= 0.0f) { sum += p; n++; }
            if (p > best_p) { best_p = p; best = (int)id; }
        }
    }
    if (n == 0 || sum <= 0.0f || !isfinite(sum)) { free(w); return -1; }
    r = p36_sample_rng_f32(rng) * sum;
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t n_b = w[cnt0 + b];
        for (uint32_t k = 0; k < n_b; k++) {
            const uint32_t id = w[idx0 + b * cap + k];
            const float p = s->sample_probs[id];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) { free(w); return (int)id; }
        }
    }
    free(w);
    return best;
#else
    (void)s; (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    return -1;
#endif
}

int p36_session_sample(p36_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s || !s->logits) return -1;
    if (!rng || temperature <= 0.0f) return p36_session_argmax(s);
#ifndef P36_NO_GPU
    /* top_k in [1,8]: skip the full P36_N_VOCAB logits readback entirely.
     * p36_gpu_topk8_tensor selects the exact top-8 on-device (see
     * topk8.comp) and only the 64-byte result crosses to host, instead of
     * the ~1 MB logits vector p36_session_ensure_logits_host would copy.
     * Falls through to the CPU path below on any GPU failure. */
    if (top_k > 0 && top_k <= 8 && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        struct { int32_t ids[8]; float vals[8]; } topk;
        if (rt && p36_gpu_topk8_tensor(rt->topk8, rt->logits, P36_N_VOCAB) &&
            p36_gpu_tensor_read(rt->topk8, 0, &topk, sizeof(topk))) {
            int ids[8];
            float vals[8];
            for (int i = 0; i < 8; i++) { ids[i] = topk.ids[i]; vals[i] = topk.vals[i]; }
            /* The GPU always returns the true top-8; a smaller requested
             * top_k just takes the prefix, since it's already sorted
             * descending and top_k <= 8 here. */
            return p36_sample_from_sorted_topk(ids, vals, top_k, temperature,
                                               top_p <= 0.0f || top_p > 1.0f ? 1.0f : top_p,
                                               min_p < 0.0f ? 0.0f : min_p, rng);
        }
    }
#endif
    {
        const int packed = p36_session_sample_packed(s, temperature, top_k,
                                                     top_p, min_p, rng);
        if (packed >= 0) return packed;
    }
    if (!p36_session_ensure_logits_host(s)) return -1;
    return p36_sample_top_p_min_p(s->logits, P36_N_VOCAB, temperature,
                                  top_k, top_p, min_p, rng, s->sample_probs);
}

int p36_session_sample_penalized(p36_session *s,
                                 float temperature, int top_k,
                                 float top_p, float min_p,
                                 const int *tokens, int n_tokens,
                                 float presence_penalty,
                                 float frequency_penalty,
                                 float repeat_penalty,
                                 uint64_t *rng) {
    if (!s || !s->logits || !rng) return -1;
    if (!tokens || n_tokens <= 0 ||
        (presence_penalty == 0.0f && frequency_penalty == 0.0f &&
         repeat_penalty == 1.0f)) {
        return p36_session_sample(s, temperature, top_k, top_p, min_p, rng);
    }
    if (!p36_session_ensure_logits_host(s)) return -1;

    return p36_sample_logits_penalized(
        s->logits, P36_N_VOCAB, temperature, top_k, top_p, min_p,
        tokens, n_tokens, presence_penalty, frequency_penalty,
        repeat_penalty, rng, s->sample_probs);
}

int p36_session_argmax_penalized_excluding(p36_session *s, int excluded_id,
                                           const int *tokens, int n_tokens,
                                           float presence, float frequency,
                                           float repeat_penalty) {
    if (!s || excluded_id < 0 || excluded_id >= (int)P36_N_VOCAB ||
        !p36_session_ensure_logits_host(s)) return -1;
    float saved = s->logits[excluded_id];
    s->logits[excluded_id] = -INFINITY;
    uint64_t rng = 1;
    int token = p36_sample_logits_penalized(s->logits, P36_N_VOCAB,
        0, 0, 1, 0, tokens, n_tokens, presence, frequency, repeat_penalty,
        &rng, s->sample_probs);
    s->logits[excluded_id] = saved;
    return token;
}

bool p36_session_in_think(p36_session *s) {
    if (!s || !s->checkpoint_valid) return false;
    const p36_vocab *vocab = &s->engine->vocab;
    for (int i = s->checkpoint.len - 1; i >= 0; i--) {
        int t = s->checkpoint.v[i];
        if (t == vocab->think_end_id) return false;
        if (t == vocab->think_start_id) return true;
    }
    return false;
}

/* An EOS while the transcript's <think> block is still unclosed would end the
 * reply with no visible text at all: quantized models near-tie </think> and
 * EOS right after the forced-open <think>.  Callers map that EOS to </think>
 * and keep decoding; at most one substitution per think block, since the
 * close itself ends the in-think state. */
int p36_session_eos_to_think_close(p36_session *s, int token) {
    if (!s || token != s->engine->vocab.eos_id) return token;
    int close = s->engine->vocab.think_end_id;
    if (close < 0 || !p36_session_in_think(s)) return token;
    return close;
}

int p36_session_token_rank(p36_session *s, int token, int max_rank) {
    if (!s || !s->logits || token < 0 || token >= (int)P36_N_VOCAB ||
        max_rank <= 0) return 0;
    if (s->gpu_top2_valid) {
        if (s->gpu_top2[0] == token) return 1;
        if (s->gpu_top2[1] == token) return max_rank >= 2 ? 2 : 0;
        if (max_rank <= 2) return 0;
    }
    if (!p36_session_ensure_logits_host(s)) return 0;
    float target = s->logits[token];
    if (!isfinite(target)) return 0;
    int rank = 1;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        if ((int)i == token || !isfinite(s->logits[i])) continue;
        if (s->logits[i] > target ||
            (s->logits[i] == target && (int)i < token)) {
            if (++rank > max_rank) return 0;
        }
    }
    return rank;
}

int p36_session_top_logprobs(p36_session *s, p36_token_score *out, int k) {
    float max_logit = P36_NEG_INF;
    double sum = 0.0;
    if (!s || !out || !s->logits || k <= 0 ||
        !p36_session_ensure_logits_host(s)) return 0;
    if (k > (int)P36_N_VOCAB) k = (int)P36_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = P36_NEG_INF;
        out[i].logprob = P36_NEG_INF;
    }
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    {
        double logsum = (double)max_logit + log(sum);
        for (int i = 0; i < k && out[i].id >= 0; i++) {
            out[i].logprob = (float)((double)out[i].logit - logsum);
        }
    }
    return k;
}

static int p36_session_eval_internal(p36_session *s, int token, bool update_mtp,
                                     char *err, size_t errlen) {
    uint32_t pos;
    if (!s) {
        if (err && errlen) snprintf(err, errlen, "missing session");
        return 1;
    }
    if (s->checkpoint.len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context exhausted");
        return 1;
    }
    if (!p36_session_reserve_kv(s, (uint32_t)s->checkpoint.len + 1u,
                                (uint32_t)s->checkpoint.len)) {
        if (err && errlen) snprintf(err, errlen, "failed to grow KV cache");
        return 1;
    }
    s->mtp_draft_valid = false;
    pos = (uint32_t)s->checkpoint.len;
    if (!(p36_engine_uses_vulkan_runtime(s->engine)
#ifndef P36_NO_GPU
          ? p36_forward_token_vulkan(s, token, pos, true)
#else
          ? false
#endif
          : p36_forward_token_cpu(s, token, pos, true))) {
        if (err && errlen) snprintf(err, errlen, "%s decode failed at pos %d",
                                     p36_backend_name(s->engine->backend),
                                     s->checkpoint.len);
        s->checkpoint_valid = false;
        return 1;
    }
#ifndef P36_NO_GPU
    if (update_mtp && p36_engine_has_mtp(s->engine) && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (!p36_mtp_eval_vulkan(s, token, pos, rt->norm)) {
            if (err && errlen) snprintf(err, errlen, "MTP draft failed at pos %u", pos);
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, P36_N_VOCAB, &s->mtp_draft_margin);
        s->mtp_draft_valid = s->mtp_draft_token >= 0;
    }
#else
    (void)update_mtp;
#endif
    p36_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

#ifndef P36_NO_GPU
/* Decode one token with the MTP draft merged into the same submission:
 * the pipeline drains once (at the draft logits read) instead of once
 * per model, which is what makes the confidence gate affordable. */
static int p36_session_eval_with_draft(p36_session *s, int token, char *err, size_t errlen) {
    p36_vulkan_runtime *rt;
    uint32_t pos;
    if (s->checkpoint.len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context exhausted");
        return 1;
    }
    if (!p36_session_reserve_kv(s, (uint32_t)s->checkpoint.len + 1u,
                                (uint32_t)s->checkpoint.len)) {
        if (err && errlen) snprintf(err, errlen, "failed to grow KV cache");
        return 1;
    }
    rt = (p36_vulkan_runtime *)s->runtime;
    s->mtp_draft_valid = false;
    pos = (uint32_t)s->checkpoint.len;
    {
        const bool timing = getenv("P36_MTP_TIMING") != NULL;
        double t0 = timing ? p36_now_sec() : 0.0, t1 = t0, t2 = t0, t3 = t0, t4 = t0;
        bool ok = p36_forward_tokens_vulkan_into(s, &token, NULL, NULL,
                                                 1, pos, rt->logits, false) &&
                  p36_vulkan_update_last_h(rt, 0);
        if (timing) t1 = p36_now_sec();
        ok = ok && p36_mtp_eval_vulkan(s, token, pos, rt->norm);
        if (timing) t2 = p36_now_sec();
        ok = ok && p36_gpu_tensor_read(rt->logits, 0, s->logits, (uint64_t)P36_N_VOCAB * sizeof(*s->logits));
        if (timing) t3 = p36_now_sec();
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "Vulkan decode+draft failed at pos %u", pos);
            s->checkpoint_valid = false;
            return 1;
        }
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
        s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, P36_N_VOCAB, &s->mtp_draft_margin);
        if (timing) {
            t4 = p36_now_sec();
            fprintf(stderr, "p36: mtp timing eval fwd=%.2f mtp=%.2f read=%.2f argmax=%.2f ms\n",
                    (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t4 - t3) * 1000.0);
        }
    }
    s->mtp_draft_valid = s->mtp_draft_token >= 0;
    p36_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}
#endif

int p36_session_eval(p36_session *s, int token, char *err, size_t errlen) {
    return p36_session_eval_internal(s, token, false, err, errlen);
}

#ifndef P36_NO_GPU
static bool p36_sessions_eval_batch_vulkan_supported(p36_decode_item *items,
                                                      int count,
                                                      p36_engine *e) {
    const char *enabled = getenv("P36_VK_SESSION_BATCH");
    if ((enabled && enabled[0] && !strcmp(enabled, "0")) ||
        !items || count < 2 || count > 8 || !e ||
        !p36_backend_uses_graph(e->backend) || e->ssd_streaming) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (s && s->vision_state) return false;
        p36_vulkan_runtime *rt = s ? s->runtime : NULL;
        if (!s || s->engine != e || !s->checkpoint_valid || !rt ||
            rt->prefill_cap < (uint32_t)count) {
            return false;
        }
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (!p36_layer_is_full_attention(il)) continue;
            p36_vulkan_full_attn_cache *cache = &rt->full[il];
            bool f16 = cache->type_k == P36_KV_CACHE_F16 &&
                       cache->type_v == P36_KV_CACHE_F16;
            bool resident = cache->type_k == P36_KV_CACHE_Q8_0 &&
                            cache->type_v == P36_KV_CACHE_Q4_0;
            if (!f16 && !resident) return false;
        }
    }
    return true;
}
#endif

int p36_sessions_eval_batch(p36_decode_item *items, int count,
                            char *err, size_t errlen) {
    if (!items || count <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty decode batch");
        return 1;
    }
    if (count == 1) {
        return p36_session_eval(items[0].session, items[0].token, err, errlen);
    }

    p36_session *first = items[0].session;
    if (!first || !first->engine) {
        if (err && errlen) snprintf(err, errlen, "decode batch has no session");
        return 1;
    }
    p36_engine *e = first->engine;
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (!s || s->engine != e) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d belongs to a different engine", i);
            }
            return 1;
        }
        if (items[i].token < 0 || items[i].token >= (int)P36_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d has an invalid token", i);
            }
            return 1;
        }
        if (s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
        if (!p36_session_reserve_kv(s, (uint32_t)s->checkpoint.len + 1u,
                                    (uint32_t)s->checkpoint.len)) {
            if (err && errlen) snprintf(err, errlen,
                                        "decode batch item %d failed to grow KV cache", i);
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "decode batch repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

#ifndef P36_NO_GPU
    if (p36_sessions_eval_batch_vulkan_supported(items, count, e)) {
        bool ok = p36_sessions_eval_batch_vulkan(items, count);
        if (!ok) {
            for (int i = 0; i < count; i++) p36_session_invalidate(items[i].session);
            if (err && errlen) {
                snprintf(err, errlen, "%s batched decode failed",
                         p36_backend_name(e->backend));
            }
            return 1;
        }
        for (int i = 0; i < count; i++) {
            p36_session *s = items[i].session;
            p36_tokens_push(&s->checkpoint, items[i].token);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
        }
        if (getenv("P36_VK_SESSION_BATCH_LOG")) {
            fprintf(stderr, "p36: %s session batch rows=%d native=1\n",
                    p36_backend_name(e->backend), count);
        }
        return 0;
    }
#endif

    for (int i = 0; i < count; i++) {
        if (p36_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) p36_session_invalidate(items[j].session);
            return 1;
        }
    }
    if (getenv("P36_VK_SESSION_BATCH_LOG")) {
        fprintf(stderr, "p36: session batch rows=%d native=0 ordered=1\n", count);
    }
    return 0;
}

int p36_sessions_eval_batch_with_prefill(
        p36_decode_item *items, int count,
        p36_session *prefill_session, const p36_tokens *prefill_prompt,
        char *err, size_t errlen) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        if (err && errlen) snprintf(err, errlen, "invalid mixed model batch");
        return 1;
    }
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len >= prefill_session->ctx_size ||
        !p36_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (!s || s == prefill_session ||
            s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size ||
            items[i].token < 0 || items[i].token >= (int)P36_N_VOCAB) {
            if (err && errlen) snprintf(err, errlen, "invalid mixed decode item %d", i);
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "mixed decode repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

    int rc = p36_session_sync(prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = p36_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) p36_session_invalidate(prefill_session);
    return rc;
}

/* P36_MTP_TIMING: one line per speculative cycle.  `drains` counts open-batch
 * flushes (each one stalls until the GPU queue is empty), and read_kb is what
 * the host pulled back in that same window. */
static void p36_mtp_timing_report(uint32_t pos, const char *exit_reason,
                                  int verify_n, int commit_n, int n_draft,
                                  double c0, double c_fwd, double c_draft,
                                  double c_snap, double c_verify, double c_commit,
                                  uint64_t r0, uint64_t f0) {
    fprintf(stderr,
            "p36: mtp cycle pos=%u %-12s verify_n=%d commit_n=%d drafts=%d "
            "fwd=%.2f draft=%.2f snap=%.2f verify=%.2f commit=%.2f total=%.2f ms "
            "read_kb=%.0f drains=%" PRIu64 "\n",
            pos, exit_reason ? exit_reason : "-", verify_n, commit_n, n_draft,
            (c_fwd - c0) * 1000.0, (c_draft - c_fwd) * 1000.0,
            (c_snap - c_draft) * 1000.0, (c_verify - c_snap) * 1000.0,
            (c_commit - c_verify) * 1000.0, (c_commit - c0) * 1000.0,
            (double)(p36_gpu_read_bytes() - r0) / 1024.0,
            p36_gpu_read_flushes() - f0);
}

int p36_session_eval_speculative_argmax(p36_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    p36_engine *e;
    if (!s || !accepted || accepted_cap <= 0 || max_tokens <= 0) return 0;
    e = s->engine;
    if (!e || !p36_engine_has_mtp(e) || e->mtp_draft_tokens <= 1 ||
        !p36_engine_uses_vulkan_runtime(e)) {
        if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifndef P36_NO_GPU
    {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        int verify[P36_MTP_MAX_DRAFT];
        int row_tops[P36_MTP_MAX_DRAFT];
        int draft_cap = e->mtp_draft_tokens - 1;
        int verify_n;
        int commit_n;
        uint32_t start_pos;
        int prev_token;
        if (!rt || !rt->last_h || !rt->logits) return 0;

        /* P36_MTP_TIMING: per-phase wall clock for one speculative cycle plus
         * the host readback volume and the GPU drains it caused.  Diagnostic
         * only: nothing below changes the tokens produced. */
        const bool mtp_timing = getenv("P36_MTP_TIMING") != NULL;
        double tc0 = 0.0, tc_fwd = 0.0, tc_draft = 0.0, tc_snap = 0.0;
        double tc_verify = 0.0;
        uint64_t tr0 = 0, tf0 = 0;
        int n_draft = 0;
        if (mtp_timing) {
            tc0 = p36_now_sec();
            tr0 = p36_gpu_read_bytes();
            tf0 = p36_gpu_read_flushes();
        }

        /* Draft backoff: after unconfident drafts, decode plainly for a
         * few tokens so hard content does not pay the draft head on every
         * token. Confident content resets to drafting every token. */
        if (s->mtp_backoff > 0) {
            s->mtp_backoff--;
            if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
            accepted[0] = first_token;
            return 1;
        }

        /* Commit the target token with its MTP draft piggybacked on the
         * same submission; the confidence gate and the first-draft check
         * against the fresh target logits are then free. */
        if (p36_session_eval_with_draft(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        if (mtp_timing) {
            tc_fwd = p36_now_sec();
            tc_draft = tc_fwd;
            tc_snap = tc_fwd;
            tc_verify = tc_fwd;
        }

        if (draft_cap > P36_MTP_MAX_DRAFT) draft_cap = P36_MTP_MAX_DRAFT;
        if (draft_cap > accepted_cap - 1) draft_cap = accepted_cap - 1;
        if (draft_cap > max_tokens - 1) draft_cap = max_tokens - 1;
        if (draft_cap > s->ctx_size - s->checkpoint.len) draft_cap = s->ctx_size - s->checkpoint.len;
        if (rt->prefill_cap > 0 && draft_cap > (int)rt->prefill_cap) draft_cap = (int)rt->prefill_cap;
        if (first_token == eos_token || draft_cap <= 0 || !s->mtp_draft_valid) {
            if (mtp_timing) {
                p36_mtp_timing_report((uint32_t)s->checkpoint.len, "no-draft", 0, 1, 0,
                                      tc0, tc_fwd, tc_draft, tc_snap, tc_verify, tc_fwd, tr0, tf0);
            }
            return 1;
        }
        if ((e->mtp_margin > 0.0f && s->mtp_draft_margin < e->mtp_margin) ||
            s->mtp_draft_token != p36_session_argmax(s)) {
            s->mtp_backoff_len = s->mtp_backoff_len ? (s->mtp_backoff_len < 4 ? s->mtp_backoff_len * 2 : 4) : 1;
            s->mtp_backoff = s->mtp_backoff_len;
            p36_mtp_stats_add(0, 0, false);
            if (mtp_timing) {
                p36_mtp_timing_report((uint32_t)s->checkpoint.len, "gate-reject", 0, 1, 0,
                                      tc0, tc_fwd, tc_draft, tc_snap, tc_verify, tc_fwd, tr0, tf0);
            }
            return 1;
        }
        s->mtp_backoff_len = 0;

        start_pos = (uint32_t)s->checkpoint.len;
        verify[0] = s->mtp_draft_token;
        verify_n = 1;
        prev_token = verify[0];
        while (verify_n < draft_cap && prev_token != eos_token) {
            float margin = 0.0f;
            if (!p36_mtp_eval_vulkan(s, prev_token, start_pos + (uint32_t)verify_n - 1u, rt->mtp_cur)) {
                if (err && errlen) snprintf(err, errlen, "MTP recursive draft failed at pos %u",
                                            start_pos + (uint32_t)verify_n - 1u);
                s->checkpoint_valid = false;
                return -1;
            }
            s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, P36_N_VOCAB, &margin);
            if (s->mtp_draft_token < 0) break;
            if (e->mtp_margin > 0.0f && margin < e->mtp_margin) break;
            verify[verify_n++] = s->mtp_draft_token;
            prev_token = s->mtp_draft_token;
            n_draft++;
        }
        if (mtp_timing) tc_draft = p36_now_sec();

        if (!p36_vulkan_spec_frontier_snapshot(rt)) return 1;
        if (mtp_timing) tc_snap = p36_now_sec();
        if (!p36_vulkan_verify_suffix_tops(s, verify, (uint32_t)verify_n, start_pos, row_tops)) {
            if (!p36_vulkan_spec_frontier_restore(rt)) p36_session_invalidate(s);
            if (err && errlen) snprintf(err, errlen, "MTP target verifier failed at pos %u", start_pos);
            return -1;
        }
        if (mtp_timing) tc_verify = p36_now_sec();

        commit_n = 1;
        for (int i = 1; i < verify_n; i++) {
            if (row_tops[i - 1] != verify[i]) break;
            commit_n++;
        }
        for (int i = 0; i < commit_n; i++) {
            if (verify[i] == eos_token) {
                commit_n = i + 1;
                break;
            }
        }

        if (commit_n == verify_n) {
            /* Full accept: the verify pass already ran every committed
             * token's forward and left the last row's logits in s->logits;
             * adopting its KV/recurrent frontier makes the replay free. */
            if (!p36_vulkan_update_last_h(rt, (uint32_t)verify_n - 1u)) {
                if (err && errlen) snprintf(err, errlen, "MTP frontier adopt failed at pos %u", start_pos);
                p36_session_invalidate(s);
                return -1;
            }
        } else {
            if (!p36_vulkan_spec_frontier_restore(rt)) {
                if (err && errlen) snprintf(err, errlen, "MTP verifier rollback failed at pos %u", start_pos);
                p36_session_invalidate(s);
                return -1;
            }
            /* Replay the accepted prefix as one batch; refreshes s->logits
             * and last_h from the last accepted row. */
            bool replay;
            p36_gpu_set_micro_batch(true);
            replay = p36_forward_tokens_vulkan(s, verify, (uint32_t)commit_n, start_pos, true);
            p36_gpu_set_micro_batch(false);
            if (!replay) {
                if (err && errlen) snprintf(err, errlen, "MTP accepted-prefix replay failed at pos %u", start_pos);
                p36_session_invalidate(s);
                return -1;
            }
        }

        for (int i = 0; i < commit_n; i++) {
            accepted[1 + i] = verify[i];
            p36_tokens_push(&s->checkpoint, verify[i]);
        }
        if (mtp_timing) {
            p36_mtp_timing_report(start_pos,
                                  commit_n == verify_n ? "full-accept" : "partial",
                                  verify_n, commit_n, n_draft, tc0, tc_fwd, tc_draft,
                                  tc_snap, tc_verify, p36_now_sec(), tr0, tf0);
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        s->mtp_draft_token = -1;
        p36_mtp_stats_add(verify_n, commit_n, commit_n == verify_n);
        return 1 + commit_n;
    }
#else
    (void)eos_token;
    if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

const float *p36_session_logits(p36_session *s, int *n_vocab) {
    if (n_vocab) *n_vocab = P36_N_VOCAB;
    return p36_session_ensure_logits_host(s) ? s->logits : NULL;
}

static void p36_session_reset_runtime(p36_session *s) {
    if (!s) return;
    s->logits_host_valid = false;
    s->gpu_top2_valid = false;
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef P36_NO_GPU
        (void)p36_vulkan_runtime_reset((p36_vulkan_runtime *)s->runtime);
#endif
    } else {
        p36_cpu_runtime_reset((p36_cpu_runtime *)s->runtime);
    }
}

void p36_session_invalidate(p36_session *s) {
    if (!s) return;
    p36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->vision_state = false;
    free(s->vision_spans);
    s->vision_spans = NULL;
    s->vision_count = 0;
    s->rope_delta = 0;
}

void p36_session_rewind(p36_session *s, int pos) {
    p36_tokens prefix = {0};
    if (!s) return;
    if (s->vision_state) {
        p36_session_invalidate(s);
        return;
    }
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    if (pos == s->checkpoint.len) return;
    for (int i = 0; i < pos; i++) p36_tokens_push(&prefix, s->checkpoint.v[i]);
    p36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    if (pos > 0) {
        if (p36_session_sync(s, &prefix, NULL, 0) != 0) {
            p36_session_invalidate(s);
        }
    }
    p36_tokens_free(&prefix);
}

int p36_session_pos(p36_session *s) {
    return s ? s->checkpoint.len : 0;
}

int p36_session_ctx(p36_session *s) {
    return s ? s->ctx_size : 0;
}

int p36_engine_routed_quant_bits(p36_engine *e) {
    return e ? e->routed_quant_bits : 0;
}

bool p36_engine_has_mtp(p36_engine *e) {
    return e && p36_backend_uses_graph(e->backend) && e->mtp_ready;
}

int p36_engine_mtp_draft_tokens(p36_engine *e) {
    return p36_engine_has_mtp(e) ? e->mtp_draft_tokens : 1;
}

const p36_tokens *p36_session_tokens(p36_session *s) {
    return s ? &s->checkpoint : NULL;
}

static void p36_payload_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "session payload error");
}

static int p36_payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                       char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    while (bytes) {
        size_t n = bytes < sizeof(buf) ? (size_t)bytes : sizeof(buf);
        if (fread(buf, 1, n, src) != n) {
            p36_payload_set_err(err, errlen, "failed to read staged session payload");
            return 1;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            p36_payload_set_err(err, errlen, "failed to write staged session payload");
            return 1;
        }
        bytes -= n;
    }
    return 0;
}

static uint64_t p36_full_k_bytes(uint32_t rows, p36_kv_cache_type type) {
    return (uint64_t)rows * p36_kv_cache_row_bytes(type, P36_N_HEAD_KV * P36_N_HEAD_DIM);
}

static uint64_t p36_full_v_bytes(uint32_t rows, p36_kv_cache_type type) {
    return (uint64_t)rows * p36_kv_cache_row_bytes(type, P36_N_HEAD_KV * P36_N_VALUE_DIM);
}

static uint64_t p36_recurrent_conv_bytes(void) {
    return (uint64_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float);
}

static uint64_t p36_recurrent_state_bytes(void) {
    return (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK * sizeof(float);
}

static int p36_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                                   char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!ptr || fwrite(ptr, 1, (size_t)bytes, fp) != bytes) {
        p36_payload_set_err(err, errlen, "failed to write session payload");
        return 1;
    }
    return 0;
}

static int p36_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                                  uint64_t *remaining, char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!remaining || *remaining < bytes || !ptr ||
        fread(ptr, 1, (size_t)bytes, fp) != bytes)
    {
        p36_payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    *remaining -= bytes;
    return 0;
}

static int p36_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    return p36_payload_write_bytes(fp, &v, sizeof(v), err, errlen);
}

static int p36_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                                char *err, size_t errlen) {
    return p36_payload_read_bytes(fp, v, sizeof(*v), remaining, err, errlen);
}

#ifndef P36_NO_GPU
static int p36_payload_write_tensor(FILE *fp, const p36_gpu_tensor *tensor,
                                    uint64_t bytes, char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (!p36_gpu_tensor_read(tensor, off, buf, n) ||
            fwrite(buf, 1, (size_t)n, fp) != n)
        {
            p36_payload_set_err(err, errlen, "failed to write GPU session payload");
            return 1;
        }
        off += n;
    }
    return 0;
}

static int p36_payload_read_tensor(p36_gpu_tensor *tensor, FILE *fp,
                                   uint64_t bytes, uint64_t *remaining,
                                   char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    if (!remaining || *remaining < bytes) {
        p36_payload_set_err(err, errlen, "truncated GPU session payload");
        return 1;
    }
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (fread(buf, 1, (size_t)n, fp) != n ||
            !p36_gpu_tensor_write(tensor, off, buf, n))
        {
            p36_payload_set_err(err, errlen, "failed to read GPU session payload");
            return 1;
        }
        off += n;
    }
    *remaining -= bytes;
    return 0;
}

static int p36_payload_write_recurrent_state(FILE *fp,
                                              const p36_gpu_tensor *tensor,
                                              bool f16,
                                              char *err, size_t errlen) {
    if (!f16)
        return p36_payload_write_tensor(fp, tensor,
                                        p36_recurrent_state_bytes(), err, errlen);
    uint16_t half[8192];
    float full[8192];
    uint64_t elems = p36_recurrent_state_bytes() / sizeof(float);
    for (uint64_t off = 0; off < elems; off += 8192u) {
        uint64_t n = elems - off;
        if (n > 8192u) n = 8192u;
        if (!p36_gpu_tensor_read(tensor, off * sizeof(uint16_t), half,
                                  n * sizeof(uint16_t))) {
            p36_payload_set_err(err, errlen, "failed to read FP16 recurrent state");
            return 1;
        }
        for (uint64_t i = 0; i < n; i++) full[i] = p36_f16_to_f32(half[i]);
        if (fwrite(full, sizeof(float), (size_t)n, fp) != n) {
            p36_payload_set_err(err, errlen, "failed to write recurrent state");
            return 1;
        }
    }
    return 0;
}

static int p36_payload_read_recurrent_state(p36_gpu_tensor *tensor, FILE *fp,
                                             bool f16, uint64_t *remaining,
                                             char *err, size_t errlen) {
    uint64_t bytes = p36_recurrent_state_bytes();
    if (!f16)
        return p36_payload_read_tensor(tensor, fp, bytes, remaining, err, errlen);
    if (!remaining || *remaining < bytes) {
        p36_payload_set_err(err, errlen, "truncated recurrent state");
        return 1;
    }
    uint16_t half[8192];
    float full[8192];
    uint64_t elems = bytes / sizeof(float);
    for (uint64_t off = 0; off < elems; off += 8192u) {
        uint64_t n = elems - off;
        if (n > 8192u) n = 8192u;
        if (fread(full, sizeof(float), (size_t)n, fp) != n) {
            p36_payload_set_err(err, errlen, "failed to read recurrent state");
            return 1;
        }
        for (uint64_t i = 0; i < n; i++) half[i] = p36_f32_to_f16(full[i]);
        if (!p36_gpu_tensor_write(tensor, off * sizeof(uint16_t), half,
                                   n * sizeof(uint16_t))) {
            p36_payload_set_err(err, errlen, "failed to upload FP16 recurrent state");
            return 1;
        }
    }
    *remaining -= bytes;
    return 0;
}
#endif

uint64_t p36_session_payload_bytes(p36_session *s) {
    if (!s || s->vision_state) return 0;
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != P36_KV_CACHE_F16 ||
         s->engine->cache_type_v != P36_KV_CACHE_F16);
    uint64_t bytes = (uint64_t)(typed ? P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS) * sizeof(uint32_t);
    bytes += (uint64_t)rows * sizeof(int32_t);
    bytes += (uint64_t)P36_N_VOCAB * sizeof(float);
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            bytes += sizeof(uint32_t) +
                     p36_full_k_bytes(rows, s->engine->cache_type_k) +
                     p36_full_v_bytes(rows, s->engine->cache_type_v);
        } else {
            bytes += p36_recurrent_conv_bytes() + p36_recurrent_state_bytes();
        }
    }
    return bytes;
}

int p36_session_write_staged_payload(const p36_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        p36_payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        p36_payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = p36_payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        p36_payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void p36_session_payload_file_free(p36_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int p36_session_stage_payload(p36_session *s, p36_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        p36_payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid || s->vision_state) {
        p36_payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
    size_t tmpl_len = strlen(tmpdir) + sizeof("/p36-session-payload.XXXXXX");
    char *tmpl = xmalloc(tmpl_len);
    snprintf(tmpl, tmpl_len, "%s/p36-session-payload.XXXXXX", tmpdir);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        free(tmpl);
        p36_payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        free(tmpl);
        if (err && errlen)
            snprintf(err, errlen, "failed to open staged session payload: %s",
                     strerror(saved));
        return 1;
    }

    int rc = p36_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        p36_payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            p36_payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        p36_payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        free(tmpl);
        return 1;
    }
    out->path = p36_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    free(tmpl);
    return 0;
}

int p36_session_save_payload(p36_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload save arguments");
        return 1;
    }
    if (!s->checkpoint_valid) {
        p36_payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->vision_state) {
        p36_payload_set_err(err, errlen, "image-conditioned checkpoints are not persistent");
        return 1;
    }
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != P36_KV_CACHE_F16 ||
         s->engine->cache_type_v != P36_KV_CACHE_F16);
    uint32_t h[P36_PAYLOAD_U32_FIELDS_TYPED_KV] = {
        P36_PAYLOAD_MAGIC,
        typed ? P36_PAYLOAD_VERSION_TYPED_KV : P36_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->engine && s->engine->backend == P36_BACKEND_CPU ?
            ((p36_cpu_runtime *)s->runtime)->prefill_cap :
#ifndef P36_NO_GPU
            ((p36_vulkan_runtime *)s->runtime)->prefill_cap,
#else
            0,
#endif
        rows,
        P36_N_VOCAB,
        P36_N_LAYER,
        P36_N_HEAD_KV,
        P36_N_HEAD_DIM,
        P36_N_VALUE_DIM,
        P36_N_SSM_CONV,
        P36_N_SSM_CONV_DIM,
        P36_N_SSM_STATE,
        P36_N_SSM_DT_RANK,
        s->engine ? (uint32_t)s->engine->cache_type_k : 0,
        s->engine ? (uint32_t)s->engine->cache_type_v : 0,
    };
    uint32_t nfields = typed ? P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS;
    for (uint32_t i = 0; i < nfields; i++) {
        if (p36_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    if (p36_payload_write_bytes(fp, s->checkpoint.v,
                                (uint64_t)rows * sizeof(int32_t),
                                err, errlen) != 0) return 1;
    if (s->engine->backend == P36_BACKEND_CPU) {
        p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_write_bytes(fp, s->logits,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                p36_full_attn_cache *c = &rt->full[il];
                if (!c->k || !c->v || c->len < rows) {
                    p36_payload_set_err(err, errlen, "CPU KV cache is shorter than checkpoint");
                    return 1;
                }
                if (p36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->k, p36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->v, p36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                    return 1;
            } else {
                p36_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_write_bytes(fp, c->conv, p36_recurrent_conv_bytes(), err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->state, p36_recurrent_state_bytes(), err, errlen) != 0)
                    return 1;
            }
        }
        return 0;
    }
#ifdef P36_NO_GPU
    return p36_unsupported_runtime(err, errlen);
#else
    p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt) return p36_unsupported_runtime(err, errlen);
    if (s->logits_host_valid) {
        if (p36_payload_write_bytes(fp, s->logits,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    err, errlen) != 0) return 1;
    } else if (p36_payload_write_tensor(fp, rt->logits,
                                        (uint64_t)P36_N_VOCAB * sizeof(float),
                                        err, errlen) != 0) {
        return 1;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            p36_vulkan_full_attn_cache *c = &rt->full[il];
            if (!c->k || !c->v || c->cap < rows) {
                p36_payload_set_err(err, errlen, "Vulkan KV cache is shorter than checkpoint");
                return 1;
            }
            if (p36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                p36_payload_write_tensor(fp, c->k, p36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                p36_payload_write_tensor(fp, c->v, p36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                return 1;
        } else {
            p36_vulkan_recurrent_cache *c = &rt->recurrent[il];
            if (p36_payload_write_tensor(fp, c->conv, p36_recurrent_conv_bytes(), err, errlen) != 0 ||
                p36_payload_write_recurrent_state(fp, c->state,
                                                  rt->recur_state_f16,
                                                  err, errlen) != 0)
                return 1;
        }
    }
    return 0;
#endif
}

int p36_session_load_payload(p36_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload load arguments");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t magic = 0, version = 0, third = 0;
    if (p36_payload_read_u32(fp, &magic, &remaining, err, errlen) != 0 ||
        p36_payload_read_u32(fp, &version, &remaining, err, errlen) != 0 ||
        p36_payload_read_u32(fp, &third, &remaining, err, errlen) != 0)
    {
        if (err && errlen) snprintf(err, errlen, "failed to read session payload header");
        return 1;
    }
    if (magic != P36_PAYLOAD_MAGIC) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload format");
        return 1;
    }
    if (version == P36_PAYLOAD_VERSION_TOKEN_ONLY) {
        uint32_t n_tokens = third;
        if (n_tokens >= (uint32_t)s->ctx_size) {
            p36_payload_set_err(err, errlen, "token-only checkpoint does not fit current context");
            return 1;
        }
        if (remaining != (uint64_t)n_tokens * sizeof(int32_t)) {
            if (err && errlen) snprintf(err, errlen, "mismatched token-only payload size");
            return 1;
        }
        p36_tokens prompt = {0};
        for (uint32_t i = 0; i < n_tokens; i++) {
            int32_t tok = 0;
            if (p36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) {
                p36_tokens_free(&prompt);
                return 1;
            }
            p36_tokens_push(&prompt, tok);
        }
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        int rc = prompt.len ? p36_session_sync(s, &prompt, err, errlen) : 0;
        if (!prompt.len) p36_session_reset_runtime(s);
        p36_tokens_free(&prompt);
        if (rc != 0) return 1;
        s->checkpoint_valid = true;
        return 0;
    }
    if (version != P36_PAYLOAD_VERSION && version != P36_PAYLOAD_VERSION_TYPED_KV) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload version");
        return 1;
    }

    uint32_t nfields = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS;
    uint32_t h[P36_PAYLOAD_U32_FIELDS_TYPED_KV] = {0};
    h[0] = magic;
    h[1] = version;
    h[2] = third;
    for (uint32_t i = 3; i < nfields; i++) {
        if (p36_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    uint32_t saved_ctx = h[2];
    uint32_t n_tokens = h[4];
    if (saved_ctx > (uint32_t)s->ctx_size || n_tokens >= (uint32_t)s->ctx_size) {
        p36_payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[5] != P36_N_VOCAB || h[6] != P36_N_LAYER ||
        h[7] != P36_N_HEAD_KV || h[8] != P36_N_HEAD_DIM ||
        h[9] != P36_N_VALUE_DIM || h[10] != P36_N_SSM_CONV ||
        h[11] != P36_N_SSM_CONV_DIM || h[12] != P36_N_SSM_STATE ||
        h[13] != P36_N_SSM_DT_RANK)
    {
        p36_payload_set_err(err, errlen, "KV checkpoint was written for a different p36 layout");
        return 1;
    }
    p36_kv_cache_type payload_type_k = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        (p36_kv_cache_type)h[14] : P36_KV_CACHE_F16;
    p36_kv_cache_type payload_type_v = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        (p36_kv_cache_type)h[15] : P36_KV_CACHE_F16;
    if (payload_type_k > P36_KV_CACHE_Q4_0 || payload_type_v > P36_KV_CACHE_Q4_0 ||
        !s->engine ||
        s->engine->cache_type_k != payload_type_k ||
        s->engine->cache_type_v != payload_type_v)
    {
        p36_payload_set_err(err, errlen, "KV checkpoint was written with a different cache type");
        return 1;
    }

    p36_session_reset_runtime(s);
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    if (!p36_session_reserve_kv(s, n_tokens, 0)) {
        p36_payload_set_err(err, errlen, "failed to grow KV cache for checkpoint");
        return 1;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        int32_t tok = 0;
        if (p36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) return 1;
        p36_tokens_push(&s->checkpoint, tok);
    }
    if (s->engine->backend == P36_BACKEND_CPU) {
        p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_read_bytes(fp, s->logits,
                                   (uint64_t)P36_N_VOCAB * sizeof(float),
                                   &remaining, err, errlen) != 0) return 1;
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                p36_full_attn_cache *c = &rt->full[il];
                if (p36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    p36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                c->len = rows;
                if (p36_payload_read_bytes(fp, c->k, p36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    p36_payload_read_bytes(fp, c->v, p36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                p36_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_read_bytes(fp, c->conv, p36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    p36_payload_read_bytes(fp, c->state, p36_recurrent_state_bytes(), &remaining, err, errlen) != 0)
                    return 1;
            }
        }
    } else {
#ifdef P36_NO_GPU
        return p36_unsupported_runtime(err, errlen);
#else
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_read_tensor(rt->logits, fp,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    &remaining, err, errlen) != 0) return 1;
        s->logits_host_valid = false;
        s->gpu_top2_valid = false;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                p36_vulkan_full_attn_cache *c = &rt->full[il];
                if (p36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    p36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                if (p36_payload_read_tensor(c->k, fp, p36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    p36_payload_read_tensor(c->v, fp, p36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                p36_vulkan_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_read_tensor(c->conv, fp, p36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    p36_payload_read_recurrent_state(c->state, fp,
                                                     rt->recur_state_f16,
                                                     &remaining, err, errlen) != 0)
                    return 1;
            }
        }
#endif
    }
    if (remaining != 0) {
        p36_payload_set_err(err, errlen, "trailing bytes in session payload");
        p36_session_invalidate(s);
        return 1;
    }
    s->checkpoint_valid = true;
    return 0;
}

int p36_session_save_snapshot(p36_session *s, p36_session_snapshot *snap, char *err, size_t errlen) {
    p36_payload_header hdr;
    uint64_t bytes;
    uint8_t *p;
    if (!s || !snap) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot save arguments");
        return 1;
    }
    bytes = sizeof(hdr) + (uint64_t)s->checkpoint.len * sizeof(int32_t);
    if (snap->cap < bytes) {
        snap->ptr = xrealloc(snap->ptr, (size_t)bytes);
        snap->cap = bytes;
    }
    hdr.magic = P36_PAYLOAD_MAGIC;
    hdr.version = P36_PAYLOAD_VERSION_TOKEN_ONLY;
    hdr.n_tokens = (uint32_t)s->checkpoint.len;
    memcpy(snap->ptr, &hdr, sizeof(hdr));
    p = snap->ptr + sizeof(hdr);
    if (hdr.n_tokens) memcpy(p, s->checkpoint.v, (size_t)hdr.n_tokens * sizeof(int32_t));
    snap->len = bytes;
    (void)err;
    (void)errlen;
    return 0;
}

int p36_session_load_snapshot(p36_session *s, const p36_session_snapshot *snap, char *err, size_t errlen) {
    const p36_payload_header *hdr;
    const int32_t *tok;
    if (!s || !snap || !snap->ptr || snap->len < sizeof(p36_payload_header)) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot payload");
        return 1;
    }
    hdr = (const p36_payload_header *)snap->ptr;
    if (hdr->magic != P36_PAYLOAD_MAGIC ||
        hdr->version != P36_PAYLOAD_VERSION_TOKEN_ONLY)
    {
        if (err && errlen) snprintf(err, errlen, "unsupported snapshot format");
        return 1;
    }
    if (snap->len != sizeof(*hdr) + (uint64_t)hdr->n_tokens * sizeof(int32_t)) {
        if (err && errlen) snprintf(err, errlen, "mismatched snapshot size");
        return 1;
    }
    tok = (const int32_t *)(snap->ptr + sizeof(*hdr));
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    for (uint32_t i = 0; i < hdr->n_tokens; i++) p36_tokens_push(&s->checkpoint, tok[i]);
    if (s->checkpoint.len > 0) {
        p36_tokens prompt = {0};
        p36_tokens_copy(&prompt, &s->checkpoint);
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        if (p36_session_sync(s, &prompt, err, errlen) != 0) {
            p36_tokens_free(&prompt);
            return 1;
        }
        p36_tokens_free(&prompt);
    } else {
        p36_session_reset_runtime(s);
    }
    s->checkpoint_valid = true;
    return 0;
}

void p36_session_snapshot_free(p36_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

int p36_qwen35_n_layer(void) {
    return P36_N_LAYER;
}

int p36_qwen35_n_embd(void) {
    return P36_N_EMBD;
}

int p36_qwen35_n_vocab(void) {
    return P36_N_VOCAB;
}

int p36_engine_prefill_layers(p36_engine *e,
                              const p36_tokens *prompt,
                              int ctx_size,
                              float *layer_out,
                              float *logits,
                              char *err,
                              size_t errlen) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    (void)layer_out;
    (void)logits;
    return p36_unsupported_runtime(err, errlen);
}

int p36_compare_prefill_layers(p36_engine *p36_e,
                               p36_engine *other_e,
                               const p36_tokens *prompt,
                               int ctx_size,
                               float *p36_layer_out,
                               float *other_layer_out,
                               float *p36_logits,
                               float *other_logits,
                               char *err,
                               size_t errlen) {
    (void)p36_e;
    (void)other_e;
    (void)prompt;
    (void)ctx_size;
    (void)p36_layer_out;
    (void)other_layer_out;
    (void)p36_logits;
    (void)other_logits;
    return p36_unsupported_runtime(err, errlen);
}
