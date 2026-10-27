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
#define P36_Q4_1_BYTES 20u
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
    P36_TENSOR_Q4_1 = 3,
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
    bool from_q4_1;          /* appended MTP block only: lossy Q4_1 -> Q8_0 for the draft head */
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
    /* Prompt-lookup speculation: k == 0 disables it. */
    int lookup_order;
    int lookup_k;
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
    uint8_t gpu_top2_slot;   /* which rt->top2 buffer the result landed in */
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
    /* Raw residual stream of the last decoded row (before the final output
     * norm): the MTP head is trained on this, not on the normalised hidden. */
    p36_gpu_tensor *last_h_raw;
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
    /* Two top-2 buffers: the reader waits on one while the next token's
     * dispatch writes the other, so a deferred read never races the GPU. */
    p36_gpu_tensor *top2[2];
    uint32_t top2_flip;
    p36_gpu_tensor *topk8;
    p36_gpu_tensor *minp_pack;
    p36_gpu_tensor *scores;
    p36_vulkan_full_attn_cache mtp_full;
    p36_vulkan_recurrent_cache spec_recurrent[P36_MAX_LAYER];
    p36_gpu_tensor *spec_logits;
    /* Top-2 of every speculative logits row, so verification reads n_draft
     * ints instead of n_draft full vocab rows. */
    p36_gpu_tensor *spec_tops;
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

