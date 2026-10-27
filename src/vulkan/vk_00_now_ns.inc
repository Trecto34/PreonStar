#include "p36_gpu.h"
#include "p36_quant.h"
#include "p36_iq2_tables_vulkan.inc"
#include "p36_ssd.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vulkan/vulkan.h>

enum {
    P36_VK_QK8_0 = 32,
    P36_VK_TENSOR_F32     = 0,
    P36_VK_TENSOR_F16     = 1,
    P36_VK_TENSOR_Q2_K    = 10,
    P36_VK_TENSOR_Q3_K    = 11,
    P36_VK_TENSOR_Q4_K    = 12,
    P36_VK_TENSOR_Q5_K    = 13,
    P36_VK_TENSOR_Q6_K    = 14,
    P36_VK_TENSOR_IQ2_XXS = 16,
    P36_VK_TENSOR_IQ2_XS  = 17,
    P36_VK_TENSOR_IQ3_XXS = 18,
    P36_VK_TENSOR_IQ1_S   = 19,
    P36_VK_TENSOR_IQ4_NL  = 20,
    P36_VK_TENSOR_IQ3_S   = 21,
    P36_VK_TENSOR_IQ2_S   = 22,
    P36_VK_TENSOR_IQ4_XS  = 23,
    P36_VK_TENSOR_IQ1_M   = 29,
    P36_VK_TENSOR_Q2_0    = 42,
    P36_VK_TENSOR_PQ2_0   = 142,
    P36_VK_TENSOR_PTQ1_0  = 143,
};

typedef struct p36_gpu_tensor {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    uint64_t bytes;
    uint64_t alloc_bytes;
    uint64_t private_charge;
    uint64_t offset;
    struct p36_gpu_tensor *owner;
    bool device_local;
    bool private_scratch;
    bool weight_only;           /* arena-backed weights: uploaded once, never
                                 * written by dispatches, skipped by the
                                 * batch hazard tracker */
    bool gpu_written;           /* some recorded command wrote this root */
    uint64_t last_use_seq;      /* submit seq of the last recorded GPU use
                                 * of this root; 0 = never referenced */
    uint64_t write_gen;         /* bumped per recorded write dispatch, so
                                 * derived caches can tell reuses of the
                                 * same buffer apart */
    struct p36_gpu_tensor *recycled_next; /* bounded dense-weight free list */
} p36_gpu_tensor;

typedef struct p36_vk_kernel {
    const char *path;
    uint32_t bindings;
    uint32_t push_bytes;
    uint32_t write_mask;        /* bit per binding the shader writes */
    VkShaderModule shader;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    bool ready;
} p36_vk_kernel;

#define P36_VK_KERNEL(path_, bindings_, push_bytes_, write_mask_) \
    ((p36_vk_kernel){ \
        .path = (path_), \
        .bindings = (bindings_), \
        .push_bytes = (push_bytes_), \
        .write_mask = (write_mask_), \
    })

typedef struct p36_vk_weight {
    const void *source;
    uint64_t bytes;
    p36_gpu_tensor *tensor;
    bool pinned;             /* retained across decode passes in bounded mode */
    struct p36_vk_weight *next;
} p36_vk_weight;

static p36_vk_kernel p36_gdn_front;

typedef struct p36_vk_packed_weight {
    const void *source;
    uint64_t src_bytes;
    uint32_t type;
    uint32_t in_dim;
    uint32_t out_dim;
    p36_gpu_tensor *tensor;
    struct p36_vk_packed_weight *next;
} p36_vk_packed_weight;

/* Weight arena: cached weights suballocate from large blocks instead of
 * one VkDeviceMemory each.  Every device memory is an amdgpu buffer object
 * the kernel validates on EVERY queue submit, and with ~700 per-weight
 * allocations that validation cost ~2 ms per submit -- the single largest
 * share of a forward pass on this stack.  A handful of arena blocks makes
 * submits cheap again.  Weights live until the model map changes, so a bump
 * allocator with whole-arena teardown is enough. */
typedef struct p36_vk_arena_block {
    p36_gpu_tensor *base;
    uint64_t used;
    struct p36_vk_arena_block *next;
} p36_vk_arena_block;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkQueryPool query_pool;
    VkDescriptorPool descriptor_pool;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    uint32_t queue_count;
    uint32_t timestamp_valid_bits;
    uint32_t api_version;
    bool ready;
    bool prof_ops;
    bool prof_kernel;
    bool have_f16;
    bool have_storage8;
    bool have_storage16;
    bool have_int_dot;
    bool subgroup_size_control;
    bool subgroup_arithmetic;
    bool subgroup_clustered;
    bool bc250;
    bool ace_queue;
    uint32_t subgroup_size;
    char shader_root[PATH_MAX];

    p36_vk_kernel matmul_f16;
    p36_vk_kernel matmul_bf16;
    p36_vk_kernel vision_matmul_f16;
    p36_vk_kernel vision_attention;
    p36_vk_kernel matmul_f32;
    p36_vk_kernel matmul_f32_fast;
    p36_vk_kernel matmul_f32_fast_w256;
    p36_vk_kernel add;
    p36_vk_kernel directional_steering;
    p36_vk_kernel rms_norm;
    p36_vk_kernel add_rms_norm;
    p36_vk_kernel add_rms_norm_q8_k;
    p36_vk_kernel swiglu;
    p36_vk_kernel swiglu_q8_k;
    p36_vk_kernel rope;
    p36_vk_kernel rope_mrope;
    p36_vk_kernel rms_norm_rope;
    p36_vk_kernel rms_norm_rope_kv;
    p36_vk_kernel copy_rows;
    p36_vk_kernel embed_row;
    p36_vk_kernel recur_window;
    p36_vk_kernel conv_silu;
    p36_vk_kernel recur_conv_silu_decode;
    p36_vk_kernel recur_norm_gate;
    p36_vk_kernel recur_norm_gate_q8_k;
    p36_vk_kernel delta_qk;
    p36_vk_kernel delta_qkv;
    p36_vk_kernel delta_gates;
    p36_vk_kernel quantize_q8_0;
    p36_vk_kernel quantize_q8_k;
    p36_vk_kernel matmul_q8_0;
    p36_vk_kernel matmul_q8_0_p36;
    p36_vk_kernel matmul_q8_0_f32b;
    p36_vk_kernel matmul_q8_0_f32b_pair;
    p36_vk_kernel shared_gate_up_decode;
    p36_vk_kernel shared_down_tail_decode;
    p36_vk_kernel matmul_q8_0_mm;
    p36_vk_kernel matmul_q8_0_mm_f16;
    p36_vk_kernel matmul_q8_0_mm_f16_out32;
    p36_vk_kernel matmul_q8_0_f32b_nx;
    p36_vk_kernel matmul_q8_0_decode;
    p36_vk_kernel matmul_q8_0_decode_p36;
    p36_vk_kernel matmul_q8_0_decode_b64;
    p36_vk_kernel matmul_kquant;
    p36_vk_kernel matmul_q5k_mmq;
    p36_vk_kernel matmul_q6k_mmq;
    p36_vk_kernel matmul_q5k_mmq_fast;
    p36_vk_kernel matmul_q6k_mmq_fast;
    p36_vk_kernel delta_net;
    p36_vk_kernel delta_net_fast;
    p36_vk_kernel delta_net_cols;
    p36_vk_kernel delta_net_cols_f16;
    p36_vk_kernel delta_net_decode;
    p36_vk_kernel delta_net_decode_reg;
    p36_vk_kernel delta_net_decode_reg_f16;
    p36_vk_kernel attn_scores;
    p36_vk_kernel attn_post;
    p36_vk_kernel attn_reduce;
    p36_vk_kernel attn_decode_fused;
    p36_vk_kernel attn_decode_split;
    p36_vk_kernel attn_prefill_qtile;
    p36_vk_kernel attn_prefill_qtile2;
    p36_vk_kernel attn_prefill_qtile2_gqa6;
    p36_vk_kernel attn_prefill_fa_gqa6;
    p36_vk_kernel attn_prefill_fa_gqa8;
    p36_vk_kernel attn_combine;
    p36_vk_kernel moe_gate_up;
    p36_vk_kernel moe_gate_up_iq2s;
    p36_vk_kernel router_topk;
    p36_vk_kernel moe_tiles;
    p36_vk_kernel kv_store;
    p36_vk_kernel kv_store_quant;
    p36_vk_kernel rms_norm_rope_kv_quant;
    p36_vk_kernel top2;
    p36_vk_kernel argmax_rows;
    p36_vk_kernel topk8;
    p36_vk_kernel logits_minp;
    p36_vk_kernel moe_gate_up_f32b;
    p36_vk_kernel moe_gate_up_f32b_iq2s;
    p36_vk_kernel moe_gate_up_decode;
    p36_vk_kernel moe_gate_up_decode_iq2s;
    p36_vk_kernel moe_gate_up_decode_iq3xxs;
    p36_vk_kernel moe_gate_up_decode_q3k;
    p36_vk_kernel moe_gate_up_decode_iq2s_iq2xxs;
    p36_vk_kernel moe_down_q2k_f32b;
    p36_vk_kernel moe_down_q2k_f32b_iq2s;
    p36_vk_kernel moe_down_q2k_sum_decode;
    p36_vk_kernel moe_down_q2k_sum_decode_iq2s;
    p36_vk_kernel moe_down_q2k_sum_decode_iq3s;
    p36_vk_kernel moe_down_q2k_sum_decode_iq3xxs;
    p36_vk_kernel moe_down_q2k_sum_decode_q3k;
    p36_vk_kernel moe_gate_up_q4k_f32b;
    p36_vk_kernel moe_down_q4k_sum_decode;
    p36_vk_kernel moe_gate_up_gemm;
    p36_vk_kernel moe_gate_up_gemm_iq2s;
    p36_vk_kernel moe_gate_up_gemm_iq3xxs;
    p36_vk_kernel moe_gate_up_gemm_q3k;
    p36_vk_kernel moe_gate_up_gemm_iq2s_iq2xxs;
    p36_vk_kernel moe_down_gemm;
    p36_vk_kernel moe_down_gemm_iq2s;
    p36_vk_kernel moe_down_gemm_iq3s;
    p36_vk_kernel moe_down_gemm_iq3xxs;
    p36_vk_kernel moe_down_gemm_q3k;
    p36_vk_kernel moe_matvec;
    p36_vk_kernel moe_matvec_fast;
    p36_vk_kernel dense_iq3_xxs_decode;
    p36_vk_kernel dense_iq3_xxs_decode_r4;
    p36_vk_kernel dense_iq3_xxs_decode_nx;
    p36_vk_kernel dense_iq3_xxs_mmq;
    p36_vk_kernel dense_iq3_xxs_mmq_pair;
    p36_vk_kernel dense_iq3_s_decode;
    p36_vk_kernel dense_iq3_s_decode_full;
    p36_vk_kernel dense_iq3_s_decode_r4;
    p36_vk_kernel dense_iq3_s_decode_r1;
    p36_vk_kernel dense_iq3_s_mmq;
    p36_vk_kernel dense_iq3_s_bm64_mmq;
    p36_vk_kernel dense_iq3_s_mmq_r4;
    p36_vk_kernel dense_iq4_xs;
    p36_vk_kernel dense_iq4_xs_decode;
    p36_vk_kernel dense_iq4_xs_rows;
    p36_vk_kernel dense_iq4_xs_mmq;
    p36_vk_kernel dense_iq1_m;
    p36_vk_kernel dense_extra_decode;
    p36_vk_kernel dense_extra_decode_q2_0;
    p36_vk_kernel dense_extra_decode_pq2_0;
    p36_vk_kernel dense_extra_small_q2_0;
    p36_vk_kernel dense_extra_small_pq2_0;
    p36_vk_kernel dense_extra_decode_ptq1_0;
    p36_vk_kernel dense_extra_decode_ptq1_0_nx;
    p36_vk_kernel dense_extra_decode_iq2s_pair;
    p36_vk_kernel dense_extra_decode_iq2s;
    p36_vk_kernel dense_extra_decode_iq2xxs;
    p36_vk_kernel dense_extra_decode_iq2xs;
    p36_vk_kernel dense_extra_mmq;
    p36_vk_kernel dense_extra_mmq_q2_0;
    p36_vk_kernel dense_extra_mmq_pq2_0;
    p36_vk_kernel dense_extra_mmq_ptq1_0;
    p36_vk_kernel dense_extra_mmq_iq2s;
    p36_vk_kernel dense_kquant_mmq;
    p36_vk_kernel predequant_b16;
    p36_vk_kernel dense_kquant_decode;
    p36_vk_kernel dense_q4k_decode;
    p36_vk_kernel dense_q5k_decode;
    p36_vk_kernel dense_q3k_decode;
    p36_vk_kernel dense_q2k_decode;
    p36_vk_kernel dense_q3k_rows;
    p36_vk_kernel dense_q4k_rows;
    p36_vk_kernel dense_q5k_rows;
    p36_vk_kernel dense_q6k_rows;
    p36_vk_kernel moe_reduce;
    p36_vk_kernel ffn_tail;
    p36_vk_kernel fwht;
    p36_vk_kernel signs_mul;
    p36_vk_kernel v_grouped_permute;
    p36_vk_kernel hadamard_prepare;
    p36_vk_weight *weights;
    p36_vk_packed_weight *packed_weights;
    p36_vk_arena_block *arena;
    p36_vk_arena_block *arena_plan;
    uint64_t *arena_plan_sizes;
    uint32_t arena_plan_count;
    uint32_t arena_plan_next;
    p36_gpu_tensor *iq_tables;
    p36_gpu_tensor *attn_part;
    uint64_t attn_part_bytes;
    p36_gpu_tensor *top2_part;
    uint64_t top2_part_bytes;
    p36_gpu_tensor *argmax_rows_part;
    uint64_t argmax_rows_part_bytes;
    p36_gpu_tensor *topk8_part;
    uint64_t topk8_part_bytes;
} p36_vulkan_runtime;

enum {
    P36_VK_QK_K = 256,
    P36_VK_MATMUL_TOK_TILE = 4,
    P36_VK_MOE_PAIR_TILE = 8,
    P36_VK_MOE_TILE_WORDS = 1 + P36_VK_MOE_PAIR_TILE,
    P36_VK_MOE_GEMM_TILE = 32,
    P36_VK_N_HEAD = 16,
    P36_VK_N_HEAD_DIM = 256,
    P36_VK_N_SSM_CONV = 4,
    P36_VK_N_SSM_STATE = 128,
    P36_VK_N_SSM_GROUP = 16,
    P36_VK_N_SSM_QK = P36_VK_N_SSM_GROUP * P36_VK_N_SSM_STATE,
    P36_VK_N_SSM_CONV_DIM = P36_VK_N_SSM_QK * 2u + 4096u,
};

typedef struct {
    uint16_t d;
    int8_t qs[P36_VK_QK8_0];
} p36_vk_block_q8_0;

typedef char p36_vk_block_q8_0_size[(sizeof(p36_vk_block_q8_0) == 34) ? 1 : -1];

typedef struct {
    float d;
    float dmin;
    int8_t qs[P36_VK_QK_K];
    int16_t bsums[P36_VK_QK_K / 16];
} p36_vk_block_q8_K;

typedef char p36_vk_block_q8_k_size[(sizeof(p36_vk_block_q8_K) == 296) ? 1 : -1];

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[P36_VK_QK_K / 8];
    uint8_t qs[P36_VK_QK_K / 2];
} p36_vk_block_q5_K_src;

typedef struct {
    uint8_t ql[P36_VK_QK_K / 2];
    uint8_t qh[P36_VK_QK_K / 4];
    int8_t scales[P36_VK_QK_K / 16];
    uint16_t d;
} p36_vk_block_q6_K_src;

typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[64];
    uint8_t mins[64];
    uint8_t qh[(P36_VK_QK_K / 8) * 8];
    uint8_t qs[(P36_VK_QK_K / 2) * 8];
} p36_vk_block_q5_Kx8;

typedef char p36_vk_block_q5_kx8_size[(sizeof(p36_vk_block_q5_Kx8) == 1440) ? 1 : -1];

typedef struct {
    uint16_t d[8];
    int8_t scales[(P36_VK_QK_K / 16) * 8];
    uint8_t ql[(P36_VK_QK_K / 2) * 8];
    uint8_t qh[(P36_VK_QK_K / 4) * 8];
} p36_vk_block_q6_Kx8;

typedef char p36_vk_block_q6_kx8_size[(sizeof(p36_vk_block_q6_Kx8) == 1680) ? 1 : -1];

static p36_vulkan_runtime p36_vk;
static pthread_mutex_t p36_vk_mu = PTHREAD_MUTEX_INITIALIZER;
static const void *p36_gpu_model_map;
static uint64_t p36_gpu_model_size;
static int p36_gpu_model_fd = -1;
static bool p36_gpu_quality;
static bool p36_gpu_dense_model;
static bool p36_gpu_commands_open;
static uint64_t p36_gpu_live_bytes;
static uint64_t p36_gpu_peak_bytes;
static p36_gpu_tensor *p36_vk_dense_recycled;

/* Routed-expert banks are cached on device like any other weight, which
 * removes the per-token expert staging memcpy.  If the heaps cannot hold
 * them the runtime falls back to per-call packing (same numerics, slower)
 * and never retries, so a full memory cannot turn into a per-call
 * vkAllocateMemory failure loop. */
static bool p36_vk_moe_bank_cache = true;

enum {
    P36_VK_STREAM_MAX_LAYERS = 40,
    P36_VK_STREAM_MAX_EXPERTS = 256,
};

typedef struct {
    bool valid;
    uint32_t layer;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t last_used;
    bool loading;               /* reserved this prepare pass, data in flight */
} p36_vk_stream_entry;

typedef struct {
    bool enabled;
    uint32_t budget;
    uint32_t full_layers;
    uint32_t runtime_cap;
    uint32_t allocation_failures;
    bool indexed;
    bool has_route_bias;
    uint64_t configured_expert_bytes;
    uint32_t cap;
    uint32_t phase;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t loads;
    uint64_t evictions;
    uint64_t lookup_steps;
    uint64_t gate_expert_bytes;
    uint64_t up_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    bool has_gate_scales;
    bool has_up_scales;
    bool has_down_scales;
    p36_gpu_tensor *gate;
    p36_gpu_tensor *up;
    p36_gpu_tensor *down;
    p36_gpu_tensor *gate_scales;
    p36_gpu_tensor *up_scales;
    p36_gpu_tensor *down_scales;
    p36_vk_stream_entry *entry;
    uint32_t slot_of[P36_VK_STREAM_MAX_LAYERS][P36_VK_STREAM_MAX_EXPERTS];
    uint32_t route_hotness[P36_VK_STREAM_MAX_LAYERS][P36_VK_STREAM_MAX_EXPERTS];
    uint32_t route_bias[P36_VK_STREAM_MAX_LAYERS][P36_VK_STREAM_MAX_EXPERTS];
    uint64_t route_tokens;
    uint64_t hotness_decay_token;
} p36_vk_stream_cache;

static p36_vk_stream_cache p36_vk_stream;

/* Decode allocates the same output-only buffers every layer. Reusing them
 * avoids a vkAllocateMemory/vkFreeMemory round-trip without CPU mapping. */
enum {
    P36_VK_POOL_CAP = 256,
    P36_VK_POOL_MAX_BYTES = 1u << 20,
    P36_VK_PRIVATE_POOL_CAP = 64,
    P36_VK_PRIVATE_LIMIT = 512u << 20,
    P36_VK_MAX_THREADS = 32,
};
static p36_gpu_tensor *p36_vk_pool[P36_VK_POOL_CAP];
static p36_gpu_tensor *p36_vk_vision_weight;
static uint32_t p36_vk_pool_n;
static p36_gpu_tensor *p36_vk_private_pool[P36_VK_PRIVATE_POOL_CAP];
static uint32_t p36_vk_private_pool_n;
static uint64_t p36_vk_private_bytes;
static uint64_t p36_vk_private_peak;

typedef struct {
    pthread_mutex_t mu;
    pthread_mutex_t submit_mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    pthread_t threads[P36_VK_MAX_THREADS];
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    p36_gpu_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} p36_vk_host_pool;

static p36_vk_host_pool p36_vk_host;
static pthread_mutex_t p36_vk_host_init_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread int p36_vk_parallel_depth;

/* Lazy submission, the scheme ds4's Metal runtime uses: dispatches are
 * recorded into one open command buffer with full compute barriers between
 * them, and the batch is submitted and fence-waited only when the host
 * actually reads a tensor.  Every host-side op fetches its pointers through
 * p36_gpu_tensor_contents(), so the flush point is automatic and the
 * host-exact parity ops keep working unchanged.  Tensors freed while a
 * batch is open are parked on the retire list until the fence signals,
 * since recorded commands still reference their memory. */
enum {
    P36_VK_BATCH_FLUSH_AT = 32,
    P36_VK_RETIRE_CAP = 1024,
};
static bool p36_vk_batch_recording;
static uint32_t p36_vk_batch_count;
static p36_gpu_tensor *p36_vk_retired[P36_VK_RETIRE_CAP];
static uint32_t p36_vk_retired_n;

/* Submission sequence: every vkQueueSubmit gets the next seq, roots are
 * stamped with the seq of the batch that last references them, and a
 * fence wait advances completed_seq (one queue, so fences signal in
 * submission order).  A root whose last_use_seq is <= completed_seq has
 * no pending GPU references and can retire without draining anything. */
static uint64_t p36_vk_submit_seq;
static uint64_t p36_vk_completed_seq;

/* P36_VK_PROF=1 wall-clock accounting: where a forward pass actually waits.
 * submit_ns covers vkQueueSubmit to fence signal (GPU batch execution as
 * seen by the host), record_ns the per-dispatch descriptor + command
 * recording.  Printed by p36_gpu_print_memory_report(). */
static uint64_t p36_vk_prof_dispatches;
static uint64_t p36_vk_prof_flushes;
/* Diagnostic counters behind p36_gpu_read_bytes/read_flushes (P36_MTP_TIMING):
 * how much the host pulls back and how many of those pulls drain the GPU. */
static uint64_t p36_vk_read_bytes_total;
static uint64_t p36_vk_flush_total;
static uint64_t p36_vk_prof_submit_ns;
/* P36_VK_PROF record-path split: descriptor plumbing, command recording and
 * per-dispatch bookkeeping, so a CPU-bound harness shows which one to fix. */
static uint64_t p36_vk_prof_desc_ns, p36_vk_prof_cmd_ns, p36_vk_prof_book_ns;
static uint64_t p36_vk_prof_record_ns;

enum {
    /* 32k timed dispatches: enough to cover a decode window, not just prefill. */
    P36_VK_PROF_QUERY_CAP = 65536,
    P36_VK_PROF_KERNEL_CAP = 64,
    P36_VK_PROF_OP_CAP = 128,
    P36_VK_PROF_SHAPE_CAP = 32,
};

typedef struct {
    const char *path;
    uint64_t dispatches;
    uint64_t groups;
    uint64_t gpu_ns;
} p36_vk_prof_kernel_row;

typedef struct {
    const char *name;
    uint64_t dispatches;
    uint64_t flushes;
    uint64_t barriers;
    uint64_t groups;
    uint64_t record_ns;
    uint64_t submit_ns;
    uint64_t gpu_ns;
    uint64_t wall_ns;
    /* Per-dispatch workgroup distribution (P36_VK_PROF_WG_HIST=1), bucketed
     * <5, 5-20, 20-40, >40 WGs.  Host-side counters only: no GPU cost, so it
     * can stay on during timed runs. */
    uint32_t wg_hist[4];
} p36_vk_prof_op_row;

typedef struct {
    const char *name;
    uint64_t t0;
    bool active;
} p36_vk_prof_scope;

static const char *p36_vk_prof_query_path[P36_VK_PROF_QUERY_CAP / 2];
static const char *p36_vk_prof_query_op[P36_VK_PROF_QUERY_CAP / 2];
static uint64_t p36_vk_prof_query_groups[P36_VK_PROF_QUERY_CAP / 2];
static uint64_t p36_vk_prof_query_host_ns[P36_VK_PROF_QUERY_CAP / 2];
static uint32_t p36_vk_prof_query_count;
/* Open op-group bracket: one timestamp pair per run of consecutive dispatches
 * of the same kernel.  Bracket indices, not dispatch indices. */
static uint32_t p36_vk_prof_grp_base = UINT32_MAX;
static const char *p36_vk_prof_grp_path = NULL;
/* Optional per-dispatch GPU timeline (P36_VK_PROF_TIMELINE=<path>).  Bounded
 * diagnostic for separating host record cadence, GPU work and dependency
 * gaps; the aggregate profile cannot show overlap or idle time. */
static FILE *p36_vk_prof_timeline_fp;
static uint64_t p36_vk_prof_timeline_seq;
static bool p36_vk_prof_timeline_tried;
static p36_vk_prof_kernel_row p36_vk_prof_kernel_rows[P36_VK_PROF_KERNEL_CAP];
static uint32_t p36_vk_prof_kernel_n;
/* P36_VK_PROF_SKIP dispatches go untimed so prefill can be excluded. */
static uint64_t p36_vk_prof_skip_left;
static p36_vk_prof_op_row p36_vk_prof_op_rows[P36_VK_PROF_OP_CAP];
static uint32_t p36_vk_prof_op_n;

typedef struct {
    uint32_t type;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    char name[80];
} p36_vk_prof_shape;

static p36_vk_prof_shape p36_vk_prof_shapes[P36_VK_PROF_SHAPE_CAP];
static uint32_t p36_vk_prof_shape_n;

enum {
    P36_VK_STREAM_PROFILE_LAYERS = 40,
    P36_VK_STREAM_PROFILE_EXPERTS = 256,
};

typedef struct {
    bool active;
    bool reported;
    const char *path;
    uint64_t hist[P36_VK_STREAM_PROFILE_LAYERS][P36_VK_STREAM_PROFILE_EXPERTS];
    uint64_t total;
} p36_vk_stream_profile;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t count;
} p36_vk_stream_profile_entry;

static p36_vk_stream_profile p36_vk_stream_prof;

static uint64_t p36_vk_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Diagnostic probe: dropping the dependency barriers measures what the token
 * would cost if the GPU could overlap independent work.  Numerics are wrong by
 * design; never enable this outside a timing experiment. */
static bool p36_vk_skip_barriers(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("P36_VK_SKIP_BARRIERS") != NULL;
    return cached == 1;
}

/* The fused expert pair at n_tok 1 skips the q8_K staging and merges gate+up
 * plus the down sum: +30-40% decode on the BC-250, validated under agentic
 * tool-calling load.  Default on.  It moves tokens against the q8 route on
 * some prompts, so P36_VK_MOE_PAIR_DECODE=0 restores the bit-exact q8 path. */
static bool p36_vk_moe_pair_decode(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("P36_VK_MOE_PAIR_DECODE");
        cached = !(env && (env[0] == '0' || !strcmp(env, "false") || !strcmp(env, "off")));
    }
    return cached == 1;
}

static void p36_vk_prof_report(void);
static void p36_vk_stream_profile_report(void);

