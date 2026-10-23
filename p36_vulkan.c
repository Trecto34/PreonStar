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
    p36_vk_kernel dense_iq4_xs_mmq;
    p36_vk_kernel dense_iq1_m;
    p36_vk_kernel dense_extra_decode;
    p36_vk_kernel dense_extra_decode_q2_0;
    p36_vk_kernel dense_extra_decode_pq2_0;
    p36_vk_kernel dense_extra_small_q2_0;
    p36_vk_kernel dense_extra_small_pq2_0;
    p36_vk_kernel dense_extra_decode_ptq1_0;
    p36_vk_kernel dense_extra_decode_iq2s_pair;
    p36_vk_kernel dense_extra_decode_iq2s;
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
static uint64_t p36_vk_prof_record_ns;

enum {
    P36_VK_PROF_QUERY_CAP = 2048,
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
    uint64_t groups;
    uint64_t record_ns;
    uint64_t submit_ns;
    uint64_t gpu_ns;
    uint64_t wall_ns;
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
/* Optional per-dispatch GPU timeline (P36_VK_PROF_TIMELINE=<path>).  Bounded
 * diagnostic for separating host record cadence, GPU work and dependency
 * gaps; the aggregate profile cannot show overlap or idle time. */
static FILE *p36_vk_prof_timeline_fp;
static uint64_t p36_vk_prof_timeline_seq;
static bool p36_vk_prof_timeline_tried;
static p36_vk_prof_kernel_row p36_vk_prof_kernel_rows[P36_VK_PROF_KERNEL_CAP];
static uint32_t p36_vk_prof_kernel_n;
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

static void p36_vk_prof_report(void);
static void p36_vk_stream_profile_report(void);

static void p36_vk_prof_reset(void) {
    p36_vk_prof_dispatches = 0;
    p36_vk_prof_flushes = 0;
    p36_vk_prof_submit_ns = 0;
    p36_vk_prof_record_ns = 0;
    p36_vk_prof_query_count = 0;
    p36_vk_prof_kernel_n = 0;
    p36_vk_prof_op_n = 0;
}

static int p36_vk_stream_profile_cmp(const void *a, const void *b) {
    const p36_vk_stream_profile_entry *ea = a;
    const p36_vk_stream_profile_entry *eb = b;
    if (ea->count < eb->count) return 1;
    if (ea->count > eb->count) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

static void p36_vk_stream_profile_init(void) {
    const char *path = getenv("P36_VK_STREAMING_EXPERT_PROFILE");
    const char *hotlist = getenv("P36_VK_STREAMING_EXPERT_HOTLIST_OUT");
    if (hotlist && hotlist[0]) path = hotlist;
    if (!path || !path[0] || p36_vk_stream_prof.active) return;
    memset(&p36_vk_stream_prof, 0, sizeof(p36_vk_stream_prof));
    p36_vk_stream_prof.active = true;
    p36_vk_stream_prof.path = path;
    atexit(p36_vk_stream_profile_report);
    fprintf(stderr, "p36: Vulkan streaming expert profiler active: %s\n", path);
}

static void p36_vk_stream_profile_record_unlocked(const p36_gpu_stream_expert_table *t,
                                                  const uint32_t *selected,
                                                  uint64_t n_selected) {
    if (!p36_vk_stream_prof.active || !t || !selected) return;
    if (t->layer >= P36_VK_STREAM_PROFILE_LAYERS) return;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= P36_VK_STREAM_PROFILE_EXPERTS) continue;
        p36_vk_stream_prof.hist[t->layer][expert]++;
        p36_vk_stream_prof.total++;
    }
}

static void p36_vk_stream_profile_report(void) {
    p36_vk_stream_profile *p = &p36_vk_stream_prof;
    p36_vk_stream_profile_entry entries[P36_VK_STREAM_PROFILE_LAYERS *
                                        P36_VK_STREAM_PROFILE_EXPERTS];
    FILE *fp;
    uint32_t n = 0;
    if (!p->active || p->reported || !p->path || !p->path[0]) return;
    p->reported = true;
    for (uint32_t il = 0; il < P36_VK_STREAM_PROFILE_LAYERS; il++) {
        for (uint32_t expert = 0; expert < P36_VK_STREAM_PROFILE_EXPERTS; expert++) {
            entries[n++] = (p36_vk_stream_profile_entry) {
                .layer = il,
                .expert = expert,
                .count = p->hist[il][expert],
            };
        }
    }
    qsort(entries, n, sizeof(entries[0]), p36_vk_stream_profile_cmp);
    fp = fopen(p->path, "wb");
    if (!fp) {
        fprintf(stderr, "p36: failed to write streaming expert profile %s: %s\n",
                p->path, strerror(errno));
        return;
    }
    fprintf(fp,
            "# p36 expert hotlist v1\n"
            "# layers %u\n"
            "# experts %u\n"
            "# selections %" PRIu64 "\n"
            "# columns: layer expert hits weight\n",
            P36_VK_STREAM_PROFILE_LAYERS,
            P36_VK_STREAM_PROFILE_EXPERTS,
            p->total);
    for (uint32_t i = 0; i < n; i++) {
        fprintf(fp, "%u %u %" PRIu64 " 0\n",
                entries[i].layer,
                entries[i].expert,
                entries[i].count);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "p36: failed to close streaming expert profile %s: %s\n",
                p->path, strerror(errno));
    } else {
        fprintf(stderr, "p36: wrote Vulkan streaming expert profile %s (%" PRIu64 " selections)\n",
                p->path, p->total);
    }
}

static const char *p36_vk_prof_op_name(const char *name) {
    return (name && name[0]) ? name : "other";
}

static const char *p36_vk_prof_iq3_shape(uint32_t type, uint32_t n_tok,
                                         uint32_t in_dim, uint32_t out_dim,
                                         const char *fallback) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("P36_VK_PROF_SHAPE") != NULL;
    if (!enabled) return fallback;
    for (uint32_t i = 0; i < p36_vk_prof_shape_n; i++) {
        p36_vk_prof_shape *s = &p36_vk_prof_shapes[i];
        if (s->type == type && s->n_tok == n_tok &&
            s->in_dim == in_dim && s->out_dim == out_dim) return s->name;
    }
    if (p36_vk_prof_shape_n == P36_VK_PROF_SHAPE_CAP) return fallback;
    p36_vk_prof_shape *s = &p36_vk_prof_shapes[p36_vk_prof_shape_n++];
    s->type = type;
    s->n_tok = n_tok;
    s->in_dim = in_dim;
    s->out_dim = out_dim;
    const char *kind = type == P36_VK_TENSOR_IQ3_S ? "iq3_s" :
                       type == P36_VK_TENSOR_IQ3_XXS ? "iq3_xxs" :
                       type == P36_VK_TENSOR_IQ2_XXS ? "iq2_xxs" :
                       type == P36_VK_TENSOR_IQ2_XS ? "iq2_xs" :
                       type == P36_VK_TENSOR_IQ2_S ? "iq2_s" :
                       type == P36_VK_TENSOR_IQ1_S ? "iq1_s" :
                       type == P36_VK_TENSOR_IQ4_NL ? "iq4_nl" :
                       type == P36_VK_TENSOR_IQ4_XS ? "iq4_xs" :
                       type == P36_VK_TENSOR_Q2_K ? "q2k" :
                       type == P36_VK_TENSOR_Q3_K ? "q3k" :
                       type == P36_VK_TENSOR_Q4_K ? "q4k" :
                       type == P36_VK_TENSOR_Q5_K ? "q5k" :
                       type == P36_VK_TENSOR_Q6_K ? "q6k" : "iq";
    snprintf(s->name, sizeof(s->name), "dense_%s_%s_%ux%u_n%u",
             kind,
             n_tok == 1 ? "decode" : "mmq", in_dim, out_dim, n_tok);
    return s->name;
}

static p36_vk_prof_op_row *p36_vk_prof_op_get(const char *name) {
    name = p36_vk_prof_op_name(name);
    for (uint32_t i = 0; i < p36_vk_prof_op_n; i++) {
        if (p36_vk_prof_op_rows[i].name == name || !strcmp(p36_vk_prof_op_rows[i].name, name)) {
            return &p36_vk_prof_op_rows[i];
        }
    }
    if (p36_vk_prof_op_n == P36_VK_PROF_OP_CAP) return NULL;
    memset(&p36_vk_prof_op_rows[p36_vk_prof_op_n], 0, sizeof(p36_vk_prof_op_rows[p36_vk_prof_op_n]));
    p36_vk_prof_op_rows[p36_vk_prof_op_n].name = name;
    return &p36_vk_prof_op_rows[p36_vk_prof_op_n++];
}

static void p36_vk_prof_op_add(const char *name,
                               uint64_t dispatches,
                               uint64_t flushes,
                               uint64_t groups,
                               uint64_t record_ns,
                               uint64_t submit_ns,
                               uint64_t gpu_ns,
                               uint64_t wall_ns) {
    if (!p36_vk.prof_ops) return;
    p36_vk_prof_op_row *r = p36_vk_prof_op_get(name);
    if (!r) return;
    r->dispatches += dispatches;
    r->flushes += flushes;
    r->groups += groups;
    r->record_ns += record_ns;
    r->submit_ns += submit_ns;
    r->gpu_ns += gpu_ns;
    r->wall_ns += wall_ns;
}

static void p36_vk_prof_scope_begin(p36_vk_prof_scope *s, const char *name) {
    if (!s) return;
    s->active = p36_vk.prof_ops;
    s->name = p36_vk_prof_op_name(name);
    s->t0 = s->active ? p36_vk_now_ns() : 0;
}

static void p36_vk_prof_scope_end(p36_vk_prof_scope *s) {
    if (!s || !s->active) return;
    p36_vk_prof_op_add(s->name, 0, 0, 0, 0, 0, 0, p36_vk_now_ns() - s->t0);
    s->active = false;
}

static void p36_vk_prof_kernel_add(const char *path, uint64_t groups, uint64_t gpu_ns) {
    for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) {
        if (p36_vk_prof_kernel_rows[i].path == path || !strcmp(p36_vk_prof_kernel_rows[i].path, path)) {
            p36_vk_prof_kernel_rows[i].dispatches++;
            p36_vk_prof_kernel_rows[i].groups += groups;
            p36_vk_prof_kernel_rows[i].gpu_ns += gpu_ns;
            return;
        }
    }
    if (p36_vk_prof_kernel_n == P36_VK_PROF_KERNEL_CAP) return;
    p36_vk_prof_kernel_rows[p36_vk_prof_kernel_n++] = (p36_vk_prof_kernel_row){
        .path = path,
        .dispatches = 1,
        .groups = groups,
        .gpu_ns = gpu_ns,
    };
}

static void p36_vk_prof_kernel_collect_unlocked(void) {
    if (!p36_vk.prof_kernel || !p36_vk.query_pool || p36_vk_prof_query_count == 0) return;

    uint64_t ticks[P36_VK_PROF_QUERY_CAP];
    VkResult rc = vkGetQueryPoolResults(p36_vk.device,
                                        p36_vk.query_pool,
                                        0,
                                        p36_vk_prof_query_count,
                                        sizeof(ticks[0]) * p36_vk_prof_query_count,
                                        ticks,
                                        sizeof(ticks[0]),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (rc != VK_SUCCESS) {
        p36_vk_prof_query_count = 0;
        return;
    }

    uint64_t mask = UINT64_MAX;
    if (p36_vk.timestamp_valid_bits > 0 && p36_vk.timestamp_valid_bits < 64) {
        mask = (1ull << p36_vk.timestamp_valid_bits) - 1ull;
    }
    if (!p36_vk_prof_timeline_tried) {
        p36_vk_prof_timeline_tried = true;
        const char *tp = getenv("P36_VK_PROF_TIMELINE");
        if (tp && tp[0]) {
            p36_vk_prof_timeline_fp = fopen(tp, "wb");
            if (p36_vk_prof_timeline_fp) {
                fprintf(p36_vk_prof_timeline_fp,
                        "# p36 vulkan dispatch timeline v1\n"
                        "# tick_period_ns=%.6f valid_bits=%u\n"
                        "# columns: seq flush path groups start_tick end_tick dur_ns host_ns\n",
                        (double)p36_vk.props.limits.timestampPeriod,
                        p36_vk.timestamp_valid_bits);
            }
        }
    }
    for (uint32_t q = 0; q + 1 < p36_vk_prof_query_count; q += 2) {
        uint64_t a = ticks[q] & mask;
        uint64_t b = ticks[q + 1u] & mask;
        uint64_t dt = (b >= a) ? (b - a) : ((mask - a) + b + 1u);
        uint64_t ns = (uint64_t)((double)dt * (double)p36_vk.props.limits.timestampPeriod);
        p36_vk_prof_kernel_add(p36_vk_prof_query_path[q / 2u], p36_vk_prof_query_groups[q / 2u], ns);
        p36_vk_prof_op_add(p36_vk_prof_query_op[q / 2u], 0, 0, 0, 0, 0, ns, 0);
        if (p36_vk_prof_timeline_fp) {
            fprintf(p36_vk_prof_timeline_fp,
                    "%" PRIu64 " %" PRIu64 " %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    p36_vk_prof_timeline_seq++,
                    p36_vk_prof_flushes,
                    p36_vk_prof_query_path[q / 2u],
                    p36_vk_prof_query_groups[q / 2u],
                    a,
                    b,
                    ns,
                    p36_vk_prof_query_host_ns[q / 2u]);
        }
    }
    if (p36_vk_prof_timeline_fp) fflush(p36_vk_prof_timeline_fp);
    p36_vk_prof_query_count = 0;
}

static void *p36_vk_host_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&p36_vk_host.mu);
        while (seen_generation == p36_vk_host.generation && !p36_vk_host.shutdown) {
            pthread_cond_wait(&p36_vk_host.work_cv, &p36_vk_host.mu);
        }
        if (p36_vk_host.shutdown) {
            pthread_mutex_unlock(&p36_vk_host.mu);
            return NULL;
        }

        seen_generation = p36_vk_host.generation;
        p36_gpu_parallel_fn fn = p36_vk_host.fn;
        void *ctx = p36_vk_host.ctx;
        uint64_t n_rows = p36_vk_host.n_rows;
        uint32_t n_threads = p36_vk_host.n_threads;
        pthread_mutex_unlock(&p36_vk_host.mu);

        {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                p36_vk_parallel_depth++;
                fn(ctx, row0, row1);
                p36_vk_parallel_depth--;
            }
        }

        pthread_mutex_lock(&p36_vk_host.mu);
        p36_vk_host.done++;
        if (p36_vk_host.done == p36_vk_host.n_workers) pthread_cond_signal(&p36_vk_host.done_cv);
        pthread_mutex_unlock(&p36_vk_host.mu);
    }
}

static void p36_vk_host_threads_init(void) {
    pthread_mutex_lock(&p36_vk_host_init_mu);
    if (p36_vk_host.initialized) {
        pthread_mutex_unlock(&p36_vk_host_init_mu);
        return;
    }

    uint32_t n_threads = 12;
    long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("P36_THREADS");
    if (online_cpus > 0) n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (n_threads > P36_VK_MAX_THREADS) n_threads = P36_VK_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&p36_vk_host.mu, NULL);
    pthread_mutex_init(&p36_vk_host.submit_mu, NULL);
    pthread_cond_init(&p36_vk_host.work_cv, NULL);
    pthread_cond_init(&p36_vk_host.done_cv, NULL);
    p36_vk_host.n_threads = 1;
    p36_vk_host.n_workers = 0;
    p36_vk_host.generation = 0;
    p36_vk_host.done = 0;
    p36_vk_host.shutdown = false;
    p36_vk_host.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&p36_vk_host.threads[i], NULL,
                           p36_vk_host_worker_main, (void *)(uintptr_t)i) != 0) {
            fprintf(stderr, "p36: Vulkan host pool started %u/%u threads\n", i, n_threads);
            break;
        }
        p36_vk_host.n_threads = i + 1u;
        p36_vk_host.n_workers = i;
    }
    pthread_mutex_unlock(&p36_vk_host_init_mu);
}

static void p36_vk_host_threads_shutdown(void) {
    pthread_mutex_lock(&p36_vk_host_init_mu);
    if (!p36_vk_host.initialized) {
        pthread_mutex_unlock(&p36_vk_host_init_mu);
        return;
    }

    pthread_mutex_lock(&p36_vk_host.submit_mu);
    pthread_mutex_lock(&p36_vk_host.mu);
    p36_vk_host.shutdown = true;
    p36_vk_host.generation++;
    pthread_cond_broadcast(&p36_vk_host.work_cv);
    pthread_mutex_unlock(&p36_vk_host.mu);
    for (uint32_t i = 1; i < p36_vk_host.n_threads; i++) pthread_join(p36_vk_host.threads[i], NULL);
    pthread_mutex_unlock(&p36_vk_host.submit_mu);

    pthread_cond_destroy(&p36_vk_host.done_cv);
    pthread_cond_destroy(&p36_vk_host.work_cv);
    pthread_mutex_destroy(&p36_vk_host.submit_mu);
    pthread_mutex_destroy(&p36_vk_host.mu);
    memset(&p36_vk_host, 0, sizeof(p36_vk_host));
    pthread_mutex_unlock(&p36_vk_host_init_mu);
}

void p36_gpu_parallel_for_rows(uint64_t n_rows,
                               uint64_t min_parallel_rows,
                               p36_gpu_parallel_fn fn,
                               void *ctx) {
    uint64_t rows_per_thread;
    uint64_t main_row1;

    if (!fn || n_rows == 0) return;
    p36_vk_host_threads_init();
    if (p36_vk_parallel_depth > 0 || p36_vk_host.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&p36_vk_host.submit_mu);
    pthread_mutex_lock(&p36_vk_host.mu);
    p36_vk_host.fn = fn;
    p36_vk_host.ctx = ctx;
    p36_vk_host.n_rows = n_rows;
    p36_vk_host.done = 0;
    p36_vk_host.generation++;
    pthread_cond_broadcast(&p36_vk_host.work_cv);
    rows_per_thread = (n_rows + p36_vk_host.n_threads - 1u) / p36_vk_host.n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&p36_vk_host.mu);

    if (main_row1 > 0) {
        p36_vk_parallel_depth++;
        fn(ctx, 0, main_row1);
        p36_vk_parallel_depth--;
    }

    pthread_mutex_lock(&p36_vk_host.mu);
    while (p36_vk_host.done < p36_vk_host.n_workers) {
        pthread_cond_wait(&p36_vk_host.done_cv, &p36_vk_host.mu);
    }
    pthread_mutex_unlock(&p36_vk_host.mu);
    pthread_mutex_unlock(&p36_vk_host.submit_mu);
}

/* Root tensors written by the open batch.  p36_gpu_tensor_contents() only
 * needs to submit when the host touches one of these (or the set
 * overflowed); reads of host-produced tensors (router selection, q8
 * staging, swiglu outputs) keep recording.  This is safe because the
 * schedulers never host-write a tensor that recorded-but-unsubmitted
 * commands read: activations flow forward, so a host write always precedes
 * the recording of its GPU consumers. */
/* A 512-token prefill chunk writes more distinct roots than this, so
 * batch_dirty_all trips once per chunk.  That used to drain the pipeline
 * on the next chunk's embed-stage map (~190ms/chunk); now contents()
 * consults the per-root gpu_written flag, so host-only tensors skip the
 * flush even under dirty_all, and scratch memory is bounded by the
 * seq-based retirement sweep on each eager submit rather than by the
 * accidental per-chunk drain. */
enum { P36_VK_WRITTEN_CAP = 64 };
static p36_gpu_tensor *p36_vk_batch_written[P36_VK_WRITTEN_CAP];
static uint32_t p36_vk_batch_written_n;
static bool p36_vk_batch_dirty_all;

/* Hazard sets since the last pipeline barrier in the open batch.  A barrier
 * is emitted only when a dispatch touches a tensor written since then, or
 * writes one that was read (RAW/WAW/WAR); independent dispatches overlap on
 * the GPU instead of draining the pipeline ~1300 times per decoded token.
 * Weight-arena roots never enter the sets: they are uploaded from the host
 * before use and no shader writes them. */
enum { P36_VK_HAZARD_CAP = 48 };
static p36_gpu_tensor *p36_vk_hazard_written[P36_VK_HAZARD_CAP];
static p36_gpu_tensor *p36_vk_hazard_read[P36_VK_HAZARD_CAP];
static uint32_t p36_vk_hazard_written_n;
static uint32_t p36_vk_hazard_read_n;
static bool p36_vk_hazard_all;

static bool p36_vk_hazard_in(p36_gpu_tensor **set, uint32_t n, const p36_gpu_tensor *root) {
    for (uint32_t i = 0; i < n; i++) {
        if (set[i] == root) return true;
    }
    return false;
}

static void p36_vk_hazard_add(p36_gpu_tensor **set, uint32_t *n, p36_gpu_tensor *root) {
    if (root->weight_only || p36_vk_hazard_in(set, *n, root)) return;
    if (*n == P36_VK_HAZARD_CAP) p36_vk_hazard_all = true;
    else set[(*n)++] = root;
}

static void p36_vk_hazard_reset(void) {
    p36_vk_hazard_written_n = 0;
    p36_vk_hazard_read_n = 0;
    p36_vk_hazard_all = false;
}

/* GPU work is outstanding while a batch records or an eager submit's fence
 * has not been waited on. Frees and host reads must respect both. */
static bool p36_vk_gpu_busy_unlocked(void);
static const char *p36_vk_flush_reason = "submit_wait_other";
static __thread const char *p36_vk_tensor_contents_reason = "submit_wait_tensor_contents";

static int p36_vk_flush_reason_unlocked(const char *reason);
static void *p36_vk_tensor_contents_labeled(p36_gpu_tensor *tensor, const char *reason);
static int p36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out);
static void p36_vk_note_written_tensor(p36_gpu_tensor *tensor);

static uint64_t p36_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static bool p36_u64_mul_ok(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool p36_gpu_tensor_range_ok(const p36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
static const unsigned char *p36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes);
static p36_gpu_tensor *p36_vk_quantize_q8_k_alloc_tensor(const p36_gpu_tensor *x,
                                                         uint64_t in_dim,
                                                         uint64_t n_tok);
static int p36_vk_run_unlocked(
        const char                 *op,
        p36_vk_kernel              *kernel,
        const p36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z);

typedef struct {
    float d;
    int8_t qs[P36_VK_QK_K];
    int16_t bsums[P36_VK_QK_K / 16];
} p36_vk_block_q8_K_packed;

enum { P36_MAX_VK_Q8_BLOCKS = 32 };

typedef struct {
    p36_vk_block_q8_K *dst;
    const float *x;
    uint64_t in_dim;
    uint64_t blocks;
} p36_vk_quantize_q8_k_ctx;

static void p36_vk_quantize_q8_k_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_quantize_q8_k_ctx *ctx = (p36_vk_quantize_q8_k_ctx *)opaque;
    p36_vk_block_q8_K_packed tmp[P36_MAX_VK_Q8_BLOCKS];

    for (uint64_t t = row0; t < row1; t++) {
        p36_vk_block_q8_K *qt = ctx->dst + t * ctx->blocks;
        p36_quant_q8_k(ctx->x + t * ctx->in_dim, tmp, (int64_t)ctx->in_dim);
        for (uint64_t b = 0; b < ctx->blocks; b++) {
            qt[b].d = tmp[b].d;
            qt[b].dmin = 0.0f;
            memcpy(qt[b].qs, tmp[b].qs, sizeof(qt[b].qs));
            memcpy(qt[b].bsums, tmp[b].bsums, sizeof(qt[b].bsums));
        }
    }
}

static void p36_vk_quantize_q8_k_host(p36_vk_block_q8_K *dst,
                                      const float *x,
                                      uint64_t n_tok,
                                      uint64_t in_dim) {
    const uint64_t blocks = in_dim / P36_VK_QK_K;
    p36_vk_quantize_q8_k_ctx ctx;

    if (blocks > P36_MAX_VK_Q8_BLOCKS) abort();
    ctx.dst = dst;
    ctx.x = x;
    ctx.in_dim = in_dim;
    ctx.blocks = blocks;
    if (n_tok * in_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 4, p36_vk_quantize_q8_k_rows, &ctx);
    else p36_vk_quantize_q8_k_rows(&ctx, 0, n_tok);
}

static void p36_vk_q5_scale_min(const uint8_t *src, uint32_t j, uint8_t *scale, uint8_t *mn) {
    const uint32_t s0 = src[j];
    if (j < 4u) {
        *scale = (uint8_t)(s0 & 63u);
        *mn = (uint8_t)(src[4u + j] & 63u);
    } else {
        const uint32_t sj4 = src[j + 4u];
        const uint32_t sjm4 = src[j - 4u];
        *scale = (uint8_t)((sj4 & 15u) | ((sjm4 >> 6) << 4));
        *mn = (uint8_t)((sj4 >> 4) | ((s0 >> 6) << 4));
    }
}

static uint64_t p36_vk_kquant_mmq_block_bytes(uint32_t type) {
    switch (type) {
    case P36_VK_TENSOR_Q5_K: return sizeof(p36_vk_block_q5_Kx8);
    case P36_VK_TENSOR_Q6_K: return sizeof(p36_vk_block_q6_Kx8);
    default: return 0;
    }
}

static bool p36_vk_use_mmq(void) {
    const char *env = getenv("P36_VK_MMQ");
    return env && env[0] && env[0] != '0';
}

static bool p36_vk_use_mmq_fast(void) {
    const char *env = getenv("P36_VK_MMQ_FAST");
    return env && env[0] && env[0] != '0';
}

/* Largest batch the Q2_0/PQ2_0 small-batch kernel takes before falling back
 * to the 128-token mmq tile.  0 disables it. */
static uint32_t p36_vk_q2_small_max(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("P36_VK_Q2_SMALL_MAX");
        long v = env && env[0] ? strtol(env, NULL, 10) : 16;
        cached = v < 0 ? 0 : (v > 4096 ? 4096 : (int)v);
    }
    return (uint32_t)cached;
}

static bool p36_vk_env_default_on(const char *name) {
    const char *env = getenv(name);
    return !env || !env[0] || env[0] != '0';
}

static bool p36_vk_have_subgroups(bool clustered) {
    return p36_vk.subgroup_size >= 32u && p36_vk.subgroup_arithmetic &&
           (!clustered || p36_vk.subgroup_clustered);
}

static bool p36_vk_use_gpu_attn_post(void) {
    return p36_vk_env_default_on("P36_VK_GPU_ATTN_POST");
}

/* Prefill batches at or above this token count route routed-MoE work
 * through the GEMM kernels; below it the matvec path keeps
 * decode-identical numerics for chunk tails and speculative windows. */
static uint32_t p36_vk_moe_gemm_min(void) {
    const char *env = getenv("P36_VK_MOE_GEMM_MIN");
    long v = env ? strtol(env, NULL, 10) : 0;
    return v > 0 ? (uint32_t)v : 128u;
}

static bool p36_vk_use_moe_down_sum_decode(void) {
    return p36_vk.have_storage16 && p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_MOE_DOWN_SUM_DECODE");
}

/* Sampled once: the runtime skips the scores scratch allocation when the
 * fused attention path is on, so the choice must not change mid-process. */
static bool p36_vk_use_attn_qtile(void) {
    static int cached = -1;
    if (cached < 0) cached = p36_vk_env_default_on("P36_VK_ATTN_QTILE") ? 1 : 0;
    return cached != 0 && p36_vk_have_subgroups(true);
}

static bool p36_vk_use_attn_qtile2(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_QTILE2");
}

/* Flash-attention replacement for attn_prefill_qtile2{_gqa6} (GQA 6 dense,
 * GQA 8 MoE).  P36_VK_ATTN_FA=0 falls back to qtile2 for parity A/B. */
static bool p36_vk_use_attn_fa(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_FA");
}

static bool p36_vk_use_attn_fused(void) {
    static int cached = -1;
    if (cached < 0) cached = p36_vk_env_default_on("P36_VK_ATTN_FUSED") ? 1 : 0;
    return cached != 0 && p36_vk_have_subgroups(true);
}

bool p36_gpu_attn_fused_enabled(void) {
    return !p36_gpu_quality && p36_vk_use_attn_fused();
}

static bool p36_vk_use_gpu_rms(void) {
    return p36_vk_env_default_on("P36_VK_GPU_RMS");
}

static bool p36_vk_use_add_rms(void) {
    return p36_vk_env_default_on("P36_VK_ADD_RMS");
}

static bool p36_vk_use_gpu_recurrent(void) {
    return p36_vk_env_default_on("P36_VK_GPU_RECURRENT");
}

static bool p36_vk_use_delta_fast_prefill(void) {
    return p36_vk_env_default_on("P36_VK_DELTA_FAST_PREFILL");
}

static bool p36_vk_use_delta_col_prefill(void) {
    const char *env = getenv("P36_VK_DELTA_COL_PREFILL");
    bool enabled = env && env[0] ? env[0] != '0' : !p36_vk.bc250;
    return p36_vk_have_subgroups(false) && p36_vk.subgroup_size_control &&
           enabled;
}

static bool p36_vk_use_delta_decode(void) {
    return p36_vk_env_default_on("P36_VK_DELTA_DECODE");
}

static bool p36_vk_use_delta_reg(void) {
    return p36_vk.subgroup_clustered &&
           p36_vk_env_default_on("P36_VK_DELTA_REG");
}

static bool p36_vk_use_attn_splitk(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_SPLITK");
}

/* Width in keys of one split-K span.  Decode dispatches n_head * n_spans
 * workgroups, so on this 40-CU board the stock 512 gives 16 * 5 = 80
 * workgroups at ctx 2048 -- two per CU, and the kernel measures ~17 GB/s
 * against a ~360 GB/s roofline, i.e. it is occupancy-bound rather than
 * bandwidth-bound.  A narrower span buys occupancy there.  At ctx 8192 and up
 * the returns are small in both directions (round-2 R5): the split already
 * dispatches n_head * n_spans workgroups and 408 workgroups at ctx 8192 (10
 * per CU, 4 waves each) fill the board, so the kernel is instruction-bound,
 * not occupancy-bound -- attn_decode_split.spv is VGPR 48 / LDS 1536 B / 20
 * subgroups per SIMD, i.e. nothing is register- or LDS-limited.
 *
 * NOT bit-exact: attn_combine walks the per-span partials sequentially in
 * f32, rescaling by exp(m - nm) at each step, so changing the span count
 * regroups that reduction and changes rounding.
 *
 * The cost of narrowing scales with context, in BOTH directions, and this is
 * why the default stays 512:
 *
 *   spans = kv_max / span, so at ctx 32768 a 128-key span puts 256 sequential
 *   f32 rescales in the chain where 512 puts 64.  Online softmax keeps that
 *   bounded, not free -- drift grows with span count, hence with context.
 *
 *   attn_combine's own cost grows the same way: measured 10.302 -> 14.870 ms
 *   at ctx 2048 going 512 -> 128, and that term scales with span count while
 *   the split-side saving does not.
 *
 * The +2.2% for span 128 was measured at ctx 2048 only (16 spans vs 4) and
 * MUST NOT be assumed to hold at long context -- neither the gain nor the
 * numerical cost has been measured beyond 2064.  Long context is measured now
 * (ctx 16384, split ms/call, 1 rep per arm, fresh process): 512 -> 0.5334 ms
 * with 792 workgroups on Swift (24 q heads) and 0.3588 ms with 528 on the
 * guard (16 q heads); 1024 -> +7.1% / +12.3%, 2048 -> +28.1% / +39.4%,
 * 4096 -> +70.5% / +96.5%, i.e. WIDENING starves the grid monotonically (per
 * key 1.32 -> 1.85 ns Swift, 1.33 -> 2.15 ns guard).  Narrowing is monotone and
 * real but small: 256 -> -4.2% and 128 -> -7.0% of the split on Swift
 * (1.315 -> 1.280 -> 1.252 ns per key), 128 -> -9.9% on the guard, and the
 * guard's 256 arm disagrees with that trend (+4.0%, single rep), so the narrow
 * side is not established.  attn_combine pays it back: per call it is
 * 0.06037 + 1.02e-4 * spans ms on Swift and 0.04006 + 1.03e-4 * spans ms on
 * the guard (residual <= 0.0003 ms over 5..129 spans) -- a fixed cost that
 * scales with the head count plus ~1 us per partial, so at span 128 the
 * combine is +15% / +22% over 512.  Net: at most ~1% of the decode token on
 * Swift, ~2-3% on the guard, both single-rep, on top of the drift above.
 * 512 stays until someone pays for an interleaved A/B plus the NLL check.
 * Evidence: karpathy/evidence/raw/r5-longctx-attn/. */
static uint32_t p36_vk_attn_span(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("P36_VK_ATTN_SPAN");
        long v = env && env[0] ? strtol(env, NULL, 10) : 512;
        if (v < 64 || v > 4096 || (v & (v - 1)) != 0) v = 512;
        cached = (int)v;
    }
    return (uint32_t)cached;
}

static uint32_t p36_vk_kv_cache_row_bytes(uint32_t type, uint32_t n) {
    if (type == 0u) return n * (uint32_t)sizeof(uint16_t);
    if (type == 1u) return ((n + P36_VK_QK8_0 - 1u) / P36_VK_QK8_0) * 34u;
    if (type == 2u) return ((n + P36_VK_QK8_0 - 1u) / P36_VK_QK8_0) * 18u;
    return 0;
}

static bool p36_vk_use_f32_fast(void) {
    return p36_vk_have_subgroups(false) && p36_vk_env_default_on("P36_VK_F32_FAST");
}

static bool p36_vk_use_gpu_ffn_tail(void) {
    return p36_vk_env_default_on("P36_VK_GPU_FFN_TAIL");
}

static bool p36_vk_use_gpu_swiglu(void) {
    return p36_vk_env_default_on("P36_VK_GPU_SWIGLU");
}

static bool p36_vk_use_moe_fast(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_MOE_FAST");
}

static bool p36_vk_use_dense_iq3_fast(void) {
    return p36_vk.have_storage8 && p36_vk.have_int_dot &&
           p36_vk.subgroup_size == 64u &&
           p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_DENSE_IQ3_FAST");
}

static bool p36_vk_use_moe_gate_up(void) {
    return p36_vk_env_default_on("P36_VK_MOE_GATE_UP");
}

static bool p36_vk_use_router_fast_prefill(void) {
    const char *env = getenv("P36_VK_ROUTER_FAST_PREFILL");
    return env && env[0] && env[0] != '0';
}

static bool p36_vk_use_q8_0_gpu(void) {
    return p36_vk_env_default_on("P36_VK_Q8_0_GPU");
}

static bool p36_vk_use_q8_0_f32b(void) {
    return p36_vk.have_storage16 && p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_Q8_0_F32B");
}

static int p36_vk_quantize_q8_k_dispatch(p36_gpu_tensor *q8,
                                         const p36_gpu_tensor *x,
                                         uint64_t in_dim,
                                         uint64_t n_tok) {
    uint64_t x_bytes = 0;
    uint64_t q8_bytes = 0;
    const uint64_t blocks = in_dim / P36_VK_QK_K;

    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0) return 0;
    if (in_dim > UINT32_MAX || n_tok > UINT32_MAX || blocks > UINT32_MAX) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !p36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes)) {
        return 0;
    }
    if (!p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(x, 0, x_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) {
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } push = { (uint32_t)in_dim, (uint32_t)blocks, (uint32_t)n_tok };
    const p36_gpu_tensor *bindings[2] = { x, q8 };

    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("q8_k_quant",
                                 &p36_vk.quantize_q8_k,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (uint32_t)blocks,
                                 (uint32_t)n_tok,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_quantize_q8_k_tensor(p36_gpu_tensor *out,
                                 const p36_gpu_tensor *x,
                                 uint64_t in_dim,
                                 uint64_t n_tok) {
    uint64_t blocks = in_dim / P36_VK_QK_K;
    uint64_t x_bytes = 0;
    uint64_t q8_count = 0;
    uint64_t q8_bytes = 0;

    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || n_tok == 0) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !p36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_count) ||
        !p36_u64_mul_ok(q8_count, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(x, 0, x_bytes) || !p36_gpu_tensor_range_ok(out, 0, q8_bytes)) return 0;

    if (!p36_gpu_quality && in_dim == 512u && n_tok == 8u && p36_vk_env_default_on("P36_VK_Q8_K_MOE_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    /* Decode/MTP-verify activations quantize on GPU too: the host quantizer
     * here was the largest remaining decode flush cause after GPU RMS. */
    if (!p36_gpu_quality && n_tok <= 8u && p36_vk_env_default_on("P36_VK_Q8_K_MOE_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    if (!p36_gpu_quality && n_tok > 8u && p36_vk_env_default_on("P36_VK_Q8_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }

    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_q8k_quant_x");
    p36_vk_block_q8_K *q8_host = p36_vk_tensor_contents_labeled(out, "submit_wait_q8k_quant_out");
    if (!x_host || !q8_host) return 0;
    p36_vk_quantize_q8_k_host(q8_host, x_host, n_tok, in_dim);
    return 1;
}


static float p36_vk_sigmoid_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}




static p36_gpu_tensor *p36_gpu_tensor_root(const p36_gpu_tensor *tensor, uint64_t *offset) {
    if (!tensor) return NULL;
    if (!tensor->owner) {
        if (offset) *offset = tensor->offset;
        return (p36_gpu_tensor *)tensor;
    }
    uint64_t root_offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor->owner, &root_offset);
    if (offset) *offset = root_offset + tensor->offset;
    return root;
}

static bool p36_gpu_tensor_range_ok(const p36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes) {
    return tensor && offset <= tensor->bytes && bytes <= tensor->bytes - offset;
}

static const unsigned char *p36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes) {
    const unsigned char *map = model_map ? (const unsigned char *)model_map :
                                           (const unsigned char *)p36_gpu_model_map;
    const uint64_t size = model_map ? model_size : p36_gpu_model_size;
    if (!map) return NULL;
    if (weight_offset > size || bytes > size - weight_offset) return NULL;
    return map + weight_offset;
}

static uint32_t p36_vk_find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < p36_vk.mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) == 0) continue;
        if ((p36_vk.mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

static uint32_t p36_vk_find_memory_type_avoiding(uint32_t bits,
                                                 VkMemoryPropertyFlags want,
                                                 VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < p36_vk.mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) == 0) continue;
        VkMemoryPropertyFlags flags = p36_vk.mem_props.memoryTypes[i].propertyFlags;
        if ((flags & want) == want && (flags & avoid) == 0) return i;
    }
    return UINT32_MAX;
}

static uint64_t p36_vk_memory_total(void) {
    uint64_t total = 0;
    const bool uma = p36_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    for (uint32_t i = 0; i < p36_vk.mem_props.memoryHeapCount; i++) {
        VkMemoryHeap heap = p36_vk.mem_props.memoryHeaps[i];
        if (uma || (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) total += heap.size;
    }
    return total;
}

static void p36_vk_kernel_destroy(p36_vk_kernel *k) {
    if (!p36_vk.device || !k) return;
    if (k->pipeline) vkDestroyPipeline(p36_vk.device, k->pipeline, NULL);
    if (k->layout) vkDestroyPipelineLayout(p36_vk.device, k->layout, NULL);
    if (k->set_layout) vkDestroyDescriptorSetLayout(p36_vk.device, k->set_layout, NULL);
    if (k->shader) vkDestroyShaderModule(p36_vk.device, k->shader, NULL);
    k->pipeline = VK_NULL_HANDLE;
    k->layout = VK_NULL_HANDLE;
    k->set_layout = VK_NULL_HANDLE;
    k->shader = VK_NULL_HANDLE;
    k->ready = false;
}

static void *p36_read_file(const char *path, size_t *len_out) {
    char rooted[PATH_MAX];
    FILE *fp = NULL;
    const char *env_dir = getenv("P36_SHADER_DIR");
    if (path && path[0] != '/' && env_dir && env_dir[0] &&
        snprintf(rooted, sizeof(rooted), "%s/%s", env_dir, path) < (int)sizeof(rooted)) {
        fp = fopen(rooted, "rb");
    }
    if (!fp && path && path[0] != '/' && p36_vk.shader_root[0] &&
        snprintf(rooted, sizeof(rooted), "%s/%s",
                 p36_vk.shader_root, path) < (int)sizeof(rooted)) {
        fp = fopen(rooted, "rb");
    }
    if (!fp && path && path[0] != '/') {
        char exe[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (n > 0) {
            exe[n] = '\0';
            char *slash = strrchr(exe, '/');
            if (slash) {
                *slash = '\0';
                if (snprintf(rooted, sizeof(rooted), "%s/%s", exe, path) < (int)sizeof(rooted)) {
                    fp = fopen(rooted, "rb");
                }
            }
        }
    }
    if (!fp) fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len <= 0 || (len & 3) != 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    void *buf = malloc((size_t)len);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *len_out = (size_t)len;
    return buf;
}

static int p36_vk_kernel_init(p36_vk_kernel *k) {
    if (k->ready) return 1;
    if (k->bindings > 16 ||
        k->bindings > p36_vk.props.limits.maxPerStageDescriptorStorageBuffers ||
        k->bindings > p36_vk.props.limits.maxDescriptorSetStorageBuffers) return 0;

    size_t code_size = 0;
    void *code = p36_read_file(k->path, &code_size);
    if (!code) {
        fprintf(stderr, "p36: failed to read Vulkan shader %s\n", k->path);
        return 0;
    }

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = code,
    };
    VkResult rc = vkCreateShaderModule(p36_vk.device, &smci, NULL, &k->shader);
    free(code);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateShaderModule failed for %s (%d)\n", k->path, rc);
        return 0;
    }

    VkDescriptorSetLayoutBinding bindings[16];
    for (uint32_t i = 0; i < k->bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = k->bindings,
        .pBindings = bindings,
    };
    rc = vkCreateDescriptorSetLayout(p36_vk.device, &dsci, NULL, &k->set_layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateDescriptorSetLayout failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = k->push_bytes,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k->set_layout,
        .pushConstantRangeCount = k->push_bytes ? 1u : 0u,
        .pPushConstantRanges = k->push_bytes ? &push : NULL,
    };
    rc = vkCreatePipelineLayout(p36_vk.device, &plci, NULL, &k->layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreatePipelineLayout failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = k->shader,
        .pName = "main",
    };
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32u,
    };
    /* Shaders that declare local_size_x = 32 and use no cross-lane operation
     * at all.  On wave64 the upper half of every wave is inactive, and since
     * nothing reads a neighbouring lane the wave width cannot change a
     * single result bit -- forcing wave32 only stops wasting half of each
     * wave.  Each entry was checked for zero cross-lane builtin uses in its
     * .comp source; do not add a shader here without that check, because
     * for a shader that does reduce across lanes the two widths are not
     * interchangeable.
     *
     * Two traps found by audit (karpathy/evidence/wave32-eligibility.md):
     * (1) a `gl_Subgroup*` READ is not a builtin *call*, but
     * `gl_SubgroupID`/`gl_SubgroupInvocationID` are just as lane-count
     * dependent -- a shader deriving an index from them is disqualified too;
     * (2) when a tile is partitioned by gl_SubgroupID with a 32-wide stride,
     * wave32 is a *correctness requirement*, not a tuning choice: at wave64
     * only 2 subgroups exist and the tail of every tile is written by
     * nobody.  That is why the whole dense `_mmq` family is forced here. */
    static const char *const p36_vk_force_wave32[] = {
        "delta_net_cols",
        "rope_qwen.spv",
        "rope_qwen_mrope.spv",
        "quantize_q8_0.spv",
    };
    /* The old suffix test was "_mmq.spv", which misses the _r4 / _bm64
     * variants because those end in "_mmq_r4.spv".  Specifically
     * dense_iq3_s_mmq_r4 -- same source, same gl_SubgroupID token formula,
     * selected at runtime for IQ3_S with out_dim == 48 && n_tok <= 128 --
     * therefore ran at wave64, where only 2 subgroups exist and tokens
     * 80..127 of every 128-token tile are written by nobody.  Match the
     * whole dense _mmq family, suffix variants included. */
    bool force_wave32 = strstr(k->path, "dense_") && strstr(k->path, "_mmq");
    for (size_t i = 0; !force_wave32 && i < sizeof(p36_vk_force_wave32) / sizeof(*p36_vk_force_wave32); i++) {
        force_wave32 = strstr(k->path, p36_vk_force_wave32[i]) != NULL;
    }
    static const char *const p36_vk_clean_scan_wave32[] = {
        "moe_matvec.spv",
        "moe_matvec_fast.spv",
        "matmul_q8_0_decode.spv",
        "matmul_q8_0_decode_b64.spv",
        "matmul_q8_0_decode_p36.spv",
        "add_rms_norm.spv",
        "rms_norm.spv",
        "rms_norm_rope_qwen.spv",
        "rms_norm_rope_kv_qwen.spv",
        "kv_store_quant.spv",
    };
    const char *clean_env = getenv("P36_VK_WAVE32_CLEAN");
    if (clean_env && clean_env[0] && clean_env[0] != '0') {
        for (size_t i = 0; !force_wave32 && i < sizeof(p36_vk_clean_scan_wave32) / sizeof(*p36_vk_clean_scan_wave32); i++) {
            force_wave32 = strstr(k->path, p36_vk_clean_scan_wave32[i]) != NULL;
        }
    }
    /* dense_extra_decode_ptq1_0 is the mirror-image trap: it partitions all
     * 128 weights of a PTQ1_0 block across every lane of the 64-wide
     * workgroup (b_start = lid>>3) and recovers the true dot product with
     * one subgroupAdd(acc[r]) over the whole workgroup. That is only
     * correct if the workgroup compiles to a single 64-wide subgroup --
     * the ptq1_0 == 64u dispatch gate above reads the device's *default*
     * subgroup size, not what RADV actually picks for this pipeline, so
     * pin it explicitly instead of trusting the heuristic. */
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size64 = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 64u,
    };
    bool force_wave64 = strstr(k->path, "dense_extra_decode_ptq1_0") != NULL;
    if (p36_vk.subgroup_size_control && force_wave32) {
        stage.pNext = &subgroup_size;
        stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    } else if (p36_vk.subgroup_size_control && p36_vk.subgroup_size == 64u && (force_wave64 || strstr(k->path,"dense_extra_decode_q2_0") != NULL || strstr(k->path,"dense_extra_decode_pq2_0") != NULL || strstr(k->path,"dense_extra_small_") != NULL)) {
        stage.pNext = &subgroup_size64;
        stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = k->layout,
    };
    /* P36_VK_SHADER_TRACE=1 names each pipeline as it is built, so an
     * RADV_DEBUG=shaderstats dump (which prints VGPR/LDS with no shader
     * name) can be attributed to a shader. */
    if (getenv("P36_VK_SHADER_TRACE")) fprintf(stderr, "p36: building pipeline %s%s\n", k->path, force_wave32 ? " [wave32]" : "");
    rc = vkCreateComputePipelines(p36_vk.device, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateComputePipelines failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    k->ready = true;
    return 1;
}

static p36_gpu_tensor *p36_vk_tensor_alloc_dl_unlocked(uint64_t bytes) {
    p36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->bytes = bytes;
    tensor->alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tensor->alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(p36_vk.device, &bci, NULL, &tensor->buffer) != VK_SUCCESS) {
        free(tensor);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p36_vk.device, tensor->buffer, &req);
    tensor->private_charge = req.size;
    uint32_t mt = p36_vk_find_memory_type_avoiding(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(p36_vk.device, &mai, NULL, &tensor->memory) != VK_SUCCESS) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (vkBindBufferMemory(p36_vk.device, tensor->buffer, tensor->memory, 0) != VK_SUCCESS) {
        vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }

    tensor->device_local = true;
    p36_gpu_live_bytes += bytes;
    if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
    return tensor;
}

static p36_gpu_tensor *p36_vk_tensor_alloc_kind_unlocked(uint64_t bytes, bool device_pref,
                                                         bool clear) {
    p36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->bytes = bytes;
    tensor->alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tensor->alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(p36_vk.device, &bci, NULL, &tensor->buffer) != VK_SUCCESS) {
        free(tensor);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p36_vk.device, tensor->buffer, &req);
    /* Prefer cached (snooped) host-visible memory: on the BC-250 the
     * write-combined GTT type is not reliably coherent for device writes
     * followed by host reads, and every host op reads GPU output.
     * Weights are the exception (device_pref): written once by memcpy and
     * only ever read by shaders, so they go to the device-local heap, which
     * is several times larger than the host-cached one and must hold the
     * routed expert banks. */
    uint32_t mt = UINT32_MAX;
    if (device_pref) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        tensor->device_local = mt != UINT32_MAX;
    }
    if (mt == UINT32_MAX) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    if (mt == UINT32_MAX) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(p36_vk.device, &mai, NULL, &tensor->memory) != VK_SUCCESS) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (vkBindBufferMemory(p36_vk.device, tensor->buffer, tensor->memory, 0) != VK_SUCCESS ||
        vkMapMemory(p36_vk.device, tensor->memory, 0, req.size, 0, (void **)&tensor->data) != VK_SUCCESS)
    {
        vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (clear) memset(tensor->data, 0, (size_t)tensor->alloc_bytes);

    p36_gpu_live_bytes += bytes;
    if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
    return tensor;
}

static void p36_vk_tensor_free_unlocked(p36_gpu_tensor *tensor);
static void p36_vk_b16_ring_destroy_unlocked(void);

/* Pooled scratch keeps its VRAM; free it all when an allocation fails so a
 * long-context prefill retries instead of dying on a full heap. */
static bool p36_vk_pools_drain_unlocked(void) {
    bool any = p36_vk_pool_n || p36_vk_private_pool_n;
    while (p36_vk_pool_n) p36_vk_tensor_free_unlocked(p36_vk_pool[--p36_vk_pool_n]);
    while (p36_vk_private_pool_n)
        p36_vk_tensor_free_unlocked(p36_vk_private_pool[--p36_vk_private_pool_n]);
    return any;
}

static p36_gpu_tensor *p36_vk_tensor_alloc_scratch_unlocked(uint64_t bytes) {
    const uint64_t alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;
    if (p36_gpu_quality) return p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    for (uint32_t i = 0; i < p36_vk_private_pool_n; i++) {
        p36_gpu_tensor *tensor = p36_vk_private_pool[i];
        if (tensor->alloc_bytes != alloc_bytes) continue;
        p36_vk_private_pool[i] = p36_vk_private_pool[--p36_vk_private_pool_n];
        tensor->bytes = bytes;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        return tensor;
    }
    if (p36_vk_private_bytes <= P36_VK_PRIVATE_LIMIT &&
        alloc_bytes <= P36_VK_PRIVATE_LIMIT - p36_vk_private_bytes) {
        p36_gpu_tensor *tensor = p36_vk_tensor_alloc_dl_unlocked(bytes);
        if (tensor && tensor->private_charge <=
                      P36_VK_PRIVATE_LIMIT - p36_vk_private_bytes) {
            tensor->private_scratch = true;
            p36_vk_private_bytes += tensor->private_charge;
            if (p36_vk_private_bytes > p36_vk_private_peak)
                p36_vk_private_peak = p36_vk_private_bytes;
            return tensor;
        }
        p36_vk_tensor_free_unlocked(tensor);
    }
    p36_gpu_tensor *t = p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    if (!t && p36_vk_pools_drain_unlocked())
        t = p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    return t;
}

static void p36_vk_tensor_free_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (!tensor->owner) {
        p36_gpu_live_bytes -= tensor->bytes;
        if (tensor->private_scratch) p36_vk_private_bytes -= tensor->private_charge;
        if (tensor->data) vkUnmapMemory(p36_vk.device, tensor->memory);
        if (tensor->memory) vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        if (tensor->buffer) vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
    }
    free(tensor);
}

/* Pool-or-free for a tensor with no pending GPU references. A full pool
 * evicts round-robin instead of rejecting: otherwise a prefill chunk fills
 * every slot with chunk-sized scratch and decode-sized buffers can never
 * enter, turning each decode step into fresh vkAllocateMemory calls. */
static uint32_t p36_vk_pool_rr;
static uint32_t p36_vk_private_pool_rr;
static void p36_vk_tensor_retire_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor->owner && tensor->private_scratch) {
        p36_gpu_live_bytes -= tensor->bytes;
        tensor->bytes = 0;
        if (p36_vk_private_pool_n < P36_VK_PRIVATE_POOL_CAP) {
            p36_vk_private_pool[p36_vk_private_pool_n++] = tensor;
        } else {
            p36_vk_private_pool[p36_vk_private_pool_rr]->bytes = 0;
            p36_vk_tensor_free_unlocked(p36_vk_private_pool[p36_vk_private_pool_rr]);
            p36_vk_private_pool[p36_vk_private_pool_rr] = tensor;
            p36_vk_private_pool_rr =
                (p36_vk_private_pool_rr + 1u) % P36_VK_PRIVATE_POOL_CAP;
        }
    } else if (!tensor->owner && !tensor->device_local && tensor->data &&
        tensor->alloc_bytes <= P36_VK_POOL_MAX_BYTES) {
        p36_gpu_live_bytes -= tensor->bytes;
        if (p36_vk_pool_n < P36_VK_POOL_CAP) {
            p36_vk_pool[p36_vk_pool_n++] = tensor;
        } else {
            /* Pooled entries already left the live-byte count at retire. */
            p36_vk_pool[p36_vk_pool_rr]->bytes = 0;
            p36_vk_tensor_free_unlocked(p36_vk_pool[p36_vk_pool_rr]);
            p36_vk_pool[p36_vk_pool_rr] = tensor;
            p36_vk_pool_rr = (p36_vk_pool_rr + 1u) % P36_VK_POOL_CAP;
        }
    } else {
        p36_vk_tensor_free_unlocked(tensor);
    }
}

/* Retire now if the tensor's last GPU reference has completed, else park
 * it on the retired queue for the next sweep. */
static void p36_vk_tensor_release_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->last_use_seq > p36_vk_completed_seq) {
        if (p36_vk_retired_n == P36_VK_RETIRE_CAP) p36_vk_flush_reason_unlocked("submit_wait_retire_cap");
        if (tensor->last_use_seq > p36_vk_completed_seq) {
            p36_vk_retired[p36_vk_retired_n++] = tensor;
            return;
        }
    }
    p36_vk_tensor_retire_unlocked(tensor);
}

static void p36_vk_stream_cache_release_unlocked(bool keep_config) {
    bool enabled = p36_vk_stream.enabled;
    uint32_t budget = p36_vk_stream.budget;
    uint32_t full_layers = p36_vk_stream.full_layers;
    uint32_t runtime_cap = p36_vk_stream.runtime_cap;
    uint32_t allocation_failures = p36_vk_stream.allocation_failures;
    uint64_t expert_bytes = p36_vk_stream.configured_expert_bytes;

    p36_vk_tensor_free_unlocked(p36_vk_stream.gate);
    p36_vk_tensor_free_unlocked(p36_vk_stream.up);
    p36_vk_tensor_free_unlocked(p36_vk_stream.down);
    p36_vk_tensor_free_unlocked(p36_vk_stream.gate_scales);
    p36_vk_tensor_free_unlocked(p36_vk_stream.up_scales);
    p36_vk_tensor_free_unlocked(p36_vk_stream.down_scales);
    free(p36_vk_stream.entry);
    memset(&p36_vk_stream, 0, sizeof(p36_vk_stream));
    if (keep_config) {
        p36_vk_stream.enabled = enabled;
        p36_vk_stream.budget = budget;
        p36_vk_stream.full_layers = full_layers;
        p36_vk_stream.runtime_cap = runtime_cap;
        p36_vk_stream.allocation_failures = allocation_failures;
        p36_vk_stream.configured_expert_bytes = expert_bytes;
    }
}

static bool p36_vk_stream_copy_model_range(const void *model_map,
                                           uint64_t model_size,
                                           uint64_t offset,
                                           void *dst,
                                           uint64_t bytes) {
    if (bytes == 0) return true;
    if (!dst || bytes > (uint64_t)SIZE_MAX) return false;
    if (p36_gpu_model_fd >= 0) {
        uint64_t done = 0;
        while (done < bytes) {
            size_t n = (size_t)(bytes - done);
            if (n > (size_t)(256u << 20)) n = (size_t)(256u << 20);
            ssize_t got = pread(p36_gpu_model_fd,
                                (unsigned char *)dst + done,
                                n,
                                (off_t)(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;
            done += (uint64_t)got;
        }
        if (done == bytes) return true;
    }

    const unsigned char *src = p36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    if (!src) return false;
    memcpy(dst, src, (size_t)bytes);
    return true;
}

/* Submit the open batch, wait for it, then settle everything that was
 * deferred while it was recording: retired tensors and the descriptor
 * pool (all its sets belong to the finished batch). */
/* Command-buffer ring: the batch cap submits eagerly WITHOUT waiting, so
 * the GPU chews on earlier work while the host records ahead; only a host
 * read waits for everything. Hazard sets survive eager submits because the
 * pending work still executes in submission order. */
enum { P36_VK_CB_RING = 4 };
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorPool pool;
    bool pending;
    uint64_t seq;
} p36_vk_cb_slot;
static p36_vk_cb_slot p36_vk_ring[P36_VK_CB_RING];
static uint32_t p36_vk_ring_cur;

/* Device-lost latch.  A gfx ring timeout (amdgpu's 10 s TDR) destroys the
 * context: every subsequent Vulkan call against this device fails, and
 * RADV's command buffer handle becomes unusable, so recording into it
 * segfaults inside vkEndCommandBuffer rather than returning an error.
 * Once any submit-path call reports VK_ERROR_DEVICE_LOST we latch here and
 * fail every later batch operation at its entry point.  The latch is
 * deliberately one-way: there is no recover-and-continue path, because the
 * dispatches that were in flight produced no results and silently resuming
 * would corrupt the forward pass.  p36_session_sync() turns the failed pass
 * into an error string for the caller. */
static bool p36_vk_device_lost;
/* P36_VK_FAULT_INJECT_LOST=<n>: force the n-th vkQueueSubmit to report
 * device loss, so the latch and its unwind can be tested without provoking
 * a real TDR.  Diagnostic only; 0 (unset) disables it. */
static uint64_t p36_vk_fault_inject_lost;
static uint64_t p36_vk_submit_calls;

static int p36_vk_ok(VkResult rc, const char *what) {
    if (rc == VK_SUCCESS) return 1;
    if (rc == VK_ERROR_DEVICE_LOST && !p36_vk_device_lost) {
        p36_vk_device_lost = true;
        fprintf(stderr,
                "p36: Vulkan device lost during %s -- the GPU context was destroyed "
                "(usually an amdgpu ring timeout; check `journalctl -k` for "
                "\"ring gfx_0.0.0 timeout\").  All further GPU work is refused; "
                "restart the process.  Lower --prefill-chunk if this reproduces.\n",
                what);
    }
    return 0;
}

static bool p36_vk_device_lost_unlocked(void) { return p36_vk_device_lost; }

static bool p36_vk_gpu_busy_unlocked(void) {
    if (p36_vk_batch_recording) return true;
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pending) return true;
    }
    return false;
}

static int p36_vk_slot_wait_unlocked(p36_vk_cb_slot *s) {
    VkResult rc = VK_SUCCESS;
    if (!s->pending) return 1;
    if (p36_vk_device_lost) { s->pending = false; return 0; }
    /* Decode batches finish in tens of microseconds; spinning on the fence
     * status dodges the syscall + thread wakeup of a blocking wait. */
    for (uint32_t spin = 0; spin < 4096; spin++) {
        rc = vkGetFenceStatus(p36_vk.device, s->fence);
        if (rc != VK_NOT_READY) break;
    }
    if (rc == VK_NOT_READY) rc = vkWaitForFences(p36_vk.device, 1, &s->fence, VK_TRUE, UINT64_MAX);
    if (rc != VK_SUCCESS) {
        /* A lost device never signals this fence; drop the slot so callers
         * unwind instead of waiting on it again. */
        s->pending = false;
        return p36_vk_ok(rc, "fence wait");
    }
    vkResetFences(p36_vk.device, 1, &s->fence);
    if (s->pool) vkResetDescriptorPool(p36_vk.device, s->pool, 0);
    s->pending = false;
    if (s->seq > p36_vk_completed_seq) p36_vk_completed_seq = s->seq;
    return 1;
}

static int p36_vk_submit_current_unlocked(bool host_read) {
    p36_vk_cb_slot *s = &p36_vk_ring[p36_vk_ring_cur];
    if (!p36_vk_batch_recording) return 1;
    if (p36_vk_device_lost) return 0;
    if (host_read) {
        /* Host ops read these buffers right after the fence.  The fence's
         * implicit host visibility is not reliable on the BC-250's RADV
         * (intermittent stale reads), so flush shader writes to host
         * explicitly. */
        VkMemoryBarrier host_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(p36_vk.command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &host_barrier, 0, NULL, 0, NULL);
    }
    {
        VkResult rc = vkEndCommandBuffer(p36_vk.command_buffer);
        if (rc != VK_SUCCESS) {
            p36_vk_batch_recording = false;
            return p36_vk_ok(rc, "vkEndCommandBuffer");
        }
    }
    {
        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &p36_vk.command_buffer,
        };
        VkResult rc = vkQueueSubmit(p36_vk.queue, 1, &si, s->fence);
        if (rc == VK_SUCCESS && p36_vk_fault_inject_lost &&
            ++p36_vk_submit_calls == p36_vk_fault_inject_lost) {
            rc = VK_ERROR_DEVICE_LOST;
        }
        if (rc != VK_SUCCESS) {
            p36_vk_batch_recording = false;
            return p36_vk_ok(rc, "vkQueueSubmit");
        }
    }
    s->pending = true;
    s->seq = ++p36_vk_submit_seq;
    p36_vk_batch_recording = false;
    p36_vk_batch_count = 0;
    return 1;
}

static void p36_vk_ring_activate_unlocked(uint32_t idx) {
    p36_vk_ring_cur = idx;
    p36_vk.command_buffer = p36_vk_ring[idx].cb;
    p36_vk.fence = p36_vk_ring[idx].fence;
    p36_vk.descriptor_pool = p36_vk_ring[idx].pool;
}

/* Retire every parked tensor whose last GPU reference has completed.
 * Runs on each eager-submit fence wait so scratch memory drains as the
 * ring advances instead of piling up until the next full flush. */
static void p36_vk_retired_sweep_unlocked(void) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < p36_vk_retired_n; i++) {
        p36_gpu_tensor *t = p36_vk_retired[i];
        if (t->last_use_seq > p36_vk_completed_seq) p36_vk_retired[kept++] = t;
        else p36_vk_tensor_retire_unlocked(t);
    }
    p36_vk_retired_n = kept;
}

static int p36_vk_submit_eager_unlocked(void) {
    uint64_t t0 = p36_vk_now_ns();
    int ok = p36_vk_submit_current_unlocked(false);
    if (ok) {
        uint32_t next = (p36_vk_ring_cur + 1u) % P36_VK_CB_RING;
        ok = p36_vk_slot_wait_unlocked(&p36_vk_ring[next]);
        p36_vk_ring_activate_unlocked(next);
        p36_vk_retired_sweep_unlocked();
    }
    p36_vk_prof_op_add("submit_eager", 0, 1, 0, 0, p36_vk_now_ns() - t0, 0, 0);
    return ok;
}

static int p36_vk_flush_unlocked(void) {
    bool had_work = p36_vk_batch_recording;
    int ok = p36_vk_submit_current_unlocked(true);
    uint64_t t0 = p36_vk_now_ns();
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        had_work = had_work || p36_vk_ring[i].pending;
        if (!p36_vk_slot_wait_unlocked(&p36_vk_ring[i])) ok = 0;
    }
    if (had_work) {
        p36_vk_prof_flushes++;
        p36_vk_flush_total++;
        {
            uint64_t submit_ns = p36_vk_now_ns() - t0;
            p36_vk_prof_submit_ns += submit_ns;
            p36_vk_prof_op_add(p36_vk_flush_reason, 0, 1, 0, 0, submit_ns, 0, 0);
        }
        if (ok) p36_vk_prof_kernel_collect_unlocked();
        p36_vk_prof_query_count = 0;
        p36_vk_batch_written_n = 0;
        p36_vk_batch_dirty_all = false;
        p36_vk_hazard_reset();
    }
    p36_vk_retired_sweep_unlocked();
    return ok;
}

static int p36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out) {
    p36_vk_cb_slot *slot = &p36_vk_ring[p36_vk_ring_cur];
    if (p36_vk_device_lost) return 0;
    if (!slot->cb) {
        VkCommandBufferAllocateInfo cai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = p36_vk.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (vkAllocateCommandBuffers(p36_vk.device, &cai, &slot->cb) != VK_SUCCESS) return 0;
    }
    if (!slot->fence) {
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(p36_vk.device, &fci, NULL, &slot->fence) != VK_SUCCESS) return 0;
    }
    p36_vk.command_buffer = slot->cb;
    p36_vk.fence = slot->fence;
    if (!p36_vk_batch_recording) {
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult rc = vkBeginCommandBuffer(p36_vk.command_buffer, &bi);
        if (rc != VK_SUCCESS) return p36_vk_ok(rc, "vkBeginCommandBuffer");
        p36_vk_batch_recording = true;
    }
    if (cmd_out) *cmd_out = p36_vk.command_buffer;
    return 1;
}

static void p36_vk_note_written_tensor(p36_gpu_tensor *tensor) {
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, NULL);
    if (!root) return;
    uint32_t i = 0;
    while (i < p36_vk_batch_written_n && p36_vk_batch_written[i] != root) i++;
    if (i < p36_vk_batch_written_n) return;
    if (p36_vk_batch_written_n == P36_VK_WRITTEN_CAP) p36_vk_batch_dirty_all = true;
    else p36_vk_batch_written[p36_vk_batch_written_n++] = root;
}

static int p36_vk_flush_reason_unlocked(const char *reason) {
    const char *prev = p36_vk_flush_reason;
    p36_vk_flush_reason = reason ? reason : "submit_wait_other";
    int ok = p36_vk_flush_unlocked();
    p36_vk_flush_reason = prev;
    return ok;
}

static void p36_vk_weight_cache_free_unlocked(void) {
    p36_vk_weight *w = p36_vk.weights;
    p36_vk_packed_weight *pw = p36_vk.packed_weights;
    p36_vk_arena_block *b = p36_vk.arena;
    p36_vk_stream_cache_release_unlocked(true);
    while (w) {
        p36_vk_weight *next = w->next;
        p36_vk_tensor_free_unlocked(w->tensor);
        free(w);
        w = next;
    }
    p36_vk.weights = NULL;
    while (pw) {
        p36_vk_packed_weight *next = pw->next;
        p36_vk_tensor_free_unlocked(pw->tensor);
        free(pw);
        pw = next;
    }
    p36_vk.packed_weights = NULL;
    while (b) {
        p36_vk_arena_block *next = b->next;
        p36_vk_tensor_free_unlocked(b->base);
        free(b);
        b = next;
    }
    while (p36_vk_dense_recycled) {
        p36_gpu_tensor *next = p36_vk_dense_recycled->recycled_next;
        p36_vk_tensor_free_unlocked(p36_vk_dense_recycled);
        p36_vk_dense_recycled = next;
    }
    p36_vk.arena = NULL;
    p36_vk.arena_plan = NULL;
    free(p36_vk.arena_plan_sizes);
    p36_vk.arena_plan_sizes = NULL;
    p36_vk.arena_plan_count = 0;
    p36_vk.arena_plan_next = 0;
}

/* Dense Qwen3.8-27B IQ4_XS is larger than the BC-250's practical resident
 * budget. Keep the normal cache for smaller models, but allow the launcher
 * to bound it for oversized dense models. Eviction happens only after a
 * full queue drain, because cached weights may still be referenced by the
 * command buffer even when the host has finished recording the operation. */
static uint64_t p36_vk_env_gib(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long gib = strtoull(env, &end, 10);
    if (end == env || *end != '\0' || gib == 0 ||
        gib > (unsigned long long)(UINT64_MAX >> 30)) return 0;
    return (uint64_t)gib << 30;
}

static uint64_t p36_vk_dense_weight_cache_limit_unlocked(void) {
    if (!p36_gpu_dense_model) return 0;
    return p36_vk_env_gib("P36_VK_DENSE_WEIGHT_CACHE_GIB");
}

static uint64_t p36_vk_dense_weight_pin_limit_unlocked(uint64_t cache_limit) {
    uint64_t configured = p36_vk_env_gib("P36_VK_DENSE_WEIGHT_PIN_GIB");
    if (configured) return configured < cache_limit ? configured : cache_limit;
    return cache_limit > (2ull << 30) ? cache_limit - (2ull << 30) : 0;
}

static uint64_t p36_vk_weight_cache_bytes_unlocked(void) {
    uint64_t total = 0;
    for (p36_vk_weight *w = p36_vk.weights; w; w = w->next) {
        if (UINT64_MAX - total < w->bytes) return UINT64_MAX;
        total += w->bytes;
    }
    for (p36_vk_packed_weight *w = p36_vk.packed_weights; w; w = w->next) {
        uint64_t bytes = w->tensor ? w->tensor->bytes : 0;
        if (UINT64_MAX - total < bytes) return UINT64_MAX;
        total += bytes;
    }
    return total;
}

static uint64_t p36_vk_weight_cache_pinned_bytes_unlocked(void) {
    uint64_t total = 0;
    for (p36_vk_weight *w = p36_vk.weights; w; w = w->next) {
        if (!w->pinned) continue;
        if (UINT64_MAX - total < w->bytes) return UINT64_MAX;
        total += w->bytes;
    }
    return total;
}

/* Remove the least-recently-used evictable weight. Bounded weights use their
 * own allocations, so unlike arena views these frees return memory that can
 * be reused by a later layer without rebuilding the retained prefix. */
static bool p36_vk_weight_cache_evict_one_unlocked(void) {
    p36_vk_weight *victim = NULL;
    p36_vk_weight *victim_prev = NULL;
    p36_vk_weight *prev = NULL;
    for (p36_vk_weight *w = p36_vk.weights; w; prev = w, w = w->next) {
        if (!w->pinned) {
            victim = w;
            victim_prev = prev;
        }
    }
    if (!victim) return false;
    if (victim_prev) victim_prev->next = victim->next;
    else p36_vk.weights = victim->next;
    p36_gpu_tensor *tensor = victim->tensor;
    if (tensor && !tensor->owner) {
        p36_gpu_live_bytes -= tensor->bytes;
        tensor->bytes = 0;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        tensor->write_gen = 0;
        tensor->recycled_next = p36_vk_dense_recycled;
        p36_vk_dense_recycled = tensor;
    } else {
        p36_vk_tensor_free_unlocked(tensor);
    }
    free(victim);
    return true;
}

/* Reuse the Vulkan allocations for streamed weights. Repeated vkAllocateMemory
 * and vkFreeMemory calls were measurable once the cache began rotating every
 * decode pass; copied bytes are overwritten completely on reuse. */
static p36_gpu_tensor *p36_vk_dense_stream_alloc_unlocked(uint64_t bytes) {
    const uint64_t need = p36_round_up_u64(bytes ? bytes : 4u, 4u);
    p36_gpu_tensor *best = NULL;
    p36_gpu_tensor *best_prev = NULL;
    p36_gpu_tensor *prev = NULL;
    for (p36_gpu_tensor *t = p36_vk_dense_recycled; t; prev = t, t = t->recycled_next) {
        if (t->alloc_bytes < need) continue;
        if (!best || t->alloc_bytes < best->alloc_bytes) {
            best = t;
            best_prev = prev;
        }
    }
    if (best) {
        if (best_prev) best_prev->recycled_next = best->recycled_next;
        else p36_vk_dense_recycled = best->recycled_next;
        best->recycled_next = NULL;
        best->bytes = bytes;
        best->owner = NULL;
        best->offset = 0;
        best->gpu_written = false;
        best->last_use_seq = 0;
        best->write_gen = 0;
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        return best;
    }
    return p36_vk_tensor_alloc_kind_unlocked(bytes, true, false);
}

/* Suballocate weight bytes from the arena; new blocks prefer the
 * device-local heap and fall back to host-cached memory when it is full,
 * like the old per-weight allocations did.  256-byte alignment satisfies
 * any minStorageBufferOffsetAlignment the spec allows. */
enum { P36_VK_ARENA_BLOCK_BYTES = 512u << 20 };

static bool p36_vk_arena_plan_unlocked(const uint64_t *sizes, uint32_t count) {
    const uint64_t cap = 2ull << 30;
    uint64_t *blocks = calloc(count ? count : 1u, sizeof(*blocks));
    uint32_t nblocks = 0;
    uint64_t used = 0;

    if (!blocks) return false;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t need = p36_round_up_u64(sizes[i] ? sizes[i] : 4u, 256u);
        if (used && (need > cap || used > cap - need)) {
            blocks[nblocks++] = used;
            used = 0;
        }
        if (need > cap) blocks[nblocks++] = need;
        else used += need;
    }
    if (used) blocks[nblocks++] = used;

    p36_vk.arena_plan_sizes = blocks;
    p36_vk.arena_plan_count = nblocks;
    p36_vk.arena_plan_next = 0;
    return true;
}

static p36_gpu_tensor *p36_vk_arena_alloc_unlocked(uint64_t bytes) {
    const uint64_t need = p36_round_up_u64(bytes ? bytes : 4, 256);
    p36_vk_arena_block *block = NULL;
    uint64_t best_left = UINT64_MAX;
    while (p36_vk.arena_plan &&
           p36_vk.arena_plan->base->alloc_bytes - p36_vk.arena_plan->used < need) {
        p36_vk.arena_plan = NULL;
    }
    if (!p36_vk.arena_plan && p36_vk.arena_plan_next < p36_vk.arena_plan_count) {
        uint64_t block_bytes = p36_vk.arena_plan_sizes[p36_vk.arena_plan_next++];
        p36_gpu_tensor *base = p36_vk_tensor_alloc_kind_unlocked(block_bytes, true, false);
        p36_vk_arena_block *planned = base ? calloc(1, sizeof(*planned)) : NULL;
        if (planned) {
            base->weight_only = true;
            planned->base = base;
            planned->next = p36_vk.arena;
            p36_vk.arena = planned;
            p36_vk.arena_plan = planned;
        } else {
            p36_vk_tensor_free_unlocked(base);
            free(p36_vk.arena_plan_sizes);
            p36_vk.arena_plan_sizes = NULL;
            p36_vk.arena_plan_count = 0;
            p36_vk.arena_plan_next = 0;
        }
    }
    if (p36_vk.arena_plan) block = p36_vk.arena_plan;
    if (!block) {
        for (p36_vk_arena_block *b = p36_vk.arena; b; b = b->next) {
            uint64_t free_bytes = b->base->alloc_bytes - b->used;
            if (free_bytes >= need && free_bytes - need < best_left) {
                block = b;
                best_left = free_bytes - need;
            }
        }
    }
    if (!block) {
        uint64_t block_bytes = need > P36_VK_ARENA_BLOCK_BYTES ? need : P36_VK_ARENA_BLOCK_BYTES;
        p36_gpu_tensor *base = p36_vk_tensor_alloc_kind_unlocked(block_bytes, true, false);
        if (!base && block_bytes > need) base = p36_vk_tensor_alloc_kind_unlocked(need, true, false);
        if (!base) base = p36_vk_tensor_alloc_kind_unlocked(need, false, false);
        if (!base) return NULL;
        block = calloc(1, sizeof(*block));
        if (!block) {
            p36_vk_tensor_free_unlocked(base);
            return NULL;
        }
        base->weight_only = true;
        block->base = base;
        block->next = p36_vk.arena;
        p36_vk.arena = block;
    }
    {
        p36_gpu_tensor *view = p36_gpu_tensor_view(block->base, block->used, bytes);
        if (view) block->used += need;
        return view;
    }
}

/* Writes into the host-visible weight arena go to write-combined memory and
 * top out well below the SSD on a single thread, so bulk staging (prewarm)
 * splits the copy across the host pool.  Inference-time lazy staging keeps
 * the plain memcpy: it runs under p36_vk_mu in paths where the pool may be
 * busy with host compute. */
static bool p36_vk_weight_copy_parallel = false;

typedef struct {
    unsigned char *dst;
    const unsigned char *src;
    uint64_t bytes;
} p36_vk_weight_copy_ctx;

static void p36_vk_weight_copy_rows(void *ctx, uint64_t row0, uint64_t row1) {
    const p36_vk_weight_copy_ctx *c = ctx;
    uint64_t off0 = row0 << 20, off1 = row1 << 20;
    if (off1 > c->bytes) off1 = c->bytes;
    if (off0 < off1) memcpy(c->dst + off0, c->src + off0, (size_t)(off1 - off0));
}

static p36_gpu_tensor *p36_vk_weight_get_unlocked(const void *source, uint64_t bytes) {
    /* Move hits to the head: the cache is a linear list scanned per matvec
     * and a token touches the same few hundred weights in layer order. */
    for (p36_vk_weight *w = p36_vk.weights, *prev = NULL; w; prev = w, w = w->next) {
        if (w->source == source && w->bytes == bytes) {
            if (prev) {
                prev->next = w->next;
                w->next = p36_vk.weights;
                p36_vk.weights = w;
            }
            return w->tensor;
        }
    }

    const uint64_t limit = p36_vk_dense_weight_cache_limit_unlocked();
    bool pinned = false;
    if (limit) {
        const uint64_t pin_limit = p36_vk_dense_weight_pin_limit_unlocked(limit);
        const uint64_t pinned_bytes = p36_vk_weight_cache_pinned_bytes_unlocked();
        pinned = pinned_bytes <= pin_limit && bytes <= pin_limit - pinned_bytes;
        if (!pinned) {
            if (bytes > limit) return NULL;
            uint64_t cached = p36_vk_weight_cache_bytes_unlocked();
            if (cached > limit - bytes) {
                if (!p36_vk_flush_reason_unlocked("dense_weight_cache_evict")) return NULL;
                do {
                    if (!p36_vk_weight_cache_evict_one_unlocked()) return NULL;
                    cached = p36_vk_weight_cache_bytes_unlocked();
                } while (cached > limit - bytes);
            }
        }
    }

    /* Retained prefix weights use the arena's low-allocation path. Streamed
     * weights are individually reclaimable so the prefix survives eviction. */
    p36_gpu_tensor *tensor = pinned ? p36_vk_arena_alloc_unlocked(bytes) :
                                      (limit ? p36_vk_dense_stream_alloc_unlocked(bytes) :
                                               p36_vk_arena_alloc_unlocked(bytes));
    if (!tensor) return NULL;
    {
        uint64_t offset = 0;
        p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
        bool copy_parallel = p36_vk_weight_copy_parallel ||
                             (limit && bytes >= (1u << 20) &&
                              p36_vk_env_default_on("P36_VK_DENSE_WEIGHT_COPY_PARALLEL"));
        if (copy_parallel) {
            p36_vk_weight_copy_ctx c = { root->data + offset, source, bytes };
            p36_gpu_parallel_for_rows((bytes + ((1u << 20) - 1)) >> 20, 8, p36_vk_weight_copy_rows, &c);
        } else {
            memcpy(root->data + offset, source, (size_t)bytes);
        }
    }

    p36_vk_weight *w = calloc(1, sizeof(*w));
    if (!w) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    w->source = source;
    w->bytes = bytes;
    w->tensor = tensor;
    w->pinned = pinned;
    w->next = p36_vk.weights;
    p36_vk.weights = w;
    return tensor;
}

typedef enum {
    P36_VK_STREAM_PART_NONE = 0,
    P36_VK_STREAM_PART_GATE,
    P36_VK_STREAM_PART_UP,
    P36_VK_STREAM_PART_DOWN,
} p36_vk_stream_part;

static bool p36_vk_stream_table_valid(const p36_gpu_stream_expert_table *t) {
    return t && t->model_map && t->model_size != 0 &&
           t->n_total_expert != 0 && t->n_total_expert <= 256 &&
           t->gate_expert_bytes != 0 &&
           t->up_expert_bytes != 0 &&
           t->down_expert_bytes != 0;
}

static uint64_t p36_vk_stream_table_expert_bytes(const p36_gpu_stream_expert_table *t) {
    if (!t) return 0;
    if (t->gate_expert_bytes > UINT64_MAX - t->up_expert_bytes) return 0;
    uint64_t sum = t->gate_expert_bytes + t->up_expert_bytes;
    if (sum > UINT64_MAX - t->down_expert_bytes) return 0;
    return sum + t->down_expert_bytes;
}

static bool p36_vk_stream_layout_matches_unlocked(const p36_gpu_stream_expert_table *t) {
    if (p36_vk_stream.cap == 0) return true;
    return p36_vk_stream.gate_expert_bytes == t->gate_expert_bytes &&
           p36_vk_stream.up_expert_bytes == t->up_expert_bytes &&
           p36_vk_stream.down_expert_bytes == t->down_expert_bytes &&
           p36_vk_stream.gate_type == t->gate_type &&
           p36_vk_stream.up_type == t->up_type &&
           p36_vk_stream.down_type == t->down_type &&
           p36_vk_stream.has_gate_scales == t->has_gate_scales &&
           p36_vk_stream.has_up_scales == t->has_up_scales &&
           p36_vk_stream.has_down_scales == t->has_down_scales;
}

static bool p36_vk_stream_cache_init_unlocked(const p36_gpu_stream_expert_table *t) {
    if (!p36_vk_stream.enabled || p36_vk_stream.budget == 0 || !p36_vk_stream_table_valid(t)) return false;
    uint64_t expert_bytes = p36_vk_stream_table_expert_bytes(t);
    if (expert_bytes == 0) return false;
    if (p36_vk_stream.configured_expert_bytes != 0 &&
        p36_vk_stream.configured_expert_bytes != expert_bytes) {
        return false;
    }
    if (p36_vk_stream.cap != 0) return p36_vk_stream_layout_matches_unlocked(t);

    uint32_t cap = p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
    while (cap != 0) {
        uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0, scale_bytes = 0;
        bool sizes_ok = p36_u64_mul_ok(cap, t->gate_expert_bytes, &gate_bytes) &&
                        p36_u64_mul_ok(cap, t->up_expert_bytes, &up_bytes) &&
                        p36_u64_mul_ok(cap, t->down_expert_bytes, &down_bytes) &&
                        p36_u64_mul_ok(cap, sizeof(float), &scale_bytes);
        if (sizes_ok) {
            p36_vk_stream.entry = calloc(cap, sizeof(p36_vk_stream.entry[0]));
            if (p36_vk_stream.entry)
                p36_vk_stream.gate = p36_vk_tensor_alloc_kind_unlocked(gate_bytes, true, true);
            if (p36_vk_stream.gate)
                p36_vk_stream.up = p36_vk_tensor_alloc_kind_unlocked(up_bytes, true, true);
            if (p36_vk_stream.up)
                p36_vk_stream.down = p36_vk_tensor_alloc_kind_unlocked(down_bytes, true, true);
            if (p36_vk_stream.down && t->has_gate_scales)
                p36_vk_stream.gate_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
            if (p36_vk_stream.down && (!t->has_gate_scales || p36_vk_stream.gate_scales) && t->has_up_scales)
                p36_vk_stream.up_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
            if (p36_vk_stream.down && (!t->has_gate_scales || p36_vk_stream.gate_scales) &&
                (!t->has_up_scales || p36_vk_stream.up_scales) && t->has_down_scales)
                p36_vk_stream.down_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
        }
        if (sizes_ok && p36_vk_stream.entry && p36_vk_stream.gate &&
            p36_vk_stream.up && p36_vk_stream.down &&
            (!t->has_gate_scales || p36_vk_stream.gate_scales) &&
            (!t->has_up_scales || p36_vk_stream.up_scales) &&
            (!t->has_down_scales || p36_vk_stream.down_scales)) {
            break;
        }

        p36_vk_stream_cache_release_unlocked(true);
        uint32_t next = p36_ssd_shrink_cache_experts(cap);
        p36_vk_stream.runtime_cap = next;
        p36_vk_stream.allocation_failures++;
        fprintf(stderr,
                "p36: Vulkan streaming expert cache allocation failed at %u experts / %.2f GiB\n",
                cap, (double)cap * (double)expert_bytes / 1073741824.0);
        if (next != 0) {
            fprintf(stderr,
                    "p36:   shrinking resident cache margin by %u experts / %.2f GiB; runtime cache cap now %u experts\n",
                    cap - next,
                    (double)(cap - next) * (double)expert_bytes / 1073741824.0,
                    next);
        }
        cap = next;
    }
    if (cap == 0) return false;

    p36_vk_stream.cap = cap;
    p36_vk_stream.gate_expert_bytes = t->gate_expert_bytes;
    p36_vk_stream.up_expert_bytes = t->up_expert_bytes;
    p36_vk_stream.down_expert_bytes = t->down_expert_bytes;
    p36_vk_stream.gate_type = t->gate_type;
    p36_vk_stream.up_type = t->up_type;
    p36_vk_stream.down_type = t->down_type;
    p36_vk_stream.has_gate_scales = t->has_gate_scales;
    p36_vk_stream.has_up_scales = t->has_up_scales;
    p36_vk_stream.has_down_scales = t->has_down_scales;
    {
        const char *env = getenv("P36_VK_STREAM_CACHE_INDEX");
        p36_vk_stream.indexed = !env || !env[0] || env[0] != '0';
    }
    return true;
}

static bool p36_vk_stream_cache_entry_matches(const p36_vk_stream_entry *e,
                                               const p36_gpu_stream_expert_table *t,
                                               uint32_t expert) {
    return e && e->valid && e->layer == t->layer && e->expert == expert &&
           e->gate_offset == t->gate_offset &&
           e->up_offset == t->up_offset &&
           e->down_offset == t->down_offset;
}

static int p36_vk_stream_cache_find_unlocked(const p36_gpu_stream_expert_table *t, uint32_t expert) {
    if (p36_vk_stream.indexed && t->layer < P36_VK_STREAM_MAX_LAYERS &&
        expert < P36_VK_STREAM_MAX_EXPERTS) {
        uint32_t mapped = p36_vk_stream.slot_of[t->layer][expert];
        p36_vk_stream.lookup_steps++;
        if (mapped == 0) return -1;
        uint32_t slot = mapped - 1u;
        if (slot < p36_vk_stream.cap &&
            p36_vk_stream_cache_entry_matches(&p36_vk_stream.entry[slot], t, expert)) {
            return (int)slot;
        }
        p36_vk_stream.slot_of[t->layer][expert] = 0;
        return -1;
    }
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
        p36_vk_stream_entry *e = &p36_vk_stream.entry[i];
        p36_vk_stream.lookup_steps++;
        if (p36_vk_stream_cache_entry_matches(e, t, expert)) return (int)i;
    }
    return -1;
}

static void p36_vk_stream_cache_unmap_slot_unlocked(uint32_t slot) {
    if (!p36_vk_stream.indexed || slot >= p36_vk_stream.cap) return;
    p36_vk_stream_entry *e = &p36_vk_stream.entry[slot];
    if (!e->valid || e->layer >= P36_VK_STREAM_MAX_LAYERS ||
        e->expert >= P36_VK_STREAM_MAX_EXPERTS) return;
    uint32_t *mapped = &p36_vk_stream.slot_of[e->layer][e->expert];
    if (*mapped == slot + 1u) *mapped = 0;
}

static uint32_t p36_vk_stream_route_hotness_unlocked(uint32_t layer, uint32_t expert) {
    if (layer >= P36_VK_STREAM_MAX_LAYERS || expert >= P36_VK_STREAM_MAX_EXPERTS) return 0;
    uint32_t hotness = p36_vk_stream.route_hotness[layer][expert];
    uint32_t bias = p36_vk_stream.route_bias[layer][expert];
    return hotness > UINT32_MAX - bias ? UINT32_MAX : hotness + bias;
}

static void p36_vk_stream_note_hotness_unlocked(uint32_t layer, uint32_t expert,
                                                uint32_t amount) {
    if (layer >= P36_VK_STREAM_MAX_LAYERS || expert >= P36_VK_STREAM_MAX_EXPERTS || amount == 0) return;
    uint32_t *hotness = &p36_vk_stream.route_hotness[layer][expert];
    *hotness = *hotness > UINT32_MAX - amount ? UINT32_MAX : *hotness + amount;
}

static void p36_vk_stream_decay_hotness_unlocked(void) {
    while (p36_vk_stream.route_tokens - p36_vk_stream.hotness_decay_token >= 16u) {
        for (uint32_t layer = 0; layer < P36_VK_STREAM_MAX_LAYERS; layer++) {
            for (uint32_t expert = 0; expert < P36_VK_STREAM_MAX_EXPERTS; expert++) {
                p36_vk_stream.route_hotness[layer][expert] >>= 1;
            }
        }
        p36_vk_stream.hotness_decay_token += 16u;
    }
}

static bool p36_vk_stream_protect_selected(void) {
    return p36_vk_env_default_on("P36_VK_STREAM_PROTECT_SELECTED");
}

static bool p36_vk_stream_evict_past_layers_first(void) {
    const char *env = getenv("P36_VK_STREAM_EVICT_PAST_LAYERS_FIRST");
    if (env) return !env[0] || env[0] != '0';
    return p36_vk_stream.has_route_bias;
}

static bool p36_vk_stream_victim_better(uint32_t slot, uint32_t best) {
    uint32_t hotness = p36_vk_stream_route_hotness_unlocked(
        p36_vk_stream.entry[slot].layer, p36_vk_stream.entry[slot].expert);
    uint32_t best_hotness = best == UINT32_MAX ? UINT32_MAX :
        p36_vk_stream_route_hotness_unlocked(
            p36_vk_stream.entry[best].layer, p36_vk_stream.entry[best].expert);
    return best == UINT32_MAX || hotness < best_hotness ||
           (hotness == best_hotness &&
            p36_vk_stream.entry[slot].last_used <
            p36_vk_stream.entry[best].last_used);
}

static uint32_t p36_vk_stream_cache_pick_slot_unlocked(uint32_t protect_layer,
                                                       const bool *protect) {
    uint32_t best = UINT32_MAX;
    /* Prefill will not revisit past layers. Decode will, so it keeps the
     * global frequency/LRU policy. */
    bool prefer_past = p36_vk_stream.phase == 0 &&
                       p36_vk_stream_evict_past_layers_first();
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[i];
        if (entry->loading) continue;
        if (!entry->valid) return i;
        if (protect && entry->layer == protect_layer &&
            entry->expert < P36_VK_STREAM_MAX_EXPERTS &&
            protect[entry->expert]) {
            continue;
        }
        if (prefer_past && entry->layer > protect_layer) continue;
        if (p36_vk_stream_victim_better(i, best)) best = i;
    }
    if (best == UINT32_MAX && prefer_past) {
        for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
            p36_vk_stream_entry *entry = &p36_vk_stream.entry[i];
            if (entry->loading || !entry->valid) continue;
            if (protect && entry->layer == protect_layer &&
                entry->expert < P36_VK_STREAM_MAX_EXPERTS &&
                protect[entry->expert]) {
                continue;
            }
            if (p36_vk_stream_victim_better(i, best)) best = i;
        }
    }
    if (best != UINT32_MAX) p36_vk_stream.evictions++;
    return best;
}

static bool p36_vk_stream_cache_copy_scale_unlocked(p36_gpu_tensor *dst,
                                                    uint32_t slot,
                                                    const p36_gpu_stream_expert_table *t,
                                                    uint64_t scale_offset,
                                                    uint32_t expert) {
    if (!dst) return true;
    float value = 1.0f;
    if (!p36_vk_stream_copy_model_range(t->model_map,
                                        t->model_size,
                                        scale_offset + (uint64_t)expert * sizeof(float),
                                        &value,
                                        sizeof(value))) {
        return false;
    }
    memcpy(dst->data + (uint64_t)slot * sizeof(float), &value, sizeof(value));
    return true;
}

/* Copy one expert's gate/up/down (+scales) into its cache slot.  Pure data
 * movement into disjoint slot ranges, so the miss batch fans out across the
 * host pool while the prepare pass holds p36_vk_mu: parallel preads keep the
 * SSD queue full where one blocking read per expert leaves it idle. */
static bool p36_vk_stream_slot_fetch(uint32_t slot,
                                     const p36_gpu_stream_expert_table *t,
                                     uint32_t expert) {
    unsigned char *gate_dst = p36_vk_stream.gate->data + (uint64_t)slot * p36_vk_stream.gate_expert_bytes;
    unsigned char *up_dst = p36_vk_stream.up->data + (uint64_t)slot * p36_vk_stream.up_expert_bytes;
    unsigned char *down_dst = p36_vk_stream.down->data + (uint64_t)slot * p36_vk_stream.down_expert_bytes;
    return p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->gate_offset + (uint64_t)expert * t->gate_expert_bytes,
                                          gate_dst,
                                          t->gate_expert_bytes) &&
           p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->up_offset + (uint64_t)expert * t->up_expert_bytes,
                                          up_dst,
                                          t->up_expert_bytes) &&
           p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->down_offset + (uint64_t)expert * t->down_expert_bytes,
                                          down_dst,
                                          t->down_expert_bytes) &&
           (!t->has_gate_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.gate_scales, slot, t, t->gate_scales_offset, expert)) &&
           (!t->has_up_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.up_scales, slot, t, t->up_scales_offset, expert)) &&
           (!t->has_down_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.down_scales, slot, t, t->down_scales_offset, expert));
}

typedef struct {
    uint32_t slot;
    uint32_t expert;
    bool ok;
} p36_vk_stream_miss;

typedef struct {
    const p36_gpu_stream_expert_table *t;
    p36_vk_stream_miss *miss;
} p36_vk_stream_fetch_ctx;

static void p36_vk_stream_fetch_rows(void *ctx, uint64_t row0, uint64_t row1) {
    p36_vk_stream_fetch_ctx *c = ctx;
    for (uint64_t i = row0; i < row1; i++) {
        c->miss[i].ok = p36_vk_stream_slot_fetch(c->miss[i].slot, c->t, c->miss[i].expert);
    }
}

/* Hit-or-reserve: hits bump recency, misses claim a slot and record the
 * pending fetch.  The reserved entry carries its final identity immediately
 * so duplicate selections in the same list hit it. */
static bool p36_vk_stream_cache_prepare_one_unlocked(const p36_gpu_stream_expert_table *t,
                                                     uint32_t expert,
                                                     uint32_t priority,
                                                     const bool *protect,
                                                     uint32_t *slot_map,
                                                     p36_vk_stream_miss *miss,
                                                     uint32_t *n_miss,
                                                     bool *need_flush) {
    if (expert >= t->n_total_expert) return false;
    p36_vk_stream_note_hotness_unlocked(t->layer, expert, priority ? priority : 1u);
    int slot = p36_vk_stream_cache_find_unlocked(t, expert);
    if (slot >= 0) {
        p36_vk_stream.hits++;
    } else {
        if (*n_miss >= 256u) return false;
        p36_vk_stream.misses++;
        uint32_t pick = p36_vk_stream_cache_pick_slot_unlocked(t->layer, protect);
        if (pick == UINT32_MAX) return false;
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[pick];
        if (entry->valid) *need_flush = true;
        p36_vk_stream_cache_unmap_slot_unlocked(pick);
        entry->valid = true;
        entry->loading = true;
        entry->layer = t->layer;
        entry->expert = expert;
        entry->gate_offset = t->gate_offset;
        entry->up_offset = t->up_offset;
        entry->down_offset = t->down_offset;
        if (p36_vk_stream.indexed && t->layer < P36_VK_STREAM_MAX_LAYERS &&
            expert < P36_VK_STREAM_MAX_EXPERTS) {
            p36_vk_stream.slot_of[t->layer][expert] = pick + 1u;
        }
        miss[*n_miss] = (p36_vk_stream_miss) { pick, expert, false };
        (*n_miss)++;
        slot = (int)pick;
    }

    p36_vk_stream_entry *entry = &p36_vk_stream.entry[slot];
    entry->last_used = ++p36_vk_stream.clock;
    if (slot_map) slot_map[expert] = (uint32_t)slot;
    return true;
}

/* Flush once if any reserved slot evicts live data, then fetch the whole
 * miss batch in parallel and finalize the entries. */
static bool p36_vk_stream_cache_finish_unlocked(const p36_gpu_stream_expert_table *t,
                                                p36_vk_stream_miss *miss,
                                                uint32_t n_miss,
                                                bool need_flush,
                                                bool prepared) {
    bool ok = prepared;
    if (n_miss == 0) return ok;
    if (ok && need_flush && !p36_vk_flush_reason_unlocked("submit_wait_stream_cache_evict")) ok = false;
    if (ok) {
        p36_vk_stream_fetch_ctx ctx = { t, miss };
        p36_gpu_parallel_for_rows(n_miss, 2, p36_vk_stream_fetch_rows, &ctx);
    }
    for (uint32_t i = 0; i < n_miss; i++) {
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[miss[i].slot];
        entry->loading = false;
        if (ok && miss[i].ok) {
            p36_vk_stream.loads++;
        } else {
            p36_vk_stream_cache_unmap_slot_unlocked(miss[i].slot);
            entry->valid = false;
            ok = false;
        }
    }
    return ok;
}

static bool p36_vk_stream_cache_prepare_u32(const p36_gpu_stream_expert_table *t,
                                            const uint32_t *selected,
                                            uint64_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || !slot_map || n_selected == 0) return false;
    for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;

    bool seen[256] = {0};
    uint32_t unique = 0;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= 256) return false;
        if (!seen[expert]) {
            seen[expert] = true;
            unique++;
        }
    }
    pthread_mutex_lock(&p36_vk_mu);
    bool ok = p36_vk_stream_cache_init_unlocked(t);
    if (ok && unique > p36_vk_stream.cap) ok = false;
    if (ok) p36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    p36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    const bool *protect = p36_vk_stream_protect_selected() ? seen : NULL;
    for (uint64_t i = 0; ok && i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= t->n_total_expert) {
            ok = false;
            break;
        }
        ok = p36_vk_stream_cache_prepare_one_unlocked(t, expert, 1, protect, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = p36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static uint32_t p36_vk_stream_cache_cap(const p36_gpu_stream_expert_table *t) {
    pthread_mutex_lock(&p36_vk_mu);
    uint32_t cap = p36_vk_stream_cache_init_unlocked(t) ? p36_vk_stream.cap : 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return cap;
}

static void p36_vk_stream_note_routes(const p36_gpu_stream_expert_table *t,
                                      const uint32_t *selected,
                                      uint64_t n_selected) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    for (uint64_t i = 0; i < n_selected; i++) {
        if (selected[i] < t->n_total_expert) {
            p36_vk_stream_note_hotness_unlocked(t->layer, selected[i], 1u);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

static bool p36_vk_stream_cache_prepare_i32(const p36_gpu_stream_expert_table *t,
                                            const int32_t *selected,
                                            const uint32_t *priorities,
                                            uint32_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || n_selected == 0) return false;
    if (slot_map) for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;
    bool selected_set[256] = {0};
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= t->n_total_expert)
            return false;
        selected_set[(uint32_t)selected[i]] = true;
    }

    pthread_mutex_lock(&p36_vk_mu);
    bool ok = p36_vk_stream_cache_init_unlocked(t);
    p36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    const bool *protect = p36_vk_stream_protect_selected() ? selected_set : NULL;
    for (uint32_t i = 0; ok && i < n_selected; i++) {
        uint32_t priority = priorities ? priorities[i] : 1u;
        ok = p36_vk_stream_cache_prepare_one_unlocked(t, (uint32_t)selected[i], priority,
                                                      protect, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = p36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static const p36_gpu_tensor *p36_vk_stream_weight_tensor_unlocked(p36_vk_stream_part part) {
    if (p36_vk_stream.cap == 0) return NULL;
    if (part == P36_VK_STREAM_PART_GATE) return p36_vk_stream.gate;
    if (part == P36_VK_STREAM_PART_UP) return p36_vk_stream.up;
    if (part == P36_VK_STREAM_PART_DOWN) return p36_vk_stream.down;
    return NULL;
}

static const p36_gpu_tensor *p36_vk_stream_scale_tensor_unlocked(p36_vk_stream_part part) {
    if (p36_vk_stream.cap == 0) return NULL;
    if (part == P36_VK_STREAM_PART_GATE) return p36_vk_stream.gate_scales;
    if (part == P36_VK_STREAM_PART_UP) return p36_vk_stream.up_scales;
    if (part == P36_VK_STREAM_PART_DOWN) return p36_vk_stream.down_scales;
    return NULL;
}

static int p36_vk_descriptor_for_tensor(const p36_gpu_tensor *tensor, VkDescriptorBufferInfo *info) {
    uint64_t root_offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &root_offset);
    if (!root || !root->buffer) return 0;
    uint64_t range = tensor->bytes ? p36_round_up_u64(tensor->bytes, 4) : 1;
    if (root_offset > root->alloc_bytes) return 0;
    if (range > root->alloc_bytes - root_offset) range = root->alloc_bytes - root_offset;
    if (range == 0) range = 1;
    *info = (VkDescriptorBufferInfo){
        .buffer = root->buffer,
        .offset = root_offset,
        .range = range,
    };
    return 1;
}

static int p36_vk_alloc_descriptor_unlocked(p36_vk_kernel *k, VkDescriptorSet *set) {
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p36_vk.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->set_layout,
    };
    VkResult rc = vkAllocateDescriptorSets(p36_vk.device, &ai, set);
    if (rc == VK_SUCCESS) return 1;
    if (rc == VK_ERROR_DEVICE_LOST) return p36_vk_ok(rc, "vkAllocateDescriptorSets");
    /* The pool's live sets belong to the open batch; rotate to the next
     * ring slot (its pool was reset when its fence completed). */
    if (!p36_vk_submit_eager_unlocked()) return 0;
    ai.descriptorPool = p36_vk.descriptor_pool;
    rc = vkAllocateDescriptorSets(p36_vk.device, &ai, set);
    if (rc == VK_ERROR_DEVICE_LOST) return p36_vk_ok(rc, "vkAllocateDescriptorSets");
    return rc == VK_SUCCESS;
}

static int p36_vk_run_unlocked(
        const char                 *op,
        p36_vk_kernel              *kernel,
        const p36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z) {
    if (p36_vk_device_lost) return 0;
    const char *prof_op = p36_vk_prof_op_name(op);
    uint64_t prof_t0 = p36_vk_now_ns();
    uint64_t groups_total = (uint64_t)groups_x * groups_y * groups_z;
    if (!p36_vk_kernel_init(kernel)) return 0;
    if (push_bytes != kernel->push_bytes) return 0;
    if (p36_vk.prof_kernel && p36_vk.query_pool && p36_vk_prof_query_count + 2u > P36_VK_PROF_QUERY_CAP) {
        if (!p36_vk_flush_reason_unlocked("submit_wait_query_pool")) return 0;
    }

    VkDescriptorBufferInfo infos[16];
    VkWriteDescriptorSet writes[16];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        if (!p36_vk_descriptor_for_tensor(bindings[i], &infos[i])) return 0;
    }

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!p36_vk_alloc_descriptor_unlocked(kernel, &set)) return 0;
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i],
        };
    }
    vkUpdateDescriptorSets(p36_vk.device, kernel->bindings, writes, 0, NULL);

    /* Record into the open batch; submission happens at the next host read
     * (p36_vk_flush_unlocked).  One reusable command buffer and fence are
     * enough because the flush always waits before recording resumes. */
    VkCommandBuffer cmd;
    if (!p36_vk_begin_batch_unlocked(&cmd)) return 0;
    p36_gpu_tensor *roots[16];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i] = p36_gpu_tensor_root(bindings[i], NULL);
    }
    {
        /* Barrier only on a real hazard against work recorded since the
         * last barrier - including work in earlier eager-submitted buffers,
         * which executes in submission order and is still covered by a
         * barrier recorded here. Independent dispatches overlap. */
        bool need_barrier = p36_vk_hazard_all;
        for (uint32_t i = 0; !need_barrier && i < kernel->bindings; i++) {
            if (roots[i]->weight_only) continue;
            if (p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, roots[i])) need_barrier = true;
            else if (((kernel->write_mask >> i) & 1u) &&
                     p36_vk_hazard_in(p36_vk_hazard_read, p36_vk_hazard_read_n, roots[i])) need_barrier = true;
        }
        if (need_barrier) {
            VkMemoryBarrier compute_barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &compute_barrier, 0, NULL, 0, NULL);
            p36_vk_hazard_reset();
        }
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout, 0, 1, &set, 0, NULL);
    if (push_bytes) vkCmdPushConstants(cmd, kernel->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    uint32_t query0 = UINT32_MAX;
    if (p36_vk.prof_kernel && p36_vk.query_pool && p36_vk_prof_query_count + 2u <= P36_VK_PROF_QUERY_CAP) {
        query0 = p36_vk_prof_query_count;
        p36_vk_prof_query_count += 2u;
        p36_vk_prof_query_path[query0 / 2u] = kernel->path;
        p36_vk_prof_query_op[query0 / 2u] = prof_op;
        p36_vk_prof_query_groups[query0 / 2u] = groups_total;
        p36_vk_prof_query_host_ns[query0 / 2u] = p36_vk_now_ns();
        vkCmdResetQueryPool(cmd, p36_vk.query_pool, query0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, p36_vk.query_pool, query0);
    }
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    if (query0 != UINT32_MAX) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, p36_vk.query_pool, query0 + 1u);
    }
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i]->last_use_seq = p36_vk_submit_seq + 1;
        if ((kernel->write_mask >> i) & 1u) {
            roots[i]->gpu_written = true;
            roots[i]->write_gen++;
            p36_vk_hazard_add(p36_vk_hazard_written, &p36_vk_hazard_written_n, roots[i]);
            p36_vk_note_written_tensor((p36_gpu_tensor *)bindings[i]);
        } else {
            p36_vk_hazard_add(p36_vk_hazard_read, &p36_vk_hazard_read_n, roots[i]);
        }
    }
    p36_vk_prof_dispatches++;
    {
        uint64_t record_ns = p36_vk_now_ns() - prof_t0;
        p36_vk_prof_record_ns += record_ns;
        p36_vk_prof_op_add(prof_op, 1, 0, groups_total, record_ns, 0, 0, 0);
    }
    if (++p36_vk_batch_count >= P36_VK_BATCH_FLUSH_AT) return p36_vk_submit_eager_unlocked();
    return 1;
}

int p36_gpu_init(void) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk.ready) {
        pthread_mutex_unlock(&p36_vk_mu);
        return 1;
    }

    memset(&p36_vk, 0, sizeof(p36_vk));
    if (!getcwd(p36_vk.shader_root, sizeof(p36_vk.shader_root)))
        p36_vk.shader_root[0] = '\0';
    p36_vk.prof_ops = getenv("P36_VK_PROF_OP") || getenv("P36_VK_PROF") || getenv("P36_VK_PROF_KERNEL");
    {
        const char *env = getenv("P36_VK_FAULT_INJECT_LOST");
        p36_vk_fault_inject_lost = env ? strtoull(env, NULL, 10) : 0;
        p36_vk_submit_calls = 0;
    }
    /* write_mask marks the bindings each shader writes; the lazy-flush
     * tracker uses it so host reads of input-only tensors do not submit
     * the open batch. */
    p36_vk.matmul_f16 = P36_VK_KERNEL("vulkan/matmul_f16.spv", 3, 16, 1u << 2);
    p36_vk.matmul_bf16 = P36_VK_KERNEL("vulkan/matmul_bf16.spv", 3, 16, 1u << 2);
    p36_gdn_front = P36_VK_KERNEL("vulkan/gdn_front.spv", 12, 24, 1u | (15u << 8));
    p36_vk.vision_matmul_f16 = P36_VK_KERNEL("vulkan/vision_matmul_f16.spv", 3, 16, 1u << 2);
    p36_vk.vision_attention = P36_VK_KERNEL("vulkan/vision_attention.spv", 2, 4, 1u << 1);
    p36_vk.matmul_f32 = P36_VK_KERNEL("vulkan/matmul_f32.spv", 3, 16, 1u << 2);
    p36_vk.matmul_f32_fast = P36_VK_KERNEL("vulkan/matmul_f32_fast.spv", 3, 16, 1u << 2);
    p36_vk.matmul_f32_fast_w256 = P36_VK_KERNEL("vulkan/matmul_f32_fast_w256.spv", 3, 16, 1u << 2);
    p36_vk.add = P36_VK_KERNEL("vulkan/add.spv", 3, 4, 1u << 2);
    p36_vk.fwht = P36_VK_KERNEL("vulkan/fwht.spv", 2, 8, 1u << 1);
    p36_vk.signs_mul = P36_VK_KERNEL("vulkan/signs_mul.spv", 3, 8, 1u << 2);
    p36_vk.v_grouped_permute = P36_VK_KERNEL("vulkan/v_grouped_permute.spv", 2, 20, 1u << 1);
    p36_vk.hadamard_prepare = P36_VK_KERNEL("vulkan/hadamard_prepare.spv", 3, 24, 1u << 2);
    p36_vk.directional_steering = P36_VK_KERNEL("vulkan/directional_steering.spv", 2, 16, 1u << 0);
    p36_vk.rms_norm = P36_VK_KERNEL("vulkan/rms_norm.spv", 3, 12, 1u << 2);
    p36_vk.add_rms_norm = P36_VK_KERNEL("vulkan/add_rms_norm.spv", 5, 8, (1u << 3) | (1u << 4));
    p36_vk.add_rms_norm_q8_k = P36_VK_KERNEL("vulkan/add_rms_norm_q8_k.spv", 6, 12, (1u << 3) | (1u << 4) | (1u << 5));
    p36_vk.swiglu = P36_VK_KERNEL("vulkan/swiglu.spv", 3, 12, 1u << 2);
    p36_vk.swiglu_q8_k = P36_VK_KERNEL("vulkan/swiglu_q8_k.spv", 4, 20,
                                        (1u << 2) | (1u << 3));
    p36_vk.rope = P36_VK_KERNEL("vulkan/rope_qwen.spv", 1, 16, 1u << 0);
    p36_vk.rope_mrope = P36_VK_KERNEL("vulkan/rope_qwen_mrope.spv", 2, 12, 1u << 0);
    p36_vk.rms_norm_rope = P36_VK_KERNEL("vulkan/rms_norm_rope_qwen.spv", 3, 24, 1u << 2);
    p36_vk.rms_norm_rope_kv = P36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen.spv", 5, 24,
                                            (1u << 2) | (1u << 4));
    p36_vk.top2 = P36_VK_KERNEL("vulkan/top2.spv", 3, 12, (1u << 1) | (1u << 2));
    p36_vk.topk8 = P36_VK_KERNEL("vulkan/topk8.spv", 3, 12, (1u << 1) | (1u << 2));
    p36_vk.logits_minp = P36_VK_KERNEL("vulkan/logits_minp_pack.spv", 2, 24, (1u << 1));
    p36_vk.copy_rows = P36_VK_KERNEL("vulkan/copy_rows.spv", 2, 20, 1u << 1);
    p36_vk.recur_window = P36_VK_KERNEL("vulkan/recur_window.spv", 3, 12, (1u << 0) | (1u << 2));
    p36_vk.conv_silu = P36_VK_KERNEL("vulkan/conv_silu.spv", 3, 12, 1u << 2);
    p36_vk.recur_conv_silu_decode = P36_VK_KERNEL("vulkan/recur_conv_silu_decode.spv", 4, 12,
                                                   (1u << 0) | (1u << 3));
    p36_vk.recur_norm_gate = P36_VK_KERNEL("vulkan/recur_norm_gate.spv", 3, 12, 1u << 0);
    p36_vk.recur_norm_gate_q8_k = P36_VK_KERNEL(
        "vulkan/recur_norm_gate_q8_k.spv", 4, 8, (1u << 0) | (1u << 3));
    p36_vk.delta_qk = P36_VK_KERNEL("vulkan/delta_qk.spv", 3, 24, (1u << 1) | (1u << 2));
    p36_vk.delta_qkv = P36_VK_KERNEL("vulkan/delta_qkv.spv", 4, 24,
                                     (1u << 1) | (1u << 2) | (1u << 3));
    p36_vk.delta_gates = P36_VK_KERNEL("vulkan/delta_gates.spv", 5, 8, 1u << 4);
    p36_vk.quantize_q8_0 = P36_VK_KERNEL("vulkan/quantize_q8_0.spv", 2, 12, 1u << 1);
    p36_vk.quantize_q8_k = P36_VK_KERNEL("vulkan/quantize_q8_k.spv", 2, 12, 1u << 1);
    p36_vk.matmul_q8_0 = P36_VK_KERNEL("vulkan/matmul_q8_0.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_p36 = P36_VK_KERNEL("vulkan/matmul_q8_0_p36.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b_pair = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b_pair.spv", 5, 20,
                                                 (1u << 3) | (1u << 4));
    p36_vk.shared_gate_up_decode = P36_VK_KERNEL("vulkan/shared_gate_up_decode.spv", 4, 16, 1u << 3);
    p36_vk.shared_down_tail_decode = P36_VK_KERNEL("vulkan/shared_down_tail_decode.spv", 4, 12, 1u << 3);
    p36_vk.matmul_q8_0_mm = P36_VK_KERNEL("vulkan/matmul_q8_0_mm.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_mm_f16 = P36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_mm_f16_out32 = P36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16_out32.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b_nx = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b_nx.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode = P36_VK_KERNEL("vulkan/matmul_q8_0_decode.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode_p36 = P36_VK_KERNEL("vulkan/matmul_q8_0_decode_p36.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode_b64 = P36_VK_KERNEL("vulkan/matmul_q8_0_decode_b64.spv", 3, 20, 1u << 2);
    p36_vk.matmul_kquant = P36_VK_KERNEL("vulkan/matmul_kquant.spv", 3, 28, 1u << 2);
    p36_vk.matmul_q5k_mmq = P36_VK_KERNEL("vulkan/matmul_q5k_mmq.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q6k_mmq = P36_VK_KERNEL("vulkan/matmul_q6k_mmq.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q5k_mmq_fast = P36_VK_KERNEL("vulkan/matmul_q5k_mmq_fast.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q6k_mmq_fast = P36_VK_KERNEL("vulkan/matmul_q6k_mmq_fast.spv", 3, 16, 1u << 2);
    p36_vk.delta_net = P36_VK_KERNEL("vulkan/delta_net.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_fast = P36_VK_KERNEL("vulkan/delta_net_fast.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_cols = P36_VK_KERNEL("vulkan/delta_net_cols.spv", 6, 16, (1u << 0) | (1u << 5));
    p36_vk.delta_net_cols_f16 = P36_VK_KERNEL("vulkan/delta_net_cols_f16.spv", 6, 16, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode = P36_VK_KERNEL("vulkan/delta_net_decode.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode_reg = P36_VK_KERNEL("vulkan/delta_net_decode_reg.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode_reg_f16 = P36_VK_KERNEL("vulkan/delta_net_decode_reg_f16.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.attn_scores = P36_VK_KERNEL("vulkan/attn_scores.spv", 3, 28, 1u << 2);
    p36_vk.attn_post = P36_VK_KERNEL("vulkan/attn_post.spv", 4, 28, (1u << 0) | (1u << 3));
    p36_vk.attn_reduce = P36_VK_KERNEL("vulkan/attn_reduce.spv", 4, 28, 1u << 3);
    p36_vk.attn_decode_fused = P36_VK_KERNEL("vulkan/attn_decode_fused.spv", 6, 40, 1u << 5);
    p36_vk.attn_decode_split = P36_VK_KERNEL("vulkan/attn_decode_split.spv", 5, 44, 1u << 4);
    p36_vk.attn_prefill_qtile = P36_VK_KERNEL("vulkan/attn_prefill_qtile.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_qtile2 = P36_VK_KERNEL("vulkan/attn_prefill_qtile2.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_qtile2_gqa6 = P36_VK_KERNEL("vulkan/attn_prefill_qtile2_gqa6.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_fa_gqa6 = P36_VK_KERNEL("vulkan/attn_prefill_fa_gqa6.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_fa_gqa8 = P36_VK_KERNEL("vulkan/attn_prefill_fa_gqa8.spv", 5, 48, 1u << 4);
    p36_vk.attn_combine = P36_VK_KERNEL("vulkan/attn_combine.spv", 4, 32, 1u << 3);
    p36_vk.moe_gate_up = P36_VK_KERNEL("vulkan/moe_gate_up.spv", 8, 28, 1u << 6);
    p36_vk.moe_gate_up_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_iq2s.spv", 8, 28, 1u << 6);
    p36_vk.router_topk = P36_VK_KERNEL("vulkan/router_topk.spv", 3, 16, (1u << 1) | (1u << 2));
    p36_vk.moe_tiles = P36_VK_KERNEL("vulkan/moe_tiles.spv", 3, 20, 1u << 1);
    p36_vk.kv_store = P36_VK_KERNEL("vulkan/kv_store.spv", 4, 16, (1u << 0) | (1u << 1));
    p36_vk.kv_store_quant = P36_VK_KERNEL("vulkan/kv_store_quant.spv", 2, 20, 1u << 0);
    p36_vk.rms_norm_rope_kv_quant = P36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen_quant.spv", 5, 32, (1u << 2) | (1u << 4));
    p36_vk.moe_gate_up_f32b = P36_VK_KERNEL("vulkan/moe_gate_up_f32b.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_f32b_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_f32b_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode = P36_VK_KERNEL("vulkan/moe_gate_up_decode.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq3xxs = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq3xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_q3k = P36_VK_KERNEL("vulkan/moe_gate_up_decode_q3k.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq2s_iq2xxs = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq2s_iq2xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_down_q2k_f32b = P36_VK_KERNEL("vulkan/moe_down_q2k_f32b.spv", 5, 24, 1u << 4);
    p36_vk.moe_down_q2k_f32b_iq2s = P36_VK_KERNEL("vulkan/moe_down_q2k_f32b_iq2s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_q2k_sum_decode = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode.spv", 6, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq2s = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq2s.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq3s = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq3s.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq3xxs = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq3xxs.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_q3k = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_q3k.spv", 6, 24, 1u << 5);
    p36_vk.moe_gate_up_q4k_f32b = P36_VK_KERNEL("vulkan/moe_gate_up_q4k_f32b.spv", 7, 24, 1u << 6);
    p36_vk.moe_down_q4k_sum_decode = P36_VK_KERNEL("vulkan/moe_down_q4k_sum_decode.spv", 6, 24, 1u << 5);
    p36_vk.moe_gate_up_gemm = P36_VK_KERNEL("vulkan/moe_gate_up_gemm.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq3xxs = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq3xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_q3k = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_q3k.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s_iq2xxs = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s_iq2xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_down_gemm = P36_VK_KERNEL("vulkan/moe_down_gemm.spv", 5, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq2s = P36_VK_KERNEL("vulkan/moe_down_gemm_iq2s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3s = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3xxs = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3xxs.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_q3k = P36_VK_KERNEL("vulkan/moe_down_gemm_q3k.spv", 5, 24, 1u << 4);
    p36_vk.moe_matvec = P36_VK_KERNEL("vulkan/moe_matvec.spv", 6, 36, 1u << 4);
    p36_vk.moe_matvec_fast = P36_VK_KERNEL("vulkan/moe_matvec_fast.spv", 6, 36, 1u << 4);
    p36_vk.dense_iq3_xxs_decode = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_xxs_decode_r4 = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode_r4.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_xxs_decode_nx = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode_nx.spv", 4, 20, 1u << 2);
    p36_vk.dense_iq3_xxs_mmq = P36_VK_KERNEL("vulkan/dense_iq3_xxs_mmq.spv", 5, 20, 1u << 2);
    /* Bindings: gate weights, up weights, gate out, up out, tables, b16. */
    p36_vk.dense_iq3_xxs_mmq_pair = P36_VK_KERNEL("vulkan/dense_iq3_xxs_mmq_pair.spv", 6, 20,
                                                  (1u << 2) | (1u << 3));
    p36_vk.dense_iq3_s_decode = P36_VK_KERNEL("vulkan/dense_iq3_s_decode.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_full = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_full.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_r4 = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_r4.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_r1 = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_r1.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_mmq = P36_VK_KERNEL("vulkan/dense_iq3_s_mmq.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq3_s_bm64_mmq = P36_VK_KERNEL("vulkan/dense_iq3_s_bm64_mmq.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq3_s_mmq_r4 = P36_VK_KERNEL("vulkan/dense_iq3_s_mmq_r4.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq4_xs = P36_VK_KERNEL("vulkan/dense_iq4_xs.spv", 3, 20, 1u << 2);
    p36_vk.dense_iq4_xs_decode = P36_VK_KERNEL("vulkan/dense_iq4_xs_decode.spv", 3, 16, 1u << 2);
    p36_vk.dense_iq4_xs_mmq = P36_VK_KERNEL("vulkan/dense_iq4_xs_mmq.spv", 3, 20, 1u << 2);
    p36_vk.dense_iq1_m = P36_VK_KERNEL("vulkan/dense_iq1_m.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode = P36_VK_KERNEL("vulkan/dense_extra_decode.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_q2_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_pq2_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_small_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_small_q2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_small_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_small_pq2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_decode_ptq1_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_ptq1_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_iq2s_pair = P36_VK_KERNEL(
        "vulkan/dense_extra_decode_iq2s_pair.spv", 6, 24, (1u << 4) | (1u << 5));
    p36_vk.dense_extra_decode_iq2s = P36_VK_KERNEL("vulkan/dense_extra_decode_iq2s.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_mmq = P36_VK_KERNEL("vulkan/dense_extra_mmq.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_q2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_pq2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_ptq1_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_ptq1_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_iq2s = P36_VK_KERNEL("vulkan/dense_extra_mmq_iq2s.spv", 4, 24, 1u << 2);
    p36_vk.dense_kquant_mmq = P36_VK_KERNEL("vulkan/dense_kquant_mmq.spv", 4, 28, 1u << 2);
    p36_vk.predequant_b16 = P36_VK_KERNEL("vulkan/predequant_b16.spv", 2, 4, 1u << 1);
    p36_vk.dense_kquant_decode = P36_VK_KERNEL("vulkan/dense_kquant_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q4k_decode = P36_VK_KERNEL("vulkan/dense_q4k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q5k_decode = P36_VK_KERNEL("vulkan/dense_q5k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q3k_decode = P36_VK_KERNEL("vulkan/dense_q3k_decode.spv", 3, 28, 1u << 2);
    p36_vk.moe_reduce = P36_VK_KERNEL("vulkan/moe_reduce.spv", 3, 8, 1u << 2);
    p36_vk.ffn_tail = P36_VK_KERNEL("vulkan/ffn_tail.spv", 3, 8, 1u << 0);

    p36_vk.api_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_instance_version =
        (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enumerate_instance_version) (void)enumerate_instance_version(&p36_vk.api_version);

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "p36",
        .applicationVersion = 1,
        .pEngineName = "p36_vulkan",
        .engineVersion = 1,
        .apiVersion = p36_vk.api_version,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &p36_vk.instance) != VK_SUCCESS) goto fail;

    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(p36_vk.instance, &ndev, NULL) != VK_SUCCESS || ndev == 0) goto fail;
    VkPhysicalDevice *devs = calloc(ndev, sizeof(devs[0]));
    if (!devs) goto fail;
    if (vkEnumeratePhysicalDevices(p36_vk.instance, &ndev, devs) != VK_SUCCESS) {
        free(devs);
        goto fail;
    }

    for (uint32_t di = 0; di < ndev && !p36_vk.physical; di++) {
        VkPhysicalDeviceProperties candidate_props;
        VkPhysicalDeviceFeatures candidate_features;
        vkGetPhysicalDeviceProperties(devs[di], &candidate_props);
        vkGetPhysicalDeviceFeatures(devs[di], &candidate_features);
        if (candidate_props.apiVersion < VK_API_VERSION_1_1 ||
            !candidate_features.shaderFloat64 || !candidate_features.shaderInt64) continue;
#ifdef P36_VULKAN_REQUIRE_BC250
        if (candidate_props.vendorID != 0x1002u || candidate_props.deviceID != 0x13feu) continue;
#endif
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, NULL);
        VkQueueFamilyProperties *qprops = calloc(nq, sizeof(qprops[0]));
        if (!qprops) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, qprops);
        uint32_t compute = UINT32_MAX;
        uint32_t ace = UINT32_MAX;
        for (uint32_t qi = 0; qi < nq; qi++) {
            if ((qprops[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            if (compute == UINT32_MAX) compute = qi;
            if ((qprops[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 &&
                ace == UINT32_MAX) ace = qi;
        }
        if (compute != UINT32_MAX) {
            const char *ace_env = getenv("P36_VK_ACE");
            bool bc250 = candidate_props.vendorID == 0x1002u &&
                         candidate_props.deviceID == 0x13feu;
            bool use_ace = bc250 && ace != UINT32_MAX &&
                           (!ace_env || !ace_env[0] || ace_env[0] != '0');
            uint32_t qi = use_ace ? ace : compute;
            p36_vk.physical = devs[di];
            p36_vk.queue_family = qi;
            p36_vk.queue_count = qprops[qi].queueCount;
            p36_vk.timestamp_valid_bits = qprops[qi].timestampValidBits;
            p36_vk.ace_queue = use_ace;
        }
        free(qprops);
    }
    free(devs);
    if (!p36_vk.physical) {
        fprintf(stderr, "p36: no compatible Vulkan compute device found\n");
        goto fail;
    }

    vkGetPhysicalDeviceProperties(p36_vk.physical, &p36_vk.props);
    if (p36_vk.api_version < VK_API_VERSION_1_1 ||
        p36_vk.props.apiVersion < VK_API_VERSION_1_1) {
        fprintf(stderr, "p36: Vulkan 1.1 or newer is required\n");
        goto fail;
    }
    VkPhysicalDeviceSubgroupProperties subgroup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    };
    if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
        p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
        subgroup.pNext = &subgroup_size_props;
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup,
    };
    vkGetPhysicalDeviceProperties2(p36_vk.physical, &props2);
    p36_vk.props = props2.properties;
    p36_vk.subgroup_size = subgroup.subgroupSize;
    p36_vk.subgroup_arithmetic =
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    p36_vk.subgroup_clustered =
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0;
    p36_vk.bc250 = p36_vk.props.vendorID == 0x1002u && p36_vk.props.deviceID == 0x13feu;
#ifdef P36_VULKAN_REQUIRE_BC250
    if (!p36_vk.bc250) {
        fprintf(stderr, "p36: this vulkan-bc250 build requires an AMD BC-250 (1002:13fe)\n");
        goto fail;
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(p36_vk.physical, &p36_vk.mem_props);

    uint64_t device_bytes = p36_vk_memory_total();
    if (device_bytes) {
        fprintf(stderr, "p36: Vulkan device %s, %.2f GiB%s\n",
                p36_vk.props.deviceName, (double)device_bytes / 1073741824.0,
                p36_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? " UMA" : " memory");
    } else {
        fprintf(stderr, "p36: Vulkan device %s\n", p36_vk.props.deviceName);
    }
    fprintf(stderr, "p36: Vulkan generic backend%s, subgroup %u\n",
            p36_vk.bc250 ? ", BC-250 fast path" : "", p36_vk.subgroup_size);
    if (p36_vk.bc250) {
        fprintf(stderr, "p36: Vulkan queue family %u: %s (%u queue%s exposed)\n",
                p36_vk.queue_family,
                p36_vk.ace_queue ? "ACE compute" : "graphics fallback",
                p36_vk.queue_count, p36_vk.queue_count == 1 ? "" : "s");
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = p36_vk.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    /* The matmul / attention / delta-net kernels accumulate in fp64 and
     * emulate correctly rounded f32 fma via int64 bit ops to stay bit-exact
     * against the CPU reference engine, so shaderFloat64 and shaderInt64 are
     * hard requirements (RADV exposes both on the BC-250 target). */
    VkPhysicalDevice16BitStorageFeatures storage16_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    };
    VkPhysicalDevice8BitStorageFeatures storage8_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    };
    VkPhysicalDeviceShaderFloat16Int8Features f16_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures int_dot_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    };
    if (p36_vk.api_version >= VK_API_VERSION_1_2 &&
        p36_vk.props.apiVersion >= VK_API_VERSION_1_2) {
        storage16_query.pNext = &storage8_query;
        storage8_query.pNext = &f16_query;
        if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
            p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
            f16_query.pNext = &int_dot_query;
        if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
            p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
            int_dot_query.pNext = &subgroup_size_query;
    }
    VkPhysicalDeviceFeatures2 feat2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &storage16_query,
    };
    vkGetPhysicalDeviceFeatures2(p36_vk.physical, &feat2);
    if (!feat2.features.shaderFloat64 || !feat2.features.shaderInt64) {
        fprintf(stderr, "p36: Vulkan shaderFloat64 and shaderInt64 are required\n");
        goto fail;
    }
    p36_vk.have_storage16 = storage16_query.storageBuffer16BitAccess != 0;
    p36_vk.have_storage8 = storage8_query.storageBuffer8BitAccess != 0;
    p36_vk.have_f16 = p36_vk.have_storage16 && f16_query.shaderFloat16;
    p36_vk.have_int_dot = int_dot_query.shaderIntegerDotProduct != 0;
    p36_vk.subgroup_size_control = subgroup_size_query.subgroupSizeControl &&
        subgroup_size_query.computeFullSubgroups &&
        subgroup_size_props.minSubgroupSize <= 32u &&
        subgroup_size_props.maxSubgroupSize >= 32u;
    VkPhysicalDeviceFeatures enabled = { .shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE };
    VkPhysicalDevice16BitStorageFeatures storage16_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .storageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDevice8BitStorageFeatures storage8_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        .storageBuffer8BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceShaderFloat16Int8Features f16_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16 = VK_TRUE,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures int_dot_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        .shaderIntegerDotProduct = VK_TRUE,
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .pEnabledFeatures = &enabled,
    };
    if (p36_vk.have_storage16) {
        dci.pNext = &storage16_enable;
        void **storage_tail = &storage16_enable.pNext;
        if (p36_vk.have_storage8) {
            storage16_enable.pNext = &storage8_enable;
            storage_tail = &storage8_enable.pNext;
        }
        if (p36_vk.have_f16) {
            *storage_tail = &f16_enable;
            if (p36_vk.have_int_dot) {
                f16_enable.pNext = &int_dot_enable;
                if (p36_vk.subgroup_size_control)
                    int_dot_enable.pNext = &subgroup_size_enable;
            } else if (p36_vk.subgroup_size_control) {
                f16_enable.pNext = &subgroup_size_enable;
            }
        } else if (p36_vk.have_int_dot) {
            *storage_tail = &int_dot_enable;
            if (p36_vk.subgroup_size_control)
                int_dot_enable.pNext = &subgroup_size_enable;
        } else if (p36_vk.subgroup_size_control) {
            *storage_tail = &subgroup_size_enable;
        }
    } else if (p36_vk.have_int_dot) {
        dci.pNext = &int_dot_enable;
        if (p36_vk.subgroup_size_control)
            int_dot_enable.pNext = &subgroup_size_enable;
    } else if (p36_vk.subgroup_size_control) {
        dci.pNext = &subgroup_size_enable;
    }
    if (vkCreateDevice(p36_vk.physical, &dci, NULL, &p36_vk.device) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(p36_vk.device, p36_vk.queue_family, 0, &p36_vk.queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = p36_vk.queue_family,
    };
    if (vkCreateCommandPool(p36_vk.device, &cpci, NULL, &p36_vk.command_pool) != VK_SUCCESS) goto fail;

    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16384,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (vkCreateDescriptorPool(p36_vk.device, &dpci, NULL, &p36_vk_ring[i].pool) != VK_SUCCESS) goto fail;
    }
    p36_vk_ring_activate_unlocked(0);

    if (getenv("P36_VK_PROF_KERNEL") && p36_vk.timestamp_valid_bits != 0) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = P36_VK_PROF_QUERY_CAP,
        };
        if (vkCreateQueryPool(p36_vk.device, &qpci, NULL, &p36_vk.query_pool) == VK_SUCCESS) {
            p36_vk.prof_kernel = true;
        }
    }

    p36_vk.ready = true;
    p36_vk_stream_profile_init();
    if (getenv("P36_VK_PROF") || getenv("P36_VK_PROF_KERNEL") || getenv("P36_VK_PROF_OP")) atexit(p36_vk_prof_report);
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;

fail:
    if (p36_vk.device) vkDeviceWaitIdle(p36_vk.device);
    if (p36_vk.query_pool) vkDestroyQueryPool(p36_vk.device, p36_vk.query_pool, NULL);
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pool) vkDestroyDescriptorPool(p36_vk.device, p36_vk_ring[i].pool, NULL);
        if (p36_vk_ring[i].fence) vkDestroyFence(p36_vk.device, p36_vk_ring[i].fence, NULL);
    }
    memset(p36_vk_ring, 0, sizeof(p36_vk_ring));
    p36_vk_ring_cur = 0;
    if (p36_vk.command_pool) vkDestroyCommandPool(p36_vk.device, p36_vk.command_pool, NULL);
    if (p36_vk.device) vkDestroyDevice(p36_vk.device, NULL);
    if (p36_vk.instance) vkDestroyInstance(p36_vk.instance, NULL);
    memset(&p36_vk, 0, sizeof(p36_vk));
    pthread_mutex_unlock(&p36_vk_mu);
    return 0;
}

static void p36_vk_prof_report(void) {
    fprintf(stderr,
            "p36: Vulkan prof dispatches=%" PRIu64 " record_ms=%.1f flushes=%" PRIu64 " submit_wait_ms=%.1f\n",
            p36_vk_prof_dispatches,
            (double)p36_vk_prof_record_ns / 1e6,
            p36_vk_prof_flushes,
            (double)p36_vk_prof_submit_ns / 1e6);
    if (p36_vk_prof_op_n != 0) {
        fprintf(stderr, "p36: Vulkan op profile\n");
        for (uint32_t i = 0; i < p36_vk_prof_op_n; i++) {
            const p36_vk_prof_op_row *r = &p36_vk_prof_op_rows[i];
            fprintf(stderr,
                    "p36:   op %-28s dispatches=%" PRIu64 " flushes=%" PRIu64 " groups=%" PRIu64
                    " record_ms=%.3f submit_wait_ms=%.3f gpu_ms=%.3f wall_ms=%.3f\n",
                    r->name,
                    r->dispatches,
                    r->flushes,
                    r->groups,
                    (double)r->record_ns / 1e6,
                    (double)r->submit_ns / 1e6,
                    (double)r->gpu_ns / 1e6,
                    (double)r->wall_ns / 1e6);
        }
    }
    if (p36_vk_prof_kernel_n != 0) {
        uint64_t total_ns = 0;
        for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) total_ns += p36_vk_prof_kernel_rows[i].gpu_ns;
        fprintf(stderr, "p36: Vulkan kernel GPU time total_ms=%.3f\n", (double)total_ns / 1e6);
        for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) {
            const p36_vk_prof_kernel_row *r = &p36_vk_prof_kernel_rows[i];
            const double pct = total_ns ? 100.0 * (double)r->gpu_ns / (double)total_ns : 0.0;
            fprintf(stderr,
                    "p36:   %-32s dispatches=%" PRIu64 " groups=%" PRIu64 " gpu_ms=%.3f pct=%.1f\n",
                    r->path,
                    r->dispatches,
                    r->groups,
                    (double)r->gpu_ns / 1e6,
                    pct);
        }
    }
    if (p36_vk_prof_timeline_fp) {
        fclose(p36_vk_prof_timeline_fp);
        p36_vk_prof_timeline_fp = NULL;
    }
}

void p36_gpu_cleanup(void) {
    pthread_mutex_lock(&p36_vk_mu);
    /* Teardown after device loss must not wait on a queue that will never
     * drain; the objects are still destroyable. */
    if (p36_vk.device && !p36_vk_device_lost) vkDeviceWaitIdle(p36_vk.device);
    p36_vk_stream_profile_report();
    p36_vk_batch_recording = false;
    p36_vk_batch_count = 0;
    p36_vk_prof_query_count = 0;
    p36_vk_batch_written_n = 0;
    p36_vk_batch_dirty_all = false;
    p36_vk_submit_seq = 0;
    p36_vk_completed_seq = 0;
    p36_vk_hazard_reset();
    p36_vk_b16_ring_destroy_unlocked();
    p36_vk_tensor_free_unlocked(p36_vk_vision_weight);
    p36_vk_vision_weight = NULL;
    while (p36_vk_retired_n) p36_vk_tensor_free_unlocked(p36_vk_retired[--p36_vk_retired_n]);
    while (p36_vk_pool_n) p36_vk_tensor_free_unlocked(p36_vk_pool[--p36_vk_pool_n]);
    while (p36_vk_private_pool_n)
        p36_vk_tensor_free_unlocked(p36_vk_private_pool[--p36_vk_private_pool_n]);
    p36_vk_weight_cache_free_unlocked();
    p36_vk_tensor_free_unlocked(p36_vk.iq_tables);
    p36_vk.iq_tables = NULL;
    p36_vk_tensor_free_unlocked(p36_vk.attn_part);
    p36_vk.attn_part = NULL;
    p36_vk.attn_part_bytes = 0;
    p36_vk_tensor_free_unlocked(p36_vk.top2_part);
    p36_vk.top2_part = NULL;
    p36_vk.top2_part_bytes = 0;
    p36_vk_tensor_free_unlocked(p36_vk.topk8_part);
    p36_vk.topk8_part = NULL;
    p36_vk.topk8_part_bytes = 0;
    p36_vk_kernel_destroy(&p36_vk.ffn_tail);
    p36_vk_kernel_destroy(&p36_vk.top2);
    p36_vk_kernel_destroy(&p36_vk.topk8);
    p36_vk_kernel_destroy(&p36_vk.logits_minp);
    p36_vk_kernel_destroy(&p36_gdn_front);
    p36_vk_kernel_destroy(&p36_vk.recur_norm_gate);
    p36_vk_kernel_destroy(&p36_vk.recur_norm_gate_q8_k);
    p36_vk_kernel_destroy(&p36_vk.moe_reduce);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_bm64_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq1_m);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_iq2s);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_small_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_small_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2s_pair);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs);
    p36_vk_kernel_destroy(&p36_vk.dense_kquant_mmq);
    p36_vk_kernel_destroy(&p36_vk.predequant_b16);
    p36_vk_kernel_destroy(&p36_vk.dense_kquant_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q4k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q5k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q3k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_r1);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_full);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_mmq_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_mmq_pair);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode_nx);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_matvec_fast);
    p36_vk_kernel_destroy(&p36_vk.moe_matvec);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_iq2s);
    p36_vk_kernel_destroy(&p36_vk.router_topk);
    p36_vk_kernel_destroy(&p36_vk.moe_tiles);
    p36_vk_kernel_destroy(&p36_vk.kv_store);
    p36_vk_kernel_destroy(&p36_vk.kv_store_quant);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope_kv_quant);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_f32b_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq2s_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_f32b_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq3s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_q4k_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q4k_sum_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq2s_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq3s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_q3k);
    p36_vk_kernel_destroy(&p36_vk.attn_decode_fused);
    p36_vk_kernel_destroy(&p36_vk.attn_decode_split);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile2);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile2_gqa6);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_fa_gqa6);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_fa_gqa8);
    p36_vk_kernel_destroy(&p36_vk.attn_combine);
    p36_vk_kernel_destroy(&p36_vk.attn_reduce);
    p36_vk_kernel_destroy(&p36_vk.attn_post);
    p36_vk_kernel_destroy(&p36_vk.attn_scores);
    p36_vk_kernel_destroy(&p36_vk.delta_net_cols);
    p36_vk_kernel_destroy(&p36_vk.delta_net_cols_f16);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode_reg);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode_reg_f16);
    p36_vk_kernel_destroy(&p36_vk.delta_net_fast);
    p36_vk_kernel_destroy(&p36_vk.delta_net);
    p36_vk_kernel_destroy(&p36_vk.delta_gates);
    p36_vk_kernel_destroy(&p36_vk.delta_qk);
    p36_vk_kernel_destroy(&p36_vk.delta_qkv);
    p36_vk_kernel_destroy(&p36_vk.recur_conv_silu_decode);
    p36_vk_kernel_destroy(&p36_vk.conv_silu);
    p36_vk_kernel_destroy(&p36_vk.recur_window);
    p36_vk_kernel_destroy(&p36_vk.copy_rows);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope_kv);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope);
    p36_vk_kernel_destroy(&p36_vk.rope_mrope);
    p36_vk_kernel_destroy(&p36_vk.rope);
    p36_vk_kernel_destroy(&p36_vk.swiglu);
    p36_vk_kernel_destroy(&p36_vk.swiglu_q8_k);
    p36_vk_kernel_destroy(&p36_vk.rms_norm);
    p36_vk_kernel_destroy(&p36_vk.add_rms_norm);
    p36_vk_kernel_destroy(&p36_vk.add_rms_norm_q8_k);
    p36_vk_kernel_destroy(&p36_vk.quantize_q8_0);
    p36_vk_kernel_destroy(&p36_vk.matmul_q6k_mmq_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_q5k_mmq_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_q6k_mmq);
    p36_vk_kernel_destroy(&p36_vk.matmul_q5k_mmq);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode_b64);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode_p36);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b_pair);
    p36_vk_kernel_destroy(&p36_vk.shared_down_tail_decode);
    p36_vk_kernel_destroy(&p36_vk.shared_gate_up_decode);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm_f16);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm_f16_out32);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b_nx);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_p36);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0);
    p36_vk_kernel_destroy(&p36_vk.matmul_kquant);
    p36_vk_kernel_destroy(&p36_vk.quantize_q8_k);
    p36_vk_kernel_destroy(&p36_vk.directional_steering);
    p36_vk_kernel_destroy(&p36_vk.add);
    p36_vk_kernel_destroy(&p36_vk.fwht);
    p36_vk_kernel_destroy(&p36_vk.signs_mul);
    p36_vk_kernel_destroy(&p36_vk.v_grouped_permute);
    p36_vk_kernel_destroy(&p36_vk.hadamard_prepare);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32_fast_w256);
    p36_vk_kernel_destroy(&p36_vk.matmul_f16);
    p36_vk_kernel_destroy(&p36_vk.matmul_bf16);
    p36_vk_kernel_destroy(&p36_vk.vision_matmul_f16);
    p36_vk_kernel_destroy(&p36_vk.vision_attention);
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pool) vkDestroyDescriptorPool(p36_vk.device, p36_vk_ring[i].pool, NULL);
        if (p36_vk_ring[i].fence) vkDestroyFence(p36_vk.device, p36_vk_ring[i].fence, NULL);
    }
    memset(p36_vk_ring, 0, sizeof(p36_vk_ring));
    p36_vk_ring_cur = 0;
    if (p36_vk.query_pool) vkDestroyQueryPool(p36_vk.device, p36_vk.query_pool, NULL);
    if (p36_vk.command_pool) vkDestroyCommandPool(p36_vk.device, p36_vk.command_pool, NULL);
    if (p36_vk.device) vkDestroyDevice(p36_vk.device, NULL);
    if (p36_vk.instance) vkDestroyInstance(p36_vk.instance, NULL);
    memset(&p36_vk, 0, sizeof(p36_vk));
    p36_gpu_model_map = NULL;
    p36_gpu_model_size = 0;
    p36_gpu_model_fd = -1;
    p36_gpu_quality = false;
    p36_gpu_commands_open = false;
    p36_vk_private_bytes = 0;
    p36_vk_private_peak = 0;
    p36_vk_moe_bank_cache = true;
    pthread_mutex_unlock(&p36_vk_mu);
    p36_vk_host_threads_shutdown();
}

static p36_gpu_tensor *p36_gpu_tensor_alloc_persistent(uint64_t bytes, bool clear) {
    if (!p36_gpu_init()) return NULL;
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tensor = NULL;
    const uint64_t alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;
    for (uint32_t i = 0; i < p36_vk_pool_n; i++) {
        if (p36_vk_pool[i]->alloc_bytes != alloc_bytes) continue;
        tensor = p36_vk_pool[i];
        p36_vk_pool[i] = p36_vk_pool[--p36_vk_pool_n];
        tensor->bytes = bytes;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        if (clear) memset(tensor->data, 0, (size_t)alloc_bytes);
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        break;
    }
    if (!tensor) tensor = p36_vk_tensor_alloc_kind_unlocked(bytes, false, clear);
    pthread_mutex_unlock(&p36_vk_mu);
    return tensor;
}

p36_gpu_tensor *p36_gpu_tensor_alloc(uint64_t bytes) {
    return p36_gpu_tensor_alloc_persistent(bytes, true);
}

p36_gpu_tensor *p36_gpu_tensor_alloc_uninitialized(uint64_t bytes) {
    return p36_gpu_tensor_alloc_persistent(bytes, false);
}

p36_gpu_tensor *p36_gpu_tensor_alloc_scratch(uint64_t bytes) {
    if (!p36_gpu_init()) return NULL;
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tensor = p36_vk_tensor_alloc_scratch_unlocked(bytes);
    pthread_mutex_unlock(&p36_vk_mu);
    return tensor;
}

p36_gpu_tensor *p36_gpu_tensor_view(const p36_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!p36_gpu_tensor_range_ok(base, offset, bytes)) return NULL;
    p36_gpu_tensor *view = calloc(1, sizeof(*view));
    if (!view) return NULL;
    view->bytes = bytes;
    view->offset = offset;
    view->owner = (p36_gpu_tensor *)base;
    return view;
}

void p36_gpu_tensor_free(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    pthread_mutex_lock(&p36_vk_mu);
    if (tensor->owner) {
        p36_vk_tensor_free_unlocked(tensor);
    } else {
        p36_vk_tensor_release_unlocked(tensor);
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

uint64_t p36_gpu_tensor_bytes(const p36_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

/* The lazy-submission sync point: any host access to tensor bytes lands
 * here first, so flushing the open batch makes every host-side op see
 * completed GPU results without explicit synchronization at call sites. */
void *p36_gpu_tensor_contents(p36_gpu_tensor *tensor) {
    uint64_t offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
    if (!root || !root->data || offset > root->bytes) return NULL;
    /* After device loss the bytes behind this pointer are whatever the
     * dead batch left there; refuse rather than hand back stale results. */
    if (p36_vk_device_lost) return NULL;
    bool flush = false;
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_gpu_busy_unlocked()) {
        if (root->gpu_written) {
            /* dirty_all only means the written set overflowed; a root no
             * dispatch ever wrote cannot have pending GPU writes. */
            flush = p36_vk_batch_dirty_all;
            for (uint32_t i = 0; !flush && i < p36_vk_batch_written_n; i++) {
                flush = p36_vk_batch_written[i] == root;
            }
        } else {
            /* Host-written staging the GPU only reads (embed stage, conv
             * window): the caller may overwrite it, so wait out a pending
             * read; with double-buffered stages this is almost never hit. */
            flush = root->last_use_seq > p36_vk_completed_seq;
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    if (flush) {
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_flush_reason_unlocked(p36_vk_tensor_contents_reason);
        pthread_mutex_unlock(&p36_vk_mu);
        if (!ok) return NULL;
    }
    return root->data + offset;
}

static void *p36_vk_tensor_contents_labeled(p36_gpu_tensor *tensor, const char *reason) {
    const char *prev = p36_vk_tensor_contents_reason;
    p36_vk_tensor_contents_reason = reason ? reason : "submit_wait_tensor_contents";
    void *p = p36_gpu_tensor_contents(tensor);
    p36_vk_tensor_contents_reason = prev;
    return p;
}

void *p36_gpu_tensor_contents_named(p36_gpu_tensor *tensor, const char *reason) {
    return p36_vk_tensor_contents_labeled(tensor, reason);
}

int p36_gpu_tensor_write(p36_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    unsigned char *dst = p36_vk_tensor_contents_labeled(tensor, "submit_wait_tensor_write");
    if (!dst || !data || !p36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(dst + offset, data, (size_t)bytes);
    return 1;
}

int p36_gpu_tensor_read(const p36_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    unsigned char *src = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)tensor, "submit_wait_tensor_read");
    if (!src || !data || !p36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(data, src + offset, (size_t)bytes);
    p36_vk_read_bytes_total += bytes;
    return 1;
}

uint64_t p36_gpu_read_bytes(void) { return p36_vk_read_bytes_total; }
uint64_t p36_gpu_read_flushes(void) { return p36_vk_flush_total; }

int p36_gpu_tensor_copy(p36_gpu_tensor *dst, uint64_t dst_offset,
                        const p36_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes) {
    if (!bytes) return 1;
    if (!p36_gpu_tensor_range_ok(dst, dst_offset, bytes) ||
        !p36_gpu_tensor_range_ok(src, src_offset, bytes)) return 0;
    if (p36_vk.ready && dst && src) {
        uint64_t root_dst_offset = 0;
        uint64_t root_src_offset = 0;
        p36_gpu_tensor *dst_root = p36_gpu_tensor_root(dst, &root_dst_offset);
        p36_gpu_tensor *src_root = p36_gpu_tensor_root(src, &root_src_offset);
        root_dst_offset += dst_offset;
        root_src_offset += src_offset;
        bool overlap = dst_root == src_root &&
                       root_dst_offset < root_src_offset + bytes &&
                       root_src_offset < root_dst_offset + bytes;
        if (dst_root && src_root && dst_root->buffer && src_root->buffer && !overlap) {
            pthread_mutex_lock(&p36_vk_mu);
            VkCommandBuffer cmd;
            int ok = p36_vk_begin_batch_unlocked(&cmd);
            if (ok) {
                bool need_barrier = p36_vk_hazard_all ||
                    p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, src_root) ||
                    p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, dst_root) ||
                    p36_vk_hazard_in(p36_vk_hazard_read, p36_vk_hazard_read_n, dst_root);
                if (need_barrier) {
                    VkMemoryBarrier barrier = {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    };
                    vkCmdPipelineBarrier(cmd,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &barrier, 0, NULL, 0, NULL);
                    p36_vk_hazard_reset();
                }
            }
            if (ok) {
                VkBufferCopy copy = {
                    .srcOffset = root_src_offset,
                    .dstOffset = root_dst_offset,
                    .size = bytes,
                };
                vkCmdCopyBuffer(cmd, src_root->buffer, dst_root->buffer, 1, &copy);
                src_root->last_use_seq = p36_vk_submit_seq + 1;
                dst_root->last_use_seq = p36_vk_submit_seq + 1;
                dst_root->gpu_written = true;
                p36_vk_hazard_add(p36_vk_hazard_read, &p36_vk_hazard_read_n, src_root);
                p36_vk_hazard_add(p36_vk_hazard_written, &p36_vk_hazard_written_n, dst_root);
                p36_vk_note_written_tensor(dst_root);
                p36_vk_prof_dispatches++;
                p36_vk_prof_op_add("tensor_copy", 1, 0, 1, 0, 0, 0, 0);
                if (++p36_vk_batch_count >= P36_VK_BATCH_FLUSH_AT) ok = p36_vk_submit_eager_unlocked();
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        }
    }
    unsigned char *dstp = p36_vk_tensor_contents_labeled(dst, "submit_wait_tensor_copy_dst");
    unsigned char *srcp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)src, "submit_wait_tensor_copy_src");
    if (!dstp || !srcp) return 0;
    memmove(dstp + dst_offset, srcp + src_offset, (size_t)bytes);
    return 1;
}

int p36_gpu_begin_commands(void) {
    p36_gpu_commands_open = true;
    return 1;
}

int p36_gpu_flush_commands(void) {
    return p36_gpu_commands_open ? 1 : 0;
}

int p36_gpu_end_commands(void) {
    if (!p36_gpu_commands_open) return 0;
    p36_gpu_commands_open = false;
    return 1;
}

int p36_gpu_synchronize(void) {
    if (!p36_vk.ready) return 1;
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_flush_reason_unlocked("submit_wait_explicit");
    pthread_mutex_unlock(&p36_vk_mu);
    if (!ok || p36_vk_device_lost) return 0;
    return p36_vk_ok(vkDeviceWaitIdle(p36_vk.device), "vkDeviceWaitIdle");
}

void p36_gpu_trim_scratch(void) {
    if (!p36_vk.ready || !p36_gpu_synchronize()) return;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_b16_ring_destroy_unlocked();
    p36_vk_retired_sweep_unlocked();
    p36_vk_pools_drain_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
}

int p36_gpu_device_lost(void) {
    pthread_mutex_lock(&p36_vk_mu);
    int lost = p36_vk_device_lost_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
    return lost;
}

int p36_gpu_device_is_gfx1013(void) {
    /* The BC-250 (1002:13fe) is the only board this backend detects, and
     * its GPU is GFX10.1/gfx1013.  p36_gpu_init() is idempotent and the
     * engine already calls it at open; calling it again keeps this correct
     * if that order ever changes, and returns 0 if the GPU is unavailable. */
    if (!p36_gpu_init()) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    int is_gfx1013 = p36_vk.bc250 ? 1 : 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return is_gfx1013;
}

int p36_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = model_map;
    p36_gpu_model_size = model_size;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_set_model_fd(int fd) {
    p36_gpu_model_fd = fd;
#if defined(POSIX_FADV_RANDOM)
    if (fd >= 0 && p36_vk_stream.enabled &&
        getenv("P36_VK_STREAM_READAHEAD") == NULL) {
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
#endif
    return fd >= 0;
}

int p36_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_offset > model_size || map_size > model_size - map_offset) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = (const unsigned char *)model_map + map_offset;
    p36_gpu_model_size = map_size;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || (!offsets && count) || (!sizes && count)) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] > model_size - offsets[i]) return 0;
    }
    if (!p36_vk_env_default_on("P36_VK_PLAN_ARENA"))
        return p36_gpu_set_model_map(model_map, model_size);
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = model_map;
    p36_gpu_model_size = model_size;
    (void)p36_vk_arena_plan_unlocked(sizes, count);
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    (void)label;
    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    int ok;
    if (!source) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_copy_parallel = true;
    ok = p36_vk_weight_get_unlocked(source, bytes) != NULL;
    p36_vk_weight_copy_parallel = false;
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_finish_model_cache(void) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk.arena_plan = NULL;
    free(p36_vk.arena_plan_sizes);
    p36_vk.arena_plan_sizes = NULL;
    p36_vk.arena_plan_count = 0;
    p36_vk.arena_plan_next = 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    (void)in_dim;
    (void)out_dim;
    return p36_gpu_cache_model_range(model_map, model_size, offset, bytes, label);
}

void p36_gpu_set_quality(bool quality) {
    p36_gpu_quality = quality;
}

static void p36_vk_prepare_dense_kernels(void) {
    /* RADV pipeline creation can take seconds.  Do it while the dense model
     * opens instead of charging the first prompt that reaches each shape. */
    p36_vk_kernel *kernels[] = {
        &p36_vk.dense_iq3_s_mmq,
        &p36_vk.dense_iq3_s_mmq_r4,
        &p36_vk.dense_iq3_xxs_mmq,
        &p36_vk.dense_iq3_xxs_mmq_pair,
        &p36_vk.dense_kquant_mmq,
        &p36_vk.predequant_b16,
        &p36_vk.dense_iq3_s_decode,
        &p36_vk.dense_iq3_s_decode_r1,
        &p36_vk.dense_iq3_s_decode_r4,
        &p36_vk.dense_iq3_xxs_decode_r4,
        &p36_vk.dense_q4k_decode,
        &p36_vk.dense_kquant_decode,
        &p36_vk.dense_extra_decode_iq2s,
        &p36_vk.dense_extra_mmq_iq2s,
        &p36_vk.dense_extra_decode_iq2s_pair,
        &p36_vk.dense_extra_mmq,
    };
    if (!p36_vk_env_default_on("P36_VK_PREPARE_DENSE")) return;
    pthread_mutex_lock(&p36_vk_mu);
    for (uint32_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++) {
        if (!p36_vk_kernel_init(kernels[i])) break;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_dense_model(bool dense) {
    p36_gpu_dense_model = dense;
    /* Prewarm regardless of model class: MoE trunk tensors reach the same
     * K-quant decode/mmq kernels (see p36_gpu_matmul_k_quant_q8_scaled_tensor)
     * and some MoE files carry non-expert IQ3_S tensors too, so skipping this
     * for dense == false just moves the multi-second RADV pipeline-compile
     * stall onto the first live decode/prefill token instead of model open. */
    p36_vk_prepare_dense_kernels();
}

void p36_gpu_set_ssd_streaming(bool enabled) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.enabled != enabled) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.enabled = enabled;
    }
    p36_vk_moe_bank_cache = !enabled;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_full_layers(uint32_t layers) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream.full_layers = layers;
    p36_vk_moe_bank_cache = !p36_vk_stream.enabled || layers != 0;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.budget != experts) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.budget = experts;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_stream.configured_expert_bytes != bytes) {
        p36_vk_stream.runtime_cap = 0;
        p36_vk_stream.allocation_failures = 0;
        p36_vk_stream_cache_release_unlocked(true);
        p36_vk_stream.configured_expert_bytes = bytes;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}
void p36_gpu_set_streaming_expert_cache_layout(
        uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    uint64_t total = 0;
    if (gate_bytes <= UINT64_MAX - up_bytes &&
        gate_bytes + up_bytes <= UINT64_MAX - down_bytes)
        total = gate_bytes + up_bytes + down_bytes;
    p36_gpu_set_streaming_expert_cache_expert_bytes(total);
}

uint64_t p36_gpu_recommended_working_set_size(void) {
    if (!p36_vk.ready) return 0;
    return p36_vk_memory_total();
}

uint32_t p36_gpu_stream_expert_cache_configured_count(void) {
    return p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
}

uint32_t p36_gpu_stream_expert_cache_current_count(void) {
    pthread_mutex_lock(&p36_vk_mu);
    uint32_t count = 0;
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) count += p36_vk_stream.entry[i].valid ? 1u : 0u;
    pthread_mutex_unlock(&p36_vk_mu);
    return count;
}

void p36_gpu_stream_expert_cache_reset_route_hotness(void) {
    pthread_mutex_lock(&p36_vk_mu);
    memset(p36_vk_stream.route_hotness, 0, sizeof(p36_vk_stream.route_hotness));
    p36_vk_stream.hotness_decay_token = p36_vk_stream.route_tokens;
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_stream_expert_cache_note_tokens(uint32_t n_tokens) {
    if (!p36_vk_stream.enabled || n_tokens == 0) return;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream.phase = n_tokens == 1u ? 1u : 0u;
    if (p36_vk_stream.route_tokens <= UINT64_MAX - n_tokens) {
        p36_vk_stream.route_tokens += n_tokens;
    } else {
        p36_vk_stream.route_tokens = UINT64_MAX;
    }
    p36_vk_stream_decay_hotness_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_stream_expert_cache_release_resident(void) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream_cache_release_unlocked(true);
    pthread_mutex_unlock(&p36_vk_mu);
}

uint32_t p36_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gate_expert_bytes, uint64_t up_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || up_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint64_t total = gate_expert_bytes + up_expert_bytes;
    if (total < gate_expert_bytes || total > UINT64_MAX - down_expert_bytes) return 0;
    total += down_expert_bytes;
    if (p36_vk_stream.configured_expert_bytes != 0 && p36_vk_stream.configured_expert_bytes != total) return 0;
    return p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
}

int p36_gpu_stream_expert_cache_seed_selected(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return p36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, n_selected, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_begin_selected_load(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return p36_gpu_stream_expert_cache_seed_selected(table, selected_ids, n_selected);
}

int p36_gpu_stream_expert_cache_prepare_selected_batch(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected) {
    uint64_t n = (uint64_t)n_tokens * n_selected;
    if (n > UINT32_MAX) return 0;
    return p36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_load_layer(const p36_gpu_stream_expert_table *table) {
    if (!table || table->n_total_expert == 0) return 0;
    uint32_t n = table->n_total_expert;
    uint32_t budget = p36_gpu_stream_expert_cache_configured_count();
    if (budget != 0 && n > budget) n = budget;
    int32_t *ids = malloc((size_t)n * sizeof(ids[0]));
    if (!ids) return 0;
    for (uint32_t i = 0; i < n; i++) ids[i] = (int32_t)i;
    int ok = p36_vk_stream_cache_prepare_i32(table, ids, NULL, n, NULL) ? 1 : 0;
    free(ids);
    return ok;
}

int p36_gpu_stream_expert_cache_seed_from_layer_selected(const p36_gpu_stream_expert_table *table, const p36_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected) {
    if (!selected || n_selected == 0) return 0;
    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    uint64_t n = (uint64_t)n_seed_tokens * n_selected;
    if (n > UINT32_MAX || !p36_gpu_tensor_range_ok(selected, 0, n * sizeof(int32_t))) return 0;
    const int32_t *ids = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_stream_seed_selected");
    if (!ids) return 0;
    return p36_vk_stream_cache_prepare_i32(table, ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_release_layer_cache(void) {
    return 1;
}

int p36_gpu_stream_expert_cache_seed_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts) {
    return p36_vk_stream_cache_prepare_i32(table, expert_ids, expert_priorities, n_experts, NULL) ? 1 : 0;
}

int p36_gpu_stream_expert_cache_bias_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts) {
    if (!table || !expert_ids || table->layer >= P36_VK_STREAM_MAX_LAYERS) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    for (uint32_t i = 0; i < n_experts; i++) {
        int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert ||
            (uint32_t)expert >= P36_VK_STREAM_MAX_EXPERTS) {
            pthread_mutex_unlock(&p36_vk_mu);
            return 0;
        }
        uint32_t priority = expert_priorities ? expert_priorities[i] : 1u;
        uint32_t *bias = &p36_vk_stream.route_bias[table->layer][expert];
        if (priority > *bias) {
            *bias = priority;
            p36_vk_stream.has_route_bias = true;
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

void p36_gpu_print_memory_report(const char *label) {
    uint64_t arena_reserved = 0, arena_used = 0, arena_max_slack = 0;
    uint32_t arena_blocks = 0;
    for (p36_vk_arena_block *b = p36_vk.arena; b; b = b->next) {
        arena_reserved += b->base->alloc_bytes;
        arena_used += b->used;
        if (b->base->alloc_bytes - b->used > arena_max_slack)
            arena_max_slack = b->base->alloc_bytes - b->used;
        arena_blocks++;
    }
    fprintf(stderr,
            "p36: Vulkan memory%s%s live=%" PRIu64 " peak=%" PRIu64
            " arena=%" PRIu64 "/%" PRIu64 " arena_blocks=%u arena_max_slack=%" PRIu64
            " scratch_private=%" PRIu64 " scratch_private_peak=%" PRIu64
            " scratch_private_limit=%" PRIu64
            " fd=%d quality=%s ssd_streaming=%s cache=%u/%u requested=%u allocation_failures=%u lookup=%s lookup_steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 " device=%s api=%u.%u.%u\n",
            label ? " " : "",
            label ? label : "",
            p36_gpu_live_bytes,
            p36_gpu_peak_bytes,
            arena_used,
            arena_reserved,
            arena_blocks,
            arena_max_slack,
            p36_vk_private_bytes,
            p36_vk_private_peak,
            (uint64_t)P36_VK_PRIVATE_LIMIT,
            p36_gpu_model_fd,
            p36_gpu_quality ? "high" : "fast",
            p36_vk_stream.enabled ? "on" : "off",
            p36_gpu_stream_expert_cache_current_count(),
            p36_vk_stream.cap ? p36_vk_stream.cap : p36_gpu_stream_expert_cache_configured_count(),
            p36_vk_stream.budget,
            p36_vk_stream.allocation_failures,
            p36_vk_stream.indexed ? "index" : "scan",
            p36_vk_stream.lookup_steps,
            p36_vk_stream.hits,
            p36_vk_stream.misses,
            p36_vk_stream.loads,
            p36_vk_stream.evictions,
            p36_vk.props.deviceName[0] ? p36_vk.props.deviceName : "uninitialized",
            VK_VERSION_MAJOR(p36_vk.api_version),
            VK_VERSION_MINOR(p36_vk.api_version),
            VK_VERSION_PATCH(p36_vk.api_version));
}

void p36_gpu_prof_reset(void) {
    pthread_mutex_lock(&p36_vk_mu);
    (void)p36_vk_flush_reason_unlocked("submit_wait_prof_reset");
    p36_vk_prof_reset();
    pthread_mutex_unlock(&p36_vk_mu);
}

void p36_gpu_prof_report(const char *label) {
    pthread_mutex_lock(&p36_vk_mu);
    (void)p36_vk_flush_reason_unlocked("submit_wait_prof_report");
    pthread_mutex_unlock(&p36_vk_mu);
    if (label && label[0]) fprintf(stderr, "p36: Vulkan prof label=%s\n", label);
    p36_vk_prof_report();
}

/* The f32 dense path carries two dispatches with very different costs and
 * reported them under one "dense_f32" profiler row, which hid what the time
 * was actually spent on: the MoE router gate [n_embd, n_expert] runs
 * n_expert workgroups, while the shared-expert gate [n_embd, 1] runs a
 * single workgroup and pays a full pipeline round-trip for 8 KB.  Tagging by
 * shape mirrors p36_vk_q8_0_op_name(); it is diagnostic naming only and
 * changes no dispatch.  Names are literals, so no allocation or thread
 * safety concerns, and the strings only materialise when profiling is on. */
static const char *p36_vk_f32_op_name(uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, bool fast) {
    if (n_tok == 1) {
        if (out_dim == 1) return fast ? "dense_f32f_d_1xN" : "dense_f32_d_1xN";
        if (in_dim == 2048 && out_dim == 256) return fast ? "dense_f32f_d_256x2048" : "dense_f32_d_256x2048";
        return fast ? "dense_f32f_d_other" : "dense_f32_d_other";
    }
    if (out_dim == 1) return fast ? "dense_f32f_p_1xN" : "dense_f32_p_1xN";
    if (in_dim == 2048 && out_dim == 256) return fast ? "dense_f32f_p_256x2048" : "dense_f32_p_256x2048";
    return fast ? "dense_f32f_p_other" : "dense_f32_p_other";
}

static int p36_vk_matmul_dense(p36_vk_kernel *kernel,
                               p36_gpu_tensor *out,
                               const void *model_map,
                               uint64_t model_size,
                               uint64_t weight_offset,
                               uint64_t weight_bytes,
                               uint64_t in_dim,
                               uint64_t out_dim,
                               const p36_gpu_tensor *x,
                               uint64_t n_tok,
                               float scale) {
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
    if (!weights) {
        pthread_mutex_unlock(&p36_vk_mu);
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
        float scale;
    } push = { (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, scale };
    const p36_gpu_tensor *bindings[3] = { weights, x, out };
    /* The bf16 shader assigns one cooperative workgroup to each row.  The
     * legacy f16/f32 shaders index rows by global id, so their trailing
     * groups simply exit. */
    const char *op = "dense_f16";
    if (kernel != &p36_vk.matmul_f16 && kernel != &p36_vk.matmul_bf16) {
        op = p36_vk.prof_ops
                 ? p36_vk_f32_op_name(in_dim, out_dim, n_tok,
                                      kernel == &p36_vk.matmul_f32_fast ||
                                      kernel == &p36_vk.matmul_f32_fast_w256)
                 : "dense_f32";
    }
    int ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                 (uint32_t)out_dim, (uint32_t)n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}


int p36_gpu_matmul_f16_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const p36_gpu_tensor *x,
                              uint64_t n_tok) {
    uint64_t elems = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    return p36_vk_matmul_dense(&p36_vk.matmul_f16, out, model_map, model_size, weight_offset,
                               elems * sizeof(uint16_t), in_dim, out_dim, x, n_tok, 1.0f);
}

int p36_gpu_matmul_bf16_scaled_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const p36_gpu_tensor *x,
                                      uint64_t n_tok,
                                      float scale) {
    uint64_t elems = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    return p36_vk_matmul_dense(&p36_vk.matmul_bf16, out, model_map, model_size, weight_offset,
                               elems * sizeof(uint16_t), in_dim, out_dim, x, n_tok, scale);
}

int p36_gpu_vision_stream_init(uint64_t max_weight_bytes) {
    if (p36_vk_vision_weight && p36_vk_vision_weight->bytes >= max_weight_bytes)
        return 1;
    p36_gpu_tensor_free(p36_vk_vision_weight);
    p36_vk_vision_weight = p36_gpu_tensor_alloc_uninitialized(max_weight_bytes);
    return p36_vk_vision_weight != NULL;
}

static unsigned char *p36_vk_vision_weight_begin(uint64_t in_dim,
                                                 uint64_t out_dim,
                                                 uint64_t n_tok) {
    uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    if (!p36_vk_vision_weight || weight_bytes > p36_vk_vision_weight->bytes ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX)
        return NULL;
    return p36_gpu_tensor_contents_named(
        p36_vk_vision_weight, "submit_wait_vision_weight_reuse");
}

static int p36_vk_vision_matmul_dispatch(p36_gpu_tensor *out,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok);

int p36_gpu_vision_matmul_f16_fill(p36_gpu_tensor *out,
                                   p36_gpu_vision_fill_fn fill, void *ctx,
                                   uint64_t in_dim, uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok) {
    if (!fill) return 0;
    unsigned char *dst = p36_vk_vision_weight_begin(in_dim, out_dim, n_tok);
    if (!dst) return 0;
    if (!fill(ctx, (uint16_t *)dst)) return 0;
    return p36_vk_vision_matmul_dispatch(out, in_dim, out_dim, x, n_tok);
}

int p36_gpu_vision_matmul_f16_disk(p36_gpu_tensor *out, int fd,
                                   uint64_t offset, uint64_t in_dim,
                                   uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok) {
    uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    unsigned char *dst = p36_vk_vision_weight_begin(in_dim, out_dim, n_tok);
    if (!dst) return 0;
    uint64_t done = 0;
    while (done < weight_bytes) {
        ssize_t n = pread(fd, dst + done, (size_t)(weight_bytes - done),
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        done += (uint64_t)n;
    }
    return p36_vk_vision_matmul_dispatch(out, in_dim, out_dim, x, n_tok);
}

static int p36_vk_vision_matmul_dispatch(p36_gpu_tensor *out,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok) {
    pthread_mutex_lock(&p36_vk_mu);
    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
        float scale;
    } push = {(uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, 1.0f};
    const p36_gpu_tensor *bindings[3] = {p36_vk_vision_weight, x, out};
    int ok = p36_vk_run_unlocked("vision_f16", &p36_vk.vision_matmul_f16,
                                 bindings, &push, sizeof(push),
                                 ((uint32_t)out_dim + 15u) / 16u,
                                 ((uint32_t)n_tok + 15u) / 16u, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_vision_attention(p36_gpu_tensor *out,
                             const p36_gpu_tensor *qkv,
                             uint32_t rows) {
    if (!out || !qkv || !rows) return 0;
    struct { uint32_t rows; } push = {rows};
    const p36_gpu_tensor *bindings[2] = {qkv, out};
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("vision_attention", &p36_vk.vision_attention,
                                 bindings, &push, sizeof(push), rows, 16u, 1u);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* The f32 tensors in this model are the small control projections (expert
 * router logits, delta-net alpha/beta, the shared-expert gate scalar):
 * out_dim is 1..256, far too little parallelism to feed the GPU, and the
 * CPU-exact serial fp64 chain is latency-bound there.  Running the same
 * loop as p36_tensor_matvec() on the host is faster than a dispatch and,
 * because the inputs are host-produced norms, costs no batch flush at
 * all -- the whole router stays off the GPU timeline. */
typedef struct {
    const float *w;
    const float *x;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    float scale;
} p36_vk_matmul_f32_ctx;

static void p36_vk_matmul_f32_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_matmul_f32_ctx *ctx = (p36_vk_matmul_f32_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *xt = ctx->x + tok * ctx->in_dim;
        float *ot = ctx->out + tok * ctx->out_dim;
        uint64_t row = 0;
        /* Four independent row chains per pass: each keeps the exact
         * sequential rounding of p36_tensor_matvec(), the interleave only
         * hides the fp64 add latency. */
        for (; row + 4 <= ctx->out_dim; row += 4) {
            const float *w0 = ctx->w + row * ctx->in_dim;
            const float *w1 = w0 + ctx->in_dim;
            const float *w2 = w1 + ctx->in_dim;
            const float *w3 = w2 + ctx->in_dim;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) {
                double xv = (double)xt[i];
                a0 += (double)w0[i] * xv;
                a1 += (double)w1[i] * xv;
                a2 += (double)w2[i] * xv;
                a3 += (double)w3[i] * xv;
            }
            ot[row] = (float)a0 * ctx->scale;
            ot[row + 1] = (float)a1 * ctx->scale;
            ot[row + 2] = (float)a2 * ctx->scale;
            ot[row + 3] = (float)a3 * ctx->scale;
        }
        for (; row < ctx->out_dim; row++) {
            const float *wr = ctx->w + row * ctx->in_dim;
            double acc = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xt[i];
            ot[row] = (float)acc * ctx->scale;
        }
    }
}

int p36_gpu_matmul_f32_scaled_tensor(p36_gpu_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const p36_gpu_tensor *x,
                                     uint64_t n_tok,
                                     float scale) {
    uint64_t elems = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!p36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    /* Fast mode keeps these small control matvecs on GPU: a host fallback
     * here would force a flush now that nothing else reads activations back
     * during decode. The vec4 f32 kernel is the normal path; --quality or
     * P36_VK_F32_FAST=0 retains the fp64 accumulation for exact routing. */
    /* The 256-thread variant is much faster but changes the router gate's
     * summation order, so it is opt-in until an eval clears it. */
    p36_vk_kernel *fast_kernel = &p36_vk.matmul_f32_fast;
    if (getenv("P36_VK_F32_FAST_WIDE") && getenv("P36_VK_F32_FAST_WIDE")[0] == '1')
        fast_kernel = &p36_vk.matmul_f32_fast_w256;
    if (!p36_gpu_quality &&
        p36_vk_matmul_dense(p36_vk_use_f32_fast() && (in_dim & 3u) == 0u
                                ? fast_kernel : &p36_vk.matmul_f32,
                            out, model_map, model_size, weight_offset,
                            elems * sizeof(float), in_dim, out_dim, x, n_tok, scale)) {
        return 1;
    }

    const float *w = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                         elems * sizeof(float));
    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_matmul_f32_x");
    float *out_host = p36_vk_tensor_contents_labeled(out, "submit_wait_matmul_f32_out");
    p36_vk_matmul_f32_ctx ctx;
    if (!w || !x_host || !out_host) return 0;

    ctx.w = w;
    ctx.x = x_host;
    ctx.out = out_host;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.scale = scale;
    if (n_tok > 1) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_matmul_f32_rows, &ctx);
    else p36_vk_matmul_f32_rows(&ctx, 0, n_tok);
    return 1;
}

int p36_gpu_matmul_f32_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const p36_gpu_tensor *x,
                              uint64_t n_tok) {
    return p36_gpu_matmul_f32_scaled_tensor(out, model_map, model_size, weight_offset,
                                            in_dim, out_dim, x, n_tok, 1.0f);
}

typedef struct {
    unsigned char *xq;
    const float *x;
    uint64_t in_dim;
    uint64_t row_bytes;
} p36_vk_quantize_q8_0_ctx;

static void p36_vk_quantize_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_quantize_q8_0_ctx *ctx = (p36_vk_quantize_q8_0_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        p36_quant_q8_0(ctx->x + t * ctx->in_dim,
                       ctx->xq + t * ctx->row_bytes,
                       (int64_t)ctx->in_dim);
    }
}

typedef struct {
    const unsigned char *w;
    const unsigned char *xq;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t n_tok;
    uint64_t row_bytes;
    float scale;
} p36_vk_matmul_q8_0_ctx;

static void p36_vk_matmul_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_matmul_q8_0_ctx *ctx = (p36_vk_matmul_q8_0_ctx *)opaque;

    for (uint64_t row = row0; row < row1; row++) {
        const void *wr = ctx->w + row * ctx->row_bytes;
        for (uint64_t tok = 0; tok < ctx->n_tok; tok++) {
            float acc = p36_quant_dot_q8_0(wr,
                                           ctx->xq + tok * ctx->row_bytes,
                                           (int)ctx->in_dim);
            ctx->out[tok * ctx->out_dim + row] = acc * ctx->scale;
        }
    }
}

static const char *p36_vk_q8_0_op_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * P36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_d_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_d_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_d_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_d_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_d_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_d_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_d_vocabx2048";
        return "dense_q8_0_d_other";
    }
    if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_p_2048x4096";
    if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_p_2048x512";
    if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_p_32x2048";
    if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_p_4096x2048";
    if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_p_512x2048";
    if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_p_8192x2048";
    if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_p_vocabx2048";
    return "dense_q8_0_p_other";
}

static const char *p36_vk_q8_0_quant_wait_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * P36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "submit_wait_q8_0_q_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "submit_wait_q8_0_q_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "submit_wait_q8_0_q_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "submit_wait_q8_0_q_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "submit_wait_q8_0_q_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "submit_wait_q8_0_q_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "submit_wait_q8_0_q_vocabx2048";
        return "submit_wait_q8_0_q_other";
    }
    return "submit_wait_q8_0_quant_x";
}

static int p36_vk_matmul_q8_0_gpu(p36_gpu_tensor *out,
                                  const unsigned char *w,
                                  uint64_t weight_bytes,
                                  const unsigned char *xq,
                                  uint64_t xq_bytes,
                                  uint64_t row_bytes,
                                  uint64_t out_dim,
                                  uint64_t n_tok,
                                  uint64_t blocks,
                                  float scale) {
    if (n_tok > 1u && !p36_vk_have_subgroups(true)) return 0;
    p36_gpu_tensor *xq_tensor = p36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;
    unsigned char *xq_host = p36_vk_tensor_contents_labeled(xq_tensor, "submit_wait_q8_0_stage_xq");
    if (!xq_host) {
        p36_gpu_tensor_free(xq_tensor);
        return 0;
    }
    memcpy(xq_host, xq, (size_t)xq_bytes);

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, xq_tensor, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            p36_vk_kernel *kernel = &p36_vk.matmul_q8_0_decode;
            if (blocks == 64u && row_bytes == 64u * 34u) kernel = &p36_vk.matmul_q8_0_decode_b64;
            ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(xq_tensor);
    return ok;
}

static bool p36_vk_use_q8_mm_f16(void) {
    return p36_vk.have_f16 && p36_vk.have_storage16 &&
           p36_vk_env_default_on("P36_VK_Q8_MM_F16");
}

static bool p36_vk_use_q8_mm_f16_out32(void) {
    return p36_vk_env_default_on("P36_VK_Q8_MM_F16_OUT32");
}

/* MTP verify/replay window: micro batches there must be bit-identical to
 * the decode matvec, not to the prefill GEMM, so speculative rows equal
 * plain greedy decode. Session-sync small extends must keep the GEMM or
 * warm resume would stop matching the chunked cold rebuild. */
static bool p36_vk_micro_batch;

void p36_gpu_set_micro_batch(bool enabled) {
    p36_vk_micro_batch = enabled;
}

/* Prefill GEMM against raw f32 activations: 64x64 shared-memory tiles,
 * weights dequantized once per k-block for all tokens. Default is the
 * packed-f16 kernel; P36_VK_Q8_MM_F16=0 keeps the f32 tiles. */
static int p36_vk_matmul_q8_0_mm(p36_gpu_tensor *out,
                                 const unsigned char *w,
                                 uint64_t weight_bytes,
                                 const p36_gpu_tensor *x,
                                 uint64_t row_bytes,
                                 uint64_t out_dim,
                                 uint64_t n_tok,
                                 uint64_t blocks,
                                 float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        int out32 = n_tok >= 32u && out_dim == 32u && p36_vk_use_q8_mm_f16() && p36_vk_use_q8_mm_f16_out32();
        int f16 = n_tok >= 32u && out_dim >= 64u && p36_vk_use_q8_mm_f16();
        p36_vk_kernel *kernel = out32 ? &p36_vk.matmul_q8_0_mm_f16_out32 :
                                f16 ? &p36_vk.matmul_q8_0_mm_f16 :
                                      &p36_vk.matmul_q8_0_mm;
        uint32_t gx = out32 ? 1u :
                      f16 ? (uint32_t)((out_dim + 127u) / 128u) :
                            (uint32_t)((out_dim + 63u) / 64u);
        uint32_t gy = (uint32_t)((n_tok + 63u) / 64u);
        ok = p36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Dense decode matvec against raw f32 activations: no activation
 * quantization, so no host read of x and no q8 staging upload. Numerics
 * differ from the CPU path (which dots q8-quantized activations), so this
 * stays out of --quality. */
static int p36_vk_matmul_q8_0_f32b(p36_gpu_tensor *out,
                                   const unsigned char *w,
                                   uint64_t weight_bytes,
                                   const p36_gpu_tensor *x,
                                   uint64_t row_bytes,
                                   uint64_t out_dim,
                                   uint64_t blocks,
                                   float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, 1u, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, 1) : "dense_q8_0";
        ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_f32b, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_q8_0_pair_scaled_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_a_dim,
                                           uint64_t out_b_dim,
                                           const p36_gpu_tensor *x,
                                           float scale_a,
                                           float scale_b) {
    uint64_t blocks, row_bytes, weight_a_bytes, weight_b_bytes;
    uint64_t in_bytes, out_a_bytes, out_b_bytes;
    if (p36_gpu_quality || !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        in_dim == 0 || (in_dim % P36_VK_QK8_0) != 0 ||
        out_b_dim == 0 || out_b_dim > out_a_dim ||
        in_dim > UINT32_MAX || out_a_dim > UINT32_MAX || out_b_dim > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_0), &row_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_a_dim, &weight_a_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_b_dim, &weight_b_bytes) ||
        !p36_u64_mul_ok(in_dim, sizeof(float), &in_bytes) ||
        !p36_u64_mul_ok(out_a_dim, sizeof(float), &out_a_bytes) ||
        !p36_u64_mul_ok(out_b_dim, sizeof(float), &out_b_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(x, 0, in_bytes) ||
        !p36_gpu_tensor_range_ok(out_a, 0, out_a_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_b_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_a_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_b_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_a;
        uint32_t out_b;
        uint32_t blocks;
        float scale_a;
        float scale_b;
    } push = { (uint32_t)out_a_dim, (uint32_t)out_b_dim, (uint32_t)blocks,
               scale_a, scale_b };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights_a = p36_vk_weight_get_unlocked(wa, weight_a_bytes);
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_b_bytes) : NULL;
    int ok = weights_a != NULL && weights_b != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { weights_a, weights_b, x, out_a, out_b };
        ok = p36_vk_run_unlocked("dense_q8_0_d_pair", &p36_vk.matmul_q8_0_f32b_pair,
                                 bindings, &push, sizeof(push),
                                 (uint32_t)out_a_dim, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Micro-batch decode matvec: 2..8 tokens accumulated in registers per
 * weight load. Weight traffic equals one matvec and each token's
 * accumulation order equals the single-token kernel, so MTP verify rows
 * are bit-identical to plain decode of the same tokens. */
static int p36_vk_matmul_q8_0_f32b_nx(p36_gpu_tensor *out,
                                      const unsigned char *w,
                                      uint64_t weight_bytes,
                                      const p36_gpu_tensor *x,
                                      uint64_t row_bytes,
                                      uint64_t out_dim,
                                      uint64_t n_tok,
                                      uint64_t blocks,
                                      float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_f32b_nx, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static int p36_vk_matmul_q8_0_gpu_quant(p36_gpu_tensor *out,
                                        const unsigned char *w,
                                        uint64_t weight_bytes,
                                        const p36_gpu_tensor *x,
                                        uint64_t row_bytes,
                                        uint64_t out_dim,
                                        uint64_t n_tok,
                                        uint64_t blocks,
                                        float scale) {
    uint64_t xq_bytes;
    if (!p36_u64_mul_ok(n_tok, blocks, &xq_bytes) ||
        !p36_u64_mul_ok(xq_bytes, 36u, &xq_bytes)) return 0;
    p36_gpu_tensor *xq_tensor = p36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } qpush = { (uint32_t)(blocks * P36_VK_QK8_0), (uint32_t)blocks, (uint32_t)n_tok };

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } mpush = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *qbindings[2] = { x, xq_tensor };
        ok = p36_vk_run_unlocked("q8_0_quant", &p36_vk.quantize_q8_0, qbindings, &qpush, sizeof(qpush),
                                 (uint32_t)blocks, (uint32_t)n_tok, 1);
    }
    if (ok) {
        const p36_gpu_tensor *mbindings[3] = { weights, xq_tensor, out };
        const char *op = p36_vk.prof_ops ? p36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_decode_p36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = p36_vk_run_unlocked(op, &p36_vk.matmul_q8_0_p36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(xq_tensor);
    return ok;
}

int p36_gpu_matmul_q8_0_scaled_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const p36_gpu_tensor *x,
                                      uint64_t n_tok,
                                      float scale) {
    uint64_t blocks;
    uint64_t row_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t xq_bytes = 0;
    unsigned char *xq;

    if (in_dim == 0 || n_tok == 0 || (in_dim % P36_VK_QK8_0) != 0 || in_dim > INT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_0), &row_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !p36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, n_tok, &xq_bytes) || xq_bytes > SIZE_MAX) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    const unsigned char *w = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    if (!p36_gpu_quality && n_tok == 1 && p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_f32b(out, w, weight_bytes, x, row_bytes, out_dim, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && p36_vk_micro_batch && n_tok >= 2 && n_tok <= 8 &&
        p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_f32b_nx(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && n_tok >= 2 && p36_vk_use_q8_0_f32b() && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_mm(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!p36_gpu_quality && n_tok > 1u && p36_vk_env_default_on("P36_VK_Q8_0_GPU_QUANT") && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_gpu_quant(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x,
                                                         p36_vk.prof_ops ? p36_vk_q8_0_quant_wait_name(out_dim, blocks, n_tok) :
                                                                          "submit_wait_q8_0_quant_x");
    if (!x_host) return 0;

    xq = malloc((size_t)xq_bytes);
    if (!xq) return 0;

    p36_vk_quantize_q8_0_ctx qctx = { xq, x_host, in_dim, row_bytes };
    if (n_tok * in_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 4, p36_vk_quantize_q8_0_rows, &qctx);
    else p36_vk_quantize_q8_0_rows(&qctx, 0, n_tok);

    if (!p36_gpu_quality && p36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        p36_vk_matmul_q8_0_gpu(out, w, weight_bytes, xq, xq_bytes, row_bytes, out_dim, n_tok, blocks, scale)) {
        free(xq);
        return 1;
    }

    float *out_host = p36_vk_tensor_contents_labeled(out, "submit_wait_q8_0_host_out");
    if (!out_host) {
        free(xq);
        return 0;
    }
    p36_vk_matmul_q8_0_ctx mctx = { w, xq, out_host, in_dim, out_dim, n_tok, row_bytes, scale };
    if (out_dim >= 64u) p36_gpu_parallel_for_rows(out_dim, 64, p36_vk_matmul_q8_0_rows, &mctx);
    else p36_vk_matmul_q8_0_rows(&mctx, 0, out_dim);
    free(xq);
    return 1;
}

int p36_gpu_shared_ffn_decode_tensor(p36_gpu_tensor *out,
                                      p36_gpu_tensor *mid,
                                      const p36_gpu_tensor *x,
                                      const p36_gpu_tensor *scalar,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      float gate_scale,
                                      float up_scale,
                                      float down_scale) {
    uint64_t in_blocks, mid_blocks, gate_bytes, down_bytes;
    if (p36_gpu_quality || !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        !out || !mid || !x || !scalar || in_dim == 0 || mid_dim == 0 || out_dim == 0 ||
        in_dim % P36_VK_QK8_0 != 0 || mid_dim % P36_VK_QK8_0 != 0) {
        return 0;
    }
    in_blocks = in_dim / P36_VK_QK8_0;
    mid_blocks = mid_dim / P36_VK_QK8_0;
    if (!p36_u64_mul_ok(in_blocks * sizeof(p36_vk_block_q8_0), mid_dim, &gate_bytes) ||
        !p36_u64_mul_ok(mid_blocks * sizeof(p36_vk_block_q8_0), out_dim, &down_bytes) ||
        !p36_gpu_tensor_range_ok(x, 0, (uint64_t)in_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(mid, 0, (uint64_t)mid_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(scalar, 0, sizeof(float)) ||
        !p36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * sizeof(float))) {
        return 0;
    }
    const unsigned char *gate = p36_gpu_weight_bytes(model_map, model_size, gate_offset, gate_bytes);
    const unsigned char *up = p36_gpu_weight_bytes(model_map, model_size, up_offset, gate_bytes);
    const unsigned char *down = p36_gpu_weight_bytes(model_map, model_size, down_offset, down_bytes);
    if (!gate || !up || !down) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *gate_tensor = p36_vk_weight_get_unlocked(gate, gate_bytes);
    p36_gpu_tensor *up_tensor = p36_vk_weight_get_unlocked(up, gate_bytes);
    p36_gpu_tensor *down_tensor = p36_vk_weight_get_unlocked(down, down_bytes);
    int ok = gate_tensor && up_tensor && down_tensor;
    if (ok) {
        struct {
            uint32_t mid_dim;
            uint32_t blocks;
            float gate_scale;
            float up_scale;
        } push = { mid_dim, (uint32_t)in_blocks, gate_scale, up_scale };
        const p36_gpu_tensor *bindings[4] = { gate_tensor, up_tensor, x, mid };
        ok = p36_vk_run_unlocked("shared_gate_up_decode", &p36_vk.shared_gate_up_decode,
                                 bindings, &push, sizeof(push), mid_dim, 1, 1);
    }
    if (ok) {
        struct {
            uint32_t out_dim;
            uint32_t blocks;
            float scale;
        } push = { out_dim, (uint32_t)mid_blocks, down_scale };
        const p36_gpu_tensor *bindings[4] = { down_tensor, mid, scalar, out };
        ok = p36_vk_run_unlocked("shared_down_tail_decode", &p36_vk.shared_down_tail_decode,
                                 bindings, &push, sizeof(push), out_dim, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static uint64_t p36_vk_kquant_row_bytes(uint32_t type) {
    switch (type) {
    case P36_VK_TENSOR_Q2_K: return 84;
    case P36_VK_TENSOR_Q3_K: return 110;
    case P36_VK_TENSOR_Q4_K: return 144;
    case P36_VK_TENSOR_Q5_K: return 176;
    case P36_VK_TENSOR_Q6_K: return 210;
    default: return 0;
    }
}

static void p36_vk_pack_q5_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            p36_vk_block_q5_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                uint8_t sc = 0;
                uint8_t mn = 0;
                if (row >= out_dim) continue;
                const p36_vk_block_q5_K_src *blk = (const p36_vk_block_q5_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                packed.dmin[r] = blk->dmin;
                for (uint32_t g = 0; g < 8u; g++) {
                    p36_vk_q5_scale_min(blk->scales, g, &sc, &mn);
                    packed.scales[g * 8u + r] = sc;
                    packed.mins[g * 8u + r] = mn;
                }
                for (uint32_t i = 0; i < P36_VK_QK_K / 8u; i++) packed.qh[i * 8u + r] = blk->qh[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 2u; i++) packed.qs[i * 8u + r] = blk->qs[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static void p36_vk_pack_q6_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            p36_vk_block_q6_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                if (row >= out_dim) continue;
                const p36_vk_block_q6_K_src *blk = (const p36_vk_block_q6_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                for (uint32_t i = 0; i < P36_VK_QK_K / 16u; i++) packed.scales[i * 8u + r] = blk->scales[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 2u; i++) packed.ql[i * 8u + r] = blk->ql[i];
                for (uint32_t i = 0; i < P36_VK_QK_K / 4u; i++) packed.qh[i * 8u + r] = blk->qh[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static p36_gpu_tensor *p36_vk_packed_weight_get_unlocked(const unsigned char *source,
                                                         uint64_t src_bytes,
                                                         uint32_t type,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim) {
    for (p36_vk_packed_weight *w = p36_vk.packed_weights; w; w = w->next) {
        if (w->source == source && w->src_bytes == src_bytes && w->type == type &&
            w->in_dim == in_dim && w->out_dim == out_dim) {
            return w->tensor;
        }
    }

    const uint32_t blocks = in_dim / P36_VK_QK_K;
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    const uint64_t pack_block_bytes = p36_vk_kquant_mmq_block_bytes(type);
    const uint64_t src_row_bytes = p36_vk_kquant_row_bytes(type) * (uint64_t)blocks;
    const uint64_t packed_bytes = (uint64_t)row_tiles * blocks * pack_block_bytes;
    if (pack_block_bytes == 0 || blocks == 0 || row_tiles == 0) return NULL;

    p36_gpu_tensor *tensor = p36_vk_arena_alloc_unlocked(packed_bytes);
    if (!tensor) return NULL;
    uint64_t offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
    if (!root) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    if (type == P36_VK_TENSOR_Q5_K) {
        p36_vk_pack_q5_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else if (type == P36_VK_TENSOR_Q6_K) {
        p36_vk_pack_q6_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    p36_vk_packed_weight *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    entry->source = source;
    entry->src_bytes = src_bytes;
    entry->type = type;
    entry->in_dim = in_dim;
    entry->out_dim = out_dim;
    entry->tensor = tensor;
    entry->next = p36_vk.packed_weights;
    p36_vk.packed_weights = entry;
    return tensor;
}

/* Expand a Q8_K activation panel once into packed f16 pairs (natural k
 * order, value = f16(d * q), the exact rounding the MMQ tile staging
 * applied).  Every 32-row output tile of a dense MMQ dispatch used to
 * re-dequantize the same panel; with the panel shared they stage the
 * token tile with plain vector copies.
 *
 * Panels live in a small persistent ring instead of the scratch pool:
 * a fresh scratch panel per matmul call sat on the retire list across
 * the whole command-buffer ring and pushed the 2k/4k peak up by half a
 * gigabyte.  Reusing a ring slot while an old batch still reads it is
 * safe because the write-after-read hazard inserts a barrier.  The
 * one-entry cache lets gate/up (same activation, same q8 write_gen)
 * share one panel and one expansion dispatch. */
enum { P36_VK_B16_RING = 6 };
static struct { p36_gpu_tensor *t; uint64_t bytes; } p36_vk_b16_ring[P36_VK_B16_RING];
static uint32_t p36_vk_b16_next;
static struct {
    const p36_gpu_tensor *root;
    uint64_t root_off;
    uint64_t write_gen;
    uint32_t n_tok;
    uint32_t blocks;
    p36_gpu_tensor *b16;
} p36_vk_b16_last;

static void p36_vk_b16_ring_destroy_unlocked(void) {
    for (uint32_t i = 0; i < P36_VK_B16_RING; i++) {
        if (p36_vk_b16_ring[i].t) p36_vk_tensor_release_unlocked(p36_vk_b16_ring[i].t);
        p36_vk_b16_ring[i].t = NULL;
        p36_vk_b16_ring[i].bytes = 0;
    }
    memset(&p36_vk_b16_last, 0, sizeof(p36_vk_b16_last));
    p36_vk_b16_next = 0;
}

static p36_gpu_tensor *p36_vk_predequant_b16_unlocked(const p36_gpu_tensor *q8,
                                                      uint32_t n_tok,
                                                      uint32_t blocks) {
    const uint64_t total = (uint64_t)n_tok * blocks;
    const uint64_t bytes = total * 512u;
    uint64_t root_off = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root((p36_gpu_tensor *)q8, &root_off);
    if (root && p36_vk_b16_last.b16 &&
        p36_vk_b16_last.root == root && p36_vk_b16_last.root_off == root_off &&
        p36_vk_b16_last.write_gen == root->write_gen &&
        p36_vk_b16_last.n_tok == n_tok && p36_vk_b16_last.blocks == blocks) {
        return p36_vk_b16_last.b16;
    }
    p36_gpu_tensor *b16 = NULL;
    for (uint32_t k = 0; k < P36_VK_B16_RING; k++) {
        uint32_t i = (p36_vk_b16_next + 1u + k) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t && p36_vk_b16_ring[i].bytes == bytes &&
            p36_vk_b16_ring[i].t != p36_vk_b16_last.b16) {
            b16 = p36_vk_b16_ring[i].t;
            p36_vk_b16_next = i;
            break;
        }
    }
    if (!b16) {
        uint32_t i = (p36_vk_b16_next + 1u) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t == p36_vk_b16_last.b16 && p36_vk_b16_ring[i].t)
            i = (i + 1u) % P36_VK_B16_RING;
        if (p36_vk_b16_ring[i].t) p36_vk_tensor_release_unlocked(p36_vk_b16_ring[i].t);
        p36_vk_b16_ring[i].t = p36_vk_tensor_alloc_scratch_unlocked(bytes);
        p36_vk_b16_ring[i].bytes = p36_vk_b16_ring[i].t ? bytes : 0;
        b16 = p36_vk_b16_ring[i].t;
        p36_vk_b16_next = i;
        if (!b16) return NULL;
    }
    struct { uint32_t total_blocks; } push = { (uint32_t)total };
    const p36_gpu_tensor *bindings[2] = { q8, b16 };
    if (!p36_vk_run_unlocked("predequant_b16", &p36_vk.predequant_b16,
                             bindings, &push, sizeof(push),
                             (uint32_t)((total * 64u + 255u) / 256u), 1, 1)) {
        return NULL;
    }
    p36_vk_b16_last.root = root;
    p36_vk_b16_last.root_off = root_off;
    p36_vk_b16_last.write_gen = root ? root->write_gen : 0;
    p36_vk_b16_last.n_tok = n_tok;
    p36_vk_b16_last.blocks = blocks;
    p36_vk_b16_last.b16 = b16;
    return b16;
}

static int p36_vk_matmul_kquant_mmq_unlocked(p36_gpu_tensor *out,
                                             const p36_gpu_tensor *weights,
                                             const p36_gpu_tensor *q8,
                                             uint32_t type,
                                             uint32_t out_dim,
                                             uint32_t n_tok,
                                             uint32_t blocks,
                                             float scale) {
    p36_vk_kernel *kernel = NULL;
    bool fast = p36_vk_use_mmq_fast();
    if (type == P36_VK_TENSOR_Q5_K) kernel = fast ? &p36_vk.matmul_q5k_mmq_fast : &p36_vk.matmul_q5k_mmq;
    else if (type == P36_VK_TENSOR_Q6_K) kernel = fast ? &p36_vk.matmul_q6k_mmq_fast : &p36_vk.matmul_q6k_mmq;
    else return 0;
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        float scale;
    } push = { out_dim, n_tok, blocks, scale };
    const p36_gpu_tensor *bindings[3] = { weights, q8, out };
    return p36_vk_run_unlocked("dense_kquant_mmq",
                               kernel,
                               bindings,
                               &push,
                               sizeof(push),
                               (out_dim + 7u) / 8u,
                               (n_tok + 3u) / 4u,
                               1);
}

int p36_gpu_matmul_k_quant_q8_scaled_tensor(p36_gpu_tensor *out,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t weight_offset,
                                            uint32_t weight_type,
                                            uint64_t in_dim,
                                            uint64_t out_dim,
                                            const p36_gpu_tensor *q8,
                                            uint64_t n_tok,
                                            float scale) {
    if (in_dim == 0 || (in_dim % 256u) != 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t block_bytes = p36_vk_kquant_row_bytes(weight_type);
    if (block_bytes == 0) return 0;
    const uint64_t blocks = in_dim / P36_VK_QK_K;
    const uint64_t row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX) return 0;

    uint64_t out_bytes = 0;
    uint64_t q8_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!p36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!p36_u64_mul_ok(n_tok, blocks, &q8_bytes) || !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) || !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;

    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    /* Named for the dense-model launch scripts that first exercised it, but
     * the kernel itself only reads this one tensor's own dims/type/scale --
     * nothing here depends on the rest of the model being dense.  MoE models
     * carry K-quant non-expert tensors too (e.g. attn_qkv/output.weight on
     * Qwen3.8-35B-A3B-IQ2_M), and those hit the AVX2-style 8x8 generic
     * matmul_kquant fallback below unconditionally because this gate was
     * never widened past p36_gpu_dense_model. */
    if (!p36_gpu_quality && n_tok == 1u &&
        p36_vk_env_default_on("P36_VK_DENSE_KQUANT_DECODE")) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        int ok = weights != NULL;
        if (ok) {
            struct {
                uint32_t in_dim;
                uint32_t out_dim;
                uint32_t n_tok;
                uint32_t blocks;
                uint32_t row_bytes;
                uint32_t type;
                float scale;
            } push = {
                (uint32_t)in_dim, (uint32_t)out_dim, 1u,
                (uint32_t)blocks, (uint32_t)row_bytes, weight_type, scale,
            };
            const p36_gpu_tensor *bindings[3] = { weights, q8, out };
            const char *op = weight_type == P36_VK_TENSOR_Q2_K ? "dense_q2k_decode" :
                             (weight_type == P36_VK_TENSOR_Q3_K ? "dense_q3k_decode" :
                             (weight_type == P36_VK_TENSOR_Q4_K ? "dense_q4k_decode" :
                             (weight_type == P36_VK_TENSOR_Q5_K ? "dense_q5k_decode" :
                                                                 "dense_q6k_decode")));
            op = p36_vk_prof_iq3_shape(
                weight_type, 1u, (uint32_t)in_dim, (uint32_t)out_dim, op);
            bool q4_full = weight_type == P36_VK_TENSOR_Q4_K &&
                           out_dim % 4u == 0u;
            bool q5_full = weight_type == P36_VK_TENSOR_Q5_K &&
                           out_dim % 4u == 0u;
            bool q3_full = weight_type == P36_VK_TENSOR_Q3_K &&
                           out_dim % 4u == 0u &&
                           p36_vk_env_default_on("P36_VK_DENSE_Q3K_DECODE");
            p36_vk_kernel *kernel = q4_full ? &p36_vk.dense_q4k_decode :
                                   q5_full ? &p36_vk.dense_q5k_decode :
                                   q3_full ? &p36_vk.dense_q3k_decode :
                                             &p36_vk.dense_kquant_decode;
            ok = p36_vk_run_unlocked(op, kernel,
                                     bindings, &push, sizeof(push),
                                     ((uint32_t)out_dim + 3u) / 4u, 1, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    /* Same widening as the decode branch above: prefill-side K-quant trunk
     * tensors are just as generic on a MoE model. */
    if (!p36_gpu_quality && n_tok > 1u &&
        p36_vk.have_f16 && p36_vk.subgroup_size_control &&
        p36_vk_env_default_on("P36_VK_DENSE_KQUANT_MMQ")) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        int ok = weights != NULL;
        if (ok) {
            struct {
                uint32_t in_dim;
                uint32_t out_dim;
                uint32_t n_tok;
                uint32_t blocks;
                uint32_t row_bytes;
                uint32_t type;
                float scale;
            } push = {
                (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok,
                (uint32_t)blocks, (uint32_t)row_bytes, weight_type, scale,
            };
            p36_gpu_tensor *b16 = p36_vk_predequant_b16_unlocked(
                q8, (uint32_t)n_tok, (uint32_t)blocks);
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, b16 };
            /* One pipeline, branch on push-constant type: label the profiler
             * row by type so per-arm changes are not lost in the shared row. */
            const char *mmq_op =
                weight_type == P36_VK_TENSOR_Q2_K ? "dense_kquant_mmq_q2k" :
                weight_type == P36_VK_TENSOR_Q3_K ? "dense_kquant_mmq_q3k" :
                weight_type == P36_VK_TENSOR_Q4_K ? "dense_kquant_mmq_q4k" :
                weight_type == P36_VK_TENSOR_Q5_K ? "dense_kquant_mmq_q5k" :
                weight_type == P36_VK_TENSOR_Q6_K ? "dense_kquant_mmq_q6k" :
                                                    "dense_kquant_mmq";
            const char *op = p36_vk_prof_iq3_shape(
                weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                (uint32_t)out_dim, mmq_op);
            ok = b16 != NULL &&
                 p36_vk_run_unlocked(op, &p36_vk.dense_kquant_mmq,
                                     bindings, &push, sizeof(push),
                                     ((uint32_t)out_dim + 31u) / 32u,
                                     ((uint32_t)n_tok + 127u) / 128u, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (!p36_gpu_quality && p36_vk_use_mmq() &&
        (weight_type == P36_VK_TENSOR_Q5_K || weight_type == P36_VK_TENSOR_Q6_K) &&
        (n_tok % 4u) == 0u && n_tok >= 4u) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *packed = p36_vk_packed_weight_get_unlocked(source, weight_bytes, weight_type,
                                                                   (uint32_t)in_dim, (uint32_t)out_dim);
        int ok = packed && p36_vk_matmul_kquant_mmq_unlocked(out, packed, q8, (uint32_t)weight_type,
                                                             (uint32_t)out_dim, (uint32_t)n_tok,
                                                             (uint32_t)blocks, scale);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
    int ok = weights != NULL;

    if (ok) {
        struct {
            uint32_t in_dim;
            uint32_t out_dim;
            uint32_t n_tok;
            uint32_t blocks;
            uint32_t row_bytes;
            uint32_t type;
            float scale;
        } mpush = {
            (uint32_t)in_dim,
            (uint32_t)out_dim,
            (uint32_t)n_tok,
            (uint32_t)blocks,
            (uint32_t)row_bytes,
            weight_type,
            scale,
        };
        const p36_gpu_tensor *mbindings[3] = { weights, q8, out };
        /* 8 rows x 8 AVX2 lanes per workgroup. */
        ok = p36_vk_run_unlocked("dense_kquant",
                                 &p36_vk.matmul_kquant,
                                 mbindings,
                                 &mpush,
                                 sizeof(mpush),
                                 ((uint32_t)out_dim + 7u) / 8u,
                                 ((uint32_t)n_tok + P36_VK_MATMUL_TOK_TILE - 1u) / P36_VK_MATMUL_TOK_TILE,
                                 1);
    }

    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_k_quant_scaled_tensor(p36_gpu_tensor *out,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t weight_offset,
                                         uint32_t weight_type,
                                         uint64_t in_dim,
                                         uint64_t out_dim,
                                         const p36_gpu_tensor *x,
                                         uint64_t n_tok,
                                         float scale) {
    p36_gpu_tensor *q8 = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!q8) return 0;
    int ok = p36_gpu_matmul_k_quant_q8_scaled_tensor(out, model_map, model_size, weight_offset,
                                                     weight_type, in_dim, out_dim, q8, n_tok, scale);
    p36_gpu_tensor_free(q8);
    return ok;
}

int p36_gpu_matmul_k_quant_tensor(p36_gpu_tensor *out,
                                  const void *model_map,
                                  uint64_t model_size,
                                  uint64_t weight_offset,
                                  uint32_t weight_type,
                                  uint64_t in_dim,
                                  uint64_t out_dim,
                                  const p36_gpu_tensor *x,
                                  uint64_t n_tok) {
    return p36_gpu_matmul_k_quant_scaled_tensor(out, model_map, model_size, weight_offset,
                                                weight_type, in_dim, out_dim, x, n_tok, 1.0f);
}

/* Row-wise RMS norm on the host, the exact float-accumulation numerics of
 * p36_ref_rms_norm() (the mode the CPU parity gates were tuned for): scalar
 * square sum, 1/sqrtf scale, optional f32 weight. */
static void p36_vk_rms_norm_row(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
    {
        float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

/* Fixed-size clone of the CPU engine's 128-wide norm: gcc compiles that
 * constant-size copy to an 8-lane ymm square-sum reduced in hsum pair
 * order; the explicit lanes reproduce those roundings exactly. */
static void p36_vk_rms_norm_row_128(float *dst, const float *src, const float *weight, float eps) {
    float lanes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float ss;
    for (uint32_t i = 0; i < 128u; i += 8u) {
        for (uint32_t k = 0; k < 8u; k++) lanes[k] += src[i + k] * src[i + k];
    }
    ss = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
       + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    {
        float scale = 1.0f / sqrtf(ss / 128.0f + eps);
        for (uint32_t i = 0; i < 128u; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

typedef struct {
    float *out;
    const float *x;
    const float *weight;
    uint32_t n;
    float eps;
} p36_vk_rms_norm_ctx;

static void p36_vk_rms_norm_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_rms_norm_ctx *ctx = (p36_vk_rms_norm_ctx *)opaque;

    for (uint64_t r = row0; r < row1; r++) {
        if (ctx->n == 128u) p36_vk_rms_norm_row_128(ctx->out + r * 128u, ctx->x + r * 128u, ctx->weight, ctx->eps);
        else p36_vk_rms_norm_row(ctx->out + r * ctx->n, ctx->x + r * ctx->n, ctx->weight, ctx->n, ctx->eps);
    }
}

static int p36_vk_rms_norm_tensor(p36_gpu_tensor *out,
                                  const p36_gpu_tensor *x,
                                  const float *weight,
                                  uint32_t n,
                                  uint32_t rows,
                                  float eps) {
    if (n == 0 || rows == 0) return 0;
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) || !p36_gpu_tensor_range_ok(x, 0, bytes)) return 0;

    /* Fast mode always norms on GPU: with f32-B dense matvecs the host no
     * longer reads activations afterwards, so a host norm here would be the
     * only remaining decode flush cause. */
    if (!p36_gpu_quality && p36_vk_use_gpu_rms()) {
        struct {
            uint32_t n;
            uint32_t has_weight;
            float eps;
        } push = { n, weight ? 1u : 0u, eps };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *wt = weight ? p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float)) : (p36_gpu_tensor *)x;
        int ok = wt != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { x, wt, out };
            ok = p36_vk_run_unlocked("norm", &p36_vk.rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_rms_norm_ctx ctx;
    const float *xp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_rms_norm_x");
    float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_rms_norm_out");
    if (!xp || !op) return 0;
    ctx.out = op;
    ctx.x = xp;
    ctx.weight = weight;
    ctx.n = n;
    ctx.eps = eps;
    if (elems >= 8192u) p36_gpu_parallel_for_rows(rows, 2, p36_vk_rms_norm_rows, &ctx);
    else p36_vk_rms_norm_rows(&ctx, 0, rows);
    return 1;
}

int p36_gpu_rms_norm_plain_rows_tensor(p36_gpu_tensor *out,
                                       const p36_gpu_tensor *x,
                                       uint32_t n,
                                       uint32_t rows,
                                       float eps) {
    return p36_vk_rms_norm_tensor(out, x, NULL, n, rows, eps);
}

int p36_gpu_rms_norm_weight_rows_tensor(p36_gpu_tensor *out,
                                         const p36_gpu_tensor *x,
                                         const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint32_t n,
                                        uint32_t rows,
                                        float eps) {
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;
    return p36_vk_rms_norm_tensor(out, x, weight, n, rows, eps);
}

/* SwiGLU on the host: p36_siluf()'s exact x / (1 + expf(-x)) with the same
 * scalar libm expf the CPU engine calls. */
typedef struct {
    float *out;
    const float *gate;
    const float *up;
    float clamp;
    float weight;
} p36_vk_swiglu_ctx;

static void p36_vk_swiglu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_swiglu_ctx *ctx = (p36_vk_swiglu_ctx *)opaque;

    for (uint64_t i = row0; i < row1; i++) {
        float g = ctx->gate[i];
        float u = ctx->up[i];
        if (ctx->clamp > 1.0e-6f) {
            if (g > ctx->clamp) g = ctx->clamp;
            if (u > ctx->clamp) u = ctx->clamp;
            if (u < -ctx->clamp) u = -ctx->clamp;
        }
        ctx->out[i] = g / (1.0f + expf(-g)) * u * ctx->weight;
    }
}

int p36_gpu_swiglu_tensor(p36_gpu_tensor *out,
                          const p36_gpu_tensor *gate,
                          const p36_gpu_tensor *up,
                          uint32_t n,
                          float clamp,
                          float weight) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !p36_gpu_tensor_range_ok(up, 0, bytes)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_swiglu()) {
        struct {
            uint32_t n;
            float clamp;
            float weight;
        } push = { n, clamp, weight };
        const p36_gpu_tensor *bindings[3] = { gate, up, out };
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_run_unlocked("swiglu", &p36_vk.swiglu, bindings, &push, sizeof(push), (n + 255u) / 256u, 1, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_swiglu_ctx ctx;
    const float *gp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)gate, "submit_wait_swiglu_gate");
    const float *upp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)up, "submit_wait_swiglu_up");
    float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_swiglu_out");
    if (!gp || !upp || !op) return 0;

    ctx.out = op;
    ctx.gate = gp;
    ctx.up = upp;
    ctx.clamp = clamp;
    ctx.weight = weight;
    if (n >= 16384u) p36_gpu_parallel_for_rows(n, 1024, p36_vk_swiglu_rows, &ctx);
    else p36_vk_swiglu_rows(&ctx, 0, n);
    return 1;
}

int p36_gpu_swiglu_q8_k_tensor(p36_gpu_tensor *out,
                                p36_gpu_tensor *q8,
                                const p36_gpu_tensor *gate,
                                const p36_gpu_tensor *up,
                                uint32_t in_dim,
                                uint32_t n_tok,
                                float clamp,
                                float weight) {
    if (p36_gpu_quality || !p36_vk_env_default_on("P36_VK_SWIGLU_Q8") ||
        !in_dim || (in_dim & 255u) || !n_tok) return 0;
    uint64_t elems = (uint64_t)in_dim * n_tok;
    uint64_t bytes = elems * sizeof(float);
    uint32_t blocks = in_dim / 256u;
    uint64_t q8_bytes = (uint64_t)n_tok * blocks * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !p36_gpu_tensor_range_ok(up, 0, bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
        float clamp;
        float weight;
    } push = { in_dim, blocks, n_tok, clamp, weight };
    const p36_gpu_tensor *bindings[4] = { gate, up, out, q8 };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("swiglu_q8_k", &p36_vk.swiglu_q8_k,
                                 bindings, &push, sizeof(push),
                                 blocks, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_ffn_tail_tensor(p36_gpu_tensor *out,
                            const p36_gpu_tensor *shared,
                            const p36_gpu_tensor *scalar,
                            uint32_t out_dim,
                            uint32_t n_tok) {
    if (out_dim == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(shared, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(scalar, 0, (uint64_t)n_tok * sizeof(float))) {
        return 0;
    }
    if (p36_gpu_quality || !p36_vk_use_gpu_ffn_tail()) {
        const float *sp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)shared, "submit_wait_ffn_tail_shared");
        const float *gp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)scalar, "submit_wait_ffn_tail_scalar");
        float *op = p36_vk_tensor_contents_labeled(out, "submit_wait_ffn_tail_out");
        if (!sp || !gp || !op) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float g = p36_vk_sigmoid_stable(gp[t]);
            if (!isfinite(g)) return 0;
            for (uint32_t j = 0; j < out_dim; j++) {
                uint64_t idx = (uint64_t)t * out_dim + j;
                op[idx] = fmaf(sp[idx], g, op[idx]);
            }
        }
        return 1;
    }
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
    } push = { out_dim, n_tok };
    const p36_gpu_tensor *bindings[3] = { out, shared, scalar };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("moe_tail", &p36_vk.ffn_tail, bindings, &push, sizeof(push), (out_dim + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_fwht_tensor(p36_gpu_tensor *dst,
                        const p36_gpu_tensor *src,
                        uint32_t width,
                        uint32_t n_rows,
                        float scale) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    if (!dst || !src || width == 0 || n_rows == 0) return 0;
    if ((width % 1024u) != 0 || width > UINT32_MAX) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(dst, 0, act_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        float    scale;
    } push = { width, scale };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("fwht", &p36_vk.fwht, bindings, &push, sizeof(push),
                                 width / 1024u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_signs_mul_tensor(p36_gpu_tensor *dst,
                             const p36_gpu_tensor *src,
                             const p36_gpu_tensor *signs,
                             uint32_t width,
                             uint32_t n_rows,
                             uint32_t sign_offset,
                             uint32_t total_signs) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    uint64_t sign_bytes = 0;
    if (!dst || !src || !signs || width == 0 || n_rows == 0) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes) ||
        !p36_u64_mul_ok(total_signs, sizeof(float), &sign_bytes)) {
        return 0;
    }
    if (sign_offset > total_signs || (uint64_t)sign_offset + width > total_signs) return 0;
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(dst, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(signs, 0, sign_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t sign_offset;
    } push = { width, sign_offset };
    const p36_gpu_tensor *bindings[3] = { src, signs, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("signs_mul", &p36_vk.signs_mul, bindings, &push,
                                 sizeof(push), (width + 255u) / 256u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_v_grouped_permute_tensor(p36_gpu_tensor *dst,
                                     const p36_gpu_tensor *src,
                                     uint32_t width,
                                     uint32_t hd,
                                     uint32_t nk,
                                     uint32_t rep,
                                     uint32_t n_rows) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    if (!dst || !src || width == 0 || n_rows == 0) return 0;
    if (hd == 0 || nk == 0 || rep == 0) return 0;
    if ((uint64_t)hd * nk * rep != width) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(dst, 0, act_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t hd;
        uint32_t nk;
        uint32_t rep;
        uint32_t n_rows;
    } push = { width, hd, nk, rep, n_rows };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("v_grouped_permute", &p36_vk.v_grouped_permute, bindings,
                                 &push, sizeof(push), (width + 255u) / 256u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_hadamard_prepare_tensor(p36_gpu_tensor *q8_dst,
                                    const p36_gpu_tensor *src,
                                    const p36_gpu_tensor *signs,
                                    uint32_t width,
                                    uint32_t n_rows,
                                    uint32_t sign_offset,
                                    uint32_t total_signs,
                                    float scale,
                                    uint32_t is_grouped) {
    uint64_t elems = 0;
    uint64_t act_bytes = 0;
    uint64_t sign_bytes = 0;
    uint64_t q8_bytes = 0;
    if (!q8_dst || !src || !signs || width == 0 || n_rows == 0) return 0;
    if ((width % 1024u) != 0 || width > UINT32_MAX || n_rows > UINT32_MAX) return 0;
    if (!p36_u64_mul_ok(width, n_rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &act_bytes) ||
        !p36_u64_mul_ok(total_signs, sizeof(float), &sign_bytes)) {
        return 0;
    }
    if (sign_offset > total_signs || (uint64_t)sign_offset + width > total_signs) return 0;
    const uint64_t blocks = width / P36_VK_QK_K;
    if (!p36_u64_mul_ok(n_rows, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(src, 0, act_bytes) ||
        !p36_gpu_tensor_range_ok(signs, 0, sign_bytes) ||
        !p36_gpu_tensor_range_ok(q8_dst, 0, q8_bytes)) {
        return 0;
    }
    struct {
        uint32_t width;
        uint32_t sign_offset;
        float    scale;
        uint32_t n_rows;
        uint32_t blocks;
        uint32_t is_grouped;
    } push = { width, sign_offset, scale, n_rows, (uint32_t)blocks, is_grouped };
    const p36_gpu_tensor *bindings[3] = { src, signs, q8_dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("hadamard_prepare", &p36_vk.hadamard_prepare, bindings,
                                 &push, sizeof(push), width / 1024u, n_rows, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_add_tensor(p36_gpu_tensor *out,
                       const p36_gpu_tensor *a,
                       const p36_gpu_tensor *b,
                       uint32_t n) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) ||
        !p36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }

    struct {
        uint32_t n;
    } push = { n };

    const p36_gpu_tensor *bindings[3] = { a, b, out };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("residual_add",
                                 &p36_vk.add,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (n + 255u) / 256u,
                                 1,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_directional_steering_project_tensor(
        p36_gpu_tensor       *x,
        const p36_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    uint64_t x_elems;
    uint64_t dir_elems;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    if (!p36_u64_mul_ok(width, rows, &x_elems) ||
        !p36_u64_mul_ok((uint64_t)layer + 1u, width, &dir_elems) ||
        !p36_gpu_tensor_range_ok(x, 0, x_elems * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(directions, 0, dir_elems * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t layer;
        uint32_t width;
        uint32_t rows;
        float scale;
    } push = { layer, width, rows, scale };
    const p36_gpu_tensor *bindings[2] = { x, directions };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("directional_steering",
                                 &p36_vk.directional_steering,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 rows,
                                 1,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Fused residual add + weighted RMS norm: out_sum = a + b (the new residual)
 * and out_norm = rmsnorm(out_sum) * w in one dispatch.  Bit-identical to
 * p36_gpu_add_tensor() + p36_gpu_rms_norm_weight_rows_tensor(), which stays
 * the fallback for quality mode and P36_VK_ADD_RMS=0.  out_sum may alias a
 * or b. */
int p36_gpu_add_rms_norm_tensor(p36_gpu_tensor *out_norm,
                                p36_gpu_tensor *out_sum,
                                const p36_gpu_tensor *a,
                                const p36_gpu_tensor *b,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint32_t n,
                                uint32_t rows,
                                float eps) {
    if (n == 0 || rows == 0) return 0;
    if (p36_gpu_quality || !p36_vk_use_add_rms() || !p36_vk_use_gpu_rms() || n > 5120u) {
        return p36_gpu_add_tensor(out_sum, a, b, n * rows) &&
               p36_gpu_rms_norm_weight_rows_tensor(out_norm, out_sum,
                                                   model_map, model_size, weight_offset,
                                                   n, rows, eps);
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out_norm, 0, bytes) || !p36_gpu_tensor_range_ok(out_sum, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) || !p36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;

    struct {
        uint32_t n;
        float eps;
    } push = { n, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { a, b, wt, out_sum, out_norm };
        ok = p36_vk_run_unlocked("add_rms_norm", &p36_vk.add_rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static bool p36_vk_use_fused_rms_q8(void) {
    const char *env = getenv("P36_VK_FUSED_RMS_Q8");
    return env && env[0] == '1';
}

int p36_gpu_add_rms_norm_q8_k_tensor(p36_gpu_tensor *out_norm,
                                     p36_gpu_tensor *out_q8,
                                     p36_gpu_tensor *out_sum,
                                     const p36_gpu_tensor *a,
                                     const p36_gpu_tensor *b,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint32_t n,
                                     uint32_t rows,
                                     float eps) {
    if (!out_q8 || p36_gpu_quality || !p36_vk_use_add_rms() || !p36_vk_use_gpu_rms() ||
        !p36_vk_use_fused_rms_q8() ||
        (n & (P36_VK_QK_K - 1u)) != 0 || n > 5120u) {
        return 0;
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    uint32_t blocks = n / P36_VK_QK_K;
    uint64_t q8_bytes = (uint64_t)rows * blocks * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(out_norm, 0, bytes) || !p36_gpu_tensor_range_ok(out_sum, 0, bytes) ||
        !p36_gpu_tensor_range_ok(a, 0, bytes) || !p36_gpu_tensor_range_ok(b, 0, bytes) ||
        !p36_gpu_tensor_range_ok(out_q8, 0, q8_bytes)) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;

    struct {
        uint32_t n;
        float eps;
        uint32_t blocks;
    } push = { n, eps, blocks };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = { a, b, wt, out_sum, out_norm, out_q8 };
        ok = p36_vk_run_unlocked("add_rms_norm_q8_k", &p36_vk.add_rms_norm_q8_k, bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}







/* ===========================================================================
 * Qwen3.6 decode kernels.
 * ===========================================================================
 *
 * Faithful GPU ports of the CPU reference path in p36.c.  Each wrapper mirrors
 * one step of p36_forward_token_cpu(): the multi-section RoPE, the full
 * attention score/softmax/weighted-sum with sinks and output gating, the
 * delta-net recurrent decode, the SSM conv window, and the routed MoE with the
 * model's per-tensor/per-expert .scale factors.
 */

/* Codebook grids for the IQ expert quants, uploaded once and bound to the
 * moe_matvec kernel.  Layout must match the TAB_* offsets in the shader. */
static p36_gpu_tensor *p36_vk_iq_tables_unlocked(void) {
    if (p36_vk.iq_tables) return p36_vk.iq_tables;
    p36_gpu_tensor *t = p36_vk_tensor_alloc_kind_unlocked(P36_VK_TAB_TOTAL_BYTES, true, true);
    if (!t) return NULL;
    p36_vk_pack_iq_tables(t->data);
    p36_vk.iq_tables = t;
    return t;
}

static float p36_vk_softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

/* In-place row L2 norm on the host: p36_l2_norm()'s float accumulation. */
static void p36_vk_l2_norm_row(float *x, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

/* Fixed-size clone, same reasoning as p36_vk_rms_norm_row_128(). */
static void p36_vk_l2_norm_row_128(float *x, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < 128u; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < 128u; i++) x[i] *= scale;
    }
}

int p36_gpu_l2_norm_rows_tensor(p36_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    uint64_t elems = 0;
    uint64_t bytes = 0;
    float *xp;
    if (n == 0 || rows == 0) return 0;
    if (!p36_u64_mul_ok(n, rows, &elems) || !p36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(x, 0, bytes)) return 0;
    xp = p36_vk_tensor_contents_labeled(x, "submit_wait_l2_norm_x");
    if (!xp) return 0;
    for (uint32_t r = 0; r < rows; r++) {
        if (n == 128u) p36_vk_l2_norm_row_128(xp + (uint64_t)r * 128u, eps);
        else p36_vk_l2_norm_row(xp + (uint64_t)r * n, n, eps);
    }
    return 1;
}

int p36_gpu_recurrent_conv_step_tensor(p36_gpu_tensor *cache_conv,
                                       const p36_gpu_tensor *cur,
                                       p36_gpu_tensor *window,
                                       uint32_t n_tok) {
    const uint32_t hist = P36_VK_N_SSM_CONV - 1u;
    uint64_t row_bytes = (uint64_t)P36_VK_N_SSM_CONV_DIM * sizeof(float);
    if (n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(cache_conv, 0, hist * row_bytes) ||
        !p36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !p36_gpu_tensor_range_ok(window, 0, (uint64_t)n_tok * P36_VK_N_SSM_CONV * row_bytes)) {
        return 0;
    }
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent()) {
        float *cachep = p36_vk_tensor_contents_labeled(cache_conv, "submit_wait_recur_window_cache");
        float *windowp = p36_vk_tensor_contents_labeled(window, "submit_wait_recur_window_out");
        float *curp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)cur, "submit_wait_recur_window_cur");
        if (!cachep || !windowp || !curp) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float *wt = windowp + (uint64_t)t * P36_VK_N_SSM_CONV * P36_VK_N_SSM_CONV_DIM;
            const float *ct = curp + (uint64_t)t * P36_VK_N_SSM_CONV_DIM;
            for (uint32_t i = 0; i < hist; i++) memcpy(wt + (uint64_t)i * P36_VK_N_SSM_CONV_DIM, cachep + (uint64_t)i * P36_VK_N_SSM_CONV_DIM, (size_t)row_bytes);
            memcpy(wt + (uint64_t)hist * P36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
            if (hist > 1u) memmove(cachep, cachep + P36_VK_N_SSM_CONV_DIM, (size_t)(hist - 1u) * (size_t)row_bytes);
            memcpy(cachep + (uint64_t)(hist - 1u) * P36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
        }
        return 1;
    }
    struct {
        uint32_t conv_dim;
        uint32_t hist;
        uint32_t n_tok;
    } push = { P36_VK_N_SSM_CONV_DIM, hist, n_tok };
    const p36_gpu_tensor *bindings[3] = { cache_conv, cur, window };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("recurrent_window", &p36_vk.recur_window, bindings, &push, sizeof(push),
                                 (P36_VK_N_SSM_CONV_DIM + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_gdn_front_tensor(p36_gpu_tensor *history,
    const p36_gpu_tensor *cur, const p36_gpu_tensor *x,
    p36_gpu_tensor *q, p36_gpu_tensor *k, p36_gpu_tensor *v, p36_gpu_tensor *gb,
    const void *map, uint64_t size, const uint64_t offsets[5],
    uint32_t width, uint32_t groups, uint32_t heads,
    float alpha_scale, float beta_scale, float eps) {
    if (!p36_gpu_dense_model || !p36_vk_env_default_on("P36_VK_GDN_FRONT") ||
        p36_gpu_quality || !p36_vk_use_gpu_recurrent() ||
        !p36_vk_use_q8_0_f32b() || !p36_vk_use_q8_0_gpu() ||
        p36_vk.props.limits.maxPerStageDescriptorStorageBuffers<12 ||
        p36_vk.props.limits.maxDescriptorSetStorageBuffers<12 ||
        p36_vk.props.limits.maxComputeWorkGroupInvocations<384 ||
        p36_vk.props.limits.maxComputeWorkGroupSize[0]<384 ||
        groups!=16 || heads!=48 || p36_vk.subgroup_size!=64 || !width || width%32) return 0;
    uint64_t channels=(2u*(uint64_t)groups+heads)*128u;
    uint64_t bytes[5]={channels*16, (uint64_t)width/32*34*heads,
                      (uint64_t)width/32*34*heads, heads*4u, heads*4u};
    if (!p36_gpu_tensor_range_ok(history,0,channels*12) ||
        !p36_gpu_tensor_range_ok(cur,0,channels*4) ||
        !p36_gpu_tensor_range_ok(x,0,(uint64_t)width*4) ||
        !p36_gpu_tensor_range_ok(q,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(k,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(v,0,(uint64_t)heads*512) ||
        !p36_gpu_tensor_range_ok(gb,0,(uint64_t)heads*8)) return 0;
    const void *w[5];
    for (unsigned i=0;i<5;i++) {
        w[i]=p36_gpu_weight_bytes(map,size,offsets[i],bytes[i]);
        if (!w[i]) return 0;
    }
    struct { uint32_t groups,heads,blocks; float alpha_scale,beta_scale,eps; }
        push={groups,heads,width/32,alpha_scale,beta_scale,eps};
    pthread_mutex_lock(&p36_vk_mu);
    const p36_gpu_tensor *b[12]={history,cur,NULL,x,NULL,NULL,NULL,NULL,q,k,v,gb};
    const unsigned slots[5]={2,4,5,6,7};
    int ok=1;
    for (unsigned i=0;i<5;i++) {
        b[slots[i]]=p36_vk_weight_get_unlocked(w[i],bytes[i]);
        if (!b[slots[i]]) { ok=0; break; }
    }
    if (ok) ok=p36_vk_run_unlocked("gdn_front",&p36_gdn_front,b,&push,sizeof(push),groups,1,1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_recurrent_conv_silu_tensor(p36_gpu_tensor *cache_conv,
                                       const p36_gpu_tensor *cur,
                                       p36_gpu_tensor *out,
                                       const void *model_map,
                                       uint64_t model_size,
                                       uint64_t kernel_offset,
                                       uint32_t conv_dim,
                                       uint32_t n_taps,
                                       uint32_t n_tok) {
    uint64_t row_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent() || conv_dim == 0 || n_taps != 4u || n_tok == 0 ||
        !p36_u64_mul_ok(row_bytes, n_taps, &kernel_bytes) ||
        !p36_gpu_tensor_range_ok(cache_conv, 0, (uint64_t)(n_taps - 1u) * row_bytes) ||
        !p36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * row_bytes)) {
        return 0;
    }
    const float *kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              kernel_offset, kernel_bytes);
    if (!kernel) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *kt = p36_vk_weight_get_unlocked(kernel, kernel_bytes);
    int ok = kt != NULL;
    if (ok) {
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        const p36_gpu_tensor *bindings[4] = { cache_conv, cur, kt, out };
        ok = p36_vk_run_unlocked("recurrent_conv_silu_decode", &p36_vk.recur_conv_silu_decode,
                                 bindings, &push, sizeof(push),
                                 (conv_dim + 255u) / 256u, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_copy_f32_tensor(p36_gpu_tensor *dst,
                            const p36_gpu_tensor *src,
                            uint32_t n_floats) {
    if (n_floats == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { n_floats, n_floats, 0u, n_floats, 1u };
    const p36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("embed_stage_copy", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (n_floats + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_extract_full_attn_q_tensor(p36_gpu_tensor *dst,
                                       const p36_gpu_tensor *qg,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { P36_VK_N_HEAD_DIM * 2u, P36_VK_N_HEAD_DIM, 0u, P36_VK_N_HEAD_DIM, n_tok * P36_VK_N_HEAD };
    const p36_gpu_tensor *bindings[2] = { qg, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_extract_q", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (P36_VK_N_HEAD_DIM + 255u) / 256u, n_tok * P36_VK_N_HEAD, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_extract_recurrent_v_tensor(p36_gpu_tensor *dst,
                                       const p36_gpu_tensor *conv,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent()) {
        const float *src = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)conv, "submit_wait_recur_extract_v_src");
        float *out = p36_vk_tensor_contents_labeled(dst, "submit_wait_recur_extract_v_out");
        if (!src || !out) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            memcpy(out + (uint64_t)t * 4096u,
                   src + (uint64_t)t * P36_VK_N_SSM_CONV_DIM + (uint64_t)P36_VK_N_SSM_QK * 2u,
                   (size_t)4096u * sizeof(float));
        }
        return 1;
    }
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { P36_VK_N_SSM_CONV_DIM, 4096u, P36_VK_N_SSM_QK * 2u, 4096u, n_tok };
    const p36_gpu_tensor *bindings[2] = { conv, dst };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("recurrent_extract_v", &p36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (4096u + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rope_qwen_rows_tensor(p36_gpu_tensor *x,
                                  uint32_t n_head,
                                  uint32_t pos0,
                                  uint32_t n_tok) {
    if (n_head == 0 || n_tok == 0) return 0;
    struct {
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t pos0;
        uint32_t n_tok;
    } push = { n_head, P36_VK_N_HEAD_DIM, pos0, n_tok };
    const p36_gpu_tensor *bindings[1] = { x };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_rope", &p36_vk.rope, bindings, &push, sizeof(push), n_head, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rope_qwen_mrope_rows_tensor(p36_gpu_tensor *x,
                                        uint32_t n_head,
                                        const uint32_t *positions,
                                        uint32_t n_tok) {
    if (n_head == 0 || n_tok == 0 || !positions) return 0;
    p36_gpu_tensor *gpos = p36_gpu_tensor_alloc((uint64_t)n_tok * 3u * sizeof(*positions));
    if (!gpos || !p36_gpu_tensor_write(gpos, 0, positions,
                                        (uint64_t)n_tok * 3u * sizeof(*positions))) {
        p36_gpu_tensor_free(gpos);
        return 0;
    }
    struct {
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t n_tok;
    } push = { n_head, P36_VK_N_HEAD_DIM, n_tok };
    const p36_gpu_tensor *bindings[2] = { x, gpos };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("attn_mrope", &p36_vk.rope_mrope,
                                 bindings, &push, sizeof(push), n_head, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(gpos);
    return ok;
}

int p36_gpu_rms_norm_rope_qwen_rows_tensor(p36_gpu_tensor *dst,
                                           const p36_gpu_tensor *src,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t src_stride,
                                           uint32_t n_head,
                                           uint32_t pos0,
                                           uint32_t n_tok,
                                           float eps) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t src_floats;
    uint64_t dst_floats;
    if (src_stride < head_dim || rows == 0) return 0;
    src_floats = (rows - 1u) * src_stride + head_dim;
    dst_floats = rows * head_dim;
    if (!p36_gpu_tensor_range_ok(src, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(dst, 0, dst_floats * sizeof(float))) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        float eps;
        uint32_t pad;
    } push = { src_stride, n_head, pos0, n_tok, eps, 0u };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { src, wt, dst };
        ok = p36_vk_run_unlocked("norm_rope", &p36_vk.rms_norm_rope,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_rms_norm_rope_qwen_kv_store_tensor(p36_gpu_tensor *k_cache,
                                                p36_gpu_tensor *v_cache,
                                                const p36_gpu_tensor *k,
                                                const p36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t cache_values = (uint64_t)cap * n_head * head_dim;
    uint64_t src_floats;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) {
        return 0;
    }
    src_floats = (rows - 1u) * src_stride + head_dim;
    if (!p36_gpu_tensor_range_ok(k, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, rows * head_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, cache_values * sizeof(uint16_t)) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, cache_values * sizeof(uint16_t))) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = p36_vk_run_unlocked("norm_rope_kv", &p36_vk.rms_norm_rope_kv,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Fused RMS norm + Qwen RoPE + quantized KV store (Q8_0 K / Q4_0 V).
 * Same semantics as p36_gpu_rms_norm_rope_qwen_kv_store_tensor but writes
 * quantized blocks directly on the GPU, eliminating the host round-trip. */
int p36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(p36_gpu_tensor *k_cache,
                                                p36_gpu_tensor *v_cache,
                                                const p36_gpu_tensor *k,
                                                const p36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps,
                                                uint32_t k_row_bytes,
                                                uint32_t v_row_bytes) {
    const uint32_t head_dim = P36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t src_floats;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) return 0;
    /* The kernel writes fixed Q8_0 K rows and Q4_0 V rows; reject any
     * other cache layout. */
    if (k_row_bytes != n_head * (head_dim / 32u) * 34u ||
        v_row_bytes != n_head * (head_dim / 32u) * 18u) return 0;
    src_floats = (rows - 1u) * src_stride + head_dim;
    if (!p36_gpu_tensor_range_ok(k, 0, src_floats * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, rows * head_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, (uint64_t)cap * k_row_bytes) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, (uint64_t)cap * v_row_bytes)) {
        return 0;
    }
    const float *weight = (const float *)p36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
        uint32_t k_row_bytes;
        uint32_t v_row_bytes;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps, k_row_bytes, v_row_bytes };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = p36_vk_run_unlocked("norm_rope_kv_quant", &p36_vk.rms_norm_rope_kv_quant,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Short causal conv1d + SiLU on the host: fp64 tap accumulation like
 * p36_ssm_conv_apply() and p36_siluf()'s exact formula with libm expf.
 * Batched over n_tok windows of n_taps rows each. */
typedef struct {
    float *out;
    const float *window;
    const float *kernel;
    uint32_t conv_dim;
    uint32_t n_taps;
} p36_vk_conv_silu_ctx;

static void p36_vk_conv_silu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_conv_silu_ctx *ctx = (p36_vk_conv_silu_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *wt = ctx->window + tok * ctx->n_taps * ctx->conv_dim;
        float *ot = ctx->out + tok * ctx->conv_dim;
        for (uint32_t c = 0; c < ctx->conv_dim; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < ctx->n_taps; t++) {
                acc += (double)wt[(uint64_t)t * ctx->conv_dim + c] * (double)ctx->kernel[(uint64_t)c * ctx->n_taps + t];
            }
            {
                float a = (float)acc;
                ot[c] = a / (1.0f + expf(-a));
            }
        }
    }
}

int p36_gpu_ssm_conv_silu_tensor(p36_gpu_tensor *out,
                                 const p36_gpu_tensor *window,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t kernel_offset,
                                 uint32_t conv_dim,
                                 uint32_t n_taps,
                                 uint32_t n_tok) {
    uint64_t dim_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes = dim_bytes * n_taps;
    if (conv_dim == 0 || n_taps == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, dim_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(window, 0, kernel_bytes * n_tok)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        const float *kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
        if (!kernel) return 0;
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *kt = p36_vk_weight_get_unlocked(kernel, kernel_bytes);
        int ok = kt != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { window, kt, out };
            ok = p36_vk_run_unlocked("recurrent_conv_silu", &p36_vk.conv_silu, bindings, &push, sizeof(push),
                                     (conv_dim + 255u) / 256u, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_conv_silu_ctx ctx;
    const float *kernel;
    const float *win;
    float *op;
    kernel = (const float *)p36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
    win = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)window, "submit_wait_recur_conv_window");
    op = p36_vk_tensor_contents_labeled(out, "submit_wait_recur_conv_out");
    if (!kernel || !win || !op) return 0;

    ctx.out = op;
    ctx.window = win;
    ctx.kernel = kernel;
    ctx.conv_dim = conv_dim;
    ctx.n_taps = n_taps;
    if ((uint64_t)n_tok * conv_dim >= 8192u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_conv_silu_rows, &ctx);
    else p36_vk_conv_silu_rows(&ctx, 0, n_tok);
    return 1;
}

/* Delta-net q/k head assembly on the host: group gather plus the
 * p36_l2_norm() float accumulation, like the loop in p36_forward_recurrent().
 * conv_stride is the per-token float stride of the conv tensor (its rows
 * carry the v part after the q/k groups). */
typedef struct {
    float *q;
    float *k;
    const float *conv;
    uint32_t n_heads;
    uint32_t n_groups;
    uint32_t state_dim;
    uint32_t conv_stride;
    uint64_t head_floats;
    float eps;
} p36_vk_delta_qk_ctx;

static void p36_vk_delta_qk_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_delta_qk_ctx *ctx = (p36_vk_delta_qk_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *ct = ctx->conv + tok * ctx->conv_stride;
        float *qt = ctx->q + tok * ctx->head_floats;
        float *kt = ctx->k + tok * ctx->head_floats;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            uint32_t g = h % ctx->n_groups;
            memcpy(qt + (uint64_t)h * ctx->state_dim, ct + (uint64_t)g * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            memcpy(kt + (uint64_t)h * ctx->state_dim, ct + ((uint64_t)ctx->n_groups + g) * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            if (ctx->state_dim == 128u) {
                p36_vk_l2_norm_row_128(qt + (uint64_t)h * 128u, ctx->eps);
                p36_vk_l2_norm_row_128(kt + (uint64_t)h * 128u, ctx->eps);
            } else {
                p36_vk_l2_norm_row(qt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
                p36_vk_l2_norm_row(kt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
            }
        }
    }
}

int p36_gpu_delta_qk_l2_norm_tensor(p36_gpu_tensor *q_out,
                                    p36_gpu_tensor *k_out,
                                    const p36_gpu_tensor *conv,
                                    uint32_t n_heads,
                                    uint32_t n_groups,
                                    uint32_t state_dim,
                                    uint32_t conv_stride,
                                    uint32_t n_tok,
                                    float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_bytes;
    if (n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim) return 0;
    conv_bytes = ((uint64_t)conv_stride * (n_tok - 1u) + 2u * (uint64_t)n_groups * state_dim) * sizeof(float);
    if (!p36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(conv, 0, conv_bytes)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        struct {
            uint32_t n_heads;
            uint32_t n_groups;
            uint32_t state_dim;
            uint32_t conv_stride;
            uint32_t n_tok;
            float eps;
        } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
        const p36_gpu_tensor *bindings[3] = { conv, q_out, k_out };
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_run_unlocked("delta_qk_l2", &p36_vk.delta_qk, bindings, &push, sizeof(push), n_heads, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_delta_qk_ctx ctx;
    const float *cp;
    float *qp;
    float *kp;
    cp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)conv, "submit_wait_delta_qk_conv");
    qp = p36_vk_tensor_contents_labeled(q_out, "submit_wait_delta_qk_q");
    kp = p36_vk_tensor_contents_labeled(k_out, "submit_wait_delta_qk_k");
    if (!cp || !qp || !kp) return 0;

    ctx.q = qp;
    ctx.k = kp;
    ctx.conv = cp;
    ctx.n_heads = n_heads;
    ctx.n_groups = n_groups;
    ctx.state_dim = state_dim;
    ctx.conv_stride = conv_stride;
    ctx.head_floats = head_floats;
    ctx.eps = eps;
    if ((uint64_t)n_tok * n_heads * state_dim >= 16384u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_delta_qk_rows, &ctx);
    else p36_vk_delta_qk_rows(&ctx, 0, n_tok);
    return 1;
}

int p36_gpu_delta_qkv_l2_norm_tensor(p36_gpu_tensor *q_out,
                                     p36_gpu_tensor *k_out,
                                     p36_gpu_tensor *v_out,
                                     const p36_gpu_tensor *conv,
                                     uint32_t n_heads,
                                     uint32_t n_groups,
                                     uint32_t state_dim,
                                     uint32_t conv_stride,
                                     uint32_t n_tok,
                                     float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_floats;
    if (p36_gpu_quality || !p36_vk_use_gpu_recurrent() ||
        n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim + head_floats) return 0;
    conv_floats = (uint64_t)conv_stride * (n_tok - 1u) +
                  2u * (uint64_t)n_groups * state_dim + head_floats;
    if (!p36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(v_out, 0, head_bytes) ||
        !p36_gpu_tensor_range_ok(conv, 0, conv_floats * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t n_groups;
        uint32_t state_dim;
        uint32_t conv_stride;
        uint32_t n_tok;
        float eps;
    } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
    const p36_gpu_tensor *bindings[4] = { conv, q_out, k_out, v_out };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("delta_qkv_l2", &p36_vk.delta_qkv, bindings,
                                 &push, sizeof(push), n_heads, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* Host-side per-head gate prep (64 scalars): decay = expf(softplus(alpha +
 * dt_bias) * a) and beta = sigmoid(beta_raw), exactly the elementwise loop
 * before p36_delta_net_decode() plus its expf, so the shader never needs a
 * GPU exp that would round differently from libm. */
typedef struct {
    float *gb;
    const float *alpha;
    const float *beta;
    const float *dt;
    const float *a;
    uint32_t n_heads;
} p36_vk_delta_gates_ctx;

static void p36_vk_delta_gates_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_delta_gates_ctx *ctx = (p36_vk_delta_gates_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *at = ctx->alpha + tok * ctx->n_heads;
        const float *bt = ctx->beta + tok * ctx->n_heads;
        float *gt = ctx->gb + tok * 2u * ctx->n_heads;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            gt[h] = expf(p36_vk_softplus(at[h] + ctx->dt[h]) * ctx->a[h]);
            gt[ctx->n_heads + h] = p36_vk_sigmoid_stable(bt[h]);
        }
    }
}

int p36_gpu_delta_net_gates_tensor(p36_gpu_tensor *gb,
                                   const p36_gpu_tensor *alpha,
                                   const p36_gpu_tensor *beta,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t dt_bias_offset,
                                   uint64_t a_offset,
                                   uint32_t n_heads,
                                   uint32_t n_tok) {
    uint64_t bytes = (uint64_t)n_heads * sizeof(float);
    if (n_heads == 0 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(gb, 0, 2u * bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(alpha, 0, bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(beta, 0, bytes * n_tok)) {
        return 0;
    }

    if (!p36_gpu_quality && p36_vk_use_gpu_recurrent()) {
        const float *dt = (const float *)p36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
        const float *a = (const float *)p36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
        if (!dt || !a) return 0;
        struct {
            uint32_t n_heads;
            uint32_t n_tok;
        } push = { n_heads, n_tok };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *dt_t = p36_vk_weight_get_unlocked(dt, bytes);
        p36_gpu_tensor *a_t = dt_t ? p36_vk_weight_get_unlocked(a, bytes) : NULL;
        int ok = dt_t != NULL && a_t != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[5] = { alpha, beta, dt_t, a_t, gb };
            ok = p36_vk_run_unlocked("delta_gates", &p36_vk.delta_gates, bindings, &push, sizeof(push),
                                     (n_heads + 63u) / 64u, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    p36_vk_delta_gates_ctx ctx;
    const float *dt;
    const float *a;
    const float *alpha_host;
    const float *beta_host;
    float *gb_host;
    dt = (const float *)p36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
    a = (const float *)p36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
    alpha_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)alpha, "submit_wait_delta_gates_alpha");
    beta_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)beta, "submit_wait_delta_gates_beta");
    gb_host = p36_vk_tensor_contents_labeled(gb, "submit_wait_delta_gates_out");
    if (!dt || !a || !alpha_host || !beta_host || !gb_host) return 0;

    ctx.gb = gb_host;
    ctx.alpha = alpha_host;
    ctx.beta = beta_host;
    ctx.dt = dt;
    ctx.a = a;
    ctx.n_heads = n_heads;
    if ((uint64_t)n_tok * n_heads >= 128u) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_delta_gates_rows, &ctx);
    else p36_vk_delta_gates_rows(&ctx, 0, n_tok);
    return 1;
}

/* n_tok sequential state updates in one dispatch: the kernel loops tokens
 * inside the workgroup, preserving the exact per-head chain order, so a
 * prefill batch costs one dispatch instead of n_tok barrier-separated
 * ones (per-dispatch overhead is what bounds this stack). */
int p36_gpu_delta_net_decode_tensor(p36_gpu_tensor *state,
                                    const p36_gpu_tensor *q,
                                    const p36_gpu_tensor *k,
                                    const p36_gpu_tensor *v,
                                    const p36_gpu_tensor *gb,
                                    p36_gpu_tensor *out,
                                    uint32_t n_heads,
                                    uint32_t state_dim,
                                    uint32_t n_tok) {
    uint64_t vec_bytes = (uint64_t)n_heads * state_dim * sizeof(float);
    uint64_t gb_bytes = 2u * (uint64_t)n_heads * sizeof(float);
    uint64_t state_bytes = vec_bytes * state_dim;
    bool state_f16 = state && state->bytes == state_bytes / 2u;
    if (n_heads == 0 || state_dim == 0 || state_dim > 128 || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(state, 0, state_f16 ? state_bytes / 2u : state_bytes) ||
        !p36_gpu_tensor_range_ok(q, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(k, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(v, 0, vec_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(gb, 0, gb_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(out, 0, vec_bytes * n_tok)) {
        return 0;
    }

    struct {
        uint32_t n_heads;
        uint32_t state_dim;
        uint32_t n_tok;
    } push = { n_heads, state_dim, n_tok };
    const p36_gpu_tensor *bindings[6] = { state, q, k, v, gb, out };
    p36_vk_kernel *kernel = &p36_vk.delta_net;
    const char *name = "delta_net";
    uint32_t groups_y = 1;
    if (!p36_gpu_quality) {
        if (n_tok >= 32u && state_dim == 128u && p36_vk_use_delta_col_prefill()) {
            /* Four columns share each q/k vector, so a 128-token window stays
             * cache-friendly while avoiding an intermediate state round-trip. */
            int ok = 1;
            pthread_mutex_lock(&p36_vk_mu);
            for (uint32_t t0 = 0; ok && t0 < n_tok; t0 += 128u) {
                struct {
                    uint32_t n_heads;
                    uint32_t state_dim;
                    uint32_t tok0;
                    uint32_t tok_end;
                } cpush = { n_heads, state_dim, t0,
                            t0 + 128u < n_tok ? t0 + 128u : n_tok };
                ok = p36_vk_run_unlocked("delta_net_cols",
                                         state_f16 ? &p36_vk.delta_net_cols_f16 :
                                                     &p36_vk.delta_net_cols,
                                         bindings, &cpush, sizeof(cpush),
                                         n_heads, (state_dim + 3u) / 4u, 1);
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        } else if (state_dim == 128u && p36_vk_use_delta_decode()) {
            bool reg = state_f16 || p36_vk_use_delta_reg();
            kernel = state_f16 ? &p36_vk.delta_net_decode_reg_f16 :
                     reg ? &p36_vk.delta_net_decode_reg :
                           &p36_vk.delta_net_decode;
            name = "delta_net_decode";
            /* COLS for the register kernel: P36_VK_DELTA_COLS selects the
             * .spv variant's column count so the grid matches it.  The default
             * 32 must match -DP36_COLS in the Makefile rule; the env override
             * only works against a .spv built with the same value. */
            uint32_t reg_cols = 32u;
            if (reg) {
                const char *env = getenv("P36_VK_DELTA_COLS");
                if (env && env[0]) {
                    unsigned long v = strtoul(env, NULL, 10);
                    if (v == 8ul || v == 16ul || v == 32ul) reg_cols = (uint32_t)v;
                }
            }
            groups_y = state_dim / (reg ? reg_cols : 32u);
        } else if (p36_vk_use_delta_fast_prefill()) {
            kernel = &p36_vk.delta_net_fast;
            name = "delta_net_fast";
        }
    }
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked(name, kernel, bindings, &push, sizeof(push),
                                 n_heads, groups_y, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

/* KV cache append, the GPU twin of p36_full_attn_cache_store(): row t of
 * k/v lands at position pos0 + t.  Rows are host-converted when the fast
 * f16 GPU append path is not applicable. */
int p36_gpu_attn_kv_store_tensor(p36_gpu_tensor *k_cache,
                                 p36_gpu_tensor *v_cache,
                                 const p36_gpu_tensor *k,
                                 const p36_gpu_tensor *v,
                                 uint32_t pos0,
                                 uint32_t n_tok,
                                 uint32_t cap,
                                 uint32_t k_row,
                                 uint32_t v_row,
                                 uint32_t k_cache_type,
                                 uint32_t v_cache_type,
                                 uint32_t k_cache_row_bytes,
                                 uint32_t v_cache_row_bytes) {
    const float *k_host;
    const float *v_host;
    unsigned char *kc;
    unsigned char *vc;
    p36_vk_prof_scope prof;
    int ok = 0;
    uint32_t k_expect;
    uint32_t v_expect;
    if (cap == 0 || n_tok == 0 || pos0 >= cap || n_tok > cap - pos0 || k_row == 0 || v_row == 0) return 0;
    if (k_cache_type > 2u || v_cache_type > 2u || k_cache_row_bytes == 0 || v_cache_row_bytes == 0) return 0;
    k_expect = p36_vk_kv_cache_row_bytes(k_cache_type, k_row);
    v_expect = p36_vk_kv_cache_row_bytes(v_cache_type, v_row);
    if (k_cache_row_bytes != k_expect || v_cache_row_bytes != v_expect) return 0;
    if (!p36_gpu_tensor_range_ok(k, 0, (uint64_t)n_tok * k_row * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(v, 0, (uint64_t)n_tok * v_row * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, (uint64_t)cap * k_cache_row_bytes) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, (uint64_t)cap * v_cache_row_bytes)) {
        return 0;
    }
    /* Fast decode appends KV on GPU; packHalf2x16 rounds to nearest-even,
     * matching P36's host half conversion. */
    if (k_cache_type == 0u && v_cache_type == 0u &&
        k_cache_row_bytes == k_row * sizeof(uint16_t) &&
        v_cache_row_bytes == v_row * sizeof(uint16_t) &&
        !p36_gpu_quality && (k_row % 2u) == 0u && (v_row % 2u) == 0u &&
        p36_vk_env_default_on("P36_VK_GPU_KV_STORE")) {
        struct {
            uint32_t k_row;
            uint32_t v_row;
            uint32_t pos0;
            uint32_t n_tok;
        } push = { k_row, v_row, pos0, n_tok };
        const p36_gpu_tensor *bindings[4] = { k_cache, v_cache, k, v };
        uint32_t row_max = k_row > v_row ? k_row : v_row;
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("attn_kv_store", &p36_vk.kv_store, bindings, &push, sizeof(push),
                                 (row_max / 2u + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    /* GPU quantized KV store: writes Q8_0/Q4_0 blocks directly on the GPU,
     * avoiding the host round-trip that costs ~200 pipeline flushes per run. */
    if (!p36_gpu_quality &&
        (k_row % 32u) == 0u && (v_row % 32u) == 0u &&
        k_row <= 1024u && v_row <= 1024u &&
        k_cache_type >= 1u && k_cache_type <= 2u &&
        v_cache_type >= 1u && v_cache_type <= 2u &&
        p36_vk_env_default_on("P36_VK_GPU_KV_STORE")) {
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } kpush =
            { k_row, k_cache_row_bytes, pos0, n_tok, k_cache_type };
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } vpush =
            { v_row, v_cache_row_bytes, pos0, n_tok, v_cache_type };
        const p36_gpu_tensor *kbind[2] = { k_cache, k };
        const p36_gpu_tensor *vbind[2] = { v_cache, v };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("kv_store_quant_k", &p36_vk.kv_store_quant,
                                 kbind, &kpush, sizeof(kpush), 1, n_tok, 1);
        if (ok) {
            ok = p36_vk_run_unlocked("kv_store_quant_v", &p36_vk.kv_store_quant,
                                     vbind, &vpush, sizeof(vpush), 1, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    p36_vk_prof_scope_begin(&prof, "attn_kv_store_host");
    k_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)k, "submit_wait_attn_kv_k");
    v_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)v, "submit_wait_attn_kv_v");
    kc = p36_vk_tensor_contents_labeled(k_cache, "submit_wait_attn_kv_k_cache");
    vc = p36_vk_tensor_contents_labeled(v_cache, "submit_wait_attn_kv_v_cache");
    if (!k_host || !v_host || !kc || !vc) goto done;

    for (uint32_t t = 0; t < n_tok; t++) {
        uint64_t pos = pos0 + t;
        const float *kt = k_host + (uint64_t)t * k_row;
        const float *vt = v_host + (uint64_t)t * v_row;
        unsigned char *kdst = kc + pos * k_cache_row_bytes;
        unsigned char *vdst = vc + pos * v_cache_row_bytes;
        if (k_cache_type == 0u) {
            uint16_t *h = (uint16_t *)kdst;
            for (uint32_t i = 0; i < k_row; i++) h[i] = p36_quant_f32_to_f16(kt[i]);
        } else if (k_cache_type == 1u) {
            p36_quant_q8_0(kt, kdst, (int64_t)k_row);
        } else {
            p36_quant_q4_0_kv(kt, kdst, (int64_t)k_row);
        }
        if (v_cache_type == 0u) {
            uint16_t *h = (uint16_t *)vdst;
            for (uint32_t i = 0; i < v_row; i++) h[i] = p36_quant_f32_to_f16(vt[i]);
        } else if (v_cache_type == 1u) {
            p36_quant_q8_0(vt, vdst, (int64_t)v_row);
        } else {
            p36_quant_q4_0_kv(vt, vdst, (int64_t)v_row);
        }
    }
    ok = 1;
done:
    p36_vk_prof_scope_end(&prof);
    return ok;
}

/* Full-attention decode: score pass then softmax/weighted-sum/gating pass,
 * mirroring the per-head loop in p36_forward_full_attn().  Causal batch:
 * query row t (absolute position pos0 + t) attends keys 0..pos0+t.  One
 * score dispatch covers the whole chunk (token on workgroup z), the softmax
 * runs on the host for every token after one flush, then one reduce
 * dispatch finishes it, so the cost is one GPU round-trip and two
 * dispatches per batch instead of per token.  scores is caller scratch of
 * n_tok * n_head * (pos0 + n_tok) floats and is clobbered. */
typedef struct {
    float *aux;
    float *scores;
    const float *qg;
    const float *sinks;
    uint64_t score_stride;
    uint64_t qg_stride;
    uint64_t aux_floats;
    uint32_t pos0;
    uint32_t n_head;
    uint32_t head_dim;
    bool has_sinks;
} p36_vk_attn_host_ctx;

static void p36_vk_attn_host_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_attn_host_ctx *ctx = (p36_vk_attn_host_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        uint32_t n_kv = ctx->pos0 + (uint32_t)tok + 1u;
        float *stok = ctx->scores + tok * ctx->score_stride;
        float *atok = ctx->aux + tok * ctx->aux_floats;
        const float *gtok = ctx->qg + tok * ctx->qg_stride;
        for (uint32_t h = 0; h < ctx->n_head; h++) {
            float *sh = stok + (uint64_t)h * n_kv;
            float max_score = ctx->has_sinks ? ctx->sinks[h] : -INFINITY;
            double denom = 0.0;
            for (uint32_t t = 0; t < n_kv; t++) {
                if (sh[t] > max_score) max_score = sh[t];
            }
            if (ctx->has_sinks) denom = exp((double)ctx->sinks[h] - (double)max_score);
            for (uint32_t t = 0; t < n_kv; t++) {
                float w = expf(sh[t] - max_score);
                denom += w;
                sh[t] = w;
            }
            atok[h] = denom > 0.0 ? (float)(1.0 / denom) : 1.0f;
            for (uint32_t j = 0; j < ctx->head_dim; j++) {
                float g = gtok[(uint64_t)h * 2u * ctx->head_dim + ctx->head_dim + j];
                atok[ctx->n_head + (uint64_t)h * ctx->head_dim + j] = 1.0f / (1.0f + expf(-g));
            }
        }
    }
}

int p36_gpu_attn_decode_tensor(p36_gpu_tensor *out,
                               const p36_gpu_tensor *q,
                               const p36_gpu_tensor *qg,
                               const p36_gpu_tensor *k_cache,
                               const p36_gpu_tensor *v_cache,
                               p36_gpu_tensor *scores,
                               const void *model_map,
                               uint64_t model_size,
                               uint64_t sinks_offset,
                               bool has_sinks,
                               uint32_t pos0,
                               uint32_t n_tok,
                               uint32_t n_head,
                               uint32_t n_head_kv,
                               uint32_t head_dim,
                               uint32_t k_cache_type,
                               uint32_t v_cache_type,
                               uint32_t k_cache_row_bytes,
                               uint32_t v_cache_row_bytes) {
    uint32_t kv_max = pos0 + n_tok;
    uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    uint64_t qg_stride = 2u * (uint64_t)n_head * head_dim;
    uint64_t k_cache_bytes = (uint64_t)kv_max * k_cache_row_bytes;
    uint64_t v_cache_bytes = (uint64_t)kv_max * v_cache_row_bytes;
    uint64_t score_stride = (uint64_t)n_head * kv_max * sizeof(float);
    uint64_t aux_floats = (uint64_t)n_head + (uint64_t)n_head * head_dim;
    const float *sinks = NULL;
    p36_gpu_tensor *aux;
    int ok = 1;
    uint32_t k_expect;
    uint32_t v_expect;
    if (n_tok == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0) return 0;
    if (n_head % n_head_kv != 0 || kv_max < n_tok ||
        k_cache_type > 2u || v_cache_type > 2u ||
        k_cache_row_bytes == 0 || v_cache_row_bytes == 0) return 0;
    k_expect = p36_vk_kv_cache_row_bytes(k_cache_type, n_head_kv * head_dim);
    v_expect = p36_vk_kv_cache_row_bytes(v_cache_type, n_head_kv * head_dim);
    if (k_cache_row_bytes != k_expect || v_cache_row_bytes != v_expect) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, q_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(q, 0, q_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(qg, 0, 2u * q_bytes * n_tok) ||
        !p36_gpu_tensor_range_ok(k_cache, 0, k_cache_bytes) ||
        !p36_gpu_tensor_range_ok(v_cache, 0, v_cache_bytes)) {
        return 0;
    }
    if (has_sinks) {
        sinks = (const float *)p36_gpu_weight_bytes(model_map, model_size, sinks_offset,
                                                    (uint64_t)n_head * sizeof(float));
        if (!sinks) return 0;
    }

    /* Fused fast path: scores, online softmax, V sum, and gating in one
     * dispatch, f32 with a per-token order that depends only on n_kv (so
     * warm==cold holds).  scores may be NULL here; the runtime skips that
     * scratch when this path is on.  --quality keeps the bit-exact fp64
     * chain below. */
    if (p36_gpu_attn_fused_enabled() &&
        head_dim <= 256 && (head_dim & 1u) == 0u) {
        /* GQA-tiled prefill: one workgroup reuses K/V across the query heads
         * sharing a KV head and folds eight spans into each partial. */
        if (n_tok >= 2u && n_head / n_head_kv <= 8u &&
            p36_vk_use_attn_qtile()) {
            uint32_t n_spans = (kv_max + 511u) / 512u;
            uint32_t n_groups = (n_spans + 7u) / 8u;
            uint32_t attn_chunk = n_tok;
            /* Eager submits bound command buffers, not a dispatch itself. */
            if (n_groups > 1u) {
                attn_chunk = (5u * 512u) / n_groups;
                if (attn_chunk > 512u) attn_chunk = 512u;
                if (attn_chunk == 0u) attn_chunk = 1u;
            }
            uint64_t part_bytes = (uint64_t)attn_chunk * n_head * n_groups *
                                  (head_dim + 2u) * sizeof(float);
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t n_head_kv;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t n_tok;
                uint32_t has_sinks;
                uint32_t k_type;
                uint32_t v_type;
                uint32_t k_row_bytes;
                uint32_t v_row_bytes;
                uint32_t tok0;
            } qpush = { pos0, n_head, n_head_kv, head_dim, n_groups, 0, has_sinks ? 1u : 0u,
                        k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes, 0 };
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t qg_stride;
                uint32_t has_sinks;
                uint32_t span_keys;
                uint32_t tok0;
            } cpush = { pos0, n_head, head_dim, n_groups,
                        (uint32_t)qg_stride, has_sinks ? 1u : 0u, 4096u, 0 };
            pthread_mutex_lock(&p36_vk_mu);
            p36_gpu_tensor *part = p36_vk.attn_part;
            if (!part || p36_vk.attn_part_bytes < part_bytes) {
                p36_vk_tensor_release_unlocked(p36_vk.attn_part);
                part = p36_vk_tensor_alloc_scratch_unlocked(part_bytes);
                p36_vk.attn_part = part;
                p36_vk.attn_part_bytes = part ? part_bytes : 0;
            }
            p36_gpu_tensor *sinks_t = sinks ?
                p36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
                (p36_gpu_tensor *)qg;
            ok = part != NULL && sinks_t != NULL;
            /* Long attention makes a dispatch-count batch exceed AMDGPU's
             * scheduler timeout.  Keep each heavy slice in its own queued
             * command buffer; the ring still records ahead of the GPU. */
            bool submit_slices = n_groups >= 5u;
            if (ok && submit_slices) ok = p36_vk_submit_eager_unlocked();
            for (uint32_t tok0 = 0; ok && tok0 < n_tok; tok0 += attn_chunk) {
                uint32_t chunk = n_tok - tok0;
                if (chunk > attn_chunk) chunk = attn_chunk;
                qpush.n_tok = chunk;
                qpush.tok0 = tok0;
                cpush.tok0 = tok0;
                const p36_gpu_tensor *qbind[5] = { q, k_cache, v_cache, sinks_t, part };
                uint32_t ratio = n_head / n_head_kv;
                bool qt2 = p36_vk_use_attn_qtile2() &&
                           head_dim == 256u &&
                           (ratio == 8u || ratio == 6u) &&
                           k_cache_type == 1u && v_cache_type == 2u;
                /* The FA kernels write the same per-group partials, 8 (GQA 6)
                 * or 4 (GQA 8) tokens per workgroup instead of 2. */
                bool fa = qt2 && p36_vk_use_attn_fa();
                if (fa) {
                    uint32_t fa_qt = ratio == 6u ? 8u : 4u;
                    ok = p36_vk_run_unlocked(ratio == 6u ? "attn_prefill_fa_gqa6" : "attn_prefill_fa_gqa8",
                                             ratio == 6u ? &p36_vk.attn_prefill_fa_gqa6
                                                         : &p36_vk.attn_prefill_fa_gqa8,
                                             qbind, &qpush, sizeof(qpush),
                                             n_head_kv, (chunk + fa_qt - 1u) / fa_qt, n_groups);
                } else {
                    ok = p36_vk_run_unlocked(qt2 ? (ratio == 6u ? "attn_prefill_qtile2_gqa6"
                                                                 : "attn_prefill_qtile2")
                                                  : "attn_prefill_qtile",
                                             qt2 ? (ratio == 6u ? &p36_vk.attn_prefill_qtile2_gqa6
                                                                : &p36_vk.attn_prefill_qtile2)
                                                 : &p36_vk.attn_prefill_qtile,
                                             qbind, &qpush, sizeof(qpush),
                                             n_head_kv, qt2 ? (chunk + 1u) / 2u : chunk, n_groups);
                }
                if (!ok) break;
                const p36_gpu_tensor *cbind[4] = { part, qg, sinks_t, out };
                ok = p36_vk_run_unlocked("attn_combine", &p36_vk.attn_combine,
                                         cbind, &cpush, sizeof(cpush),
                                         n_head, chunk, 1);
                if (ok && submit_slices) ok = p36_vk_submit_eager_unlocked();
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        }

        /* Split-K when the causal range spans multiple 512-key windows:
         * per-span partials then an ordered combine.  Decode occupancy goes
         * from n_head workgroups to n_head * spans.  A single-span split is
         * bit-identical to the fused kernel (span 0 seeds its max with the
         * sink), so gating on kv_max keeps per-token n_tok-invariance. */
        uint32_t span_keys = p36_vk_attn_span();
        uint32_t n_spans = (kv_max + span_keys - 1u) / span_keys;
        if (n_spans > 1u && head_dim <= 256u && p36_vk_use_attn_splitk()) {
            uint64_t part_bytes = (uint64_t)n_tok * n_head * n_spans * (head_dim + 2u) * sizeof(float);
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t n_head_kv;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t has_sinks;
                uint32_t k_type;
                uint32_t v_type;
                uint32_t k_row_bytes;
                uint32_t v_row_bytes;
                uint32_t span_keys;
            } spush = { pos0, n_head, n_head_kv, head_dim, n_spans, has_sinks ? 1u : 0u,
                        k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes,
                        span_keys };
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t qg_stride;
                uint32_t has_sinks;
                uint32_t span_keys;
                uint32_t tok0;
            } cpush = { pos0, n_head, head_dim, n_spans,
                        (uint32_t)qg_stride, has_sinks ? 1u : 0u, span_keys, 0 };
            pthread_mutex_lock(&p36_vk_mu);
            /* The partials scratch lives on the runtime and only grows: an
             * alloc/free per attention call showed up as ~0.2ms of host
             * record time per dispatch. */
            p36_gpu_tensor *part = p36_vk.attn_part;
            if (!part || p36_vk.attn_part_bytes < part_bytes) {
                p36_vk_tensor_release_unlocked(p36_vk.attn_part);
                part = p36_vk_tensor_alloc_scratch_unlocked(part_bytes);
                p36_vk.attn_part = part;
                p36_vk.attn_part_bytes = part ? part_bytes : 0;
            }
            p36_gpu_tensor *sinks_t = sinks ?
                p36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
                (p36_gpu_tensor *)qg;
            ok = part != NULL && sinks_t != NULL;
            if (ok) {
                const p36_gpu_tensor *sbind[5] = { q, k_cache, v_cache, sinks_t, part };
                ok = p36_vk_run_unlocked("attn_decode_split", &p36_vk.attn_decode_split,
                                         sbind, &spush, sizeof(spush),
                                         n_head, n_tok, n_spans);
            }
            if (ok) {
                const p36_gpu_tensor *cbind[4] = { part, qg, sinks_t, out };
                ok = p36_vk_run_unlocked("attn_combine", &p36_vk.attn_combine,
                                         cbind, &cpush, sizeof(cpush),
                                         n_head, n_tok, 1);
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        }
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t qg_stride;
            uint32_t has_sinks;
            uint32_t k_type;
            uint32_t v_type;
            uint32_t k_row_bytes;
            uint32_t v_row_bytes;
        } fpush = {
            pos0, n_head, n_head_kv, head_dim,
            (uint32_t)qg_stride, has_sinks ? 1u : 0u,
            k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes,
        };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *sinks_t = sinks ?
            p36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
            (p36_gpu_tensor *)qg;
        ok = sinks_t != NULL;
        if (ok) {
            const p36_gpu_tensor *fbindings[6] = { q, qg, k_cache, v_cache, sinks_t, out };
            ok = p36_vk_run_unlocked("attn_decode_fused", &p36_vk.attn_decode_fused,
                                     fbindings, &fpush, sizeof(fpush),
                                     n_head, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (!p36_gpu_tensor_range_ok(scores, 0, score_stride * n_tok)) return 0;

    {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t k_type;
            uint32_t k_row_bytes;
        } spush = { pos0, n_head, n_head_kv, head_dim, (uint32_t)(score_stride / sizeof(float)),
                    k_cache_type, k_cache_row_bytes };
        const p36_gpu_tensor *sbindings[3] = { q, k_cache, scores };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("attn_scores", &p36_vk.attn_scores, sbindings, &spush, sizeof(spush),
                                 (kv_max + 63u) / 64u, n_head, n_tok);
        pthread_mutex_unlock(&p36_vk_mu);
    }
    if (!ok) return 0;
    aux = p36_gpu_quality ?
        p36_gpu_tensor_alloc(aux_floats * n_tok * sizeof(float)) :
        p36_gpu_tensor_alloc_scratch(aux_floats * n_tok * sizeof(float));
    if (!aux) return 0;

    if (!p36_gpu_quality && p36_vk_use_gpu_attn_post()) {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t qg_stride;
            uint32_t aux_stride;
            uint32_t has_sinks;
        } ppush = {
            pos0,
            n_head,
            head_dim,
            (uint32_t)(score_stride / sizeof(float)),
            (uint32_t)qg_stride,
            (uint32_t)aux_floats,
            has_sinks ? 1u : 0u,
        };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *sinks_t = sinks ? p36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) : (p36_gpu_tensor *)qg;
        ok = sinks_t != NULL;
        if (ok) {
            const p36_gpu_tensor *pbindings[4] = { scores, qg, sinks_t, aux };
            ok = p36_vk_run_unlocked("attn_post", &p36_vk.attn_post, pbindings, &ppush, sizeof(ppush), n_head, n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
    } else {
        p36_vk_prof_scope prof;
        p36_vk_prof_scope_begin(&prof, "attn_softmax_host");
        float *aux_host = p36_vk_tensor_contents_labeled(aux, "submit_wait_attn_softmax_aux");
        float *scores_host = p36_vk_tensor_contents_labeled(scores, "submit_wait_attn_softmax_scores");
        const float *qg_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)qg, "submit_wait_attn_softmax_qg");
        p36_vk_attn_host_ctx hctx;
        if (!aux_host || !scores_host || !qg_host) {
            p36_vk_prof_scope_end(&prof);
            p36_gpu_tensor_free(aux);
            return 0;
        }
        hctx.aux = aux_host;
        hctx.scores = scores_host;
        hctx.qg = qg_host;
        hctx.sinks = sinks;
        hctx.score_stride = score_stride / sizeof(float);
        hctx.qg_stride = qg_stride;
        hctx.aux_floats = aux_floats;
        hctx.pos0 = pos0;
        hctx.n_head = n_head;
        hctx.head_dim = head_dim;
        hctx.has_sinks = has_sinks;
        if (n_tok > 1) p36_gpu_parallel_for_rows(n_tok, 2, p36_vk_attn_host_rows, &hctx);
        else p36_vk_attn_host_rows(&hctx, 0, n_tok);
        p36_vk_prof_scope_end(&prof);
    }
    if (!ok) {
        p36_gpu_tensor_free(aux);
        return 0;
    }

    {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t v_type;
            uint32_t v_row_bytes;
        } rpush = { pos0, n_head, n_head_kv, head_dim, (uint32_t)(score_stride / sizeof(float)),
                    v_cache_type, v_cache_row_bytes };
        const p36_gpu_tensor *rbindings[4] = { scores, v_cache, aux, out };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("attn_reduce", &p36_vk.attn_reduce, rbindings, &rpush, sizeof(rpush),
                                 (head_dim + 63u) / 64u, n_head, n_tok);
        pthread_mutex_unlock(&p36_vk_mu);
    }
    p36_gpu_tensor_free(aux);
    return ok;
}

/* Expert router on the host: softmax, top-k, and routing-weight
 * normalization replicate p36_softmax_inplace() + p36_topk_indices() and the
 * weight loop in p36_forward_ffn() operation for operation.  256 scalars per
 * call; doing it on the host makes expert selection bit-equal to the CPU
 * engine, so a near-tie can never route to a different expert. */
typedef struct {
    uint32_t *selected;
    float *weights;
    const float *logits;
    uint32_t n_expert;
    uint32_t n_used;
    float route_scale;
} p36_vk_router_ctx;

static void p36_vk_router_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_router_ctx *ctx = (p36_vk_router_ctx *)opaque;
    float prob[256];
    uint32_t idx[8];
    float val[8];

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *lt = ctx->logits + tok * ctx->n_expert;
        uint32_t *st = ctx->selected + tok * ctx->n_used;
        float *wt = ctx->weights + tok * ctx->n_used;
        {
            float maxv = -INFINITY;
            double sum = 0.0;
            for (uint32_t i = 0; i < ctx->n_expert; i++) if (lt[i] > maxv) maxv = lt[i];
            for (uint32_t i = 0; i < ctx->n_expert; i++) {
                prob[i] = expf(lt[i] - maxv);
                sum += prob[i];
            }
            if (sum != 0.0) {
                float inv = (float)(1.0 / sum);
                for (uint32_t i = 0; i < ctx->n_expert; i++) prob[i] *= inv;
            }
        }
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            idx[i] = 0;
            val[i] = -INFINITY;
        }
        for (uint32_t i = 0; i < ctx->n_expert; i++) {
            float v = prob[i];
            for (uint32_t j = 0; j < ctx->n_used; j++) {
                if (v > val[j]) {
                    for (uint32_t t = ctx->n_used - 1; t > j; t--) {
                        idx[t] = idx[t - 1];
                        val[t] = val[t - 1];
                    }
                    idx[j] = i;
                    val[j] = v;
                    break;
                }
            }
        }
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < ctx->n_used; i++) sum += val[i];
            if (sum < 6.103515625e-5) sum = 6.103515625e-5;
            for (uint32_t i = 0; i < ctx->n_used; i++) {
                st[i] = idx[i];
                wt[i] = (float)((double)val[i] / sum) * ctx->route_scale;
            }
        }
    }
}

static void p36_vk_router_fast_rows(void *opaque, uint64_t row0, uint64_t row1) {
    p36_vk_router_ctx *ctx = (p36_vk_router_ctx *)opaque;
    uint32_t idx[8];
    float val[8];
    float top_exp[8];

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *lt = ctx->logits + tok * ctx->n_expert;
        uint32_t *st = ctx->selected + tok * ctx->n_used;
        float *wt = ctx->weights + tok * ctx->n_used;

        for (uint32_t i = 0; i < ctx->n_used; i++) {
            idx[i] = 0;
            val[i] = -INFINITY;
        }
        for (uint32_t i = 0; i < ctx->n_expert; i++) {
            float v = lt[i];
            for (uint32_t j = 0; j < ctx->n_used; j++) {
                if (v > val[j]) {
                    for (uint32_t t = ctx->n_used - 1; t > j; t--) {
                        idx[t] = idx[t - 1];
                        val[t] = val[t - 1];
                    }
                    idx[j] = i;
                    val[j] = v;
                    break;
                }
            }
        }

        double sum = 0.0;
        float maxv = val[0];
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            top_exp[i] = expf(val[i] - maxv);
            sum += top_exp[i];
        }
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            st[i] = idx[i];
            wt[i] = (float)((double)top_exp[i] / sum) * ctx->route_scale;
        }
    }
}

int p36_gpu_router_topk_tensor(p36_gpu_tensor *selected,
                               p36_gpu_tensor *weights,
                               const p36_gpu_tensor *logits,
                               uint32_t n_expert,
                               uint32_t n_used,
                               uint32_t n_tok,
                               float route_scale) {
    const float *logits_host;
    uint32_t *selected_host;
    float *weights_host;
    p36_vk_router_ctx ctx;
    p36_vk_prof_scope prof;
    int ok = 0;
    if (n_expert == 0 || n_expert > 256 || n_used == 0 || n_used > 8 || n_used > n_expert || n_tok == 0) return 0;
    if (!p36_gpu_tensor_range_ok(logits, 0, (uint64_t)n_tok * n_expert * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, (uint64_t)n_tok * n_used * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(weights, 0, (uint64_t)n_tok * n_used * sizeof(float))) {
        return 0;
    }
    /* Fast-mode routing runs on GPU for every batch size so the expert
     * selection never crosses the host and a token gets identical routing
     * numerics whether it arrives in a prefill chunk, a chunk tail, or a
     * single decode step (warm/cold session equality depends on that). */
    if (!p36_gpu_quality && p36_vk_env_default_on("P36_VK_ROUTER_GPU")) {
        struct {
            uint32_t n_expert;
            uint32_t n_used;
            uint32_t n_tok;
            float route_scale;
        } push = { n_expert, n_used, n_tok, route_scale };
        const p36_gpu_tensor *bindings[3] = { logits, selected, weights };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("router_topk", &p36_vk.router_topk, bindings, &push, sizeof(push),
                                 n_tok, 1, 1);
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) return 1;
    }

    p36_vk_prof_scope_begin(&prof, "router_topk_host");
    logits_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)logits, "submit_wait_router_logits");
    selected_host = p36_vk_tensor_contents_labeled(selected, "submit_wait_router_selected");
    weights_host = p36_vk_tensor_contents_labeled(weights, "submit_wait_router_weights");
    if (!logits_host || !selected_host || !weights_host) goto done;

    ctx.selected = selected_host;
    ctx.weights = weights_host;
    ctx.logits = logits_host;
    ctx.n_expert = n_expert;
    ctx.n_used = n_used;
    ctx.route_scale = route_scale;
    p36_gpu_parallel_fn fn = (!p36_gpu_quality && n_tok >= 32u && p36_vk_use_router_fast_prefill()) ? p36_vk_router_fast_rows : p36_vk_router_rows;
    if ((uint64_t)n_tok * n_expert >= 1024u) p36_gpu_parallel_for_rows(n_tok, 2, fn, &ctx);
    else fn(&ctx, 0, n_tok);
    ok = 1;
done:
    p36_vk_prof_scope_end(&prof);
    return ok;
}

int p36_gpu_top2_tensor(p36_gpu_tensor *out_ids,
                        const p36_gpu_tensor *logits,
                        uint32_t count) {
    if (!p36_vk_env_default_on("P36_VK_ARGMAX")) return 0;
    const uint32_t groups = count < 32768u ?
        (count + 255u) / 256u : 128u;
    const uint64_t bytes = (uint64_t)groups * 16u;
    int ok = count != 0u && groups <= 256u &&
             p36_gpu_tensor_range_ok(logits, 0, (uint64_t)count * sizeof(float)) &&
             p36_gpu_tensor_range_ok(out_ids, 0, 2u * sizeof(int32_t));
    if (!ok) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.top2_part || p36_vk.top2_part_bytes < bytes) {
        p36_vk_tensor_release_unlocked(p36_vk.top2_part);
        p36_vk.top2_part = p36_vk_tensor_alloc_scratch_unlocked(bytes);
        p36_vk.top2_part_bytes = p36_vk.top2_part ? bytes : 0;
    }
    if (p36_vk.top2_part) {
        const p36_gpu_tensor *bindings[3] = {
            logits, p36_vk.top2_part, out_ids,
        };
        struct {
            uint32_t count;
            uint32_t groups;
            uint32_t pass;
        } push = { count, groups, 0u };
        ok = p36_vk_run_unlocked("top2_stage1", &p36_vk.top2,
                                 bindings, &push, sizeof(push), groups, 1, 1);
        if (ok) {
            push.pass = 1u;
            ok = p36_vk_run_unlocked("top2_stage2", &p36_vk.top2,
                                     bindings, &push, sizeof(push), 1, 1, 1);
        }
    } else {
        ok = 0;
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_logits_minp_pack(p36_gpu_tensor *out, const p36_gpu_tensor *logits,
                             uint32_t count, float temperature,
                             float reject_scaled) {
    if (!p36_vk_env_default_on("P36_VK_MINP_PACK")) return 0;
    const uint32_t nb = P36_GPU_MINP_NB(count);
    const uint64_t bytes = P36_GPU_MINP_BYTES(count);
    int ok = count != 0u && nb != 0u && temperature > 0.0f &&
             p36_gpu_tensor_range_ok(logits, 0, (uint64_t)count * sizeof(float)) &&
             p36_gpu_tensor_range_ok(out, 0, bytes);
    if (!ok) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    {
        /* One entry point, three passes: a block max, a global max over those,
         * then the filter and the per-block compaction (see the shader). */
        const p36_gpu_tensor *bindings[2] = { logits, out };
        struct {
            uint32_t count;
            uint32_t nb;
            uint32_t cap;
            uint32_t pass;
            float    temperature;
            float    reject_scaled;
        } push = { count, nb, P36_GPU_MINP_CAP, 0u, temperature, reject_scaled };
        ok = p36_vk_run_unlocked("logits_minp_max", &p36_vk.logits_minp,
                                 bindings, &push, sizeof(push), nb, 1, 1);
        if (ok) {
            push.pass = 1u;
            ok = p36_vk_run_unlocked("logits_minp_reduce", &p36_vk.logits_minp,
                                     bindings, &push, sizeof(push), 1, 1, 1);
        }
        if (ok) {
            push.pass = 2u;
            ok = p36_vk_run_unlocked("logits_minp_pack", &p36_vk.logits_minp,
                                     bindings, &push, sizeof(push), nb, 1, 1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_topk8_tensor(p36_gpu_tensor *out, const p36_gpu_tensor *logits,
                         uint32_t count) {
    if (!p36_vk_env_default_on("P36_VK_TOPK8")) return 0;
    const uint32_t groups = count < 32768u ?
        (count + 255u) / 256u : 128u;
    const uint64_t bytes = (uint64_t)groups * (8u * (sizeof(float) + sizeof(int32_t)));
    int ok = count != 0u && groups <= 256u &&
             p36_gpu_tensor_range_ok(logits, 0, (uint64_t)count * sizeof(float)) &&
             p36_gpu_tensor_range_ok(out, 0, 8u * (sizeof(int32_t) + sizeof(float)));
    if (!ok) return 0;

    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.topk8_part || p36_vk.topk8_part_bytes < bytes) {
        p36_vk_tensor_release_unlocked(p36_vk.topk8_part);
        p36_vk.topk8_part = p36_vk_tensor_alloc_scratch_unlocked(bytes);
        p36_vk.topk8_part_bytes = p36_vk.topk8_part ? bytes : 0;
    }
    if (p36_vk.topk8_part) {
        /* Output (binding 2) is one buffer: int ids[8] then float vals[8],
         * matching topk8.comp's Output block exactly — the caller reads both
         * halves out of the same `out` tensor at their respective offsets. */
        const p36_gpu_tensor *bindings[3] = {
            logits, p36_vk.topk8_part, out,
        };
        struct {
            uint32_t count;
            uint32_t groups;
            uint32_t pass;
        } push = { count, groups, 0u };
        ok = p36_vk_run_unlocked("topk8_stage1", &p36_vk.topk8,
                                 bindings, &push, sizeof(push), groups, 1, 1);
        if (ok) {
            push.pass = 1u;
            ok = p36_vk_run_unlocked("topk8_stage2", &p36_vk.topk8,
                                     bindings, &push, sizeof(push), 1, 1, 1);
        }
    } else {
        ok = 0;
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_recurrent_norm_gate_tensor(
        p36_gpu_tensor *state, const p36_gpu_tensor *gate,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t width,
        uint32_t rows, float eps) {
    if (p36_gpu_quality || !p36_vk_env_default_on("P36_VK_RECURRENT_NORM_GATE") ||
        width == 0u || rows == 0u) return 0;
    uint64_t elems;
    uint64_t bytes;
    if (!p36_u64_mul_ok(width, rows, &elems) ||
        !p36_u64_mul_ok(elems, sizeof(float), &bytes) ||
        !p36_gpu_tensor_range_ok(state, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes)) return 0;
    const float *w = (const float *)p36_gpu_weight_bytes(
        model_map, model_size, weight_offset, (uint64_t)width * sizeof(float));
    if (!w) return 0;

    struct {
        uint32_t width;
        uint32_t rows;
        float eps;
    } push = { width, rows, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(
        w, (uint64_t)width * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[3] = { state, gate, wt };
        ok = p36_vk_run_unlocked("recurrent_norm_gate", &p36_vk.recur_norm_gate,
                                 bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_recurrent_norm_gate_q8_k_tensor(
        p36_gpu_tensor *state, p36_gpu_tensor *q8,
        const p36_gpu_tensor *gate,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, float eps) {
    if (p36_gpu_quality ||
        !p36_vk_env_default_on("P36_VK_RECURRENT_NORM_GATE_Q8") ||
        !rows || (rows & 1u)) return 0;
    uint64_t bytes = (uint64_t)rows * 128u * sizeof(float);
    uint64_t q8_bytes = (uint64_t)(rows / 2u) * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(state, 0, bytes) ||
        !p36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    const float *w = (const float *)p36_gpu_weight_bytes(
        model_map, model_size, weight_offset, 128u * sizeof(float));
    if (!w) return 0;
    struct {
        uint32_t rows;
        float eps;
    } push = { rows, eps };
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *wt = p36_vk_weight_get_unlocked(w, 128u * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[4] = { state, gate, wt, q8 };
        ok = p36_vk_run_unlocked(
            "recurrent_norm_gate_q8_k", &p36_vk.recur_norm_gate_q8_k,
            bindings, &push, sizeof(push), rows / 2u, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static uint64_t p36_vk_iq_block_bytes(uint32_t type) {
    switch (type) {
    case P36_VK_TENSOR_Q2_K: return 84;
    case P36_VK_TENSOR_Q3_K: return 110;
    case P36_VK_TENSOR_Q4_K: return 144;
    case P36_VK_TENSOR_Q5_K: return 176;
    case P36_VK_TENSOR_Q6_K: return 210;
    case P36_VK_TENSOR_IQ2_XXS: return 66;
    case P36_VK_TENSOR_IQ2_XS: return 74;
    case P36_VK_TENSOR_IQ3_XXS: return 98;
    case P36_VK_TENSOR_IQ1_S: return 50;
    case P36_VK_TENSOR_IQ4_NL: return 144;
    case P36_VK_TENSOR_IQ3_S: return 110;
    case P36_VK_TENSOR_IQ2_S: return 82;
    case P36_VK_TENSOR_IQ4_XS: return 136;
    case P36_VK_TENSOR_IQ1_M: return 56;
    case P36_VK_TENSOR_Q2_0: return 72;
    case P36_VK_TENSOR_PQ2_0: return 68;
    case P36_VK_TENSOR_PTQ1_0: return 56;
    default: return 0;
    }
}

static p36_gpu_tensor *p36_vk_quantize_q8_k_alloc_tensor(const p36_gpu_tensor *x,
                                                         uint64_t in_dim,
                                                         uint64_t n_tok) {
    uint64_t blocks = in_dim / P36_VK_QK_K;
    uint64_t q8_count = 0;
    uint64_t q8_bytes = 0;
    p36_gpu_tensor *q8;

    if (!p36_u64_mul_ok(n_tok, blocks, &q8_count) ||
        !p36_u64_mul_ok(q8_count, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return NULL;
    }
    q8 = p36_gpu_tensor_alloc_scratch(q8_bytes);
    if (!q8) return NULL;
    if (!p36_gpu_quantize_q8_k_tensor(q8, x, in_dim, n_tok)) {
        p36_gpu_tensor_free(q8);
        return NULL;
    }
    return q8;
}

/* Expert-major routed-pair tiles for moe_matvec: workgroup y owns one expert
 * and up to P36_VK_MOE_PAIR_TILE pair ids, so the shader can load each expert
 * row block once and update that many accumulators before advancing. */
/* Upper bound on the packed tile count without reading the schedule back:
 * sum(ceil(c_e/4)) <= floor(n_slot/4) + #experts hit.  Expert-major packing
 * is dense from tile 0, so consumers can dispatch this bound instead of the
 * n_slot worst case and skip most dead-tile workgroups. */
static uint32_t p36_vk_moe_tile_bound(uint32_t n_slot, uint32_t n_expert, uint32_t pair_tile) {
    uint32_t live = n_expert < n_slot ? n_expert : n_slot;
    uint32_t bound = n_slot / pair_tile + live;
    return bound < n_slot ? bound : n_slot;
}

/* GPU tile schedule straight from the router selection, no host read.
 * Unused trailing tiles keep the invalid marker and the kernels exit early.
 * With slot_map (SSD streaming) the tile expert fields carry cache slot ids
 * so the consumers index the stream cache buffers directly. */
static p36_gpu_tensor *p36_vk_moe_tiles_gpu(const p36_gpu_tensor *selected, uint32_t n_slot, uint32_t n_expert,
                                            const uint32_t *slot_map, uint32_t pair_tile) {
    uint32_t tile_bound;
    uint64_t tile_words;
    uint64_t words;
    if (n_slot == 0 || n_slot > 8192u || n_expert == 0 || n_expert > 256u) return NULL;
    tile_bound = p36_vk_moe_tile_bound(n_slot, n_expert, pair_tile);
    tile_words = 1u + pair_tile;
    if (!p36_u64_mul_ok(tile_bound, tile_words, &words)) return NULL;
    p36_gpu_tensor *tiles = p36_gpu_tensor_alloc_scratch(words * sizeof(uint32_t));
    p36_gpu_tensor *map = NULL;
    if (!tiles) return NULL;
    if (slot_map) {
        map = p36_gpu_tensor_alloc(256u * sizeof(uint32_t));
        if (!map || !p36_gpu_tensor_write(map, 0, slot_map, 256u * sizeof(uint32_t))) {
            p36_gpu_tensor_free(map);
            p36_gpu_tensor_free(tiles);
            return NULL;
        }
    }
    struct {
        uint32_t n_slot;
        uint32_t n_expert;
        uint32_t streamed;
        uint32_t pair_tile;
        uint32_t tile_bound;
    } push = { n_slot, n_expert, slot_map ? 1u : 0u, pair_tile, tile_bound };
    const p36_gpu_tensor *bindings[3] = { selected, tiles, map ? map : tiles };
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("moe_tiles", &p36_vk.moe_tiles, bindings, &push, sizeof(push), 1, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(map);
    if (!ok) {
        p36_gpu_tensor_free(tiles);
        return NULL;
    }
    return tiles;
}

static p36_gpu_tensor *p36_vk_moe_build_tiles_ex(const uint32_t *selected_host,
                                                  uint64_t n_slot,
                                                  uint32_t n_expert,
                                                  const uint32_t *expert_slot_map,
                                                  uint32_t pair_tile,
                                                  bool skip_unmapped,
                                                  uint32_t *tile_count_out) {
    uint32_t counts[256] = {0};
    uint32_t offsets[257];
    uint32_t next[256];
    uint32_t *grouped_pairs = NULL;
    p36_gpu_tensor *tiles = NULL;
    uint32_t *tiles_host;
    uint64_t words = 0;
    uint64_t bytes = 0;
    uint32_t tile = 0;

    if (!selected_host || !tile_count_out || n_expert == 0 || n_expert > 256 || n_slot == 0 || n_slot > UINT32_MAX) {
        return NULL;
    }
    for (uint64_t pair = 0; pair < n_slot; pair++) {
        uint32_t expert = selected_host[pair];
        if (skip_unmapped && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) continue;
        counts[expert]++;
    }
    offsets[0] = 0;
    *tile_count_out = 0;
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        offsets[expert + 1u] = offsets[expert] + counts[expert];
        *tile_count_out += (counts[expert] + pair_tile - 1u) / pair_tile;
    }
    if (*tile_count_out == 0) return NULL;

    memcpy(next, offsets, (size_t)n_expert * sizeof(next[0]));
    grouped_pairs = malloc((size_t)n_slot * sizeof(grouped_pairs[0]));
    if (!grouped_pairs) return NULL;
    for (uint32_t pair = 0; pair < (uint32_t)n_slot; pair++) {
        uint32_t expert = selected_host[pair];
        if (skip_unmapped && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) continue;
        grouped_pairs[next[expert]++] = pair;
    }
    if (!p36_u64_mul_ok(*tile_count_out, 1u + pair_tile, &words) ||
        !p36_u64_mul_ok(words, sizeof(uint32_t), &bytes)) {
        free(grouped_pairs);
        return NULL;
    }

    tiles = p36_gpu_tensor_alloc(bytes);
    tiles_host = tiles ? p36_vk_tensor_contents_labeled(tiles, "submit_wait_moe_tiles_build") : NULL;
    if (!tiles || !tiles_host) {
        p36_gpu_tensor_free(tiles);
        free(grouped_pairs);
        return NULL;
    }
    memset(tiles_host, 0xff, (size_t)bytes);
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        if (counts[expert] != 0 && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) {
            free(grouped_pairs);
            p36_gpu_tensor_free(tiles);
            return NULL;
        }
        for (uint32_t i = offsets[expert]; i < offsets[expert + 1u]; i += pair_tile) {
            uint32_t base = tile * (1u + pair_tile);
            tiles_host[base] = expert_slot_map ? expert_slot_map[expert] : expert;
            for (uint32_t u = 0; u < pair_tile && i + u < offsets[expert + 1u]; u++) {
                tiles_host[base + 1u + u] = grouped_pairs[i + u];
            }
            tile++;
        }
    }
    free(grouped_pairs);
    return tiles;
}

static p36_gpu_tensor *p36_vk_moe_build_tiles(const uint32_t *selected_host,
                                               uint64_t n_slot,
                                               uint32_t n_expert,
                                               const uint32_t *expert_slot_map,
                                               uint32_t *tile_count_out) {
    return p36_vk_moe_build_tiles_ex(selected_host, n_slot, n_expert,
                                     expert_slot_map, P36_VK_MOE_PAIR_TILE,
                                     false, tile_count_out);
}

/* One routed expert matvec batch over n_tok tokens.  Pairs are grouped by
 * expert on the host so workgroup y can own one expert and up to
 * P36_VK_MOE_PAIR_TILE pair ids, loading each expert row block once and
 * updating that many q8 accumulation chains before advancing.
 *
 * The whole expert bank and its .scale tensor are cached on device through
 * the weight cache and the router's expert ids index them directly, so a
 * decode step moves no expert bytes at all.  If the bank does not fit the
 * heaps the runtime falls back to packing one copy per selected expert. */
static int p36_vk_moe_matvec(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const p36_gpu_moe_weight *w,
                              const p36_gpu_tensor *qx,
                              bool x_per_expert,
                              bool fast_allowed,
                              p36_vk_stream_part stream_part,
                              const uint32_t *stream_slot_map,
                              bool resident_bank,
                              const p36_gpu_tensor *selected,
                              uint32_t n_used,
                              uint32_t n_tok,
                             uint32_t in_dim,
                             uint32_t out_dim,
                             uint32_t n_expert) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(w->type);
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    uint64_t packed_weight_bytes = 0;
    uint64_t packed_scale_bytes = 0;
    const unsigned char *source;
    const unsigned char *scales_source = NULL;
    const uint32_t *selected_host;
    p36_gpu_tensor *tiles = NULL;
    p36_gpu_tensor *packed_weights = NULL;
    p36_gpu_tensor *packed_scales = NULL;
    const uint32_t *tiles_host;
    uint32_t tile_count = 0;
    uint32_t packed_count = 0;
    uint32_t packed_expert[256];
    uint32_t packed_slot[256];
    unsigned char *packed_weight_host;
    float *packed_scales_host = NULL;

    if (block_bytes == 0 || in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || out_dim == 0 || n_tok == 0) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * out_dim;
    if (expert_stride > UINT32_MAX ||
        !p36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    source = p36_gpu_weight_bytes(model_map, model_size, w->offset, weight_bytes);
    if (!source) return 0;
    if (w->has_scales) {
        scales_source = p36_gpu_weight_bytes(model_map, model_size, w->scales_offset, scale_bytes);
        if (!scales_source) return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, n_slot * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (x_per_expert ? n_slot : n_tok) * blocks * sizeof(p36_vk_block_q8_K))) {
        return 0;
    }

    tiles_host = NULL;
    if (!p36_gpu_quality && n_slot <= 4096u && stream_part == P36_VK_STREAM_PART_NONE &&
        (!p36_vk_stream.enabled || resident_bank) && p36_vk_moe_bank_cache) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, P36_VK_MOE_PAIR_TILE);
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, P36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_moe_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = p36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_part != P36_VK_STREAM_PART_NONE ? stream_slot_map : NULL,
                                       &tile_count);
        tiles_host = tiles ? p36_vk_tensor_contents_labeled(tiles, "submit_wait_moe_tiles_read") : NULL;
        if (!tiles || !tiles_host || tile_count == 0) {
            p36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t type;
        uint32_t n_used;
        uint32_t has_scales;
        uint32_t x_per_expert;
        uint32_t packed_weights;
    } push = {
        out_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        w->type, n_used, w->has_scales ? 1u : 0u, x_per_expert ? 1u : 0u, 0u,
    };

    if (stream_part != P36_VK_STREAM_PART_NONE && stream_slot_map) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        const p36_gpu_tensor *bank = p36_vk_stream_weight_tensor_unlocked(stream_part);
        const p36_gpu_tensor *bank_scales = p36_vk_stream_scale_tensor_unlocked(stream_part);
        int ok = tables && bank && (!w->has_scales || bank_scales);
        if (ok) {
            const p36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, w->has_scales ? bank_scales : bank, out, tables,
            };
            p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
            ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down_stream" :
                                     (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec_stream" : "moe_matvec_stream"),
                                     kernel, bindings, &push, sizeof(push),
                                     (out_dim + 7u) / 8u, tile_count, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) {
            p36_gpu_tensor_free(tiles);
            return 1;
        }
    }

    if ((!p36_vk_stream.enabled || resident_bank) && p36_vk_moe_bank_cache) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *bank = tables ? p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        p36_gpu_tensor *bank_scales = NULL;
        if (bank && w->has_scales) {
            bank_scales = p36_vk_weight_get_unlocked(scales_source, scale_bytes);
            if (!bank_scales) bank = NULL;
        }
        if (bank) {
            const p36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, bank_scales ? bank_scales : bank, out, tables,
            };
            p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
            int ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                         (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                         kernel, bindings, &push, sizeof(push),
                                         (out_dim + 7u) / 8u, tile_count, 1);
            pthread_mutex_unlock(&p36_vk_mu);
            p36_gpu_tensor_free(tiles);
            return ok;
        }
        p36_vk_moe_bank_cache = false;
        pthread_mutex_unlock(&p36_vk_mu);
        fprintf(stderr, "p36: Vulkan expert bank cache exhausted device memory, "
                        "falling back to per-call expert staging\n");
    }

    /* Packed per-call staging needs expert ids on the host; GPU-built tiles
     * only serve the resident-bank dispatch above. */
    if (!tiles_host) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    memset(packed_slot, 0xff, sizeof(packed_slot));
    for (uint64_t slot = 0; slot < n_slot; slot++) {
        uint32_t expert = selected_host[slot];
        if (packed_slot[expert] != UINT32_MAX) continue;
        packed_slot[expert] = packed_count;
        packed_expert[packed_count++] = expert;
    }
    p36_gpu_tensor_free(tiles);
    tiles = p36_vk_moe_build_tiles(selected_host, n_slot, n_expert,
                                   packed_slot, &tile_count);
    if (!tiles || tile_count == 0 ||
        !p36_u64_mul_ok(packed_count, expert_stride, &packed_weight_bytes)) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    packed_weights = p36_gpu_tensor_alloc(packed_weight_bytes);
    if (w->has_scales) {
        packed_scale_bytes = (uint64_t)packed_count * sizeof(float);
        packed_scales = p36_gpu_tensor_alloc(packed_scale_bytes);
    }
    packed_weight_host = packed_weights ? p36_vk_tensor_contents_labeled(packed_weights, "submit_wait_moe_pack_weights") : NULL;
    packed_scales_host = packed_scales ? p36_vk_tensor_contents_labeled(packed_scales, "submit_wait_moe_pack_scales") : NULL;
    if (!packed_weights || !packed_weight_host ||
        (w->has_scales && (!packed_scales || !packed_scales_host))) {
        p36_gpu_tensor_free(packed_scales);
        p36_gpu_tensor_free(packed_weights);
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    for (uint32_t slot = 0; slot < packed_count; slot++) {
        const uint32_t expert = packed_expert[slot];
        memcpy(packed_weight_host + (uint64_t)slot * expert_stride,
               source + (uint64_t)expert * expert_stride,
               (size_t)expert_stride);
        if (packed_scales_host) packed_scales_host[slot] = ((const float *)scales_source)[expert];
    }
    push.packed_weights = 0u;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    int ok = tables != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            packed_weights,
            qx,
            tiles,
            w->has_scales ? packed_scales : packed_weights,
            out,
            tables,
        };
        p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
        ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                 (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                 kernel, bindings, &push, sizeof(push),
                                 (out_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(packed_scales);
    p36_gpu_tensor_free(packed_weights);
    p36_gpu_tensor_free(tiles);
    return ok;
}

/* IQ2_S sibling of p36_gpu_matmul_q8_0_pair_scaled_tensor: two
 * independently-scaled IQ2_S rows sharing one Q8_K activation read, one
 * dispatch. q8 must already be a valid Q8_K quantization of x at in_dim;
 * the caller owns quantizing it once and reusing it for both legs, which
 * is itself part of the win over two separate calls each re-quantizing. */
int p36_gpu_matmul_iq2s_pair_scaled_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_a_dim,
                                           uint64_t out_b_dim,
                                           const p36_gpu_tensor *q8,
                                           float scale_a,
                                           float scale_b) {
    if (p36_gpu_quality || !p36_vk_env_default_on("P36_VK_IQ2S_PAIR") ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        out_a_dim == 0 || out_b_dim == 0 ||
        in_dim > UINT32_MAX || out_a_dim > UINT32_MAX || out_b_dim > UINT32_MAX) return 0;
    uint64_t blocks = in_dim / P36_VK_QK_K;
    uint64_t block_bytes = p36_vk_iq_block_bytes(P36_VK_TENSOR_IQ2_S);
    uint64_t row_bytes, weight_a_bytes, weight_b_bytes, out_a_bytes, out_b_bytes, q8_bytes;
    if (!p36_u64_mul_ok(blocks, block_bytes, &row_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_a_dim, &weight_a_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_b_dim, &weight_b_bytes) ||
        !p36_u64_mul_ok(out_a_dim, sizeof(float), &out_a_bytes) ||
        !p36_u64_mul_ok(out_b_dim, sizeof(float), &out_b_bytes) ||
        !p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_K), &q8_bytes) ||
        row_bytes > UINT32_MAX) return 0;
    if (!p36_gpu_tensor_range_ok(q8, 0, q8_bytes) ||
        !p36_gpu_tensor_range_ok(out_a, 0, out_a_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_b_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_a_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_b_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_a;
        uint32_t out_b;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale_a;
        float scale_b;
    } push = { (uint32_t)out_a_dim, (uint32_t)out_b_dim, (uint32_t)blocks,
               (uint32_t)row_bytes, scale_a, scale_b };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights_a = tables ? p36_vk_weight_get_unlocked(wa, weight_a_bytes) : NULL;
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_b_bytes) : NULL;
    int ok = weights_a != NULL && weights_b != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = { weights_a, weights_b, q8, tables, out_a, out_b };
        uint32_t groups = (uint32_t)(out_a_dim > out_b_dim ? out_a_dim : out_b_dim);
        ok = p36_vk_run_unlocked("dense_extra_decode_iq2s_pair",
                                 &p36_vk.dense_extra_decode_iq2s_pair,
                                 bindings, &push, sizeof(push), groups, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_iq_quant_q8_scaled_tensor(p36_gpu_tensor *out,
                                              const void *model_map,
                                              uint64_t model_size,
                                              uint64_t weight_offset,
                                              uint32_t weight_type,
                                              uint64_t in_dim,
                                              uint64_t out_dim,
                                              const p36_gpu_tensor *q8,
                                              uint64_t n_tok,
                                              float scale) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(weight_type);
    uint64_t blocks, row_bytes, weight_bytes, out_bytes, q8_bytes;
    uint32_t tile_count;
    p36_gpu_tensor *tiles;
    uint32_t *tp;
    const unsigned char *source;
    int ok;

bool extra_type = weight_type == P36_VK_TENSOR_IQ2_XXS ||
                      weight_type == P36_VK_TENSOR_IQ2_XS ||
                      weight_type == P36_VK_TENSOR_IQ2_S ||
                      weight_type == P36_VK_TENSOR_IQ1_S ||
                      weight_type == P36_VK_TENSOR_IQ4_NL ||
                      weight_type == P36_VK_TENSOR_PQ2_0 ||
                      weight_type == P36_VK_TENSOR_PTQ1_0 ||
                      /* IQ1_M prefill uses the shared 128-token mmq tile; the
                       * naive dense_iq1_m re-read the weights once per token.
                       * Decode and 2..8-row batches (MTP verify) stay on
                       * dense_iq1_m, row-identical to one-token decode. */
                      (weight_type == P36_VK_TENSOR_IQ1_M && n_tok > 8u &&
                       !p36_gpu_quality && p36_vk_env_default_on("P36_VK_DENSE_IQ1M_MMQ"));
    if ((weight_type != P36_VK_TENSOR_IQ3_XXS &&
         weight_type != P36_VK_TENSOR_IQ3_S &&
         weight_type != P36_VK_TENSOR_IQ4_XS &&
         weight_type != P36_VK_TENSOR_IQ1_M && !extra_type) ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        out_dim == 0 || n_tok == 0 || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes) ||
        weight_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(n_tok, out_dim, &out_bytes) ||
        !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    if (extra_type) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *weights = tables ?
            p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            const char *op = p36_vk_prof_iq3_shape(
                weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                (uint32_t)out_dim, n_tok == 1u ?
                    "dense_extra_decode" : "dense_extra_mmq");
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    uint32_t type;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, weight_type, scale,
                };
bool q2_family = weight_type == P36_VK_TENSOR_Q2_0 ||
                                  weight_type == P36_VK_TENSOR_PQ2_0;
                bool ptq1_0 = weight_type == P36_VK_TENSOR_PTQ1_0;
                /* The generic dense_extra_decode shader knows Q2_0 but not
                 * PQ2_0, and silently misreading type 142 as type 42 would
                 * produce plausible garbage.  Refuse loudly instead: the
                 * specialized kernel is the only correct PQ2_0 path, and
                 * teaching the generic shader would change SPIR-V that the
                 * IQ2/IQ3 models also run. */
                if (weight_type == P36_VK_TENSOR_PQ2_0 &&
                    !(p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                      p36_vk.subgroup_arithmetic)) {
                    fprintf(stderr, "p36: pq2_0 needs integer dot product, "
                                    "subgroup size 64 and subgroup arithmetic\n");
                    pthread_mutex_unlock(&p36_vk_mu);
                    return 0;
                }
                if (ptq1_0 &&
                    !(p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                      p36_vk.subgroup_arithmetic)) {
                    fprintf(stderr, "p36: ptq1_0 needs integer dot product, "
                                    "subgroup size 64 and subgroup arithmetic\n");
                    pthread_mutex_unlock(&p36_vk_mu);
                    return 0;
                }
                bool iq2s_fast = weight_type == P36_VK_TENSOR_IQ2_S &&
                                 p36_vk_env_default_on("P36_VK_IQ2S_EXTRA");
                p36_vk_kernel *decode_kernel =
                    q2_family &&
                    p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                    p36_vk.subgroup_arithmetic ?
                        (weight_type == P36_VK_TENSOR_PQ2_0 ?
                            &p36_vk.dense_extra_decode_pq2_0 :
                            &p36_vk.dense_extra_decode_q2_0) :
                    ptq1_0 && p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                    p36_vk.subgroup_arithmetic ?
                        &p36_vk.dense_extra_decode_ptq1_0 :
                    iq2s_fast ? &p36_vk.dense_extra_decode_iq2s :
                        &p36_vk.dense_extra_decode;
                ok = p36_vk_run_unlocked(
                    op, decode_kernel,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    uint32_t type;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes,
                    weight_type, scale,
                };
                bool is_pq2_0 = weight_type == P36_VK_TENSOR_PQ2_0;
                bool is_ptq1_0 = weight_type == P36_VK_TENSOR_PTQ1_0;
                bool is_q2_0 = weight_type == P36_VK_TENSOR_Q2_0 || is_pq2_0;
                /* Small batches (MTP verify, prompt tails, short incremental
                 * prefills) would pay a whole 128-token mmq tile.  Stream the
                 * weights once per 8-token group instead; each row is
                 * bit-identical to the one-token decode kernel. */
                bool small_ok = is_q2_0 && p36_vk.have_int_dot &&
                                p36_vk.subgroup_size == 64u && p36_vk.subgroup_arithmetic;
                /* A ragged tail past whole 128-token tiles (the end of any
                 * prompt) would cost another full tile; hand it to the small
                 * kernel and let mmq take only the aligned head. */
                uint32_t small_first = 0;
                if (small_ok && n_tok > p36_vk_q2_small_max()) {
                    uint32_t rem = (uint32_t)n_tok % 128u;
                    if (rem == 0u || rem > p36_vk_q2_small_max()) small_ok = false;
                    else small_first = (uint32_t)n_tok - rem;
                }
                if (small_ok && small_first) {
                    push.n_tok = small_first;
                    ok = p36_vk_run_unlocked(
                        op, is_pq2_0 ? &p36_vk.dense_extra_mmq_pq2_0 : &p36_vk.dense_extra_mmq_q2_0,
                        bindings, &push, sizeof(push),
                        ((uint32_t)out_dim + 63u) / 64u, small_first / 128u, 1);
                }
                if (small_ok) {
                    p36_vk_kernel *small = is_pq2_0 ? &p36_vk.dense_extra_small_pq2_0 :
                                                      &p36_vk.dense_extra_small_q2_0;
                    for (uint32_t tok0 = small_first; ok && tok0 < (uint32_t)n_tok; tok0 += 8u) {
                        uint32_t group = (uint32_t)n_tok - tok0;
                        if (group > 8u) group = 8u;
                        struct {
                            uint32_t out_dim;
                            uint32_t n_tok;
                            uint32_t tok0;
                            uint32_t blocks;
                            uint32_t row_bytes;
                            float scale;
                        } spush = {
                            (uint32_t)out_dim, group, tok0,
                            (uint32_t)blocks, (uint32_t)row_bytes, scale,
                        };
                        ok = p36_vk_run_unlocked(op, small, bindings, &spush, sizeof(spush),
                                                 ((uint32_t)out_dim + 1u) / 2u, 1, 1);
                    }
                    pthread_mutex_unlock(&p36_vk_mu);
                    return ok;
                }
                bool is_iq2s_fast = weight_type == P36_VK_TENSOR_IQ2_S &&
                                    p36_vk_env_default_on("P36_VK_IQ2S_EXTRA");
                p36_vk_kernel *mmq_kernel = is_ptq1_0 ?
                    &p36_vk.dense_extra_mmq_ptq1_0 :
                    (is_pq2_0 ?
                        &p36_vk.dense_extra_mmq_pq2_0 :
                        (is_q2_0 ? &p36_vk.dense_extra_mmq_q2_0 :
                         is_iq2s_fast ? &p36_vk.dense_extra_mmq_iq2s :
                                   &p36_vk.dense_extra_mmq));
                ok = p36_vk_run_unlocked(
                    op, mmq_kernel,
                    bindings, &push, sizeof(push),
                    is_ptq1_0 ? ((uint32_t)out_dim + 31u) / 32u :
                    (is_q2_0 ? ((uint32_t)out_dim + 63u) / 64u :
                                ((uint32_t)out_dim + 31u) / 32u),
                    is_ptq1_0 ? ((uint32_t)n_tok + 63u) / 64u :
                                ((uint32_t)n_tok + 127u) / 128u,
                    1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (weight_type == P36_VK_TENSOR_IQ4_XS && !p36_gpu_quality &&
        p36_vk_use_dense_iq3_fast() && (n_tok != 1u || out_dim % 4u == 0u)) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { weights, q8, out };
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, scale,
                };
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq4_xs_decode");
                ok = p36_vk_run_unlocked(
                    op, &p36_vk.dense_iq4_xs_decode,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                ok = p36_vk_run_unlocked(
                    "dense_iq4_xs_mmq", &p36_vk.dense_iq4_xs_mmq,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 31u) / 32u,
                    ((uint32_t)n_tok + 127u) / 128u, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (weight_type == P36_VK_TENSOR_IQ4_XS ||
        weight_type == P36_VK_TENSOR_IQ1_M) {
        struct {
            uint32_t out_dim;
            uint32_t n_tok;
            uint32_t blocks;
            uint32_t row_bytes;
            float scale;
        } push = {
            (uint32_t)out_dim, (uint32_t)n_tok,
            (uint32_t)blocks, (uint32_t)row_bytes, scale,
        };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = weight_type == P36_VK_TENSOR_IQ1_M ?
            p36_vk_iq_tables_unlocked() : NULL;
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        ok = weights != NULL &&
             (weight_type != P36_VK_TENSOR_IQ1_M || tables != NULL);
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ4_XS ?
                &p36_vk.dense_iq4_xs : &p36_vk.dense_iq1_m;
            ok = p36_vk_run_unlocked(
                weight_type == P36_VK_TENSOR_IQ4_XS ? "dense_iq4_xs" : "dense_iq1_m",
                kernel, bindings, &push, sizeof(push),
                (uint32_t)out_dim, (uint32_t)n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if ((weight_type == P36_VK_TENSOR_IQ3_XXS ||
         weight_type == P36_VK_TENSOR_IQ3_S) && !p36_gpu_quality &&
        p36_vk_use_dense_iq3_fast()) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *weights = tables ?
            p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, scale,
                };
                bool rows1 = weight_type == P36_VK_TENSOR_IQ3_S &&
                             out_dim == 48u;
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u r4/mv\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                bool rows4 = weight_type == P36_VK_TENSOR_IQ3_S &&
                             ((out_dim == 5120 && in_dim >= 6144) ||
                              out_dim == 6144 ||
                              out_dim == 10240 || out_dim == 17408);
                bool xxs_rows4 = weight_type == P36_VK_TENSOR_IQ3_XXS &&
                                 out_dim % 4u == 0u;
                bool full_rows5 = weight_type == P36_VK_TENSOR_IQ3_S &&
                                  !rows1 && !rows4 && out_dim % 5u == 0u;
                p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ3_XXS ?
                    (xxs_rows4 ? &p36_vk.dense_iq3_xxs_decode_r4 :
                                 &p36_vk.dense_iq3_xxs_decode) :
                    (rows1 ? &p36_vk.dense_iq3_s_decode_r1 :
                     rows4 ? &p36_vk.dense_iq3_s_decode_r4 :
                     full_rows5 ? &p36_vk.dense_iq3_s_decode_full :
                                  &p36_vk.dense_iq3_s_decode);
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    weight_type == P36_VK_TENSOR_IQ3_XXS ?
                        "dense_iq3_xxs_decode" : "dense_iq3_s_decode");
                ok = p36_vk_run_unlocked(
                    op,
                    kernel, bindings, &push, sizeof(push),
                    rows1 ? (uint32_t)out_dim :
                    rows4 || xxs_rows4 ?
                        ((uint32_t)out_dim + 3u) / 4u :
                        ((uint32_t)out_dim + 4u) / 5u,
                    1, 1);
            } else if (weight_type == P36_VK_TENSOR_IQ3_XXS && out_dim % 4u == 0u &&
                       n_tok >= 2u && n_tok <= 8u &&
                       p36_vk_env_default_on("P36_VK_DENSE_IQ3_NX")) {
                /* Small-batch activation rows share one weight unpack.
                 * dense_iq3_xxs_decode_r4 is ALU-bound (P4: 5.4 VALU/weight), and
                 * the 128-row MMQ tile costs a full tile for 2..8 rows
                 * (dense_iq3_xxs_mmq + predequant_b16).  Only 2 rows is
                 * reachable today (P7: the verify is 2 rows, never 3); 3..8 is
                 * P6's recorded reconsider_if condition, bit-exact at 3/4/8 in
                 * tests/test_dense_iq3xxs_nx.c, and 8 is the audit's scope. */
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq3_xxs_decode_nx");
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u nx\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                ok = p36_vk_run_unlocked(
                    op, &p36_vk.dense_iq3_xxs_decode_nx,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                bool small_rows = weight_type == P36_VK_TENSOR_IQ3_S &&
                                  out_dim == 48u && n_tok <= 128u;
                bool wide_rows = weight_type == P36_VK_TENSOR_IQ3_S &&
                                 !small_rows && n_tok > 128u;
                p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ3_XXS ?
                    &p36_vk.dense_iq3_xxs_mmq :
                    small_rows ? &p36_vk.dense_iq3_s_mmq_r4 :
                    wide_rows ? &p36_vk.dense_iq3_s_bm64_mmq :
                                 &p36_vk.dense_iq3_s_mmq;
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq3_mmq");
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u mmq\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                p36_gpu_tensor *b16 = p36_vk_predequant_b16_unlocked(
                    q8, (uint32_t)n_tok, (uint32_t)blocks);
                const p36_gpu_tensor *mmq_bindings[5] = {
                    weights, q8, out, tables, b16,
                };
                ok = b16 != NULL &&
                     p36_vk_run_unlocked(
                    op, kernel,
                    mmq_bindings, &push, sizeof(push),
                    small_rows ? ((uint32_t)out_dim + 3u) / 4u :
                    wide_rows ? ((uint32_t)out_dim + 63u) / 64u :
                                 ((uint32_t)out_dim + 31u) / 32u,
                    ((uint32_t)n_tok + 127u) / 128u, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    tile_count = ((uint32_t)n_tok + P36_VK_MOE_PAIR_TILE - 1u) /
                 P36_VK_MOE_PAIR_TILE;
    tiles = p36_gpu_tensor_alloc((uint64_t)tile_count *
                                 (1u + P36_VK_MOE_PAIR_TILE) * sizeof(uint32_t));
    tp = tiles ? p36_vk_tensor_contents_labeled(tiles, "submit_wait_dense_iq_tiles") : NULL;
    if (!tiles || !tp) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    memset(tp, 0xff, (size_t)p36_gpu_tensor_bytes(tiles));
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t base = tile * (1u + P36_VK_MOE_PAIR_TILE);
        tp[base] = 0;
        for (uint32_t u = 0; u < P36_VK_MOE_PAIR_TILE; u++) {
            uint32_t tok = tile * P36_VK_MOE_PAIR_TILE + u;
            if (tok < n_tok) tp[base + 1u + u] = tok;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t type;
        uint32_t n_used;
        uint32_t has_scales;
        uint32_t x_per_expert;
        uint32_t packed_weights;
    } push = {
        (uint32_t)out_dim, (uint32_t)blocks, (uint32_t)row_bytes,
        (uint32_t)weight_bytes, weight_type, 1u, 0u, 0u, 0u,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights = tables ?
        p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
    ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            weights, q8, tiles, weights, out, tables,
        };
        p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast()) ?
            &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
        ok = p36_vk_run_unlocked("dense_iq_matmul", kernel, bindings,
                                 &push, sizeof(push),
                                 ((uint32_t)out_dim + 7u) / 8u,
                                 tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(tiles);
    if (ok && scale != 1.0f) {
        float *p = p36_vk_tensor_contents_labeled(out, "submit_wait_dense_iq_scale");
        if (!p) return 0;
        for (uint64_t i = 0; i < n_tok * out_dim; i++) p[i] *= scale;
    }
    return ok;
}

/* Dense gate+up prefill in one dispatch (campaign W6).  Both weight matrices
 * must be the same type with the same in_dim/out_dim, which is what the dense
 * FFN pair is; the shader decodes both A tiles against one staged activation
 * tile.  Falls back to the two single-projection dispatches on any refusal. */
int p36_gpu_matmul_iq3_xxs_pair_mmq_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_dim,
                                           const p36_gpu_tensor *q8,
                                           uint64_t n_tok,
                                           float scale) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(P36_VK_TENSOR_IQ3_XXS);
    uint64_t blocks, row_bytes, weight_bytes, out_bytes, q8_bytes;
    if (p36_gpu_quality || !p36_vk_use_dense_iq3_fast() ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || out_dim == 0 ||
        n_tok < 2 || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes) ||
        weight_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(n_tok, out_dim, &out_bytes) ||
        !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!out_a || !out_b || !p36_gpu_tensor_range_ok(out_a, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = {
        (uint32_t)out_dim, (uint32_t)n_tok,
        (uint32_t)blocks, (uint32_t)row_bytes, scale,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights_a = tables ?
        p36_vk_weight_get_unlocked(wa, weight_bytes) : NULL;
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_bytes) : NULL;
    p36_gpu_tensor *b16 = weights_b ?
        p36_vk_predequant_b16_unlocked(q8, (uint32_t)n_tok, (uint32_t)blocks) : NULL;
    int ok = b16 != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            weights_a, weights_b, out_a, out_b, tables, b16,
        };
        const char *op = p36_vk_prof_iq3_shape(
            P36_VK_TENSOR_IQ3_XXS, (uint32_t)n_tok, (uint32_t)in_dim,
            (uint32_t)out_dim, "dense_iq3_xxs_pair_mmq");
        ok = p36_vk_run_unlocked(op, &p36_vk.dense_iq3_xxs_mmq_pair,
                                 bindings, &push, sizeof(push),
                                 ((uint32_t)out_dim + 31u) / 32u,
                                 ((uint32_t)n_tok + 127u) / 128u, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_iq_quant_scaled_tensor(p36_gpu_tensor *out,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t weight_type,
                                           uint64_t in_dim,
                                           uint64_t out_dim,
                                           const p36_gpu_tensor *x,
                                           uint64_t n_tok,
                                           float scale) {
    p36_gpu_tensor *q8 = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!q8) return 0;
    int ok = p36_gpu_matmul_iq_quant_q8_scaled_tensor(
        out, model_map, model_size, weight_offset, weight_type,
        in_dim, out_dim, q8, n_tok, scale);
    p36_gpu_tensor_free(q8);
    return ok;
}

static int p36_vk_moe_gate_up_iq2_swiglu(p36_gpu_tensor *out,
                                          const void *model_map,
                                          uint64_t model_size,
                                          const p36_gpu_moe_weight *gate,
                                          const p36_gpu_moe_weight *up,
                                          const p36_gpu_tensor *selected,
                                          const p36_gpu_tensor *qx,
                                          const uint32_t *stream_slot_map,
                                          bool resident_bank,
                                          uint32_t n_used,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                         uint32_t mid_dim,
                                         uint32_t n_expert) {
    uint64_t block_bytes = 66;
    bool iq2s = false;
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    const unsigned char *gate_source;
    const unsigned char *up_source;
    const unsigned char *gate_scales_source = NULL;
    const unsigned char *up_scales_source = NULL;
    const uint32_t *selected_host;
    p36_gpu_tensor *tiles = NULL;
    uint32_t tile_count = 0;
    int ok = 0;

    if (p36_gpu_quality || !p36_vk_use_moe_gate_up() || (n_tok > 1u && n_tok < 32u) ||
        (!p36_vk_moe_bank_cache && !stream_slot_map)) return 0;
    if (!gate || !up || !selected || !qx || !out || n_used == 0 || n_tok == 0 || n_expert == 0) return 0;
    /* IQ2_S experts (Qwen3.8-35B-A3B-IQ2_M) decode with the same Q8_K
     * activation staging and the same 0.125 scale folding; only the 82-byte
     * superblock decode differs, so the shader is the same source built with
     * -DP36_MOE_IQ2S. */
    if (gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_S) {
        iq2s = true;
        block_bytes = 82;
    } else if (gate->type != P36_VK_TENSOR_IQ2_XXS || up->type != P36_VK_TENSOR_IQ2_XXS) {
        return 0;
    }
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || mid_dim == 0) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * mid_dim;
    if (expert_stride > UINT32_MAX || row_bytes > UINT32_MAX || blocks > UINT32_MAX ||
        !p36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    gate_source = p36_gpu_weight_bytes(model_map, model_size, gate->offset, weight_bytes);
    up_source = p36_gpu_weight_bytes(model_map, model_size, up->offset, weight_bytes);
    if (!gate_source || !up_source) return 0;
    if (gate->has_scales) {
        gate_scales_source = p36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes);
        if (!gate_scales_source) return 0;
    }
    if (up->has_scales) {
        up_scales_source = p36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes);
        if (!up_scales_source) return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, n_slot * mid_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * blocks * sizeof(p36_vk_block_q8_K))) {
        return 0;
    }

    if (n_slot <= 4096u && !stream_slot_map &&
        (!p36_vk_stream.enabled || resident_bank)) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, P36_VK_MOE_PAIR_TILE);
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, P36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_moe_gate_up_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = p36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_slot_map,
                                       &tile_count);
        if (!tiles || tile_count == 0) {
            p36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t n_used;
        uint32_t has_gate_scales;
        uint32_t has_up_scales;
    } push = {
        mid_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        n_used, gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    const p36_gpu_tensor *gate_bank = NULL;
    const p36_gpu_tensor *up_bank = NULL;
    const p36_gpu_tensor *gate_scales = NULL;
    const p36_gpu_tensor *up_scales = NULL;
    if (stream_slot_map) {
        gate_bank = p36_vk_stream_weight_tensor_unlocked(P36_VK_STREAM_PART_GATE);
        up_bank = p36_vk_stream_weight_tensor_unlocked(P36_VK_STREAM_PART_UP);
        gate_scales = p36_vk_stream_scale_tensor_unlocked(P36_VK_STREAM_PART_GATE);
        up_scales = p36_vk_stream_scale_tensor_unlocked(P36_VK_STREAM_PART_UP);
    } else {
        gate_bank = tables ? p36_vk_weight_get_unlocked(gate_source, weight_bytes) : NULL;
        up_bank = gate_bank ? p36_vk_weight_get_unlocked(up_source, weight_bytes) : NULL;
        if (up_bank && gate->has_scales) gate_scales = p36_vk_weight_get_unlocked(gate_scales_source, scale_bytes);
        if (up_bank && (!gate->has_scales || gate_scales) && up->has_scales) up_scales = p36_vk_weight_get_unlocked(up_scales_source, scale_bytes);
    }
    if (tables && up_bank && (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales)) {
        const p36_gpu_tensor *bindings[8] = {
            gate_bank,
            up_bank,
            qx,
            tiles,
            gate->has_scales ? gate_scales : gate_bank,
            up->has_scales ? up_scales : up_bank,
            out,
            tables,
        };
        ok = p36_vk_run_unlocked(iq2s ? "moe_iq2s_gate_up" : "moe_iq2_gate_up",
                                 iq2s ? &p36_vk.moe_gate_up_iq2s : &p36_vk.moe_gate_up,
                                 bindings, &push, sizeof(push),
                                 (mid_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(tiles);
    return ok;
}

/* Routed expert FFN, the routed half of p36_forward_ffn() including all
 * .scale applications, batched over n_tok rows of x with per-token expert
 * selections.  The fast IQ2_XXS path fuses gate/up/SwiGLU into one dispatch;
 * other layouts and --quality keep the separate exact gate/up/SwiGLU route. */
int p36_gpu_moe_ffn_q8_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const p36_gpu_moe_weight *gate,
                              const p36_gpu_moe_weight *up,
                              const p36_gpu_moe_weight *down,
                              const p36_gpu_tensor *selected,
                              const p36_gpu_tensor *weights,
                              uint32_t layer,
                              uint32_t n_used,
                              const p36_gpu_tensor *qx,
                              uint32_t n_tok,
                              uint32_t in_dim,
                              uint32_t mid_dim,
                              uint32_t out_dim,
                              uint32_t n_expert) {
    uint64_t mid_bytes = (uint64_t)n_used * mid_dim * sizeof(float);
    uint64_t down_bytes = (uint64_t)n_used * out_dim * sizeof(float);
    p36_gpu_tensor *gate8 = NULL;
    p36_gpu_tensor *up8 = NULL;
    p36_gpu_tensor *mid8 = NULL;
    p36_gpu_tensor *qmid = NULL;
    p36_gpu_tensor *down8 = NULL;
    uint32_t stream_slot_map[256];
    const uint32_t *stream_slots = NULL;
    bool resident_bank = p36_vk_stream.enabled && layer < p36_vk_stream.full_layers;
    int ok = 0;

    if (!gate || !up || !down || !qx || n_used == 0 || n_expert == 0 || n_tok == 0) return 0;
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        mid_dim == 0 || (mid_dim % P36_VK_QK_K) != 0 || out_dim == 0) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * (in_dim / P36_VK_QK_K) * sizeof(p36_vk_block_q8_K)) ||
        !p36_gpu_tensor_range_ok(selected, 0, (uint64_t)n_tok * n_used * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(weights, 0, (uint64_t)n_tok * n_used * sizeof(float))) {
        return 0;
    }

    if (p36_vk_stream.enabled && !resident_bank && n_expert <= 256) {
        uint64_t gate_block = p36_vk_iq_block_bytes(gate->type);
        uint64_t up_block = p36_vk_iq_block_bytes(up->type);
        uint64_t down_block = p36_vk_iq_block_bytes(down->type);
        uint64_t gate_row = gate_block * (in_dim / P36_VK_QK_K);
        uint64_t up_row = up_block * (in_dim / P36_VK_QK_K);
        uint64_t down_row = down_block * (mid_dim / P36_VK_QK_K);
        const uint32_t *selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected,
                                                                       "submit_wait_moe_stream_selected");
        p36_gpu_stream_expert_table table;
        memset(&table, 0, sizeof(table));
        table.model_map = model_map;
        table.model_size = model_size;
        table.layer = layer;
        table.n_total_expert = n_expert;
        table.gate_offset = gate->offset;
        table.up_offset = up->offset;
        table.down_offset = down->offset;
        table.gate_scales_offset = gate->scales_offset;
        table.up_scales_offset = up->scales_offset;
        table.down_scales_offset = down->scales_offset;
        table.gate_type = gate->type;
        table.up_type = up->type;
        table.down_type = down->type;
        table.has_gate_scales = gate->has_scales;
        table.has_up_scales = up->has_scales;
        table.has_down_scales = down->has_scales;
        if (gate_block && up_block && down_block && selected_host &&
            !p36_u64_mul_ok(gate_row, mid_dim, &table.gate_expert_bytes)) {
            selected_host = NULL;
        }
        if (selected_host && !p36_u64_mul_ok(up_row, mid_dim, &table.up_expert_bytes)) selected_host = NULL;
        if (selected_host && !p36_u64_mul_ok(down_row, out_dim, &table.down_expert_bytes)) selected_host = NULL;
        if (selected_host && p36_vk_stream_cache_prepare_u32(&table,
                                                             selected_host,
                                                             (uint64_t)n_tok * n_used,
                                                             stream_slot_map)) {
            stream_slots = stream_slot_map;
        }
    }

    mid8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
    down8 = p36_gpu_tensor_alloc_scratch(down_bytes * n_tok);
    ok = qx && mid8 && down8 &&
          p36_vk_moe_gate_up_iq2_swiglu(mid8, model_map, model_size, gate, up,
                                         selected, qx, stream_slots, resident_bank,
                                         n_used, n_tok, in_dim, mid_dim, n_expert);
    if (!ok && mid8 && down8) {
        gate8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
        up8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
        ok = gate8 && up8 &&
             p36_vk_moe_matvec(gate8, model_map, model_size, gate, qx, false, false,
                                stream_slots ? P36_VK_STREAM_PART_GATE : P36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             p36_vk_moe_matvec(up8, model_map, model_size, up, qx, false, true,
                                stream_slots ? P36_VK_STREAM_PART_UP : P36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             p36_gpu_swiglu_tensor(mid8, gate8, up8, n_tok * n_used * mid_dim, 0.0f, 1.0f);
    }
    if (ok) {
        qmid = p36_vk_quantize_q8_k_alloc_tensor(mid8, mid_dim, (uint64_t)n_tok * n_used);
        ok = qmid &&
               p36_vk_moe_matvec(down8, model_map, model_size, down, qmid, true,
                                  down->type == P36_VK_TENSOR_Q2_K ||
                                  down->type == P36_VK_TENSOR_Q3_K,
                                  stream_slots ? P36_VK_STREAM_PART_DOWN : P36_VK_STREAM_PART_NONE,
                                  stream_slots, resident_bank,
                                  selected, n_used, n_tok, mid_dim, out_dim, n_expert);
    }
    if (ok) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } push = { out_dim, n_used };
        const p36_gpu_tensor *bindings[3] = { down8, weights, out };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("moe_reduce", &p36_vk.moe_reduce, bindings, &push, sizeof(push),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
    }

    p36_gpu_tensor_free(down8);
    p36_gpu_tensor_free(qmid);
    p36_gpu_tensor_free(mid8);
    p36_gpu_tensor_free(up8);
    p36_gpu_tensor_free(gate8);
    return ok;
}

/* Whole routed FFN against f32 activations for decode and prefill chunks:
 * GPU tile schedule, fused gate/up/SwiGLU, specialized routed down, weighted
 * reduce.  Besides the native IQ2_XXS/Q2_K recipe this covers APEX's
 * IQ3_XXS/Q3_K, IQ2_S/IQ3_XXS and mixed IQ2_S+IQ2_XXS/IQ3_XXS recipes.
 * No q8_K stage anywhere, no host read of the selection. */
int p36_gpu_moe_ffn_f32_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      const p36_gpu_moe_weight *gate,
                                      const p36_gpu_moe_weight *up,
                                      const p36_gpu_moe_weight *down,
                                      const p36_gpu_tensor *selected,
                                      const p36_gpu_tensor *weights,
                                      uint32_t layer,
                                      uint32_t n_used,
                                      const p36_gpu_tensor *x,
                                      uint32_t n_tok,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      uint32_t n_expert) {
    bool q4k = gate->type == P36_VK_TENSOR_Q4_K && up->type == P36_VK_TENSOR_Q4_K &&
               down->type == P36_VK_TENSOR_Q4_K;
    /* IQ2_S experts (Qwen3.8-35B-A3B-IQ2_M): the fused f32 expert path carries
     * the decode pair and the prefill GEMM pair.  Layers 0-2 of that file mix
     * an IQ3_S down under IQ2_S gate/up, so the down side picks the IQ3_S
     * variant of the same kernels (110-byte superblocks instead of 82). */
    bool gu_iq2s = gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_S;
    bool gu_iq3xxs = gate->type == P36_VK_TENSOR_IQ3_XXS && up->type == P36_VK_TENSOR_IQ3_XXS;
    bool gu_iq2s_iq2xxs = gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_XXS;
    bool gu_q3k = gate->type == P36_VK_TENSOR_Q3_K && up->type == P36_VK_TENSOR_Q3_K;
    bool down_iq2s = down->type == P36_VK_TENSOR_IQ2_S;
    bool down_iq3s = down->type == P36_VK_TENSOR_IQ3_S;
    bool down_iq3xxs = down->type == P36_VK_TENSOR_IQ3_XXS;
    bool down_q3k = down->type == P36_VK_TENSOR_Q3_K;
    bool iq2s = gu_iq2s && (down_iq2s || down_iq3s);
    bool apex_iq3xxs_q3k = gu_iq3xxs && down_q3k &&
                           p36_vk_env_default_on("P36_VK_MOE_APEX_IQ3XXS_Q3K");
    bool apex_iq2s_iq3xxs = gu_iq2s && down_iq3xxs &&
                            p36_vk_env_default_on("P36_VK_MOE_APEX_IQ2S_IQ3XXS");
    bool apex_iq2s_iq2xxs_iq3xxs = gu_iq2s_iq2xxs && down_iq3xxs &&
                                   p36_vk_env_default_on("P36_VK_MOE_APEX_MIXED");
    /* Prefill batches only: the f32-activation GEMM is what makes IQ3_XXS/Q3_K
     * prefill ~4x faster, but it moves logits against the q8 route (~0.4-1.0
     * at decode), so decode and small batches keep the q8 route. */
    const bool apex_prefill = n_tok >= p36_vk_moe_gemm_min();
    bool apex_iq3xxs = apex_prefill && gu_iq3xxs && down_iq3xxs &&
                       p36_vk_env_default_on("P36_VK_MOE_APEX_IQ3XXS");
    bool apex_q3k = apex_prefill && gu_q3k && down_q3k &&
                    p36_vk_env_default_on("P36_VK_MOE_APEX_Q3K");
    bool apex = apex_iq3xxs_q3k || apex_iq2s_iq3xxs || apex_iq2s_iq2xxs_iq3xxs || apex_q3k || apex_iq3xxs;
    uint32_t gate_block_bytes = q4k ? 144u : gate->type == P36_VK_TENSOR_IQ2_S ? 82u :
                                gate->type == P36_VK_TENSOR_IQ3_XXS ? 98u :
                                gate->type == P36_VK_TENSOR_Q3_K ? 110u : 66u;
    uint32_t up_block_bytes = q4k ? 144u : up->type == P36_VK_TENSOR_IQ2_S ? 82u :
                              up->type == P36_VK_TENSOR_IQ3_XXS ? 98u :
                              up->type == P36_VK_TENSOR_Q3_K ? 110u : 66u;
    uint64_t gate_stride = (uint64_t)(in_dim / P36_VK_QK_K) * gate_block_bytes * mid_dim;
    uint64_t up_stride = (uint64_t)(in_dim / P36_VK_QK_K) * up_block_bytes * mid_dim;
    uint64_t down_stride = (uint64_t)(mid_dim / P36_VK_QK_K) *
                           (q4k ? 144u : down_iq2s ? 82u : down_iq3s || down_q3k ? 110u :
                            down_iq3xxs ? 98u : 84u) * out_dim;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    p36_gpu_tensor *tiles = NULL;
    p36_gpu_tensor *sel_slots = NULL;
    p36_gpu_tensor *mid8 = NULL;
    p36_gpu_tensor *down8 = NULL;
    const uint32_t *selected_host = NULL;
    p36_gpu_stream_expert_table stream_table = {0};
    uint32_t selected_unique[256];
    uint32_t unique = 0;
    uint32_t stream_cap = 0;
    bool overflow = false;
    bool streamed = p36_vk_stream.enabled && layer >= p36_vk_stream.full_layers;
    int ok = 0;

    if (p36_gpu_quality || !p36_vk.have_storage16 || !p36_vk_have_subgroups(false) ||
        n_tok == 0 || n_used == 0 || n_used > 8u || n_slot > 8192u) return 0;
    if (!p36_vk_env_default_on("P36_VK_MOE_F32B")) return 0;
    if (!streamed && !p36_vk_moe_bank_cache) return 0;
    if (!q4k && !iq2s && !apex &&
        (gate->type != P36_VK_TENSOR_IQ2_XXS || up->type != P36_VK_TENSOR_IQ2_XXS ||
         down->type != P36_VK_TENSOR_Q2_K)) return 0;
    /* Q4_K rides the same identity-schedule decode kernels; batches and
     * streaming keep the q8_K fallback. */
    if (q4k && (streamed || n_tok != 1u || !p36_vk_use_moe_down_sum_decode() ||
                !p36_vk_env_default_on("P36_VK_MOE_Q4K_F32B"))) return 0;
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || (in_dim % 4u) != 0 ||
        mid_dim == 0 || (mid_dim % P36_VK_QK_K) != 0 || out_dim == 0) return 0;
    if (gate_stride * n_expert > UINT32_MAX || up_stride * n_expert > UINT32_MAX ||
        down_stride * n_expert > UINT32_MAX) return 0;

    const unsigned char *gate_src = p36_gpu_weight_bytes(model_map, model_size, gate->offset, gate_stride * n_expert);
    const unsigned char *up_src = p36_gpu_weight_bytes(model_map, model_size, up->offset, up_stride * n_expert);
    const unsigned char *down_src = p36_gpu_weight_bytes(model_map, model_size, down->offset, down_stride * n_expert);
    if (!gate_src || !up_src || !down_src) return 0;
    const unsigned char *gate_sc = gate->has_scales ? p36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes) : NULL;
    const unsigned char *up_sc = up->has_scales ? p36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes) : NULL;
    const unsigned char *down_sc = down->has_scales ? p36_gpu_weight_bytes(model_map, model_size, down->scales_offset, scale_bytes) : NULL;
    if ((gate->has_scales && !gate_sc) || (up->has_scales && !up_sc) || (down->has_scales && !down_sc)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(x, 0, (uint64_t)n_tok * in_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(weights, 0, n_slot * sizeof(float))) {
        return 0;
    }

    /* SSD streaming: the selected experts stream into the cache and the
     * tile schedule carries cache slot ids, so the same fused kernels read
     * the stream buffers at slot * stride — identical bytes, identical
     * math, no q8_K staging.  The selection read is a host sync, but the
     * cache prepare needs it on the host anyway. */
    uint32_t stream_slot_map[256];
    if (streamed) {
        if (n_expert > 256u) return 0;
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected,
                                                       "submit_wait_moe_stream_selected");
        if (!selected_host) return 0;
        stream_table.model_map = model_map;
        stream_table.model_size = model_size;
        stream_table.layer = layer;
        stream_table.n_total_expert = n_expert;
        stream_table.gate_offset = gate->offset;
        stream_table.up_offset = up->offset;
        stream_table.down_offset = down->offset;
        stream_table.gate_scales_offset = gate->scales_offset;
        stream_table.up_scales_offset = up->scales_offset;
        stream_table.down_scales_offset = down->scales_offset;
        stream_table.gate_type = gate->type;
        stream_table.up_type = up->type;
        stream_table.down_type = down->type;
        stream_table.has_gate_scales = gate->has_scales;
        stream_table.has_up_scales = up->has_scales;
        stream_table.has_down_scales = down->has_scales;
        stream_table.gate_expert_bytes = gate_stride;
        stream_table.up_expert_bytes = up_stride;
        stream_table.down_expert_bytes = down_stride;
        stream_cap = p36_vk_stream_cache_cap(&stream_table);
        if (stream_cap == 0) return 0;
        bool seen[256] = {0};
        for (uint64_t i = 0; i < n_slot; i++) {
            uint32_t expert = selected_host[i];
            if (expert >= n_expert) return 0;
            if (!seen[expert]) {
                seen[expert] = true;
                selected_unique[unique++] = expert;
            }
        }
        overflow = unique > stream_cap;
        if (overflow) p36_vk_stream_note_routes(&stream_table, selected_host, n_slot);
        if (!overflow && !p36_vk_stream_cache_prepare_u32(
                &stream_table, selected_host, n_slot, stream_slot_map)) return 0;
        /* The kernels compute weight offsets in 32-bit halfword units, so
         * the whole slot arena must stay indexable. */
        if (gate_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u ||
            up_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u ||
            down_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u) return 0;
        if (!overflow && n_tok == 1u) {
            /* Decode identity schedule with slot ids instead of expert ids. */
            uint32_t mapped[8];
            for (uint32_t i = 0; i < (uint32_t)n_slot; i++) mapped[i] = stream_slot_map[selected_host[i]];
            sel_slots = p36_gpu_tensor_alloc(n_slot * sizeof(uint32_t));
            if (!sel_slots || !p36_gpu_tensor_write(sel_slots, 0, mapped, n_slot * sizeof(uint32_t))) goto done;
        }
    }

    /* Decode: one distinct expert per slot, so the selection itself is the
     * schedule and the tile build dispatch is skipped. */
    bool identity = n_tok == 1u && !overflow;
    bool down_sum_decode = identity && p36_vk_use_moe_down_sum_decode();
    /* Prefill batches route to the register-tiled GEMM kernels: wider
     * expert token tiles amortize the dequant and weight reads that bound
     * the matvec path. Small batches (chunk tails, MTP verify windows)
     * keep the matvec kernels, whose per-slot arithmetic order matches
     * decode bit for bit; the GEMM's k order is fixed but differs from
     * decode, which the relaxed warm==cold session gate absorbs. */
    bool gemm = !identity && !p36_vk_micro_batch &&
                n_tok >= p36_vk_moe_gemm_min() &&
                p36_vk_env_default_on("P36_VK_MOE_GEMM");
    /* IQ2_S experts: the prefill GEMM pair, the identity decode pair, and the
     * small-batch (2..127 token) f32 pair.  The f32 pair needs an IQ2_S down
     * as well, so the IQ3_S-down layers (IQ2_M il 0-2) still fall back to the
     * q8 route outside the GEMM range. */
    if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;
    /* APEX has dedicated wide-prefill and identity-decode kernels.  Tiny
     * routed batches retain the generic path until their f32 pair kernels
     * are selected below as well. */
    if (apex && !(gemm || (identity && down_sum_decode))) return 0;
    uint32_t pair_tile = gemm ? P36_VK_MOE_GEMM_TILE : P36_VK_MOE_PAIR_TILE;
    uint32_t tile_count = (uint32_t)n_slot;
    if (!identity && !overflow) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert,
                                     streamed ? stream_slot_map : NULL, pair_tile);
        if (!tiles) goto done;
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, pair_tile);
    }
    mid8 = p36_gpu_tensor_alloc_scratch(n_slot * mid_dim * sizeof(float));
    if (!down_sum_decode) down8 = p36_gpu_tensor_alloc_scratch(n_slot * out_dim * sizeof(float));
    if (!mid8 || (!down_sum_decode && !down8)) goto done;

    uint32_t wave = 0;
    ok = 1;
    /* Overflow runs the original full-batch expert tiles in cache-sized
     * waves and reduces once. Pair arithmetic is unchanged, while a small
     * resident cache never selects the Q8 fallback. */
    do {
        p36_gpu_tensor *wave_tiles = tiles;
        uint32_t wave_tile_count = tile_count;
        uint32_t wave_experts = unique - wave;
        if (overflow) {
            if (wave_experts > stream_cap) wave_experts = stream_cap;
            if (!p36_vk_stream_cache_prepare_u32(&stream_table,
                                                  selected_unique + wave,
                                                  wave_experts,
                                                  stream_slot_map)) {
                ok = 0;
                break;
            }
            wave_tiles = p36_vk_moe_build_tiles_ex(selected_host, n_slot,
                                                    n_expert, stream_slot_map,
                                                    pair_tile, true,
                                                    &wave_tile_count);
            if (!wave_tiles || wave_tile_count == 0) {
                p36_gpu_tensor_free(wave_tiles);
                ok = 0;
                break;
            }
        }

        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *gate_bank, *up_bank, *down_bank;
        p36_gpu_tensor *gate_scales, *up_scales, *down_scales;
        if (streamed) {
            gate_bank = p36_vk_stream.gate;
            up_bank = p36_vk_stream.up;
            down_bank = p36_vk_stream.down;
            gate_scales = gate->has_scales ? p36_vk_stream.gate_scales : NULL;
            up_scales = up->has_scales ? p36_vk_stream.up_scales : NULL;
            down_scales = down->has_scales ? p36_vk_stream.down_scales : NULL;
            ok = tables && gate_bank && up_bank && down_bank;
        } else {
            gate_bank = tables ? p36_vk_weight_get_unlocked(gate_src, gate_stride * n_expert) : NULL;
            up_bank = gate_bank ? p36_vk_weight_get_unlocked(up_src, up_stride * n_expert) : NULL;
            down_bank = up_bank ? p36_vk_weight_get_unlocked(down_src, down_stride * n_expert) : NULL;
            gate_scales = gate_sc && down_bank ? p36_vk_weight_get_unlocked(gate_sc, scale_bytes) : NULL;
            up_scales = up_sc && down_bank ? p36_vk_weight_get_unlocked(up_sc, scale_bytes) : NULL;
            down_scales = down_sc && down_bank ? p36_vk_weight_get_unlocked(down_sc, scale_bytes) : NULL;
            ok = tables && down_bank;
        }
        ok = ok &&
             (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales) && (!down->has_scales || down_scales);
        if (ok) {
            struct {
                uint32_t mid_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_gate_scales;
                uint32_t has_up_scales;
                uint32_t identity;
            } gpush = { mid_dim, in_dim / P36_VK_QK_K, n_used,
                        gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const p36_gpu_tensor *gb[8] = {
                gate_bank, up_bank, x, identity ? (sel_slots ? sel_slots : selected) : wave_tiles,
                gate_scales ? gate_scales : gate_bank,
                up_scales ? up_scales : up_bank,
                mid8, tables,
            };
            if (q4k) {
                const p36_gpu_tensor *qgb[7] = {
                    gate_bank, up_bank, x, sel_slots ? sel_slots : selected,
                    gate_scales ? gate_scales : gate_bank,
                    up_scales ? up_scales : up_bank,
                    mid8,
                };
                ok = p36_vk_run_unlocked("moe_q4k_gate_up", &p36_vk.moe_gate_up_q4k_f32b, qgb, &gpush, sizeof(gpush),
                                         mid_dim, wave_tile_count, 1);
            } else {
                ok = gemm ? p36_vk_run_unlocked(gu_q3k ? "moe_q3k_gate_up_gemm" :
                                                gu_iq3xxs ? "moe_iq3xxs_gate_up_gemm" :
                                                (gu_iq2s_iq2xxs ? "moe_iq2s_iq2xxs_gate_up_gemm" :
                                                 (gu_iq2s ? "moe_iq2s_gate_up_gemm" : "moe_iq2_gate_up_gemm")),
                                                gu_q3k ? &p36_vk.moe_gate_up_gemm_q3k :
                                                gu_iq3xxs ? &p36_vk.moe_gate_up_gemm_iq3xxs :
                                                (gu_iq2s_iq2xxs ? &p36_vk.moe_gate_up_gemm_iq2s_iq2xxs :
                                                 (gu_iq2s ? &p36_vk.moe_gate_up_gemm_iq2s : &p36_vk.moe_gate_up_gemm)),
                                                gb, &gpush, sizeof(gpush),
                                                (mid_dim + 63u) / 64u, wave_tile_count, 1)
                          : p36_vk_run_unlocked(identity ? (gu_q3k ? "moe_q3k_gate_up_decode" :
                                                              gu_iq3xxs ? "moe_iq3xxs_gate_up_decode" :
                                                               (gu_iq2s_iq2xxs ? "moe_iq2s_iq2xxs_gate_up_decode" :
                                                                (gu_iq2s ? "moe_iq2s_gate_up_decode" : "moe_iq2_gate_up_decode")))
                                                         : (gu_iq2s ? "moe_iq2s_gate_up_f32b" : "moe_iq2_gate_up"),
                                                identity ? (gu_q3k ? &p36_vk.moe_gate_up_decode_q3k :
                                                            gu_iq3xxs ? &p36_vk.moe_gate_up_decode_iq3xxs :
                                                            (gu_iq2s_iq2xxs ? &p36_vk.moe_gate_up_decode_iq2s_iq2xxs :
                                                             (gu_iq2s ? &p36_vk.moe_gate_up_decode_iq2s
                                                                      : &p36_vk.moe_gate_up_decode)))
                                                         : (gu_iq2s ? &p36_vk.moe_gate_up_f32b_iq2s
                                                                     : &p36_vk.moe_gate_up_f32b),
                                                gb, &gpush, sizeof(gpush), mid_dim, wave_tile_count, 1);
            }
        }
        if (ok) {
            struct {
                uint32_t out_dim;
                uint32_t in_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_scales;
                uint32_t identity;
            } dpush = { out_dim, mid_dim, mid_dim / P36_VK_QK_K, n_used, down->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const p36_gpu_tensor *schedule = identity ? (sel_slots ? sel_slots : selected) : wave_tiles;
            if (down_sum_decode) {
                const p36_gpu_tensor *db[7] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    weights, out, tables,
                };
                const char *dname = q4k ? "moe_q4k_down_sum_decode" :
                                    down_q3k ? "moe_q3k_down_sum_decode" :
                                    down_iq3xxs ? "moe_iq3xxs_down_sum_decode" :
                                    down_iq3s ? "moe_iq3s_down_sum_decode" :
                                    down_iq2s ? "moe_iq2s_down_sum_decode" : "moe_q2k_down_sum_decode";
                p36_vk_kernel *dkernel = q4k ? &p36_vk.moe_down_q4k_sum_decode :
                                          down_q3k ? &p36_vk.moe_down_q2k_sum_decode_q3k :
                                          down_iq3xxs ? &p36_vk.moe_down_q2k_sum_decode_iq3xxs :
                                          down_iq3s ? &p36_vk.moe_down_q2k_sum_decode_iq3s :
                                          down_iq2s ? &p36_vk.moe_down_q2k_sum_decode_iq2s :
                                                       &p36_vk.moe_down_q2k_sum_decode;
                ok = p36_vk_run_unlocked(dname, dkernel,
                                         db, &dpush, sizeof(dpush),
                                         out_dim, 1, 1);
            } else {
                const p36_gpu_tensor *db[6] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    down8, tables,
                };
                const char *dname = down_q3k ? "moe_q3k_down_gemm" :
                                    down_iq3xxs ? "moe_iq3xxs_down_gemm" :
                                    down_iq3s ? "moe_iq3s_down_gemm" :
                                    down_iq2s ? "moe_iq2s_down_gemm" : "moe_q2k_down_gemm";
                p36_vk_kernel *dkernel = down_q3k ? &p36_vk.moe_down_gemm_q3k :
                                          down_iq3xxs ? &p36_vk.moe_down_gemm_iq3xxs :
                                          down_iq3s ? &p36_vk.moe_down_gemm_iq3s :
                                          down_iq2s ? &p36_vk.moe_down_gemm_iq2s : &p36_vk.moe_down_gemm;
                ok = gemm ? p36_vk_run_unlocked(dname, dkernel, db, &dpush, sizeof(dpush),
                                                (out_dim + 63u) / 64u, wave_tile_count, 1)
                          : p36_vk_run_unlocked(down_iq2s ? "moe_iq2s_down_f32b" : "moe_q2k_down",
                                                down_iq2s ? &p36_vk.moe_down_q2k_f32b_iq2s
                                                          : &p36_vk.moe_down_q2k_f32b,
                                                db, &dpush, sizeof(dpush),
                                                out_dim, wave_tile_count, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (overflow) p36_gpu_tensor_free(wave_tiles);
        wave += wave_experts;
    } while (ok && overflow && wave < unique);

    if (ok && !down_sum_decode) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } rpush = { out_dim, n_used };
        const p36_gpu_tensor *rb[3] = { down8, weights, out };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("moe_reduce", &p36_vk.moe_reduce, rb, &rpush, sizeof(rpush),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
    }

done:
    p36_gpu_tensor_free(down8);
    p36_gpu_tensor_free(mid8);
    p36_gpu_tensor_free(tiles);
    p36_gpu_tensor_free(sel_slots);
    return ok;
}

int p36_gpu_moe_ffn_tensor(p36_gpu_tensor *out,
                           const void *model_map,
                           uint64_t model_size,
                           const p36_gpu_moe_weight *gate,
                           const p36_gpu_moe_weight *up,
                           const p36_gpu_moe_weight *down,
                           const p36_gpu_tensor *selected,
                           const p36_gpu_tensor *weights,
                           uint32_t layer,
                           uint32_t n_used,
                           const p36_gpu_tensor *x,
                           uint32_t n_tok,
                           uint32_t in_dim,
                           uint32_t mid_dim,
                           uint32_t out_dim,
                           uint32_t n_expert) {
    p36_gpu_tensor *qx = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!qx) return 0;
    int ok = p36_gpu_moe_ffn_q8_tensor(out, model_map, model_size,
                                       gate, up, down,
                                       selected, weights, layer, n_used,
                                       qx, n_tok, in_dim, mid_dim, out_dim, n_expert);
    p36_gpu_tensor_free(qx);
    return ok;
}
