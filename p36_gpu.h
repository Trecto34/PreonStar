#ifndef P36_GPU_H
#define P36_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the P36-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV/SSM state, and
 * scratch buffers stay device-owned across the whole prefill/decode command
 * sequence.
 */
typedef struct p36_gpu_tensor p36_gpu_tensor;

int p36_gpu_init(void);
void p36_gpu_cleanup(void);

/* Host-side row splitter for Vulkan parity ops. Each callback must preserve
 * the serial arithmetic order inside every row/token it handles. */
typedef void (*p36_gpu_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);
void p36_gpu_parallel_for_rows(uint64_t n_rows,
                               uint64_t min_parallel_rows,
                               p36_gpu_parallel_fn fn,
                               void *ctx);

p36_gpu_tensor *p36_gpu_tensor_alloc(uint64_t bytes);
/* Persistent storage whose rows are written before use. */
p36_gpu_tensor *p36_gpu_tensor_alloc_uninitialized(uint64_t bytes);
/* Output-only scratch. Contents are undefined until a GPU command writes it. */
p36_gpu_tensor *p36_gpu_tensor_alloc_scratch(uint64_t bytes);
p36_gpu_tensor *p36_gpu_tensor_view(const p36_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void p36_gpu_tensor_free(p36_gpu_tensor *tensor);
uint64_t p36_gpu_tensor_bytes(const p36_gpu_tensor *tensor);
void *p36_gpu_tensor_contents(p36_gpu_tensor *tensor);
/* Same, but the flush this mapping may force shows up in the op profile
 * under `reason` instead of the generic submit_wait_tensor_contents. */
void *p36_gpu_tensor_contents_named(p36_gpu_tensor *tensor, const char *reason);
int p36_gpu_tensor_write(p36_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int p36_gpu_tensor_read(const p36_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
/* Diagnostic: bytes pulled back to the host, and how many of those reads forced
 * an open-batch flush (a GPU drain).  Used by P36_MTP_TIMING. */
uint64_t p36_gpu_read_bytes(void);
uint64_t p36_gpu_read_flushes(void);
int p36_gpu_tensor_copy(p36_gpu_tensor *dst, uint64_t dst_offset,
                          const p36_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int p36_gpu_vision_stream_init(uint64_t max_weight_bytes);
int p36_gpu_vision_matmul_f16_disk(p36_gpu_tensor *out, int fd,
                                   uint64_t offset, uint64_t in_dim,
                                   uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok);
/* Same F16 vision matmul, but the caller writes the [out_dim][in_dim] F16
 * weight into the mapped stream buffer (used to dequantize F32/Q8_0 sidecar
 * weights on the host without a dedicated shader). Return nonzero on success. */
typedef int (*p36_gpu_vision_fill_fn)(void *ctx, uint16_t *dst);
int p36_gpu_vision_matmul_f16_fill(p36_gpu_tensor *out,
                                   p36_gpu_vision_fill_fn fill, void *ctx,
                                   uint64_t in_dim, uint64_t out_dim,
                                   const p36_gpu_tensor *x,
                                   uint64_t n_tok);
int p36_gpu_vision_attention(p36_gpu_tensor *out,
                             const p36_gpu_tensor *qkv,
                             uint32_t rows);

int p36_gpu_begin_commands(void);
int p36_gpu_flush_commands(void);
int p36_gpu_end_commands(void);
int p36_gpu_synchronize(void);
/* Free idle pooled scratch so large one-off allocations (KV growth) fit. */
void p36_gpu_trim_scratch(void);

/* True once the GPU context has been destroyed under us (an amdgpu ring
 * timeout is the usual cause).  Latched and one-way: every later GPU call
 * fails, and the process must be restarted.  Lets callers tell "the device
 * died" apart from an ordinary op failure when reporting an error. */
int p36_gpu_device_lost(void);

/* True when the active device is the BC-250's GFX1013 GPU.  Callers use it
 * to apply the watchdog-driven prefill-chunk cap; safe to call before the
 * GPU is up, since it initialises on demand. */
int p36_gpu_device_is_gfx1013(void);

int p36_gpu_set_model_map(const void *model_map, uint64_t model_size);
int p36_gpu_set_model_fd(int fd);
int p36_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
int p36_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int p36_gpu_finish_model_cache(void);
int p36_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int p36_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
#ifdef P36_METAL
int p36_gpu_prepare_q5_k_output_cache(const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim);
#endif
void p36_gpu_set_quality(bool quality);
void p36_gpu_set_dense_model(bool dense);
void p36_gpu_set_micro_batch(bool enabled);
/* True when the fused attention path will serve p36_gpu_attn_decode_tensor;
 * the runtime then skips the per-context scores scratch. Sampled once. */
bool p36_gpu_attn_fused_enabled(void);
void p36_gpu_set_ssd_streaming(bool enabled);
void p36_gpu_set_streaming_full_layers(uint32_t layers);
void p36_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void p36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
void p36_gpu_set_streaming_expert_cache_layout(uint64_t gate_bytes,
                                               uint64_t up_bytes,
                                               uint64_t down_bytes);
uint64_t p36_gpu_recommended_working_set_size(void);
uint32_t p36_gpu_stream_expert_cache_configured_count(void);
uint32_t p36_gpu_stream_expert_cache_current_count(void);

typedef struct p36_gpu_stream_expert_table {
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_scales_offset;
    uint64_t up_scales_offset;
    uint64_t down_scales_offset;
    uint64_t gate_expert_bytes;
    uint64_t up_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    bool has_gate_scales;
    bool has_up_scales;
    bool has_down_scales;
} p36_gpu_stream_expert_table;

void p36_gpu_stream_expert_cache_reset_route_hotness(void);
void p36_gpu_stream_expert_cache_note_tokens(uint32_t n_tokens);
void p36_gpu_stream_expert_cache_release_resident(void);
uint32_t p36_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gate_expert_bytes, uint64_t up_expert_bytes, uint64_t down_expert_bytes);
int p36_gpu_stream_expert_cache_seed_selected(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected);
int p36_gpu_stream_expert_cache_begin_selected_load(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected);
int p36_gpu_stream_expert_cache_prepare_selected_batch(const p36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected);
int p36_gpu_stream_expert_cache_load_layer(const p36_gpu_stream_expert_table *table);
int p36_gpu_stream_expert_cache_seed_from_layer_selected(const p36_gpu_stream_expert_table *table, const p36_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected);
int p36_gpu_stream_expert_cache_release_layer_cache(void);
int p36_gpu_stream_expert_cache_seed_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts);
int p36_gpu_stream_expert_cache_bias_experts(const p36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts);
void p36_gpu_print_memory_report(const char *label);
void p36_gpu_prof_reset(void);
void p36_gpu_prof_report(const char *label);

/* =========================================================================
 * Dense Projections and Norms.
 * =========================================================================
 *
 * Matvec weights are read straight from the mmapped GGUF at weight_offset and
 * cached on device on first use.  The *_scaled variants apply the scalar from
 * the matching ".scale" model tensor to the output, like p36_scale_inplace()
 * after each CPU matvec.  The k-quant path quantizes activations to q8_K and
 * uses the same integer dot ggml's CPU kernels use.
 */

int p36_gpu_quantize_q8_k_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *x,
        uint64_t                in_dim,
        uint64_t                n_tok);

int p36_gpu_swiglu_q8_k_tensor(
        p36_gpu_tensor       *out,
        p36_gpu_tensor       *q8,
        const p36_gpu_tensor *gate,
        const p36_gpu_tensor *up,
        uint32_t                in_dim,
        uint32_t                n_tok,
        float                   clamp,
        float                   weight);

int p36_gpu_matmul_f16_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok);

int p36_gpu_matmul_bf16_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_f32_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok);

int p36_gpu_matmul_f32_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_q8_0_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_q8_0_pair_scaled_tensor(
        p36_gpu_tensor       *out_a,
        p36_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_a_dim,
        uint64_t                out_b_dim,
        const p36_gpu_tensor *x,
        float                   scale_a,
        float                   scale_b);

int p36_gpu_matmul_iq2s_pair_scaled_tensor(
        p36_gpu_tensor       *out_a,
        p36_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_a_dim,
        uint64_t                out_b_dim,
        const p36_gpu_tensor *q8,
        float                   scale_a,
        float                   scale_b);

int p36_gpu_matmul_iq3_s_pair_tensor(
        p36_gpu_tensor       *out_a,
        p36_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x);

int p36_gpu_matmul_iq3_xxs_pair_mmq_tensor(
        p36_gpu_tensor       *out_a,
        p36_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *q8,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_iq3_xxs_pair_tensor(
        p36_gpu_tensor       *out_a,
        p36_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x);

int p36_gpu_matmul_k_quant_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok);

int p36_gpu_matmul_k_quant_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_k_quant_q8_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *q8,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_iq_quant_q8_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *q8,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_matmul_iq_quant_scaled_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const p36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int p36_gpu_rms_norm_plain_rows_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int p36_gpu_rms_norm_weight_rows_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int p36_gpu_recurrent_norm_gate_tensor(
        p36_gpu_tensor       *state,
        const p36_gpu_tensor *gate,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                width,
        uint32_t                rows,
        float                   eps);

int p36_gpu_recurrent_norm_gate_q8_k_tensor(
        p36_gpu_tensor       *state,
        p36_gpu_tensor       *q8,
        const p36_gpu_tensor *gate,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                rows,
        float                   eps);

int p36_gpu_swiglu_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *gate,
        const p36_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int p36_gpu_ffn_tail_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *shared,
        const p36_gpu_tensor *scalar,
        uint32_t                out_dim,
        uint32_t                n_tok);

int p36_gpu_add_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *a,
        const p36_gpu_tensor *b,
        uint32_t                n);

int p36_gpu_directional_steering_project_tensor(
        p36_gpu_tensor       *x,
        const p36_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/* Normalized Walsh-Hadamard transform over each contiguous 1024-element block
 * of every f32 row: dst = H src / sqrt(1024). */
int p36_gpu_fwht_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *src,
        uint32_t                width,
        uint32_t                n_rows,
        float                   scale);

/* Elementwise multiply of every f32 row by the +/-1 sign vector stored at
 * sign_offset (broadcast across rows). dst may alias src. */
int p36_gpu_signs_mul_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *src,
        const p36_gpu_tensor *signs,
        uint32_t                width,
        uint32_t                n_rows,
        uint32_t                sign_offset,
        uint32_t                total_signs);

/* Tiled -> grouped V-head reorder within each f32 row of width hd*nk*rep:
 * dst[hd + hd*(rep + rep*nk)] = src[hd + hd*(nk + nk*rep)]. */
int p36_gpu_v_grouped_permute_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *src,
        uint32_t                width,
        uint32_t                hd,
        uint32_t                nk,
        uint32_t                rep,
        uint32_t                n_rows);

/* Fused activation prep: (optional grouped permute) -> signs multiply ->
 * normalized 1024-block FWHT -> Q8_K quantize into q8_dst. */
int p36_gpu_hadamard_prepare_tensor(
        p36_gpu_tensor       *q8_dst,
        const p36_gpu_tensor *src,
        const p36_gpu_tensor *signs,
        uint32_t              width,
        uint32_t              n_rows,
        uint32_t              sign_offset,
        uint32_t              total_signs,
        float                 scale,
        uint32_t              is_grouped);

/* out_sum = a + b, out_norm = rmsnorm(out_sum) * weight, fused per row.
 * out_sum may alias a or b. */
int p36_gpu_add_rms_norm_tensor(
        p36_gpu_tensor       *out_norm,
        p36_gpu_tensor       *out_sum,
        const p36_gpu_tensor *a,
        const p36_gpu_tensor *b,
        const void            *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

/* out_sum = a + b, out_norm = rmsnorm(out_sum) * weight, out_q8 = q8_K(out_norm), fused per row.
 * out_sum may alias a or b. */
int p36_gpu_add_rms_norm_q8_k_tensor(
        p36_gpu_tensor       *out_norm,
        p36_gpu_tensor       *out_q8,
        p36_gpu_tensor       *out_sum,
        const p36_gpu_tensor *a,
        const p36_gpu_tensor *b,
        const void            *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

/* =========================================================================
 * Qwen3.6 Decode Kernels.
 * =========================================================================
 *
 * Faithful GPU ports of the CPU reference path in p36.c, one entry per step
 * of p36_forward_token_cpu().
 */

/* p36_l2_norm() over rows, in place. */
int p36_gpu_l2_norm_rows_tensor(
        p36_gpu_tensor *x,
        uint32_t          n,
        uint32_t          rows,
        float             eps);

int p36_gpu_recurrent_conv_step_tensor(
        p36_gpu_tensor       *cache_conv,
        const p36_gpu_tensor *cur,
        p36_gpu_tensor       *window,
        uint32_t                n_tok);

int p36_gpu_gdn_front_tensor(p36_gpu_tensor *history,
    const p36_gpu_tensor *cur, const p36_gpu_tensor *x,
    p36_gpu_tensor *q, p36_gpu_tensor *k, p36_gpu_tensor *v, p36_gpu_tensor *gb,
    const void *map, uint64_t size, const uint64_t offsets[5],
    uint32_t width, uint32_t groups, uint32_t heads,
    float alpha_scale, float beta_scale, float eps);

int p36_gpu_recurrent_conv_silu_tensor(
        p36_gpu_tensor       *cache_conv,
        const p36_gpu_tensor *cur,
        p36_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              kernel_offset,
        uint32_t              conv_dim,
        uint32_t              n_taps,
        uint32_t              n_tok);

int p36_gpu_extract_full_attn_q_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *qg,
        uint32_t                n_tok);

/* Device-side contiguous f32 copy (dst[0..n) = src[0..n)), so host-staged
 * data reaches a GPU-written tensor without mapping it. */
int p36_gpu_copy_f32_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *src,
        uint32_t                n_floats);

/* Dequantize one Q8_0 embedding row on the device.  When `ids` is given the
 * row index is read from device memory, so the caller does not have to know
 * the token before recording the dispatch. */
/* Upload the quantized embedding table once, ahead of the decode loop. */
void p36_gpu_embed_table_prepare(const void *model_map, uint64_t model_size,
                                 uint64_t table_offset, uint64_t table_bytes);

int p36_gpu_embed_row_tensor(
        p36_gpu_tensor       *dst,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              table_offset,
        uint64_t              table_bytes,
        uint32_t              n_embd,
        uint32_t              type,
        const p36_gpu_tensor *ids,
        uint32_t              row);

int p36_gpu_extract_recurrent_v_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *conv,
        uint32_t                n_tok);

int p36_gpu_rope_qwen_rows_tensor(
        p36_gpu_tensor *x,
        uint32_t          n_head,
        uint32_t          pos0,
        uint32_t          n_tok);

int p36_gpu_rope_qwen_mrope_rows_tensor(
        p36_gpu_tensor *x,
        uint32_t          n_head,
        const uint32_t   *positions,
        uint32_t          n_tok);

int p36_gpu_rms_norm_rope_qwen_rows_tensor(
        p36_gpu_tensor       *dst,
        const p36_gpu_tensor *src,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        float                 eps);

int p36_gpu_rms_norm_rope_qwen_kv_store_tensor(
        p36_gpu_tensor       *k_cache,
        p36_gpu_tensor       *v_cache,
        const p36_gpu_tensor *k,
        const p36_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        uint32_t              cap,
        float                 eps);

int p36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
        p36_gpu_tensor       *k_cache,
        p36_gpu_tensor       *v_cache,
        const p36_gpu_tensor *k,
        const p36_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        uint32_t              cap,
        float                 eps,
        uint32_t              k_row_bytes,
        uint32_t              v_row_bytes);

int p36_gpu_shared_ffn_decode_tensor(
        p36_gpu_tensor       *out,
        p36_gpu_tensor       *mid,
        const p36_gpu_tensor *x,
        const p36_gpu_tensor *scalar,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              in_dim,
        uint32_t              mid_dim,
        uint32_t              out_dim,
        float                 gate_scale,
        float                 up_scale,
        float                 down_scale);


/* p36_ssm_conv_apply() + SiLU over n_tok windows: each window holds n_taps
 * rows of conv_dim, oldest first; the kernel weight is the model ssm_conv1d
 * tensor. */
int p36_gpu_ssm_conv_silu_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *window,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                kernel_offset,
        uint32_t                conv_dim,
        uint32_t                n_taps,
        uint32_t                n_tok);

/* Broadcast the conv output groups into per-head q/k and L2-normalize each
 * head (the gather + p36_l2_norm() loop in p36_forward_recurrent()), for
 * n_tok conv rows of conv_stride floats each. */
int p36_gpu_delta_qk_l2_norm_tensor(
        p36_gpu_tensor       *q_out,
        p36_gpu_tensor       *k_out,
        const p36_gpu_tensor *conv,
        uint32_t                n_heads,
        uint32_t                n_groups,
        uint32_t                state_dim,
        uint32_t                conv_stride,
        uint32_t                n_tok,
        float                   eps);

int p36_gpu_delta_qkv_l2_norm_tensor(
        p36_gpu_tensor       *q_out,
        p36_gpu_tensor       *k_out,
        p36_gpu_tensor       *v_out,
        const p36_gpu_tensor *conv,
        uint32_t              n_heads,
        uint32_t              n_groups,
        uint32_t              state_dim,
        uint32_t              conv_stride,
        uint32_t              n_tok,
        float                 eps);

/* Per-head decay/beta prep over n_tok rows: gb[h] = softplus(alpha+dt_bias)*a
 * and gb[n_heads+h] = sigmoid(beta). */
int p36_gpu_delta_net_gates_tensor(
        p36_gpu_tensor       *gb,
        const p36_gpu_tensor *alpha,
        const p36_gpu_tensor *beta,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                dt_bias_offset,
        uint64_t                a_offset,
        uint32_t                n_heads,
        uint32_t                n_tok);

/* p36_delta_net_decode(): n_tok recurrent state updates + readouts, applied
 * strictly in token order. */
int p36_gpu_delta_net_decode_tensor(
        p36_gpu_tensor       *state,
        const p36_gpu_tensor *q,
        const p36_gpu_tensor *k,
        const p36_gpu_tensor *v,
        const p36_gpu_tensor *gb,
        p36_gpu_tensor       *out,
        uint32_t                n_heads,
        uint32_t                state_dim,
        uint32_t                n_tok);

/* p36_full_attn_cache_store(): append n_tok K/V row pairs starting at pos0. */
int p36_gpu_attn_kv_store_tensor(
        p36_gpu_tensor       *k_cache,
        p36_gpu_tensor       *v_cache,
        const p36_gpu_tensor *k,
        const p36_gpu_tensor *v,
        uint32_t                pos0,
        uint32_t                n_tok,
        uint32_t                cap,
        uint32_t                k_row,
        uint32_t                v_row,
        uint32_t                k_cache_type,
        uint32_t                v_cache_type,
        uint32_t                k_cache_row_bytes,
        uint32_t                v_cache_row_bytes);

/* Full-attention decode over the typed KV cache: scores, softmax with optional
 * attention sinks, weighted V sum, and sigmoid output gating from the gate
 * half of the fused q projection (qg rows are [head][2*head_dim]).  Causal
 * batch: query row t sits at position pos0 + t and attends keys 0..pos0+t.
 * scores is caller scratch of n_tok*n_head*(pos0+n_tok) floats, clobbered. */
int p36_gpu_attn_decode_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *q,
        const p36_gpu_tensor *qg,
        const p36_gpu_tensor *k_cache,
        const p36_gpu_tensor *v_cache,
        p36_gpu_tensor       *scores,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        bool                    has_sinks,
        uint32_t                pos0,
        uint32_t                n_tok,
        uint32_t                n_head,
        uint32_t                n_head_kv,
        uint32_t                head_dim,
        uint32_t                k_cache_type,
        uint32_t                v_cache_type,
        uint32_t                k_cache_row_bytes,
        uint32_t                v_cache_row_bytes);

/* Router softmax + top-k + weight normalization over n_tok logit rows;
 * route_scale is the model expert_weights_scale. */
int p36_gpu_router_topk_tensor(
        p36_gpu_tensor       *selected,
        p36_gpu_tensor       *weights,
        const p36_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_used,
        uint32_t                n_tok,
        float                   route_scale);

/* Find the highest and second-highest F32 entries on the GPU.  The result is
 * two int32 token ids in descending logit order.  Unsupported backends return
 * zero so their existing host-logit path remains unchanged. */
int p36_gpu_top2_tensor(
        p36_gpu_tensor       *out_ids,
        const p36_gpu_tensor *logits,
        uint32_t              count);

/* Top-2 of every row of a `rows x count` logits block, plus a device copy of
 * the last row into `last_logits`.  Speculative verification then needs one
 * small int readback instead of `rows` full-vocabulary host reads, each of
 * which drains the queue.  Unsupported backends return zero so callers keep
 * their per-row readback path. */
int p36_gpu_spec_verify_tops(
        p36_gpu_tensor       *last_logits,
        p36_gpu_tensor       *out_ids,
        const p36_gpu_tensor *spec_logits,
        uint32_t              rows,
        uint32_t              count);
/* Exact top-8 F32 entries on the GPU, descending by value (ties broken by
 * lower index, matching the CPU top-k sampler). `out` is one 64-byte buffer:
 * int32 ids[8] at offset 0, then float32 vals[8] at offset 32. Unsupported
 * backends return zero so the caller's host-logit path remains unchanged. */
int p36_gpu_topk8_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *logits,
        uint32_t              count);

/* GPU min-p prefilter for the default sampler (vulkan/logits_minp_pack.comp).
 * Writes, per `P36_GPU_MINP_BLOCK`-logit block, the ascending-index survivors
 * of `(v - max)/temperature > reject_scaled` into a fixed-capacity bucket, so
 * the host can run p36_sample_full_vocab's expf/sum/draw over the survivors
 * only and stay bit-identical. `out` must hold P36_GPU_MINP_BYTES(count).
 * Layout, all uint32 words: w[0] = max as an order-preserving uint (only pass 2
 * reads it), then nb block maxima, then nb per-block survivor counts (a count
 * above P36_GPU_MINP_CAP means overflow -- the caller must fall back), then
 * nb*cap survivor indices, then nb*cap f32 bit patterns of the scaled logits.
 * Returns zero when unsupported, so callers keep their scalar path. */
#define P36_GPU_MINP_BLOCK     256u  /* logits per workgroup */
#define P36_GPU_MINP_CAP       16u   /* survivors one block may record */
#define P36_GPU_MINP_HDR_WORDS 4u
#define P36_GPU_MINP_NB(count) \
    (((count) + P36_GPU_MINP_BLOCK - 1u) / P36_GPU_MINP_BLOCK)
#define P36_GPU_MINP_WORDS(count) \
    (P36_GPU_MINP_HDR_WORDS + 2u * P36_GPU_MINP_NB(count) + \
     2u * P36_GPU_MINP_NB(count) * P36_GPU_MINP_CAP)
#define P36_GPU_MINP_BYTES(count) \
    ((uint64_t)P36_GPU_MINP_WORDS(count) * sizeof(uint32_t))

int p36_gpu_logits_minp_pack(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *logits,
        uint32_t              count,
        float                 temperature,
        float                 reject_scaled);

/* One routed expert weight: GGUF offset/type of the 3D expert tensor plus the
 * optional per-expert ".scale" tensor (n_expert f32). */
typedef struct {
    uint64_t offset;
    uint32_t type;
    uint64_t scales_offset;
    bool has_scales;
} p36_gpu_moe_weight;

/* Routed expert FFN over n_tok rows of x with per-token expert selections,
 * mirroring the routed half of p36_forward_ffn() including all .scale
 * applications.  Supports Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ2_XXS / IQ3_XXS / IQ2_S / IQ3_S expert quants. */
int p36_gpu_moe_ffn_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const p36_gpu_moe_weight *gate,
        const p36_gpu_moe_weight *up,
        const p36_gpu_moe_weight *down,
        const p36_gpu_tensor *selected,
        const p36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const p36_gpu_tensor *x,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

int p36_gpu_moe_ffn_q8_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const p36_gpu_moe_weight *gate,
        const p36_gpu_moe_weight *up,
        const p36_gpu_moe_weight *down,
        const p36_gpu_tensor *selected,
        const p36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const p36_gpu_tensor *qx,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

int p36_gpu_moe_ffn_f32_tensor(
        p36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const p36_gpu_moe_weight *gate,
        const p36_gpu_moe_weight *up,
        const p36_gpu_moe_weight *down,
        const p36_gpu_tensor *selected,
        const p36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const p36_gpu_tensor *x,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

#endif
