/* =========================================================================
 * Inkling-2-Mini (inkling2_loop) Vulkan kernels and dispatch wrappers.
 * ========================================================================= */

static p36_vk_kernel p36_vk_ink_mix;
static p36_vk_kernel p36_vk_ink_post_attn_mix;
static p36_vk_kernel p36_vk_ink_qkv_rel_q4k;
static p36_vk_kernel p36_vk_ink_qkv_rel_q6k;
static p36_vk_kernel p36_vk_ink_wo_q4k_dec;
static p36_vk_kernel p36_vk_ink_f32_gemm;
static p36_vk_kernel p36_vk_ink_f32_gemm_exact;
static p36_vk_kernel p36_vk_ink_ctrl_prep;
static p36_vk_kernel p36_vk_ink_ctrl_prep_down;
static p36_vk_kernel p36_vk_ink_ctrl_gate;
static p36_vk_kernel p36_vk_ink_ctrl_gate_dec;
static p36_vk_kernel p36_vk_ink_f32_to_f16;
static p36_vk_kernel p36_vk_ink_conv;
static p36_vk_kernel p36_vk_ink_conv_kv_store;
static p36_vk_kernel p36_vk_ink_clip_norm;
static p36_vk_kernel p36_vk_ink_rms_norm_q8k;
static p36_vk_kernel p36_vk_ink_router_topk;
static p36_vk_kernel p36_vk_ink_moe_reduce;
static p36_vk_kernel p36_vk_ink_softcap;
static p36_vk_kernel p36_vk_ink_attn_decode;
static p36_vk_kernel p36_vk_ink_attn_combine;
static p36_vk_kernel p36_vk_ink_attn_prefill;
static p36_gpu_tensor *p36_vk_ink_attn_parts = NULL;
static p36_vk_kernel p36_vk_ink_gu_dec_iq2xxs;
static p36_vk_kernel p36_vk_ink_gu_dec_iq2xs;
static p36_vk_kernel p36_vk_ink_gu_dec_iq2s;
static p36_vk_kernel p36_vk_ink_gu_dec_iq3xxs;
static p36_vk_kernel p36_vk_ink_gu_dec_iq3s;
static p36_vk_kernel p36_vk_ink_gu_dec_q4k;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq2xxs;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq2xs;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq2s;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq3xxs;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq3s;
static p36_vk_kernel p36_vk_ink_gu_gemm_q4k;
static p36_vk_kernel p36_vk_ink_down_dec_iq4nl;
static p36_vk_kernel p36_vk_ink_down_dec_q5_0;
static p36_vk_kernel p36_vk_ink_down_dec_nvfp4;
static p36_vk_kernel p36_vk_ink_down_red_conv;
static p36_vk_kernel p36_vk_ink_down_gemm_iq4nl;
static p36_vk_kernel p36_vk_ink_down_gemm_q5_0;
static p36_vk_kernel p36_vk_ink_down_gemm_nvfp4;
static p36_vk_kernel p36_vk_ink_down_dec_iq4nl_split;
static p36_vk_kernel p36_vk_ink_down_gemm_iq4nl_split;
static p36_vk_kernel p36_vk_ink_gu_dec_q4k_sm, p36_vk_ink_gu_dec_iq3xxs_sm, p36_vk_ink_gu_gemm_q4k_sm, p36_vk_ink_gu_gemm_iq3xxs_sm;
static bool p36_vk_ink_kernels_inited = false;

static void p36_vk_inkling2_init_kernels_unlocked(void) {
    if (p36_vk_ink_kernels_inited) return;
    p36_vk_ink_mix            = P36_VK_KERNEL("vulkan/inkling2_mix.spv", 6, 40, (1u << 0) | (1u << 3) | (1u << 5));
    p36_vk_ink_post_attn_mix  = P36_VK_KERNEL("vulkan/inkling2_post_attn_mix.spv", 9, 32, (1u << 0) | (1u << 4) | (1u << 5) | (1u << 8));
    p36_vk_ink_qkv_rel_q4k    = P36_VK_KERNEL("vulkan/inkling2_qkv_rel_q4k_decode.spv", 11, 20, (1u << 7) | (1u << 8) | (1u << 9) | (1u << 10));
    p36_vk_ink_qkv_rel_q6k    = P36_VK_KERNEL("vulkan/inkling2_qkv_rel_q6k_decode.spv", 11, 20, (1u << 7) | (1u << 8) | (1u << 9) | (1u << 10));
    p36_vk_ink_wo_q4k_dec     = P36_VK_KERNEL("vulkan/inkling2_wo_q4k_decode.spv", 3, 8, 1u << 2);
    p36_vk_ink_f32_gemm       = P36_VK_KERNEL("vulkan/inkling2_f32_gemm.spv", 3, 12, 1u << 2);
    p36_vk_ink_f32_gemm_exact = P36_VK_KERNEL("vulkan/inkling2_f32_gemm_exact.spv", 3, 12, 1u << 2);
    p36_vk_ink_ctrl_prep      = P36_VK_KERNEL("vulkan/inkling2_control_prep.spv", 3, 4, 1u << 2);
    p36_vk_ink_ctrl_prep_down = P36_VK_KERNEL("vulkan/inkling2_control_prep_down.spv", 6, 16, 1u << 3);
    p36_vk_ink_ctrl_gate      = P36_VK_KERNEL("vulkan/inkling2_control_gate.spv", 10, 28, 1u << 5);
    p36_vk_ink_ctrl_gate_dec  = P36_VK_KERNEL("vulkan/inkling2_control_gate_decode.spv", 10, 28, 1u << 5);
    p36_vk_ink_f32_to_f16     = P36_VK_KERNEL("vulkan/inkling2_f32_to_f16.spv", 2, 4, 1u << 1);
    p36_vk_ink_conv           = P36_VK_KERNEL("vulkan/inkling2_conv.spv", 6, 28, (1u << 2) | (1u << 5));
    /* Int8 KV by default; must match p36_ink_kv_row_bytes() in core_inkling2.inc. */
    const bool kv_f16 = getenv("P36_INK_KV_F16") != NULL;
    p36_vk_ink_conv_kv_store  = P36_VK_KERNEL(kv_f16 ? "vulkan/inkling2_conv_kv_store.spv" : "vulkan/inkling2_conv_kv_store_q8.spv", 10, 32, (1u << 4) | (1u << 5) | (1u << 8) | (1u << 9));
    p36_vk_ink_clip_norm      = P36_VK_KERNEL("vulkan/inkling2_clip_norm.spv", 4, 28, (1u << 0) | (1u << 1));
    p36_vk_ink_rms_norm_q8k   = P36_VK_KERNEL("vulkan/inkling2_rms_norm_q8k.spv", 6, 24, (1u << 2) | (1u << 3));
    p36_vk_ink_router_topk    = P36_VK_KERNEL("vulkan/inkling2_router_topk.spv", 4, 16, (1u << 2) | (1u << 3));
    p36_vk_ink_moe_reduce     = P36_VK_KERNEL("vulkan/inkling2_moe_reduce.spv", 3, 8, 1u << 2);
    p36_vk_ink_softcap        = P36_VK_KERNEL("vulkan/inkling2_softcap.spv", 1, 12, 1u << 0);
    p36_vk_ink_attn_decode    = P36_VK_KERNEL(kv_f16 ? "vulkan/inkling2_attn_decode.spv" : "vulkan/inkling2_attn_decode_q8.spv", 9, 32, (1u << 7) | (1u << 8));
    p36_vk_ink_attn_combine   = P36_VK_KERNEL("vulkan/inkling2_attn_combine.spv", 3, 4, 1u << 2);
    p36_vk_ink_attn_prefill   = P36_VK_KERNEL(kv_f16 ? "vulkan/inkling2_attn_prefill.spv" : "vulkan/inkling2_attn_prefill_q8.spv", 10, 32, 1u << 7);
    p36_vk_ink_gu_dec_iq2xxs  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq2xxs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_iq2xs   = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq2xs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_iq2s    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq2s.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_iq3xxs  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq3xxs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_iq3s    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq3s.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_q4k     = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_q4k.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq2xxs = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq2xxs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq2xs  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq2xs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq2s   = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq2s.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq3xxs = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq3xxs.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq3s   = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq3s.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_q4k    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_q4k.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_q4k_sm     = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_q4k_sm.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_dec_iq3xxs_sm  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq3xxs_sm.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_q4k_sm    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_q4k_sm.spv", 6, 32, 1u << 3);
    p36_vk_ink_gu_gemm_iq3xxs_sm = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq3xxs_sm.spv", 6, 32, 1u << 3);
    p36_vk_ink_down_dec_iq4nl = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_iq4nl.spv", 6, 20, 1u << 4);
    p36_vk_ink_down_dec_q5_0  = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_q5_0.spv", 6, 20, 1u << 4);
    p36_vk_ink_down_dec_nvfp4 = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_nvfp4.spv", 6, 20, 1u << 4);
    p36_vk_ink_down_red_conv  = P36_VK_KERNEL("vulkan/inkling2_moe_down_reduce_conv.spv", 5, 12, (1u << 1) | (1u << 3));
    p36_vk_ink_down_gemm_iq4nl= P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_iq4nl.spv", 5, 20, 1u << 3);
    p36_vk_ink_down_gemm_q5_0 = P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_q5_0.spv", 5, 20, 1u << 3);
    p36_vk_ink_down_gemm_nvfp4= P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_nvfp4.spv", 5, 20, 1u << 3);
    p36_vk_ink_down_dec_iq4nl_split = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_iq4nl_split.spv", 6, 20, 1u << 4);
    p36_vk_ink_down_gemm_iq4nl_split = P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_iq4nl_split.spv", 5, 20, 1u << 3);
    p36_vk_ink_kernels_inited = true;
}

static void p36_vk_inkling2_cleanup_unlocked(void) {
    if (!p36_vk_ink_kernels_inited) return;
    p36_vk_kernel_destroy(&p36_vk_ink_mix);
    p36_vk_kernel_destroy(&p36_vk_ink_post_attn_mix);
    p36_vk_kernel_destroy(&p36_vk_ink_qkv_rel_q4k);
    p36_vk_kernel_destroy(&p36_vk_ink_qkv_rel_q6k);
    p36_vk_kernel_destroy(&p36_vk_ink_wo_q4k_dec);
    p36_vk_kernel_destroy(&p36_vk_ink_f32_gemm);
    p36_vk_kernel_destroy(&p36_vk_ink_f32_gemm_exact);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_prep);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_prep_down);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_gate);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_gate_dec);
    p36_vk_kernel_destroy(&p36_vk_ink_f32_to_f16);
    p36_vk_kernel_destroy(&p36_vk_ink_conv);
    p36_vk_kernel_destroy(&p36_vk_ink_conv_kv_store);
    p36_vk_kernel_destroy(&p36_vk_ink_clip_norm);
    p36_vk_kernel_destroy(&p36_vk_ink_rms_norm_q8k);
    p36_vk_kernel_destroy(&p36_vk_ink_router_topk);
    p36_vk_kernel_destroy(&p36_vk_ink_moe_reduce);
    p36_vk_kernel_destroy(&p36_vk_ink_softcap);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_decode);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_combine);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_prefill);
    if (p36_vk_ink_attn_parts) {
        p36_vk_tensor_free_unlocked(p36_vk_ink_attn_parts);
        p36_vk_ink_attn_parts = NULL;
    }
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq2xs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq2s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq3s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_q4k);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq2xs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq3s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_q4k);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_q4k_sm);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq3xxs_sm);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_q4k_sm);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq3xxs_sm);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_iq4nl);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_q5_0);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_nvfp4);
    p36_vk_kernel_destroy(&p36_vk_ink_down_red_conv);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_iq4nl);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_q5_0);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_nvfp4);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_iq4nl_split);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_iq4nl_split);
    p36_vk_ink_kernels_inited = false;
}

/* Default on (P36_INK_DOWN_SPLIT=0 disables): mark an IQ4_NL ffn_down_exps tensor [in_dim, out_dim, n_expert]
 * to be stored in the split layout (see p36_vk_ink_split_rows).  Must run before
 * anything caches the tensor (the engine calls it before prewarm); a tensor that is
 * already resident in the GGUF layout is refused, so the two layouts never mix.
 * Returns 1 if registered. */
static int p36_vk_ink_split_add(const void *model_map, uint64_t model_size, uint64_t offset,
                                p36_vk_ink_split_entry e) {
    const uint64_t expert_bytes = (uint64_t)e.out_dim * e.blocks * e.block_bytes;
    if (!model_map || !expert_bytes || e.bytes % expert_bytes || e.bytes > UINT32_MAX) return 0;
    e.src = p36_gpu_weight_bytes(model_map, model_size, offset, e.bytes);
    if (!e.src) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    int ok = 0;
    if (p36_vk_ink_split_map != model_map || p36_vk_ink_split_map_size != model_size) {
        p36_vk_ink_split_n = 0;
        p36_vk_ink_split_map = model_map;
        p36_vk_ink_split_map_size = model_size;
    }
    bool resident = false;
    for (p36_vk_weight *w = p36_vk.weights; w; w = w->next) resident |= w->source == e.src;
    bool dup = false;
    for (uint32_t i = 0; i < p36_vk_ink_split_n; i++) dup |= p36_vk_ink_split[i].src == e.src;
    if (!resident && !dup && p36_vk_ink_split_n < P36_VK_INK_SPLIT_CAP) {
        p36_vk_ink_split[p36_vk_ink_split_n++] = e;
        ok = 1;
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_down_split_register(const void *model_map, uint64_t model_size,
                                         uint64_t offset, uint64_t bytes,
                                         uint32_t out_dim, uint32_t in_dim) {
    const uint32_t blocks = in_dim / 32u;
    if (in_dim % 32u || blocks % 4u || ((uint64_t)out_dim * blocks) % 8u) return 0;
    return p36_vk_ink_split_add(model_map, model_size, offset, (p36_vk_ink_split_entry){
        .bytes = bytes, .kind = P36_VK_INK_SPLIT_DOWN, .out_dim = out_dim, .blocks = blocks, .block_bytes = 18u });
}

/* Default on (P36_INK_GU_STEPMAJOR=0 disables): store a Q4_K or IQ3_XXS
 * ffn_gate_up_exps tensor [in_dim, rows, n_expert] step-major (kind GU).
 * type is the GGML type (12 = Q4_K, 18 = IQ3_XXS).  Same rules as above. */
int p36_gpu_inkling2_gate_up_sm_register(const void *model_map, uint64_t model_size,
                                         uint64_t offset, uint64_t bytes,
                                         uint32_t rows, uint32_t in_dim, uint32_t type) {
    const uint32_t bb = type == 12u ? 144u : type == 18u ? 98u : 0u;
    if (!bb || in_dim % 256u || rows % 2u) return 0;
    return p36_vk_ink_split_add(model_map, model_size, offset, (p36_vk_ink_split_entry){
        .bytes = bytes, .kind = P36_VK_INK_SPLIT_GU, .out_dim = rows, .blocks = in_dim / 256u, .block_bytes = bb });
}

/* Loop-controller weights (F32 in every Inkling GGUF, 227 MB read per decode
 * token): the decode kernels read an F16 copy (+57 MB), made once on the GPU
 * from the resident F32 tensor (a CPU conversion re-read the mmap from disk,
 * ~0.6 s). Prefill keeps reading F32. The copy lives in packed_weights, so a
 * model switch frees it. Caller holds p36_vk_mu. */
static p36_gpu_tensor *p36_vk_ink_f16w_unlocked(const unsigned char *src, uint64_t f32_bytes) {
    for (p36_vk_packed_weight *w = p36_vk.packed_weights; w; w = w->next) {
        if (w->source == src && w->src_bytes == f32_bytes && w->type == P36_VK_TENSOR_F16)
            return w->tensor;
    }
    if ((f32_bytes % 16u) != 0) return NULL;
    p36_gpu_tensor *f32_t = p36_vk_weight_get_unlocked(src, f32_bytes);
    p36_gpu_tensor *t = f32_t ? p36_vk_arena_alloc_unlocked(f32_bytes / 2u) : NULL;
    p36_vk_packed_weight *entry = t ? calloc(1, sizeof(*entry)) : NULL;
    if (!entry) {
        if (t) p36_vk_tensor_free_unlocked(t);
        return NULL;
    }
    const uint32_t n4 = (uint32_t)(f32_bytes / 16u);
    const p36_gpu_tensor *bindings[2] = { f32_t, t };
    if (!p36_vk_run_unlocked("inkling2_f32_to_f16", &p36_vk_ink_f32_to_f16, bindings, &n4, sizeof(n4),
                             (n4 + 255u) / 256u, 1, 1)) {
        free(entry);
        p36_vk_tensor_free_unlocked(t);
        return NULL;
    }
    entry->source = src;
    entry->src_bytes = f32_bytes;
    entry->type = P36_VK_TENSOR_F16;
    entry->tensor = t;
    entry->next = p36_vk.packed_weights;
    p36_vk.packed_weights = entry;
    return t;
}

static p36_gpu_tensor *p36_vk_ink_f32w_unlocked(const unsigned char *src, uint64_t f32_bytes) {
    return p36_vk_weight_get_unlocked(src, f32_bytes);
}

int p36_gpu_inkling2_f16_weight_warm(const void *model_map, uint64_t model_size,
                                     uint64_t offset, uint64_t f32_bytes) {
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *src = p36_gpu_weight_bytes(model_map, model_size, offset, f32_bytes);
    int ok = src && p36_vk_ink_f16w_unlocked(src, f32_bytes) != NULL;
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_mix_tensor(
        p36_gpu_tensor       *blocks_normed,
        const p36_gpu_tensor *cur,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              query_offset,
        uint64_t              query_bytes,
        p36_gpu_tensor       *out,
        uint32_t              n_tok,
        uint32_t              n_embd,
        uint32_t              n_blocks,
        uint32_t              include_cur,
        uint32_t              n_q,
        uint32_t              q_offset,
        uint32_t              state_stride,
        uint32_t              out_stride,
        uint32_t              push_slot) {
    if (n_tok == 0 || n_embd == 0) return 0;
    uint64_t tok_bytes = (uint64_t)n_tok * n_embd * sizeof(float);
    if ((include_cur != 0u || push_slot != 0xffffffffu) &&
        !p36_gpu_tensor_range_ok(cur, 0, tok_bytes)) {
        fprintf(stderr, "p36: inkling2_mix cur OOB (need %llu B)\n", (unsigned long long)tok_bytes);
        return 0;
    }
    if (n_blocks > 0u || push_slot != 0xffffffffu) {
        uint32_t max_blk = n_blocks > 0u ? (n_blocks - 1u) : 0u;
        if (push_slot != 0xffffffffu && push_slot > max_blk) max_blk = push_slot;
        uint64_t blk_need = ((uint64_t)max_blk * state_stride + (uint64_t)n_tok * n_embd) * sizeof(float);
        if (!p36_gpu_tensor_range_ok(blocks_normed, 0, blk_need)) {
            fprintf(stderr, "p36: inkling2_mix blocks_normed OOB (slot=%u stride=%u need=%llu B)\n",
                    max_blk, state_stride, (unsigned long long)blk_need);
            return 0;
        }
    }
    if (n_q > 0u) {
        uint64_t out_need = ((uint64_t)(n_q - 1u) * out_stride + (uint64_t)n_tok * n_embd) * sizeof(float);
        if (!out || !p36_gpu_tensor_range_ok(out, 0, out_need)) {
            fprintf(stderr, "p36: inkling2_mix out OOB (n_q=%u out_stride=%u need=%llu B)\n",
                    n_q, out_stride, (unsigned long long)out_need);
            return 0;
        }
        uint64_t q_need = (uint64_t)(q_offset + n_q) * n_embd * sizeof(float);
        if (query_bytes < q_need) {
            fprintf(stderr, "p36: inkling2_mix query OOB (q_off=%u n_q=%u need=%llu B got=%llu B)\n",
                    q_offset, n_q, (unsigned long long)q_need, (unsigned long long)query_bytes);
            return 0;
        }
    }
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const p36_gpu_tensor *q_tensor = cur;
    if (n_q > 0 && query_bytes > 0) {
        const unsigned char *src = p36_gpu_weight_bytes(model_map, model_size, query_offset, query_bytes);
        if (!src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
        q_tensor = p36_vk_weight_get_unlocked(src, query_bytes);
        if (!q_tensor) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    }
    const p36_gpu_tensor *bindings[6] = {
        blocks_normed, cur, q_tensor, out ? out : blocks_normed, cur, out ? out : blocks_normed
    };
    struct {
        uint32_t n_embd, n_blocks, include_cur, n_q;
        uint32_t q_offset, state_stride, out_stride, push_slot;
        uint32_t norm_mode;
        float eps;
    } pc = { n_embd, n_blocks, include_cur, n_q, q_offset, state_stride, out_stride, push_slot, 0u, 1.0e-6f };
    ok = p36_vk_run_unlocked("inkling2_mix", &p36_vk_ink_mix, bindings, &pc, sizeof(pc), n_tok, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_mix_norm_tensor(
        p36_gpu_tensor       *blocks_normed,
        const p36_gpu_tensor *cur,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              query_offset,
        uint64_t              query_bytes,
        uint64_t              norm_w_offset,
        p36_gpu_tensor       *norm_out,
        p36_gpu_tensor       *q8_out,
        uint32_t              n_tok,
        uint32_t              n_embd,
        uint32_t              n_blocks,
        uint32_t              include_cur,
        uint32_t              state_stride,
        float                 eps) {
    if (n_tok == 0 || n_embd == 0) return 0;
    uint64_t tok_bytes = (uint64_t)n_tok * n_embd * sizeof(float);
    if ((include_cur != 0u && !p36_gpu_tensor_range_ok(cur, 0, tok_bytes)) ||
        !p36_gpu_tensor_range_ok(norm_out, 0, tok_bytes) ||
        (n_blocks > 0u && !p36_gpu_tensor_range_ok(blocks_normed, 0,
            ((uint64_t)(n_blocks - 1u) * state_stride + (uint64_t)n_tok * n_embd) * sizeof(float))) ||
        (q8_out && !p36_gpu_tensor_range_ok(q8_out, 0, (uint64_t)n_tok * (n_embd / 256u) * sizeof(p36_vk_block_q8_K)))) {
        fprintf(stderr, "p36: inkling2_mix_norm buffer OOB (n_tok=%u n_blocks=%u stride=%u)\n",
                n_tok, n_blocks, state_stride);
        return 0;
    }
    int ok = 0;
    uint64_t nw_bytes = (uint64_t)n_embd * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *q_src = p36_gpu_weight_bytes(model_map, model_size, query_offset, query_bytes);
    const unsigned char *nw_src = p36_gpu_weight_bytes(model_map, model_size, norm_w_offset, nw_bytes);
    if (!q_src || !nw_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *q_t = p36_vk_weight_get_unlocked(q_src, query_bytes);
    p36_gpu_tensor *nw_t = q_t ? p36_vk_weight_get_unlocked(nw_src, nw_bytes) : NULL;
    if (!nw_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = {
        blocks_normed, cur, q_t, norm_out, nw_t, q8_out ? q8_out : norm_out
    };
    struct {
        uint32_t n_embd, n_blocks, include_cur, n_q;
        uint32_t q_offset, state_stride, out_stride, push_slot;
        uint32_t norm_mode;
        float eps;
    } pc = {
        n_embd, n_blocks, include_cur, 1u,
        0u, state_stride, state_stride, 0xffffffffu,
        q8_out ? 2u : 1u, eps
    };
    ok = p36_vk_run_unlocked("inkling2_mix_norm", &p36_vk_ink_mix, bindings, &pc, sizeof(pc), n_tok, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_post_attn_mix_tensor(
        p36_gpu_tensor       *blocks_normed,
        const p36_gpu_tensor *wo_out,
        const p36_gpu_tensor *inp_sa,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_w_offset,
        p36_gpu_tensor       *conv_state,
        p36_gpu_tensor       *ffn_inp,
        uint64_t              query_offset,
        uint64_t              query_bytes,
        uint64_t              norm_w_offset,
        bool                  has_norm,
        p36_gpu_tensor       *out,
        uint32_t              n_embd,
        uint32_t              n_blocks,
        uint32_t              state_stride,
        uint32_t              push_slot,
        uint32_t              has_res,
        float                 inv_scale,
        float                 eps) {
    int ok = 0;
    uint64_t cw_bytes = (uint64_t)n_embd * 4u * sizeof(float);
    uint64_t nw_bytes = (uint64_t)n_embd * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *cw_src = p36_gpu_weight_bytes(model_map, model_size, conv_w_offset, cw_bytes);
    const unsigned char *q_src = p36_gpu_weight_bytes(model_map, model_size, query_offset, query_bytes);
    if (!cw_src || !q_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *cw_t = p36_vk_weight_get_unlocked(cw_src, cw_bytes);
    p36_gpu_tensor *q_t = cw_t ? p36_vk_weight_get_unlocked(q_src, query_bytes) : NULL;
    const p36_gpu_tensor *nw_t = wo_out;
    if (has_norm) {
        const unsigned char *nw_src = p36_gpu_weight_bytes(model_map, model_size, norm_w_offset, nw_bytes);
        nw_t = (q_t && nw_src) ? p36_vk_weight_get_unlocked(nw_src, nw_bytes) : NULL;
    }
    if (!q_t || !nw_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[9] = {
        blocks_normed, wo_out, inp_sa, cw_t, conv_state, ffn_inp, q_t, nw_t, out
    };
    struct {
        uint32_t n_embd, n_blocks, state_stride, push_slot, has_res, has_norm;
        float inv_scale, eps;
    } pc = {
        n_embd, n_blocks, state_stride, push_slot, has_res, has_norm ? 1u : 0u, inv_scale, eps
    };
    ok = p36_vk_run_unlocked("inkling2_post_attn_mix", &p36_vk_ink_post_attn_mix, bindings, &pc, sizeof(pc),
                             1, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_control_prep_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *cur,
        const p36_gpu_tensor *msg,
        uint32_t              n_tok,
        uint32_t              n_embd) {
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const p36_gpu_tensor *bindings[3] = { cur, msg, out };
    uint32_t pc = n_embd;
    ok = p36_vk_run_unlocked("inkling2_ctrl_prep", &p36_vk_ink_ctrl_prep, bindings, &pc, sizeof(pc), n_tok, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_control_prep_down_tensor(
        p36_gpu_tensor       *h_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              down_offset,
        const p36_gpu_tensor *cur,
        const p36_gpu_tensor *msg,
        uint32_t              n_embd,
        uint32_t              rank) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)2u * n_embd * rank * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *w_src = p36_gpu_weight_bytes(model_map, model_size, down_offset, w_bytes);
    if (!w_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w_t = p36_vk_ink_f16w_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { w_t, cur, msg, h_out, w_t, w_t };
    struct { uint32_t n_embd, cur_stride4, msg_stride4, h_stride; } pc = { n_embd, 0u, 0u, 0u };
    ok = p36_vk_run_unlocked("inkling2_ctrl_pd", &p36_vk_ink_ctrl_prep_down, bindings, &pc, sizeof(pc), rank, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_control_prep_down3_tensor(
        p36_gpu_tensor       *h_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              down0_offset,
        uint64_t              down1_offset,
        uint64_t              down2_offset,
        const p36_gpu_tensor *cur,
        const p36_gpu_tensor *msg,
        uint32_t              n_embd,
        uint32_t              rank,
        uint32_t              cur_stride,
        uint32_t              msg_stride,
        uint32_t              h_stride) {
    if (n_embd == 0 || rank == 0) return 0;
    uint64_t cur_need = ((uint64_t)2u * cur_stride + n_embd) * sizeof(float);
    uint64_t msg_need = ((uint64_t)2u * msg_stride + n_embd) * sizeof(float);
    uint64_t h_need   = ((uint64_t)2u * h_stride + rank) * sizeof(float);
    if (!p36_gpu_tensor_range_ok(cur, 0, cur_need) ||
        !p36_gpu_tensor_range_ok(msg, 0, msg_need) ||
        !p36_gpu_tensor_range_ok(h_out, 0, h_need)) {
        fprintf(stderr, "p36: inkling2_ctrl_pd3 OOB (cur_str=%u msg_str=%u h_str=%u)\n",
                cur_stride, msg_stride, h_stride);
        return 0;
    }
    int ok = 0;
    uint64_t w_bytes = (uint64_t)2u * n_embd * rank * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *s0 = p36_gpu_weight_bytes(model_map, model_size, down0_offset, w_bytes);
    const unsigned char *s1 = p36_gpu_weight_bytes(model_map, model_size, down1_offset, w_bytes);
    const unsigned char *s2 = p36_gpu_weight_bytes(model_map, model_size, down2_offset, w_bytes);
    if (!s0 || !s1 || !s2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w0 = p36_vk_ink_f16w_unlocked(s0, w_bytes);
    p36_gpu_tensor *w1 = w0 ? p36_vk_ink_f16w_unlocked(s1, w_bytes) : NULL;
    p36_gpu_tensor *w2 = w1 ? p36_vk_ink_f16w_unlocked(s2, w_bytes) : NULL;
    if (!w2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { w0, cur, msg, h_out, w1, w2 };
    struct { uint32_t n_embd, cur_stride4, msg_stride4, h_stride; } pc = {
        n_embd, cur_stride >> 2u, msg_stride >> 2u, h_stride
    };
    ok = p36_vk_run_unlocked("inkling2_ctrl_pd", &p36_vk_ink_ctrl_prep_down, bindings, &pc, sizeof(pc), rank, 3, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_conv_kv_store_tensor(
        const p36_gpu_tensor *k_in,
        const p36_gpu_tensor *v_in,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              k_weight_offset,
        uint64_t              v_weight_offset,
        p36_gpu_tensor       *k_state,
        p36_gpu_tensor       *v_state,
        const p36_gpu_tensor *k_dyn,
        const p36_gpu_tensor *v_dyn,
        p36_gpu_tensor       *cache_k,
        p36_gpu_tensor       *cache_v,
        uint32_t              n_head_kv,
        uint32_t              n_tok,
        uint32_t              pos0,
        uint32_t              kv_cap,
        uint32_t              is_swa,
        float                 inv_scale,
        float                 qk_clip) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_head_kv * 128u * 4u * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *kw_src = p36_gpu_weight_bytes(model_map, model_size, k_weight_offset, w_bytes);
    const unsigned char *vw_src = p36_gpu_weight_bytes(model_map, model_size, v_weight_offset, w_bytes);
    if (!kw_src || !vw_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *kw_t = p36_vk_weight_get_unlocked(kw_src, w_bytes);
    p36_gpu_tensor *vw_t = kw_t ? p36_vk_weight_get_unlocked(vw_src, w_bytes) : NULL;
    if (!vw_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[10] = {
        k_in, v_in, kw_t, vw_t, k_state, v_state,
        k_dyn ? k_dyn : k_in, v_dyn ? v_dyn : v_in, cache_k, cache_v
    };
    struct {
        uint32_t n_head_kv, n_tok, pos0, kv_cap, is_swa, has_dyn;
        float inv_scale, qk_clip;
    } pc = { n_head_kv, n_tok, pos0, kv_cap, is_swa, k_dyn != NULL ? 1u : 0u, inv_scale, qk_clip };
    uint32_t gy = (n_tok >= 3u) ? ((n_tok + 7u) / 8u) : 1u;
    ok = p36_vk_run_unlocked("inkling2_conv_kv", &p36_vk_ink_conv_kv_store, bindings, &pc, sizeof(pc),
                             n_head_kv, gy, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    if (ok && n_tok >= 3u) {
        uint64_t row_bytes = (uint64_t)n_head_kv * 128u * sizeof(float);
        uint64_t src_off = (uint64_t)(n_tok - 3u) * row_bytes;
        ok = p36_gpu_tensor_copy(k_state, 0, k_in, src_off, 3u * row_bytes) &&
             p36_gpu_tensor_copy(v_state, 0, v_in, src_off, 3u * row_bytes);
    }
    return ok;
}

int p36_gpu_inkling2_control_gate_tensor(
        p36_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              forget_offset,
        uint64_t              gain_offset,
        const p36_gpu_tensor *h,
        const p36_gpu_tensor *cur,
        const p36_gpu_tensor *msg,
        uint32_t              n_tok,
        uint32_t              n_embd,
        uint32_t              rank) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_embd * rank * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *f_src = p36_gpu_weight_bytes(model_map, model_size, forget_offset, w_bytes);
    const unsigned char *g_src = p36_gpu_weight_bytes(model_map, model_size, gain_offset, w_bytes);
    if (!f_src || !g_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    /* The decode kernel reads F16 copies, the prefill kernel the F32 tensors. */
    p36_gpu_tensor *(*get)(const unsigned char *, uint64_t) =
        (n_tok == 1u) ? p36_vk_ink_f16w_unlocked : p36_vk_ink_f32w_unlocked;
    p36_gpu_tensor *f_t = get(f_src, w_bytes);
    p36_gpu_tensor *g_t = f_t ? get(g_src, w_bytes) : NULL;
    if (!g_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[10] = { f_t, g_t, h, cur, msg, out, f_t, g_t, f_t, g_t };
    struct { uint32_t n_embd, rank, n_tok, cur_stride, msg_stride, h_stride4, out_stride; } pc = {
        n_embd, rank, n_tok, 0u, 0u, 0u, 0u
    };
    uint32_t gx = (n_tok == 1u) ? ((n_embd + 3u) / 4u) : ((n_embd + 63u) / 64u);
    uint32_t gy = (n_tok == 1u) ? 1u : ((n_tok + 15u) / 16u);
    ok = p36_vk_run_unlocked("inkling2_ctrl_gate",
                             (n_tok == 1u) ? &p36_vk_ink_ctrl_gate_dec : &p36_vk_ink_ctrl_gate,
                             bindings, &pc, sizeof(pc), gx, gy, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_control_gate3_tensor(
        p36_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              f0_offset,
        uint64_t              g0_offset,
        uint64_t              f1_offset,
        uint64_t              g1_offset,
        uint64_t              f2_offset,
        uint64_t              g2_offset,
        const p36_gpu_tensor *h,
        const p36_gpu_tensor *cur,
        const p36_gpu_tensor *msg,
        uint32_t              n_embd,
        uint32_t              rank,
        uint32_t              cur_stride,
        uint32_t              msg_stride,
        uint32_t              h_stride,
        uint32_t              out_stride) {
    if (n_embd == 0 || rank == 0) return 0;
    uint64_t h_need   = ((uint64_t)2u * h_stride + rank) * sizeof(float);
    uint64_t cur_need = ((uint64_t)2u * cur_stride + n_embd) * sizeof(float);
    uint64_t msg_need = ((uint64_t)2u * msg_stride + n_embd) * sizeof(float);
    uint64_t out_need = ((uint64_t)2u * out_stride + n_embd) * sizeof(float);
    if (!p36_gpu_tensor_range_ok(h, 0, h_need) ||
        !p36_gpu_tensor_range_ok(cur, 0, cur_need) ||
        !p36_gpu_tensor_range_ok(msg, 0, msg_need) ||
        !p36_gpu_tensor_range_ok(out, 0, out_need)) {
        fprintf(stderr, "p36: inkling2_ctrl_gate3 OOB (cur_str=%u msg_str=%u h_str=%u out_str=%u)\n",
                cur_stride, msg_stride, h_stride, out_stride);
        return 0;
    }
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_embd * rank * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *fs0 = p36_gpu_weight_bytes(model_map, model_size, f0_offset, w_bytes);
    const unsigned char *gs0 = p36_gpu_weight_bytes(model_map, model_size, g0_offset, w_bytes);
    const unsigned char *fs1 = p36_gpu_weight_bytes(model_map, model_size, f1_offset, w_bytes);
    const unsigned char *gs1 = p36_gpu_weight_bytes(model_map, model_size, g1_offset, w_bytes);
    const unsigned char *fs2 = p36_gpu_weight_bytes(model_map, model_size, f2_offset, w_bytes);
    const unsigned char *gs2 = p36_gpu_weight_bytes(model_map, model_size, g2_offset, w_bytes);
    if (!fs0 || !gs0 || !fs1 || !gs1 || !fs2 || !gs2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *f0 = p36_vk_ink_f16w_unlocked(fs0, w_bytes);
    p36_gpu_tensor *g0 = f0 ? p36_vk_ink_f16w_unlocked(gs0, w_bytes) : NULL;
    p36_gpu_tensor *f1 = g0 ? p36_vk_ink_f16w_unlocked(fs1, w_bytes) : NULL;
    p36_gpu_tensor *g1 = f1 ? p36_vk_ink_f16w_unlocked(gs1, w_bytes) : NULL;
    p36_gpu_tensor *f2 = g1 ? p36_vk_ink_f16w_unlocked(fs2, w_bytes) : NULL;
    p36_gpu_tensor *g2 = f2 ? p36_vk_ink_f16w_unlocked(gs2, w_bytes) : NULL;
    if (!g2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[10] = { f0, g0, h, cur, msg, out, f1, g1, f2, g2 };
    struct { uint32_t n_embd, rank, n_tok, cur_stride, msg_stride, h_stride4, out_stride; } pc = {
        n_embd, rank, 1u, cur_stride, msg_stride, h_stride >> 2u, out_stride
    };
    ok = p36_vk_run_unlocked("inkling2_ctrl_gate", &p36_vk_ink_ctrl_gate_dec, bindings, &pc, sizeof(pc),
                             (n_embd + 3u) / 4u, 3, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_conv_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *cur,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        p36_gpu_tensor       *state,
        const p36_gpu_tensor *dyn,
        const p36_gpu_tensor *res,
        uint32_t              n_channels,
        uint32_t              n_tok,
        uint32_t              n_heads,
        uint32_t              head_dim,
        float                 inv_scale) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_channels * 4u * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *w_src = p36_gpu_weight_bytes(model_map, model_size, weight_offset, w_bytes);
    if (!w_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w_t = p36_vk_weight_get_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { cur, w_t, state, dyn ? dyn : cur, res ? res : cur, out };
    struct {
        uint32_t n_channels, n_tok, has_dyn, n_heads, head_dim, has_res;
        float inv_scale;
    } pc = { n_channels, n_tok, dyn != NULL ? 1u : 0u, n_heads, head_dim, res != NULL ? 1u : 0u, inv_scale };
    uint32_t gy = (n_tok >= 3u) ? ((n_tok + 15u) / 16u) : 1u;
    ok = p36_vk_run_unlocked("inkling2_conv", &p36_vk_ink_conv, bindings, &pc, sizeof(pc),
                             (n_channels + 63u) / 64u, gy, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    if (ok && n_tok >= 3u) {
        uint64_t row_bytes = (uint64_t)n_channels * sizeof(float);
        ok = p36_gpu_tensor_copy(state, 0, cur, (uint64_t)(n_tok - 3u) * row_bytes, 3u * row_bytes);
    }
    return ok;
}

int p36_gpu_inkling2_clip_qr_tensor(
        p36_gpu_tensor       *q_buf,
        p36_gpu_tensor       *r_buf,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_scale_offset,
        uint64_t              q_scale_bytes,
        uint64_t              r_scale_offset,
        uint64_t              r_scale_bytes,
        uint32_t              n_heads,
        uint32_t              n_tok,
        uint32_t              pos0,
        uint32_t              use_tau,
        float                 qk_clip,
        float                 rel_clip,
        float                 log_floor,
        float                 log_alpha) {
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *qs_src = p36_gpu_weight_bytes(model_map, model_size, q_scale_offset, q_scale_bytes);
    const unsigned char *rs_src = p36_gpu_weight_bytes(model_map, model_size, r_scale_offset, r_scale_bytes);
    if (!qs_src || !rs_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *qs_t = p36_vk_weight_get_unlocked(qs_src, q_scale_bytes);
    p36_gpu_tensor *rs_t = qs_t ? p36_vk_weight_get_unlocked(rs_src, r_scale_bytes) : NULL;
    if (!rs_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[4] = { q_buf, r_buf, qs_t, rs_t };
    struct {
        uint32_t n_heads, pos0, use_tau;
        float qk_clip, rel_clip, log_floor, log_alpha;
    } pc = { n_heads, pos0, use_tau, qk_clip, rel_clip, log_floor, log_alpha };
    ok = p36_vk_run_unlocked("inkling2_clip_qr", &p36_vk_ink_clip_norm, bindings, &pc, sizeof(pc),
                             n_heads, n_tok, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_rms_norm_q8k_tensor(
        p36_gpu_tensor       *norm_out,
        p36_gpu_tensor       *q8_out,
        const p36_gpu_tensor *inp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_embd,
        uint32_t              n_tok,
        float                 eps) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_embd * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *w_src = p36_gpu_weight_bytes(model_map, model_size, weight_offset, w_bytes);
    if (!w_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w_t = p36_vk_weight_get_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { inp, w_t, norm_out, q8_out ? q8_out : norm_out, w_t, w_t };
    uint32_t blks = n_embd / 256u;
    struct { uint32_t n_embd, blocks, tok_stride4, q8_stride_blocks, multi_w; float eps; } pc = {
        n_embd, q8_out ? blks : 0u, n_embd >> 2u, blks, 0u, eps
    };
    ok = p36_vk_run_unlocked("inkling2_norm_q8k", &p36_vk_ink_rms_norm_q8k, bindings, &pc, sizeof(pc),
                             n_tok, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_rms_norm_q8k3_tensor(
        p36_gpu_tensor       *norm_out,
        p36_gpu_tensor       *q8_out,
        const p36_gpu_tensor *inp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w0_offset,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              n_embd,
        uint32_t              tok_stride,
        uint32_t              q8_stride_blocks,
        float                 eps) {
    if (n_embd == 0 || (n_embd % 256u) != 0u) return 0;
    uint64_t vec_need = ((uint64_t)2u * tok_stride + n_embd) * sizeof(float);
    uint64_t q8_need  = ((uint64_t)2u * q8_stride_blocks + (n_embd / 256u)) * sizeof(p36_vk_block_q8_K);
    if (!p36_gpu_tensor_range_ok(inp, 0, vec_need) ||
        !p36_gpu_tensor_range_ok(norm_out, 0, vec_need) ||
        !p36_gpu_tensor_range_ok(q8_out, 0, q8_need)) {
        fprintf(stderr, "p36: inkling2_norm_q8k3 OOB (tok_str=%u q8_str_blks=%u)\n",
                tok_stride, q8_stride_blocks);
        return 0;
    }
    int ok = 0;
    uint64_t w_bytes = (uint64_t)n_embd * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *s0 = p36_gpu_weight_bytes(model_map, model_size, w0_offset, w_bytes);
    const unsigned char *s1 = p36_gpu_weight_bytes(model_map, model_size, w1_offset, w_bytes);
    const unsigned char *s2 = p36_gpu_weight_bytes(model_map, model_size, w2_offset, w_bytes);
    if (!s0 || !s1 || !s2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w0 = p36_vk_weight_get_unlocked(s0, w_bytes);
    p36_gpu_tensor *w1 = w0 ? p36_vk_weight_get_unlocked(s1, w_bytes) : NULL;
    p36_gpu_tensor *w2 = w1 ? p36_vk_weight_get_unlocked(s2, w_bytes) : NULL;
    if (!w2) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { inp, w0, norm_out, q8_out, w1, w2 };
    struct { uint32_t n_embd, blocks, tok_stride4, q8_stride_blocks, multi_w; float eps; } pc = {
        n_embd, n_embd / 256u, tok_stride >> 2u, q8_stride_blocks,
        (w0_offset == w1_offset && w0_offset == w2_offset) ? 0u : 1u, eps
    };
    ok = p36_vk_run_unlocked("inkling2_norm_q8k", &p36_vk_ink_rms_norm_q8k, bindings, &pc, sizeof(pc),
                             3, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_attn_tensor(
        p36_gpu_tensor       *out,
        const p36_gpu_tensor *q,
        const p36_gpu_tensor *r,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              rel_bias_offset,
        uint64_t              rel_bias_bytes,
        const p36_gpu_tensor *sinks,
        const p36_gpu_tensor *gate,
        const p36_gpu_tensor *cache_k,
        const p36_gpu_tensor *cache_v,
        const p36_gpu_tensor *new_k,
        const p36_gpu_tensor *new_v,
        uint32_t              n_head,
        uint32_t              n_head_kv,
        uint32_t              n_tok,
        uint32_t              pos0,
        uint32_t              kv_cap,
        uint32_t              is_swa,
        uint32_t              rel_extent,
        float                 scale) {
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *rb_src = p36_gpu_weight_bytes(model_map, model_size, rel_bias_offset, rel_bias_bytes);
    if (!rb_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *rb_t = p36_vk_weight_get_unlocked(rb_src, rel_bias_bytes);
    if (!rb_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[10] = {
        q, r, rb_t, sinks, gate, cache_k, cache_v, out,
        new_k ? new_k : cache_k, new_v ? new_v : cache_v
    };
    if (n_tok == 1u) {
        uint32_t k_len = (is_swa && pos0 >= rel_extent) ? rel_extent : (pos0 + 1u);
        uint32_t n_splits = (k_len >= 128u) ? ((k_len + 63u) / 64u) : 1u;
        /* 64 splits (512 keys each at 32k) beat 16 by -15% attn+combine GPU time with int8 KV; 128 loses in combine. */
        if (n_splits > 64u) n_splits = 64u;
        if (n_splits > 1u && !p36_vk_ink_attn_parts) {
            pthread_mutex_unlock(&p36_vk_mu);
            p36_gpu_tensor *p = p36_gpu_tensor_alloc_scratch((uint64_t)24u * 64u * 132u * sizeof(float));
            pthread_mutex_lock(&p36_vk_mu);
            if (!p36_vk_ink_attn_parts) p36_vk_ink_attn_parts = p;
            else if (p) p36_vk_tensor_free_unlocked(p);
        }
        const p36_gpu_tensor *part_b = p36_vk_ink_attn_parts ? p36_vk_ink_attn_parts : out;
        if (n_splits > 1u && !p36_vk_ink_attn_parts) n_splits = 1u;
        const p36_gpu_tensor *dec_b[9] = { q, r, rb_t, sinks, gate, cache_k, cache_v, out, part_b };
        struct {
            uint32_t n_head, n_head_kv, pos, kv_cap, is_swa, rel_extent, n_splits;
            float scale;
        } pc = { n_head, n_head_kv, pos0, kv_cap, is_swa, rel_extent, n_splits, scale };
        if (p36_vk.prof_kernel && kv_cap)
            p36_vk_prof_next_bytes = ((cache_k->bytes + cache_v->bytes) / kv_cap) * k_len;
        ok = p36_vk_run_unlocked("inkling2_attn_dec", &p36_vk_ink_attn_decode, dec_b, &pc, sizeof(pc),
                                 n_head_kv, n_splits, 1);
        if (ok && n_splits > 1u) {
            const p36_gpu_tensor *cb[3] = { part_b, gate, out };
            ok = p36_vk_run_unlocked("inkling2_attn_comb", &p36_vk_ink_attn_combine, cb, &n_splits, sizeof(n_splits),
                                     n_head, 1, 1);
        }
    } else {
        struct {
            uint32_t n_head, n_head_kv, n_tok, pos0, kv_cap, is_swa, rel_extent;
            float scale;
        } pc = { n_head, n_head_kv, n_tok, pos0, kv_cap, is_swa, rel_extent, scale };
        ok = p36_vk_run_unlocked("inkling2_attn_pf", &p36_vk_ink_attn_prefill, bindings, &pc, sizeof(pc),
                                 n_head_kv, (n_tok + 7u) / 8u, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_router_topk_tensor(
        p36_gpu_tensor       *selected,
        p36_gpu_tensor       *weights,
        const p36_gpu_tensor *logits,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        bool                  has_bias,
        uint32_t              n_tok,
        uint32_t              n_expert,
        uint32_t              n_used,
        float                 weight_scale) {
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const p36_gpu_tensor *bias_t = logits;
    if (has_bias) {
        uint64_t b_bytes = (uint64_t)n_expert * sizeof(float);
        const unsigned char *b_src = p36_gpu_weight_bytes(model_map, model_size, bias_offset, b_bytes);
        if (!b_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
        bias_t = p36_vk_weight_get_unlocked(b_src, b_bytes);
        if (!bias_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    }
    const p36_gpu_tensor *bindings[4] = { logits, bias_t, selected, weights };
    struct { uint32_t n_expert, n_used, has_bias; float weight_scale; } pc = {
        n_expert, n_used, has_bias ? 1u : 0u, weight_scale
    };
    ok = p36_vk_run_unlocked("inkling2_router", &p36_vk_ink_router_topk, bindings, &pc, sizeof(pc),
                             n_tok, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_moe_ffn_tensor(
        p36_gpu_tensor       *out,
        p36_gpu_tensor       *mid_scratch,
        p36_gpu_tensor       *down_scratch,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_up_offset,
        uint32_t              gate_up_type,
        uint64_t              gate_up_bytes,
        uint64_t              gate_up_scale_offset,
        bool                  has_gate_up_scale,
        uint64_t              down_offset,
        uint32_t              down_type,
        uint64_t              down_bytes,
        uint64_t              down_scale_offset,
        bool                  has_down_scale,
        uint64_t              conv_w_offset,
        p36_gpu_tensor       *conv_state,
        const p36_gpu_tensor *res,
        float                 conv_inv_scale,
        const p36_gpu_tensor *x,
        const p36_gpu_tensor *selected,
        const p36_gpu_tensor *weights,
        uint32_t              n_tok,
        uint32_t              in_dim,
        uint32_t              mid_dim,
        uint32_t              out_dim,
        uint32_t              n_expert,
        uint32_t              n_used,
        float                 situ_beta,
        float                 situ_lb) {
    p36_gpu_tensor *tiles = NULL;
    uint32_t tile_count = 0;
    if (n_tok > 1u) {
        uint32_t n_slot = n_tok * n_used;
        tile_count = p36_vk_moe_tile_bound(n_slot, n_expert, 64u);
        tiles = p36_vk_moe_tiles_gpu(selected, n_slot, n_expert, NULL, 64u);
        if (!tiles) {
            const uint32_t *sel_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "ink_moe_sel");
            if (!sel_host) return 0;
            tiles = p36_vk_moe_build_tiles_ex(sel_host, n_slot, n_expert, NULL, 64u, false, &tile_count);
            if (!tiles || tile_count == 0) {
                p36_gpu_tensor_free(tiles);
                return 0;
            }
        }
    }

    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();

    const unsigned char *gu_src = p36_gpu_weight_bytes(model_map, model_size, gate_up_offset, gate_up_bytes);
    const unsigned char *dn_src = p36_gpu_weight_bytes(model_map, model_size, down_offset, down_bytes);
    if (!gu_src || !dn_src) { pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0; }
    p36_gpu_tensor *gu_t = p36_vk_weight_get_unlocked(gu_src, gate_up_bytes);
    p36_gpu_tensor *dn_t = gu_t ? p36_vk_weight_get_unlocked(dn_src, down_bytes) : NULL;
    p36_gpu_tensor *tables = dn_t ? p36_vk_iq_tables_unlocked() : NULL;
    if (!tables) { pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0; }

    const p36_gpu_tensor *gu_sc_t = weights;
    const p36_gpu_tensor *dn_sc_t = weights;
    uint64_t sc_bytes = (uint64_t)n_expert * sizeof(float);
    if (has_gate_up_scale) {
        const unsigned char *sc_src = p36_gpu_weight_bytes(model_map, model_size, gate_up_scale_offset, sc_bytes);
        gu_sc_t = sc_src ? p36_vk_weight_get_unlocked(sc_src, sc_bytes) : NULL;
        if (!gu_sc_t) { pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0; }
    }
    if (has_down_scale) {
        const unsigned char *sc_src = p36_gpu_weight_bytes(model_map, model_size, down_scale_offset, sc_bytes);
        dn_sc_t = sc_src ? p36_vk_weight_get_unlocked(sc_src, sc_bytes) : NULL;
        if (!dn_sc_t) { pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0; }
    }

    p36_vk_kernel *gu_k = NULL;
    p36_vk_kernel *dn_k = NULL;
    /* The cached copy's layout picks the down kernel (P36_INK_DOWN_SPLIT); a split
     * tensor must be IQ4_NL with exactly the dims it was re-laid out for. */
    const p36_vk_ink_split_entry *dn_split = p36_vk_ink_split_find(dn_src, down_bytes);
    if (dn_split && (dn_split->kind != P36_VK_INK_SPLIT_DOWN || down_type != P36_VK_TENSOR_IQ4_NL ||
                     dn_split->out_dim != out_dim || dn_split->blocks != mid_dim / 32u)) {
        pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
    }
    /* Same for gate_up (P36_INK_GU_STEPMAJOR): only Q4_K / IQ3_XXS have _sm kernels. */
    const p36_vk_ink_split_entry *gu_split = p36_vk_ink_split_find(gu_src, gate_up_bytes);
    if (gu_split && (gu_split->kind != P36_VK_INK_SPLIT_GU ||
                     (gate_up_type != P36_VK_TENSOR_Q4_K && gate_up_type != P36_VK_TENSOR_IQ3_XXS) ||
                     gu_split->out_dim != 2u * mid_dim || gu_split->blocks != in_dim / 256u)) {
        pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
    }
    if (n_tok == 1u) {
        switch (gate_up_type) {
            case P36_VK_TENSOR_IQ2_XXS: gu_k = &p36_vk_ink_gu_dec_iq2xxs; break;
            case P36_VK_TENSOR_IQ2_XS:  gu_k = &p36_vk_ink_gu_dec_iq2xs; break;
            case P36_VK_TENSOR_IQ2_S:   gu_k = &p36_vk_ink_gu_dec_iq2s; break;
            case P36_VK_TENSOR_IQ3_XXS: gu_k = gu_split ? &p36_vk_ink_gu_dec_iq3xxs_sm : &p36_vk_ink_gu_dec_iq3xxs; break;
            case P36_VK_TENSOR_IQ3_S:   gu_k = &p36_vk_ink_gu_dec_iq3s; break;
            case P36_VK_TENSOR_Q4_K:    gu_k = gu_split ? &p36_vk_ink_gu_dec_q4k_sm : &p36_vk_ink_gu_dec_q4k; break;
            default: pthread_mutex_unlock(&p36_vk_mu); return 0;
        }
        switch (down_type) {
            case P36_VK_TENSOR_IQ4_NL:  dn_k = dn_split ? &p36_vk_ink_down_dec_iq4nl_split : &p36_vk_ink_down_dec_iq4nl; break;
            case P36_VK_TENSOR_Q5_0:    dn_k = &p36_vk_ink_down_dec_q5_0; break;
            case P36_VK_TENSOR_NVFP4:   dn_k = &p36_vk_ink_down_dec_nvfp4; break;
            default: pthread_mutex_unlock(&p36_vk_mu); return 0;
        }
        struct {
            uint32_t mid_dim, blocks, n_used;
            float situ_beta, inv_beta, situ_lb, inv_lb;
            uint32_t has_exp_scale;
        } gu_pc = { mid_dim, in_dim / 256u, n_used, situ_beta, 1.0f / situ_beta, situ_lb, 1.0f / situ_lb, has_gate_up_scale ? 1u : 0u };
        const p36_gpu_tensor *gu_b[6] = { gu_t, x, selected, mid_scratch, tables, gu_sc_t };
        uint32_t gu_rows = (gate_up_type == P36_VK_TENSOR_IQ2_S) ? 16u : 4u;
        if (p36_vk.prof_kernel && n_expert)
            p36_vk_prof_next_bytes = (gate_up_bytes / n_expert) * n_used;
        ok = p36_vk_run_unlocked("inkling2_moe_gu_dec", gu_k, gu_b, &gu_pc, sizeof(gu_pc),
                                 mid_dim / gu_rows, n_used, 1);
        if (ok) {
            struct { uint32_t out_dim, in_dim, blocks, n_used, has_exp_scale; } dn_pc = {
                out_dim, mid_dim, mid_dim / 32u, n_used, has_down_scale ? 1u : 0u
            };
            const p36_gpu_tensor *dn_b[6] = {
                dn_t, mid_scratch, selected, weights, down_scratch, dn_sc_t
            };
            if (p36_vk.prof_kernel && n_expert)
                p36_vk_prof_next_bytes = (down_bytes / n_expert) * n_used;
            ok = p36_vk_run_unlocked("inkling2_moe_dn_dec", dn_k, dn_b, &dn_pc, sizeof(dn_pc),
                                     out_dim / 16u, n_used, 1);
            if (ok) {
                uint64_t cw_bytes = (uint64_t)out_dim * 4u * sizeof(float);
                const unsigned char *cw_src = p36_gpu_weight_bytes(model_map, model_size, conv_w_offset, cw_bytes);
                p36_gpu_tensor *cw_t = cw_src ? p36_vk_weight_get_unlocked(cw_src, cw_bytes) : NULL;
                if (!cw_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
                struct { uint32_t out_dim, n_used; float inv_scale; } rc_pc = {
                    out_dim, n_used, conv_inv_scale
                };
                const p36_gpu_tensor *rc_b[5] = {
                    down_scratch, out, cw_t, conv_state, res
                };
                ok = p36_vk_run_unlocked("inkling2_moe_dn_rc", &p36_vk_ink_down_red_conv, rc_b, &rc_pc, sizeof(rc_pc),
                                         (out_dim + 63u) / 64u, 1, 1);
            }
        }
    } else {
        switch (gate_up_type) {
            case P36_VK_TENSOR_IQ2_XXS: gu_k = &p36_vk_ink_gu_gemm_iq2xxs; break;
            case P36_VK_TENSOR_IQ2_XS:  gu_k = &p36_vk_ink_gu_gemm_iq2xs; break;
            case P36_VK_TENSOR_IQ2_S:   gu_k = &p36_vk_ink_gu_gemm_iq2s; break;
            case P36_VK_TENSOR_IQ3_XXS: gu_k = gu_split ? &p36_vk_ink_gu_gemm_iq3xxs_sm : &p36_vk_ink_gu_gemm_iq3xxs; break;
            case P36_VK_TENSOR_IQ3_S:   gu_k = &p36_vk_ink_gu_gemm_iq3s; break;
            case P36_VK_TENSOR_Q4_K:    gu_k = gu_split ? &p36_vk_ink_gu_gemm_q4k_sm : &p36_vk_ink_gu_gemm_q4k; break;
            default: pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
        }
        switch (down_type) {
            case P36_VK_TENSOR_IQ4_NL:  dn_k = dn_split ? &p36_vk_ink_down_gemm_iq4nl_split : &p36_vk_ink_down_gemm_iq4nl; break;
            case P36_VK_TENSOR_Q5_0:    dn_k = &p36_vk_ink_down_gemm_q5_0; break;
            case P36_VK_TENSOR_NVFP4:   dn_k = &p36_vk_ink_down_gemm_nvfp4; break;
            default: pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
        }
        struct {
            uint32_t mid_dim, blocks, n_used;
            float situ_beta, inv_beta, situ_lb, inv_lb;
            uint32_t has_exp_scale;
        } gu_pc = { mid_dim, in_dim / 256u, n_used, situ_beta, 1.0f / situ_beta, situ_lb, 1.0f / situ_lb, has_gate_up_scale ? 1u : 0u };
        const p36_gpu_tensor *gu_b[6] = { gu_t, x, tiles, mid_scratch, tables, gu_sc_t };
        ok = p36_vk_run_unlocked("inkling2_moe_gu_gemm", gu_k, gu_b, &gu_pc, sizeof(gu_pc),
                                 (mid_dim + 31u) / 32u, tile_count, 1);
        if (ok) {
            struct { uint32_t out_dim, in_dim, blocks, n_used, has_exp_scale; } dn_pc = {
                out_dim, mid_dim, mid_dim / 32u, n_used, has_down_scale ? 1u : 0u
            };
            const p36_gpu_tensor *dn_b[5] = { dn_t, mid_scratch, tiles, down_scratch, dn_sc_t };
            ok = p36_vk_run_unlocked("inkling2_moe_dn_gemm", dn_k, dn_b, &dn_pc, sizeof(dn_pc),
                                     (out_dim + 63u) / 64u, tile_count, 1);
        }
        if (ok) {
            uint32_t n4 = out_dim / 4u;
            struct { uint32_t n4, n_used; } red_pc = { n4, n_used };
            const p36_gpu_tensor *red_b[3] = { down_scratch, weights, out };
            ok = p36_vk_run_unlocked("inkling2_moe_reduce", &p36_vk_ink_moe_reduce, red_b, &red_pc, sizeof(red_pc),
                                     (n4 + 63u) / 64u, n_tok, 1);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    if (tiles) p36_gpu_tensor_free(tiles);
    return ok;
}

int p36_gpu_inkling2_softcap_tensor(
        p36_gpu_tensor *logits,
        uint32_t        n,
        float           cap) {
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const p36_gpu_tensor *bindings[1] = { logits };
    struct { uint32_t n; float cap, inv_cap; } pc = { n, cap, 1.0f / cap };
    ok = p36_vk_run_unlocked("inkling2_softcap", &p36_vk_ink_softcap, bindings, &pc, sizeof(pc),
                             (n + 63u) / 64u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_qkv_rel_q4k_decode_tensor(
        p36_gpu_tensor       *yq,
        p36_gpu_tensor       *yk,
        p36_gpu_tensor       *yv,
        p36_gpu_tensor       *yr,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              wq_offset,
        uint64_t              wk_offset,
        uint64_t              wv_offset,
        uint64_t              wr_offset,
        const p36_gpu_tensor *q8_0,
        const p36_gpu_tensor *q8_1,
        const p36_gpu_tensor *q8_2,
        uint32_t              in_dim,
        uint32_t              out_q,
        uint32_t              out_k,
        uint32_t              out_v,
        uint32_t              out_r) {
    if ((in_dim % 256u) != 0u || (out_q % 8u) != 0u || (out_k % 8u) != 0u ||
        (out_v % 8u) != 0u || (out_r % 8u) != 0u) return 0;
    uint32_t blocks = in_dim / 256u;
    uint64_t row_bytes = (uint64_t)blocks * 144u;
    uint64_t q_bytes = row_bytes * out_q;
    uint64_t k_bytes = row_bytes * out_k;
    uint64_t v_bytes = row_bytes * out_v;
    uint64_t r_bytes = row_bytes * out_r;
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *q_src = p36_gpu_weight_bytes(model_map, model_size, wq_offset, q_bytes);
    const unsigned char *k_src = p36_gpu_weight_bytes(model_map, model_size, wk_offset, k_bytes);
    const unsigned char *v_src = p36_gpu_weight_bytes(model_map, model_size, wv_offset, v_bytes);
    const unsigned char *r_src = p36_gpu_weight_bytes(model_map, model_size, wr_offset, r_bytes);
    if (!q_src || !k_src || !v_src || !r_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *wq_t = p36_vk_weight_get_unlocked(q_src, q_bytes);
    p36_gpu_tensor *wk_t = wq_t ? p36_vk_weight_get_unlocked(k_src, k_bytes) : NULL;
    p36_gpu_tensor *wv_t = wk_t ? p36_vk_weight_get_unlocked(v_src, v_bytes) : NULL;
    p36_gpu_tensor *wr_t = wv_t ? p36_vk_weight_get_unlocked(r_src, r_bytes) : NULL;
    if (!wr_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[11] = {
        wq_t, wk_t, wv_t, wr_t, q8_0, q8_1, q8_2, yq, yk, yv, yr
    };
    uint32_t wg_q = out_q / 8u;
    uint32_t wg_qk = wg_q + out_k / 8u;
    uint32_t wg_qkv = wg_qk + out_v / 8u;
    uint32_t wg_tot = wg_qkv + out_r / 8u;
    struct { uint32_t blocks, row_words, wg_q, wg_qk, wg_qkv; } pc = {
        blocks, (uint32_t)(row_bytes / 4u), wg_q, wg_qk, wg_qkv
    };
    ok = p36_vk_run_unlocked("inkling2_qkv_rel_q4k", &p36_vk_ink_qkv_rel_q4k, bindings, &pc, sizeof(pc),
                             wg_tot, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_qkv_rel_q6k_decode_tensor(
        p36_gpu_tensor       *yq,
        p36_gpu_tensor       *yk,
        p36_gpu_tensor       *yv,
        p36_gpu_tensor       *yr,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              wq_offset,
        uint64_t              wk_offset,
        uint64_t              wv_offset,
        uint64_t              wr_offset,
        const p36_gpu_tensor *q8_0,
        const p36_gpu_tensor *q8_1,
        const p36_gpu_tensor *q8_2,
        uint32_t              in_dim,
        uint32_t              out_q,
        uint32_t              out_k,
        uint32_t              out_v,
        uint32_t              out_r) {
    /* The shader's block loop is unrolled for in_dim 2304 (9 blocks); anything
     * else takes the generic per-matrix path. */
    if (in_dim != 2304u || (out_q % 8u) != 0u || (out_k % 8u) != 0u ||
        (out_v % 8u) != 0u || (out_r % 8u) != 0u) return 0;
    uint32_t blocks = in_dim / 256u;
    uint64_t row_bytes = (uint64_t)blocks * 210u;
    uint64_t q_bytes = row_bytes * out_q;
    uint64_t k_bytes = row_bytes * out_k;
    uint64_t v_bytes = row_bytes * out_v;
    uint64_t r_bytes = row_bytes * out_r;
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *q_src = p36_gpu_weight_bytes(model_map, model_size, wq_offset, q_bytes);
    const unsigned char *k_src = p36_gpu_weight_bytes(model_map, model_size, wk_offset, k_bytes);
    const unsigned char *v_src = p36_gpu_weight_bytes(model_map, model_size, wv_offset, v_bytes);
    const unsigned char *r_src = p36_gpu_weight_bytes(model_map, model_size, wr_offset, r_bytes);
    if (!q_src || !k_src || !v_src || !r_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *wq_t = p36_vk_weight_get_unlocked(q_src, q_bytes);
    p36_gpu_tensor *wk_t = wq_t ? p36_vk_weight_get_unlocked(k_src, k_bytes) : NULL;
    p36_gpu_tensor *wv_t = wk_t ? p36_vk_weight_get_unlocked(v_src, v_bytes) : NULL;
    p36_gpu_tensor *wr_t = wv_t ? p36_vk_weight_get_unlocked(r_src, r_bytes) : NULL;
    if (!wr_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[11] = {
        wq_t, wk_t, wv_t, wr_t, q8_0, q8_1, q8_2, yq, yk, yv, yr
    };
    uint32_t wg_q = out_q / 8u;
    uint32_t wg_qk = wg_q + out_k / 8u;
    uint32_t wg_qkv = wg_qk + out_v / 8u;
    uint32_t wg_tot = wg_qkv + out_r / 8u;
    struct { uint32_t blocks, row_u16, wg_q, wg_qk, wg_qkv; } pc = {
        blocks, (uint32_t)(row_bytes / 2u), wg_q, wg_qk, wg_qkv
    };
    ok = p36_vk_run_unlocked("inkling2_qkv_rel_q6k", &p36_vk_ink_qkv_rel_q6k, bindings, &pc, sizeof(pc),
                             wg_tot, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_wo_q4k_decode_tensor(
        p36_gpu_tensor       *y,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w_offset,
        const p36_gpu_tensor *x,
        uint32_t              in_dim,
        uint32_t              out_dim) {
    if ((in_dim % 256u) != 0u || (out_dim % 4u) != 0u) return 0;
    uint32_t blocks = in_dim / 256u;
    uint64_t row_bytes = (uint64_t)blocks * 144u;
    uint64_t w_bytes = row_bytes * out_dim;
    int ok = 0;
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *w_src = p36_gpu_weight_bytes(model_map, model_size, w_offset, w_bytes);
    if (!w_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w_t = p36_vk_weight_get_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[3] = { w_t, x, y };
    struct { uint32_t blocks, row_words; } pc = { blocks, (uint32_t)(row_bytes / 4u) };
    ok = p36_vk_run_unlocked("inkling2_wo_q4k_dec", &p36_vk_ink_wo_q4k_dec, bindings, &pc, sizeof(pc),
                             out_dim / 4u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_inkling2_f32_gemm_tensor(
        p36_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const p36_gpu_tensor *x,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_tok) {
    int ok = 0;
    uint64_t w_bytes = (uint64_t)in_dim * out_dim * sizeof(float);
    pthread_mutex_lock(&p36_vk_mu);
    if (!p36_vk.ready) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_vk_inkling2_init_kernels_unlocked();
    const unsigned char *w_src = p36_gpu_weight_bytes(model_map, model_size, weight_offset, w_bytes);
    if (!w_src) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    p36_gpu_tensor *w_t = p36_vk_weight_get_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[3] = { w_t, x, out };
    struct { uint32_t in_dim, out_dim, n_tok; } pc = { in_dim, out_dim, n_tok };
    /* Keep MoE router (out_dim > 64) in exact F32 to preserve top-k routing; use packed f16 for loop_down. */
    p36_vk_kernel *k = (out_dim > 64u) ? &p36_vk_ink_f32_gemm_exact : &p36_vk_ink_f32_gemm;
    ok = p36_vk_run_unlocked("inkling2_f32_gemm", k, bindings, &pc, sizeof(pc),
                             out_dim / 32u, (n_tok + 31u) / 32u, 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}
