/* =========================================================================
 * Inkling-2-Mini (inkling2_loop) Vulkan kernels and dispatch wrappers.
 * ========================================================================= */

static p36_vk_kernel p36_vk_ink_mix;
static p36_vk_kernel p36_vk_ink_ctrl_prep;
static p36_vk_kernel p36_vk_ink_ctrl_prep_down;
static p36_vk_kernel p36_vk_ink_ctrl_gate;
static p36_vk_kernel p36_vk_ink_conv;
static p36_vk_kernel p36_vk_ink_conv_kv_store;
static p36_vk_kernel p36_vk_ink_clip_norm;
static p36_vk_kernel p36_vk_ink_rms_norm_q8k;
static p36_vk_kernel p36_vk_ink_router_topk;
static p36_vk_kernel p36_vk_ink_softcap;
static p36_vk_kernel p36_vk_ink_attn_decode;
static p36_vk_kernel p36_vk_ink_attn_combine;
static p36_vk_kernel p36_vk_ink_attn_prefill;
static p36_gpu_tensor *p36_vk_ink_attn_parts = NULL;
static p36_vk_kernel p36_vk_ink_gu_dec_iq2xxs;
static p36_vk_kernel p36_vk_ink_gu_dec_iq2s;
static p36_vk_kernel p36_vk_ink_gu_dec_iq3xxs;
static p36_vk_kernel p36_vk_ink_gu_dec_iq3s;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq2xxs;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq2s;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq3xxs;
static p36_vk_kernel p36_vk_ink_gu_gemm_iq3s;
static p36_vk_kernel p36_vk_ink_down_dec_iq4nl;
static p36_vk_kernel p36_vk_ink_down_dec_q5_0;
static p36_vk_kernel p36_vk_ink_down_gemm_iq4nl;
static p36_vk_kernel p36_vk_ink_down_gemm_q5_0;
static bool p36_vk_ink_kernels_inited = false;

static void p36_vk_inkling2_init_kernels_unlocked(void) {
    if (p36_vk_ink_kernels_inited) return;
    p36_vk_ink_mix            = P36_VK_KERNEL("vulkan/inkling2_mix.spv", 6, 40, (1u << 0) | (1u << 3) | (1u << 5));
    p36_vk_ink_ctrl_prep      = P36_VK_KERNEL("vulkan/inkling2_control_prep.spv", 3, 4, 1u << 2);
    p36_vk_ink_ctrl_prep_down = P36_VK_KERNEL("vulkan/inkling2_control_prep_down.spv", 4, 4, 1u << 3);
    p36_vk_ink_ctrl_gate      = P36_VK_KERNEL("vulkan/inkling2_control_gate.spv", 6, 12, 1u << 5);
    p36_vk_ink_conv           = P36_VK_KERNEL("vulkan/inkling2_conv.spv", 6, 28, (1u << 2) | (1u << 5));
    p36_vk_ink_conv_kv_store  = P36_VK_KERNEL("vulkan/inkling2_conv_kv_store.spv", 10, 32, (1u << 4) | (1u << 5) | (1u << 8) | (1u << 9));
    p36_vk_ink_clip_norm      = P36_VK_KERNEL("vulkan/inkling2_clip_norm.spv", 4, 28, (1u << 0) | (1u << 1));
    p36_vk_ink_rms_norm_q8k   = P36_VK_KERNEL("vulkan/inkling2_rms_norm_q8k.spv", 4, 12, (1u << 2) | (1u << 3));
    p36_vk_ink_router_topk    = P36_VK_KERNEL("vulkan/inkling2_router_topk.spv", 4, 16, (1u << 2) | (1u << 3));
    p36_vk_ink_softcap        = P36_VK_KERNEL("vulkan/inkling2_softcap.spv", 1, 12, 1u << 0);
    p36_vk_ink_attn_decode    = P36_VK_KERNEL("vulkan/inkling2_attn_decode.spv", 9, 32, (1u << 7) | (1u << 8));
    p36_vk_ink_attn_combine   = P36_VK_KERNEL("vulkan/inkling2_attn_combine.spv", 3, 4, 1u << 2);
    p36_vk_ink_attn_prefill   = P36_VK_KERNEL("vulkan/inkling2_attn_prefill.spv", 8, 32, 1u << 7);
    p36_vk_ink_gu_dec_iq2xxs  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq2xxs.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_dec_iq2s    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq2s.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_dec_iq3xxs  = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq3xxs.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_dec_iq3s    = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_decode_iq3s.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_gemm_iq2xxs = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq2xxs.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_gemm_iq2s   = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq2s.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_gemm_iq3xxs = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq3xxs.spv", 5, 28, 1u << 3);
    p36_vk_ink_gu_gemm_iq3s   = P36_VK_KERNEL("vulkan/inkling2_moe_gate_up_gemm_iq3s.spv", 5, 28, 1u << 3);
    p36_vk_ink_down_dec_iq4nl = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_iq4nl.spv", 5, 16, 1u << 4);
    p36_vk_ink_down_dec_q5_0  = P36_VK_KERNEL("vulkan/inkling2_moe_down_decode_q5_0.spv", 5, 16, 1u << 4);
    p36_vk_ink_down_gemm_iq4nl= P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_iq4nl.spv", 4, 16, 1u << 3);
    p36_vk_ink_down_gemm_q5_0 = P36_VK_KERNEL("vulkan/inkling2_moe_down_gemm_q5_0.spv", 4, 16, 1u << 3);
    p36_vk_ink_kernels_inited = true;
}

static void p36_vk_inkling2_cleanup_unlocked(void) {
    if (!p36_vk_ink_kernels_inited) return;
    p36_vk_kernel_destroy(&p36_vk_ink_mix);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_prep);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_prep_down);
    p36_vk_kernel_destroy(&p36_vk_ink_ctrl_gate);
    p36_vk_kernel_destroy(&p36_vk_ink_conv);
    p36_vk_kernel_destroy(&p36_vk_ink_conv_kv_store);
    p36_vk_kernel_destroy(&p36_vk_ink_clip_norm);
    p36_vk_kernel_destroy(&p36_vk_ink_rms_norm_q8k);
    p36_vk_kernel_destroy(&p36_vk_ink_router_topk);
    p36_vk_kernel_destroy(&p36_vk_ink_softcap);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_decode);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_combine);
    p36_vk_kernel_destroy(&p36_vk_ink_attn_prefill);
    if (p36_vk_ink_attn_parts) {
        p36_vk_tensor_free_unlocked(p36_vk_ink_attn_parts);
        p36_vk_ink_attn_parts = NULL;
    }
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq2s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_dec_iq3s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk_ink_gu_gemm_iq3s);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_iq4nl);
    p36_vk_kernel_destroy(&p36_vk_ink_down_dec_q5_0);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_iq4nl);
    p36_vk_kernel_destroy(&p36_vk_ink_down_gemm_q5_0);
    p36_vk_ink_kernels_inited = false;
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
    p36_gpu_tensor *w_t = p36_vk_weight_get_unlocked(w_src, w_bytes);
    if (!w_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[4] = { w_t, cur, msg, h_out };
    uint32_t pc = n_embd;
    ok = p36_vk_run_unlocked("inkling2_ctrl_pd", &p36_vk_ink_ctrl_prep_down, bindings, &pc, sizeof(pc), rank, 1, 1);
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
    ok = p36_vk_run_unlocked("inkling2_conv_kv", &p36_vk_ink_conv_kv_store, bindings, &pc, sizeof(pc),
                             n_head_kv, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
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
    p36_gpu_tensor *f_t = p36_vk_weight_get_unlocked(f_src, w_bytes);
    p36_gpu_tensor *g_t = f_t ? p36_vk_weight_get_unlocked(g_src, w_bytes) : NULL;
    if (!g_t) { pthread_mutex_unlock(&p36_vk_mu); return 0; }
    const p36_gpu_tensor *bindings[6] = { f_t, g_t, h, cur, msg, out };
    struct { uint32_t n_embd, rank, n_tok; } pc = { n_embd, rank, n_tok };
    uint32_t gx = (n_tok == 1u) ? ((n_embd + 3u) / 4u) : ((n_embd + 63u) / 64u);
    uint32_t gy = (n_tok == 1u) ? 1u : ((n_tok + 15u) / 16u);
    ok = p36_vk_run_unlocked("inkling2_ctrl_gate", &p36_vk_ink_ctrl_gate, bindings, &pc, sizeof(pc),
                             gx, gy, 1);
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
    ok = p36_vk_run_unlocked("inkling2_conv", &p36_vk_ink_conv, bindings, &pc, sizeof(pc),
                             (n_channels + 63u) / 64u, 1, 1);
    pthread_mutex_unlock(&p36_vk_mu);
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
    const p36_gpu_tensor *bindings[4] = { inp, w_t, norm_out, q8_out };
    struct { uint32_t n_embd, blocks; float eps; } pc = { n_embd, n_embd / 256u, eps };
    ok = p36_vk_run_unlocked("inkling2_norm_q8k", &p36_vk_ink_rms_norm_q8k, bindings, &pc, sizeof(pc),
                             n_tok, 1, 1);
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
    const p36_gpu_tensor *bindings[8] = { q, r, rb_t, sinks, gate, cache_k, cache_v, out };
    if (n_tok == 1u) {
        uint32_t k_len = (is_swa && pos0 >= kv_cap) ? kv_cap : (pos0 + 1u);
        uint32_t n_splits = (k_len >= 128u) ? ((k_len + 63u) / 64u) : 1u;
        if (n_splits > 16u) n_splits = 16u;
        if (n_splits > 1u && !p36_vk_ink_attn_parts) {
            pthread_mutex_unlock(&p36_vk_mu);
            p36_gpu_tensor *p = p36_gpu_tensor_alloc_scratch((uint64_t)24u * 16u * 130u * sizeof(float));
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
        uint64_t              down_offset,
        uint32_t              down_type,
        uint64_t              down_bytes,
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
        tile_count = p36_vk_moe_tile_bound(n_slot, n_expert, 32u);
        tiles = p36_vk_moe_tiles_gpu(selected, n_slot, n_expert, NULL, 32u);
        if (!tiles) {
            const uint32_t *sel_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "ink_moe_sel");
            if (!sel_host) return 0;
            tiles = p36_vk_moe_build_tiles_ex(sel_host, n_slot, n_expert, NULL, 32u, false, &tile_count);
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

    p36_vk_kernel *gu_k = NULL;
    p36_vk_kernel *dn_k = NULL;
    if (n_tok == 1u) {
        switch (gate_up_type) {
            case P36_VK_TENSOR_IQ2_XXS: gu_k = &p36_vk_ink_gu_dec_iq2xxs; break;
            case P36_VK_TENSOR_IQ2_S:   gu_k = &p36_vk_ink_gu_dec_iq2s; break;
            case P36_VK_TENSOR_IQ3_XXS: gu_k = &p36_vk_ink_gu_dec_iq3xxs; break;
            case P36_VK_TENSOR_IQ3_S:   gu_k = &p36_vk_ink_gu_dec_iq3s; break;
            default: pthread_mutex_unlock(&p36_vk_mu); return 0;
        }
        switch (down_type) {
            case P36_VK_TENSOR_IQ4_NL:  dn_k = &p36_vk_ink_down_dec_iq4nl; break;
            case P36_VK_TENSOR_Q5_0:    dn_k = &p36_vk_ink_down_dec_q5_0; break;
            default: pthread_mutex_unlock(&p36_vk_mu); return 0;
        }
        struct {
            uint32_t mid_dim, blocks, n_used;
            float situ_beta, inv_beta, situ_lb, inv_lb;
        } gu_pc = { mid_dim, in_dim / 256u, n_used, situ_beta, 1.0f / situ_beta, situ_lb, 1.0f / situ_lb };
        const p36_gpu_tensor *gu_b[5] = { gu_t, x, selected, mid_scratch, tables };
        ok = p36_vk_run_unlocked("inkling2_moe_gu_dec", gu_k, gu_b, &gu_pc, sizeof(gu_pc),
                                 mid_dim / 4u, n_used, 1);
        if (ok) {
            struct { uint32_t out_dim, in_dim, blocks, n_used; } dn_pc = {
                out_dim, mid_dim, mid_dim / 32u, n_used
            };
            const p36_gpu_tensor *dn_b[5] = { dn_t, mid_scratch, selected, weights, out };
            ok = p36_vk_run_unlocked("inkling2_moe_dn_dec", dn_k, dn_b, &dn_pc, sizeof(dn_pc),
                                     out_dim / 4u, 1, 1);
        }
    } else {
        switch (gate_up_type) {
            case P36_VK_TENSOR_IQ2_XXS: gu_k = &p36_vk_ink_gu_gemm_iq2xxs; break;
            case P36_VK_TENSOR_IQ2_S:   gu_k = &p36_vk_ink_gu_gemm_iq2s; break;
            case P36_VK_TENSOR_IQ3_XXS: gu_k = &p36_vk_ink_gu_gemm_iq3xxs; break;
            case P36_VK_TENSOR_IQ3_S:   gu_k = &p36_vk_ink_gu_gemm_iq3s; break;
            default: pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
        }
        switch (down_type) {
            case P36_VK_TENSOR_IQ4_NL:  dn_k = &p36_vk_ink_down_gemm_iq4nl; break;
            case P36_VK_TENSOR_Q5_0:    dn_k = &p36_vk_ink_down_gemm_q5_0; break;
            default: pthread_mutex_unlock(&p36_vk_mu); p36_gpu_tensor_free(tiles); return 0;
        }
        struct {
            uint32_t mid_dim, blocks, n_used;
            float situ_beta, inv_beta, situ_lb, inv_lb;
        } gu_pc = { mid_dim, in_dim / 256u, n_used, situ_beta, 1.0f / situ_beta, situ_lb, 1.0f / situ_lb };
        const p36_gpu_tensor *gu_b[5] = { gu_t, x, tiles, mid_scratch, tables };
        ok = p36_vk_run_unlocked("inkling2_moe_gu_gemm", gu_k, gu_b, &gu_pc, sizeof(gu_pc),
                                 (mid_dim + 63u) / 64u, tile_count, 1);
        if (ok) {
            struct { uint32_t out_dim, in_dim, blocks, n_used; } dn_pc = {
                out_dim, mid_dim, mid_dim / 32u, n_used
            };
            const p36_gpu_tensor *dn_b[4] = { dn_t, mid_scratch, tiles, down_scratch };
            ok = p36_vk_run_unlocked("inkling2_moe_dn_gemm", dn_k, dn_b, &dn_pc, sizeof(dn_pc),
                                     (out_dim + 127u) / 128u, tile_count, 1);
        }
        if (ok) {
            struct { uint32_t n, n_used; } red_pc = { out_dim, n_used };
            const p36_gpu_tensor *red_b[3] = { down_scratch, weights, out };
            ok = p36_vk_run_unlocked("inkling2_moe_reduce", &p36_vk.moe_reduce, red_b, &red_pc, sizeof(red_pc),
                                     (out_dim + 255u) / 256u, n_tok, 1);
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
