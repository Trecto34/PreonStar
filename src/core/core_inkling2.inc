/* =========================================================================
 * Inkling-2-Mini (inkling2_loop) model binding, Vulkan runtime, and forward.
 * ========================================================================= */

#define P36_INK_MAX_PHYS 32
#define P36_INK_MAX_LOGICAL 64
#define P36_INK_MAX_BLOCKS 8

typedef struct {
    uint32_t n_phys_layer;
    uint32_t n_logical_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    uint32_t n_vocab_padded;
    uint32_t n_vocab_unpadded;
    uint32_t n_swa;
    uint32_t n_relative;
    uint32_t n_relative_extent;
    uint32_t n_conv;
    uint32_t attn_res_block_size;
    uint32_t loop_start;
    uint32_t loop_end;
    uint32_t loop_count;
    uint32_t loop_gate_rank;
    float rms_eps;
    float qk_clip;
    float rel_clip;
    float log_scale_floor;
    float log_scale_alpha;
    float situ_beta;
    float situ_linear_beta;
    float expert_weights_scale;
    float final_logit_softcapping;
    bool is_swa[P36_INK_MAX_PHYS];
} p36_inkling2_hparams;

typedef struct {
    p36_tensor *attn_norm;
    p36_tensor *attn_q_inp_norm;
    p36_tensor *attn_k_inp_norm;
    p36_tensor *attn_v_inp_norm;
    p36_tensor *attn_k_conv_proj;
    p36_tensor *attn_v_conv_proj;
    p36_tensor *attn_q;
    p36_tensor *attn_k;
    p36_tensor *attn_v;
    p36_tensor *attn_output;
    p36_tensor *attn_gate;
    p36_tensor *attn_q_clip;
    p36_tensor *attn_sink_proj;
    p36_tensor *attn_rel;
    p36_tensor *attn_rel_bias;
    p36_tensor *attn_rel_clip;
    p36_tensor *attn_k_conv;
    p36_tensor *attn_v_conv;
    p36_tensor *attn_post_conv;
    p36_tensor *ffn_post_conv;
    p36_tensor *attn_res_query;
    p36_tensor *ffn_res_query;
    p36_tensor *ffn_norm;
    p36_tensor *ffn_gate_inp;
    p36_tensor *exp_probs_b;
    p36_tensor *ffn_gate_up_exps;
    p36_tensor *ffn_gate_up_exps_scale;
    p36_tensor *ffn_down_exps;
    p36_tensor *ffn_down_exps_scale;
    p36_tensor *loop_msg_query;
    p36_tensor *loop_down[4];
    p36_tensor *loop_forget[4];
    p36_tensor *loop_gain[4];
} p36_inkling2_layer_weights;

typedef struct {
    p36_tensor *token_embd;
    p36_tensor *token_embd_norm;
    p36_tensor *output;
    p36_tensor *output_norm;
    p36_tensor *output_res_query;
    p36_tensor *loop_res_query;
    p36_tensor *loop_kv_res_query;
    p36_inkling2_layer_weights layer[P36_INK_MAX_PHYS];
} p36_inkling2_weights;

static bool p36_inkling2_init_engine(p36_engine *e) {
    const p36_model *m = &e->model;
    p36_inkling2_hparams *hp = xcalloc(1, sizeof(*hp));
    p36_inkling2_weights *w = xcalloc(1, sizeof(*w));
    e->ink_hparams = hp;
    e->ink_weights = w;

    p36_str arch = {0};
    model_get_string(m, "general.architecture", &arch);
    bool is_inkling2_arch = p36_streq(arch, "inkling2");

#define INK_GET_U32(suf, dst) do { \
    if (!model_get_u32(m, "inkling2_loop." suf, &(dst))) \
        model_get_u32(m, "inkling2." suf, &(dst)); \
} while (0)
#define INK_GET_F32(suf, dst) do { \
    if (!model_get_f32_compat(m, "inkling2_loop." suf, &(dst))) \
        model_get_f32_compat(m, "inkling2." suf, &(dst)); \
} while (0)

    uint32_t block_count = is_inkling2_arch ? 48u : 24u;
    INK_GET_U32("block_count", block_count);
    hp->n_embd = 2304;
    INK_GET_U32("embedding_length", hp->n_embd);
    hp->n_head = 24;
    hp->n_head_kv = 8;
    hp->n_head_dim = 128;
    INK_GET_U32("attention.key_length", hp->n_head_dim);
    hp->n_expert = 128;
    INK_GET_U32("expert_count", hp->n_expert);
    hp->n_expert_used = 8;
    INK_GET_U32("expert_used_count", hp->n_expert_used);
    hp->n_ff_exp = 1152;
    INK_GET_U32("expert_feed_forward_length", hp->n_ff_exp);
    hp->n_vocab_padded = 201024;
    INK_GET_U32("vocab_size", hp->n_vocab_padded);
    hp->n_vocab_unpadded = 200058;
    INK_GET_U32("vocab_size_unpadded", hp->n_vocab_unpadded);
    hp->n_swa = 1024;
    INK_GET_U32("attention.sliding_window", hp->n_swa);
    hp->n_relative = 16;
    INK_GET_U32("attention.relative_length", hp->n_relative);
    hp->n_relative_extent = 1024;
    INK_GET_U32("attention.relative_extent", hp->n_relative_extent);
    hp->n_conv = 4;
    INK_GET_U32("shortconv.l_cache", hp->n_conv);
    hp->attn_res_block_size = 6;
    INK_GET_U32("attn_res.block_size", hp->attn_res_block_size);
    hp->loop_start = 6;
    if (!model_get_u32(m, "inkling2_loop.loop.start", &hp->loop_start))
        model_get_u32(m, "inkling2.loop.core_start", &hp->loop_start);
    hp->loop_end = 18;
    if (!model_get_u32(m, "inkling2_loop.loop.end", &hp->loop_end))
        model_get_u32(m, "inkling2.loop.core_end", &hp->loop_end);
    hp->loop_count = 3;
    if (!model_get_u32(m, "inkling2_loop.loop.count", &hp->loop_count))
        model_get_u32(m, "inkling2.num_loops", &hp->loop_count);
    hp->loop_gate_rank = 64;
    INK_GET_U32("loop.gate_rank", hp->loop_gate_rank);

    hp->rms_eps = 1.0e-6f;
    INK_GET_F32("attention.layer_norm_rms_epsilon", hp->rms_eps);
    hp->qk_clip = 22.6274166f;
    INK_GET_F32("attention.qk_clip", hp->qk_clip);
    hp->rel_clip = 8.0f;
    INK_GET_F32("attention.relative_clip", hp->rel_clip);
    hp->log_scale_floor = 8192.0f;
    INK_GET_F32("attention.log_scale_floor", hp->log_scale_floor);
    hp->log_scale_alpha = 0.1f;
    INK_GET_F32("attention.log_scale_alpha", hp->log_scale_alpha);
    hp->situ_beta = 4.0f;
    INK_GET_F32("activation.situ_beta", hp->situ_beta);
    hp->situ_linear_beta = 25.0f;
    INK_GET_F32("activation.situ_linear_beta", hp->situ_linear_beta);
    hp->expert_weights_scale = 8.0f;
    INK_GET_F32("expert_weights_scale", hp->expert_weights_scale);
    hp->final_logit_softcapping = 30.0f;
    INK_GET_F32("final_logit_softcapping", hp->final_logit_softcapping);

#undef INK_GET_U32
#undef INK_GET_F32

    uint32_t n_core = hp->loop_count > 1 ? (hp->loop_end - hp->loop_start) : 0;
    uint32_t extra_core = n_core * (hp->loop_count > 1 ? (hp->loop_count - 1u) : 0u);
    if (is_inkling2_arch) {
        hp->n_logical_layer = block_count;
        hp->n_phys_layer = block_count - extra_core;
    } else {
        hp->n_phys_layer = block_count;
        hp->n_logical_layer = block_count + extra_core;
    }

    /* Read sliding_window_pattern array (or fallback to every 6th full-attn). */
    p36_array_ref pat;
    bool got_pat = model_get_array(m, "inkling2_loop.attention.sliding_window_pattern", &pat) ||
                   model_get_array(m, "inkling2.attention.sliding_window_pattern", &pat);
    if (got_pat && pat.len == hp->n_phys_layer && pat.type == GGUF_VALUE_BOOL) {
        p36_cursor c = cursor_at(m, pat.data_pos);
        for (uint32_t i = 0; i < hp->n_phys_layer; i++) {
            uint8_t b = 1;
            if (cursor_read(&c, &b, 1)) hp->is_swa[i] = (b != 0);
        }
    } else if (got_pat && pat.len == hp->n_logical_layer && pat.type == GGUF_VALUE_BOOL) {
        p36_cursor c = cursor_at(m, pat.data_pos);
        for (uint32_t il = 0; il < hp->n_logical_layer; il++) {
            uint8_t b = 1;
            if (!cursor_read(&c, &b, 1)) break;
            uint32_t ip;
            if (n_core > 0 && il >= hp->loop_start && il < hp->loop_start + n_core * hp->loop_count) {
                ip = hp->loop_start + ((il - hp->loop_start) % n_core);
            } else if (n_core > 0 && il >= hp->loop_start + n_core * hp->loop_count) {
                ip = hp->loop_end + (il - (hp->loop_start + n_core * hp->loop_count));
            } else {
                ip = il;
            }
            if (ip < P36_INK_MAX_PHYS) hp->is_swa[ip] = (b != 0);
        }
    } else {
        for (uint32_t i = 0; i < hp->n_phys_layer; i++) {
            hp->is_swa[i] = ((i % 6u) != 5u);
        }
    }

    w->token_embd = required_tensor(m, "token_embd.weight");
    w->token_embd_norm = required_tensor(m, "token_embd_norm.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output_res_query = required_tensor(m, "output_res_query.weight");
    w->loop_res_query = model_find_tensor(m, "loop_res_query.weight");
    w->loop_kv_res_query = model_find_tensor(m, "loop_kv_res_query.weight");

    /* Populate generic e->weights fields for helper compatibility. */
    e->weights.token_embd = w->token_embd;
    e->weights.output = w->output;
    e->weights.output_norm = w->output_norm;
    e->expert_weights_scale = hp->expert_weights_scale;

    for (uint32_t ip = 0; ip < hp->n_phys_layer; ip++) {
        p36_inkling2_layer_weights *l = &w->layer[ip];
        l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", ip);
        l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", ip);
        l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", ip);
        l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", ip);
        l->attn_gate = required_tensorf(m, "blk.%u.attn_gate.weight", ip);
        l->attn_q_clip = required_tensorf(m, "blk.%u.attn_q_clip.weight", ip);
        l->attn_sink_proj = required_tensorf(m, "blk.%u.attn_sink_proj.weight", ip);
        l->attn_rel = required_tensorf(m, "blk.%u.attn_rel.weight", ip);
        l->attn_rel_bias = required_tensorf(m, "blk.%u.attn_rel_bias.weight", ip);
        l->attn_rel_clip = required_tensorf(m, "blk.%u.attn_rel_clip.weight", ip);
        l->attn_k_conv = required_tensorf(m, "blk.%u.attn_k_conv.weight", ip);
        l->attn_v_conv = required_tensorf(m, "blk.%u.attn_v_conv.weight", ip);
        l->attn_post_conv = required_tensorf(m, "blk.%u.attn_post_conv.weight", ip);
        l->ffn_post_conv = required_tensorf(m, "blk.%u.ffn_post_conv.weight", ip);
        if (hp->is_swa[ip]) {
            l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", ip);
        } else {
            l->attn_q_inp_norm = required_tensorf(m, "blk.%u.attn_q_inp_norm.weight", ip);
            l->attn_k_inp_norm = required_tensorf(m, "blk.%u.attn_k_inp_norm.weight", ip);
            l->attn_v_inp_norm = required_tensorf(m, "blk.%u.attn_v_inp_norm.weight", ip);
            l->attn_k_conv_proj = required_tensorf(m, "blk.%u.attn_k_conv_proj.weight", ip);
            l->attn_v_conv_proj = required_tensorf(m, "blk.%u.attn_v_conv_proj.weight", ip);
        }
        if (ip > 0) {
            l->attn_res_query = required_tensorf(m, "blk.%u.attn_res_query.weight", ip);
        }
        l->ffn_res_query = required_tensorf(m, "blk.%u.ffn_res_query.weight", ip);
        l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", ip);
        l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", ip);
        l->exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", ip);
        l->ffn_gate_up_exps = required_tensorf(m, "blk.%u.ffn_gate_up_exps.weight", ip);
        l->ffn_gate_up_exps_scale = tensor_by_namef(m, "blk.%u.ffn_gate_up_exps.scale", ip);
        l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", ip);
        l->ffn_down_exps_scale = tensor_by_namef(m, "blk.%u.ffn_down_exps.scale", ip);

        if (hp->loop_count > 1 && ip >= hp->loop_start && ip < hp->loop_end) {
            l->loop_msg_query = tensor_by_namef(m, "blk.%u.loop_msg_query.weight", ip);
            static const char *const m_apex[4] = { "q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl" };
            static const char *const m_pr[4]   = { "attn",   "k",      "v",      "mlp" };
            for (int k = 0; k < 4; k++) {
                char nm[96];
                snprintf(nm, sizeof(nm), "blk.%u.loop_%s_down.weight", ip, m_apex[k]);
                l->loop_down[k] = model_find_tensor(m, nm);
                if (!l->loop_down[k]) {
                    snprintf(nm, sizeof(nm), "blk.%u.loop_%s_down.weight", ip, m_pr[k]);
                    l->loop_down[k] = required_tensor(m, nm);
                }
                snprintf(nm, sizeof(nm), "blk.%u.loop_%s_forget.weight", ip, m_apex[k]);
                l->loop_forget[k] = model_find_tensor(m, nm);
                if (!l->loop_forget[k]) {
                    snprintf(nm, sizeof(nm), "blk.%u.loop_%s_forget.weight", ip, m_pr[k]);
                    l->loop_forget[k] = required_tensor(m, nm);
                }
                snprintf(nm, sizeof(nm), "blk.%u.loop_%s_gain.weight", ip, m_apex[k]);
                l->loop_gain[k] = model_find_tensor(m, nm);
                if (!l->loop_gain[k]) {
                    snprintf(nm, sizeof(nm), "blk.%u.loop_%s_gain.weight", ip, m_pr[k]);
                    l->loop_gain[k] = required_tensor(m, nm);
                }
            }
        }
    }
    e->routed_quant_bits = p36_quant_bits_from_type(w->layer[0].ffn_gate_up_exps->type);
    return true;
}

#ifndef P36_NO_GPU
typedef struct {
    uint32_t prefill_cap;
    uint32_t full_kv_cap;
    uint32_t swa_kv_cap;
    p36_gpu_tensor *kv_k[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *kv_v[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_k[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_v[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_attn_post[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_ffn_post[P36_INK_MAX_LOGICAL];
    bool is_swa_logical[P36_INK_MAX_LOGICAL];
    uint32_t n_logical;

    p36_gpu_tensor *cur;
    p36_gpu_tensor *inp_sa;
    p36_gpu_tensor *ffn_inp;
    p36_gpu_tensor *prelude_cur;
    p36_gpu_tensor *blocks_normed;     /* [4, prefill_cap, n_embd] */
    p36_gpu_tensor *bank_normed;       /* [4, prefill_cap, n_embd] */
    p36_gpu_tensor *msg_buf;           /* [4, prefill_cap, n_embd] */
    p36_gpu_tensor *msg_view[4];
    p36_gpu_tensor *msg_view_12;       /* [2, prefill_cap, n_embd] view of slices 1..2 */
    p36_gpu_tensor *attn_in_buf;       /* [3, prefill_cap, n_embd] */
    p36_gpu_tensor *attn_in_view[3];
    p36_gpu_tensor *ctrl_in;           /* [prefill_cap, 2 * n_embd] */
    p36_gpu_tensor *ctrl_h_buf;        /* [3, prefill_cap, 64] contiguous */
    p36_gpu_tensor *ctrl_h[3];         /* 3x [prefill_cap, 64] views for parallel Q/K/V */
    p36_gpu_tensor *ctrl_out_buf;      /* [4, prefill_cap, n_embd] contiguous */
    p36_gpu_tensor *ctrl_out[4];       /* [prefill_cap, n_embd] views */
    p36_gpu_tensor *norm_buf;          /* [3, prefill_cap, n_embd] contiguous */
    p36_gpu_tensor *norm[3];           /* [prefill_cap, n_embd] views */
    p36_gpu_tensor *q8_buf;            /* [3, prefill_cap, q8_k(n_embd)] contiguous */
    p36_gpu_tensor *q8[3];             /* [prefill_cap, q8_k(n_embd)] views */
    p36_gpu_tensor *q_buf;             /* [prefill_cap, 3072] */
    p36_gpu_tensor *k_buf;             /* [prefill_cap, 1024] */
    p36_gpu_tensor *v_buf;             /* [prefill_cap, 1024] */
    p36_gpu_tensor *r_buf;             /* [prefill_cap, 384] */
    p36_gpu_tensor *sinks_buf;         /* [prefill_cap, 24] */
    p36_gpu_tensor *gate_buf;          /* [prefill_cap, 3072] */
    p36_gpu_tensor *k_dyn;             /* [prefill_cap, 32] */
    p36_gpu_tensor *v_dyn;             /* [prefill_cap, 32] */
    p36_gpu_tensor *attn_out;          /* [prefill_cap, 3072] */
    p36_gpu_tensor *wo_out;            /* [prefill_cap, 2304] */
    p36_gpu_tensor *router_logits;     /* [prefill_cap, 128] */
    p36_gpu_tensor *selected;          /* [prefill_cap, 8] u32 */
    p36_gpu_tensor *router_w;          /* [prefill_cap, 8] f32 */
    p36_gpu_tensor *moe_mid;           /* [prefill_cap * 8, 1152] */
    p36_gpu_tensor *moe_down;          /* [prefill_cap * 8, 2304] */
    p36_gpu_tensor *moe_out;           /* [prefill_cap, 2304] */
} p36_inkling2_runtime;

static bool p36_inkling2_vulkan_runtime_reset(p36_vulkan_runtime *rt) {
    p36_inkling2_runtime *ink = (p36_inkling2_runtime *)rt->ink;
    if (!ink) return false;
    rt->inp_q8_valid = false;
    for (uint32_t il = 0; il < ink->n_logical; il++) {
        if (!p36_gpu_tensor_zero(ink->conv_k[il]) ||
            !p36_gpu_tensor_zero(ink->conv_v[il]) ||
            !p36_gpu_tensor_zero(ink->conv_attn_post[il]) ||
            !p36_gpu_tensor_zero(ink->conv_ffn_post[il])) {
            return false;
        }
    }
    return true;
}

static void p36_inkling2_vulkan_runtime_free(p36_vulkan_runtime *rt) {
    p36_inkling2_runtime *ink = (p36_inkling2_runtime *)rt->ink;
    if (ink) {
        for (uint32_t il = 0; il < ink->n_logical; il++) {
            p36_gpu_tensor_free(ink->kv_k[il]);
            p36_gpu_tensor_free(ink->kv_v[il]);
            p36_gpu_tensor_free(ink->conv_k[il]);
            p36_gpu_tensor_free(ink->conv_v[il]);
            p36_gpu_tensor_free(ink->conv_attn_post[il]);
            p36_gpu_tensor_free(ink->conv_ffn_post[il]);
        }
        for (int i = 0; i < 4; i++) {
            p36_gpu_tensor_free(ink->msg_view[i]);
            p36_gpu_tensor_free(ink->ctrl_out[i]);
        }
        p36_gpu_tensor_free(ink->msg_view_12);
        for (int i = 0; i < 3; i++) {
            p36_gpu_tensor_free(ink->attn_in_view[i]);
            p36_gpu_tensor_free(ink->ctrl_h[i]);
            p36_gpu_tensor_free(ink->norm[i]);
            p36_gpu_tensor_free(ink->q8[i]);
        }
        p36_gpu_tensor_free(ink->cur);
        p36_gpu_tensor_free(ink->inp_sa);
        p36_gpu_tensor_free(ink->ffn_inp);
        p36_gpu_tensor_free(ink->prelude_cur);
        p36_gpu_tensor_free(ink->blocks_normed);
        p36_gpu_tensor_free(ink->bank_normed);
        p36_gpu_tensor_free(ink->msg_buf);
        p36_gpu_tensor_free(ink->attn_in_buf);
        p36_gpu_tensor_free(ink->ctrl_in);
        p36_gpu_tensor_free(ink->ctrl_h_buf);
        p36_gpu_tensor_free(ink->ctrl_out_buf);
        p36_gpu_tensor_free(ink->norm_buf);
        p36_gpu_tensor_free(ink->q8_buf);
        p36_gpu_tensor_free(ink->q_buf);
        p36_gpu_tensor_free(ink->k_buf);
        p36_gpu_tensor_free(ink->v_buf);
        p36_gpu_tensor_free(ink->r_buf);
        p36_gpu_tensor_free(ink->sinks_buf);
        p36_gpu_tensor_free(ink->gate_buf);
        p36_gpu_tensor_free(ink->k_dyn);
        p36_gpu_tensor_free(ink->v_dyn);
        p36_gpu_tensor_free(ink->attn_out);
        p36_gpu_tensor_free(ink->wo_out);
        p36_gpu_tensor_free(ink->router_logits);
        p36_gpu_tensor_free(ink->selected);
        p36_gpu_tensor_free(ink->router_w);
        p36_gpu_tensor_free(ink->moe_mid);
        p36_gpu_tensor_free(ink->moe_down);
        p36_gpu_tensor_free(ink->moe_out);
        free(ink);
        rt->ink = NULL;
    }
    p36_gpu_tensor_free(rt->hidden);
    p36_gpu_tensor_free(rt->embed_stage[0]);
    p36_gpu_tensor_free(rt->embed_stage[1]);
    p36_gpu_tensor_free(rt->norm);
    p36_gpu_tensor_free(rt->last_h);
    p36_gpu_tensor_free(rt->logits);
    p36_gpu_tensor_free(rt->top2[0]);
    p36_gpu_tensor_free(rt->top2[1]);
    p36_gpu_tensor_free(rt->topk8);
    p36_gpu_tensor_free(rt->minp_pack);
    free(rt);
}

static p36_vulkan_runtime *p36_inkling2_vulkan_runtime_create(int ctx_size, uint32_t prefill_cap) {
    p36_vulkan_runtime *rt = xcalloc(1, sizeof(*rt));
    p36_inkling2_runtime *ink = xcalloc(1, sizeof(*ink));
    rt->ink = ink;
    if (prefill_cap < 1u) prefill_cap = 1u;
    if (prefill_cap > 1024u) prefill_cap = 1024u;
    rt->prefill_cap = prefill_cap;
    ink->prefill_cap = prefill_cap;
    ink->full_kv_cap = (uint32_t)ctx_size;
    ink->swa_kv_cap = 2048u;
    ink->n_logical = 48u;

    const uint32_t D = 2304u;
    const uint32_t Q = 3072u;
    const uint32_t KV = 1024u;
    const uint64_t tok_embd_bytes = (uint64_t)prefill_cap * D * sizeof(float);
    const uint64_t q8_bytes = (uint64_t)prefill_cap * (D / 256u) * P36_VK_Q8_K_BYTES;

    /* Common rt buffers expected by p36 session/sampling */
    rt->hidden = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    rt->embed_stage[0] = p36_gpu_tensor_alloc(tok_embd_bytes);
    rt->embed_stage[1] = p36_gpu_tensor_alloc(tok_embd_bytes);
    rt->norm = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    rt->last_h = p36_gpu_tensor_alloc_scratch((uint64_t)D * sizeof(float));
    rt->logits = p36_gpu_tensor_alloc((uint64_t)201024u * sizeof(float));
    rt->top2[0] = p36_gpu_tensor_alloc(2u * sizeof(uint32_t));
    rt->top2[1] = p36_gpu_tensor_alloc(2u * sizeof(uint32_t));
    rt->topk8 = p36_gpu_tensor_alloc(16u * sizeof(float));
    rt->minp_pack = p36_gpu_tensor_alloc(((uint64_t)P36_N_VOCAB * 2u + 4u) * sizeof(uint32_t));
    if (!rt->hidden || !rt->embed_stage[0] || !rt->embed_stage[1] || !rt->norm || !rt->last_h ||
        !rt->logits || !rt->top2[0] || !rt->top2[1] || !rt->topk8 || !rt->minp_pack) {
        goto fail;
    }

    /* Inkling2 activations & scratch */
    ink->cur = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    ink->inp_sa = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    ink->ffn_inp = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    ink->prelude_cur = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    ink->blocks_normed = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
    ink->bank_normed = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
    ink->msg_buf = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
    ink->attn_in_buf = p36_gpu_tensor_alloc_scratch(3u * tok_embd_bytes);
    ink->ctrl_in = p36_gpu_tensor_alloc_scratch(2u * tok_embd_bytes);
    ink->q_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * Q * sizeof(float));
    ink->k_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * KV * sizeof(float));
    ink->v_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * KV * sizeof(float));
    ink->r_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 384u * sizeof(float));
    ink->sinks_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 24u * sizeof(float));
    ink->gate_buf = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * Q * sizeof(float));
    ink->k_dyn = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 32u * sizeof(float));
    ink->v_dyn = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 32u * sizeof(float));
    ink->attn_out = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * Q * sizeof(float));
    ink->wo_out = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    ink->router_logits = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 128u * sizeof(float));
    ink->selected = p36_gpu_tensor_alloc((uint64_t)prefill_cap * 8u * sizeof(uint32_t));
    ink->router_w = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 8u * sizeof(float));
    ink->moe_mid = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 8u * 1152u * sizeof(float));
    ink->moe_down = p36_gpu_tensor_alloc_scratch((uint64_t)prefill_cap * 8u * D * sizeof(float));
    ink->moe_out = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);

    if (!ink->cur || !ink->inp_sa || !ink->ffn_inp || !ink->prelude_cur ||
        !ink->blocks_normed || !ink->bank_normed || !ink->msg_buf || !ink->attn_in_buf ||
        !ink->ctrl_in || !ink->q_buf || !ink->k_buf || !ink->v_buf ||
        !ink->r_buf || !ink->sinks_buf || !ink->gate_buf || !ink->k_dyn || !ink->v_dyn ||
        !ink->attn_out || !ink->wo_out || !ink->router_logits || !ink->selected ||
        !ink->router_w || !ink->moe_mid || !ink->moe_down || !ink->moe_out) {
        goto fail;
    }

    uint64_t h_slice_bytes = (uint64_t)prefill_cap * 64u * sizeof(float);
    ink->ctrl_out_buf = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
    ink->ctrl_h_buf = p36_gpu_tensor_alloc_scratch(3u * h_slice_bytes);
    ink->norm_buf = p36_gpu_tensor_alloc_scratch(3u * tok_embd_bytes);
    ink->q8_buf = p36_gpu_tensor_alloc_scratch(3u * q8_bytes);
    if (!ink->ctrl_out_buf || !ink->ctrl_h_buf || !ink->norm_buf || !ink->q8_buf) goto fail;

    for (int i = 0; i < 4; i++) {
        ink->msg_view[i] = p36_gpu_tensor_view(ink->msg_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        ink->ctrl_out[i] = p36_gpu_tensor_view(ink->ctrl_out_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        if (!ink->msg_view[i] || !ink->ctrl_out[i]) goto fail;
    }
    ink->msg_view_12 = p36_gpu_tensor_view(ink->msg_buf, tok_embd_bytes, 2u * tok_embd_bytes);
    if (!ink->msg_view_12) goto fail;
    for (int i = 0; i < 3; i++) {
        ink->attn_in_view[i] = p36_gpu_tensor_view(ink->attn_in_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        ink->ctrl_h[i] = p36_gpu_tensor_view(ink->ctrl_h_buf, (uint64_t)i * h_slice_bytes, h_slice_bytes);
        ink->norm[i] = p36_gpu_tensor_view(ink->norm_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        ink->q8[i] = p36_gpu_tensor_view(ink->q8_buf, (uint64_t)i * q8_bytes, q8_bytes);
        if (!ink->attn_in_view[i] || !ink->ctrl_h[i] || !ink->norm[i] || !ink->q8[i]) goto fail;
    }

    /* Per-logical-layer KV cache (F16) and shortconv state (F32, 3 history slots). */
    for (uint32_t il = 0; il < ink->n_logical; il++) {
        uint32_t ip;
        if (il < 6u) ip = il;
        else if (il < 42u) ip = 6u + ((il - 6u) % 12u);
        else ip = 18u + (il - 42u);
        bool is_swa = ((ip % 6u) != 5u);
        ink->is_swa_logical[il] = is_swa;
        uint32_t cap = is_swa ? ink->swa_kv_cap : ink->full_kv_cap;
        uint64_t kv_bytes = (uint64_t)cap * KV * sizeof(uint16_t);
        ink->kv_k[il] = p36_gpu_tensor_alloc_uninitialized(kv_bytes);
        ink->kv_v[il] = p36_gpu_tensor_alloc_uninitialized(kv_bytes);
        ink->conv_k[il] = p36_gpu_tensor_alloc(3u * KV * sizeof(float));
        ink->conv_v[il] = p36_gpu_tensor_alloc(3u * KV * sizeof(float));
        ink->conv_attn_post[il] = p36_gpu_tensor_alloc(3u * D * sizeof(float));
        ink->conv_ffn_post[il] = p36_gpu_tensor_alloc(3u * D * sizeof(float));
        if (!ink->kv_k[il] || !ink->kv_v[il] || !ink->conv_k[il] || !ink->conv_v[il] ||
            !ink->conv_attn_post[il] || !ink->conv_ffn_post[il]) {
            goto fail;
        }
    }

    if (!p36_inkling2_vulkan_runtime_reset(rt)) {
        fprintf(stderr, "p36: inkling2 runtime reset failed\n");
        goto fail;
    }
    return rt;

fail:
    fprintf(stderr, "p36: inkling2 runtime alloc failed (ctx=%d prefill=%u)\n", ctx_size, prefill_cap);
    p36_inkling2_vulkan_runtime_free(rt);
    return NULL;
}

static bool p36_inkling2_control_vk(const p36_model *m,
                                    const p36_inkling2_layer_weights *l,
                                    const p36_inkling2_hparams *hp,
                                    p36_inkling2_runtime *ink,
                                    p36_gpu_tensor *out,
                                    const p36_gpu_tensor *cur,
                                    const p36_gpu_tensor *msg,
                                    int mod,
                                    uint32_t n_tok) {
    bool down_ok;
    if (n_tok == 1u) {
        down_ok = p36_gpu_inkling2_control_prep_down_tensor(ink->ctrl_h[0], m->map, m->size,
                                                            l->loop_down[mod]->abs_offset,
                                                            cur, msg, hp->n_embd, hp->loop_gate_rank) != 0;
    } else {
        down_ok = p36_gpu_inkling2_control_prep_tensor(ink->ctrl_in, cur, msg, n_tok, hp->n_embd) &&
                  ((n_tok >= 16u && l->loop_down[mod]->type == P36_TENSOR_F32 && (hp->loop_gate_rank % 64u) == 0u)
                       ? (p36_gpu_inkling2_f32_gemm_tensor(ink->ctrl_h[0], m->map, m->size,
                                                           l->loop_down[mod]->abs_offset,
                                                           ink->ctrl_in, 2u * hp->n_embd, hp->loop_gate_rank, n_tok) != 0)
                       : p36_gpu_tensor_matmul_scaled(m, l->loop_down[mod], ink->ctrl_in, ink->ctrl_h[0],
                                                      2u * hp->n_embd, hp->loop_gate_rank, n_tok, 1.0f));
    }
    return down_ok &&
           p36_gpu_inkling2_control_gate_tensor(out, m->map, m->size,
                                                l->loop_forget[mod]->abs_offset,
                                                l->loop_gain[mod]->abs_offset,
                                                ink->ctrl_h[0], cur, msg,
                                                n_tok, hp->n_embd, hp->loop_gate_rank);
}

static bool p36_forward_tokens_inkling2_vulkan(p36_session *s,
                                               const int *tokens,
                                               const float *embeddings,
                                               uint32_t n_tok,
                                               uint32_t pos0,
                                               p36_gpu_tensor *logits_out,
                                               bool logits_all_rows) {
    p36_engine *e = s->engine;
    p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
    p36_inkling2_runtime *ink = (p36_inkling2_runtime *)rt->ink;
    const p36_inkling2_hparams *hp = (const p36_inkling2_hparams *)e->ink_hparams;
    const p36_inkling2_weights *w = (const p36_inkling2_weights *)e->ink_weights;
    const p36_model *m = &e->model;

    if (!ink || !hp || !w || n_tok == 0 || n_tok > ink->prefill_cap) {
        fprintf(stderr, "p36: inkling2 fwd bad args n_tok=%u cap=%u\n", n_tok, ink ? ink->prefill_cap : 0u);
        return false;
    }
#define INK_CHK(expr, tag) do { if (!(expr)) { fprintf(stderr, "p36: inkling2 fail at %s (il=%u ip=%u n_tok=%u)\n", (tag), il, ip, n_tok); return false; } } while (0)

    const uint32_t D = hp->n_embd;
    const uint32_t Q = hp->n_head * hp->n_head_dim;
    const uint32_t KV = hp->n_head_kv * hp->n_head_dim;
    const uint32_t state_stride = ink->prefill_cap * D;
    const float inv_sqrt_5 = 1.0f / sqrtf(5.0f);
    const float inv_sqrt_9 = 1.0f / 3.0f;
    const float attn_scale = 1.0f / sqrtf((float)hp->n_head_dim);

    /* 1. Embed + token_embd_norm */
    {
        uint32_t il = 0, ip = 0;
        p36_gpu_tensor *stage = rt->embed_stage[rt->embed_flip & 1u];
        rt->embed_flip ^= 1u;
        if (embeddings) {
            /* Raw vision embeddings bypass token_embd_norm (matches llama-inkling2). */
            INK_CHK(p36_gpu_tensor_write(stage, 0, embeddings, (uint64_t)n_tok * D * sizeof(float)), "embed_write");
            INK_CHK(p36_gpu_copy_f32_tensor(ink->cur, stage, n_tok * D), "embed_copy_raw");
        } else {
            INK_CHK(p36_gpu_embed_tokens(m, w->token_embd, tokens, n_tok, stage), "embed_tokens");
            INK_CHK(p36_gpu_copy_f32_tensor(rt->hidden, stage, n_tok * D), "embed_copy");
            INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(ink->cur, NULL, rt->hidden, m->map, m->size,
                                                         w->token_embd_norm->abs_offset, D, n_tok, hp->rms_eps), "tok_norm");
        }
    }

    uint32_t n_blocks = 0;
    uint32_t prelude_n_blocks = 0;
    uint32_t bank_n = 0;
    const uint32_t n_core = hp->loop_count > 1 ? (hp->loop_end - hp->loop_start) : 0;

    for (uint32_t il = 0; il < hp->n_logical_layer; il++) {
        uint32_t ip;
        int pass = -1;
        if (n_core > 0 && il >= hp->loop_start && il < hp->loop_start + n_core * hp->loop_count) {
            pass = (int)((il - hp->loop_start) / n_core);
            ip = hp->loop_start + ((il - hp->loop_start) % n_core);
        } else if (n_core > 0 && il >= hp->loop_start + n_core * hp->loop_count) {
            ip = hp->loop_end + (il - (hp->loop_start + n_core * hp->loop_count));
        } else {
            ip = il;
        }

        const p36_inkling2_layer_weights *l = &w->layer[ip];
        const bool is_swa = hp->is_swa[ip];
        const uint32_t kv_cap = is_swa ? ink->swa_kv_cap : ink->full_kv_cap;

        /* Loop boundary handling */
        if (n_core > 0 && ip == hp->loop_start && pass >= 0) {
            if (pass == 0) {
                prelude_n_blocks = n_blocks;
                INK_CHK(p36_gpu_copy_f32_tensor(ink->prelude_cur, ink->cur, n_tok * D), "prelude_copy");
            } else {
                bank_n = 0;
                for (uint32_t bi = prelude_n_blocks + 1u; bi < n_blocks; bi++) {
                    uint64_t src_off = (uint64_t)bi * state_stride * sizeof(float);
                    uint64_t dst_off = (uint64_t)bank_n * state_stride * sizeof(float);
                    INK_CHK(p36_gpu_tensor_copy(ink->bank_normed, dst_off, ink->blocks_normed, src_off,
                                                (uint64_t)n_tok * D * sizeof(float)), "bank_copy");
                    bank_n++;
                }
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->bank_normed, ink->cur, m->map, m->size,
                                                    0, 0, NULL, n_tok, D, 0, 1, 0, 0,
                                                    state_stride, state_stride, bank_n), "bank_push");
                bank_n++;
                n_blocks = prelude_n_blocks;
                INK_CHK(p36_gpu_copy_f32_tensor(ink->cur, ink->prelude_cur, n_tok * D), "prelude_restore");
            }
        }

        /* Compute loop messages for pass > 0 */
        if (pass > 0) {
            if (l->loop_msg_query) {
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->bank_normed, ink->cur, m->map, m->size,
                                                    l->loop_msg_query->abs_offset, l->loop_msg_query->bytes,
                                                    ink->msg_buf, n_tok, D, bank_n, 0, 4, 0,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg");
            } else {
                uint32_t row = 2u * (ip - hp->loop_start);
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->bank_normed, ink->cur, m->map, m->size,
                                                    w->loop_res_query->abs_offset, w->loop_res_query->bytes,
                                                    ink->msg_view[0], n_tok, D, bank_n, 0, 1, row,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg0");
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->bank_normed, ink->cur, m->map, m->size,
                                                    w->loop_res_query->abs_offset, w->loop_res_query->bytes,
                                                    ink->msg_view[3], n_tok, D, bank_n, 0, 1, row + 1u,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg3");
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->bank_normed, ink->cur, m->map, m->size,
                                                    w->loop_kv_res_query->abs_offset, w->loop_kv_res_query->bytes,
                                                    ink->msg_view_12, n_tok, D, bank_n, 0, 2, row,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg12");
            }
        }

        /* Zero-copy inpSA: ink->cur is untouched until FFN mix overwrites ink->cur */
        p36_gpu_tensor *inp_sa_t = ink->cur;

        /* Depth-residual mix + loop control + RMSNorm + Q8_K for attention inputs */
        p36_gpu_tensor *n_in[3];
        p36_gpu_tensor *q8_in[3];
        if (is_swa && pass <= 0 && l->attn_res_query) {
            /* Fused mix + RMSNorm + Q8_K (1 dispatch instead of 3!) */
            INK_CHK(p36_gpu_inkling2_mix_norm_tensor(ink->blocks_normed, inp_sa_t, m->map, m->size,
                                                     l->attn_res_query->abs_offset, l->attn_res_query->bytes,
                                                     l->attn_norm->abs_offset,
                                                     ink->norm[0], ink->q8[0], n_tok, D, n_blocks, 1,
                                                     state_stride, hp->rms_eps), "attn_mix_norm_q8");
            n_in[0] = n_in[1] = n_in[2] = ink->norm[0];
            q8_in[0] = q8_in[1] = q8_in[2] = ink->q8[0];
        } else {
            p36_gpu_tensor *inp[3] = { inp_sa_t, inp_sa_t, inp_sa_t };
            if (l->attn_res_query) {
                uint32_t n_q = is_swa ? 1u : 3u;
                INK_CHK(p36_gpu_inkling2_mix_tensor(ink->blocks_normed, inp_sa_t, m->map, m->size,
                                                    l->attn_res_query->abs_offset, l->attn_res_query->bytes,
                                                    ink->attn_in_buf, n_tok, D, n_blocks, 1, n_q, 0,
                                                    state_stride, state_stride, 0xffffffffu), "attn_mix");
                if (is_swa) {
                    inp[0] = inp[1] = inp[2] = ink->attn_in_view[0];
                } else {
                    inp[0] = ink->attn_in_view[0];
                    inp[1] = ink->attn_in_view[1];
                    inp[2] = ink->attn_in_view[2];
                }
            }
            if (pass > 0) {
                if (n_tok == 1u) {
                    /* Fused 3-way Q/K/V loop controllers: 2 dispatches instead of 6! (-96 disp/tok) */
                    uint32_t cur_str = is_swa ? 0u : state_stride;
                    uint32_t h_str = ink->prefill_cap * 64u;
                    const p36_gpu_tensor *ctrl_cur = is_swa ? inp[0] : ink->attn_in_buf;
                    INK_CHK(p36_gpu_inkling2_control_prep_down3_tensor(
                                ink->ctrl_h_buf, m->map, m->size,
                                l->loop_down[0]->abs_offset, l->loop_down[1]->abs_offset, l->loop_down[2]->abs_offset,
                                ctrl_cur, ink->msg_buf, D, hp->loop_gate_rank,
                                cur_str, state_stride, h_str), "ctrl_pd3");
                    INK_CHK(p36_gpu_inkling2_control_gate3_tensor(
                                ink->ctrl_out_buf, m->map, m->size,
                                l->loop_forget[0]->abs_offset, l->loop_gain[0]->abs_offset,
                                l->loop_forget[1]->abs_offset, l->loop_gain[1]->abs_offset,
                                l->loop_forget[2]->abs_offset, l->loop_gain[2]->abs_offset,
                                ink->ctrl_h_buf, ctrl_cur, ink->msg_buf, D, hp->loop_gate_rank,
                                cur_str, state_stride, h_str, state_stride), "ctrl_g3");
                    inp[0] = ink->ctrl_out[0];
                    inp[1] = ink->ctrl_out[1];
                    inp[2] = ink->ctrl_out[2];
                } else {
                    for (int j = 0; j < 3; j++) {
                        INK_CHK(p36_inkling2_control_vk(m, l, hp, ink, ink->ctrl_out[j], inp[j], ink->msg_view[j], j, n_tok), "ctrl_qkv");
                        inp[j] = ink->ctrl_out[j];
                    }
                }
            }
            if (is_swa && pass <= 0) {
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(ink->norm[0], ink->q8[0], inp[0],
                                                             m->map, m->size, l->attn_norm->abs_offset,
                                                             D, n_tok, hp->rms_eps), "attn_norm_q8_swa");
                n_in[0] = n_in[1] = n_in[2] = ink->norm[0];
                q8_in[0] = q8_in[1] = q8_in[2] = ink->q8[0];
            } else if (n_tok == 1u) {
                const p36_tensor *nw[3] = {
                    is_swa ? l->attn_norm : l->attn_q_inp_norm,
                    is_swa ? l->attn_norm : l->attn_k_inp_norm,
                    is_swa ? l->attn_norm : l->attn_v_inp_norm,
                };
                p36_gpu_tensor *inp_base = (pass > 0) ? ink->ctrl_out_buf : ink->attn_in_buf;
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k3_tensor(
                            ink->norm_buf, ink->q8_buf, inp_base, m->map, m->size,
                            nw[0]->abs_offset, nw[1]->abs_offset, nw[2]->abs_offset,
                            D, state_stride, ink->prefill_cap * (D / 256u), hp->rms_eps), "attn_norm_q8_3");
                for (int j = 0; j < 3; j++) {
                    n_in[j] = ink->norm[j];
                    q8_in[j] = ink->q8[j];
                }
            } else {
                const p36_tensor *nw[3] = {
                    is_swa ? l->attn_norm : l->attn_q_inp_norm,
                    is_swa ? l->attn_norm : l->attn_k_inp_norm,
                    is_swa ? l->attn_norm : l->attn_v_inp_norm,
                };
                for (int j = 0; j < 3; j++) {
                    INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(ink->norm[j], ink->q8[j], inp[j],
                                                                 m->map, m->size, nw[j]->abs_offset,
                                                                 D, n_tok, hp->rms_eps), "attn_norm_q8_3");
                    n_in[j] = ink->norm[j];
                    q8_in[j] = ink->q8[j];
                }
            }
        }

        /* Attention projections */
        bool fused_gate = false;
        if (n_tok == 1u &&
            l->attn_q->type == P36_TENSOR_Q4_K && l->attn_k->type == P36_TENSOR_Q4_K &&
            l->attn_v->type == P36_TENSOR_Q4_K && l->attn_rel->type == P36_TENSOR_Q4_K &&
            p36_gpu_inkling2_qkv_rel_q4k_decode_tensor(ink->q_buf, ink->k_buf, ink->v_buf, ink->r_buf,
                                                       m->map, m->size,
                                                       l->attn_q->abs_offset, l->attn_k->abs_offset,
                                                       l->attn_v->abs_offset, l->attn_rel->abs_offset,
                                                       q8_in[0], q8_in[1], q8_in[2],
                                                       D, Q, KV, KV, hp->n_head * hp->n_relative)) {
            /* Fused 0%-tail Q4_K decode: Q + K + V + rel in 1 dispatch! (-144 disp/tok) */
        } else {
            if (n_tok == 1u &&
                l->attn_q->type == P36_TENSOR_Q8_0 && l->attn_gate->type == P36_TENSOR_Q8_0 &&
                p36_gpu_matmul_q8_0_pair_scaled_tensor(ink->q_buf, ink->gate_buf, m->map, m->size,
                                                       l->attn_q->abs_offset, l->attn_gate->abs_offset,
                                                       D, Q, Q, n_in[0], 1.0f, 1.0f)) {
                fused_gate = true;
            } else {
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_q, n_in[0], q8_in[0], ink->q_buf, D, Q, n_tok, 1.0f), "attn_q");
            }
            if (n_tok == 1u && n_in[1] == n_in[2] &&
                l->attn_k->type == P36_TENSOR_Q8_0 && l->attn_v->type == P36_TENSOR_Q8_0 &&
                p36_gpu_matmul_q8_0_pair_scaled_tensor(ink->k_buf, ink->v_buf, m->map, m->size,
                                                       l->attn_k->abs_offset, l->attn_v->abs_offset,
                                                       D, KV, KV, n_in[1], 1.0f, 1.0f)) {
                /* Fused Q8_0 pair: attn_k + attn_v on SWA pass 0 */
            } else {
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k, n_in[1], q8_in[1], ink->k_buf, D, KV, n_tok, 1.0f), "attn_k");
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v, n_in[2], q8_in[2], ink->v_buf, D, KV, n_tok, 1.0f), "attn_v");
            }
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_rel, n_in[0], q8_in[0], ink->r_buf, D, hp->n_head * hp->n_relative, n_tok, 1.0f), "attn_rel");
        }
        if (fused_gate) {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_sink_proj, n_in[0], q8_in[0], ink->sinks_buf, D, hp->n_head, n_tok, 1.0f), "attn_sink");
        } else if (n_tok == 1u &&
            l->attn_gate->type == P36_TENSOR_Q8_0 && l->attn_sink_proj->type == P36_TENSOR_Q8_0 &&
            p36_gpu_matmul_q8_0_pair_scaled_tensor(ink->gate_buf, ink->sinks_buf, m->map, m->size,
                                                   l->attn_gate->abs_offset, l->attn_sink_proj->abs_offset,
                                                   D, Q, hp->n_head,
                                                   n_in[0], 1.0f, 1.0f)) {
            /* Fused Q8_0 pair: attn_gate + attn_sink_proj in 1 dispatch! (-48 disp/tok) */
        } else {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_gate, n_in[0], q8_in[0], ink->gate_buf, D, Q, n_tok, 1.0f), "attn_gate");
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_sink_proj, n_in[0], q8_in[0], ink->sinks_buf, D, hp->n_head, n_tok, 1.0f), "attn_sink");
        }
        if (!is_swa) {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k_conv_proj, n_in[1], q8_in[1], ink->k_dyn, D, hp->n_conv * hp->n_head_kv, n_tok, 1.0f), "k_dyn");
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v_conv_proj, n_in[2], q8_in[2], ink->v_dyn, D, hp->n_conv * hp->n_head_kv, n_tok, 1.0f), "v_dyn");
        }

        /* Fused K/V short-conv + K clip + V rmsnorm + F16 KV store (1 dispatch instead of 3!) */
        const float kv_conv_scale = is_swa ? inv_sqrt_5 : inv_sqrt_9;
        INK_CHK(p36_gpu_inkling2_conv_kv_store_tensor(ink->k_buf, ink->v_buf, m->map, m->size,
                                                      l->attn_k_conv->abs_offset, l->attn_v_conv->abs_offset,
                                                      ink->conv_k[il], ink->conv_v[il],
                                                      is_swa ? NULL : ink->k_dyn, is_swa ? NULL : ink->v_dyn,
                                                      ink->kv_k[il], ink->kv_v[il],
                                                      hp->n_head_kv, n_tok, pos0, kv_cap, is_swa ? 1u : 0u,
                                                      kv_conv_scale, hp->qk_clip), "conv_kv");

        /* Fused Q clip + R clip (1 dispatch instead of 2!) */
        INK_CHK(p36_gpu_inkling2_clip_qr_tensor(ink->q_buf, ink->r_buf, m->map, m->size,
                                                l->attn_q_clip->abs_offset, l->attn_q_clip->bytes,
                                                l->attn_rel_clip->abs_offset, l->attn_rel_clip->bytes,
                                                hp->n_head, n_tok, pos0, is_swa ? 0u : 1u,
                                                hp->qk_clip, hp->rel_clip,
                                                hp->log_scale_floor, hp->log_scale_alpha), "clip_qr");

        /* FlashAttention + fused sigmoid gate */
        INK_CHK(p36_gpu_inkling2_attn_tensor(ink->attn_out, ink->q_buf, ink->r_buf,
                                             m->map, m->size, l->attn_rel_bias->abs_offset, l->attn_rel_bias->bytes,
                                             ink->sinks_buf, ink->gate_buf, ink->kv_k[il], ink->kv_v[il],
                                             hp->n_head, hp->n_head_kv, n_tok, pos0, kv_cap,
                                             is_swa ? 1u : 0u, hp->n_relative_extent, attn_scale), "attn");

        /* Output projection */
        INK_CHK(p36_gpu_tensor_matmul_scaled(m, l->attn_output, ink->attn_out, ink->wo_out, Q, D, n_tok, 1.0f), "attn_out");

        /* Block residual boundary + attn_post_conv + FFN mix (+ optional ffn_norm) */
        if (n_tok == 1u) {
            bool is_boundary = ((ip % hp->attn_res_block_size) == 0u);
            uint32_t push_slot = is_boundary ? n_blocks : 0xffffffffu;
            uint32_t has_res = is_boundary ? 0u : 1u;
            INK_CHK(p36_gpu_inkling2_post_attn_mix_tensor(ink->blocks_normed, ink->wo_out, inp_sa_t,
                                                          m->map, m->size,
                                                          l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                          ink->ffn_inp,
                                                          l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                          l->ffn_norm->abs_offset, pass <= 0,
                                                          pass <= 0 ? ink->norm[0] : ink->cur,
                                                          D, n_blocks, state_stride, push_slot, has_res,
                                                          inv_sqrt_5, hp->rms_eps), "post_attn_mix");
            if (is_boundary) n_blocks++;
            if (pass > 0) {
                if (!p36_inkling2_control_vk(m, l, hp, ink, ink->ctrl_out[3], ink->cur, ink->msg_view[3], 3, 1u)) {
                    return false;
                }
                if (!p36_gpu_inkling2_rms_norm_q8k_tensor(ink->norm[0], NULL, ink->ctrl_out[3], m->map, m->size,
                                                          l->ffn_norm->abs_offset, D, 1u, hp->rms_eps)) {
                    return false;
                }
            }
        } else {
            if ((ip % hp->attn_res_block_size) == 0u) {
                if (!p36_gpu_inkling2_mix_tensor(ink->blocks_normed, inp_sa_t, m->map, m->size,
                                                 0, 0, NULL, n_tok, D, 0, 1, 0, 0,
                                                 state_stride, state_stride, n_blocks)) {
                    return false;
                }
                n_blocks++;
                if (!p36_gpu_inkling2_conv_tensor(ink->ffn_inp, ink->wo_out, m->map, m->size,
                                                  l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                  NULL, NULL, D, n_tok, 0, 0, inv_sqrt_5)) {
                    return false;
                }
            } else {
                if (!p36_gpu_inkling2_conv_tensor(ink->ffn_inp, ink->wo_out, m->map, m->size,
                                                  l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                  NULL, inp_sa_t, D, n_tok, 0, 0, inv_sqrt_5)) {
                    return false;
                }
            }
            if (pass <= 0) {
                if (!p36_gpu_inkling2_mix_norm_tensor(ink->blocks_normed, ink->ffn_inp, m->map, m->size,
                                                      l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                      l->ffn_norm->abs_offset,
                                                      ink->norm[0], NULL, n_tok, D, n_blocks, 1,
                                                      state_stride, hp->rms_eps)) {
                    return false;
                }
            } else {
                if (!p36_gpu_inkling2_mix_tensor(ink->blocks_normed, ink->ffn_inp, m->map, m->size,
                                                 l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                 ink->cur, n_tok, D, n_blocks, 1, 1, 0,
                                                 state_stride, state_stride, 0xffffffffu)) {
                    return false;
                }
                if (!p36_inkling2_control_vk(m, l, hp, ink, ink->ctrl_out[3], ink->cur, ink->msg_view[3], 3, n_tok)) {
                    return false;
                }
                if (!p36_gpu_inkling2_rms_norm_q8k_tensor(ink->norm[0], NULL, ink->ctrl_out[3], m->map, m->size,
                                                          l->ffn_norm->abs_offset, D, n_tok, hp->rms_eps)) {
                    return false;
                }
            }
        }

        /* MoE router + top-k + gate_up/SITU + down + ffn_post_conv + residual add */
        if (n_tok >= 16u && l->ffn_gate_inp->type == P36_TENSOR_F32 && (hp->n_expert % 64u) == 0u) {
            INK_CHK(p36_gpu_inkling2_f32_gemm_tensor(ink->router_logits, m->map, m->size,
                                                     l->ffn_gate_inp->abs_offset,
                                                     ink->norm[0], D, hp->n_expert, n_tok), "router_gemm");
        } else {
            INK_CHK(p36_gpu_tensor_matmul_scaled(m, l->ffn_gate_inp, ink->norm[0], ink->router_logits, D, hp->n_expert, n_tok, 1.0f), "router_mm");
        }
        INK_CHK(p36_gpu_inkling2_router_topk_tensor(ink->selected, ink->router_w, ink->router_logits,
                                                    m->map, m->size,
                                                    l->exp_probs_b ? l->exp_probs_b->abs_offset : 0,
                                                    l->exp_probs_b != NULL,
                                                    n_tok, hp->n_expert, hp->n_expert_used, hp->expert_weights_scale), "router_topk");
        INK_CHK(p36_gpu_inkling2_moe_ffn_tensor(n_tok == 1u ? ink->cur : ink->moe_out,
                                                ink->moe_mid, ink->moe_down,
                                                m->map, m->size,
                                                l->ffn_gate_up_exps->abs_offset, l->ffn_gate_up_exps->type, l->ffn_gate_up_exps->bytes,
                                                l->ffn_gate_up_exps_scale ? l->ffn_gate_up_exps_scale->abs_offset : 0,
                                                l->ffn_gate_up_exps_scale != NULL,
                                                l->ffn_down_exps->abs_offset, l->ffn_down_exps->type, l->ffn_down_exps->bytes,
                                                l->ffn_down_exps_scale ? l->ffn_down_exps_scale->abs_offset : 0,
                                                l->ffn_down_exps_scale != NULL,
                                                l->ffn_post_conv->abs_offset, ink->conv_ffn_post[il], ink->ffn_inp, inv_sqrt_5,
                                                ink->norm[0], ink->selected, ink->router_w,
                                                n_tok, D, hp->n_ff_exp, D, hp->n_expert, hp->n_expert_used,
                                                hp->situ_beta, hp->situ_linear_beta), "moe_ffn");
        if (n_tok > 1u) {
            INK_CHK(p36_gpu_inkling2_conv_tensor(ink->cur, ink->moe_out, m->map, m->size,
                                                 l->ffn_post_conv->abs_offset, ink->conv_ffn_post[il],
                                                 NULL, ink->ffn_inp, D, n_tok, 0, 0, inv_sqrt_5), "conv_ffn");
        }
    }

    /* Final fused depth-residual mix + output_norm (+ Q8_K on decode!) */
    {
        uint32_t il = hp->n_logical_layer, ip = 0;
        p36_gpu_tensor *out_q8 = (n_tok == 1u && logits_out) ? ink->q8[0] : NULL;
        INK_CHK(p36_gpu_inkling2_mix_norm_tensor(ink->blocks_normed, ink->cur, m->map, m->size,
                                                 w->output_res_query->abs_offset, w->output_res_query->bytes,
                                                 w->output_norm->abs_offset,
                                                 rt->norm, out_q8, n_tok, D, n_blocks, 1,
                                                 state_stride, hp->rms_eps), "out_mix_norm");

        /* Final LM head projection + softcapping */
        if (logits_out) {
            p36_gpu_tensor *last = NULL;
            const p36_gpu_tensor *out_in = rt->norm;
            uint32_t rows = n_tok;
            if (!logits_all_rows && n_tok > 1u) {
                last = p36_gpu_tensor_view(rt->norm, (uint64_t)(n_tok - 1u) * D * sizeof(float), (uint64_t)D * sizeof(float));
                out_in = last;
                rows = 1u;
            }
            bool ok = out_in && (out_q8
                ? p36_gpu_tensor_matmul_q8_or_float_scaled(m, w->output, out_in, out_q8, logits_out,
                                                           D, hp->n_vocab_padded, 1u, 1.0f)
                : p36_gpu_tensor_matmul_scaled(m, w->output, out_in, logits_out,
                                               D, hp->n_vocab_padded, rows, 1.0f));
            p36_gpu_tensor_free(last);
            INK_CHK(ok, "lm_head");
            if (hp->final_logit_softcapping > 0.0f) {
                /* Rows are n_vocab_padded apart; cover every row's real
                 * vocab (the padded tail in between is capped harmlessly).
                 * rows == 1 is unchanged: exactly n_vocab_unpadded. */
                const uint32_t cap_n = (rows - 1u) * hp->n_vocab_padded + hp->n_vocab_unpadded;
                INK_CHK(p36_gpu_inkling2_softcap_tensor(logits_out, cap_n, hp->final_logit_softcapping), "softcap");
            }
        }
    }
    return true;
}
#endif
