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
    uint32_t cap;
    p36_gpu_tensor *slab;              /* NULL for unaliased decode scratch; single aliased slab for prefill */
    p36_gpu_tensor *embed_stage;       /* host-visible [cap, D] f32 (prefill only; decode uses rt->embed_stage) */
    p36_gpu_tensor *embed_hidden;      /* [cap, D] f32 */
    p36_gpu_tensor *out_norm;          /* [cap, D] f32 */
    p36_gpu_tensor *cur;
    p36_gpu_tensor *ffn_inp;
    p36_gpu_tensor *prelude_cur;
    p36_gpu_tensor *blocks_normed;     /* [4, cap, n_embd] */
    p36_gpu_tensor *bank_normed;       /* [4, cap, n_embd] */
    p36_gpu_tensor *msg_buf;           /* [4, cap, n_embd] */
    p36_gpu_tensor *msg_view[4];
    p36_gpu_tensor *msg_view_12;       /* [2, cap, n_embd] view of slices 1..2 */
    p36_gpu_tensor *attn_in_buf;       /* [3, cap, n_embd] */
    p36_gpu_tensor *attn_in_view[3];
    p36_gpu_tensor *ctrl_in;           /* [cap, 2 * n_embd] */
    p36_gpu_tensor *ctrl_h_buf;        /* [3, cap, 64] contiguous */
    p36_gpu_tensor *ctrl_h[3];         /* 3x [cap, 64] views for parallel Q/K/V */
    p36_gpu_tensor *ctrl_out_buf;      /* [4, cap, n_embd] contiguous */
    p36_gpu_tensor *ctrl_out[4];       /* [cap, n_embd] views */
    p36_gpu_tensor *norm_buf;          /* [3, cap, n_embd] contiguous */
    p36_gpu_tensor *norm[3];           /* [cap, n_embd] views */
    p36_gpu_tensor *q8_buf;            /* [3, cap, q8_k(n_embd)] contiguous */
    p36_gpu_tensor *q8[3];             /* [cap, q8_k(n_embd)] views */
    p36_gpu_tensor *q_buf;             /* [cap, 3072] */
    p36_gpu_tensor *k_buf;             /* [cap, 1024] */
    p36_gpu_tensor *v_buf;             /* [cap, 1024] */
    p36_gpu_tensor *r_buf;             /* [cap, 384] */
    p36_gpu_tensor *sinks_buf;         /* [cap, 24] */
    p36_gpu_tensor *gate_buf;          /* [cap, 3072] */
    p36_gpu_tensor *k_dyn;             /* [cap, 32] */
    p36_gpu_tensor *v_dyn;             /* [cap, 32] */
    p36_gpu_tensor *attn_out;          /* [cap, 3072] */
    p36_gpu_tensor *wo_out;            /* [cap, 2304] */
    p36_gpu_tensor *swa_tmp_k;         /* [cap] KV rows, same format as kv_k/kv_v (prefill SWA staging) */
    p36_gpu_tensor *swa_tmp_v;         /* [cap] KV rows, same format as kv_k/kv_v (prefill SWA staging) */
    p36_gpu_tensor *router_logits;     /* [cap, 128] */
    p36_gpu_tensor *selected;          /* [cap, 8] u32 (host-visible) */
    p36_gpu_tensor *router_w;          /* [cap, 8] f32 */
    p36_gpu_tensor *moe_mid;           /* [cap * 8, 1152] */
    p36_gpu_tensor *moe_down;          /* [cap * 8, 2304] */
    p36_gpu_tensor *moe_out;           /* [cap, 2304] */
} p36_inkling2_scratch;

typedef struct {
    uint32_t prefill_cap;
    uint32_t full_kv_cap;
    uint32_t full_kv_init_cap;
    uint32_t full_kv_limit;
    uint32_t full_kv_layer_cap[P36_INK_MAX_LOGICAL];
    uint32_t swa_kv_cap;
    p36_gpu_tensor *kv_k[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *kv_v[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_k[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_v[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_attn_post[P36_INK_MAX_LOGICAL];
    p36_gpu_tensor *conv_ffn_post[P36_INK_MAX_LOGICAL];
    bool is_swa_logical[P36_INK_MAX_LOGICAL];
    uint32_t n_logical;

    p36_inkling2_scratch decode;       /* permanent cap=1 (~0.3 MiB, zero false aliasing) */
    p36_inkling2_scratch prefill;      /* lazy aliased slab (freed on decode/reset) */
} p36_inkling2_runtime;

/* KV row = 8 heads x 128 dims. Default int8 (128 x int8 + 4 f16 scales = 136 B/head) halves the
 * KV bytes that decode attention reads; P36_INK_KV_F16=1 keeps F16 (vk_14 reads the same env). */
static inline uint64_t p36_ink_kv_row_bytes(void) {
    static int q8 = -1;
    if (q8 < 0) q8 = getenv("P36_INK_KV_F16") ? 0 : 1;
    return q8 ? 8ull * 136ull : 1024ull * sizeof(uint16_t);
}

static inline uint64_t p36_ink_align256(uint64_t n) {
    return (n + 255ull) & ~255ull;
}

static void p36_inkling2_scratch_free(p36_inkling2_scratch *sc) {
    if (!sc) return;
    for (int i = 0; i < 4; i++) {
        p36_gpu_tensor_free(sc->msg_view[i]);
        p36_gpu_tensor_free(sc->ctrl_out[i]);
    }
    p36_gpu_tensor_free(sc->msg_view_12);
    for (int i = 0; i < 3; i++) {
        p36_gpu_tensor_free(sc->attn_in_view[i]);
        p36_gpu_tensor_free(sc->ctrl_h[i]);
        p36_gpu_tensor_free(sc->norm[i]);
        p36_gpu_tensor_free(sc->q8[i]);
    }
    p36_gpu_tensor_free(sc->embed_hidden);
    p36_gpu_tensor_free(sc->out_norm);
    p36_gpu_tensor_free(sc->cur);
    p36_gpu_tensor_free(sc->ffn_inp);
    p36_gpu_tensor_free(sc->prelude_cur);
    p36_gpu_tensor_free(sc->blocks_normed);
    p36_gpu_tensor_free(sc->bank_normed);
    p36_gpu_tensor_free(sc->msg_buf);
    p36_gpu_tensor_free(sc->attn_in_buf);
    p36_gpu_tensor_free(sc->ctrl_in);
    p36_gpu_tensor_free(sc->ctrl_h_buf);
    p36_gpu_tensor_free(sc->ctrl_out_buf);
    p36_gpu_tensor_free(sc->norm_buf);
    p36_gpu_tensor_free(sc->q8_buf);
    p36_gpu_tensor_free(sc->q_buf);
    p36_gpu_tensor_free(sc->k_buf);
    p36_gpu_tensor_free(sc->v_buf);
    p36_gpu_tensor_free(sc->r_buf);
    p36_gpu_tensor_free(sc->sinks_buf);
    p36_gpu_tensor_free(sc->gate_buf);
    p36_gpu_tensor_free(sc->k_dyn);
    p36_gpu_tensor_free(sc->v_dyn);
    p36_gpu_tensor_free(sc->attn_out);
    p36_gpu_tensor_free(sc->wo_out);
    p36_gpu_tensor_free(sc->swa_tmp_k);
    p36_gpu_tensor_free(sc->swa_tmp_v);
    p36_gpu_tensor_free(sc->router_logits);
    p36_gpu_tensor_free(sc->selected);
    p36_gpu_tensor_free(sc->router_w);
    p36_gpu_tensor_free(sc->moe_mid);
    p36_gpu_tensor_free(sc->moe_down);
    p36_gpu_tensor_free(sc->moe_out);
    p36_gpu_tensor_free(sc->embed_stage);
    p36_gpu_tensor_free(sc->slab);
    memset(sc, 0, sizeof(*sc));
}

static bool p36_inkling2_scratch_init(p36_inkling2_scratch *sc, uint32_t cap, bool aliased) {
    memset(sc, 0, sizeof(*sc));
    const uint32_t D = 2304u;
    const uint32_t Q = 3072u;
    const uint32_t KV = 1024u;
    const uint64_t tok_embd_bytes = (uint64_t)cap * D * sizeof(float);
    const uint64_t q8_bytes = (uint64_t)cap * (D / 256u) * P36_VK_Q8_K_BYTES;
    const uint64_t h_slice_bytes = (uint64_t)cap * 64u * sizeof(float);
    const uint64_t q_bytes = (uint64_t)cap * Q * sizeof(float);
    const uint64_t kv_f32_bytes = (uint64_t)cap * KV * sizeof(float);
    const uint64_t kv_cache_bytes = (uint64_t)cap * p36_ink_kv_row_bytes();
    const uint64_t r_bytes = (uint64_t)cap * 384u * sizeof(float);
    const uint64_t sinks_bytes = (uint64_t)cap * 24u * sizeof(float);
    const uint64_t dyn_bytes = (uint64_t)cap * 32u * sizeof(float);
    const uint64_t rlog_bytes = (uint64_t)cap * 128u * sizeof(float);
    const uint64_t rw_bytes = (uint64_t)cap * 8u * sizeof(float);
    const uint64_t moe_mid_bytes = (uint64_t)cap * 8u * 1152u * sizeof(float);
    const uint64_t moe_down_bytes = (uint64_t)cap * 8u * D * sizeof(float);

    sc->cap = cap;
    sc->selected = p36_gpu_tensor_alloc((uint64_t)cap * 8u * sizeof(uint32_t));
    if (!sc->selected) goto fail;
    if (aliased) {
        sc->embed_stage = p36_gpu_tensor_alloc(tok_embd_bytes);
        if (!sc->embed_stage) goto fail;

        /* Region A: persistent across layers/passes within a forward call. */
        uint64_t off = 0;
        uint64_t o_cur = off;           off += p36_ink_align256(tok_embd_bytes);
        uint64_t o_prelude = off;       off += p36_ink_align256(tok_embd_bytes);
        uint64_t o_blocks = off;        off += p36_ink_align256(4u * tok_embd_bytes);
        uint64_t o_bank = off;          off += p36_ink_align256(4u * tok_embd_bytes);
        uint64_t o_ffn_inp = off;       off += p36_ink_align256(tok_embd_bytes);
        uint64_t reg_b = off;

        /* Region B: phase-aliased workspace.
         * - norm_buf[0..2] sits at reg_b (embed_hidden, out_norm, ctrl_in alias reg_b).
         * - q8_buf and msg_buf follow norm_buf (live through pre-attn & ctrl_ffn, dead during MoE).
         * - Sub-phase 1 (attn_in_buf + ctrl_out_buf + ctrl_h_buf) and Sub-phase 2 (Q/K/V/Attn/wo/swa_tmp)
         *   do not overlap and share b3.
         * - Sub-phase 3 (MoE: router_logits, router_w, moe_mid, moe_down, moe_out) starts right
         *   after norm[0] (reg_b + tok_embd_bytes), since norm[1..2]/q8/msg/attn are dead. */
        uint64_t o_norm = reg_b;
        uint64_t b1 = reg_b + p36_ink_align256(3u * tok_embd_bytes);
        uint64_t o_q8 = b1;             b1 += p36_ink_align256(3u * q8_bytes);
        uint64_t o_msg = b1;            b1 += p36_ink_align256(4u * tok_embd_bytes);
        uint64_t b3 = b1;

        uint64_t p1 = b3;
        uint64_t o_attn_in = p1;        p1 += p36_ink_align256(3u * tok_embd_bytes);
        uint64_t o_ctrl_out = p1;       p1 += p36_ink_align256(4u * tok_embd_bytes);
        uint64_t o_ctrl_h = p1;         p1 += p36_ink_align256(3u * h_slice_bytes);

        uint64_t p2 = b3;
        uint64_t o_q = p2;              p2 += p36_ink_align256(q_bytes);
        uint64_t o_k = p2;              p2 += p36_ink_align256(kv_f32_bytes);
        uint64_t o_v = p2;              p2 += p36_ink_align256(kv_f32_bytes);
        uint64_t o_r = p2;              p2 += p36_ink_align256(r_bytes);
        uint64_t o_sinks = p2;          p2 += p36_ink_align256(sinks_bytes);
        uint64_t o_gate = p2;           p2 += p36_ink_align256(q_bytes);
        uint64_t o_k_dyn = p2;          p2 += p36_ink_align256(dyn_bytes);
        uint64_t o_v_dyn = p2;          p2 += p36_ink_align256(dyn_bytes);
        uint64_t o_attn_out = p2;       p2 += p36_ink_align256(q_bytes);
        uint64_t o_wo_out = p2;         p2 += p36_ink_align256(tok_embd_bytes);
        uint64_t o_swa_k = p2;          p2 += p36_ink_align256(kv_cache_bytes);
        uint64_t o_swa_v = p2;          p2 += p36_ink_align256(kv_cache_bytes);

        uint64_t p3 = reg_b + p36_ink_align256(tok_embd_bytes);
        uint64_t o_rlog = p3;           p3 += p36_ink_align256(rlog_bytes);
        uint64_t o_rw = p3;             p3 += p36_ink_align256(rw_bytes);
        uint64_t o_moe_mid = p3;        p3 += p36_ink_align256(moe_mid_bytes);
        uint64_t o_moe_down = p3;       p3 += p36_ink_align256(moe_down_bytes);
        uint64_t o_moe_out = p3;        p3 += p36_ink_align256(tok_embd_bytes);

        uint64_t slab_bytes = p1;
        if (p2 > slab_bytes) slab_bytes = p2;
        if (p3 > slab_bytes) slab_bytes = p3;

        sc->slab = p36_gpu_tensor_alloc_scratch(slab_bytes);
        if (!sc->slab) goto fail;

        sc->cur = p36_gpu_tensor_view(sc->slab, o_cur, tok_embd_bytes);
        sc->prelude_cur = p36_gpu_tensor_view(sc->slab, o_prelude, tok_embd_bytes);
        sc->blocks_normed = p36_gpu_tensor_view(sc->slab, o_blocks, 4u * tok_embd_bytes);
        sc->bank_normed = p36_gpu_tensor_view(sc->slab, o_bank, 4u * tok_embd_bytes);
        sc->ffn_inp = p36_gpu_tensor_view(sc->slab, o_ffn_inp, tok_embd_bytes);

        sc->embed_hidden = p36_gpu_tensor_view(sc->slab, reg_b, tok_embd_bytes);
        sc->out_norm = p36_gpu_tensor_view(sc->slab, reg_b, tok_embd_bytes);
        sc->ctrl_in = p36_gpu_tensor_view(sc->slab, reg_b, 2u * tok_embd_bytes);
        sc->norm_buf = p36_gpu_tensor_view(sc->slab, o_norm, 3u * tok_embd_bytes);
        sc->q8_buf = p36_gpu_tensor_view(sc->slab, o_q8, 3u * q8_bytes);
        sc->msg_buf = p36_gpu_tensor_view(sc->slab, o_msg, 4u * tok_embd_bytes);

        sc->attn_in_buf = p36_gpu_tensor_view(sc->slab, o_attn_in, 3u * tok_embd_bytes);
        sc->ctrl_out_buf = p36_gpu_tensor_view(sc->slab, o_ctrl_out, 4u * tok_embd_bytes);
        sc->ctrl_h_buf = p36_gpu_tensor_view(sc->slab, o_ctrl_h, 3u * h_slice_bytes);

        sc->q_buf = p36_gpu_tensor_view(sc->slab, o_q, q_bytes);
        sc->k_buf = p36_gpu_tensor_view(sc->slab, o_k, kv_f32_bytes);
        sc->v_buf = p36_gpu_tensor_view(sc->slab, o_v, kv_f32_bytes);
        sc->r_buf = p36_gpu_tensor_view(sc->slab, o_r, r_bytes);
        sc->sinks_buf = p36_gpu_tensor_view(sc->slab, o_sinks, sinks_bytes);
        sc->gate_buf = p36_gpu_tensor_view(sc->slab, o_gate, q_bytes);
        sc->k_dyn = p36_gpu_tensor_view(sc->slab, o_k_dyn, dyn_bytes);
        sc->v_dyn = p36_gpu_tensor_view(sc->slab, o_v_dyn, dyn_bytes);
        sc->attn_out = p36_gpu_tensor_view(sc->slab, o_attn_out, q_bytes);
        sc->wo_out = p36_gpu_tensor_view(sc->slab, o_wo_out, tok_embd_bytes);
        sc->swa_tmp_k = p36_gpu_tensor_view(sc->slab, o_swa_k, kv_cache_bytes);
        sc->swa_tmp_v = p36_gpu_tensor_view(sc->slab, o_swa_v, kv_cache_bytes);

        sc->router_logits = p36_gpu_tensor_view(sc->slab, o_rlog, rlog_bytes);
        sc->router_w = p36_gpu_tensor_view(sc->slab, o_rw, rw_bytes);
        sc->moe_mid = p36_gpu_tensor_view(sc->slab, o_moe_mid, moe_mid_bytes);
        sc->moe_down = p36_gpu_tensor_view(sc->slab, o_moe_down, moe_down_bytes);
        sc->moe_out = p36_gpu_tensor_view(sc->slab, o_moe_out, tok_embd_bytes);
    } else {
        /* Separate tiny allocations for decode (cap=1, ~0.3 MiB total) so the
         * Vulkan hazard tracker never sees false aliasing between concurrent ops. */
        sc->embed_hidden = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->out_norm = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->cur = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->ffn_inp = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->prelude_cur = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->blocks_normed = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
        sc->bank_normed = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
        sc->msg_buf = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
        sc->attn_in_buf = p36_gpu_tensor_alloc_scratch(3u * tok_embd_bytes);
        sc->ctrl_in = p36_gpu_tensor_alloc_scratch(2u * tok_embd_bytes);
        sc->ctrl_out_buf = p36_gpu_tensor_alloc_scratch(4u * tok_embd_bytes);
        sc->ctrl_h_buf = p36_gpu_tensor_alloc_scratch(3u * h_slice_bytes);
        sc->norm_buf = p36_gpu_tensor_alloc_scratch(3u * tok_embd_bytes);
        sc->q8_buf = p36_gpu_tensor_alloc_scratch(3u * q8_bytes);
        sc->q_buf = p36_gpu_tensor_alloc_scratch(q_bytes);
        sc->k_buf = p36_gpu_tensor_alloc_scratch(kv_f32_bytes);
        sc->v_buf = p36_gpu_tensor_alloc_scratch(kv_f32_bytes);
        sc->r_buf = p36_gpu_tensor_alloc_scratch(r_bytes);
        sc->sinks_buf = p36_gpu_tensor_alloc_scratch(sinks_bytes);
        sc->gate_buf = p36_gpu_tensor_alloc_scratch(q_bytes);
        sc->k_dyn = p36_gpu_tensor_alloc_scratch(dyn_bytes);
        sc->v_dyn = p36_gpu_tensor_alloc_scratch(dyn_bytes);
        sc->attn_out = p36_gpu_tensor_alloc_scratch(q_bytes);
        sc->wo_out = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
        sc->router_logits = p36_gpu_tensor_alloc_scratch(rlog_bytes);
        sc->router_w = p36_gpu_tensor_alloc_scratch(rw_bytes);
        sc->moe_mid = p36_gpu_tensor_alloc_scratch(moe_mid_bytes);
        sc->moe_down = p36_gpu_tensor_alloc_scratch(moe_down_bytes);
        sc->moe_out = p36_gpu_tensor_alloc_scratch(tok_embd_bytes);
    }

    if (!sc->embed_hidden || !sc->out_norm || !sc->cur || !sc->ffn_inp || !sc->prelude_cur ||
        !sc->blocks_normed || !sc->bank_normed || !sc->msg_buf || !sc->attn_in_buf ||
        !sc->ctrl_in || !sc->ctrl_out_buf || !sc->ctrl_h_buf || !sc->norm_buf || !sc->q8_buf ||
        !sc->q_buf || !sc->k_buf || !sc->v_buf || !sc->r_buf || !sc->sinks_buf ||
        !sc->gate_buf || !sc->k_dyn || !sc->v_dyn || !sc->attn_out || !sc->wo_out ||
        !sc->router_logits || !sc->router_w || !sc->moe_mid || !sc->moe_down || !sc->moe_out ||
        (aliased && (!sc->swa_tmp_k || !sc->swa_tmp_v))) {
        goto fail;
    }

    for (int i = 0; i < 4; i++) {
        sc->msg_view[i] = p36_gpu_tensor_view(sc->msg_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        sc->ctrl_out[i] = p36_gpu_tensor_view(sc->ctrl_out_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        if (!sc->msg_view[i] || !sc->ctrl_out[i]) goto fail;
    }
    sc->msg_view_12 = p36_gpu_tensor_view(sc->msg_buf, tok_embd_bytes, 2u * tok_embd_bytes);
    if (!sc->msg_view_12) goto fail;
    for (int i = 0; i < 3; i++) {
        sc->attn_in_view[i] = p36_gpu_tensor_view(sc->attn_in_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        sc->ctrl_h[i] = p36_gpu_tensor_view(sc->ctrl_h_buf, (uint64_t)i * h_slice_bytes, h_slice_bytes);
        sc->norm[i] = p36_gpu_tensor_view(sc->norm_buf, (uint64_t)i * tok_embd_bytes, tok_embd_bytes);
        sc->q8[i] = p36_gpu_tensor_view(sc->q8_buf, (uint64_t)i * q8_bytes, q8_bytes);
        if (!sc->attn_in_view[i] || !sc->ctrl_h[i] || !sc->norm[i] || !sc->q8[i]) goto fail;
    }
    return true;

fail:
    p36_inkling2_scratch_free(sc);
    return false;
}

static bool p36_inkling2_vulkan_runtime_reserve_kv(p36_vulkan_runtime *rt,
                                                   uint32_t need,
                                                   uint32_t limit,
                                                   uint32_t rows) {
    if (!rt || !rt->ink || need > limit || rows > need) return false;
    p36_inkling2_runtime *ink = (p36_inkling2_runtime *)rt->ink;
    if (need <= ink->full_kv_cap) return true;

    const uint64_t row_bytes = p36_ink_kv_row_bytes();
    uint32_t target_cap = p36_kv_next_cap(ink->full_kv_cap, need, limit);
    if (!target_cap) return false;

    /* Free lazy prefill scratch and drain pools before growing KV to minimize peak UMA usage. */
    if (ink->prefill.cap != 0) {
        p36_gpu_synchronize();
        p36_inkling2_scratch_free(&ink->prefill);
    }
    p36_gpu_trim_scratch();

    /* Grow one full-attention layer at a time (peak extra RAM = 1 layer <= 128 MiB at 32k,
     * instead of all 8 layers = 1024 MiB). Existing KV rows remain intact on OOM. */
    uint32_t min_cap = UINT32_MAX;
    for (uint32_t il = 0; il < ink->n_logical; il++) {
        if (ink->is_swa_logical[il]) continue;
        if (ink->full_kv_layer_cap[il] >= need) {
            if (ink->full_kv_layer_cap[il] < min_cap) min_cap = ink->full_kv_layer_cap[il];
            continue;
        }
        uint32_t try_cap = target_cap;
        p36_gpu_tensor *nk = p36_gpu_tensor_alloc_uninitialized((uint64_t)try_cap * row_bytes);
        p36_gpu_tensor *nv = nk ? p36_gpu_tensor_alloc_uninitialized((uint64_t)try_cap * row_bytes) : NULL;
        if ((!nk || !nv) && try_cap > need) {
            p36_gpu_tensor_free(nk);
            p36_gpu_tensor_free(nv);
            p36_gpu_trim_scratch();
            try_cap = need;
            nk = p36_gpu_tensor_alloc_uninitialized((uint64_t)try_cap * row_bytes);
            nv = nk ? p36_gpu_tensor_alloc_uninitialized((uint64_t)try_cap * row_bytes) : NULL;
        }
        if (!nk || !nv) {
            p36_gpu_tensor_free(nk);
            p36_gpu_tensor_free(nv);
            fprintf(stderr,
                    "p36: inkling2 KV cache OOM growing layer %u from %u to %u tokens (%.1f MiB needed)\n",
                    il, ink->full_kv_layer_cap[il], try_cap,
                    (double)(2ull * try_cap * row_bytes) / (1024.0 * 1024.0));
            return false;
        }
        if (rows > 0) {
            uint64_t copy_bytes = (uint64_t)rows * row_bytes;
            if (!p36_gpu_tensor_copy(nk, 0, ink->kv_k[il], 0, copy_bytes) ||
                !p36_gpu_tensor_copy(nv, 0, ink->kv_v[il], 0, copy_bytes) ||
                !p36_gpu_synchronize()) {
                p36_gpu_tensor_free(nk);
                p36_gpu_tensor_free(nv);
                fprintf(stderr, "p36: inkling2 KV cache copy failed growing layer %u to %u tokens\n",
                        il, try_cap);
                return false;
            }
        }
        p36_gpu_tensor_free(ink->kv_k[il]);
        p36_gpu_tensor_free(ink->kv_v[il]);
        ink->kv_k[il] = nk;
        ink->kv_v[il] = nv;
        ink->full_kv_layer_cap[il] = try_cap;
        if (try_cap < min_cap) min_cap = try_cap;
    }
    if (min_cap != UINT32_MAX) ink->full_kv_cap = min_cap;
    return true;
}

static bool p36_inkling2_vulkan_runtime_reset(p36_vulkan_runtime *rt) {
    p36_inkling2_runtime *ink = (p36_inkling2_runtime *)rt->ink;
    if (!ink) return false;
    rt->inp_q8_valid = false;
    if (ink->prefill.cap != 0) {
        p36_gpu_synchronize();
        p36_inkling2_scratch_free(&ink->prefill);
        p36_gpu_trim_scratch();
    }
    {
        bool any_grown = false;
        for (uint32_t il = 0; il < ink->n_logical; il++) {
            if (!ink->is_swa_logical[il] && ink->full_kv_layer_cap[il] > ink->full_kv_init_cap) {
                any_grown = true;
                break;
            }
        }
        if (any_grown) {
            const uint64_t kv_bytes = (uint64_t)ink->full_kv_init_cap * p36_ink_kv_row_bytes();
            p36_gpu_synchronize();
            for (uint32_t il = 0; il < ink->n_logical; il++) {
                if (ink->is_swa_logical[il] || ink->full_kv_layer_cap[il] <= ink->full_kv_init_cap) continue;
                p36_gpu_tensor_free(ink->kv_k[il]);
                p36_gpu_tensor_free(ink->kv_v[il]);
                ink->kv_k[il] = p36_gpu_tensor_alloc_uninitialized(kv_bytes);
                ink->kv_v[il] = p36_gpu_tensor_alloc_uninitialized(kv_bytes);
                if (!ink->kv_k[il] || !ink->kv_v[il]) return false;
                ink->full_kv_layer_cap[il] = ink->full_kv_init_cap;
            }
            ink->full_kv_cap = ink->full_kv_init_cap;
        }
    }
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
        p36_inkling2_scratch_free(&ink->decode);
        p36_inkling2_scratch_free(&ink->prefill);
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
    ink->full_kv_limit = (uint32_t)ctx_size;
    ink->full_kv_init_cap = (uint32_t)ctx_size < 2048u ? (uint32_t)ctx_size : 2048u;
    ink->full_kv_cap = ink->full_kv_init_cap;
    ink->swa_kv_cap = 1024u;
    ink->n_logical = 48u;

    const uint32_t D = 2304u;
    const uint32_t KV = 1024u;

    /* Common rt buffers expected by p36 session/sampling (decode-sized embed_stage: 9 KiB each) */
    rt->embed_stage[0] = p36_gpu_tensor_alloc((uint64_t)D * sizeof(float));
    rt->embed_stage[1] = p36_gpu_tensor_alloc((uint64_t)D * sizeof(float));
    rt->last_h = p36_gpu_tensor_alloc_scratch((uint64_t)D * sizeof(float));
    rt->logits = p36_gpu_tensor_alloc((uint64_t)201024u * sizeof(float));
    rt->top2[0] = p36_gpu_tensor_alloc(2u * sizeof(uint32_t));
    rt->top2[1] = p36_gpu_tensor_alloc(2u * sizeof(uint32_t));
    rt->topk8 = p36_gpu_tensor_alloc(16u * sizeof(float));
    rt->minp_pack = p36_gpu_tensor_alloc(((uint64_t)P36_N_VOCAB * 2u + 4u) * sizeof(uint32_t));
    if (!rt->embed_stage[0] || !rt->embed_stage[1] || !rt->last_h ||
        !rt->logits || !rt->top2[0] || !rt->top2[1] || !rt->topk8 || !rt->minp_pack) {
        goto fail;
    }

    /* Permanent decode scratch (cap=1, ~0.3 MiB); prefill scratch is allocated lazily on n_tok > 1. */
    if (!p36_inkling2_scratch_init(&ink->decode, 1u, false)) goto fail;

    /* Per-logical-layer KV cache (int8 or F16, see p36_ink_kv_row_bytes) and shortconv state (F32, 3 history slots). */
    for (uint32_t il = 0; il < ink->n_logical; il++) {
        uint32_t ip;
        if (il < 6u) ip = il;
        else if (il < 42u) ip = 6u + ((il - 6u) % 12u);
        else ip = 18u + (il - 42u);
        bool is_swa = ((ip % 6u) != 5u);
        ink->is_swa_logical[il] = is_swa;
        uint32_t cap = is_swa ? ink->swa_kv_cap : ink->full_kv_cap;
        if (!is_swa) ink->full_kv_layer_cap[il] = cap;
        uint64_t kv_bytes = (uint64_t)cap * p36_ink_kv_row_bytes();
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

static bool p36_inkling2_embed_tokens_host(const p36_model *m, const p36_tensor *t,
                                           const int *tokens, uint32_t n_tok,
                                           uint32_t n_embd, uint32_t n_vocab,
                                           p36_gpu_tensor *dst) {
    float *p = p36_gpu_tensor_contents_named(dst, "submit_wait_embed_tokens");
    if (!p) return false;
    for (uint32_t i = 0; i < n_tok; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)n_vocab) return false;
        float *row = p + (uint64_t)i * n_embd;
        if (!p36_tensor_row_to_float(m, t, (uint64_t)tokens[i], row, n_embd)) return false;
        if (t->hadamard_inverse) p36_hadamard_inverse_host(row, n_embd);
    }
    return true;
}

static bool p36_inkling2_control_vk(const p36_model *m,
                                    const p36_inkling2_layer_weights *l,
                                    const p36_inkling2_hparams *hp,
                                    p36_inkling2_scratch *sc,
                                    p36_gpu_tensor *out,
                                    const p36_gpu_tensor *cur,
                                    const p36_gpu_tensor *msg,
                                    int mod,
                                    uint32_t n_tok) {
    bool down_ok;
    if (n_tok == 1u) {
        down_ok = p36_gpu_inkling2_control_prep_down_tensor(sc->ctrl_h[0], m->map, m->size,
                                                            l->loop_down[mod]->abs_offset,
                                                            cur, msg, hp->n_embd, hp->loop_gate_rank) != 0;
    } else {
        down_ok = p36_gpu_inkling2_control_prep_tensor(sc->ctrl_in, cur, msg, n_tok, hp->n_embd) &&
                  ((n_tok >= 16u && l->loop_down[mod]->type == P36_TENSOR_F32 && (hp->loop_gate_rank % 64u) == 0u)
                       ? (p36_gpu_inkling2_f32_gemm_tensor(sc->ctrl_h[0], m->map, m->size,
                                                           l->loop_down[mod]->abs_offset,
                                                           sc->ctrl_in, 2u * hp->n_embd, hp->loop_gate_rank, n_tok) != 0)
                       : p36_gpu_tensor_matmul_scaled(m, l->loop_down[mod], sc->ctrl_in, sc->ctrl_h[0],
                                                      2u * hp->n_embd, hp->loop_gate_rank, n_tok, 1.0f));
    }
    return down_ok &&
           p36_gpu_inkling2_control_gate_tensor(out, m->map, m->size,
                                                l->loop_forget[mod]->abs_offset,
                                                l->loop_gain[mod]->abs_offset,
                                                sc->ctrl_h[0], cur, msg,
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
    if (pos0 + n_tok > ink->full_kv_cap) {
        if (!p36_inkling2_vulkan_runtime_reserve_kv(rt, pos0 + n_tok, ink->full_kv_limit, pos0)) {
            fprintf(stderr, "p36: inkling2 failed to grow full KV cache to %u\n", pos0 + n_tok);
            return false;
        }
    }
    p36_inkling2_scratch *sc;
    if (n_tok == 1u) {
        if (ink->prefill.cap != 0) {
            p36_gpu_synchronize();
            p36_inkling2_scratch_free(&ink->prefill);
            p36_gpu_trim_scratch();
        }
        sc = &ink->decode;
    } else {
        uint32_t want_cap = 64u;
        while (want_cap < n_tok && want_cap < ink->prefill_cap) want_cap <<= 1u;
        if (want_cap > ink->prefill_cap) want_cap = ink->prefill_cap;
        if (ink->prefill.cap < want_cap) {
            if (ink->prefill.cap != 0) {
                p36_gpu_synchronize();
                p36_inkling2_scratch_free(&ink->prefill);
                p36_gpu_trim_scratch();
            }
            if (!p36_inkling2_scratch_init(&ink->prefill, want_cap, true)) {
                fprintf(stderr, "p36: inkling2 prefill scratch alloc failed (cap=%u)\n", want_cap);
                return false;
            }
        }
        sc = &ink->prefill;
    }
#define INK_CHK(expr, tag) do { if (!(expr)) { fprintf(stderr, "p36: inkling2 fail at %s (il=%u ip=%u n_tok=%u)\n", (tag), il, ip, n_tok); return false; } } while (0)

    const uint32_t D = hp->n_embd;
    const uint32_t Q = hp->n_head * hp->n_head_dim;
    const uint32_t KV = hp->n_head_kv * hp->n_head_dim;
    const uint32_t state_stride = sc->cap * D;
    const float inv_sqrt_5 = 1.0f / sqrtf(5.0f);
    const float inv_sqrt_9 = 1.0f / 3.0f;
    const float attn_scale = 1.0f / sqrtf((float)hp->n_head_dim);

    /* 1. Embed (host mmap row lookup — avoids 248.5 MiB GPU embedding copy) + token_embd_norm */
    {
        uint32_t il = 0, ip = 0;
        p36_gpu_tensor *stage;
        if (n_tok == 1u) {
            stage = rt->embed_stage[rt->embed_flip & 1u];
            rt->embed_flip ^= 1u;
        } else {
            stage = sc->embed_stage;
        }
        if (embeddings) {
            /* Raw vision embeddings bypass token_embd_norm (matches llama-inkling2). */
            INK_CHK(p36_gpu_tensor_write(stage, 0, embeddings, (uint64_t)n_tok * D * sizeof(float)), "embed_write");
            INK_CHK(p36_gpu_copy_f32_tensor(sc->cur, stage, n_tok * D), "embed_copy_raw");
        } else {
            INK_CHK(p36_inkling2_embed_tokens_host(m, w->token_embd, tokens, n_tok, D, hp->n_vocab_padded, stage), "embed_tokens");
            INK_CHK(p36_gpu_copy_f32_tensor(sc->embed_hidden, stage, n_tok * D), "embed_copy");
            INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(sc->cur, NULL, sc->embed_hidden, m->map, m->size,
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
                INK_CHK(p36_gpu_copy_f32_tensor(sc->prelude_cur, sc->cur, n_tok * D), "prelude_copy");
            } else {
                bank_n = 0;
                for (uint32_t bi = prelude_n_blocks + 1u; bi < n_blocks; bi++) {
                    uint64_t src_off = (uint64_t)bi * state_stride * sizeof(float);
                    uint64_t dst_off = (uint64_t)bank_n * state_stride * sizeof(float);
                    INK_CHK(p36_gpu_tensor_copy(sc->bank_normed, dst_off, sc->blocks_normed, src_off,
                                                (uint64_t)n_tok * D * sizeof(float)), "bank_copy");
                    bank_n++;
                }
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->bank_normed, sc->cur, m->map, m->size,
                                                    0, 0, NULL, n_tok, D, 0, 1, 0, 0,
                                                    state_stride, state_stride, bank_n), "bank_push");
                bank_n++;
                n_blocks = prelude_n_blocks;
                INK_CHK(p36_gpu_copy_f32_tensor(sc->cur, sc->prelude_cur, n_tok * D), "prelude_restore");
            }
        }

        /* Compute loop messages for pass > 0 */
        if (pass > 0) {
            if (l->loop_msg_query) {
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->bank_normed, sc->cur, m->map, m->size,
                                                    l->loop_msg_query->abs_offset, l->loop_msg_query->bytes,
                                                    sc->msg_buf, n_tok, D, bank_n, 0, 4, 0,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg");
            } else {
                uint32_t row = 2u * (ip - hp->loop_start);
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->bank_normed, sc->cur, m->map, m->size,
                                                    w->loop_res_query->abs_offset, w->loop_res_query->bytes,
                                                    sc->msg_view[0], n_tok, D, bank_n, 0, 1, row,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg0");
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->bank_normed, sc->cur, m->map, m->size,
                                                    w->loop_res_query->abs_offset, w->loop_res_query->bytes,
                                                    sc->msg_view[3], n_tok, D, bank_n, 0, 1, row + 1u,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg3");
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->bank_normed, sc->cur, m->map, m->size,
                                                    w->loop_kv_res_query->abs_offset, w->loop_kv_res_query->bytes,
                                                    sc->msg_view_12, n_tok, D, bank_n, 0, 2, row,
                                                    state_stride, state_stride, 0xffffffffu), "loop_msg12");
            }
        }

        /* Zero-copy inpSA: sc->cur is untouched until FFN mix overwrites sc->cur */
        p36_gpu_tensor *inp_sa_t = sc->cur;

        /* Depth-residual mix + loop control + RMSNorm + Q8_K for attention inputs */
        p36_gpu_tensor *n_in[3];
        p36_gpu_tensor *q8_in[3];
        if (is_swa && pass <= 0 && l->attn_res_query) {
            /* Fused mix + RMSNorm + Q8_K (1 dispatch instead of 3!) */
            INK_CHK(p36_gpu_inkling2_mix_norm_tensor(sc->blocks_normed, inp_sa_t, m->map, m->size,
                                                     l->attn_res_query->abs_offset, l->attn_res_query->bytes,
                                                     l->attn_norm->abs_offset,
                                                     sc->norm[0], sc->q8[0], n_tok, D, n_blocks, 1,
                                                     state_stride, hp->rms_eps), "attn_mix_norm_q8");
            n_in[0] = n_in[1] = n_in[2] = sc->norm[0];
            q8_in[0] = q8_in[1] = q8_in[2] = sc->q8[0];
        } else {
            p36_gpu_tensor *inp[3] = { inp_sa_t, inp_sa_t, inp_sa_t };
            if (l->attn_res_query) {
                uint32_t n_q = is_swa ? 1u : 3u;
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->blocks_normed, inp_sa_t, m->map, m->size,
                                                    l->attn_res_query->abs_offset, l->attn_res_query->bytes,
                                                    sc->attn_in_buf, n_tok, D, n_blocks, 1, n_q, 0,
                                                    state_stride, state_stride, 0xffffffffu), "attn_mix");
                if (is_swa) {
                    inp[0] = inp[1] = inp[2] = sc->attn_in_view[0];
                } else {
                    inp[0] = sc->attn_in_view[0];
                    inp[1] = sc->attn_in_view[1];
                    inp[2] = sc->attn_in_view[2];
                }
            }
            if (pass > 0) {
                if (n_tok == 1u) {
                    /* Fused 3-way Q/K/V loop controllers: 2 dispatches instead of 6! (-96 disp/tok) */
                    uint32_t cur_str = is_swa ? 0u : state_stride;
                    uint32_t h_str = sc->cap * 64u;
                    const p36_gpu_tensor *ctrl_cur = is_swa ? inp[0] : sc->attn_in_buf;
                    INK_CHK(p36_gpu_inkling2_control_prep_down3_tensor(
                                sc->ctrl_h_buf, m->map, m->size,
                                l->loop_down[0]->abs_offset, l->loop_down[1]->abs_offset, l->loop_down[2]->abs_offset,
                                ctrl_cur, sc->msg_buf, D, hp->loop_gate_rank,
                                cur_str, state_stride, h_str), "ctrl_pd3");
                    INK_CHK(p36_gpu_inkling2_control_gate3_tensor(
                                sc->ctrl_out_buf, m->map, m->size,
                                l->loop_forget[0]->abs_offset, l->loop_gain[0]->abs_offset,
                                l->loop_forget[1]->abs_offset, l->loop_gain[1]->abs_offset,
                                l->loop_forget[2]->abs_offset, l->loop_gain[2]->abs_offset,
                                sc->ctrl_h_buf, ctrl_cur, sc->msg_buf, D, hp->loop_gate_rank,
                                cur_str, state_stride, h_str, state_stride), "ctrl_g3");
                    inp[0] = sc->ctrl_out[0];
                    inp[1] = sc->ctrl_out[1];
                    inp[2] = sc->ctrl_out[2];
                } else {
                    for (int j = 0; j < 3; j++) {
                        INK_CHK(p36_inkling2_control_vk(m, l, hp, sc, sc->ctrl_out[j], inp[j], sc->msg_view[j], j, n_tok), "ctrl_qkv");
                        inp[j] = sc->ctrl_out[j];
                    }
                }
            }
            if (is_swa && pass <= 0) {
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(sc->norm[0], sc->q8[0], inp[0],
                                                             m->map, m->size, l->attn_norm->abs_offset,
                                                             D, n_tok, hp->rms_eps), "attn_norm_q8_swa");
                n_in[0] = n_in[1] = n_in[2] = sc->norm[0];
                q8_in[0] = q8_in[1] = q8_in[2] = sc->q8[0];
            } else if (n_tok == 1u) {
                const p36_tensor *nw[3] = {
                    is_swa ? l->attn_norm : l->attn_q_inp_norm,
                    is_swa ? l->attn_norm : l->attn_k_inp_norm,
                    is_swa ? l->attn_norm : l->attn_v_inp_norm,
                };
                p36_gpu_tensor *inp_base = (pass > 0) ? sc->ctrl_out_buf : sc->attn_in_buf;
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k3_tensor(
                            sc->norm_buf, sc->q8_buf, inp_base, m->map, m->size,
                            nw[0]->abs_offset, nw[1]->abs_offset, nw[2]->abs_offset,
                            D, state_stride, sc->cap * (D / 256u), hp->rms_eps), "attn_norm_q8_3");
                for (int j = 0; j < 3; j++) {
                    n_in[j] = sc->norm[j];
                    q8_in[j] = sc->q8[j];
                }
            } else {
                const p36_tensor *nw[3] = {
                    is_swa ? l->attn_norm : l->attn_q_inp_norm,
                    is_swa ? l->attn_norm : l->attn_k_inp_norm,
                    is_swa ? l->attn_norm : l->attn_v_inp_norm,
                };
                for (int j = 0; j < 3; j++) {
                    INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(sc->norm[j], sc->q8[j], inp[j],
                                                                 m->map, m->size, nw[j]->abs_offset,
                                                                 D, n_tok, hp->rms_eps), "attn_norm_q8_3");
                    n_in[j] = sc->norm[j];
                    q8_in[j] = sc->q8[j];
                }
            }
        }

        /* Attention projections */
        bool fused_gate = false;
        if (n_tok == 1u &&
            l->attn_q->type == P36_TENSOR_Q4_K && l->attn_k->type == P36_TENSOR_Q4_K &&
            l->attn_v->type == P36_TENSOR_Q4_K && l->attn_rel->type == P36_TENSOR_Q4_K &&
            p36_gpu_inkling2_qkv_rel_q4k_decode_tensor(sc->q_buf, sc->k_buf, sc->v_buf, sc->r_buf,
                                                       m->map, m->size,
                                                       l->attn_q->abs_offset, l->attn_k->abs_offset,
                                                       l->attn_v->abs_offset, l->attn_rel->abs_offset,
                                                       q8_in[0], q8_in[1], q8_in[2],
                                                       D, Q, KV, KV, hp->n_head * hp->n_relative)) {
            /* Fused 0%-tail Q4_K decode: Q + K + V + rel in 1 dispatch! (-144 disp/tok) */
        } else if (n_tok == 1u &&
            l->attn_q->type == P36_TENSOR_Q6_K && l->attn_k->type == P36_TENSOR_Q6_K &&
            l->attn_v->type == P36_TENSOR_Q6_K && l->attn_rel->type == P36_TENSOR_Q6_K &&
            p36_gpu_inkling2_qkv_rel_q6k_decode_tensor(sc->q_buf, sc->k_buf, sc->v_buf, sc->r_buf,
                                                       m->map, m->size,
                                                       l->attn_q->abs_offset, l->attn_k->abs_offset,
                                                       l->attn_v->abs_offset, l->attn_rel->abs_offset,
                                                       q8_in[0], q8_in[1], q8_in[2],
                                                       D, Q, KV, KV, hp->n_head * hp->n_relative)) {
            /* Fused 0%-tail Q6_K decode: Q + K + V + rel in 1 dispatch! (-144 disp/tok) */
        } else {
            if (n_tok == 1u &&
                l->attn_q->type == P36_TENSOR_Q8_0 && l->attn_gate->type == P36_TENSOR_Q8_0 &&
                p36_gpu_matmul_q8_0_pair_scaled_tensor(sc->q_buf, sc->gate_buf, m->map, m->size,
                                                       l->attn_q->abs_offset, l->attn_gate->abs_offset,
                                                       D, Q, Q, n_in[0], 1.0f, 1.0f)) {
                fused_gate = true;
            } else {
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_q, n_in[0], q8_in[0], sc->q_buf, D, Q, n_tok, 1.0f), "attn_q");
            }
            if (n_tok == 1u && n_in[1] == n_in[2] &&
                l->attn_k->type == P36_TENSOR_Q8_0 && l->attn_v->type == P36_TENSOR_Q8_0 &&
                p36_gpu_matmul_q8_0_pair_scaled_tensor(sc->k_buf, sc->v_buf, m->map, m->size,
                                                       l->attn_k->abs_offset, l->attn_v->abs_offset,
                                                       D, KV, KV, n_in[1], 1.0f, 1.0f)) {
                /* Fused Q8_0 pair: attn_k + attn_v on SWA pass 0 */
            } else {
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k, n_in[1], q8_in[1], sc->k_buf, D, KV, n_tok, 1.0f), "attn_k");
                INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v, n_in[2], q8_in[2], sc->v_buf, D, KV, n_tok, 1.0f), "attn_v");
            }
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_rel, n_in[0], q8_in[0], sc->r_buf, D, hp->n_head * hp->n_relative, n_tok, 1.0f), "attn_rel");
        }
        if (fused_gate) {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_sink_proj, n_in[0], q8_in[0], sc->sinks_buf, D, hp->n_head, n_tok, 1.0f), "attn_sink");
        } else if (n_tok == 1u &&
            l->attn_gate->type == P36_TENSOR_Q8_0 && l->attn_sink_proj->type == P36_TENSOR_Q8_0 &&
            p36_gpu_matmul_q8_0_pair_scaled_tensor(sc->gate_buf, sc->sinks_buf, m->map, m->size,
                                                   l->attn_gate->abs_offset, l->attn_sink_proj->abs_offset,
                                                   D, Q, hp->n_head,
                                                   n_in[0], 1.0f, 1.0f)) {
            /* Fused Q8_0 pair: attn_gate + attn_sink_proj in 1 dispatch! (-48 disp/tok) */
        } else {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_gate, n_in[0], q8_in[0], sc->gate_buf, D, Q, n_tok, 1.0f), "attn_gate");
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_sink_proj, n_in[0], q8_in[0], sc->sinks_buf, D, hp->n_head, n_tok, 1.0f), "attn_sink");
        }
        if (!is_swa) {
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k_conv_proj, n_in[1], q8_in[1], sc->k_dyn, D, hp->n_conv * hp->n_head_kv, n_tok, 1.0f), "k_dyn");
            INK_CHK(p36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v_conv_proj, n_in[2], q8_in[2], sc->v_dyn, D, hp->n_conv * hp->n_head_kv, n_tok, 1.0f), "v_dyn");
        }

        /* Fused K/V short-conv + K clip + V rmsnorm + F16 KV store (1 dispatch instead of 3!)
         * For SWA prefill (n_tok > 1), stage new chunk in swa_tmp_k/v so swa_kv_cap=1024
         * ring slots aren't overwritten before attn_prefill reads past keys. */
        const float kv_conv_scale = is_swa ? inv_sqrt_5 : inv_sqrt_9;
        const bool swa_stage = is_swa && (n_tok > 1u);
        INK_CHK(p36_gpu_inkling2_conv_kv_store_tensor(sc->k_buf, sc->v_buf, m->map, m->size,
                                                      l->attn_k_conv->abs_offset, l->attn_v_conv->abs_offset,
                                                      ink->conv_k[il], ink->conv_v[il],
                                                      is_swa ? NULL : sc->k_dyn, is_swa ? NULL : sc->v_dyn,
                                                      swa_stage ? sc->swa_tmp_k : ink->kv_k[il],
                                                      swa_stage ? sc->swa_tmp_v : ink->kv_v[il],
                                                      hp->n_head_kv, n_tok,
                                                      swa_stage ? 0u : pos0,
                                                      kv_cap,
                                                      swa_stage ? 0u : (is_swa ? 1u : 0u),
                                                      kv_conv_scale, hp->qk_clip), "conv_kv");

        /* Fused Q clip + R clip (1 dispatch instead of 2!) */
        INK_CHK(p36_gpu_inkling2_clip_qr_tensor(sc->q_buf, sc->r_buf, m->map, m->size,
                                                l->attn_q_clip->abs_offset, l->attn_q_clip->bytes,
                                                l->attn_rel_clip->abs_offset, l->attn_rel_clip->bytes,
                                                hp->n_head, n_tok, pos0, is_swa ? 0u : 1u,
                                                hp->qk_clip, hp->rel_clip,
                                                hp->log_scale_floor, hp->log_scale_alpha), "clip_qr");

        /* FlashAttention + fused sigmoid gate */
        INK_CHK(p36_gpu_inkling2_attn_tensor(sc->attn_out, sc->q_buf, sc->r_buf,
                                             m->map, m->size, l->attn_rel_bias->abs_offset, l->attn_rel_bias->bytes,
                                             sc->sinks_buf, sc->gate_buf, ink->kv_k[il], ink->kv_v[il],
                                             swa_stage ? sc->swa_tmp_k : NULL, swa_stage ? sc->swa_tmp_v : NULL,
                                             hp->n_head, hp->n_head_kv, n_tok, pos0, kv_cap,
                                             is_swa ? 1u : 0u, hp->n_relative_extent, attn_scale), "attn");

        /* Commit staged SWA prefill K/V into the 1024-slot ring buffer */
        if (swa_stage) {
            uint32_t start = pos0 & (ink->swa_kv_cap - 1u);
            uint32_t first = ink->swa_kv_cap - start;
            if (first > n_tok) first = n_tok;
            uint64_t kv_row_bytes = p36_ink_kv_row_bytes();
            INK_CHK(p36_gpu_tensor_copy(ink->kv_k[il], (uint64_t)start * kv_row_bytes,
                                        sc->swa_tmp_k, 0, (uint64_t)first * kv_row_bytes), "swa_commit_k1");
            INK_CHK(p36_gpu_tensor_copy(ink->kv_v[il], (uint64_t)start * kv_row_bytes,
                                        sc->swa_tmp_v, 0, (uint64_t)first * kv_row_bytes), "swa_commit_v1");
            if (n_tok > first) {
                uint32_t second = n_tok - first;
                INK_CHK(p36_gpu_tensor_copy(ink->kv_k[il], 0, sc->swa_tmp_k,
                                            (uint64_t)first * kv_row_bytes, (uint64_t)second * kv_row_bytes), "swa_commit_k2");
                INK_CHK(p36_gpu_tensor_copy(ink->kv_v[il], 0, sc->swa_tmp_v,
                                            (uint64_t)first * kv_row_bytes, (uint64_t)second * kv_row_bytes), "swa_commit_v2");
            }
        }

        /* Output projection */
        if (n_tok == 1u && l->attn_output->type == P36_TENSOR_Q4_K &&
            p36_gpu_inkling2_wo_q4k_decode_tensor(sc->wo_out, m->map, m->size,
                                                  l->attn_output->abs_offset, sc->attn_out, Q, D)) {
            /* Direct F32->Q4_K decode: eliminates 40 quantize_q8_k dispatches + barriers/tok */
        } else {
            INK_CHK(p36_gpu_tensor_matmul_scaled(m, l->attn_output, sc->attn_out, sc->wo_out, Q, D, n_tok, 1.0f), "attn_out");
        }

        /* Block residual boundary + attn_post_conv + FFN mix (+ optional ffn_norm) */
        if (n_tok == 1u) {
            bool is_boundary = ((ip % hp->attn_res_block_size) == 0u);
            uint32_t push_slot = is_boundary ? n_blocks : 0xffffffffu;
            uint32_t has_res = is_boundary ? 0u : 1u;
            INK_CHK(p36_gpu_inkling2_post_attn_mix_tensor(sc->blocks_normed, sc->wo_out, inp_sa_t,
                                                          m->map, m->size,
                                                          l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                          sc->ffn_inp,
                                                          l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                          l->ffn_norm->abs_offset, pass <= 0,
                                                          pass <= 0 ? sc->norm[0] : sc->cur,
                                                          D, n_blocks, state_stride, push_slot, has_res,
                                                          inv_sqrt_5, hp->rms_eps), "post_attn_mix");
            if (is_boundary) n_blocks++;
            if (pass > 0) {
                INK_CHK(p36_inkling2_control_vk(m, l, hp, sc, sc->ctrl_out[3], sc->cur, sc->msg_view[3], 3, 1u), "ctrl_ffn_1");
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(sc->norm[0], NULL, sc->ctrl_out[3], m->map, m->size,
                                                             l->ffn_norm->abs_offset, D, 1u, hp->rms_eps), "ffn_norm_1");
            }
        } else {
            if ((ip % hp->attn_res_block_size) == 0u) {
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->blocks_normed, inp_sa_t, m->map, m->size,
                                                    0, 0, NULL, n_tok, D, 0, 1, 0, 0,
                                                    state_stride, state_stride, n_blocks), "blk_push");
                n_blocks++;
                INK_CHK(p36_gpu_inkling2_conv_tensor(sc->ffn_inp, sc->wo_out, m->map, m->size,
                                                     l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                     NULL, NULL, D, n_tok, 0, 0, inv_sqrt_5), "post_conv_b");
            } else {
                INK_CHK(p36_gpu_inkling2_conv_tensor(sc->ffn_inp, sc->wo_out, m->map, m->size,
                                                     l->attn_post_conv->abs_offset, ink->conv_attn_post[il],
                                                     NULL, inp_sa_t, D, n_tok, 0, 0, inv_sqrt_5), "post_conv");
            }
            if (pass <= 0) {
                INK_CHK(p36_gpu_inkling2_mix_norm_tensor(sc->blocks_normed, sc->ffn_inp, m->map, m->size,
                                                         l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                         l->ffn_norm->abs_offset,
                                                         sc->norm[0], NULL, n_tok, D, n_blocks, 1,
                                                         state_stride, hp->rms_eps), "ffn_mix_norm");
            } else {
                INK_CHK(p36_gpu_inkling2_mix_tensor(sc->blocks_normed, sc->ffn_inp, m->map, m->size,
                                                    l->ffn_res_query->abs_offset, l->ffn_res_query->bytes,
                                                    sc->cur, n_tok, D, n_blocks, 1, 1, 0,
                                                    state_stride, state_stride, 0xffffffffu), "ffn_mix");
                INK_CHK(p36_inkling2_control_vk(m, l, hp, sc, sc->ctrl_out[3], sc->cur, sc->msg_view[3], 3, n_tok), "ctrl_ffn");
                INK_CHK(p36_gpu_inkling2_rms_norm_q8k_tensor(sc->norm[0], NULL, sc->ctrl_out[3], m->map, m->size,
                                                             l->ffn_norm->abs_offset, D, n_tok, hp->rms_eps), "ffn_norm");
            }
        }

        /* MoE router + top-k + gate_up/SITU + down + ffn_post_conv + residual add */
        if (n_tok >= 16u && l->ffn_gate_inp->type == P36_TENSOR_F32 && (hp->n_expert % 64u) == 0u) {
            INK_CHK(p36_gpu_inkling2_f32_gemm_tensor(sc->router_logits, m->map, m->size,
                                                     l->ffn_gate_inp->abs_offset,
                                                     sc->norm[0], D, hp->n_expert, n_tok), "router_gemm");
        } else {
            INK_CHK(p36_gpu_tensor_matmul_scaled(m, l->ffn_gate_inp, sc->norm[0], sc->router_logits, D, hp->n_expert, n_tok, 1.0f), "router_mm");
        }
        INK_CHK(p36_gpu_inkling2_router_topk_tensor(sc->selected, sc->router_w, sc->router_logits,
                                                    m->map, m->size,
                                                    l->exp_probs_b ? l->exp_probs_b->abs_offset : 0,
                                                    l->exp_probs_b != NULL,
                                                    n_tok, hp->n_expert, hp->n_expert_used, hp->expert_weights_scale), "router_topk");
        const bool moe_rows = p36_gpu_inkling2_moe_rows(n_tok);
        INK_CHK(p36_gpu_inkling2_moe_ffn_tensor(moe_rows ? sc->cur : sc->moe_out,
                                                sc->moe_mid, sc->moe_down,
                                                m->map, m->size,
                                                l->ffn_gate_up_exps->abs_offset, l->ffn_gate_up_exps->type, l->ffn_gate_up_exps->bytes,
                                                l->ffn_gate_up_exps_scale ? l->ffn_gate_up_exps_scale->abs_offset : 0,
                                                l->ffn_gate_up_exps_scale != NULL,
                                                l->ffn_down_exps->abs_offset, l->ffn_down_exps->type, l->ffn_down_exps->bytes,
                                                l->ffn_down_exps_scale ? l->ffn_down_exps_scale->abs_offset : 0,
                                                l->ffn_down_exps_scale != NULL,
                                                l->ffn_post_conv->abs_offset, ink->conv_ffn_post[il], sc->ffn_inp, inv_sqrt_5,
                                                sc->norm[0], sc->selected, sc->router_w,
                                                n_tok, D, hp->n_ff_exp, D, hp->n_expert, hp->n_expert_used,
                                                hp->situ_beta, hp->situ_linear_beta), "moe_ffn");
        if (!moe_rows) {
            INK_CHK(p36_gpu_inkling2_conv_tensor(sc->cur, sc->moe_out, m->map, m->size,
                                                 l->ffn_post_conv->abs_offset, ink->conv_ffn_post[il],
                                                 NULL, sc->ffn_inp, D, n_tok, 0, 0, inv_sqrt_5), "conv_ffn");
        }
    }

    /* Final fused depth-residual mix + output_norm (+ Q8_K on decode!) */
    {
        uint32_t il = hp->n_logical_layer, ip = 0;
        p36_gpu_tensor *out_q8 = (n_tok == 1u && logits_out) ? sc->q8[0] : NULL;
        INK_CHK(p36_gpu_inkling2_mix_norm_tensor(sc->blocks_normed, sc->cur, m->map, m->size,
                                                 w->output_res_query->abs_offset, w->output_res_query->bytes,
                                                 w->output_norm->abs_offset,
                                                 sc->out_norm, out_q8, n_tok, D, n_blocks, 1,
                                                 state_stride, hp->rms_eps), "out_mix_norm");

        /* Final LM head projection + softcapping */
        if (logits_out) {
            p36_gpu_tensor *last = NULL;
            const p36_gpu_tensor *out_in = sc->out_norm;
            uint32_t rows = n_tok;
            if (!logits_all_rows && n_tok > 1u) {
                last = p36_gpu_tensor_view(sc->out_norm, (uint64_t)(n_tok - 1u) * D * sizeof(float), (uint64_t)D * sizeof(float));
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
                INK_CHK(p36_gpu_inkling2_softcap_tensor(logits_out, rows * hp->n_vocab_unpadded, hp->final_logit_softcapping), "softcap");
            }
        }
    }
    return true;
}
#endif
