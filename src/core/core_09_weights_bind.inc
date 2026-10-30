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
            l->ssm_conv1d_right = tensor_by_namef(m, "blk.%u.ssm_conv1d.weight_right", il);
            l->ssm_bd_logit_scale = tensor_by_namef(m, "blk.%u.ssm_bd_logit_scale.weight", il);
            l->ssm_bd_history_bias = tensor_by_namef(m, "blk.%u.ssm_bd_history.bias", il);
            l->ssm_bd_history_weight = tensor_by_namef(m, "blk.%u.ssm_bd_history.weight", il);
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
/* The nextn block reuses the target layer's dense shared-expert kernels, so it
 * may carry any quant those kernels dispatch (a Q4_K head block is normal). */
static void tensor_expect_dense_quant(const p36_tensor *t, uint32_t ndim,
                                      uint64_t d0, uint64_t d1) {
    if (!t) p36_die("internal error: missing tensor while validating MTP layout");
    if (t->type != P36_TENSOR_Q4_K && t->type != P36_TENSOR_Q5_K &&
        t->type != P36_TENSOR_Q6_K && t->type != P36_TENSOR_Q8_0) {
        fprintf(stderr, "p36: tensor %.*s has type %s, expected a dense quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, 0);
}

static void mtp_weights_validate_layout(const p36_mtp_weights *w) {
    const p36_layer_weights *l;
    if (!w) p36_die("internal error: missing MTP weights while validating layout");
    l = &w->block;
    /* The shared token table and projection are the target model's own
     * tensors, already validated with the target: any dense matrix type the
     * main model runs is valid here too. */
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

    tensor_expect_layout_or_q8_0(l->ffn_gate_inp, P36_TENSOR_F32, 2, P36_N_EMBD, P36_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_gate_inp_shexp, P36_TENSOR_F32, 1, P36_N_EMBD, 0, 0);
    tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
    tensor_expect_dense_quant(l->ffn_gate_shexp, 2, P36_N_EMBD, P36_N_FF_SHARED);
    tensor_expect_routed_gate_up(l->ffn_up_exps, 3, P36_N_EMBD, P36_N_FF_EXP, P36_N_EXPERT);
    tensor_expect_dense_quant(l->ffn_up_shexp, 2, P36_N_EMBD, P36_N_FF_SHARED);
    tensor_expect_routed_down(l->ffn_down_exps, 3, P36_N_FF_EXP, P36_N_EMBD, P36_N_EXPERT);
    tensor_expect_dense_quant(l->ffn_down_shexp, 2, P36_N_FF_SHARED, P36_N_EMBD);
    tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, P36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
}

/* `head` carries the shared token table and output projection: the draft
 * head embeds and projects with the target model's own tensors, and a
 * support GGUF's copies are dead weight (1 GiB of Q8_0 on a 16 GiB board).
 * Only the nextn block and its norms come from the support file. */
static void mtp_weights_bind(p36_mtp_weights *w, const p36_model *m,
                             const p36_model *head) {
    const uint32_t il = P36_N_LAYER;
    p36_layer_weights *l;
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(head, "token_embd.weight");
    w->output_norm = required_tensor(head, "output_norm.weight");
    w->output = required_tensor(head, "output.weight");
    w->output_scale = model_find_tensor(head, "output.scale");
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
     * upload (~1.2s spread over the first replies). The head only reads the
     * nextn block plus the token table: the output projection and its norm
     * come from the main model, so caching this file's copies would pin an
     * unused ~0.5 GiB on 16 GiB boards. */
    if (e->mtp_ready) {
        for (uint64_t i = 0; i < e->mtp_model.n_tensors; i++) {
            const p36_tensor *t = &e->mtp_model.tensors[i];
            if (t == e->mtp_weights.token_embd || t == e->mtp_weights.output ||
                t == e->mtp_weights.output_norm || t == e->mtp_weights.output_scale ||
                t->bytes == 0)
                continue;
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

