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

