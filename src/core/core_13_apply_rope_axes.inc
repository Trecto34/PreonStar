static void p36_apply_rope_axes(float *x, uint32_t head_dim,
                                uint32_t t, uint32_t h, uint32_t w) {
    const uint32_t n_rot = P36_N_ROT;
    const uint32_t half = n_rot / 2u;
    const float base = 10000000.0f;
    const uint32_t sect_dims = P36_ROPE_SECTIONS[0] + P36_ROPE_SECTIONS[1] + P36_ROPE_SECTIONS[2] + P36_ROPE_SECTIONS[3];
    const float theta_scale = powf(base, -2.0f / (float)n_rot);
    float theta[4] = {(float)t, (float)h, (float)w, 0.0f};
    if (head_dim < n_rot) return;
    if (sect_dims == 0 || sect_dims > half) return;
    for (uint32_t pair = 0; pair < half; pair++) {
        uint32_t axis = 3;
        uint32_t sector = pair % sect_dims;
        if (sector % 3u == 1u && sector < 3u * P36_ROPE_SECTIONS[1]) axis = 1;
        else if (sector % 3u == 2u && sector < 3u * P36_ROPE_SECTIONS[2]) axis = 2;
        else if (sector % 3u == 0u && sector < 3u * P36_ROPE_SECTIONS[0]) axis = 0;
        {
            float c = cosf(theta[axis]);
            float s = sinf(theta[axis]);
            float x0 = x[pair];
            float x1 = x[pair + half];
            x[pair] = x0 * c - x1 * s;
            x[pair + half] = x0 * s + x1 * c;
        }
        for (uint32_t i = 0; i < 4; i++) theta[i] *= theta_scale;
    }
}

static void p36_apply_rope_one(float *x, uint32_t head_dim, uint32_t pos) {
    p36_apply_rope_axes(x, head_dim, pos, pos, pos);
}

static void p36_full_attn_cache_store(p36_full_attn_cache *cache, uint32_t pos, const float *k, const float *v) {
    uint8_t *dstk;
    uint8_t *dstv;
    if (!cache || pos >= cache->cap) return;
    dstk = cache->k + (uint64_t)pos * cache->k_row_bytes;
    dstv = cache->v + (uint64_t)pos * cache->v_row_bytes;
    p36_kv_cache_store_row(dstk, cache->type_k, k, P36_N_HEAD_KV * P36_N_HEAD_DIM);
    p36_kv_cache_store_row(dstv, cache->type_v, v, P36_N_HEAD_KV * P36_N_VALUE_DIM);
    if (cache->len <= pos) cache->len = pos + 1u;
}

static void p36_recurrent_conv_step(p36_recurrent_cache *cache, const float *cur, float *window) {
    uint32_t hist = P36_N_SSM_CONV - 1u;
    for (uint32_t i = 0; i < hist; i++) {
        memcpy(window + (uint64_t)i * P36_N_SSM_CONV_DIM,
               cache->conv + (uint64_t)i * P36_N_SSM_CONV_DIM,
               (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(window + (uint64_t)hist * P36_N_SSM_CONV_DIM, cur, (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
    if (hist > 1u) {
        memmove(cache->conv,
                cache->conv + P36_N_SSM_CONV_DIM,
                (size_t)(hist - 1u) * P36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(cache->conv + (uint64_t)(hist - 1u) * P36_N_SSM_CONV_DIM, cur, (size_t)P36_N_SSM_CONV_DIM * sizeof(float));
}

static void p36_ssm_conv_apply(const float *window, const float *kernel, float *out) {
    for (uint32_t c = 0; c < P36_N_SSM_CONV_DIM; c++) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < P36_N_SSM_CONV; t++) {
            acc += window[(uint64_t)t * P36_N_SSM_CONV_DIM + c] * kernel[(uint64_t)c * P36_N_SSM_CONV + t];
        }
        out[c] = acc;
    }
}

static void p36_delta_net_decode(float *state, const float *q, const float *k, const float *v,
                                 const float *g, const float *b, float *out) {
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        float *s = state + (uint64_t)h * P36_N_SSM_STATE * P36_N_SSM_STATE;
        const float *qh = q + (uint64_t)h * P36_N_SSM_STATE;
        const float *kh = k + (uint64_t)h * P36_N_SSM_STATE;
        const float *vh = v + (uint64_t)h * P36_N_SSM_STATE;
        float decay = expf(g[h]);
        float beta = b[h];
        float sk[P36_N_SSM_STATE];
        float d[P36_N_SSM_STATE];
        for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) {
            for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) s[(uint64_t)i * P36_N_SSM_STATE + j] *= decay;
        }
        for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * P36_N_SSM_STATE + j] * (double)kh[i];
            sk[j] = (float)acc;
            d[j] = (vh[j] - sk[j]) * beta;
        }
        for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) {
            float ki = kh[i];
            for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) s[(uint64_t)i * P36_N_SSM_STATE + j] += ki * d[j];
        }
        for (uint32_t j = 0; j < P36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < P36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * P36_N_SSM_STATE + j] * (double)qh[i];
            out[(uint64_t)h * P36_N_SSM_STATE + j] = (float)(acc / sqrt((double)P36_N_SSM_STATE));
        }
    }
}

static bool p36_forward_ffn(const p36_engine *e, const p36_layer_weights *l, const float *inp, float *out,
                            p36_cpu_runtime *rt) {
    float gate_logits[P36_N_EXPERT];
    uint32_t top_idx[P36_N_EXPERT_USED];
    float top_val[P36_N_EXPERT_USED];
    float top_w[P36_N_EXPERT_USED];
    uint8_t inpq[P36_MAX_Q8_K_BYTES];
    const uint8_t *xq;
    p36_activation_quant_kind inpq_kind;
    float *gate = rt->work2;
    float *up = rt->work3;
    float *mid = rt->work4;
    float *rowbuf = rt->work5;
    float shared_gate;
    if (P36_MODEL_DENSE) {
        const p36_tensor *rot_gate = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ? l->ffn_gate_shexp :
                                     (p36_hadamard_rotate_enabled(l->ffn_up_shexp) ? l->ffn_up_shexp : NULL);
        const float *ffn_in = rot_gate ? p36_hadamard_forward_rows(rot_gate, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
        const float *down_in;
        inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (!p36_quantize_activation_row(inpq_kind, ffn_in, inpq, P36_N_EMBD)) return false;
        xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
        if (!p36_tensor_matvec_pair_prequant(e,
                                             l->ffn_gate_shexp,
                                             l->ffn_up_shexp,
                                             ffn_in, xq, gate, up, rowbuf,
                                             P36_N_EMBD, P36_N_FF_SHARED,
                                             1.0f, 1.0f)) {
            if (!p36_tensor_matvec_prequant(
                    e, l->ffn_gate_shexp, ffn_in, xq, gate, rowbuf,
                    P36_N_EMBD, P36_N_FF_SHARED) ||
                !p36_tensor_matvec_prequant(
                    e, l->ffn_up_shexp, ffn_in, xq, up, rowbuf,
                    P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
        }
        for (uint32_t j = 0; j < P36_N_FF_SHARED; j++)
            mid[j] = p36_siluf(gate[j]) * up[j];
        down_in = p36_hadamard_forward_rows(l->ffn_down_shexp, mid, rt->batch_had, P36_N_FF_SHARED, 1u);
        return p36_tensor_matvec(e, l->ffn_down_shexp, down_in, out, rowbuf,
                                 P36_N_FF_SHARED, P36_N_EMBD);
    }
    if (!p36_tensor_matvec(e, l->ffn_gate_inp, inp, gate_logits, rowbuf, P36_N_EMBD, P36_N_EXPERT)) return false;
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!p36_quantize_activation_row(inpq_kind, inp, inpq, P36_N_EMBD)) return false;
    xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
    p36_softmax_inplace(gate_logits, P36_N_EXPERT);
    p36_topk_indices(gate_logits, P36_N_EXPERT, P36_N_EXPERT_USED, top_idx, top_val);
    {
        double sum = 0.0;
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) sum += top_val[i];
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * e->expert_weights_scale;
    }
    memset(out, 0, (size_t)P36_N_EMBD * sizeof(float));
    for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) {
        float gate_scale = p36_tensor_index_or(&e->model, l->ffn_gate_exps_scale, top_idx[i], 1.0f);
        float up_scale = p36_tensor_index_or(&e->model, l->ffn_up_exps_scale, top_idx[i], 1.0f);
        if (!p36_tensor_expert_matvec_pair_prequant(e,
                                                    l->ffn_gate_exps,
                                                    l->ffn_up_exps,
                                                    top_idx[i],
                                                    inp,
                                                    xq,
                                                    gate,
                                                    up,
                                                    rowbuf,
                                                    P36_N_EMBD,
                                                    P36_N_FF_EXP,
                                                    gate_scale,
                                                    up_scale)) {
            if (!p36_tensor_expert_matvec_prequant(e, l->ffn_gate_exps, top_idx[i], inp, xq,
                                                   gate, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                return false;
            }
            p36_scale_inplace(gate, P36_N_FF_EXP, gate_scale);
            if (!p36_tensor_expert_matvec_prequant(e, l->ffn_up_exps, top_idx[i], inp, xq,
                                                   up, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                return false;
            }
            p36_scale_inplace(up, P36_N_FF_EXP, up_scale);
        }
        for (uint32_t j = 0; j < P36_N_FF_EXP; j++) mid[j] = p36_siluf(gate[j]) * up[j];
        if (!p36_tensor_expert_matvec(e, l->ffn_down_exps, top_idx[i], mid, gate, rowbuf, P36_N_FF_EXP, P36_N_EMBD)) return false;
        p36_scale_inplace(gate, P36_N_EMBD,
                          p36_tensor_index_or(&e->model, l->ffn_down_exps_scale, top_idx[i], 1.0f));
        for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += gate[j] * top_w[i];
    }
    {
        float gate_scale = p36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = p36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        p36_activation_quant_kind shared_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!p36_quantize_activation_row(shared_kind, inp, inpq, P36_N_EMBD)) return false;
            xq = shared_kind == P36_ACTIVATION_QUANT_NONE ? NULL : inpq;
            inpq_kind = shared_kind;
        }
        if (!p36_tensor_matvec_pair_prequant(e,
                                             l->ffn_gate_shexp,
                                             l->ffn_up_shexp,
                                             inp,
                                             xq,
                                             gate,
                                             up,
                                             rowbuf,
                                             P36_N_EMBD,
                                             P36_N_FF_SHARED,
                                             gate_scale,
                                             up_scale)) {
            if (!p36_tensor_matvec_prequant(e, l->ffn_gate_shexp, inp, xq,
                                            gate, rowbuf, P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
            p36_scale_inplace(gate, P36_N_FF_SHARED, gate_scale);
            if (!p36_tensor_matvec_prequant(e, l->ffn_up_shexp, inp, xq,
                                            up, rowbuf, P36_N_EMBD, P36_N_FF_SHARED)) {
                return false;
            }
            p36_scale_inplace(up, P36_N_FF_SHARED, up_scale);
        }
    }
    for (uint32_t j = 0; j < P36_N_FF_SHARED; j++)
        mid[j] = gate[j] / (1.0f + expf(-gate[j])) * up[j];
    if (!p36_tensor_matvec(e, l->ffn_down_shexp, mid, gate, rowbuf, P36_N_FF_SHARED, P36_N_EMBD)) return false;
    p36_scale_inplace(gate, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f));
    if (!p36_tensor_get_plain(&e->model, l->ffn_gate_inp_shexp, rowbuf, P36_N_EMBD)) return false;
    {
        double acc = 0.0;
        for (uint32_t i = 0; i < P36_N_EMBD; i++) acc += (double)rowbuf[i] * (double)inp[i];
        up[0] = (float)acc;
    }
    shared_gate = p36_sigmoidf(up[0]);
    for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += gate[j] * shared_gate;
    return true;
}

static bool p36_forward_full_attn(const p36_engine *e, const p36_layer_weights *l, uint32_t il, uint32_t pos,
                                  const float *inp, float *out, p36_cpu_runtime *rt) {
    float *qg = rt->work5;
    float *q = rt->work0;
    float *k = rt->work1;
    float *v = rt->work2;
    float *acc = rt->work3;
    float sinks[P36_N_HEAD];
    p36_full_attn_cache *cache = &rt->full[il];
    bool have_sinks = false;
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_q) ? l->attn_q :
                                (p36_hadamard_rotate_enabled(l->attn_k) ? l->attn_k :
                                (p36_hadamard_rotate_enabled(l->attn_v) ? l->attn_v : NULL));
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
    if (!p36_tensor_matvec(e, l->attn_q, qkv_in, qg, rt->work4, P36_N_EMBD, P36_N_HEAD * P36_N_HEAD_DIM * 2u)) {
        fprintf(stderr, "p36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(qg, P36_N_HEAD * P36_N_HEAD_DIM * 2u,
                      p36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_k, qkv_in, k, rt->work4, P36_N_EMBD, P36_N_HEAD_KV * P36_N_HEAD_DIM)) {
        fprintf(stderr, "p36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(k, P36_N_HEAD_KV * P36_N_HEAD_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_v, qkv_in, v, rt->work4, P36_N_EMBD, P36_N_HEAD_KV * P36_N_VALUE_DIM)) {
        fprintf(stderr, "p36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    p36_scale_inplace(v, P36_N_HEAD_KV * P36_N_VALUE_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f));
    for (uint32_t h = 0; h < P36_N_HEAD; h++) {
        p36_ref_rms_norm(q + (uint64_t)h * P36_N_HEAD_DIM,
                         qg + (uint64_t)h * P36_N_HEAD_DIM * 2u,
                         (const float *)(e->model.map + l->attn_q_norm->abs_offset),
                         P36_N_HEAD_DIM,
                         P36_RMS_EPS);
        p36_apply_rope_one(q + (uint64_t)h * P36_N_HEAD_DIM, P36_N_HEAD_DIM, pos);
    }
    for (uint32_t h = 0; h < P36_N_HEAD_KV; h++) {
        p36_ref_rms_norm(k + (uint64_t)h * P36_N_HEAD_DIM,
                         k + (uint64_t)h * P36_N_HEAD_DIM,
                         (const float *)(e->model.map + l->attn_k_norm->abs_offset),
                         P36_N_HEAD_DIM,
                         P36_RMS_EPS);
        p36_apply_rope_one(k + (uint64_t)h * P36_N_HEAD_DIM, P36_N_HEAD_DIM, pos);
    }
    if (l->attn_sinks) have_sinks = p36_tensor_get_plain(&e->model, l->attn_sinks, sinks, P36_N_HEAD);
    p36_full_attn_cache_store(cache, pos, k, v);
    for (uint32_t h = 0; h < P36_N_HEAD; h++) {
        uint32_t kvh = h / (P36_N_HEAD / P36_N_HEAD_KV);
        float *head_out = acc + (uint64_t)h * P36_N_VALUE_DIM;
        float max_score = have_sinks ? sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)P36_N_VALUE_DIM * sizeof(float));
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *kc = cache->k + (uint64_t)t * cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(cache->type_k, P36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t i = 0; i < P36_N_HEAD_DIM; i++) dot += (double)q[(uint64_t)h * P36_N_HEAD_DIM + i] * (double)p36_kv_cache_at(kc, cache->type_k, i);
            rt->scores[t] = (float)(dot / sqrt((double)P36_N_HEAD_DIM));
            if (rt->scores[t] > max_score) max_score = rt->scores[t];
        }
        if (have_sinks) denom = exp((double)sinks[h] - (double)max_score);
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *vc = cache->v + (uint64_t)t * cache->v_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(cache->type_v, P36_N_VALUE_DIM);
            float w = expf(rt->scores[t] - max_score);
            denom += w;
            for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) head_out[i] += w * p36_kv_cache_at(vc, cache->type_v, i);
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) head_out[i] *= inv;
        }
        for (uint32_t i = 0; i < P36_N_VALUE_DIM; i++) {
            uint64_t gate_off = (uint64_t)h * P36_N_HEAD_DIM * 2u + P36_N_HEAD_DIM + i;
            head_out[i] *= p36_sigmoidf(qg[gate_off]);
        }
    }
    {
        const float *ao_in = p36_hadamard_forward_rows(l->attn_output, acc, rt->batch_had, P36_N_SSM_INNER, 1u);
        if (!p36_tensor_matvec(e, l->attn_output, ao_in, out, rt->work4, P36_N_SSM_INNER, P36_N_EMBD)) {
            fprintf(stderr, "p36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos);
            return false;
        }
    }
    p36_scale_inplace(out, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
    return true;
}

static bool p36_forward_recurrent(const p36_engine *e, const p36_layer_weights *l, uint32_t il,
                                  const float *inp, float *out, p36_cpu_runtime *rt) {
    p36_recurrent_cache *cache = &rt->recurrent[il];
    float *qkv = rt->work5;
    float *window = rt->work4;
    float *conv = rt->work3;
    float *z = rt->work2;
    float *beta = rt->work1;
    float *q = rt->work5;
    float *k = rt->work4;
    float *v = rt->work0;
    float *proj = rt->work3;
    float *gate = rt->work1 + P36_N_SSM_DT_RANK;
    const float *kernel = (const float *)(e->model.map + l->ssm_conv1d->abs_offset);
    const p36_tensor *rot_qkv = p36_hadamard_rotate_enabled(l->attn_qkv) ? l->attn_qkv :
                                (p36_hadamard_rotate_enabled(l->attn_gate) ? l->attn_gate : NULL);
    const float *qkv_in = rot_qkv ? p36_hadamard_forward_rows(rot_qkv, inp, rt->batch_had, P36_N_EMBD, 1u) : inp;
    if (!p36_tensor_matvec(e, l->attn_qkv, qkv_in, qkv, z, P36_N_EMBD, P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent attn_qkv failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(qkv, P36_N_SSM_CONV_DIM,
                      p36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f));
    p36_recurrent_conv_step(cache, qkv, window);
    p36_ssm_conv_apply(window, kernel, conv);
    for (uint32_t i = 0; i < P36_N_SSM_CONV_DIM; i++) conv[i] = p36_siluf(conv[i]);
    if (!p36_all_finite(conv, P36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "p36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_tensor_matvec(e, l->ssm_beta, inp, beta, z, P36_N_EMBD, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent ssm_beta failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(beta, P36_N_SSM_DT_RANK,
                      p36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->ssm_alpha, inp, gate, z, P36_N_EMBD, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent ssm_alpha failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(gate, P36_N_SSM_DT_RANK,
                      p36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    if (!p36_tensor_matvec(e, l->attn_gate, qkv_in, z, window, P36_N_EMBD, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent attn_gate failed at layer=%u\n", il);
        return false;
    }
    p36_scale_inplace(z, P36_N_SSM_INNER,
                      p36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
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
        gate[h] = p36_softplusf(gate[h] + ((const float *)(e->model.map + l->ssm_dt->abs_offset))[h]) * ((const float *)(e->model.map + l->ssm_a->abs_offset))[h];
    }
    if (!p36_all_finite(beta, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent beta non-finite at layer=%u\n", il);
        return false;
    }
    if (!p36_all_finite(gate, P36_N_SSM_DT_RANK)) {
        fprintf(stderr, "p36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    p36_delta_net_decode(cache->state, q, k, v, gate, beta, proj);
    if (!p36_all_finite(proj, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    for (uint32_t h = 0; h < P36_N_SSM_DT_RANK; h++) {
        p36_ref_rms_norm(proj + (uint64_t)h * P36_N_SSM_STATE,
                         proj + (uint64_t)h * P36_N_SSM_STATE,
                         (const float *)(e->model.map + l->ssm_norm->abs_offset),
                         P36_N_SSM_STATE,
                         P36_RMS_EPS);
    }
    for (uint32_t i = 0; i < P36_N_SSM_INNER; i++) proj[i] *= p36_siluf(z[i]);
    if (!p36_all_finite(proj, P36_N_SSM_INNER)) {
        fprintf(stderr, "p36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    {
        const float *ssm_in = p36_hadamard_forward_rows(l->ssm_out, proj, rt->batch_had, P36_N_SSM_INNER, 1u);
        if (!p36_tensor_matvec(e, l->ssm_out, ssm_in, out, q, P36_N_SSM_INNER, P36_N_EMBD)) {
            fprintf(stderr, "p36: recurrent ssm_out failed at layer=%u\n", il);
            return false;
        }
    }
    p36_scale_inplace(out, P36_N_EMBD,
                      p36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
    return true;
}

typedef struct {
    float *q;
    const float *qg;
    const float *weight;
    uint32_t pos0;
} p36_full_attn_q_rows_ctx;

typedef struct {
    float *k;
    const float *weight;
    uint32_t pos0;
} p36_full_attn_k_rows_ctx;

typedef struct {
    float *out;
    const float *q;
    const float *qg;
    const p36_full_attn_cache *cache;
    const float *sinks;
    uint32_t pos0;
    bool have_sinks;
} p36_full_attn_scores_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_layer_weights *l;
    const float *inp;
    float *gate_logits;
    const uint8_t *xq;
    uint32_t xq_row_bytes;
} p36_ffn_routed_batch_ctx;

typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} p36_ffn_combine_batch_ctx;

static void p36_full_attn_q_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_q_rows_ctx *ctx = (p36_full_attn_q_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD);
        const float *src = ctx->qg + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u;
        float *dst = ctx->q + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        p36_ref_rms_norm(dst, src, ctx->weight, P36_N_HEAD_DIM, P36_RMS_EPS);
        p36_apply_rope_one(dst, P36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void p36_full_attn_k_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_k_rows_ctx *ctx = (p36_full_attn_k_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD_KV);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD_KV);
        float *dst = ctx->k + (uint64_t)t * P36_N_HEAD_KV * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        p36_ref_rms_norm(dst, dst, ctx->weight, P36_N_HEAD_DIM, P36_RMS_EPS);
        p36_apply_rope_one(dst, P36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void p36_full_attn_scores_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_full_attn_scores_ctx *ctx = (p36_full_attn_scores_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / P36_N_HEAD);
        uint32_t h = (uint32_t)(idx % P36_N_HEAD);
        uint32_t kvh = h / (P36_N_HEAD / P36_N_HEAD_KV);
        uint32_t pos = ctx->pos0 + t;
        const float *qh = ctx->q + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM + (uint64_t)h * P36_N_HEAD_DIM;
        const float *gate = ctx->qg + (uint64_t)t * P36_N_HEAD * P36_N_HEAD_DIM * 2u + (uint64_t)h * P36_N_HEAD_DIM * 2u + P36_N_HEAD_DIM;
        float *head_out = ctx->out + ((uint64_t)t * P36_N_HEAD + h) * P36_N_VALUE_DIM;
        float max_score = ctx->have_sinks ? ctx->sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)P36_N_VALUE_DIM * sizeof(float));
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_k, P36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < P36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)p36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float score = (float)(dot / sqrt((double)P36_N_HEAD_DIM));
                if (score > max_score) max_score = score;
            }
        }
        if (ctx->have_sinks) denom = exp((double)ctx->sinks[h] - (double)max_score);
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_k, P36_N_HEAD_DIM);
            const uint8_t *vc = ctx->cache->v + (uint64_t)i * ctx->cache->v_row_bytes + (uint64_t)kvh * p36_kv_cache_row_bytes(ctx->cache->type_v, P36_N_VALUE_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < P36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)p36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float w = expf((float)(dot / sqrt((double)P36_N_HEAD_DIM)) - max_score);
                denom += w;
                for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] += w * p36_kv_cache_at(vc, ctx->cache->type_v, j);
            }
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] *= inv;
        }
        for (uint32_t j = 0; j < P36_N_VALUE_DIM; j++) head_out[j] *= p36_sigmoidf(gate[j]);
    }
}

static void p36_ffn_routed_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_ffn_routed_batch_ctx *ctx = (p36_ffn_routed_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float top_val[P36_N_EXPERT_USED];
        float top_w[P36_N_EXPERT_USED];
        uint32_t top_idx[P36_N_EXPERT_USED];
        float gate[P36_N_FF_EXP];
        float up[P36_N_FF_EXP];
        float mid[P36_N_FF_EXP];
        float down[P36_N_EMBD];
        float rowbuf[P36_N_SSM_INNER];
        float *logits = ctx->gate_logits + t * P36_N_EXPERT;
        float *out = ctx->out + t * P36_N_EMBD;
        const float *inp = ctx->inp + t * P36_N_EMBD;
        const uint8_t *inpq = ctx->xq ? ctx->xq + (uint64_t)t * ctx->xq_row_bytes : NULL;
        p36_softmax_inplace(logits, P36_N_EXPERT);
        p36_topk_indices(logits, P36_N_EXPERT, P36_N_EXPERT_USED, top_idx, top_val);
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) sum += top_val[i];
            if (sum < 6.103515625e-5) sum = 6.103515625e-5;
            for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * ctx->e->expert_weights_scale;
        }
        memset(out, 0, (size_t)P36_N_EMBD * sizeof(float));
        for (uint32_t i = 0; i < P36_N_EXPERT_USED; i++) {
            float gate_scale = p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_gate_exps_scale, top_idx[i], 1.0f);
            float up_scale = p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_up_exps_scale, top_idx[i], 1.0f);
            if (!p36_tensor_expert_matvec_pair_prequant(ctx->e,
                                                        ctx->l->ffn_gate_exps,
                                                        ctx->l->ffn_up_exps,
                                                        top_idx[i],
                                                        inp,
                                                        inpq,
                                                        gate,
                                                        up,
                                                        rowbuf,
                                                        P36_N_EMBD,
                                                        P36_N_FF_EXP,
                                                        gate_scale,
                                                        up_scale)) {
                if (!p36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_gate_exps, top_idx[i], inp, inpq,
                                                       gate, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                    continue;
                }
                p36_scale_inplace(gate, P36_N_FF_EXP, gate_scale);
                if (!p36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_up_exps, top_idx[i], inp, inpq,
                                                       up, rowbuf, P36_N_EMBD, P36_N_FF_EXP)) {
                    continue;
                }
                p36_scale_inplace(up, P36_N_FF_EXP, up_scale);
            }
            for (uint32_t j = 0; j < P36_N_FF_EXP; j++) mid[j] = p36_siluf(gate[j]) * up[j];
            if (!p36_tensor_expert_matvec(ctx->e, ctx->l->ffn_down_exps, top_idx[i], mid, down, rowbuf, P36_N_FF_EXP, P36_N_EMBD)) continue;
            p36_scale_inplace(down, P36_N_EMBD,
                              p36_tensor_index_or(&ctx->e->model, ctx->l->ffn_down_exps_scale, top_idx[i], 1.0f));
            for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += down[j] * top_w[i];
        }
    }
}

static void p36_ffn_combine_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_ffn_combine_batch_ctx *ctx = (p36_ffn_combine_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float g = p36_sigmoidf(ctx->scalar[t]);
        float *out = ctx->out + t * P36_N_EMBD;
        const float *shared = ctx->shared + t * P36_N_EMBD;
        for (uint32_t j = 0; j < P36_N_EMBD; j++) out[j] += shared[j] * g;
    }
}

static bool p36_forward_ffn_batch(const p36_engine *e,
                                  const p36_layer_weights *l,
                                  const float *inp,
                                  float *out,
                                  uint32_t n_tok,
                                  p36_cpu_runtime *rt) {
    p36_ffn_routed_batch_ctx routed_ctx;
    p36_ffn_combine_batch_ctx combine_ctx;
    const float *shared_gate_inp;
    const uint8_t *gate_inp_xq = NULL;
    p36_activation_quant_kind inpq_kind;
    if (P36_MODEL_DENSE) {
        const p36_tensor *rot_gate = p36_hadamard_rotate_enabled(l->ffn_gate_shexp) ? l->ffn_gate_shexp :
                                     (p36_hadamard_rotate_enabled(l->ffn_up_shexp) ? l->ffn_up_shexp : NULL);
        const float *ffn_in = rot_gate ? p36_hadamard_forward_rows(rot_gate, inp, rt->batch_had, P36_N_EMBD, n_tok) : inp;
        const float *down_in;
        inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (!p36_quantize_activation_batch(inpq_kind, ffn_in, rt->batch_xq,
                                           n_tok, P36_N_EMBD, e->n_threads)) return false;
        gate_inp_xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : rt->batch_xq;
        if (!p36_tensor_matmul_pair_batch_prequant(
                e, l->ffn_gate_shexp, l->ffn_up_shexp,
                ffn_in, gate_inp_xq,
                rt->batch_ffn_shared_gate, rt->batch_ffn_shared_up,
                n_tok, P36_N_EMBD, P36_N_FF_SHARED, 1.0f, 1.0f)) {
            if (!p36_tensor_matmul_batch_prequant(
                    e, l->ffn_gate_shexp, ffn_in, gate_inp_xq,
                    rt->batch_ffn_shared_gate, n_tok,
                    P36_N_EMBD, P36_N_FF_SHARED, 1.0f) ||
                !p36_tensor_matmul_batch_prequant(
                    e, l->ffn_up_shexp, ffn_in, gate_inp_xq,
                    rt->batch_ffn_shared_up, n_tok,
                    P36_N_EMBD, P36_N_FF_SHARED, 1.0f)) {
                return false;
            }
        }
        p36_swiglu_rows(rt->batch_ffn_shared_mid,
                        rt->batch_ffn_shared_gate,
                        rt->batch_ffn_shared_up,
                        n_tok, P36_N_FF_SHARED, e->n_threads);
        down_in = p36_hadamard_forward_rows(l->ffn_down_shexp, rt->batch_ffn_shared_mid,
                                            rt->batch_had, P36_N_FF_SHARED, n_tok);
        if (!p36_quantize_activation_batch_for_type(
                l->ffn_down_shexp->type, down_in,
                rt->batch_xq, n_tok, P36_N_FF_SHARED, e->n_threads)) {
            return false;
        }
        return p36_tensor_matmul_batch_prequant(
                e, l->ffn_down_shexp, down_in,
                rt->batch_xq, out, n_tok,
                P36_N_FF_SHARED, P36_N_EMBD, 1.0f);
    }
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_inp->type);
    if (inpq_kind != P36_ACTIVATION_QUANT_NONE) {
        if (!p36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
        gate_inp_xq = rt->batch_xq;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ffn_gate_inp, inp, gate_inp_xq,
                                          rt->batch_ffn_gate_logits,
                                          n_tok, P36_N_EMBD, P36_N_EXPERT, 1.0f)) return false;
    routed_ctx.out = out;
    routed_ctx.e = e;
    routed_ctx.l = l;
    routed_ctx.inp = inp;
    routed_ctx.gate_logits = rt->batch_ffn_gate_logits;
    inpq_kind = p36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!p36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
    routed_ctx.xq = inpq_kind == P36_ACTIVATION_QUANT_NONE ? NULL : rt->batch_xq;
    routed_ctx.xq_row_bytes = p36_activation_quant_row_bytes(inpq_kind, P36_N_EMBD);
    p36_parallel_for_rows(n_tok, 2, e->n_threads, p36_ffn_routed_batch_worker, &routed_ctx);

    {
        float gate_scale = p36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = p36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        p36_activation_quant_kind shared_kind = p36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!p36_quantize_activation_batch(shared_kind, inp, rt->batch_xq, n_tok, P36_N_EMBD, e->n_threads)) return false;
            inpq_kind = shared_kind;
        }
        if (!p36_tensor_matmul_pair_batch_prequant(e,
                                                   l->ffn_gate_shexp,
                                                   l->ffn_up_shexp,
                                                   inp,
                                                   rt->batch_xq,
                                                   rt->batch_ffn_shared_gate,
                                                   rt->batch_ffn_shared_up,
                                                   n_tok,
                                                   P36_N_EMBD,
                                                   P36_N_FF_SHARED,
                                                   gate_scale,
                                                   up_scale)) {
            if (!p36_tensor_matmul_batch_prequant(e, l->ffn_gate_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_gate,
                                                  n_tok, P36_N_EMBD, P36_N_FF_SHARED,
                                                  gate_scale)) {
                return false;
            }
            if (!p36_tensor_matmul_batch_prequant(e, l->ffn_up_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_up,
                                                  n_tok, P36_N_EMBD, P36_N_FF_SHARED,
                                                  up_scale)) {
                return false;
            }
        }
    }
    p36_swiglu_rows(rt->batch_ffn_shared_mid,
                    rt->batch_ffn_shared_gate,
                    rt->batch_ffn_shared_up,
                    n_tok,
                    P36_N_FF_SHARED,
                    e->n_threads);

    if (!p36_quantize_activation_batch_for_type(l->ffn_down_shexp->type, rt->batch_ffn_shared_mid,
                                               rt->batch_xq, n_tok, P36_N_FF_SHARED, e->n_threads)) {
        return false;
    }
    if (!p36_tensor_matmul_batch_prequant(e, l->ffn_down_shexp, rt->batch_ffn_shared_mid, rt->batch_xq,
                                          rt->batch_ffn_shared_out,
                                          n_tok, P36_N_FF_SHARED, P36_N_EMBD,
                                          p36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }

    shared_gate_inp = (const float *)(e->model.map + l->ffn_gate_inp_shexp->abs_offset);
    p36_tensor_vector_dot_batch(rt->batch_ffn_scalar, shared_gate_inp, inp, n_tok, P36_N_EMBD, e->n_threads);
    combine_ctx.out = out;
    combine_ctx.shared = rt->batch_ffn_shared_out;
    combine_ctx.scalar = rt->batch_ffn_scalar;
    p36_parallel_for_rows(n_tok, 2, e->n_threads, p36_ffn_combine_batch_worker, &combine_ctx);
    return true;
}

