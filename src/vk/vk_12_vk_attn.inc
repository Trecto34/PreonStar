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

