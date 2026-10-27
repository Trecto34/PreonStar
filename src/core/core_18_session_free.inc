void p36_session_free(p36_session *s) {
    if (!s) return;
    s->engine->session_bytes -= s->reserved_context_bytes;
    s->engine->live_sessions--;
    p36_tokens_free(&s->checkpoint);
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef P36_NO_GPU
        p36_vulkan_runtime_free((p36_vulkan_runtime *)s->runtime);
#endif
    } else {
        p36_cpu_runtime_free((p36_cpu_runtime *)s->runtime);
    }
    free(s->logits);
    free(s->sample_probs);
    free(s->mtp_logits);
    free(s->vision_spans);
    free(s);
}

void p36_session_set_progress(p36_session *s, p36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

int p36_session_power(p36_session *s) {
    return s ? p36_engine_power(s->engine) : 100;
}

static bool p36_session_reserve_kv(p36_session *s, uint32_t need,
                                   uint32_t rows) {
    if (!s || need > (uint32_t)s->ctx_size) return false;
#ifdef P36_NO_GPU
    (void)rows;
#endif
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifdef P36_NO_GPU
        return false;
#else
        return p36_vulkan_runtime_reserve_kv(
            (p36_vulkan_runtime *)s->runtime, need,
            (uint32_t)s->ctx_size, rows);
#endif
    }
    return p36_cpu_runtime_reserve_kv(
        (p36_cpu_runtime *)s->runtime, need, (uint32_t)s->ctx_size);
}

int p36_session_set_power(p36_session *s, int power_percent) {
    return s ? p36_engine_set_power(s->engine, power_percent) : 1;
}

float p36_session_directional_steering_ffn(p36_session *s) {
    return s && s->engine ? s->engine->directional_steering_ffn_scale : 0.0f;
}

int p36_session_set_directional_steering_ffn(p36_session *s, float scale) {
    if (!s || !s->engine || !isfinite(scale) || scale < -100.0f || scale > 100.0f)
        return 1;
    bool loaded = s->engine->directional_steering_dirs != NULL;
#ifndef P36_NO_GPU
    loaded = loaded || s->engine->directional_steering_gpu != NULL;
#endif
    if (scale != 0.0f && !loaded) {
        fprintf(stderr, "p36: live FFN steering needs a vector loaded at startup\n");
        return 1;
    }
    s->engine->directional_steering_ffn_scale = scale;
    s->mtp_draft_valid = false;
    return 0;
}

void p36_session_set_display_progress(p36_session *s, p36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void p36_session_set_cancel(p36_session *s, p36_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

/* Evaluate prompt tokens [start, len) on top of the current checkpoint
 * (start must equal s->checkpoint.len), leaving the last token's logits in
 * s->logits.  The Vulkan runtime walks the range in prefill-sized batches;
 * the CPU path uses smaller layer-major chunks that keep recurrent state
 * updates ordered while batching projections and FFN work. */
static int p36_session_prefill_range(p36_session *s, const p36_tokens *prompt, int start,
                                     char *err, size_t errlen) {
    int i = start;
#ifndef P36_NO_GPU
    if (start == 0 && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_gpu_stream_expert_cache_reset_route_hotness();
    }
#endif
    while (i < prompt->len) {
        if (s->cancel && s->cancel(s->cancel_ud)) {
            if (err && errlen) snprintf(err, errlen, "interrupted");
            return P36_SESSION_SYNC_INTERRUPTED;
        }
        int n = prompt->len - i;
        bool compute_logits;
        bool ok;
#ifndef P36_NO_GPU
        if (p36_engine_uses_vulkan_runtime(s->engine)) {
            p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
        } else
#endif
        {
            p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
        }
        if (!p36_session_reserve_kv(s, (uint32_t)(i + n),
                                    (uint32_t)s->checkpoint.len)) {
            if (err && errlen) snprintf(err, errlen, "failed to grow KV cache");
            return 1;
        }
        compute_logits = (i + n) == prompt->len;
#ifndef P36_NO_GPU
        if (p36_engine_uses_vulkan_runtime(s->engine)) {
            ok = p36_forward_tokens_vulkan(s, prompt->v + i, (uint32_t)n,
                                           (uint32_t)i, compute_logits);
        } else
#endif
        {
            p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
            ok = !p36_engine_uses_vulkan_runtime(s->engine) &&
                 (n == 1 && rt && rt->prefill_cap == 1
                  ? p36_forward_token_cpu(s, prompt->v[i], (uint32_t)i, compute_logits)
                  : p36_forward_tokens_cpu(s, prompt->v + i, (uint32_t)n, (uint32_t)i, compute_logits));
        }
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "%s prefill failed at token %d",
                                         p36_backend_name(s->engine->backend), i);
            s->checkpoint_valid = false;
            return 1;
        }
        for (int j = 0; j < n; j++) p36_tokens_push(&s->checkpoint, prompt->v[i + j]);
        s->checkpoint_valid = true;
        i += n;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_chunk", i, prompt->len);
    }
    return 0;
}

/* Image pads alone cannot identify the image that conditioned the live state. */
const p36_vision_span *p36_session_vision_spans(const p36_session *s, size_t *count) {
    if (count) *count = s && s->checkpoint_valid ? s->vision_count : 0;
    return s && s->checkpoint_valid ? s->vision_spans : NULL;
}

bool p36_session_vision_prefix_matches(const p36_session *s,
                                       const p36_vision_span *images, size_t count) {
    if (!s || !s->checkpoint_valid || (count && !images)) return false;
    if (count < s->vision_count) return false;
    for (size_t i = 0; i < s->vision_count; i++) {
        const p36_vision_span *a = &s->vision_spans[i], *b = &images[i];
        if (a->token_start != b->token_start ||
            a->embedding.token_count != b->embedding.token_count ||
            a->embedding.grid_width != b->embedding.grid_width ||
            a->embedding.grid_height != b->embedding.grid_height ||
            memcmp(a->embedding.fingerprint, b->embedding.fingerprint, 32)) return false;
    }
    if (s->vision_state && !s->vision_count) return false;
    for (size_t i = s->vision_count; i < count; i++) {
        if (!images[i].token_start || images[i].token_start - 1u < (uint32_t)s->checkpoint.len)
            return false;
    }
    return true;
}

int p36_session_sync(p36_session *s, const p36_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt) {
        if (err && errlen) snprintf(err, errlen, "invalid sync arguments");
        return 1;
    }
    if (prompt->len <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (!s->vision_state && s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        p36_tokens_starts_with(prompt, &s->checkpoint))
    {
        return p36_session_prefill_range(s, prompt, s->checkpoint.len, err, errlen);
    }
    p36_session_reset_runtime(s);
    s->vision_state = false;
    free(s->vision_spans);
    s->vision_spans = NULL;
    s->vision_count = 0;
    s->rope_delta = 0;
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    return p36_session_prefill_range(s, prompt, 0, err, errlen);
}

int p36_session_sync_vision(p36_session *s, const p36_tokens *prompt,
                            const p36_vision_span *images, size_t image_count,
                            char *err, size_t errlen) {
    if (!s || !prompt || !images || image_count == 0 ||
        !p36_engine_has_vision(s->engine)) {
        if (err && errlen) snprintf(err, errlen, "invalid vision sync arguments");
        return 1;
    }
    if (!p36_engine_uses_vulkan_runtime(s->engine)) {
        if (err && errlen) snprintf(err, errlen, "vision prompts require the Vulkan backend");
        return 1;
    }
    if (prompt->len <= 0 || prompt->len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "vision prompt exceeds context");
        return 1;
    }
    bool reuse = p36_session_vision_prefix_matches(s, images, image_count) &&
                 prompt->len >= s->checkpoint.len &&
                 p36_tokens_starts_with(prompt, &s->checkpoint);
    uint64_t previous_end = 0;
    for (size_t k = 0; k < image_count; k++) {
        const p36_vision_span *span = &images[k];
        uint64_t end = (uint64_t)span->token_start + span->embedding.token_count;
        if ((!span->embedding.data && !(reuse && k < s->vision_count)) ||
            !span->embedding.token_count ||
            !span->embedding.grid_width || !span->embedding.grid_height ||
            (uint64_t)span->embedding.grid_width * span->embedding.grid_height !=
                span->embedding.token_count ||
            span->token_start == 0 || span->token_start - 1u < previous_end ||
            end >= (uint64_t)prompt->len ||
            prompt->v[span->token_start - 1] != s->engine->vocab.vision_start_id ||
            prompt->v[end] != s->engine->vocab.vision_end_id) {
            if (err && errlen) snprintf(err, errlen, "malformed vision span");
            return 1;
        }
        for (uint64_t i = span->token_start; i < end; i++) {
            if (prompt->v[i] != s->engine->vocab.image_pad_id) {
                if (err && errlen) snprintf(err, errlen, "vision span does not cover image tokens");
                return 1;
            }
        }
        previous_end = end + 1;
    }

    int pos = 0;
    size_t first_image = 0;
    if (reuse) {
        pos = s->checkpoint.len;
        first_image = s->vision_count;
    } else {
        p36_session_invalidate(s);
    }
    p36_vision_span *identities = realloc(s->vision_spans, image_count * sizeof(*identities));
    if (!identities) {
        if (err && errlen) snprintf(err, errlen, "unable to allocate image identities");
        p36_session_invalidate(s);
        return 1;
    }
    s->vision_spans = identities;
    s->vision_state = true;
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
#ifndef P36_NO_GPU
    if (!reuse) p36_gpu_stream_expert_cache_reset_route_hotness();
    for (size_t k = first_image; k < image_count; k++) {
        const p36_vision_span *span = &images[k];
        p36_tokens prefix = *prompt;
        prefix.len = (int)span->token_start;
        int rc = p36_session_prefill_range(s, &prefix, pos, err, errlen);
        if (rc != 0) return rc;
        pos = prefix.len;
        uint32_t base = (uint32_t)((int32_t)pos + s->rope_delta);
        uint32_t *positions = malloc((uint64_t)span->embedding.token_count *
                                     3u * sizeof(*positions));
        if (!positions) {
            if (err && errlen) snprintf(err, errlen,
                                        "unable to allocate vision positions");
            return 1;
        }
        for (uint32_t i = 0; i < span->embedding.token_count; i++) {
            positions[i * 3u] = base;
            positions[i * 3u + 1u] = base + i / span->embedding.grid_width;
            positions[i * 3u + 2u] = base + i % span->embedding.grid_width;
        }
        uint32_t done = 0;
        p36_vulkan_runtime *rt = s->runtime;
        while (done < span->embedding.token_count) {
            if (s->cancel && s->cancel(s->cancel_ud)) {
                if (err && errlen) snprintf(err, errlen, "interrupted");
                free(positions);
                return P36_SESSION_SYNC_INTERRUPTED;
            }
            uint32_t n = span->embedding.token_count - done;
            if (n > rt->prefill_cap) n = rt->prefill_cap;
            if (!p36_session_reserve_kv(s, (uint32_t)pos + n,
                                        (uint32_t)s->checkpoint.len) ||
                !p36_forward_embeddings_vulkan(
                    s, prompt->v + pos,
                    span->embedding.data + (uint64_t)done * P36_N_EMBD,
                    positions + (uint64_t)done * 3u,
                    n, (uint32_t)pos,
                    pos + (int)n == prompt->len)) {
                if (err && errlen) snprintf(err, errlen,
                                             "vision prefill failed at token %d", pos);
                s->checkpoint_valid = false;
                free(positions);
                return 1;
            }
            for (uint32_t i = 0; i < n; i++)
                p36_tokens_push(&s->checkpoint, prompt->v[pos + (int)i]);
            s->checkpoint_valid = true;
            pos += (int)n;
            done += n;
            if (s->progress) s->progress(s->progress_ud, "prefill_chunk", pos, prompt->len);
            if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_chunk", pos, prompt->len);
        }
        free(positions);
        uint32_t extent = span->embedding.grid_width > span->embedding.grid_height ?
                          span->embedding.grid_width : span->embedding.grid_height;
        s->rope_delta = (int32_t)(base + extent) - pos;
        s->vision_spans[k] = *span;
        s->vision_spans[k].embedding.data = NULL;
        s->vision_count = k + 1;
    }
    return p36_session_prefill_range(s, prompt, pos, err, errlen);
#else
    (void)previous_end;
    (void)pos;
    (void)first_image;
    return 1;
#endif
}

bool p36_session_has_vision_state(const p36_session *s) {
    return s && s->vision_state;
}

bool p36_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

p36_session_rewrite_result p36_session_rewrite_from_common(p36_session *s, const p36_tokens *prompt, int common,
                                                           char *err, size_t errlen) {
    if (!s || !prompt || common < 0 || common > prompt->len) {
        if (err && errlen) snprintf(err, errlen, "invalid rewrite arguments");
        return P36_SESSION_REWRITE_ERROR;
    }
    if (p36_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        return p36_session_sync(s, prompt, err, errlen) == 0 ? P36_SESSION_REWRITE_OK : P36_SESSION_REWRITE_ERROR;
    }
    if (common == s->checkpoint.len && common == prompt->len) return P36_SESSION_REWRITE_OK;
    s->checkpoint.len = common;
    s->checkpoint_valid = true;
    if (p36_session_prefill_range(s, prompt, common, err, errlen) != 0) {
        return P36_SESSION_REWRITE_ERROR;
    }
    return P36_SESSION_REWRITE_OK;
}

int p36_session_common_prefix(p36_session *s, const p36_tokens *prompt) {
    int n;
    int i;
    if (!s || !prompt || !s->checkpoint_valid) return 0;
    n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

static bool p36_session_ensure_logits_host(p36_session *s) {
    if (!s || !s->logits) return false;
    if (s->logits_host_valid) return true;
#ifndef P36_NO_GPU
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (rt && p36_gpu_tensor_read(
                rt->logits, 0, s->logits,
                (uint64_t)P36_N_VOCAB * sizeof(*s->logits))) {
            s->logits_host_valid = true;
            return true;
        }
    }
#endif
    return false;
}

int p36_session_argmax(p36_session *s) {
    if (!s || !s->logits) return -1;
    if (s->gpu_top2_valid) return s->gpu_top2[0];
    if (!p36_session_ensure_logits_host(s)) return -1;
    return sample_argmax(s->logits, P36_N_VOCAB);
}

int p36_session_argmax_excluding(p36_session *s, int excluded_id) {
    int best = -1;
    float best_logit = P36_NEG_INF;
    if (!s || !s->logits) return -1;
    if (s->gpu_top2_valid) {
        return s->gpu_top2[0] == excluded_id
            ? s->gpu_top2[1] : s->gpu_top2[0];
    }
    if (!p36_session_ensure_logits_host(s)) return -1;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v;
        if ((int)i == excluded_id) continue;
        v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

/* GPU min-p prefilter path for the default sampler: the same arithmetic as
 * p36_sample_full_vocab's `top_p >= 1.0f` branch, except that the max and the
 * `scaled > reject_scaled` cut are done on the GPU (vulkan/logits_minp_pack.comp)
 * and the host walks only the survivors. The expf, the sum and the draw stay
 * here so the token stream is bit-identical to the scalar path. Returns the
 * token, or -1 when the packed list cannot reproduce that arithmetic exactly
 * (a block over its cap, no survivor, a non-Vulkan backend), in which case the
 * caller falls back to the scalar path. */
static int p36_session_sample_packed(p36_session *s, float temperature,
                                     int top_k, float top_p, float min_p,
                                     uint64_t *rng) {
#ifndef P36_NO_GPU
    p36_vulkan_runtime *rt;
    const uint32_t nb = P36_GPU_MINP_NB(P36_N_VOCAB);
    const uint32_t cap = P36_GPU_MINP_CAP;
    const uint32_t words = P36_GPU_MINP_WORDS(P36_N_VOCAB);
    const uint32_t cnt0 = P36_GPU_MINP_HDR_WORDS + nb;
    const uint32_t idx0 = cnt0 + nb;
    const uint32_t scl0 = idx0 + nb * cap;
    uint32_t *w;
    float reject_scaled = P36_NEG_INF;
    float sum = 0.0f, best_p = -1.0f, r;
    int best = -1, n = 0;
    bool have_reject = false;

    if (top_k != 0 || !(top_p >= 1.0f)) return -1;
    if (!(min_p > 0.0f) || !(min_p <= 1.0f) || !isfinite(min_p)) return -1;
    if (!p36_engine_uses_vulkan_runtime(s->engine)) return -1;
    rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt || !rt->logits || !rt->minp_pack) return -1;

    /* Same retreated cutoff p36_sample_full_vocab derives. */
    {
        float cutoff = logf(min_p);
        for (int i = 0; i < 8 && isfinite(cutoff); i++) {
            cutoff = nextafterf(cutoff, -FLT_MAX);
            if (expf(cutoff) < min_p) {
                reject_scaled = cutoff;
                have_reject = true;
                break;
            }
        }
    }
    if (!have_reject) return -1;
    if (!p36_gpu_logits_minp_pack(rt->minp_pack, rt->logits, P36_N_VOCAB,
                                  temperature, reject_scaled)) return -1;

    w = xmalloc((size_t)words * sizeof(w[0]));
    if (!p36_gpu_tensor_read(rt->minp_pack, 0, w, (uint64_t)words * sizeof(w[0]))) {
        free(w);
        return -1;
    }
    for (uint32_t b = 0; b < nb; b++) {
        if (w[cnt0 + b] > cap) { free(w); return -1; }
    }
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t n_b = w[cnt0 + b];
        for (uint32_t k = 0; k < n_b; k++) {
            const uint32_t slot = b * cap + k;
            const uint32_t id = w[idx0 + slot];
            float scaled, p;
            memcpy(&scaled, &w[scl0 + slot], sizeof(scaled));
            p = -1.0f;
            if (scaled > reject_scaled) {
                p = expf(scaled);
                if (p < min_p) p = -1.0f;
            }
            s->sample_probs[id] = p;
            if (p >= 0.0f) { sum += p; n++; }
            if (p > best_p) { best_p = p; best = (int)id; }
        }
    }
    if (n == 0 || sum <= 0.0f || !isfinite(sum)) { free(w); return -1; }
    r = p36_sample_rng_f32(rng) * sum;
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t n_b = w[cnt0 + b];
        for (uint32_t k = 0; k < n_b; k++) {
            const uint32_t id = w[idx0 + b * cap + k];
            const float p = s->sample_probs[id];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) { free(w); return (int)id; }
        }
    }
    free(w);
    return best;
#else
    (void)s; (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    return -1;
#endif
}

int p36_session_sample(p36_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s || !s->logits) return -1;
    if (!rng || temperature <= 0.0f) return p36_session_argmax(s);
#ifndef P36_NO_GPU
    /* top_k in [1,8]: skip the full P36_N_VOCAB logits readback entirely.
     * p36_gpu_topk8_tensor selects the exact top-8 on-device (see
     * topk8.comp) and only the 64-byte result crosses to host, instead of
     * the ~1 MB logits vector p36_session_ensure_logits_host would copy.
     * Falls through to the CPU path below on any GPU failure. */
    if (top_k > 0 && top_k <= 8 && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        struct { int32_t ids[8]; float vals[8]; } topk;
        if (rt && p36_gpu_topk8_tensor(rt->topk8, rt->logits, P36_N_VOCAB) &&
            p36_gpu_tensor_read(rt->topk8, 0, &topk, sizeof(topk))) {
            int ids[8];
            float vals[8];
            for (int i = 0; i < 8; i++) { ids[i] = topk.ids[i]; vals[i] = topk.vals[i]; }
            /* The GPU always returns the true top-8; a smaller requested
             * top_k just takes the prefix, since it's already sorted
             * descending and top_k <= 8 here. */
            return p36_sample_from_sorted_topk(ids, vals, top_k, temperature,
                                               top_p <= 0.0f || top_p > 1.0f ? 1.0f : top_p,
                                               min_p < 0.0f ? 0.0f : min_p, rng);
        }
    }
#endif
    {
        const int packed = p36_session_sample_packed(s, temperature, top_k,
                                                     top_p, min_p, rng);
        if (packed >= 0) return packed;
    }
    if (!p36_session_ensure_logits_host(s)) return -1;
    return p36_sample_top_p_min_p(s->logits, P36_N_VOCAB, temperature,
                                  top_k, top_p, min_p, rng, s->sample_probs);
}

int p36_session_sample_penalized(p36_session *s,
                                 float temperature, int top_k,
                                 float top_p, float min_p,
                                 const int *tokens, int n_tokens,
                                 float presence_penalty,
                                 float frequency_penalty,
                                 float repeat_penalty,
                                 uint64_t *rng) {
    if (!s || !s->logits || !rng) return -1;
    if (!tokens || n_tokens <= 0 ||
        (presence_penalty == 0.0f && frequency_penalty == 0.0f &&
         repeat_penalty == 1.0f)) {
        return p36_session_sample(s, temperature, top_k, top_p, min_p, rng);
    }
    if (!p36_session_ensure_logits_host(s)) return -1;

    return p36_sample_logits_penalized(
        s->logits, P36_N_VOCAB, temperature, top_k, top_p, min_p,
        tokens, n_tokens, presence_penalty, frequency_penalty,
        repeat_penalty, rng, s->sample_probs);
}

int p36_session_argmax_penalized_excluding(p36_session *s, int excluded_id,
                                           const int *tokens, int n_tokens,
                                           float presence, float frequency,
                                           float repeat_penalty) {
    if (!s || excluded_id < 0 || excluded_id >= (int)P36_N_VOCAB ||
        !p36_session_ensure_logits_host(s)) return -1;
    float saved = s->logits[excluded_id];
    s->logits[excluded_id] = -INFINITY;
    uint64_t rng = 1;
    int token = p36_sample_logits_penalized(s->logits, P36_N_VOCAB,
        0, 0, 1, 0, tokens, n_tokens, presence, frequency, repeat_penalty,
        &rng, s->sample_probs);
    s->logits[excluded_id] = saved;
    return token;
}

bool p36_session_in_think(p36_session *s) {
    if (!s || !s->checkpoint_valid) return false;
    const p36_vocab *vocab = &s->engine->vocab;
    for (int i = s->checkpoint.len - 1; i >= 0; i--) {
        int t = s->checkpoint.v[i];
        if (t == vocab->think_end_id) return false;
        if (t == vocab->think_start_id) return true;
    }
    return false;
}

/* An EOS while the transcript's <think> block is still unclosed would end the
 * reply with no visible text at all: quantized models near-tie </think> and
 * EOS right after the forced-open <think>.  Callers map that EOS to </think>
 * and keep decoding; at most one substitution per think block, since the
 * close itself ends the in-think state. */
int p36_session_eos_to_think_close(p36_session *s, int token) {
    if (!s || token != s->engine->vocab.eos_id) return token;
    int close = s->engine->vocab.think_end_id;
    if (close < 0 || !p36_session_in_think(s)) return token;
    return close;
}

int p36_session_token_rank(p36_session *s, int token, int max_rank) {
    if (!s || !s->logits || token < 0 || token >= (int)P36_N_VOCAB ||
        max_rank <= 0) return 0;
    if (s->gpu_top2_valid) {
        if (s->gpu_top2[0] == token) return 1;
        if (s->gpu_top2[1] == token) return max_rank >= 2 ? 2 : 0;
        if (max_rank <= 2) return 0;
    }
    if (!p36_session_ensure_logits_host(s)) return 0;
    float target = s->logits[token];
    if (!isfinite(target)) return 0;
    int rank = 1;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        if ((int)i == token || !isfinite(s->logits[i])) continue;
        if (s->logits[i] > target ||
            (s->logits[i] == target && (int)i < token)) {
            if (++rank > max_rank) return 0;
        }
    }
    return rank;
}

int p36_session_top_logprobs(p36_session *s, p36_token_score *out, int k) {
    float max_logit = P36_NEG_INF;
    double sum = 0.0;
    if (!s || !out || !s->logits || k <= 0 ||
        !p36_session_ensure_logits_host(s)) return 0;
    if (k > (int)P36_N_VOCAB) k = (int)P36_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = P36_NEG_INF;
        out[i].logprob = P36_NEG_INF;
    }
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;
    for (uint32_t i = 0; i < P36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    {
        double logsum = (double)max_logit + log(sum);
        for (int i = 0; i < k && out[i].id >= 0; i++) {
            out[i].logprob = (float)((double)out[i].logit - logsum);
        }
    }
    return k;
}

static int p36_session_eval_internal(p36_session *s, int token, bool update_mtp,
                                     char *err, size_t errlen) {
    uint32_t pos;
    if (!s) {
        if (err && errlen) snprintf(err, errlen, "missing session");
        return 1;
    }
    if (s->checkpoint.len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context exhausted");
        return 1;
    }
    if (!p36_session_reserve_kv(s, (uint32_t)s->checkpoint.len + 1u,
                                (uint32_t)s->checkpoint.len)) {
        if (err && errlen) snprintf(err, errlen, "failed to grow KV cache");
        return 1;
    }
    s->mtp_draft_valid = false;
    pos = (uint32_t)s->checkpoint.len;
    if (!(p36_engine_uses_vulkan_runtime(s->engine)
#ifndef P36_NO_GPU
          ? p36_forward_token_vulkan(s, token, pos, true)
#else
          ? false
#endif
          : p36_forward_token_cpu(s, token, pos, true))) {
        if (err && errlen) snprintf(err, errlen, "%s decode failed at pos %d",
                                     p36_backend_name(s->engine->backend),
                                     s->checkpoint.len);
        s->checkpoint_valid = false;
        return 1;
    }
#ifndef P36_NO_GPU
    if (update_mtp && p36_engine_has_mtp(s->engine) && p36_engine_uses_vulkan_runtime(s->engine)) {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (!p36_mtp_eval_vulkan(s, token, pos, rt->norm)) {
            if (err && errlen) snprintf(err, errlen, "MTP draft failed at pos %u", pos);
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, P36_N_VOCAB, &s->mtp_draft_margin);
        s->mtp_draft_valid = s->mtp_draft_token >= 0;
    }
#else
    (void)update_mtp;
#endif
    p36_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

#ifndef P36_NO_GPU
#endif

int p36_session_eval(p36_session *s, int token, char *err, size_t errlen) {
    return p36_session_eval_internal(s, token, false, err, errlen);
}

#ifndef P36_NO_GPU
static bool p36_sessions_eval_batch_vulkan_supported(p36_decode_item *items,
                                                      int count,
                                                      p36_engine *e) {
    const char *enabled = getenv("P36_VK_SESSION_BATCH");
    if ((enabled && enabled[0] && !strcmp(enabled, "0")) ||
        !items || count < 2 || count > 8 || !e ||
        !p36_backend_uses_graph(e->backend) || e->ssd_streaming) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (s && s->vision_state) return false;
        p36_vulkan_runtime *rt = s ? s->runtime : NULL;
        if (!s || s->engine != e || !s->checkpoint_valid || !rt ||
            rt->prefill_cap < (uint32_t)count) {
            return false;
        }
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (!p36_layer_is_full_attention(il)) continue;
            p36_vulkan_full_attn_cache *cache = &rt->full[il];
            bool f16 = cache->type_k == P36_KV_CACHE_F16 &&
                       cache->type_v == P36_KV_CACHE_F16;
            bool resident = cache->type_k == P36_KV_CACHE_Q8_0 &&
                            cache->type_v == P36_KV_CACHE_Q4_0;
            if (!f16 && !resident) return false;
        }
    }
    return true;
}
#endif

int p36_sessions_eval_batch(p36_decode_item *items, int count,
                            char *err, size_t errlen) {
    if (!items || count <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty decode batch");
        return 1;
    }
    if (count == 1) {
        return p36_session_eval(items[0].session, items[0].token, err, errlen);
    }

    p36_session *first = items[0].session;
    if (!first || !first->engine) {
        if (err && errlen) snprintf(err, errlen, "decode batch has no session");
        return 1;
    }
    p36_engine *e = first->engine;
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (!s || s->engine != e) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d belongs to a different engine", i);
            }
            return 1;
        }
        if (items[i].token < 0 || items[i].token >= (int)P36_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d has an invalid token", i);
            }
            return 1;
        }
        if (s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
        if (!p36_session_reserve_kv(s, (uint32_t)s->checkpoint.len + 1u,
                                    (uint32_t)s->checkpoint.len)) {
            if (err && errlen) snprintf(err, errlen,
                                        "decode batch item %d failed to grow KV cache", i);
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "decode batch repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

#ifndef P36_NO_GPU
    if (p36_sessions_eval_batch_vulkan_supported(items, count, e)) {
        bool ok = p36_sessions_eval_batch_vulkan(items, count);
        if (!ok) {
            for (int i = 0; i < count; i++) p36_session_invalidate(items[i].session);
            if (err && errlen) {
                snprintf(err, errlen, "%s batched decode failed",
                         p36_backend_name(e->backend));
            }
            return 1;
        }
        for (int i = 0; i < count; i++) {
            p36_session *s = items[i].session;
            p36_tokens_push(&s->checkpoint, items[i].token);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
        }
        if (getenv("P36_VK_SESSION_BATCH_LOG")) {
            fprintf(stderr, "p36: %s session batch rows=%d native=1\n",
                    p36_backend_name(e->backend), count);
        }
        return 0;
    }
#endif

    for (int i = 0; i < count; i++) {
        if (p36_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) p36_session_invalidate(items[j].session);
            return 1;
        }
    }
    if (getenv("P36_VK_SESSION_BATCH_LOG")) {
        fprintf(stderr, "p36: session batch rows=%d native=0 ordered=1\n", count);
    }
    return 0;
}

int p36_sessions_eval_batch_with_prefill(
        p36_decode_item *items, int count,
        p36_session *prefill_session, const p36_tokens *prefill_prompt,
        char *err, size_t errlen) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        if (err && errlen) snprintf(err, errlen, "invalid mixed model batch");
        return 1;
    }
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len >= prefill_session->ctx_size ||
        !p36_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        p36_session *s = items[i].session;
        if (!s || s == prefill_session ||
            s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size ||
            items[i].token < 0 || items[i].token >= (int)P36_N_VOCAB) {
            if (err && errlen) snprintf(err, errlen, "invalid mixed decode item %d", i);
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "mixed decode repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

    int rc = p36_session_sync(prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = p36_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) p36_session_invalidate(prefill_session);
    return rc;
}

/* P36_MTP_TIMING: one line per speculative cycle.  `drains` counts open-batch
 * flushes (each one stalls until the GPU queue is empty), and read_kb is what
 * the host pulled back in that same window. */
