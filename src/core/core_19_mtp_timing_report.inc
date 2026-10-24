static void p36_mtp_timing_report(uint32_t pos, const char *exit_reason,
                                  int verify_n, int commit_n, int n_draft,
                                  double c0, double c_fwd, double c_draft,
                                  double c_snap, double c_verify, double c_commit,
                                  uint64_t r0, uint64_t f0) {
    fprintf(stderr,
            "p36: mtp cycle pos=%u %-12s verify_n=%d commit_n=%d drafts=%d "
            "fwd=%.2f draft=%.2f snap=%.2f verify=%.2f commit=%.2f total=%.2f ms "
            "read_kb=%.0f drains=%" PRIu64 "\n",
            pos, exit_reason ? exit_reason : "-", verify_n, commit_n, n_draft,
            (c_fwd - c0) * 1000.0, (c_draft - c_fwd) * 1000.0,
            (c_snap - c_draft) * 1000.0, (c_verify - c_snap) * 1000.0,
            (c_commit - c_verify) * 1000.0, (c_commit - c0) * 1000.0,
            (double)(p36_gpu_read_bytes() - r0) / 1024.0,
            p36_gpu_read_flushes() - f0);
}

/* Shared speculative commit: verify[] has just been forwarded as one batch
 * from start_pos, with row_tops[i] the argmax predicted after verify[i].
 * Accepts the longest prefix the target agrees with, leaves the KV frontier
 * and s->logits consistent with that prefix, and pushes it into the
 * checkpoint.  Both draft sources (MTP head, prompt lookup) end here so the
 * accept/rollback numerics exist once.  Returns 1 + commit_n, or -1. */
static int p36_spec_commit_tokens(p36_session *s, const int *verify, int verify_n,
                                  uint32_t start_pos, const int *row_tops,
                                  int eos_token, int *accepted,
                                  char *err, size_t errlen) {
    p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
    int commit_n = 1;
    if (!rt) return -1;

    for (int i = 1; i < verify_n; i++) {
        if (row_tops[i - 1] != verify[i]) break;
        commit_n++;
    }
    for (int i = 0; i < commit_n; i++) {
        if (verify[i] == eos_token) {
            commit_n = i + 1;
            break;
        }
    }

    if (commit_n == verify_n) {
        /* Full accept: the verify pass already ran every committed token's
         * forward and left the last row's logits in s->logits; adopting its
         * KV/recurrent frontier makes the replay free. */
        if (!p36_vulkan_update_last_h(rt, (uint32_t)verify_n - 1u)) {
            if (err && errlen)
                snprintf(err, errlen, "spec frontier adopt failed at pos %u", start_pos);
            p36_session_invalidate(s);
            return -1;
        }
    } else {
        if (!p36_vulkan_spec_frontier_restore(rt)) {
            if (err && errlen)
                snprintf(err, errlen, "spec rollback failed at pos %u", start_pos);
            p36_session_invalidate(s);
            return -1;
        }
        /* Replay the accepted prefix as one batch; refreshes s->logits
         * and last_h from the last accepted row. */
        bool replay;
        p36_gpu_set_micro_batch(true);
        replay = p36_forward_tokens_vulkan(s, verify, (uint32_t)commit_n, start_pos, true);
        p36_gpu_set_micro_batch(false);
        if (!replay) {
            if (err && errlen)
                snprintf(err, errlen, "spec accepted-prefix replay failed at pos %u", start_pos);
            p36_session_invalidate(s);
            return -1;
        }
    }

    for (int i = 0; i < commit_n; i++) {
        accepted[1 + i] = verify[i];
        p36_tokens_push(&s->checkpoint, verify[i]);
    }
    s->checkpoint_valid = true;
    p36_mtp_stats_add(verify_n, commit_n, commit_n == verify_n);
    return 1 + commit_n;
}

/* Prompt lookup: the newest earlier copy of the last `order` tokens predicts
 * the continuation.  Pure host work over the live token history -- no model,
 * no GPU -- so a repeat-heavy turn (code blocks, file rewrites, echoed tool
 * output) gets drafts for free.  Returns the number of proposed tokens. */
static int p36_lookup_propose(const p36_session *s, int order, int k, int *out) {
    const int *v;
    uint32_t len, suffix, i, avail;
    if (!s || !out || order <= 0 || k <= 0) return 0;
    v = s->checkpoint.v;
    len = s->checkpoint.len;
    if (!v || len <= (uint32_t)order) return 0;
    suffix = len - (uint32_t)order;
    for (i = suffix; i-- > 0u; ) {
        if (memcmp(v + i, v + suffix, (size_t)order * sizeof(*v)) != 0) continue;
        avail = len - (i + (uint32_t)order);
        if (avail == 0u) continue;
        if (avail > (uint32_t)k) avail = (uint32_t)k;
        memcpy(out, v + i + (uint32_t)order, (size_t)avail * sizeof(*out));
        return (int)avail;
    }
    return 0;
}

/* One prompt-lookup speculative step: evaluate the pending token, draft from
 * the token history, then reuse the MTP verifier and commit path so the
 * accept/rollback numerics never fork by draft source. */
static int p36_session_eval_lookup(p36_session *s, int first_token, int max_tokens,
                                   int eos_token, int *accepted, int accepted_cap,
                                   char *err, size_t errlen) {
    p36_engine *e = s->engine;
    p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
    int verify[P36_MTP_MAX_DRAFT];
    int row_tops[P36_MTP_MAX_DRAFT];
    int k, verify_n, rc;
    uint32_t start_pos;
    if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    if (!rt || !rt->last_h || !rt->logits) return 1;
    k = e->lookup_k;
    if (k > P36_MTP_MAX_DRAFT) k = P36_MTP_MAX_DRAFT;
    if (k > accepted_cap - 1) k = accepted_cap - 1;
    if (k > max_tokens - 1) k = max_tokens - 1;
    if (k > s->ctx_size - s->checkpoint.len) k = s->ctx_size - s->checkpoint.len;
    if (first_token == eos_token || k <= 1) return 1;
    verify_n = p36_lookup_propose(s, e->lookup_order, k, verify);
    if (verify_n <= 1) return 1;
    /* The greedy token at this position is already known from the logits of
     * the token just committed.  A proposal that disagrees cannot be right,
     * so skip the verify batch rather than pay for it. */
    if (verify[0] != p36_session_argmax(s)) return 1;
    start_pos = (uint32_t)s->checkpoint.len;
    if (!p36_vulkan_spec_frontier_snapshot(rt)) return 1;
    if (!p36_vulkan_verify_suffix_tops(s, verify, (uint32_t)verify_n, start_pos, row_tops)) {
        if (!p36_vulkan_spec_frontier_restore(rt)) p36_session_invalidate(s);
        if (err && errlen) snprintf(err, errlen, "lookup verifier failed at pos %u", start_pos);
        return -1;
    }
    rc = p36_spec_commit_tokens(s, verify, verify_n, start_pos, row_tops,
                                eos_token, accepted, err, errlen);
    return rc < 0 ? -1 : rc;
}
int p36_session_eval_speculative_argmax(p36_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    p36_engine *e;
    if (!s || !accepted || accepted_cap <= 0 || max_tokens <= 0) return 0;
    e = s->engine;
    if (!e || !p36_engine_has_mtp(e) || e->mtp_draft_tokens <= 1 ||
        !p36_engine_uses_vulkan_runtime(e)) {
        /* No MTP head available: prompt lookup is the only draft source. */
        if (e && e->lookup_k > 1 && p36_engine_uses_vulkan_runtime(e))
            return p36_session_eval_lookup(s, first_token, max_tokens, eos_token,
                                           accepted, accepted_cap, err, errlen);
        if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifndef P36_NO_GPU
    {
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        int verify[P36_MTP_MAX_DRAFT];
        int row_tops[P36_MTP_MAX_DRAFT];
        int draft_cap = e->mtp_draft_tokens - 1;
        int verify_n;
        int commit_n;
        uint32_t start_pos;
        int prev_token;
        if (!rt || !rt->last_h || !rt->logits) return 0;

        /* P36_MTP_TIMING: per-phase wall clock for one speculative cycle plus
         * the host readback volume and the GPU drains it caused.  Diagnostic
         * only: nothing below changes the tokens produced. */
        const bool mtp_timing = getenv("P36_MTP_TIMING") != NULL;
        double tc0 = 0.0, tc_fwd = 0.0, tc_draft = 0.0, tc_snap = 0.0;
        double tc_verify = 0.0;
        uint64_t tr0 = 0, tf0 = 0;
        int n_draft = 0;
        if (mtp_timing) {
            tc0 = p36_now_sec();
            tr0 = p36_gpu_read_bytes();
            tf0 = p36_gpu_read_flushes();
        }

        /* Draft backoff: after unconfident drafts, decode plainly for a
         * few tokens so hard content does not pay the draft head on every
         * token. Confident content resets to drafting every token. */
        if (s->mtp_backoff > 0) {
            s->mtp_backoff--;
            if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
            accepted[0] = first_token;
            return 1;
        }

        /* Commit the target token with its MTP draft piggybacked on the
         * same submission; the confidence gate and the first-draft check
         * against the fresh target logits are then free. */
        if (p36_session_eval_with_draft(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        if (mtp_timing) {
            tc_fwd = p36_now_sec();
            tc_draft = tc_fwd;
            tc_snap = tc_fwd;
            tc_verify = tc_fwd;
        }

        if (draft_cap > P36_MTP_MAX_DRAFT) draft_cap = P36_MTP_MAX_DRAFT;
        if (draft_cap > accepted_cap - 1) draft_cap = accepted_cap - 1;
        if (draft_cap > max_tokens - 1) draft_cap = max_tokens - 1;
        if (draft_cap > s->ctx_size - s->checkpoint.len) draft_cap = s->ctx_size - s->checkpoint.len;
        if (rt->prefill_cap > 0 && draft_cap > (int)rt->prefill_cap) draft_cap = (int)rt->prefill_cap;
        if (first_token == eos_token || draft_cap <= 0 || !s->mtp_draft_valid) {
            if (mtp_timing) {
                p36_mtp_timing_report((uint32_t)s->checkpoint.len, "no-draft", 0, 1, 0,
                                      tc0, tc_fwd, tc_draft, tc_snap, tc_verify, tc_fwd, tr0, tf0);
            }
            return 1;
        }
        if ((e->mtp_margin > 0.0f && s->mtp_draft_margin < e->mtp_margin) ||
            s->mtp_draft_token != p36_session_argmax(s)) {
            s->mtp_backoff_len = s->mtp_backoff_len ? (s->mtp_backoff_len < 4 ? s->mtp_backoff_len * 2 : 4) : 1;
            s->mtp_backoff = s->mtp_backoff_len;
            p36_mtp_stats_add(0, 0, false);
            if (mtp_timing) {
                p36_mtp_timing_report((uint32_t)s->checkpoint.len, "gate-reject", 0, 1, 0,
                                      tc0, tc_fwd, tc_draft, tc_snap, tc_verify, tc_fwd, tr0, tf0);
            }
            return 1;
        }
        s->mtp_backoff_len = 0;

        start_pos = (uint32_t)s->checkpoint.len;
        verify[0] = s->mtp_draft_token;
        verify_n = 1;
        prev_token = verify[0];
        while (verify_n < draft_cap && prev_token != eos_token) {
            float margin = 0.0f;
            if (!p36_mtp_eval_vulkan(s, prev_token, start_pos + (uint32_t)verify_n - 1u, rt->mtp_cur)) {
                if (err && errlen) snprintf(err, errlen, "MTP recursive draft failed at pos %u",
                                            start_pos + (uint32_t)verify_n - 1u);
                s->checkpoint_valid = false;
                return -1;
            }
            s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, P36_N_VOCAB, &margin);
            if (s->mtp_draft_token < 0) break;
            if (e->mtp_margin > 0.0f && margin < e->mtp_margin) break;
            verify[verify_n++] = s->mtp_draft_token;
            prev_token = s->mtp_draft_token;
            n_draft++;
        }
        if (mtp_timing) tc_draft = p36_now_sec();

        if (!p36_vulkan_spec_frontier_snapshot(rt)) return 1;
        if (mtp_timing) tc_snap = p36_now_sec();
        const bool prof_verify = getenv("P36_VK_PROF_VERIFY") != NULL;
        static uint32_t prof_verify_n;
        if (prof_verify) p36_gpu_prof_reset();
        if (!p36_vulkan_verify_suffix_tops(s, verify, (uint32_t)verify_n, start_pos, row_tops)) {
            if (!p36_vulkan_spec_frontier_restore(rt)) p36_session_invalidate(s);
            if (err && errlen) snprintf(err, errlen, "MTP target verifier failed at pos %u", start_pos);
            return -1;
        }
        if (prof_verify) {
            char plabel[32];
            snprintf(plabel, sizeof(plabel), "verify%u_n%d", prof_verify_n++, verify_n);
            p36_gpu_prof_report(plabel);
        }
        if (mtp_timing) tc_verify = p36_now_sec();

        {
            int rc = p36_spec_commit_tokens(s, verify, verify_n, start_pos, row_tops,
                                            eos_token, accepted, err, errlen);
            if (rc < 0) return -1;
            commit_n = rc - 1;
        }
        if (mtp_timing) {
            p36_mtp_timing_report(start_pos,
                                  commit_n == verify_n ? "full-accept" : "partial",
                                  verify_n, commit_n, n_draft, tc0, tc_fwd, tc_draft,
                                  tc_snap, tc_verify, p36_now_sec(), tr0, tf0);
        }
        s->mtp_draft_valid = false;
        s->mtp_draft_token = -1;
        return 1 + commit_n;
    }
#else
    (void)eos_token;
    if (p36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

const float *p36_session_logits(p36_session *s, int *n_vocab) {
    if (n_vocab) *n_vocab = P36_N_VOCAB;
    return p36_session_ensure_logits_host(s) ? s->logits : NULL;
}

static void p36_session_reset_runtime(p36_session *s) {
    if (!s) return;
    s->logits_host_valid = false;
    s->gpu_top2_valid = false;
    if (p36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef P36_NO_GPU
        (void)p36_vulkan_runtime_reset((p36_vulkan_runtime *)s->runtime);
#endif
    } else {
        p36_cpu_runtime_reset((p36_cpu_runtime *)s->runtime);
    }
}

void p36_session_invalidate(p36_session *s) {
    if (!s) return;
    p36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->vision_state = false;
    free(s->vision_spans);
    s->vision_spans = NULL;
    s->vision_count = 0;
    s->rope_delta = 0;
}

void p36_session_rewind(p36_session *s, int pos) {
    p36_tokens prefix = {0};
    if (!s) return;
    if (s->vision_state) {
        p36_session_invalidate(s);
        return;
    }
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    if (pos == s->checkpoint.len) return;
    for (int i = 0; i < pos; i++) p36_tokens_push(&prefix, s->checkpoint.v[i]);
    p36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    if (pos > 0) {
        if (p36_session_sync(s, &prefix, NULL, 0) != 0) {
            p36_session_invalidate(s);
        }
    }
    p36_tokens_free(&prefix);
}

int p36_session_pos(p36_session *s) {
    return s ? s->checkpoint.len : 0;
}

int p36_session_ctx(p36_session *s) {
    return s ? s->ctx_size : 0;
}

int p36_engine_routed_quant_bits(p36_engine *e) {
    return e ? e->routed_quant_bits : 0;
}

bool p36_engine_has_mtp(p36_engine *e) {
    return e && p36_backend_uses_graph(e->backend) && e->mtp_ready;
}

int p36_engine_mtp_draft_tokens(p36_engine *e) {
    return p36_engine_has_mtp(e) ? e->mtp_draft_tokens : 1;
}

/* Drafts per prompt-lookup step; 0 disables the lookup route. */
int p36_engine_lookup_k(p36_engine *e) {
    return e && p36_engine_uses_vulkan_runtime(e) ? e->lookup_k : 0;
}

/* Greedy decode can speculate when any draft source is armed. */
bool p36_engine_spec_enabled(p36_engine *e) {
    return p36_engine_mtp_draft_tokens(e) > 1 || p36_engine_lookup_k(e) > 1;
}

const p36_tokens *p36_session_tokens(p36_session *s) {
    return s ? &s->checkpoint : NULL;
}

static void p36_payload_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "session payload error");
}

static int p36_payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                       char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    while (bytes) {
        size_t n = bytes < sizeof(buf) ? (size_t)bytes : sizeof(buf);
        if (fread(buf, 1, n, src) != n) {
            p36_payload_set_err(err, errlen, "failed to read staged session payload");
            return 1;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            p36_payload_set_err(err, errlen, "failed to write staged session payload");
            return 1;
        }
        bytes -= n;
    }
    return 0;
}

static uint64_t p36_full_k_bytes(uint32_t rows, p36_kv_cache_type type) {
    return (uint64_t)rows * p36_kv_cache_row_bytes(type, P36_N_HEAD_KV * P36_N_HEAD_DIM);
}

static uint64_t p36_full_v_bytes(uint32_t rows, p36_kv_cache_type type) {
    return (uint64_t)rows * p36_kv_cache_row_bytes(type, P36_N_HEAD_KV * P36_N_VALUE_DIM);
}

static uint64_t p36_recurrent_conv_bytes(void) {
    return (uint64_t)(P36_N_SSM_CONV - 1u) * P36_N_SSM_CONV_DIM * sizeof(float);
}

static uint64_t p36_recurrent_state_bytes(void) {
    return (uint64_t)P36_N_SSM_STATE * P36_N_SSM_STATE * P36_N_SSM_DT_RANK * sizeof(float);
}

static int p36_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                                   char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!ptr || fwrite(ptr, 1, (size_t)bytes, fp) != bytes) {
        p36_payload_set_err(err, errlen, "failed to write session payload");
        return 1;
    }
    return 0;
}

static int p36_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                                  uint64_t *remaining, char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!remaining || *remaining < bytes || !ptr ||
        fread(ptr, 1, (size_t)bytes, fp) != bytes)
    {
        p36_payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    *remaining -= bytes;
    return 0;
}

static int p36_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    return p36_payload_write_bytes(fp, &v, sizeof(v), err, errlen);
}

static int p36_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                                char *err, size_t errlen) {
    return p36_payload_read_bytes(fp, v, sizeof(*v), remaining, err, errlen);
}

#ifndef P36_NO_GPU
static int p36_payload_write_tensor(FILE *fp, const p36_gpu_tensor *tensor,
                                    uint64_t bytes, char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (!p36_gpu_tensor_read(tensor, off, buf, n) ||
            fwrite(buf, 1, (size_t)n, fp) != n)
        {
            p36_payload_set_err(err, errlen, "failed to write GPU session payload");
            return 1;
        }
        off += n;
    }
    return 0;
}

static int p36_payload_read_tensor(p36_gpu_tensor *tensor, FILE *fp,
                                   uint64_t bytes, uint64_t *remaining,
                                   char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    if (!remaining || *remaining < bytes) {
        p36_payload_set_err(err, errlen, "truncated GPU session payload");
        return 1;
    }
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (fread(buf, 1, (size_t)n, fp) != n ||
            !p36_gpu_tensor_write(tensor, off, buf, n))
        {
            p36_payload_set_err(err, errlen, "failed to read GPU session payload");
            return 1;
        }
        off += n;
    }
    *remaining -= bytes;
    return 0;
}

static int p36_payload_write_recurrent_state(FILE *fp,
                                              const p36_gpu_tensor *tensor,
                                              bool f16,
                                              char *err, size_t errlen) {
    if (!f16)
        return p36_payload_write_tensor(fp, tensor,
                                        p36_recurrent_state_bytes(), err, errlen);
    uint16_t half[8192];
    float full[8192];
    uint64_t elems = p36_recurrent_state_bytes() / sizeof(float);
    for (uint64_t off = 0; off < elems; off += 8192u) {
        uint64_t n = elems - off;
        if (n > 8192u) n = 8192u;
        if (!p36_gpu_tensor_read(tensor, off * sizeof(uint16_t), half,
                                  n * sizeof(uint16_t))) {
            p36_payload_set_err(err, errlen, "failed to read FP16 recurrent state");
            return 1;
        }
        for (uint64_t i = 0; i < n; i++) full[i] = p36_f16_to_f32(half[i]);
        if (fwrite(full, sizeof(float), (size_t)n, fp) != n) {
            p36_payload_set_err(err, errlen, "failed to write recurrent state");
            return 1;
        }
    }
    return 0;
}

static int p36_payload_read_recurrent_state(p36_gpu_tensor *tensor, FILE *fp,
                                             bool f16, uint64_t *remaining,
                                             char *err, size_t errlen) {
    uint64_t bytes = p36_recurrent_state_bytes();
    if (!f16)
        return p36_payload_read_tensor(tensor, fp, bytes, remaining, err, errlen);
    if (!remaining || *remaining < bytes) {
        p36_payload_set_err(err, errlen, "truncated recurrent state");
        return 1;
    }
    uint16_t half[8192];
    float full[8192];
    uint64_t elems = bytes / sizeof(float);
    for (uint64_t off = 0; off < elems; off += 8192u) {
        uint64_t n = elems - off;
        if (n > 8192u) n = 8192u;
        if (fread(full, sizeof(float), (size_t)n, fp) != n) {
            p36_payload_set_err(err, errlen, "failed to read recurrent state");
            return 1;
        }
        for (uint64_t i = 0; i < n; i++) half[i] = p36_f32_to_f16(full[i]);
        if (!p36_gpu_tensor_write(tensor, off * sizeof(uint16_t), half,
                                   n * sizeof(uint16_t))) {
            p36_payload_set_err(err, errlen, "failed to upload FP16 recurrent state");
            return 1;
        }
    }
    *remaining -= bytes;
    return 0;
}
#endif

uint64_t p36_session_payload_bytes(p36_session *s) {
    if (!s || s->vision_state) return 0;
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != P36_KV_CACHE_F16 ||
         s->engine->cache_type_v != P36_KV_CACHE_F16);
    uint64_t bytes = (uint64_t)(typed ? P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS) * sizeof(uint32_t);
    bytes += (uint64_t)rows * sizeof(int32_t);
    bytes += (uint64_t)P36_N_VOCAB * sizeof(float);
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            bytes += sizeof(uint32_t) +
                     p36_full_k_bytes(rows, s->engine->cache_type_k) +
                     p36_full_v_bytes(rows, s->engine->cache_type_v);
        } else {
            bytes += p36_recurrent_conv_bytes() + p36_recurrent_state_bytes();
        }
    }
    return bytes;
}

int p36_session_write_staged_payload(const p36_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        p36_payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        p36_payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = p36_payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        p36_payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void p36_session_payload_file_free(p36_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int p36_session_stage_payload(p36_session *s, p36_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        p36_payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid || s->vision_state) {
        p36_payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
    size_t tmpl_len = strlen(tmpdir) + sizeof("/p36-session-payload.XXXXXX");
    char *tmpl = xmalloc(tmpl_len);
    snprintf(tmpl, tmpl_len, "%s/p36-session-payload.XXXXXX", tmpdir);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        free(tmpl);
        p36_payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        free(tmpl);
        if (err && errlen)
            snprintf(err, errlen, "failed to open staged session payload: %s",
                     strerror(saved));
        return 1;
    }

    int rc = p36_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        p36_payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            p36_payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        p36_payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        free(tmpl);
        return 1;
    }
    out->path = p36_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    free(tmpl);
    return 0;
}

int p36_session_save_payload(p36_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload save arguments");
        return 1;
    }
    if (!s->checkpoint_valid) {
        p36_payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->vision_state) {
        p36_payload_set_err(err, errlen, "image-conditioned checkpoints are not persistent");
        return 1;
    }
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != P36_KV_CACHE_F16 ||
         s->engine->cache_type_v != P36_KV_CACHE_F16);
    uint32_t h[P36_PAYLOAD_U32_FIELDS_TYPED_KV] = {
        P36_PAYLOAD_MAGIC,
        typed ? P36_PAYLOAD_VERSION_TYPED_KV : P36_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->engine && s->engine->backend == P36_BACKEND_CPU ?
            ((p36_cpu_runtime *)s->runtime)->prefill_cap :
#ifndef P36_NO_GPU
            ((p36_vulkan_runtime *)s->runtime)->prefill_cap,
#else
            0,
#endif
        rows,
        P36_N_VOCAB,
        P36_N_LAYER,
        P36_N_HEAD_KV,
        P36_N_HEAD_DIM,
        P36_N_VALUE_DIM,
        P36_N_SSM_CONV,
        P36_N_SSM_CONV_DIM,
        P36_N_SSM_STATE,
        P36_N_SSM_DT_RANK,
        s->engine ? (uint32_t)s->engine->cache_type_k : 0,
        s->engine ? (uint32_t)s->engine->cache_type_v : 0,
    };
    uint32_t nfields = typed ? P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS;
    for (uint32_t i = 0; i < nfields; i++) {
        if (p36_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    if (p36_payload_write_bytes(fp, s->checkpoint.v,
                                (uint64_t)rows * sizeof(int32_t),
                                err, errlen) != 0) return 1;
    if (s->engine->backend == P36_BACKEND_CPU) {
        p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_write_bytes(fp, s->logits,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                p36_full_attn_cache *c = &rt->full[il];
                if (!c->k || !c->v || c->len < rows) {
                    p36_payload_set_err(err, errlen, "CPU KV cache is shorter than checkpoint");
                    return 1;
                }
                if (p36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->k, p36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->v, p36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                    return 1;
            } else {
                p36_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_write_bytes(fp, c->conv, p36_recurrent_conv_bytes(), err, errlen) != 0 ||
                    p36_payload_write_bytes(fp, c->state, p36_recurrent_state_bytes(), err, errlen) != 0)
                    return 1;
            }
        }
        return 0;
    }
#ifdef P36_NO_GPU
    return p36_unsupported_runtime(err, errlen);
#else
    p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
    if (!rt) return p36_unsupported_runtime(err, errlen);
    if (s->logits_host_valid) {
        if (p36_payload_write_bytes(fp, s->logits,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    err, errlen) != 0) return 1;
    } else if (p36_payload_write_tensor(fp, rt->logits,
                                        (uint64_t)P36_N_VOCAB * sizeof(float),
                                        err, errlen) != 0) {
        return 1;
    }
    for (uint32_t il = 0; il < P36_N_LAYER; il++) {
        if (p36_layer_is_full_attention(il)) {
            p36_vulkan_full_attn_cache *c = &rt->full[il];
            if (!c->k || !c->v || c->cap < rows) {
                p36_payload_set_err(err, errlen, "Vulkan KV cache is shorter than checkpoint");
                return 1;
            }
            if (p36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                p36_payload_write_tensor(fp, c->k, p36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                p36_payload_write_tensor(fp, c->v, p36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                return 1;
        } else {
            p36_vulkan_recurrent_cache *c = &rt->recurrent[il];
            if (p36_payload_write_tensor(fp, c->conv, p36_recurrent_conv_bytes(), err, errlen) != 0 ||
                p36_payload_write_recurrent_state(fp, c->state,
                                                  rt->recur_state_f16,
                                                  err, errlen) != 0)
                return 1;
        }
    }
    return 0;
#endif
}

int p36_session_load_payload(p36_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload load arguments");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t magic = 0, version = 0, third = 0;
    if (p36_payload_read_u32(fp, &magic, &remaining, err, errlen) != 0 ||
        p36_payload_read_u32(fp, &version, &remaining, err, errlen) != 0 ||
        p36_payload_read_u32(fp, &third, &remaining, err, errlen) != 0)
    {
        if (err && errlen) snprintf(err, errlen, "failed to read session payload header");
        return 1;
    }
    if (magic != P36_PAYLOAD_MAGIC) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload format");
        return 1;
    }
    if (version == P36_PAYLOAD_VERSION_TOKEN_ONLY) {
        uint32_t n_tokens = third;
        if (n_tokens >= (uint32_t)s->ctx_size) {
            p36_payload_set_err(err, errlen, "token-only checkpoint does not fit current context");
            return 1;
        }
        if (remaining != (uint64_t)n_tokens * sizeof(int32_t)) {
            if (err && errlen) snprintf(err, errlen, "mismatched token-only payload size");
            return 1;
        }
        p36_tokens prompt = {0};
        for (uint32_t i = 0; i < n_tokens; i++) {
            int32_t tok = 0;
            if (p36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) {
                p36_tokens_free(&prompt);
                return 1;
            }
            p36_tokens_push(&prompt, tok);
        }
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        int rc = prompt.len ? p36_session_sync(s, &prompt, err, errlen) : 0;
        if (!prompt.len) p36_session_reset_runtime(s);
        p36_tokens_free(&prompt);
        if (rc != 0) return 1;
        s->checkpoint_valid = true;
        return 0;
    }
    if (version != P36_PAYLOAD_VERSION && version != P36_PAYLOAD_VERSION_TYPED_KV) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload version");
        return 1;
    }

    uint32_t nfields = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        P36_PAYLOAD_U32_FIELDS_TYPED_KV : P36_PAYLOAD_U32_FIELDS;
    uint32_t h[P36_PAYLOAD_U32_FIELDS_TYPED_KV] = {0};
    h[0] = magic;
    h[1] = version;
    h[2] = third;
    for (uint32_t i = 3; i < nfields; i++) {
        if (p36_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    uint32_t saved_ctx = h[2];
    uint32_t n_tokens = h[4];
    if (saved_ctx > (uint32_t)s->ctx_size || n_tokens >= (uint32_t)s->ctx_size) {
        p36_payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[5] != P36_N_VOCAB || h[6] != P36_N_LAYER ||
        h[7] != P36_N_HEAD_KV || h[8] != P36_N_HEAD_DIM ||
        h[9] != P36_N_VALUE_DIM || h[10] != P36_N_SSM_CONV ||
        h[11] != P36_N_SSM_CONV_DIM || h[12] != P36_N_SSM_STATE ||
        h[13] != P36_N_SSM_DT_RANK)
    {
        p36_payload_set_err(err, errlen, "KV checkpoint was written for a different p36 layout");
        return 1;
    }
    p36_kv_cache_type payload_type_k = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        (p36_kv_cache_type)h[14] : P36_KV_CACHE_F16;
    p36_kv_cache_type payload_type_v = version == P36_PAYLOAD_VERSION_TYPED_KV ?
        (p36_kv_cache_type)h[15] : P36_KV_CACHE_F16;
    if (payload_type_k > P36_KV_CACHE_Q4_0 || payload_type_v > P36_KV_CACHE_Q4_0 ||
        !s->engine ||
        s->engine->cache_type_k != payload_type_k ||
        s->engine->cache_type_v != payload_type_v)
    {
        p36_payload_set_err(err, errlen, "KV checkpoint was written with a different cache type");
        return 1;
    }

    p36_session_reset_runtime(s);
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    if (!p36_session_reserve_kv(s, n_tokens, 0)) {
        p36_payload_set_err(err, errlen, "failed to grow KV cache for checkpoint");
        return 1;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        int32_t tok = 0;
        if (p36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) return 1;
        p36_tokens_push(&s->checkpoint, tok);
    }
    if (s->engine->backend == P36_BACKEND_CPU) {
        p36_cpu_runtime *rt = (p36_cpu_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_read_bytes(fp, s->logits,
                                   (uint64_t)P36_N_VOCAB * sizeof(float),
                                   &remaining, err, errlen) != 0) return 1;
        s->logits_host_valid = true;
        s->gpu_top2_valid = false;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                p36_full_attn_cache *c = &rt->full[il];
                if (p36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    p36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                c->len = rows;
                if (p36_payload_read_bytes(fp, c->k, p36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    p36_payload_read_bytes(fp, c->v, p36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                p36_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_read_bytes(fp, c->conv, p36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    p36_payload_read_bytes(fp, c->state, p36_recurrent_state_bytes(), &remaining, err, errlen) != 0)
                    return 1;
            }
        }
    } else {
#ifdef P36_NO_GPU
        return p36_unsupported_runtime(err, errlen);
#else
        p36_vulkan_runtime *rt = (p36_vulkan_runtime *)s->runtime;
        if (!rt) return p36_unsupported_runtime(err, errlen);
        if (p36_payload_read_tensor(rt->logits, fp,
                                    (uint64_t)P36_N_VOCAB * sizeof(float),
                                    &remaining, err, errlen) != 0) return 1;
        s->logits_host_valid = false;
        s->gpu_top2_valid = false;
        for (uint32_t il = 0; il < P36_N_LAYER; il++) {
            if (p36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                p36_vulkan_full_attn_cache *c = &rt->full[il];
                if (p36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    p36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                if (p36_payload_read_tensor(c->k, fp, p36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    p36_payload_read_tensor(c->v, fp, p36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                p36_vulkan_recurrent_cache *c = &rt->recurrent[il];
                if (p36_payload_read_tensor(c->conv, fp, p36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    p36_payload_read_recurrent_state(c->state, fp,
                                                     rt->recur_state_f16,
                                                     &remaining, err, errlen) != 0)
                    return 1;
            }
        }
#endif
    }
    if (remaining != 0) {
        p36_payload_set_err(err, errlen, "trailing bytes in session payload");
        p36_session_invalidate(s);
        return 1;
    }
    s->checkpoint_valid = true;
    return 0;
}

int p36_session_save_snapshot(p36_session *s, p36_session_snapshot *snap, char *err, size_t errlen) {
    p36_payload_header hdr;
    uint64_t bytes;
    uint8_t *p;
    if (!s || !snap) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot save arguments");
        return 1;
    }
    bytes = sizeof(hdr) + (uint64_t)s->checkpoint.len * sizeof(int32_t);
    if (snap->cap < bytes) {
        snap->ptr = xrealloc(snap->ptr, (size_t)bytes);
        snap->cap = bytes;
    }
    hdr.magic = P36_PAYLOAD_MAGIC;
    hdr.version = P36_PAYLOAD_VERSION_TOKEN_ONLY;
    hdr.n_tokens = (uint32_t)s->checkpoint.len;
    memcpy(snap->ptr, &hdr, sizeof(hdr));
    p = snap->ptr + sizeof(hdr);
    if (hdr.n_tokens) memcpy(p, s->checkpoint.v, (size_t)hdr.n_tokens * sizeof(int32_t));
    snap->len = bytes;
    (void)err;
    (void)errlen;
    return 0;
}

int p36_session_load_snapshot(p36_session *s, const p36_session_snapshot *snap, char *err, size_t errlen) {
    const p36_payload_header *hdr;
    const int32_t *tok;
    if (!s || !snap || !snap->ptr || snap->len < sizeof(p36_payload_header)) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot payload");
        return 1;
    }
    hdr = (const p36_payload_header *)snap->ptr;
    if (hdr->magic != P36_PAYLOAD_MAGIC ||
        hdr->version != P36_PAYLOAD_VERSION_TOKEN_ONLY)
    {
        if (err && errlen) snprintf(err, errlen, "unsupported snapshot format");
        return 1;
    }
    if (snap->len != sizeof(*hdr) + (uint64_t)hdr->n_tokens * sizeof(int32_t)) {
        if (err && errlen) snprintf(err, errlen, "mismatched snapshot size");
        return 1;
    }
    tok = (const int32_t *)(snap->ptr + sizeof(*hdr));
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    for (uint32_t i = 0; i < hdr->n_tokens; i++) p36_tokens_push(&s->checkpoint, tok[i]);
    if (s->checkpoint.len > 0) {
        p36_tokens prompt = {0};
        p36_tokens_copy(&prompt, &s->checkpoint);
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        if (p36_session_sync(s, &prompt, err, errlen) != 0) {
            p36_tokens_free(&prompt);
            return 1;
        }
        p36_tokens_free(&prompt);
    } else {
        p36_session_reset_runtime(s);
    }
    s->checkpoint_valid = true;
    return 0;
}

void p36_session_snapshot_free(p36_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

int p36_qwen35_n_layer(void) {
    return P36_N_LAYER;
}

int p36_qwen35_n_embd(void) {
    return P36_N_EMBD;
}

int p36_qwen35_n_vocab(void) {
    return P36_N_VOCAB;
}

int p36_engine_prefill_layers(p36_engine *e,
                              const p36_tokens *prompt,
                              int ctx_size,
                              float *layer_out,
                              float *logits,
                              char *err,
                              size_t errlen) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    (void)layer_out;
    (void)logits;
    return p36_unsupported_runtime(err, errlen);
}

int p36_compare_prefill_layers(p36_engine *p36_e,
                               p36_engine *other_e,
                               const p36_tokens *prompt,
                               int ctx_size,
                               float *p36_layer_out,
                               float *other_layer_out,
                               float *p36_logits,
                               float *other_logits,
                               char *err,
                               size_t errlen) {
    (void)p36_e;
    (void)other_e;
    (void)prompt;
    (void)ctx_size;
    (void)p36_layer_out;
    (void)other_layer_out;
    (void)p36_logits;
    (void)other_logits;
    return p36_unsupported_runtime(err, errlen);
}
