static bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    if (!mkdir_p(dir)) {
        server_log(P36_LOG_DEFAULT, "p36-server: failed to create KV cache directory %s: %s", dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = xstrdup(dir);
    if (budget_mb == 0) budget_mb = KV_CACHE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    kv_cache_evict(kc, NULL);
    server_log(P36_LOG_KVCACHE,
               "p36-server: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d)",
               kc->dir,
               (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
               reject_different_quant ? "reject" : "accept",
               kc->opt.min_tokens,
               kc->opt.cold_max_tokens,
               kc->opt.continued_interval_tokens,
               kc->opt.boundary_trim_tokens,
               kc->opt.boundary_align_tokens);
    return true;
}

static void kv_cache_close(kv_disk_cache *kc) {
    kv_cache_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

static char *render_tokens_text(p36_engine *engine, const p36_tokens *tokens, size_t *out_len) {
    buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = p36_token_text(engine, tokens->v[i], &len);
        buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return buf_take(&b);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

static void tokens_copy_prefix(p36_tokens *dst, const p36_tokens *src, int n) {
    dst->len = 0;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) p36_tokens_push(dst, src->v[i]);
}

static void tokens_append(p36_tokens *dst, const p36_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) p36_tokens_push(dst, src->v[i]);
}

static void build_prompt_from_exact_prefix_and_text_suffix(
        p36_engine *engine,
        const p36_tokens *exact_prefix,
        const char *suffix_text,
        p36_tokens *out)
{
    p36_tokens_copy(out, exact_prefix);

    p36_tokens suffix = {0};
    /* The suffix may start with Qwen chat markers such as <|im_start|> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    p36_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    p36_tokens_free(&suffix);
}

static int kv_cache_store_len(const kv_disk_cache *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

static int kv_cache_continued_step(const kv_disk_cache *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

static int kv_cache_continued_store_target_for(const kv_disk_cache *kc,
                                               int last_store_tokens,
                                               int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;

    /* Continued checkpoints cannot roll the live graph backward.  Save only
     * at absolute aligned frontiers, not relative to the last cold/evict file.
     * Otherwise an early cold checkpoint can shift the whole schedule and leave
     * long generations with no recent durable restart point. */
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= last_store_tokens) return 0;
    return live_tokens;
}

#ifdef P36_SERVER_TEST
static int kv_cache_continued_store_target(const kv_disk_cache *kc,
                                           int live_tokens) {
    return kv_cache_continued_store_target_for(
            kc, kc ? kc->continued_last_store_tokens : 0, live_tokens);
}
#endif

/* A same-text-prefix file can be reused by a larger context, but not by a
 * smaller one: the payload was validated against the context capacity recorded
 * in the file.  If the existing file cannot be used by this server, replace it
 * so this context can still populate its own cache. */
static bool kv_cache_file_text_matches(const char *path, const char *ident, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    kv_cache_text_sha(ident, stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_existing_compatible(kv_disk_cache *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    kv_entry e = {0};
    if (!kv_read_entry_file(path, sha, &e)) return false;
    bool compatible = (!kc->reject_different_quant || e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, kc->ident, sha, text, text_len);
    kv_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            server_log(P36_LOG_KVCACHE, "p36-server: kv cache replaced incompatible file %s", path);
        }
        return false;
    }
    return true;
}

static void kv_cache_rewrite_tool_map(server *s, const char *path, const char *text) {
    if (!s || !path || !text || tool_memory_count_qwen_tool_in_text(s, text) == 0) return;
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes);
    uint64_t end = KV_CACHE_FIXED_HEADER + 4ull + (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_tool_map_write(s, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[KV_CACHE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            kv_fill_header(h, hdr.quant_bits, hdr.reason,
                           (uint8_t)(hdr.ext_flags | KV_EXT_TOOL_MAP),
                           hdr.tokens, hdr.hits, hdr.ctx_size,
                           hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)ok;
}

static bool kv_cache_store_live_prefix(server *s, server_slot *slot,
                                       const p36_tokens *tokens,
                                       int store_len, const char *reason) {
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled || p36_session_has_vision_state(slot->session)) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    p36_tokens store_tokens = {0};
    tokens_copy_prefix(&store_tokens, tokens, store_len);

    const int quant_bits = p36_engine_routed_quant_bits(s->engine);
    if (!kv_quant_bits_valid(quant_bits)) {
        p36_tokens_free(&store_tokens);
        return false;
    }
    char err[160] = {0};
    /* Disk cache persistence must observe the graph exactly as-is.  If callers
     * want a shorter prefix, they first prefill to that prefix and only then call
     * this function.  This keeps cache population from doing hidden inference. */
    const p36_tokens *live_tokens = p36_session_tokens(slot->session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !p36_tokens_starts_with(live_tokens, &store_tokens))
    {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                   store_tokens.len,
                   reason,
                   live_tokens ? live_tokens->len : -1);
        p36_tokens_free(&store_tokens);
        return false;
    }

    uint64_t payload_bytes = p36_session_payload_bytes(slot->session);
    if (payload_bytes == 0) {
        p36_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = render_tokens_text(s->engine, &store_tokens, &text_len);
    if (text_len > UINT32_MAX) {
        server_log(P36_LOG_KVCACHE, "p36-server: kv cache skipped tokens=%d because rendered text is too large", store_tokens.len);
        free(text);
        p36_tokens_free(&store_tokens);
        return false;
    }

    char sha[41];
    kv_cache_text_sha(kc->ident, text, text_len, sha);
    char *path = kv_path_for_sha(kc, sha);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     quant_bits, p36_session_ctx(slot->session))) {
        kv_cache_rewrite_tool_map(s, path, text);
        free(text);
        free(path);
        p36_tokens_free(&store_tokens);
        return true;
    }

    buf tmpb = {0};
    buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = buf_take(&tmpb);
    const double save_t0 = now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        server_log(P36_LOG_KVCACHE, "p36-server: kv cache failed to create %s: %s save=%.1f ms",
                   tmp, strerror(errno), (now_sec() - save_t0) * 1000.0);
        free(tmp);
        free(text);
        free(path);
        p36_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[KV_CACHE_FIXED_HEADER];
    uint8_t ext_flags = tool_memory_count_qwen_tool_in_text(s, text) > 0 ? KV_EXT_TOOL_MAP : 0;
    kv_fill_header(h, (uint8_t)quant_bits, kv_reason_code(reason), ext_flags,
                   (uint32_t)store_tokens.len, 0,
                   (uint32_t)p36_session_ctx(slot->session), now, now, payload_bytes);
    uint8_t tb[4];
    le_put32(tb, (uint32_t)text_len);
    uint64_t tool_map_bytes = 0;
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              p36_session_save_payload(slot->session, fp, err, sizeof(err)) == 0 &&
              kv_tool_map_write(s, fp, text, &tool_map_bytes) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (now_sec() - save_t0) * 1000.0;
    if (!ok) {
        server_log(P36_LOG_KVCACHE, "p36-server: kv cache store failed (%s): %s save=%.1f ms",
                   reason,
                   saved_errno ? strerror(saved_errno) : (err[0] ? err : "unknown error"),
                   save_ms);
        unlink(tmp);
    } else {
        server_log(P36_LOG_KVCACHE,
                   "p36-server: kv cache stored tokens=%d trimmed=%d reason=%s size=%.2f MiB save=%.1f ms",
                   store_tokens.len,
                   original_len - store_tokens.len,
                   reason,
                   (double)(KV_CACHE_FIXED_HEADER + 4ull + text_len + payload_bytes + tool_map_bytes) / (1024.0 * 1024.0),
                   save_ms);
        kv_cache_evict(kc, live_tokens);
    }
    free(tmp);
    free(text);
    free(path);
    p36_tokens_free(&store_tokens);
    return ok;
}

static void kv_cache_store_current(server *s, server_slot *slot,
                                   const char *reason) {
    const p36_tokens *tokens = p36_session_tokens(slot->session);
    if (tokens) kv_cache_store_live_prefix(s, slot, tokens, tokens->len, reason);
}

static void kv_cache_note_store(server_slot *slot, int tokens) {
    if (tokens > slot->continued_last_store_tokens) {
        slot->continued_last_store_tokens = tokens;
    }
}

static void kv_cache_maybe_store_continued(server *s, server_slot *slot) {
    kv_disk_cache *kc = &s->kv;
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    const p36_tokens *tokens = p36_session_tokens(slot->session);
    if (!tokens) goto done;
    const int target = kv_cache_continued_store_target_for(
            kc, slot->continued_last_store_tokens, tokens->len);
    if (target == 0) goto done;
    if (kv_cache_store_live_prefix(s, slot, tokens, target, "continued")) {
        kv_cache_note_store(slot, target);
    }
done:
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
}

static int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        kv_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            kv_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        kv_cache_text_sha(kc->ident, prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

static void strip_assistant_reasoning_blocks(buf *out, const char *text, size_t len);

static bool kv_read_entry_text(const kv_entry *e, char **out_text, uint32_t *out_len) {
    if (!e || !e->path) return false;
    FILE *fp = fopen(e->path, "rb");
    if (!fp) return false;
    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes);
    char *text = NULL;
    if (ok) {
        text = xmalloc((size_t)text_bytes + 1);
        ok = fread(text, 1, text_bytes, fp) == text_bytes;
        if (ok) text[text_bytes] = '\0';
    }
    fclose(fp);
    if (!ok) {
        free(text);
        return false;
    }
    *out_text = text;
    if (out_len) *out_len = text_bytes;
    return true;
}

/* Fallback for clients that drop reasoning_content: an entry's stored text
 * carries the sampled reasoning block, but the resent prompt does not, so the
 * full-text hash (kv_cache_find_text_prefix) misses.  Match the
 * reasoning-stripped stored text as a literal prompt prefix instead, mirroring
 * live_text_prefix_prompt for the in-memory checkpoint. */
static int kv_cache_find_stripped_prefix(kv_disk_cache *kc, const char *prompt_text,
                                         int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < kc->len; i++) {
        kv_entry *e = &kc->entry[i];
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (e->text_bytes == 0) continue;
        char *text = NULL;
        uint32_t text_bytes = 0;
        if (!kv_read_entry_text(e, &text, &text_bytes)) continue;
        /* Reject a checkpoint captured before a turn generated anything: its
         * text ends on an open generation prompt, and a reasoning-less suffix
         * would be prefilled into that think block as if it were reasoning. */
        static const char open_gen[] = "<|im_start|>assistant\n<think>\n";
        if (text_bytes >= sizeof(open_gen) - 1 &&
            !memcmp(text + text_bytes - (sizeof(open_gen) - 1), open_gen, sizeof(open_gen) - 1)) {
            free(text);
            continue;
        }
        buf stripped = {0};
        strip_assistant_reasoning_blocks(&stripped, text, text_bytes);
        bool ok = stripped.len > 0 && stripped.len < text_bytes &&
                  stripped.len <= prompt_bytes &&
                  byte_prefix_match(prompt_text, prompt_bytes, stripped.ptr, stripped.len);
        if (ok && (best < 0 || stripped.len > best_len ||
                   (stripped.len == best_len && e->tokens > kc->entry[best].tokens))) {
            best = i;
            best_len = stripped.len;
        }
        buf_free(&stripped);
        free(text);
    }
    return best;
}

static int kv_cache_try_load_text(server *s, server_slot *slot,
                                  const char *prompt_text, bool allow_stripped,
                                  p36_tokens *effective_prompt,
                                  char **loaded_path_out) {
    if (loaded_path_out) *loaded_path_out = NULL;
    if (effective_prompt) effective_prompt->len = 0;
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = p36_engine_routed_quant_bits(s->engine);
    if (!kv_quant_bits_valid(quant_bits)) return 0;
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = kv_cache_find_text_prefix(kc, prompt_text, quant_bits,
                                        p36_session_ctx(slot->session));
    bool stripped_mode = false;
    if (idx < 0 && allow_stripped) {
        idx = kv_cache_find_stripped_prefix(kc, prompt_text, quant_bits,
                                            p36_session_ctx(slot->session));
        stripped_mode = idx >= 0;
    }
    if (idx < 0) return 0;

    kv_entry e = kc->entry[idx];
    char *path = xstrdup(e.path);
    const double load_t0 = now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    kv_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = kv_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    buf stripped_text = {0};
    size_t prefix_bytes = text_bytes;
    if (header_ok) {
        cached_text = xmalloc((size_t)text_bytes + 1);
        if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
            header_ok = false;
            fail_reason = "truncated cached text";
        } else {
            cached_text[text_bytes] = '\0';
            char text_sha[41];
            kv_cache_text_sha(kc->ident, cached_text, text_bytes, text_sha);
            if (strcmp(text_sha, e.sha)) {
                header_ok = false;
                fail_reason = "cached text hash mismatch";
            } else if (stripped_mode) {
                /* The stored text still carries the sampled reasoning, so it can
                 * be longer than the resent prompt; match its stripped form. */
                strip_assistant_reasoning_blocks(&stripped_text, cached_text, text_bytes);
                if (stripped_text.len == 0 || stripped_text.len >= text_bytes ||
                    !byte_prefix_match(prompt_text, prompt_bytes,
                                       stripped_text.ptr, stripped_text.len)) {
                    header_ok = false;
                    fail_reason = "cached stripped text prefix mismatch";
                } else {
                    prefix_bytes = stripped_text.len;
                }
            } else if (!byte_prefix_match(prompt_text, prompt_bytes,
                                          cached_text, text_bytes)) {
                header_ok = false;
                fail_reason = "cached text prefix mismatch";
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    if (header_ok && p36_session_load_payload(slot->session, fp, hdr.payload_bytes, err, sizeof(err)) == 0) {
        const p36_tokens *loaded_tokens = p36_session_tokens(slot->session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt we give p36_session_sync() from that exact history and
                 * tokenize only the text suffix after the byte prefix. */
                build_prompt_from_exact_prefix_and_text_suffix(
                    s->engine, loaded_tokens, prompt_text + prefix_bytes,
                    effective_prompt);
            }
            if (hdr.ext_flags & KV_EXT_TOOL_MAP) kv_tool_map_load_from_pos(s, fp, NULL);
        } else {
            p36_session_invalidate(slot->session);
            unlink(path);
            server_log(P36_LOG_KVCACHE, "p36-server: kv cache discarded corrupt text-prefix payload %s", path);
        }
    } else {
        if (header_ok) p36_session_invalidate(slot->session);
        server_log(P36_LOG_KVCACHE,
                   "p36-server: kv cache load failed %s: %s load=%.1f ms",
                   path,
                   header_ok ? err : fail_reason,
                   (now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (now_sec() - load_t0) * 1000.0;
        if (loaded_path_out) *loaded_path_out = xstrdup(path);
        slot->continued_last_store_tokens = loaded;
        kv_cache_touch_file(path, hdr.hits + 1);
        server_log(P36_LOG_KVCACHE,
                   "p36-server: kv cache hit text tokens=%d text=%u%s quant=%u load=%.1f ms file=%s",
                   loaded, text_bytes, stripped_mode ? " stripped=1" : "",
                   hdr.quant_bits, load_ms, path);
    }
    buf_free(&stripped_text);
    free(cached_text);
    free(path);
    return loaded;
}

/* A thinking-on assistant turn is rendered as
 *   "<|im_start|>assistant\n<think>\nREASONING\n</think>\n\nCONTENT<|im_end|>\n"
 * and the live graph samples exactly that block.  Clients that drop
 * reasoning_content from resent history render the turn without it, so the
 * live text is no longer a byte prefix and the whole conversation would be
 * re-prefilled.  Removing the block recovers the bytes those clients rendered,
 * keeping the live checkpoint authoritative for the reasoning state. */
static void strip_assistant_reasoning_blocks(buf *out, const char *text, size_t len) {
    static const char assistant[] = "<|im_start|>assistant\n";
    static const char open_tag[] = "<think>\n";
    static const char close_tag[] = "</think>";
    const char *p = text;
    const char *end = text + len;
    while (p < end) {
        const char *block = strstr(p, "<|im_start|>assistant\n<think>\n");
        if (!block) break;
        const char *body = block + strlen(assistant) + strlen(open_tag);
        const char *close_at = strstr(body, close_tag);
        const char *end_mark = strstr(body, "<|im_end|>");
        const char *next_mark = strstr(body, "<|im_start|>");
        if (next_mark && (!end_mark || next_mark < end_mark)) end_mark = next_mark;
        buf_append(out, p, (size_t)(block - p));
        buf_puts(out, assistant);
        const char *after;
        if (close_at && (!end_mark || close_at < end_mark)) {
            after = close_at + strlen(close_tag);
            for (int nl = 0; nl < 2 && after < end && *after == '\n'; nl++) after++;
        } else {
            /* A turn cut off at max_tokens has no closing tag and no content;
             * its reasoning ends at the next structural marker. */
            after = end_mark ? end_mark : end;
        }
        p = after;
    }
    if (p < end) buf_append(out, p, (size_t)(end - p));
}

static int live_text_prefix_prompt(server *s, server_slot *slot,
                                   const request *req,
                                   p36_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    const p36_tokens *live_tokens = p36_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, live_tokens, &live_text_len);
    const size_t prompt_text_len = strlen(req->prompt_text);
    /* This is the core text-prefix case.  The live graph is authoritative, so
     * keep its sampled tokenization and tokenize only the request bytes that
     * come after it.  Reusing req->prompt's token suffix would be wrong: full
     * prompt BPE may have merged across this byte boundary. */
    const char *suffix = NULL;
    buf stripped = {0};
    if (byte_prefix_match(req->prompt_text, prompt_text_len, live_text, live_text_len)) {
        suffix = req->prompt_text + live_text_len;
    } else if (req->preserve_thinking) {
        strip_assistant_reasoning_blocks(&stripped, live_text, live_text_len);
        if (stripped.len > 0 && stripped.len < live_text_len &&
            byte_prefix_match(req->prompt_text, prompt_text_len, stripped.ptr, stripped.len) &&
            /* If the client sent the reasoning back, the suffix would restart
             * the think block and duplicate the reasoning already in the live
             * graph; leave that case to the cold path. */
            strncmp(req->prompt_text + stripped.len, "<think>", 7) != 0)
            suffix = req->prompt_text + stripped.len;
    }
    if (!suffix) {
        buf_free(&stripped);
        free(live_text);
        return 0;
    }
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, suffix, effective_prompt);
    buf_free(&stripped);
    free(live_text);
    return live_tokens->len;
}

static bool server_kv_store_live_prefix(server *s, server_slot *slot,
                                        const p36_tokens *tokens,
                                        int store_len, const char *reason) {
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    bool ok = kv_cache_store_live_prefix(s, slot, tokens, store_len, reason);
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
    return ok;
}

static void server_kv_store_current(server *s, server_slot *slot,
                                    const char *reason) {
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    kv_cache_store_current(s, slot, reason);
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
}

static int server_kv_try_load_text(server *s, server_slot *slot,
                                   const char *prompt_text, bool allow_stripped,
                                   p36_tokens *effective_prompt,
                                   char **loaded_path_out) {
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    int loaded = kv_cache_try_load_text(s, slot, prompt_text, allow_stripped,
                                        effective_prompt, loaded_path_out);
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
    return loaded;
}

static int server_kv_try_load(server *s, server_slot *slot,
                              const request *req,
                              p36_tokens *effective_prompt,
                              char **loaded_path_out) {
    return server_kv_try_load_text(s, slot,
            req ? req->prompt_text : NULL, req ? req->preserve_thinking : false,
            effective_prompt, loaded_path_out);
}

/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical QWEN_TOOL block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

static void trace_cache_capture(
        trace_cache_diag *d,
        const p36_tokens *live,
        const p36_tokens *prompt,
        int old_pos,
        int common)
{
    memset(d, 0, sizeof(*d));
    d->valid = true;
    d->old_pos = old_pos;
    d->prompt_len = prompt ? prompt->len : 0;
    d->common = common;

    const int live_len = live ? live->len : 0;
    const int prompt_len = prompt ? prompt->len : 0;
    int max_len = live_len > prompt_len ? live_len : prompt_len;
    int start = common - TRACE_CACHE_BEFORE;
    if (start < 0) start = 0;
    int end = common + TRACE_CACHE_AFTER + 1;
    if (end > max_len) end = max_len;
    if (end < start) end = start;

    d->start = start;
    d->count = end - start;
    if (d->count > TRACE_CACHE_WINDOW) d->count = TRACE_CACHE_WINDOW;
    for (int i = 0; i < d->count; i++) {
        int pos = start + i;
        d->live_id[i] = live && pos < live->len ? live->v[pos] : -1;
        d->prompt_id[i] = prompt && pos < prompt->len ? prompt->v[pos] : -1;
    }
}

static const char *trace_cache_miss_reason(const trace_cache_diag *d) {
    if (!d || !d->valid) return "unknown";
    if (d->old_pos == 0) return "no-live-checkpoint";
    if (d->common != d->old_pos) return "token-mismatch";
    if (d->prompt_len < d->old_pos) return "incoming-prompt-shorter-than-live-checkpoint";
    return "live-prefix-match";
}

static void trace_write_escaped_bytes(FILE *fp, const char *p, size_t len) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)p[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20 || c == 0x7f) {
            fputs("\\x", fp);
            fputc(hex[c >> 4], fp);
            fputc(hex[c & 15], fp);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void trace_write_token(FILE *fp, p36_engine *engine, int token) {
    if (token < 0) {
        fputs("- <none>", fp);
        return;
    }
    size_t len = 0;
    char *piece = p36_token_text(engine, token, &len);
    fprintf(fp, "%d ", token);
    trace_write_escaped_bytes(fp, piece, len);
    free(piece);
}

static void trace_write_cache_diag(
        server *s,
        const trace_cache_diag *d,
        const tool_replay_stats *tool_replay,
        int cached,
        const char *cache_source,
        int disk_cached,
        const char *disk_path)
{
    fprintf(s->trace,
            "\n--- cache decision ---\n"
            "live_tokens_before: %d\n"
            "prompt_tokens: %d\n"
            "live_prompt_common: %d\n"
            "memory_token_reusable: %d\n"
            "memory_miss_reason: %s\n"
            "tool_replay: mem=%d disk=%d canonical=%d missing_ids=%d\n"
            "cache_source: %s\n"
            "cached_tokens: %d\n"
            "disk_cached_tokens: %d\n",
            d && d->valid ? d->old_pos : 0,
            d && d->valid ? d->prompt_len : 0,
            d && d->valid ? d->common : 0,
            d && d->valid && d->old_pos > 0 &&
                d->common == d->old_pos && d->prompt_len >= d->old_pos ? 1 : 0,
            trace_cache_miss_reason(d),
            tool_replay ? tool_replay->mem : 0,
            tool_replay ? tool_replay->disk : 0,
            tool_replay ? tool_replay->canonical : 0,
            tool_replay ? tool_replay->missing_ids : 0,
            cache_source ? cache_source : "none",
            cached,
            disk_cached);
    if (disk_path && disk_path[0]) fprintf(s->trace, "disk_cache_file: %s\n", disk_path);

    if (!d || !d->valid || d->old_pos == 0 ||
        (d->common == d->old_pos && d->prompt_len >= d->old_pos))
    {
        return;
    }

    fprintf(s->trace,
            "\nfirst_mismatch_token: %d\n"
            "token_window: [%d..%d)\n",
            d->common,
            d->start,
            d->start + d->count);
    for (int i = 0; i < d->count; i++) {
        int pos = d->start + i;
        int live = d->live_id[i];
        int prompt = d->prompt_id[i];
        const char *mark;
        if (live < 0) mark = "prompt-only";
        else if (prompt < 0) mark = "live-only";
        else mark = live == prompt ? "==" : "!=";

        fprintf(s->trace, "%7d %-11s live ", pos, mark);
        trace_write_token(s->trace, s->engine, live);
        fputs(" | prompt ", s->trace);
        trace_write_token(s->trace, s->engine, prompt);
        fputc('\n', s->trace);
    }
}

static void trace_time(FILE *fp) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    fputs(buf, fp);
}

static uint64_t trace_begin(
        server *s,
        const job *j,
        int cached,
        int effective_prompt_tokens,
        const trace_cache_diag *cache_diag,
        const char *cache_source,
        int disk_cached,
        const char *disk_path) {
    if (!s->trace) return 0;

    pthread_mutex_lock(&s->trace_mu);
    uint64_t id = ++s->trace_seq;
    fprintf(s->trace, "\n===== request %llu ", (unsigned long long)id);
    trace_time(s->trace);
    fprintf(s->trace,
            " =====\nkind: %s\nmodel: %s\nstream: %d\ntools: %d\nthink_mode: %s\nthinking_budget: %d\npreserve_thinking: %d\nprompt_tokens: %d\neffective_prompt_tokens: %d\ncached_tokens: %d\nmax_tokens: %d\ntemperature: %.3f\ntop_k: %d\ntop_p: %.3f\nmin_p: %.3f\npresence_penalty: %.3f\nfrequency_penalty: %.3f\nrepeat_penalty: %.3f\nseed: %llu\n",
            j->req.kind == REQ_CHAT ? "chat" : "completion",
            j->req.model ? j->req.model : "",
            j->req.stream ? 1 : 0,
            j->req.has_tools ? 1 : 0,
            p36_think_mode_name(j->req.think_mode),
            j->req.thinking_budget,
            j->req.preserve_thinking ? 1 : 0,
            j->req.prompt.len,
            effective_prompt_tokens,
            cached,
            j->req.max_tokens,
            j->req.temperature,
            j->req.top_k,
            j->req.top_p,
            j->req.min_p,
            j->req.presence_penalty,
            j->req.frequency_penalty,
            j->req.repeat_penalty,
            (unsigned long long)j->req.seed);
    fprintf(s->trace, "stream_include_usage: %d\n",
            j->req.stream_include_usage ? 1 : 0);
    trace_write_cache_diag(s, cache_diag, &j->req.tool_replay, cached,
                           cache_source, disk_cached, disk_path);
    if (j->req.raw_body) {
        fputs("\n--- raw request json ---\n", s->trace);
        fputs(j->req.raw_body, s->trace);
        if (!j->req.raw_body[0] || j->req.raw_body[strlen(j->req.raw_body) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    if (j->req.prompt_text) {
        fputs("\n--- rendered prompt ---\n", s->trace);
        fputs(j->req.prompt_text, s->trace);
        if (!j->req.prompt_text[0] || j->req.prompt_text[strlen(j->req.prompt_text) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    fputs("\n--- generated text ---\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
    return id;
}

static void trace_piece(server *s, uint64_t id, const char *piece, size_t len) {
    if (!s->trace || !id || !piece || !len) return;
    pthread_mutex_lock(&s->trace_mu);
    fwrite(piece, 1, len, s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

