static void generate_job(server *s, server_slot *slot, job *j) {
    char err[160];
    err[0] = '\0';
    bool continued_vision = request_continue_vision(s, slot, &j->req);
    slot_pending_clear(slot);
    if (continued_vision && request_exceeds_context(&j->req, p36_session_ctx(slot->session))) {
        http_error_context_length_exceeded(j->fd, &j->req, j->req.prompt.len,
                                           p36_session_ctx(slot->session));
        return;
    }
    const bool multimodal = j->req.image_count != 0;
    if ((multimodal || p36_session_has_vision_state(slot->session)) &&
        !p36_session_vision_prefix_matches(slot->session, j->req.images, j->req.image_count)) {
        pthread_mutex_lock(&s->inference_mu);
        p36_session_invalidate(slot->session);
        pthread_mutex_unlock(&s->inference_mu);
    }
    const int old_pos = p36_session_pos(slot->session);
    const int common = p36_session_common_prefix(slot->session, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    if (s->trace) {
        trace_cache_capture(&cache_diag, p36_session_tokens(slot->session),
                            &j->req.prompt, old_pos, common);
    }
    p36_tokens effective_prompt = {0};
    const p36_tokens *prompt_for_sync = &j->req.prompt;
    int cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
    const char *cache_source = cached > 0 ? "memory-token" : "none";
    int disk_cached = 0;
    char *disk_cache_path = NULL;
    if (cached == 0) {
        int text_cached = live_text_prefix_prompt(s, slot, &j->req, &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            int delta = effective_prompt.len - j->req.prompt.len;
            for (size_t i = 0; i < j->req.image_count; i++) {
                if (j->req.images[i].token_start >= (uint32_t)cached)
                    j->req.images[i].token_start = (uint32_t)((int)j->req.images[i].token_start + delta);
            }
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0) slot->continued_last_store_tokens = 0;
    if (!multimodal && s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the resident slot. Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        server_kv_store_current(s, slot, "evict");
    }
    const int sys_len = (!multimodal && s->kv.enabled && s->kv.opt.system_checkpoint &&
                         j->req.kind == REQ_CHAT) ? request_system_boundary(s, &j->req) : 0;
    j->req.sys_ckpt.enabled = s->kv.enabled && s->kv.opt.system_checkpoint && j->req.kind == REQ_CHAT;
    j->req.sys_ckpt.tokens = sys_len;
    if (!multimodal && cached == 0) {
        const double restore_t0 = now_sec();
        disk_cached = server_kv_try_load(s, slot, &j->req, &effective_prompt,
                                        &disk_cache_path);
        if (disk_cached > 0 && sys_len > 0 && disk_cached == sys_len) {
            j->req.sys_ckpt.hit = true;
            j->req.sys_ckpt.restore_ms = (now_sec() - restore_t0) * 1000.0;
        }
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    const int prompt_tokens = prompt_for_sync->len;
    j->req.cached_tokens = cached;
    const double t0 = now_sec();
    uint64_t trace_id = trace_begin(s, j, cached, prompt_tokens, &cache_diag,
                                    cache_source, disk_cached, disk_cache_path);
    free(disk_cache_path);
    char ctx_span[48];
    request_ctx_span(ctx_span, sizeof(ctx_span), cached, prompt_tokens);
    server_prefill_progress progress = {
        .srv = s,
        .slot = slot,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .t0 = t0,
    };
    snprintf(progress.ctx, sizeof(progress.ctx), "%s", ctx_span);
    char req_flags[64];
    log_flags(req_flags, sizeof(req_flags), j->req.has_tools, false, false, false);
    server_log(P36_LOG_PREFILL,
               "p36-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags);
    p36_session_set_progress(slot->session, server_progress_cb, &progress);

    /* System+tools checkpoint: prefill to the end of the system turn, store it,
     * then keep going (same forward-only pattern as the cold store below).  A
     * conversation that restored anything from disk or memory skips this. */
    int sys_built_len = 0;
    if (cached == 0 && sys_len > 0 && s->kv.enabled && sys_len >= s->kv.opt.min_tokens &&
        sys_len < prompt_for_sync->len &&
        prompt_for_sync == &j->req.prompt)
    {
        const double build_t0 = now_sec();
        p36_tokens prefix = {0};
        tokens_copy_prefix(&prefix, prompt_for_sync, sys_len);
        if (server_session_sync(s, slot, &prefix, err, sizeof(err)) != 0) {
            p36_tokens_free(&prefix);
            p36_tokens_free(&effective_prompt);
            p36_session_set_progress(slot->session, NULL, NULL);
            trace_event(s, trace_id, "prefill failed: %s", err);
            http_error(j->fd, 500, err);
            return;
        }
        if (server_kv_store_live_prefix(s, slot, prompt_for_sync, sys_len, "system"))
            kv_cache_note_store(slot, sys_len);
        j->req.sys_ckpt.build_ms = (now_sec() - build_t0) * 1000.0;
        sys_built_len = sys_len;
        p36_tokens_free(&prefix);
    }

    int cold_store_len = 0;
    if (!multimodal && cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens &&
        s->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= s->kv.opt.cold_max_tokens)
    {
        cold_store_len = kv_cache_store_len(&s->kv, prompt_for_sync->len);
        /* The live graph only moves forward: a cold prefix shorter than the
         * system checkpoint we just built would force a full re-prefill. */
        if (sys_built_len && cold_store_len <= sys_built_len) cold_store_len = 0;
    }

    if (s->kv.enabled &&
        cold_store_len >= s->kv.opt.min_tokens &&
        cold_store_len < prompt_for_sync->len)
    {
        p36_tokens prefix = {0};
        tokens_copy_prefix(&prefix, prompt_for_sync, cold_store_len);
        if (server_session_sync(s, slot, &prefix, err, sizeof(err)) != 0) {
            p36_tokens_free(&prefix);
            p36_tokens_free(&effective_prompt);
            p36_session_set_progress(slot->session, NULL, NULL);
            trace_event(s, trace_id, "prefill failed: %s", err);
            http_error(j->fd, 500, err);
            return;
        }
        if (server_kv_store_live_prefix(s, slot, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(slot, cold_store_len);
        }
        p36_tokens_free(&prefix);
    }

    if (server_session_sync_multimodal(s, slot, prompt_for_sync,
                                       j->req.images, j->req.image_count, err, sizeof(err)) != 0) {
        p36_tokens_free(&effective_prompt);
        p36_session_set_progress(slot->session, NULL, NULL);
        trace_event(s, trace_id, "prefill failed: %s", err);
        http_error(j->fd, 500, err);
        return;
    }
    p36_session_set_progress(slot->session, NULL, NULL);
    server_log(P36_LOG_PREFILL,
               "p36-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags,
               now_sec() - t0);
    if (cold_store_len == prompt_for_sync->len) {
        if (server_kv_store_live_prefix(s, slot, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(slot, cold_store_len);
        }
    }
    char id[96];
    if (j->req.api == API_RESPONSES) {
        responses_random_id(id, sizeof(id), "resp_");
    } else {
        responses_random_id(id, sizeof(id),
                            j->req.kind == REQ_CHAT ? "chatcmpl-" : "cmpl-");
    }

    bool structured_stream = request_uses_structured_stream(&j->req);
    anthropic_stream anthropic_live = {0};
    openai_stream openai_live = {0};
    const bool openai_live_chat = request_uses_openai_live_stream(&j->req);
    if (j->req.stream) {
        if (!sse_headers(j->fd)) {
            server_log(P36_LOG_GENERATION, "p36-server: %s ctx=%s sse headers failed", j->req.kind == REQ_CHAT ? "chat" : "completion", ctx_span);
            p36_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, id,
                                      prompt_tokens, &anthropic_live)) {
            server_log(P36_LOG_GENERATION, "p36-server: chat ctx=%s anthropic stream start failed", ctx_span);
            p36_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, id, NULL, NULL)) {
            server_log(P36_LOG_GENERATION, "p36-server: chat ctx=%s openai role chunk failed", ctx_span);
            p36_tokens_free(&effective_prompt);
            return;
        }
        if (openai_live_chat) openai_stream_start(&j->req, &openai_live);
    }

    buf text = {0};
    p36_tokens generated_tokens = {0};
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    const char *finish = "length";
    int completion = 0;
    int max_tokens = j->req.max_tokens;
    int room = p36_session_ctx(slot->session) - p36_session_pos(slot->session);
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    size_t tool_scan_from = 0;
    int next_tool_progress = 128;
    int next_decode_log = 50;
    uint64_t rng = j->req.seed;
    if (!rng && !random_bytes(&rng, sizeof(rng)))
        rng = ((uint64_t)time(NULL) << 32) ^ (uint64_t)(uintptr_t)j;
    if (!rng) rng = UINT64_C(0x9e3779b97f4a7c15);
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > room) max_tokens = room;
    trace_event(s, trace_id, "prefill done; decode_max=%d ctx_room=%d", max_tokens, room);
    const double decode_t0 = now_sec();
    double last_decode_log_t = decode_t0;
    int last_decode_log_completion = 0;
    thinking_state thinking = thinking_state_from_prompt(&j->req);
    /* Agent controls: independent thinking and action phases, and an
     * execution boundary at the first complete tool call. */
    agent_action_leak *leak = calloc(1, sizeof(*leak));
    int thinking_tokens = 0;
    int think_close_id = -1;
    if (j->req.thinking_budget > 0) {
        p36_tokens close = {0};
        p36_tokenize_rendered_chat(s->engine, "</think>", &close);
        if (close.len == 1) think_close_id = close.v[0];
        p36_tokens_free(&close);
    }
    const bool thinking_gates_tool_markers = p36_think_mode_enabled(j->req.think_mode);
    bool tool_scan_waiting_for_think_close =
        thinking_gates_tool_markers && thinking.inside;
    size_t think_recovery_scan_from = 0;
    qwen_tool_decode_tracker qwen_tool_tracker;
    qwen_tool_decode_tracker_init(&qwen_tool_tracker);

    server_generation_enter(s);
    while (!g_stop_requested && completion < max_tokens &&
           p36_session_pos(slot->session) < p36_session_ctx(slot->session)) {
        qwen_tool_decode_state qwen_tool_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            qwen_tool_tracker.decode : QWEN_TOOL_DECODE_OUTSIDE;
        const bool in_tool_call = qwen_tool_decode_state_is_tool(qwen_tool_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s, slot);
        }
        float temperature, top_p, min_p;
        int top_k;
        request_sampling(&j->req, &temperature, &top_k, &top_p, &min_p);
        const float payload_temperature = temperature;
        const bool greedy_tool_syntax = in_tool_call &&
            !qwen_tool_decode_state_uses_payload_sampling(qwen_tool_state);
        if (greedy_tool_syntax) temperature = 0.0f;
        float presence_penalty = j->req.presence_penalty;
        float frequency_penalty = j->req.frequency_penalty;
        float repeat_penalty = j->req.repeat_penalty;
        if (in_tool_call && !qwen_tool_decode_state_uses_payload_sampling(qwen_tool_state)) {
            presence_penalty = 0.0f;
            frequency_penalty = 0.0f;
            repeat_penalty = 1.0f;
        }
        int close_rank = 0;
        const bool hard_think_close = think_close_id >= 0 && j->req.thinking_budget_hard &&
            p36_session_in_think(slot->session) && thinking_tokens >= j->req.thinking_budget;
        if (!hard_think_close && think_close_id >= 0 && p36_session_in_think(slot->session)) {
            /* A hard budget (top-level thinking_budget) follows the local agent's
             * schedule: soft </think> bias from 75% of the budget, forced at 100%. */
            const int hb = j->req.thinking_budget;
            const bool hard = j->req.thinking_budget_hard;
            const int soft_start = hard ? (hb <= 1 ? hb : (int)(((int64_t)hb * 3) / 4)) : hb;
            const int basis = hard ? thinking_tokens : completion;
            int limit = p36_think_close_rank_limit(basis, soft_start);
            if (limit > 0 && basis == soft_start)
                trace_event(s, trace_id, "thinking closure ranking starts at token=%d",
                            basis);
            if (limit > 0)
                close_rank = p36_session_token_rank(slot->session, think_close_id, limit);
        }
        int token = close_rank > 0 || hard_think_close ? think_close_id :
            p36_session_sample_penalized(
                slot->session, temperature, top_k, top_p, min_p,
                generated_tokens.v, generated_tokens.len,
                presence_penalty, frequency_penalty, repeat_penalty, &rng);
        if (hard_think_close)
            trace_event(s, trace_id, "hard thinking budget close at thinking_tokens=%d",
                        thinking_tokens);
        else if (close_rank > 0)
            trace_event(s, trace_id, "closing thinking at token=%d rank=%d",
                        completion, close_rank);
        if (j->req.ignore_eos && token == p36_token_eos(s->engine))
            token = p36_session_argmax_penalized_excluding(slot->session, p36_token_eos(s->engine),
                generated_tokens.v, generated_tokens.len, presence_penalty, frequency_penalty,
                repeat_penalty);
        else token = p36_session_eos_to_think_close(slot->session, token);
        if (token < 0) {
            finish = "error";
            snprintf(err, sizeof(err), "failed to select a token");
            break;
        }
        if (token == p36_token_eos(s->engine)) {
            finish = "stop";
            break;
        }

        int toks[17];
        int ntok = 0;
        const int block_start = p36_session_pos(slot->session);
        if (!multimodal && !s->batched_mode && temperature <= 0.0f &&
            presence_penalty == 0.0f && frequency_penalty == 0.0f &&
            repeat_penalty == 1.0f &&
            p36_engine_mtp_draft_tokens(s->engine) > 1 &&
            getenv("P36_MTP_SPEC_DISABLE") == NULL)
        {
            ntok = p36_session_eval_speculative_argmax(slot->session,
                                                       token,
                                                       max_tokens - completion,
                                                       p36_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else {
            if (server_eval_token(s, slot, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        if (!ntok) {
            finish = "error";
            snprintf(err, sizeof(err), "decode returned no tokens");
            break;
        }
        bool stop_decode = false, text_stop = false;
        int kept = 0;
        for (int ti = 0; ti < ntok && completion < max_tokens; ti++) {
            token = toks[ti];
            if (token == p36_token_eos(s->engine)) {
                if (j->req.ignore_eos) break;
                finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = p36_token_text(s->engine, token, &piece_len);
            completion++;
            kept++;
            p36_tokens_push(&generated_tokens, token);

            trace_piece(s, trace_id, piece, piece_len);
            buf_append(&text, piece, piece_len);
            bool was_thinking = thinking.inside;
            if (!qwen_tool_decode_state_is_tool(qwen_tool_tracker.decode))
                thinking_state_feed(&thinking, piece, piece_len);
            if (was_thinking) thinking_tokens++;
            else if (j->req.action_budget > 0 && j->req.kind == REQ_CHAT &&
                     !thinking.inside && !saw_tool_start &&
                     !qwen_tool_decode_state_is_tool(qwen_tool_tracker.decode) &&
                     agent_action_leak_feed(leak, j->req.action_budget, piece, piece_len)) {
                finish = "action_budget";
                stop_decode = true;
                trace_event(s, trace_id, "action_budget leak abort after %d content tokens",
                            leak->tokens);
            }
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking.inside) {
                    qwen_tool_decode_tracker_init(&qwen_tool_tracker);
                    qwen_tool_tracker.pos = text.len;
                } else {
                    if (was_thinking) {
                        const char *end = find_tool_structural_text(text.ptr, "</think>", true);
                        qwen_tool_tracker.pos = end ? (size_t)(end + 8 - text.ptr) : text.len;
                    }
                    qwen_tool_decode_tracker_update(&qwen_tool_tracker, text.ptr, text.len);
                }
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, text.ptr,
                                                stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, text.len);
            if (stream_len > text.len) stream_len = text.len;
            stream_len = utf8_stream_safe_len(text.ptr, plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                stop_scan_from = text.len > hold ? text.len - hold : 0;
            }

            if (j->req.stream && !structured_stream && stream_len > plain_stream_pos) {
                char *delta = xstrndup(text.ptr + plain_stream_pos, stream_len - plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, id, delta, NULL);
                free(delta);
                if (!ok) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, &j->req, id,
                                             &anthropic_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, id,
                                          &openai_live, text.ptr, stream_len,
                                          false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (stop_decode) break;
            if (hit_stop) {
                (void)stop_len;
                finish = "stop";
                text.len = stop_pos;
                text.ptr[text.len] = '\0';
                p36_session_invalidate(slot->session);
                text_stop = true;
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking_gates_tool_markers && thinking.inside) {
                    if (complete_tool_call_inside_thinking(
                            text.ptr, text.len, &think_recovery_scan_from)) {
                        saw_tool_start = true;
                        saw_tool_end = true;
                        finish = "tool_calls";
                        stop_decode = true;
                        server_log(P36_LOG_WARNING,
                                   "p36-server: chat ctx=%s recovered a complete tool call "
                                   "from unclosed reasoning after %d generated tokens",
                                   ctx_span, completion);
                        trace_event(s, trace_id,
                                    "recovered complete tool call from unclosed reasoning "
                                    "after %d generated tokens",
                                    completion);
                        break;
                    }
                    tool_scan_waiting_for_think_close = true;
                    tool_scan_from = text.len;
                } else {
                    if (tool_scan_waiting_for_think_close) {
                        const char *think_end = find_tool_structural_text(text.ptr, "</think>", true);
                        tool_scan_from = think_end ?
                            (size_t)((think_end + strlen("</think>")) - text.ptr) : text.len;
                        if (tool_scan_from > text.len) tool_scan_from = text.len;
                        tool_scan_waiting_for_think_close = false;
                    }
                    if (tool_scan_from > text.len) tool_scan_from = text.len;
                    const char *tool_scan = text.ptr ? text.ptr + tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = saw_tool_start;
                    bool old_end = saw_tool_end;
                    observe_tool_markers(&qwen_tool_tracker, tool_scan, &saw_tool_start, &saw_tool_end, &orphan_end);
                    if (orphan_end && !saw_orphan_tool_end) {
                        saw_orphan_tool_end = true;
                        server_log(P36_LOG_WARNING,
                                   "p36-server: chat ctx=%s ignored orphan tool-call end marker after %d generated tokens",
                                   ctx_span,
                                   completion);
                        trace_event(s, trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    completion);
                    }
                    if (saw_tool_start && !old_start) {
                        trace_event(s, trace_id, "entered tool-call block after %d generated tokens", completion);
                    }
                    if (saw_tool_end && !old_end) {
                        trace_event(s, trace_id, "closed tool-call block after %d generated tokens", completion);
                        if (j->req.stop_after_tool_call) {
                            finish = "stop";
                            stop_decode = true;
                            break;
                        }
                    }
                    const size_t marker_hold = 80;
                    tool_scan_from = text.len > marker_hold ? text.len - marker_hold : 0;
                }
                if (s->trace && completion >= next_tool_progress) {
                    trace_event(s, trace_id,
                                "progress gen=%d qwen_tool_start=%d qwen_tool_end=%d",
                                completion, saw_tool_start ? 1 : 0, saw_tool_end ? 1 : 0);
                    next_tool_progress += 128;
                }
            }

            if (completion >= next_decode_log) {
                log_decode_progress(j->req.kind, ctx_span, completion,
                                    j->req.has_tools,
                                    thinking.inside,
                                    saw_tool_start,
                                    saw_tool_end,
                                    decode_t0,
                                    &last_decode_log_t,
                                    &last_decode_log_completion);
                next_decode_log += 50;
            }



            bool next_greedy = qwen_tool_decode_state_is_tool(qwen_tool_tracker.decode) &&
                !qwen_tool_decode_state_uses_payload_sampling(qwen_tool_tracker.decode);
            if (ti + 1 < ntok && payload_temperature > 0.0f &&
                next_greedy != greedy_tool_syntax) break;
        }
        if (kept < ntok && !text_stop && strcmp(finish, "error")) {
            pthread_mutex_lock(&s->inference_mu);
            p36_session_rewind(slot->session, block_start + kept);
            bool ok = p36_session_pos(slot->session) == block_start + kept;
            pthread_mutex_unlock(&s->inference_mu);
            if (!ok) {
                finish = "error";
                snprintf(err, sizeof(err), "failed to restore speculative boundary");
                stop_decode = true;
            }
        }
        if (stop_decode) break;
    }

    server_generation_leave(s);
    free(leak);
    j->req.reasoning_tokens = thinking_tokens;
    j->req.prefill_s = decode_t0 - t0;
    j->req.decode_s = now_sec() - decode_t0;
    if (g_stop_requested && strcmp(finish, "error") != 0) {
        finish = "error";
        snprintf(err, sizeof(err), "shutdown requested");
    }

    if (saw_tool_start && !saw_tool_end && strcmp(finish, "error"))
        trace_event(s, trace_id, "incomplete tool: finish=%s generated=%d limit=%d context_room=%d",
                    finish, completion, j->req.max_tokens, room);

    if (completion > last_decode_log_completion) {
        log_decode_progress(j->req.kind, ctx_span, completion,
                            j->req.has_tools,
                            thinking.inside,
                            saw_tool_start,
                            saw_tool_end,
                            decode_t0,
                            &last_decode_log_t,
                            &last_decode_log_completion);
    }

    if (j->req.stream && !structured_stream && text.len > plain_stream_pos) {
        char *tail = xstrndup(text.ptr + plain_stream_pos, text.len - plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, id, tail, NULL)) finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    bool incomplete_tool_call = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            text.ptr ? text.ptr : "",
            p36_think_mode_enabled(j->req.think_mode),
            j->req.has_tools,
            saw_tool_start,
            &final_finish,
            err,
            sizeof(err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure,
            &incomplete_tool_call);
        j->req.incomplete_tool_call = incomplete_tool_call;
        if (incomplete_tool_call)
            trace_event(s, trace_id, "incomplete tool call quarantined (finish=%s, complete_calls=%d)",
                        final_finish, parsed_calls.len);
        if (!parsed_ok && recovered_tool_parse_failure) {
            server_log(P36_LOG_WARNING,
                       "p36-server: chat ctx=%s invalid tool call returned as assistant text finish=%s",
                       ctx_span,
                       final_finish);
            trace_event(s, trace_id,
                        "invalid tool call returned as assistant text finish=%s",
                        final_finish);
        }
        if (parsed_calls.len) {
            if (openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &openai_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            /* Known tool IDs can replay Qwen's empty thinking prelude exactly,
             * preserving recurrent and vision state without a full rebuild. */
            parsed_calls.replay_empty_think = !s->disable_exact_tool_replay &&
                !j->req.kat_coder && !p36_think_mode_enabled(j->req.think_mode);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        }
    }
    log_tool_calls_summary(ctx_span, &parsed_calls);

    trace_finish(s, trace_id, &j->req, final_finish, completion,
                 saw_tool_start, saw_tool_end,
                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                 parsed_reasoning, &parsed_calls, now_sec() - t0);

    if (j->req.kind == REQ_CHAT && parsed_calls.len) {
        canonicalize_tool_checkpoint(s, slot, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
    } else if (!parsed_calls.len &&
               should_canonicalize_thinking_checkpoint(&j->req, &thinking, final_finish)) {
        canonicalize_thinking_checkpoint(s, slot, j, ctx_span, trace_id,
                                         parsed_content ? parsed_content : "");
    }

    slot_remember_vision_tools(s, slot, &j->req, &parsed_calls);

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_RESPONSES) {
            response_ok = responses_sse_finish(
                j->fd, &j->req, id,
                parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                parsed_reasoning, &parsed_calls, final_finish,
                prompt_tokens, completion);
        } else if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, &j->req, id, &anthropic_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    &parsed_calls, final_finish, completion);
        } else if (openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, id, &openai_live,
                                                 text.ptr ? text.ptr : "", text.len,
                                                 &parsed_calls, final_finish,
                                                 prompt_tokens, completion);
        } else if (structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, id,
                                          parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          prompt_tokens, completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, id, prompt_tokens, completion);
        }
        if (!response_ok) {
            server_log(P36_LOG_DEFAULT,
                       "p36-server: %s ctx=%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span);
        }
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(
            j->fd, &j->req, id,
            parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
            parsed_reasoning, &parsed_calls, final_finish,
            prompt_tokens, completion);
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else {
        final_response(j->fd, &j->req, id,
                       parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       prompt_tokens, completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  true,
                  thinking.inside,
                  saw_tool_start,
                  saw_tool_end);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(P36_LOG_GENERATION,
                       "p36-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(P36_LOG_GENERATION,
                       "p36-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  j->req.has_tools,
                  thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(P36_LOG_GENERATION,
                       "p36-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(P36_LOG_GENERATION,
                       "p36-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    openai_stream_free(&openai_live);
    buf_free(&text);
    p36_tokens_free(&generated_tokens);
    p36_tokens_free(&effective_prompt);
}

static int job_slot_score_values(bool busy, bool assigned,
                                 int common, int live) {
    if (busy || assigned) return INT_MIN;
    if (common == live && live > 0) return INT_MAX / 2 + live;
    return common;
}

static int job_slot_score(server_slot *slot, const job *j) {
    if (!slot || !j) return INT_MIN;
    if (slot->busy || slot->assigned) return INT_MIN;
    if (slot_pending_matches(slot, &j->req)) return INT_MAX;
    int common = p36_session_common_prefix(slot->session, &j->req.prompt);
    if (common && !p36_session_vision_prefix_matches(
            slot->session, j->req.images, j->req.image_count)) common = 0;
    return job_slot_score_values(
            false, false, common,
            p36_session_pos(slot->session));
}

static void dispatch_jobs_locked(server *s) {
    if (!s || !s->batched_mode) return;
    for (;;) {
        job *chosen = NULL, *chosen_prev = NULL;
        server_slot *chosen_slot = NULL;
        job *prev = NULL;
        for (job *j = s->head; j; prev = j, j = j->next) {
            int best_score = INT_MIN;
            server_slot *best = NULL;
            for (int i = 0; i < s->slot_count; i++) {
                int score = job_slot_score(&s->slots[i], j);
                if (score > best_score) {
                    best_score = score;
                    best = &s->slots[i];
                }
            }
            if (best) {
                chosen = j;
                chosen_prev = prev;
                chosen_slot = best;
                break;
            }
        }
        if (!chosen) break;
        if (chosen_prev) chosen_prev->next = chosen->next;
        else s->head = chosen->next;
        if (s->tail == chosen) s->tail = chosen_prev;
        chosen->next = NULL;
        chosen_slot->assigned = chosen;
        chosen_slot->busy = true;
        pthread_cond_broadcast(&s->cv);
    }
}

static bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    if (s->batched_mode) {
        dispatch_jobs_locked(s);
        pthread_cond_broadcast(&s->cv);
    } else {
        pthread_cond_signal(&s->cv);
    }
    pthread_mutex_unlock(&s->mu);
    return true;
}

static job *dequeue(server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
    if (!s->head) {
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }
    job *j = s->head;
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;
    return j;
}

static void *worker_main(void *arg) {
    server *s = arg;
    for (;;) {
        job *j = dequeue(s);
        if (!j) break;
        generate_job(s, &s->slots[0], j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

static void *slot_worker_main(void *arg) {
    server_slot *slot = arg;
    server *s = slot->srv;
    for (;;) {
        pthread_mutex_lock(&s->mu);
        while (!slot->assigned && (!s->stopping || s->head))
            pthread_cond_wait(&s->cv, &s->mu);
        if (!slot->assigned && s->stopping && !s->head) {
            pthread_mutex_unlock(&s->mu);
            break;
        }
        job *j = slot->assigned;
        slot->assigned = NULL;
        pthread_mutex_unlock(&s->mu);

        generate_job(s, slot, j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);

        pthread_mutex_lock(&s->mu);
        slot->busy = false;
        dispatch_jobs_locked(s);
        pthread_mutex_unlock(&s->mu);
    }
    return NULL;
}

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

