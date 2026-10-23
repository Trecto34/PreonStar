static int worker_run_turn(agent_worker *w, const char *user_text) {
    if (w->cfg->server_url) return worker_run_turn_remote(w, user_text);
    agent_config *cfg = w->cfg;
    pthread_mutex_lock(&w->mu);
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    char compact_err[160] = {0};
    bool next_thinking = w->thinking_enabled;
    char *user_content = agent_chat_control_text(user_text, &next_thinking);
    const char *effort_note = NULL;
    if (p36_engine_is_qwen38(w->engine) && next_thinking) {
        p36_think_mode mode = enabled_think_mode(cfg);
        effort_note = p36_qwen38_effort_instruction(mode);
        if (mode == P36_THINK_MEDIUM)
            effort_note = "Reasoning effort is set to medium. Use your normal reasoning depth for this turn.";
    }
    p36_tokens incoming = {0};
    if (effort_note)
        p36_chat_append_message(w->engine, &incoming, "system", effort_note);
    p36_chat_append_message(w->engine, &incoming, "user", user_content);
    bool fits = agent_worker_user_fits(w, &incoming, compact_err, sizeof(compact_err));
    p36_tokens_free(&incoming);
    if (!fits) {
        free(user_content);
        agent_set_error(w, compact_err);
        return 1;
    }
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
    {
        free(user_content);
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
        return 1;
    }
    agent_worker_maybe_append_datetime_context(w);
    agent_worker_append_hints_context(w);
    w->thinking_enabled = next_thinking;
    if (effort_note) {
        p36_chat_append_message(w->engine, &w->transcript, "system", effort_note);
        agent_trace_text(w, "reasoning-effort", effort_note, strlen(effort_note));
    }
    agent_trace_text(w, "user", user_content, strlen(user_content));
    agent_st_note_goal(&w->st, user_content);
    if (!agent_worker_append_user(w, user_content, compact_err, sizeof(compact_err))) {
        free(user_content);
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err);
        return 1;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_prompt(user_content, 0);
        w->session_created_at = (uint64_t)time(NULL);
        agent_session_identity_sha(w->session_title, w->session_created_at,
                                   w->session_sha);
    }
    free(user_content);

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->user_activity = true;
    w->session_dirty = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    /* A user turn may contain any number of assistant/tool/assistant rounds.
     * Coding agents naturally perform long read/edit/test loops, so there is
     * deliberately no artificial "too many tool calls" ceiling here: context
     * pressure, compaction, user Ctrl+C, and the model's final answer are the
     * real stopping conditions.  The transcript is the single source of truth:
     * after a Qwen tool stanza completes we terminate that assistant message, append
     * the tool result as a tool message, then ask the model to continue. */
    int consecutive_failures = 0;
    int carried_generation = 0;
    int carried_thinking_tokens = 0;
    bool resume_assistant = false, resume_in_think = false;
    int resume_start = 0;
    agent_repetition_detector thinking_repetition = {0};
    agent_recovery rec = {.max = cfg->gen.max_recoveries};
    int think_level = agent_think_level_for_prompt(user_text);
    int think_budget = agent_think_budget_for(cfg, think_level);
    agent_vw_begin_turn(&w->st, &cfg->gen);
    w->wd_ncalls = 0;
    w->wd_stagnant = 0;
    w->wd_last_err = 0;
    w->wd_prev_hash = 0;
    w->wd_prev_run = 0;
    for (int tool_round = 0; ; tool_round++) {
        think_level = agent_think_level_raise(think_level, consecutive_failures, rec.recoveries > 0);
        think_budget = agent_think_budget_for(cfg, think_level);
        if (!resume_assistant &&
            !agent_worker_compact_if_needed(w, "soft limit before generation",
                                            compact_err, sizeof(compact_err)))
        {
            if (agent_err_is_interrupted(compact_err)) {
                worker_clear_interrupt(w);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
            agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
            return 1;
        }
        if (!resume_assistant) {
            agent_worker_maybe_append_system_prompt_reminder(w);
            agent_worker_append_hints_context(w);
        }
        p36_think_mode think_mode = w->thinking_enabled && !rec.nothink_next ?
            enabled_think_mode(cfg) : P36_THINK_NONE;
        rec.nothink_next = false;
        const bool continuing_assistant = resume_assistant;
        const int assistant_start = continuing_assistant ? resume_start : w->transcript.len;
        if (!resume_assistant)
            p36_chat_append_assistant_prefix(w->engine, &w->transcript, think_mode);

        const p36_tokens *prompt_for_sync = &w->transcript;
        int old_pos = p36_session_pos(w->session);
        int common = p36_session_common_prefix(w->session, &w->transcript);
        int cached = common == old_pos && w->transcript.len >= old_pos ? common : 0;

        int suffix = prompt_for_sync->len - cached;
        agent_trace(w, "prefill tool_round=%d transcript=%d prompt=%d cached=%d suffix=%d think=%s",
                    tool_round, w->transcript.len, prompt_for_sync->len,
                    cached, suffix, p36_think_mode_name(think_mode));
        agent_trace_tokens(w, "prefill_suffix", prompt_for_sync, cached);

        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        char err[160];
        p36_session_set_progress(w->session, worker_progress_cb, w);
        p36_session_set_display_progress(w->session, worker_progress_cb, w);
        p36_session_set_cancel(w->session, worker_cancel_session_cb, w);
        const double prefill_t0 = now_sec();
        int sync_rc = w->image_count ?
            p36_session_sync_vision(w->session, prompt_for_sync, w->images, w->image_count, err, sizeof(err)) :
            p36_session_sync(w->session, prompt_for_sync, err, sizeof(err));
        w->m.prefill_ms += (now_sec() - prefill_t0) * 1000.0;
        w->m.prompt_tokens += prompt_for_sync->len;
        w->m.cached_tokens += cached;
        p36_session_set_cancel(w->session, NULL, NULL);
        p36_session_set_progress(w->session, NULL, NULL);
        p36_session_set_display_progress(w->session, NULL, NULL);
        if (sync_rc == P36_SESSION_SYNC_INTERRUPTED) {
            agent_publish_system_status(
                w, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
            p36_tokens_push(&w->transcript, p36_token_eos(w->engine));
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (sync_rc != 0) {
            agent_set_error(w, err);
            return 1;
        }

        int max_tokens = cfg->gen.n_predict - carried_generation;
        agent_trace(w, "prefill sync done tool_round=%d pos=%d", tool_round,
                    p36_session_pos(w->session));
        int room = p36_session_ctx(w->session) - p36_session_pos(w->session);
        int generation_room = room - agent_compact_reserve_tokens(w);
        bool context_limited = generation_room < max_tokens;
        if (context_limited) max_tokens = generation_room;
        if (max_tokens <= 0 && cfg->gen.n_predict > 0) {
            agent_set_error(w, "context has no generation room after compaction; use a larger --ctx");
            return 1;
        }

        bool use_color = isatty(STDOUT_FILENO) != 0;
        bool in_think = continuing_assistant ? resume_in_think : p36_think_mode_enabled(think_mode);
        int thinking_tokens = continuing_assistant ? carried_thinking_tokens : 0;
        if (!continuing_assistant) memset(&thinking_repetition, 0, sizeof(thinking_repetition));
        resume_assistant = false;
        agent_token_renderer renderer = {
            .engine = w->engine,
            .worker = w,
            .format_thinking = p36_think_mode_enabled(think_mode),
            .format_markdown = use_color,
            .in_think = in_think,
            .use_color = use_color,
            .last_output_newline = true,
        };
        agent_qwen_tool_parser qwen_tool = {.state = AGENT_QWEN_TOOL_SEARCH};
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &qwen_tool,
            .in_think = in_think,
        };
        agent_edit_upto_forcer upto_forcer = {0};
        bool got_tool = false;
        bool malformed_tool = false;
        bool early_tool_error = false;
        bool repetitive_tool = false;
        bool repetition_close_thinking = false;
        bool model_stopped = false;
        int compaction_lookahead = 0;
        int generated = 0;
        size_t tool_repetition_raw_seen = 0;
        agent_repetition_detector tool_repetition = {0};
        agent_action_leak leak = {0};
        bool action_leak = false;
        /* The action phase starts when thinking ends, or at token 0 when the
         * round runs without thinking (including recovery retries). */
        const bool leak_guard = cfg->gen.action_budget > 0;
        double t0 = now_sec();

        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        bool status_greedy_sampling = false;
        int think_close_id = agent_special_token_id(w->engine, "</think>");
        int think_soft_close_start = agent_think_soft_close_start(
            think_budget);
        agent_trace(w, "active_frequency_penalty=%.3f",
                    (double)cfg->gen.frequency_penalty);
        agent_trace(w, "active_repeat_penalty=%.3f",
                    (double)cfg->gen.repeat_penalty);
        while (generated < max_tokens && !worker_should_interrupt(w)) {
            worker_apply_pending_power(w);
            bool greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }
            int close_rank = 0;
            int close_rank_limit = 0;
            bool in_think = think_close_id >= 0 &&
                            p36_session_in_think(w->session);
            bool hard_budget_close = in_think &&
                agent_think_hard_budget_reached(
                    thinking_tokens, think_budget);
            if (in_think && !hard_budget_close && !repetition_close_thinking) {
                close_rank_limit = agent_think_close_rank_limit(
                    thinking_tokens, think_budget);
                if (close_rank_limit > 0 && thinking_tokens == think_soft_close_start)
                    agent_trace(w, "thinking closure ranking starts at token=%d",
                                think_soft_close_start);
                if (close_rank_limit > 0)
                    close_rank = p36_session_token_rank(
                        w->session, think_close_id, close_rank_limit);
            }
            bool output_hard_close = in_think && !context_limited &&
                                     generated + 1 >= max_tokens;
            int token = close_rank > 0 || hard_budget_close ||
                        repetition_close_thinking || output_hard_close ?
                think_close_id :
                worker_sample_with_mode(w, cfg, greedy_sampling, generated, &rng);
            if (close_rank > 0)
                agent_trace(w,
                    "closing thinking at thinking_token=%d rank=%d accepted_rank=%d",
                    thinking_tokens, close_rank, close_rank_limit);
            else if (hard_budget_close)
                agent_trace(w,
                    "thinking_hard_budget_close thinking_tokens=%d hard_budget=%d",
                    thinking_tokens, think_budget);
            else if (output_hard_close)
                agent_trace(w, "closing thinking at hard token budget=%d",
                            max_tokens);
            token = p36_session_eos_to_think_close(w->session, token);
            if (token == p36_token_eos(w->engine)) {
                model_stopped = true;
                break;
            }

            size_t text_len = 0;
            char *text = p36_token_text(w->engine, token, &text_len);
            if (cfg->edit_upto &&
                agent_edit_upto_forcer_should_replace(&upto_forcer, &qwen_tool,
                                                       text, text_len))
            {
                agent_trace(w, "edit old auto-upto replaced token=%d text=%.*s",
                            token, (int)(text_len > 80 ? 80 : text_len), text);
                free(text);
                if (worker_force_generated_text(w, "[upto]\n", max_tokens,
                                                &generated, t0, &stream,
                                                err, sizeof(err)) != 0) {
                    agent_qwen_tool_parser_free(&qwen_tool);
                    agent_set_error(w, err);
                    return 1;
                }
            } else {
                if (worker_accept_generated_token(w, token, &generated, t0,
                                                  &stream, err, sizeof(err)) != 0) {
                    free(text);
                    agent_qwen_tool_parser_free(&qwen_tool);
                    agent_set_error(w, err);
                    return 1;
                }
                if (in_think && token != think_close_id) {
                    thinking_tokens++;
                    if (!repetition_close_thinking &&
                        agent_repetition_detector_feed(
                            &thinking_repetition, text, text_len)) {
                        repetition_close_thinking = true;
                        agent_trace(w,
                            "repetition_detected_thinking thinking_tokens=%d",
                            thinking_tokens);
                    }
                } else if (leak_guard && !in_think && token != think_close_id &&
                           qwen_tool.state == AGENT_QWEN_TOOL_SEARCH &&
                           agent_action_leak_feed(&leak, cfg->gen.action_budget,
                                                  text, text_len)) {
                    action_leak = true;
                    agent_trace(w, "action_leak content_tokens=%d budget=%d",
                                leak.tokens, cfg->gen.action_budget);
                }
                free(text);
            }

            if (qwen_tool.raw_len < tool_repetition_raw_seen)
                tool_repetition_raw_seen = 0;
            if (qwen_tool.raw_len > tool_repetition_raw_seen) {
                if (agent_repetition_detector_feed(
                        &tool_repetition,
                        qwen_tool.raw + tool_repetition_raw_seen,
                        qwen_tool.raw_len - tool_repetition_raw_seen)) {
                    repetitive_tool = true;
                    agent_trace(w,
                        "repetition_detected_tool generated=%d raw_bytes=%zu",
                        generated, qwen_tool.raw_len);
                }
                tool_repetition_raw_seen = qwen_tool.raw_len;
            }

            /* Cut an oversized write while it streams, not after the whole
             * payload was generated and then rejected. */
            if (!stream.tool_preflight_error &&
                qwen_tool.state == AGENT_QWEN_TOOL_PARAM_VALUE &&
                qwen_tool.param_name && !strcmp(qwen_tool.param_name, "content") &&
                qwen_tool.current.name && !strcmp(qwen_tool.current.name, "write") &&
                qwen_tool.raw_len > qwen_tool.param_value_start +
                                    AGENT_WRITE_PAYLOAD_MAX + 64) {
                stream.tool_preflight_error = true;
                const char *wp = agent_tool_arg_value(&qwen_tool.current, "path");
                snprintf(stream.tool_preflight_error_msg,
                         sizeof(stream.tool_preflight_error_msg),
                         "WRITE_PAYLOAD_TOO_LARGE path=%.120s max_bytes=%d "
                         "next=write the first part, then write append=true for the rest",
                         wp ? wp : "?", AGENT_WRITE_PAYLOAD_MAX);
                agent_trace(w, "write payload preflight: %s",
                            stream.tool_preflight_error_msg);
            }

            greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }

            if (qwen_tool.state == AGENT_QWEN_TOOL_DONE) {
                got_tool = true;
                break;
            }
            if (repetitive_tool || action_leak) break;
            if (stream.tool_preflight_error) {
                early_tool_error = true;
                break;
            }
            if (qwen_tool.state == AGENT_QWEN_TOOL_ERROR) {
                malformed_tool = true;
                break;
            }
            if (stream.qwen_tool_in_think) {
                malformed_tool = true;
                break;
            }
            if (context_limited && generated == max_tokens &&
                compaction_lookahead < 32 && agent_stream_compaction_needs_lookahead(&stream)) {
                max_tokens++;
                compaction_lookahead++;
                if (max_tokens == cfg->gen.n_predict - carried_generation)
                    context_limited = false;
            }
        }

        agent_trace(w, "generation finished tool_round=%d generated=%d carried=%d context_limited=%d",
                    tool_round, generated, carried_generation, context_limited);
        {
            const char *fin = got_tool ? "tool_calls" : action_leak ? "action_budget" :
                repetitive_tool ? "repetition_abort" : malformed_tool || early_tool_error ? "tool_error" :
                model_stopped ? "stop" : context_limited ? "length" : "stop";
            int action_tok = generated - thinking_tokens;
            w->m.steps++;
            w->m.reasoning_tokens += thinking_tokens;
            w->m.action_tokens += action_tok > 0 ? action_tok : 0;
            w->m.decode_ms += (now_sec() - t0) * 1000.0;
            snprintf(w->m.finish, sizeof(w->m.finish), "%s", fin);
            agent_trace(w, "[THINK] tokens=%d budget=%d", thinking_tokens, think_budget);
            agent_trace(w, "[ACTION] tokens=%d finish=%s ctx=%d/%d", action_tok > 0 ? action_tok : 0,
                        fin, p36_session_pos(w->session), p36_session_ctx(w->session));
        }
        bool interrupted = worker_should_interrupt(w);
        const bool partial_tool = agent_stream_has_partial_tool(&stream);
        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);
        worker_set_greedy_sampling(w, false);
        if (interrupted) {
            if (partial_tool) {
                agent_rollback_partial_tool(
                    w, assistant_start, "partial_tool_interrupt_rollback",
                    agent_partial_tool_interrupt_warning);
                agent_qwen_tool_parser_free(&qwen_tool);
                agent_publish_system_status(w, "Stopped by user; incomplete tool call discarded");
                worker_clear_interrupt(w);

                char *queued_user = worker_request_queued_user_drain(w);
                if (queued_user && queued_user[0]) {
                    char *content = agent_chat_control_text(
                        queued_user, &w->thinking_enabled);
                    agent_trace_text(w, "queued_user_after_partial_tool_interrupt",
                                     content, strlen(content));
                    if (!agent_worker_append_user(w, content, compact_err,
                                                  sizeof(compact_err))) {
                        free(content);
                        free(queued_user);
                        agent_set_error(w, compact_err);
                        return 1;
                    }
                    free(content);
                    free(queued_user);
                    consecutive_failures = 0;
                    carried_generation = 0;
                    carried_thinking_tokens = 0;
                    continue;
                }
                free(queued_user);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
            p36_tokens_push(&w->transcript, p36_token_eos(w->engine));
            agent_qwen_tool_parser_free(&qwen_tool);
            agent_publish_system_status(w, "Stopped by user");
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (action_leak) {
            /* Drop the whole failed round (thinking + monologue); retry once
             * with thinking off so the model goes straight to an action. */
            agent_qwen_tool_parser_free(&qwen_tool);
            carried_generation = 0;
            carried_thinking_tokens = 0;
            w->m.recoveries++;
            if (!agent_recovery_next(&rec))
                return agent_turn_blocked(w, "post-think reasoning leak, recoveries exhausted");
            agent_rollback_assistant_suffix(w, assistant_start);
            p36_chat_append_message(w->engine, &w->transcript, "user",
                                    agent_action_required_msg);
            agent_trace(w, "[RECOVERY] n=%d reason=action_leak", rec.recoveries);
            continue;
        }
        if (repetitive_tool) {
            agent_rollback_partial_tool(
                w, assistant_start, "repetition_detected_tool rollback=1",
                agent_tool_repetition_warning);
            agent_qwen_tool_parser_free(&qwen_tool);
            carried_generation = 0;
            carried_thinking_tokens = 0;
            continue;
        }
        if (context_limited && generated == max_tokens && !model_stopped &&
            !got_tool && !malformed_tool && !early_tool_error) {
            carried_generation += generated;
            if (partial_tool) {
                /* No tool from this unfinished round has run. Remove the
                 * partial arguments before asking the model to retry. */
                w->transcript.len = assistant_start;
                p36_session_invalidate(w->session);
            }
            agent_qwen_tool_parser_free(&qwen_tool);
            agent_trace(w, "generation context boundary: generated=%d partial_tool=%d",
                        generated, partial_tool);
            int open_assistant = assistant_start;
            if (!agent_worker_compact_transcript(w, "generation reached compaction reserve",
                                       partial_tool ? NULL : &open_assistant,
                                       compact_err, sizeof(compact_err))) {
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err);
                return 1;
            }
            if (partial_tool) {
                p36_chat_append_message(w->engine, &w->transcript, "user",
                "The last tool call was interrupted by context compaction and was NOT executed. "
                "Continue the user's task, respecting its constraints. If a tool is needed, "
                "retry with smaller arguments or smaller edits.\n");
            } else {
                resume_assistant = true;
                resume_start = open_assistant;
                resume_in_think = stream.in_think;
                carried_thinking_tokens = thinking_tokens;
            }
            continue;
        }
        if (stream.qwen_tool_in_think) {
            got_tool = false;
            malformed_tool = true;
            early_tool_error = false;
            snprintf(qwen_tool.error, sizeof(qwen_tool.error),
                     "tool calling is not allowed inside <think></think>");
        } else if (!malformed_tool && qwen_tool.state == AGENT_QWEN_TOOL_ERROR) {
            malformed_tool = true;
        } else if (!got_tool && !malformed_tool && !early_tool_error &&
                   !interrupted &&
                   (qwen_tool.state == AGENT_QWEN_TOOL_STRUCTURAL ||
                    qwen_tool.state == AGENT_QWEN_TOOL_PARAM_VALUE))
        {
            malformed_tool = true;
            snprintf(qwen_tool.error, sizeof(qwen_tool.error),
                     "incomplete Qwen tool call");
        }

        p36_tokens_push(&w->transcript, p36_token_eos(w->engine));

        if (!got_tool && !malformed_tool && !early_tool_error) {
            agent_qwen_tool_parser_free(&qwen_tool);
            agent_trace(w, "[FINAL] finish=%s", w->m.finish);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        agent_tool_observation observation;
        agent_tool_observation_init(&observation);
        if (early_tool_error) {
            agent_tool_observation_puts(&observation, "Tool error: ");
            agent_tool_observation_puts(
                &observation, stream.tool_preflight_error_msg[0] ?
                stream.tool_preflight_error_msg :
                "edit old selector failed before new was generated");
            agent_tool_observation_puts(&observation, "\n");
        } else if (malformed_tool) {
            agent_tool_observation_puts(
                &observation, "Tool error: invalid Qwen tool call: ");
            agent_tool_observation_puts(
                &observation, qwen_tool.error[0] ? qwen_tool.error : "parse error");
            agent_tool_observation_puts(&observation, "\n");
            if (stream.qwen_tool_in_think)
                agent_tool_observation_puts(&observation,
                    "Close thinking with </think> before retrying the tool call.\n");
            agent_tool_observation_puts(
                &observation, p36_engine_chat_template(w->engine) ?
                agent_json_tool_syntax_reminder : agent_qwen_tool_syntax_reminder);
        } else {
            agent_tool_observation_free(&observation);
            for (int ci = 0; ci < qwen_tool.calls.len; ci++)
                agent_trace(w, "[TOOL CALL] %s", qwen_tool.calls.v[ci].name ? qwen_tool.calls.v[ci].name : "?");
            const int blocked_before = w->wd_blocked;
            observation = agent_execute_tool_observation(w, &qwen_tool.calls);
            w->m.tools += qwen_tool.calls.len;
            w->m.blocked_calls += w->wd_blocked - blocked_before;
        }
        agent_buf observed_text = {0};
        for (size_t i = 0; i < observation.part_count; i++)
            agent_buf_puts(&observed_text, observation.parts[i].text ? observation.parts[i].text : "");
        char *warning = agent_tool_response_add_warning(agent_buf_take(&observed_text),
                                                       &consecutive_failures);
        if (consecutive_failures > 0) {
            const char *start = strstr(warning, "\n\n⚠️ SYSTEM WARNING:");
            if (start) agent_tool_observation_puts(&observation, start);
        }
        free(warning);
        if (w->wd_stagnant >= cfg->gen.max_stagnant_turns) {
            w->wd_stagnant = 0;
            w->m.recoveries++;
            if (!agent_recovery_next(&rec)) {
                agent_tool_observation_free(&observation);
                agent_qwen_tool_parser_free(&qwen_tool);
                return agent_turn_blocked(w, "stagnation: repeated blocked/failing calls");
            }
            agent_tool_observation_puts(&observation, agent_action_required_msg);
            agent_tool_observation_puts(&observation, "\n");
            agent_trace(w, "[RECOVERY] n=%d reason=stagnation", rec.recoveries);
        }
        int projected_tokens = 0;
        int result_reserve = AGENT_TOOL_RESULT_RESERVE_TOKENS;
        char append_err[160] = {0};
        int fits = agent_tool_observation_fits(w, &observation, result_reserve,
                                               &projected_tokens, append_err, sizeof(append_err));
        if (fits < 0) goto observation_error;
        if (!fits)
        {
            if (!agent_worker_compact(w, "tool result would exceed context",
                                      compact_err, sizeof(compact_err)))
            {
                agent_tool_observation_free(&observation);
                agent_qwen_tool_parser_free(&qwen_tool);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
                return 1;
            }
            fits = agent_tool_observation_fits(w, &observation, result_reserve,
                                                &projected_tokens, append_err, sizeof(append_err));
            if (fits < 0) goto observation_error;
            if (!fits)
            {
                agent_tool_observation_free(&observation);
                agent_tool_observation_init(&observation);
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Tool error: tool result still does not fit after context compaction "
                         "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
                         "Retry with a smaller read/search/bash output.\n",
                         projected_tokens, w->cfg->gen.ctx_size,
                         result_reserve);
                agent_tool_observation_puts(&observation, msg);
                fits = agent_tool_observation_fits(w, &observation, 16, NULL,
                                                   append_err, sizeof(append_err));
                if (fits < 0) goto observation_error;
                if (!fits) {
                    agent_tool_observation_free(&observation);
                    agent_qwen_tool_parser_free(&qwen_tool);
                    agent_set_error(w, "context full after compaction");
                    return 1;
                }
            }
        }
        if (!agent_tool_observation_commit(w, &observation,
                                           append_err, sizeof(append_err)))
            goto observation_error;
        agent_tool_observation_free(&observation);
        agent_qwen_tool_parser_free(&qwen_tool);
        carried_generation = 0;
        char *queued_user = worker_request_queued_user_drain(w);
        if (queued_user && queued_user[0]) {
            char *content = agent_chat_control_text(queued_user,
                                                    &w->thinking_enabled);
            agent_trace_text(w, "queued_user", content, strlen(content));
            if (!agent_worker_append_user(w, content, compact_err, sizeof(compact_err))) {
                free(content);
                free(queued_user);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err);
                return 1;
            }
            free(content);
            consecutive_failures = 0;
            pthread_mutex_lock(&w->mu);
            w->user_activity = true;
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        }
        free(queued_user);
        if (agent_vw_force_final(w)) return 0;
        continue;
observation_error:
        agent_tool_observation_free(&observation);
        agent_qwen_tool_parser_free(&qwen_tool);
        agent_set_error(w, append_err[0] ? append_err : "unable to append tool observation");
        return 1;
    }
}

