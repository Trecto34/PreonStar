static bool agent_vw_force_final(agent_worker *w) {
    if (!w->st.vw.finalize || w->st.vw.rejected < AGENT_VW_MAX_REJECTS) return false;
    char *text = agent_vw_final_text(&w->st);
    w->m.forced_finalizations++;
    snprintf(w->m.finish, sizeof(w->m.finish), w->st.vw.impl_stop ? "impl_limit" : "verify_budget");
    agent_trace(w, "[VERIFY] forced finalization after %d refused tool calls", w->st.vw.rejected);
    agent_publishf(w, "\n%s", text);
    free(text);
    agent_set_status(w, AGENT_WORKER_IDLE);
    return true;
}

static agent_tool_observation agent_execute_tool_observation(
        agent_worker *w, const agent_tool_calls *calls) {
    agent_tool_observation obs;
    agent_tool_observation_init(&obs);
    for (int i = 0; i < calls->len; i++) {
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "Tool result %d (%s):\n", i + 1,
                 calls->v[i].name ? calls->v[i].name : "unknown");
        agent_tool_observation_puts(&obs, hdr);
        if (w->st.vw.finalize) {
            /* FINALIZE_REQUIRED: refuse without executing and keep it short. */
            w->st.vw.rejected++;
            char *note = agent_vw_notice(&w->st.vw);
            agent_trace(w, "[VERIFY] refused tool call %d while FINALIZE_REQUIRED", w->st.vw.rejected);
            agent_tool_observation_puts(&obs, note);
            free(note);
            continue;
        }
        char *blocked = agent_ws_gate(w, &calls->v[i]);
        if (blocked) {
            /* Refused escapes still spend budget, so probing the host cannot loop forever. */
            agent_trace(w, "[WORKSPACE] escape blocked: %s", calls->v[i].name);
            blocked = agent_vw_after(w, &calls->v[i], NULL, false, blocked);
            agent_tool_observation_puts(&obs, blocked);
            free(blocked);
            continue;
        }
        if (calls->v[i].name && !strcmp(calls->v[i].name, "view_image")) {
            agent_tool_view_image(w, &calls->v[i], &obs);
            /* Screenshots are verification too: they spend the same budget. */
            bool img_dup = false;
            if (agent_vw_step(&w->st.vw, &w->m, AGENT_VW_VERIFY,
                              agent_vw_signature(&w->st.vw, &calls->v[i], NULL), NULL, true, &img_dup)) {
                char *note = agent_vw_notice(&w->st.vw);
                agent_tool_observation_puts(&obs, "\n");
                agent_tool_observation_puts(&obs, note);
                free(note);
                agent_trace(w, "[VERIFY] FINALIZE_REQUIRED no target change since step %d",
                            w->st.vw.last_target_step);
            }
            continue;
        }
        char *res;
        if (agent_watchdog_blocks(w, &calls->v[i], w->cfg->gen.max_repeat_tool)) {
            w->wd_blocked++;
            w->wd_stagnant++;
            res = xstrdup("Tool error: REPEAT_CALL_BLOCKED\n"
                          "reason=identical call, workspace unchanged since last result\n"
                          "next=use the earlier result; edit/write/run, or give the final answer\n");
            res = agent_vw_after(w, &calls->v[i], NULL, false, res);
        } else {
            agent_vw_pre vwpre;
            agent_vw_before(&w->st, &calls->v[i], &vwpre);
            res = agent_execute_tool_call(w, &calls->v[i]);
            uint64_t eh = 0;
            if (!strncmp(res, "Tool error", 10)) {
                eh = agent_fnv1a(1469598103934665603ULL, res);
                w->wd_stagnant = eh == w->wd_last_err ? w->wd_stagnant + 1 : 0;
            } else {
                w->wd_stagnant = 0;
            }
            w->wd_last_err = eh;
            agent_st_note_tool(&w->st, &calls->v[i], res);
            res = agent_vw_after(w, &calls->v[i], &vwpre, true, res);
        }
        agent_trace_text(w, "[TOOL RESULT]", res, strlen(res) > 300 ? 300 : strlen(res));
        agent_tool_observation_puts(&obs, res);
        if (res[0] && res[strlen(res) - 1] != '\n')
            agent_tool_observation_puts(&obs, "\n");
        free(res);
    }
    if (calls->len == 0)
        agent_tool_observation_puts(&obs, "Tool error: empty tool call block\n");
    return obs;
}

static void agent_vision_spans_free(p36_vision_span *spans, size_t count) {
    if (!spans) return;
    for (size_t i = 0; i < count; i++)
        p36_vision_embedding_free(&spans[i].embedding);
    free(spans);
}

static bool agent_tool_observation_build(agent_worker *w,
                                         const agent_tool_observation *obs,
                                         p36_tokens *tokens,
                                         p36_vision_span **spans_out,
                                         char *err, size_t err_len) {
    const char **parts = xmalloc(obs->part_count * sizeof(parts[0]));
    for (size_t i = 0; i < obs->part_count; i++)
        parts[i] = obs->parts[i].text ? obs->parts[i].text : "";
    p36_vision_embedding *images = NULL;
    p36_vision_span *spans = NULL;
    if (obs->image_count) {
        images = xmalloc(obs->image_count * sizeof(images[0]));
        spans = xmalloc(obs->image_count * sizeof(spans[0]));
        memset(images, 0, obs->image_count * sizeof(images[0]));
        memset(spans, 0, obs->image_count * sizeof(spans[0]));
        const int n_embd = p36_engine_embd_dim(w->engine);
        for (size_t i = 0; i < obs->image_count; i++) {
            const p36_vision_embedding *src = &obs->images[i];
            if (!src->data || src->token_count == 0 || n_embd <= 0 ||
                (uint64_t)src->token_count >
                    SIZE_MAX / (uint64_t)n_embd / sizeof(float)) {
                snprintf(err, err_len, "invalid image observation embedding");
                for (size_t j = 0; j < i; j++)
                    p36_vision_embedding_free(&images[j]);
                free(images);
                free(spans);
                free(parts);
                p36_tokens_free(tokens);
                return false;
            }
            const size_t bytes = (size_t)src->token_count *
                                 (size_t)n_embd * sizeof(float);
            images[i] = *src;
            images[i].data = malloc(bytes);
            if (!images[i].data) {
                snprintf(err, err_len, "unable to copy image observation embedding");
                for (size_t j = 0; j <= i; j++)
                    p36_vision_embedding_free(&images[j]);
                free(images);
                free(spans);
                free(parts);
                p36_tokens_free(tokens);
                return false;
            }
            memcpy(images[i].data, src->data, bytes);
        }
    }
    p36_tokens_copy(tokens, &w->transcript);
    bool ok = p36_chat_append_multimodal_message(
        w->engine, tokens, "tool",
        parts, images, obs->image_count, spans,
        err, err_len) != 0;
    free(parts);
    if (!ok) {
        for (size_t i = 0; i < obs->image_count; i++) {
            p36_vision_embedding_free(&images[i]);
            p36_vision_embedding_free(&spans[i].embedding);
        }
    }
    free(images);
    if (!ok) {
        free(spans);
        p36_tokens_free(tokens);
        return false;
    }
    *spans_out = spans;
    return true;
}

static int agent_tool_observation_fits(agent_worker *w,
                                        const agent_tool_observation *obs,
                                        int reserve_tokens,
                                        int *tokens_out,
                                        char *err, size_t err_len) {
    p36_tokens tmp = {0};
    p36_vision_span *spans = NULL;
    if (err_len) err[0] = '\0';
    if (!agent_tool_observation_build(w, obs, &tmp, &spans,
                                      err, err_len))
        return -1;
    int tokens = tmp.len;
    p36_tokens_free(&tmp);
    agent_vision_spans_free(spans, obs->image_count);
    if (tokens_out) *tokens_out = tokens;
    int ctx = w->cfg->gen.ctx_size;
    int compact = agent_compact_reserve_tokens(w) + 128;
    if (reserve_tokens < compact) reserve_tokens = compact;
    return ctx > 0 && tokens < ctx - reserve_tokens;
}

static bool agent_tool_observation_commit(agent_worker *w,
                                          agent_tool_observation *obs,
                                          char *err, size_t err_len) {
    p36_tokens next = {0};
    p36_vision_span *spans = NULL;
    if (!agent_tool_observation_build(w, obs, &next, &spans, err, err_len))
        return false;
    p36_tokens_free(&w->transcript);
    w->transcript = next;
    agent_worker_images_append(w, spans, obs->image_count);
    free(spans);
    return true;
}

/* If compaction happens while a bash process is still alive, inject a small
 * tool-role reminder into the rebuilt transcript.  Otherwise the summary could
 * preserve the user's task but lose the fact that an external process still
 * needs status/wait/stop handling. */
static char *agent_bash_jobs_compaction_observation(agent_worker *w) {
    if (!w->bash_jobs) return NULL;
    agent_buf out = {0};
    agent_buf_puts(&out,
        "Bash job update after context compaction. Running jobs still need explicit bash_status or bash_stop if relevant.\n");
    for (agent_bash_job *job = w->bash_jobs, *next = NULL; job; job = next) {
        next = job->next;
        bool done = false;
        char *obs = agent_bash_observation(job, true, &done);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "\nJob %d:\n", job->id);
        agent_buf_puts(&out, hdr);
        agent_buf_puts(&out, obs);
        free(obs);
        if (done) agent_bash_remove_job(w, job);
    }
    return agent_buf_take(&out);
}

/* ============================================================================
 * Context Compaction
 * ============================================================================
 *
 * Compaction asks the model for durable task state, then rebuilds the live
 * transcript as: system prompt + summary + recent verbatim tail.  This keeps
 * the active KV usable while avoiding unbounded transcript growth.
 */

/* Decide when to compact before an ordinary turn or before appending a large
 * tool result.  The fixed free-token threshold is capped proportionally for
 * smaller contexts so tests with tiny contexts still compact rather than fail. */
static bool agent_worker_should_compact(agent_worker *w) {
    int ctx = w->cfg->gen.ctx_size, used = w->transcript.len;
    return agent_context_should_compact(ctx, used) ||
           (used > 0 && ctx - used <= agent_compact_reserve_tokens(w) + 128);
}

static int agent_special_token_id(p36_engine *engine, const char *rendered) {
    p36_tokens t = {0};
    p36_tokenize_rendered_chat(engine, rendered, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    p36_tokens_free(&t);
    return id;
}

static int agent_compact_tail_boundary(const p36_tokens *tokens, int bottom,
                                       int sys_len, int budget,
                                       const p36_tokens *user) {
    int target = bottom - budget;
    if (target < sys_len) target = sys_len;
    int earliest = bottom - 2 * budget;
    if (earliest < sys_len) earliest = sys_len;
    int found = -1;
    for (int i = earliest; i + user->len <= bottom && user->len > 0; i++) {
        if (!memcmp(tokens->v + i, user->v, (size_t)user->len * sizeof(int)))
            found = i;
    }
    return found >= 0 ? found : target;
}

/* Retain the most recent complete user turn when it fits within twice the tail budget. */
static int agent_compact_tail_start(agent_worker *w, int bottom, int sys_len) {
    int budget = w->cfg->gen.ctx_size / AGENT_COMPACT_TAIL_DIVISOR;
    if (budget > AGENT_COMPACT_TAIL_CAP_TOKENS) budget = AGENT_COMPACT_TAIL_CAP_TOKENS;
    if (budget < 1) budget = 1;
    p36_tokens user = {0};
    p36_tokenize_rendered_chat(w->engine, "<|im_start|>user\n", &user);
    int start = agent_compact_tail_boundary(&w->transcript, bottom, sys_len, budget, &user);
    p36_tokens_free(&user);
    return start;
}

static void agent_tokens_append_range(p36_tokens *dst, const p36_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) p36_tokens_push(dst, src->v[i]);
}

/* Build the private prompt used to ask the model for durable state.  The prompt
 * explicitly forbids tool calls because the result is consumed internally, not
 * delivered as an assistant turn. */
static char *agent_compact_make_prompt(const char *reason) {
    agent_buf b = {0};
    agent_buf_puts(&b,
        "Internal p36-agent context compaction request. This is not a user request.\n"
        "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
        "- user goals, constraints, and preferences\n"
        "- files inspected or edited\n"
        "- commands run and important results\n"
        "- decisions, rejected approaches, known bugs, and pending next steps\n"
        "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
        "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
        "Do not solve unfinished tasks, calculate new results, or finish incomplete code in the summary. "
        "Record the user's requirements and what actually happened, leaving unfinished work pending.\n"
        "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or Qwen tool markup.\n"
        "Output only the compact summary.\n");
    if (reason && reason[0]) {
        agent_buf_puts(&b, "\nCompaction reason: ");
        agent_buf_puts(&b, reason);
        agent_buf_puts(&b, "\n");
    }
    return agent_buf_take(&b);
}

static int agent_compact_summary_budget(int ctx) {
    int budget = ctx / 8;
    if (budget < 256) budget = 256;
    if (budget > AGENT_COMPACT_SUMMARY_MAX_TOKENS)
        budget = AGENT_COMPACT_SUMMARY_MAX_TOKENS;
    return budget;
}

static int agent_compact_reserve_tokens(agent_worker *w) {
    if (w->compact_reserve) return w->compact_reserve;
    char *text = agent_compact_make_prompt("context pressure before continuing the current task");
    p36_tokens tokens = {0};
    p36_chat_append_message(w->engine, &tokens, "user", text);
    p36_chat_append_assistant_prefix(w->engine, &tokens, P36_THINK_NONE);
    int reserve = tokens.len + 64 +
                  agent_compact_summary_budget(w->cfg->gen.ctx_size);
    free(text);
    p36_tokens_free(&tokens);
    w->compact_reserve = reserve;
    return reserve;
}

static int agent_compact_image_boundary(const p36_vision_span *images, size_t count,
                                        int pos) {
    for (size_t i = 0; i < count; i++) {
        const p36_vision_span *span = &images[i];
        int wrapper = 1;
        int start = (int)span->token_start - wrapper;
        uint64_t end = (uint64_t)span->token_start +
                       span->embedding.token_count + (unsigned)wrapper;
        if (pos > start && (uint64_t)pos < end) return start;
    }
    return pos;
}

/* Perform the full compaction exchange and rebuild the live P36 session from
 * the compacted transcript.  Any failure invalidates live KV because the model
 * may have just seen private compaction instructions that are not part of the
 * real conversation. */
/* Copy a transcript range, dropping the content of closed <think> blocks so
 * the retained tail carries actions and results, not deliberation. */
static void agent_tokens_append_range_no_think(p36_engine *engine, p36_tokens *dst,
                                               const p36_tokens *src, int from, int to) {
    int ts = agent_special_token_id(engine, "<think>");
    int te = agent_special_token_id(engine, "</think>");
    for (int i = from; i < to; i++) {
        if (src->v[i] == ts) {
            int j = i + 1;
            while (j < to && src->v[j] != te) j++;
            if (j < to) {            /* closed block: keep tags, drop body */
                p36_tokens_push(dst, ts);
                p36_tokens_push(dst, te);
                i = j;
                continue;
            }
        }
        p36_tokens_push(dst, src->v[i]);
    }
}

static bool agent_worker_compact_transcript(agent_worker *w, const char *reason,
                                  int *open_assistant,
                                  char *err, size_t err_len) {
    const int bottom = w->transcript.len;
    if (bottom <= 0) return true;

    p36_tokens sys = {0};
    (void)agent_worker_build_system_tokens(w, &sys);
    if (bottom <= sys.len) {
        p36_tokens_free(&sys);
        return true;
    }

    /* Harness-maintained state replaces the model-written narrative, so no
     * summary generation (and none of its indecision) enters the context. */
    const bool det_state = !w->cfg->gen.compact_llm && !agent_st_empty(&w->st);
    agent_buf summary = {0};
    int summary_bottom = bottom;
    const int ctx = w->cfg->gen.ctx_size;
    if (det_state) {
        agent_publishf(w, "\n\x1b[1;95mCOMPACTING\x1b[0m %s: keeping structured task state\n",
                       reason && reason[0] ? reason : "context");
        summary.ptr = agent_st_render(&w->st);
        summary.len = strlen(summary.ptr);
        summary.cap = summary.len + 1;
        goto have_summary;
    }
    agent_publishf(w,
        "\n\x1b[1;95mCOMPACTING\x1b[0m %s: summarizing durable task state\n\x1b[38;5;245m",
        reason && reason[0] ? reason : "context");

    char *prompt_text = agent_compact_make_prompt(reason);
    p36_tokens summary_suffix = {0};
    p36_chat_append_message(w->engine, &summary_suffix, "user", prompt_text);
    free(prompt_text);
    p36_chat_append_assistant_prefix(w->engine, &summary_suffix, P36_THINK_NONE);
    int summary_budget = agent_compact_summary_budget(ctx);
    if (summary_bottom > ctx - summary_suffix.len - summary_budget - 16)
        summary_bottom = ctx - summary_suffix.len - summary_budget - 16;
    summary_bottom = agent_compact_image_boundary(w->images, w->image_count, summary_bottom);
    if (summary_bottom <= sys.len) {
        snprintf(err, err_len, "context too small to summarize this conversation; use a larger --ctx");
        p36_tokens_free(&summary_suffix);
        p36_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }
    /* A full restored session may have no summary room. Summarize a bounded
     * prefix and retain EVERY unsummarized token verbatim in the new tail. */
    p36_tokens prompt = {0};
    agent_tokens_append_range(&prompt, &w->transcript, 0, summary_bottom);
    /* A bounded prefix may end inside a user message, thinking, or code.
     * Close it only in the private summary prompt; the real tail is untouched. */
    int im_start = agent_special_token_id(w->engine, "<|im_start|>");
    int im_end = agent_special_token_id(w->engine, "<|im_end|>");
    int think_start = agent_special_token_id(w->engine, "<think>");
    int think_end = agent_special_token_id(w->engine, "</think>");
    bool message_open = false, think_open = false;
    for (int i = 0; i < prompt.len; i++) {
        int token = prompt.v[i];
        if (token == im_start) { message_open = true; think_open = false; }
        if (token == im_end) { message_open = false; think_open = false; }
        if (token == think_start) think_open = true;
        if (token == think_end) think_open = false;
    }
    if (message_open) {
        if (think_open) p36_tokenize_rendered_chat(w->engine, "</think>\n", &prompt);
        p36_tokenize_rendered_chat(w->engine, "<|im_end|>\n", &prompt);
    }
    agent_tokens_append_range(&prompt, &summary_suffix, 0, summary_suffix.len);
    p36_tokens_free(&summary_suffix);
    agent_trace(w, "compaction summary prefix=%d total=%d retained_unsummarized=%d",
                summary_bottom, bottom, bottom - summary_bottom);

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_COMPACTING;
    w->progress_started_at = now_sec();
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    int summary_room = w->cfg->gen.ctx_size - prompt.len - 1;
    if (summary_room < 256) {
        snprintf(err, err_len, "not enough context left to request compaction summary");
        p36_tokens_free(&prompt);
        p36_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }
    int summary_max = summary_room < summary_budget ? summary_room : summary_budget;

    p36_session_set_progress(w->session, worker_progress_cb, w);
    p36_session_set_display_progress(w->session, worker_progress_cb, w);
    p36_session_set_cancel(w->session, worker_cancel_session_cb, w);
    size_t summary_images = 0;
    while (summary_images < w->image_count &&
           (uint64_t)w->images[summary_images].token_start +
           w->images[summary_images].embedding.token_count < (uint64_t)summary_bottom)
        summary_images++;
    int sync_rc;
    if (summary_images) {
        sync_rc = p36_session_sync_vision(w->session, &prompt,
                                         w->images, summary_images, err, err_len);
    } else {
        sync_rc = p36_session_sync(w->session, &prompt, err, err_len);
    }
    p36_session_set_cancel(w->session, NULL, NULL);
    p36_session_set_progress(w->session, NULL, NULL);
    p36_session_set_display_progress(w->session, NULL, NULL);
    if (sync_rc == P36_SESSION_SYNC_INTERRUPTED) {
        p36_session_invalidate(w->session);
        snprintf(err, err_len, "interrupted");
        agent_publish_system_status(
            w, "Compaction interrupted; keeping the previous conversation state.");
        p36_tokens_free(&prompt);
        p36_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        worker_clear_interrupt(w);
        return false;
    }
    if (sync_rc != 0) {
        p36_session_invalidate(w->session);
        p36_tokens_free(&prompt);
        p36_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }

    /* From here until the final rebuild, the live KV contains the internal
     * compaction prompt/summary, while w->transcript still contains the real
     * conversation.  If anything fails, invalidate live KV so the next turn
     * cannot accidentally continue from the private compaction exchange. */
    char eval_err[160] = {0};
    int think_end_id = agent_special_token_id(w->engine, "</think>");
    double t0 = now_sec();
    for (int i = 0; i < summary_max; i++) {
        if (worker_should_interrupt(w)) {
            snprintf(err, err_len, "interrupted");
            p36_session_invalidate(w->session);
            p36_tokens_free(&prompt);
            p36_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            agent_publish_system_status(
                w, "Compaction interrupted; keeping the previous conversation state.");
            worker_clear_interrupt(w);
            return false;
        }
        int token = p36_session_argmax(w->session);
        if (token == p36_token_eos(w->engine)) break;
        if (token == think_end_id) {
            agent_trace(w, "compaction summary stopped before control token id=%d", token);
            break;
        }
        if (p36_session_eval(w->session, token, eval_err, sizeof(eval_err)) != 0) {
            snprintf(err, err_len, "%s", eval_err);
            p36_session_invalidate(w->session);
            p36_tokens_free(&prompt);
            p36_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            return false;
        }

        size_t text_len = 0;
        char *text = p36_token_text(w->engine, token, &text_len);
        agent_buf_append(&summary, text, text_len);
        agent_publish(w, text, text_len);
        free(text);
        char *tool = summary.ptr ? strstr(summary.ptr, "<tool_call>") : NULL;
        if (tool) {
            summary.len = (size_t)(tool - summary.ptr);
            summary.ptr[summary.len] = '\0';
            agent_trace(w, "compaction summary stopped before tool call");
            break;
        }

        double dt = now_sec() - t0;
        pthread_mutex_lock(&w->mu);
        w->status.generated = i + 1;
        w->status.gen_tps = dt > 0.0 ? (double)(i + 1) / dt : 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }
    agent_publish(w, "\x1b[0m\n", 5);
    p36_tokens_free(&prompt);

    bool summary_empty = true;
    for (size_t i = 0; i < summary.len; i++) {
        if (!isspace((unsigned char)summary.ptr[i])) {
            summary_empty = false;
            break;
        }
    }
    if (summary_empty) {
        agent_trace(w, "compaction summary empty; retaining recent transcript without a summary");
        agent_publishf(w, "COMPACTING summary unavailable; retaining recent conversation only\n");
        free(summary.ptr);
        memset(&summary, 0, sizeof(summary));
        agent_buf_puts(&summary,
            "Automatic summary was unavailable. Earlier conversation may be missing; "
            "use the recent transcript below and ask for missing details if needed.\n");
    }

have_summary:
    agent_trace_text(w, "compaction-summary", summary.ptr, summary.len);
    int tail_start = agent_compact_tail_start(w, bottom, sys.len);
    if (tail_start > summary_bottom) tail_start = summary_bottom;
    tail_start = agent_compact_image_boundary(w->images, w->image_count, tail_start);
    p36_tokens compacted = {0};
    p36_tokens_copy(&compacted, &sys);

    agent_buf summary_msg = {0};
    agent_buf_puts(&summary_msg,
        "\n\n[p36-agent compacted earlier conversation. Durable task-state summary follows.]\n");
    agent_buf_puts(&summary_msg, summary.ptr);
    if (summary_msg.len && summary_msg.ptr[summary_msg.len - 1] != '\n')
        agent_buf_puts(&summary_msg, "\n");
    agent_buf_puts(&summary_msg, "[End compacted summary. Recent conversation continues verbatim below.]\n\n");
    p36_chat_append_message(w->engine, &compacted, "user", summary_msg.ptr);
    free(summary_msg.ptr);
    free(summary.ptr);

    int resumed_start = -1;
    /* Generation resumes inside the SAME assistant message, including any
     * partial word or code line. A synthetic user request to "continue" can
     * make the model skip the unfinished fragment or start a different task. */
    if (open_assistant && tail_start > *open_assistant) {
        int think_start = agent_special_token_id(w->engine, "<think>");
        int think_end = agent_special_token_id(w->engine, "</think>");
        bool in_think = false;
        for (int i = *open_assistant; i < tail_start; i++) {
            if (w->transcript.v[i] == think_start) in_think = true;
            if (w->transcript.v[i] == think_end) in_think = false;
        }
        resumed_start = compacted.len;
        p36_chat_append_assistant_prefix(w->engine, &compacted,
            in_think ? P36_THINK_HIGH : P36_THINK_NONE);
    } else if (tail_start < bottom &&
        w->transcript.v[tail_start] != agent_special_token_id(w->engine, "<|im_start|>")) {
        p36_chat_append_message(w->engine, &compacted, "user",
            "[Verbatim tail of the previous conversation; it may begin mid-message.]\n");
    }
    const int tail_dst_start = compacted.len;
    if (open_assistant && resumed_start < 0)
        resumed_start = tail_dst_start + *open_assistant - tail_start;
    if (det_state && !open_assistant && !w->image_count)
        agent_tokens_append_range_no_think(w->engine, &compacted, &w->transcript,
                                           tail_start, bottom);
    else
        agent_tokens_append_range(&compacted, &w->transcript, tail_start, bottom);
    agent_trace_tokens(w, "compacted_transcript", &compacted, 0);
    if (compacted.len >= ctx - agent_compact_reserve_tokens(w) - 128) {
        snprintf(err, err_len, "compacted conversation leaves no working room; use a larger --ctx");
        p36_tokens_free(&compacted);
        p36_tokens_free(&sys);
        p36_session_invalidate(w->session);
        return false;
    }


    p36_vision_span *old_images = w->images;
    size_t old_image_count = w->image_count;
    size_t old_image_cap = w->image_cap;
    p36_vision_span *new_images = old_image_count ?
        xmalloc(old_image_count * sizeof(new_images[0])) : NULL;
    bool *kept_images = old_image_count ?
        calloc(old_image_count, sizeof(kept_images[0])) : NULL;
    if (old_image_count && !kept_images) {
        free(new_images);
        p36_tokens_free(&compacted);
        p36_tokens_free(&sys);
        p36_session_invalidate(w->session);
        snprintf(err, err_len, "out of memory retaining compacted images");
        return false;
    }
    size_t new_image_count = 0;
    for (size_t i = 0; i < old_image_count; i++) {
        uint64_t image_end = (uint64_t)old_images[i].token_start +
                             old_images[i].embedding.token_count;
        if (old_images[i].token_start >= (uint32_t)tail_start &&
            image_end <= (uint64_t)bottom) {
            new_images[new_image_count] = old_images[i];
            new_images[new_image_count].token_start =
                (uint32_t)(tail_dst_start +
                           (int)old_images[i].token_start - tail_start);
            new_image_count++;
            kept_images[i] = true;
        }
    }

    agent_publishf(w,
        "\x1b[1;95mCOMPACTING\x1b[0m rebuilding context: old=%d summary+tail=%d tail=%d\n",
        bottom, compacted.len, bottom - tail_start);

    p36_tokens old_transcript = {0};
    p36_tokens_copy(&old_transcript, &w->transcript);
    p36_tokens_free(&w->transcript);
    w->transcript = compacted;
    w->images = new_images;
    w->image_count = new_image_count;
    w->image_cap = old_image_count;
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
        p36_session_invalidate(w->session);
        p36_tokens_free(&w->transcript);
        w->transcript = old_transcript;
        free(new_images);
        w->images = old_images;
        w->image_count = old_image_count;
        w->image_cap = old_image_cap;
        free(kept_images);
        p36_tokens_free(&sys);
        return false;
    }
    for (size_t i = 0; i < old_image_count; i++) {
        if (!kept_images[i]) p36_vision_embedding_free(&old_images[i].embedding);
    }
    free(old_images);
    free(kept_images);
    pthread_mutex_lock(&w->mu);
    agent_hints_compacted(&w->hints);
    pthread_mutex_unlock(&w->mu);
    agent_worker_note_system_prompt_seen(w);
    p36_tokens_free(&old_transcript);
    p36_tokens_free(&sys);
    /* Do not insert an observation inside a response we are still emitting. */
    char *bash_update = open_assistant ? NULL : agent_bash_jobs_compaction_observation(w);
    if (bash_update) {
        p36_chat_append_message(w->engine, &w->transcript, "tool", bash_update);
        w->session_dirty = true;
        agent_trace_text(w, "tool-after-compaction", bash_update, strlen(bash_update));
        agent_publish(w, "\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n",
                      strlen("\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n"));
        free(bash_update);
    }
    w->m.compactions++;
    agent_trace(w, "[COMPACTION] reason=\"%s\" old=%d new=%d tail_start=%d tail=%d",
                reason ? reason : "", bottom, w->transcript.len,
                tail_start, bottom - tail_start);
    if (open_assistant) *open_assistant = resumed_start;
    return true;
}

static bool agent_worker_compact(agent_worker *w, const char *reason,
                                  char *err, size_t err_len) {
    return agent_worker_compact_transcript(w, reason, NULL, err, err_len);
}

static bool agent_worker_user_fits(agent_worker *w, const p36_tokens *message,
                                   char *err, size_t err_len) {
    p36_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    int ctx = w->cfg->gen.ctx_size;
    bool ok = message->len < ctx - sys.len - 4;
    p36_tokens_free(&sys);
    if (!ok) snprintf(err, err_len,
        "user message is too large for --ctx %d; use a larger context or a smaller message", ctx);
    return ok;
}

static bool agent_worker_append_user(agent_worker *w, const char *text,
                                     char *err, size_t err_len) {
    p36_tokens message = {0};
    p36_chat_append_message(w->engine, &message, "user", text);
    bool ok = agent_worker_user_fits(w, &message, err, err_len);
    int ctx = w->cfg->gen.ctx_size;
    if (ok && message.len >= ctx - w->transcript.len - 4) {
        ok = agent_worker_compact(w, "make room for incoming user message", err, err_len);
        if (ok && message.len >= ctx - w->transcript.len - 4) {
            snprintf(err, err_len, "user message does not fit after compaction; use a larger --ctx");
            ok = false;
        }
    }
    if (ok) agent_tokens_append_range(&w->transcript, &message, 0, message.len);
    p36_tokens_free(&message);
    return ok;
}

static bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len) {
    if (!agent_worker_should_compact(w)) return true;
    return agent_worker_compact(w, reason, err, err_len);
}

static int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len) {
    if (p36_session_eval(w->session, token, err, err_len) != 0)
        return 1;

    p36_tokens_push(&w->transcript, token);

    size_t text_len = 0;
    char *text = p36_token_text(w->engine, token, &text_len);
    agent_trace_token(w, token, text, text_len, *generated + 1);
    agent_stream_text(stream, text, text_len, false);
    free(text);
    (*generated)++;

    double dt = now_sec() - t0;
    pthread_mutex_lock(&w->mu);
    w->status.generated = *generated;
    w->status.gen_tps = dt > 0.0 ? (double)*generated / dt : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

static int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len) {
    p36_tokens tokens = {0};
    p36_tokenize_text(w->engine, text, &tokens);
    if (tokens.len > max_tokens - *generated) {
        snprintf(err, err_len, "not enough generation room to force %s", text);
        p36_tokens_free(&tokens);
        return 1;
    }
    for (int i = 0; i < tokens.len && *generated < max_tokens; i++) {
        if (worker_accept_generated_token(w, tokens.v[i], generated, t0,
                                          stream, err, err_len) != 0) {
            p36_tokens_free(&tokens);
            return 1;
        }
    }
    p36_tokens_free(&tokens);
    return 0;
}

/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* Qwen tool structure is a machine-readable grammar, so once the model has clearly
 * started a tool stanza we decode grammar bytes greedily.  Parameter values are
 * different: they can be shell commands, code, file contents, or edit bodies,
 * and should keep the configured sampling behavior.  The only exception is a
 * parameter closing tag once it is clearly Qwen tool syntax, not ordinary text such
 * as HTML/XML/code containing "</".
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy. */
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    if (!sr || !sr->parser) return false;
    if (sr->parser->state == AGENT_QWEN_TOOL_ERROR ||
        sr->parser->state == AGENT_QWEN_TOOL_DONE)
        return false;

    /* A possible opening marker is being held back by the start detector.  A
     * single '<' is too common in prose/code to justify forcing argmax; after
     * the second byte, the buffered prefix still matching here is specifically
     * Qwen tool-shaped ("<tool_call..."). */
    if (sr->qwen_tool_start_len > 1) return true;
    if (!sr->qwen_tool_active) return false;

    if (sr->parser->state == AGENT_QWEN_TOOL_STRUCTURAL)
        return true;
    if (sr->parser->state != AGENT_QWEN_TOOL_PARAM_VALUE)
        return false;

    return sr->parser->param_close_prefix;
}

static bool agent_stream_compaction_needs_lookahead(const agent_stream_renderer *sr) {
    return !sr->qwen_tool_active && sr->parser->state == AGENT_QWEN_TOOL_SEARCH &&
           (sr->pending_len || sr->qwen_tool_start_len);
}

static bool agent_stream_has_partial_tool(const agent_stream_renderer *sr) {
    if (!sr || !sr->parser) return false;
    return sr->qwen_tool_active || sr->qwen_tool_start_len > 0 ||
           sr->parser->state == AGENT_QWEN_TOOL_STRUCTURAL ||
           sr->parser->state == AGENT_QWEN_TOOL_PARAM_VALUE;
}

typedef struct {
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float frequency_penalty;
    float repeat_penalty;
    const int *penalty_tokens;
    int penalty_token_count;
} agent_sampling_request;

static agent_sampling_request agent_sampling_request_build(
        const agent_config *cfg, bool greedy,
        const p36_tokens *transcript, int generated) {
    agent_sampling_request request = {
        .temperature = greedy ? 0.0f : cfg->gen.temperature,
        .top_k = greedy ? 0 : cfg->gen.top_k,
        .top_p = greedy ? 1.0f : cfg->gen.top_p,
        .min_p = greedy ? 0.0f : cfg->gen.min_p,
        .frequency_penalty = greedy ? 0.0f : cfg->gen.frequency_penalty,
        .repeat_penalty = greedy ? 1.0f : cfg->gen.repeat_penalty,
    };
    if (!greedy && transcript && generated > 0 && generated <= transcript->len) {
        request.penalty_tokens = transcript->v + transcript->len - generated;
        request.penalty_token_count = generated;
    }
    return request;
}

static int worker_sample_with_mode(agent_worker *w, const agent_config *cfg,
                                   bool greedy, int generated, uint64_t *rng) {
    agent_sampling_request request = agent_sampling_request_build(
        cfg, greedy, &w->transcript, generated);
    return p36_session_sample_penalized(
        w->session, request.temperature, request.top_k,
        request.top_p, request.min_p,
        request.penalty_tokens, request.penalty_token_count,
        0.0f, request.frequency_penalty, request.repeat_penalty, rng);
}

static const char agent_partial_tool_interrupt_warning[] =
    "The previous tool call was interrupted by the user and was NOT executed. "
    "The workspace is unchanged by that tool call.";

static const char agent_tool_repetition_warning[] =
    "Your previous tool call entered a repetitive generation loop and was NOT "
    "executed. Preserve the intended change, but rewrite it compactly using "
    "loops, arrays, helpers, or data-driven construction.";


static void agent_rollback_assistant_suffix(agent_worker *w,
                                            int assistant_start) {
    if (assistant_start < 0) assistant_start = 0;
    if (assistant_start > w->transcript.len)
        assistant_start = w->transcript.len;
    w->transcript.len = assistant_start;
    p36_session_invalidate(w->session);
    w->session_dirty = true;
}

static void agent_rollback_partial_tool(agent_worker *w, int assistant_start,
                                        const char *trace_event,
                                        const char *warning) {
    agent_rollback_assistant_suffix(w, assistant_start);
    agent_trace(w, "%s assistant_start=%d", trace_event, assistant_start);
    p36_chat_append_message(w->engine, &w->transcript, "system", warning);
}

