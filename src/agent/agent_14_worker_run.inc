static void worker_set_greedy_sampling(agent_worker *w, bool greedy) {
    pthread_mutex_lock(&w->mu);
    if (w->status.greedy_sampling != greedy) {
        w->status.greedy_sampling = greedy;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}

/* ---- Deterministic thinking policy (--think auto) ----
 * Level 0 trivial / 1 normal / 2 hard, chosen from the first prompt and raised
 * (never lowered within a user turn) by tool failures and recoveries.  No LLM
 * router: the signals are cheap and the cost of a wrong guess is one retry. */
static const int agent_think_level_budget[3] = {256, 1024, 2048};

static int agent_think_level_for_prompt(const char *text) {
    static const char *const hard[] = {"debug", "refactor", "architect", "design", "race",
        "deadlock", "performance", "optimi", "migrate", "why does", "root cause",
        "multiple files", "across"};
    static const char *const easy[] = {"list ", "show ", "read ", "cat ", "grep ", "find ",
        "rename ", "typo", "print ", "what is in", "open "};
    if (!text) return 1;
    char lower[1500];
    size_t n = strlen(text);
    if (n >= sizeof(lower)) n = sizeof(lower) - 1;
    for (size_t i = 0; i < n; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[n] = 0;
    int nhard = 0, neasy = 0;
    for (size_t i = 0; i < sizeof(hard) / sizeof(hard[0]); i++) if (strstr(lower, hard[i])) nhard++;
    for (size_t i = 0; i < sizeof(easy) / sizeof(easy[0]); i++) if (strstr(lower, easy[i])) neasy++;
    if (nhard >= 2 || strlen(text) > 1200) return 2;
    if (nhard == 0 && neasy > 0 && strlen(text) < 200) return 0;
    return 1;
}

static int agent_think_level_raise(int level, int consecutive_failures, bool recovered) {
    int want = consecutive_failures >= 2 ? 2 : (consecutive_failures == 1 || recovered ? 1 : 0);
    return want > level ? want : level;
}

/* Effective per-round budget: the explicit/default cap, tightened by --think auto. */
static int agent_think_budget_for(const agent_config *cfg, int level) {
    int cap = cfg->gen.thinking_budget;
    if (!cfg->gen.think_auto) return cap;
    int b = agent_think_level_budget[level < 0 ? 0 : level > 2 ? 2 : level];
    return cap > 0 && cap < b ? cap : b;
}

/* Concise end-of-run summary (also used by the remote backend). */
static char *agent_metrics_summary(const agent_worker *w) {
    const agent_metrics *m = &w->m;
    char b[1900];
    long pf_tokens = m->prompt_tokens - m->cached_tokens;
    double pf = m->prefill_ms > 0 ? (double)pf_tokens / (m->prefill_ms / 1000.0) : 0.0;
    long dec_tokens = m->reasoning_tokens + m->action_tokens;
    double dc = m->decode_ms > 0 ? (double)dec_tokens / (m->decode_ms / 1000.0) : 0.0;
    snprintf(b, sizeof(b),
        "\n--- session summary ---\n"
        "steps: %d\ntools: %d\nreasoning tokens: %ld\naction tokens: %ld\n"
        "recoveries: %d\nrepeated calls blocked: %d\ncompactions: %d\n"
        "target mutations: %d\nverification helper mutations: %d\nverification steps: %d\n"
        "verification budget exhaustions: %d\nno-progress steps: %d\nforced finalizations: %d\n"
        "implementation steps: %d\nimplementation step limit hits: %d\n"
        "controller phase transitions: IMPLEMENT->VERIFY %d, VERIFY->IMPLEMENT %d, VERIFY->FINALIZE_REQUIRED %d\n"
        "final controller phase: %s\n"
        "prompt tokens: %ld (%ld cached, %.0f%%)\n"
        "wall time: %.1fs\nprefill: %.1f tok/s\ndecode: %.1f tok/s\nlast finish: %s\n",
        m->steps, m->tools, m->reasoning_tokens, m->action_tokens, m->recoveries,
        m->blocked_calls, m->compactions,
        m->target_mutations, m->helper_mutations, m->verify_steps, m->verify_exhaustions,
        m->no_progress_steps, m->forced_finalizations,
        m->impl_steps, m->impl_limit_hits,
        m->ph_iv, m->ph_vi, m->ph_vf, agent_vw_phase(&w->st.vw),
        m->prompt_tokens, m->cached_tokens,
        m->prompt_tokens ? 100.0 * (double)m->cached_tokens / (double)m->prompt_tokens : 0.0,
        now_sec() - w->session_t0, pf, dc, m->finish[0] ? m->finish : "-");
    return xstrdup(b);
}

static int agent_turn_blocked(agent_worker *w, const char *why) {
    char msg[160];
    snprintf(msg, sizeof(msg), "BLOCKED: %s", why);
    agent_trace(w, "%s", msg);
    agent_publishf(w, "\n%s\n", msg);   /* visible in non-interactive runs too */
    agent_set_status(w, AGENT_WORKER_IDLE);
    return 0;
}


/* ============================================================================
 * Remote backend (--server): the model, template and KV live on a p36-server;
 * tools, watchdog, recovery and compaction state stay here.  Messages are kept
 * as OpenAI-format JSON so the server's canonical template and exact tool-call
 * replay stay in charge of the token stream.
 * ============================================================================
 */

static void agent_json_puts(char **out, size_t *len, size_t *cap, const char *s) {
    size_t n = strlen(s);
    if (*len + n + 1 > *cap) {
        *cap = (*len + n + 1) * 2;
        *out = xrealloc(*out, *cap);
    }
    memcpy(*out + *len, s, n + 1);
    *len += n;
}

static void agent_rmsg_push(agent_worker *w, char *json) {
    if (w->nrmsg == w->rmsg_cap) {
        w->rmsg_cap = w->rmsg_cap ? w->rmsg_cap * 2 : 32;
        w->rmsg = xrealloc(w->rmsg, (size_t)w->rmsg_cap * sizeof(w->rmsg[0]));
    }
    w->rmsg[w->nrmsg++] = json;
}

static char *agent_rmsg_text(const char *role, const char *content, const char *extra) {
    char *out = NULL;
    size_t len = 0, cap = 0;
    agent_json_puts(&out, &len, &cap, "{\"role\":");
    p36_remote_json_quote(&out, &len, &cap, role);
    agent_json_puts(&out, &len, &cap, ",\"content\":");
    p36_remote_json_quote(&out, &len, &cap, content ? content : "");
    if (extra) agent_json_puts(&out, &len, &cap, extra);
    agent_json_puts(&out, &len, &cap, "}");
    return out;
}

/* Tool schemas are the JSON blocks already embedded in the local prompt, so the
 * remote and local agents can never drift apart. */
static char *agent_remote_tools_json(void) {
    const char *a = strstr(agent_tools_prompt_after_edit, "<tools>\n");
    const char *b = strstr(agent_tools_prompt_after_edit, "</tools>");
    if (!a || !b) return xstrdup("[]");
    a += 8;
    char *out = NULL;
    size_t len = 0, cap = 0;
    agent_json_puts(&out, &len, &cap, "[");
    for (const char *p = a; p < b; p++) {
        if (p[0] == '\n' && p + 1 < b && p[1] == '\n') {
            agent_json_puts(&out, &len, &cap, ",");
            p++;
        } else {
            char c[2] = {*p, 0};
            agent_json_puts(&out, &len, &cap, c);
        }
    }
    agent_json_puts(&out, &len, &cap, "]");
    return out;
}

/* Same guidance as the local prompt minus the Qwen tool-call syntax, which the
 * server's template renders itself. */
static char *agent_remote_system_prompt(const agent_config *cfg) {
    agent_buf b = {0};
    const char *intro = agent_tools_prompt_intro;
    const char *first_end = strstr(intro, "\n\n");
    if (first_end) agent_buf_append(&b, intro, (size_t)(first_end - intro) + 2);
    const char *read_help = strstr(intro, "</IMPORTANT>\n\n");
    if (read_help) agent_buf_puts(&b, read_help + strlen("</IMPORTANT>\n\n"));
    agent_buf_puts(&b, cfg->edit_upto ? agent_tools_prompt_edit_upto : agent_tools_prompt_edit_exact);
    const char *tools = strstr(agent_tools_prompt_after_edit, "<tools>");
    if (tools) agent_buf_append(&b, agent_tools_prompt_after_edit,
                                (size_t)(tools - agent_tools_prompt_after_edit));
    const char *rules = strstr(agent_tools_prompt_after_edit, "- This system runs");
    if (rules) { agent_buf_puts(&b, "# Rules\n\n"); agent_buf_puts(&b, rules); }
    if (cfg->gen.system && cfg->gen.system[0]) {
        bool t = true;
        char *plain = agent_chat_control_text(cfg->gen.system, &t);
        agent_buf_puts(&b, "\n\n");
        agent_buf_puts(&b, plain);
        free(plain);
    }
    return agent_buf_take(&b);
}

static char *agent_remote_request(agent_worker *w, bool thinking, int think_budget) {
    const agent_config *cfg = w->cfg;
    char *out = NULL, *tools = agent_remote_tools_json();
    size_t len = 0, cap = 0;
    char num[256];
    agent_json_puts(&out, &len, &cap, "{\"model\":");
    p36_remote_json_quote(&out, &len, &cap, cfg->server_model ? cfg->server_model : "");
    agent_json_puts(&out, &len, &cap, ",\"messages\":[");
    for (int i = 0; i < w->nrmsg; i++) {
        if (i) agent_json_puts(&out, &len, &cap, ",");
        agent_json_puts(&out, &len, &cap, w->rmsg[i]);
    }
    agent_json_puts(&out, &len, &cap, "],\"tools\":");
    agent_json_puts(&out, &len, &cap, tools);
    free(tools);
    snprintf(num, sizeof(num), ",\"stream\":true,\"stream_options\":{\"include_usage\":true},"
             "\"stop_after_tool_call\":true,\"max_tokens\":%d", cfg->gen.n_predict);
    agent_json_puts(&out, &len, &cap, num);
    if (cfg->gen.action_budget > 0) {
        snprintf(num, sizeof(num), ",\"action_budget\":%d", cfg->gen.action_budget);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (thinking && think_budget > 0) {
        snprintf(num, sizeof(num), ",\"thinking_budget\":%d", think_budget);
        agent_json_puts(&out, &len, &cap, num);
    }
    snprintf(num, sizeof(num), ",\"chat_template_kwargs\":{\"enable_thinking\":%s}",
             thinking ? "true" : "false");
    agent_json_puts(&out, &len, &cap, num);
    if (cfg->gen.temperature_set) {
        snprintf(num, sizeof(num), ",\"temperature\":%g", (double)cfg->gen.temperature);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.top_k_set) {
        snprintf(num, sizeof(num), ",\"top_k\":%d", cfg->gen.top_k);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.top_p_set) {
        snprintf(num, sizeof(num), ",\"top_p\":%g", (double)cfg->gen.top_p);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.min_p_set) {
        snprintf(num, sizeof(num), ",\"min_p\":%g", (double)cfg->gen.min_p);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.frequency_penalty != 0.0f) {
        snprintf(num, sizeof(num), ",\"frequency_penalty\":%g", (double)cfg->gen.frequency_penalty);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.repeat_penalty != 1.0f) {
        snprintf(num, sizeof(num), ",\"repeat_penalty\":%g", (double)cfg->gen.repeat_penalty);
        agent_json_puts(&out, &len, &cap, num);
    }
    if (cfg->gen.seed) {
        snprintf(num, sizeof(num), ",\"seed\":%llu", (unsigned long long)cfg->gen.seed);
        agent_json_puts(&out, &len, &cap, num);
    }
    agent_json_puts(&out, &len, &cap, "}");
    return out;
}

typedef struct {
    agent_worker *w;
    int last_kind;
} agent_remote_stream;

static void agent_remote_delta(void *ud, int kind, const char *s, size_t n) {
    agent_remote_stream *rs = ud;
    if (kind != rs->last_kind) {
        if (kind == 1) agent_publish(rs->w, "\x1b[38;5;245m", 11);
        else agent_publish(rs->w, "\x1b[0m\n", 5);
        rs->last_kind = kind;
    }
    agent_publish(rs->w, s, n);
}

static bool agent_remote_cancelled(void *ud) {
    return worker_should_interrupt(((agent_remote_stream *)ud)->w);
}

/* Compaction without a model call: system prompt + harness state + the last
 * few messages (never splitting an assistant/tool pair). */
static void agent_remote_compact(agent_worker *w, const char *reason) {
    if (w->nrmsg < 4 || agent_st_empty(&w->st)) return;
    int keep = w->nrmsg - 4;
    if (keep < 1) keep = 1;
    while (keep > 1 && strstr(w->rmsg[keep], "\"role\":\"tool\"")) keep--;
    char *state = agent_st_render(&w->st);
    agent_buf msg = {0};
    agent_buf_puts(&msg, "[p36-agent compacted earlier conversation. Durable task state follows.]\n");
    agent_buf_puts(&msg, state);
    agent_buf_puts(&msg, "[End compacted state. Recent conversation continues verbatim below.]\n");
    free(state);
    char *state_msg = agent_rmsg_text("user", msg.ptr, NULL);
    free(msg.ptr);
    int old = w->nrmsg;
    for (int i = 1; i < keep; i++) free(w->rmsg[i]);
    w->rmsg[keep - 1] = state_msg;
    memmove(w->rmsg + 1, w->rmsg + keep - 1, (size_t)(w->nrmsg - keep + 1) * sizeof(w->rmsg[0]));
    w->nrmsg -= keep - 2;
    w->m.compactions++;
    agent_trace(w, "[COMPACTION] reason=\"%s\" messages=%d->%d", reason, old, w->nrmsg);
    agent_publishf(w, "\n\x1b[1;95mCOMPACTING\x1b[0m %s: kept structured task state (%d -> %d messages)\n",
                   reason, old, w->nrmsg);
}

static int worker_run_turn_remote(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    p36_remote_cfg rc = {cfg->server_url, cfg->api_key, cfg->server_timeout};
    bool next_thinking = w->thinking_enabled;
    char *user_content = agent_chat_control_text(user_text, &next_thinking);
    w->thinking_enabled = next_thinking;
    agent_st_note_goal(&w->st, user_content);
    agent_trace_text(w, "user", user_content, strlen(user_content));
    agent_rmsg_push(w, agent_rmsg_text("user", user_content, NULL));
    free(user_content);
    agent_set_status(w, AGENT_WORKER_GENERATING);

    agent_recovery rec = {.max = cfg->gen.max_recoveries};
    int consecutive_failures = 0;
    int think_level = agent_think_level_for_prompt(user_text);
    agent_vw_begin_turn(&w->st, &cfg->gen);
    w->wd_ncalls = 0;
    w->wd_stagnant = 0;
    w->wd_last_err = 0;
    w->wd_prev_hash = 0;
    w->wd_prev_run = 0;
    for (;;) {
        if (w->remote_ctx > 0 && agent_context_should_compact(w->remote_ctx, w->remote_used))
            agent_remote_compact(w, "soft limit before generation");
        const bool thinking = w->thinking_enabled && !rec.nothink_next;
        rec.nothink_next = false;
        think_level = agent_think_level_raise(think_level, consecutive_failures, rec.recoveries > 0);
        char *req = agent_remote_request(w, thinking, agent_think_budget_for(cfg, think_level));
        const char *dump = getenv("P36_AGENT_DUMP_REQ");   /* debugging aid: last request body */
        if (dump && dump[0]) {
            FILE *df = fopen(dump, "wb");
            if (df) { fputs(req, df); fclose(df); }
        }
        agent_remote_stream rs = {w, 0};
        p36_remote_result res;
        double t0 = now_sec();
        int rcode = p36_remote_chat(&rc, req, agent_remote_delta, agent_remote_cancelled, &rs, &res);
        free(req);
        if (rs.last_kind) agent_publish(w, "\x1b[0m", 4);
        agent_publish(w, "\n", 1);
        if (rcode != 0) {
            bool interrupted = worker_should_interrupt(w);
            if (interrupted) {
                worker_clear_interrupt(w);
                agent_publish_system_status(w, "Stopped by user");
                agent_set_status(w, AGENT_WORKER_IDLE);
            } else if (strstr(res.err, "context") && w->nrmsg > 4 && !agent_st_empty(&w->st)) {
                agent_remote_compact(w, "server rejected the context size");
                p36_remote_result_free(&res);
                continue;
            } else {
                agent_set_error(w, res.err);
            }
            p36_remote_result_free(&res);
            return interrupted ? 0 : 1;
        }
        int action_tokens = res.completion_tokens - res.reasoning_tokens;
        w->m.steps++;
        w->m.reasoning_tokens += res.reasoning_tokens;
        w->m.action_tokens += action_tokens > 0 ? action_tokens : 0;
        w->m.prompt_tokens += res.prompt_tokens;
        w->m.cached_tokens += res.cached_tokens;
        w->m.prefill_ms += res.prefill_ms;
        w->m.decode_ms += res.decode_ms;
        w->m.wall_s += now_sec() - t0;
        snprintf(w->m.finish, sizeof(w->m.finish), "%s", res.finish);
        w->remote_used = res.prompt_tokens + res.completion_tokens;
        agent_trace(w, "[STEP] finish=%s prompt=%d cached=%d reasoning=%d action=%d prefill_ms=%.0f decode_ms=%.0f",
                    res.finish, res.prompt_tokens, res.cached_tokens, res.reasoning_tokens,
                    action_tokens, res.prefill_ms, res.decode_ms);
        if (res.reasoning) agent_trace_text(w, "[THINK]", res.reasoning, strlen(res.reasoning));

        if (!strcmp(res.finish, "action_budget")) {
            /* Drop the round; ask for an action with thinking off. */
            p36_remote_result_free(&res);
            w->m.recoveries++;
            if (!agent_recovery_next(&rec))
                return agent_turn_blocked(w, "post-think reasoning leak, recoveries exhausted");
            agent_rmsg_push(w, agent_rmsg_text("user", agent_action_required_msg, NULL));
            agent_trace(w, "[RECOVERY] n=%d reason=action_leak", rec.recoveries);
            agent_publish_system_status(w, "reasoning leak aborted; requesting an action");
            continue;
        }

        const bool has_call = res.ncalls > 0 && res.calls[0].name;
        /* A truncated tool call (discarded by the server) or output cut off
         * before any content/call is a failed generation, not an answer: never
         * put it in the transcript; retry through the recovery machinery. */
        bool blank_content = true;
        for (const char *cp = res.content; cp && *cp; cp++)
            if (!isspace((unsigned char)*cp)) { blank_content = false; break; }
        const bool cut_empty = !has_call && !strcmp(res.finish, "length") && blank_content;
        if (!has_call && (res.incomplete_tool || cut_empty)) {
            const bool tool_cut = res.incomplete_tool;
            p36_remote_result_free(&res);
            w->m.recoveries++;
            if (!agent_recovery_next(&rec))
                return agent_turn_blocked(w, tool_cut ? "tool call keeps getting cut off, recoveries exhausted"
                                                      : "output keeps getting cut off, recoveries exhausted");
            agent_rmsg_push(w, agent_rmsg_text("user", tool_cut ? agent_tool_cut_msg : agent_action_required_msg, NULL));
            agent_trace(w, "[RECOVERY] n=%d reason=%s", rec.recoveries, tool_cut ? "incomplete_tool_call" : "output_cut");
            agent_publish_system_status(w, tool_cut ? "tool call was cut off; retrying smaller"
                                                    : "output was cut off; requesting an action");
            continue;
        }
        char *extra = NULL;
        if (has_call) {
            const p36_remote_call *rcall = &res.calls[0];
            char idbuf[32];
            snprintf(idbuf, sizeof(idbuf), "call_%d", w->m.tools + 1);
            const char *cid = rcall->id ? rcall->id : idbuf;
            size_t el = 0, ec = 0;
            agent_json_puts(&extra, &el, &ec, ",\"tool_calls\":[{\"id\":");
            p36_remote_json_quote(&extra, &el, &ec, cid);
            agent_json_puts(&extra, &el, &ec, ",\"type\":\"function\",\"function\":{\"name\":");
            p36_remote_json_quote(&extra, &el, &ec, rcall->name);
            agent_json_puts(&extra, &el, &ec, ",\"arguments\":");
            p36_remote_json_quote(&extra, &el, &ec, rcall->arguments ? rcall->arguments : "{}");
            agent_json_puts(&extra, &el, &ec, "}}]");
        }
        /* Send the model's own reasoning back: the server keeps it for assistant
         * turns after the last user message, exactly as the local transcript does. */
        if (res.reasoning && res.reasoning[0]) {
            size_t xl = extra ? strlen(extra) : 0, xc = xl + 1;
            agent_json_puts(&extra, &xl, &xc, ",\"reasoning_content\":");
            p36_remote_json_quote(&extra, &xl, &xc, res.reasoning);
        }
        agent_rmsg_push(w, agent_rmsg_text("assistant", res.content, extra));
        free(extra);
        if (!has_call) {
            if (!strcmp(res.finish, "length"))
                agent_publishf(w, "\n[output token limit reached]\n");
            agent_trace(w, "[FINAL] finish=%s", res.finish);
            p36_remote_result_free(&res);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        /* Convert the structured call into the local tool representation. */
        agent_tool_calls calls = {0};
        agent_tool_call tc = {0};
        tc.name = xstrdup(res.calls[0].name);
        p36_jv *args = p36_jv_parse(res.calls[0].arguments ? res.calls[0].arguments : "{}");
        for (int i = 0; i < p36_jv_count(args); i++) {
            char scratch[40];
            const char *val = p36_jv_text(p36_jv_at(args, i), scratch, sizeof(scratch));
            if (tc.argc == tc.argcap) {
                tc.argcap = tc.argcap ? tc.argcap * 2 : 4;
                tc.args = xrealloc(tc.args, (size_t)tc.argcap * sizeof(tc.args[0]));
            }
            tc.args[tc.argc].name = xstrdup(p36_jv_key(args, i));
            tc.args[tc.argc].value = xstrdup(val ? val : "");
            tc.argc++;
        }
        p36_jv_free(args);
        calls.v = xmalloc(sizeof(calls.v[0]));
        calls.v[0] = tc;
        calls.len = calls.cap = 1;
        char *tool_id = xstrdup(res.calls[0].id ? res.calls[0].id : "");
        if (!tool_id[0]) { free(tool_id); char idbuf[32]; snprintf(idbuf, sizeof(idbuf), "call_%d", w->m.tools + 1); tool_id = xstrdup(idbuf); }
        agent_trace(w, "[TOOL CALL] %s", tc.name);
        p36_remote_result_free(&res);

        int blocked_before = w->wd_blocked;
        agent_tool_observation obs = agent_execute_tool_observation(w, &calls);
        w->m.tools++;
        w->m.blocked_calls += w->wd_blocked - blocked_before;
        agent_buf joined = {0};
        for (size_t i = 0; i < obs.part_count; i++)
            agent_buf_puts(&joined, obs.parts[i].text ? obs.parts[i].text : "");
        agent_tool_observation_free(&obs);
        agent_tool_calls_free(&calls);
        char *text = agent_tool_response_add_warning(agent_buf_take(&joined), &consecutive_failures);
        if (w->wd_stagnant >= cfg->gen.max_stagnant_turns) {
            w->wd_stagnant = 0;
            w->m.recoveries++;
            if (!agent_recovery_next(&rec)) {
                free(text);
                free(tool_id);
                return agent_turn_blocked(w, "stagnation: repeated blocked/failing calls");
            }
            agent_buf more = {0};
            agent_buf_puts(&more, text);
            agent_buf_puts(&more, "\n");
            agent_buf_puts(&more, agent_action_required_msg);
            agent_buf_puts(&more, "\n");
            free(text);
            text = agent_buf_take(&more);
            agent_trace(w, "[RECOVERY] n=%d reason=stagnation", rec.recoveries);
        }
        char *tool_extra = NULL;
        size_t tl = 0, tcap = 0;
        agent_json_puts(&tool_extra, &tl, &tcap, ",\"tool_call_id\":");
        p36_remote_json_quote(&tool_extra, &tl, &tcap, tool_id);
        agent_rmsg_push(w, agent_rmsg_text("tool", text, tool_extra));
        free(tool_extra);
        free(tool_id);
        free(text);
        if (agent_vw_force_final(w)) return 0;
        if (worker_should_interrupt(w)) {
            worker_clear_interrupt(w);
            agent_publish_system_status(w, "Stopped by user");
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
    }
}

/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model-native Qwen tool iteration without a client/server protocol. */
