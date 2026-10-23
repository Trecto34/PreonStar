static void agent_bash_publish_observation(agent_worker *w, const char *obs) {
    if (!obs || !obs[0]) return;
    const char *body = NULL;
    const char *label = strstr(obs, "\n<head ");
    const char *close = NULL;
    if (label) {
        close = "</head>";
    } else {
        label = strstr(obs, "\n<tail ");
        if (label) close = "</tail>";
    }
    if (label) {
        const char *tag_end = strstr(label, ">\n");
        if (tag_end) {
            agent_publish(w, "\x1b[90m", 5);
            if (strstr(label, "\n<head ") == label)
                agent_publish(w, "[showing first output lines]\n",
                              strlen("[showing first output lines]\n"));
            else
                agent_publish(w, "[showing last output lines]\n",
                              strlen("[showing last output lines]\n"));
            agent_publish(w, "\x1b[0m", 4);
            body = tag_end + 2;
        }
    } else {
        label = strstr(obs, "\n<output>\n");
        if (label) {
            body = label + strlen("\n<output>\n");
            close = "</output>";
        }
    }
    if (!body || !body[0]) return;
    const char *end = close ? strstr(body, close) : NULL;
    size_t n = end ? (size_t)(end - body) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        char *safe = agent_terminal_safe_text(body, n);
        agent_publish(w, safe, strlen(safe));
        if (safe[0] && safe[strlen(safe) - 1] != '\n') agent_publish(w, "\n", 1);
        agent_publish(w, "\x1b[0m", 4);
        free(safe);
    }
}

static void agent_bash_refresh_for(agent_worker *w, agent_bash_job *job,
                                   int refresh_sec) {
    double start = now_sec();
    while (agent_bash_is_running(job) && now_sec() - start < refresh_sec) {
        agent_bash_drain_terminal(job);
        if (worker_should_interrupt(w)) break;
        usleep(20000);
    }
}

/* Common implementation for bash, bash_status, and bash_stop. */
static char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done) {
    if (stop && agent_bash_is_running(job)) {
        agent_bash_signal(job, SIGTERM);
        double start = now_sec();
        while (agent_bash_is_running(job) && now_sec() - start < 1.0) {
            usleep(20000);
        }
        agent_bash_signal(job, SIGKILL);
    }
    if (wait || stop) agent_bash_refresh_for(w, job, refresh_sec);

    bool done = false;
    char *obs = agent_bash_observation(job, true, &done);
    agent_bash_publish_observation(w, obs);
    if (remove_if_done && done) agent_bash_remove_job(w, job);
    return obs;
}

static int agent_tool_job_id(const agent_tool_call *call) {
    return agent_parse_int_default(agent_tool_arg_value(call, "job"), 0, 0, INT_MAX);
}

static pid_t agent_tool_pid(const agent_tool_call *call) {
    return (pid_t)agent_parse_int_default(agent_tool_arg_value(call, "pid"), 0, 0, INT_MAX);
}

/* ============================================================================
 * Tool Dispatch
 * ============================================================================
 */

/* Execute one parsed Qwen tool call and return the text that will be appended as
 * the tool-role result.  UI visualization already happened while streaming; this
 * function is only about side effects and the model-visible observation. */

/* ---- Deterministic task state (Phase 5) ---- */
static char *agent_st_clip(const char *s, size_t max) {
    size_t n = s ? strlen(s) : 0;
    while (n && isspace((unsigned char)s[n - 1])) n--;
    bool cut = n > max;
    if (cut) n = max;
    char *out = xmalloc(n + 4);
    memcpy(out, s ? s : "", n);
    for (size_t i = 0; i < n; i++) if (out[i] == '\n' || out[i] == '\r') out[i] = ' ';
    if (cut) memcpy(out + n, "...", 4); else out[n] = 0;
    return out;
}

static void agent_st_push(char **v, int *n, char *owned) {
    if (*n && !strcmp(v[*n - 1], owned)) { free(owned); return; }
    if (*n == AGENT_ST_LINES) {
        free(v[0]);
        memmove(v, v + 1, (AGENT_ST_LINES - 1) * sizeof(v[0]));
        (*n)--;
    }
    v[(*n)++] = owned;
}

static agent_st_file *agent_st_file_get(agent_task_state *st, const char *path) {
    for (int i = 0; i < st->nfiles; i++)
        if (!strcmp(st->files[i].path, path)) return &st->files[i];
    if (st->nfiles == AGENT_ST_FILES) {
        memmove(st->files, st->files + 1, (AGENT_ST_FILES - 1) * sizeof(st->files[0]));
        st->nfiles--;
    }
    agent_st_file *f = &st->files[st->nfiles++];
    memset(f, 0, sizeof(*f));
    snprintf(f->path, sizeof(f->path), "%s", path);
    return f;
}

static void agent_st_note_goal(agent_task_state *st, const char *user_text) {
    if (!user_text || !user_text[0]) return;
    if (st->ngoals == AGENT_ST_GOALS) {
        free(st->goals[0]);
        memmove(st->goals, st->goals + 1, (AGENT_ST_GOALS - 1) * sizeof(st->goals[0]));
        st->ngoals--;
    }
    st->goals[st->ngoals++] = agent_st_clip(user_text, 600);
}

static void agent_st_note_tool(agent_task_state *st, const agent_tool_call *c,
                               const char *res) {
    const char *name = c->name ? c->name : "?";
    const char *path = NULL;
    for (int i = 0; i < c->argc; i++)
        if (c->args[i].name && !strcmp(c->args[i].name, "path")) path = c->args[i].value;
    char line[320];
    st->last_failed = !strncmp(res, "Tool error", 10);
    if (st->last_failed) {
        char *e = agent_st_clip(res + 12, 140);
        snprintf(line, sizeof(line), "%s%s%s: %s", name, path ? " " : "", path ? path : "", e);
        free(e);
        agent_st_push(st->failed, &st->nfailed, xstrdup(line));
        return;
    }
    if (!strcmp(name, "read") && path) {
        agent_st_file *f = agent_st_file_get(st, path);
        const char *ol = strstr(res, "\noffending_line="), *lb = strstr(res, "\nline_bytes=");
        if (ol && lb) snprintf(f->blob, sizeof(f->blob), "line %d, %lld bytes",
                               atoi(ol + 16), atoll(lb + 12));
        const char *a = agent_tool_arg_value(c, "start_line");
        const char *m = agent_tool_arg_value(c, "max_lines");
        char r[40];
        if (agent_tool_arg_value(c, "whole")) snprintf(r, sizeof(r), "all");
        else snprintf(r, sizeof(r), "%s+%s", a ? a : "1", m ? m : "500");
        if (!strstr(f->ranges, r) && strlen(f->ranges) + strlen(r) + 3 < sizeof(f->ranges)) {
            if (f->ranges[0]) strcat(f->ranges, "; ");
            strcat(f->ranges, r);
        }
    } else if ((!strcmp(name, "write") || !strcmp(name, "edit")) && path) {
        agent_st_file *f = agent_st_file_get(st, path);
        f->ranges[0] = 0;   /* earlier reads are stale now */
        char *first = agent_st_clip(res, 90);
        snprintf(f->note, sizeof(f->note), "%s", first);
        snprintf(line, sizeof(line), "%s %s", name, path);
        free(first);
        agent_st_push(st->done, &st->ndone, xstrdup(line));
    } else if (!strcmp(name, "bash")) {
        const char *cmd = agent_tool_arg_value(c, "command");
        char *cc = agent_st_clip(cmd, 120);
        const char *ex = strstr(res, "exit code: ");
        const char *es = strstr(res, "exit_status=");   /* what bash jobs actually report */
        char code[24] = "ok";
        if (ex) snprintf(code, sizeof(code), "exit %d", atoi(ex + 11));
        else if (es && atoi(es + 12) != 0) snprintf(code, sizeof(code), "exit %d", atoi(es + 12));
        snprintf(line, sizeof(line), "`%s` -> %s", cc, code);
        free(cc);
        agent_st_push(st->facts, &st->nfacts, xstrdup(line));
        for (int i = 0; i < st->nfiles; i++) st->files[i].ranges[0] = 0; /* shell may change anything */
    } else if (!strcmp(name, "search") || !strcmp(name, "list")) {
        const char *q = agent_tool_arg_value(c, "query");
        snprintf(line, sizeof(line), "%s %s%s%s", name, q ? q : "", q && path ? " in " : "", path ? path : "");
        agent_st_push(st->done, &st->ndone, xstrdup(line));
    }
}

static bool agent_vw_stagnant(const agent_verify_state *v);

static const char *agent_vw_phase(const agent_verify_state *v) {
    return v->finalize ? "FINALIZE_REQUIRED" : v->phase == AGENT_PH_VERIFY ? "VERIFY" : "IMPLEMENT";
}

static char *agent_st_render(const agent_task_state *st) {
    agent_buf b = {0};
    char tmp[512 + PATH_MAX];
    if (st->ws_root[0]) {
        snprintf(tmp, sizeof(tmp), "WORKSPACE_ROOT: %s\n(controller-fixed project directory; a file found elsewhere never replaces it)\n\n", st->ws_root);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "GOAL:\n");
    for (int i = 0; i < st->ngoals; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->goals[i]);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "\nVERIFIED FACTS:\n");
    for (int i = 0; i < st->nfacts; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->facts[i]);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "\nCOMPLETED:\n");
    for (int i = 0; i < st->ndone; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->done[i]);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "\nFILES / LOCATIONS:\n");
    for (int i = 0; i < st->nfiles; i++) {
        snprintf(tmp, sizeof(tmp), "- %s%s%s\n", st->files[i].path,
                 st->files[i].note[0] ? " (modified)" : "",
                 st->files[i].ranges[0] ? " (read)" : "");
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "\nFAILED ATTEMPTS:\n");
    for (int i = 0; i < st->nfailed; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->failed[i]);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "\nCURRENT EDIT STATE:\n");
    for (int i = 0; i < st->nfiles; i++) {
        if (!st->files[i].note[0]) continue;
        snprintf(tmp, sizeof(tmp), "- %s: %s\n", st->files[i].path, st->files[i].note);
        agent_buf_puts(&b, tmp);
    }
    const agent_verify_state *vw = &st->vw;
    if (vw->armed) {
        /* Counters live on the worker, so compaction renders them but never resets them. */
        agent_buf_puts(&b, "\nVERIFICATION STATE:\n");
        snprintf(tmp, sizeof(tmp), "- CURRENT_PHASE: %s\n", agent_vw_phase(vw));
        agent_buf_puts(&b, tmp);
        if (vw->npending) {
            snprintf(tmp, sizeof(tmp), "- IMPLEMENTATION_PENDING: %s%s\n", vw->pending, vw->npending > 1 ? " (+more)" : "");
            agent_buf_puts(&b, tmp);
        }
        if (vw->last_target_step)
            snprintf(tmp, sizeof(tmp), "- LAST_TARGET_MUTATION: step %d (%s)\n",
                     vw->last_target_step, vw->last_target);
        else
            snprintf(tmp, sizeof(tmp), "- LAST_TARGET_MUTATION: none\n");
        agent_buf_puts(&b, tmp);
        snprintf(tmp, sizeof(tmp),
                 "- VERIFY_STEPS_USED: %d\n- MAX_VERIFY_STEPS: %d\n- NO_PROGRESS_STEPS: %d (ceiling %d)\n"
                 "- STAGNANT: %s\n",
                 vw->verify_used, vw->max_verify, vw->no_progress, vw->max_noprog,
                 agent_vw_stagnant(vw) ? "yes" : "no");
        agent_buf_puts(&b, tmp);
    }
    if (st->nunresolved) {
        agent_buf_puts(&b, "\nUNRESOLVED:\n");
        for (int i = 0; i < st->nunresolved; i++) {
            snprintf(tmp, sizeof(tmp), "- %s\n", st->unresolved[i]);
            agent_buf_puts(&b, tmp);
        }
    }
    agent_buf_puts(&b, "\nNEXT REQUIRED ACTION:\n");
    if (vw->finalize && vw->impl_stop) {
        agent_buf_puts(&b, "- implementation step limit reached: give a concise INCOMPLETE/STOPPED report now, no more tools\n");
    } else if (vw->finalize) {
        agent_buf_puts(&b, "- verification budget exhausted: give the final answer now, no more tools; "
                           "report what is confirmed working, confirmed failing and unresolved\n");
    } else if (vw->npending) {
        snprintf(tmp, sizeof(tmp), "- implement what is missing: %s\n", vw->pending);
        agent_buf_puts(&b, tmp);
    } else if (st->last_failed && st->nfailed) {
        snprintf(tmp, sizeof(tmp), "- fix and retry: %s\n", st->failed[st->nfailed - 1]);
        agent_buf_puts(&b, tmp);
    } else {
        agent_buf_puts(&b, "- continue the GOAL from CURRENT EDIT STATE; act, or give the final answer\n");
    }
    agent_buf_puts(&b, "\nDO NOT REPEAT:\n");
    for (int i = 0; i < st->nfiles; i++) {
        if (!st->files[i].ranges[0]) continue;
        snprintf(tmp, sizeof(tmp), "- already read %s [%s], unchanged\n",
                 st->files[i].path, st->files[i].ranges);
        agent_buf_puts(&b, tmp);
    }
    for (int i = 0; i < st->nfiles; i++) {
        if (!st->files[i].blob[0]) continue;
        snprintf(tmp, sizeof(tmp), "- do not read the minified/vendor blob in %s (%s) again; "
                 "use search/grep or targeted ranges excluding it\n",
                 st->files[i].path, st->files[i].blob);
        agent_buf_puts(&b, tmp);
    }
    for (int i = 0; i < st->ndnr; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->dnr[i]);
        agent_buf_puts(&b, tmp);
    }
    return agent_buf_take(&b);
}

static bool agent_st_empty(const agent_task_state *st) { return st->ngoals == 0; }

static void agent_tool_observation_init(agent_tool_observation *obs) {
    memset(obs, 0, sizeof(*obs));
    obs->parts = xmalloc(sizeof(obs->parts[0]));
    memset(obs->parts, 0, sizeof(obs->parts[0]));
    obs->part_count = 1;
    obs->part_cap = 1;
}

static void agent_tool_observation_puts(agent_tool_observation *obs,
                                        const char *text) {
    if (!obs->part_count) agent_tool_observation_init(obs);
    agent_tool_text_part *part = &obs->parts[obs->part_count - 1];
    size_t n = text ? strlen(text) : 0;
    if (part->len + n + 1 > part->cap) {
        size_t cap = part->cap ? part->cap : 128;
        while (cap < part->len + n + 1) cap *= 2;
        part->text = xrealloc(part->text, cap);
        part->cap = cap;
    }
    if (n) memcpy(part->text + part->len, text, n);
    part->len += n;
    part->text[part->len] = '\0';
}

static void agent_tool_observation_add_image(agent_tool_observation *obs,
                                             p36_vision_embedding *embedding) {
    if (obs->image_count == obs->image_cap) {
        size_t cap = obs->image_cap ? obs->image_cap * 2 : 2;
        obs->images = xrealloc(obs->images, cap * sizeof(obs->images[0]));
        obs->image_cap = cap;
    }
    obs->images[obs->image_count++] = *embedding;
    memset(embedding, 0, sizeof(*embedding));
    if (obs->part_count == obs->part_cap) {
        size_t cap = obs->part_cap ? obs->part_cap * 2 : 2;
        obs->parts = xrealloc(obs->parts, cap * sizeof(obs->parts[0]));
        obs->part_cap = cap;
    }
    memset(&obs->parts[obs->part_count++], 0, sizeof(obs->parts[0]));
}

static void agent_tool_observation_free(agent_tool_observation *obs) {
    if (!obs) return;
    for (size_t i = 0; i < obs->part_count; i++) free(obs->parts[i].text);
    for (size_t i = 0; i < obs->image_count; i++)
        p36_vision_embedding_free(&obs->images[i]);
    free(obs->parts);
    free(obs->images);
    memset(obs, 0, sizeof(*obs));
}

static void agent_tool_view_image(agent_worker *w,
                                  const agent_tool_call *call,
                                  agent_tool_observation *obs) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) {
        agent_tool_observation_puts(obs, "Tool error: view_image requires path\n");
        return;
    }
    if (!p36_engine_has_vision(w->engine)) {
        agent_tool_observation_puts(
            obs, "Tool error: view_image requires p36-agent --vision FILE\n");
        return;
    }
    p36_vision_embedding embedding = {0};
    char err[256] = {0};
    char expanded[PATH_MAX];
    int encoded = agent_expand_home_path(path, expanded, sizeof(expanded)) &&
        p36_engine_vision_encode_file(w->engine, expanded, &embedding, err, sizeof(err));
    if (!encoded) {
        agent_tool_observation_puts(obs, "Tool error: view_image failed: ");
        agent_tool_observation_puts(obs, err[0] ? err : "unable to decode image");
        agent_tool_observation_puts(obs, "\n");
        return;
    }

    char shown[PATH_MAX + 80];
    snprintf(shown, sizeof(shown), "\n[tool:view_image] %s\n", path);
    agent_publish(w, shown, strlen(shown));
    agent_tool_observation_add_image(obs, &embedding);
    char meta[PATH_MAX + 192];
    snprintf(meta, sizeof(meta),
             "\nImage observation attached (%ux%u, %u visual tokens).\n",
             obs->images[obs->image_count - 1].width,
             obs->images[obs->image_count - 1].height,
             obs->images[obs->image_count - 1].token_count);
    agent_tool_observation_puts(obs, meta);
}

static char *agent_execute_tool_call(agent_worker *w, const agent_tool_call *call) {
    agent_buf result = {0};
    if (!call->name) return xstrdup("Tool error: missing tool name\n");

    if (!strcmp(call->name, "read")) return agent_tool_read(w, call);
    if (!strcmp(call->name, "more")) return agent_tool_more(w, call);
    if (!strcmp(call->name, "write")) return agent_tool_write(w, call);
    if (!strcmp(call->name, "list")) return agent_tool_list(call);
    if (!strcmp(call->name, "edit")) return agent_tool_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tool_search(w, call);
    if (!strcmp(call->name, "google_search")) return agent_tool_google_search(w, call);
    if (!strcmp(call->name, "visit_page")) return agent_tool_visit_page(w, call);

    if (!strcmp(call->name, "bash")) {
        const char *cmd = agent_tool_arg_value(call, "command");
        if (!cmd || !cmd[0]) return xstrdup("Tool error: bash requires command\n");
        int timeout = agent_parse_timeout(agent_tool_arg_value(call, "timeout_sec"));
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        char err[160] = {0};
        agent_bash_job *job = agent_bash_start(w, cmd, timeout, err, sizeof(err));
        if (!job) {
            agent_buf_puts(&result, "Tool error: bash failed to start: ");
            agent_buf_puts(&result, err[0] ? err : "unknown error");
            agent_buf_puts(&result, "\n");
            return agent_buf_take(&result);
        }
        return agent_bash_job_tool_result(w, job, true, refresh, false, true);
    }

    if (!strcmp(call->name, "bash_status") ||
        !strcmp(call->name, "bash_stop"))
    {
        int job_id = agent_tool_job_id(call);
        pid_t pid = agent_tool_pid(call);
        agent_bash_job *job = agent_bash_find_job(w, job_id, pid);
        if (!job) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                     job_id, (long)pid);
            return xstrdup(msg);
        }
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              0, 0, 3600);
        bool stop = !strcmp(call->name, "bash_stop");
        bool wait = stop || refresh > 0;
        if (stop && refresh == 0) refresh = 1;
        return agent_bash_job_tool_result(w, job, wait, refresh, stop, true);
    }

    {
        char header[256];
        snprintf(header, sizeof(header), "\n[tool:%s] unknown tool\n", call->name);
        agent_publish(w, header, strlen(header));
        agent_buf_puts(&result, "Tool error: unknown tool: ");
        agent_buf_puts(&result, call->name);
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
}

/* Execute all tool calls from one Qwen tool block, preserving per-call labels in the
 * combined result so the model can associate observations with calls. */
/* Per-user-turn recovery state shared by the leak and stagnation paths. */
typedef struct {
    int recoveries;
    int max;
    bool nothink_next;   /* next round skips thinking: go straight to ACTION */
} agent_recovery;

/* Count one recovery; false means the turn must end BLOCKED. */
static bool agent_recovery_next(agent_recovery *r) {
    r->nothink_next = true;
    return ++r->recoveries <= r->max;
}

static const char agent_tool_cut_msg[] =
    "Your last tool call was cut off by the output limit and was NOT executed. "
    "Retry with smaller arguments: write at most 12 KiB per call and use append=true for more.";

static const char agent_action_required_msg[] =
    "ACTION REQUIRED. Use the verified state. Execute one tool call or give "
    "the final answer. Do not repeat analysis.";

static uint64_t agent_fnv1a(uint64_t h, const char *s) {
    for (; s && *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
    return h;
}

static bool agent_tool_is_readonly(const char *name) {
    static const char *const ro[] = {"read", "more", "list", "search",
        "google_search", "visit_page", "bash_status", "view_image"};
    for (size_t i = 0; name && i < sizeof(ro) / sizeof(ro[0]); i++)
        if (!strcmp(name, ro[i])) return true;
    return false;
}

static uint64_t agent_tool_call_hash(const agent_tool_call *c) {
    uint64_t h = agent_fnv1a(1469598103934665603ULL, c->name);
    for (int i = 0; i < c->argc; i++) {
        h = agent_fnv1a(h, c->args[i].name);
        h = agent_fnv1a(h, "=");
        h = agent_fnv1a(h, c->args[i].value);
        h = agent_fnv1a(h, ";");
    }
    return h;
}

/* True when this call repeats an identical one already executed against an
 * unchanged workspace more than max_repeat times.  Any mutating tool resets
 * the table (state changed, so a re-read is legitimate). */
