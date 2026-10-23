static void worker_request_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->save_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_compact(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->compact_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_power(agent_worker *w, int power) {
    pthread_mutex_lock(&w->mu);
    w->requested_power = power;
    w->power_requested = true;
    w->status.power_percent = power;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_take_save_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->save_requested && !w->paused;
    if (requested) w->save_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static bool worker_take_compact_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->compact_requested && !w->paused;
    if (requested) w->compact_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static bool worker_take_power_requested(agent_worker *w, int *power) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->power_requested;
    if (requested) {
        if (power) *power = w->requested_power;
        w->power_requested = false;
    }
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static void worker_apply_pending_power(agent_worker *w) {
    int power = 0;
    if (!worker_take_power_requested(w, &power)) return;
    if (p36_session_set_power(w->session, power) != 0) {
        agent_publishf(w, "\npower change failed\n");
        return;
    }
    pthread_mutex_lock(&w->mu);
    w->cfg->engine.power_percent = power;
    w->status.power_percent = power;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_set_status(w, AGENT_WORKER_SAVING);
    char err[160] = {0};
    char sha[41];
    int tokens = 0;
    if (agent_worker_save_session_now(w, sha, &tokens, err, sizeof(err)))
        agent_publishf(w, "\nsaved session %.8s (%d tokens)\n", sha, tokens);
    else
        agent_publishf(w, "\nsave failed: %s\n", err[0] ? err : "unknown error");
    agent_set_status(w, AGENT_WORKER_IDLE);
}

static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    if (w->cfg->server_url) {
        agent_remote_compact(w, "user requested compaction");
        return;
    }
    if (!agent_worker_has_user_session(w)) {
        agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        return;
    }

    int before = w->transcript.len;
    char err[160] = {0};
    if (agent_worker_compact(w, "user requested compaction", err, sizeof(err))) {
        if (w->transcript.len != before) {
            pthread_mutex_lock(&w->mu);
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        } else {
            agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        }
        agent_set_status(w, AGENT_WORKER_IDLE);
    } else {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return;
        }
        agent_set_error(w, err[0] ? err : "context compaction failed");
    }
}

/* Status can become IDLE before the worker finishes deferred operations.
 * Publish ownership separately so the UI can safely save or switch sessions. */
static bool worker_wait_for_work(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->active = false;
    pthread_cond_broadcast(&w->cond);
    agent_wake_locked(w);
    while (!w->stop && !w->cmd_text &&
           (w->paused || (!w->save_requested && !w->compact_requested &&
                          !w->power_requested)))
        pthread_cond_wait(&w->cond, &w->mu);
    bool run = !w->stop;
    w->active = run;
    pthread_mutex_unlock(&w->mu);
    return run;
}

/* Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all P36 session mutation, tool execution, and compaction. */
static void *worker_main(void *arg) {
    agent_worker *w = arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                w->cfg->gen.ctx_size,
                p36_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->gen.trace_path ? w->cfg->gen.trace_path : "");
    char init_err[160] = {0};
    if (w->cfg->server_url) {
        w->thinking_enabled = agent_initial_thinking_enabled(w->cfg);
        w->remote_ctx = w->cfg->gen.ctx_size;   /* clamped to the server's context in main() */
        char *sys = agent_remote_system_prompt(w->cfg);
        agent_rmsg_push(w, agent_rmsg_text("system", sys, NULL));
        agent_trace_text(w, "initial_system_prompt", sys, strlen(sys));
        free(sys);
    } else {
        if (!agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err))) {
            agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
        }
        agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);
    }
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    while (worker_wait_for_work(w)) {
        pthread_mutex_lock(&w->mu);
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        if (w->power_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_apply_pending_power(w);
            continue;
        }
        if (!w->cmd_text && w->save_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_save(w);
            continue;
        }
        if (!w->cmd_text && w->compact_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_compact(w);
            continue;
        }
        char *cmd = w->cmd_text;
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        worker_run_turn(w, cmd);
        free(cmd);
        if (w->cfg->non_interactive && w->m.steps) {
            char *sum = agent_metrics_summary(w);
            agent_trace_text(w, "[SUMMARY]", sum, strlen(sum));
            agent_publish(w, sum, strlen(sum));
            free(sum);
        }
        worker_apply_pending_power(w);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}

/* ============================================================================
 * Worker/UI Synchronization Helpers
 * ============================================================================
 */

static int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}

/* Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode. */
static bool worker_check_raw_mode_restore(agent_worker *w) {
    bool needs = false;
    pthread_mutex_lock(&w->mu);
    if (w->raw_mode_needs_restore) {
        w->raw_mode_needs_restore = false;
        needs = true;
    }
    pthread_mutex_unlock(&w->mu);
    return needs;
}

static void drain_wake_fd(int fd) {
    char buf[128];
    for (;;) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) continue;
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

/* Submit one user turn if the worker is idle.  Busy submissions are rejected so
 * the UI can keep the typed text editable instead of silently queueing it. */
static bool worker_submit(agent_worker *w, const char *text) {
    pthread_mutex_lock(&w->mu);
    bool ok = w->initialized && !w->active && !w->paused &&
        w->status.state == AGENT_WORKER_IDLE && !w->cmd_text &&
        !w->save_requested && !w->compact_requested && !w->power_requested;
    if (ok) {
        w->interrupt = false;
        w->cmd_text = xstrdup(text);
        /* A submitted turn is no longer idle, even if the worker thread has
         * not yet reached its real prefill accounting.  Non-interactive mode
         * depends on this to avoid exiting in the small handoff window between
         * accepting stdin and starting generation. */
        w->status.state = AGENT_WORKER_PREFILL;
        w->status.prefill_done = 0;
        w->status.prefill_total = 0;
        w->status.prefill_label = agent_next_prefill_label();
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}

static int worker_status_power_locked(agent_worker *w) {
    if (w->power_requested) return w->requested_power;
    int power = w->cfg->engine.power_percent;
    return power > 0 ? power : 100;
}

/* Request interruption at the next model/tool polling point. */
static void worker_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = true;
    pthread_cond_broadcast(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* Stop the worker thread. */
static void worker_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    pthread_cond_broadcast(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* Finish an interrupted turn, including any accepted prompt, before asking
 * about exit. Condition waits also release workers awaiting UI responses. */
static void worker_pause(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->paused = true;
    if (w->active || w->cmd_text) w->interrupt = true;
    pthread_cond_broadcast(&w->cond);
    while (w->active || w->cmd_text)
        pthread_cond_wait(&w->cond, &w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}

static void worker_resume(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->paused = false;
    pthread_cond_broadcast(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order. */
static void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    if (out) {
        *out = w->out;
        *out_len = w->out_len;
        w->out = NULL;
        w->out_len = 0;
        w->out_cap = 0;
    }
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    w->status.power_percent = worker_status_power_locked(w);
    w->status.think_mode = w->cfg->gen.think_mode;
    w->status.thinking_budget = w->cfg->gen.thinking_budget;
    w->status.qwen38 = p36_engine_is_qwen38(w->engine);
    if (status) *status = w->status;
    w->wake_pending = false;
    pthread_mutex_unlock(&w->mu);
}

static void worker_get_status(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    w->status.power_percent = worker_status_power_locked(w);
    w->status.think_mode = w->cfg->gen.think_mode;
    w->status.thinking_budget = w->cfg->gen.thinking_budget;
    w->status.qwen38 = p36_engine_is_qwen38(w->engine);
    *status = w->status;
    pthread_mutex_unlock(&w->mu);
}

static bool worker_is_idle(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool idle = w->initialized && !w->active && !w->cmd_text &&
        (w->paused || (!w->save_requested && !w->compact_requested &&
                       !w->power_requested)) &&
        (w->status.state == AGENT_WORKER_IDLE ||
         w->status.state == AGENT_WORKER_ERROR);
    pthread_mutex_unlock(&w->mu);
    return idle;
}

static bool worker_is_initialized(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    w->status.power_percent = worker_status_power_locked(w);
    if (status) *status = w->status;
    bool initialized = w->initialized;
    pthread_mutex_unlock(&w->mu);
    return initialized;
}

static bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}

static char *agent_format_user_prompt_echo(const char *text) {
    agent_buf b = {0};
    if (stdout_is_tty()) {
        agent_buf_puts(&b, "\x1b[1;91m*\x1b[1;97m ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\x1b[0m\n\n");
    } else {
        agent_buf_puts(&b, "* ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\n\n");
    }
    return agent_buf_take(&b);
}

static void agent_echo_user_prompt(const char *text) {
    char *msg = agent_format_user_prompt_echo(text);
    printf("%s", msg);
    fflush(stdout);
    free(msg);
}

/* ============================================================================
 * Terminal Prompt, Status Footer, And Async Output Rendering
 * ============================================================================
 */

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len);
#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

static void agent_progress_append(char *buf, size_t len, size_t *pos,
                                  const char *s) {
    if (len == 0 || *pos >= len - 1) return;
    size_t avail = len - *pos;
    int n = snprintf(buf + *pos, avail, "%s", s);
    if (n <= 0) return;
    if ((size_t)n >= avail) *pos = len - 1;
    else *pos += (size_t)n;
}

static void build_prompt_text(const agent_status *st, char *buf, size_t len) {
    (void)st;
    snprintf(buf, len, "p36-agent> ");
}

static void agent_progress_bar(int done, int total, double tps,
                               char *buf, size_t len, bool color) {
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * AGENT_PROGRESS_BAR_WIDTH) / total);
    if (filled < 0) filled = 0;
    if (filled > AGENT_PROGRESS_BAR_WIDTH) filled = AGENT_PROGRESS_BAR_WIDTH;
    if (color && filled == 0 && done < total) filled = 1;
    char rate[32] = {0};
    size_t rate_len = 0;
    if (tps > 0.0 && filled < AGENT_PROGRESS_BAR_WIDTH) {
        snprintf(rate, sizeof(rate), " %.0ft/s", tps);
        rate_len = strlen(rate);
    }
    size_t pos = 0;
    agent_progress_append(buf, len, &pos, "[");
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_BAR_FILL);
    for (int i = 0; i < AGENT_PROGRESS_BAR_WIDTH && pos + 1 < len; i++) {
        if (color && i == filled) {
            agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
        }
        if (i >= filled && rate_len > 0 && (size_t)(i - filled) < rate_len) {
            char ch[2] = {rate[i - filled], '\0'};
            agent_progress_append(buf, len, &pos, ch);
        } else {
            agent_progress_append(buf, len, &pos, i < filled ? "▶" : "·");
        }
    }
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
    agent_progress_append(buf, len, &pos, "]");
    buf[pos < len ? pos : len - 1] = '\0';
}

static void agent_power_status_suffix(const agent_status *st,
                                      char *buf, size_t len) {
    if (len == 0) return;
    if (st->power_percent > 0 && st->power_percent < 100)
        snprintf(buf, len, " | ⚡ %d%%", st->power_percent);
    else
        buf[0] = '\0';
}

static unsigned agent_next_prefill_label(void) {
    static unsigned next;
    return next++;
}

/* Keep each prefill operation on a single playful label so the footer does not
 * visually churn while progress updates stream in. */
static const char *agent_prefill_label(const agent_status *st) {
    static const char *labels[] = {
        "reading",
        "absorbing",
        "studying",
        "gathering",
        "crunching",
        "scrutinizing",
    };
    size_t n = sizeof(labels) / sizeof(labels[0]);
    return labels[(st ? st->prefill_label : 0u) % n];
}

/* Build the one-line footer shown below the prompt.  It is intentionally compact
 * because linenoise redraws it on every progress update. */
static void build_status_text(const agent_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32];
    char power[32];
    agent_format_ctx_size(st->ctx_used, used, sizeof(used));
    agent_format_ctx_size(st->ctx_size, total_ctx, sizeof(total_ctx));
    agent_power_status_suffix(st, power, sizeof(power));

    switch (st->state) {
    case AGENT_WORKER_PREFILL: {
        int done = st->prefill_done;
        int total = st->prefill_total > 0 ? st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[AGENT_PROGRESS_BAR_MAX_BYTES];
        agent_progress_bar(done, total, st->prefill_tps, bar, sizeof(bar),
                           stdout_is_tty());
        snprintf(buf, len, "ctx %s/%s | %s %s %d/%d %.1f%%%s",
                 used, total_ctx, agent_prefill_label(st), bar,
                 done, total, pct, power);
        break;
    }
    case AGENT_WORKER_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens%s %.1f t/s%s",
                 used, total_ctx, st->generated,
                 st->greedy_sampling ? " ❄️" : "", st->gen_tps, power);
        break;
    case AGENT_WORKER_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s",
                 used, total_ctx, st->generated, st->gen_tps, power);
        break;
    case AGENT_WORKER_SAVING:
        snprintf(buf, len, "ctx %s/%s | saving session%s", used, total_ctx, power);
        break;
    case AGENT_WORKER_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s%s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error", power);
        break;
    case AGENT_WORKER_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted%s", used, total_ctx, power);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle%s", used, total_ctx, power);
        break;
    }
    size_t n = strlen(buf);
    const char *mode = st->think_mode == P36_THINK_NONE ? "off" :
                       st->qwen38 ? p36_think_mode_name(st->think_mode) : "on";
    if (n < len) {
        if (st->think_mode == P36_THINK_NONE)
            snprintf(buf + n, len - n, " | think %s (Tab)", mode);
        else
            snprintf(buf + n, len - n, " | think %s %dk (Tab)", mode,
                     st->thinking_budget / 1000);
    }
}

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

static void agent_prompt_queue_push(agent_prompt_queue *q, const char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    q->v[q->len++] = xstrdup(text ? text : "");
}

static char *agent_prompt_queue_pop(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    char *text = q->v[0];
    memmove(q->v, q->v + 1, (q->len - 1) * sizeof(q->v[0]));
    q->len--;
    return text;
}

static void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    memmove(q->v + 1, q->v, q->len * sizeof(q->v[0]));
    q->v[0] = text;
    q->len++;
}

static char *agent_prompt_queue_take_all(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    if (q->len == 1) return agent_prompt_queue_pop(q);

    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char hdr[64];
        if (i) agent_buf_puts(&b, "\n\n");
        snprintf(hdr, sizeof(hdr), "Queued user message %zu:\n", i + 1);
        agent_buf_puts(&b, hdr);
        agent_buf_puts(&b, q->v[i]);
        free(q->v[i]);
    }
    q->len = 0;
    return agent_buf_take(&b);
}

static char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char *echo = agent_format_user_prompt_echo(q->v[i]);
        agent_buf_puts(&b, echo);
        free(echo);
    }
    return agent_buf_take(&b);
}

static const char *agent_prompt_queue_peek(const agent_prompt_queue *q) {
    return q->len ? q->v[0] : NULL;
}

static void agent_prompt_queue_free(agent_prompt_queue *q) {
    for (size_t i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

static bool agent_footer_is_multiline(const char *status) {
    return status && strchr(status, '\n');
}

/* Build the editable footer.  With queued prompts, the footer becomes multiple
 * rows: a compact queue preview first, then the normal status row. */
static void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len) {
    char status[512];
    build_status_text(st, status, sizeof(status));
    if (!queue || !queue->len) {
        snprintf(buf, len, "%s", status);
        return;
    }

    const char *queued = agent_prompt_queue_peek(queue);
    if (cols < 1) cols = 1;
    int max_rows = 3;
    size_t budget = (size_t)cols * (size_t)max_rows;
    const char *plain_suffix = " (ctrl+x to edit, ESC to send ASAP)";
    size_t queued_len = strlen(queued), pos = 0, cells = 8;
    size_t reserve = strlen(plain_suffix) + 4;
    size_t preview_bytes = len > 1024 ? len - 1024 : 0;
    agent_buf msg = {0};
    agent_buf_puts(&msg, "queued: ");
    while (pos < queued_len) {
        int width;
        size_t n = linenoiseNextGrapheme(queued + pos, queued_len - pos, &width);
        if (!n) break;
        bool control = (unsigned char)queued[pos] < 32 || queued[pos] == 127;
        if (control) width = 1;
        if (cells + (size_t)width + reserve > budget || msg.len + n > preview_bytes) break;
        agent_buf_append(&msg, control ? " " : queued + pos, control ? 1 : n);
        cells += (size_t)width;
        pos += n;
    }
    if (pos < queued_len) agent_buf_puts(&msg, "...");
    agent_buf_puts(&msg, plain_suffix);
    char *preview = agent_buf_take(&msg);

    agent_buf out = {0};
    pos = 0;
    size_t preview_len = strlen(preview);
    for (int row = 0; row < max_rows && pos < preview_len; row++) {
        if (row) agent_buf_puts(&out, "\n");
        if (stdout_is_tty()) agent_buf_puts(&out, AGENT_QUEUE_STYLE);
        cells = 0;
        while (pos < preview_len) {
            int width;
            size_t n = linenoiseNextGrapheme(preview + pos, preview_len - pos, &width);
            if (!n || cells + (size_t)width > (size_t)cols) break;
            agent_buf_append(&out, preview + pos, n);
            cells += (size_t)width;
            pos += n;
        }
        if (stdout_is_tty()) agent_buf_puts(&out, "\x1b[0m");
    }
    agent_buf_puts(&out, "\n");
    if (stdout_is_tty()) agent_buf_puts(&out, AGENT_STATUS_STYLE_START);
    agent_buf_puts(&out, status);
    snprintf(buf, len, "%s", out.ptr ? out.ptr : "");
    free(preview);
    free(out.ptr);
}

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    bool prompt_dirty, status_dirty;
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool output_pending_wrap;
    char output_utf8[4];
    size_t output_utf8_len;
    int output_escape;
    bool output_zwj, output_regional;
    int output_glyph_width;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    agent_input_buf deferred_output;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len);
static void editor_hide(agent_editor *ed);
static void editor_show(agent_editor *ed);
static bool editor_write_scroll_output_preserve_prompt(agent_editor *ed,
                                                       const char *text,
                                                       size_t len,
                                                       bool settle_boundary);

typedef enum {
    CPR_INVALID,
    CPR_PARTIAL,
    CPR_COMPLETE,
} cpr_state;

/* Classify a possible terminal cursor-position reply (ESC[row;colR).  User
 * keystrokes can arrive interleaved with these replies, so we only swallow bytes
 * when they are definitely part of a complete CPR sequence. */
static cpr_state cpr_candidate_state(const char *buf, size_t len) {
    if (len == 0) return CPR_PARTIAL;
    if ((unsigned char)buf[0] != 0x1b) return CPR_INVALID;
    if (len == 1) return CPR_PARTIAL;
    if (buf[1] != '[') return CPR_INVALID;
    if (len == 2) return CPR_PARTIAL;

    size_t p = 2;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    if (buf[p++] != ';') return CPR_INVALID;
    if (p == len) return CPR_PARTIAL;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    return p + 1 == len && buf[p] == 'R' ? CPR_COMPLETE : CPR_INVALID;
}

