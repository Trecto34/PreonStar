static const char *agent_yes_no_auto_name(agent_yes_no_auto answer) {
    switch (answer) {
    case AGENT_YES_NO_AUTO_NO: return "no";
    case AGENT_YES_NO_AUTO_YES: return "yes";
    default: return "";
    }
}

static void agent_clear_password(void *ptr, size_t len) {
    volatile unsigned char *p = ptr;
    while (len--) *p++ = 0;
}

/* This reader deliberately bypasses linenoise, history, and the prompt queue.
 * The only destination for its input is the command's private terminal. */
static bool agent_prompt_password(agent_worker *w,
                                   const agent_password_request *request) {
    struct termios saved, raw, child;
    if (tcgetattr(STDIN_FILENO, &saved) != 0) return false;
    raw = saved;
    cfmakeraw(&raw);
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return false;

    printf("\r\nPassword for bash job %d (input hidden; Ctrl+C cancels)\r\n",
           request->job_id);
    for (const char *p = request->prompt; *p; p++) {
        if (*p == '\n') putchar('\r');
        putchar(*p);
    }
    fflush(stdout);

    char password[1024] = {0};
    size_t len = 0;
    bool ok = false;
    while (now_sec() < request->deadline && !worker_should_interrupt(w) &&
           !agent_sigint) {
        struct pollfd pfd[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = request->fd, .events = 0},
        };
        int rc = poll(pfd, 2, 100);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0 || (pfd[0].revents & (POLLHUP | POLLERR | POLLNVAL)) ||
            (pfd[1].revents & (POLLHUP | POLLERR | POLLNVAL))) break;
        if (tcgetattr(request->fd, &child) != 0 || (child.c_lflag & ECHO)) break;
        if (!(pfd[0].revents & POLLIN)) continue;
        unsigned char c;
        if (read(STDIN_FILENO, &c, 1) != 1) break;
        if (c == 3 || c == 4 || c == 27) break;
        if (c == '\r' || c == '\n') {
            password[len++] = '\n';
            size_t sent = 0;
            while (sent < len) {
                ssize_t n = write(request->fd, password + sent, len - sent);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) break;
                sent += (size_t)n;
            }
            ok = sent == len;
            break;
        }
        if (c == 127 || c == 8) {
            if (len) {
                do { len--; } while (len && (password[len] & 0xc0) == 0x80);
            }
        } else if (c == 21) {
            len = 0;
        } else if (c >= 32 && len + 1 < sizeof(password)) {
            password[len++] = c;
        }
    }
    agent_clear_password(password, sizeof(password));
    /* Discard input following Enter/cancellation so a pasted secret suffix
     * cannot become a chat message when the editor resumes. */
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved);
    printf("\n%s", ok ? "" : "Password entry cancelled.\n");
    fflush(stdout);
    return ok;
}

static int editor_prompt_password(agent_editor *editor, agent_worker *w,
                                   const agent_password_request *request,
                                   const char *prompt, const char *status) {
    char *saved_input = NULL;
    if (editor->active && editor->edit.buf && editor->edit.len)
        saved_input = xstrndup(editor->edit.buf, editor->edit.len);
    editor_stop(editor);
    editor_restore_terminal_layout(editor);
    bool ok = agent_prompt_password(w, request);
    worker_answer_password(w, ok);
    int rc = editor_start(editor, prompt, status, saved_input);
    free(saved_input);
    return rc;
}

/* Shared y/n prompt.  By default it blocks forever like the historical helper;
 * callers that cannot safely stall the agent can request an automatic answer
 * after timeout_sec seconds. */
static bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out) {
    char buf[32];
    int timeout_sec = opts ? opts->timeout_sec : 0;
    agent_yes_no_auto auto_answer = opts ?
        opts->timeout_answer : AGENT_YES_NO_AUTO_NONE;
    bool use_timeout = timeout_sec > 0 && auto_answer != AGENT_YES_NO_AUTO_NONE;
    double deadline = use_timeout ? now_sec() + timeout_sec : 0.0;

    if (timed_out) *timed_out = false;
    for (;;) {
        printf("%s", prompt);
        if (use_timeout) {
            int rem = (int)(deadline - now_sec() + 0.999);
            if (rem < 0) rem = 0;
            printf("[auto-%s in %ds] ", agent_yes_no_auto_name(auto_answer), rem);
        }
        fflush(stdout);
        if (use_timeout) {
            double rem_sec = deadline - now_sec();
            if (rem_sec <= 0.0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
            int timeout_ms = (int)(rem_sec * 1000.0) + 1;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout_ms);
            } while (rc < 0 && errno == EINTR);
            if (rc == 0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            if (rc < 0) return false;
        }
        /* stdin may be in non-blocking mode (set by editor_start).
         * Temporarily switch to blocking so fgets can wait for input. */
        int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags & ~O_NONBLOCK);
        }
        bool got_line = fgets(buf, sizeof(buf), stdin) != NULL;
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags);
        }
        if (!got_line) return false;
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == 'y' || *p == 'Y') return true;
        if (*p == 'n' || *p == 'N') return false;
    }
}

static bool agent_prompt_yes_no(const char *prompt) {
    return agent_prompt_yes_no_ex(prompt, NULL, NULL);
}

/* Ask before discarding a dirty user session.  Fresh sessions that contain only
 * the system prompt are deliberately ignored. */
static bool agent_maybe_save_before_leaving_session(agent_worker *w) {
    if (!agent_worker_needs_save(w)) return true;
    if (!agent_prompt_yes_no("Save current session? (y/n) ")) return true;
    char err[160] = {0};
    if (agent_worker_save_session(w, err, sizeof(err))) return true;
    printf("save failed: %s\n", err);
    return agent_prompt_yes_no("Continue anyway? (y/n) ");
}

typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

/* Process exit is different from /new or /switch: once the terminal is already
 * restored, declining the save can terminate immediately and let the OS reclaim
 * model/runtime resources instead of waiting for orderly teardown. */
static agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w) {
    worker_pause(w);
    if (!agent_worker_needs_save(w)) return AGENT_EXIT_CLEAN;
    if (!agent_prompt_yes_no("Save current session? (y/n) ")) return AGENT_EXIT_NOW;
    char err[160] = {0};
    if (agent_worker_save_session(w, err, sizeof(err))) return AGENT_EXIT_CLEAN;
    printf("save failed: %s\n", err);
    if (agent_prompt_yes_no("Continue anyway? (y/n) ")) return AGENT_EXIT_NOW;
    worker_resume(w);
    return AGENT_EXIT_CANCEL;
}

/* ============================================================================
 * Interactive Runtime Loop
 * ============================================================================
 */

static void agent_noninteractive_marker(const char *msg) {
    write_all(STDERR_FILENO, msg, strlen(msg));
    write_all(STDERR_FILENO, "\n", 1);
}

static int agent_read_stdin_available(agent_input_buf *in, bool *eof) {
    char buf[4096];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            agent_input_buf_append(in, buf, (size_t)n);
            continue;
        }
        if (n == 0) {
            *eof = true;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        perror("p36-agent: read stdin");
        return -1;
    }
}

/* Headless mode is intentionally just another front-end for the same worker.
 * With -p/--prompt it is a one-shot execution.  Without -p it becomes a small
 * stdin protocol: announce readiness on stderr, collect bytes until stdin has
 * been quiet for 200 ms, submit that buffer as one prompt, and keep reading so
 * later input can be queued while the model is still working. */
static int run_agent_non_interactive(p36_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    const bool one_shot = cfg->gen.prompt != NULL;
    bool one_shot_submitted = false;
    bool stdin_eof = false;
    bool waiting_announced = false;
    bool stdin_nonblock = false;
    int old_stdin_flags = 0;
    agent_input_buf input = {0};
    agent_prompt_queue queue = {0};
    double quiet_deadline = 0.0;
    int rc = 0;

    if (!one_shot) {
        if (set_nonblock(STDIN_FILENO, true, &old_stdin_flags) != 0) {
            perror("p36-agent: nonblocking stdin");
            agent_worker_free(&worker);
            return 1;
        }
        stdin_nonblock = true;
    }

    while (true) {
        bool initialized = worker_is_initialized(&worker, NULL);
        bool idle = worker_is_idle(&worker);

        if (one_shot && !one_shot_submitted && initialized) {
            if (worker_submit(&worker, cfg->gen.prompt))
                one_shot_submitted = true;
            idle = false;
        }

        if (!one_shot && queue.len && idle) {
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                idle = false;
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(queued);
        }

        if (!one_shot && initialized && idle && !queue.len &&
            input.len == 0 && !stdin_eof && !waiting_announced)
        {
            agent_noninteractive_marker("+QUARKSTAR_WAITING");
            waiting_announced = true;
        }

        int timeout_ms = -1;
        if (!one_shot && input.len > 0) {
            double rem = quiet_deadline - now_sec();
            timeout_ms = rem <= 0.0 ? 0 : (int)(rem * 1000.0) + 1;
        }

        struct pollfd pfd[2];
        int nfds = 0;
        int wake_idx = nfds;
        pfd[nfds++] = (struct pollfd){.fd = worker.wake_fd[0], .events = POLLIN};
        int stdin_idx = -1;
        if (!one_shot && initialized && !stdin_eof) {
            stdin_idx = nfds;
            pfd[nfds++] = (struct pollfd){.fd = STDIN_FILENO, .events = POLLIN};
        }

        int prc = poll(pfd, (nfds_t)nfds, timeout_ms);
        if (prc < 0) {
            if (errno == EINTR) continue;
            perror("p36-agent: poll");
            rc = 1;
            break;
        }
        if (pfd[wake_idx].revents & POLLIN) drain_wake_fd(worker.wake_fd[0]);
        if (stdin_idx >= 0 && (pfd[stdin_idx].revents & (POLLIN | POLLHUP))) {
            size_t old_len = input.len;
            if (agent_read_stdin_available(&input, &stdin_eof) != 0) {
                rc = 1;
                break;
            }
            if (input.len != old_len) {
                quiet_deadline = now_sec() + 0.200;
                waiting_announced = false;
            }
        }

        char *out = NULL;
        size_t out_len = 0;
        agent_status st = {0};
        worker_consume(&worker, &out, &out_len, &st);
        if (out && out_len) {
            write_all(STDOUT_FILENO, out, out_len);
            fflush(stdout);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *queued = agent_prompt_queue_take_all(&queue);
            worker_answer_queued_user_drain(&worker, queued);
        }

        if (st.state == AGENT_WORKER_ERROR) {
            fprintf(stderr, "p36-agent: %s\n",
                    st.error[0] ? st.error : "worker error");
            rc = 1;
            break;
        }

        if (!one_shot && input.len > 0 &&
            (stdin_eof || now_sec() >= quiet_deadline))
        {
            char *prompt = agent_input_buf_take(&input);
            if (worker_is_idle(&worker) && queue.len == 0) {
                if (!worker_submit(&worker, prompt)) {
                    agent_prompt_queue_push(&queue, prompt);
                    agent_noninteractive_marker("+QUARKSTAR_QUEUED");
                }
            } else {
                agent_prompt_queue_push(&queue, prompt);
                agent_noninteractive_marker("+QUARKSTAR_QUEUED");
            }
            free(prompt);
            waiting_announced = false;
        }

        if (one_shot && one_shot_submitted && worker_is_idle(&worker)) break;
        if (!one_shot && stdin_eof && input.len == 0 &&
            queue.len == 0 && worker_is_idle(&worker))
            break;
    }

    /* Drain anything published between the final status transition and the
     * loop exit.  This keeps stdout complete without adding another protocol. */
    char *out = NULL;
    size_t out_len = 0;
    worker_consume(&worker, &out, &out_len, NULL);
    if (out && out_len) {
        write_all(STDOUT_FILENO, out, out_len);
        fflush(stdout);
    }
    free(out);

    if (stdin_nonblock) fcntl(STDIN_FILENO, F_SETFL, old_stdin_flags);
    agent_input_buf_free(&input);
    agent_prompt_queue_free(&queue);
    agent_worker_free(&worker);
    return rc;
}

/* Main UI loop.  poll() multiplexes stdin with the worker wake pipe; all
 * terminal writes go through editor_write_async() so linenoise, status footer,
 * model output, and tool output never race each other. */
static int run_agent(p36_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    char hist[PATH_MAX];
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(hist, sizeof(hist), "%s/.p36_agent_history", home);
    /* The agent uses ANSI scroll regions when possible: model/tool output
     * scrolls above the live linenoise prompt and status footer, so streaming
     * tokens do not require repainting the bottom rows.  Terminals without
     * scroll-region support fall back to the older prompt-below-output path. */
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    agent_completion_worker = &worker;
    linenoiseSetCompletionCallback(agent_switch_completion_callback);

    agent_status st;
    worker_get_status(&worker, &st);
    char prompt[160];
    char statusline[4096];
    build_prompt_text(&st, prompt, sizeof(prompt));
    build_footer_text(&st, NULL, 80, statusline, sizeof(statusline));

    agent_editor editor = {0};
    agent_prompt_queue queue = {0};
    if (editor_start(&editor, prompt, statusline, NULL) != 0) {
        fprintf(stderr, "p36-agent: failed to start line editor\n");
        agent_worker_free(&worker);
        return 1;
    }
    editor_write_welcome_banner(&editor, cfg, prompt, statusline);

    char *initial_pending = cfg->gen.prompt && cfg->gen.prompt[0] ?
                            xstrdup(cfg->gen.prompt) : NULL;

    bool running = true;
    bool exit_save_handled = false;
    bool show_welcome_after_restart = false;
    bool force_status_redraw_after_restart = false;
    char *restore_line = NULL;
resume_editor:
    while (running) {
        /* If a bash child process changed the terminal mode (e.g., from raw
         * to cooked), restore raw mode so linenoise continues to work. */
        if (worker_check_raw_mode_restore(&worker)) {
            linenoiseRestoreRawMode();
        }
        struct pollfd pfd[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = worker.wake_fd[0], .events = POLLIN},
        };
        int timeout = (!editor.paste_open && !editor.paste_start_pending &&
                       linenoiseEditQueuedInput(&editor.edit) > 0) ? 0 : 100;
        int rc = poll(pfd, 2, timeout);
        if (rc < 0 && errno != EINTR) break;

        if (agent_sigint) {
            agent_sigint = 0;
            if (worker_is_idle(&worker)) {
                editor_cancel_input_with_hint(&editor, prompt, statusline);
            } else {
                worker_interrupt(&worker);
            }
        }

        if (rc > 0 && (pfd[0].revents & POLLIN)) editor_read_stdin(&editor);

        /* Linenoise runs the terminal in raw mode, so Ctrl+C normally arrives
         * as byte 3 instead of SIGINT.  Handle it before worker output is
         * drained and repainted; otherwise a busy decoding stream can leave the
         * interrupt waiting behind a large terminal-output backlog. */
        if (editor_take_queued_byte(&editor, 3)) { /* Ctrl+C */
            if (!worker_is_idle(&worker)) {
                worker_interrupt(&worker);
            } else {
                editor_cancel_input_with_hint(&editor, prompt, statusline);
            }
        }

        while (!editor.paste_open && !editor.paste_start_pending &&
            editor.edit.len == 0 &&
            editor.edit.queued_input_pos < editor.edit.queued_input_len &&
            editor.edit.queued_input[editor.edit.queued_input_pos] == '\t' &&
            worker_is_idle(&worker))
        {
            editor_take_queued_byte(&editor, '\t');
            pthread_mutex_lock(&worker.mu);
            bool qwen38 = p36_engine_is_qwen38(worker.engine);
            agent_cycle_think_mode(cfg, qwen38);
            worker.thinking_enabled = cfg->gen.think_mode != P36_THINK_NONE;
            agent_wake_locked(&worker);
            pthread_mutex_unlock(&worker.mu);
        }

        if (rc > 0 && (pfd[1].revents & POLLIN)) drain_wake_fd(worker.wake_fd[0]);

        char *out = NULL;
        size_t out_len = 0;
        worker_consume(&worker, &out, &out_len, &st);
        build_prompt_text(&st, prompt, sizeof(prompt));
        int footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
        build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
        if ((out && out_len) || editor.deferred_output.len) {
            bool force_show = st.state == AGENT_WORKER_IDLE ||
                              st.state == AGENT_WORKER_ERROR ||
                              st.state == AGENT_WORKER_STOPPED;
            editor_write_async(&editor, out, out_len, prompt, statusline, force_show);
        } else {
            editor_set_prompt_status(&editor, prompt, statusline);
            editor_flush_prompt_status(&editor, st.state == AGENT_WORKER_IDLE ||
                                       st.state == AGENT_WORKER_ERROR || st.state == AGENT_WORKER_STOPPED);
            if (editor.hidden && (st.state == AGENT_WORKER_IDLE ||
                                  st.state == AGENT_WORKER_ERROR ||
                                  st.state == AGENT_WORKER_STOPPED))
                editor_show(&editor);
        }
        if (st.state == AGENT_WORKER_ERROR && st.error[0]) {
            char msg[320];
            int n = snprintf(msg, sizeof(msg), "\np36-agent: %s\n", st.error);
            editor_write_async(&editor, msg, n > 0 ? (size_t)n : 0,
                               prompt, statusline, true);
            pthread_mutex_lock(&worker.mu);
            worker.status.state = AGENT_WORKER_IDLE;
            worker.status.prefill_tps = 0.0;
            worker.status.greedy_sampling = false;
            worker.status.error[0] = '\0';
            pthread_mutex_unlock(&worker.mu);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (echo) {
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
                free(echo);
            }
            worker_answer_queued_user_drain(&worker, queued);
            continue;
        }

        agent_password_request password_request;
        if (worker_take_password_request(&worker, &password_request)) {
            if (editor_prompt_password(&editor, &worker, &password_request,
                                       prompt, statusline) != 0) break;
            continue;
        }

        char web_approval_msg[256];
        if (worker_take_web_approval_request(&worker, web_approval_msg,
                                             sizeof(web_approval_msg)))
        {
            char *saved_input = NULL;
            if (editor.active && editor.edit.buf && editor.edit.len)
                saved_input = xstrndup(editor.edit.buf, editor.edit.len);
            editor_stop(&editor);
            editor_restore_terminal_layout(&editor);
            agent_yes_no_options approval_opts = {
                .timeout_sec = 30,
                .timeout_answer = AGENT_YES_NO_AUTO_NO,
            };
            bool approval_timed_out = false;
            bool allow = agent_prompt_yes_no_ex(web_approval_msg,
                                                &approval_opts,
                                                &approval_timed_out);
            worker_answer_web_approval(&worker, allow,
                approval_timed_out ? "browser start approval timed out" : NULL);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
            editor_start(&editor, prompt, statusline, saved_input);
            free(saved_input);
            continue;
        }

        if (initial_pending && worker_is_idle(&worker)) {
            if (worker_submit(&worker, initial_pending)) {
                free(initial_pending);
                initial_pending = NULL;
            }
        }

        if (!initial_pending && queue.len && worker_is_idle(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                linenoiseHistoryAdd(queued);
                linenoiseHistorySave(hist);
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                if (echo)
                    editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(echo);
            free(queued);
        }

        if (queue.len && editor_take_queued_byte(&editor, 24)) { /* Ctrl+X */
            char *queued = agent_prompt_queue_pop(&queue);
            editor_replace_input(&editor, queued);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
            editor_set_prompt_status(&editor, prompt, statusline);
            free(queued);
        }
        if (queue.len && !worker_is_idle(&worker) && editor_take_bare_escape(&editor)) {
            worker_interrupt(&worker);
        }

        if (!editor.paste_open && !editor.paste_start_pending &&
            linenoiseEditQueuedInput(&editor.edit) > 0)
        {
            if (editor.hidden) {
                /* A user key while the model is in the middle of a partial
                 * output line means the prompt must become visible again. End
                 * the model line explicitly; otherwise linenoise would redraw
                 * on top of generated text. */
                editor_show(&editor);
            }
            errno = 0;
            char *line = linenoiseEditFeed(&editor.edit);
            if (line == linenoiseEditMore) {
                /* Still editing. */
            } else if (!line) {
                if (errno == EAGAIN) {
                    if (!worker_is_idle(&worker)) {
                        worker_interrupt(&worker);
                    } else {
                        editor_cancel_input_with_hint(&editor, prompt, statusline);
                    }
                } else {
                    running = false;
                }
            } else {
                char *cmd = line;
                while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r' || *cmd == '\n') cmd++;
                char *end = cmd + strlen(cmd);
                while (end > cmd && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
                *end = '\0';

                bool was_below_output = editor.prompt_below_output;
                bool had_output_line_open = editor.output_line_open;
                int saved_output_col = editor.output_col;
                editor_stop(&editor);
                bool busy = !worker_is_idle(&worker);
                if (!cmd[0]) {
                    /* Empty input: just reopen the editor. */
                } else if (!strcmp(cmd, "/help")) {
                    runtime_help();
                } else if (!strcmp(cmd, "/save")) {
                    if (busy) {
                        worker_request_save(&worker);
                        printf("save scheduled at next safe point\n");
                    } else {
                        char err[160] = {0};
                        if (!agent_worker_save_session(&worker, err, sizeof(err)))
                            printf("save failed: %s\n", err);
                    }
                } else if (!strcmp(cmd, "/compact")) {
                    worker_request_compact(&worker);
                    if (busy)
                        printf("compaction scheduled at next safe point\n");
                } else if (!strcmp(cmd, "/list")) {
                    agent_worker_list_sessions(&worker);
                } else if (!strncmp(cmd, "/power", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /power <1..100>\n");
                    } else {
                        int power = 0;
                        if (!parse_power_percent(arg, &power)) {
                            printf("usage: /power <1..100>\n");
                        } else {
                            worker_request_power(&worker, power);
                        }
                    }
                } else if (agent_slash_command_with_args(cmd, "/hints")) {
                    char *arg = cmd + strlen("/hints");
                    while (*arg == ' ' || *arg == '\t') arg++;
                    bool enabled;
                    if (!agent_parse_hints(arg, &enabled)) {
                        printf("usage: /hints on|off\n");
                    } else {
                        pthread_mutex_lock(&worker.mu);
                        worker.hints.enabled = enabled;
                        pthread_mutex_unlock(&worker.mu);
                        printf("hints %s (applies at the next conversation boundary)\n",
                               enabled ? "on" : "off");
                    }
                } else if (!strncmp(cmd, "/steer", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    if (busy) {
                        printf("command requires the model to be idle: %s\n", cmd);
                    } else {
                        char *arg = cmd + 6;
                        while (*arg == ' ' || *arg == '\t') arg++;
                        if (!arg[0]) {
                            printf("Steering FFN: %g.\n",
                                   (double)p36_session_directional_steering_ffn(
                                       worker.session));
                        } else {
                            float scale = 0.0f;
                            if (!parse_steering_level(arg, &scale)) {
                                printf("usage: /steer <-100..100>\n");
                            } else if (p36_session_set_directional_steering_ffn(
                                           worker.session, scale) == 0) {
                                worker.cfg->engine.directional_steering_ffn = scale;
                                printf("Steering FFN: %g.\n", (double)scale);
                            }
                        }
                    }
                } else if (cmd[0] == '/' && !agent_slash_command_known(cmd)) {
                    ssize_t ignored = write(STDOUT_FILENO, "\a", 1);
                    (void)ignored;
                    restore_line = xstrdup(cmd);
                } else if (cmd[0] == '/' && busy) {
                    printf("command requires the model to be idle: %s\n", cmd);
                } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
                    /* Stop the editor so raw mode and non-blocking stdin are
                     * disabled before we prompt the user.  Then restore the
                     * ANSI scroll region too; AGENT_EXIT_NOW exits directly. */
                    editor_stop(&editor);
                    editor_restore_terminal_layout(&editor);
                    agent_exit_save_result exit_save =
                        agent_maybe_save_before_exiting(&worker);
                    if (exit_save == AGENT_EXIT_NOW) {
                        exit(0);
                    } else if (exit_save == AGENT_EXIT_CLEAN) {
                        exit_save_handled = true;
                        running = false;
                    }
                } else if (!strcmp(cmd, "/new")) {
                    editor_restore_terminal_layout(&editor);
                    if (agent_maybe_save_before_leaving_session(&worker)) {
                        char err[160] = {0};
                        if (!agent_worker_reset_to_sysprompt(&worker, err, sizeof(err))) {
                            printf("new session failed: %s\n", err);
                        } else {
                            show_welcome_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/switch", 7) &&
                           (cmd[7] == '\0' || cmd[7] == ' ' || cmd[7] == '\t')) {
                    char *arg = cmd + 7;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /switch <sha-prefix>\n");
                    } else {
                        editor_restore_terminal_layout(&editor);
                        if (agent_maybe_save_before_leaving_session(&worker)) {
                            char *sha = arg;
                            while (*arg && *arg != ' ' && *arg != '\t') arg++;
                            if (*arg) *arg = '\0';
                            char err[160] = {0};
                            if (!agent_worker_switch_session(&worker, sha,
                                                             AGENT_HISTORY_DEFAULT_TURNS,
                                                             err, sizeof(err)))
                                printf("switch failed: %s\n", err);
                            else
                                force_status_redraw_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/del", 4) &&
                           (cmd[4] == '\0' || cmd[4] == ' ' || cmd[4] == '\t')) {
                    char *arg = cmd + 4;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /del <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        char err[160] = {0};
                        if (agent_worker_delete_session(&worker, sha_arg,
                                                        sha, err, sizeof(err)))
                            printf("deleted session %.8s\n", sha);
                        else
                            printf("delete failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/strip", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /strip <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        uint32_t tokens = 0;
                        char err[160] = {0};
                        if (agent_worker_strip_session(&worker, sha_arg,
                                                       sha, &tokens,
                                                       err, sizeof(err)))
                            printf("stripped session %.8s (%u tokens)\n",
                                   sha, tokens);
                        else
                            printf("strip failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/history", 8) &&
                           (cmd[8] == '\0' || cmd[8] == ' ' || cmd[8] == '\t')) {
                    char *arg = cmd + 8;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    int history_turns = arg[0] ?
                        agent_parse_int_default(arg, AGENT_HISTORY_DEFAULT_TURNS,
                                                1, AGENT_HISTORY_MAX_TURNS) :
                        AGENT_HISTORY_DEFAULT_TURNS;
                    char err[160] = {0};
                    if (!agent_worker_show_history(&worker, history_turns,
                                                   err, sizeof(err)))
                        printf("history failed: %s\n", err);
                } else if (busy) {
                    agent_prompt_queue_push(&queue, cmd);
                } else {
                    linenoiseHistoryAdd(cmd);
                    linenoiseHistorySave(hist);
                    if (worker_submit(&worker, cmd)) {
                        agent_echo_user_prompt(cmd);
                    } else {
                        restore_line = xstrdup(cmd);
                    }
                }
                linenoiseFree(line);

                if (running) {
                    worker_get_status(&worker, &st);
                    build_prompt_text(&st, prompt, sizeof(prompt));
                    int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
                    build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
                    editor_start(&editor, prompt, statusline, restore_line);
                    if (!editor.scroll_region && was_below_output) {
                        editor.output_line_open = had_output_line_open;
                        editor.prompt_below_output = was_below_output;
                        editor.output_col = saved_output_col;
                    }
                    if (show_welcome_after_restart) {
                        editor_write_welcome_banner(&editor, cfg, prompt, statusline);
                        show_welcome_after_restart = false;
                    }
                    if (force_status_redraw_after_restart) {
                        editor_write_async(&editor, "", 0, prompt, statusline, true);
                        force_status_redraw_after_restart = false;
                    }
                    free(restore_line);
                    restore_line = NULL;
                }
            }
        }
    }

    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    if (!exit_save_handled) {
        agent_exit_save_result exit_save =
            agent_maybe_save_before_exiting(&worker);
        if (exit_save == AGENT_EXIT_NOW) exit(0);
        if (exit_save == AGENT_EXIT_CANCEL) {
            running = true;
            editor_start(&editor, prompt, statusline, NULL);
            goto resume_editor;
        }
    }
    free(initial_pending);
    free(restore_line);
    agent_prompt_queue_free(&queue);
    linenoiseSetCompletionCallback(NULL);
    agent_completion_worker = NULL;
    agent_worker_free(&worker);
    return 0;
}

#ifndef P36_AGENT_TEST_NO_MAIN
int main(int argc, char **argv) {
    p36_cli_expand_long_equals(&argc, &argv, "p36-agent");
    agent_config cfg = parse_options(argc, argv);
    char workdir[PATH_MAX] = {0};
    if (cfg.workspace_path && !cfg.chdir_path) cfg.chdir_path = cfg.workspace_path;
    if (cfg.chdir_path) {
        struct stat st;
        if (!agent_expand_home_path(cfg.chdir_path, workdir, sizeof(workdir))) {
            fprintf(stderr, "p36-agent: failed to chdir to %s: path is too long\n",
                    cfg.chdir_path);
            return 1;
        }
        if (stat(workdir, &st) != 0) {
            fprintf(stderr, "p36-agent: failed to chdir to %s: %s\n",
                    cfg.chdir_path, strerror(errno));
            return 1;
        }
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "p36-agent: failed to chdir to %s: not a directory\n",
                    cfg.chdir_path);
            return 1;
        }
    }
    p36_engine *engine = NULL;
    cfg.engine.context_size = cfg.gen.ctx_size;
    if (cfg.server_url) {
        /* No local model: the server owns weights, template and KV. */
        char err[300] = {0}, *model = NULL;
        int server_ctx = 0;
        p36_remote_cfg rc = {cfg.server_url, cfg.api_key, cfg.server_timeout};
        if (p36_remote_models(&rc, &model, &server_ctx, err, sizeof(err)) != 0) {
            fprintf(stderr, "p36-agent: cannot reach %s: %s\n", cfg.server_url, err);
            return 1;
        }
        if (!cfg.server_model) cfg.server_model = model;
        if (server_ctx > 0 && server_ctx < cfg.gen.ctx_size) cfg.gen.ctx_size = server_ctx;
        agent_resolve_thinking_config(&cfg, false);
    } else {
    if (p36_engine_open(&engine, &cfg.engine) != 0) return 1;
    agent_resolve_thinking_config(&cfg, p36_engine_is_qwen38(engine));
    }
    if (cfg.chdir_path) {
        if (chdir(workdir) != 0) {
            fprintf(stderr, "p36-agent: failed to chdir to %s: %s\n",
                    cfg.chdir_path, strerror(errno));
            p36_engine_close(engine);
            return 1;
        }
    }
    /* Establish WORKSPACE_ROOT once; from here on only the config holds it. */
    {
        char want[PATH_MAX], cwd[PATH_MAX];
        if (cfg.workspace_path ? !agent_expand_home_path(cfg.workspace_path, want, sizeof(want))
                               : !getcwd(want, sizeof(want))) want[0] = 0;
        if (!want[0] || !realpath(want, cfg.ws_root) || !getcwd(cwd, sizeof(cwd)) ||
            !agent_ws_inside(cfg.ws_root, cwd)) {
            fprintf(stderr, "p36-agent: cannot establish the workspace root (%s)\n",
                    cfg.workspace_path ? cfg.workspace_path : "current directory");
            p36_engine_close(engine);
            return 1;
        }
        if (cfg.yolo)
            fprintf(stderr, "p36-agent: WARNING --yolo: workspace jail OFF. File tools and the shell can read and "
                            "modify the whole host filesystem.\n");
        else if (cfg.unsafe_fs)
            fprintf(stderr, "p36-agent: WARNING --unsafe-filesystem: file tools may access paths outside %s "
                            "(the shell stays confined).\n", cfg.ws_root);
    }
    if (!cfg.server_url) {
        agent_apply_sampling_defaults(&cfg, engine);
        log_context_memory(cfg.engine.backend,
                           cfg.gen.ctx_size,
                           cfg.engine.prefill_chunk,
                           cfg.engine.cache_type_k,
                           cfg.engine.cache_type_v);
    }

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = agent_sigint_handler;
    bool sigint_installed = !cfg.non_interactive &&
        sigaction(SIGINT, &sa, &old_int) == 0;

    int rc = cfg.non_interactive ?
        run_agent_non_interactive(engine, &cfg) :
        run_agent(engine, &cfg);

    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    if (engine) p36_engine_close(engine);
    p36_prompt_prefix_free(&cfg.gen.prefix);
    return rc;
}
#endif
