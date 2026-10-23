static char *agent_tool_search(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    char expanded[PATH_MAX];
    if (!agent_expand_home_path(path, expanded, sizeof(expanded)))
        return xstrdup("Tool error: search path is too long\n");
    const char *mode = agent_tool_arg_value(call, "mode");
    if (mode && strcmp(mode, "literal") && strcmp(mode, "regex"))
        return xstrdup("Tool error: search mode must be literal or regex (POSIX extended)\n");
    struct stat root;
    if (stat(expanded, &root) != 0 || (!S_ISDIR(root.st_mode) && !S_ISREG(root.st_mode)))
        return xstrdup("Tool error: search path is missing, unreadable, or not a regular file/directory\n");
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tool_arg_value(call, "glob"),
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = agent_parse_bool_default(agent_tool_arg_value(call, "case_sensitive"), true),
        .context = agent_parse_int_default(agent_tool_arg_value(call, "context"), 0, 0, 5),
        .max_results = agent_parse_int_default(agent_tool_arg_value(call, "max_results"), 50, 1, 500),
        .out = {.limit = AGENT_TOOL_MAX_BYTES},
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, expanded, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches in searched text\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    ctx.out.limit = 0;
    if (ctx.skipped || ctx.results >= ctx.max_results) {
        bool truncated = ctx.out.truncated;
        ctx.out.truncated = false;
        char note[PATH_MAX + 256];
        snprintf(note, sizeof(note), "\nSearch incomplete: %zu skipped paths%s. %s\n",
                 ctx.skipped, ctx.results >= ctx.max_results ? "; match limit reached" : "", ctx.first_skip);
        agent_buf_puts(&ctx.out, note);
        ctx.out.truncated = truncated;
    }
    return agent_buf_take(&ctx.out);
}

/* ============================================================================
 * Browser Web Tools
 * ============================================================================
 *
 * The browser subsystem lives in p36_web.c: it owns a visible browser and CDP.  The
 * agent side only asks for permission, dispatches tools, and caps visit_page
 * output using the same "head plus temp file" shape as bash.
 */

#define AGENT_WEB_HEAD_BYTES (8*1024)
#define AGENT_WEB_HEAD_LINES 100

static int agent_count_lines(const char *s) {
    if (!s || !s[0]) return 0;
    int lines = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') lines++;
    }
    if (s[strlen(s) - 1] != '\n') lines++;
    return lines;
}

static char *agent_string_head(const char *s, int max_lines, size_t max_bytes,
                               int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!s) return xstrdup("");
    size_t used = 0;
    int lines = 0;
    while (s[used] && used < max_bytes && lines < max_lines) {
        if (s[used++] == '\n') lines++;
    }
    if (s[used] && used >= max_bytes && byte_limited) *byte_limited = true;
    if (used && s[used - 1] != '\n' && lines < max_lines) lines++;
    if (lines_read) *lines_read = lines;
    return xstrndup(s, used);
}

static bool agent_write_temp_text(const char *prefix, const char *text,
                                  char *path, size_t path_len,
                                  char *err, size_t err_len) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(err, err_len, "failed to create temporary file: %s", strerror(errno));
        return false;
    }
    size_t len = text ? strlen(text) : 0;
    const char *p = text ? text : "";
    size_t left = len;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            snprintf(err, err_len, "failed to write temporary file: %s", strerror(errno));
            close(fd);
            unlink(tmpl);
            return false;
        }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) != 0) {
        snprintf(err, err_len, "failed to close temporary file: %s", strerror(errno));
        unlink(tmpl);
        return false;
    }
    snprintf(path, path_len, "%s", tmpl);
    return true;
}

static char *agent_tool_google_search(agent_worker *w, const agent_tool_call *call) {
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: google_search requires query\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Searching Google for %s...", query);
    char *md = p36_web_google_search(w->web, query, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: google_search failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    return md;
}

static char *agent_tool_visit_page(agent_worker *w, const agent_tool_call *call) {
    const char *url = agent_tool_arg_value(call, "url");
    if (!url || !url[0]) return xstrdup("Tool error: visit_page requires url\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Opening page %s...", url);
    char *md = p36_web_visit_page(w->web, url, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char path[PATH_MAX];
    if (!agent_write_temp_text("p36_agent_web", md, path, sizeof(path),
                               err, sizeof(err)))
    {
        free(md);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "could not store rendered page");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int total_lines = agent_count_lines(md);
    int shown_lines = 0;
    bool byte_limited = false;
    char *head = agent_string_head(md, AGENT_WEB_HEAD_LINES, AGENT_WEB_HEAD_BYTES,
                                   &shown_lines, &byte_limited);
    bool truncated = byte_limited || shown_lines < total_lines;
    agent_buf out = {0};
    char line[PATH_MAX + 256];
    snprintf(line, sizeof(line),
             "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n",
             url, path, strlen(md), total_lines);
    agent_buf_puts(&out, line);
    if (truncated) {
        snprintf(line, sizeof(line), "<head -%d %s>\n",
                 AGENT_WEB_HEAD_LINES, path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</head>\n");
        agent_buf_puts(&out,
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n");
    } else {
        agent_buf_puts(&out, "<markdown>\n");
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</markdown>\n");
    }
    free(head);
    free(md);
    return agent_buf_take(&out);
}

/* ============================================================================
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report the total captured output size.
 */

#define AGENT_BASH_HEAD_BYTES (8*1024)
#define AGENT_BASH_HEAD_LINES 100
#define AGENT_BASH_TAIL_BYTES (32*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
#define AGENT_BASH_FINAL_TAIL_LINES 20

struct agent_bash_job {
    pthread_t thread;
    pthread_mutex_t mu;
    bool thread_started;
    int id;
    pid_t pid;
    int pipe_fd;
    int tty_fd;
    int tmp_fd;
    char path[PATH_MAX];
    double start_time;
    double end_time;
    double stop_deadline;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    bool reaped, pipe_eof;
    int child_status;
    int output_error;
    struct agent_bash_job *next;
    agent_worker *worker;  /* back-pointer for terminal state restoration */
};

static int agent_bash_display_lines(const agent_bash_job *job) {
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}

static void agent_bash_note_output(agent_bash_job *job, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n' && job->newline_count < INT_MAX - 1) job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}

static bool agent_bash_is_running(agent_bash_job *job) {
    pthread_mutex_lock(&job->mu);
    bool running = job->running;
    pthread_mutex_unlock(&job->mu);
    return running;
}

static void agent_bash_signal(agent_bash_job *job, int sig) {
    pthread_mutex_lock(&job->mu);
    if (job->running) {
        if (sig == SIGKILL && !job->stop_deadline) job->stop_deadline = now_sec() + 1;
        kill(-job->pid, sig);
        if (!job->reaped) kill(job->pid, sig);
    }
    pthread_mutex_unlock(&job->mu);
}

static void agent_bash_job_free(agent_bash_job *job) {
    if (!job) return;
    agent_bash_signal(job, SIGKILL);
    if (job->thread_started) pthread_join(job->thread, NULL);
    else if (job->pid > 0)
        while (waitpid(job->pid, NULL, 0) < 0 && errno == EINTR) {}
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tty_fd >= 0) close(job->tty_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    pthread_mutex_destroy(&job->mu);
    free(job);
}

static void agent_bash_jobs_free(agent_worker *w) {
    agent_bash_job *job = w->bash_jobs;
    while (job) {
        agent_bash_job *next = job->next;
        agent_bash_job_free(job);
        job = next;
    }
    w->bash_jobs = NULL;
}

static agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid) {
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}

static void agent_bash_remove_job(agent_worker *w, agent_bash_job *target) {
    agent_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            agent_bash_job_free(target);
            return;
        }
        link = &(*link)->next;
    }
}

static void agent_bash_drain(agent_bash_job *job) {
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    /* Yield even for an endless writer, so its deadline is still checked. */
    for (size_t drained = 0; drained < 256 * 1024;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            drained += (size_t)n;
            size_t pos = 0;
            while (pos < (size_t)n) {
                ssize_t wr = write(job->tmp_fd, tmp + pos, (size_t)n - pos);
                if (wr < 0 && errno == EINTR) continue;
                if (wr <= 0) {
                    job->output_error = wr < 0 ? errno : EIO;
                    return;
                }
                agent_bash_note_output(job, tmp + pos, (size_t)wr);
                pos += (size_t)wr;
            }
            continue;
        }
        if (!n) job->pipe_eof = true;
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            job->output_error = errno;
        break;
    }
}

static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}

static void agent_bash_finalize(agent_bash_job *job, int status) {
    agent_bash_drain(job);
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    job->end_time = now_sec();
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the UI thread to verify raw mode at
     * a safe point instead of touching linenoise from the worker path. */
    agent_worker_note_terminal_mode_may_have_changed(job->worker);
}

static void agent_bash_drain_terminal(agent_bash_job *job) {
    if (job->tty_fd < 0) return;
    agent_password_request request = {
        .fd = job->tty_fd,
        .job_id = job->id,
        .deadline = job->start_time + job->timeout_sec,
    };
    size_t used = 0;
    char buf[512];
    ssize_t n;
    while ((n = read(job->tty_fd, buf, sizeof(buf))) > 0) {
        /* Terminal prompts are UI-only. Do not copy them to the model's
         * output file; some password readers enable masked terminal echo. */
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (used + 1 < sizeof(request.prompt) &&
                ((c >= 32 && c != 127) || c == '\n' || c == '\t'))
                request.prompt[used++] = c;
        }
    }
    request.prompt[used] = '\0';
    struct termios mode;
    if (!used || tcgetattr(job->tty_fd, &mode) != 0) return;
    if (mode.c_lflag & ECHO) {
        agent_publish(job->worker, request.prompt, used);
        return;
    }
    if (!agent_request_password(job->worker, &request)) {
        pthread_mutex_lock(&job->mu);
        job->timed_out = now_sec() >= request.deadline;
        const char *msg = job->timed_out ? "Password entry timed out.\n" :
            "Password entry cancelled or unavailable.\n";
        if (job->running && job->tmp_fd >= 0) {
            agent_bash_note_output(job, msg, strlen(msg));
            if (write(job->tmp_fd, msg, strlen(msg)) < 0) job->output_error = errno;
        }
        pthread_mutex_unlock(&job->mu);
        agent_bash_signal(job, SIGKILL);
    }
}

/* Drain available output, notice process exit, and enforce timeout.  This is
 * called opportunistically by status/wait/compaction instead of a background
 * reaper thread, keeping all bash job state owned by the agent worker. */
static void agent_bash_poll(agent_bash_job *job) {
    if (!job || !job->running) return;
    agent_bash_drain(job);

    int status = 0;
    pid_t rc = job->reaped ? 0 : waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        job->reaped = true;
        job->child_status = status;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        job->end_time = now_sec();
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_worker_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (job->reaped && job->pipe_eof) {
        agent_bash_finalize(job, job->child_status);
        return;
    }
    if (!job->timed_out && (job->output_error || now_sec() - job->start_time >= job->timeout_sec)) {
        job->timed_out = !job->output_error;
        kill(-job->pid, SIGKILL);
        if (!job->reaped) {
            kill(job->pid, SIGKILL);
            while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
            job->reaped = true;
            job->child_status = status;
        }
    }
    if (job->output_error || (job->timed_out && now_sec() - job->start_time > job->timeout_sec + 1) ||
        (job->stop_deadline && now_sec() >= job->stop_deadline)) {
        /* A descendant can deliberately escape the shell process group. Do
         * not let its inherited stdout prevent bounded job cleanup. */
        if (!job->output_error && !job->pipe_eof) job->output_error = EIO;
        agent_bash_finalize(job, job->child_status);
    }
}

static void *agent_bash_monitor(void *arg) {
    agent_bash_job *job = arg;
    for (;;) {
        pthread_mutex_lock(&job->mu);
        agent_bash_poll(job);
        bool running = job->running;
        int fd = job->pipe_fd;
        pthread_mutex_unlock(&job->mu);
        if (!running) break;
        struct pollfd pfd = {.fd = fd, .events = POLLIN};
        int rc = poll(&pfd, 1, 50);
        /* A closed stdout must not turn the deadline monitor into a busy loop. */
        if (rc > 0 && (pfd.revents & POLLHUP) && !(pfd.revents & POLLIN))
            usleep(50000);
    }
    return NULL;
}

/* ---- Workspace jail ----------------------------------------------------------
 * WORKSPACE_ROOT is the canonical realpath of the launch directory (or --workspace),
 * fixed in main() and read from the config on every call, so neither the model nor
 * compaction can redefine it.  File tools are gated on the canonical path; bash runs
 * under bubblewrap with only the workspace writable and system dirs read-only, so
 * `find /` cannot see other projects.  --yolo lifts both, --unsafe-filesystem the
 * file-tool gate only.  The path gate is check-then-use (TOCTOU); the sandbox is the
 * real barrier for shell-created links. */
static bool agent_ws_inside(const char *root, const char *p) {
    size_t n = strlen(root);
    if (n == 1 && root[0] == '/') return true;
    return !strncmp(p, root, n) && (p[n] == '/' || p[n] == '\0');
}

/* Canonicalize path even when it does not exist yet: realpath() the longest existing
 * prefix and append the missing tail.  A missing tail containing ".." or a dangling
 * symlink cannot be proven to stay inside, so both fail. */
static bool agent_ws_canon(const char *path, char *out, size_t out_len) {
    char exp[PATH_MAX], head[PATH_MAX], tail[PATH_MAX] = "", res[PATH_MAX];
    if (!agent_expand_home_path(path, exp, sizeof(exp))) return false;
    if (exp[0] == '/') {
        snprintf(head, sizeof(head), "%s", exp);
    } else {
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof(cwd)) ||
            snprintf(head, sizeof(head), "%s/%s", cwd, exp) >= (int)sizeof(head)) return false;
    }
    for (;;) {
        if (realpath(head, res)) break;
        struct stat st;
        if (lstat(head, &st) == 0 && S_ISLNK(st.st_mode)) return false;   /* dangling link */
        char *slash = strrchr(head, '/');
        if (!slash) return false;
        const char *comp = slash + 1;
        if (!strcmp(comp, "..")) return false;
        if (comp[0] && strcmp(comp, ".")) {
            char t[PATH_MAX];
            snprintf(t, sizeof(t), "%s%s%s", comp, tail[0] ? "/" : "", tail);
            snprintf(tail, sizeof(tail), "%s", t);
        }
        if (slash == head) { snprintf(res, sizeof(res), "/"); break; }
        *slash = 0;
    }
    int n = snprintf(out, out_len, "%s%s%s", res, tail[0] && strcmp(res, "/") ? "/" : "", tail);
    return n > 0 && (size_t)n < out_len;
}

static bool agent_ws_jailed(const agent_config *cfg) {
    return cfg && cfg->ws_root[0] && !cfg->yolo;
}

/* NULL when the call may run; otherwise the deterministic refusal. */
static char *agent_ws_gate(agent_worker *w, const agent_tool_call *c) {
    const agent_config *cfg = w->cfg;
    if (cfg && cfg->ws_root[0]) snprintf(w->st.ws_root, sizeof(w->st.ws_root), "%s", cfg->ws_root);
    if (!agent_ws_jailed(cfg) || cfg->unsafe_fs || !c->name) return NULL;
    static const char *const tools[] = {"read", "write", "edit", "search", "list", "view_image"};
    bool file_tool = false;
    for (size_t i = 0; i < sizeof(tools) / sizeof(tools[0]); i++)
        if (!strcmp(c->name, tools[i])) file_tool = true;
    if (!file_tool) return NULL;
    const char *path = agent_tool_arg_value(c, "path");
    if (!path || !path[0]) path = ".";
    char canon[PATH_MAX];
    if (agent_ws_canon(path, canon, sizeof(canon)) && agent_ws_inside(cfg->ws_root, canon)) return NULL;
    char b[2 * PATH_MAX + 200];
    snprintf(b, sizeof(b),
             "Tool error: WORKSPACE_ESCAPE_BLOCKED\nrequested_path: %s\nworkspace_root: %s\n"
             "next=use paths inside workspace_root; files elsewhere on the host are not part of this project\n",
             path, cfg->ws_root);
    return xstrdup(b);
}

/* bwrap argv for one shell command, or NULL when no jail applies.  Storage lives in
 * st so the child can exec without allocating. */
typedef struct { char *argv[64]; char buf[12][PATH_MAX + 16]; int n, nb; } agent_ws_argv;

static const char *agent_ws_arg(agent_ws_argv *a, const char *s) {
    snprintf(a->buf[a->nb], sizeof(a->buf[0]), "%s", s);
    return a->argv[a->n++] = a->buf[a->nb++];
}

static bool agent_ws_sandbox_argv(const agent_config *cfg, const char *cmd, agent_ws_argv *a) {
    if (!agent_ws_jailed(cfg)) return false;
    memset(a, 0, sizeof(*a));
    static const char *const fixed[] = {"bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc",
                                        "--unshare-uts", "--proc", "/proc", "--dev", "/dev",
                                        "--tmpfs", "/tmp"};
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); i++) a->argv[a->n++] = (char *)fixed[i];
    static const char *const sys[] = {"/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc"};
    for (size_t i = 0; i < sizeof(sys) / sizeof(sys[0]); i++) {
        char tgt[PATH_MAX];
        ssize_t l = readlink(sys[i], tgt, sizeof(tgt) - 1);
        if (l > 0) {
            tgt[l] = 0;
            a->argv[a->n++] = (char *)"--symlink";
            agent_ws_arg(a, tgt);
            a->argv[a->n++] = (char *)sys[i];
        } else if (access(sys[i], F_OK) == 0) {
            a->argv[a->n++] = (char *)"--ro-bind";
            a->argv[a->n++] = (char *)sys[i];
            a->argv[a->n++] = (char *)sys[i];
        }
    }
    char tmpd[PATH_MAX + 16];
    snprintf(tmpd, sizeof(tmpd), "%s/.p36-tmp", cfg->ws_root);
    (void)mkdir(tmpd, 0700);                       /* scratch space that stays inside the workspace */
    a->argv[a->n++] = (char *)"--bind";
    const char *root = agent_ws_arg(a, cfg->ws_root);
    a->argv[a->n++] = (char *)root;
    a->argv[a->n++] = (char *)"--chdir";
    a->argv[a->n++] = (char *)root;
    a->argv[a->n++] = (char *)"--setenv";
    a->argv[a->n++] = (char *)"TMPDIR";
    const char *t = agent_ws_arg(a, tmpd);
    a->argv[a->n++] = (char *)"--setenv";
    a->argv[a->n++] = (char *)"HOME";
    a->argv[a->n++] = (char *)t;
    a->argv[a->n++] = (char *)"--";
    a->argv[a->n++] = (char *)"sh";
    a->argv[a->n++] = (char *)"-c";
    a->argv[a->n++] = (char *)(cmd ? cmd : "");
    a->argv[a->n] = NULL;
    return true;
}

/* A separate session prevents /dev/tty readers such as sudo from stealing
 * the editor's input. Interactive jobs get a private controlling terminal. */
static agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len) {
    char tmp_path[] = "/tmp/p36_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    int ttyfd = -1, slavefd = -1;
    if (w->cfg && !w->cfg->non_interactive && isatty(STDIN_FILENO)) {
        ttyfd = posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC);
        if (ttyfd >= 0 && grantpt(ttyfd) == 0 && unlockpt(ttyfd) == 0) {
            const char *name = ptsname(ttyfd);
            if (name) slavefd = open(name, O_RDWR | O_NOCTTY | O_CLOEXEC);
        }
        if (slavefd < 0) {
            snprintf(err, err_len, "failed to create command terminal: %s", strerror(errno));
            if (ttyfd >= 0) close(ttyfd);
            close(pipefd[0]);
            close(pipefd[1]);
            close(tmpfd);
            unlink(tmp_path);
            return NULL;
        }
    }
    fcntl(tmpfd, F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
    agent_ws_argv sandbox;                         /* built before fork: the child must not allocate */
    const bool jailed = agent_ws_sandbox_argv(w->cfg, cmd, &sandbox);
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        if (ttyfd >= 0) close(ttyfd);
        if (slavefd >= 0) close(slavefd);
        unlink(tmp_path);
        return NULL;
    }
    if (pid == 0) {
        if (setsid() < 0) _exit(127);
        if (slavefd >= 0) {
            if (ioctl(slavefd, TIOCSCTTY, 0) < 0) _exit(127);
            close(slavefd);
            close(ttyfd);
        }
        close(tmpfd);
        /* Ordinary stdin reads still get EOF. Only explicit /dev/tty password
         * readers use the private terminal. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        if (jailed) execvp("bwrap", sandbox.argv);   /* fail closed: never fall back to an open shell */
        else execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    if (slavefd >= 0) close(slavefd);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);
    if (ttyfd >= 0) set_nonblock(ttyfd, true, NULL);

    agent_bash_job *job = xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->pipe_fd = pipefd[0];
    job->tty_fd = ttyfd;
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    pthread_mutex_init(&job->mu, NULL);
    job->start_time = now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    int rc = pthread_create(&job->thread, NULL, agent_bash_monitor, job);
    if (rc != 0) {
        snprintf(err, err_len, "failed to start shell monitor: %s", strerror(rc));
        agent_bash_job_free(job);
        unlink(tmp_path);
        return NULL;
    }
    job->thread_started = true;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}


/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
static char *agent_bash_read_head(const agent_bash_job *job, int max_lines,
                                  size_t max_bytes, int *lines_read,
                                  bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    size_t available = job->bytes < max_bytes ? job->bytes : max_bytes;
    while (lines < max_lines && out.len < available) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && out.len < job->bytes && byte_limited) *byte_limited = true;
    fclose(fp);
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}

/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
static char *agent_bash_read_tail_lines(const agent_bash_job *job, int max_lines) {
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    size_t bytes = job->bytes;
    size_t take = bytes < AGENT_BASH_TAIL_BYTES ? bytes : AGENT_BASH_TAIL_BYTES;
    if (fseeko(fp, (off_t)(bytes - take), SEEK_SET) != 0) {
        fclose(fp);
        return xstrdup("<failed to seek output file>\n");
    }
    char tmp[2048];
    while (take) {
        size_t want = take < sizeof(tmp) ? take : sizeof(tmp);
        size_t n = fread(tmp, 1, want, fp);
        if (n) agent_buf_append(&tail, tmp, n);
        take -= n;
        if (n < want) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    char *start = tail.ptr;
    int newlines = tail.ptr[tail.len - 1] == '\n' ? 0 : 1;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    /* A byte-limited tail may begin inside a UTF-8 character. */
    while (((unsigned char)*start & 0xc0) == 0x80) start++;
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}

/* The first observation uses the head; subsequent ones use a bounded tail. */
static char *agent_bash_observation(agent_bash_job *job, bool mark_observed, bool *done) {
    pthread_mutex_lock(&job->mu);
    agent_bash_job snapshot = {
        .id = job->id, .pid = job->pid,
        .start_time = job->start_time, .end_time = job->end_time,
        .timeout_sec = job->timeout_sec, .bytes = job->bytes,
        .newline_count = job->newline_count, .last_byte = job->last_byte,
        .observed_once = job->observed_once, .exit_status = job->exit_status,
        .running = job->running, .timed_out = job->timed_out,
        .output_error = job->output_error,
    };
    memcpy(snapshot.path, job->path, sizeof(snapshot.path));
    if (mark_observed) job->observed_once = true;
    pthread_mutex_unlock(&job->mu);
    /* Disk reads must not hold up the monitor's deadline checks. Only the
     * immutable observation fields above are read from this local snapshot. */
    job = &snapshot;
    if (done) *done = !snapshot.running;
    bool first_observation = !job->observed_once;
    int display_lines = agent_bash_display_lines(job);
    double elapsed = (job->running ? now_sec() : job->end_time) - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }
    if (job->output_error) {
        agent_buf_puts(&out, "Tool error: command output could not be captured completely: ");
        agent_buf_puts(&out, strerror(job->output_error));
        agent_buf_puts(&out, "\n");
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = agent_bash_read_head(job, AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = agent_bash_read_tail_lines(job, tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    return agent_buf_take(&out);
}

/* Shell bytes are data, not instructions to our terminal. Keep only SGR color
 * sequences; OSC, cursor motion, erasure and other controls stay in the log. */
static char *agent_terminal_safe_text(const char *text, size_t len) {
    agent_buf out = {0};
    for (size_t i = 0; i < len;) {
        unsigned char c = (unsigned char)text[i++];
        if (c >= 0x80) {
            uint32_t cp;
            size_t n = linenoiseUtf8Decode(text + i - 1, len - i + 1, &cp);
            if (n > 1 && !(cp >= 0x80 && cp <= 0x9f)) {
                agent_buf_append(&out, text + i - 1, n);
                i += n - 1;
                continue;
            }
            char escaped[5];
            snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            agent_buf_puts(&out, escaped);
            continue;
        }
        if (c == 0x1b) {
            size_t start = i - 1;
            if (i == len) break;
            char kind = text[i++];
            if (kind == '[') {
                bool sgr = true;
                while (i < len && (unsigned char)text[i] < 0x40) {
                    if (!isdigit((unsigned char)text[i]) && text[i] != ';' && text[i] != ':') sgr = false;
                    i++;
                }
                if (i < len) {
                    if (text[i] == 'm' && sgr && i - start < 96)
                        agent_buf_append(&out, text + start, i - start + 1);
                    i++;
                }
            } else if (kind == ']' || kind == 'P' || kind == '_' || kind == '^' || kind == 'X') {
                while (i < len) {
                    if (text[i++] == '\a' && kind == ']') break;
                    if (i >= 2 && text[i - 2] == 0x1b && text[i - 1] == '\\') break;
                }
            } else {
                while (kind >= 0x20 && kind <= 0x2f && i < len) kind = text[i++];
            }
            continue;
        }
        if ((c < 32 && c != '\n' && c != '\r' && c != '\t') || c == 127) {
            char escaped[5];
            snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            agent_buf_puts(&out, escaped);
        } else {
            char ch = (char)c;
            agent_buf_append(&out, &ch, 1);
        }
    }
    return agent_buf_take(&out);
}

