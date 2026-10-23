static void agent_history_ptrs_push(agent_history_ptrs *p, const char *s,
                                    agent_history_mark mark) {
    if (p->len == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 16;
        p->v = xrealloc(p->v, (size_t)p->cap * sizeof(p->v[0]));
        p->mark = xrealloc(p->mark, (size_t)p->cap * sizeof(p->mark[0]));
    }
    p->v[p->len] = s;
    p->mark[p->len] = mark;
    p->len++;
}

static const char *agent_memmem(const char *hay, size_t hay_len,
                                const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    const char first = needle[0];
    const char *end = hay + hay_len - needle_len + 1;
    for (const char *p = hay; p < end; p++) {
        if (*p == first && memcmp(p, needle, needle_len) == 0) return p;
    }
    return NULL;
}

static const char *agent_history_next_marker(const char *p, const char *end,
                                             agent_history_mark *mark,
                                             size_t *mark_len) {
    static const char user_mark[] = "<|im_start|>user\n";
    static const char assistant_mark[] = "<|im_start|>assistant\n";
    static const char eos_mark[] = "<|im_end|>\n";
    const char *u = agent_memmem(p, (size_t)(end - p),
                                 user_mark, sizeof(user_mark) - 1);
    const char *a = agent_memmem(p, (size_t)(end - p),
                                 assistant_mark, sizeof(assistant_mark) - 1);
    const char *e = agent_memmem(p, (size_t)(end - p),
                                 eos_mark, sizeof(eos_mark) - 1);
    if (!u && !a && !e) return NULL;
    if (u && (!a || u < a) && (!e || u < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_USER;
        if (mark_len) *mark_len = sizeof(user_mark) - 1;
        return u;
    }
    if (a && (!e || a < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_ASSISTANT;
        if (mark_len) *mark_len = sizeof(assistant_mark) - 1;
        return a;
    }
    if (mark) *mark = AGENT_HISTORY_MARK_EOS;
    if (mark_len) *mark_len = sizeof(eos_mark) - 1;
    return e;
}

static void agent_history_trim(const char **p, const char **end) {
    while (*p < *end && isspace((unsigned char)**p)) (*p)++;
    while (*end > *p && isspace((unsigned char)(*end)[-1])) (*end)--;
}

static bool agent_history_has_prefix(const char *p, const char *end,
                                     const char *prefix) {
    size_t n = strlen(prefix);
    return (size_t)(end - p) >= n && memcmp(p, prefix, n) == 0;
}

/* Tool messages are rendered as user turns in the transcript.  Return the
 * inner payload for the current <tool_result> wrapper so /history skips these
 * pseudo-user turns and displays their content without leaking the wrapper. */
static bool agent_history_tool_result_payload(const char **p, const char **end) {
    const char *s = *p, *e = *end;
    agent_history_trim(&s, &e);

    const char *open = "<tool_result>";
    const char *close = "</tool_result>";
    const size_t open_len = strlen(open);
    const size_t close_len = strlen(close);
    if (!agent_history_has_prefix(s, e, open)) return false;

    s += open_len;
    if ((size_t)(e - s) >= close_len &&
        memcmp(e - close_len, close, close_len) == 0)
    {
        e -= close_len;
    }
    *p = s;
    *end = e;
    return true;
}

static bool agent_history_is_tool_user(const char *p, const char *end) {
    agent_history_trim(&p, &end);
    return agent_history_tool_result_payload(&p, &end) ||
           agent_history_has_prefix(p, end, "Tool:") ||
           agent_history_has_prefix(p, end, "Tool result");
}

static void agent_history_ptrs_free(agent_history_ptrs *p) {
    free(p->v);
    free(p->mark);
    memset(p, 0, sizeof(*p));
}

/* Find the oldest rendered-chat marker needed to show the last N user turns.
 * Tool-result pseudo-user turns are skipped while human turns exist, so
 * /history stays centered on the human conversation.  Compacted sessions can
 * legitimately have a tail made only of tool result turns; in that case we
 * fall back to recent tool/assistant events instead of showing an empty
 * history. */
static const char *agent_history_start_for_turns(const char *text, size_t len,
                                                 int user_turns,
                                                 bool *tool_only) {
    const char *end = text + len;
    agent_history_ptrs marks = {0};
    agent_history_ptrs users = {0};
    agent_history_ptrs all_users = {0};
    const char *p = text;
    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        agent_history_ptrs_push(&marks, m, mark);
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        if (mark == AGENT_HISTORY_MARK_USER) {
            agent_history_ptrs_push(&all_users, m, mark);
            if (!agent_history_is_tool_user(content, content_end))
                agent_history_ptrs_push(&users, m, mark);
        }
        p = content_end;
    }

    const char *start = end;
    if (tool_only) *tool_only = false;
    if (users.len > 0) {
        int idx = users.len - user_turns;
        if (idx < 0) idx = 0;
        start = users.v[idx];
    } else if (all_users.len > 0) {
        int idx = all_users.len - user_turns;
        if (idx < 0) idx = 0;
        start = all_users.v[idx];
        if (tool_only) *tool_only = true;

        /* Tool result messages are stored as user-role turns after the
         * assistant Qwen tool stanza that produced them.  Include that preceding
         * assistant marker when it is still in the retained tail, otherwise
         * replay shows the result but hides the call that caused it. */
        for (int i = marks.len - 1; i >= 0; i--) {
            if (marks.v[i] >= start) continue;
            if (marks.mark[i] == AGENT_HISTORY_MARK_USER) break;
            if (marks.mark[i] == AGENT_HISTORY_MARK_ASSISTANT) {
                start = marks.v[i];
                break;
            }
        }
    }
    agent_history_ptrs_free(&marks);
    agent_history_ptrs_free(&users);
    agent_history_ptrs_free(&all_users);
    return start;
}

static bool agent_history_latest_compaction_summary(const char *text,
                                                    size_t len,
                                                    const char **sum_start,
                                                    const char **sum_end) {
    static const char start_mark[] =
        "[p36-agent compacted earlier conversation. Durable task-state summary follows.]";
    static const char end_mark[] =
        "[End compacted summary. Recent conversation continues verbatim below.]";
    const char *end = text + len;
    const char *scan = text;
    const char *best_start = NULL;
    const char *best_end = NULL;
    while (scan < end) {
        const char *s = agent_memmem(scan, (size_t)(end - scan),
                                     start_mark, sizeof(start_mark) - 1);
        if (!s) break;
        const char *content = s + sizeof(start_mark) - 1;
        const char *e = agent_memmem(content, (size_t)(end - content),
                                     end_mark, sizeof(end_mark) - 1);
        if (!e) break;
        best_start = content;
        best_end = e;
        scan = e + sizeof(end_mark) - 1;
    }
    if (!best_start || !best_end) return false;
    agent_history_trim(&best_start, &best_end);
    if (best_start >= best_end) return false;
    if (sum_start) *sum_start = best_start;
    if (sum_end) *sum_end = best_end;
    return true;
}

static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes);

static void agent_history_render_compaction_summary(agent_worker *w,
                                                    const char *text,
                                                    size_t len) {
    const char *p = NULL, *end = NULL;
    if (!agent_history_latest_compaction_summary(text, len, &p, &end)) return;
    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) {
        const char *s = "\n\x1b[1;95mCompacted Summary:\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "\nCompacted Summary:\n",
                      strlen("\nCompacted Summary:\n"));
    }
    agent_history_publish_limited(w, p, end, 80, 12000);
}

static const char *agent_history_skip_utf8_continuation(const char *p,
                                                        const char *end) {
    while (p < end && (((unsigned char)*p) & 0xc0) == 0x80) p++;
    return p;
}

static const char *agent_history_tail_start(const char *p, const char *end,
                                            int max_lines, size_t max_bytes,
                                            bool *truncated) {
    *truncated = false;
    if (p >= end) return p;

    const char *start = p;
    size_t len = (size_t)(end - p);
    if (max_bytes && len > max_bytes) {
        start = end - max_bytes;
        *truncated = true;
    }

    if (max_lines > 0) {
        const char *scan = end;
        if (scan > p && scan[-1] == '\n') scan--;
        const char *line_start = p;
        int lines = 0;
        while (scan > p) {
            scan--;
            if (*scan == '\n' && ++lines == max_lines) {
                line_start = scan + 1;
                break;
            }
        }
        if (line_start > p) *truncated = true;
        if (line_start > start) start = line_start;
    }

    return agent_history_skip_utf8_continuation(start, end);
}

static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes) {
    bool truncated = false;
    const char *start = agent_history_tail_start(p, end, max_lines, max_bytes,
                                                 &truncated);
    if (truncated)
        agent_publish(w, "\n... earlier history truncated; showing tail ...\n",
                      strlen("\n... earlier history truncated; showing tail ...\n"));
    agent_publish(w, start, (size_t)(end - start));
    if (end > start && end[-1] != '\n') agent_publish(w, "\n", 1);
}

static void agent_history_render_assistant(agent_worker *w,
                                           const char *p, const char *end) {
    agent_history_trim(&p, &end);
    if (p >= end) return;
    bool source_truncated = false;
    (void)agent_history_tail_start(p, end,
                                   AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                   AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                   &source_truncated);
    bool use_color = isatty(STDOUT_FILENO) != 0;
    agent_tail_capture tail = {
        .cap = source_truncated ? AGENT_HISTORY_ASSISTANT_MAX_BYTES : 0,
    };
    agent_token_renderer renderer = {
        .engine = w->engine,
        .worker = w,
        .format_thinking = true,
        /* History replay should look like the original live output: the user is
         * switching back to a session, not reading a different transcript
         * format.  Tool calls are still dry-rendered below, so replay never
         * executes tools or mutates transcript state. */
        .format_markdown = true,
        .use_color = use_color && !source_truncated,
        .last_output_newline = true,
        .capture = source_truncated ? &tail : NULL,
    };
    agent_qwen_tool_parser qwen_tool = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &qwen_tool,
        .replay = true,
    };

    /* Dry-run replay: the same streaming projection hides Qwen tool and renders
     * semantic tool lines, but no tool is executed and no transcript state is
     * changed.  The saved KV payload remains the only authority for resume. */
    agent_stream_text(&stream, p, (size_t)(end - p), true);
    renderer_finish(&renderer);
    agent_qwen_tool_parser_free(&qwen_tool);

    if (source_truncated) {
        size_t tail_len = 0;
        char *tail_text = agent_tail_capture_take(&tail, &tail_len);
        bool rendered_truncated = tail.total > tail_len;
        bool line_truncated = false;
        const char *tail_start =
            agent_history_tail_start(tail_text, tail_text + tail_len,
                                     AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                     AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                     &line_truncated);
        if (use_color) agent_publish(w, "\x1b[90m", 5);
        agent_publish(w,
                      "\n... earlier assistant history truncated; showing tail ...\n",
                      strlen("\n... earlier assistant history truncated; showing tail ...\n"));
        (void)rendered_truncated;
        agent_publish(w, tail_start, (size_t)(tail_text + tail_len - tail_start));
        if (tail_len && tail_text[tail_len - 1] != '\n') agent_publish(w, "\n", 1);
        if (use_color) agent_publish(w, "\x1b[0m", 4);
        free(tail_text);
    }
}

/* Re-render saved transcript text for /history and /switch.  It intentionally
 * uses the same assistant/token renderer as live output, so restored history
 * looks like the original terminal stream instead of raw rendered-chat text. */
static void agent_history_render_text(agent_worker *w, const char *text,
                                      size_t len, int user_turns) {
    if (user_turns <= 0) return;
    if (user_turns > AGENT_HISTORY_MAX_TURNS)
        user_turns = AGENT_HISTORY_MAX_TURNS;

    const char *end = text + len;
    agent_history_render_compaction_summary(w, text, len);

    bool tool_only = false;
    const char *p = agent_history_start_for_turns(text, len, user_turns,
                                                  &tool_only);
    if (p >= end) {
        agent_publish(w, "\n(no user history)\n", strlen("\n(no user history)\n"));
        return;
    }

    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) agent_publish(w, "\n\x1b[90m", strlen("\n\x1b[90m"));
    else agent_publish(w, "\n", 1);
    if (tool_only) {
        agent_publishf(w, "--- session history: recent tool/assistant events ---\n");
    } else {
        agent_publishf(w, "--- session history: last %d user turn%s ---\n",
                       user_turns, user_turns == 1 ? "" : "s");
    }
    if (color) agent_publish(w, "\x1b[0m", 4);

    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        const char *tp = content, *te = content_end;
        agent_history_trim(&tp, &te);

        if (mark == AGENT_HISTORY_MARK_USER) {
            if (agent_history_is_tool_user(tp, te)) {
                const char *payload_start = tp;
                const char *payload_end = te;
                (void)agent_history_tool_result_payload(&payload_start,
                                                        &payload_end);
                if (color) {
                    const char *s = "\x1b[90mTool result:\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "Tool result:\n", strlen("Tool result:\n"));
                }
                agent_history_publish_limited(w, payload_start, payload_end,
                                              12, 3000);
                if (color) agent_publish(w, "\x1b[0m", 4);
            } else {
                if (color) {
                    const char *s = "\x1b[1;32mUser:\x1b[0m\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "User:\n", strlen("User:\n"));
                }
                agent_history_publish_limited(w, tp, te, 24, 6000);
            }
        } else if (mark == AGENT_HISTORY_MARK_ASSISTANT) {
            if (color) {
                const char *s = "\x1b[1;37mAssistant:\x1b[0m\n";
                agent_publish(w, s, strlen(s));
            } else {
                agent_publish(w, "Assistant:\n", strlen("Assistant:\n"));
            }
            agent_history_render_assistant(w, tp, te);
        }
        p = content_end;
    }

    if (color) {
        const char *s = "\x1b[90m--- end history ---\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "--- end history ---\n", strlen("--- end history ---\n"));
    }
}

/* Render recent saved transcript text without mutating the live session. */
static bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    size_t text_len = 0;
    char *text = p36_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    agent_history_render_text(w, text, text_len, user_turns);
    free(text);
    return true;
}

typedef struct {
    p36_kvstore_entry entry;
    char *title;
} agent_session_list_item;

static int agent_session_list_cmp_recent(const void *a, const void *b) {
    const agent_session_list_item *sa = a, *sb = b;
    uint64_t ta = sa->entry.last_used ? sa->entry.last_used : sa->entry.created_at;
    uint64_t tb = sb->entry.last_used ? sb->entry.last_used : sb->entry.created_at;
    if (ta < tb) return 1;
    if (ta > tb) return -1;
    return strcmp(sa->entry.sha, sb->entry.sha);
}

static void agent_session_list_free(agent_session_list_item *v, int n) {
    for (int i = 0; i < n; i++) {
        p36_kvstore_entry_free(&v[i].entry);
        free(v[i].title);
    }
    free(v);
}

static void agent_session_list_push(agent_session_list_item **v, int *len,
                                    int *cap, p36_kvstore_entry entry,
                                    char *title) {
    if (*len == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *v = xrealloc(*v, (size_t)*cap * sizeof((*v)[0]));
    }
    (*v)[(*len)++] = (agent_session_list_item){
        .entry = entry,
        .title = title,
    };
}

/* Print resumable sessions from ~/.p36/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session. */
static void agent_worker_list_sessions(agent_worker *w) {
    DIR *d = opendir(w->cache_dir);
    if (!d) {
        printf("no sessions: %s\n", strerror(errno));
        return;
    }

    int cols = renderer_terminal_cols();
    size_t title_budget = cols > 16 ? (size_t)(cols - 12) : 20;
    if (title_budget > 160) title_budget = 160;

    agent_session_list_item *sessions = NULL;
    int sessions_len = 0, sessions_cap = 0;
    const uint8_t model_id = (uint8_t)p36_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!p36_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = p36_kvstore_path_join(w->cache_dir, de->d_name);
        p36_kvstore_entry e = {0};
        if (p36_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) {
                char *title = agent_session_title_from_file(path, title_budget);
                agent_session_list_push(&sessions, &sessions_len, &sessions_cap,
                                        e, title);
            } else {
                p36_kvstore_entry_free(&e);
            }
        }
        free(path);
    }
    closedir(d);
    if (!sessions_len) {
        printf("no saved sessions\n");
        return;
    }

    qsort(sessions, (size_t)sessions_len, sizeof(sessions[0]),
          agent_session_list_cmp_recent);

    bool color = isatty(STDOUT_FILENO) != 0;
    const char *sha_on = color ? "\x1b[1;96m" : "";
    const char *title_on = color ? "\x1b[1;97m" : "";
    const char *help_on = color ? "\x1b[97m" : "";
    const char *dim = color ? "\x1b[90m" : "";
    const char *reset = color ? "\x1b[0m" : "";

    for (int i = 0; i < sessions_len; i++) {
        p36_kvstore_entry *e = &sessions[i].entry;
        char age[32];
        agent_format_age(e->last_used ? e->last_used : e->created_at,
                         age, sizeof(age));
        printf("%s%.8s%s %s>%s %s%s%s\n",
               sha_on, e->sha, reset, dim, reset,
               title_on, sessions[i].title, reset);
        printf("         %s> %s, %u tokens, %.2f MB%s%s\n\n",
               dim, age, e->tokens,
               (double)e->file_size / (1024.0 * 1024.0),
               e->payload_bytes == 0 ? ", stripped" : "",
               reset);
    }
    printf("%sUse /switch <id> to select a session, /del <id> to remove, "
           "/strip <id> to strip KV cache.%s\n",
           help_on, reset);
    agent_session_list_free(sessions, sessions_len);
}

typedef struct {
    char sha[41];
    uint64_t last_used;
} agent_completion_session;

typedef struct {
    agent_completion_session *v;
    int len;
    int cap;
} agent_completion_sessions;

static void agent_completion_sessions_push(agent_completion_sessions *s,
                                           const char sha[41],
                                           uint64_t last_used) {
    if (s->len == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->v = xrealloc(s->v, (size_t)s->cap * sizeof(s->v[0]));
    }
    memcpy(s->v[s->len].sha, sha, 41);
    s->v[s->len].last_used = last_used;
    s->len++;
}

static int agent_completion_session_cmp(const void *a, const void *b) {
    const agent_completion_session *sa = a, *sb = b;
    if (sa->last_used < sb->last_used) return 1;
    if (sa->last_used > sb->last_used) return -1;
    return strcmp(sa->sha, sb->sha);
}

/* Tab completion for /switch.  Suggestions are sorted by recent use and accept
 * either an empty prefix or any unambiguous hex prefix. */
static void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc) {
    agent_worker *w = agent_completion_worker;
    static const char cmd[] = "/switch";
    const size_t cmd_len = sizeof(cmd) - 1;
    if (!w || !buf || strncmp(buf, cmd, cmd_len) != 0) return;

    const char *p = buf + cmd_len;
    if (*p && *p != ' ' && *p != '\t') return;
    while (*p == ' ' || *p == '\t') p++;

    const char *prefix = p;
    size_t prefix_len = strlen(prefix);
    for (size_t i = 0; i < prefix_len; i++) {
        if (!isxdigit((unsigned char)prefix[i])) return;
    }
    if (prefix_len > 40) return;

    DIR *d = opendir(w->cache_dir);
    if (!d) return;

    agent_completion_sessions sessions = {0};
    const uint8_t model_id = (uint8_t)p36_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!p36_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (prefix_len && strncasecmp(sha, prefix, prefix_len) != 0) continue;

        uint64_t last_used = 0;
        char *path = p36_kvstore_path_join(w->cache_dir, de->d_name);
        p36_kvstore_entry e = {0};
        if (p36_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) last_used = e.last_used;
            else last_used = UINT64_MAX;
            p36_kvstore_entry_free(&e);
        } else {
            last_used = UINT64_MAX;
        }
        free(path);
        if (last_used == UINT64_MAX) continue;
        agent_completion_sessions_push(&sessions, sha, last_used);
    }
    closedir(d);

    qsort(sessions.v, (size_t)sessions.len, sizeof(sessions.v[0]),
          agent_completion_session_cmp);
    for (int i = 0; i < sessions.len; i++) {
        char line[64];
        int sha_chars = prefix_len > 8 ? 40 : 8;
        snprintf(line, sizeof(line), "/switch %.*s",
                 sha_chars, sessions.v[i].sha);
        linenoiseAddCompletion(lc, line);
    }
    free(sessions.v);
}

/* Resolve a user-provided SHA prefix to exactly one saved session file. */
static bool agent_worker_find_session(agent_worker *w, const char *prefix,
                                      char sha_out[41], char **path_out,
                                      char *err, size_t err_len) {
    size_t plen = strlen(prefix);
    if (plen == 0 || plen > 40) {
        snprintf(err, err_len, "invalid session SHA prefix");
        return false;
    }
    for (size_t i = 0; i < plen; i++) {
        if (!isxdigit((unsigned char)prefix[i])) {
            snprintf(err, err_len, "invalid session SHA prefix");
            return false;
        }
    }

    DIR *d = opendir(w->cache_dir);
    if (!d) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    int matches = 0;
    char match_sha[41] = {0};
    char *match_path = NULL;
    const uint8_t model_id = (uint8_t)p36_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!p36_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (strncasecmp(sha, prefix, plen) != 0) continue;
        char *path = p36_kvstore_path_join(w->cache_dir, de->d_name);
        p36_kvstore_entry e = {0};
        bool same_model = p36_kvstore_read_entry_file(path, sha, &e) &&
                          e.model_id == model_id;
        p36_kvstore_entry_free(&e);
        if (!same_model) {
            free(path);
            continue;
        }
        matches++;
        if (matches == 1) {
            memcpy(match_sha, sha, sizeof(match_sha));
            match_path = path;
        } else {
            free(path);
        }
    }
    closedir(d);
    if (matches == 0) {
        snprintf(err, err_len, "no saved session matches %.40s", prefix);
        return false;
    }
    if (matches > 1) {
        snprintf(err, err_len, "session prefix %.40s is ambiguous", prefix);
        free(match_path);
        return false;
    }
    memcpy(sha_out, match_sha, 41);
    *path_out = match_path;
    return true;
}

static bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len) {
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;
    if (unlink(path) != 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }
    if (sha_out) memcpy(sha_out, sha, 41);
    free(path);
    return true;
}

/* Strip the heavy backend payload from a saved session while preserving its
 * rendered transcript. Loading such a file later tokenizes the text and
 * rebuilds the live KV with a full prefill. */
static bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }

    p36_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *title = NULL;
    bool ok = p36_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    if (ok && (hdr.ext_flags & P36_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    fclose(fp);
    if (!ok) {
        if (!err[0]) snprintf(err, err_len, "failed to read session");
        free(title);
        free(text);
        free(path);
        return false;
    }

    char actual_sha[41];
    agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
    if (strcmp(actual_sha, sha)) {
        snprintf(err, err_len, "cached session identity does not match file name");
        free(title);
        free(text);
        free(path);
        return false;
    }

    p36_tokens stripped_tokens = {0};
    p36_tokenize_rendered_chat(w->engine, text, &stripped_tokens);
    uint32_t stripped_token_count = (uint32_t)stripped_tokens.len;
    p36_tokens_free(&stripped_tokens);

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    uint8_t h[P36_KVSTORE_FIXED_HEADER];
    uint64_t now = (uint64_t)time(NULL);
    p36_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason, hdr.ext_flags,
                            stripped_token_count, hdr.hits, hdr.ctx_size,
                            hdr.created_at, now, 0);
    uint8_t tb[4];
    p36_kvstore_le_put32(tb, text_bytes);

    errno = 0;
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
         fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
         fwrite(text, 1, text_bytes, fp) == text_bytes &&
         (!(hdr.ext_flags & P36_KVSTORE_EXT_SESSION_TITLE) ||
          agent_kv_write_title_trailer(fp, title, err, err_len)) &&
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
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) : "failed to write stripped session");
        unlink(tmp);
    } else {
        if (sha_out) memcpy(sha_out, sha, 41);
        if (tokens_out) *tokens_out = stripped_token_count;
    }

    free(tmp);
    free(title);
    free(text);
    free(path);
    return ok;
}

/* Load a saved session KV into the live transcript and optionally replay recent
 * history for the human. */
static bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    bool stripped = false;
    p36_kvstore_entry entry = {0};
    if (p36_kvstore_read_entry_file(path, sha, &entry)) {
        stripped = entry.payload_bytes == 0;
        p36_kvstore_entry_free(&entry);
    }
    if (stripped) {
        printf("rebuilding stripped session %.8s from rendered text...\n", sha);
        fflush(stdout);
    }

    p36_tokens loaded = {0};
    agent_kv_session_meta meta = {0};
    bool ok = agent_kv_load_path(w, path, sha, NULL, 0, &loaded, &meta,
                                 err, err_len);
    if (ok) {
        agent_worker_images_clear(w);
        p36_tokens_free(&w->transcript);
        w->transcript = loaded;
        free(w->session_title);
        w->session_title = meta.title ? xstrdup(meta.title) : xstrdup("(no user prompt)");
        w->session_created_at = meta.created_at ? meta.created_at : (uint64_t)time(NULL);
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = meta.legacy_identity ? xstrdup(path) : NULL;
        agent_worker_note_system_prompt_seen(w);
        w->datetime_context_injected = true;
        w->thinking_enabled = agent_initial_thinking_enabled(w->cfg);
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.ctx_used = w->transcript.len;
        w->status.ctx_size = w->cfg->gen.ctx_size;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        w->status.error[0] = '\0';
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("switched to session %.8s (%d tokens%s)\n",
               sha, w->transcript.len, stripped ? ", rebuilt from text" : "");
        if (history_turns > 0)
            (void)agent_worker_show_history(w, history_turns, err, err_len);
    } else {
        p36_tokens_free(&loaded);
    }
    agent_kv_session_meta_free(&meta);
    free(path);
    return ok;
}

/* ============================================================================
 * Tool Argument Parsing And File Tool Helpers
 * ============================================================================
 */

static int agent_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}

