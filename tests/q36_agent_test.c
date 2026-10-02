#define Q36_AGENT_TEST
#define Q36_AGENT_TEST_NO_MAIN
#include <stdarg.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "../q36_agent.c"
#include <sys/resource.h>

static void test_tool_arg(agent_tool_call *call, const char *name, const char *value) {
    agent_tool_call_add_arg(call, name, value, strlen(value));
}

static int test_write_file(const char *path, const char *data, size_t len,
                           char *err, size_t errlen) {
    return agent_replace_file(path, data, len, NULL, 0, err, errlen);
}

static void test_atomic_file_tools(void) {
    char dir[] = "/tmp/q36-agent-files-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[PATH_MAX], linkpath[PATH_MAX], err[256];
    snprintf(path, sizeof(path), "%s/file", dir);
    snprintf(linkpath, sizeof(linkpath), "%s/link", dir);
    char original[4096];
    memset(original, 'x', sizeof(original));
    AGENT_TEST_ASSERT(test_write_file(path, original, sizeof(original), err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(chmod(path, 0751) == 0);
#ifdef __APPLE__
    AGENT_TEST_ASSERT(setxattr(path, "com.q36.agent-test", "keep", 4, 0, 0) == 0);
#elif defined(__linux__)
    AGENT_TEST_ASSERT(setxattr(path, "user.q36-agent-test", "keep", 4, 0) == 0);
#endif

    pid_t child = fork();
    AGENT_TEST_ASSERT(child >= 0);
    if (child == 0) {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit = {64, 64};
        if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
        int rc = agent_replace_file(path, original, sizeof(original),
                                     original, sizeof(original), err, sizeof(err));
        _exit(rc == -1 ? 0 : 3);
    }
    int status = 0;
    if (child > 0) waitpid(child, &status, 0);
    AGENT_TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char *data = NULL;
    size_t len = 0;
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == sizeof(original) && !memcmp(data, original, len));
    free(data);
    AGENT_TEST_ASSERT(agent_replace_file(path, "bad", 3, "stale", 5, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "changed") != NULL);

    AGENT_TEST_ASSERT(symlink("file", linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(linkpath, "new", 3, err, sizeof(err)) == 0);
    struct stat st;
    AGENT_TEST_ASSERT(lstat(linkpath, &st) == 0 && S_ISLNK(st.st_mode));
    AGENT_TEST_ASSERT(stat(path, &st) == 0 && (st.st_mode & 0777) == 0751);
    AGENT_TEST_ASSERT(st.st_uid == getuid());
#ifdef __APPLE__
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "com.q36.agent-test", attribute, sizeof(attribute), 0, 0) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#elif defined(__linux__)
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "user.q36-agent-test", attribute, sizeof(attribute)) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#endif
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == 3 && !memcmp(data, "new", 3));
    free(data);
    unlink(linkpath);
    AGENT_TEST_ASSERT(link(path, linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(path, "bad", 3, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "hard-linked") != NULL);
    unlink(linkpath);
    unlink(path);
    /* Failed replacements must not leave temporary files behind. */
    AGENT_TEST_ASSERT(rmdir(dir) == 0);

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = true;
    AGENT_TEST_ASSERT(agent_edit_find_old_span("literal [upto] text", 19,
                         "[upto]", false, &match, &match_len, &anchored, err, sizeof(err)));
    AGENT_TEST_ASSERT(!anchored && match_len == 6 && !memcmp(match, "[upto]", 6));
}

static void test_streaming_file_tools(void) {
    char path[] = "/tmp/q36-agent-read-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    /* Larger than the old whole-file cap, with matches at both ends. */
    fputs("needle first\r\n", fp);
    for (int i = 0; i < 1024 * 1024; i++) fputs("0123456789abcdef\n", fp);
    fputs("needle last\r", fp);
    fclose(fp);
    agent_worker w = {0};
    char *text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "needle first") && w.more_valid);
    AGENT_TEST_ASSERT(w.more_next_line == 2 && w.more_byte_offset == 14);
    free(text);
    agent_tool_call more = {0};
    test_tool_arg(&more, "count", "1");
    text = agent_tool_more(&w, &more);
    AGENT_TEST_ASSERT(strstr(text, "2 0123456789abcdef"));
    free(text);
    agent_tool_call_free(&more);

    agent_tool_call call = {0};
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    char linkpath[PATH_MAX];
    snprintf(linkpath, sizeof(linkpath), "%s-link", path);
    AGENT_TEST_ASSERT(symlink(path, linkpath) == 0);
    agent_tool_call linked = {0};
    test_tool_arg(&linked, "path", linkpath);
    test_tool_arg(&linked, "query", "needle");
    text = agent_tool_search(&w, &linked);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    agent_tool_call_free(&linked);
    unlink(linkpath);
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    test_tool_arg(&call, "mode", "regexp");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc('x', fp);
    fclose(fp);
    /* A 256 KiB single line is a blob: bounded output, no continuation. */
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_READ_MAX_BYTES && !w.more_valid);
    AGENT_TEST_ASSERT(strstr(text, "MINIFIED_OR_VENDOR_BLOB OMITTED: 262144 bytes"));
    free(text);
    text = agent_read_range(&w, path, 1, INT_MAX, true, true, true);
    AGENT_TEST_ASSERT(strstr(text, "READ_OUTPUT_TRUNCATED") && !strstr(text, "Tool error"));
    free(text);
    fp = fopen(path, "wb");
    for (int i = 0; i < 4096; i++) fputs("0123456789abcdef\n", fp);
    fclose(fp);
    text = agent_read_range(&w, path, 1, INT_MAX, true, true, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error: whole read") && !w.more_valid);
    free(text);
    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc(0x80, fp);
    fclose(fp);
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_READ_MAX_BYTES);
    free(text);
    unlink(path);
    AGENT_TEST_ASSERT(mkfifo(path, 0600) == 0);
    double started = now_sec();
    text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:") && now_sec() - started < 0.5);
    free(text);
    unlink(path);
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    agent_buf b = {0};
    char *large = xmalloc(200000);
    memset(large, 'q', 200000);
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strlen(text) == 200000);
    free(text);
    b.limit = 100;
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strstr(text, "Output truncated") != NULL);
    free(large);
    free(text);
    b.limit = 3;
    agent_buf_puts(&b, "a\xe4\xb8\xad" "b");
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(!strncmp(text, "a\n[Output truncated", 19));
    free(text);
}

/* index.html with a 600 KB minified line in the middle (the real-world bug). */
static void test_read_blob_cap(void) {
    char path[] = "/tmp/q36-agent-blob-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    for (int i = 1; i <= 215; i++) fprintf(fp, "<p>normal line %d</p>\n", i);
    fputs("<script>", fp);
    for (int i = 0; i < 600 * 1024 / 8; i++) fputs("var a=1;", fp);
    fputs("</script>\n<p>after blob</p>\n", fp);
    fclose(fp);
    agent_worker w = {0};
    for (int bare = 0; bare < 2; bare++) {
        char *text = agent_read_range(&w, path, 210, 11, false, bare, true);
        AGENT_TEST_ASSERT(strlen(text) < AGENT_READ_MAX_BYTES);
        AGENT_TEST_ASSERT(strstr(text, "normal line 215") && strstr(text, "after blob"));
        AGENT_TEST_ASSERT(strstr(text, "[MINIFIED_OR_VENDOR_BLOB OMITTED: 614417 bytes]"));
        AGENT_TEST_ASSERT(strstr(text, "READ_OUTPUT_TRUNCATED\npath="));
        AGENT_TEST_ASSERT(strstr(text, "offending_line=216\nline_bytes=614417\nreturned_bytes="));
        AGENT_TEST_ASSERT(strstr(text, "requested_lines=210-217"));
        if (!bare) {
            /* Compaction state remembers it and forbids a re-read. */
            agent_task_state st = {0};
            agent_tool_call rd = {0};
            rd.name = xstrdup("read");
            test_tool_arg(&rd, "path", path);
            agent_st_note_goal(&st, "fix spin");
            agent_st_note_tool(&st, &rd, text);
            char *r = agent_st_render(&st);
            AGENT_TEST_ASSERT(strstr(r, "do not read the minified/vendor blob") && strstr(r, "line 216"));
            free(r);
            agent_tool_call_free(&rd);
        }
        free(text);
    }
    /* search clips the matching blob line too */
    agent_tool_call call = {0};
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "var a=1;");
    char *text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_READ_MAX_BYTES && strstr(text, "exceeds"));
    free(text);
    fp = fopen(path, "wb");
    fputs("var a=1;", fp);
    for (int i = 0; i < 10000; i++) fputc('y', fp);
    fputc('\n', fp);
    fclose(fp);
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_READ_MAX_BYTES && strstr(text, "BLOB OMITTED: 10008"));
    free(text);
    agent_tool_call_free(&call);
    unlink(path);
}

static void test_background_jobs(void) {
    agent_worker w = {0};
    pthread_mutex_init(&w.mu, NULL);
    w.wake_fd[0] = w.wake_fd[1] = -1;
    char err[256], marker[] = "/tmp/q36-agent-deadline-XXXXXX";
    int fd = mkstemp(marker);
    close(fd);
    unlink(marker);
    char cmd[PATH_MAX + 128];
    snprintf(cmd, sizeof(cmd), "sleep 2; printf late > %s", marker);
    agent_bash_job *job = agent_bash_start(&w, cmd, 1, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    /* Deliberately no status polling: this stands in for model generation. */
    usleep(2400000);
    AGENT_TEST_ASSERT(access(marker, F_OK) != 0);
    bool finished = false;
    char *obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "timed_out=1"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "(sleep 0.1; printf descendant-output) &", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(350000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "descendant-output"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "head -c 2097152 /dev/zero | tr '\\000' x", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(900000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0"));
    AGENT_TEST_ASSERT(job->bytes == 2097152);
    free(obs);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(strlen(obs) < AGENT_BASH_TAIL_BYTES + 2048);
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "sleep 0.3; printf completed", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    char output_path[PATH_MAX];
    snprintf(output_path, sizeof(output_path), "%s", job->path);
    agent_tool_call call = {.name = xstrdup("bash_status")};
    char id[32];
    snprintf(id, sizeof(id), "%d", job->id);
    test_tool_arg(&call, "job", id);
    test_tool_arg(&call, "refresh_sec", "1");
    double start = now_sec();
    obs = agent_execute_tool_call(&w, &call);
    AGENT_TEST_ASSERT(now_sec() - start >= 0.2);
    AGENT_TEST_ASSERT(strstr(obs, "status=done") && strstr(obs, "completed"));
    AGENT_TEST_ASSERT(w.bash_jobs == NULL);
    free(obs);
    agent_tool_call_free(&call);
    unlink(output_path);

    /* Two monitors must make progress together, and stopping one must neither
     * block shutdown nor kill the other job's process group. */
    job = agent_bash_start(&w, "sleep 30", 60, err, sizeof(err));
    agent_bash_job *other = agent_bash_start(&w, "sleep 0.2; printf independent", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job && other);
    if (!job || !other) goto done;
    start = now_sec();
    agent_bash_signal(job, SIGKILL);
    unlink(job->path);
    agent_bash_remove_job(&w, job);
    AGENT_TEST_ASSERT(now_sec() - start < 2);
    while (agent_bash_is_running(other) && now_sec() - start < 3) usleep(10000);
    obs = agent_bash_observation(other, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0") && strstr(obs, "independent"));
    free(obs);
    unlink(other->path);
    agent_bash_remove_job(&w, other);
done:
    agent_bash_jobs_free(&w);
    unlink(marker);
    free(w.out);
    pthread_mutex_destroy(&w.mu);
}

static void test_shell_terminal_controls(void) {
    const char malicious[] = "before\x1b[2Jafter\x1b[H!\x1b]52;c;secret\a"
                             "\x1bPdata\x1b\\\x1b[31mred\x1b[0m\b\n";
    char *safe = agent_terminal_safe_text(malicious, sizeof(malicious) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "beforeafter!\x1b[31mred\x1b[0m\\x08\n"));
    free(safe);
    const char c1[] = "\xe4\xb8\xad\xc2\x9b" "2J\x9b" "2J";
    safe = agent_terminal_safe_text(c1, sizeof(c1) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "\xe4\xb8\xad\\xc2\\x9b2J\\x9b2J"));
    free(safe);
    for (size_t i = 0; i < sizeof(malicious); i++) {
        safe = agent_terminal_safe_text(malicious, i);
        AGENT_TEST_ASSERT(!strstr(safe, "\x1b[2J") && !strstr(safe, "\x1b]52"));
        free(safe);
    }
}

static const char *test_output_dir;

static void test_fixture(const char *name, const char *data, size_t len) {
    if (!test_output_dir) return;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", test_output_dir, name);
    FILE *fp = fopen(path, "wb");
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    AGENT_TEST_ASSERT(fwrite(data, 1, len, fp) == len);
    AGENT_TEST_ASSERT(fclose(fp) == 0);
}

static void test_completion(const char *text, linenoiseCompletions *completions) {
    (void)text;
    linenoiseAddCompletion(completions, "example");
}

static void test_fragmented_terminal_input(void) {
    int input[2];
    AGENT_TEST_ASSERT(pipe(input) == 0);
    fcntl(input[0], F_SETFL, O_NONBLOCK);
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) { close(input[0]); close(input[1]); return; }
    setenv("LINENOISE_ASSUME_TTY", "1", 1);
    struct linenoiseState l = {0};
    char buffer[1024] = "";
    l.ifd = input[0]; l.ofd = fileno(sink);
    l.buf = buffer; l.buflen = sizeof(buffer) - 1;
    l.cols = 80; l.prompt = "";
    const char *samples[] = {"\xc3\xa9", "\xe4\xb8\xad", "\xf0\x9f\x98\x80"};
    for (size_t s = 0; s < sizeof(samples)/sizeof(samples[0]); s++) {
        const char *sample = samples[s];
        for (size_t split = 1; split < strlen(sample); split++) {
            linenoiseEditClear(&l);
            for (size_t i = 0; i < split; i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            for (size_t i = split; i < strlen(sample); i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(!strcmp(l.buf, sample));
        }
    }
    linenoiseEditClear(&l);
    const char sequence[] = "ab\x1b[DZ\x1b[3~\x1b[H!";
    for (size_t i = 0; i < sizeof(sequence) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sequence[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "!aZ"));
    linenoiseEditClear(&l);
    const char paste[] = "\x1b[200~one\r\n\xe4\xb8\xad\nthree\x1b[201~";
    for (size_t i = 0; i < sizeof(paste) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, paste[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
        if (i < sizeof(paste) - 2) AGENT_TEST_ASSERT(l.len == 0);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "one\n\xe4\xb8\xad\nthree"));
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\xe4');
    linenoiseEditFeedByte(&l, 'X');
    AGENT_TEST_ASSERT(!strcmp(l.buf, "\xef\xbf\xbdX"));
    linenoiseEditClear(&l);
    const char invalid_paste[] = "\x1b[200~bad\xe4\x1b[201~";
    for (size_t i = 0; i < sizeof(invalid_paste) - 1; i++)
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, invalid_paste[i]) == linenoiseEditMore);
    AGENT_TEST_ASSERT(l.len == 0 && !l.paste_active);
    linenoiseSetCompletionCallback(test_completion);
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    AGENT_TEST_ASSERT(l.in_completion);
    linenoiseEditFeedByte(&l, '\x1b');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e"));
    linenoiseEditFeedByte(&l, '[');
    linenoiseEditFeedByte(&l, 'D');
    AGENT_TEST_ASSERT(l.pos == 0);
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\t');
    const char unicode[] = "\xe4\xb8\xad";
    for (size_t i = 0; i < sizeof(unicode) - 1; i++) linenoiseEditFeedByte(&l, unicode[i]);
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "example\xe4\xb8\xad"));
    linenoiseEditClear(&l);
    l.buflen = 3;
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    linenoiseEditFeedByte(&l, ' ');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e") && l.len == 1);
    l.buflen = sizeof(buffer) - 1;
    linenoiseSetCompletionCallback(NULL);
    free(l.queued_input);
    free(l.paste_buf);
    close(input[0]); close(input[1]); fclose(sink);
    unsetenv("LINENOISE_ASSUME_TTY");
}

static void test_markdown_literals(void) {
    const char *input[] = {"Use *.c files.", "The literal is \\*.", "An unmatched `tick",
                          "**bold** and *italic* and `code`.", "``a ` b``", "*unclosed",
                          "* list item\n", "trailing \\", "**unclosed", "`a``", "\\`literal\\`",
                          "> **Hint:** Check `errno`.\n"};
    const char *expected[] = {"Use *.c files.", "The literal is *.", "An unmatched `tick",
                             "bold and italic and code.", "a ` b", "*unclosed",
                             "* list item\n", "trailing \\", "**unclosed", "`a``", "`literal`",
                             "> Hint: Check errno.\n"};
    for (size_t i = 0; i < sizeof(input)/sizeof(input[0]); i++) {
        agent_tail_capture capture = {.cap = 16384};
        agent_token_renderer r = {.capture = &capture, .format_markdown = true};
        for (size_t j = 0; j < strlen(input[i]); j++) renderer_markdown_feed(&r, input[i][j]);
        renderer_markdown_finish(&r);
        renderer_flush_utf8(&r);
        size_t len;
        char *out = agent_tail_capture_take(&capture, &len);
        AGENT_TEST_ASSERT(!strcmp(out, expected[i]));
        if (strcmp(out, expected[i])) fprintf(stderr, "markdown: %s => %s\n", input[i], out);
        free(out);
    }
    agent_tail_capture capture = {.cap = 20000};
    agent_token_renderer r = {.capture = &capture, .format_markdown = true};
    renderer_markdown_feed(&r, '*');
    for (int i = 0; i < 8192; i++) renderer_markdown_feed(&r, 'x');
    renderer_markdown_finish(&r);
    size_t len;
    char *out = agent_tail_capture_take(&capture, &len);
    AGENT_TEST_ASSERT(len == 8193 && out[0] == '*');
    free(out);
}

static void test_unicode_output_and_footer(void) {
    agent_editor ed = {0};
    ed.edit.cols = 80;
    char ascii[78];
    memset(ascii, 'a', sizeof(ascii));
    editor_note_output(&ed, ascii, sizeof(ascii));
    editor_note_output(&ed, "\xe4", 1);
    AGENT_TEST_ASSERT(ed.output_col == 78 && ed.output_utf8_len == 1);
    editor_note_output(&ed, "\xb8\xad", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "\xcc\x81", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "x", 1);
    AGENT_TEST_ASSERT(ed.output_col == 1 && !ed.output_pending_wrap);
    editor_note_output(&ed, "\r", 1);
    AGENT_TEST_ASSERT(ed.output_col == 0 && !ed.output_pending_wrap);
    const char family[] = "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb";
    for (size_t i = 0; i < sizeof(family) - 1; i++) editor_note_output(&ed, family + i, 1);
    AGENT_TEST_ASSERT(ed.output_col == 2);
    int width;
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(family, sizeof(family) - 1, &width) == sizeof(family) - 1 && width == 2);
    const char flag[] = "\xf0\x9f\x87\xae\xf0\x9f\x87\xb9";
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(flag, sizeof(flag) - 1, &width) == sizeof(flag) - 1 && width == 2);
    editor_note_output(&ed, flag, sizeof(flag) - 1);
    AGENT_TEST_ASSERT(ed.output_col == 4);
    editor_note_output(&ed, "\x1b[", 2);
    AGENT_TEST_ASSERT(ed.output_escape == 2);
    editor_note_output(&ed, "31mX", 4);
    AGENT_TEST_ASSERT(ed.output_col == 5 && !ed.output_escape);

    agent_prompt_queue q = {0};
    char queued[181];
    for (int i = 0; i < 60; i++) memcpy(queued + i * 3, "\xe4\xb8\xad", 3);
    queued[180] = 0;
    agent_prompt_queue_push(&q, queued);
    agent_status st = {0};
    char footer[4096];
    build_footer_text(&st, &q, 40, footer, sizeof(footer));
    test_fixture("queue-footer.txt", footer, strlen(footer));
    for (size_t pos = 0; pos < strlen(footer);) {
        uint32_t cp;
        size_t n = linenoiseUtf8Decode(footer + pos, strlen(footer) - pos, &cp);
        AGENT_TEST_ASSERT(n && cp != 0xfffd);
        if (!n) break;
        pos += n;
    }
    agent_prompt_queue_free(&q);
}

static void test_footer_only_updates(void) {
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) return;
    int saved = dup(STDOUT_FILENO);
    dup2(fileno(sink), STDOUT_FILENO);
    agent_editor ed = {.active = true, .scroll_region = true, .term_rows = 24,
                       .term_cols = 80, .output_bottom = 22, .prompt_row = 23};
    snprintf(ed.prompt, sizeof(ed.prompt), "q36-agent> ");
    snprintf(ed.status, sizeof(ed.status), "generation 0");
    char buffer[] = "draft";
    ed.edit = (struct linenoiseState){.ifd = -1, .ofd = STDOUT_FILENO,
        .buf = buffer, .buflen = sizeof(buffer), .len = 5, .pos = 5, .oldpos = 5,
        .prompt = ed.prompt, .plen = strlen(ed.prompt), .cols = 80,
        .oldrows = 1, .oldstatusrows = 1, .oldrpos = 1,
        .screen_cursor_row = 23, .screen_cursor_col = 17};
    linenoiseEditSetStatus(&ed.edit, ed.status, "", "");
    const char initial[] = "\x1b[23;1Hq36-agent> draft\r\ngeneration 0\x1b[23;17H";
    write_all(STDOUT_FILENO, initial, sizeof(initial) - 1);
    editor_set_prompt_status(&ed, ed.prompt, "generation 1");
    off_t first = lseek(STDOUT_FILENO, 0, SEEK_CUR);
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(lseek(STDOUT_FILENO, 0, SEEK_CUR) == first && ed.status_dirty);
    ed.last_prompt_redraw_time -= 1;
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(!ed.status_dirty);
    editor_set_prompt_status(&ed, ed.prompt, "done");
    editor_flush_prompt_status(&ed, true);
    AGENT_TEST_ASSERT(!ed.status_dirty && !strcmp(buffer, "draft"));
    dup2(saved, STDOUT_FILENO);
    close(saved);
    fseek(sink, 0, SEEK_END);
    size_t len = (size_t)ftell(sink);
    rewind(sink);
    char *text = xmalloc(len + 1);
    AGENT_TEST_ASSERT(fread(text, 1, len, sink) == len);
    text[len] = 0;
    AGENT_TEST_ASSERT(strstr(text, "\x1b[?2026h") && !strstr(text, "\x1b[0K"));
    test_fixture("status.ansi", text, len);
    free(text);
    free(ed.edit.status); free(ed.edit.status_start); free(ed.edit.status_end);
    fclose(sink);
}

static int test_terminal_driver(void) {
    agent_editor ed = {0};
    linenoiseSetMultiLine(1);
    if (editor_start(&ed, "q36-agent> ", "ready", NULL)) return 2;
    double start = now_sec(), next = start;
    char *answer = NULL;
    unsigned tick = 0;
    while (now_sec() - start < 10 && !answer) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        poll(&pfd, 1, 10);
        if (pfd.revents & POLLIN) editor_read_stdin(&ed);
        while (linenoiseEditQueuedInput(&ed.edit)) {
            char *line = linenoiseEditFeed(&ed.edit);
            if (line == linenoiseEditMore) continue;
            if (line) answer = line;
            else answer = xstrdup("<input error>");
            break;
        }
        if (now_sec() >= next) {
            char status[80];
            snprintf(status, sizeof(status), "generation %u", ++tick);
            if (tick % 4 == 0) {
                const char output[] = "model output \xe4\xb8\xad\n";
                editor_write_async(&ed, output, sizeof(output) - 1, "q36-agent> ", status, false);
            } else editor_set_prompt_status(&ed, "q36-agent> ", status);
            next = now_sec() + 0.05;
        }
    }
    editor_stop(&ed);
    editor_restore_terminal_layout(&ed);
    if (!answer) return 3;
    printf("\nRESULT:");
    for (size_t i = 0; i < strlen(answer); i++) printf("%02x", (unsigned char)answer[i]);
    puts("");
    free(answer);
    return 0;
}

static char *test_hint_capture(const char *text, size_t split, bool markdown,
                               bool color, int *calls) {
    agent_tail_capture capture = {.cap = 32768};
    agent_token_renderer renderer = {.capture = &capture, .format_thinking = true,
        .format_markdown = markdown, .use_color = color, .last_output_newline = true};
    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &parser};
    size_t n = strlen(text);
    if (split <= n) {
        agent_stream_text(&stream, text, split, false);
        /* A prompt redraw resets terminal colors between generated fragments. */
        if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        agent_stream_text(&stream, text + split, n - split, false);
    } else {
        for (size_t i = 0; i < n; i++) {
            agent_stream_text(&stream, text + i, 1, false);
            if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        }
    }
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    AGENT_TEST_ASSERT(!renderer.md_hint && !renderer.md_hint_prefix_len && !renderer.color_open);
    if (calls) {
        *calls = (int)parser.calls.len;
        AGENT_TEST_ASSERT(parser.calls.len == 1);
        if (parser.calls.len == 1) {
            AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].name, "bash"));
            AGENT_TEST_ASSERT(parser.calls.v[0].argc == 1);
            if (parser.calls.v[0].argc == 1)
                AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].args[0].value, "printf HINT_OK"));
        }
    } else AGENT_TEST_ASSERT(parser.calls.len == 0);
    agent_qwen_tool_parser_free(&parser);
    return agent_tail_capture_take(&capture, NULL);
}

static void test_hint_rendering(void) {
    const char *badge = "\x1b[1;97;48;5;23m Hint \x1b[0m";
    const char *sample =
        "The error path is fixed.\n\n"
        "> **Hint:** Save `errno` before **cleanup**; calls can overwrite it.\n"
        "> Keep the original error for reporting.\n"
        "> Unicode stays intact: caf\xc3\xa9, \xe4\xb8\xad.\n"
        "Normal prose resumes here.\n\n"
        "```text\n> **Hint:** This is literal code.\n```\n"
        "Another normal line.\n";
    for (size_t split = 0; split <= strlen(sample) + 1; split++) {
        char *out = test_hint_capture(sample, split, true, true, NULL);
        const char *label = strstr(out, badge);
        AGENT_TEST_ASSERT(label && !strstr(label + strlen(badge), badge));
        AGENT_TEST_ASSERT(strstr(out, "\x1b[1mcleanup") || split > strlen(sample));
        if (split == strlen(sample)) test_fixture("hints.ansi", out, strlen(out));
        if (split > strlen(sample)) test_fixture("hints-fragmented.ansi", out, strlen(out));
        free(out);
    }
    const char *literal[] = {"> quoted text", "> **Hinting:** not a hint",
        "Inline > **Hint:** not an aside", "`> **Hint:** literal`",
        "\\> **Hint:** escaped", "<think>> **Hint:** hidden reasoning</think>Normal."};
    for (size_t i = 0; i < sizeof(literal) / sizeof(literal[0]); i++) {
        char *out = test_hint_capture(literal[i], strlen(literal[i]), true, true,
                                      NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge));
        free(out);
    }
    const char marker[] = "> **Hint:**";
    for (size_t i = 0; i < sizeof(marker) - 1; i++) {
        char partial[sizeof(marker)];
        memcpy(partial, marker, i);
        partial[i] = 0;
        char *out = test_hint_capture(partial, i, true, true, NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge) && !strncmp(out, partial, i));
        free(out);
    }
    char *out = test_hint_capture(sample, strlen(sample), false, false, NULL);
    AGENT_TEST_ASSERT(!strncmp(out, sample, strlen(sample)) && !strchr(out, '\x1b'));
    free(out);
    out = test_hint_capture("> **Hint:** Check `errno`.\n", 0, true, false, NULL);
    AGENT_TEST_ASSERT(!strcmp(out, "> Hint: Check errno.\n\n"));
    free(out);
    const char *tool = "> **Hint:** Keep shell checks reproducible.\n"
        "<tool_call><function=bash><parameter=command>printf HINT_OK"
        "</parameter></function></tool_call>";
    for (size_t split = 0; split <= strlen(tool); split++) {
        int calls = 0;
        out = test_hint_capture(tool, split, true, true, &calls);
        AGENT_TEST_ASSERT(calls == 1 && strstr(out, badge));
        free(out);
    }
}

/* Behave like sudo's terminal password reader without using real credentials
 * or requiring privilege. stdout/stderr still belong to the bash tool. */
static int password_child(void) {
    int fd = open("/dev/tty", O_RDWR);
    if (fd < 0) { puts("NO_TERMINAL"); return 1; }
    struct termios saved, mode;
    if (tcgetattr(fd, &saved) != 0) return 2;
    mode = saved;
    mode.c_lflag &= ~(ECHO | ECHONL);
    for (int attempt = 0; attempt < 3; attempt++) {
        if (tcsetattr(fd, TCSAFLUSH, &mode) != 0) return 3;
        write_all(fd, "[sudo] password for test: ", 26);
        char buf[1024] = {0};
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        tcsetattr(fd, TCSANOW, &saved);
        if (n > 0 && !strcmp(buf, "q36-test-secret\n")) {
            close(fd);
            puts("PASSWORD_OK");
            return 0;
        }
        write_all(fd, "\nSorry, try again.\n", 19);
    }
    close(fd);
    return 1;
}

static void test_worker_init(agent_worker *w, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->cfg = cfg;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    if (pipe(w->wake_fd) != 0) abort();
    set_nonblock(w->wake_fd[0], true, NULL);
    set_nonblock(w->wake_fd[1], true, NULL);
}

static void test_worker_free(agent_worker *w) {
    agent_bash_jobs_free(w);
    free(w->out);
    close(w->wake_fd[0]);
    close(w->wake_fd[1]);
    pthread_mutex_destroy(&w->mu);
    pthread_cond_destroy(&w->cond);
}

static void test_worker_ownership(void) {
    agent_config cfg = {0};
    agent_worker w;
    test_worker_init(&w, &cfg);
    w.initialized = true;
    AGENT_TEST_ASSERT(worker_is_idle(&w));
    w.active = true;
    AGENT_TEST_ASSERT(!worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    w.active = false;
    w.save_requested = true;
    AGENT_TEST_ASSERT(!worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    worker_pause(&w);
    AGENT_TEST_ASSERT(worker_is_idle(&w));
    AGENT_TEST_ASSERT(!worker_take_save_requested(&w) && w.save_requested);
    AGENT_TEST_ASSERT(!worker_submit(&w, "next turn"));
    worker_resume(&w);
    AGENT_TEST_ASSERT(worker_take_save_requested(&w));
    w.interrupt = true;
    AGENT_TEST_ASSERT(worker_submit(&w, "next turn"));
    AGENT_TEST_ASSERT(!w.interrupt && !worker_is_idle(&w));
    free(w.cmd_text);
    test_worker_free(&w);
}

static void *test_worker_ui_wait(void *arg) {
    agent_worker *w = arg;
    if (w->cfg->gen.seed == 0) {
        free(worker_request_queued_user_drain(w));
    } else if (w->cfg->gen.seed == 1) {
        agent_password_request request = {0};
        agent_request_password(w, &request);
    } else {
        char err[160];
        agent_web_confirm(w, "Allow?", err, sizeof(err));
    }
    worker_clear_interrupt(w);
    agent_set_status(w, AGENT_WORKER_IDLE);
    while (worker_wait_for_work(w)) {
        pthread_mutex_lock(&w->mu);
        free(w->cmd_text);
        w->cmd_text = NULL;
        w->status.state = AGENT_WORKER_IDLE;
        pthread_mutex_unlock(&w->mu);
    }
    return NULL;
}

static void test_worker_pause_ui_waits(void) {
    for (int request = 0; request < 3; request++) {
        agent_config cfg = {0};
        cfg.gen.seed = request;
        agent_worker w;
        test_worker_init(&w, &cfg);
        w.initialized = true;
        w.active = true;
        w.status.state = AGENT_WORKER_GENERATING;
        pthread_t thread;
        AGENT_TEST_ASSERT(pthread_create(&thread, NULL, test_worker_ui_wait, &w) == 0);
        struct pollfd pfd = {.fd = w.wake_fd[0], .events = POLLIN};
        AGENT_TEST_ASSERT(poll(&pfd, 1, 2000) == 1);
        worker_pause(&w);
        AGENT_TEST_ASSERT(worker_is_idle(&w) && !w.interrupt);
        worker_resume(&w);
        AGENT_TEST_ASSERT(worker_submit(&w, "continue after cancelled exit"));
        worker_pause(&w);
        AGENT_TEST_ASSERT(worker_is_idle(&w));
        worker_stop(&w);
        pthread_join(thread, NULL);
        test_worker_free(&w);
    }
}

static void *password_job_thread(void *arg) {
    agent_worker *w = arg;
    agent_bash_refresh_for(w, w->bash_jobs, 10);
    agent_set_status(w, AGENT_WORKER_IDLE);
    return NULL;
}

static int password_driver(const char *cmd, int timeout, bool noninteractive) {
    agent_config cfg = {.non_interactive = noninteractive};
    agent_worker w;
    test_worker_init(&w, &cfg);
    agent_editor editor = {0};
    const char *prompt = "q36-agent> ";
    if (editor_start(&editor, prompt, "test", "unfinished draft") != 0) return 1;
    char err[160];
    agent_bash_job *job = agent_bash_start(&w, cmd, timeout, err, sizeof(err));
    if (!job) { fprintf(stderr, "%s\n", err); return 2; }
    w.status.state = AGENT_WORKER_GENERATING;
    pthread_t thread;
    if (pthread_create(&thread, NULL, password_job_thread, &w) != 0) return 3;
    for (;;) {
        struct pollfd pfd = {.fd = w.wake_fd[0], .events = POLLIN};
        poll(&pfd, 1, 100);
        drain_wake_fd(w.wake_fd[0]);
        agent_password_request request;
        if (worker_take_password_request(&w, &request)) {
            if (editor_prompt_password(&editor, &w, &request, prompt, "test") != 0)
                abort();
        }
        pthread_mutex_lock(&w.mu);
        bool done = w.status.state == AGENT_WORKER_IDLE;
        w.wake_pending = false;
        pthread_mutex_unlock(&w.mu);
        if (done) break;
    }
    pthread_join(thread, NULL);
    bool draft_ok = !strcmp(editor.edit.buf, "unfinished draft");
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    char *obs = agent_bash_observation(job, false, NULL);
    printf("\nDRAFT_OK=%d\n%s", draft_ok, obs);
    free(obs);
    unlink(job->path);
    test_worker_free(&w);
    return draft_ok ? 0 : 4;
}

static void test_bash_noninteractive(void) {
    agent_config cfg = {.non_interactive = true};
    agent_worker w;
    test_worker_init(&w, &cfg);
    char err[160];
    agent_bash_job *job = agent_bash_start(&w,
        "printf stdout; printf stderr >&2; read value; test $? -ne 0", 5,
        err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (job) {
        agent_bash_refresh_for(&w, job, 5);
        AGENT_TEST_ASSERT(!job->running && job->exit_status == 0);
        char *obs = agent_bash_observation(job, false, NULL);
        AGENT_TEST_ASSERT(strstr(obs, "stdoutstderr") != NULL);
        free(obs);
        unlink(job->path);
    }
    test_worker_free(&w);
}

static void test_compaction_boundaries(void) {
    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_stream_renderer stream = {.parser = &parser};
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 0;
    stream.qwen_tool_start_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.qwen_tool_active = true;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.qwen_tool_active = false;
    parser.state = AGENT_QWEN_TOOL_PARAM_VALUE;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    AGENT_TEST_ASSERT(agent_stream_has_partial_tool(&stream));
    stream.qwen_tool_start_len = 0;
    parser.state = AGENT_QWEN_TOOL_DONE;
    AGENT_TEST_ASSERT(!agent_stream_has_partial_tool(&stream));
    parser.state = AGENT_QWEN_TOOL_SEARCH;
    int data[1000] = {0}, role[] = {42, 43, 44};
    q36_tokens tokens = {.v = data, .len = 1000};
    q36_tokens user = {.v = role, .len = 3};
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 900);
    memcpy(data + 850, role, sizeof(role));
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 850);
    memcpy(data + 950, role, sizeof(role));
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 950);
    data[951] = 0;
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, &user) == 850);
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 150, 100, 100, &user) == 100);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(4096) == 512);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(100000) == 4096);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(1024) == 256);
}

static void test_partial_tool_interrupt_rollback(void) {
    char path[] = "/tmp/q36-agent-interrupt-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    AGENT_TEST_ASSERT(write(fd, "original\n", 9) == 9);
    close(fd);

    char partial[PATH_MAX + 256];
    snprintf(partial, sizeof(partial),
        "<tool_call>\n<function=write>\n"
        "<parameter=path>\n%s\n</parameter>\n"
        "<parameter=content>\nreplacement that never closes", path);
    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_qwen_tool_feed(&parser, partial, strlen(partial));
    AGENT_TEST_ASSERT(parser.state == AGENT_QWEN_TOOL_PARAM_VALUE);
    AGENT_TEST_ASSERT(parser.calls.len == 0);

    agent_worker worker = {0};
    for (int i = 0; i < 8; i++) q36_tokens_push(&worker.transcript, 100 + i);
    agent_rollback_assistant_suffix(&worker, 3);
    AGENT_TEST_ASSERT(worker.transcript.len == 3);
    AGENT_TEST_ASSERT(worker.session_dirty);
    AGENT_TEST_ASSERT(strstr(agent_partial_tool_interrupt_warning,
                             "was NOT executed") != NULL);
    AGENT_TEST_ASSERT(strstr(agent_partial_tool_interrupt_warning,
                             "workspace is unchanged") != NULL);

    char *contents = NULL;
    size_t contents_len = 0;
    char err[160] = {0};
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &contents, &contents_len,
                                            err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(contents_len == 9 && !memcmp(contents, "original\n", 9));
    free(contents);
    agent_qwen_tool_parser_free(&parser);
    q36_tokens_free(&worker.transcript);
    unlink(path);
}

static void test_action_leak_and_watchdog(void) {
    /* A real final answer past the budget is not a leak. */
    static agent_action_leak l;
    memset(&l, 0, sizeof(l));
    bool hit = false;
    for (int i = 0; i < 200 && !hit; i++)
        hit = agent_action_leak_feed(&l, 8, "The file was updated and tests pass. ", 37) && i < 3;
    AGENT_TEST_ASSERT(!hit);
    /* Reasoning restarts past the budget are. */
    memset(&l, 0, sizeof(l));
    const char *chunks[] = {"Wait, let me reconsider the file.\n", "Actually, I should read it again.\n",
                            "Let me read the file once more.\n", "Hmm, maybe not.\n"};
    hit = false;
    for (int i = 0; i < 400 && !hit; i++)
        hit = agent_action_leak_feed(&l, 8, chunks[i % 4], strlen(chunks[i % 4]));
    AGENT_TEST_ASSERT(hit);
    /* Under budget never trips. */
    memset(&l, 0, sizeof(l));
    for (int i = 0; i < 8; i++)
        AGENT_TEST_ASSERT(!agent_action_leak_feed(&l, 8, chunks[i % 4], strlen(chunks[i % 4])));

    /* Watchdog: second identical read blocked; mutation resets. */
    static agent_worker w;
    memset(&w, 0, sizeof(w));
    agent_tool_arg a = {"path", "x.c"};
    agent_tool_call rd = {"read", &a, 1, 1}, wr = {"write", &a, 1, 1};
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &rd, 1));
    AGENT_TEST_ASSERT(agent_watchdog_blocks(&w, &rd, 1));
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &wr, 1));
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &rd, 1));
    /* Back-to-back identical mutating calls are loops too. */
    memset(&w, 0, sizeof(w));
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &wr, 1));
    AGENT_TEST_ASSERT(agent_watchdog_blocks(&w, &wr, 1));
    AGENT_TEST_ASSERT(agent_watchdog_blocks(&w, &wr, 1));
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &rd, 1));   /* different call resets the run */
    AGENT_TEST_ASSERT(!agent_watchdog_blocks(&w, &wr, 1));
}

/* Deterministic replay of the recovery protocol: scripted "model" rounds go
 * through the same leak detector and recovery counter the worker loop uses. */
static void test_leak_recovery_protocol(void) {
    const char *leaky[] = {"Wait, let me reconsider.\n", "Actually, let me read it again.\n",
                           "Hmm, let me check once more.\n"};
    const char *good = "Done: index.html written.\n";
    static agent_action_leak l;
    agent_recovery rec = {.max = 2};
    /* Rounds 0..2 leak (round 2 exceeds max -> BLOCKED); a healthy round after
     * a recovery passes and a retry round never starts with thinking. */
    int blocked_at = -1;
    for (int round = 0; round < 3; round++) {
        memset(&l, 0, sizeof(l));
        bool leaked = false;
        for (int i = 0; i < 300 && !leaked; i++)
            leaked = agent_action_leak_feed(&l, 16, leaky[i % 3], strlen(leaky[i % 3]));
        AGENT_TEST_ASSERT(leaked);
        AGENT_TEST_ASSERT(l.tokens > 16 && l.tokens < 120); /* aborted promptly */
        if (!agent_recovery_next(&rec)) { blocked_at = round; break; }
        AGENT_TEST_ASSERT(rec.nothink_next);
    }
    AGENT_TEST_ASSERT(blocked_at == 2);
    agent_recovery ok = {.max = 2};
    AGENT_TEST_ASSERT(agent_recovery_next(&ok));
    memset(&l, 0, sizeof(l));
    for (int i = 0; i < 200; i++) {
        char line[96];
        snprintf(line, sizeof(line), "%s step %d wrote section %d of the report. ", good, i, i * 7);
        AGENT_TEST_ASSERT(!agent_action_leak_feed(&l, 16, line, strlen(line)));
    }
    /* Past 4x budget a lone restart marker is enough. */
    memset(&l, 0, sizeof(l));
    bool hit = false;
    for (int i = 0; i < 200 && !hit; i++) {
        char buf[96];
        if (i == 150) snprintf(buf, sizeof(buf), "Wait, one more thing.\n");
        else snprintf(buf, sizeof(buf), "Point %d: the answer is in file %d. ", i, i * 13);
        hit = agent_action_leak_feed(&l, 16, buf, strlen(buf));
    }
    AGENT_TEST_ASSERT(hit);
}

static char *test_call(const char *name, ...) {
    agent_tool_arg args[6];
    int n = 0;
    va_list ap;
    va_start(ap, name);
    const char *k;
    while (n < 6 && (k = va_arg(ap, const char *))) {
        args[n].name = (char *)k;
        args[n].value = (char *)va_arg(ap, const char *);
        n++;
    }
    va_end(ap);
    agent_tool_call c = {(char *)name, args, n, n};
    if (!strcmp(name, "write")) return agent_tool_write(NULL, &c);
    return agent_tool_edit(NULL, &c);
}

static void test_write_append_and_edit_lines(void) {
    char dir[] = "/tmp/q36agentXXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[256];
    snprintf(path, sizeof(path), "%s/f.txt", dir);
    char *r = test_call("write", "path", path, "content", "one\n", NULL);
    AGENT_TEST_ASSERT(strstr(r, "OK write") && strstr(r, "total=4"));
    free(r);
    r = test_call("write", "path", path, "content", "two\nthree\n", "append", "true", NULL);
    AGENT_TEST_ASSERT(strstr(r, "total=14") && strstr(r, "append=1"));
    free(r);
    char *big = xmalloc(AGENT_WRITE_PAYLOAD_MAX + 2);
    memset(big, 'x', AGENT_WRITE_PAYLOAD_MAX + 1);
    big[AGENT_WRITE_PAYLOAD_MAX + 1] = 0;
    r = test_call("write", "path", path, "content", big, NULL);
    AGENT_TEST_ASSERT(strstr(r, "WRITE_PAYLOAD_TOO_LARGE") && strstr(r, "max_bytes=12288"));
    free(r);
    free(big);
    /* whitespace-sensitive edit: model's old text has the wrong indentation */
    r = test_call("write", "path", path, "content", "int f(void) {\n        return 1;\n}\n", NULL);
    free(r);
    r = test_call("edit", "path", path, "old", "return  1 ;", "new", "x", NULL);
    AGENT_TEST_ASSERT(strstr(r, "EDIT_MATCH_NOT_FOUND") && strstr(r, "hint=whitespace_differs") &&
                      strstr(r, "actual_matches=0") && strstr(r, "path="));
    free(r);
    r = test_call("edit", "path", path, "start_line", "2", "end_line", "2",
                  "new", "    return 2;", NULL);
    AGENT_TEST_ASSERT(strstr(r, "Tool error") == NULL);
    free(r);
    char *data = NULL; size_t len = 0; char err[128];
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(data && !strcmp(data, "int f(void) {\n    return 2;\n}\n"));
    free(data);
    r = test_call("edit", "path", path, "start_line", "4", "end_line", "3", "new", "// end", NULL);
    free(r);   /* insert at EOF keeps one line per line */
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(data && !strcmp(data, "int f(void) {\n    return 2;\n}\n// end\n"));
    free(data);
    r = test_call("edit", "path", path, "start_line", "2", "end_line", "2", "new", "    return 2;", NULL);
    AGENT_TEST_ASSERT(strstr(r, "EDIT_NO_CHANGE") != NULL);
    free(r);
    r = test_call("edit", "path", path, "start_line", "9", "new", "z", NULL);
    AGENT_TEST_ASSERT(strstr(r, "EDIT_LINE_RANGE_INVALID") && strstr(r, "lines=4"));
    free(r);
    unlink(path);
    rmdir(dir);
}

static void test_task_state_render(void) {
    static agent_task_state st;
    memset(&st, 0, sizeof(st));
    agent_st_note_goal(&st, "Fix the cube spin.\nWait, maybe not");
    agent_tool_arg pa = {"path", "index.html"};
    agent_tool_call rd = {"read", &pa, 1, 1}, ed = {"edit", &pa, 1, 1};
    agent_st_note_tool(&st, &rd, "1 <html>");
    char *r = agent_st_render(&st);
    AGENT_TEST_ASSERT(strstr(r, "- already read index.html"));
    free(r);
    agent_st_note_tool(&st, &ed, "Edited index.html using line-range replacement\nTouched old lines 3-3");
    r = agent_st_render(&st);
    AGENT_TEST_ASSERT(!strstr(r, "already read"));          /* edit invalidates the read */
    AGENT_TEST_ASSERT(strstr(r, "CURRENT EDIT STATE:\n- index.html: Edited index.html"));
    AGENT_TEST_ASSERT(strstr(r, "- edit index.html"));
    free(r);
    agent_st_note_tool(&st, &ed, "Tool error: EDIT_MATCH_NOT_FOUND part=old text expected_matches=1");
    r = agent_st_render(&st);
    AGENT_TEST_ASSERT(strstr(r, "fix and retry: edit index.html: EDIT_MATCH_NOT_FOUND"));
    free(r);
    AGENT_TEST_ASSERT(!agent_st_empty(&st));
}

/* ---- Verification-convergence watchdog ----
 * These run the real tools through agent_execute_tool_observation, the one
 * place both agent loops execute tools, so the kind of each call is decided by
 * the same file/hash logic as in production.  The "target" lives in a dir under
 * the cwd (a /tmp path would be a helper by definition). */
typedef struct {
    agent_config cfg;
    agent_worker w;
    char wd[64];     /* workspace under cwd */
    char hd[64];     /* helper dir under /tmp */
    char target[160];
} vw_fixture;

static char *vw_run(agent_worker *w, const char *name, ...) {
    agent_tool_arg args[6];
    int n = 0;
    va_list ap;
    va_start(ap, name);
    const char *k;
    while (n < 6 && (k = va_arg(ap, const char *))) {
        args[n].name = (char *)k;
        args[n].value = (char *)va_arg(ap, const char *);
        n++;
    }
    va_end(ap);
    agent_tool_call c = {(char *)name, args, n, n};
    agent_tool_calls calls = {&c, 1, 1};
    agent_tool_observation obs = agent_execute_tool_observation(w, &calls);
    agent_buf b = {0};
    for (size_t i = 0; i < obs.part_count; i++)
        agent_buf_puts(&b, obs.parts[i].text ? obs.parts[i].text : "");
    agent_tool_observation_free(&obs);
    return agent_buf_take(&b);
}

static void vw_start(vw_fixture *f, int max_verify, int max_noprog) {
    memset(f, 0, sizeof(*f));
    f->cfg.non_interactive = true;
    f->cfg.gen.max_repeat_tool = 2;
    f->cfg.gen.max_verify_steps = max_verify;
    f->cfg.gen.max_no_progress_steps = max_noprog;
    test_worker_init(&f->w, &f->cfg);
    snprintf(f->wd, sizeof(f->wd), "q36-vwtest-XXXXXX");
    AGENT_TEST_ASSERT(mkdtemp(f->wd) != NULL);
    snprintf(f->hd, sizeof(f->hd), "/tmp/q36vwh-XXXXXX");
    AGENT_TEST_ASSERT(mkdtemp(f->hd) != NULL);
    snprintf(f->target, sizeof(f->target), "%s/main.html", f->wd);
    agent_vw_begin_turn(&f->w.st, &f->cfg.gen);
}

static void vw_finish(vw_fixture *f) {
    char cmd[200];
    test_worker_free(&f->w);
    snprintf(cmd, sizeof(cmd), "rm -rf '%s' '%s'", f->wd, f->hd);
    AGENT_TEST_ASSERT(system(cmd) == 0);
}

static void vw_helper_path(const vw_fixture *f, const char *name, char *out, size_t n) {
    snprintf(out, n, "%s/%s", f->hd, name);
}

/* The observed failure, replayed: the target is fixed once, then the agent keeps
 * rebuilding its own verification harness.  It must end in FINALIZE_REQUIRED
 * (not BLOCKED) long before the 157 steps of the real run. */
static void test_vw_replays_the_observed_pathology(void) {
    static vw_fixture f;
    vw_start(&f, 8, 16);
    agent_verify_state *v = &f.w.st.vw;
    char h1[200], h2[200], cmd1[400], cmd2[400];
    vw_helper_path(&f, "verify.sh", h1, sizeof(h1));
    vw_helper_path(&f, "verify_mock2.sh", h2, sizeof(h2));
    snprintf(cmd1, sizeof(cmd1), "sh %s", h1);
    snprintf(cmd2, sizeof(cmd2), "sh %s", h2);

    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>fixed</html>\n", NULL);
    AGENT_TEST_ASSERT(strstr(r, "OK write")); free(r);
    AGENT_TEST_ASSERT(v->armed && v->last_target_step == 1 && f.w.m.target_mutations == 1);

    r = vw_run(&f.w, "bash", "command", "echo overlayHidden:true # pytest", NULL); free(r);   /* strong evidence */
    r = vw_run(&f.w, "bash", "command", "echo pageerrors:none # pytest", NULL); free(r);
    AGENT_TEST_ASSERT(v->verify_used == 2 && !v->finalize);

    /* Harness edits are not progress: the target counters do not move. */
    r = vw_run(&f.w, "write", "path", h1, "content", "exit 0\n", NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 1 && f.w.m.helper_mutations == 1);
    AGENT_TEST_ASSERT(v->last_target_step == 1 && v->verify_used == 3);
    r = vw_run(&f.w, "read", "path", h1, NULL); free(r);          /* inspect the helper */
    r = vw_run(&f.w, "bash", "command", cmd1, NULL); free(r);     /* alternate verification */
    r = vw_run(&f.w, "write", "path", h2, "content", "exit 0\n", NULL); free(r);  /* new harness */
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 1 && f.w.m.helper_mutations == 2);
    AGENT_TEST_ASSERT(v->verify_used == 6 && !v->finalize);

    /* Same check through another harness file: a near-identical re-check costs double. */
    r = vw_run(&f.w, "bash", "command", cmd2, NULL);
    AGENT_TEST_ASSERT(v->finalize && f.w.m.verify_exhaustions == 1);
    AGENT_TEST_ASSERT(strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") &&
                      strstr(r, "No target files changed since step 1.") &&
                      strstr(r, "Do not call additional tools."));
    AGENT_TEST_ASSERT(strstr(r, "confirmed failing"));      /* asks for all three outcomes */
    free(r);
    AGENT_TEST_ASSERT(v->step == 8 && f.w.m.target_mutations == 1);

    /* Further tools are refused, not executed. */
    r = vw_run(&f.w, "read", "path", f.target, NULL);
    AGENT_TEST_ASSERT(strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") && !strstr(r, "fixed"));
    free(r);
    AGENT_TEST_ASSERT(v->rejected == 1 && !agent_vw_force_final(&f.w));
    r = vw_run(&f.w, "bash", "command", "echo again", NULL); free(r);
    AGENT_TEST_ASSERT(v->rejected == 2);

    /* The model ignored the notice twice: the controller answers, never BLOCKED. */
    AGENT_TEST_ASSERT(agent_vw_force_final(&f.w));
    AGENT_TEST_ASSERT(f.w.m.forced_finalizations == 1 && !strcmp(f.w.m.finish, "verify_budget"));
    AGENT_TEST_ASSERT(f.w.out && strstr(f.w.out, "verification budget is exhausted") &&
                      strstr(f.w.out, "not a claim") && !strstr(f.w.out, "BLOCKED"));
    AGENT_TEST_ASSERT(f.w.m.verify_steps + f.w.m.target_mutations + f.w.m.helper_mutations < 20);
    vw_finish(&f);
}

/* A: a real failure that is then repaired resets the budget. */
static void test_vw_target_repair_resets_budget(void) {
    static vw_fixture f;
    vw_start(&f, 8, 16);
    agent_verify_state *v = &f.w.st.vw;
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>v1</html>\n", NULL); free(r);
    r = vw_run(&f.w, "bash", "command", "echo check one # pytest", NULL); free(r);
    r = vw_run(&f.w, "bash", "command", "test -s /no/such/file/at/all", NULL);   /* fails */
    free(r);
    AGENT_TEST_ASSERT(v->verify_used == 2 && f.w.st.nunresolved == 1);
    r = vw_run(&f.w, "write", "path", f.target, "content", "<html>v2</html>\n", NULL); free(r);
    AGENT_TEST_ASSERT(v->verify_used == 0 && v->no_progress == 0 && v->dups == 0);
    AGENT_TEST_ASSERT(v->last_target_step == 4 && f.w.m.target_mutations == 2 && !v->finalize);
    /* A rewrite with identical content is not a mutation. */
    r = vw_run(&f.w, "write", "path", f.target, "content", "<html>v2</html>\n", NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 2 && v->last_target_step == 4 && v->verify_used == 0 &&
                      v->no_progress == 1);   /* back in IMPLEMENT: not a verification step */
    vw_finish(&f);
}

/* B: repeating the same check with nothing changed is forced to finalize quickly. */
static void test_vw_repeated_verification_finalizes(void) {
    static vw_fixture f;
    vw_start(&f, 8, 16);
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>ok</html>\n", NULL); free(r);
    bool hit = false;
    for (int i = 1; i <= 12 && !hit; i++) {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "echo run %d # make test", i);   /* digits are masked: same check */
        r = vw_run(&f.w, "bash", "command", cmd, NULL);
        hit = strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") != NULL;
        free(r);
    }
    AGENT_TEST_ASSERT(hit && f.w.st.vw.finalize && f.w.st.vw.step <= 6);
    AGENT_TEST_ASSERT(f.w.st.vw.dups >= 3);   /* VERIFY_STAGNANT territory */
    AGENT_TEST_ASSERT(f.w.m.verify_exhaustions == 1);
    vw_finish(&f);
}

/* C: the agent's own probe fails while the target is fine. */
static void test_vw_harness_failure_is_bounded_and_reported(void) {
    static vw_fixture f;
    vw_start(&f, 6, 16);
    char h[200], cmd[400];
    vw_helper_path(&f, "fps_probe.sh", h, sizeof(h));
    snprintf(cmd, sizeof(cmd), "sh %s", h);
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>ok</html>\n", NULL); free(r);
    r = vw_run(&f.w, "write", "path", h, "content", "echo fps=21; exit 1\n", NULL); free(r);
    bool hit = false;
    for (int i = 0; i < 8 && !hit; i++) {
        r = vw_run(&f.w, "bash", "command", cmd, NULL);   /* the probe's >40 FPS assumption fails */
        hit = strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") != NULL;
        free(r);
    }
    AGENT_TEST_ASSERT(hit && f.w.st.vw.finalize && f.w.m.target_mutations == 1);
    AGENT_TEST_ASSERT(f.w.st.nunresolved >= 1 && strstr(f.w.st.unresolved[0], "[TEST_HARNESS_FAILURE_OR_ASSUMPTION]"));
    char *fin = agent_vw_final_text(&f.w.st);
    AGENT_TEST_ASSERT(strstr(fin, "Unresolved:") && strstr(fin, "TEST_HARNESS_FAILURE_OR_ASSUMPTION") &&
                      strstr(fin, "not a claim that every check passed"));
    free(fin);
    vw_finish(&f);
}

/* D: compaction renders the counters and the lists; it cannot reset them. */
static void test_vw_state_survives_compaction_render(void) {
    static vw_fixture f;
    vw_start(&f, 8, 16);
    char h[200], cmd[400];
    vw_helper_path(&f, "verify.sh", h, sizeof(h));
    snprintf(cmd, sizeof(cmd), "sh %s", h);
    agent_st_note_goal(&f.w.st, "Fix the loading overlay.");
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>ok</html>\n", NULL); free(r);
    r = vw_run(&f.w, "write", "path", h, "content", "exit 1\n", NULL); free(r);
    r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);       /* repeated check */
    const agent_verify_state before = f.w.st.vw;
    char *s = agent_st_render(&f.w.st);
    AGENT_TEST_ASSERT(strstr(s, "LAST_TARGET_MUTATION: step 1") && strstr(s, "VERIFY_STEPS_USED:") &&
                      strstr(s, "MAX_VERIFY_STEPS: 8") && strstr(s, "NO_PROGRESS_STEPS:") &&
                      strstr(s, "UNRESOLVED:") && strstr(s, "DO NOT REPEAT:") &&
                      strstr(s, "do not repeat: bash") && strstr(s, "PHASE: VERIFY"));
    free(s);
    AGENT_TEST_ASSERT(!memcmp(&before, &f.w.st.vw, sizeof(before)));   /* rendering is read-only */
    /* Nothing but a new user turn resets the counters. */
    agent_st_note_goal(&f.w.st, "second goal");
    AGENT_TEST_ASSERT(f.w.st.vw.verify_used == before.verify_used && f.w.st.vw.step == before.step);
    agent_vw_begin_turn(&f.w.st, &f.cfg.gen);
    AGENT_TEST_ASSERT(f.w.st.vw.verify_used == 0 && !f.w.st.vw.armed && f.w.st.nunresolved == 0);
    vw_finish(&f);
}

/* Classification edges: bash mutations, product files named like helpers, limits off. */
static void test_vw_classification_edges(void) {
    static vw_fixture f;
    vw_start(&f, 8, 16);
    agent_verify_state *v = &f.w.st.vw;
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>a</html>\n", NULL); free(r);
    r = vw_run(&f.w, "read", "path", f.target, NULL); free(r);
    /* A shell edit of a file the agent already touched is a target mutation. */
    char cmd[400];
    snprintf(cmd, sizeof(cmd), "printf '<!-- x -->' >> %s", f.target);
    r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 2 && v->verify_used == 0);
    /* A shell write into a helper dir is a helper mutation. */
    snprintf(cmd, sizeof(cmd), "echo 1 > %s/probe.txt", f.hd);
    r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.helper_mutations == 1 && f.w.m.target_mutations == 2);
    /* A pre-existing product file with "verify" in its name is a target, not a helper. */
    char prod[200];
    snprintf(prod, sizeof(prod), "%s/verifyToken.js", f.wd);
    AGENT_TEST_ASSERT(test_write_file(prod, "old\n", 4, cmd, sizeof(cmd)) == 0);
    r = vw_run(&f.w, "write", "path", prod, "content", "new\n", NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 3);
    /* ...but one the agent creates under that name is a helper, and so are its later edits. */
    snprintf(prod, sizeof(prod), "%s/verify_run.mjs", f.wd);
    r = vw_run(&f.w, "write", "path", prod, "content", "1\n", NULL); free(r);
    r = vw_run(&f.w, "write", "path", prod, "content", "2\n", NULL); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 3 && f.w.m.helper_mutations == 3);
    vw_finish(&f);

    /* Before the first target edit the budget is not armed; 0 disables it entirely. */
    vw_start(&f, 8, 16);
    for (int i = 0; i < 30; i++) {
        snprintf(cmd, sizeof(cmd), "echo probe-a%c", 'a' + i % 20);
        r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    }
    AGENT_TEST_ASSERT(!f.w.st.vw.armed && !f.w.st.vw.finalize && f.w.m.verify_steps == 0);
    vw_finish(&f);
    vw_start(&f, 0, 0);
    r = vw_run(&f.w, "write", "path", f.target, "content", "x\n", NULL); free(r);
    for (int i = 0; i < 30; i++) {
        snprintf(cmd, sizeof(cmd), "echo run %c", 'a' + i % 20);
        r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    }
    AGENT_TEST_ASSERT(!f.w.st.vw.finalize && f.w.m.verify_exhaustions == 0);
    vw_finish(&f);
    /* The no-progress ceiling still bites when the verify budget is raised. */
    vw_start(&f, 100, 5);
    r = vw_run(&f.w, "write", "path", f.target, "content", "x\n", NULL); free(r);
    for (int i = 0; i < 5; i++) {
        snprintf(cmd, sizeof(cmd), "echo distinct%c # pytest", 'a' + i);
        r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    }
    AGENT_TEST_ASSERT(f.w.st.vw.finalize && f.w.st.vw.no_progress == 5);
    vw_finish(&f);
}

/* An agent whose workspace is itself under /tmp still edits real targets there;
 * only temp paths outside the workspace are helpers. */
static void test_vw_workspace_under_tmp(void) {
    char ws[] = "/tmp/q36vwws-XXXXXX", other[] = "/tmp/q36vwot-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(ws) != NULL && mkdtemp(other) != NULL);
    char old[PATH_MAX];
    AGENT_TEST_ASSERT(getcwd(old, sizeof(old)) != NULL && chdir(ws) == 0);
    agent_config cfg = {.non_interactive = true};
    cfg.gen.max_repeat_tool = 2;
    agent_worker w;
    test_worker_init(&w, &cfg);
    agent_vw_begin_turn(&w.st, &cfg.gen);
    char inside[PATH_MAX], outside[PATH_MAX];
    snprintf(inside, sizeof(inside), "%s/main.c", ws);
    snprintf(outside, sizeof(outside), "%s/check.sh", other);
    char *r = vw_run(&w, "write", "path", "rel.c", "content", "int x;\n", NULL); free(r);
    r = vw_run(&w, "write", "path", inside, "content", "int y;\n", NULL); free(r);
    AGENT_TEST_ASSERT(w.m.target_mutations == 2 && w.m.helper_mutations == 0 && w.st.vw.armed);
    r = vw_run(&w, "write", "path", outside, "content", "exit 0\n", NULL); free(r);
    AGENT_TEST_ASSERT(w.m.target_mutations == 2 && w.m.helper_mutations == 1);
    test_worker_free(&w);
    AGENT_TEST_ASSERT(chdir(old) == 0);
    char cmd[200];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s' '%s'", ws, other);
    AGENT_TEST_ASSERT(system(cmd) == 0);
}

/* The task state must not record a failed shell command as "ok", and image views
 * (browser screenshots) spend the same budget as other checks. */
static void test_vw_failed_bash_fact_and_image_budget(void) {
    static agent_task_state st;
    memset(&st, 0, sizeof(st));
    agent_tool_arg ca = {"command", "node probe.mjs"};
    agent_tool_call bash = {"bash", &ca, 1, 1};
    agent_st_note_tool(&st, &bash, "status=done\nexit_status=1\nfps=21\n");
    agent_st_note_tool(&st, &bash, "status=done\nexit_status=0\nfine\n");
    AGENT_TEST_ASSERT(st.nfacts == 2 && strstr(st.facts[0], "-> exit 1") && strstr(st.facts[1], "-> ok"));

    agent_verify_state v = {.armed = true, .max_verify = 4, .max_noprog = 16};
    agent_metrics m = {0};
    agent_tool_arg ia = {"path", "shot.png"};
    agent_tool_call img = {"view_image", &ia, 1, 1};
    uint64_t sig = agent_vw_signature(&v, &img, NULL);
    bool dup = false;
    AGENT_TEST_ASSERT(!agent_vw_step(&v, &m, AGENT_VW_VERIFY, sig, NULL, true, &dup) && !dup);
    AGENT_TEST_ASSERT(!agent_vw_step(&v, &m, AGENT_VW_VERIFY, sig, NULL, true, &dup) && dup);   /* repeat costs 2 */
    AGENT_TEST_ASSERT(agent_vw_step(&v, &m, AGENT_VW_VERIFY, sig, NULL, true, &dup) && v.finalize);
    AGENT_TEST_ASSERT(m.verify_exhaustions == 1 && m.verify_steps == 3);
    AGENT_TEST_ASSERT(!agent_vw_step(&v, &m, AGENT_VW_VERIFY, sig, NULL, true, &dup));        /* fires once */
}

static void test_vw_flags_and_summary(void) {
    char *a[] = {"q36-agent", "--max-verify-steps", "3", "--max-no-progress-steps", "9"};
    agent_config c = parse_options(5, a);
    AGENT_TEST_ASSERT(c.gen.max_verify_steps == 3 && c.gen.max_no_progress_steps == 9);
    char *z[] = {"q36-agent", "--max-verify-steps", "0", "--max-no-progress-steps", "0"};
    c = parse_options(5, z);                                       /* 0 turns the budget off */
    AGENT_TEST_ASSERT(c.gen.max_verify_steps == 0 && c.gen.max_no_progress_steps == 0);
    char *b[] = {"q36-agent"};
    c = parse_options(1, b);
    AGENT_TEST_ASSERT(c.gen.max_verify_steps == AGENT_MAX_VERIFY_DEFAULT &&
                      c.gen.max_no_progress_steps == AGENT_MAX_NOPROGRESS_DEFAULT);
    agent_worker w = {0};
    w.m.target_mutations = 2; w.m.helper_mutations = 3; w.m.verify_steps = 7;
    w.m.verify_exhaustions = 1; w.m.no_progress_steps = 7; w.m.forced_finalizations = 1;
    w.m.ph_iv = 2; w.m.ph_vi = 1; w.m.ph_vf = 1;
    w.st.vw.phase = AGENT_PH_VERIFY;
    char *s = agent_metrics_summary(&w);
    AGENT_TEST_ASSERT(strstr(s, "IMPLEMENT->VERIFY 2, VERIFY->IMPLEMENT 1, VERIFY->FINALIZE_REQUIRED 1") &&
                      strstr(s, "final controller phase: VERIFY"));
    AGENT_TEST_ASSERT(strstr(s, "target mutations: 2") && strstr(s, "verification helper mutations: 3") &&
                      strstr(s, "verification steps: 7") && strstr(s, "verification budget exhaustions: 1") &&
                      strstr(s, "no-progress steps: 7") && strstr(s, "forced finalizations: 1"));
    free(s);
}

/* The false positive seen in a real run: edits, compaction, then reads and searches that
 * rebuild state.  That is IMPLEMENT recovery, never VERIFY, and never finalizes. */
static void test_vw_compaction_recovery_stays_in_implement(void) {
    static vw_fixture f;
    vw_start(&f, 3, 16);                       /* tiny verify budget: any leak would show */
    agent_verify_state *v = &f.w.st.vw;
    agent_st_note_goal(&f.w.st, "Build the Three.js scene.");
    char *r = vw_run(&f.w, "write", "path", f.target,
                     "content", "<html>function buildFallenPetals(){}\n</html>\n", NULL); free(r);
    r = vw_run(&f.w, "edit", "path", f.target, "old", "buildFallenPetals(){}", "new", "buildFallenPetals(){ /*2*/ }", NULL);
    free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 2 && v->phase == AGENT_PH_IMPLEMENT);

    /* Deterministic compaction renders the state and must keep IMPLEMENT. */
    char *s = agent_st_render(&f.w.st);
    AGENT_TEST_ASSERT(strstr(s, "CURRENT_PHASE: IMPLEMENT") && strstr(s, "LAST_TARGET_MUTATION: step 2") &&
                      !strstr(s, "FINALIZE_REQUIRED"));
    free(s);

    /* Post-compaction recovery: two reads, a search, more reads. */
    r = vw_run(&f.w, "read", "path", f.target, NULL); free(r);
    r = vw_run(&f.w, "read", "path", f.target, "start_line", "1", "end_line", "1", NULL); free(r);
    r = vw_run(&f.w, "search", "query", "requestAnimationFrame", "path", f.wd, NULL); free(r);
    char cmd[200];
    snprintf(cmd, sizeof(cmd), "grep -c requestAnimationFrame %s; wc -l %s", f.target, f.target);
    r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    AGENT_TEST_ASSERT(v->phase == AGENT_PH_IMPLEMENT && f.w.m.verify_steps == 0 && v->verify_used == 0 &&
                      !v->finalize && f.w.m.verify_exhaustions == 0 && f.w.m.ph_iv == 0);

    /* The missing implementation is then added and succeeds. */
    r = vw_run(&f.w, "edit", "path", f.target, "old", "/*2*/", "new", "requestAnimationFrame(loop);", NULL);
    AGENT_TEST_ASSERT(!strstr(r, "Tool error")); free(r);
    AGENT_TEST_ASSERT(f.w.m.target_mutations == 3 && v->phase == AGENT_PH_IMPLEMENT);

    /* Now checking starts: VERIFY, and the budget applies normally. */
    r = vw_run(&f.w, "bash", "command", "echo ok # node --check", NULL); free(r);
    AGENT_TEST_ASSERT(v->phase == AGENT_PH_VERIFY && f.w.m.ph_iv == 1 && v->verify_used == 1);
    r = vw_run(&f.w, "bash", "command", "echo other # pytest", NULL); free(r);
    AGENT_TEST_ASSERT(!v->finalize && v->verify_used == 2);
    r = vw_run(&f.w, "bash", "command", "echo third # make test", NULL);
    AGENT_TEST_ASSERT(strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") && v->finalize && f.w.m.ph_vf == 1); free(r);
    vw_finish(&f);
}

/* A target edit during VERIFY goes back to IMPLEMENT; concrete MISSING evidence does too and
 * is never finalized from the verify budget. */
static void test_vw_incomplete_evidence_and_reentry(void) {
    static vw_fixture f;
    vw_start(&f, 2, 16);
    agent_verify_state *v = &f.w.st.vw;
    char *r = vw_run(&f.w, "write", "path", f.target, "content", "<html>a</html>\n", NULL); free(r);
    r = vw_run(&f.w, "bash", "command", "echo a # pytest", NULL); free(r);
    AGENT_TEST_ASSERT(v->phase == AGENT_PH_VERIFY);
    r = vw_run(&f.w, "write", "path", f.target, "content", "<html>b</html>\n", NULL); free(r);
    AGENT_TEST_ASSERT(v->phase == AGENT_PH_IMPLEMENT && f.w.m.ph_vi == 1 && v->verify_used == 0);
    r = vw_run(&f.w, "bash", "command", "echo b # pytest", NULL); free(r);
    /* This check spends the last unit but reports missing work: no finalization. */
    r = vw_run(&f.w, "bash", "command", "printf 'MISSING:\\n- render loop\\n' # make test", NULL);
    AGENT_TEST_ASSERT(!strstr(r, "VERIFICATION_BUDGET_EXHAUSTED") && !v->finalize && v->npending == 1 &&
                      v->phase == AGENT_PH_IMPLEMENT && f.w.m.verify_exhaustions == 0);
    free(r);
    char *s = agent_st_render(&f.w.st);
    AGENT_TEST_ASSERT(strstr(s, "IMPLEMENTATION_PENDING: render loop") && strstr(s, "implement what is missing"));
    free(s);
    for (int i = 0; i < 6; i++) {            /* checks cannot enter VERIFY while work is pending */
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "echo c%c # pytest", 'a' + i);
        r = vw_run(&f.w, "bash", "command", cmd, NULL); free(r);
    }
    AGENT_TEST_ASSERT(v->phase == AGENT_PH_IMPLEMENT && !v->finalize);
    r = vw_run(&f.w, "write", "path", f.target, "content", "<html>c</html>\n", NULL); free(r);
    AGENT_TEST_ASSERT(v->npending == 0 && v->phase == AGENT_PH_IMPLEMENT);
    vw_finish(&f);
}

/* In-process fake server: replies to one request with a canned body. */
typedef struct { int fd; const char *reply; size_t reply_len; } fake_srv;

static void *fake_srv_thread(void *arg) {
    fake_srv *f = arg;
    int c = accept(f->fd, NULL, NULL);
    char buf[16384];
    ssize_t n = 0, total = 0;
    while (total < (ssize_t)sizeof(buf) - 1 && (n = recv(c, buf + total, sizeof(buf) - 1 - (size_t)total, 0)) > 0) {
        total += n;
        buf[total] = 0;
        if (strstr(buf, "\r\n\r\n") && strstr(buf, "stream")) break;
    }
    size_t off = 0;
    while (off < f->reply_len) {   /* dribble to exercise reassembly across reads */
        size_t chunk = f->reply_len - off < 37 ? f->reply_len - off : 37;
        if (send(c, f->reply + off, chunk, MSG_NOSIGNAL) < 0) break;
        off += chunk;
        usleep(200);
    }
    close(c);
    close(f->fd);
    return NULL;
}

static void fake_remote_roundtrip(const char *reply, bool chunked) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t al = sizeof(a);
    AGENT_TEST_ASSERT(bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0);
    AGENT_TEST_ASSERT(listen(fd, 1) == 0);
    getsockname(fd, (struct sockaddr *)&a, &al);
    fake_srv f = {fd, reply, strlen(reply)};
    pthread_t th;
    pthread_create(&th, NULL, fake_srv_thread, &f);
    char url[64];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d", ntohs(a.sin_port));
    q36_remote_cfg cfg = {url, "k", 5};
    q36_remote_result r;
    int rc = q36_remote_chat(&cfg, "{\"stream\":true}", NULL, NULL, NULL, &r);
    pthread_join(th, NULL);
    AGENT_TEST_ASSERT(rc == 0);
    AGENT_TEST_ASSERT(r.content && !strcmp(r.content, "Hello wörld"));
    AGENT_TEST_ASSERT(r.reasoning && !strcmp(r.reasoning, "think"));
    AGENT_TEST_ASSERT(r.ncalls == 1 && r.calls[0].name && !strcmp(r.calls[0].name, "read"));
    AGENT_TEST_ASSERT(r.calls[0].arguments && !strcmp(r.calls[0].arguments, "{\"path\":\"a.c\"}"));
    AGENT_TEST_ASSERT(r.calls[0].id && !strcmp(r.calls[0].id, "call_7"));
    AGENT_TEST_ASSERT(!strcmp(r.finish, "tool_calls"));
    AGENT_TEST_ASSERT(r.prompt_tokens == 100 && r.cached_tokens == 90 && r.reasoning_tokens == 3);
    AGENT_TEST_ASSERT(r.decode_ms > 11.0 && r.decode_ms < 13.0);
    q36_remote_result_free(&r);
    (void)chunked;
}

static void test_remote_client_sse(void) {
    const char *body =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thi\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"nk\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello w\\u00f6rld\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_7\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\"\"}}]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":\\\"a.c\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":9,"
        "\"prompt_tokens_details\":{\"cached_tokens\":90},\"completion_tokens_details\":{\"reasoning_tokens\":3},"
        "\"timings\":{\"prefill_ms\":5.5,\"decode_ms\":12.0}}}\n\n"
        "data: [DONE]\n\n";
    char plain[4096], chunked[4600];
    snprintf(plain, sizeof(plain), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n%s", body);
    fake_remote_roundtrip(plain, false);
    /* same stream, chunk-framed in uneven pieces */
    size_t bl = strlen(body), off = 0, n = (size_t)snprintf(chunked, sizeof(chunked),
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\n\r\n");
    size_t piece = 61;
    while (off < bl) {
        size_t c = bl - off < piece ? bl - off : piece;
        n += (size_t)snprintf(chunked + n, sizeof(chunked) - n, "%zx\r\n", c);
        memcpy(chunked + n, body + off, c);
        n += c;
        memcpy(chunked + n, "\r\n", 2);
        n += 2;
        off += c;
    }
    memcpy(chunked + n, "0\r\n\r\n", 6);
    fake_remote_roundtrip(chunked, true);
}

static void test_remote_errors_and_json(void) {
    q36_remote_cfg cfg = {"https://x", NULL, 1};
    q36_remote_result r;
    AGENT_TEST_ASSERT(q36_remote_chat(&cfg, "{}", NULL, NULL, NULL, &r) != 0);
    AGENT_TEST_ASSERT(strstr(r.err, "http://"));
    q36_jv *j = q36_jv_parse("{\"a\":[1,2.5,true,null,\"x\\n\\ud83d\\ude00\"],\"b\":{}}");
    AGENT_TEST_ASSERT(j && q36_jv_count(q36_jv_get(j, "a")) == 5);
    char sc[40];
    AGENT_TEST_ASSERT(!strcmp(q36_jv_text(q36_jv_at(q36_jv_get(j, "a"), 0), sc, sizeof(sc)), "1"));
    AGENT_TEST_ASSERT(!strcmp(q36_jv_text(q36_jv_at(q36_jv_get(j, "a"), 4), sc, sizeof(sc)), "x\n\xf0\x9f\x98\x80"));
    q36_jv_free(j);
    AGENT_TEST_ASSERT(q36_jv_parse("{\"a\":") == NULL);
    AGENT_TEST_ASSERT(q36_jv_parse("{} x") == NULL);
}

static void test_think_policy_and_flags(void) {
    char *a1[] = {"q36-agent", "--think", "auto", "--action-budget", "100", "--max-recoveries", "5"};
    agent_config c = parse_options(7, a1);
    AGENT_TEST_ASSERT(c.gen.think_auto && c.gen.think_mode == Q36_THINK_HIGH);
    AGENT_TEST_ASSERT(c.gen.action_budget == 100 && c.gen.max_recoveries == 5);
    char *a2[] = {"q36-agent", "--think", "low"};
    c = parse_options(3, a2);
    AGENT_TEST_ASSERT(!c.gen.think_auto && c.gen.think_mode == Q36_THINK_LOW);
    char *a3[] = {"q36-agent", "--think", "--vulkan"};
    c = parse_options(3, a3);
    AGENT_TEST_ASSERT(!c.gen.think_auto && c.gen.think_mode == Q36_THINK_HIGH);
    AGENT_TEST_ASSERT(c.gen.action_budget == AGENT_ACTION_BUDGET_DEFAULT &&
                      c.gen.max_repeat_tool == 1 && c.gen.max_stagnant_turns == 2);
    char *a4[] = {"q36-agent", "--server", "http://h:1", "--server-model", "m", "--think", "auto"};
    c = parse_options(7, a4);
    AGENT_TEST_ASSERT(c.server_url && !strcmp(c.server_model, "m"));

    AGENT_TEST_ASSERT(agent_think_level_for_prompt("list the files in src/") == 0);
    AGENT_TEST_ASSERT(agent_think_level_for_prompt("Implement a feature that parses dates") == 1);
    AGENT_TEST_ASSERT(agent_think_level_for_prompt("debug the race and refactor the locking across files") == 2);
    AGENT_TEST_ASSERT(agent_think_level_raise(0, 0, false) == 0);
    AGENT_TEST_ASSERT(agent_think_level_raise(0, 1, false) == 1);
    AGENT_TEST_ASSERT(agent_think_level_raise(1, 2, false) == 2);
    AGENT_TEST_ASSERT(agent_think_level_raise(2, 0, true) == 2);       /* never lowered */
    c.gen.thinking_budget = 50000;
    c.gen.think_auto = true;
    AGENT_TEST_ASSERT(agent_think_budget_for(&c, 0) == 256 && agent_think_budget_for(&c, 2) == 2048);
    c.gen.thinking_budget = 500;                                       /* explicit cap wins */
    AGENT_TEST_ASSERT(agent_think_budget_for(&c, 2) == 500);
    c.gen.think_auto = false;
    AGENT_TEST_ASSERT(agent_think_budget_for(&c, 0) == 500);
}

static void test_repetitive_tool_aborted_before_execution(void) {
    char path[] = "/tmp/q36-agent-repetition-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    AGENT_TEST_ASSERT(write(fd, "original\n", 9) == 9);
    close(fd);

    agent_buf partial = {0};
    agent_buf_puts(&partial,
        "<tool_call>\n<function=write>\n<parameter=path>\n");
    agent_buf_puts(&partial, path);
    agent_buf_puts(&partial,
        "\n</parameter>\n<parameter=content>\n");
    for (int i = 0; i < AGENT_REPETITION_EXACT_LINE_RUN; i++)
        agent_buf_puts(&partial, "scene.add(makeTree({ x: 10, y: 20 }));\n");

    agent_qwen_tool_parser parser = {.state = AGENT_QWEN_TOOL_SEARCH};
    agent_qwen_tool_feed(&parser, partial.ptr, partial.len);
    AGENT_TEST_ASSERT(parser.state == AGENT_QWEN_TOOL_PARAM_VALUE);
    AGENT_TEST_ASSERT(parser.calls.len == 0);
    AGENT_TEST_ASSERT(agent_repetition_text_detected(parser.raw, parser.raw_len));
    AGENT_TEST_ASSERT(strstr(agent_tool_repetition_warning,
                             "was NOT executed") != NULL);

    agent_worker worker = {0};
    for (int i = 0; i < 6; i++) q36_tokens_push(&worker.transcript, 200 + i);
    agent_rollback_assistant_suffix(&worker, 2);
    AGENT_TEST_ASSERT(worker.transcript.len == 2);

    char *contents = NULL;
    size_t contents_len = 0;
    char err[160] = {0};
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &contents, &contents_len,
                                            err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(contents_len == 9 && !memcmp(contents, "original\n", 9));
    free(contents);
    agent_qwen_tool_parser_free(&parser);
    free(partial.ptr);
    q36_tokens_free(&worker.transcript);
    unlink(path);
}

static void test_agent_frequency_penalty_sampling_path(void) {
    int token_data[] = {11, 12, 13, 14, 15};
    q36_tokens transcript = {.v = token_data, .len = 5};
    agent_config cfg = {0};
    cfg.gen.temperature = 0.7f;
    cfg.gen.top_k = 20;
    cfg.gen.top_p = 0.9f;
    cfg.gen.min_p = 0.05f;
    cfg.gen.frequency_penalty = 0.15f;

    agent_sampling_request request = agent_sampling_request_build(
        &cfg, false, &transcript, 3);
    AGENT_TEST_ASSERT(request.frequency_penalty == 0.15f);
    AGENT_TEST_ASSERT(request.penalty_token_count == 3);
    AGENT_TEST_ASSERT(request.penalty_tokens == token_data + 2);
    AGENT_TEST_ASSERT(request.penalty_tokens[0] == 13);

    request = agent_sampling_request_build(&cfg, true, &transcript, 3);
    AGENT_TEST_ASSERT(request.frequency_penalty == 0.0f);
    AGENT_TEST_ASSERT(request.penalty_token_count == 0);
    AGENT_TEST_ASSERT(request.penalty_tokens == NULL);
}

static void test_observation_error_is_not_context_exhaustion(void) {
    agent_worker worker = {0};
    q36_tokens_push(&worker.transcript, 42);
    agent_tool_observation observation;
    agent_tool_observation_init(&observation);
    agent_tool_observation_puts(&observation, "image result");
    q36_vision_embedding invalid = {0};
    agent_tool_observation_add_image(&observation, &invalid);
    char err[160] = {0};
    int count = -1;
    AGENT_TEST_ASSERT(agent_tool_observation_fits(&worker, &observation, 16,
                                                 &count, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "invalid image observation") != NULL);
    AGENT_TEST_ASSERT(count == -1);
    AGENT_TEST_ASSERT(worker.transcript.len == 1 && worker.transcript.v[0] == 42);
    agent_tool_observation_free(&observation);
    q36_tokens_free(&worker.transcript);
}

int main(int argc, char **argv) {
    q36_cli_expand_long_equals(&argc, &argv, "q36_agent_test");
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts("usage: q36_agent_test [--terminal-fixtures DIR]");
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--password-child")) return password_child();
    if (argc == 4 && !strcmp(argv[1], "--password-driver"))
        return password_driver(argv[2], atoi(argv[3]), false);
    if (argc == 4 && !strcmp(argv[1], "--noninteractive-driver"))
        return password_driver(argv[2], atoi(argv[3]), true);
    if (argc == 2 && !strcmp(argv[1], "--terminal-driver")) return test_terminal_driver();
    if (argc == 2 && !strcmp(argv[1], "--terminal-fixtures")) {
        fputs("q36_agent_test: missing value for --terminal-fixtures\n", stderr);
        return 2;
    }
    if (argc == 3 && !strcmp(argv[1], "--terminal-fixtures")) {
        if (q36_cli_token_is_option(argv[2])) {
            fputs("q36_agent_test: missing value for --terminal-fixtures\n", stderr);
            return 2;
        }
        test_output_dir = argv[2];
    }
    else if (argc != 1) {
        fprintf(stderr, "q36_agent_test: unknown option or invalid arguments: %s\n",
                argv[1]);
        return 2;
    }
    q36_agent_unit_tests_run();
    test_worker_ownership();
    test_worker_pause_ui_waits();
    test_compaction_boundaries();
    test_partial_tool_interrupt_rollback();
    test_repetitive_tool_aborted_before_execution();
    test_action_leak_and_watchdog();
    test_think_policy_and_flags();
    test_remote_client_sse();
    test_remote_errors_and_json();
    test_task_state_render();
    test_vw_replays_the_observed_pathology();
    test_vw_target_repair_resets_budget();
    test_vw_repeated_verification_finalizes();
    test_vw_harness_failure_is_bounded_and_reported();
    test_vw_state_survives_compaction_render();
    test_vw_classification_edges();
    test_vw_workspace_under_tmp();
    test_vw_failed_bash_fact_and_image_budget();
    test_vw_flags_and_summary();
    test_vw_compaction_recovery_stays_in_implement();
    test_vw_incomplete_evidence_and_reentry();
    test_write_append_and_edit_lines();
    test_leak_recovery_protocol();
    test_agent_frequency_penalty_sampling_path();
    test_observation_error_is_not_context_exhaustion();
    test_atomic_file_tools();
    test_streaming_file_tools();
    test_read_blob_cap();
    test_background_jobs();
    test_shell_terminal_controls();
    test_fragmented_terminal_input();
    test_markdown_literals();
    test_hint_rendering();
    test_unicode_output_and_footer();
    test_footer_only_updates();
    test_bash_noninteractive();
    if (agent_test_failures) {
        fprintf(stderr, "q36-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("q36-agent tests: ok");
    return 0;
}
