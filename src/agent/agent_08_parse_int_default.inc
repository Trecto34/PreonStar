static int agent_parse_int_default(const char *s, int def, int min, int max) {
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return def;
    if (v < min) v = min;
    if (v > max) v = max;
    return (int)v;
}

static bool agent_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}

#define AGENT_FILE_MAX_BYTES (16*1024*1024)
#define AGENT_READ_DEFAULT_LINES 500
/* Deterministic read caps (~4k tokens total).  A line longer than
 * AGENT_READ_LINE_MAX is treated as a minified/vendor blob: only
 * AGENT_READ_BLOB_KEEP bytes of it reach the model, plus an omission marker. */
#define AGENT_READ_MAX_BYTES (16 * 1024)
#define AGENT_READ_LINE_MAX 4096
#define AGENT_READ_BLOB_KEEP 256
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2
/* Leave room for the private summary without crossing the next KV growth tier. */
#define AGENT_COMPACT_SOFT_PERCENT 70
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
#define AGENT_COMPACT_TAIL_DIVISOR 10
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096

static bool agent_context_should_compact(int ctx, int used) {
    if (ctx <= 0 || used <= 0) return false;
    if (used >= (ctx * AGENT_COMPACT_SOFT_PERCENT) / 100) return true;
    int free_threshold = AGENT_COMPACT_MIN_FREE_TOKENS;
    int proportional = ctx / 4;
    if (free_threshold > proportional) free_threshold = proportional;
    return ctx - used <= free_threshold;
}

static int agent_think_soft_close_start(int hard_budget) {
    if (hard_budget <= 1) return hard_budget;
    int start = (int)(((int64_t)hard_budget * 3) / 4);
    return start > 0 ? start : 1;
}

static bool agent_think_hard_budget_reached(int think_tokens, int hard_budget) {
    return hard_budget > 0 && think_tokens >= hard_budget;
}

static int agent_think_close_rank_limit(int think_tokens, int hard_budget) {
    return p36_think_close_rank_limit(
        think_tokens, agent_think_soft_close_start(hard_budget));
}

#include "p36_loopguard.h"

static FILE *agent_open_regular_file(const char *path) {
    char expanded[PATH_MAX];
    if (!agent_expand_home_path(path, expanded, sizeof(expanded))) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    int fd = open(expanded, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    int rc = fstat(fd, &st);
    if (rc != 0 || !S_ISREG(st.st_mode)) {
        int saved = rc != 0 ? errno : EINVAL;
        close(fd);
        errno = saved;
        return NULL;
    }
    FILE *fp = fdopen(fd, "rb");
    if (!fp) { int saved = errno; close(fd); errno = saved; }
    return fp;
}

static int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen) {
    FILE *fp = agent_open_regular_file(path);
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    char *buf = NULL;
    size_t used = 0, cap = 0;
    char tmp[8192];
    while (true) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (used + n > AGENT_FILE_MAX_BYTES) {
                fclose(fp);
                free(buf);
                snprintf(err, errlen, "file too large: %s exceeds %d bytes",
                         path, AGENT_FILE_MAX_BYTES);
                return -1;
            }
            if (used + n + 1 > cap) {
                cap = cap ? cap * 2 : 8192;
                while (cap < used + n + 1) cap *= 2;
                buf = xrealloc(buf, cap);
            }
            memcpy(buf + used, tmp, n);
            used += n;
            buf[used] = '\0';
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                snprintf(err, errlen, "read %s: %s", path, strerror(errno));
                fclose(fp);
                free(buf);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    if (!buf) buf = xstrdup("");
    *data = buf;
    *len = used;
    return 0;
}

static int agent_line_for_offset(const agent_line_spans *spans, size_t offset) {
    if (!spans || spans->len <= 0) return 1;
    for (int i = 0; i < spans->len; i++) {
        if (offset < spans->v[i].end) return i + 1;
    }
    return spans->len;
}

static bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta) {
    agent_line_spans old_spans = {0};
    agent_line_spans new_spans = {0};
    agent_split_lines(old_data, old_len, &old_spans);
    agent_split_lines(new_data, new_len, &new_spans);
    bool ok = old_spans.len > 0;
    if (ok) {
        size_t old_last = edit_offset;
        if (replaced_len > 0) old_last = edit_offset + replaced_len - 1;
        if (old_last >= old_len) old_last = old_len ? old_len - 1 : 0;
        if (start_line) *start_line = agent_line_for_offset(&old_spans, edit_offset);
        if (end_line) *end_line = agent_line_for_offset(&old_spans, old_last);
        if (delta) *delta = new_spans.len - old_spans.len;
    }
    agent_line_spans_free(&old_spans);
    agent_line_spans_free(&new_spans);
    return ok;
}

static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);

static char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind) {
    agent_buf b = {.limit = AGENT_TOOL_MAX_BYTES};
    char msg[PATH_MAX + 180];
    snprintf(msg, sizeof(msg), "Edited %s using %s\n", path, kind);
    agent_buf_puts(&b, msg);
    if (start_line > 0 && end_line >= start_line) {
        snprintf(msg, sizeof(msg),
                 "Touched old lines %d-%d; current post-edit context follows.\n",
                 start_line, end_line);
        agent_buf_puts(&b, msg);
        if (delta != 0) {
            snprintf(msg, sizeof(msg),
                     "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n",
                     end_line, delta, end_line + 1, end_line + 1 + delta);
            agent_buf_puts(&b, msg);
        }
    }
    if (start_line > 0 && end_line >= start_line) {
        int new_anchor_end = end_line + delta;
        if (new_anchor_end < start_line) new_anchor_end = start_line;
        agent_edit_result_append_context(&b, path, new_data, new_len,
                                         start_line, new_anchor_end);
    }
    return agent_buf_take(&b);
}

static void agent_worker_set_more(agent_worker *w, const char *path,
                                  int next_line, bool bare) {
    if (path != w->more_path)
        snprintf(w->more_path, sizeof(w->more_path), "%s", path ? path : "");
    w->more_next_line = next_line;
    w->more_bare = bare;
    w->more_valid = path && path[0] && next_line > 0;
    w->more_byte_offset = 0;
    w->more_mid_line = false;
}


/* Read file text for the model.  Normal mode shows plain line numbers.  Raw
 * mode is reserved for cases where line decoration would corrupt the payload
 * being inspected. */
static char *agent_read_range_from(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more, off_t offset, bool mid_line) {
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    agent_buf out = {0}, body = {0};
    FILE *fp = agent_open_regular_file(path);
    if (!fp) goto failed;
    struct stat st;
    if (fstat(fileno(fp), &st) != 0) goto failed;
    if (!S_ISREG(st.st_mode)) { errno = EINVAL; goto failed; }
    if (start_line < 1) start_line = 1;
    if (max_lines <= 0) max_lines = AGENT_READ_DEFAULT_LINES;
    int c, line = 1, lines = 0;
    if (offset > 0) {
        if (fseeko(fp, offset, SEEK_SET) != 0) goto failed;
        line = start_line;
    }
    while (line < start_line && (c = fgetc(fp)) != EOF) {
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
        }
        if (c == '\n' || c == '\r') line++;
    }
    bool beginning = !mid_line, prefix = true;
    int last_line = 0;
    size_t line_start = 0, line_bytes = 0, blob_bytes = 0;
    int blob_line = 0;
    bool in_blob = false;
    while ((whole_file || lines < max_lines) && (c = fgetc(fp)) != EOF) {
        if (in_blob && c != '\n' && c != '\r') { line_bytes++; continue; }
        if (!in_blob && body.len >= AGENT_READ_MAX_BYTES &&
            ((c & 0xc0) != 0x80 || body.len >= AGENT_READ_MAX_BYTES + 3)) {
            ungetc(c, fp);
            break;
        }
        if (!c) {
            agent_buf_puts(&out, "Tool error: read encountered binary data\n");
            goto done;
        }
        last_line = line;
        if (prefix && !bare) {
            char label[80];
            snprintf(label, sizeof(label), "%d%s ", line, beginning ? "" : " (continued)");
            agent_buf_puts(&body, label);
        }
        if (prefix) { line_start = body.len; line_bytes = 0; }
        prefix = false;
        beginning = false;
        char ch = (char)c;
        if (c != '\n' && c != '\r' && ++line_bytes > AGENT_READ_LINE_MAX) {
            /* Keep a short head of the line, drop the rest until its newline. */
            size_t keep = line_start + AGENT_READ_BLOB_KEEP;
            if (!bare) { char *nl = memchr(body.ptr + line_start, ' ', body.len - line_start);
                         if (nl) keep = (size_t)(nl - body.ptr) + 1 + AGENT_READ_BLOB_KEEP; }
            if (keep < body.len) {
                while (keep > line_start && ((unsigned char)body.ptr[keep] & 0xc0) == 0x80) keep--;
                body.len = keep;
            }
            body.ptr[body.len] = '\0';
            in_blob = true;
            if (!blob_line) blob_line = line;
            continue;
        }
        if (in_blob) {
            /* Newline closing a blob line: marker goes before the line break. */
            char marker[96];
            snprintf(marker, sizeof(marker), "\n[MINIFIED_OR_VENDOR_BLOB OMITTED: %zu bytes]", line_bytes);
            if (!blob_bytes) blob_bytes = line_bytes;
            agent_buf_puts(&body, marker);
            in_blob = false;
        }
        if (c == '\r') {
            int next = fgetc(fp);
            if (bare) agent_buf_append(&body, &ch, 1);
            if (next == '\n') {
                ch = '\n';
            } else {
                if (next != EOF) ungetc(next, fp);
                if (bare) ch = 0;
                else ch = '\n';
            }
        }
        if (ch) agent_buf_append(&body, &ch, 1);
        if (c == '\n' || c == '\r') {
            if (line == INT_MAX) { errno = EOVERFLOW; goto failed; }
            line++;
            lines++;
            beginning = prefix = true;
        }
    }
    if (in_blob) {
        char marker[96];
        snprintf(marker, sizeof(marker), "\n[MINIFIED_OR_VENDOR_BLOB OMITTED: %zu bytes]", line_bytes);
        if (!blob_bytes) blob_bytes = line_bytes;
        agent_buf_puts(&body, marker);
    }
    if (ferror(fp)) goto failed;
    off_t next_offset = ftello(fp);
    if (next_offset < 0) goto failed;
    c = fgetc(fp);
    bool more = c != EOF;
    if (ferror(fp)) goto failed;
    if (whole_file && more) {
        agent_buf_puts(&out, "Tool error: whole read exceeds the 16 KiB output limit; use read and more for chunks\n");
        goto done;
    }
    if (!bare) {
        char header[PATH_MAX + 128];
        snprintf(header, sizeof(header), "%s: lines %d-%d%s\n", path,
                 last_line ? start_line : 0, last_line, more ? " (partial read)" : " (end of file)");
        agent_buf_puts(&out, header);
    }
    agent_buf_append(&out, body.ptr, body.len);
    if (blob_line) {
        char meta[PATH_MAX + 320];
        snprintf(meta, sizeof(meta),
                 "\nREAD_OUTPUT_TRUNCATED\npath=%s\nrequested_lines=%d-%d\noffending_line=%d\n"
                 "line_bytes=%zu\nreturned_bytes=%zu\n"
                 "next=do not read that line again; use search or a line range that excludes it\n",
                 path, start_line, last_line, blob_line, blob_bytes, body.len);
        agent_buf_puts(&out, meta);
    }
    if (more) {
        char note[256];
        snprintf(note, sizeof(note), "\n[Read truncated. continue_offset=%d; continue_byte_offset=%lld%s. Call more to continue.]\n",
                 line, (long long)next_offset, beginning ? "" : " (within line)");
        agent_buf_puts(&out, note);
    }
    if (set_more) {
        agent_worker_set_more(w, more ? path : NULL, more ? line : 0, bare);
        w->more_byte_offset = next_offset;
        w->more_mid_line = !beginning;
    }
    free(body.ptr);
    fclose(fp);
    return agent_buf_take(&out);
failed:
    agent_buf_puts(&out, "Tool error: read failed: ");
    agent_buf_puts(&out, strerror(errno));
    agent_buf_puts(&out, "\n");
done:
    if (fp) fclose(fp);
    free(body.ptr);
    if (set_more) agent_worker_set_more(w, NULL, 0, false);
    return agent_buf_take(&out);
}

static char *agent_read_range(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more) {
    return agent_read_range_from(w, path, start_line, max_lines, whole_file,
                                 bare, set_more, 0, false);
}

static char *agent_tool_read(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    bool whole = agent_parse_bool_default(agent_tool_arg_value(call, "whole"), false);
    int start = agent_parse_int_default(agent_tool_arg_value(call, "start_line"),
                                        1, 1, INT_MAX);
    int count = agent_parse_int_default(agent_tool_arg_value(call, "max_lines"),
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    bool raw = agent_parse_bool_default(agent_tool_arg_value(call, "raw"), false);
    return agent_read_range(w, path, start, count, whole, raw, true);
}

static char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    int count = agent_parse_int_default(agent_tool_arg_value(call, "count"),
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range_from(w, w->more_path, w->more_next_line, count, false,
                                 w->more_bare, true, w->more_byte_offset, w->more_mid_line);
}

static char *agent_tool_write(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    const char *content = agent_tool_arg_value(call, "content");
    if (!path || !path[0]) return xstrdup("Tool error: write requires path\n");
    if (!content) return xstrdup("Tool error: write requires content\n");
    size_t len = strlen(content);
    char msg[PATH_MAX + 200];
    if (len > AGENT_WRITE_PAYLOAD_MAX) {
        snprintf(msg, sizeof(msg),
                 "Tool error: WRITE_PAYLOAD_TOO_LARGE path=%s bytes=%zu max_bytes=%d "
                 "next=write the first part, then write append=true for the rest\n",
                 path, len, AGENT_WRITE_PAYLOAD_MAX);
        return xstrdup(msg);
    }
    bool append = agent_parse_bool_default(agent_tool_arg_value(call, "append"), false);
    char err[256];
    char *old = NULL;
    size_t old_len = 0;
    char *out = (char *)content;
    size_t out_len = len;
    if (append && agent_read_file_bytes(path, &old, &old_len, err, sizeof(err)) == 0) {
        out = xmalloc(old_len + len + 1);
        memcpy(out, old, old_len);
        memcpy(out + old_len, content, len + 1);
        out_len = old_len + len;
    }
    int rc = agent_replace_file(path, out, out_len, old, old_len, err, sizeof(err));
    if (out != content) free(out);
    free(old);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    snprintf(msg, sizeof(msg), "OK write path=%s bytes=%zu total=%zu append=%d max_payload=%d\n",
             path, len, out_len, append, AGENT_WRITE_PAYLOAD_MAX);
    return xstrdup(msg);
}

static char *agent_tool_list(const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    char expanded[PATH_MAX];
    if (!agent_expand_home_path(path, expanded, sizeof(expanded)))
        return xstrdup("Tool error: list path is too long\n");
    DIR *dir = opendir(expanded);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {.limit = AGENT_TOOL_MAX_BYTES};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        if (snprintf(full, sizeof(full), "%s/%s", expanded, de->d_name) >=
            (int)sizeof(full)) continue;
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

/* ============================================================================
 * Edit And Search Tools
 * ============================================================================
 */

static bool agent_same_file_version(const struct stat *a, const struct stat *b) {
#ifdef __APPLE__
    struct timespec am = a->st_mtimespec, bm = b->st_mtimespec;
    struct timespec ac = a->st_ctimespec, bc = b->st_ctimespec;
#else
    struct timespec am = a->st_mtim, bm = b->st_mtim;
    struct timespec ac = a->st_ctim, bc = b->st_ctim;
#endif
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && a->st_nlink == b->st_nlink &&
           am.tv_sec == bm.tv_sec && am.tv_nsec == bm.tv_nsec &&
           ac.tv_sec == bc.tv_sec && ac.tv_nsec == bc.tv_nsec;
}

#ifdef __linux__
/* Linux stores ACLs as xattrs too. A metadata copy failure must abort the
 * replacement, not silently remove access rules from the edited file. */
static int agent_copy_file_xattrs(int src, int dst) {
    if (fremovexattr(dst, "system.posix_acl_access") != 0 &&
        errno != ENODATA && errno != ENOTSUP) return -1;
    ssize_t count = flistxattr(src, NULL, 0);
    if (count < 0) return errno == ENOTSUP ? 0 : -1;
    if (!count) return 0;
    char *names = xmalloc((size_t)count);
    count = flistxattr(src, names, (size_t)count);
    int rc = -1, saved_errno;
    if (count < 0) goto done;
    for (ssize_t pos = 0; pos < count;) {
        const char *name = names + pos;
        ssize_t len = fgetxattr(src, name, NULL, 0);
        if (len < 0) goto done;
        void *value = xmalloc(len ? (size_t)len : 1);
        ssize_t readlen = fgetxattr(src, name, value, (size_t)len);
        int copied = readlen < 0 ? -1 : fsetxattr(dst, name, value, (size_t)readlen, 0);
        int saved_errno = errno;
        free(value);
        errno = saved_errno;
        if (copied != 0) goto done;
        pos += (ssize_t)strlen(name) + 1;
    }
    rc = 0;
done:
    saved_errno = errno;
    free(names);
    errno = saved_errno;
    return rc;
}
#endif

/* Commit only complete files. Follow existing symlinks, but refuse hard links:
 * an atomic rename cannot preserve their shared-inode semantics. The version
 * checks catch concurrent edits, though unrelated writers do not take a lock. */
static int agent_replace_file(const char *path, const char *data, size_t len,
                              const char *expected, size_t expected_len,
                              char *err, size_t errlen) {
    char expanded[PATH_MAX];
    if (!agent_expand_home_path(path, expanded, sizeof(expanded))) {
        snprintf(err, errlen, "path is too long");
        return -1;
    }
    path = expanded;
    char target[PATH_MAX], temp[PATH_MAX] = "";
    struct stat before, after;
    int src = -1, fd = -1, rc = -1;
    bool exists = lstat(path, &before) == 0;
    if (!exists && errno != ENOENT) goto failed;
    if (exists) {
        if (!realpath(path, target)) goto failed;
        src = open(target, O_RDWR | O_NONBLOCK | O_NOFOLLOW);
        if (src < 0 || fstat(src, &before) != 0) goto failed;
        if (!S_ISREG(before.st_mode) || before.st_nlink != 1) {
            snprintf(err, errlen, "refusing to replace non-regular or hard-linked file: %.160s", path);
            goto done;
        }
        if (expected) {
            if (before.st_size < 0 || (uintmax_t)before.st_size != expected_len)
                goto changed;
            char buf[8192];
            size_t pos = 0;
            while (pos < expected_len) {
                size_t count = expected_len - pos;
                if (count > sizeof(buf)) count = sizeof(buf);
                ssize_t n = read(src, buf, count);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) goto failed;
                if (!n || memcmp(buf, expected + pos, (size_t)n)) goto changed;
                pos += (size_t)n;
            }
        }
    } else {
        if (expected) goto changed;
        if (snprintf(target, sizeof(target), "%s", path) >= (int)sizeof(target)) {
            errno = ENAMETOOLONG;
            goto failed;
        }
    }
    if (snprintf(temp, sizeof(temp), "%s.p36-XXXXXX", target) >= (int)sizeof(temp)) {
        temp[0] = 0;
        errno = ENAMETOOLONG;
        goto failed;
    }
    fd = mkstemp(temp);
    if (fd < 0) { temp[0] = 0; goto failed; }
    if (!exists) {
        /* Let open apply the process umask without changing it in this thread. */
        if (unlink(temp) != 0) goto failed;
        close(fd);
        fd = open(temp, O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd < 0) { temp[0] = 0; goto failed; }
    }
    for (size_t pos = 0; pos < len;) {
        ssize_t n = write(fd, data + pos, len - pos);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; goto failed; }
        pos += (size_t)n;
    }
    if (exists) {
        if (fchown(fd, before.st_uid, before.st_gid) != 0) goto failed;
        if (fchmod(fd, before.st_mode & 07777) != 0) goto failed;
#ifdef __APPLE__
        if (fcopyfile(src, fd, NULL, COPYFILE_ACL | COPYFILE_XATTR) != 0) goto failed;
#elif defined(__linux__)
        if (agent_copy_file_xattrs(src, fd) != 0) goto failed;
#endif
    }
    if (fsync(fd) != 0) goto failed;
    if (close(fd) != 0) { fd = -1; goto failed; }
    fd = -1;
    if (exists) {
        char resolved[PATH_MAX];
        if (!realpath(path, resolved) || strcmp(resolved, target) ||
            stat(target, &after) != 0 || !agent_same_file_version(&before, &after))
            goto changed;
        if (rename(temp, target) != 0) goto failed;
    } else {
        /* Unlike rename, link will not overwrite a concurrently created file. */
        if (link(temp, target) != 0) goto failed;
        unlink(temp);
    }
    temp[0] = 0;
    rc = 0;
    goto done;
changed:
    snprintf(err, errlen, "file changed while editing; read it again: %.160s", path);
    goto done;
failed:
    snprintf(err, errlen, "replace %.160s: %.64s", path, strerror(errno));
done:
    if (fd >= 0) close(fd);
    if (src >= 0) close(src);
    if (temp[0]) unlink(temp);
    return rc;
}


static void agent_edit_result_append_line(agent_buf *b, const char *data,
                                          const agent_line_span *sp,
                                          int line) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%d ", line);
    agent_buf_puts(b, prefix);
    agent_buf_append(b, data + sp->start, sp->content_end - sp->start);
    agent_buf_puts(b, "\n");
}

/* Successful edits return the nearby post-edit file shape.  This spends cheap
 * prefill tokens to save expensive model retries: the model immediately sees
 * shifted line numbers, braces, semicolons, and accidental duplication. */
static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end) {
    enum {
        CONTEXT_BEFORE = 5,
        CONTEXT_AFTER = 8,
        EDITED_CONTEXT_HEAD = 18,
        EDITED_CONTEXT_TAIL = 18
    };

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (spans.len <= 0) {
        agent_line_spans_free(&spans);
        return;
    }

    if (anchor_start < 1) anchor_start = 1;
    if (anchor_start > spans.len) anchor_start = spans.len;
    if (anchor_end < anchor_start) anchor_end = anchor_start;
    if (anchor_end > spans.len) anchor_end = spans.len;

    int ctx_start = anchor_start - CONTEXT_BEFORE;
    if (ctx_start < 1) ctx_start = 1;
    int ctx_end = anchor_end + CONTEXT_AFTER;
    if (ctx_end > spans.len) ctx_end = spans.len;

    char hdr[PATH_MAX + 160];
    snprintf(hdr, sizeof(hdr),
             "Current file around edit: %s lines %d-%d of %d\n",
             path, ctx_start, ctx_end, spans.len);
    agent_buf_puts(b, hdr);

    int edited_lines = anchor_end - anchor_start + 1;
    if (edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL) {
        for (int line = ctx_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    } else {
        int head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        int tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for (int line = ctx_start; line <= head_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
        snprintf(hdr, sizeof(hdr),
                 "... %d edited lines omitted ...\n",
                 tail_start - head_end - 1);
        agent_buf_puts(b, hdr);
        for (int line = tail_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    }

    agent_line_spans_free(&spans);
}

static const char *agent_memmem_simple(const char *hay, size_t hay_len,
                                       const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    size_t last = hay_len - needle_len;
    for (size_t i = 0; i <= last; i++) {
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len))
            return hay + i;
    }
    return NULL;
}

/* Needle differs from the file only in whitespace?  Say so, so the model
 * switches to start_line/end_line instead of re-counting spaces. */
static const char *agent_whitespace_hint(const char *data, size_t len,
                                         const char *needle, size_t needle_len) {
    char *a = xmalloc(len + 1), *b = xmalloc(needle_len + 1);
    size_t an = 0, bn = 0;
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)data[i])) a[an++] = data[i];
    }
    for (size_t i = 0; i < needle_len; i++) {
        if (!isspace((unsigned char)needle[i])) b[bn++] = needle[i];
    }
    bool hit = bn && agent_memmem_simple(a, an, b, bn);
    free(a);
    free(b);
    return hit ? " hint=whitespace_differs use=start_line/end_line" : "";
}

static bool agent_find_unique(const char *data, size_t len,
                              const char *needle, size_t needle_len,
                              const char **match, const char *label,
                              char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "EDIT_%s_EMPTY", label);
        return false;
    }
    const char *first = agent_memmem_simple(data, len, needle, needle_len);
    if (!first) {
        snprintf(err, err_len,
                 "EDIT_MATCH_NOT_FOUND part=%s expected_matches=1 actual_matches=0%s",
                 label, agent_whitespace_hint(data, len, needle, needle_len));
        return false;
    }
    int found = 1;
    const char *at = first;
    while (found < 9) {
        size_t off = (size_t)(at - data) + 1;
        at = off <= len ? agent_memmem_simple(data + off, len - off, needle, needle_len) : NULL;
        if (!at) break;
        found++;
    }
    if (found > 1) {
        snprintf(err, err_len,
                 "EDIT_MATCH_NOT_UNIQUE part=%s expected_matches=1 actual_matches=%d%s",
                 label, found, found == 9 ? "+" : "");
        return false;
    }
    *match = first;
    return true;
}

/* Find an anchor only in the suffix after start.
 *
 * Anchored edits use "head [upto] tail": the head fixes the edit start, and
 * the tail should delimit the first unique end point after that start. A tail
 * may legitimately appear earlier in the file, so checking global uniqueness
 * would reject valid edits.
 */
static bool agent_find_unique_after(const char *data, size_t len,
                                    const char *start,
                                    const char *needle, size_t needle_len,
                                    const char **match, const char *label,
                                    char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "EDIT_%s_EMPTY", label);
        return false;
    }
    if (start < data || start > data + len) {
        snprintf(err, err_len, "%s search starts outside file", label);
        return false;
    }
    size_t off = (size_t)(start - data);
    const char *first = agent_memmem_simple(data + off, len - off,
                                            needle, needle_len);
    if (!first) {
        snprintf(err, err_len,
                 "EDIT_MATCH_NOT_FOUND part=%s expected_matches=1 actual_matches=0 after=head",
                 label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len,
                 "EDIT_MATCH_NOT_UNIQUE part=%s expected_matches=1 actual_matches=2+ after=head",
                 label);
        return false;
    }
    *match = first;
    return true;
}

static bool agent_edit_old_may_be_closing_tag(const char *text, size_t len) {
    size_t i = 0;
    while (i < len && (text[i] == ' ' || text[i] == '\t' ||
                       text[i] == '\r' || text[i] == '\n'))
        i++;
    return i < len && text[i] == '<';
}

static bool agent_edit_old_is_only_space(const char *text, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (text[i] != ' ' && text[i] != '\t' &&
            text[i] != '\r' && text[i] != '\n')
            return false;
    }
    return true;
}

static bool agent_span_has_nonspace(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)s[i])) return true;
    }
    return false;
}

static bool agent_edit_old_prefix_mature_for_upto(const char *old, size_t old_len) {
    if (old_len < AGENT_EDIT_UPTO_MIN_PREFIX_BYTES) return false;
    int nonempty_lines = 0;
    bool line_has_text = false;
    for (size_t i = 0; i < old_len; i++) {
        if (old[i] == '\n') {
            if (line_has_text) nonempty_lines++;
            line_has_text = false;
        } else if (!isspace((unsigned char)old[i])) {
            line_has_text = true;
        }
    }
    return nonempty_lines >= AGENT_EDIT_UPTO_MIN_PREFIX_LINES;
}

static bool agent_edit_old_ready_for_upto(const char *old, size_t old_len) {
    if (!old_len || strstr(old, "[upto]")) return false;
    if (!agent_edit_old_prefix_mature_for_upto(old, old_len)) return false;
    size_t end = old_len;
    while (end > 0 && (old[end - 1] == ' ' || old[end - 1] == '\t' ||
                       old[end - 1] == '\r'))
        end--;
    return end > 0 && old[end - 1] == '\n';
}

/* While the model streams an edit old=... argument, stop it from retyping a
 * large exact old block once the emitted prefix is already a unique file
 * anchor.  The next sampled token is inspected before eval: if it would keep
 * writing old text rather than close the parameter, the caller evaluates a
 * complete "[upto]" marker line instead. */
static bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_qwen_tool_parser *p,
                                                  const char *next_text,
                                                  size_t next_len) {
    if (!forcer || !p) return false;
    bool in_edit_old = p->state == AGENT_QWEN_TOOL_PARAM_VALUE &&
        p->current.name && strcmp(p->current.name, "edit") == 0 &&
        p->param_name && strcmp(p->param_name, "old") == 0;
    if (!in_edit_old) {
        forcer->active = false;
        forcer->done = false;
        return false;
    }
    if (!forcer->active) {
        forcer->active = true;
        forcer->done = false;
    }
    if (forcer->done)
        return false;
    if (agent_edit_old_may_be_closing_tag(next_text, next_len) ||
        agent_edit_old_is_only_space(next_text, next_len))
        return false;

    const char *path = agent_tool_arg_value(&p->current, "path");
    if (!path || !path[0] || p->param_value_start > p->raw_len) return false;

    const char *old = p->raw + p->param_value_start;
    size_t old_len = p->raw_len - p->param_value_start;
    if (!agent_edit_old_ready_for_upto(old, old_len)) return false;

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0)
        return false;

    const char *match = NULL;
    bool unique = agent_find_unique(data, len, old, old_len, &match,
                                    "old prefix", err, sizeof(err));
    free(data);
    if (unique) {
        forcer->done = true;
        return true;
    }
    return false;
}

