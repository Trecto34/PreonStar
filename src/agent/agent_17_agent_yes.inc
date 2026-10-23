static void editor_flush_cpr_candidate(agent_editor *ed) {
    if (!ed->cpr_len) return;
    linenoiseEditQueueInput(&ed->edit, ed->cpr_buf, ed->cpr_len);
    ed->cpr_len = 0;
}

static bool agent_tail_ends_with(const char *tail, size_t tail_len,
                                 const char *seq, size_t seq_len) {
    return tail_len >= seq_len &&
           memcmp(tail + tail_len - seq_len, seq, seq_len) == 0;
}

static bool agent_tail_has_seq_prefix(const char *tail, size_t tail_len,
                                      const char *seq, size_t seq_len) {
    size_t max = tail_len < seq_len - 1 ? tail_len : seq_len - 1;
    for (size_t n = max; n > 0; n--) {
        if (memcmp(tail + tail_len - n, seq, n) == 0) return true;
    }
    return false;
}

/* Track bracketed paste markers outside linenoise.  The nonblocking event loop
 * may receive a paste in chunks; pausing linenoiseEditFeed() until ESC[201~
 * arrives prevents pasted newlines from being interpreted as Enter. */
static void editor_track_bracketed_paste(agent_editor *ed, char c) {
    static const char start[] = "\x1b[200~";
    static const char end[] = "\x1b[201~";

    if (ed->paste_tail_len == sizeof(ed->paste_tail)) {
        memmove(ed->paste_tail, ed->paste_tail + 1, sizeof(ed->paste_tail) - 1);
        ed->paste_tail_len--;
    }
    ed->paste_tail[ed->paste_tail_len++] = c;

    /* The blocking linenoise() path waits inside linenoiseEditPaste() until it
     * sees ESC[201~. In the agent the outer event loop reads stdin in
     * non-blocking chunks; if we let linenoise start parsing ESC[200~ before
     * the closing marker has arrived, pasted newlines can be interpreted as
     * Enter. Keep feeding bytes into linenoise's queue, but don't call
     * linenoiseEditFeed() while the terminal paste envelope is still open. */
    if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                             start, sizeof(start) - 1))
    {
        ed->paste_open = true;
        ed->paste_start_pending = false;
    } else if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                                    end, sizeof(end) - 1))
    {
        ed->paste_open = false;
        ed->paste_start_pending = false;
    } else {
        ed->paste_start_pending =
            !ed->paste_open &&
            agent_tail_has_seq_prefix(ed->paste_tail, ed->paste_tail_len,
                                      start, sizeof(start) - 1);
    }
}

/* Separate late CPR replies from real user input before handing bytes to
 * linenoise. */
static void editor_filter_input_byte(agent_editor *ed, char c) {
    if (ed->cpr_len || (unsigned char)c == 0x1b) {
        if (ed->cpr_len == sizeof(ed->cpr_buf)) {
            editor_flush_cpr_candidate(ed);
        }
        ed->cpr_buf[ed->cpr_len++] = c;
        cpr_state st = cpr_candidate_state(ed->cpr_buf, ed->cpr_len);
        if (st == CPR_COMPLETE) {
            ed->cpr_len = 0; /* Late terminal cursor report: discard it. */
        } else if (st == CPR_INVALID) {
            editor_flush_cpr_candidate(ed);
        }
        return;
    }
    linenoiseEditQueueInput(&ed->edit, &c, 1);
}

/* Queue raw terminal bytes into linenoise while preserving paste envelopes and
 * filtering cursor-position replies. */
static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        editor_track_bracketed_paste(ed, buf[i]);
        editor_filter_input_byte(ed, buf[i]);
    }
}

/* Drain stdin in nonblocking mode.  The outer event loop decides when queued
 * bytes are fed to linenoiseEditFeed(). */
static void editor_read_stdin(agent_editor *ed) {
    char buf[256];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            editor_queue_bytes(ed, buf, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

static bool editor_take_queued_byte(agent_editor *ed, unsigned char byte) {
    struct linenoiseState *l = &ed->edit;
    for (size_t i = l->queued_input_pos; i < l->queued_input_len; i++) {
        if ((unsigned char)l->queued_input[i] != byte) continue;
        memmove(l->queued_input + i, l->queued_input + i + 1,
                l->queued_input_len - i - 1);
        l->queued_input_len--;
        if (l->queued_input_pos > l->queued_input_len)
            l->queued_input_pos = l->queued_input_len;
        return true;
    }
    return false;
}

static bool editor_take_bare_escape(agent_editor *ed) {
    if (ed->cpr_len == 1 && (unsigned char)ed->cpr_buf[0] == 0x1b) {
        ed->cpr_len = 0;
        return true;
    }
    return false;
}

static void editor_replace_input(agent_editor *ed, const char *text) {
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    if (text && text[0]) linenoiseEditInsert(&ed->edit, text, strlen(text));
}

/* Track streamed text with the same widths as linenoise. Partial UTF-8 and
 * escape sequences remain pending across worker-output chunks. */
static void editor_note_output(agent_editor *ed, const char *text, size_t len) {
    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (ed->output_escape) {
            if (ed->output_escape == 1) ed->output_escape = c == '[' ? 2 : 0;
            else if (c >= 0x40 && c <= 0x7e) ed->output_escape = 0;
            continue;
        }
        if (c == 0x1b && !ed->output_utf8_len) { ed->output_escape = 1; continue; }
        uint32_t cp = c;
        if (ed->output_utf8_len || c >= 0x80) {
            if (ed->output_utf8_len && (c & 0xc0) != 0x80) {
                ed->output_utf8_len = 0;
                ed->output_col = (ed->output_col + 1) % cols;
            }
            ed->output_utf8[ed->output_utf8_len++] = (char)c;
            size_t n = linenoiseUtf8Decode(ed->output_utf8, ed->output_utf8_len, &cp);
            if (!n) continue;
            ed->output_utf8_len = 0;
        }
        if (cp == '\n' || cp == '\r') {
            ed->output_col = 0;
            ed->output_pending_wrap = false;
            if (cp == '\n') ed->output_line_open = false;
            ed->output_zwj = ed->output_regional = false;
            ed->output_glyph_width = 0;
            continue;
        }
        if (cp == '\b') {
            if (ed->output_pending_wrap) ed->output_col = cols - 1;
            else if (ed->output_col > 0) ed->output_col--;
            ed->output_pending_wrap = false;
            continue;
        }
        if (cp == '\t') {
            int col = ed->output_pending_wrap ? cols - 1 : ed->output_col;
            ed->output_col = (col | 7) + 1;
            if (ed->output_col >= cols) ed->output_col = cols - 1;
            ed->output_pending_wrap = false;
            ed->output_line_open = true;
            continue;
        }
        int width = linenoiseCharacterWidth(cp);
        bool regional = cp >= 0x1f1e6 && cp <= 0x1f1ff;
        if (cp == 0x200d) {
            ed->output_zwj = true;
            continue;
        }
        if (cp == 0xfe0f && ed->output_glyph_width == 1) {
            width = 1;
            ed->output_glyph_width = 2;
        } else if (width) {
            if (ed->output_zwj || (regional && ed->output_regional)) {
                width = 0;
                ed->output_zwj = false;
                ed->output_regional = false;
            } else {
                ed->output_glyph_width = width;
                ed->output_regional = regional;
            }
        }
        if (width > 0) {
            if (ed->output_col + width > cols) ed->output_col = 0;
            ed->output_col = (ed->output_col + width) % cols;
            ed->output_pending_wrap = ed->output_col == 0;
            ed->output_line_open = true;
        }
    }
}

/* Normalize generated LF to CRLF for terminal output without changing the text
 * stored in the transcript. */
static void editor_write_terminal_text(const char *text, size_t len) {
    size_t start = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] != '\n') continue;
        if (i > start) write_all(STDOUT_FILENO, text + start, i - start);
        write_all(STDOUT_FILENO, "\r\n", 2);
        start = i + 1;
    }
    if (start < len) write_all(STDOUT_FILENO, text + start, len - start);
}

/* Locate a CPR reply inside a mixed stdin buffer.  Bytes before/after the reply
 * are user input and must be queued back into linenoise. */
static bool find_cpr_reply(const char *buf, size_t len, size_t *start, size_t *end,
                           int *row, int *col) {
    for (size_t i = 0; i + 5 < len; i++) {
        if ((unsigned char)buf[i] != 0x1b || buf[i + 1] != '[') continue;
        size_t p = i + 2;
        int r = 0, c = 0;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            r = r * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p++] != ';') continue;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            c = c * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p] != 'R') continue;
        *start = i;
        *end = p + 1;
        *row = r;
        *col = c;
        return true;
    }
    return false;
}

/* Ask the terminal for the cursor column after writing model output.  Any user
 * bytes read while waiting for the CPR reply are queued back into linenoise so
 * typing during generation is not lost. */
static bool editor_query_cursor(agent_editor *ed, int *col_out) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    char buf[512];
    size_t len = 0, start = 0, end = 0;
    int row = 0, col = 0;
    write_all(STDOUT_FILENO, "\x1b[6n", 4);

    for (int attempt = 0; attempt < 8; attempt++) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        int rc = poll(&pfd, 1, 5);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) continue;
        for (;;) {
            ssize_t n = read(STDIN_FILENO, buf + len, sizeof(buf) - len);
            if (n > 0) {
                len += (size_t)n;
                if (find_cpr_reply(buf, len, &start, &end, &row, &col)) {
                    if (start) editor_queue_bytes(ed, buf, start);
                    if (end < len) editor_queue_bytes(ed, buf + end, len - end);
                    (void)row;
                    *col_out = col;
                    return col > 0;
                }
                if (len == sizeof(buf)) break;
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            break;
        }
    }

    if (len) editor_queue_bytes(ed, buf, len);
    return false;
}

static void editor_move_to_output_cursor(agent_editor *ed) {
    char seq[64];
    write_all(STDOUT_FILENO, "\x1b[1A", 4);
    int n = snprintf(seq, sizeof(seq), "\x1b[%dG", ed->output_col + 1);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static bool editor_get_terminal_size(int *rows, int *cols) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0) return false;
    if (ws.ws_row < 1 || ws.ws_col < 1) return false;
    *rows = ws.ws_row;
    *cols = ws.ws_col;
    return true;
}

static void editor_csi_cursor(int row, int col) {
    char seq[64];
    int n = snprintf(seq, sizeof(seq), "\x1b[%d;%dH", row, col);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_save_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\0337", 2);
    ed->output_cursor_saved = true;
}

static void editor_restore_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->output_cursor_saved) {
        write_all(STDOUT_FILENO, "\0338", 2);
    } else {
        editor_csi_cursor(ed->output_bottom, 1);
    }
}

static void editor_move_to_prompt_row(agent_editor *ed) {
    if (!ed->scroll_region) return;
    editor_csi_cursor(ed->prompt_row, 1);
}

static void editor_move_to_prompt_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->edit.screen_cursor_row > 0 && ed->edit.screen_cursor_col > 0) {
        editor_csi_cursor(ed->edit.screen_cursor_row, ed->edit.screen_cursor_col);
    } else {
        editor_move_to_prompt_row(ed);
    }
}

static void editor_clear_row(int row) {
    editor_csi_cursor(row, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K", 5);
}

static void editor_clear_prompt_region(agent_editor *ed) {
    if (!ed->scroll_region) return;
    for (int row = ed->prompt_row; row <= ed->term_rows; row++)
        editor_clear_row(row);

    /* In scroll-region mode p36-agent owns the absolute prompt/status rows.
     * Clearing them directly is more reliable than asking linenoise to clean
     * relative to whatever cursor position the last worker/status transition
     * left behind.  Reset linenoise's render bookkeeping so the next show is a
     * pure write into the reserved rows. */
    ed->edit.oldrows = 0;
    ed->edit.oldstatusrows = 0;
    ed->edit.oldrpos = 1;
    ed->edit.oldpos = ed->edit.pos;
}

static void editor_set_scroll_margin(int bottom) {
    char seq[96];
    int n = snprintf(seq, sizeof(seq), "\x1b[1;%dr", bottom);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_scroll_output_up(int bottom, int lines) {
    if (lines <= 0) return;
    editor_set_scroll_margin(bottom);
    editor_csi_cursor(bottom, 1);
    for (int i = 0; i < lines; i++)
        write_all(STDOUT_FILENO, "\n", 1);
}

static bool editor_set_scroll_layout(agent_editor *ed, int reserved_rows,
                                     bool allow_shrink,
                                     bool scroll_on_grow) {
    if (!ed->scroll_region) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;
    if (reserved_rows < 2) reserved_rows = 2;
    if (reserved_rows > rows - 2) reserved_rows = rows - 2;
    if (!allow_shrink && ed->reserved_rows > 0 &&
        ed->term_rows == rows && ed->term_cols == cols &&
        reserved_rows < ed->reserved_rows)
    {
        reserved_rows = ed->reserved_rows;
    }

    int output_bottom = rows - reserved_rows;
    int prompt_row = output_bottom + 1;
    bool changed = ed->term_rows != rows ||
                   ed->term_cols != cols ||
                   ed->output_bottom != output_bottom ||
                   ed->prompt_row != prompt_row ||
                   ed->reserved_rows != reserved_rows;
    if (!changed) return true;

    /* If the prompt grows, rows that were output rows become prompt rows.  Do
     * not simply clear them: first scroll the old output region upward by the
     * number of newly reserved rows, exactly as if the model had printed more
     * lines.  If the prompt shrinks, no output is restored; the output region
     * simply grows downward and the prompt/status block remains bottom
     * anchored. */
    bool scrolled_output = false;
    if (scroll_on_grow &&
        ed->term_rows == rows && ed->term_cols == cols &&
        ed->output_bottom > 0 && output_bottom < ed->output_bottom)
    {
        editor_scroll_output_up(ed->output_bottom,
                                ed->output_bottom - output_bottom);
        scrolled_output = true;
    }

    editor_set_scroll_margin(output_bottom);

    ed->term_rows = rows;
    ed->term_cols = cols;
    ed->output_bottom = output_bottom;
    ed->prompt_row = prompt_row;
    ed->reserved_rows = reserved_rows;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = scrolled_output;

    for (int row = prompt_row; row <= rows; row++)
        editor_clear_row(row);

    /* If the prompt grew while generated output was in the middle of a line,
     * the scroll above moved that partial line up with its column intact.
     * Preserve that column when saving the new output cursor; otherwise the
     * next token resumes at column 1 and overwrites the line it was extending. */
    int output_col = ed->output_line_open ? ed->output_col + 1 : 1;
    if (output_col < 1) output_col = 1;
    if (output_col > cols) output_col = cols;
    editor_csi_cursor(output_bottom, output_col);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static int editor_linenoise_layout_changed(struct linenoiseState *l,
                                           size_t prompt_rows,
                                           size_t status_rows,
                                           void *privdata) {
    (void)l;
    agent_editor *ed = privdata;
    if (!ed || !ed->scroll_region) return 0;
    if (prompt_rows < 1) prompt_rows = 1;
    int reserved = (int)(prompt_rows + status_rows);
    if (!editor_set_scroll_layout(ed, reserved, true, true)) return 0;
    return ed->prompt_row;
}

/* Keep generated output inside a scroll region that excludes the live prompt
 * and status footer.  This lets terminals scroll model/tool output naturally
 * without rewriting the prompt on every streamed token, which is especially
 * important over SSH where full redraws are visibly expensive. */
static bool editor_configure_scroll_region(agent_editor *ed) {
    if (ed->scroll_region) return true;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;

    ed->term_rows = 0;
    ed->term_cols = 0;
    ed->output_bottom = 0;
    ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = false;
    ed->scroll_region = true;
    if (!editor_set_scroll_layout(ed, 2, true, false)) return false;

    /* The agent prints backend startup lines before the editor exists.  Once
     * the scroll region is installed, create an append line at the bottom of
     * that region instead of guessing that the old terminal cursor was already
     * there.  Without this first scroll, the first agent/model output can
     * overwrite the last visible startup line. */
    editor_scroll_output_up(ed->output_bottom, 1);
    ed->output_cursor_saved = false;
    editor_csi_cursor(ed->output_bottom, 1);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static void editor_restore_terminal_layout(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    write_all(STDOUT_FILENO, "\x1b[r", 3);
    editor_csi_cursor(ed->term_rows, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K\r\n", 7);
    ed->scroll_region = false;
    ed->output_cursor_saved = false;
    ed->term_rows = ed->term_cols = 0;
    ed->output_bottom = ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_at_scroll_boundary = false;
}

/* Start linenoise in nonblocking mode and install the status footer. */
static int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial) {
    char *input = xmalloc(AGENT_INPUT_INITIAL_BUFLEN);
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool had_scroll_region = ed->scroll_region;
    bool use_scroll_region = editor_configure_scroll_region(ed);
    if (use_scroll_region) {
        if (had_scroll_region)
            editor_set_scroll_layout(ed, 2, true, false);
        editor_move_to_prompt_row(ed);
    }
    if (linenoiseEditStart(&ed->edit, STDIN_FILENO, STDOUT_FILENO,
                           input, AGENT_INPUT_INITIAL_BUFLEN, ed->prompt) != 0)
    {
        editor_restore_terminal_layout(ed);
        free(input);
        return -1;
    }
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
    linenoiseEditSetLayoutCallback(&ed->edit, editor_linenoise_layout_changed, ed);
    if (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")) {
        linenoiseHide(&ed->edit);
        linenoiseShow(&ed->edit);
    }
    ed->input = input;
    ed->edit.buflen_max = AGENT_INPUT_MAX_BUFLEN;
    ed->active = true;
    if (set_nonblock(STDIN_FILENO, true, &ed->old_stdin_flags) != 0)
        ed->old_stdin_flags = -1;
    if (initial && initial[0]) linenoiseEditInsert(&ed->edit, initial, strlen(initial));
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->output_pending_wrap = false;
    ed->output_utf8_len = 0;
    ed->output_escape = 0;
    ed->output_zwj = ed->output_regional = false;
    ed->output_glyph_width = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
    return 0;
}

/* Stop the live editor and restore stdin flags. */
static void editor_stop(agent_editor *ed) {
    if (!ed->active) return;
    if (ed->deferred_output.len)
        editor_write_scroll_output_preserve_prompt(ed, NULL, 0, true);
    /* p36-agent treats linenoise as a live input widget, not as persistent
     * command scrollback.  Clear it before shutdown so submitting a line and
     * immediately reopening the editor does not leave the accepted
     * prompt+input duplicated above the fresh prompt. */
    if (!ed->hidden && (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")))
        editor_hide(ed);
    linenoiseEditStop(&ed->edit);
    if (ed->old_stdin_flags >= 0) fcntl(STDIN_FILENO, F_SETFL, ed->old_stdin_flags);
    free(ed->edit.buf);
    ed->input = NULL;
    ed->active = false;
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->output_pending_wrap = false;
    ed->output_utf8_len = 0;
    ed->output_escape = 0;
    ed->output_zwj = ed->output_regional = false;
    ed->output_glyph_width = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
}

/* Hide the live prompt before model output is written.  In scroll-region mode
 * the output cursor was saved before the prompt was drawn, so restoring it is
 * enough to append more model/tool bytes without touching the prompt rows. */
static void editor_hide(agent_editor *ed) {
    if (!ed->active || ed->hidden) return;
    if (ed->scroll_region) {
        editor_clear_prompt_region(ed);
        editor_restore_output_cursor(ed);
        ed->hidden = true;
        return;
    }
    linenoiseHide(&ed->edit);
    if (ed->prompt_below_output) {
        editor_move_to_output_cursor(ed);
        ed->prompt_below_output = false;
    }
    ed->hidden = true;
}

/* Restore the live prompt after output.  The primary path draws it in the
 * reserved bottom rows; the fallback path keeps the older one-row-below-output
 * trick for terminals where scroll regions are unavailable. */
static void editor_show(agent_editor *ed) {
    if (!ed->active || !ed->hidden) return;
    if (ed->scroll_region) {
        editor_save_output_cursor(ed);
        editor_move_to_prompt_row(ed);
        write_all(STDOUT_FILENO, "\x1b[0m", 4);
        linenoiseShow(&ed->edit);
        ed->hidden = false;
        return;
    }
    if (ed->output_line_open) {
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->prompt_below_output = true;
    } else {
        ed->prompt_below_output = false;
    }
    /* Model/tool output can leave SGR attributes active while it streams.
     * Redrawing linenoise always starts from normal attributes; tool rendering
     * re-emits its own color on the next streamed byte if it is still inside a
     * colored parameter. */
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    ed->hidden = false;
}

static void editor_update_prompt(agent_editor *ed, const char *prompt) {
    if (strcmp(ed->prompt, prompt)) ed->prompt_dirty = true;
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    ed->edit.prompt = ed->prompt;
    ed->edit.plen = strlen(ed->prompt);
}

static void editor_update_status(agent_editor *ed, const char *status) {
    if (strcmp(ed->status, status ? status : "")) ed->status_dirty = true;
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
}

static void editor_redraw_visible_prompt(agent_editor *ed) {
    if (!ed->active || !ed->scroll_region) return;
    linenoiseBeginUpdate(STDOUT_FILENO);
    editor_clear_prompt_region(ed);
    editor_move_to_prompt_row(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    linenoiseEndUpdate();
}

static bool editor_prompt_redraw_due(agent_editor *ed) {
    double now = now_sec();
    if (ed->last_prompt_redraw_time <= 0.0 ||
        now - ed->last_prompt_redraw_time >= AGENT_STATUS_REDRAW_INTERVAL_SEC)
    {
        return true;
    }
    return false;
}

static void editor_flush_prompt_status(agent_editor *ed, bool force) {
    if (!ed->active || ed->hidden) return;
    int rows, cols;
    if (ed->scroll_region && editor_get_terminal_size(&rows, &cols) &&
        (rows != ed->term_rows || cols != ed->term_cols)) {
        ed->edit.cols = cols;
        ed->prompt_dirty = true;
        force = true;
    }
    if (!ed->prompt_dirty && !ed->status_dirty) return;
    if (!force && !editor_prompt_redraw_due(ed)) return;
    linenoiseBeginUpdate(STDOUT_FILENO);
    if (ed->scroll_region) {
        if (ed->prompt_dirty || !linenoiseRefreshStatus(&ed->edit))
            editor_redraw_visible_prompt(ed);
    } else {
        editor_hide(ed);
        editor_show(ed);
    }
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    linenoiseEndUpdate();
}

static void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status) {
    if (strcmp(ed->prompt, prompt)) editor_update_prompt(ed, prompt);
    if (strcmp(ed->status, status ? status : "")) editor_update_status(ed, status);
    editor_flush_prompt_status(ed, false);
}

static bool editor_write_scroll_output_preserve_prompt(agent_editor *ed,
                                                       const char *text,
                                                       size_t len,
                                                       bool settle_boundary) {
    agent_input_buf_append(&ed->deferred_output, text, len);
    if (!ed->deferred_output.len) return false;

    agent_editor predicted = *ed;
    editor_note_output(&predicted, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if ((predicted.output_utf8_len || predicted.output_escape) && settle_boundary) {
        if (predicted.output_utf8_len) {
            ed->deferred_output.len -= predicted.output_utf8_len;
            agent_input_buf_append(&ed->deferred_output, "\xef\xbf\xbd", 3);
        } else {
            while (ed->deferred_output.len &&
                   ed->deferred_output.ptr[--ed->deferred_output.len] != 0x1b) {}
        }
        predicted = *ed;
        editor_note_output(&predicted, ed->deferred_output.ptr, ed->deferred_output.len);
    }
    bool at_boundary = predicted.output_pending_wrap;
    if (predicted.output_utf8_len || predicted.output_escape) return false;
    /* DECSC/DECRC does not reliably preserve pending auto-wrap.  Keep a chunk
     * that ends at the margin until the next content can trigger the wrap in
     * the same terminal write. */
    if (at_boundary && !settle_boundary) return false;

    linenoiseBeginUpdate(STDOUT_FILENO);
    editor_restore_output_cursor(ed);
    editor_write_terminal_text(ed->deferred_output.ptr,
                               ed->deferred_output.len);
    editor_note_output(ed, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if (at_boundary) {
        /* ESC 7/8 does not preserve a terminal's pending auto-wrap flag.  At
         * the end of a turn, settle it explicitly before saving the cursor. */
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->output_col = 0;
        ed->output_line_open = false;
        ed->output_pending_wrap = false;
    }
    agent_input_buf_free(&ed->deferred_output);
    editor_save_output_cursor(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    editor_move_to_prompt_cursor(ed);
    linenoiseEndUpdate();
    ed->output_at_scroll_boundary = true;
    return true;
}

/* Serialize async model/tool output with linenoise.  This is the central
 * terminal contract.  In scroll-region mode the live prompt stays painted:
 * output is appended in the upper scroll area, then the cursor is returned to
 * linenoise's remembered prompt position.  The fallback path still hides and
 * redraws because it has no protected prompt rows. */
static void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show) {
    if (ed->scroll_region && ed->active && !ed->hidden &&
        (len || ed->deferred_output.len)) {
        editor_flush_prompt_status(ed, false);
        bool prompt_changed = strcmp(ed->prompt, prompt) != 0;
        bool status_changed = strcmp(ed->status, status ? status : "") != 0;

        linenoiseBeginUpdate(STDOUT_FILENO);
        editor_write_scroll_output_preserve_prompt(ed, text, len, force_show);
        if (prompt_changed) editor_update_prompt(ed, prompt);
        if (status_changed) editor_update_status(ed, status);
        editor_flush_prompt_status(ed, force_show);
        linenoiseEndUpdate();
        return;
    }

    editor_hide(ed);
    if (len) {
        editor_write_terminal_text(text, len);
        if (ed->scroll_region) ed->output_at_scroll_boundary = true;
        if (!ed->scroll_region) {
            if (text[len - 1] == '\n' || text[len - 1] == '\r') {
                ed->output_col = 0;
                ed->output_line_open = false;
            } else {
                int col = 0;
                if (editor_query_cursor(ed, &col)) {
                    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
                    ed->output_col = col > 0 ? col - 1 : 0;
                    ed->output_line_open = true;
                    if (ed->output_col + 1 >= cols) {
                        write_all(STDOUT_FILENO, "\r\n", 2);
                        ed->output_col = 0;
                    }
                } else {
                    editor_note_output(ed, text, len);
                }
            }
        }
    }
    if (ed->active) {
        editor_update_prompt(ed, prompt);
        editor_update_status(ed, status);
        /* In scroll-region mode this saves the current output cursor and
         * redraws linenoise in the fixed prompt rows.  In fallback mode it may
         * put the prompt below an unfinished generated line. */
        if (force_show || len) editor_show(ed);
    }
}

/* Ctrl+C while idle is an edit-cancel key, not an exit key.  Clear the real
 * linenoise buffer so stale text cannot be submitted later, then leave a short
 * visible hint about the explicit EOF exit path. */
static void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status) {
    if (!ed->active) return;
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    const char *msg = stdout_is_tty() ?
        "\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n" :
        "press Ctrl+D to exit\n";
    editor_write_async(ed, msg, strlen(msg), prompt, status, true);
}

static void runtime_help(void) {
    puts("Commands:");
    puts("  /help        Show this help.");
    puts("  /save        Save the current session.");
    puts("  /compact     Compact the current session context now.");
    puts("  /list        List saved sessions.");
    puts("  /switch SHA  Load a saved session and show recent history.");
    puts("  /del SHA     Delete a saved session.");
    puts("  /strip SHA   Strip KV payload; /switch rebuilds it by prefill.");
    puts("  /history [N] Show N recent user turns from the current session.");
    puts("  /power N     Set GPU duty cycle percentage, 1..100.");
    puts("  /hints on|off Enable or disable brief programming hints; starts off.");
    puts("  /steer [F]   Show or set FFN steering for subsequent tokens.");
    puts("  /new         Start a fresh session from the system prompt.");
    puts("  /quit, /exit Exit.");
    puts("  Ctrl+C       Interrupt generation; clear edited text.");
    puts("  Tab          Cycle thinking mode at an empty idle prompt; keep budget.");
    puts("  Enter        Queue text while the agent is busy.");
    puts("  Ctrl+X       Edit the first queued prompt.");
    puts("  ESC          Interrupt and send queued prompt immediately.");
    puts("  Ctrl+D       Exit from an empty prompt.");
}

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len) {
    if (ctx_size >= 1000) {
        if (ctx_size % 1000 == 0) snprintf(buf, len, "%dk", ctx_size / 1000);
        else snprintf(buf, len, "%.1fk", (double)ctx_size / 1000.0);
    } else {
        snprintf(buf, len, "%d", ctx_size);
    }
}

static void agent_format_welcome_banner(const agent_config *cfg,
                                        char *buf, size_t len) {
    char ctx[32];
    agent_format_ctx_size(cfg->gen.ctx_size, ctx, sizeof(ctx));
    if (stdout_is_tty()) {
        snprintf(buf, len,
                 "\x1b[1;97mQuark\x1b[1;95mStar\x1b[0m 🔮 Agent, context %s tokens\n\n",
                 ctx);
    } else {
        snprintf(buf, len, "QuarkStar Agent, context %s tokens\n\n", ctx);
    }
}

static void editor_write_welcome_banner(agent_editor *editor,
                                        const agent_config *cfg,
                                        const char *prompt,
                                        const char *statusline) {
    char banner[256];
    agent_format_welcome_banner(cfg, banner, sizeof(banner));
    editor_write_async(editor, banner, strlen(banner), prompt, statusline, true);
}

/* Initialize the worker, cache directory, sysprompt checkpoint path, trace file,
 * and model thread.  After this returns, all P36 session mutation happens on
 * the worker thread. */
static int agent_worker_init(agent_worker *w, p36_engine *engine, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->engine = engine;
    w->cfg = cfg;
    w->wake_fd[0] = -1;
    w->wake_fd[1] = -1;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    w->status.state = AGENT_WORKER_IDLE;
    w->active = true;
    w->session_t0 = now_sec();
    if (pipe(w->wake_fd) != 0) return -1;
    int old_flags;
    set_nonblock(w->wake_fd[0], true, &old_flags);
    set_nonblock(w->wake_fd[1], true, &old_flags);
    if (!cfg->server_url &&
        p36_session_create(&w->session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "p36-agent: session backend is required\n");
        return -1;
    }
    w->cache_dir = agent_default_cache_dir();
    if (!agent_mkdir_p(w->cache_dir)) {
        fprintf(stderr, "p36-agent: failed to create %s: %s\n",
                w->cache_dir, strerror(errno));
        return -1;
    }
    p36_web_config web_cfg = {
        .home_dir = getenv("HOME"),
        .port = 9333,
        .confirm = agent_web_confirm,
        .confirm_privdata = w,
        .log = agent_web_log,
        .log_privdata = w,
        .cancel = agent_web_cancel,
        .cancel_privdata = w,
    };
    w->web = p36_web_create(&web_cfg);
    w->sysprompt_path = p36_kvstore_path_join(w->cache_dir, "sysprompt.kv");
    if (cfg->gen.trace_path && cfg->gen.trace_path[0]) {
        w->trace = fopen(cfg->gen.trace_path, "ab");
        if (!w->trace) {
            fprintf(stderr, "p36-agent: failed to open trace %s: %s\n",
                    cfg->gen.trace_path, strerror(errno));
            return -1;
        }
    }
    if (pthread_create(&w->thread, NULL, worker_main, w) != 0) return -1;
    return 0;
}

/* Shut down the worker and release owned resources, including any live bash
 * process groups. */
static void agent_worker_free(agent_worker *w) {
    worker_stop(w);
    if (!w->cfg->non_interactive && w->m.steps) {
        char *sum = agent_metrics_summary(w);
        fputs(sum, stderr);
        free(sum);
    }
    if (w->thread) pthread_join(w->thread, NULL);
    agent_bash_jobs_free(w);
    p36_web_free(w->web);
    if (w->session) p36_session_free(w->session);
    for (int i = 0; i < w->nrmsg; i++) free(w->rmsg[i]);
    free(w->rmsg);
    p36_tokens_free(&w->transcript);
    agent_worker_images_clear(w);
    free(w->cache_dir);
    free(w->sysprompt_path);
    free(w->session_title);
    free(w->legacy_session_path_to_delete);
    free(w->queued_user_drain_text);
    if (w->wake_fd[0] >= 0) close(w->wake_fd[0]);
    if (w->wake_fd[1] >= 0) close(w->wake_fd[1]);
    if (w->trace) fclose(w->trace);
    free(w->cmd_text);
    free(w->out);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}

typedef enum {
    AGENT_YES_NO_AUTO_NONE,
    AGENT_YES_NO_AUTO_NO,
    AGENT_YES_NO_AUTO_YES,
} agent_yes_no_auto;

typedef struct {
    int timeout_sec;
    agent_yes_no_auto timeout_answer;
} agent_yes_no_options;

