static bool agent_tool_response_is_error(const char *content) {
    char head[121];
    content = content ? content : "";
    size_t len = strlen(content);
    size_t head_len = len < sizeof(head) - 1 ? len : sizeof(head) - 1;
    for (size_t i = 0; i < head_len; i++)
        head[i] = (char)tolower((unsigned char)content[i]);
    head[head_len] = '\0';

    bool code = agent_ascii_contains_ci(content, "throw new ") ||
                agent_ascii_contains_ci(content, "throw error") ||
                agent_ascii_contains_ci(content, "console.error") ||
                agent_ascii_contains_ci(content, "logger.error") ||
                agent_ascii_contains_ci(content, "logging.error") ||
                strstr(head, "import ") || strstr(head, "def ") ||
                strstr(head, "function ");
    if (code) return false;

    bool exit_zero = strstr(head, "exit code: 0") ||
                     strstr(head, "process exited with code 0");
    bool error_ok = strstr(head, "\"error\": null") ||
                    strstr(head, "\"error\":null") ||
                    strstr(head, "\"error\": false") ||
                    strstr(head, "\"error\":false") ||
                    strstr(head, "\"error\": \"\"") ||
                    strstr(head, "\"error\":\"\"");
    bool strong = (strstr(head, "\"error\":") && !error_ok) ||
                  strstr(head, "\"status\": \"error\"") ||
                  strstr(head, "\"status\":\"error\"") ||
                  strstr(head, "traceback (most recent call last):") ||
                  strstr(head, "command not found") ||
                  strstr(head, "invalid syntax") || strstr(head, "fatal:") ||
                  ((strstr(head, "exit code: ") ||
                    strstr(head, "process exited with code")) && !exit_zero) ||
                  !strncmp(head, "exception:", 10) ||
                  !strncmp(head, "failed to ", 10);
    bool weak = strstr(head, "error:") || strstr(head, "err!");
    bool weak_suppressed = strstr(head, "$ ") || strstr(head, "took ") || len >= 600;
    return strong || (weak && !weak_suppressed);
}

/* ============================================================================
 * System Prompt Rendering And Worker Output Queues
 * ============================================================================
 */

#define AGENT_TOOL_CONTRACTS \
    "Read output is limited to 16 KiB and 4 KiB per line; longer lines are omitted as blobs, so use search or targeted ranges for them. Use more to continue; " \
    "whole=true fails rather than returning an excerpt. raw=true omits line numbers but still reports truncation.\n" \
    "Search mode is literal (default) or regex (POSIX extended). Defaults: path=., case_sensitive=true, context=0, max_results=50. " \
    "context is 0-5, max_results is 1-500. The search skips .git and nested symlinks, and reports incomplete coverage.\n" \
    "bash timeout_sec defaults to 3600; refresh_sec defaults to 60 and waits up to that many seconds. " \
    "bash_status returns immediately unless refresh_sec is given. Jobs keep running and timeouts remain active between calls.\n" \
    "write content is limited to 12 KiB per call: write the first part, then call write with append=true for each next part. " \
    "To edit by line numbers (safe for whitespace), call edit with start_line, end_line and new, without old.\n\n"

static const char agent_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "# Tools\n\n"
    "You have access to the following functions. If you choose to call a function ONLY reply "
    "in the following format with NO suffix:\n\n"
    "<think>\n"
    "Brief explanation of tool call\n"
    "</think>\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_name>\n"
    "value\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- You can use the <think></think> block to plan your next tool call OR to synthesize data "
    "and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n"
    "- Function calls MUST use an inner <function=...></function> block nested within "
    "<tool_call></tool_call> XML tags. Required parameters MUST be specified.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
    "thinking, with NO conversational text before it.\n"
    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with "
    "NO spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user without "
    "any tool call.\n"
    "</IMPORTANT>\n\n"
    "Read defaults to a bounded chunk: path alone returns the first 500 lines, not the whole file. "
    "If read says more lines are available, call more with count=<lines> to read the next chunk; "
    "more defaults to the next 500 lines. "
    "Use more for exact continuation; a byte-limited chunk can end within a line. "
    "If the user explicitly asks you to read a complete file into context, call read with whole=true. "
    "A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n"
    AGENT_TOOL_CONTRACTS;

static const char agent_tools_prompt_edit_exact[] =
    "## Editing files\n\n"
    "Use edit with path, old, and new for changes. Always put the edited file path first. "
    "The old text must match exactly once in the current file; otherwise edit fails for safety. "
    "Read enough of the file to provide the exact old text being replaced.\n"
    "To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n"
    "Use read raw=true only when you need plain file text without line numbers.\n\n";

static const char agent_tools_prompt_edit_upto[] =
    "## Editing files\n\n"
    "Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. "
    "For edit, always put the edited file path as the first parameter. "
    "The old text must match exactly once in the current file; otherwise edit fails for safety.\n"
    "For large replacements, prefer anchored old text: write the first lines, then [upto], then the final lines. "
    "The tool replaces everything from the head through the tail. If the head or tail is ambiguous, the edit fails.\n"
    "After [upto], always write unique final lines before closing old; never close old immediately after [upto].\n"
    "Do not use a generic tail anchor like:\n"
    "- BigNum bignum_add(BigNum *a, BigNum *b) {\n"
    "- [upto]\n"
    "- }\n"
    "because the closing brace may match many functions. Instead include final lines that are unique near that function, "
    "for example its last calculation and return line before the brace.\n"
    "Example anchored edit:\n"
    "<tool_call>\n"
    "<function=edit>\n"
    "<parameter=path>\n/tmp/example.c\n</parameter>\n"
    "<parameter=old>\nstatic int parse(void) {\n"
    "    int ok = 0;\n"
    "[upto]\n"
    "    return ok;\n"
    "}\n</parameter>\n"
    "<parameter=new>\nstatic int parse(void) {\n"
    "    return parse_impl();\n"
    "}\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n"
    "Use read raw=true only when you need plain file text without line numbers.\n\n";

static const char agent_tools_prompt_after_edit[] =
    "For long-running bash commands, pass refresh_sec. If a bash job is still running, use "
    "bash_status to check it early or bash_stop to terminate it.\n\n"
    "Run sudo normally when needed. In interactive mode, terminal password requests open a private "
    "password prompt for the user. Never ask for passwords in chat or put them in tool arguments.\n\n"
    "Use google_search to find web pages. Use visit_page to read a known URL with a visible browser. "
    "The first web call may ask the user for permission to start a visible browser.\n\n"
    "<tools>\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"google_search\",\n"
    "    \"description\": \"Search Google in a visible browser and return compact Markdown links.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"visit_page\",\n"
    "    \"description\": \"Open a URL in a visible browser and return rendered page Markdown.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"url\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"url\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash\",\n"
    "    \"description\": \"Run a shell command.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"command\": {\"type\": \"string\"},\n"
    "        \"timeout_sec\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"command\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_status\",\n"
    "    \"description\": \"Report current status and recent output for a bash job.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"integer\"},\n"
    "        \"pid\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_stop\",\n"
    "    \"description\": \"Terminate a running bash job and report its final output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"integer\"},\n"
    "        \"pid\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"read\",\n"
    "    \"description\": \"Read a text file or a range of lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"start_line\": {\"type\": \"integer\"},\n"
    "        \"max_lines\": {\"type\": \"integer\"},\n"
    "        \"whole\": {\"type\": \"boolean\"},\n"
    "        \"raw\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"more\",\n"
    "    \"description\": \"Continue the previous read-like output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"count\": {\"type\": \"number\"}\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"write\",\n"
    "    \"description\": \"Create or overwrite a text file.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"content\": {\"type\": \"string\"},\n"
    "        \"append\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"content\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"edit\",\n"
    "    \"description\": \"Replace exactly one old text match.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"old\": {\"type\": \"string\"},\n"
    "        \"new\": {\"type\": \"string\"},\n"
    "        \"start_line\": {\"type\": \"integer\"},\n"
    "        \"end_line\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"new\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"search\",\n"
    "    \"description\": \"Search files and return compact edit-friendly matches.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"},\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"mode\": {\"type\": \"string\", \"enum\": [\"literal\", \"regex\"]},\n"
    "        \"glob\": {\"type\": \"string\"},\n"
    "        \"context\": {\"type\": \"integer\"},\n"
    "        \"max_results\": {\"type\": \"integer\"},\n"
    "        \"case_sensitive\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"list\",\n"
    "    \"description\": \"List one directory compactly.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n"
    "</tools>\n\n"
    "# Rules\n\n"
    "- Always use the exact Qwen3.6 tool-call syntax above.\n"
    "- This system runs on local inference of a few hundred tokens/s of prefill, "
    "and a few tens of tokens/s decoding speed. Use read/search to get the exact "
    "text you need, then use edit instead of rewriting whole files.\n"
    "- Write code that is reliable and works well; always have a mental model of "
    "what is going on in complex parts of the code.\n"
    "- Work in a way that preserves the current system configuration integrity, "
    "unless explicitly asked otherwise by the user.\n";

static const char agent_tools_prompt_vision[] =
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"view_image\",\n"
    "    \"description\": \"Open a local PNG or JPEG as a visual observation.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n";

static char *agent_build_tools_prompt(bool edit_upto, bool vision) {
    const char *intro = agent_tools_prompt_intro;
    const char *edit = edit_upto ? agent_tools_prompt_edit_upto : agent_tools_prompt_edit_exact;
    const char *tail = strstr(agent_tools_prompt_after_edit, "</tools>");
    size_t a = strlen(intro), b = strlen(edit);
    size_t c = (size_t)(tail - agent_tools_prompt_after_edit);
    size_t d = vision ? strlen(agent_tools_prompt_vision) : 0;
    char *out = xmalloc(a + b + c + d + strlen(tail) + 1);
    memcpy(out, intro, a);
    memcpy(out + a, edit, b);
    memcpy(out + a + b, agent_tools_prompt_after_edit, c);
    if (vision) memcpy(out + a + b + c, agent_tools_prompt_vision, d);
    strcpy(out + a + b + c + d, tail);
    return out;
}

static char *agent_build_system_prompt(const char *extra, bool *thinking,
                                       bool edit_upto, bool vision) {
    char *tools = agent_build_tools_prompt(edit_upto, vision);
    char *plain = agent_chat_control_text(extra, thinking);
    if (!plain[0]) {
        free(plain);
        return tools;
    }

    size_t a = strlen(tools);
    size_t b = strlen(plain);
    char *out = xmalloc(a + 2 + b + 1);
    memcpy(out, tools, a);
    memcpy(out + a, "\n\n", 2);
    memcpy(out + a + 2, plain, b + 1);
    free(tools);
    free(plain);
    return out;
}

static const char agent_qwen_tool_syntax_reminder[] =
    "Qwen3.6 tool syntax reminder:\n"
    "<tool_call>\n"
    "<function=$TOOL_NAME>\n"
    "<parameter=$PARAMETER_NAME>\n"
    "$PARAMETER_VALUE\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n";

static const char agent_json_tool_syntax_reminder[] =
    "Tool syntax reminder:\n"
    "<tool_call>\n"
    "{\"name\": \"$TOOL_NAME\", \"arguments\": {\"$PARAMETER_NAME\": \"$PARAMETER_VALUE\"}}\n"
    "</tool_call>\n";

#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000

static const char agent_hints_prompt[] =
    "The user enabled programming hints. Occasionally explain a useful concept "
    "behind the current work: a design choice, failure mechanism, trade-off, "
    "or way to verify correctness. Base each hint on code or results you have "
    "actually examined. Write one or two sentences as a normal Markdown "
    "blockquote starting with > **Hint:**. Hints are prose, not tool calls. "
    "Put them after the relevant discovery, including between tool calls. "
    "Prefer no hint to routine narration, repetition, or generic advice. "
    "Do not invent personal experience or a learner profile. Keep working "
    "without waiting for an answer. Later system notes may enable or disable "
    "hints; follow the latest setting.\n";

static bool agent_parse_hints(const char *arg, bool *enabled) {
    if (!strcmp(arg, "on")) *enabled = true;
    else if (!strcmp(arg, "off")) *enabled = false;
    else return false;
    return true;
}

/* The full instructions are sent once per live context, not on each enable. */
static const char *agent_hints_note(agent_hints *h) {
    const char *note;
    if (h->enabled && !h->instructed) {
        note = agent_hints_prompt;
        h->instructed = true;
    } else if (h->applied != AGENT_HINTS_UNSET &&
               h->applied != (h->enabled ? AGENT_HINTS_ON : AGENT_HINTS_OFF)) {
        note = h->enabled ?
            "The user enabled programming hints. Produce them from now on.\n" :
            "Programming hints are disabled. Do not produce teaching asides; "
            "continue the task normally.\n";
    } else {
        return NULL;
    }
    h->applied = h->enabled ? AGENT_HINTS_ON : AGENT_HINTS_OFF;
    return note;
}

static void agent_hints_compacted(agent_hints *h) {
    /* The retained tail may still contain an old on/off note. */
    if (h->applied != AGENT_HINTS_UNSET) h->applied = AGENT_HINTS_UNKNOWN;
    h->instructed = false;
}

static void agent_worker_append_hints_context(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    const char *note = agent_hints_note(&w->hints);
    pthread_mutex_unlock(&w->mu);
    if (!note) return;
    p36_chat_append_message(w->engine, &w->transcript, "system", note);
    agent_trace_text(w, "hints-context", note, strlen(note));
    pthread_mutex_lock(&w->mu);
    w->session_dirty = true;
    pthread_mutex_unlock(&w->mu);
}

static char *agent_build_system_prompt_reminder(bool edit_upto, bool vision) {
    char *tools = agent_build_tools_prompt(edit_upto, vision);
    const char *start = "\n\n[System prompt reminder follows.]\n";
    const char *end = "[End system prompt reminder.]\n\n";
    size_t len = strlen(start) + strlen(tools) + strlen(end) + 1;
    char *out = xmalloc(len);
    out[0] = '\0';
    strcat(out, start);
    strcat(out, tools);
    strcat(out, end);
    free(tools);
    return out;
}

static char *agent_collect_tool_schemas(bool vision) {
    const char *open_tag = strstr(agent_tools_prompt_after_edit, "<tools>\n");
    const char *close_tag = strstr(agent_tools_prompt_after_edit, "\n</tools>");
    if (!open_tag || !close_tag || close_tag <= open_tag + 8) return NULL;
    const char *body = open_tag + 8;
    size_t body_len = (size_t)(close_tag - body);
    size_t v_len = vision ? strlen(agent_tools_prompt_vision) : 0;
    char *buf = xmalloc(body_len + (v_len ? v_len + 2 : 0) + 1);
    memcpy(buf, body, body_len);
    size_t total = body_len;
    if (v_len) {
        memcpy(buf + total, "\n\n", 2);
        total += 2;
        memcpy(buf + total, agent_tools_prompt_vision, v_len);
        total += v_len;
    }
    buf[total] = '\0';
    return buf;
}

static void agent_append_system_prompt(p36_engine *engine, p36_tokens *tokens,
                                       const char *extra, bool *thinking,
                                       bool edit_upto, p36_think_mode think_mode) {
    const char *chat_tmpl = p36_engine_chat_template(engine);
    if (chat_tmpl) {
        char *plain = agent_chat_control_text(extra, thinking);
        char *tools_buf = agent_collect_tool_schemas(p36_engine_has_vision(engine));
        bool en_think = (!thinking || *thinking) && think_mode != P36_THINK_NONE;
        p36_think_mode eff_mode = (think_mode == P36_THINK_NONE && en_think) ?
                                  P36_THINK_HIGH : think_mode;
        char *rendered = p36_jinja_render_system_block(
            chat_tmpl, plain, tools_buf, eff_mode, en_think);
        free(tools_buf);
        free(plain);
        if (rendered) {
            if (rendered[0]) p36_tokenize_rendered_chat(engine, rendered, tokens);
            free(rendered);
            return;
        }
    }
    /* The built-in prompt is trusted P36 control text.  User supplied system
     * text remains ordinary chat content. */
    char *prompt = agent_build_system_prompt(extra, thinking, edit_upto,
                                             p36_engine_has_vision(engine));
    p36_chat_append_message(engine, tokens, "system", prompt);
    free(prompt);
}

static void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}

static void agent_worker_maybe_append_datetime_context(agent_worker *w) {
    if (w->datetime_context_injected) return;

    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);

    char when[128];
    if (strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", &tm) == 0)
        snprintf(when, sizeof(when), "%lld", (long long)now);

    char msg[256];
    snprintf(msg, sizeof(msg),
             "Current local date and time at session start: %s. "
             "Use this only when date or time matters.", when);
    p36_chat_append_message(w->engine, &w->transcript, "system", msg);
    agent_trace_text(w, "datetime-context", msg, strlen(msg));
    w->datetime_context_injected = true;
}

/* Refresh the trusted tool prompt in very long sessions. */
static void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w) {
    if (w->last_system_prompt_reminder_at <= 0) {
        agent_worker_note_system_prompt_seen(w);
        return;
    }
    if (w->transcript.len - w->last_system_prompt_reminder_at <
        AGENT_SYSTEM_PROMPT_REMINDER_TOKENS)
    {
        return;
    }

    char *reminder = agent_build_system_prompt_reminder(
        w->cfg->edit_upto,
        p36_engine_has_vision(w->engine));
    agent_publish_system_status(w, "Re-injecting system prompt reminder...");
    agent_trace(w, "system prompt reminder injected at transcript=%d",
                w->transcript.len);
    p36_chat_append_message(w->engine, &w->transcript, "system", reminder);
    free(reminder);

    const char *extra = w->cfg->gen.system;
    if (extra && extra[0]) {
        bool thinking = w->thinking_enabled;
        char *plain = agent_chat_control_text(extra, &thinking);
        if (plain[0])
            p36_chat_append_message(w->engine, &w->transcript, "system", plain);
        free(plain);
    }
    agent_worker_note_system_prompt_seen(w);
}

/* Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe. */
static void agent_wake_locked(agent_worker *w) {
    if (w->wake_pending) return;
    w->wake_pending = true;
    char c = 'x';
    ssize_t wr = write(w->wake_fd[1], &c, 1);
    (void)wr;
}

/* Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place. */
static void agent_publish(agent_worker *w, const char *s, size_t n) {
    if (!n) return;
    pthread_mutex_lock(&w->mu);
    if (w->out_len + n + 1 > w->out_cap) {
        size_t cap = w->out_cap ? w->out_cap * 2 : 4096;
        while (cap < w->out_len + n + 1) cap *= 2;
        char *p = realloc(w->out, cap);
        if (!p) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        w->out = p;
        w->out_cap = cap;
    }
    memcpy(w->out + w->out_len, s, n);
    w->out_len += n;
    w->out[w->out_len] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

