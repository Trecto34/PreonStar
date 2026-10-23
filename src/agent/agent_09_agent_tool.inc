static bool agent_edit_find_old_span(const char *data, size_t len,
                                     const char *old, bool allow_upto,
                                     const char **match,
                                     size_t *match_len, bool *anchored,
                                     char *err, size_t err_len) {
    static const char marker[] = "[upto]";
    size_t old_len = strlen(old);
    const char *upto = strstr(old, marker);
    if (!allow_upto || !upto) {
        *anchored = false;
        if (!agent_find_unique(data, len, old, old_len, match, "old text",
                               err, err_len))
            return false;
        *match_len = old_len;
        return true;
    }
    if (strstr(upto + strlen(marker), marker)) {
        snprintf(err, err_len, "old text contains more than one [upto] marker");
        return false;
    }
    size_t head_len = (size_t)(upto - old);
    const char *tail = upto + strlen(marker);
    size_t tail_len = old_len - head_len - strlen(marker);
    /* Strip leading newline/CR from tail before searching.  The head already
     * includes the newline at its end, so the extra \n that follows [upto] in
     * the old text (whether injected by the forcer or written by the model)
     * must not be part of the tail needle -- the file after the head has no
     * duplicate newline. */
    while (tail_len > 0 && (*tail == '\n' || *tail == '\r')) {
        tail++;
        tail_len--;
    }
    if (!agent_span_has_nonspace(tail, tail_len)) {
        snprintf(err, err_len,
                 "old text after [upto] must include a unique tail anchor");
        return false;
    }
    const char *head_pos = NULL;
    const char *tail_pos = NULL;
    if (!agent_find_unique(data, len, old, head_len, &head_pos, "old head",
                           err, err_len))
        return false;
    if (!agent_find_unique_after(data, len, head_pos + head_len,
                                 tail, tail_len, &tail_pos, "old tail",
                                 err, err_len))
        return false;
    *anchored = true;
    *match = head_pos;
    *match_len = (size_t)(tail_pos - head_pos) + tail_len;
    return true;
}

#ifdef P36_AGENT_TEST
static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static void test_agent_cache_lengths_are_bounded_by_file(void) {
    FILE *fp = tmpfile();
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    AGENT_TEST_ASSERT(fwrite("abc", 1, 3, fp) == 3);
    rewind(fp);
    char *text = NULL;
    char err[128] = {0};
    AGENT_TEST_ASSERT(!agent_kv_read_text(fp, UINT32_MAX, &text,
                                         err, sizeof(err)));
    AGENT_TEST_ASSERT(text == NULL);
    fclose(fp);

    fp = tmpfile();
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    uint8_t huge[4] = {0xff, 0xff, 0xff, 0xff};
    AGENT_TEST_ASSERT(fwrite(huge, 1, sizeof(huge), fp) == sizeof(huge));
    rewind(fp);
    p36_kvstore_entry hdr = {0};
    char *title = NULL;
    AGENT_TEST_ASSERT(!agent_kv_read_title_trailer(fp, &hdr, &title,
                                                   err, sizeof(err)));
    AGENT_TEST_ASSERT(title == NULL);
    fclose(fp);
}

static void test_agent_edit_upto_tail_newline_is_not_part_of_anchor(void) {
    const char *data =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "clean:\n"
        "\trm -f bc\n";
    const char *old =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "[upto]\n"
        "clean:\n";

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};
    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old, false,
                                                &match, &match_len, &anchored,
                                                err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "EDIT_MATCH_NOT_FOUND") != NULL);
    err[0] = '\0';
    AGENT_TEST_ASSERT(agent_edit_find_old_span(data, strlen(data), old, true,
                                               &match, &match_len, &anchored,
                                               err, sizeof(err)));
    AGENT_TEST_ASSERT(anchored);
    AGENT_TEST_ASSERT(match == data);
    AGENT_TEST_ASSERT(match_len == strlen(data) - strlen("\trm -f bc\n"));
}

static void test_agent_edit_upto_requires_tail_after_newline_strip(void) {
    const char *data = "head\nbody\ntail\n";
    const char *old = "head\n[upto]\n";
    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};

    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old, true,
                                                &match, &match_len, &anchored,
                                                err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "must include a unique tail anchor") != NULL);
}

static void test_agent_qwen_tool_parser(void) {
    const char *raw =
        "<tool_call>\n"
        "<function=edit>\n"
        "<parameter=path>\n/tmp/a.c\n</parameter>\n"
        "<parameter=old>\nline one\nline two\n</parameter>\n"
        "<parameter=new>\nreplacement\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    agent_qwen_tool_parser p = {.state = AGENT_QWEN_TOOL_SEARCH};
    for (size_t i = 0; raw[i]; i++) agent_qwen_tool_feed(&p, raw + i, 1);
    AGENT_TEST_ASSERT(p.state == AGENT_QWEN_TOOL_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(p.calls.v[0].name, "edit"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "/tmp/a.c"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "old"),
                              "line one\nline two"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "new"),
                              "replacement"));
    agent_qwen_tool_parser_free(&p);

    const char *json_raw =
        "<tool_call>\n"
        "{\"name\": \"edit\", \"arguments\": {\"path\": \"/tmp/b.c\", \"old\": \"x\", \"new\": \"y\"}}\n"
        "</tool_call>";
    p = (agent_qwen_tool_parser){.state = AGENT_QWEN_TOOL_SEARCH};
    for (size_t i = 0; json_raw[i]; i++) agent_qwen_tool_feed(&p, json_raw + i, 1);
    AGENT_TEST_ASSERT(p.state == AGENT_QWEN_TOOL_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(p.calls.v[0].name, "edit"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "/tmp/b.c"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "old"), "x"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "new"), "y"));
    agent_qwen_tool_parser_free(&p);
}

static void test_agent_chat_template_controls(void) {
    bool thinking = true;
    char *text = agent_chat_control_text(
        "  inspect this <|think_off|> carefully  ", &thinking);
    AGENT_TEST_ASSERT(!thinking);
    AGENT_TEST_ASSERT(!strcmp(text, "inspect this  carefully"));
    free(text);

    text = agent_chat_control_text("<|think_on|> continue", &thinking);
    AGENT_TEST_ASSERT(thinking);
    AGENT_TEST_ASSERT(!strcmp(text, "continue"));
    free(text);

    char *system = agent_build_system_prompt(
        "<|think_off|> custom system instructions", &thinking, false, false);
    AGENT_TEST_ASSERT(!thinking);
    AGENT_TEST_ASSERT(strstr(system, "# Tools") != NULL);
    AGENT_TEST_ASSERT(strstr(system, "custom system instructions") != NULL);
    AGENT_TEST_ASSERT(strstr(system, "<|think_off|>") == NULL);
    free(system);
}

static void test_agent_edit_upto_prompt_is_opt_in(void) {
    char *exact = agent_build_tools_prompt(false, false);
    char *upto = agent_build_tools_prompt(true, true);
    AGENT_TEST_ASSERT(strstr(exact,
        "To call multiple functions, output a separate, completely closed") != NULL);
    AGENT_TEST_ASSERT(strstr(exact, "Do NOT nest <tool_call> blocks.") != NULL);
    AGENT_TEST_ASSERT(strstr(exact,
        "provide your final answer directly to the user without any tool call") != NULL);
    AGENT_TEST_ASSERT(strstr(exact, "[upto]") == NULL);
    AGENT_TEST_ASSERT(strstr(upto, "[upto]") != NULL);
    AGENT_TEST_ASSERT(strstr(exact, "view_image") == NULL);
    AGENT_TEST_ASSERT(strstr(upto, "view_image") != NULL);
    free(exact);
    free(upto);
}

static void test_agent_tool_error_recovery(void) {
    AGENT_TEST_ASSERT(agent_tool_response_is_error(
        "Tool result 1 (edit):\nTool error: old text did not match\n"));
    AGENT_TEST_ASSERT(!agent_tool_response_is_error(
        "command took 2.0s\nerror: appears in captured output\n"));
    AGENT_TEST_ASSERT(!agent_tool_response_is_error(
        "{\"error\": null, \"result\": \"ok\"}"));
    AGENT_TEST_ASSERT(!agent_tool_response_is_error("throw new Error('example')"));
    char traceback[900];
    memset(traceback, 'x', sizeof(traceback));
    memcpy(traceback, "Traceback (most recent call last):", 34);
    traceback[sizeof(traceback) - 1] = '\0';
    AGENT_TEST_ASSERT(agent_tool_response_is_error(traceback));

    int failures = 0;
    char *text = agent_tool_response_add_warning(
        xstrdup("Tool error: first failure"), &failures);
    AGENT_TEST_ASSERT(failures == 1);
    AGENT_TEST_ASSERT(strstr(text,
        "⚠️ SYSTEM WARNING: The previous tool call returned an error. "
        "Diagnose the failure and retry with completely corrected arguments.") != NULL);
    free(text);

    text = agent_tool_response_add_warning(
        xstrdup("Tool error: second failure"), &failures);
    AGENT_TEST_ASSERT(failures == 2);
    AGENT_TEST_ASSERT(strstr(text,
        "⚠️ SYSTEM WARNING: 2 consecutive tool errors detected. "
        "Your previous approach is incorrect. You MUST use a fundamentally "
        "different approach or corrected arguments.") != NULL);
    free(text);

    text = agent_tool_response_add_warning(xstrdup("Tool result: ok"), &failures);
    AGENT_TEST_ASSERT(failures == 0);
    AGENT_TEST_ASSERT(!strcmp(text, "Tool result: ok"));
    free(text);
}

static void test_agent_streaming_defaults(void) {
    char *defv[] = {"p36-agent"};
    char *streamv[] = {"p36-agent", "--ssd-streaming"};
    char *streamctxv[] = {"p36-agent", "--ctx", "65536", "--ssd-streaming"};
    char *ctxstreamv[] = {"p36-agent", "--ssd-streaming", "--ctx", "49152"};
    char *streamcachev[] = {"p36-agent", "--ssd-streaming", "-ctk", "q8_0", "-ctv", "q4_0"};
    char *cachev[] = {"p36-agent", "-ctk", "q4_0", "-ctv", "f16"};
    char *cpuv[] = {"p36-agent", "--cpu"};
    char *metalv[] = {"p36-agent", "--metal"};
    char *vulkanv[] = {"p36-agent", "--vulkan"};
    char *chunkv[] = {"p36-agent", "--prefill-chunk", "1024"};
    char *samplingv[] = {"p36-agent", "--temp", "0.7", "--top-k", "20",
                         "--top-p", "0.95", "--min-p", "0.05"};
    char *overridev[] = {"p36-agent", "--temp", "0.2", "--top-k", "7",
                         "--top-p", "0.6", "--min-p", "0.01"};
    char *budgetv[] = {"p36-agent", "--thinking-budget", "32000"};
    char *penaltyv[] = {"p36-agent", "--default-frequency-penalty", "0.15"};
    char *repeatv[] = {"p36-agent", "--repeat-penalty", "1.1"};
    char *uptov[] = {"p36-agent", "--edit-upto"};
    agent_config def = parse_options(1, defv);
    agent_config stream = parse_options(2, streamv);
    agent_config streamctx = parse_options(4, streamctxv);
    agent_config ctxstream = parse_options(4, ctxstreamv);
    agent_config streamcache = parse_options(6, streamcachev);
    agent_config cache = parse_options(5, cachev);
    agent_config cpu = parse_options(2, cpuv);
    agent_config metal = parse_options(2, metalv);
    agent_config vulkan = parse_options(2, vulkanv);
    agent_config chunk = parse_options(3, chunkv);
    agent_config sampling = parse_options(9, samplingv);
    agent_config override = parse_options(9, overridev);
    agent_config budget = parse_options(3, budgetv);
    agent_config penalty = parse_options(3, penaltyv);
    agent_config repeat = parse_options(3, repeatv);
    agent_config upto = parse_options(2, uptov);
    agent_apply_sampling_defaults(&override, NULL);

    AGENT_TEST_ASSERT(!def.engine.ssd_streaming);
    AGENT_TEST_ASSERT(def.gen.n_predict == 100000);
    AGENT_TEST_ASSERT(def.gen.thinking_budget == 50000);
    AGENT_TEST_ASSERT(budget.gen.thinking_budget == 32000);
    AGENT_TEST_ASSERT(def.gen.frequency_penalty == 0.0f);
    AGENT_TEST_ASSERT(penalty.gen.frequency_penalty == 0.15f);
    AGENT_TEST_ASSERT(def.gen.repeat_penalty == 1.0f);
    AGENT_TEST_ASSERT(!def.gen.repeat_penalty_set);
    AGENT_TEST_ASSERT(repeat.gen.repeat_penalty == 1.1f);
    AGENT_TEST_ASSERT(repeat.gen.repeat_penalty_set);
    AGENT_TEST_ASSERT(!def.edit_upto);
    AGENT_TEST_ASSERT(upto.edit_upto);
    AGENT_TEST_ASSERT(def.gen.ctx_size == AGENT_RESIDENT_CTX);
    AGENT_TEST_ASSERT(metal.gen.ctx_size == AGENT_RESIDENT_CTX);
    AGENT_TEST_ASSERT(vulkan.gen.ctx_size == AGENT_RESIDENT_CTX);
    AGENT_TEST_ASSERT(def.engine.prefill_chunk == 0);
#ifndef P36_NO_GPU
    p36_context_memory auto_metal = p36_context_memory_estimate_configured(
        P36_BACKEND_METAL, 4096, def.engine.prefill_chunk,
        P36_KV_CACHE_Q8_0, P36_KV_CACHE_Q4_0);
    AGENT_TEST_ASSERT(auto_metal.prefill_cap == 1024);
#endif
#ifdef P36_NO_GPU
    AGENT_TEST_ASSERT(def.engine.cache_type_k == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(def.engine.cache_type_v == P36_KV_CACHE_F16);
#else
    AGENT_TEST_ASSERT(def.engine.cache_type_k == P36_KV_CACHE_Q8_0);
    AGENT_TEST_ASSERT(def.engine.cache_type_v == P36_KV_CACHE_Q4_0);
#endif
    AGENT_TEST_ASSERT(stream.engine.ssd_streaming);
    AGENT_TEST_ASSERT(stream.gen.ctx_size == AGENT_STREAMING_CTX);
    AGENT_TEST_ASSERT(stream.engine.prefill_chunk == 0);
    AGENT_TEST_ASSERT(chunk.engine.prefill_chunk == 1024);
    AGENT_TEST_ASSERT(stream.engine.cache_type_k == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(stream.engine.cache_type_v == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(streamctx.engine.ssd_streaming);
    AGENT_TEST_ASSERT(streamctx.gen.ctx_size == 65536);
    AGENT_TEST_ASSERT(ctxstream.engine.ssd_streaming);
    AGENT_TEST_ASSERT(ctxstream.gen.ctx_size == 49152);
    AGENT_TEST_ASSERT(streamcache.engine.cache_type_k == P36_KV_CACHE_Q8_0);
    AGENT_TEST_ASSERT(streamcache.engine.cache_type_v == P36_KV_CACHE_Q4_0);
    AGENT_TEST_ASSERT(cache.engine.cache_type_k == P36_KV_CACHE_Q4_0);
    AGENT_TEST_ASSERT(cache.engine.cache_type_v == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(!cpu.engine.ssd_streaming);
    AGENT_TEST_ASSERT(cpu.engine.cache_type_k == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(cpu.engine.cache_type_v == P36_KV_CACHE_F16);
    AGENT_TEST_ASSERT(sampling.gen.temperature == 0.7f);
    AGENT_TEST_ASSERT(sampling.gen.top_k == 20);
    AGENT_TEST_ASSERT(sampling.gen.top_p == 0.95f);
    AGENT_TEST_ASSERT(sampling.gen.min_p == 0.05f);
    AGENT_TEST_ASSERT(override.gen.temperature == 0.2f);
    AGENT_TEST_ASSERT(override.gen.top_k == 7);
    AGENT_TEST_ASSERT(override.gen.top_p == 0.6f);
    AGENT_TEST_ASSERT(override.gen.min_p == 0.01f);
    AGENT_TEST_ASSERT(override.gen.temperature_set);
    AGENT_TEST_ASSERT(override.gen.top_k_set);
    AGENT_TEST_ASSERT(override.gen.top_p_set);
    AGENT_TEST_ASSERT(override.gen.min_p_set);
}

static void test_agent_thinking_budget_modes(void) {
    char *defv[] = {"p36-agent"};
    char *lowv[] = {"p36-agent", "--thinking-budget", "8000"};
    char *mediumv[] = {"p36-agent", "--thinking-budget", "16000"};
    char *highv[] = {"p36-agent", "--thinking-budget", "24000"};
    char *xhighv[] = {"p36-agent", "--thinking-budget", "24001"};
    char *explicitv[] = {"p36-agent", "--think-medium"};
    agent_config cfg = parse_options(1, defv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_XHIGH);
    AGENT_TEST_ASSERT(cfg.gen.thinking_budget == 50000);

    cfg = parse_options(3, lowv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_LOW);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(5999, cfg.gen.thinking_budget) == 0);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(6000, cfg.gen.thinking_budget) == 1);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(8000, 50000) == 0);

    cfg = parse_options(3, mediumv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_MEDIUM);
    cfg = parse_options(3, highv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_HIGH);
    AGENT_TEST_ASSERT(strstr(p36_qwen38_effort_instruction(cfg.gen.think_mode),
                             "xhigh") != NULL);
    cfg = parse_options(3, xhighv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_XHIGH);
    cfg = parse_options(2, explicitv);
    agent_resolve_thinking_config(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_MEDIUM);
    AGENT_TEST_ASSERT(cfg.gen.thinking_budget == 50000);

    p36_think_mode mode = P36_THINK_NONE;
    mode = agent_next_think_mode(mode, true);
    AGENT_TEST_ASSERT(mode == P36_THINK_LOW);
    mode = agent_next_think_mode(mode, true);
    AGENT_TEST_ASSERT(mode == P36_THINK_MEDIUM);
    mode = agent_next_think_mode(mode, true);
    AGENT_TEST_ASSERT(mode == P36_THINK_HIGH);
    mode = agent_next_think_mode(mode, true);
    AGENT_TEST_ASSERT(mode == P36_THINK_XHIGH);
    mode = agent_next_think_mode(mode, true);
    AGENT_TEST_ASSERT(mode == P36_THINK_LOW);
    AGENT_TEST_ASSERT(agent_next_think_mode(P36_THINK_NONE, false) == P36_THINK_HIGH);
    AGENT_TEST_ASSERT(agent_next_think_mode(P36_THINK_HIGH, false) == P36_THINK_NONE);

    cfg = parse_options(1, defv);
    agent_resolve_thinking_config(&cfg, true);
    agent_cycle_think_mode(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_LOW);
    AGENT_TEST_ASSERT(cfg.gen.thinking_budget == 50000);
    agent_cycle_think_mode(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_MEDIUM);
    AGENT_TEST_ASSERT(cfg.gen.thinking_budget == 50000);

    cfg = parse_options(3, lowv);
    agent_resolve_thinking_config(&cfg, true);
    agent_cycle_think_mode(&cfg, true);
    AGENT_TEST_ASSERT(cfg.gen.think_mode == P36_THINK_MEDIUM);
    AGENT_TEST_ASSERT(cfg.gen.thinking_budget == 8000);
}

static void test_agent_context_pressure_helpers(void) {
    AGENT_TEST_ASSERT(!agent_context_should_compact(100000, 69999));
    AGENT_TEST_ASSERT(agent_context_should_compact(100000, 70000));
    AGENT_TEST_ASSERT(!agent_context_should_compact(64000, 44799));
    AGENT_TEST_ASSERT(agent_context_should_compact(64000, 44800));
    AGENT_TEST_ASSERT(agent_think_soft_close_start(32) == 24);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(23, 32) == 0);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(24, 32) == 1);
    AGENT_TEST_ASSERT(!agent_think_hard_budget_reached(31, 32));
    AGENT_TEST_ASSERT(agent_think_hard_budget_reached(32, 32));
    AGENT_TEST_ASSERT(agent_think_soft_close_start(64) == 48);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(47, 64) == 0);
    AGENT_TEST_ASSERT(agent_think_close_rank_limit(48, 64) == 1);
    AGENT_TEST_ASSERT(!agent_think_hard_budget_reached(63, 64));
    AGENT_TEST_ASSERT(agent_think_hard_budget_reached(64, 64));
    AGENT_TEST_ASSERT(p36_kvstore_quant_bits_valid(1));
    AGENT_TEST_ASSERT(p36_kvstore_quant_bits_valid(3));
    AGENT_TEST_ASSERT(p36_kvstore_quant_bits_valid(8));
    AGENT_TEST_ASSERT(!p36_kvstore_quant_bits_valid(0));
    AGENT_TEST_ASSERT(!p36_kvstore_quant_bits_valid(7));
    AGENT_TEST_ASSERT(!p36_kvstore_quant_bits_valid(9));

    char path[PATH_MAX];
    const char *home = getenv("HOME");
    AGENT_TEST_ASSERT(agent_expand_home_path("/tmp/x", path, sizeof(path)));
    AGENT_TEST_ASSERT(!strcmp(path, "/tmp/x"));
    if (home && home[0]) {
        AGENT_TEST_ASSERT(agent_expand_home_path("~/x", path, sizeof(path)));
        char expected[PATH_MAX];
        snprintf(expected, sizeof(expected), "%s/x", home);
        AGENT_TEST_ASSERT(!strcmp(path, expected));
    }
}

static void test_agent_repetition_detector(void) {
    agent_repetition_detector detector = {0};
    const char *line = "scene.add(makeTree({ x: 10, y: 20 }));\n";
    bool detected = false;
    for (int i = 0; i < AGENT_REPETITION_EXACT_LINE_RUN; i++)
        detected = agent_repetition_detector_feed(&detector, line, strlen(line));
    AGENT_TEST_ASSERT(detected);

    memset(&detector, 0, sizeof(detector));
    detected = false;
    char varied[96];
    for (int i = 0; i < AGENT_REPETITION_NEAR_LINE_RUN; i++) {
        snprintf(varied, sizeof(varied),
                 "scene.add(makeTree({ x: %02d, y: 20 }));\n", i);
        detected = agent_repetition_detector_feed(
            &detector, varied, strlen(varied));
    }
    AGENT_TEST_ASSERT(detected);

    memset(&detector, 0, sizeof(detector));
    detected = false;
    for (int i = 0; i < 64; i++) {
        snprintf(varied, sizeof(varied),
                 "items[%d] = build_item(%d, %d);\n", i, i, i + 1);
        detected |= agent_repetition_detector_feed(
            &detector, varied, strlen(varied));
    }
    AGENT_TEST_ASSERT(!detected);
}

static void test_agent_welcome_banner(void) {
    agent_config cfg = {0};
    char banner[256];
    cfg.gen.ctx_size = 100000;
    agent_format_welcome_banner(&cfg, banner, sizeof(banner));
    AGENT_TEST_ASSERT(strstr(banner, "Quark") != NULL);
    AGENT_TEST_ASSERT(strstr(banner, "Star") != NULL);
    AGENT_TEST_ASSERT(strstr(banner, "context 100k tokens") != NULL);
}

static void test_agent_hints_state(void) {
    bool enabled = false;
    AGENT_TEST_ASSERT(agent_parse_hints("on", &enabled) && enabled);
    AGENT_TEST_ASSERT(agent_parse_hints("off", &enabled) && !enabled);
    AGENT_TEST_ASSERT(!agent_parse_hints("", &enabled));
    AGENT_TEST_ASSERT(!agent_parse_hints("toggle", &enabled));
    AGENT_TEST_ASSERT(!agent_parse_hints("on off", &enabled));
    AGENT_TEST_ASSERT(agent_slash_command_known("/hints on"));
    AGENT_TEST_ASSERT(!agent_slash_command_known("/hintson"));

    agent_hints h = {0};
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = true;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = false;
    const char *note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    h.enabled = true;
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && note != agent_hints_prompt);
    AGENT_TEST_ASSERT(note && strstr(note, "enabled"));

    agent_hints_compacted(&h);
    AGENT_TEST_ASSERT(h.enabled && !h.instructed);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);
    h.enabled = false;
    agent_hints_compacted(&h);
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    AGENT_TEST_ASSERT(!h.instructed);
    agent_hints_compacted(&h);
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));

    /* A restored session must override old on notes without restoring a setting. */
    h = (agent_hints){.applied = AGENT_HINTS_UNKNOWN};
    note = agent_hints_note(&h);
    AGENT_TEST_ASSERT(!h.enabled && !h.instructed);
    AGENT_TEST_ASSERT(note && strstr(note, "disabled"));
    h.enabled = true;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == agent_hints_prompt);

    /* Multiple changes before a safe point coalesce to the latest request. */
    h = (agent_hints){0};
    h.enabled = true;
    h.enabled = false;
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);
    agent_hints_compacted(&h);
    AGENT_TEST_ASSERT(agent_hints_note(&h) == NULL);

    char *tools = agent_build_tools_prompt(false, false);
    AGENT_TEST_ASSERT(strstr(tools, "programming hints") == NULL);
    free(tools);
}

static void p36_agent_unit_tests_run(void) {
    test_agent_hints_state();
    test_agent_cache_lengths_are_bounded_by_file();
    test_agent_edit_upto_tail_newline_is_not_part_of_anchor();
    test_agent_edit_upto_requires_tail_after_newline_strip();
    test_agent_qwen_tool_parser();
    test_agent_chat_template_controls();
    test_agent_edit_upto_prompt_is_opt_in();
    test_agent_tool_error_recovery();
    test_agent_streaming_defaults();
    test_agent_thinking_budget_modes();
    test_agent_context_pressure_helpers();
    test_agent_repetition_detector();
    test_agent_welcome_banner();
}
#endif

static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return true; /* Cannot preflight until path is known. */

    const char *old = agent_tool_arg_value(call, "old");
    if (!old || !old[0]) {
        snprintf(err, err_len, "edit requires non-empty old text");
        return false;
    }

    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, err_len) != 0)
        return false;

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    bool allow_upto = w && w->cfg && w->cfg->edit_upto;
    bool ok = agent_edit_find_old_span(data, len, old, allow_upto,
                                       &match, &match_len,
                                       &anchored, err, err_len);
    free(data);
    return ok;
}

static char *agent_apply_file_splice(const char *path,
                                     const char *data, size_t len,
                                     size_t offset, size_t remove_len,
                                     const char *insert, const char *kind) {
    char err[256];
    if (!insert) insert = "";
    size_t insert_len = strlen(insert);
    size_t out_len = offset + insert_len + (len - offset - remove_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, offset);
    memcpy(out + offset, insert, insert_len);
    memcpy(out + offset + insert_len, data + offset + remove_len,
           len - offset - remove_len);
    out[out_len] = '\0';

    if (out_len == len && !memcmp(out, data, len)) {
        char msg[PATH_MAX + 80];
        snprintf(msg, sizeof(msg), "Tool error: EDIT_NO_CHANGE path=%s reason=%s\n", path,
                 "replacement is identical to the existing text");
        free(out);
        return xstrdup(msg);
    }
    int rc = agent_replace_file(path, out, out_len, data, len, err, sizeof(err));
    if (rc != 0) {
        free(out);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int start_line = 0, end_line = 0, delta = 0;
    agent_old_new_line_effect(data, len, out, out_len, offset, remove_len,
                              &start_line, &end_line, &delta);
    char *result = agent_edit_result(path, start_line, end_line, delta,
                                     out, out_len, kind);
    free(out);
    return result;
}

/* Old/new editing is intentionally conservative: exact old text must be unique.
 * For large replacements, old may contain one [upto] marker: the head must be
 * unique, and the tail must be unique after that head before the whole span is
 * replaced. */
/* Replace lines [start,end] (1-based, inclusive) of path with new_text.  end =
 * start-1 inserts before start.  Needs no exact old text, so indentation never
 * has to be reproduced by the model. */
static char *agent_tool_edit_lines(const char *path, const char *start_arg,
                                   const char *end_arg, const char *new_text) {
    char err[256], msg[PATH_MAX + 160];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        snprintf(msg, sizeof(msg), "Tool error: %s\n", err);
        return xstrdup(msg);
    }
    int lines = 0;
    for (size_t i = 0; i < len; i++) if (data[i] == '\n') lines++;
    if (len && data[len - 1] != '\n') lines++;
    int start = agent_parse_int_default(start_arg, -1, -1, INT_MAX);
    int end = agent_parse_int_default(end_arg, start, -1, INT_MAX);
    if (start < 1 || start > lines + 1 || end < start - 1 || end > lines) {
        snprintf(msg, sizeof(msg),
                 "Tool error: EDIT_LINE_RANGE_INVALID path=%s start=%s end=%s lines=%d\n",
                 path, start_arg, end_arg ? end_arg : start_arg, lines);
        free(data);
        return xstrdup(msg);
    }
    size_t off, end_off;
    int cur = 1;
    off = (start == 1) ? 0 : len;
    end_off = len;
    for (size_t i = 0; i < len; i++) {
        if (data[i] != '\n') continue;
        cur++;
        if (cur == start) off = i + 1;
        if (cur == end + 1) { end_off = i + 1; break; }
    }
    if (end < start) end_off = off;
    size_t nl = strlen(new_text);
    char *ins = xmalloc(nl + 2);
    memcpy(ins, new_text, nl);
    bool eof_no_nl = end_off == len && (!len || data[len - 1] != '\n');
    if (nl && new_text[nl - 1] != '\n' && !eof_no_nl) ins[nl++] = '\n';
    ins[nl] = '\0';
    char *result = agent_apply_file_splice(path, data, len, off, end_off - off,
                                           ins, "line-range replacement");
    free(ins);
    free(data);
    return result;
}

static char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *old = agent_tool_arg_value(call, "old");
    const char *new_text = agent_tool_arg_value(call, "new");
    const char *start_arg = agent_tool_arg_value(call, "start_line");
    if (!new_text) return xstrdup("Tool error: edit requires new text\n");
    if (start_arg && start_arg[0] && (!old || !old[0]))
        return agent_tool_edit_lines(path, start_arg,
                                     agent_tool_arg_value(call, "end_line"), new_text);
    if (!old || !old[0]) return xstrdup("Tool error: EDIT_OLD_EMPTY use=old or start_line/end_line\n");

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    bool allow_upto = w && w->cfg && w->cfg->edit_upto;
    if (!agent_edit_find_old_span(data, len, old, allow_upto,
                                  &match, &match_len,
                                  &anchored, err, sizeof(err)))
    {
        free(data);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, " path=");
        agent_buf_puts(&b, path);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char *result = agent_apply_file_splice(path, data, len,
                                           (size_t)(match - data), match_len,
                                           new_text,
                                           anchored ? "anchored old/new replacement"
                                                    : "old/new replacement");
    free(data);
    return result;
}

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    size_t skipped;
    char first_skip[PATH_MAX + 128];
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    size_t n = strlen(data);
    if (n > AGENT_READ_LINE_MAX) {
        char marker[96];
        size_t keep = AGENT_READ_BLOB_KEEP;
        while (keep && ((unsigned char)data[keep] & 0xc0) == 0x80) keep--;
        agent_buf_append(&ctx->out, data, keep);
        snprintf(marker, sizeof(marker), "\n[MINIFIED_OR_VENDOR_BLOB OMITTED: %zu bytes]", n);
        agent_buf_puts(&ctx->out, marker);
    } else {
        agent_buf_append(&ctx->out, data, n);
    }
    agent_buf_puts(&ctx->out, "\n");
}

static void agent_search_skip(agent_search_ctx *ctx, const char *path, const char *why) {
    ctx->skipped++;
    if (!ctx->first_skip[0])
        snprintf(ctx->first_skip, sizeof(ctx->first_skip), "%s: %s", path, why);
}

/* A file can be arbitrarily large; only the current line and a small context
 * ring are resident. Oversized or binary lines are reported, never ignored. */
static int agent_search_read_line(FILE *fp, char **text) {
    agent_buf line = {0};
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!c || line.len == AGENT_TOOL_MAX_BYTES) {
            free(line.ptr);
            return -2;
        }
        if (c == '\n' || c == '\r') {
            if (c == '\r') {
                int next = fgetc(fp);
                if (next != '\n' && next != EOF) ungetc(next, fp);
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&line, &ch, 1);
    }
    if (ferror(fp)) { free(line.ptr); return -1; }
    if (c == EOF && !line.len) { free(line.ptr); return 0; }
    *text = agent_buf_take(&line);
    return 1;
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    FILE *fp = agent_open_regular_file(path);
    if (!fp) { agent_search_skip(ctx, path, strerror(errno)); return; }
    char *ring[6] = {0};
    bool printed_file = false;
    int line = 0, last_emitted = 0, after = 0;
    while ((ctx->results < ctx->max_results || after) && !ctx->out.truncated) {
        char *text = NULL;
        int rc = agent_search_read_line(fp, &text);
        if (!rc) break;
        if (rc < 0) {
            agent_search_skip(ctx, path, rc == -2 ? "binary data or line exceeds 128 KiB" : strerror(errno));
            break;
        }
        if (line == INT_MAX) {
            free(text);
            agent_search_skip(ctx, path, "line number limit reached");
            break;
        }
        line++;
        free(ring[line % 6]);
        ring[line % 6] = text;
        bool match = ctx->results < ctx->max_results &&
                     agent_search_line_matches(ctx, text, strlen(text));
        if (match && !printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        if (match) {
            int from = line - ctx->context;
            if (from <= last_emitted) from = last_emitted + 1;
            for (int j = from; j <= line; j++)
                agent_search_emit_line(ctx, ring[j % 6], j);
            last_emitted = line;
            after = ctx->context;
            ctx->results++;
        } else if (after) {
            agent_search_emit_line(ctx, text, line);
            last_emitted = line;
            after--;
        }
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    for (int i = 0; i < 6; i++) free(ring[i]);
    fclose(fp);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (depth > 24) { agent_search_skip(ctx, path, "directory depth limit reached"); return; }
    struct stat st;
    if ((depth == 0 ? stat(path, &st) : lstat(path, &st)) != 0) {
        agent_search_skip(ctx, path, strerror(errno));
        return;
    }
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        agent_search_skip(ctx, path, "not a regular file or directory (nested symlinks are not followed)");
        return;
    }
    DIR *dir = opendir(path);
    if (!dir) { agent_search_skip(ctx, path, strerror(errno)); return; }
    struct dirent *de;
    while (ctx->results < ctx->max_results && !ctx->out.truncated) {
        errno = 0;
        de = readdir(dir);
        if (!de) {
            if (errno) agent_search_skip(ctx, path, strerror(errno));
            break;
        }
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        if (snprintf(child, sizeof(child), "%s/%s", path, de->d_name) >= (int)sizeof(child)) {
            agent_search_skip(ctx, path, "path exceeds PATH_MAX");
            continue;
        }
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

/* Implement the search tool using either literal matching or POSIX regex. */
