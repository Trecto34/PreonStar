static agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = P36_DEFAULT_MODEL_PATH,
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside p36-agent.",
            .n_predict = 100000,
            .thinking_budget = AGENT_THINKING_BUDGET_DEFAULT,
            .action_budget = AGENT_ACTION_BUDGET_DEFAULT,
            .max_repeat_tool = AGENT_MAX_REPEAT_TOOL_DEFAULT,
            .max_stagnant_turns = AGENT_MAX_STAGNANT_DEFAULT,
            .max_recoveries = AGENT_MAX_RECOVERIES_DEFAULT,
            .max_verify_steps = AGENT_MAX_VERIFY_DEFAULT,
            .max_no_progress_steps = AGENT_MAX_NOPROGRESS_DEFAULT,
            .max_implementation_steps = AGENT_MAX_IMPL_DEFAULT,
            .ctx_size = AGENT_RESIDENT_CTX,
            .temperature = P36_DEFAULT_TEMPERATURE,
            .top_k = 0,
            .top_p = P36_DEFAULT_TOP_P,
            .min_p = P36_DEFAULT_MIN_P,
            .repeat_penalty = 1.0f,
            .think_mode = P36_THINK_HIGH,
        },
    };

    bool steering_scale_set = false;
    bool ctx_set = false;
    bool cache_type_k_set = false;
    bool cache_type_v_set = false;
    bool show_help = false;
    const char *help_topic = NULL;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            show_help = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') help_topic = argv[++i];
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "--edit-upto")) {
            c.edit_upto = true;
        } else if (!strcmp(arg, "--prefix-file")) {
            if (c.gen.prefix.count) {
                fprintf(stderr, "specify --prefix-file only once\n");
                exit(2);
            }
            char err[256] = {0};
            if (p36_prompt_prefix_load(&c.gen.prefix,
                    need_arg(&i, argc, argv, arg), err, sizeof(err)) != 0) {
                fprintf(stderr, "%s\n", err);
                exit(2);
            }
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.gen.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-template")) {
            c.engine.chat_template = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vision")) {
            c.engine.vision_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
            ctx_set = true;
        } else if (!strcmp(arg, "-ctk") || !strcmp(arg, "--cache-type-k")) {
            if (!p36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.engine.cache_type_k)) {
                fprintf(stderr, "p36-agent: invalid KV cache K type\n");
                exit(2);
            }
            cache_type_k_set = true;
        } else if (!strcmp(arg, "-ctv") || !strcmp(arg, "--cache-type-v")) {
            if (!p36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.engine.cache_type_v)) {
                fprintf(stderr, "p36-agent: invalid KV cache V type\n");
                exit(2);
            }
            cache_type_v_set = true;
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--action-budget")) {
            c.gen.action_budget = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-repeat-tool")) {
            c.gen.max_repeat_tool = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-stagnant-turns")) {
            c.gen.max_stagnant_turns = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--server")) {
            c.server_url = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--server-model")) {
            c.server_model = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--api-key")) {
            c.api_key = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--server-timeout")) {
            c.server_timeout = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--server-protocol")) {
            const char *proto = need_arg(&i, argc, argv, arg);
            if (strcmp(proto, "auto") && strcmp(proto, "chat")) {
                fprintf(stderr, "p36-agent: --server-protocol %s is not supported (use auto or chat)\n", proto);
                exit(2);
            }
        } else if (!strcmp(arg, "--compact-llm")) {
            c.gen.compact_llm = true;
        } else if (!strcmp(arg, "--max-recoveries")) {
            c.gen.max_recoveries = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-verify-steps")) {
            c.gen.max_verify_steps = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--max-no-progress-steps")) {
            c.gen.max_no_progress_steps = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--thinking-budget")) {
            c.gen.thinking_budget = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.gen.temperature_set = true;
        } else if (!strcmp(arg, "--top-k")) {
            c.gen.top_k = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.gen.top_k_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.min_p_set = true;
        } else if (!strcmp(arg, "--frequency-penalty") ||
                   !strcmp(arg, "--default-frequency-penalty")) {
            c.gen.frequency_penalty = parse_float_range(
                need_arg(&i, argc, argv, arg), arg, -2.0f, 2.0f);
        } else if (!strcmp(arg, "--repeat-penalty") ||
                   !strcmp(arg, "--default-repeat-penalty")) {
            c.gen.repeat_penalty = parse_float_range(
                need_arg(&i, argc, argv, arg), arg, 0.5f, 2.0f);
            c.gen.repeat_penalty_set = true;
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = P36_THINK_HIGH;
            c.gen.think_mode_set = true;
            /* optional level: --think auto|low|medium|high */
            const char *lv = i + 1 < argc ? argv[i + 1] : NULL;
            if (lv && (!strcmp(lv, "auto") || !strcmp(lv, "low") ||
                       !strcmp(lv, "medium") || !strcmp(lv, "high"))) {
                i++;
                if (!strcmp(lv, "auto")) c.gen.think_auto = true;
                else if (!strcmp(lv, "low")) c.gen.think_mode = P36_THINK_LOW;
                else if (!strcmp(lv, "medium")) c.gen.think_mode = P36_THINK_MEDIUM;
            }
        } else if (!strcmp(arg, "--think-low")) {
            c.gen.think_mode = P36_THINK_LOW;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-medium")) {
            c.gen.think_mode = P36_THINK_MEDIUM;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-xhigh")) {
            c.gen.think_mode = P36_THINK_XHIGH;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = P36_THINK_MAX;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = P36_THINK_NONE;
            c.gen.think_mode_set = true;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--vulkan")) {
            c.engine.backend = P36_BACKEND_VULKAN;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = P36_BACKEND_METAL;
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = P36_BACKEND_CPU;
        } else if (!strcmp(arg, "--cuda") ||
                   !strcmp(arg, "--role") || !strcmp(arg, "--layers") ||
                   !strcmp(arg, "--listen") || !strcmp(arg, "--coordinator")) {
            fprintf(stderr, "p36-agent: %s is not supported in this p36 port\n", arg);
            exit(2);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--workspace")) {
            c.workspace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--yolo")) {
            c.yolo = true;
        } else if (!strcmp(arg, "--unsafe-filesystem")) {
            c.unsafe_fs = true;
        } else if (!strcmp(arg, "--max-implementation-steps")) {
            c.gen.max_implementation_steps = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.engine.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.engine.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!p36_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "p36-agent: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            c.engine.ssd_streaming = true;
            c.engine.ssd_streaming_full_layers =
                (uint32_t)parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "p36-agent: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!p36_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "p36-agent: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "p36-agent: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "p36-agent: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "p36-agent: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (show_help) {
        if (help_topic && !p36_help_topic_valid(P36_HELP_AGENT, help_topic)) {
            fprintf(stderr, "p36-agent: unknown help topic: %s\n", help_topic);
            exit(2);
        }
        usage(stdout, help_topic);
        exit(0);
    }
    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    if (!ctx_set) {
        c.gen.ctx_size = c.engine.ssd_streaming ?
            AGENT_STREAMING_CTX : AGENT_RESIDENT_CTX;
    }
    if (c.gen.ctx_size > P36_CONTEXT_MAX) {
        fprintf(stderr, "p36-agent: --ctx must not exceed %d\n", P36_CONTEXT_MAX);
        exit(2);
    }
    if (!cache_type_k_set)
        c.engine.cache_type_k = p36_default_kv_cache_type_k(c.engine.backend, c.engine.ssd_streaming);
    if (!cache_type_v_set)
        c.engine.cache_type_v = p36_default_kv_cache_type_v(c.engine.backend, c.engine.ssd_streaming);
    return c;
}

static void agent_apply_sampling_defaults(agent_config *c, p36_engine *engine) {
    float temperature, top_p, min_p;
    int top_k;
    p36_engine_sampling_defaults(engine, &temperature, &top_k, &top_p, &min_p);
    if (!c->gen.temperature_set) c->gen.temperature = temperature;
    if (!c->gen.top_k_set) c->gen.top_k = top_k;
    if (!c->gen.top_p_set) c->gen.top_p = top_p;
    if (!c->gen.min_p_set) c->gen.min_p = min_p;
}

static void log_context_memory(p36_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               p36_kv_cache_type cache_type_k,
                               p36_kv_cache_type cache_type_v) {
    p36_context_memory m =
        p36_context_memory_estimate_configured(backend,
                                               ctx_size,
                                               prefill_chunk,
                                               cache_type_k,
                                               cache_type_v);
    const uint64_t kv_bytes = m.raw_bytes + m.compressed_bytes;
    const bool color = p36_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";
    fprintf(stderr,
            "%sp36: memory: KV %.2f GiB (raw %.2f + compressed %.2f) "
            "+ buffers %.2f GiB = %s%.2f GiB context%s\n",
            green,
            (double)kv_bytes / 1073741824.0,
            (double)m.raw_bytes / 1073741824.0,
            (double)m.compressed_bytes / 1073741824.0,
            (double)m.scratch_bytes / 1073741824.0,
            bright_green,
            (double)m.total_bytes / 1073741824.0,
            reset);
    fprintf(stderr,
            "%sp36: memory detail: ctx=%d prefill_cap=%u raw_kv_rows=%u "
            "compressed_kv_rows=%u backend=%s%s\n",
            green,
            ctx_size,
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap,
            p36_backend_name(backend),
            reset);
}

static void agent_resolve_thinking_config(agent_config *cfg, bool qwen38) {
    if (qwen38 && !cfg->gen.think_mode_set)
        cfg->gen.think_mode = p36_qwen38_mode_for_budget(cfg->gen.thinking_budget);
}

static p36_think_mode agent_next_think_mode(p36_think_mode mode, bool qwen38) {
    if (!qwen38) return mode == P36_THINK_NONE ? P36_THINK_HIGH : P36_THINK_NONE;
    switch (mode) {
    case P36_THINK_NONE: return P36_THINK_LOW;
    case P36_THINK_LOW: return P36_THINK_MEDIUM;
    case P36_THINK_MEDIUM: return P36_THINK_HIGH;
    case P36_THINK_HIGH: return P36_THINK_XHIGH;
    default: return P36_THINK_LOW;
    }
}

static void agent_cycle_think_mode(agent_config *cfg, bool qwen38) {
    cfg->gen.think_mode = agent_next_think_mode(cfg->gen.think_mode, qwen38);
    cfg->gen.think_mode_set = true;
}

static p36_think_mode effective_think_mode(const agent_config *cfg) {
    return p36_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

static p36_think_mode enabled_think_mode(const agent_config *cfg) {
    p36_think_mode mode = cfg->gen.think_mode;
    if (mode == P36_THINK_NONE) mode = P36_THINK_HIGH;
    return p36_think_mode_for_context(mode, cfg->gen.ctx_size);
}

static char *agent_chat_control_text(const char *text, bool *thinking) {
    static const char think_off[] = "<|think_off|>";
    static const char think_on[] = "<|think_on|>";
    const char *remove = NULL;
    size_t remove_len = 0;

    text = text ? text : "";
    if (strstr(text, think_off)) {
        if (thinking) *thinking = false;
        remove = think_off;
        remove_len = sizeof(think_off) - 1;
    } else if (strstr(text, think_on)) {
        if (thinking) *thinking = true;
        remove = think_on;
        remove_len = sizeof(think_on) - 1;
    }

    size_t len = strlen(text);
    char *out = xmalloc(len + 1);
    size_t used = 0;
    while (remove) {
        const char *p = strstr(text, remove);
        if (!p) break;
        size_t n = (size_t)(p - text);
        memcpy(out + used, text, n);
        used += n;
        text = p + remove_len;
    }
    strcpy(out + used, text);
    used += strlen(text);

    char *start = out;
    char *end = out + used;
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    char *trimmed = xstrndup(start, (size_t)(end - start));
    free(out);
    return trimmed;
}

static bool agent_initial_thinking_enabled(const agent_config *cfg) {
    bool thinking = p36_think_mode_enabled(effective_think_mode(cfg));
    char *plain = agent_chat_control_text(cfg->gen.system, &thinking);
    free(plain);
    return thinking;
}

static bool agent_ascii_contains_ci(const char *s, const char *needle) {
    size_t n = strlen(needle);
    if (!n) return true;
    for (; *s; s++) {
        size_t i = 0;
        while (i < n && s[i] &&
               tolower((unsigned char)s[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == n) return true;
    }
    return false;
}

