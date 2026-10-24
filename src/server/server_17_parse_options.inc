static server_config parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = P36_DEFAULT_MODEL_PATH,
            .backend = default_server_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
            .lookup_order = 3,
            .lookup_k = 0,
        },
        .host = "127.0.0.1",
        .port = 8000,
        .ctx_size = 32768,
        .default_tokens = P36_CONTEXT_MAX,
        .tool_memory_max_ids = P36_TOOL_MEMORY_DEFAULT_MAX_IDS,
        .mixed_prefill_quantum = server_prefill_quantum_default(true),
    };
    c.kv_cache = kv_cache_default_options();

    bool directional_steering_scale_set = false;
    bool cache_type_k_set = false;
    bool cache_type_v_set = false;
    bool show_help = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            show_help = true;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-template")) {
            c.engine.chat_template = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vision")) {
            c.engine.vision_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--lookup-order")) {
            c.engine.lookup_order = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--lookup-k")) {
            c.engine.lookup_k = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-ctk") || !strcmp(arg, "--cache-type-k")) {
            if (!p36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.engine.cache_type_k)) {
                server_log(P36_LOG_DEFAULT, "p36-server: invalid cache type for %s", arg);
                exit(2);
            }
            cache_type_k_set = true;
        } else if (!strcmp(arg, "-ctv") || !strcmp(arg, "--cache-type-v")) {
            if (!p36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.engine.cache_type_v)) {
                server_log(P36_LOG_DEFAULT, "p36-server: invalid cache type for %s", arg);
                exit(2);
            }
            cache_type_v_set = true;
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens") || !strcmp(arg, "--default-tokens")) {
            c.default_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--presence-penalty") || !strcmp(arg, "--default-presence-penalty")) {
            c.default_presence_penalty = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -2.0f, 2.0f);
            c.default_presence_penalty_set = true;
        } else if (!strcmp(arg, "--frequency-penalty") || !strcmp(arg, "--default-frequency-penalty")) {
            c.default_frequency_penalty = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -2.0f, 2.0f);
            c.default_frequency_penalty_set = true;
        } else if (!strcmp(arg, "--temp") || !strcmp(arg, "--temperature")) {
            c.default_temperature = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.default_temperature_set = true;
        } else if (!strcmp(arg, "--top-k")) {
            c.default_top_k = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.default_top_k_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.default_top_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.default_top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.default_min_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.default_min_p_set = true;
        } else if (!strcmp(arg, "--repeat-penalty")) {
            c.default_repeat_penalty = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.5f, 2.0f);
            c.default_repeat_penalty_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--f32-fast-wide")) {
            p36_set_gpu_fast_path_env("P36_VK_F32_FAST_WIDE", "1");
        } else if (!strcmp(arg, "--attn-span")) {
            p36_set_gpu_fast_path_env("P36_VK_ATTN_SPAN", need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.engine.prefill_chunk = (uint32_t)parse_int_arg(
                    need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--batched-session")) {
            c.batched_sessions = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mixed-prefill-quantum")) {
            c.mixed_prefill_quantum = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cors")) {
            c.enable_cors = true;
        } else if (!strcmp(arg, "--nothink") || !strcmp(arg, "--no-think")) {
            g_force_nothink = true;
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--models-dir")) {
            c.models_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-system-checkpoint")) {
            c.kv_cache.system_checkpoint = true;
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--disable-exact-tool-replay")) {
            c.disable_exact_tool_replay = true;
        } else if (!strcmp(arg, "--tool-memory-max-ids")) {
            c.tool_memory_max_ids = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
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
                server_log(P36_LOG_DEFAULT,
                           "p36-server: --ssd-streaming-cache-experts must be a positive count or <number>GB");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            c.engine.ssd_streaming = true;
            c.engine.ssd_streaming_full_layers = (uint32_t)
                parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            c.engine.ssd_streaming_preload_experts = (uint32_t)
                parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!p36_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                server_log(P36_LOG_DEFAULT,
                           "p36-server: --simulate-used-memory must be a positive GiB value, e.g. 8GB");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--vulkan")) {
            c.engine.backend = P36_BACKEND_VULKAN;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = P36_BACKEND_METAL;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = P36_BACKEND_CPU;
        } else if (!strcmp(arg, "--cuda")) {
            server_log(P36_LOG_DEFAULT, "p36-server: %s is not supported; use --metal, --vulkan, or --cpu", arg);
            exit(2);
        } else {
            server_log(P36_LOG_DEFAULT, "p36-server: unknown option: %s", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (c.kv_cache.system_checkpoint && !c.kv_disk_dir && !show_help) {
        server_log(P36_LOG_DEFAULT, "p36-server: --kv-system-checkpoint requires --kv-disk-dir DIR");
        exit(2);
    }
    if (show_help) {
        usage(stdout);
        exit(0);
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens");
        exit(2);
    }
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.ctx_size > P36_CONTEXT_MAX) {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: --ctx must not exceed %d", P36_CONTEXT_MAX);
        exit(2);
    }
    if (!cache_type_k_set)
        c.engine.cache_type_k = p36_default_kv_cache_type_k(c.engine.backend, c.engine.ssd_streaming);
    if (!cache_type_v_set)
        c.engine.cache_type_v = p36_default_kv_cache_type_v(c.engine.backend, c.engine.ssd_streaming);
    return c;
}

#ifndef P36_SERVER_TEST
int main(int argc, char **argv) {
    p36_cli_expand_long_equals(&argc, &argv, "p36-server");
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    g_enable_cors = cfg.enable_cors;
    if (cfg.models_dir && cfg.batched_sessions > 0) {
        server_log(P36_LOG_DEFAULT, "p36-server: --models-dir cannot be combined with --batched-session");
        return 2;
    }
    /* Probe the catalog before any GPU state exists: probes fork. */
    server s;
    memset(&s, 0, sizeof(s));
    if (cfg.models_dir) {
        s.model_initial = server_models_scan(&s, cfg.models_dir, cfg.engine.model_path);
        if (s.model_initial < 0) {
            server_log(P36_LOG_DEFAULT, "p36-server: cannot resolve startup model %s",
                       cfg.engine.model_path);
            server_models_free(&s);
            return 1;
        }
        s.model_current = s.model_initial;
    }
    bool mtp_requested = cfg.engine.mtp_path != NULL || cfg.engine.mtp_draft_tokens > 1;
    if (cfg.batched_sessions > 0) {
        cfg.engine.mtp_path = NULL;
        cfg.engine.mtp_draft_tokens = 1;
    }

    p36_engine *engine = NULL;
    cfg.engine.context_size = cfg.ctx_size;
    cfg.engine.session_count = cfg.batched_sessions > 0 ? cfg.batched_sessions : 1;
    if (p36_engine_open(&engine, &cfg.engine) != 0) return 1;

    const int slot_count = cfg.batched_sessions > 0 ? cfg.batched_sessions : 1;
    log_context_memory(cfg.engine.backend, cfg.ctx_size,
                       cfg.engine.prefill_chunk, slot_count,
                       cfg.engine.cache_type_k, cfg.engine.cache_type_v);

    s.engine = engine;
    s.engine_opt = cfg.engine;
    s.ctx_size = cfg.ctx_size;
    s.slot_count = slot_count;
    s.batched_mode = cfg.batched_sessions > 0;
    s.mixed_prefill_quantum = cfg.mixed_prefill_quantum;
    s.last_prefill_slot = slot_count - 1;
    s.slots = xmalloc((size_t)slot_count * sizeof(*s.slots));
    memset(s.slots, 0, (size_t)slot_count * sizeof(*s.slots));
    if (s.batched_mode) {
        s.slot_threads = xmalloc((size_t)slot_count * sizeof(*s.slot_threads));
        memset(s.slot_threads, 0, (size_t)slot_count * sizeof(*s.slot_threads));
    }
    s.default_tokens = cfg.default_tokens;
    s.default_presence_penalty = cfg.default_presence_penalty;
    s.default_frequency_penalty = cfg.default_frequency_penalty;
    s.default_presence_penalty_set = cfg.default_presence_penalty_set;
    s.default_frequency_penalty_set = cfg.default_frequency_penalty_set;
    s.default_temperature = cfg.default_temperature;
    s.default_top_k = cfg.default_top_k;
    s.default_top_p = cfg.default_top_p;
    s.default_min_p = cfg.default_min_p;
    s.default_repeat_penalty = cfg.default_repeat_penalty;
    s.default_temperature_set = cfg.default_temperature_set;
    s.default_top_k_set = cfg.default_top_k_set;
    s.default_top_p_set = cfg.default_top_p_set;
    s.default_min_p_set = cfg.default_min_p_set;
    s.default_repeat_penalty_set = cfg.default_repeat_penalty_set;
    s.disable_exact_tool_replay = cfg.disable_exact_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    if (cfg.kv_disk_dir && s.model_count > 0) {
        /* One cache directory per model: checkpoints never cross models. */
        s.kv_root = xstrdup(cfg.kv_disk_dir);
        s.kv_budget_mb = cfg.kv_disk_space_mb;
        s.kv_reject_different_quant = cfg.kv_cache_reject_different_quant;
        s.kv_opt = cfg.kv_cache;
        server_kv_open_for_model(&s, s.model_current);
    } else if (cfg.kv_disk_dir) {
        kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                      cfg.kv_cache_reject_different_quant, cfg.kv_cache);
        server_kv_set_identity(&s, cfg.engine.model_path);
    }
    if (s.disable_exact_tool_replay) {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: exact tool replay disabled; tool history uses canonical JSON rendering");
    }
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutexattr_t recursive_attr;
    pthread_mutexattr_init(&recursive_attr);
    pthread_mutexattr_settype(&recursive_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&s.kv_mu, &recursive_attr);
    pthread_mutex_init(&s.inference_mu, &recursive_attr);
    pthread_mutexattr_destroy(&recursive_attr);
    pthread_mutex_init(&s.model_mu, NULL);
    pthread_cond_init(&s.model_cv, NULL);
    {
        pthread_rwlockattr_t rwa;
        pthread_rwlockattr_init(&rwa);
#ifdef __GLIBC__
        /* A queued swap must not starve behind a steady stream of readers. */
        pthread_rwlockattr_setkind_np(&rwa, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
#endif
        pthread_rwlock_init(&s.swap_lock, &rwa);
        pthread_rwlockattr_destroy(&rwa);
    }
    pthread_mutex_init(&s.trace_mu, NULL);
    for (int i = 0; i < slot_count; i++) {
        server_slot *slot = &s.slots[i];
        slot->srv = &s;
        slot->id = i;
        if (p36_session_create(&slot->session, engine, cfg.ctx_size) != 0) {
            server_log(P36_LOG_DEFAULT,
                       "p36-server: failed to create %s session %d/%d",
                       p36_backend_name(cfg.engine.backend), i + 1, slot_count);
            server_close_resources(&s);
            return 1;
        }
    }
    s.session = s.slots[0].session;
    if (s.batched_mode) {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: batched mode resident_sessions=%d prefill_quantum=%d mixed_prefill_quantum=%d decode_coalesce_us=%ld",
                   slot_count, server_prefill_quantum_for(&s, false),
                   server_prefill_quantum_for(&s, true), server_decode_coalesce_us());
        if (mtp_requested) {
            server_log(P36_LOG_DEFAULT,
                       "p36-server: MTP is disabled while multi-session batching is active");
        }
    }
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(P36_LOG_DEFAULT, "p36-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(P36_LOG_DEFAULT, "p36-server: tracing session to %s", cfg.trace_path);
    }

    pthread_t worker = (pthread_t){0};
    int slot_threads_started = 0;
    bool decode_thread_started = false;
    if (s.batched_mode) {
        if (pthread_create(&s.decode_thread, NULL, decode_worker_main, &s) != 0)
            die("failed to start decode coordinator");
        decode_thread_started = true;
        for (int i = 0; i < slot_count; i++) {
            if (pthread_create(&s.slot_threads[i], NULL, slot_worker_main,
                               &s.slots[i]) != 0)
                die("failed to start session worker");
            slot_threads_started++;
        }
    } else if (pthread_create(&worker, NULL, worker_main, &s) != 0) {
        die("failed to start worker");
    }

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(P36_LOG_DEFAULT, "p36-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        if (s.batched_mode) {
            for (int i = 0; i < slot_threads_started; i++) pthread_join(s.slot_threads[i], NULL);
            pthread_mutex_lock(&s.model_mu);
            s.model_stopping = true;
            pthread_cond_broadcast(&s.model_cv);
            pthread_mutex_unlock(&s.model_mu);
            if (decode_thread_started) pthread_join(s.decode_thread, NULL);
        } else {
            pthread_join(worker, NULL);
        }
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(P36_LOG_DEFAULT, "p36-server: listening on http://%s:%d", cfg.host, cfg.port);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(P36_LOG_DEFAULT, "p36-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        client_arg *ca = xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_mutex_lock(&s.mu);
        s.clients++;
        pthread_mutex_unlock(&s.mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(P36_LOG_DEFAULT, "p36-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_mutex_lock(&s.model_mu);
    server_cancel_decode_waiters_locked(&s);
    pthread_cond_broadcast(&s.model_cv);
    pthread_mutex_unlock(&s.model_mu);
    if (s.batched_mode) {
        for (int i = 0; i < slot_threads_started; i++) pthread_join(s.slot_threads[i], NULL);
        pthread_mutex_lock(&s.model_mu);
        s.model_stopping = true;
        pthread_cond_broadcast(&s.model_cv);
        pthread_mutex_unlock(&s.model_mu);
        if (decode_thread_started) pthread_join(s.decode_thread, NULL);
    } else {
        pthread_join(worker, NULL);
    }
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    for (int i = 0; s.kv.enabled && i < s.slot_count; i++) {
        server_slot *slot = &s.slots[i];
        const p36_tokens *tokens = p36_session_tokens(slot->session);
        if (!tokens || tokens->len < s.kv.opt.min_tokens) continue;
        server_log(P36_LOG_KVCACHE,
                   "p36-server: persisting resident KV cache slot=%d tokens=%d",
                   i, tokens->len);
        server_kv_store_current(&s, slot, "shutdown");
    }
    server_close_resources(&s);
    return 0;
}
#else

static int test_failures = 0;

static void test_assert(bool cond, const char *file, int line, const char *expr) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    test_failures++;
}

#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)

static void test_tool_schema_order_from_anthropic_schema(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    const tool_schema_order *order = tool_schema_orders_find(&orders, "bash");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "command"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "description"));
    tool_schema_orders_free(&orders);
}

static void test_tool_schema_order_from_openai_tools(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{"
        "\"type\":\"object\",\"properties\":{"
        "\"filePath\":{\"type\":\"string\"},"
        "\"oldString\":{\"type\":\"string\"},"
        "\"newString\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"edit\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "edit");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 3);
    TEST_ASSERT(order && !strcmp(order->prop[0], "filePath"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "oldString"));
    TEST_ASSERT(order && !strcmp(order->prop[2], "newString"));
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_tool_schemas_minified_and_sorted(void) {
    const char *json =
        "[\n"
        "  {\n"
        "    \"type\": \"function\",\n"
        "    \"function\": {\n"
        "      \"name\": \"write\",\n"
        "      \"description\": \"Write file\",\n"
        "      \"parameters\": {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "          \"path\": {\"type\": \"string\"}\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  },\n"
        "  {\n"
        "    \"name\": \"bash\",\n"
        "    \"input_schema\": {\n"
        "      \"type\": \"object\",\n"
        "      \"properties\": {\n"
        "        \"command\": {\"type\": \"string\"}\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas != NULL);
    char *bash_pos = strstr(schemas, "\"name\":\"bash\"");
    char *write_pos = strstr(schemas, "\"name\":\"write\"");
    TEST_ASSERT(bash_pos != NULL);
    TEST_ASSERT(write_pos != NULL);
    TEST_ASSERT(bash_pos < write_pos);
    TEST_ASSERT(strstr(schemas, "    ") == NULL);
    TEST_ASSERT(strstr(schemas, "  ") == NULL);
    free(schemas);
    tool_schema_orders_free(&orders);
}

static tool_calls make_swapped_bash_call(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&calls, tc);
    return calls;
}

static tool_schema_orders make_bash_order(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    return orders;
}

static char *read_socket_text(int fd) {
    buf b = {0};
    char tmp[1024];
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n > 0) {
            buf_append(&b, tmp, (size_t)n);
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    return buf_take(&b);
}

static void test_context_length_error_uses_protocol_standard_shape(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.prompt.len = 16;
    TEST_ASSERT(request_exceeds_context(&r, 16));
    TEST_ASSERT(!request_exceeds_context(&r, 17));

    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], &r, 16, 16));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 400") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"code\":\"context_length_exceeded\"") != NULL);
        TEST_ASSERT(strstr(out, "\"param\":\"messages\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":16") != NULL);
        TEST_ASSERT(strstr(out, "\"n_ctx\":16") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&r);

    request a;
    request_init(&a, REQ_CHAT, 128);
    a.api = API_ANTHROPIC;

    sv[0] = sv[1] = -1;
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], &a, 20, 20));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\",\"error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":20") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&a);
}

static void test_anthropic_live_stream_sends_incremental_blocks(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = P36_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_test", 10, &st));
    const char *raw1 = "need a tool</think>Hello.\n\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], &r, "msg_test", &st,
                                            raw1, strlen(raw1), false));

    const char *raw =
        "need a tool</think>Hello.\n\n"
        P36_TOOL_CALLS_START "\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], &r, "msg_test", &st,
                                            raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], &r, "msg_test", &st,
                                          raw, strlen(raw), &calls,
                                          "tool_calls", 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *msg_start = strstr(out, "event: message_start");
    const char *thinking = strstr(out, "\"thinking\":\"need a tool\"");
    const char *signature = strstr(out, "\"type\":\"signature_delta\"");
    const char *text = strstr(out, "\"text\":\"Hello.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *stop = strstr(out, "event: message_stop");
    TEST_ASSERT(msg_start != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(signature != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(msg_start < thinking);
    TEST_ASSERT(thinking < signature);
    TEST_ASSERT(signature < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < stop);
    TEST_ASSERT(strstr(out, P36_TOOL_CALLS_START) == NULL);

    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_sends_incremental_text(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_test", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "<think>need a tool</think>Hello.\n\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw1, strlen(raw1), false));

    const char *raw =
        "<think>need a tool</think>Hello.\n\n"
        P36_TOOL_CALLS_START "\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st,
                                       raw, strlen(raw), &calls,
                                       "tool_calls", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *thinking = strstr(out, "\"reasoning_content\":\"need a tool\"");
    const char *text = strstr(out, "\"content\":\"Hello.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < thinking);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < done);
    TEST_ASSERT(strstr(out, P36_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, "<think>") == NULL);

    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_chat_stream_splits_reasoning_without_tools(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_HIGH;
    r.has_tools = false;

    TEST_ASSERT(request_uses_structured_stream(&r));
    TEST_ASSERT(request_uses_openai_live_stream(&r));
    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_title", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "We need to generate a title";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_title", &st,
                                         raw1, strlen(raw1), false));

    const char *raw2 =
        "We need to generate a title</think>Free disk space check";
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_title", &st,
                                       raw2, strlen(raw2), NULL,
                                       "stop", 12, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *reasoning1 = strstr(out, "\"reasoning_content\":\"We need to generate \"");
    const char *reasoning2 = strstr(out, "\"reasoning_content\":\"a title\"");
    const char *content = strstr(out, "\"content\":\"Free disk space check\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(reasoning1 != NULL);
    TEST_ASSERT(reasoning2 != NULL);
    TEST_ASSERT(content != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < reasoning1);
    TEST_ASSERT(reasoning1 < reasoning2);
    TEST_ASSERT(reasoning2 < content);
    TEST_ASSERT(content < done);
    TEST_ASSERT(strstr(out, "\"content\":\"We need to generate a title") == NULL);
    TEST_ASSERT(strstr(out, "</think>") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_qwen_tool_stream_sends_answer_before_finish(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_HIGH;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *partial = "<think>first pass</thi";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_second_think",
                                         &st, partial, strlen(partial), false));
    const char *complete = "<think>first pass</think>first answer chunk";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_second_think",
                                         &st, complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"reasoning_content\":") != NULL);
    TEST_ASSERT(strstr(out, "\"content\":\"first answer chunk\"") != NULL);
    TEST_ASSERT(strstr(out, "</thi") == NULL);
    TEST_ASSERT(strstr(out, "[DONE]") == NULL);
    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_anthropic_qwen_tool_stream_sends_answer_before_finish(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = P36_THINK_HIGH;
    r.has_tools = true;

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_second_think", 5, &st));
    const char *partial = "first pass</thi";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], &r, "msg_second_think",
                                            &st, partial, strlen(partial), false));
    const char *complete = "first pass</think>first answer chunk";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], &r, "msg_second_think",
                                            &st, complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"thinking\":") != NULL);
    TEST_ASSERT(strstr(out, "\"text\":\"first answer chunk\"") != NULL);
    TEST_ASSERT(strstr(out, "</thi") == NULL);
    TEST_ASSERT(strstr(out, "message_stop") == NULL);
    free(out);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_sends_partial_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_partial_tool", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        "Before.\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\necho partial";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\necho partial done" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(raw_complete, &parsed_content, &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_openai_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "call_", 5));
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                       raw_complete, strlen(raw_complete), &calls,
                                       "tool_calls", 10, 4));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"content\":\"Before.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *arguments = strstr(out, "\"arguments\":\"echo partial done\"");
    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(arguments != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < arguments);
    TEST_ASSERT(tool_id_count == 1);
    TEST_ASSERT(strstr(out, P36_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, P36_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_waits_for_incomplete_tool_tags(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_invoke = P36_TOOL_CALLS_START "\n" P36_INVOKE_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_invoke, strlen(raw_invoke), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == OPENAI_TOOL_BETWEEN_INVOKES);

    const char *raw_param =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_param, strlen(raw_param), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == OPENAI_TOOL_BETWEEN_PARAMS);

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, P36_PARAM_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_sends_partial_raw_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "edits>\n[1,2,3]\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_raw_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"name\":\"edit\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"edits\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"[1,2,3]\"") != NULL);
    TEST_ASSERT(strstr(out, P36_TOOL_CALLS_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_preserves_ampersands(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_partial =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\necho &";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_partial, strlen(raw_partial), false));

    const char *raw_complete =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\necho & done\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"echo & done\"") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_holds_partial_utf8_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char prefix[] =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "write>\n"
        P36_PARAM_START "content>\nflag ";
    const char suffix[] =
        " done" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    const char flag_utf8[] = {(char)0xf0, (char)0x9f, (char)0x9a, (char)0xa9, 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    buf partial = {0};
    buf_append(&partial, prefix, strlen(prefix));
    buf_putc(&partial, (char)0xf0);
    buf_putc(&partial, (char)0x9f);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         partial.ptr, partial.len, false));

    buf complete = {0};
    buf_append(&complete, prefix, strlen(prefix));
    buf_append(&complete, flag_utf8, 4);
    buf_append(&complete, suffix, strlen(suffix));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         complete.ptr, complete.len, false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, flag_utf8) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    buf_free(&partial);
    buf_free(&complete);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_handles_multiple_calls(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "read>\n"
        P36_PARAM_START "path>\na.c\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END "\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\nwc -l a.c\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_multi_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(tool_id_count == 2);
    TEST_ASSERT(strstr(out, "\"name\":\"read\"") != NULL);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"path\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\\\"command\\\":") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_streaming_holds_partial_utf8(void) {
    const char partial[] = {'A', ' ', (char)0xf0, (char)0x9f, 0};
    const char complete[] = {'A', ' ', (char)0xf0, (char)0x9f,
                             (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char flag_done[] = {(char)0xf0, (char)0x9f,
                              (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    TEST_ASSERT(utf8_stream_safe_len(partial, 0, strlen(partial), false) == 2);
    TEST_ASSERT(utf8_stream_safe_len(complete, 0, strlen(complete), false) == strlen(complete));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = P36_THINK_NONE;

    openai_stream st;
    openai_stream_start(&r, &st);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         partial, strlen(partial), false));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"content\":\"A \"") != NULL);
    TEST_ASSERT(strstr(out, flag_done) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_request_defaults_match_qwen_api(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    TEST_ASSERT(r.think_mode == P36_THINK_HIGH);
    TEST_ASSERT(r.temperature == P36_DEFAULT_TEMPERATURE);
    TEST_ASSERT(r.top_p == P36_DEFAULT_TOP_P);
    TEST_ASSERT(r.top_k == 0);
    TEST_ASSERT(r.min_p == P36_DEFAULT_MIN_P);
    TEST_ASSERT(r.preserve_thinking);
    TEST_ASSERT(!r.temperature_set && !r.top_p_set && !r.top_k_set && !r.min_p_set);

    float temperature, top_p, min_p;
    int top_k;
    p36_engine_sampling_defaults(NULL, &temperature, &top_k, &top_p, &min_p);
    TEST_ASSERT(temperature == P36_DEFAULT_TEMPERATURE);
    TEST_ASSERT(top_p == P36_DEFAULT_TOP_P);
    TEST_ASSERT(top_k == 0);
    TEST_ASSERT(min_p == P36_DEFAULT_MIN_P);
    request_free(&r);
}

static void test_explicit_sampling_wins_in_thinking_mode(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    float temperature, top_p, min_p;
    int top_k;

    request_sampling(&r, &temperature, &top_k, &top_p, &min_p);
    TEST_ASSERT(temperature == P36_DEFAULT_TEMPERATURE);
    TEST_ASSERT(top_p == P36_DEFAULT_TOP_P);
    TEST_ASSERT(top_k == 0);
    TEST_ASSERT(min_p == P36_DEFAULT_MIN_P);

    r.temperature = 0.0f;
    r.top_p = 0.75f;
    r.top_k = 17;
    r.min_p = 0.01f;
    r.temperature_set = r.top_p_set = r.top_k_set = r.min_p_set = true;
    request_sampling(&r, &temperature, &top_k, &top_p, &min_p);
    TEST_ASSERT(temperature == 0.0f);
    TEST_ASSERT(top_p == 0.75f);
    TEST_ASSERT(top_k == 17);
    TEST_ASSERT(min_p == 0.01f);
    request_free(&r);
}

static void test_request_peek_model(void) {
    char *m = request_peek_model("{\"model\":\"RavenX\",\"messages\":[]}");
    TEST_ASSERT(m && !strcmp(m, "RavenX"));
    free(m);
    /* A nested "model" key (tool schema) must not win over the top level. */
    m = request_peek_model("{\"tools\":[{\"function\":{\"parameters\":{\"model\":\"x\"}}}],"
                           "\"messages\":[{\"content\":\"say \\\"model\\\"\"}],\"model\":\"B\"}");
    TEST_ASSERT(m && !strcmp(m, "B"));
    free(m);
    m = request_peek_model("{\"tools\":[{\"model\":\"x\"}]}");
    TEST_ASSERT(m == NULL);
    m = request_peek_model(" { \"stream\" : true , \"model\" : \"C\" } ");
    TEST_ASSERT(m && !strcmp(m, "C"));
    free(m);
    TEST_ASSERT(request_peek_model("{\"model\":42}") == NULL);
    TEST_ASSERT(request_peek_model("not json") == NULL);
    TEST_ASSERT(request_peek_model("{\"model\":") == NULL);
    TEST_ASSERT(request_peek_model(NULL) == NULL);
}

static void test_reasoning_effort_mapping(void) {
    p36_think_mode mode = P36_THINK_NONE;
    TEST_ASSERT(parse_reasoning_effort_name("none", &mode) && mode == P36_THINK_NONE);
    TEST_ASSERT(think_mode_from_enabled(true, mode) == P36_THINK_NONE);
    TEST_ASSERT(parse_reasoning_effort_name("low", &mode) && mode == P36_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("minimal", &mode) && mode == P36_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("medium", &mode) && mode == P36_THINK_MEDIUM);
    TEST_ASSERT(parse_reasoning_effort_name("high", &mode) && mode == P36_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("xhigh", &mode) && mode == P36_THINK_XHIGH);
    TEST_ASSERT(parse_reasoning_effort_name("max", &mode) && mode == P36_THINK_MAX);
    TEST_ASSERT(parse_reasoning_effort_name("0", &mode) && mode == P36_THINK_NONE);
    TEST_ASSERT(parse_reasoning_effort_name("1", &mode) && mode == P36_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("33", &mode) && mode == P36_THINK_LOW);
    TEST_ASSERT(parse_reasoning_effort_name("34", &mode) && mode == P36_THINK_MEDIUM);
    TEST_ASSERT(parse_reasoning_effort_name("50", &mode) && mode == P36_THINK_MEDIUM);
    TEST_ASSERT(parse_reasoning_effort_name("66", &mode) && mode == P36_THINK_MEDIUM);
    TEST_ASSERT(parse_reasoning_effort_name("67", &mode) && mode == P36_THINK_XHIGH);
    TEST_ASSERT(parse_reasoning_effort_name("100", &mode) && mode == P36_THINK_XHIGH);
    TEST_ASSERT(parse_reasoning_effort_name("001", &mode) && mode == P36_THINK_LOW);
    TEST_ASSERT(think_mode_from_enabled(false, mode) == P36_THINK_NONE);
    TEST_ASSERT(!parse_reasoning_effort_name("banana", &mode));
    TEST_ASSERT(!parse_reasoning_effort_name("", &mode));
    TEST_ASSERT(!parse_reasoning_effort_name("101", &mode));
    TEST_ASSERT(!parse_reasoning_effort_name("-1", &mode));
    TEST_ASSERT(!parse_reasoning_effort_name("1.5", &mode));
    TEST_ASSERT(!parse_reasoning_effort_name("99999999999999999999", &mode));
    uint32_t min_ctx = p36_think_max_min_context();
    TEST_ASSERT(min_ctx == 98304);
    TEST_ASSERT(p36_think_mode_for_context(P36_THINK_MAX, (int)min_ctx - 1) == P36_THINK_HIGH);
    TEST_ASSERT(p36_think_mode_for_context(P36_THINK_MAX, (int)min_ctx) == P36_THINK_MAX);
    TEST_ASSERT(think_mode_from_enabled(true, P36_THINK_HIGH) == P36_THINK_HIGH);
    g_force_nothink = true;
    TEST_ASSERT(think_mode_from_enabled(true, P36_THINK_HIGH) == P36_THINK_NONE);
    TEST_ASSERT(think_mode_from_enabled(true, P36_THINK_MAX) == P36_THINK_NONE);
    g_force_nothink = false;
}

static void test_api_thinking_controls_parse(void) {
    bool enabled = true;
    int budget = 0;
    const char *thinking = "{\"type\":\"disabled\",\"budget_tokens\":1024}";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled, &budget));
    TEST_ASSERT(!enabled);
    TEST_ASSERT(budget == 1024);
    thinking = "true";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled, &budget));
    TEST_ASSERT(enabled);
    thinking = "{\"budget_tokens\":8000.5}";
    TEST_ASSERT(!parse_thinking_control_value(&thinking, &enabled, &budget));
    thinking = "{\"budget_tokens\":-1}";
    TEST_ASSERT(!parse_thinking_control_value(&thinking, &enabled, &budget));

    p36_think_mode mode = P36_THINK_HIGH;
    const char *anth_effort = "{\"effort\":\"max\",\"other\":true}";
    TEST_ASSERT(parse_output_config_effort(&anth_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_MAX);

    const char *openai_effort = "\"xhigh\"";
    mode = P36_THINK_HIGH;
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_XHIGH);
    openai_effort = "\"50\"";
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_MEDIUM);
    openai_effort = "\"0\"";
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_NONE);
    const char *responses_effort = "{\"effort\":\"none\"}";
    TEST_ASSERT(parse_responses_reasoning(&responses_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_NONE);
    responses_effort = "{\"effort\":\"100\"}";
    TEST_ASSERT(parse_responses_reasoning(&responses_effort, &mode));
    TEST_ASSERT(mode == P36_THINK_XHIGH);

    request r = {.qwen38 = true, .thinking_budget = 8000};
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_HIGH) == P36_THINK_LOW);
    r.thinking_budget = 16000;
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_HIGH) == P36_THINK_MEDIUM);
    r.thinking_budget = 24000;
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_HIGH) == P36_THINK_HIGH);
    r.thinking_budget = 24001;
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_HIGH) == P36_THINK_XHIGH);
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_NONE) == P36_THINK_NONE);
    r.qwen38 = false;
    TEST_ASSERT(think_mode_from_budget(&r, P36_THINK_HIGH) == P36_THINK_HIGH);
}

static void test_qwen38_effort_prompt(void) {
    chat_msgs msgs = {0};
    chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("system"),
                                     .content = xstrdup("Be precise.")});
    chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("user"),
                                     .content = xstrdup("Hello")});
    char *low = render_chat_prompt_text_model(&msgs, NULL, NULL,
        P36_THINK_LOW, false, true, true);
    char *medium = render_chat_prompt_text_model(&msgs, NULL, NULL,
        P36_THINK_MEDIUM, false, true, true);
    char *xhigh = render_chat_prompt_text_model(&msgs, NULL, NULL,
        P36_THINK_XHIGH, false, true, true);
    char *off = render_chat_prompt_text_model(&msgs, NULL, NULL,
        P36_THINK_NONE, false, true, true);
    char *moe = render_chat_prompt_text_model(&msgs, NULL, NULL,
        P36_THINK_LOW, false, false, true);
    TEST_ASSERT(strstr(low, "<|im_start|>system\nReasoning effort is set to low.") == low);
    TEST_ASSERT(strstr(low, "\n\nBe precise.<|im_end|>") != NULL);
    TEST_ASSERT(strstr(medium, "Reasoning effort is set to") == NULL);
    TEST_ASSERT(strstr(xhigh, "<|im_start|>system\nReasoning effort is set to xhigh.") == xhigh);
    TEST_ASSERT(strstr(off, "Reasoning effort is set to") == NULL);
    TEST_ASSERT(strstr(off, "<think>\n\n</think>\n\n") != NULL);
    TEST_ASSERT(strstr(moe, "Reasoning effort is set to") == NULL);
    free(low);
    free(medium);
    free(xhigh);
    free(off);
    free(moe);
    chat_msgs_free(&msgs);
}

static void test_render_think_max_prompt_prefix(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_MAX);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, p36_think_max_prefix()) != NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>system\nYou are terse.<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>") != NULL);
    TEST_ASSERT(strstr(prompt, "</think>") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_non_thinking_prompt_closes_think(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_NONE);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, p36_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_HIGH, false, false);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>assistant\nfirst answer<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n<think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, P36_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>\ntool reasoning\n</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_response>\n/tmp\n</tool_response>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>\ntool reasoning\n</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_response>\n/tmp\n</tool_response>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_tool_prompt_args_preserve_call_order(void) {
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_qwen_tool_calls_text(&b, &calls, NULL, false);
    const char *command = strstr(b.ptr, "<parameter=command>");
    const char *description = strstr(b.ptr, "<parameter=description>");
    const char *timeout = strstr(b.ptr, "<parameter=timeout>");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
}

static void test_openai_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_tool_calls_json(&b, &calls, "test", &r.tool_orders);
    const char *command = strstr(b.ptr, "\\\"command\\\"");
    const char *description = strstr(b.ptr, "\\\"description\\\"");
    const char *timeout = strstr(b.ptr, "\\\"timeout\\\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_anthropic_thinking_and_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_anthropic_content(&b, "done", "thinking text", &calls, "msg_1", &r.tool_orders);
    const char *thinking = strstr(b.ptr, "\"type\":\"thinking\"");
    const char *text = strstr(b.ptr, "\"type\":\"text\"");
    const char *tool = strstr(b.ptr, "\"type\":\"tool_use\"");
    const char *command = strstr(b.ptr, "\"command\"");
    const char *description = strstr(b.ptr, "\"description\"");
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_parse_short_qwen_tool_and_canonical_suffix(void) {
    const char *generated =
        "<think>need a tool</think>"
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=description>\nlist files\n</parameter>\n"
        "<parameter=command>\nls -la\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_HIGH;
    r.tool_orders = make_bash_order();
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    const char *command = strstr(suffix, "<parameter=command>");
    const char *description = strstr(suffix, "<parameter=description>");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(strstr(suffix, "</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<|im_end|>\n") != NULL);

    free(suffix);
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_qwen_tool_parser_preserves_multiline_parameters(void) {
    const char *generated =
        "review done\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "path>\n/private/tmp/tetris.c\n" P36_PARAM_END "\n"
        P36_PARAM_START "oldText>\nold <text>\nsecond line\n" P36_PARAM_END "\n"
        P36_PARAM_START "newText>\nnew text\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "review done"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\":\"/private/tmp/tetris.c\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"oldText\":\"old <text>\\nsecond line\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"newText\":\"new text\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

/* ---- Truncated tool call / reasoning semantics (chat completions) ---- */
typedef struct {
    char *content, *reasoning;
    tool_calls calls;
    const char *finish;
    bool incomplete, recovered;
} trunc_result;

static void trunc_parse(trunc_result *o, const char *text, bool thinking, bool has_tools,
                        const char *finish) {
    memset(o, 0, sizeof(*o));
    o->finish = finish;
    char err[64];
    parse_generated_message_for_response(text, thinking, has_tools, strstr(text, P36_TOOL_CALLS_START) != NULL,
                                         &o->finish, err, sizeof(err), &o->content, &o->reasoning,
                                         &o->calls, &o->recovered, &o->incomplete);
}

static void trunc_free(trunc_result *o) {
    free(o->content); free(o->reasoning); tool_calls_free(&o->calls);
}

#define TC_CALL "<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n" \
                "<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>"

static void test_truncated_generation_semantics(void) {
    trunc_result o;
    /* 1. cut inside reasoning */
    trunc_parse(&o, "I should write the file", true, true, "length");
    TEST_ASSERT(o.content && !o.content[0] && o.reasoning && !strcmp(o.reasoning, "I should write the file"));
    TEST_ASSERT(o.calls.len == 0 && !o.incomplete && !strcmp(o.finish, "length"));
    trunc_free(&o);
    /* 2a. cut exactly after </think>; 2b. cut inside the delimiter */
    trunc_parse(&o, "plan</think>", true, true, "length");
    TEST_ASSERT(o.content && !o.content[0] && o.reasoning && !strcmp(o.reasoning, "plan"));
    trunc_free(&o);
    trunc_parse(&o, "plan</thi", true, true, "length");
    TEST_ASSERT(o.content && !o.content[0] && o.reasoning && !strcmp(o.reasoning, "plan"));
    trunc_free(&o);
    /* 3. after reasoning, before the tool call starts */
    trunc_parse(&o, "plan</think>\n\n", true, true, "length");
    TEST_ASSERT(o.content && !o.content[0] && o.reasoning && !strcmp(o.reasoning, "plan"));
    TEST_ASSERT(o.calls.len == 0 && !o.incomplete);
    trunc_free(&o);
    trunc_parse(&o, "plan</think>\n\n<tool_ca", true, true, "length");
    TEST_ASSERT(o.content && !o.content[0] && o.calls.len == 0);
    trunc_free(&o);
    /* 4. halfway through the tool name; 5. halfway through the arguments */
    const char *cuts[] = {
        "plan</think>\n\n<tool_call>\n<function=wri",
        "plan</think>\n\n<tool_call>\n<function=write>\n<parameter=pa",
        "plan</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhel",
    };
    for (size_t i = 0; i < sizeof(cuts) / sizeof(cuts[0]); i++) {
        trunc_parse(&o, cuts[i], true, true, "length");
        TEST_ASSERT(o.incomplete && o.calls.len == 0);
        TEST_ASSERT(o.content && !o.content[0]);
        TEST_ASSERT(o.reasoning && !strcmp(o.reasoning, "plan"));
        TEST_ASSERT(!strcmp(o.finish, "length") && !o.recovered);
        trunc_free(&o);
    }
    /* 6. complete call that exactly fills the budget */
    trunc_parse(&o, "plan</think>\n\n" TC_CALL, true, true, "length");
    TEST_ASSERT(!o.incomplete && o.calls.len == 1 && !strcmp(o.calls.v[0].name, "write"));
    TEST_ASSERT(o.content && !o.content[0]);
    trunc_free(&o);
    /* 7. complete call followed by attempted prose: the call wins, prose is dropped */
    trunc_parse(&o, "plan</think>\n\n" TC_CALL "\n\nDone, the file is written.", true, true, "stop");
    TEST_ASSERT(o.calls.len == 1 && o.content && !o.content[0] && !o.incomplete);
    trunc_free(&o);
    /* two calls, the second cut off: keep the first, flag the loss */
    trunc_parse(&o, "plan</think>\n\n" TC_CALL "\n<tool_call>\n<function=wri", true, true, "length");
    TEST_ASSERT(o.calls.len == 1 && o.incomplete);
    trunc_free(&o);
    /* 8. action_budget abort mid-marker never leaves a fragment in content */
    trunc_parse(&o, "I will now do it. <tool_ca", false, true, "action_budget");
    TEST_ASSERT(o.content && !strcmp(o.content, "I will now do it. ") && o.calls.len == 0);
    trunc_free(&o);
    /* non-thinking request, truncated call after prose */
    trunc_parse(&o, "Sure.\n<tool_call>\n<function=wr", false, true, "length");
    TEST_ASSERT(o.incomplete && o.content && !strcmp(o.content, "Sure.") && !o.reasoning);
    trunc_free(&o);
    /* closed but malformed call keeps the legacy text recovery */
    trunc_parse(&o, "x\n<tool_call>\n<function=>\n</tool_call>", false, true, "tool_calls");
    TEST_ASSERT(o.recovered && !o.incomplete && strstr(o.content, "<tool_call>"));
    trunc_free(&o);
    /* 10. normal chat and tool-less truncation are unchanged */
    trunc_parse(&o, "think</think>\n\nHello world", true, false, "stop");
    TEST_ASSERT(o.content && !strcmp(o.content, "\n\nHello world") && !strcmp(o.reasoning, "think"));
    trunc_free(&o);
    trunc_parse(&o, "Partial answer that was cut", false, false, "length");
    TEST_ASSERT(o.content && !strcmp(o.content, "Partial answer that was cut") && !o.incomplete);
    trunc_free(&o);
    trunc_parse(&o, "Say <tool_call> literally", false, false, "length");   /* no tools: plain text */
    TEST_ASSERT(o.content && strstr(o.content, "<tool_call>"));
    trunc_free(&o);
}

/* 9. Streaming: whatever prefix the stream is cut at, no delimiter or tool
 * markup reaches user-visible content, and the final chunk flags the loss. */
static void test_truncated_stream_never_leaks_markup(void) {
    const char *full = "plan the work</think>\n\n" TC_CALL;
    size_t n = strlen(full);
    for (size_t cut = 1; cut <= n; cut += 3) {
        int sv[2];
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        if (sv[0] < 0 || sv[1] < 0) return;
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_OPENAI;
        r.stream = true;
        r.think_mode = P36_THINK_HIGH;
        r.has_tools = true;
        r.tool_orders = make_bash_order();
        openai_stream st;
        openai_stream_start(&r, &st);
        char *raw = xstrndup(full, cut);
        TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_cut", &st, raw, cut, false));
        trunc_result o;
        trunc_parse(&o, raw, true, true, "length");
        r.incomplete_tool_call = o.incomplete;
        TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_cut", &st, raw, cut,
                                           &o.calls, o.calls.len ? "tool_calls" : o.finish, 10, 4));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        /* user-visible fields only: reasoning_content may carry text, never delimiters */
        TEST_ASSERT(strstr(out, "</think>") == NULL);
        TEST_ASSERT(strstr(out, "<think>") == NULL);
        TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
        TEST_ASSERT(strstr(out, "<function=") == NULL);
        TEST_ASSERT(strstr(out, "<parameter=") == NULL);
        TEST_ASSERT(strstr(out, "\"content\":\"plan") == NULL);   /* reasoning never relabelled as content */
        bool complete = cut == n;
        TEST_ASSERT(complete || o.calls.len == 0);
        if (o.incomplete) TEST_ASSERT(strstr(out, "\"incomplete_tool_call\":true") != NULL);
        else TEST_ASSERT(strstr(out, "incomplete_tool_call") == NULL);
        if (complete) TEST_ASSERT(o.calls.len == 1 && strstr(out, "\"finish_reason\":\"tool_calls\"") != NULL);
        free(out); free(raw);
        trunc_free(&o);
        openai_stream_free(&st);
        request_free(&r);
        close(sv[0]);
        close(sv[1]);
    }
}

static void test_tool_parse_failure_returns_recoverable_finish(void) {
    const char *generated =
        "trying a tool\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START ">\n"
        P36_TOOL_CALLS_END;

    char err[128] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    const char *finish = "tool_calls";
    bool recovered = false;

    TEST_ASSERT(!parse_generated_message_for_response(generated,
                                                       false,
                                                       true,
                                                       true,
                                                       &finish,
                                                       err,
                                                       sizeof(err),
                                                       &content,
                                                       &reasoning,
                                                       &calls,
                                                       &recovered,
                                                       NULL));
    TEST_ASSERT(recovered);
    TEST_ASSERT(!strcmp(finish, "stop"));
    TEST_ASSERT(!strcmp(err, "invalid tool call"));
    TEST_ASSERT(content && strstr(content, P36_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(reasoning == NULL);
    TEST_ASSERT(calls.len == 0);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_thinking_tool_markers_are_not_executable(void) {
    const char *valid =
        "<think>reasoning " P36_TOOL_CALLS_START "</think>\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>pwd" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};

    TEST_ASSERT(parse_generated_message_ex(valid, true, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && strstr(reasoning, P36_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "bash"));
    free(content);
    free(reasoning);
    tool_calls_free(&calls);

    const char *unclosed = "reasoning " P36_TOOL_CALLS_START;
    content = reasoning = NULL;
    memset(&calls, 0, sizeof(calls));
    TEST_ASSERT(parse_generated_message_ex(unclosed, true, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    TEST_ASSERT(content && !strcmp(content, ""));
    TEST_ASSERT(reasoning && !strcmp(reasoning, unclosed));
    free(content);
    free(reasoning);
    tool_calls_free(&calls);

    const char *recovered =
        "<think>reasoning\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>pwd" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    content = reasoning = NULL;
    memset(&calls, 0, sizeof(calls));
    TEST_ASSERT(parse_generated_message_ex(recovered, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, ""));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "reasoning"));
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "bash"));
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_tool_checkpoint_suffix_is_future_prompt_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{},\"timeout\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, P36_THINK_HIGH);

    const char *generated =
        "need a tool</think>\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\ncd /tmp && git diff 2>/dev/null\n" P36_PARAM_END "\n"
        P36_PARAM_START "timeout>\n10\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(strstr(calls.v[0].arguments, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "&amp;&amp;") == NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(suffix, "&amp;&amp;") == NULL);
    TEST_ASSERT(strstr(suffix, "2&gt;/dev/null") == NULL);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, P36_THINK_HIGH);

    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, canonical.len));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_checkpoint_minifies_json_parameters(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}");
    const char *tool_schemas =
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("edit");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, P36_THINK_HIGH);

    const char *generated =
        "need edit</think>\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "path>\n/tmp/file\n" P36_PARAM_END "\n"
        P36_PARAM_START "edits>\n"
        "[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]"
        "\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("edit");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, P36_THINK_HIGH);

    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, canonical.len));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_memory_replays_sampled_qwen_tool(void) {
    const char *generated =
        "<think>need shell</think>\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "command>\nls -la\n" P36_PARAM_END "\n"
        P36_PARAM_START "timeout>\n10\n" P36_PARAM_END "\n"
        P36_PARAM_START "description>\nlist files\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls sampled = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    assign_tool_call_ids(&s, &sampled, API_OPENAI);
    TEST_ASSERT(sampled.v[0].id != NULL);
    TEST_ASSERT(!strncmp(sampled.v[0].id, "call_", 5));
    tool_memory_remember(&s, &sampled);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    tool_call tc = {0};
    tc.id = xstrdup(sampled.v[0].id);
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.disk == 0);
    TEST_ASSERT(stats.canonical == 0);
    TEST_ASSERT(stats.missing_ids == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    const char *command = strstr(prompt, "<parameter=command>");
    const char *timeout = strstr(prompt, "<parameter=timeout>");
    const char *description = strstr(prompt, "<parameter=description>");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < timeout);
    TEST_ASSERT(timeout < description);

    free(prompt);
    chat_msgs_free(&msgs);
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_memory_preserves_empty_think_per_id(void) {
    const char *generated =
        "<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n"
        "</function>\n</tool_call>";
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_calls sampled = {0};
    char *content = NULL, *reasoning = NULL;
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);
    for (int i = 0; i < 2; i++) {
        free(sampled.v[0].id);
        sampled.v[0].id = xstrdup(i ? "call_thinking" : "call_empty");
        sampled.replay_empty_think = i == 0;
        tool_memory_remember(&s, &sampled);
    }
    /* Identical tool text shares a block, but each ID retains its own prelude. */
    const char *ids[] = {"call_empty", "call_thinking", "call_unknown"};
    for (int i = 0; i < 3; i++) {
        chat_msgs msgs = {0};
        chat_msg assistant = {.role = xstrdup("assistant")};
        tool_call call = {.id = xstrdup(ids[i]), .name = xstrdup("bash"),
                          .arguments = xstrdup("{\"command\":\"pwd\"}")};
        tool_calls_push(&assistant.calls, call);
        chat_msgs_push(&msgs, assistant);
        tool_memory_attach_to_messages(&s, &msgs, NULL);
        TEST_ASSERT(msgs.v[0].calls.replay_empty_think == (i == 0));
        char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
        const char *start = strstr(prompt, "<|im_start|>assistant\n");
        TEST_ASSERT(start != NULL);
        if (start) {
            start += strlen("<|im_start|>assistant\n");
            TEST_ASSERT((strncmp(start, "<think>\n\n</think>\n\n", 19) == 0) == (i == 0));
        }
        free(prompt);
        chat_msgs_free(&msgs);
    }
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_exact_tool_replay_can_be_disabled(void) {
    const char *tool_text =
        "\n\n<tool_call>\n"
        "<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n"
        "</tool_call>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "call_disabled", tool_text);
    s.disable_exact_tool_replay = true;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disabled");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"canonical\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 123;
    TEST_ASSERT(kv_tool_map_write(&s, fp, tool_text, &bytes));
    TEST_ASSERT(bytes == 0);

    if (fp) fclose(fp);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_qwen_tool_decode_state_separates_structure_and_payload(void) {
    qwen_tool_decode_tracker tracker;
    qwen_tool_decode_tracker_init(&tracker);

    const char *prefix =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n";
    TEST_ASSERT(qwen_tool_decode_state_for_text(prefix, strlen(prefix)) ==
                QWEN_TOOL_DECODE_STRUCTURAL);
    qwen_tool_decode_tracker_update(&tracker, prefix, strlen(prefix));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_STRUCTURAL);

    const char *path_param =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "path>\n/tmp/a.py";
    TEST_ASSERT(qwen_tool_decode_state_for_text(path_param, strlen(path_param)) ==
                QWEN_TOOL_DECODE_PAYLOAD);
    qwen_tool_decode_tracker_update(&tracker, path_param, strlen(path_param));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_PAYLOAD);

    const char *path_closing =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "path>\n/tmp/a.py</";
    TEST_ASSERT(qwen_tool_decode_state_for_text(path_closing, strlen(path_closing)) ==
                QWEN_TOOL_DECODE_STRUCTURAL);
    qwen_tool_decode_tracker_update(&tracker, path_closing, strlen(path_closing));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_STRUCTURAL);

    const char *json_struct =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "edits>\n[{";
    TEST_ASSERT(qwen_tool_decode_state_for_text(json_struct, strlen(json_struct)) ==
                QWEN_TOOL_DECODE_PAYLOAD);
    qwen_tool_decode_tracker_init(&tracker);
    qwen_tool_decode_tracker_update(&tracker, json_struct, strlen(json_struct));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_PAYLOAD);

    const char *json_string =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "edits>\n[{\"newText\":\"for i in";
    TEST_ASSERT(qwen_tool_decode_state_for_text(json_string, strlen(json_string)) ==
                QWEN_TOOL_DECODE_PAYLOAD);
    qwen_tool_decode_tracker_init(&tracker);
    qwen_tool_decode_tracker_update(&tracker, json_string, strlen(json_string));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_PAYLOAD);

    const char *done =
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "edit>\n"
        P36_PARAM_START "edits>\n[]"
        P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    TEST_ASSERT(qwen_tool_decode_state_for_text(done, strlen(done)) ==
                QWEN_TOOL_DECODE_OUTSIDE);
    qwen_tool_decode_tracker_init(&tracker);
    qwen_tool_decode_tracker_update(&tracker, done, strlen(done));
    TEST_ASSERT(tracker.decode == QWEN_TOOL_DECODE_OUTSIDE);
}

static void test_tool_memory_max_ids_prunes_oldest(void) {
    const char *a_tool = "\n\n<tool_call>\n<function=bash>\n<parameter=command>\na\n</parameter>\n</function>\n</tool_call>";
    const char *b_tool = "\n\n<tool_call>\n<function=bash>\n<parameter=command>\nb\n</parameter>\n</function>\n</tool_call>";
    const char *c_tool = "\n\n<tool_call>\n<function=bash>\n<parameter=command>\nc\n</parameter>\n</function>\n</tool_call>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_mem.max_entries = 2;
    tool_memory_put(&s, "call_a", a_tool);
    tool_memory_put(&s, "call_b", b_tool);
    tool_memory_put(&s, "call_c", c_tool);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {.id = xstrdup("call_a"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_qwen_tool_call_rendering(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("get_weather");
    tc.arguments = xstrdup("{\"location\":\"Rome\",\"units\":\"celsius\"}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_qwen_tool_calls_text(&b, &calls, NULL, false);
    TEST_ASSERT(strstr(b.ptr, "<tool_call>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "</tool_call>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "<function=get_weather>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "<parameter=location>\nRome\n</parameter>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "<parameter=units>\ncelsius\n</parameter>") != NULL);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls parsed = {0};
    char *cp = xstrdup(b.ptr);
    TEST_ASSERT(parse_generated_message(cp, &content, &reasoning, &parsed));
    TEST_ASSERT(parsed.len == 1);
    TEST_ASSERT(parsed.v[0].name && !strcmp(parsed.v[0].name, "get_weather"));
    TEST_ASSERT(strstr(parsed.v[0].arguments, "\"location\":\"Rome\"") != NULL);

    free(cp);
    free(content);
    free(reasoning);
    tool_calls_free(&parsed);
    buf_free(&b);
    tool_calls_free(&calls);

    tool_calls multi = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"ls\"}");
    tool_calls_push(&multi, tc);
    tc.name = xstrdup("edit");
    tc.arguments = xstrdup("{\"filePath\":\"/tmp/a\"}");
    tool_calls_push(&multi, tc);

    b = (buf){0};
    append_qwen_tool_calls_text(&b, &multi, NULL, false);
    const char *rendered_multi = b.ptr ? b.ptr : "";
    TEST_ASSERT(strstr(rendered_multi, "<tool_call>") != NULL);
    TEST_ASSERT(strstr(rendered_multi, "</tool_call>\n<tool_call>") != NULL);
    TEST_ASSERT(strstr(rendered_multi, "</tool_call>\n\n<tool_call>") == NULL);
    int count = 0;
    for (const char *p = rendered_multi; (p = strstr(p, "<tool_call>")) != NULL; p++) count++;
    TEST_ASSERT(count == 2);

    buf_free(&b);
    tool_calls_free(&multi);

    tool_calls scalar = {0};
    tc = (tool_call){
        .name = xstrdup("odd_tool"),
        .arguments = xstrdup("[1,true,null]"),
    };
    tool_calls_push(&scalar, tc);
    append_qwen_tool_calls_text(&b, &scalar, NULL, false);
    TEST_ASSERT(strstr(b.ptr, "<function=odd_tool>\n[1,true,null]</function>") != NULL);
    buf_free(&b);
    tool_calls_free(&scalar);
}

static void test_tool_separator_whitespace_is_not_content(void) {
    const char *generated =
        "<think>need a tool</think>"
        "I will inspect the files.\n\n\n\n"
        P36_TOOL_CALLS_START "\n"
        P36_INVOKE_START "bash>\n"
        P36_PARAM_START "description>\nlist files\n" P36_PARAM_END "\n"
        P36_PARAM_START "command>\nls -la\n" P36_PARAM_END "\n"
        P36_INVOKE_END "\n"
        P36_TOOL_CALLS_END;
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message(generated, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && !strcmp(content, "I will inspect the files."));
    TEST_ASSERT(calls.len == 1);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_qwen_tool_prompt_preserves_tool_supplied_text(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo 2>&1 && echo done\",\"count\":1}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_qwen_tool_calls_text(&b, &calls, NULL, false);
    TEST_ASSERT(strstr(b.ptr, "echo 2>&1 && echo done") != NULL);
    TEST_ASSERT(strstr(b.ptr, "2&gt;&amp;1") == NULL);
    TEST_ASSERT(strstr(b.ptr, "&amp;&amp;") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    memset(&calls, 0, sizeof(calls));
    memset(&tc, 0, sizeof(tc));
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"printf native-qwen\",\"count\":1}");
    tool_calls_push(&calls, tc);

    append_qwen_tool_calls_text(&b, &calls, NULL, false);
    TEST_ASSERT(strstr(b.ptr, "printf native-qwen") != NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("<tool_call>not a real tool call");
    chat_msgs_push(&msgs, tool);
    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, P36_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_response>\n<tool_call>not a real tool call\n</tool_response>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_stop_list_parses_all_sequences(void) {
    stop_list stops = {0};
    const char *json = "[\"END\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));
    TEST_ASSERT(stops.len == 2);
    TEST_ASSERT(stops.max_len == 4);

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "hello STOP tail END", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("hello "));
    TEST_ASSERT(len == strlen("STOP"));
    TEST_ASSERT(stop_list_stream_safe_len(&stops, strlen("abcdef")) == 3);
    stop_list_clear(&stops);
    free(stops.v);
}

static void test_stop_list_streaming_holds_and_trims_stop_text(void) {
    stop_list stops = {0};
    const char *json = "[\"</END>\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));

    size_t safe = stop_list_stream_safe_len(&stops, strlen("hello </"));
    TEST_ASSERT(safe == strlen("hel"));

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "answer STOP hidden", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("answer "));
    TEST_ASSERT(len == strlen("STOP"));

    stop_list_clear(&stops);
    free(stops.v);
}

static char *test_nested_json_array(int depth) {
    buf b = {0};
    for (int i = 0; i < depth; i++) buf_putc(&b, '[');
    buf_putc(&b, '0');
    for (int i = 0; i < depth; i++) buf_putc(&b, ']');
    return buf_take(&b);
}

static void test_json_skip_has_nesting_limit(void) {
    char *ok = test_nested_json_array(JSON_MAX_NESTING);
    const char *p = ok;
    TEST_ASSERT(json_skip_value(&p));
    TEST_ASSERT(*p == '\0');
    free(ok);

    char *bad = test_nested_json_array(JSON_MAX_NESTING + 1);
    p = bad;
    TEST_ASSERT(!json_skip_value(&p));
    free(bad);
}

static void test_json_string_handles_surrogates(void) {
    const char *p = "\"paired \\ud83d\\ude80 lone \\ud83d text badlow \\ud83d\\u0041 low \\ude80\"";
    char *s = NULL;
    TEST_ASSERT(json_string(&p, &s));
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strstr(s, "paired \xf0\x9f\x9a\x80") != NULL);
    TEST_ASSERT(strstr(s, "lone \xef\xbf\xbd text") != NULL);
    TEST_ASSERT(strstr(s, "badlow \xef\xbf\xbd" "A") != NULL);
    TEST_ASSERT(strstr(s, "low \xef\xbf\xbd") != NULL);
    TEST_ASSERT(*p == '\0');
    free(s);
}

static void test_json_owned_replacement_is_atomic(void) {
    char *value = xstrdup("old");
    const char *p = "\"new\"";
    TEST_ASSERT(json_string_replace(&p, &value));
    TEST_ASSERT(!strcmp(value, "new"));

    p = "\"unterminated";
    TEST_ASSERT(!json_string_replace(&p, &value));
    TEST_ASSERT(!strcmp(value, "new"));
    free(value);

    int integer = 0;
    p = "NaN";
    TEST_ASSERT(!json_int(&p, &integer));
    p = "Infinity";
    TEST_ASSERT(!json_int(&p, &integer));

    double number = 0.0;
    p = "NaN";
    TEST_ASSERT(!json_number(&p, &number));
    p = "-Infinity";
    TEST_ASSERT(!json_number(&p, &number));
    p = "1e9999";
    TEST_ASSERT(!json_number(&p, &number));
}

static void test_chat_agent_budget_fields_parse(void) {
    request r;
    char err[160];
    /* Malformed extension values are rejected; valid ones are exercised live. */
    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[],\"action_budget\":\"x\"}", 128, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[],\"stop_after_tool_call\":3}", 128, 4096, &r, err, sizeof(err)));
}

static void test_api_parsers_reject_nonfinite_numbers(void) {
    request r;
    char err[160];

    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[],\"temperature\":NaN}", 128, 4096,
        &r, err, sizeof(err)));
    TEST_ASSERT(!parse_responses_request(NULL, NULL,
        "{\"input\":[],\"top_p\":Infinity}", 128, 4096,
        &r, err, sizeof(err)));
    TEST_ASSERT(!parse_anthropic_request(NULL, NULL,
        "{\"messages\":[],\"presence_penalty\":-Infinity}", 128, 4096,
        &r, err, sizeof(err)));
    TEST_ASSERT(!parse_completion_request(NULL, NULL,
        "{\"prompt\":\"hello\",\"frequency_penalty\":1e9999}", 128, 4096,
        &r, err, sizeof(err)));
}

static void test_api_parsers_reject_malformed_duplicate_strings(void) {
    request r;
    char err[160];

    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[],\"model\":\"first\",\"model\":\"unterminated}",
        128, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_responses_request(NULL, NULL,
        "{\"input\":[],\"instructions\":\"first\",\"instructions\":\"unterminated}",
        128, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_anthropic_request(NULL, NULL,
        "{\"messages\":[],\"system\":\"first\",\"system\":\"unterminated}",
        128, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_completion_request(NULL, NULL,
        "{\"prompt\":\"hello\",\"model\":\"first\",\"model\":\"unterminated}",
        128, 4096, &r, err, sizeof(err)));
}

static void test_model_metadata_clamps_completion_to_context(void) {
    buf b = {0};
    append_model_json_values(&b, 32768, 262144);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":32768") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":32768") != NULL);
    buf_free(&b);

    append_model_json_values(&b, 100000, 4096);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":100000") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":4096") != NULL);
    buf_free(&b);
}

static void test_gguf_counts_are_rejected_before_allocation(void) {
    char path[] = "/tmp/p36-bad-counts.XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) return;
    uint8_t header[32] = {0};
    memcpy(header, "GGUF", 4);
    header[4] = 3;
    memset(header + 16, 0xff, 8);
    TEST_ASSERT(write(fd, header, sizeof(header)) == (ssize_t)sizeof(header));
    close(fd);

    pid_t pid = fork();
    TEST_ASSERT(pid >= 0);
    if (pid == 0) {
        p36_engine *engine = NULL;
        p36_engine_options opt = {
            .model_path = path,
            .backend = P36_BACKEND_CPU,
        };
        int rc = p36_engine_open(&engine, &opt);
        p36_engine_close(engine);
        _exit(rc == 0 ? 0 : 1);
    }
    if (pid > 0) {
        int status = 0;
        TEST_ASSERT(waitpid(pid, &status, 0) == pid);
        TEST_ASSERT(WIFEXITED(status));
        TEST_ASSERT(WEXITSTATUS(status) != 0);
    }
    if (access("./gguf-tools/qwen36-quantize", X_OK) == 0) {
        pid = fork();
        TEST_ASSERT(pid >= 0);
        if (pid == 0) {
            execl("./gguf-tools/qwen36-quantize", "qwen36-quantize",
                  "--in", path, "--allow-synthetic-imatrix", "--dry-run",
                  (char *)NULL);
            _exit(127);
        }
        if (pid > 0) {
            int status = 0;
            TEST_ASSERT(waitpid(pid, &status, 0) == pid);
            TEST_ASSERT(WIFEXITED(status));
            TEST_ASSERT(WEXITSTATUS(status) != 0);
        }
    }
    unlink(path);
}

static void test_client_socket_nonblocking_flag(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    set_client_socket_nonblocking(sv[0]);
    int flags = fcntl(sv[0], F_GETFL, 0);
    TEST_ASSERT(flags >= 0);
    TEST_ASSERT((flags & O_NONBLOCK) != 0);
    close(sv[0]);
    close(sv[1]);
}

static void test_thinking_state_tracks_prompt_and_generated_tags(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_HIGH;
    r.prompt_text = xstrdup("<|im_start|>assistant\n<think>");
    thinking_state st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "reasoning body", strlen("reasoning body"));
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "</thi", strlen("</thi"));
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "nk>answer", strlen("nk>answer"));
    TEST_ASSERT(st.inside == false);
    thinking_state_feed(&st, "<thi", strlen("<thi"));
    TEST_ASSERT(st.inside == false);
    thinking_state_feed(&st, "nk>more", strlen("nk>more"));
    TEST_ASSERT(st.inside == true);
    request_free(&r);

    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_NONE;
    r.prompt_text = xstrdup("<|im_start|>assistant\n</think>");
    st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == false);
    request_free(&r);
}

static void test_strip_assistant_reasoning_blocks(void) {
    const char *terminated =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    buf out = {0};
    strip_assistant_reasoning_blocks(&out, terminated, strlen(terminated));
    TEST_ASSERT(out.ptr && !strcmp(out.ptr,
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n"));
    buf_free(&out);

    const char *truncated =
        "<|im_start|>assistant\n<think>\npartial reasoning<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    out = (buf){0};
    strip_assistant_reasoning_blocks(&out, truncated, strlen(truncated));
    TEST_ASSERT(out.ptr && !strcmp(out.ptr,
        "<|im_start|>assistant\n<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n"));
    buf_free(&out);

    const char *plain = "<|im_start|>assistant\nplain<|im_end|>\n";
    out = (buf){0};
    strip_assistant_reasoning_blocks(&out, plain, strlen(plain));
    TEST_ASSERT(out.ptr && !strcmp(out.ptr, plain));
    buf_free(&out);
}

static void test_thinking_checkpoint_canonicalization_gate(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = P36_THINK_HIGH;
    thinking_state st = {.inside = true};

    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "stop"));

    st.inside = false;
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(should_canonicalize_thinking_checkpoint(&r, &st, "stop"));

    r.prompt_preserves_reasoning = true;
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "stop"));
    r.prompt_preserves_reasoning = false;
    r.has_tools = true;
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "stop"));
    r.has_tools = false;
    r.think_mode = P36_THINK_NONE;
    /* Preserved reasoning (the default) keeps the empty think block in history,
     * so the live graph already matches the next render: nothing to canonicalize. */
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "stop"));
    r.preserve_thinking = false;
    TEST_ASSERT(should_canonicalize_thinking_checkpoint(&r, &st, "stop"));
    TEST_ASSERT(!should_canonicalize_thinking_checkpoint(&r, &st, "length"));

    request_free(&r);
}

static void test_tool_marker_state_ignores_orphan_end(void) {
    bool start = false, end = false, orphan = false;
    qwen_tool_decode_tracker tracker;
    qwen_tool_decode_tracker_init(&tracker);
    observe_tool_markers(&tracker, "reasoning </tool_call>", &start, &end, &orphan);
    TEST_ASSERT(!start && !end && orphan);
    const char *raw = "<tool_call><function=bash></function></tool_call>";
    for (size_t n = 1; n <= strlen(raw); n++)
        qwen_tool_decode_tracker_update(&tracker, raw, n);
    orphan = false;
    observe_tool_markers(&tracker, raw, &start, &end, &orphan);
    TEST_ASSERT(start && end && !orphan);
}
static void test_canonical_rewrite_rebuilds_when_live_tail_changes(void) {
    /* Regression for the first canonical-KV rewrite attempt: replacing a small
     * live suffix looks tempting because the raw SWA ring may still contain the
     * needed rows, but compressed KV counters and compressor/indexer frontiers
     * are already past the shared prefix.  Until those graph frontiers can be
     * restored exactly, every rewrite behind the live end must rebuild or load a
     * disk checkpoint. */
    TEST_ASSERT(p36_session_rewrite_requires_rebuild(19296, 19290, 19081));
    TEST_ASSERT(p36_session_rewrite_requires_rebuild(1024, 1030, 1000));
    TEST_ASSERT(p36_session_rewrite_requires_rebuild(1024, 900, 900));

    TEST_ASSERT(!p36_session_rewrite_requires_rebuild(1024, 1024, 1024));
    TEST_ASSERT(!p36_session_rewrite_requires_rebuild(1024, 1100, 1024));
}

static void test_kv_key_includes_identity(void) {
    char plain[41], a1[41], a2[41], b[41], legacy[41];
    sha1_bytes_hex("system text", 11, plain);
    kv_cache_text_sha("", "system text", 11, legacy);
    TEST_ASSERT(!strcmp(plain, legacy));                        /* empty identity = legacy key */
    kv_cache_text_sha("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "system text", 11, a1);
    kv_cache_text_sha("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "system text", 11, a2);
    kv_cache_text_sha("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "system text", 11, b);
    TEST_ASSERT(!strcmp(a1, a2));                               /* deterministic */
    TEST_ASSERT(strcmp(a1, b) != 0 && strcmp(a1, plain) != 0);  /* other model => other key */
    kv_cache_text_sha("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "system texu", 11, b);
    TEST_ASSERT(strcmp(a1, b) != 0);                            /* other prompt/tools => other key */
}

static void write_fake_model(const char *path, size_t size, size_t flip_at, bool flip) {
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;
    for (size_t i = 0; i < size; i++) {
        unsigned char c = (unsigned char)((i * 2654435761u) >> 13);
        if (flip && i == flip_at) c ^= 0xff;
        fputc(c, fp);
    }
    fclose(fp);
}

static void test_kv_model_fingerprint_detects_changes(void) {
    char dir[] = "/tmp/p36fpXXXXXX";
    TEST_ASSERT(mkdtemp(dir) != NULL);
    char m1[128], m2[128], m3[128], m4[128], st1[128], st2[128];
    snprintf(m1, sizeof(m1), "%s/a.gguf", dir); snprintf(m2, sizeof(m2), "%s/b.gguf", dir);
    snprintf(m3, sizeof(m3), "%s/c.gguf", dir); snprintf(m4, sizeof(m4), "%s/d.gguf", dir);
    snprintf(st1, sizeof(st1), "%s/s1", dir); snprintf(st2, sizeof(st2), "%s/s2", dir);
    const size_t size = 18u << 20;
    write_fake_model(m1, size, 0, false);
    write_fake_model(m2, size, 5000, true);              /* metadata region differs */
    write_fake_model(m3, size, size - 10, true);         /* tail differs */
    write_fake_model(m4, size + 1024, 0, false);         /* size differs */
    char f1[41], f1b[41], f2[41], f3[41], f4[41], fs1[41], fs2[41];
    TEST_ASSERT(kv_model_fingerprint(m1, NULL, f1) && kv_model_fingerprint(m1, NULL, f1b));
    TEST_ASSERT(!strcmp(f1, f1b));
    TEST_ASSERT(kv_model_fingerprint(m2, NULL, f2) && strcmp(f1, f2) != 0);
    TEST_ASSERT(kv_model_fingerprint(m3, NULL, f3) && strcmp(f1, f3) != 0);
    TEST_ASSERT(kv_model_fingerprint(m4, NULL, f4) && strcmp(f1, f4) != 0);
    FILE *fp = fopen(st1, "wb"); fputs("vector-one", fp); fclose(fp);
    fp = fopen(st2, "wb"); fputs("vector-two", fp); fclose(fp);
    TEST_ASSERT(kv_model_fingerprint(m1, st1, fs1) && kv_model_fingerprint(m1, st2, fs2));
    TEST_ASSERT(strcmp(fs1, fs2) != 0 && strcmp(fs1, f1) != 0);   /* steering is part of the identity */
    TEST_ASSERT(!kv_model_fingerprint("/nonexistent/model.gguf", NULL, f1));
    unlink(m1); unlink(m2); unlink(m3); unlink(m4); unlink(st1); unlink(st2); rmdir(dir);
}

static void test_system_boundary_tokens(void) {
    enum { START = 1, SYS = 2, END = 3, NL = 4, USER = 5, X = 9 };
    int ok[] = {START, SYS, NL, X, X, X, END, NL, START, USER, NL, X};
    p36_tokens t = {.v = ok, .len = (int)(sizeof(ok) / sizeof(ok[0])), .cap = 12};
    TEST_ASSERT(system_boundary_tokens(&t, START, SYS, END, NL) == 8);   /* up to and incl. "<|im_end|>\n" */
    int no_sys[] = {START, USER, NL, X, X, END, NL, X};
    t.v = no_sys; t.len = 8;
    TEST_ASSERT(system_boundary_tokens(&t, START, SYS, END, NL) == 0);   /* first turn is not system */
    int only_sys[] = {START, SYS, NL, X, X, END, NL};
    t.v = only_sys; t.len = 7;
    TEST_ASSERT(system_boundary_tokens(&t, START, SYS, END, NL) == 0);   /* nothing follows: no checkpoint */
    int open_sys[] = {START, SYS, NL, X, X, X, X, X};
    t.v = open_sys; t.len = 8;
    TEST_ASSERT(system_boundary_tokens(&t, START, SYS, END, NL) == 0);   /* unterminated */
    int nested[] = {START, SYS, NL, X, START, USER, END, NL, X};
    t.v = nested; t.len = 9;
    TEST_ASSERT(system_boundary_tokens(&t, START, SYS, END, NL) == 0);   /* malformed: new turn before end */
}

static void test_kv_cache_store_len_uses_configured_boundary(void) {
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    TEST_ASSERT(kv_cache_store_len(&kc, 11011) == 10240);
    TEST_ASSERT(kv_cache_store_len(&kc, 1695) == 1695);

    kc.opt.boundary_trim_tokens = 0;
    kc.opt.boundary_align_tokens = 1000;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3000);

    kc.opt.boundary_align_tokens = 0;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3500);
}

static void test_kv_quant_bits(void) {
    TEST_ASSERT(kv_quant_bits_valid(1));
    TEST_ASSERT(kv_quant_bits_valid(3));
    TEST_ASSERT(kv_quant_bits_valid(8));
    TEST_ASSERT(!kv_quant_bits_valid(0));
    TEST_ASSERT(!kv_quant_bits_valid(7));
}

static void test_kv_cache_continued_uses_aligned_frontiers(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10239) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 4096;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 24576;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30720) == 30720);

    kc.continued_last_store_tokens = 10240;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 18432) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20480) == 20480);

    kc.opt.boundary_align_tokens = 0;
    kc.continued_last_store_tokens = 20480;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 29999) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30000) == 30000);
}

static void test_sha1_bytes_hex_matches_known_vector(void) {
    char sha[41];
    sha1_bytes_hex("abc", 3, sha);
    TEST_ASSERT(!strcmp(sha, "a9993e364706816aba3e25717850c26c9cd0d89d"));
}

static void test_kv_stub_file(const char *dir, const char *sha,
                              uint8_t reason, uint32_t tokens, uint32_t hits,
                              uint64_t last_used, uint64_t payload_bytes) {
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, reason, 0, tokens, hits, 32768, 100, last_used, payload_bytes);
    uint8_t text_len[4] = {0};
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}

static void test_kv_cache_entry_can_be_touched_twice(void) {
    char tmpl[] = "/tmp/p36-kv-retain-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sha = "1111111111111111111111111111111111111111";
    test_kv_stub_file(dir, sha, KV_REASON_COLD, 512, 0, 100, 0);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);

    TEST_ASSERT(kv_cache_touch_file(path, 1));
    TEST_ASSERT(access(path, F_OK) == 0);
    TEST_ASSERT(kv_cache_touch_file(path, 2));
    TEST_ASSERT(access(path, F_OK) == 0);

    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        kv_entry e = {0};
        uint32_t text_bytes = 0;
        TEST_ASSERT(kv_read_header(fp, &e, &text_bytes));
        TEST_ASSERT(e.hits == 2);
        fclose(fp);
    }

    unlink(path);
    free(path);
    rmdir(dir);
}

static void test_kv_text_stub_file(const char *dir, const char *text,
                                   uint32_t tokens, uint64_t payload_bytes) {
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, KV_REASON_COLD, 0, tokens, 0, 32768, 100, 100, payload_bytes);
    uint8_t text_len[4];
    le_put32(text_len, (uint32_t)strlen(text));
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}

static void test_kv_cache_lookup_uses_longest_text_prefix(void) {
    char tmpl[] = "/tmp/p36-kv-text-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *short_text = "transcript prefix";
    const char *long_text = "transcript prefix with sampled token bytes";
    test_kv_text_stub_file(dir, short_text, 512, 0);
    test_kv_text_stub_file(dir, long_text, 768, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    int idx = kv_cache_find_text_prefix(&kc,
        "transcript prefix with sampled token bytes and suffix",
        2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].tokens == 768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].text_bytes == strlen(long_text));
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, "transcript prefiX", 2, 32768) < 0);

    kv_cache_close(&kc);
    char short_sha[41], long_sha[41];
    sha1_bytes_hex(short_text, strlen(short_text), short_sha);
    sha1_bytes_hex(long_text, strlen(long_text), long_sha);
    char short_name[44], long_name[44];
    snprintf(short_name, sizeof(short_name), "%.40s.kv", short_sha);
    snprintf(long_name, sizeof(long_name), "%.40s.kv", long_sha);
    char *short_path = path_join(dir, short_name);
    char *long_path = path_join(dir, long_name);
    unlink(short_path);
    unlink(long_path);
    free(short_path);
    free(long_path);
    rmdir(dir);
}

static void test_kv_cache_lookup_matches_stripped_reasoning(void) {
    char tmpl[] = "/tmp/p36-kv-strip-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *stored =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer";
    const char *stripped =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nanswer";
    const char *plain = "<|im_start|>assistant\nno reasoning";
    test_kv_text_stub_file(dir, stored, 512, 0);
    test_kv_text_stub_file(dir, plain, 512, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    char drop_prompt[256];
    snprintf(drop_prompt, sizeof(drop_prompt), "%s<|im_start|>user\nnext", stripped);
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, drop_prompt, 2, 32768) < 0);
    int idx = kv_cache_find_stripped_prefix(&kc, drop_prompt, 2, 32768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].tokens == 512);

    char resend_prompt[256];
    snprintf(resend_prompt, sizeof(resend_prompt), "%s<|im_start|>user\nnext", stored);
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, resend_prompt, 2, 32768) >= 0);

    /* A stored text without reasoning has no stripped form to match. */
    char plain_prompt[128];
    snprintf(plain_prompt, sizeof(plain_prompt), "%s<|im_start|>user\nnext", plain);
    TEST_ASSERT(kv_cache_find_stripped_prefix(&kc, plain_prompt, 2, 32768) < 0);

    kv_cache_close(&kc);
    char sha[41];
    char name[44];
    sha1_bytes_hex(stored, strlen(stored), sha);
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    sha1_bytes_hex(plain, strlen(plain), sha);
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}

static void test_kv_cache_stripped_lookup_rejects_open_generation_prompt(void) {
    char tmpl[] = "/tmp/p36-kv-open-gen-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    test_kv_text_stub_file(dir, cold, 512, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    const char *prompt =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
        "<|im_start|>user\nnext";
    TEST_ASSERT(kv_cache_find_stripped_prefix(&kc, prompt, 2, 32768) < 0);

    kv_cache_close(&kc);
    char sha[41];
    char name[44];
    sha1_bytes_hex(cold, strlen(cold), sha);
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}

static void test_kv_tool_map_filters_by_qwen_tool_text(void) {
    const char *qwen_tool_keep =
        "\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "</function>\n</tool_call>";
    const char *qwen_tool_drop =
        "\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\nzzzz\n</parameter>\n"
        "</function>\n</tool_call>";

    server src = {0}, dst = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    pthread_mutex_init(&dst.tool_mu, NULL);
    tool_memory_put(&src, "call_invalid", "<tool_call>{}</tool_call>");
    TEST_ASSERT(!tool_memory_has_id(&src, "call_invalid"));
    tool_memory_put(&src, "call_keep", qwen_tool_keep);
    tool_memory_put(&src, "call_drop", qwen_tool_drop);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 0;
    TEST_ASSERT(kv_tool_map_write(&src, fp, qwen_tool_keep, &bytes));
    TEST_ASSERT(bytes > 0);
    rewind(fp);
    TEST_ASSERT(kv_tool_map_load_from_pos(&dst, fp, NULL) == 1);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call keep = {.id = xstrdup("call_keep"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, keep);
    chat_msgs_push(&msgs, a);
    chat_msg b = {0};
    b.role = xstrdup("assistant");
    tool_call drop = {.id = xstrdup("call_drop"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&b.calls, drop);
    chat_msgs_push(&msgs, b);
    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&dst, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text != NULL);
    TEST_ASSERT(msgs.v[1].calls.raw_tool_text == NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_tool_text, "pwd") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_tool_text, "zzzz") == NULL);

    chat_msgs_free(&msgs);
    if (fp) fclose(fp);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
}

static void test_tool_id_set_handles_large_history(void) {
    enum { CALLS = 4096 };
    chat_msgs msgs = {0};
    chat_msg msg = {.role = xstrdup("assistant")};
    char id[64];

    for (int i = 0; i < CALLS; i++) {
        snprintf(id, sizeof(id), "call_%d", i);
        tool_call call = {
            .id = xstrdup(id),
            .name = xstrdup("tool"),
            .arguments = xstrdup("{}"),
        };
        tool_calls_push(&msg.calls, call);
    }
    chat_msgs_push(&msgs, msg);

    id_set ids = {0};
    id_set_init(&ids);
    collect_tool_call_ids(&msgs, &ids);
    TEST_ASSERT(raxSize(ids.ids) == CALLS);
    for (int i = 0; i < CALLS; i++) {
        snprintf(id, sizeof(id), "call_%d", i);
        TEST_ASSERT(id_set_contains(&ids, id));
    }
    id_set_add(&ids, "call_0");
    TEST_ASSERT(raxSize(ids.ids) == CALLS);

    id_set_free(&ids);
    chat_msgs_free(&msgs);
}

static void test_kv_tool_map_restores_before_prompt_render(void) {
    char tmpl[] = "/tmp/p36-kv-tool-map-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sha = "3333333333333333333333333333333333333333";
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    const char *qwen_tool =
        "\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\necho exact\n</parameter>\n"
        "</function>\n</tool_call>";
    const char *text = qwen_tool;

    server src = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    tool_memory_put(&src, "call_disk", qwen_tool);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_CONTINUED, KV_EXT_TOOL_MAP, 512, 0, 32768, 100, 100, 0);
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        uint64_t ignored = 0;
        TEST_ASSERT(kv_tool_map_write(&src, fp, qwen_tool, &ignored));
        TEST_ASSERT(fclose(fp) == 0);
    }

    server dst = {0};
    pthread_mutex_init(&dst.tool_mu, NULL);
    dst.kv.enabled = true;
    dst.kv.dir = xstrdup(dir);
    dst.kv.opt = kv_cache_default_options();

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disk");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo canonical\"}");
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    kv_cache_restore_tool_memory_for_messages(&dst, &msgs);
    TEST_ASSERT(access(path, F_OK) == 0);
    kv_cache_restore_tool_memory_for_messages(&dst, &msgs);
    TEST_ASSERT(access(path, F_OK) == 0);
    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&dst, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text != NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "echo exact") != NULL);
    TEST_ASSERT(strstr(prompt, "echo canonical") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    kv_cache_close(&dst.kv);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
    unlink(path);
    free(path);
    rmdir(dir);
}

static void test_kv_cache_eviction_values_fresh_snapshots(void) {
    char tmpl[] = "/tmp/p36-kv-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    test_kv_stub_file(dir, old_sha, KV_REASON_UNKNOWN, 512, 0, 100, 4096);
    test_kv_stub_file(dir, new_sha, KV_REASON_UNKNOWN, 2048, 0, 200, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

static void test_kv_cache_eviction_keeps_aligned_continued_frontiers(void) {
    char tmpl[] = "/tmp/p36-kv-live-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 512, 0, 200, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, 300, 2048);

    char cold_name[44], continued_name[44];
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *cold_path = path_join(dir, cold_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL);

    TEST_ASSERT(access(cold_path, F_OK) != 0);
    TEST_ASSERT(access(continued_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(cold_path);
    unlink(continued_path);
    free(cold_path);
    free(continued_path);
    rmdir(dir);
}

static void test_thinking_checkpoint_canonical_matches_future_prompt(void) {
    /* Simulate: user sends a single message, thinking mode on, no tools.
     * Model generates reasoning + content.  The next request will drop the
     * reasoning from this turn.  Verify that:
     *   prompt_text[:-len("<think>")] + "</think>" + content + "<|im_end|>\n"
     * equals what render_chat_prompt_text produces for the history. */

    chat_msgs prefix_msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&prefix_msgs, user1);

    /* This is what prompt_text looks like for the first generation */
    char *prompt_text = render_chat_prompt_text_profile(
        &prefix_msgs, NULL, NULL, P36_THINK_HIGH, false, false);
    /* prompt_text should end with <think> and its canonical newline. */
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= 8);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "<think>\n", 8));

    /* The model generates: reasoning + </think> + content */
    const char *reasoning = "Let me think... 2+2 = 4";
    const char *content = "The answer is 4.";

    /* Build the canonical checkpoint text (what we'd produce after canonicalization) */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 8);
    buf_puts(&canonical, content);
    buf_puts(&canonical, "<|im_end|>\n");

    /* Now build what the NEXT request would render: history includes this
     * assistant message, plus a new user message.  Extract just the prefix
     * up to and including the eos of the assistant turn. */
    chat_msgs history_msgs = {0};
    chat_msg h_user1 = {0};
    h_user1.role = xstrdup("user");
    h_user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&history_msgs, h_user1);
    chat_msg h_asst = {0};
    h_asst.role = xstrdup("assistant");
    h_asst.reasoning = xstrdup(reasoning);
    h_asst.content = xstrdup(content);
    chat_msgs_push(&history_msgs, h_asst);
    chat_msg h_user2 = {0};
    h_user2.role = xstrdup("user");
    h_user2.content = xstrdup("Thanks!");
    chat_msgs_push(&history_msgs, h_user2);

    char *future_prompt = render_chat_prompt_text_profile(
        &history_msgs, NULL, NULL, P36_THINK_HIGH, false, false);

    /* The future prompt should START with our canonical text */
    size_t clen = canonical.len;
    TEST_ASSERT(strlen(future_prompt) > clen);
    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, clen));

    /* And what comes after is the new user turn + assistant prefix */
    const char *rest = future_prompt + clen;
    TEST_ASSERT(strstr(rest, "Thanks!") != NULL);
    TEST_ASSERT(strstr(rest, "<think>") != NULL);  /* new turn starts thinking */

    /* Verify reasoning is NOT in the future prompt for this turn */
    const char *asst_turn = strstr(future_prompt, "<|im_start|>assistant\n");
    TEST_ASSERT(asst_turn != NULL);
    TEST_ASSERT(strstr(future_prompt, reasoning) == NULL);  /* reasoning dropped */

    free(future_prompt);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&prefix_msgs);
    chat_msgs_free(&history_msgs);
}

static void test_thinking_canonical_empty_content(void) {
    /* Edge case: model thinks but produces empty content (e.g. tool-less
     * thinking where answer is entirely in reasoning).  Canonical should
     * still be valid: prompt_text[:-7] + "</think><|im_end|>\n" */
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Think about life");
    chat_msgs_push(&msgs, user);

    char *prompt_text = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_HIGH, false, false);
    size_t pt_len = strlen(prompt_text);

    /* Build canonical with empty content */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 8);
    /* empty content */
    buf_puts(&canonical, "<|im_end|>\n");

    /* Future prompt with empty content assistant message */
    chat_msgs history = {0};
    chat_msg h_u = {0};
    h_u.role = xstrdup("user");
    h_u.content = xstrdup("Think about life");
    chat_msgs_push(&history, h_u);
    chat_msg h_a = {0};
    h_a.role = xstrdup("assistant");
    h_a.reasoning = xstrdup("Deep thoughts about existence...");
    h_a.content = xstrdup("");
    chat_msgs_push(&history, h_a);
    chat_msg h_u2 = {0};
    h_u2.role = xstrdup("user");
    h_u2.content = xstrdup("Continue");
    chat_msgs_push(&history, h_u2);

    char *future = render_chat_prompt_text_profile(
        &history, NULL, NULL, P36_THINK_HIGH, false, false);
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));
    /* reasoning dropped */
    TEST_ASSERT(strstr(future, "Deep thoughts") == NULL);

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_multi_turn(void) {
    /* Multi-turn: 3 user messages, 2 assistant responses with reasoning.
     * Both prior assistant turns should have reasoning dropped.
     * The canonical after the SECOND generation should produce text that
     * matches the start of a 3rd-turn future prompt. */
    chat_msgs turn2_prefix = {0};
    chat_msg u1 = {0};
    u1.role = xstrdup("user");
    u1.content = xstrdup("Hello");
    chat_msgs_push(&turn2_prefix, u1);
    chat_msg a1 = {0};
    a1.role = xstrdup("assistant");
    a1.reasoning = xstrdup("first reasoning");
    a1.content = xstrdup("Hi there");
    chat_msgs_push(&turn2_prefix, a1);
    chat_msg u2 = {0};
    u2.role = xstrdup("user");
    u2.content = xstrdup("How are you?");
    chat_msgs_push(&turn2_prefix, u2);

    /* prompt_text for the 2nd generation (includes 1st assistant turn) */
    char *prompt_text = render_chat_prompt_text_profile(
        &turn2_prefix, NULL, NULL, P36_THINK_HIGH, false, false);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "<think>\n", 8));

    /* 1st turn reasoning is already dropped in this prompt_text */
    TEST_ASSERT(strstr(prompt_text, "first reasoning") == NULL);
    TEST_ASSERT(strstr(prompt_text, "Hi there") != NULL);

    /* After 2nd generation: canonical drops 2nd reasoning too */
    const char *content2 = "I'm doing well";
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 8);
    buf_puts(&canonical, content2);
    buf_puts(&canonical, "<|im_end|>\n");

    /* Future: 3rd user message arrives */
    chat_msgs future_msgs = {0};
    chat_msg fu1 = {0}; fu1.role = xstrdup("user"); fu1.content = xstrdup("Hello");
    chat_msgs_push(&future_msgs, fu1);
    chat_msg fa1 = {0}; fa1.role = xstrdup("assistant");
    fa1.reasoning = xstrdup("first reasoning");
    fa1.content = xstrdup("Hi there");
    chat_msgs_push(&future_msgs, fa1);
    chat_msg fu2 = {0}; fu2.role = xstrdup("user"); fu2.content = xstrdup("How are you?");
    chat_msgs_push(&future_msgs, fu2);
    chat_msg fa2 = {0}; fa2.role = xstrdup("assistant");
    fa2.reasoning = xstrdup("second reasoning");
    fa2.content = xstrdup(content2);
    chat_msgs_push(&future_msgs, fa2);
    chat_msg fu3 = {0}; fu3.role = xstrdup("user"); fu3.content = xstrdup("Great");
    chat_msgs_push(&future_msgs, fu3);

    char *future = render_chat_prompt_text_profile(
        &future_msgs, NULL, NULL, P36_THINK_HIGH, false, false);
    /* Both reasonings dropped */
    TEST_ASSERT(strstr(future, "first reasoning") == NULL);
    TEST_ASSERT(strstr(future, "second reasoning") == NULL);
    /* Canonical is a prefix of future */
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&turn2_prefix);
    chat_msgs_free(&future_msgs);
}

static void test_thinking_canonical_with_tools_preserves_reasoning(void) {
    /* When tools ARE present, reasoning is preserved in re-render.
     * The thinking canonicalization should NOT fire (has_tools gate),
     * and the tool canonicalization handles it.  Verify the template
     * preserves reasoning when tool_context is true. */
    const char *tool_schemas = "{\"name\":\"bash\"}";

    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("run ls");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, tool_schemas, NULL, P36_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "<think>\n", 8));

    /* With tools, next render KEEPS reasoning */
    chat_msgs history = {0};
    chat_msg hu = {0}; hu.role = xstrdup("user"); hu.content = xstrdup("run ls");
    chat_msgs_push(&history, hu);
    chat_msg ha = {0}; ha.role = xstrdup("assistant");
    ha.reasoning = xstrdup("I should run bash");
    ha.content = xstrdup("Here you go");
    chat_msgs_push(&history, ha);
    chat_msg hu2 = {0}; hu2.role = xstrdup("user"); hu2.content = xstrdup("thanks");
    chat_msgs_push(&history, hu2);

    char *future = render_chat_prompt_text(&history, tool_schemas, NULL, P36_THINK_HIGH);
    /* Reasoning IS preserved when tools present */
    TEST_ASSERT(strstr(future, "I should run bash") != NULL);
    TEST_ASSERT(strstr(future, "<think>\nI should run bash\n</think>") != NULL);

    free(future);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_non_thinking_mode_noop(void) {
    /* When thinking is disabled via the explicit non-thinking alias, prompt_text ends with
     * </think> not <think>.  The canonicalization should be a no-op
     * (early return on memcmp check). */
    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("Hello");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_NONE);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= strlen("<think>\n\n</think>\n\n"));
    TEST_ASSERT(!memcmp(prompt_text + pt_len - strlen("<think>\n\n</think>\n\n"),
                        "<think>\n\n</think>\n\n",
                        strlen("<think>\n\n</think>\n\n")));

    free(prompt_text);
    chat_msgs_free(&msgs);
}

static void test_kat_template_thinking_controls(void) {
    chat_msgs msgs = {0};
    chat_msg system = {.role = xstrdup("system"), .content = xstrdup("  sys  ")};
    chat_msg user = {.role = xstrdup("user"), .content = xstrdup("  hello  ")};
    chat_msgs_push(&msgs, system);
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_HIGH, true, false);
    TEST_ASSERT(!strcmp(prompt,
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"));
    free(prompt);

    prompt = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_NONE, true, false);
    TEST_ASSERT(strstr(prompt,
        "<|im_start|>assistant\n<think>\n\n</think>\n\n") != NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>assistant\n</think>") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_kat_template_preserve_thinking(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {.role = xstrdup("user"), .content = xstrdup("first")};
    chat_msg assistant = {
        .role = xstrdup("assistant"),
        .content = xstrdup("answer"),
        .reasoning = xstrdup("old reasoning"),
    };
    chat_msg user2 = {.role = xstrdup("user"), .content = xstrdup("next")};
    chat_msgs_push(&msgs, user1);
    chat_msgs_push(&msgs, assistant);
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_HIGH, true, false);
    TEST_ASSERT(strstr(prompt, "old reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>assistant\nanswer<|im_end|>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text_profile(
        &msgs, NULL, NULL, P36_THINK_HIGH, true, true);
    TEST_ASSERT(strstr(prompt,
        "<|im_start|>assistant\n<think>\nold reasoning\n</think>\n\nanswer<|im_end|>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_kat_template_tool_round(void) {
    chat_msgs msgs = {0};
    chat_msg user = {.role = xstrdup("user"), .content = xstrdup("inspect")};
    chat_msg assistant = {
        .role = xstrdup("assistant"),
        .reasoning = xstrdup("need reads"),
    };
    tool_call call = {
        .name = xstrdup("read"),
        .arguments = xstrdup("{\"path\":\"a.c\"}"),
    };
    tool_calls_push(&assistant.calls, call);
    chat_msg tool1 = {.role = xstrdup("tool"), .content = xstrdup("one")};
    chat_msg tool2 = {.role = xstrdup("tool"), .content = xstrdup("two")};
    chat_msgs_push(&msgs, user);
    chat_msgs_push(&msgs, assistant);
    chat_msgs_push(&msgs, tool1);
    chat_msgs_push(&msgs, tool2);

    char *prompt = render_chat_prompt_text_profile(
        &msgs, "{\"name\":\"read\"}", NULL, P36_THINK_HIGH, true, false);
    TEST_ASSERT(strstr(prompt, "<IMPORTANT>\nReminder:") != NULL);
    TEST_ASSERT(strstr(prompt, "Do NOT nest <tool_call> blocks.") != NULL);
    TEST_ASSERT(strstr(prompt,
        "<think>\nneed reads\n</think>\n\n<tool_call>\n<function=read>") != NULL);
    TEST_ASSERT(strstr(prompt,
        "<|im_start|>user\n<tool_response>\none\n</tool_response>"
        "\n<tool_response>\ntwo\n</tool_response><|im_end|>\n") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_fixed_template_inline_controls(void) {
    for (int kat = 0; kat < 2; kat++) {
        chat_msgs msgs = {0};
        chat_msg system = {
            .role = xstrdup("system"),
            .content = xstrdup("sys <|think_off|>"),
        };
        chat_msg user = {
            .role = xstrdup("user"),
            .content = xstrdup("hello"),
        };
        chat_msgs_push(&msgs, system);
        chat_msgs_push(&msgs, user);

        bool thinking = true;
        apply_chat_thinking_controls(&msgs, &thinking);
        TEST_ASSERT(!thinking);
        char *prompt = render_chat_prompt_text_profile(
            &msgs, NULL, NULL, P36_THINK_HIGH, kat, true);
        TEST_ASSERT(strstr(prompt, "think_off") == NULL);
        TEST_ASSERT(strstr(prompt, "system\nsys<|im_end|>") != NULL);
        TEST_ASSERT(strstr(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") != NULL);
        free(prompt);
        chat_msgs_free(&msgs);

        msgs = (chat_msgs){0};
        system = (chat_msg){
            .role = xstrdup("system"),
            .content = xstrdup("  <|think_off|>  "),
        };
        user = (chat_msg){
            .role = xstrdup("user"),
            .content = xstrdup("hello"),
        };
        chat_msgs_push(&msgs, system);
        chat_msgs_push(&msgs, user);
        prompt = render_chat_prompt_text_profile(
            &msgs, NULL, NULL, P36_THINK_HIGH, kat, true);
        TEST_ASSERT(!strncmp(prompt, "<|im_start|>user\nhello",
                             strlen("<|im_start|>user\nhello")));
        TEST_ASSERT(strstr(prompt, "<|im_start|>system") == NULL);
        free(prompt);
        chat_msgs_free(&msgs);
    }
}

static void test_fixed_template_chronological_system_messages(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {.role = xstrdup("user"), .content = xstrdup("first")};
    chat_msg developer = {.role = xstrdup("developer"), .content = xstrdup("middle")};
    chat_msg user2 = {.role = xstrdup("user"), .content = xstrdup("last")};
    chat_msgs_push(&msgs, user1);
    chat_msgs_push(&msgs, developer);
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    const char *first = strstr(prompt, "<|im_start|>user\nfirst");
    const char *middle = strstr(prompt, "<|im_start|>system\nmiddle");
    const char *last = strstr(prompt, "<|im_start|>user\nlast");
    TEST_ASSERT(first && middle && last && first < middle && middle < last);
    TEST_ASSERT(!strncmp(prompt, "<|im_start|>user\n", strlen("<|im_start|>user\n")));
    free(prompt);

    chat_msg_free(&msgs.v[1]);
    msgs.v[1].role = xstrdup("system");
    msgs.v[1].content = xstrdup("CLIENT_SYSTEM");
    prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA", NULL, P36_THINK_HIGH);
    const char *tools = strstr(prompt, "TOOL_SCHEMA");
    const char *client = strstr(prompt, "CLIENT_SYSTEM");
    first = strstr(prompt, "<|im_start|>user\nfirst");
    TEST_ASSERT(tools && client && first && tools < first && first < client);
    free(prompt);
    chat_msgs_free(&msgs);

    msgs = (chat_msgs){0};
    chat_msg system = {.role = xstrdup("system"), .content = xstrdup("first system")};
    developer = (chat_msg){
        .role = xstrdup("developer"),
        .content = xstrdup("second system"),
    };
    user1 = (chat_msg){.role = xstrdup("user"), .content = xstrdup("question")};
    chat_msgs_push(&msgs, system);
    chat_msgs_push(&msgs, developer);
    chat_msgs_push(&msgs, user1);
    prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA", NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt,
        "</IMPORTANT>\n\nfirst system\n\nsecond system<|im_end|>") != NULL);
    TEST_ASSERT(strstr(prompt, "<|im_start|>system\nsecond system") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_fixed_template_tool_error_recovery(void) {
    for (int kat = 0; kat < 2; kat++) {
        chat_msgs msgs = {0};
        chat_msg user = {.role = xstrdup("user"), .content = xstrdup("run")};
        chat_msg tool1 = {.role = xstrdup("tool"), .content = xstrdup("Error: bad argument")};
        chat_msg tool2 = {.role = xstrdup("tool"), .content = xstrdup("failed to open")};
        chat_msgs_push(&msgs, user);
        chat_msgs_push(&msgs, tool1);
        chat_msgs_push(&msgs, tool2);

        char *prompt = render_chat_prompt_text_profile(
            &msgs, NULL, NULL, P36_THINK_HIGH, kat, true);
        TEST_ASSERT(strstr(prompt,
            "⚠️ SYSTEM WARNING: The previous tool call returned an error. "
            "Diagnose the failure and retry with completely corrected arguments.") != NULL);
        TEST_ASSERT(strstr(prompt,
            "⚠️ SYSTEM WARNING: 2 consecutive tool errors detected. "
            "Your previous approach is incorrect. You MUST use a fundamentally "
            "different approach or corrected arguments.") != NULL);
        TEST_ASSERT(strstr(prompt,
            "<|im_start|>assistant\n<think>\n") != NULL);
        TEST_ASSERT(strstr(prompt,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n") == NULL);
        TEST_ASSERT(strstr(prompt,
            "</tool_response>\n<tool_response>") != NULL);
        free(prompt);
        chat_msgs_free(&msgs);
    }

    TEST_ASSERT(!tool_response_is_error("{\"error_count\": 0, \"result\": \"ok\"}"));
    TEST_ASSERT(!tool_response_is_error("{\"error\": null, \"result\": \"ok\"}"));
    TEST_ASSERT(!tool_response_is_error("Process exited with code 0"));
    TEST_ASSERT(!tool_response_is_error("throw new Error('example')"));
    TEST_ASSERT(!tool_response_is_error("$ printf 'error: example'"));
    TEST_ASSERT(!tool_response_is_error("error: appears in output; command took 1.2s"));
    char traceback[900];
    memset(traceback, 'x', sizeof(traceback));
    memcpy(traceback, "Traceback (most recent call last):", 34);
    traceback[sizeof(traceback) - 1] = '\0';
    TEST_ASSERT(tool_response_is_error(traceback));
}

static void test_fixed_template_reasoning_extraction(void) {
    chat_msgs msgs = {0};
    chat_msg user = {.role = xstrdup("user"), .content = xstrdup("question")};
    chat_msg assistant = {
        .role = xstrdup("assistant"),
        .content = xstrdup("<think>\nold thought\n</ think>\nanswer"),
    };
    chat_msg user2 = {.role = xstrdup("user"), .content = xstrdup("next")};
    chat_msgs_push(&msgs, user);
    chat_msgs_push(&msgs, assistant);
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "<think>\nold thought\n</think>\n\nanswer") != NULL);
    free(prompt);

    free(msgs.v[1].content);
    msgs.v[1].content = xstrdup("The literal string </think> is documentation.");
    prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "The literal string </think> is documentation.") != NULL);
    free(prompt);

    free(msgs.v[1].content);
    msgs.v[1].content = xstrdup("<think>single line thought</think>answer");
    prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt,
        "<think>\nsingle line thought\n</think>\n\nanswer") != NULL);
    free(prompt);

    msgs.v[1].reasoning = xstrdup("explicit thought");
    prompt = render_chat_prompt_text(&msgs, NULL, NULL, P36_THINK_HIGH);
    TEST_ASSERT(strstr(prompt,
        "<think>\nexplicit thought\n</think>\n\nanswer") != NULL);
    TEST_ASSERT(strstr(prompt, "single line thought") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_reasoning_message_aliases(void) {
    const char *p =
        "[{\"role\":\"assistant\",\"reasoning_content\":\"openai\","
        "\"thinking\":\"anthropic\",\"reasoning\":\"responses\","
        "\"content\":\"answer\"}]";
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].reasoning && !strcmp(msgs.v[0].reasoning, "openai"));
    chat_msgs_free(&msgs);

    p = "[{\"role\":\"assistant\",\"reasoning\":\"vllm\",\"content\":\"answer\"}]";
    TEST_ASSERT(parse_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].reasoning && !strcmp(msgs.v[0].reasoning, "vllm"));
    chat_msgs_free(&msgs);
}

static void test_chat_template_kwargs_parse(void) {
    const char *p = "{\"enable_thinking\":false,\"preserve_thinking\":true,\"future\":1}";
    bool thinking = true;
    bool got_thinking = false;
    bool preserve = false;
    TEST_ASSERT(parse_chat_template_kwargs(&p, &thinking, &got_thinking, &preserve));
    TEST_ASSERT(!thinking);
    TEST_ASSERT(got_thinking);
    TEST_ASSERT(preserve);
}

static void test_responses_input_parses_qwen_tool_continuation(void) {
    const char *json =
        "[{\"type\":\"message\",\"role\":\"user\","
        "\"content\":[{\"type\":\"input_text\",\"text\":\"inspect\"}]},"
        "{\"type\":\"function_call\",\"call_id\":\"call_1\","
        "\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.c\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\","
        "\"output\":\"contents\"}]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_responses_input(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.len > 0 && !strcmp(msgs.v[0].role, "user"));
    TEST_ASSERT(msgs.len > 0 && !strcmp(msgs.v[0].content, "inspect"));
    TEST_ASSERT(msgs.len > 1 && msgs.v[1].calls.len == 1);
    TEST_ASSERT(msgs.len > 1 && !strcmp(msgs.v[1].calls.v[0].name, "read"));
    TEST_ASSERT(msgs.len > 2 && !strcmp(msgs.v[2].role, "tool"));
    TEST_ASSERT(msgs.len > 2 && !strcmp(msgs.v[2].tool_call_id, "call_1"));
    TEST_ASSERT(msgs.len > 2 && !strcmp(msgs.v[2].content, "contents"));
    chat_msgs_free(&msgs);

    p = "[{\"role\":\"assistant\",\"content\":\"Reading both files.\"},"
        "{\"type\":\"function_call\",\"call_id\":\"call_a\",\"name\":\"read\",\"arguments\":\"{}\"},"
        "{\"type\":\"function_call\",\"call_id\":\"call_b\",\"name\":\"read\",\"arguments\":\"{}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_a\",\"output\":\"a\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_b\",\"output\":\"b\"}]";
    TEST_ASSERT(parse_responses_input(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.v[0].calls.len == 2);
    TEST_ASSERT(!strcmp(msgs.v[0].content, "Reading both files."));
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].id, "call_a"));
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[1].id, "call_b"));
    chat_msgs_free(&msgs);
}

static void test_responses_output_is_protocol_native(void) {
    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    tool_calls calls = {0};
    tool_call tc = {
        .id = xstrdup("call_native"),
        .name = xstrdup("read"),
        .arguments = xstrdup("{\"path\":\"a.c\"}"),
    };
    tool_calls_push(&calls, tc);
    TEST_ASSERT(responses_final_response(sv[0], &r, "resp_test", "done", "why",
                                         &calls, "tool_calls", 12, 3));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"object\":\"response\"") != NULL);
    TEST_ASSERT(strstr(out, "\"type\":\"reasoning\"") != NULL);
    TEST_ASSERT(strstr(out, "\"type\":\"message\"") != NULL);
    TEST_ASSERT(strstr(out, "\"type\":\"function_call\"") != NULL);
    TEST_ASSERT(strstr(out, "\"call_id\":\"call_native\"") != NULL);
    TEST_ASSERT(strstr(out, "chat.completion") == NULL);
    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_responses_stream_uses_responses_events(void) {
    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.stream = true;
    TEST_ASSERT(request_uses_structured_stream(&r));
    TEST_ASSERT(responses_sse_finish(sv[0], &r, "resp_stream", "hello", NULL,
                                     NULL, "stop", 4, 1));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "event: response.created") != NULL);
    TEST_ASSERT(strstr(out, "event: response.output_text.delta") != NULL);
    TEST_ASSERT(strstr(out, "event: response.completed") != NULL);
    TEST_ASSERT(strstr(out, "chat.completion") == NULL);
    free(out);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_responses_stream_item_lifecycle(void) {
    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    request r;
    request_init(&r, REQ_CHAT, 128);
    tool_calls calls = {0};
    for (int i = 0; i < 2; i++) {
        tool_call call = {.id = xstrdup(i ? "call_second" : "call_first"),
                          .name = xstrdup("read"),
                          .arguments = xstrdup("{\"path\":\"text.png\"}")};
        tool_calls_push(&calls, call);
    }
    TEST_ASSERT(responses_sse_finish(sv[0], &r, "resp_items", "hello", "why",
                                     &calls, "tool_calls", 4, 1));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    const char *cursor = out;
    for (int i = 0; i < 4; i++) {
        const char *added = strstr(cursor, "event: response.output_item.added\n");
        TEST_ASSERT(added != NULL);
        if (!added) break;
        const char *done = strstr(added, "event: response.output_item.done\n");
        TEST_ASSERT(done != NULL);
        if (!done) break;
        const char *payload = strstr(added, i < 2 ? "text.delta\"" : "arguments.done\"");
        TEST_ASSERT(payload && payload < done);
        char index[40];
        snprintf(index, sizeof(index), "\"output_index\":%d,", i);
        const char *position = strstr(added, index);
        TEST_ASSERT(position && position < done);
        cursor = done + 1;
    }
    TEST_ASSERT(strstr(cursor, "event: response.completed\n") != NULL);
    TEST_ASSERT(strstr(out, "\"call_id\":\"call_first\"") != NULL);
    TEST_ASSERT(strstr(out, "\"call_id\":\"call_second\"") != NULL);
    for (const char *p = out; (p = strstr(p, "\ndata: ")); ) {
        p += 7;
        TEST_ASSERT(json_skip_value(&p));
    }
    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_cors_headers_are_opt_in(void) {
    int sv[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    bool old = g_enable_cors;
    g_enable_cors = true;
    TEST_ASSERT(http_response(sv[0], 200, "text/plain", ""));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Methods: GET, POST, OPTIONS") != NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Headers: *") != NULL);
    g_enable_cors = old;
    free(out);
    close(sv[0]);
    close(sv[1]);
}

static void test_server_streaming_options(void) {
    char *argv[] = {
        "p36-server",
        "--ssd-streaming-full-layers", "0",
        "--ssd-streaming-cache-experts", "512",
        "--ssd-streaming-cold",
    };
    server_config c = parse_options((int)(sizeof(argv) / sizeof(argv[0])), argv);
    TEST_ASSERT(c.engine.ssd_streaming);
    TEST_ASSERT(c.engine.ssd_streaming_cold);
    TEST_ASSERT(c.engine.ssd_streaming_full_layers == 0);
    TEST_ASSERT(c.engine.ssd_streaming_full_layers_set);
    TEST_ASSERT(c.engine.ssd_streaming_cache_experts == 512);
    TEST_ASSERT(c.engine.cache_type_k == P36_KV_CACHE_F16);
    TEST_ASSERT(c.engine.cache_type_v == P36_KV_CACHE_F16);

    char *resident_argv[] = {"p36-server", "--vulkan"};
    c = parse_options(2, resident_argv);
    TEST_ASSERT(c.engine.cache_type_k == P36_KV_CACHE_Q8_0);
    TEST_ASSERT(c.engine.cache_type_v == P36_KV_CACHE_Q4_0);

    char *cpu_argv[] = {"p36-server", "--cpu"};
    c = parse_options(2, cpu_argv);
    TEST_ASSERT(c.engine.cache_type_k == P36_KV_CACHE_F16);
    TEST_ASSERT(c.engine.cache_type_v == P36_KV_CACHE_F16);

    char *explicit_argv[] = {
        "p36-server", "--ssd-streaming", "-ctk", "q8_0", "-ctv", "q4_0",
    };
    c = parse_options(6, explicit_argv);
    TEST_ASSERT(c.engine.cache_type_k == P36_KV_CACHE_Q8_0);
    TEST_ASSERT(c.engine.cache_type_v == P36_KV_CACHE_Q4_0);

    char *nothink_argv[] = {"p36-server", "--nothink"};
    c = parse_options(2, nothink_argv);
    TEST_ASSERT(g_force_nothink);
    g_force_nothink = false;

    char *nothink_dash_argv[] = {"p36-server", "--no-think"};
    c = parse_options(2, nothink_dash_argv);
    TEST_ASSERT(g_force_nothink);
    g_force_nothink = false;

    char *batched_argv[] = {
        "p36-server", "--batched-session", "4",
        "--mixed-prefill-quantum", "64",
    };
    c = parse_options(5, batched_argv);
    TEST_ASSERT(c.batched_sessions == 4);
    TEST_ASSERT(c.mixed_prefill_quantum == 64);
    server s = {.mixed_prefill_quantum = c.mixed_prefill_quantum};
    TEST_ASSERT(server_prefill_quantum_for(&s, true) == 64);
}

static void test_batched_prefill_round_robin(void) {
    server s = {0};
    server_slot slots[4] = {0};
    s.slots = slots;
    s.slot_count = 4;
    s.last_prefill_slot = 0;
    slots[0].prefill_waiting = true;
    slots[2].prefill_waiting = true;
    slots[3].prefill_waiting = true;
    TEST_ASSERT(server_next_prefill_slot_locked(&s) == 2);
    s.last_prefill_slot = 2;
    TEST_ASSERT(server_next_prefill_slot_locked(&s) == 3);
    slots[3].prefill_waiting = false;
    TEST_ASSERT(server_next_prefill_slot_locked(&s) == 0);
    slots[0].prefill_waiting = false;
    slots[2].prefill_waiting = false;
    TEST_ASSERT(server_next_prefill_slot_locked(&s) == -1);
}

static void test_batched_slot_scoring(void) {
    TEST_ASSERT(job_slot_score_values(false, false, 80, 80) >
                job_slot_score_values(false, false, 79, 80));
    TEST_ASSERT(job_slot_score_values(false, false, 80, 80) >
                job_slot_score_values(false, false, 40, 40));
    TEST_ASSERT(job_slot_score_values(false, false, 40, 80) >
                job_slot_score_values(false, false, 10, 80));
    TEST_ASSERT(job_slot_score_values(true, false, 80, 80) == INT_MIN);
    TEST_ASSERT(job_slot_score_values(false, true, 80, 80) == INT_MIN);
    TEST_ASSERT(job_slot_score_values(false, false, 0, 0) == 0);
}

static void test_batched_decode_cancellation(void) {
    server s = {0};
    server_slot slots[2] = {0};
    s.slots = slots;
    s.slot_count = 2;
    s.decode_pending = 2;
    slots[0].decode_pending = true;
    slots[1].decode_pending = true;
    pthread_cond_init(&s.model_cv, NULL);
    server_cancel_decode_waiters_locked(&s);
    TEST_ASSERT(s.decode_pending == 0);
    TEST_ASSERT(!slots[0].decode_pending && slots[0].decode_done);
    TEST_ASSERT(!slots[1].decode_pending && slots[1].decode_done);
    TEST_ASSERT(slots[0].decode_rc != 0 && slots[1].decode_rc != 0);
    pthread_cond_destroy(&s.model_cv);
}

static const char test_inline_png_base64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8DQwMDAwMAEIkAYABglAYOd/VRoAAAAAElFTkSuQmCC";

static void test_openai_inline_image_content(void) {
    buf json = {0};
    buf_puts(&json,
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\","
        "\"text\":\"describe \"},{\"type\":\"image_url\","
        "\"image_url\":{\"url\":\"data:image/png;base64,");
    buf_puts(&json, test_inline_png_base64);
    buf_puts(&json, "\"}},{\"type\":\"text\",\"text\":\" please\"}]}]");
    const char *p = json.ptr;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].images.len == 1);
    TEST_ASSERT(strstr(msgs.v[0].content, "describe ") == msgs.v[0].content);
    TEST_ASSERT(strstr(msgs.v[0].content, " please") != NULL);
    TEST_ASSERT(msgs.v[0].images.v[0].encoded_len >= 8);
    TEST_ASSERT(!memcmp(msgs.v[0].images.v[0].encoded, "\x89PNG\r\n\x1a\n", 8));
    chat_msgs_free(&msgs);
    buf_free(&json);
}

static void test_http_image_paths_and_urls_are_rejected(void) {
    const char *cases[] = {
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\","
        "\"image_url\":{\"url\":\"https://example.com/a.png\"}}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\","
        "\"image_url\":{\"url\":\"/tmp/a.png\"}}]}]",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char *p = cases[i];
        chat_msgs msgs = {0};
        TEST_ASSERT(!parse_messages(&p, &msgs));
        chat_msgs_free(&msgs);
    }
}

static void test_anthropic_inline_image_content(void) {
    buf json = {0};
    buf_puts(&json,
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image\","
        "\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\","
        "\"data\":\"");
    buf_puts(&json, test_inline_png_base64);
    buf_puts(&json, "\"}},{\"type\":\"text\",\"text\":\"describe\"}]}]");
    const char *p = json.ptr;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].images.len == 1);
    TEST_ASSERT(strstr(msgs.v[0].content, "describe") != NULL);
    chat_msgs_free(&msgs);
    buf_free(&json);
}

static void test_responses_inline_image_content(void) {
    buf json = {0};
    buf_puts(&json, "[{\"type\":\"input_text\",\"text\":\"describe \"},"
                   "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,");
    buf_puts(&json, test_inline_png_base64);
    buf_puts(&json, "\"}]");
    const char *p = json.ptr;
    chat_msg msg = {0};
    TEST_ASSERT(parse_openai_content(&p, &msg));
    TEST_ASSERT(msg.images.len == 1);
    TEST_ASSERT(strstr(msg.content, "describe ") == msg.content);
    chat_msg_free(&msg);
    buf_free(&json);
}

static void test_anthropic_tool_image_output(void) {
    buf json = {0};
    buf_puts(&json, "[{\"content\":[{\"type\":\"tool_result\","
                    "\"tool_use_id\":\"tool_test\",\"content\":["
                    "{\"type\":\"text\",\"text\":\"read image\"},"
                    "{\"type\":\"image\",\"source\":{\"type\":\"base64\","
                    "\"media_type\":\"image/png\",\"data\":\"");
    buf_puts(&json, test_inline_png_base64);
    buf_puts(&json, "\"}}]}],\"role\":\"user\"}]");
    const char *p = json.ptr;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 1 && msgs.v[0].images.len == 1);
    TEST_ASSERT(strstr(msgs.v[0].content, "<tool_response>\nread image") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].content, msgs.v[0].images.v[0].marker) != NULL);
    chat_msgs_free(&msgs);
    buf_free(&json);
    const char *invalid[] = {
        "[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"content\":["
        "{\"type\":\"tool_use\",\"name\":\"bash\",\"input\":{}}]}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"content\":["
        "{\"type\":\"tool_result\",\"content\":\"nested\"}]}]}]",
    };
    for (size_t i = 0; i < 2; i++) {
        p = invalid[i];
        TEST_ASSERT(!parse_anthropic_messages(&p, &msgs));
        chat_msgs_free(&msgs);
    }
}

static void test_responses_tool_image_output(void) {
    const char *types[] = {"function_call_output", "custom_tool_call_output", "reasoning"};
    for (size_t i = 0; i < 3; i++) {
        buf json = {0};
        buf_puts(&json, "[{\"type\":\"");
        buf_puts(&json, types[i]);
        buf_puts(&json, "\",\"call_id\":\"call_test\",\"output\":["
                       "{\"type\":\"input_text\",\"text\":\"read image\"},"
                       "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,");
        buf_puts(&json, test_inline_png_base64);
        buf_puts(&json, "\"}]}]");
        const char *p = json.ptr;
        chat_msgs msgs = {0};
        bool ok = parse_responses_input(&p, &msgs);
        TEST_ASSERT(ok == (i < 2));
        if (ok) {
            TEST_ASSERT(msgs.len == 1 && !strcmp(msgs.v[0].role, "tool"));
            TEST_ASSERT(msgs.v[0].images.len == 1);
            TEST_ASSERT(strstr(msgs.v[0].content, msgs.v[0].images.v[0].marker) != NULL);
        }
        chat_msgs_free(&msgs);
        buf_free(&json);
    }
}

static void test_tool_only_continuation_binding(void) {
    const char *p = "[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-secret\",\"content\":\"done\"}]}]";
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    request r = {.api = API_ANTHROPIC};
    request_note_tool_continuation(&r, &msgs, "tools");
    TEST_ASSERT(r.continuation_count == 1);
    TEST_ASSERT(!strcmp(r.continuation_ids[0], "call-secret"));
    char *ids[] = {"call-secret"};
    server_slot slot = {.pending_ids = ids, .pending_count = 1,
                        .pending_api = API_ANTHROPIC, .pending_tools = "tools"};
    TEST_ASSERT(slot_pending_matches(&slot, &r));
    slot.pending_pos = 1;
    TEST_ASSERT(!slot_pending_matches(&slot, &r));
    slot.pending_pos = 0;
    slot.pending_tools = "different tools";
    TEST_ASSERT(!slot_pending_matches(&slot, &r));
    slot.pending_tools = "tools";
    ids[0] = "other-call";
    TEST_ASSERT(!slot_pending_matches(&slot, &r));
    request_free(&r);
    chat_msgs_free(&msgs);

    p = "[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-secret\",\"content\":\"done\"},\"new user instruction\"]}]";
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    r.api = API_ANTHROPIC;
    request_note_tool_continuation(&r, &msgs, "tools");
    TEST_ASSERT(r.continuation_count == 0);
    request_free(&r);
    chat_msgs_free(&msgs);

    p = "[{\"role\":\"user\",\"content\":\"<tool_response>call-secret</tool_response>\"}]";
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    r.api = API_ANTHROPIC;
    request_note_tool_continuation(&r, &msgs, "tools");
    TEST_ASSERT(r.continuation_count == 0);
    request_free(&r);
    chat_msgs_free(&msgs);
}

static void test_server_image_embedding_cache(void) {
    server_image_cache cache = {0};
    uint8_t key = 1;
    float data[2] = {1.25f, -2.5f};
    server_image_input input = {.encoded = &key, .encoded_len = 1};
    p36_vision_embedding src = {.data = data, .token_count = 1,
                               .width = 42, .fingerprint = {7}};
    p36_vision_embedding out = {0};
    const size_t budget = 2 * (sizeof(data) + 1);
    TEST_ASSERT(!server_image_cache_get(&cache, &input, &out));
    server_image_cache_put(&cache, &input, &src, 2, budget);
    TEST_ASSERT(server_image_cache_get(&cache, &input, &out));
    TEST_ASSERT(out.data != data && !memcmp(out.data, data, sizeof(data)));
    TEST_ASSERT(out.width == 42 && out.fingerprint[0] == 7);
    out.data[0] = 99;
    p36_vision_embedding_free(&out);
    key = 2;
    TEST_ASSERT(!server_image_cache_get(&cache, &input, &out));
    server_image_cache_put(&cache, &input, &src, 2, budget);
    key = 1;
    TEST_ASSERT(server_image_cache_get(&cache, &input, &out));
    TEST_ASSERT(out.data[0] == data[0]);
    p36_vision_embedding_free(&out);
    key = 3;
    server_image_cache_put(&cache, &input, &src, 2, budget);
    TEST_ASSERT(cache.bytes == budget);
    key = 2;
    TEST_ASSERT(!server_image_cache_get(&cache, &input, &out));
    key = 1;
    TEST_ASSERT(server_image_cache_get(&cache, &input, &out));
    p36_vision_embedding_free(&out);
    server_image_cache_clear(&cache);
    TEST_ASSERT(cache.bytes == 0 && cache.clock == 0);
    for (key = 1; key <= SERVER_IMAGE_CACHE_ENTRIES + 1; key++)
        server_image_cache_put(&cache, &input, &src, 2, 4096);
    TEST_ASSERT(cache.bytes == SERVER_IMAGE_CACHE_ENTRIES * (sizeof(data) + 1));
    key = 1;
    TEST_ASSERT(!server_image_cache_get(&cache, &input, &out));
    server_image_cache_clear(&cache);
    src.token_count = UINT32_MAX;
    server_image_cache_put(&cache, &input, &src, UINT32_MAX, SIZE_MAX);
    TEST_ASSERT(cache.bytes == 0);
    src.token_count = 1;
    server_image_cache_put(&cache, &input, &src, 2, sizeof(data));
    TEST_ASSERT(cache.bytes == 0);
}

static void test_jinja_chat_template_and_json_tool_call(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {.role = xstrdup("system"), .content = xstrdup("Be helpful.")};
    chat_msg u1 = {.role = xstrdup("user"), .content = xstrdup("Check files")};
    chat_msg a1 = {.role = xstrdup("assistant"), .content = xstrdup("Running ls.")};
    tool_call tc1 = {
        .id = xstrdup("call_1"),
        .name = xstrdup("bash"),
        .arguments = xstrdup("{\"command\":\"ls\"}"),
    };
    tool_calls_push(&a1.calls, tc1);
    chat_msg t1 = {
        .role = xstrdup("tool"),
        .content = xstrdup("README.md"),
        .tool_call_id = xstrdup("call_1"),
    };
    chat_msg u2 = {.role = xstrdup("user"), .content = xstrdup("What did it find?")};
    chat_msgs_push(&msgs, sys);
    chat_msgs_push(&msgs, u1);
    chat_msgs_push(&msgs, a1);
    chat_msgs_push(&msgs, t1);
    chat_msgs_push(&msgs, u2);

    const char *sample_jinja =
        "{%- for m in messages %}\n"
        "{{- '<|im_start|>' + m.role + '\n' + m.content }}\n"
        "{%- if m.tool_calls is defined %}\n"
        "{%- for tc in m.tool_calls %}\n"
        "{{- '\n<tool_call>\n{\"name\": \"' + tc.name + '\", \"arguments\": ' + (tc.arguments | tojson) + '}\n</tool_call>' }}\n"
        "{%- endfor %}\n"
        "{%- endif %}\n"
        "{{- '<|im_end|>\n' }}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "{{- '<|im_start|>assistant\n' }}\n"
        "{%- if enable_thinking is defined and enable_thinking is false %}\n"
        "{{- '<think>\n\n</think>\n\n' }}\n"
        "{%- else %}\n"
        "{{- '<think>\n' }}\n"
        "{%- endif %}\n"
        "{%- endif %}";

    const char *tools_json = "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{\"type\":\"object\"}}}";
    char *prompt_high = render_jinja_chat_prompt_text(sample_jinja, &msgs, tools_json, P36_THINK_HIGH, true);
    TEST_ASSERT(strstr(prompt_high, "<|im_start|>system\nBe helpful.<|im_end|>\n") != NULL);
    TEST_ASSERT(strstr(prompt_high, "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\":\"ls\"}}\n</tool_call>") != NULL);
    const char *tail_high = "<|im_start|>assistant\n<think>\n";
    TEST_ASSERT(strlen(prompt_high) >= strlen(tail_high) &&
                strcmp(prompt_high + strlen(prompt_high) - strlen(tail_high), tail_high) == 0);
    free(prompt_high);

    char *prompt_none = render_jinja_chat_prompt_text(sample_jinja, &msgs, NULL, P36_THINK_NONE, true);
    const char *tail_none = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    TEST_ASSERT(strlen(prompt_none) >= strlen(tail_none) &&
                strcmp(prompt_none + strlen(prompt_none) - strlen(tail_none), tail_none) == 0);
    free(prompt_none);
    chat_msgs_free(&msgs);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls parsed_calls = {0};
    const char *gen = "Thought step</think>\n\nLet me run bash.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>";
    bool ok = parse_generated_message_ex(gen, true, &content, &reasoning, &parsed_calls);
    TEST_ASSERT(ok);
    TEST_ASSERT(reasoning && strcmp(reasoning, "Thought step") == 0);
    TEST_ASSERT(content && strstr(content, "Let me run bash.") != NULL);
    TEST_ASSERT(parsed_calls.len == 1);
    TEST_ASSERT(strcmp(parsed_calls.v[0].name, "bash") == 0);
    TEST_ASSERT(strstr(parsed_calls.v[0].arguments, "\"command\"") != NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.chat_template = sample_jinja;
    r.think_mode = P36_THINK_HIGH;
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &parsed_calls);
    TEST_ASSERT(strstr(suffix, "<tool_call>\n{\"name\": \"bash\", \"arguments\": ") != NULL);
    free(suffix);
    request_free(&r);
    free(content);
    free(reasoning);
    tool_calls_free(&parsed_calls);
}

static void p36_server_unit_tests_run(void) {
    test_jinja_chat_template_and_json_tool_call();
    test_openai_inline_image_content();
    test_http_image_paths_and_urls_are_rejected();
    test_anthropic_inline_image_content();
    test_responses_inline_image_content();
    test_anthropic_tool_image_output();
    test_responses_tool_image_output();
    test_server_image_embedding_cache();
    test_tool_only_continuation_binding();
    test_batched_prefill_round_robin();
    test_batched_slot_scoring();
    test_batched_decode_cancellation();
    test_request_defaults_match_qwen_api();
    test_explicit_sampling_wins_in_thinking_mode();
    test_request_peek_model();
    test_reasoning_effort_mapping();
    test_qwen38_effort_prompt();
    test_api_thinking_controls_parse();
    test_chat_agent_budget_fields_parse();
    test_truncated_generation_semantics();
    test_truncated_stream_never_leaks_markup();
    test_render_think_max_prompt_prefix();
    test_render_non_thinking_prompt_closes_think();
    test_render_drops_old_reasoning_without_tools();
    test_render_preserves_reasoning_with_tools();
    test_tool_schema_order_from_anthropic_schema();
    test_tool_schema_order_from_openai_tools();
    test_tool_schemas_minified_and_sorted();
    test_responses_input_parses_qwen_tool_continuation();
    test_responses_output_is_protocol_native();
    test_responses_stream_uses_responses_events();
    test_responses_stream_item_lifecycle();
    test_context_length_error_uses_protocol_standard_shape();
    test_cors_headers_are_opt_in();
    test_server_streaming_options();
    test_tool_prompt_args_preserve_call_order();
    test_openai_tool_args_preserve_call_order();
    test_anthropic_thinking_and_tool_args_preserve_call_order();
    test_anthropic_live_stream_sends_incremental_blocks();
    test_anthropic_qwen_tool_stream_sends_answer_before_finish();
    test_openai_tool_stream_sends_incremental_text();
    test_openai_chat_stream_splits_reasoning_without_tools();
    test_openai_qwen_tool_stream_sends_answer_before_finish();
    test_openai_tool_stream_sends_partial_arguments();
    test_openai_tool_stream_waits_for_incomplete_tool_tags();
    test_openai_tool_stream_sends_partial_raw_arguments();
    test_openai_tool_stream_preserves_ampersands();
    test_openai_tool_stream_holds_partial_utf8_arguments();
    test_openai_tool_stream_handles_multiple_calls();
    test_streaming_holds_partial_utf8();
    test_parse_short_qwen_tool_and_canonical_suffix();
    test_qwen_tool_parser_preserves_multiline_parameters();
    test_tool_parse_failure_returns_recoverable_finish();
    test_thinking_tool_markers_are_not_executable();
    test_tool_checkpoint_suffix_is_future_prompt_canonical();
    test_tool_checkpoint_minifies_json_parameters();
    test_tool_memory_replays_sampled_qwen_tool();
    test_tool_memory_preserves_empty_think_per_id();
    test_exact_tool_replay_can_be_disabled();
    test_qwen_tool_decode_state_separates_structure_and_payload();
    test_tool_memory_max_ids_prunes_oldest();
    test_kv_quant_bits();
    test_kv_tool_map_filters_by_qwen_tool_text();
    test_kv_tool_map_restores_before_prompt_render();
    test_thinking_checkpoint_canonical_matches_future_prompt();
    test_thinking_canonical_empty_content();
    test_thinking_canonical_multi_turn();
    test_thinking_canonical_with_tools_preserves_reasoning();
    test_thinking_canonical_non_thinking_mode_noop();
    test_kat_template_thinking_controls();
    test_kat_template_preserve_thinking();
    test_kat_template_tool_round();
    test_fixed_template_inline_controls();
    test_fixed_template_chronological_system_messages();
    test_fixed_template_tool_error_recovery();
    test_fixed_template_reasoning_extraction();
    test_reasoning_message_aliases();
    test_chat_template_kwargs_parse();
    test_qwen_tool_call_rendering();
    test_tool_separator_whitespace_is_not_content();
    test_qwen_tool_prompt_preserves_tool_supplied_text();
    test_stop_list_parses_all_sequences();
    test_stop_list_streaming_holds_and_trims_stop_text();
    test_json_skip_has_nesting_limit();
    test_json_string_handles_surrogates();
    test_json_owned_replacement_is_atomic();
    test_api_parsers_reject_nonfinite_numbers();
    test_api_parsers_reject_malformed_duplicate_strings();
    test_model_metadata_clamps_completion_to_context();
    test_gguf_counts_are_rejected_before_allocation();
    test_client_socket_nonblocking_flag();
    test_thinking_state_tracks_prompt_and_generated_tags();
    test_strip_assistant_reasoning_blocks();
    test_thinking_checkpoint_canonicalization_gate();
    test_tool_marker_state_ignores_orphan_end();
    test_canonical_rewrite_rebuilds_when_live_tail_changes();
    test_kv_cache_store_len_uses_configured_boundary();
    test_kv_cache_continued_uses_aligned_frontiers();
    test_sha1_bytes_hex_matches_known_vector();
    test_kv_key_includes_identity();
    test_kv_model_fingerprint_detects_changes();
    test_system_boundary_tokens();
    test_tool_id_set_handles_large_history();
    test_kv_cache_lookup_uses_longest_text_prefix();
    test_kv_cache_lookup_matches_stripped_reasoning();
    test_kv_cache_stripped_lookup_rejects_open_generation_prompt();
    test_kv_cache_entry_can_be_touched_twice();
    test_kv_cache_eviction_values_fresh_snapshots();
    test_kv_cache_eviction_keeps_aligned_continued_frontiers();
}

#ifndef P36_SERVER_TEST_NO_MAIN
int main(int argc, char **argv) {
    int cli_rc = p36_cli_require_no_arguments(argc, argv, "p36_server_test");
    if (cli_rc) return cli_rc;
    p36_server_unit_tests_run();
    if (test_failures) {
        fprintf(stderr, "p36-server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("p36-server tests: ok");
    return 0;
}
#endif

#endif
