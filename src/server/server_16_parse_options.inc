static bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    if (q) *q = '\0';

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

typedef struct {
    server *srv;
    int fd;
} client_arg;

static void append_model_json_profile(buf *b, int ctx, int default_tokens,
                                      const char *id, const char *name,
                                      bool vision) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_puts(b, "{\"id\":");
    json_escape(b, id);
    buf_puts(b, ",\"object\":\"model\",\"created\":1767225600,"
                "\"owned_by\":\"p36.c\",\"name\":");
    json_escape(b, name);
    buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"meta\":{\"n_ctx\":%d,\"n_ctx_train\":%d},"
        "\"architecture\":{\"modality\":\"%s\","
            "\"input_modalities\":[%s],"
            "\"output_modalities\":[\"text\"]},"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"ignore_eos\","
            "\"presence_penalty\","
            "\"frequency_penalty\","
            "\"repeat_penalty\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"chat_template_kwargs\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        ctx,
        vision ? "text+image->text" : "text->text",
        vision ? "\"text\",\"image\"" : "\"text\"",
        ctx,
        max_completion);
}

#ifdef P36_SERVER_TEST
static void append_model_json_values(buf *b, int ctx, int default_tokens) {
    append_model_json_profile(b, ctx, default_tokens,
                              "qwen3.6-35b-a3b", "Qwen 3.6 35B A3B", false);
}
#endif

static void append_model_json(buf *b, const server *s) {
    const bool kat = p36_engine_is_kat_coder(s->engine);
    append_model_json_profile(b, p36_session_ctx(s->session), s->default_tokens,
                              p36_engine_model_name(s->engine),
                              kat ? "KAT-Coder V2.5 Dev" : "Qwen 3.6 35B A3B",
                              p36_engine_has_vision(s->engine));
}

/* llama-server compatible /props: clients such as little-coder/pi probe it
 * for the context window and the vision modality. */
static bool send_props(server *s, int fd) {
    const int ctx = p36_session_ctx(s->session);
    const bool vision = p36_engine_has_vision(s->engine);
    const char *id = s->model_count > 0 ? s->models[s->model_current].id
                                        : p36_engine_model_name(s->engine);
    buf b = {0};
    buf_printf(&b,
        "{\"default_generation_settings\":{\"n_ctx\":%d,\"params\":{\"n_predict\":%d}},"
        "\"n_ctx\":%d,\"total_slots\":%d,\"model_alias\":",
        ctx, s->default_tokens < ctx ? s->default_tokens : ctx, ctx,
        s->slot_count > 0 ? s->slot_count : 1);
    json_escape(&b, id);
    buf_puts(&b, ",\"model_path\":");
    json_escape(&b, id);
    buf_printf(&b, ",\"modalities\":{\"vision\":%s,\"audio\":false}}\n",
               vision ? "true" : "false");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool send_model(server *s, int fd) {
    buf b = {0};
    append_model_json(&b, s);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool send_models(server *s, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    if (s->model_count == 0) {
        append_model_json(&b, s);
    } else {
        /* The resident model first, then the rest of the catalog. */
        const int ctx = p36_session_ctx(s->session);
        append_model_json_profile(&b, ctx, s->default_tokens,
                                  s->models[s->model_current].id,
                                  s->models[s->model_current].id,
                                  p36_engine_has_vision(s->engine));
        for (int i = 0; i < s->model_count; i++) {
            if (i == s->model_current) continue;
            buf_putc(&b, ',');
            append_model_json_profile(&b, ctx, s->default_tokens,
                                      s->models[i].id, s->models[i].id,
                                      i == s->model_initial &&
                                          s->engine_opt.vision_path &&
                                          s->engine_opt.vision_path[0]);
        }
    }
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static int server_model_find(const server *s, const char *id) {
    if (!id) return -1;
    for (int i = 0; i < s->model_count; i++)
        if (!strcmp(s->models[i].id, id)) return i;
    return -1;
}

/* The top-level "model" string of a request body, or NULL.  A real scan, not
 * strstr: tool schemas can carry nested "model" keys. */
static char *request_peek_model(const char *body) {
    const char *p = body;
    if (!p) return NULL;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    if (*p == '}') return NULL;
    for (;;) {
        char *key = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        json_ws(&p);
        if (!strcmp(key, "model") && *p == '"') {
            char *val = NULL;
            free(key);
            return json_string(&p, &val) ? val : NULL;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p != ',') return NULL;
        p++;
    }
}

/* Validate a GGUF without risking the server: the engine's model checks
 * exit() on a mismatched shape, so they run in a child process. */
static bool server_model_probe(const char *path) {
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) {
            dup2(nul, STDOUT_FILENO);
            dup2(nul, STDERR_FILENO);
        }
        _exit(p36_inspect_model(path) == 0 ? 0 : 1);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static int server_model_cmp(const void *a, const void *b) {
    return strcmp(((const server_model *)a)->id, ((const server_model *)b)->id);
}

/* Build the catalog from dir plus the startup model; returns the startup
 * model's index, or -1.  Files that resolve to the same inode appear once. */
static int server_models_scan(server *s, const char *dir, const char *initial_path) {
    char initial_real[PATH_MAX];
    if (!realpath(initial_path, initial_real)) return -1;
    DIR *d = opendir(dir);
    if (!d) {
        server_log(P36_LOG_DEFAULT, "p36-server: cannot open --models-dir %s: %s",
                   dir, strerror(errno));
        return -1;
    }
    int cap = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        size_t n = strlen(de->d_name);
        if (n <= 5 || strcmp(de->d_name + n - 5, ".gguf") != 0) continue;
        buf pb = {0};
        buf_printf(&pb, "%s/%s", dir, de->d_name);
        if (s->model_count == cap) {
            cap = cap ? cap * 2 : 16;
            s->models = realloc(s->models, (size_t)cap * sizeof(*s->models));
            if (!s->models) die("out of memory");
        }
        s->models[s->model_count].path = buf_take(&pb);
        s->models[s->model_count].id = xstrndup(de->d_name, n - 5);
        s->model_count++;
    }
    closedir(d);
    qsort(s->models, (size_t)s->model_count, sizeof(*s->models), server_model_cmp);

    /* Several names for one file (symlinks): list it once, under the name of
     * the regular file when there is one. */
    int uniq = 0;
    for (int i = 0; i < s->model_count; i++) {
        char real[PATH_MAX];
        struct stat st;
        bool link = lstat(s->models[i].path, &st) == 0 && S_ISLNK(st.st_mode);
        int same = -1;
        if (realpath(s->models[i].path, real)) {
            for (int j = 0; j < uniq && same < 0; j++) {
                char other[PATH_MAX];
                if (realpath(s->models[j].path, other) && !strcmp(other, real)) same = j;
            }
        } else {
            link = true;  /* dangling: drop it */
            same = uniq;
        }
        if (same < 0) {
            s->models[uniq++] = s->models[i];
            continue;
        }
        struct stat kept;
        bool kept_link = same < uniq && lstat(s->models[same].path, &kept) == 0 &&
                         S_ISLNK(kept.st_mode);
        if (same < uniq && kept_link && !link) {
            server_model dropped = s->models[same];
            s->models[same] = s->models[i];
            s->models[i] = dropped;
        }
        free(s->models[i].id);
        free(s->models[i].path);
    }
    s->model_count = uniq;

    /* Probe everything except the startup model, which the real open checks. */
    int initial = -1;
    int kept = 0;
    for (int i = 0; i < s->model_count; i++) {
        char real[PATH_MAX];
        bool is_initial = realpath(s->models[i].path, real) && !strcmp(real, initial_real);
        if (!is_initial && !server_model_probe(s->models[i].path)) {
            server_log(P36_LOG_DEFAULT, "p36-server: models: skipping %s (not loadable)",
                       s->models[i].id);
            free(s->models[i].id);
            free(s->models[i].path);
            continue;
        }
        if (is_initial) initial = kept;
        s->models[kept++] = s->models[i];
    }
    s->model_count = kept;
    if (initial < 0) {
        /* -m lives outside the directory: add it under its own file name. */
        const char *base = strrchr(initial_path, '/');
        base = base ? base + 1 : initial_path;
        size_t n = strlen(base);
        if (n > 5 && !strcmp(base + n - 5, ".gguf")) n -= 5;
        s->models = realloc(s->models, (size_t)(s->model_count + 1) * sizeof(*s->models));
        if (!s->models) die("out of memory");
        s->models[s->model_count].id = xstrndup(base, n);
        s->models[s->model_count].path = xstrdup(initial_path);
        initial = s->model_count++;
    }
    for (int i = 0; i < s->model_count; i++)
        server_log(P36_LOG_DEFAULT, "p36-server: models: %s%s", s->models[i].id,
                   i == initial ? " (loaded)" : "");
    return initial;
}

static void server_models_free(server *s) {
    for (int i = 0; i < s->model_count; i++) {
        free(s->models[i].id);
        free(s->models[i].path);
    }
    free(s->models);
    s->models = NULL;
    s->model_count = 0;
    free(s->kv_root);
    s->kv_root = NULL;
}

/* Fold the model content fingerprint into every cache key.  If the model file
 * cannot be read the cache is switched off rather than keyed by a guess. */
static void server_kv_set_identity(server *s, const char *model_path) {
    if (!s->kv.enabled) return;
    if (!kv_model_fingerprint(model_path, s->engine_opt.directional_steering_file, s->kv.ident)) {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: KV disk cache disabled: cannot fingerprint model %s",
                   model_path ? model_path : "(none)");
        kv_cache_close(&s->kv);
        return;
    }
    server_log(P36_LOG_KVCACHE, "p36-server: KV cache identity %s", s->kv.ident);
}

static void server_kv_open_for_model(server *s, int idx) {
    if (!s->kv_root) return;
    buf db = {0};
    buf_printf(&db, "%s/%s", s->kv_root, s->models[idx].id);
    char *dir = buf_take(&db);
    kv_cache_open(&s->kv, dir, s->kv_budget_mb, s->kv_reject_different_quant, s->kv_opt);
    server_kv_set_identity(s, s->models[idx].path);
    free(dir);
}

static bool server_model_open_engine(server *s, int idx) {
    p36_engine_options opt = s->engine_opt;
    opt.model_path = s->models[idx].path;
    if (idx != s->model_initial) {
        /* Sidecars are built for one model; only the startup model gets them. */
        opt.mtp_path = NULL;
        opt.vision_path = NULL;
        opt.directional_steering_file = NULL;
    }
    p36_engine *engine = NULL;
    if (p36_engine_open(&engine, &opt) != 0) return false;
    p36_session *session = NULL;
    if (p36_session_create(&session, engine, s->ctx_size) != 0) {
        p36_engine_close(engine);
        return false;
    }
    s->engine = engine;
    s->slots[0].session = session;
    s->session = session;
    s->model_current = idx;
    return true;
}

/* Swap the resident model.  Caller holds swap_lock for writing, so no request
 * is parsing, queued or generating and the worker is idle. */
static bool server_model_swap(server *s, int idx) {
    if (idx == s->model_current) return true;
    const int prev = s->model_current;
    const double t0 = now_sec();
    server_log(P36_LOG_DEFAULT, "p36-server: models: switching %s -> %s",
               s->models[prev].id, s->models[idx].id);

    server_slot *slot = &s->slots[0];
    const p36_tokens *live = p36_session_tokens(slot->session);
    if (s->kv.enabled && live && live->len >= s->kv.opt.min_tokens)
        server_kv_store_current(s, slot, "evict");
    kv_cache_close(&s->kv);
    const int tool_max = s->tool_mem.max_entries;
    tool_memory_free(&s->tool_mem);
    s->tool_mem.max_entries = tool_max;
    server_image_cache_clear(&s->image_cache);
    slot_pending_clear(slot);
    p36_session_free(slot->session);
    slot->session = NULL;
    s->session = NULL;
    p36_engine_close(s->engine);
    s->engine = NULL;

    bool ok = server_model_open_engine(s, idx);
    if (!ok) {
        server_log(P36_LOG_DEFAULT, "p36-server: models: failed to load %s; restoring %s",
                   s->models[idx].id, s->models[prev].id);
        if (!server_model_open_engine(s, prev))
            die("p36-server: could not restore the previous model");
    }
    server_kv_open_for_model(s, s->model_current);
    server_log(P36_LOG_DEFAULT, "p36-server: models: %s resident after %.1f s",
               s->models[s->model_current].id, now_sec() - t0);
    return ok;
}

static void client_done(server *s) {
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}

static void set_client_socket_nonblocking(int fd);

static void *client_main(void *arg) {
    client_arg *ca = arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);
    bool model_locked = false;

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, 200, "text/plain", "");
        http_request_free(&hr);
        goto done;
    }

    /* With a catalog, hold the resident model (read side) from here until the
     * job completes; a request naming another catalog model swaps it in first. */
    if (s->model_count > 0) {
        int want = -1;
        if (!strcmp(hr.method, "POST")) {
            char *name = request_peek_model(hr.body);
            want = server_model_find(s, name);
            free(name);
        }
        for (int tries = 0;; tries++) {
            pthread_rwlock_rdlock(&s->swap_lock);
            if (want < 0 || want == s->model_current || tries >= 3) break;
            pthread_rwlock_unlock(&s->swap_lock);
            pthread_rwlock_wrlock(&s->swap_lock);
            bool swapped = server_model_swap(s, want);
            pthread_rwlock_unlock(&s->swap_lock);
            if (!swapped) {
                http_error(fd, 503, "requested model failed to load");
                http_request_free(&hr);
                goto done;
            }
        }
        model_locked = true;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        send_models(s, fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") &&
        (!strcmp(hr.path, "/props") || !strcmp(hr.path, "/v1/props"))) {
        send_props(s, fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strncmp(hr.path, "/v1/models/", 11) &&
        s->model_count > 0) {
        int idx = server_model_find(s, hr.path + 11);
        if (idx < 0) {
            http_error(fd, 404, "unknown model");
        } else {
            buf b = {0};
            append_model_json_profile(&b, p36_session_ctx(s->session), s->default_tokens,
                                      s->models[idx].id, s->models[idx].id,
                                      idx == s->model_current
                                          ? p36_engine_has_vision(s->engine)
                                          : (idx == s->model_initial &&
                                             s->engine_opt.vision_path &&
                                             s->engine_opt.vision_path[0]));
            buf_putc(&b, '\n');
            http_response(fd, 200, "application/json", b.ptr);
            buf_free(&b);
        }
        http_request_free(&hr);
        goto done;
    }
    char model_path[320];
    snprintf(model_path, sizeof(model_path), "/v1/models/%s",
             p36_engine_model_name(s->engine));
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, model_path)) {
        send_model(s, fd);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok = false;
    const int ctx_size = p36_session_ctx(s->session);
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, s, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, 400, err);
        goto done;
    }
    request_apply_model_sampling_defaults(s->engine, &req);
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    pthread_mutex_lock(&j.mu);
    if (!enqueue(s, &j)) {
        pthread_mutex_unlock(&j.mu);
        http_error(fd, 503, "server shutting down");
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        request_free(&j.req);
        goto done;
    }
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);

    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    if (model_locked) pthread_rwlock_unlock(&s->swap_lock);
    close(fd);
    client_done(s);
    return NULL;
}

static int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = P36_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

typedef struct {
    p36_engine_options engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    float default_presence_penalty;
    float default_frequency_penalty;
    bool default_presence_penalty_set;
    bool default_frequency_penalty_set;
    float default_temperature;
    int default_top_k;
    float default_top_p;
    float default_min_p;
    float default_repeat_penalty;
    bool default_temperature_set;
    bool default_top_k_set;
    bool default_top_p_set;
    bool default_min_p_set;
    bool default_repeat_penalty_set;
    const char *trace_path;
    const char *kv_disk_dir;
    const char *models_dir;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
    int batched_sessions;
    int mixed_prefill_quantum;
} server_config;

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(P36_LOG_DEFAULT, "p36-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(P36_LOG_DEFAULT, "p36-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(P36_LOG_DEFAULT, "p36-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc || p36_cli_token_is_option(argv[*i + 1])) {
        server_log(P36_LOG_DEFAULT, "p36-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void log_context_memory(p36_backend backend, int ctx_size,
                               uint32_t prefill_chunk, int session_count,
                               p36_kv_cache_type cache_type_k,
                               p36_kv_cache_type cache_type_v) {
    p36_context_memory m = p36_context_memory_estimate_configured(
            backend, ctx_size, prefill_chunk, cache_type_k, cache_type_v);
    server_log(P36_LOG_DEFAULT,
               "p36-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)",
               (double)m.total_bytes / (1024.0 * 1024.0),
               ctx_size,
               p36_backend_name(backend),
               m.prefill_cap,
               m.raw_cap,
               m.comp_cap);
    if (session_count > 1) {
        server_log(P36_LOG_DEFAULT,
                   "p36-server: %d resident sessions request about %.2f GiB of context buffers",
                   session_count,
                   (double)m.total_bytes * (double)session_count /
                       (1024.0 * 1024.0 * 1024.0));
    }
}

static void server_close_resources(server *s) {
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    server_image_cache_clear(&s->image_cache);
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    for (int i = 0; i < s->slot_count; i++) {
        slot_pending_clear(&s->slots[i]);
        p36_session_free(s->slots[i].session);
    }
    free(s->slot_threads);
    free(s->slots);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->kv_mu);
    pthread_mutex_destroy(&s->inference_mu);
    pthread_mutex_destroy(&s->model_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->model_cv);
    pthread_rwlock_destroy(&s->swap_lock);
    server_models_free(s);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    p36_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: p36-server [options]\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: " P36_DEFAULT_MODEL_PATH "\n"
        "  --chat-template FILE\n"
        "      Load an external .jinja chat template file (disabled by default).\n"
        "  --mtp FILE\n"
        "      Optional MTP support GGUF used for draft-token probes.\n"
        "  --mtp-draft N\n"
        "      Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F\n"
        "      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n"
        "  -c, --ctx N\n"
        "      Context size allocated at startup. Default: 32768\n"
        "  -ctk, --cache-type-k TYPE\n"
        "      KV cache K type: f16, q8_0, or q4_0. Default: Vulkan/Metal resident q8_0, otherwise f16\n"
        "  -ctv, --cache-type-v TYPE\n"
        "      KV cache V type: f16, q8_0, or q4_0. Default: Vulkan/Metal resident q4_0, otherwise f16\n"
        "  -n, --tokens N          (alias: --default-tokens)\n"
        "      Default max output tokens when the client omits a limit. Default: 262144 (256K)\n"
        "  --presence-penalty F    (alias: --default-presence-penalty)\n"
        "      Default presence_penalty when the client omits one. Range -2..2. Default: 0 (1.5 for KAT-Coder).\n"
        "      Guards against a request degenerating into verbatim repetition. Explicit client values win.\n"
        "  --frequency-penalty F   (alias: --default-frequency-penalty)\n"
        "      Default frequency_penalty when the client omits one. Range -2..2. Default: 0. Explicit client values win.\n"
        "  --temp F                (alias: --temperature)\n"
        "      Default temperature when the client omits one. Range 0..100. Default: model default. Explicit client values win.\n"
        "  --top-k N\n"
        "      Default top_k when the client omits one. Range 0 or more (0 = off). Default: model default. Explicit client values win.\n"
        "  --top-p F\n"
        "      Default top_p when the client omits one. Range 0..1. Default: model default. Explicit client values win.\n"
        "  --min-p F\n"
        "      Default min_p when the client omits one. Range 0..1. Default: model default. Explicit client values win.\n"
        "  --repeat-penalty F\n"
        "      Default repeat_penalty when the client omits one. Range 0.5..2.0. Default: 1 (off). Explicit client values win.\n"
        "  -t, --threads N\n"
        "      CPU helper threads for lightweight host-side work.\n"
        "  --prefill-chunk N\n"
        "      GPU graph prefill chunk size. Default: auto; resident GPU resolves to 1024.\n"
        "      Disk KV checkpoints land on multiples of it; 512 caches more of a short system prompt.\n"
        "  --f32-fast-wide\n"
        "      Opt-in 256-thread f32 matvec. Faster, but reassociates the MoE router gate.\n"
        "      Benchmarking only: observed to break long-context chat. See README.md.\n"
        "  --attn-span N\n"
        "      Attention split-K span in keys. Default: 512. 128 measured +2.2%% at ctx 2048 only;\n"
        "      error and combine cost both scale with context/span. Not validated for long context.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load an n_layer x n_embd f32 direction matrix for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before serving. Slower startup, fewer first-use stalls.\n"
        "  --ssd-streaming\n"
        "      Keep non-routed weights resident and stream selected routed experts from SSD.\n"
        "  --ssd-streaming-cold\n"
        "      Start SSD streaming with an empty dynamic expert cache.\n"
        "  --ssd-streaming-cache-experts N|NGB\n"
        "      Dynamic expert cache as an expert count or GiB budget.\n"
        "  --ssd-streaming-full-layers N\n"
        "      Keep the first N routed layers resident. Use 0 to disable.\n"
        "  --ssd-streaming-preload-experts N\n"
        "      Override the number of dynamic expert slots preloaded at startup.\n"
        "  --simulate-used-memory NGB\n"
        "      Reserve memory before auto-sizing the streaming cache.\n"
        "  --metal | --vulkan | --cpu | --backend NAME\n"
        "      Select Metal on Apple Silicon, Vulkan on Linux, or CPU.\n"
        "\n"
        "HTTP API:\n"
        "  --vision FILE\n"
        "      Accept PNG/JPEG images using a disk-streamed Vulkan vision sidecar.\n"
        "  --host HOST\n"
        "      Bind address. Default: 127.0.0.1\n"
        "  --port N\n"
        "      Bind port. Default: 8000\n"
        "  --trace FILE\n"
        "      Write a human-readable session trace: prompts, cache decisions, output, tool calls.\n"
        "  --batched-session N\n"
        "      Keep N resident sessions and batch decode-ready GPU requests.\n"
        "  --mixed-prefill-quantum N\n"
        "      Prefill tokens per scheduling turn while generation is active. Default: 128\n"
        "  --models-dir DIR\n"
        "      Expose every loadable *.gguf in DIR on /v1/models. A request whose \"model\" names\n"
        "      one of them swaps it in (one model resident at a time; -m is loaded first).\n"
        "      Unknown model names use the resident model. Not with --batched-session.\n"
        "  --cors\n"
        "      Add Access-Control-Allow-* headers and answer browser preflight requests.\n"
        "\n"
        "Thinking and sampling:\n"
        "  --nothink, --no-think\n"
        "      Force-disable thinking/reasoning for all models, overriding client thinking controls.\n"
        "  ignore_eos=true requires explicit temperature=0; stop strings and context limits still apply.\n"
        "  Chat requests default to thinking mode (xhigh on Qwen3.8).\n"
        "  Numeric reasoning_effort strings: 0 off; 1..33 low; 34..66 medium; 67..100 xhigh.\n"
        "  thinking.budget_tokens starts adaptive thinking closure and selects dense effort at 8k/16k/24k.\n"
        "  Agent controls on /v1/chat/completions (all optional, off unless sent):\n"
        "    thinking_budget N     hard cap: soft </think> bias from 75%%, forced at N reasoning tokens.\n"
        "    action_budget N       after thinking, N content tokens with no tool call start a reasoning-leak\n"
        "                          check; a leak ends the reply with finish_reason=action_budget.\n"
        "    stop_after_tool_call  end decoding at the first complete tool call.\n"
        "  A tool call cut off by max_tokens is discarded, never returned as text: the choice gets\n"
        "  incomplete_tool_call=true (finish_reason stays length). Cut reasoning stays in reasoning_content.\n"
        "  usage adds completion_tokens_details.{reasoning_tokens,content_tokens} and\n"
        "  timings.{prefill_ms,decode_ms} (plus system_checkpoint with --kv-system-checkpoint).\n"
        "  Only reasoning_effort=max or output_config.effort=max requests Think Max.\n"
        "  Think Max requires --ctx >= 98304; smaller contexts use high.\n"
        "  thinking={type:disabled}, think=false, chat_template_kwargs.enable_thinking=false, or a -nothink model alias selects non-thinking mode.\n"
        "  Qwen and KAT-Coder preserve prior thinking by default; chat_template_kwargs.preserve_thinking=false strips it.\n"
        "  Qwen defaults are temperature=1, top_k=0, top_p=1, min_p=0.05.\n"
        "  KAT defaults are temperature=0.7, top_k=20, top_p=0.8, min_p=0.05.\n"
        "  Model defaults apply only to omitted sampling knobs; explicit client values win.\n"
        "\n"
        "Disk KV cache:\n"
        "  --kv-disk-dir DIR\n"
        "      Enable disk KV checkpoints in DIR. The directory is created if needed.\n"
        "  --kv-disk-space-mb N\n"
        "      Disk budget for checkpoint files. Default when enabled: 4096\n"
        "  --kv-system-checkpoint\n"
        "      Store the system-prompt+tools prefix once so new conversations skip its prefill.\n"
        "      Needs --kv-disk-dir. Keys include a model content fingerprint; ~80 MiB per checkpoint.\n"
        "  --kv-cache-min-tokens N\n"
        "      Do not save or load checkpoints shorter than N tokens. Default: 512\n"
        "  --kv-cache-cold-max-tokens N\n"
        "      Cold first prompts in [min,N] are saved automatically. 0 disables cold saves. Default: 30000\n"
        "  --kv-cache-continued-interval-tokens N\n"
        "      Save at absolute aligned frontiers spaced about N tokens apart. 0 disables. Default: 10000\n"
        "  --kv-cache-boundary-trim-tokens N\n"
        "      Trim this many tail tokens before cold boundary saves to avoid tokenizer boundary merges. Default: 32\n"
        "  --kv-cache-boundary-align-tokens N\n"
        "      Align cold boundary saves down to this token multiple. 0 disables alignment. Default: 2048\n"
        "  --kv-cache-reject-different-quant\n"
        "      Refuse checkpoints written by the same model with a different routed-expert quantization.\n"
        "  --disable-exact-tool-replay\n"
        "      Disable the tool-id -> exact sampled <tool_call> map. Tool history falls back to canonical JSON rendering.\n"
        "  --tool-memory-max-ids N\n"
        "      Maximum exact tool-call IDs kept in RAM for replay. Default: 100000\n"
        "\n"
        "  Cache triggers:\n"
        "      cold       save a stable prefix of a long first prompt before generation starts\n"
        "      continued  save absolute aligned restart frontiers during long prefill or generation\n"
        "      evict      save the live conversation before another request replaces it\n"
        "      shutdown   save the live conversation when the server exits cleanly\n"
        "\n"
        "Normal server command:\n"
        "  ./p36-server --ctx 100000 --kv-disk-dir /tmp/p36-kv --kv-disk-space-mb 8192\n"
        "\n"
        "Notes:\n"
        "  Use /v1/chat/completions, /v1/responses, /v1/completions, or /v1/messages.\n"
        "  Larger --ctx values allocate more KV memory at startup; the startup log prints the estimate.\n"
        "  Disk KV caching is best for agents that resend long prompts with stable prefixes.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

static p36_backend parse_backend_arg(const char *s, const char *arg) {
    if (!strcmp(s, "metal")) return P36_BACKEND_METAL;
    if (!strcmp(s, "vulkan")) return P36_BACKEND_VULKAN;
    if (!strcmp(s, "cpu")) return P36_BACKEND_CPU;
    server_log(P36_LOG_DEFAULT, "p36-server: invalid %s value: %s", arg, s);
    server_log(P36_LOG_DEFAULT, "p36-server: valid server backends are: metal, vulkan, cpu");
    exit(2);
}

static p36_backend default_server_backend(void) {
#ifdef P36_NO_GPU
    return P36_BACKEND_CPU;
#elif defined(__APPLE__)
    return P36_BACKEND_METAL;
#else
    return P36_BACKEND_VULKAN;
#endif
}

