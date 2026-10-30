static int p36_sample_top_p_min_p(const float *logits, uint32_t n_vocab,
                                  float temperature, int top_k,
                                  float top_p, float min_p,
                                  uint64_t *rng, float *prob_scratch) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        const bool owned_scratch = prob_scratch == NULL;
        if (owned_scratch) {
            prob_scratch = xmalloc((size_t)n_vocab * sizeof(prob_scratch[0]));
        }
        const int token = p36_sample_full_vocab(logits, n_vocab, temperature,
                                                top_p, min_p, rng, prob_scratch);
        if (owned_scratch) free(prob_scratch);
        return token;
    }
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    {
        int ids[1024];
        float vals[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            {
                int j = n < top_k ? n++ : n - 1;
                while (j > 0 && vals[j - 1] < v) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                    j--;
                }
                vals[j] = v;
                ids[j] = (int)i;
            }
        }
        if (n == 0) return sample_argmax(logits, n_vocab);
        return p36_sample_from_sorted_topk(ids, vals, n, temperature, top_p, min_p, rng);
    }
}

static int p36_sample_logits_penalized(float *logits, uint32_t n_vocab,
                                       float temperature, int top_k,
                                       float top_p, float min_p,
                                       const int *tokens, int n_tokens,
                                       float presence_penalty,
                                       float frequency_penalty,
                                       float repeat_penalty,
                                       uint64_t *rng, float *scratch) {
    if (!tokens || n_tokens <= 0 ||
        (presence_penalty == 0.0f && frequency_penalty == 0.0f &&
         repeat_penalty == 1.0f)) {
        return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                      top_p, min_p, rng, scratch);
    }

    /* Distinct seen ids plus their original logits, so the logits can be
     * restored exactly after sampling (no float add-back drift). */
    struct { int id; float logit; } *saved =
        malloc((size_t)n_tokens * sizeof(*saved));
    if (!saved) {
        return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                      top_p, min_p, rng, scratch);
    }
    int n_saved = 0;
    memset(scratch, 0, (size_t)n_vocab * sizeof(scratch[0]));
    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        if (token < 0 || (uint32_t)token >= n_vocab) continue;
        if (scratch[token] != 0.0f) continue;
        scratch[token] = 1.0f;
        saved[n_saved].id = token;
        saved[n_saved].logit = logits[token];
        n_saved++;
    }
    /* llama.cpp-style multiplicative penalty: positive logits shrink, negative grow. */
    for (int k = 0; k < n_saved; k++) {
        float *l = &logits[saved[k].id];
        if (repeat_penalty != 1.0f) *l = *l > 0.0f ? *l / repeat_penalty : *l * repeat_penalty;
    }
    /* Scratch counts occurrences for frequency/presence; first hit applies presence. */
    memset(scratch, 0, (size_t)n_vocab * sizeof(scratch[0]));
    for (int i = 0; i < n_tokens; i++) {
        int token = tokens[i];
        if (token < 0 || (uint32_t)token >= n_vocab) continue;
        if (scratch[token] == 0.0f) logits[token] -= presence_penalty;
        scratch[token] += 1.0f;
        logits[token] -= frequency_penalty;
    }
    int sampled = p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                         top_p, min_p, rng, scratch);
    for (int k = 0; k < n_saved; k++) logits[saved[k].id] = saved[k].logit;
    free(saved);
    return sampled;
}

#ifdef P36_TEST_HOOKS
int p36_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch) {
    if (!logits || !rng || n_vocab == 0) return -1;
    return p36_sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                                  top_p, min_p, rng, prob_scratch);
}

int p36_test_sample_logits_penalized(float *logits, uint32_t n_vocab,
                                     float temperature, int top_k,
                                     float top_p, float min_p,
                                     const int *tokens, int n_tokens,
                                     float presence_penalty,
                                     float frequency_penalty,
                                     float repeat_penalty,
                                     uint64_t *rng, float *prob_scratch) {
    if (!logits || !rng || !prob_scratch || n_vocab == 0) return -1;
    return p36_sample_logits_penalized(
        logits, n_vocab, temperature, top_k, top_p, min_p,
        tokens, n_tokens, presence_penalty, frequency_penalty,
        repeat_penalty, rng, prob_scratch);
}

uint32_t p36_test_kv_next_cap(uint32_t cap, uint32_t need, uint32_t limit) {
    return p36_kv_next_cap(cap, need, limit);
}
#endif

static bool p36_load_directional_steering(p36_engine *e) {
    uint64_t count;
    uint64_t bytes;
    struct stat st;
    FILE *fp;
    if (!e || (e->directional_steering_attn_scale == 0.0f &&
               e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }
    if (!e->directional_steering_file || !e->directional_steering_file[0]) {
        fprintf(stderr, "p36: directional steering needs --dir-steering-file\n");
        return false;
    }
    count = (uint64_t)P36_N_LAYER * P36_N_EMBD;
    bytes = count * sizeof(float);
    fp = fopen(e->directional_steering_file, "rb");
    if (!fp) {
        fprintf(stderr, "p36: failed to open directional steering file %s: %s\n",
                e->directional_steering_file, strerror(errno));
        return false;
    }
    if (fstat(fileno(fp), &st) != 0 || st.st_size < 0 || (uint64_t)st.st_size != bytes) {
        fprintf(stderr,
                "p36: directional steering file must contain exactly %u x %u f32 values (%" PRIu64 " bytes)\n",
                P36_N_LAYER, P36_N_EMBD, bytes);
        fclose(fp);
        return false;
    }
    e->directional_steering_dirs = xmalloc((size_t)bytes);
    if (fread(e->directional_steering_dirs, sizeof(float), (size_t)count, fp) != count ||
        fclose(fp) != 0) {
        fprintf(stderr, "p36: failed to read directional steering vectors from %s\n",
                e->directional_steering_file);
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        return false;
    }
#ifndef P36_NO_GPU
    if (p36_backend_uses_graph(e->backend)) {
        e->directional_steering_gpu = p36_gpu_tensor_alloc(bytes);
        if (!e->directional_steering_gpu ||
            !p36_gpu_tensor_write(e->directional_steering_gpu, 0,
                                  e->directional_steering_dirs, bytes)) {
            fprintf(stderr, "p36: failed to upload directional steering vectors\n");
            return false;
        }
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
    }
#endif
    fprintf(stderr, "p36: directional steering enabled: %s attn=%g ffn=%g\n",
            e->directional_steering_file,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

int p36_engine_open(p36_engine **out, const p36_engine_options *opt) {
    p36_engine *e;
    if (!out || !opt || !opt->model_path || !opt->model_path[0]) return 1;
#ifdef P36_NO_GPU
    if (p36_backend_uses_graph(opt->backend)) return 1;
#else
#if defined(P36_METAL) && !defined(P36_METAL_TEST_COMPAT)
    if (opt->backend == P36_BACKEND_VULKAN) {
        fprintf(stderr, "p36: Vulkan was requested from a Metal binary; use --metal or rebuild with 'make gpu'\n");
        return 1;
    }
#elif !defined(P36_METAL)
    if (opt->backend == P36_BACKEND_METAL) {
        fprintf(stderr, "p36: Metal was requested from a Vulkan binary; use --vulkan or build with 'make metal'\n");
        return 1;
    }
#endif
    if (p36_backend_uses_graph(opt->backend) && !p36_gpu_init()) return 1;
#endif
    e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->vision_model.fd = -1;
#if defined(P36_METAL) && defined(P36_METAL_TEST_COMPAT)
    /* The GPU ABI is shared by both native runtimes.  Normalize legacy
     * Vulkan-named test callers to the backend linked into this binary. */
    e->backend = p36_backend_uses_graph(opt->backend)
        ? P36_BACKEND_METAL : opt->backend;
#else
    e->backend = opt->backend;
#endif
    e->n_threads = p36_resolve_thread_count(opt->n_threads);
    e->cpu_prefill_cap = opt->prefill_chunk ? opt->prefill_chunk : p36_default_cpu_prefill_cap();
    e->prefill_cap_override = opt->prefill_chunk;
    e->context_size = opt->context_size > 0 ? opt->context_size : 4096;
    e->planned_sessions = opt->session_count > 0 ? (uint32_t)opt->session_count : 1;
    e->cache_type_k = opt->cache_type_k;
    e->cache_type_v = opt->cache_type_v;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    e->quality = opt->quality;
    {
        const char *mmq_contract = getenv("P36_CPU_MMQ_CONTRACT");
        e->cpu_mmq_contract = mmq_contract && mmq_contract[0] && mmq_contract[0] != '0';
    }
    e->ssd_streaming = opt->ssd_streaming;
    e->ssd_streaming_cold = opt->ssd_streaming_cold;
    e->ssd_streaming_full_layers_set = opt->ssd_streaming_full_layers_set;
    e->ssd_streaming_cache_experts = opt->ssd_streaming_cache_experts;
    e->ssd_streaming_full_layers = opt->ssd_streaming_full_layers;
    e->ssd_streaming_cache_bytes = opt->ssd_streaming_cache_bytes;
    e->ssd_streaming_preload_experts = opt->ssd_streaming_preload_experts;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > P36_MTP_MAX_DRAFT) e->mtp_draft_tokens = P36_MTP_MAX_DRAFT;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 0.0f;
    e->lookup_order = opt->lookup_order > 0 ? opt->lookup_order : 3;
    e->lookup_k = opt->lookup_k > 0 ? opt->lookup_k : 0;
    if (e->lookup_k > P36_MTP_MAX_DRAFT) e->lookup_k = P36_MTP_MAX_DRAFT;
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = p36_strdup(opt->directional_steering_file);
    }
    e->directional_steering_attn_scale = opt->directional_steering_attn;
    e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    if (opt->simulate_used_memory_bytes != 0 &&
        !p36_ssd_memory_lock_acquire(&e->simulated_memory,
                                     opt->simulate_used_memory_bytes)) {
        p36_engine_close(e);
        return 1;
    }
    model_open(&e->model, opt->model_path, p36_backend_uses_graph(opt->backend));
    {
        p36_str name = {0};
        if (model_get_string(&e->model, "general.name", &name)) {
            e->kat_coder = p36_str_contains(name, "KAT Coder") ||
                           p36_str_contains(name, "KAT-Coder");
        }
    }
    {
        const char *tmpl_opt = opt->chat_template;
        if (!tmpl_opt || !tmpl_opt[0]) tmpl_opt = getenv("P36_CHAT_TEMPLATE");
        if (tmpl_opt && tmpl_opt[0] &&
            strcasecmp(tmpl_opt, "off") != 0 &&
            strcasecmp(tmpl_opt, "none") != 0 &&
            strcasecmp(tmpl_opt, "false") != 0 &&
            strcasecmp(tmpl_opt, "0") != 0 &&
            strcasecmp(tmpl_opt, "default") != 0) {
            FILE *fp = fopen(tmpl_opt, "rb");
            if (!fp) {
                fprintf(stderr, "p36: failed to open chat template %s: %s\n",
                        tmpl_opt, strerror(errno));
                p36_engine_close(e);
                return 1;
            }
            if (fseek(fp, 0, SEEK_END) != 0) {
                fclose(fp);
                fprintf(stderr, "p36: failed to seek chat template %s\n", tmpl_opt);
                p36_engine_close(e);
                return 1;
            }
            long fsize = ftell(fp);
            if (fsize <= 0 || fsize > 1024 * 1024 || fseek(fp, 0, SEEK_SET) != 0) {
                fclose(fp);
                fprintf(stderr, "p36: invalid chat template size in %s\n", tmpl_opt);
                p36_engine_close(e);
                return 1;
            }
            char *buf = xmalloc((size_t)fsize + 1);
            size_t nread = fread(buf, 1, (size_t)fsize, fp);
            fclose(fp);
            buf[nread] = '\0';
            char err[256] = {0};
            if (!p36_jinja_validate(buf, err, sizeof(err))) {
                free(buf);
                fprintf(stderr, "p36: invalid chat template in %s: %s\n",
                        tmpl_opt, err[0] ? err : "validation failed");
                p36_engine_close(e);
                return 1;
            }
            e->chat_template = buf;
        }
    }
    config_validate_model(&e->model);
    hadamard_parse(&e->model);
    e->variant = g_p36_shape.variant;
    if (!p36_load_directional_steering(e)) {
        p36_engine_close(e);
        return 1;
    }
    if (opt->vision_path && opt->vision_path[0]) {
        model_open(&e->vision_model, opt->vision_path, false);
        p36_str arch = required_string(&e->vision_model, "general.architecture");
        p36_str projector = required_string(&e->vision_model, "clip.projector_type");
        p36_tensor *projection = model_find_tensor(&e->vision_model, "mm.2.weight");
        if (!p36_streq(arch, "clip") || !p36_streq(projector, "qwen3vl_merger") ||
            !projection ||
            (projection->type != P36_TENSOR_F16 && projection->type != P36_TENSOR_F32 &&
             projection->type != P36_TENSOR_Q8_0) ||
            projection->dim[0] != 4608u || projection->dim[1] != P36_N_EMBD) {
            fprintf(stderr, "p36: vision sidecar does not match the language model\n");
            p36_engine_close(e);
            return 1;
        }
#ifdef MADV_RANDOM
        madvise((void *)e->vision_model.map, (size_t)e->vision_model.size, MADV_RANDOM);
#endif
#ifdef MADV_DONTNEED
        madvise((void *)e->vision_model.map, (size_t)e->vision_model.size, MADV_DONTNEED);
#endif
#if !defined(P36_NO_GPU) && !defined(P36_METAL)
        /* Non-F16 matrices are expanded to F16 in the stream buffer. */
        uint64_t vision_stream_bytes = e->vision_model.max_tensor_bytes;
        for (uint64_t i = 0; i < e->vision_model.n_tensors; i++) {
            const p36_tensor *vt = &e->vision_model.tensors[i];
            if (vt->ndim < 2) continue;
            uint64_t f16_bytes = vt->elements * sizeof(uint16_t);
            if (f16_bytes > vision_stream_bytes) vision_stream_bytes = f16_bytes;
        }
        if (e->backend == P36_BACKEND_VULKAN &&
            !p36_gpu_vision_stream_init(vision_stream_bytes)) {
            fprintf(stderr, "p36: unable to allocate streamed vision weight buffer\n");
            p36_engine_close(e);
            return 1;
        }
#endif
        e->vision_ready = true;
        fprintf(stderr, "p36: vision sidecar mapped from disk: %s\n", opt->vision_path);
    }
    if (P36_MODEL_DENSE && e->ssd_streaming) {
        fprintf(stderr, "p36: Qwen3.6 27B dense does not use expert SSD streaming\n");
        p36_engine_close(e);
        return 1;
    }
    vocab_load(&e->vocab, &e->model);
    weights_bind(&e->weights, &e->model);
    if (P36_MODEL_DENSE && getenv("P36_DUMP_DENSE_TYPES")) {
        uint64_t iq4_xs_tensors = 0;
        uint64_t iq4_xs_bytes = 0;
        uint64_t iq4_xs_blocks = 0;
        for (uint32_t il = 0; il < P36_N_LAYER; ++il) {
            const p36_layer_weights *l = &e->weights.layer[il];
            if (l->kind == P36_LAYER_FULL_ATTN) {
                fprintf(stderr,
                        "p36: dense layer %u kind=attention q=%s k=%s v=%s o=%s ffn_gate=%s ffn_up=%s ffn_down=%s\n",
                        il, tensor_type_name(l->attn_q->type),
                        tensor_type_name(l->attn_k->type),
                        tensor_type_name(l->attn_v->type),
                        tensor_type_name(l->attn_output->type),
                        tensor_type_name(l->ffn_gate_shexp->type),
                        tensor_type_name(l->ffn_up_shexp->type),
                        tensor_type_name(l->ffn_down_shexp->type));
            } else {
                fprintf(stderr,
                        "p36: dense layer %u kind=recurrent qkv=%s z=%s beta=%s alpha=%s o=%s ffn_gate=%s ffn_up=%s ffn_down=%s\n",
                        il, tensor_type_name(l->attn_qkv->type),
                        tensor_type_name(l->attn_gate->type),
                        tensor_type_name(l->ssm_beta->type),
                        tensor_type_name(l->ssm_alpha->type),
                        tensor_type_name(l->ssm_out->type),
                        tensor_type_name(l->ffn_gate_shexp->type),
                        tensor_type_name(l->ffn_up_shexp->type),
                        tensor_type_name(l->ffn_down_shexp->type));
            }
        }
        for (uint64_t it = 0; it < e->model.n_tensors; ++it) {
            const p36_tensor *t = &e->model.tensors[it];
            if (t->type != P36_TENSOR_IQ4_XS) continue;
            iq4_xs_tensors++;
            iq4_xs_bytes += t->bytes;
            iq4_xs_blocks += t->elements / P36_QK_K;
        }
        fprintf(stderr,
                "p36: dense IQ4_XS tensors=%" PRIu64 " bytes=%" PRIu64
                " blocks=%" PRIu64 " exact-scale-expanded-bytes=%" PRIu64
                " delta=%" PRIu64 "\n",
                iq4_xs_tensors, iq4_xs_bytes, iq4_xs_blocks,
                iq4_xs_bytes + iq4_xs_blocks * 2u,
                iq4_xs_blocks * 2u);
    }
    if (e->ssd_streaming && !p36_backend_supports_ssd_streaming(e->backend)) {
        fprintf(stderr,
                "p36: --ssd-streaming requires a Metal or Vulkan GPU graph backend\n");
        p36_engine_close(e);
        return 1;
    }
    e->routed_quant_bits = p36_quant_bits_from_type(
        P36_MODEL_DENSE ? e->weights.layer[0].ffn_gate_shexp->type :
                          e->weights.layer[0].ffn_gate_exps->type);
    e->expert_weights_scale = 1.0f;
    if (!model_get_f32_compat(&e->model, "qwen35moe.expert_weights_scale", &e->expert_weights_scale))
        model_get_f32_compat(&e->model, "clover_moe.expert_weights_scale", &e->expert_weights_scale);
    e->mtp_expert_weights_scale = e->expert_weights_scale;
    if (e->variant == P36_VARIANT_CLOVER_35B_A3B) {
        uint32_t bs = 32;
        model_get_u32(&e->model, "clover_moe.block_size", &bs);
        e->clover_block_size = bs ? bs : 32;
        e->clover_mask_token_id = 151669;
        e->clover_coreset_enabled = true;
        e->clover_coreset_top_p = 0.9f;
        model_get_f32_compat(&e->model, "clover_moe.moe_block_coreset_top_p", &e->clover_coreset_top_p);
    }
    if (opt->mtp_path && opt->mtp_path[0]) {
        if (P36_MODEL_DENSE) {
            fprintf(stderr, "p36: Qwen3.6 27B dense does not support a MoE MTP model\n");
            p36_engine_close(e);
            return 1;
        }
        if (!p36_backend_uses_graph(opt->backend)) {
            fprintf(stderr, "p36: --mtp requires a GPU graph backend\n");
            p36_engine_close(e);
            return 1;
        }
#ifdef P36_NO_GPU
        fprintf(stderr, "p36: --mtp is not available in CPU-only builds\n");
        p36_engine_close(e);
        return 1;
#else
        model_open(&e->mtp_model, opt->mtp_path, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model, &e->model);
        model_get_f32_compat(&e->mtp_model, "qwen35moe.expert_weights_scale", &e->mtp_expert_weights_scale);
        e->mtp_ready = true;
        fprintf(stderr, "p36: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
#endif
    } else if (P36_MODEL_DENSE && e->mtp_draft_tokens > 1 &&
               p36_backend_uses_graph(opt->backend) &&
               tensor_by_namef(&e->model, "blk.%u.nextn.eh_proj.weight", P36_N_LAYER) &&
               tensor_by_namef(&e->model, "blk.%u.nextn.eh_proj.weight", P36_N_LAYER)->bytes != 0) {
#ifndef P36_NO_GPU
        e->mtp_model = e->model;
        e->mtp_model.owned = false;
        mtp_weights_bind_dense(&e->mtp_weights, &e->model);
        e->mtp_ready = true;
        fprintf(stderr, "p36: in-file MTP head bound (blk.%u nextn), draft=%d\n",
                P36_N_LAYER,
                e->mtp_draft_tokens);
#endif
    }
#ifndef P36_NO_GPU
    if (p36_backend_uses_graph(opt->backend)) {
        p36_gpu_set_quality(opt->quality);
        p36_gpu_set_dense_model(P36_MODEL_DENSE);
        p36_gpu_set_ssd_streaming(e->ssd_streaming);
        if (!p36_engine_configure_streaming_auto_cache(e)) {
            p36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming_full_layers > P36_N_LAYER) {
            fprintf(stderr, "p36: --ssd-streaming-full-layers must be between 0 and %u\n",
                    P36_N_LAYER);
            p36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming && e->ssd_streaming_full_layers != 0) {
            uint32_t reserve = 0;
            uint64_t resident_bytes = 0;
            if (!p36_streaming_full_layer_budget(&e->weights,
                                                  e->ssd_streaming_full_layers,
                                                  &reserve,
                                                  &resident_bytes)) {
                fprintf(stderr, "p36: SSD streaming full-layer byte accounting failed\n");
                p36_engine_close(e);
                return 1;
            }
            if ((uint64_t)reserve + P36_N_EXPERT > e->ssd_streaming_cache_experts) {
                uint32_t old = e->ssd_streaming_full_layers;
                e->ssd_streaming_full_layers = e->ssd_streaming_cache_experts > P36_N_EXPERT ?
                    (e->ssd_streaming_cache_experts - P36_N_EXPERT) / P36_N_EXPERT : 0;
                if (!p36_streaming_full_layer_budget(&e->weights, e->ssd_streaming_full_layers,
                                                      &reserve, &resident_bytes)) {
                    p36_engine_close(e);
                    return 1;
                }
                fprintf(stderr, "p36: SSD full layers reduced %u -> %u for runtime headroom\n",
                        old, e->ssd_streaming_full_layers);
            }
            e->ssd_streaming_cache_experts -= (uint32_t)reserve;
            fprintf(stderr,
                    "p36: SSD streaming keeps %u full routed layers resident "
                    "(%.2f GiB); dynamic cache %u experts\n",
                    e->ssd_streaming_full_layers,
                    (double)resident_bytes / 1073741824.0,
                    e->ssd_streaming_cache_experts);
        } else if (e->ssd_streaming && e->ssd_streaming_full_layers_set) {
            fprintf(stderr, "p36: SSD streaming full resident layers disabled\n");
        }
        p36_gpu_set_streaming_full_layers(e->ssd_streaming ?
                                          e->ssd_streaming_full_layers : 0);
        p36_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
#ifdef P36_METAL
        p36_streaming_lock_static_weights(e);
#endif
        if (e->ssd_streaming) {
            uint64_t slab_gate_bytes = 0, slab_up_bytes = 0;
            uint64_t slab_down_bytes = 0, slab_expert_bytes = 0;
            if (p36_streaming_routed_expert_layout(
                    &e->weights, &slab_gate_bytes, &slab_up_bytes,
                    &slab_down_bytes, &slab_expert_bytes)) {
                p36_gpu_set_streaming_expert_cache_layout(
                    slab_gate_bytes, slab_up_bytes, slab_down_bytes);
                uint32_t routed = 0;
                for (uint32_t il = 0; il < P36_N_LAYER; il++) {
                    const p36_layer_weights *l = &e->weights.layer[il];
                    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) continue;
                    routed++;
                }
                uint32_t cache_floor = 2u * routed * P36_N_EXPERT_USED;
                if (e->ssd_streaming_cache_experts < cache_floor) {
                    fprintf(stderr,
                            "p36: WARNING: SSD streaming cache has %u experts; %u are recommended to retain two decode routes per cacheable layer\n",
                            e->ssd_streaming_cache_experts, cache_floor);
                }
            }
        }
        if (!p36_vulkan_set_model_spans(e) ||
            !p36_gpu_set_model_fd(e->model.fd)) {
            p36_engine_close(e);
            return 1;
        }
#ifdef P36_METAL
        if (e->ssd_streaming && e->ssd_streaming_full_layers != 0) {
            for (uint32_t il = 0;
                 il < e->ssd_streaming_full_layers; il++) {
                p36_gpu_stream_expert_table table =
                    p36_stream_expert_table_make(
                        &e->model, &e->weights.layer[il], il);
                if (!p36_gpu_stream_expert_cache_load_layer(&table)) {
                    fprintf(stderr,
                            "p36: Metal failed to load full resident routed layer %u\n",
                            il);
                    p36_engine_close(e);
                    return 1;
                }
            }
        }
#endif
        p36_vulkan_prewarm_weights(e);
        if (!p36_gpu_finish_model_cache()) {
            p36_engine_close(e);
            return 1;
        }
        if (!p36_engine_seed_streaming_expert_cache(e)) {
            p36_engine_close(e);
            return 1;
        }
    }
#endif
    *out = e;
    return 0;
}

void p36_engine_close(p36_engine *e) {
    if (!e) return;
#ifndef P36_NO_GPU
    p36_gpu_tensor_free(e->directional_steering_gpu);
    p36_gpu_tensor_free(e->session_batch_logits);
    if (p36_backend_uses_graph(e->backend)) p36_gpu_cleanup();
#endif
    free(e->directional_steering_dirs);
    vocab_free(&e->vocab);
    model_close(&e->vision_model);
    model_close(&e->mtp_model);
    model_close(&e->model);
    free(e->directional_steering_file);
    free(e->chat_template);
    p36_ssd_memory_lock_release(&e->simulated_memory);
    free(e);
}

void p36_engine_summary(p36_engine *e) {
    if (!e) return;
    model_summary(&e->model);
}

bool p36_engine_has_vision(p36_engine *e) {
    return e && e->vision_ready;
}

void p36_vision_embedding_free(p36_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->data);
    memset(embedding, 0, sizeof(*embedding));
}

static int p36_engine_vision_encode(p36_engine *e, p36_image *image,
                                    p36_vision_embedding *out,
                                    char *err, size_t errlen) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!e || !e->vision_ready) {
        if (err && errlen) snprintf(err, errlen, "vision is not enabled; pass --vision FILE");
        return 0;
    }
    if (!p36_engine_uses_vulkan_runtime(e)) {
        if (err && errlen) snprintf(err, errlen, "vision requires the Vulkan backend");
        return 0;
    }
    return p36_vision_encode_image(e, image, out, err, errlen);
}

int p36_engine_vision_encode_file(p36_engine *e, const char *path,
                                  p36_vision_embedding *out,
                                  char *err, size_t errlen) {
    p36_image image = {0};
    if (!p36_image_decode_file(&image, path, err, errlen)) return 0;
    int ok = p36_engine_vision_encode(e, &image, out, err, errlen);
    p36_image_free(&image);
    return ok;
}

int p36_engine_vision_encode_memory(p36_engine *e,
                                    const uint8_t *data, size_t len,
                                    p36_vision_embedding *out,
                                    char *err, size_t errlen) {
    p36_image image = {0};
    if (!p36_image_decode_memory(&image, data, len, err, errlen)) return 0;
    int ok = p36_engine_vision_encode(e, &image, out, err, errlen);
    p36_image_free(&image);
    return ok;
}

int p36_prompt_append_vision(p36_engine *e, p36_tokens *prompt,
                             p36_vision_span *span,
                             p36_vision_embedding *embedding,
                             char *err, size_t errlen) {
    if (!e || !prompt || !span || !embedding || !embedding->data ||
        !embedding->token_count || e->vocab.vision_start_id < 0 ||
        e->vocab.vision_end_id < 0 || e->vocab.image_pad_id < 0) {
        if (err && errlen) snprintf(err, errlen, "invalid vision prompt");
        return 0;
    }
    p36_tokens_push(prompt, e->vocab.vision_start_id);
    span->token_start = (uint32_t)prompt->len;
    for (uint32_t i = 0; i < embedding->token_count; i++)
        p36_tokens_push(prompt, e->vocab.image_pad_id);
    p36_tokens_push(prompt, e->vocab.vision_end_id);
    span->embedding = *embedding;
    memset(embedding, 0, sizeof(*embedding));
    return 1;
}

int p36_engine_embd_dim(p36_engine *e) {
    return e ? (int)P36_N_EMBD : 0;
}

int p36_engine_vocab_size(p36_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

int p36_engine_power(p36_engine *e) {
    return e && e->power_percent > 0 ? e->power_percent : 100;
}

int p36_engine_set_power(p36_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *p36_engine_model_name(p36_engine *e) {
    if (e && e->kat_coder) return "kat-coder-v2.5-dev";
    if (e && e->variant == P36_VARIANT_27B) return "qwen3.6-27b";
    if (e && e->variant == P36_VARIANT_CLOVER_35B_A3B) return "clover-35b-a3b";
    return "qwen3.6-35b-a3b";
}

int p36_engine_model_id(p36_engine *e) {
    if (e && e->kat_coder) return 2;
    if (e && e->variant == P36_VARIANT_CLOVER_35B_A3B) return 4;
    return e && e->variant == P36_VARIANT_27B ? 3 : 1;
}

uint32_t p36_engine_prefill_chunk(p36_engine *e) {
    if (!e) return 0;
#ifndef P36_NO_GPU
    if (p36_engine_uses_vulkan_runtime(e)) return p36_engine_gpu_prefill_cap(e);
#endif
    return e->cpu_prefill_cap;
}

bool p36_engine_is_qwen38(p36_engine *e) {
    return e && e->variant == P36_VARIANT_27B;
}

bool p36_engine_is_kat_coder(p36_engine *e) {
    return e && e->kat_coder;
}

const char *p36_engine_chat_template(p36_engine *e) {
    return e ? e->chat_template : NULL;
}

void p36_engine_sampling_defaults(p36_engine *e, float *temperature,
                                  int *top_k, float *top_p, float *min_p) {
    if (!temperature || !top_k || !top_p || !min_p) return;
    *temperature = P36_DEFAULT_TEMPERATURE;
    *top_k = 0;
    *top_p = P36_DEFAULT_TOP_P;
    *min_p = P36_DEFAULT_MIN_P;
    if (e && e->variant == P36_VARIANT_CLOVER_35B_A3B) {
        *temperature = 0.0f;
        *top_k = 1;
        *top_p = 1.0f;
        *min_p = 0.0f;
        return;
    }
    if (!e || !e->kat_coder) return;
    *temperature = 0.7f;
    *top_k = 20;
    *top_p = 0.8f;
    *min_p = 0.05f;
}

int p36_inspect_model(const char *model_path) {
    p36_engine *e = NULL;
    p36_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = P36_BACKEND_CPU;
    if (p36_engine_open(&e, &opt) != 0) return 1;
    p36_engine_summary(e);
    p36_engine_close(e);
    return 0;
}

void p36_engine_dump_tokens(p36_engine *e, const p36_tokens *tokens) {
    if (!e || !tokens) return;
    dump_tokens_fp(stdout, &e->vocab, tokens);
}

int p36_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    p36_engine *e = NULL;
    p36_engine_options opt;
    p36_tokens tokens = {0};
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = P36_BACKEND_CPU;
    if (!fp) fp = stdout;
    if (p36_engine_open(&e, &opt) != 0) return 1;
    p36_tokenize_text(e, text ? text : "", &tokens);
    dump_tokens_fp(fp, &e->vocab, &tokens);
    p36_tokens_free(&tokens);
    p36_engine_close(e);
    return 0;
}

static int p36_unsupported_runtime(char *err, size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "Qwen inference/session path is not wired on the new baseline yet");
    return 1;
}

int p36_engine_generate_argmax(p36_engine *e, const p36_tokens *prompt,
                               int n_predict, int ctx_size,
                               p36_token_emit_fn emit,
                               p36_generation_done_fn done,
                               void *emit_ud,
                               p36_session_progress_fn progress,
                               void *progress_ud) {
    p36_session *s = NULL;
    char err[160] = {0};
    int eos;
    int generated = 0;
    double t_prefill0;
    double t_prefill1;
    double t_decode0;
    double t_decode1;
    int rc = 1;
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return 1;
    if (p36_session_create(&s, e, ctx_size) != 0 || !s) return 1;
    p36_session_set_progress(s, progress, progress_ud);
    t_prefill0 = p36_now_sec();
    if (p36_session_sync(s, prompt, err, sizeof(err)) != 0) goto done;
    t_prefill1 = p36_now_sec();
    eos = p36_token_eos(e);
    const bool prof_decode = getenv("P36_VK_PROF_DECODE") != NULL;
    if (prof_decode) p36_gpu_prof_reset();
    t_decode0 = p36_now_sec();
    int room = p36_session_ctx(s) - p36_session_pos(s);
    int max_tokens = room <= 1 ? 0 : n_predict;
    if (max_tokens > room - 1) max_tokens = room - 1;
    for (int i = 0; i < max_tokens; i++) {
        int token = p36_session_argmax(s);
        if (token < 0) goto done;
        token = p36_session_eos_to_think_close(s, token);
        if (token == eos) break;
        if (emit) emit(emit_ud, token);
        if (p36_session_eval(s, token, err, sizeof(err)) != 0) goto done;
        generated++;
    }
    t_decode1 = p36_now_sec();
    if (prof_decode) p36_gpu_prof_report("decode");
    if (done) done(emit_ud);
    p36_log(stderr,
            P36_LOG_TIMING,
            "p36: prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)generated / (t_decode1 - t_decode0) : 0.0);
    rc = 0;
done:
    p36_session_free(s);
    return rc;
}

int p36_engine_head_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_first_token_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_full_test(p36_engine *e, const p36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int p36_engine_vulkan_graph_prompt_test(p36_engine *e, const p36_tokens *prompt, int ctx_size) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    return 1;
}

int p36_engine_debug_tensor_row(p36_engine *e, const char *tensor_name, uint64_t row, float *dst, uint32_t n) {
    p36_tensor *t;
    if (!e || !tensor_name || !dst) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t) return 1;
    return p36_tensor_row_to_float(&e->model, t, row, dst, n) ? 0 : 1;
}

int p36_engine_debug_first_tensor_of_type(p36_engine *e, uint32_t type, char *name, size_t name_cap, uint32_t *n) {
    if (!e || !name || name_cap == 0) return 1;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        if (t->name.len + 1 > name_cap) return 1;
        memcpy(name, t->name.ptr, t->name.len);
        name[t->name.len] = '\0';
        if (n) *n = (uint32_t)t->dim[0];
        return 0;
    }
    return 1;
}

int p36_engine_debug_tensor_of_type_at(p36_engine *e, uint32_t type, uint32_t ordinal,
                                       char *name, size_t name_cap,
                                       uint32_t *in_dim, uint32_t *out_dim) {
    uint32_t seen = 0;
    if (!e || !name || name_cap == 0) return 1;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        if (seen++ != ordinal) continue;
        if (t->name.len + 1 > name_cap) return 1;
        memcpy(name, t->name.ptr, t->name.len);
        name[t->name.len] = '\0';
        if (in_dim) *in_dim = (uint32_t)t->dim[0];
        if (out_dim) *out_dim = t->ndim > 1 ? (uint32_t)t->dim[1] : 1u;
        return 0;
    }
    return 1;
}

uint32_t p36_engine_debug_count_tensors_of_type(p36_engine *e, uint32_t type) {
    uint32_t n = 0;
    if (!e) return 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        p36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        n++;
    }
    return n;
}

int p36_engine_debug_tensor_row_packed(p36_engine *e, const char *tensor_name, uint64_t row,
                                       void *dst, uint64_t dst_cap,
                                       uint64_t *row_bytes, uint32_t *type, uint32_t *n) {
    p36_tensor *t;
    uint64_t got_row_bytes;
    uint64_t row_count;
    const uint8_t *src;
    if (!e || !tensor_name) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t || t->ndim == 0) return 1;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return 1;
    if (row_bytes) *row_bytes = got_row_bytes;
    if (type) *type = t->type;
    if (n) *n = (uint32_t)t->dim[0];
    if (!dst) return 0;
    if (dst_cap < got_row_bytes) return 1;
    src = e->model.map + t->abs_offset + row * got_row_bytes;
    memcpy(dst, src, (size_t)got_row_bytes);
    return 0;
}

int p36_session_create(p36_session **out, p36_engine *e, int ctx_size) {
    p36_session *s;
    if (!out || !e || ctx_size <= 0 || ctx_size > P36_CONTEXT_ALLOC_MAX) return 1;
    uint64_t reserved = 0;
#ifndef P36_NO_GPU
    if (!p36_streaming_reserve_session(e, ctx_size, &reserved)) {
        fprintf(stderr, "p36: requested context leaves insufficient SSD streaming memory\n");
        return 1;
    }
#endif
    s = xcalloc(1, sizeof(*s));
    s->reserved_context_bytes = reserved;
    s->engine = e;
    s->ctx_size = ctx_size;
    s->logits = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->logits));
    s->sample_probs = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->sample_probs));
    memset(s->logits, 0, (size_t)P36_N_VOCAB * sizeof(*s->logits));
    s->logits_host_valid = true;
    s->gpu_top2_valid = false;
    if (p36_engine_uses_vulkan_runtime(e)) {
#ifdef P36_NO_GPU
        s->runtime = NULL;
#else
        s->runtime = p36_vulkan_runtime_create(ctx_size,
                                               p36_engine_gpu_prefill_cap(e),
                                               e->mtp_ready,
                                               e->lookup_k > 1 || e->variant == P36_VARIANT_CLOVER_35B_A3B,
                                               e->quality,
                                               e->ssd_streaming,
                                               e->cache_type_k,
                                               e->cache_type_v);
        if (s->runtime) {
            ((p36_vulkan_runtime *)s->runtime)->clover_block_size =
                e->clover_block_size ? e->clover_block_size : 32u;
        }
#endif
    } else {
        s->runtime = p36_cpu_runtime_create(ctx_size,
                                            e->cpu_prefill_cap,
                                            e->cache_type_k,
                                            e->cache_type_v);
    }
    if (!s->runtime) {
        free(s->logits);
        free(s->sample_probs);
        free(s->mtp_logits);
        free(s->vision_spans);
        free(s);
        return 1;
    }
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)P36_N_VOCAB * sizeof(*s->mtp_logits));
        memset(s->mtp_logits, 0, (size_t)P36_N_VOCAB * sizeof(*s->mtp_logits));
        s->mtp_draft_token = -1;
    }
    p36_session_reset_runtime(s);
    e->session_bytes += reserved;
    e->live_sessions++;
    *out = s;
    return 0;
}

