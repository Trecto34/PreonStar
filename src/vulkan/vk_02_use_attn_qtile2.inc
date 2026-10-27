static bool p36_vk_use_attn_qtile2(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_QTILE2");
}

/* Flash-attention replacement for attn_prefill_qtile2{_gqa6} (GQA 6 dense,
 * GQA 8 MoE).  P36_VK_ATTN_FA=0 falls back to qtile2 for parity A/B. */
static bool p36_vk_use_attn_fa(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_FA");
}

static bool p36_vk_use_attn_fused(void) {
    static int cached = -1;
    if (cached < 0) cached = p36_vk_env_default_on("P36_VK_ATTN_FUSED") ? 1 : 0;
    return cached != 0 && p36_vk_have_subgroups(true);
}

bool p36_gpu_attn_fused_enabled(void) {
    return !p36_gpu_quality && p36_vk_use_attn_fused();
}

static bool p36_vk_use_gpu_rms(void) {
    return p36_vk_env_default_on("P36_VK_GPU_RMS");
}

static bool p36_vk_use_add_rms(void) {
    return p36_vk_env_default_on("P36_VK_ADD_RMS");
}

static bool p36_vk_use_gpu_recurrent(void) {
    return p36_vk_env_default_on("P36_VK_GPU_RECURRENT");
}

static bool p36_vk_use_delta_fast_prefill(void) {
    return p36_vk_env_default_on("P36_VK_DELTA_FAST_PREFILL");
}

static bool p36_vk_use_delta_col_prefill(void) {
    const char *env = getenv("P36_VK_DELTA_COL_PREFILL");
    bool enabled = env && env[0] ? env[0] != '0' : !p36_vk.bc250;
    return p36_vk_have_subgroups(false) && p36_vk.subgroup_size_control &&
           enabled;
}

static bool p36_vk_use_delta_decode(void) {
    return p36_vk_env_default_on("P36_VK_DELTA_DECODE");
}

static bool p36_vk_use_delta_reg(void) {
    return p36_vk.subgroup_clustered &&
           p36_vk_env_default_on("P36_VK_DELTA_REG");
}

static bool p36_vk_use_attn_splitk(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_ATTN_SPLITK");
}

/* Width in keys of one split-K span.  Decode dispatches n_head * n_spans
 * workgroups, so on this 40-CU board the stock 512 gives 16 * 5 = 80
 * workgroups at ctx 2048 -- two per CU, and the kernel measures ~17 GB/s
 * against a ~360 GB/s roofline, i.e. it is occupancy-bound rather than
 * bandwidth-bound.  A narrower span buys occupancy there.  At ctx 8192 and up
 * the returns are small in both directions (round-2 R5): the split already
 * dispatches n_head * n_spans workgroups and 408 workgroups at ctx 8192 (10
 * per CU, 4 waves each) fill the board, so the kernel is instruction-bound,
 * not occupancy-bound -- attn_decode_split.spv is VGPR 48 / LDS 1536 B / 20
 * subgroups per SIMD, i.e. nothing is register- or LDS-limited.
 *
 * NOT bit-exact: attn_combine walks the per-span partials sequentially in
 * f32, rescaling by exp(m - nm) at each step, so changing the span count
 * regroups that reduction and changes rounding.
 *
 * The cost of narrowing scales with context, in BOTH directions, and this is
 * why the default stays 512:
 *
 *   spans = kv_max / span, so at ctx 32768 a 128-key span puts 256 sequential
 *   f32 rescales in the chain where 512 puts 64.  Online softmax keeps that
 *   bounded, not free -- drift grows with span count, hence with context.
 *
 *   attn_combine's own cost grows the same way: measured 10.302 -> 14.870 ms
 *   at ctx 2048 going 512 -> 128, and that term scales with span count while
 *   the split-side saving does not.
 *
 * The +2.2% for span 128 was measured at ctx 2048 only (16 spans vs 4) and
 * MUST NOT be assumed to hold at long context -- neither the gain nor the
 * numerical cost has been measured beyond 2064.  Long context is measured now
 * (ctx 16384, split ms/call, 1 rep per arm, fresh process): 512 -> 0.5334 ms
 * with 792 workgroups on Swift (24 q heads) and 0.3588 ms with 528 on the
 * guard (16 q heads); 1024 -> +7.1% / +12.3%, 2048 -> +28.1% / +39.4%,
 * 4096 -> +70.5% / +96.5%, i.e. WIDENING starves the grid monotonically (per
 * key 1.32 -> 1.85 ns Swift, 1.33 -> 2.15 ns guard).  Narrowing is monotone and
 * real but small: 256 -> -4.2% and 128 -> -7.0% of the split on Swift
 * (1.315 -> 1.280 -> 1.252 ns per key), 128 -> -9.9% on the guard, and the
 * guard's 256 arm disagrees with that trend (+4.0%, single rep), so the narrow
 * side is not established.  attn_combine pays it back: per call it is
 * 0.06037 + 1.02e-4 * spans ms on Swift and 0.04006 + 1.03e-4 * spans ms on
 * the guard (residual <= 0.0003 ms over 5..129 spans) -- a fixed cost that
 * scales with the head count plus ~1 us per partial, so at span 128 the
 * combine is +15% / +22% over 512.  Net: at most ~1% of the decode token on
 * Swift, ~2-3% on the guard, both single-rep, on top of the drift above.
 * 512 stays until someone pays for an interleaved A/B plus the NLL check.
 * Evidence: karpathy/evidence/raw/r5-longctx-attn/. */
static uint32_t p36_vk_attn_span(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("P36_VK_ATTN_SPAN");
        long v = env && env[0] ? strtol(env, NULL, 10) : 512;
        if (v < 64 || v > 4096 || (v & (v - 1)) != 0) v = 512;
        cached = (int)v;
    }
    return (uint32_t)cached;
}

static uint32_t p36_vk_kv_cache_row_bytes(uint32_t type, uint32_t n) {
    if (type == 0u) return n * (uint32_t)sizeof(uint16_t);
    if (type == 1u) return ((n + P36_VK_QK8_0 - 1u) / P36_VK_QK8_0) * 34u;
    if (type == 2u) return ((n + P36_VK_QK8_0 - 1u) / P36_VK_QK8_0) * 18u;
    return 0;
}

static bool p36_vk_use_f32_fast(void) {
    return p36_vk_have_subgroups(false) && p36_vk_env_default_on("P36_VK_F32_FAST");
}

static bool p36_vk_use_gpu_ffn_tail(void) {
    return p36_vk_env_default_on("P36_VK_GPU_FFN_TAIL");
}

static bool p36_vk_use_gpu_swiglu(void) {
    return p36_vk_env_default_on("P36_VK_GPU_SWIGLU");
}

static bool p36_vk_use_moe_fast(void) {
    return p36_vk_have_subgroups(true) && p36_vk_env_default_on("P36_VK_MOE_FAST");
}

static bool p36_vk_use_dense_iq3_fast(void) {
    return p36_vk.have_storage8 && p36_vk.have_int_dot &&
           p36_vk.subgroup_size == 64u &&
           p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_DENSE_IQ3_FAST");
}

static bool p36_vk_use_moe_gate_up(void) {
    return p36_vk_env_default_on("P36_VK_MOE_GATE_UP");
}

static bool p36_vk_use_router_fast_prefill(void) {
    const char *env = getenv("P36_VK_ROUTER_FAST_PREFILL");
    return env && env[0] && env[0] != '0';
}

static bool p36_vk_use_q8_0_gpu(void) {
    return p36_vk_env_default_on("P36_VK_Q8_0_GPU");
}

static bool p36_vk_use_q8_0_f32b(void) {
    return p36_vk.have_storage16 && p36_vk_have_subgroups(false) &&
           p36_vk_env_default_on("P36_VK_Q8_0_F32B");
}

static int p36_vk_quantize_q8_k_dispatch(p36_gpu_tensor *q8,
                                         const p36_gpu_tensor *x,
                                         uint64_t in_dim,
                                         uint64_t n_tok) {
    uint64_t x_bytes = 0;
    uint64_t q8_bytes = 0;
    const uint64_t blocks = in_dim / P36_VK_QK_K;

    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0) return 0;
    if (in_dim > UINT32_MAX || n_tok > UINT32_MAX || blocks > UINT32_MAX) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !p36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes)) {
        return 0;
    }
    if (!p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(x, 0, x_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) {
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } push = { (uint32_t)in_dim, (uint32_t)blocks, (uint32_t)n_tok };
    const p36_gpu_tensor *bindings[2] = { x, q8 };

    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_run_unlocked("q8_k_quant",
                                 &p36_vk.quantize_q8_k,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (uint32_t)blocks,
                                 (uint32_t)n_tok,
                                 1);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_quantize_q8_k_tensor(p36_gpu_tensor *out,
                                 const p36_gpu_tensor *x,
                                 uint64_t in_dim,
                                 uint64_t n_tok) {
    uint64_t blocks = in_dim / P36_VK_QK_K;
    uint64_t x_bytes = 0;
    uint64_t q8_count = 0;
    uint64_t q8_bytes = 0;

    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || n_tok == 0) return 0;
    if (!p36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !p36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_count) ||
        !p36_u64_mul_ok(q8_count, sizeof(p36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(x, 0, x_bytes) || !p36_gpu_tensor_range_ok(out, 0, q8_bytes)) return 0;

    if (!p36_gpu_quality && in_dim == 512u && n_tok == 8u && p36_vk_env_default_on("P36_VK_Q8_K_MOE_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    /* Decode/MTP-verify activations quantize on GPU too: the host quantizer
     * here was the largest remaining decode flush cause after GPU RMS. */
    if (!p36_gpu_quality && n_tok <= 8u && p36_vk_env_default_on("P36_VK_Q8_K_MOE_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    if (!p36_gpu_quality && n_tok > 8u && p36_vk_env_default_on("P36_VK_Q8_GPU") &&
        p36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }

    const float *x_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)x, "submit_wait_q8k_quant_x");
    p36_vk_block_q8_K *q8_host = p36_vk_tensor_contents_labeled(out, "submit_wait_q8k_quant_out");
    if (!x_host || !q8_host) return 0;
    p36_vk_quantize_q8_k_host(q8_host, x_host, n_tok, in_dim);
    return 1;
}


static float p36_vk_sigmoid_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}




static p36_gpu_tensor *p36_gpu_tensor_root(const p36_gpu_tensor *tensor, uint64_t *offset) {
    if (!tensor) return NULL;
    if (!tensor->owner) {
        if (offset) *offset = tensor->offset;
        return (p36_gpu_tensor *)tensor;
    }
    uint64_t root_offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor->owner, &root_offset);
    if (offset) *offset = root_offset + tensor->offset;
    return root;
}

static bool p36_gpu_tensor_range_ok(const p36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes) {
    return tensor && offset <= tensor->bytes && bytes <= tensor->bytes - offset;
}

static const unsigned char *p36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes) {
    const unsigned char *map = model_map ? (const unsigned char *)model_map :
                                           (const unsigned char *)p36_gpu_model_map;
    const uint64_t size = model_map ? model_size : p36_gpu_model_size;
    if (!map) return NULL;
    if (weight_offset > size || bytes > size - weight_offset) return NULL;
    return map + weight_offset;
}

static uint32_t p36_vk_find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < p36_vk.mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) == 0) continue;
        if ((p36_vk.mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

static uint32_t p36_vk_find_memory_type_avoiding(uint32_t bits,
                                                 VkMemoryPropertyFlags want,
                                                 VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < p36_vk.mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) == 0) continue;
        VkMemoryPropertyFlags flags = p36_vk.mem_props.memoryTypes[i].propertyFlags;
        if ((flags & want) == want && (flags & avoid) == 0) return i;
    }
    return UINT32_MAX;
}

static uint64_t p36_vk_memory_total(void) {
    uint64_t total = 0;
    const bool uma = p36_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

    for (uint32_t i = 0; i < p36_vk.mem_props.memoryHeapCount; i++) {
        VkMemoryHeap heap = p36_vk.mem_props.memoryHeaps[i];
        if (uma || (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) total += heap.size;
    }
    return total;
}

static void p36_vk_kernel_destroy(p36_vk_kernel *k) {
    if (!p36_vk.device || !k) return;
    if (k->pipeline) vkDestroyPipeline(p36_vk.device, k->pipeline, NULL);
    if (k->layout) vkDestroyPipelineLayout(p36_vk.device, k->layout, NULL);
    if (k->set_layout) vkDestroyDescriptorSetLayout(p36_vk.device, k->set_layout, NULL);
    if (k->shader) vkDestroyShaderModule(p36_vk.device, k->shader, NULL);
    k->pipeline = VK_NULL_HANDLE;
    k->layout = VK_NULL_HANDLE;
    k->set_layout = VK_NULL_HANDLE;
    k->shader = VK_NULL_HANDLE;
    k->ready = false;
}

static void *p36_read_file(const char *path, size_t *len_out) {
    char rooted[PATH_MAX];
    FILE *fp = NULL;
    const char *env_dir = getenv("P36_SHADER_DIR");
    if (path && path[0] != '/' && env_dir && env_dir[0] &&
        snprintf(rooted, sizeof(rooted), "%s/%s", env_dir, path) < (int)sizeof(rooted)) {
        fp = fopen(rooted, "rb");
    }
    if (!fp && path && path[0] != '/' && p36_vk.shader_root[0] &&
        snprintf(rooted, sizeof(rooted), "%s/%s",
                 p36_vk.shader_root, path) < (int)sizeof(rooted)) {
        fp = fopen(rooted, "rb");
    }
    if (!fp && path && path[0] != '/') {
        char exe[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (n > 0) {
            exe[n] = '\0';
            char *slash = strrchr(exe, '/');
            if (slash) {
                *slash = '\0';
                if (snprintf(rooted, sizeof(rooted), "%s/%s", exe, path) < (int)sizeof(rooted)) {
                    fp = fopen(rooted, "rb");
                }
            }
        }
    }
    if (!fp) fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len <= 0 || (len & 3) != 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    void *buf = malloc((size_t)len);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *len_out = (size_t)len;
    return buf;
}

static int p36_vk_kernel_init(p36_vk_kernel *k) {
    if (k->ready) return 1;
    if (k->bindings > 16 ||
        k->bindings > p36_vk.props.limits.maxPerStageDescriptorStorageBuffers ||
        k->bindings > p36_vk.props.limits.maxDescriptorSetStorageBuffers) return 0;

    size_t code_size = 0;
    void *code = p36_read_file(k->path, &code_size);
    if (!code) {
        fprintf(stderr, "p36: failed to read Vulkan shader %s\n", k->path);
        return 0;
    }

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = code,
    };
    VkResult rc = vkCreateShaderModule(p36_vk.device, &smci, NULL, &k->shader);
    free(code);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateShaderModule failed for %s (%d)\n", k->path, rc);
        return 0;
    }

    VkDescriptorSetLayoutBinding bindings[16];
    for (uint32_t i = 0; i < k->bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = k->bindings,
        .pBindings = bindings,
    };
    rc = vkCreateDescriptorSetLayout(p36_vk.device, &dsci, NULL, &k->set_layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateDescriptorSetLayout failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = k->push_bytes,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k->set_layout,
        .pushConstantRangeCount = k->push_bytes ? 1u : 0u,
        .pPushConstantRanges = k->push_bytes ? &push : NULL,
    };
    rc = vkCreatePipelineLayout(p36_vk.device, &plci, NULL, &k->layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreatePipelineLayout failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = k->shader,
        .pName = "main",
    };
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32u,
    };
    /* Shaders that declare local_size_x = 32 and use no cross-lane operation
     * at all.  On wave64 the upper half of every wave is inactive, and since
     * nothing reads a neighbouring lane the wave width cannot change a
     * single result bit -- forcing wave32 only stops wasting half of each
     * wave.  Each entry was checked for zero cross-lane builtin uses in its
     * .comp source; do not add a shader here without that check, because
     * for a shader that does reduce across lanes the two widths are not
     * interchangeable.
     *
     * Two traps found by audit (karpathy/evidence/wave32-eligibility.md):
     * (1) a `gl_Subgroup*` READ is not a builtin *call*, but
     * `gl_SubgroupID`/`gl_SubgroupInvocationID` are just as lane-count
     * dependent -- a shader deriving an index from them is disqualified too;
     * (2) when a tile is partitioned by gl_SubgroupID with a 32-wide stride,
     * wave32 is a *correctness requirement*, not a tuning choice: at wave64
     * only 2 subgroups exist and the tail of every tile is written by
     * nobody.  That is why the whole dense `_mmq` family is forced here. */
    static const char *const p36_vk_force_wave32[] = {
        "delta_net_cols",
        "rope_qwen.spv",
        "rope_qwen_mrope.spv",
        "quantize_q8_0.spv",
    };
    /* The old suffix test was "_mmq.spv", which misses the _r4 / _bm64
     * variants because those end in "_mmq_r4.spv".  Specifically
     * dense_iq3_s_mmq_r4 -- same source, same gl_SubgroupID token formula,
     * selected at runtime for IQ3_S with out_dim == 48 && n_tok <= 128 --
     * therefore ran at wave64, where only 2 subgroups exist and tokens
     * 80..127 of every 128-token tile are written by nobody.  Match the
     * whole dense _mmq family, suffix variants included. */
    bool force_wave32 = strstr(k->path, "dense_") && strstr(k->path, "_mmq");
    for (size_t i = 0; !force_wave32 && i < sizeof(p36_vk_force_wave32) / sizeof(*p36_vk_force_wave32); i++) {
        force_wave32 = strstr(k->path, p36_vk_force_wave32[i]) != NULL;
    }
    static const char *const p36_vk_clean_scan_wave32[] = {
        "moe_matvec.spv",
        "moe_matvec_fast.spv",
        "matmul_q8_0_decode.spv",
        "matmul_q8_0_decode_b64.spv",
        "matmul_q8_0_decode_p36.spv",
        "add_rms_norm.spv",
        "rms_norm.spv",
        "rms_norm_rope_qwen.spv",
        "rms_norm_rope_kv_qwen.spv",
        "kv_store_quant.spv",
    };
    const char *clean_env = getenv("P36_VK_WAVE32_CLEAN");
    if (clean_env && clean_env[0] && clean_env[0] != '0') {
        for (size_t i = 0; !force_wave32 && i < sizeof(p36_vk_clean_scan_wave32) / sizeof(*p36_vk_clean_scan_wave32); i++) {
            force_wave32 = strstr(k->path, p36_vk_clean_scan_wave32[i]) != NULL;
        }
    }
    /* moe_down_q2k_sum_decode declares local_size_x = 32 and reduces with one
     * subgroupAdd per expert.  At wave64 half of every wave is inactive, which
     * is exactly the shape this kernel shows in the profile: 2048 workgroups,
     * 32 threads each, 56 GB/s against 132 GB/s for its 64-thread sibling
     * moe_gate_up_decode on the same access pattern.  Env-gated until the
     * measurement and the logits md5 both clear. */
    if (!force_wave32 && getenv("P36_VK_WAVE32_MOE_DOWN") &&
        strstr(k->path, "moe_down_q2k_sum_decode"))
        force_wave32 = true;
    /* dense_extra_decode_ptq1_0 is the mirror-image trap: it partitions all
     * 128 weights of a PTQ1_0 block across every lane of the 64-wide
     * workgroup (b_start = lid>>3) and recovers the true dot product with
     * one subgroupAdd(acc[r]) over the whole workgroup. That is only
     * correct if the workgroup compiles to a single 64-wide subgroup --
     * the ptq1_0 == 64u dispatch gate above reads the device's *default*
     * subgroup size, not what RADV actually picks for this pipeline, so
     * pin it explicitly instead of trusting the heuristic. */
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size64 = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 64u,
    };
    bool force_wave64 = strstr(k->path, "dense_extra_decode_ptq1_0") != NULL;
    if (p36_vk.subgroup_size_control && force_wave32) {
        stage.pNext = &subgroup_size;
        stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    } else if (p36_vk.subgroup_size_control && p36_vk.subgroup_size == 64u && (force_wave64 || strstr(k->path,"dense_extra_decode_q2_0") != NULL || strstr(k->path,"dense_extra_decode_pq2_0") != NULL || strstr(k->path,"dense_extra_small_") != NULL)) {
        stage.pNext = &subgroup_size64;
        stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = k->layout,
    };
    /* P36_VK_SHADER_TRACE=1 names each pipeline as it is built, so an
     * RADV_DEBUG=shaderstats dump (which prints VGPR/LDS with no shader
     * name) can be attributed to a shader. */
    if (getenv("P36_VK_SHADER_TRACE")) fprintf(stderr, "p36: building pipeline %s%s\n", k->path, force_wave32 ? " [wave32]" : "");
    rc = vkCreateComputePipelines(p36_vk.device, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "p36: vkCreateComputePipelines failed for %s (%d)\n", k->path, rc);
        p36_vk_kernel_destroy(k);
        return 0;
    }

    k->ready = true;
    return 1;
}

static p36_gpu_tensor *p36_vk_tensor_alloc_dl_unlocked(uint64_t bytes) {
    p36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->bytes = bytes;
    tensor->alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tensor->alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(p36_vk.device, &bci, NULL, &tensor->buffer) != VK_SUCCESS) {
        free(tensor);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p36_vk.device, tensor->buffer, &req);
    tensor->private_charge = req.size;
    uint32_t mt = p36_vk_find_memory_type_avoiding(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(p36_vk.device, &mai, NULL, &tensor->memory) != VK_SUCCESS) {
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (vkBindBufferMemory(p36_vk.device, tensor->buffer, tensor->memory, 0) != VK_SUCCESS) {
        vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }

    tensor->device_local = true;
    p36_gpu_live_bytes += bytes;
    if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
    return tensor;
}

