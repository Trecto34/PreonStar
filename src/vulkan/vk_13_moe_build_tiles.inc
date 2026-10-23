static p36_gpu_tensor *p36_vk_moe_build_tiles(const uint32_t *selected_host,
                                               uint64_t n_slot,
                                               uint32_t n_expert,
                                               const uint32_t *expert_slot_map,
                                               uint32_t *tile_count_out) {
    return p36_vk_moe_build_tiles_ex(selected_host, n_slot, n_expert,
                                     expert_slot_map, P36_VK_MOE_PAIR_TILE,
                                     false, tile_count_out);
}

/* One routed expert matvec batch over n_tok tokens.  Pairs are grouped by
 * expert on the host so workgroup y can own one expert and up to
 * P36_VK_MOE_PAIR_TILE pair ids, loading each expert row block once and
 * updating that many q8 accumulation chains before advancing.
 *
 * The whole expert bank and its .scale tensor are cached on device through
 * the weight cache and the router's expert ids index them directly, so a
 * decode step moves no expert bytes at all.  If the bank does not fit the
 * heaps the runtime falls back to packing one copy per selected expert. */
static int p36_vk_moe_matvec(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const p36_gpu_moe_weight *w,
                              const p36_gpu_tensor *qx,
                              bool x_per_expert,
                              bool fast_allowed,
                              p36_vk_stream_part stream_part,
                              const uint32_t *stream_slot_map,
                              bool resident_bank,
                              const p36_gpu_tensor *selected,
                              uint32_t n_used,
                              uint32_t n_tok,
                             uint32_t in_dim,
                             uint32_t out_dim,
                             uint32_t n_expert) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(w->type);
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    uint64_t packed_weight_bytes = 0;
    uint64_t packed_scale_bytes = 0;
    const unsigned char *source;
    const unsigned char *scales_source = NULL;
    const uint32_t *selected_host;
    p36_gpu_tensor *tiles = NULL;
    p36_gpu_tensor *packed_weights = NULL;
    p36_gpu_tensor *packed_scales = NULL;
    const uint32_t *tiles_host;
    uint32_t tile_count = 0;
    uint32_t packed_count = 0;
    uint32_t packed_expert[256];
    uint32_t packed_slot[256];
    unsigned char *packed_weight_host;
    float *packed_scales_host = NULL;

    if (block_bytes == 0 || in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || out_dim == 0 || n_tok == 0) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * out_dim;
    if (expert_stride > UINT32_MAX ||
        !p36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    source = p36_gpu_weight_bytes(model_map, model_size, w->offset, weight_bytes);
    if (!source) return 0;
    if (w->has_scales) {
        scales_source = p36_gpu_weight_bytes(model_map, model_size, w->scales_offset, scale_bytes);
        if (!scales_source) return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, n_slot * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (x_per_expert ? n_slot : n_tok) * blocks * sizeof(p36_vk_block_q8_K))) {
        return 0;
    }

    tiles_host = NULL;
    if (!p36_gpu_quality && n_slot <= 4096u && stream_part == P36_VK_STREAM_PART_NONE &&
        (!p36_vk_stream.enabled || resident_bank) && p36_vk_moe_bank_cache) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, P36_VK_MOE_PAIR_TILE);
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, P36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_moe_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = p36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_part != P36_VK_STREAM_PART_NONE ? stream_slot_map : NULL,
                                       &tile_count);
        tiles_host = tiles ? p36_vk_tensor_contents_labeled(tiles, "submit_wait_moe_tiles_read") : NULL;
        if (!tiles || !tiles_host || tile_count == 0) {
            p36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t type;
        uint32_t n_used;
        uint32_t has_scales;
        uint32_t x_per_expert;
        uint32_t packed_weights;
    } push = {
        out_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        w->type, n_used, w->has_scales ? 1u : 0u, x_per_expert ? 1u : 0u, 0u,
    };

    if (stream_part != P36_VK_STREAM_PART_NONE && stream_slot_map) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        const p36_gpu_tensor *bank = p36_vk_stream_weight_tensor_unlocked(stream_part);
        const p36_gpu_tensor *bank_scales = p36_vk_stream_scale_tensor_unlocked(stream_part);
        int ok = tables && bank && (!w->has_scales || bank_scales);
        if (ok) {
            const p36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, w->has_scales ? bank_scales : bank, out, tables,
            };
            p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
            ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down_stream" :
                                     (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec_stream" : "moe_matvec_stream"),
                                     kernel, bindings, &push, sizeof(push),
                                     (out_dim + 7u) / 8u, tile_count, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (ok) {
            p36_gpu_tensor_free(tiles);
            return 1;
        }
    }

    if ((!p36_vk_stream.enabled || resident_bank) && p36_vk_moe_bank_cache) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *bank = tables ? p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        p36_gpu_tensor *bank_scales = NULL;
        if (bank && w->has_scales) {
            bank_scales = p36_vk_weight_get_unlocked(scales_source, scale_bytes);
            if (!bank_scales) bank = NULL;
        }
        if (bank) {
            const p36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, bank_scales ? bank_scales : bank, out, tables,
            };
            p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
            int ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                         (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                         kernel, bindings, &push, sizeof(push),
                                         (out_dim + 7u) / 8u, tile_count, 1);
            pthread_mutex_unlock(&p36_vk_mu);
            p36_gpu_tensor_free(tiles);
            return ok;
        }
        p36_vk_moe_bank_cache = false;
        pthread_mutex_unlock(&p36_vk_mu);
        fprintf(stderr, "p36: Vulkan expert bank cache exhausted device memory, "
                        "falling back to per-call expert staging\n");
    }

    /* Packed per-call staging needs expert ids on the host; GPU-built tiles
     * only serve the resident-bank dispatch above. */
    if (!tiles_host) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    memset(packed_slot, 0xff, sizeof(packed_slot));
    for (uint64_t slot = 0; slot < n_slot; slot++) {
        uint32_t expert = selected_host[slot];
        if (packed_slot[expert] != UINT32_MAX) continue;
        packed_slot[expert] = packed_count;
        packed_expert[packed_count++] = expert;
    }
    p36_gpu_tensor_free(tiles);
    tiles = p36_vk_moe_build_tiles(selected_host, n_slot, n_expert,
                                   packed_slot, &tile_count);
    if (!tiles || tile_count == 0 ||
        !p36_u64_mul_ok(packed_count, expert_stride, &packed_weight_bytes)) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    packed_weights = p36_gpu_tensor_alloc(packed_weight_bytes);
    if (w->has_scales) {
        packed_scale_bytes = (uint64_t)packed_count * sizeof(float);
        packed_scales = p36_gpu_tensor_alloc(packed_scale_bytes);
    }
    packed_weight_host = packed_weights ? p36_vk_tensor_contents_labeled(packed_weights, "submit_wait_moe_pack_weights") : NULL;
    packed_scales_host = packed_scales ? p36_vk_tensor_contents_labeled(packed_scales, "submit_wait_moe_pack_scales") : NULL;
    if (!packed_weights || !packed_weight_host ||
        (w->has_scales && (!packed_scales || !packed_scales_host))) {
        p36_gpu_tensor_free(packed_scales);
        p36_gpu_tensor_free(packed_weights);
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    for (uint32_t slot = 0; slot < packed_count; slot++) {
        const uint32_t expert = packed_expert[slot];
        memcpy(packed_weight_host + (uint64_t)slot * expert_stride,
               source + (uint64_t)expert * expert_stride,
               (size_t)expert_stride);
        if (packed_scales_host) packed_scales_host[slot] = ((const float *)scales_source)[expert];
    }
    push.packed_weights = 0u;

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    int ok = tables != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            packed_weights,
            qx,
            tiles,
            w->has_scales ? packed_scales : packed_weights,
            out,
            tables,
        };
        p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast() && fast_allowed) ? &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
        ok = p36_vk_run_unlocked(w->type == P36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                 (w->type == P36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                 kernel, bindings, &push, sizeof(push),
                                 (out_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(packed_scales);
    p36_gpu_tensor_free(packed_weights);
    p36_gpu_tensor_free(tiles);
    return ok;
}

/* IQ2_S sibling of p36_gpu_matmul_q8_0_pair_scaled_tensor: two
 * independently-scaled IQ2_S rows sharing one Q8_K activation read, one
 * dispatch. q8 must already be a valid Q8_K quantization of x at in_dim;
 * the caller owns quantizing it once and reusing it for both legs, which
 * is itself part of the win over two separate calls each re-quantizing. */
int p36_gpu_matmul_iq2s_pair_scaled_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_a_dim,
                                           uint64_t out_b_dim,
                                           const p36_gpu_tensor *q8,
                                           float scale_a,
                                           float scale_b) {
    if (p36_gpu_quality || !p36_vk_env_default_on("P36_VK_IQ2S_PAIR") ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        out_a_dim == 0 || out_b_dim == 0 ||
        in_dim > UINT32_MAX || out_a_dim > UINT32_MAX || out_b_dim > UINT32_MAX) return 0;
    uint64_t blocks = in_dim / P36_VK_QK_K;
    uint64_t block_bytes = p36_vk_iq_block_bytes(P36_VK_TENSOR_IQ2_S);
    uint64_t row_bytes, weight_a_bytes, weight_b_bytes, out_a_bytes, out_b_bytes, q8_bytes;
    if (!p36_u64_mul_ok(blocks, block_bytes, &row_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_a_dim, &weight_a_bytes) ||
        !p36_u64_mul_ok(row_bytes, out_b_dim, &weight_b_bytes) ||
        !p36_u64_mul_ok(out_a_dim, sizeof(float), &out_a_bytes) ||
        !p36_u64_mul_ok(out_b_dim, sizeof(float), &out_b_bytes) ||
        !p36_u64_mul_ok(blocks, sizeof(p36_vk_block_q8_K), &q8_bytes) ||
        row_bytes > UINT32_MAX) return 0;
    if (!p36_gpu_tensor_range_ok(q8, 0, q8_bytes) ||
        !p36_gpu_tensor_range_ok(out_a, 0, out_a_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_b_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_a_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_b_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_a;
        uint32_t out_b;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale_a;
        float scale_b;
    } push = { (uint32_t)out_a_dim, (uint32_t)out_b_dim, (uint32_t)blocks,
               (uint32_t)row_bytes, scale_a, scale_b };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights_a = tables ? p36_vk_weight_get_unlocked(wa, weight_a_bytes) : NULL;
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_b_bytes) : NULL;
    int ok = weights_a != NULL && weights_b != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = { weights_a, weights_b, q8, tables, out_a, out_b };
        uint32_t groups = (uint32_t)(out_a_dim > out_b_dim ? out_a_dim : out_b_dim);
        ok = p36_vk_run_unlocked("dense_extra_decode_iq2s_pair",
                                 &p36_vk.dense_extra_decode_iq2s_pair,
                                 bindings, &push, sizeof(push), groups, 1, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_iq_quant_q8_scaled_tensor(p36_gpu_tensor *out,
                                              const void *model_map,
                                              uint64_t model_size,
                                              uint64_t weight_offset,
                                              uint32_t weight_type,
                                              uint64_t in_dim,
                                              uint64_t out_dim,
                                              const p36_gpu_tensor *q8,
                                              uint64_t n_tok,
                                              float scale) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(weight_type);
    uint64_t blocks, row_bytes, weight_bytes, out_bytes, q8_bytes;
    uint32_t tile_count;
    p36_gpu_tensor *tiles;
    uint32_t *tp;
    const unsigned char *source;
    int ok;

bool extra_type = weight_type == P36_VK_TENSOR_IQ2_XXS ||
                      weight_type == P36_VK_TENSOR_IQ2_XS ||
                      weight_type == P36_VK_TENSOR_IQ2_S ||
                      weight_type == P36_VK_TENSOR_IQ1_S ||
                      weight_type == P36_VK_TENSOR_IQ4_NL ||
                      weight_type == P36_VK_TENSOR_PQ2_0 ||
                      weight_type == P36_VK_TENSOR_PTQ1_0 ||
                      /* IQ1_M prefill uses the shared 128-token mmq tile; the
                       * naive dense_iq1_m re-read the weights once per token.
                       * Decode and 2..8-row batches (MTP verify) stay on
                       * dense_iq1_m, row-identical to one-token decode. */
                      (weight_type == P36_VK_TENSOR_IQ1_M && n_tok > 8u &&
                       !p36_gpu_quality && p36_vk_env_default_on("P36_VK_DENSE_IQ1M_MMQ"));
    if ((weight_type != P36_VK_TENSOR_IQ3_XXS &&
         weight_type != P36_VK_TENSOR_IQ3_S &&
         weight_type != P36_VK_TENSOR_IQ4_XS &&
         weight_type != P36_VK_TENSOR_IQ1_M && !extra_type) ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        out_dim == 0 || n_tok == 0 || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes) ||
        weight_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(n_tok, out_dim, &out_bytes) ||
        !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    source = p36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    if (extra_type) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *weights = tables ?
            p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            const char *op = p36_vk_prof_iq3_shape(
                weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                (uint32_t)out_dim, n_tok == 1u ?
                    "dense_extra_decode" : "dense_extra_mmq");
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    uint32_t type;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, weight_type, scale,
                };
bool q2_family = weight_type == P36_VK_TENSOR_Q2_0 ||
                                  weight_type == P36_VK_TENSOR_PQ2_0;
                bool ptq1_0 = weight_type == P36_VK_TENSOR_PTQ1_0;
                /* The generic dense_extra_decode shader knows Q2_0 but not
                 * PQ2_0, and silently misreading type 142 as type 42 would
                 * produce plausible garbage.  Refuse loudly instead: the
                 * specialized kernel is the only correct PQ2_0 path, and
                 * teaching the generic shader would change SPIR-V that the
                 * IQ2/IQ3 models also run. */
                if (weight_type == P36_VK_TENSOR_PQ2_0 &&
                    !(p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                      p36_vk.subgroup_arithmetic)) {
                    fprintf(stderr, "p36: pq2_0 needs integer dot product, "
                                    "subgroup size 64 and subgroup arithmetic\n");
                    pthread_mutex_unlock(&p36_vk_mu);
                    return 0;
                }
                if (ptq1_0 &&
                    !(p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                      p36_vk.subgroup_arithmetic)) {
                    fprintf(stderr, "p36: ptq1_0 needs integer dot product, "
                                    "subgroup size 64 and subgroup arithmetic\n");
                    pthread_mutex_unlock(&p36_vk_mu);
                    return 0;
                }
                bool iq2s_fast = weight_type == P36_VK_TENSOR_IQ2_S &&
                                 p36_vk_env_default_on("P36_VK_IQ2S_EXTRA");
                p36_vk_kernel *decode_kernel =
                    q2_family &&
                    p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                    p36_vk.subgroup_arithmetic ?
                        (weight_type == P36_VK_TENSOR_PQ2_0 ?
                            &p36_vk.dense_extra_decode_pq2_0 :
                            &p36_vk.dense_extra_decode_q2_0) :
                    ptq1_0 && p36_vk.have_int_dot && p36_vk.subgroup_size == 64u &&
                    p36_vk.subgroup_arithmetic ?
                        &p36_vk.dense_extra_decode_ptq1_0 :
                    iq2s_fast ? &p36_vk.dense_extra_decode_iq2s :
                        &p36_vk.dense_extra_decode;
                ok = p36_vk_run_unlocked(
                    op, decode_kernel,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    uint32_t type;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes,
                    weight_type, scale,
                };
                bool is_pq2_0 = weight_type == P36_VK_TENSOR_PQ2_0;
                bool is_ptq1_0 = weight_type == P36_VK_TENSOR_PTQ1_0;
                bool is_q2_0 = weight_type == P36_VK_TENSOR_Q2_0 || is_pq2_0;
                /* Small batches (MTP verify, prompt tails, short incremental
                 * prefills) would pay a whole 128-token mmq tile.  Stream the
                 * weights once per 8-token group instead; each row is
                 * bit-identical to the one-token decode kernel. */
                bool small_ok = is_q2_0 && p36_vk.have_int_dot &&
                                p36_vk.subgroup_size == 64u && p36_vk.subgroup_arithmetic;
                /* A ragged tail past whole 128-token tiles (the end of any
                 * prompt) would cost another full tile; hand it to the small
                 * kernel and let mmq take only the aligned head. */
                uint32_t small_first = 0;
                if (small_ok && n_tok > p36_vk_q2_small_max()) {
                    uint32_t rem = (uint32_t)n_tok % 128u;
                    if (rem == 0u || rem > p36_vk_q2_small_max()) small_ok = false;
                    else small_first = (uint32_t)n_tok - rem;
                }
                if (small_ok && small_first) {
                    push.n_tok = small_first;
                    ok = p36_vk_run_unlocked(
                        op, is_pq2_0 ? &p36_vk.dense_extra_mmq_pq2_0 : &p36_vk.dense_extra_mmq_q2_0,
                        bindings, &push, sizeof(push),
                        ((uint32_t)out_dim + 63u) / 64u, small_first / 128u, 1);
                }
                if (small_ok) {
                    p36_vk_kernel *small = is_pq2_0 ? &p36_vk.dense_extra_small_pq2_0 :
                                                      &p36_vk.dense_extra_small_q2_0;
                    for (uint32_t tok0 = small_first; ok && tok0 < (uint32_t)n_tok; tok0 += 8u) {
                        uint32_t group = (uint32_t)n_tok - tok0;
                        if (group > 8u) group = 8u;
                        struct {
                            uint32_t out_dim;
                            uint32_t n_tok;
                            uint32_t tok0;
                            uint32_t blocks;
                            uint32_t row_bytes;
                            float scale;
                        } spush = {
                            (uint32_t)out_dim, group, tok0,
                            (uint32_t)blocks, (uint32_t)row_bytes, scale,
                        };
                        ok = p36_vk_run_unlocked(op, small, bindings, &spush, sizeof(spush),
                                                 ((uint32_t)out_dim + 1u) / 2u, 1, 1);
                    }
                    pthread_mutex_unlock(&p36_vk_mu);
                    return ok;
                }
                bool is_iq2s_fast = weight_type == P36_VK_TENSOR_IQ2_S &&
                                    p36_vk_env_default_on("P36_VK_IQ2S_EXTRA");
                p36_vk_kernel *mmq_kernel = is_ptq1_0 ?
                    &p36_vk.dense_extra_mmq_ptq1_0 :
                    (is_pq2_0 ?
                        &p36_vk.dense_extra_mmq_pq2_0 :
                        (is_q2_0 ? &p36_vk.dense_extra_mmq_q2_0 :
                         is_iq2s_fast ? &p36_vk.dense_extra_mmq_iq2s :
                                   &p36_vk.dense_extra_mmq));
                ok = p36_vk_run_unlocked(
                    op, mmq_kernel,
                    bindings, &push, sizeof(push),
                    is_ptq1_0 ? ((uint32_t)out_dim + 31u) / 32u :
                    (is_q2_0 ? ((uint32_t)out_dim + 63u) / 64u :
                                ((uint32_t)out_dim + 31u) / 32u),
                    is_ptq1_0 ? ((uint32_t)n_tok + 63u) / 64u :
                                ((uint32_t)n_tok + 127u) / 128u,
                    1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (weight_type == P36_VK_TENSOR_IQ4_XS && !p36_gpu_quality &&
        p36_vk_use_dense_iq3_fast() && (n_tok != 1u || out_dim % 4u == 0u)) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[3] = { weights, q8, out };
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, scale,
                };
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq4_xs_decode");
                ok = p36_vk_run_unlocked(
                    op, &p36_vk.dense_iq4_xs_decode,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                ok = p36_vk_run_unlocked(
                    "dense_iq4_xs_mmq", &p36_vk.dense_iq4_xs_mmq,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 31u) / 32u,
                    ((uint32_t)n_tok + 127u) / 128u, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if (weight_type == P36_VK_TENSOR_IQ4_XS ||
        weight_type == P36_VK_TENSOR_IQ1_M) {
        struct {
            uint32_t out_dim;
            uint32_t n_tok;
            uint32_t blocks;
            uint32_t row_bytes;
            float scale;
        } push = {
            (uint32_t)out_dim, (uint32_t)n_tok,
            (uint32_t)blocks, (uint32_t)row_bytes, scale,
        };
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = weight_type == P36_VK_TENSOR_IQ1_M ?
            p36_vk_iq_tables_unlocked() : NULL;
        p36_gpu_tensor *weights = p36_vk_weight_get_unlocked(source, weight_bytes);
        ok = weights != NULL &&
             (weight_type != P36_VK_TENSOR_IQ1_M || tables != NULL);
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ4_XS ?
                &p36_vk.dense_iq4_xs : &p36_vk.dense_iq1_m;
            ok = p36_vk_run_unlocked(
                weight_type == P36_VK_TENSOR_IQ4_XS ? "dense_iq4_xs" : "dense_iq1_m",
                kernel, bindings, &push, sizeof(push),
                (uint32_t)out_dim, (uint32_t)n_tok, 1);
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    if ((weight_type == P36_VK_TENSOR_IQ3_XXS ||
         weight_type == P36_VK_TENSOR_IQ3_S) && !p36_gpu_quality &&
        p36_vk_use_dense_iq3_fast()) {
        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *weights = tables ?
            p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        ok = weights != NULL;
        if (ok) {
            const p36_gpu_tensor *bindings[4] = { weights, q8, out, tables };
            if (n_tok == 1u) {
                struct {
                    uint32_t out_dim;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)blocks,
                    (uint32_t)row_bytes, scale,
                };
                bool rows1 = weight_type == P36_VK_TENSOR_IQ3_S &&
                             out_dim == 48u;
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u r4/mv\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                bool rows4 = weight_type == P36_VK_TENSOR_IQ3_S &&
                             ((out_dim == 5120 && in_dim >= 6144) ||
                              out_dim == 6144 ||
                              out_dim == 10240 || out_dim == 17408);
                bool xxs_rows4 = weight_type == P36_VK_TENSOR_IQ3_XXS &&
                                 out_dim % 4u == 0u;
                bool full_rows5 = weight_type == P36_VK_TENSOR_IQ3_S &&
                                  !rows1 && !rows4 && out_dim % 5u == 0u;
                p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ3_XXS ?
                    (xxs_rows4 ? &p36_vk.dense_iq3_xxs_decode_r4 :
                                 &p36_vk.dense_iq3_xxs_decode) :
                    (rows1 ? &p36_vk.dense_iq3_s_decode_r1 :
                     rows4 ? &p36_vk.dense_iq3_s_decode_r4 :
                     full_rows5 ? &p36_vk.dense_iq3_s_decode_full :
                                  &p36_vk.dense_iq3_s_decode);
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    weight_type == P36_VK_TENSOR_IQ3_XXS ?
                        "dense_iq3_xxs_decode" : "dense_iq3_s_decode");
                ok = p36_vk_run_unlocked(
                    op,
                    kernel, bindings, &push, sizeof(push),
                    rows1 ? (uint32_t)out_dim :
                    rows4 || xxs_rows4 ?
                        ((uint32_t)out_dim + 3u) / 4u :
                        ((uint32_t)out_dim + 4u) / 5u,
                    1, 1);
            } else if (weight_type == P36_VK_TENSOR_IQ3_XXS && out_dim % 4u == 0u &&
                       n_tok >= 2u && n_tok <= 8u &&
                       p36_vk_env_default_on("P36_VK_DENSE_IQ3_NX")) {
                /* Small-batch activation rows share one weight unpack.
                 * dense_iq3_xxs_decode_r4 is ALU-bound (P4: 5.4 VALU/weight), and
                 * the 128-row MMQ tile costs a full tile for 2..8 rows
                 * (dense_iq3_xxs_mmq + predequant_b16).  Only 2 rows is
                 * reachable today (P7: the verify is 2 rows, never 3); 3..8 is
                 * P6's recorded reconsider_if condition, bit-exact at 3/4/8 in
                 * tests/test_dense_iq3xxs_nx.c, and 8 is the audit's scope. */
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq3_xxs_decode_nx");
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u nx\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                ok = p36_vk_run_unlocked(
                    op, &p36_vk.dense_iq3_xxs_decode_nx,
                    bindings, &push, sizeof(push),
                    ((uint32_t)out_dim + 3u) / 4u, 1, 1);
            } else {
                struct {
                    uint32_t out_dim;
                    uint32_t n_tok;
                    uint32_t blocks;
                    uint32_t row_bytes;
                    float scale;
                } push = {
                    (uint32_t)out_dim, (uint32_t)n_tok,
                    (uint32_t)blocks, (uint32_t)row_bytes, scale,
                };
                bool small_rows = weight_type == P36_VK_TENSOR_IQ3_S &&
                                  out_dim == 48u && n_tok <= 128u;
                bool wide_rows = weight_type == P36_VK_TENSOR_IQ3_S &&
                                 !small_rows && n_tok > 128u;
                p36_vk_kernel *kernel = weight_type == P36_VK_TENSOR_IQ3_XXS ?
                    &p36_vk.dense_iq3_xxs_mmq :
                    small_rows ? &p36_vk.dense_iq3_s_mmq_r4 :
                    wide_rows ? &p36_vk.dense_iq3_s_bm64_mmq :
                                 &p36_vk.dense_iq3_s_mmq;
                const char *op = p36_vk_prof_iq3_shape(
                    weight_type, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, "dense_iq3_mmq");
                if (getenv("P36_VK_DBG_IQ3N"))
                    fprintf(stderr, "IQ3F n_tok=%u in=%u out=%u mmq\n",
                            (uint32_t)n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
                p36_gpu_tensor *b16 = p36_vk_predequant_b16_unlocked(
                    q8, (uint32_t)n_tok, (uint32_t)blocks);
                const p36_gpu_tensor *mmq_bindings[5] = {
                    weights, q8, out, tables, b16,
                };
                ok = b16 != NULL &&
                     p36_vk_run_unlocked(
                    op, kernel,
                    mmq_bindings, &push, sizeof(push),
                    small_rows ? ((uint32_t)out_dim + 3u) / 4u :
                    wide_rows ? ((uint32_t)out_dim + 63u) / 64u :
                                 ((uint32_t)out_dim + 31u) / 32u,
                    ((uint32_t)n_tok + 127u) / 128u, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        return ok;
    }

    tile_count = ((uint32_t)n_tok + P36_VK_MOE_PAIR_TILE - 1u) /
                 P36_VK_MOE_PAIR_TILE;
    tiles = p36_gpu_tensor_alloc((uint64_t)tile_count *
                                 (1u + P36_VK_MOE_PAIR_TILE) * sizeof(uint32_t));
    tp = tiles ? p36_vk_tensor_contents_labeled(tiles, "submit_wait_dense_iq_tiles") : NULL;
    if (!tiles || !tp) {
        p36_gpu_tensor_free(tiles);
        return 0;
    }
    memset(tp, 0xff, (size_t)p36_gpu_tensor_bytes(tiles));
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t base = tile * (1u + P36_VK_MOE_PAIR_TILE);
        tp[base] = 0;
        for (uint32_t u = 0; u < P36_VK_MOE_PAIR_TILE; u++) {
            uint32_t tok = tile * P36_VK_MOE_PAIR_TILE + u;
            if (tok < n_tok) tp[base + 1u + u] = tok;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t type;
        uint32_t n_used;
        uint32_t has_scales;
        uint32_t x_per_expert;
        uint32_t packed_weights;
    } push = {
        (uint32_t)out_dim, (uint32_t)blocks, (uint32_t)row_bytes,
        (uint32_t)weight_bytes, weight_type, 1u, 0u, 0u, 0u,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights = tables ?
        p36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
    ok = weights != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            weights, q8, tiles, weights, out, tables,
        };
        p36_vk_kernel *kernel = (!p36_gpu_quality && p36_vk_use_moe_fast()) ?
            &p36_vk.moe_matvec_fast : &p36_vk.moe_matvec;
        ok = p36_vk_run_unlocked("dense_iq_matmul", kernel, bindings,
                                 &push, sizeof(push),
                                 ((uint32_t)out_dim + 7u) / 8u,
                                 tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(tiles);
    if (ok && scale != 1.0f) {
        float *p = p36_vk_tensor_contents_labeled(out, "submit_wait_dense_iq_scale");
        if (!p) return 0;
        for (uint64_t i = 0; i < n_tok * out_dim; i++) p[i] *= scale;
    }
    return ok;
}

/* Dense gate+up prefill in one dispatch (campaign W6).  Both weight matrices
 * must be the same type with the same in_dim/out_dim, which is what the dense
 * FFN pair is; the shader decodes both A tiles against one staged activation
 * tile.  Falls back to the two single-projection dispatches on any refusal. */
int p36_gpu_matmul_iq3_xxs_pair_mmq_tensor(p36_gpu_tensor *out_a,
                                           p36_gpu_tensor *out_b,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_a_offset,
                                           uint64_t weight_b_offset,
                                           uint64_t in_dim,
                                           uint64_t out_dim,
                                           const p36_gpu_tensor *q8,
                                           uint64_t n_tok,
                                           float scale) {
    uint64_t block_bytes = p36_vk_iq_block_bytes(P36_VK_TENSOR_IQ3_XXS);
    uint64_t blocks, row_bytes, weight_bytes, out_bytes, q8_bytes;
    if (p36_gpu_quality || !p36_vk_use_dense_iq3_fast() ||
        in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || out_dim == 0 ||
        n_tok < 2 || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > UINT32_MAX) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(row_bytes, out_dim, &weight_bytes) ||
        weight_bytes > UINT32_MAX ||
        !p36_u64_mul_ok(n_tok, out_dim, &out_bytes) ||
        !p36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes) ||
        !p36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !p36_u64_mul_ok(q8_bytes, sizeof(p36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!out_a || !out_b || !p36_gpu_tensor_range_ok(out_a, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(out_b, 0, out_bytes) ||
        !p36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;
    const unsigned char *wa = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_a_offset, weight_bytes);
    const unsigned char *wb = p36_gpu_weight_bytes(model_map, model_size,
                                                   weight_b_offset, weight_bytes);
    if (!wa || !wb) return 0;

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = {
        (uint32_t)out_dim, (uint32_t)n_tok,
        (uint32_t)blocks, (uint32_t)row_bytes, scale,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    p36_gpu_tensor *weights_a = tables ?
        p36_vk_weight_get_unlocked(wa, weight_bytes) : NULL;
    p36_gpu_tensor *weights_b = weights_a ?
        p36_vk_weight_get_unlocked(wb, weight_bytes) : NULL;
    p36_gpu_tensor *b16 = weights_b ?
        p36_vk_predequant_b16_unlocked(q8, (uint32_t)n_tok, (uint32_t)blocks) : NULL;
    int ok = b16 != NULL;
    if (ok) {
        const p36_gpu_tensor *bindings[6] = {
            weights_a, weights_b, out_a, out_b, tables, b16,
        };
        const char *op = p36_vk_prof_iq3_shape(
            P36_VK_TENSOR_IQ3_XXS, (uint32_t)n_tok, (uint32_t)in_dim,
            (uint32_t)out_dim, "dense_iq3_xxs_pair_mmq");
        ok = p36_vk_run_unlocked(op, &p36_vk.dense_iq3_xxs_mmq_pair,
                                 bindings, &push, sizeof(push),
                                 ((uint32_t)out_dim + 31u) / 32u,
                                 ((uint32_t)n_tok + 127u) / 128u, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_matmul_iq_quant_scaled_tensor(p36_gpu_tensor *out,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t weight_type,
                                           uint64_t in_dim,
                                           uint64_t out_dim,
                                           const p36_gpu_tensor *x,
                                           uint64_t n_tok,
                                           float scale) {
    p36_gpu_tensor *q8 = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!q8) return 0;
    int ok = p36_gpu_matmul_iq_quant_q8_scaled_tensor(
        out, model_map, model_size, weight_offset, weight_type,
        in_dim, out_dim, q8, n_tok, scale);
    p36_gpu_tensor_free(q8);
    return ok;
}

static int p36_vk_moe_gate_up_iq2_swiglu(p36_gpu_tensor *out,
                                          const void *model_map,
                                          uint64_t model_size,
                                          const p36_gpu_moe_weight *gate,
                                          const p36_gpu_moe_weight *up,
                                          const p36_gpu_tensor *selected,
                                          const p36_gpu_tensor *qx,
                                          const uint32_t *stream_slot_map,
                                          bool resident_bank,
                                          uint32_t n_used,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                         uint32_t mid_dim,
                                         uint32_t n_expert) {
    uint64_t block_bytes = 66;
    bool iq2s = false;
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    const unsigned char *gate_source;
    const unsigned char *up_source;
    const unsigned char *gate_scales_source = NULL;
    const unsigned char *up_scales_source = NULL;
    const uint32_t *selected_host;
    p36_gpu_tensor *tiles = NULL;
    uint32_t tile_count = 0;
    int ok = 0;

    if (p36_gpu_quality || !p36_vk_use_moe_gate_up() || (n_tok > 1u && n_tok < 32u) ||
        (!p36_vk_moe_bank_cache && !stream_slot_map)) return 0;
    if (!gate || !up || !selected || !qx || !out || n_used == 0 || n_tok == 0 || n_expert == 0) return 0;
    /* IQ2_S experts (Qwen3.8-35B-A3B-IQ2_M) decode with the same Q8_K
     * activation staging and the same 0.125 scale folding; only the 82-byte
     * superblock decode differs, so the shader is the same source built with
     * -DP36_MOE_IQ2S. */
    if (gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_S) {
        iq2s = true;
        block_bytes = 82;
    } else if (gate->type != P36_VK_TENSOR_IQ2_XXS || up->type != P36_VK_TENSOR_IQ2_XXS) {
        return 0;
    }
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || mid_dim == 0) return 0;
    blocks = in_dim / P36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * mid_dim;
    if (expert_stride > UINT32_MAX || row_bytes > UINT32_MAX || blocks > UINT32_MAX ||
        !p36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    gate_source = p36_gpu_weight_bytes(model_map, model_size, gate->offset, weight_bytes);
    up_source = p36_gpu_weight_bytes(model_map, model_size, up->offset, weight_bytes);
    if (!gate_source || !up_source) return 0;
    if (gate->has_scales) {
        gate_scales_source = p36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes);
        if (!gate_scales_source) return 0;
    }
    if (up->has_scales) {
        up_scales_source = p36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes);
        if (!up_scales_source) return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, n_slot * mid_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * blocks * sizeof(p36_vk_block_q8_K))) {
        return 0;
    }

    if (n_slot <= 4096u && !stream_slot_map &&
        (!p36_vk_stream.enabled || resident_bank)) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, P36_VK_MOE_PAIR_TILE);
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, P36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected, "submit_wait_moe_gate_up_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = p36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_slot_map,
                                       &tile_count);
        if (!tiles || tile_count == 0) {
            p36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t n_used;
        uint32_t has_gate_scales;
        uint32_t has_up_scales;
    } push = {
        mid_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        n_used, gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
    };

    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
    const p36_gpu_tensor *gate_bank = NULL;
    const p36_gpu_tensor *up_bank = NULL;
    const p36_gpu_tensor *gate_scales = NULL;
    const p36_gpu_tensor *up_scales = NULL;
    if (stream_slot_map) {
        gate_bank = p36_vk_stream_weight_tensor_unlocked(P36_VK_STREAM_PART_GATE);
        up_bank = p36_vk_stream_weight_tensor_unlocked(P36_VK_STREAM_PART_UP);
        gate_scales = p36_vk_stream_scale_tensor_unlocked(P36_VK_STREAM_PART_GATE);
        up_scales = p36_vk_stream_scale_tensor_unlocked(P36_VK_STREAM_PART_UP);
    } else {
        gate_bank = tables ? p36_vk_weight_get_unlocked(gate_source, weight_bytes) : NULL;
        up_bank = gate_bank ? p36_vk_weight_get_unlocked(up_source, weight_bytes) : NULL;
        if (up_bank && gate->has_scales) gate_scales = p36_vk_weight_get_unlocked(gate_scales_source, scale_bytes);
        if (up_bank && (!gate->has_scales || gate_scales) && up->has_scales) up_scales = p36_vk_weight_get_unlocked(up_scales_source, scale_bytes);
    }
    if (tables && up_bank && (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales)) {
        const p36_gpu_tensor *bindings[8] = {
            gate_bank,
            up_bank,
            qx,
            tiles,
            gate->has_scales ? gate_scales : gate_bank,
            up->has_scales ? up_scales : up_bank,
            out,
            tables,
        };
        ok = p36_vk_run_unlocked(iq2s ? "moe_iq2s_gate_up" : "moe_iq2_gate_up",
                                 iq2s ? &p36_vk.moe_gate_up_iq2s : &p36_vk.moe_gate_up,
                                 bindings, &push, sizeof(push),
                                 (mid_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&p36_vk_mu);
    p36_gpu_tensor_free(tiles);
    return ok;
}

/* Routed expert FFN, the routed half of p36_forward_ffn() including all
 * .scale applications, batched over n_tok rows of x with per-token expert
 * selections.  The fast IQ2_XXS path fuses gate/up/SwiGLU into one dispatch;
 * other layouts and --quality keep the separate exact gate/up/SwiGLU route. */
int p36_gpu_moe_ffn_q8_tensor(p36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const p36_gpu_moe_weight *gate,
                              const p36_gpu_moe_weight *up,
                              const p36_gpu_moe_weight *down,
                              const p36_gpu_tensor *selected,
                              const p36_gpu_tensor *weights,
                              uint32_t layer,
                              uint32_t n_used,
                              const p36_gpu_tensor *qx,
                              uint32_t n_tok,
                              uint32_t in_dim,
                              uint32_t mid_dim,
                              uint32_t out_dim,
                              uint32_t n_expert) {
    uint64_t mid_bytes = (uint64_t)n_used * mid_dim * sizeof(float);
    uint64_t down_bytes = (uint64_t)n_used * out_dim * sizeof(float);
    p36_gpu_tensor *gate8 = NULL;
    p36_gpu_tensor *up8 = NULL;
    p36_gpu_tensor *mid8 = NULL;
    p36_gpu_tensor *qmid = NULL;
    p36_gpu_tensor *down8 = NULL;
    uint32_t stream_slot_map[256];
    const uint32_t *stream_slots = NULL;
    bool resident_bank = p36_vk_stream.enabled && layer < p36_vk_stream.full_layers;
    int ok = 0;

    if (!gate || !up || !down || !qx || n_used == 0 || n_expert == 0 || n_tok == 0) return 0;
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 ||
        mid_dim == 0 || (mid_dim % P36_VK_QK_K) != 0 || out_dim == 0) {
        return 0;
    }
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * (in_dim / P36_VK_QK_K) * sizeof(p36_vk_block_q8_K)) ||
        !p36_gpu_tensor_range_ok(selected, 0, (uint64_t)n_tok * n_used * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(weights, 0, (uint64_t)n_tok * n_used * sizeof(float))) {
        return 0;
    }

    if (p36_vk_stream.enabled && !resident_bank && n_expert <= 256) {
        uint64_t gate_block = p36_vk_iq_block_bytes(gate->type);
        uint64_t up_block = p36_vk_iq_block_bytes(up->type);
        uint64_t down_block = p36_vk_iq_block_bytes(down->type);
        uint64_t gate_row = gate_block * (in_dim / P36_VK_QK_K);
        uint64_t up_row = up_block * (in_dim / P36_VK_QK_K);
        uint64_t down_row = down_block * (mid_dim / P36_VK_QK_K);
        const uint32_t *selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected,
                                                                       "submit_wait_moe_stream_selected");
        p36_gpu_stream_expert_table table;
        memset(&table, 0, sizeof(table));
        table.model_map = model_map;
        table.model_size = model_size;
        table.layer = layer;
        table.n_total_expert = n_expert;
        table.gate_offset = gate->offset;
        table.up_offset = up->offset;
        table.down_offset = down->offset;
        table.gate_scales_offset = gate->scales_offset;
        table.up_scales_offset = up->scales_offset;
        table.down_scales_offset = down->scales_offset;
        table.gate_type = gate->type;
        table.up_type = up->type;
        table.down_type = down->type;
        table.has_gate_scales = gate->has_scales;
        table.has_up_scales = up->has_scales;
        table.has_down_scales = down->has_scales;
        if (gate_block && up_block && down_block && selected_host &&
            !p36_u64_mul_ok(gate_row, mid_dim, &table.gate_expert_bytes)) {
            selected_host = NULL;
        }
        if (selected_host && !p36_u64_mul_ok(up_row, mid_dim, &table.up_expert_bytes)) selected_host = NULL;
        if (selected_host && !p36_u64_mul_ok(down_row, out_dim, &table.down_expert_bytes)) selected_host = NULL;
        if (selected_host && p36_vk_stream_cache_prepare_u32(&table,
                                                             selected_host,
                                                             (uint64_t)n_tok * n_used,
                                                             stream_slot_map)) {
            stream_slots = stream_slot_map;
        }
    }

    mid8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
    down8 = p36_gpu_tensor_alloc_scratch(down_bytes * n_tok);
    ok = qx && mid8 && down8 &&
          p36_vk_moe_gate_up_iq2_swiglu(mid8, model_map, model_size, gate, up,
                                         selected, qx, stream_slots, resident_bank,
                                         n_used, n_tok, in_dim, mid_dim, n_expert);
    if (!ok && mid8 && down8) {
        gate8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
        up8 = p36_gpu_tensor_alloc_scratch(mid_bytes * n_tok);
        ok = gate8 && up8 &&
             p36_vk_moe_matvec(gate8, model_map, model_size, gate, qx, false, false,
                                stream_slots ? P36_VK_STREAM_PART_GATE : P36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             p36_vk_moe_matvec(up8, model_map, model_size, up, qx, false, true,
                                stream_slots ? P36_VK_STREAM_PART_UP : P36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             p36_gpu_swiglu_tensor(mid8, gate8, up8, n_tok * n_used * mid_dim, 0.0f, 1.0f);
    }
    if (ok) {
        qmid = p36_vk_quantize_q8_k_alloc_tensor(mid8, mid_dim, (uint64_t)n_tok * n_used);
        ok = qmid &&
               p36_vk_moe_matvec(down8, model_map, model_size, down, qmid, true,
                                  down->type == P36_VK_TENSOR_Q2_K ||
                                  down->type == P36_VK_TENSOR_Q3_K,
                                  stream_slots ? P36_VK_STREAM_PART_DOWN : P36_VK_STREAM_PART_NONE,
                                  stream_slots, resident_bank,
                                  selected, n_used, n_tok, mid_dim, out_dim, n_expert);
    }
    if (ok) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } push = { out_dim, n_used };
        const p36_gpu_tensor *bindings[3] = { down8, weights, out };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("moe_reduce", &p36_vk.moe_reduce, bindings, &push, sizeof(push),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
    }

    p36_gpu_tensor_free(down8);
    p36_gpu_tensor_free(qmid);
    p36_gpu_tensor_free(mid8);
    p36_gpu_tensor_free(up8);
    p36_gpu_tensor_free(gate8);
    return ok;
}

/* Whole routed FFN against f32 activations for decode and prefill chunks:
 * GPU tile schedule, fused gate/up/SwiGLU, specialized routed down, weighted
 * reduce.  Besides the native IQ2_XXS/Q2_K recipe this covers APEX's
 * IQ3_XXS/Q3_K, IQ2_S/IQ3_XXS and mixed IQ2_S+IQ2_XXS/IQ3_XXS recipes.
 * No q8_K stage anywhere, no host read of the selection. */
int p36_gpu_moe_ffn_f32_tensor(p36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      const p36_gpu_moe_weight *gate,
                                      const p36_gpu_moe_weight *up,
                                      const p36_gpu_moe_weight *down,
                                      const p36_gpu_tensor *selected,
                                      const p36_gpu_tensor *weights,
                                      uint32_t layer,
                                      uint32_t n_used,
                                      const p36_gpu_tensor *x,
                                      uint32_t n_tok,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      uint32_t n_expert) {
    bool q4k = gate->type == P36_VK_TENSOR_Q4_K && up->type == P36_VK_TENSOR_Q4_K &&
               down->type == P36_VK_TENSOR_Q4_K;
    /* IQ2_S experts (Qwen3.8-35B-A3B-IQ2_M): the fused f32 expert path carries
     * the decode pair and the prefill GEMM pair.  Layers 0-2 of that file mix
     * an IQ3_S down under IQ2_S gate/up, so the down side picks the IQ3_S
     * variant of the same kernels (110-byte superblocks instead of 82). */
    bool gu_iq2s = gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_S;
    bool gu_iq3xxs = gate->type == P36_VK_TENSOR_IQ3_XXS && up->type == P36_VK_TENSOR_IQ3_XXS;
    bool gu_iq2s_iq2xxs = gate->type == P36_VK_TENSOR_IQ2_S && up->type == P36_VK_TENSOR_IQ2_XXS;
    bool gu_q3k = gate->type == P36_VK_TENSOR_Q3_K && up->type == P36_VK_TENSOR_Q3_K;
    bool down_iq2s = down->type == P36_VK_TENSOR_IQ2_S;
    bool down_iq3s = down->type == P36_VK_TENSOR_IQ3_S;
    bool down_iq3xxs = down->type == P36_VK_TENSOR_IQ3_XXS;
    bool down_q3k = down->type == P36_VK_TENSOR_Q3_K;
    bool iq2s = gu_iq2s && (down_iq2s || down_iq3s);
    bool apex_iq3xxs_q3k = gu_iq3xxs && down_q3k &&
                           p36_vk_env_default_on("P36_VK_MOE_APEX_IQ3XXS_Q3K");
    bool apex_iq2s_iq3xxs = gu_iq2s && down_iq3xxs &&
                            p36_vk_env_default_on("P36_VK_MOE_APEX_IQ2S_IQ3XXS");
    bool apex_iq2s_iq2xxs_iq3xxs = gu_iq2s_iq2xxs && down_iq3xxs &&
                                   p36_vk_env_default_on("P36_VK_MOE_APEX_MIXED");
    /* Prefill batches only: the f32-activation GEMM is what makes IQ3_XXS/Q3_K
     * prefill ~4x faster, but it moves logits against the q8 route (~0.4-1.0
     * at decode), so decode and small batches keep the q8 route. */
    const bool apex_prefill = n_tok >= p36_vk_moe_gemm_min();
    bool apex_iq3xxs = apex_prefill && gu_iq3xxs && down_iq3xxs &&
                       p36_vk_env_default_on("P36_VK_MOE_APEX_IQ3XXS");
    bool apex_q3k = apex_prefill && gu_q3k && down_q3k &&
                    p36_vk_env_default_on("P36_VK_MOE_APEX_Q3K");
    bool apex = apex_iq3xxs_q3k || apex_iq2s_iq3xxs || apex_iq2s_iq2xxs_iq3xxs || apex_q3k || apex_iq3xxs;
    uint32_t gate_block_bytes = q4k ? 144u : gate->type == P36_VK_TENSOR_IQ2_S ? 82u :
                                gate->type == P36_VK_TENSOR_IQ3_XXS ? 98u :
                                gate->type == P36_VK_TENSOR_Q3_K ? 110u : 66u;
    uint32_t up_block_bytes = q4k ? 144u : up->type == P36_VK_TENSOR_IQ2_S ? 82u :
                              up->type == P36_VK_TENSOR_IQ3_XXS ? 98u :
                              up->type == P36_VK_TENSOR_Q3_K ? 110u : 66u;
    uint64_t gate_stride = (uint64_t)(in_dim / P36_VK_QK_K) * gate_block_bytes * mid_dim;
    uint64_t up_stride = (uint64_t)(in_dim / P36_VK_QK_K) * up_block_bytes * mid_dim;
    uint64_t down_stride = (uint64_t)(mid_dim / P36_VK_QK_K) *
                           (q4k ? 144u : down_iq2s ? 82u : down_iq3s || down_q3k ? 110u :
                            down_iq3xxs ? 98u : 84u) * out_dim;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    p36_gpu_tensor *tiles = NULL;
    p36_gpu_tensor *sel_slots = NULL;
    p36_gpu_tensor *mid8 = NULL;
    p36_gpu_tensor *down8 = NULL;
    const uint32_t *selected_host = NULL;
    p36_gpu_stream_expert_table stream_table = {0};
    uint32_t selected_unique[256];
    uint32_t unique = 0;
    uint32_t stream_cap = 0;
    bool overflow = false;
    bool streamed = p36_vk_stream.enabled && layer >= p36_vk_stream.full_layers;
    int ok = 0;

    if (p36_gpu_quality || !p36_vk.have_storage16 || !p36_vk_have_subgroups(false) ||
        n_tok == 0 || n_used == 0 || n_used > 8u || n_slot > 8192u) return 0;
    if (!p36_vk_env_default_on("P36_VK_MOE_F32B")) return 0;
    if (!streamed && !p36_vk_moe_bank_cache) return 0;
    if (!q4k && !iq2s && !apex &&
        (gate->type != P36_VK_TENSOR_IQ2_XXS || up->type != P36_VK_TENSOR_IQ2_XXS ||
         down->type != P36_VK_TENSOR_Q2_K)) return 0;
    /* Q4_K rides the same identity-schedule decode kernels; batches and
     * streaming keep the q8_K fallback. */
    if (q4k && (streamed || n_tok != 1u || !p36_vk_use_moe_down_sum_decode() ||
                !p36_vk_env_default_on("P36_VK_MOE_Q4K_F32B"))) return 0;
    if (in_dim == 0 || (in_dim % P36_VK_QK_K) != 0 || (in_dim % 4u) != 0 ||
        mid_dim == 0 || (mid_dim % P36_VK_QK_K) != 0 || out_dim == 0) return 0;
    if (gate_stride * n_expert > UINT32_MAX || up_stride * n_expert > UINT32_MAX ||
        down_stride * n_expert > UINT32_MAX) return 0;

    const unsigned char *gate_src = p36_gpu_weight_bytes(model_map, model_size, gate->offset, gate_stride * n_expert);
    const unsigned char *up_src = p36_gpu_weight_bytes(model_map, model_size, up->offset, up_stride * n_expert);
    const unsigned char *down_src = p36_gpu_weight_bytes(model_map, model_size, down->offset, down_stride * n_expert);
    if (!gate_src || !up_src || !down_src) return 0;
    const unsigned char *gate_sc = gate->has_scales ? p36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes) : NULL;
    const unsigned char *up_sc = up->has_scales ? p36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes) : NULL;
    const unsigned char *down_sc = down->has_scales ? p36_gpu_weight_bytes(model_map, model_size, down->scales_offset, scale_bytes) : NULL;
    if ((gate->has_scales && !gate_sc) || (up->has_scales && !up_sc) || (down->has_scales && !down_sc)) return 0;
    if (!p36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(x, 0, (uint64_t)n_tok * in_dim * sizeof(float)) ||
        !p36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !p36_gpu_tensor_range_ok(weights, 0, n_slot * sizeof(float))) {
        return 0;
    }

    /* SSD streaming: the selected experts stream into the cache and the
     * tile schedule carries cache slot ids, so the same fused kernels read
     * the stream buffers at slot * stride — identical bytes, identical
     * math, no q8_K staging.  The selection read is a host sync, but the
     * cache prepare needs it on the host anyway. */
    uint32_t stream_slot_map[256];
    if (streamed) {
        if (n_expert > 256u) return 0;
        selected_host = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)selected,
                                                       "submit_wait_moe_stream_selected");
        if (!selected_host) return 0;
        stream_table.model_map = model_map;
        stream_table.model_size = model_size;
        stream_table.layer = layer;
        stream_table.n_total_expert = n_expert;
        stream_table.gate_offset = gate->offset;
        stream_table.up_offset = up->offset;
        stream_table.down_offset = down->offset;
        stream_table.gate_scales_offset = gate->scales_offset;
        stream_table.up_scales_offset = up->scales_offset;
        stream_table.down_scales_offset = down->scales_offset;
        stream_table.gate_type = gate->type;
        stream_table.up_type = up->type;
        stream_table.down_type = down->type;
        stream_table.has_gate_scales = gate->has_scales;
        stream_table.has_up_scales = up->has_scales;
        stream_table.has_down_scales = down->has_scales;
        stream_table.gate_expert_bytes = gate_stride;
        stream_table.up_expert_bytes = up_stride;
        stream_table.down_expert_bytes = down_stride;
        stream_cap = p36_vk_stream_cache_cap(&stream_table);
        if (stream_cap == 0) return 0;
        bool seen[256] = {0};
        for (uint64_t i = 0; i < n_slot; i++) {
            uint32_t expert = selected_host[i];
            if (expert >= n_expert) return 0;
            if (!seen[expert]) {
                seen[expert] = true;
                selected_unique[unique++] = expert;
            }
        }
        overflow = unique > stream_cap;
        if (overflow) p36_vk_stream_note_routes(&stream_table, selected_host, n_slot);
        if (!overflow && !p36_vk_stream_cache_prepare_u32(
                &stream_table, selected_host, n_slot, stream_slot_map)) return 0;
        /* The kernels compute weight offsets in 32-bit halfword units, so
         * the whole slot arena must stay indexable. */
        if (gate_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u ||
            up_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u ||
            down_stride * p36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u) return 0;
        if (!overflow && n_tok == 1u) {
            /* Decode identity schedule with slot ids instead of expert ids. */
            uint32_t mapped[8];
            for (uint32_t i = 0; i < (uint32_t)n_slot; i++) mapped[i] = stream_slot_map[selected_host[i]];
            sel_slots = p36_gpu_tensor_alloc(n_slot * sizeof(uint32_t));
            if (!sel_slots || !p36_gpu_tensor_write(sel_slots, 0, mapped, n_slot * sizeof(uint32_t))) goto done;
        }
    }

    /* Decode: one distinct expert per slot, so the selection itself is the
     * schedule and the tile build dispatch is skipped. */
    bool identity = n_tok == 1u && !overflow;
    bool down_sum_decode = identity && p36_vk_use_moe_down_sum_decode();
    /* Prefill batches route to the register-tiled GEMM kernels: wider
     * expert token tiles amortize the dequant and weight reads that bound
     * the matvec path. Small batches (chunk tails, MTP verify windows)
     * keep the matvec kernels, whose per-slot arithmetic order matches
     * decode bit for bit; the GEMM's k order is fixed but differs from
     * decode, which the relaxed warm==cold session gate absorbs. */
    bool gemm = !identity && !p36_vk_micro_batch &&
                n_tok >= p36_vk_moe_gemm_min() &&
                p36_vk_env_default_on("P36_VK_MOE_GEMM");
    /* IQ2_S experts: the prefill GEMM pair, the identity decode pair, and the
     * small-batch (2..127 token) f32 pair.  The f32 pair needs an IQ2_S down
     * as well, so the IQ3_S-down layers (IQ2_M il 0-2) still fall back to the
     * q8 route outside the GEMM range. */
    if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;
    /* APEX has dedicated wide-prefill and identity-decode kernels.  Tiny
     * routed batches retain the generic path until their f32 pair kernels
     * are selected below as well. */
    if (apex && !(gemm || (identity && down_sum_decode))) return 0;
    uint32_t pair_tile = gemm ? P36_VK_MOE_GEMM_TILE : P36_VK_MOE_PAIR_TILE;
    uint32_t tile_count = (uint32_t)n_slot;
    if (!identity && !overflow) {
        tiles = p36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert,
                                     streamed ? stream_slot_map : NULL, pair_tile);
        if (!tiles) goto done;
        tile_count = p36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, pair_tile);
    }
    mid8 = p36_gpu_tensor_alloc_scratch(n_slot * mid_dim * sizeof(float));
    if (!down_sum_decode) down8 = p36_gpu_tensor_alloc_scratch(n_slot * out_dim * sizeof(float));
    if (!mid8 || (!down_sum_decode && !down8)) goto done;

    uint32_t wave = 0;
    ok = 1;
    /* Overflow runs the original full-batch expert tiles in cache-sized
     * waves and reduces once. Pair arithmetic is unchanged, while a small
     * resident cache never selects the Q8 fallback. */
    do {
        p36_gpu_tensor *wave_tiles = tiles;
        uint32_t wave_tile_count = tile_count;
        uint32_t wave_experts = unique - wave;
        if (overflow) {
            if (wave_experts > stream_cap) wave_experts = stream_cap;
            if (!p36_vk_stream_cache_prepare_u32(&stream_table,
                                                  selected_unique + wave,
                                                  wave_experts,
                                                  stream_slot_map)) {
                ok = 0;
                break;
            }
            wave_tiles = p36_vk_moe_build_tiles_ex(selected_host, n_slot,
                                                    n_expert, stream_slot_map,
                                                    pair_tile, true,
                                                    &wave_tile_count);
            if (!wave_tiles || wave_tile_count == 0) {
                p36_gpu_tensor_free(wave_tiles);
                ok = 0;
                break;
            }
        }

        pthread_mutex_lock(&p36_vk_mu);
        p36_gpu_tensor *tables = p36_vk_iq_tables_unlocked();
        p36_gpu_tensor *gate_bank, *up_bank, *down_bank;
        p36_gpu_tensor *gate_scales, *up_scales, *down_scales;
        if (streamed) {
            gate_bank = p36_vk_stream.gate;
            up_bank = p36_vk_stream.up;
            down_bank = p36_vk_stream.down;
            gate_scales = gate->has_scales ? p36_vk_stream.gate_scales : NULL;
            up_scales = up->has_scales ? p36_vk_stream.up_scales : NULL;
            down_scales = down->has_scales ? p36_vk_stream.down_scales : NULL;
            ok = tables && gate_bank && up_bank && down_bank;
        } else {
            gate_bank = tables ? p36_vk_weight_get_unlocked(gate_src, gate_stride * n_expert) : NULL;
            up_bank = gate_bank ? p36_vk_weight_get_unlocked(up_src, up_stride * n_expert) : NULL;
            down_bank = up_bank ? p36_vk_weight_get_unlocked(down_src, down_stride * n_expert) : NULL;
            gate_scales = gate_sc && down_bank ? p36_vk_weight_get_unlocked(gate_sc, scale_bytes) : NULL;
            up_scales = up_sc && down_bank ? p36_vk_weight_get_unlocked(up_sc, scale_bytes) : NULL;
            down_scales = down_sc && down_bank ? p36_vk_weight_get_unlocked(down_sc, scale_bytes) : NULL;
            ok = tables && down_bank;
        }
        ok = ok &&
             (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales) && (!down->has_scales || down_scales);
        if (ok) {
            struct {
                uint32_t mid_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_gate_scales;
                uint32_t has_up_scales;
                uint32_t identity;
            } gpush = { mid_dim, in_dim / P36_VK_QK_K, n_used,
                        gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const p36_gpu_tensor *gb[8] = {
                gate_bank, up_bank, x, identity ? (sel_slots ? sel_slots : selected) : wave_tiles,
                gate_scales ? gate_scales : gate_bank,
                up_scales ? up_scales : up_bank,
                mid8, tables,
            };
            if (q4k) {
                const p36_gpu_tensor *qgb[7] = {
                    gate_bank, up_bank, x, sel_slots ? sel_slots : selected,
                    gate_scales ? gate_scales : gate_bank,
                    up_scales ? up_scales : up_bank,
                    mid8,
                };
                ok = p36_vk_run_unlocked("moe_q4k_gate_up", &p36_vk.moe_gate_up_q4k_f32b, qgb, &gpush, sizeof(gpush),
                                         mid_dim, wave_tile_count, 1);
            } else {
                ok = gemm ? p36_vk_run_unlocked(gu_q3k ? "moe_q3k_gate_up_gemm" :
                                                gu_iq3xxs ? "moe_iq3xxs_gate_up_gemm" :
                                                (gu_iq2s_iq2xxs ? "moe_iq2s_iq2xxs_gate_up_gemm" :
                                                 (gu_iq2s ? "moe_iq2s_gate_up_gemm" : "moe_iq2_gate_up_gemm")),
                                                gu_q3k ? &p36_vk.moe_gate_up_gemm_q3k :
                                                gu_iq3xxs ? &p36_vk.moe_gate_up_gemm_iq3xxs :
                                                (gu_iq2s_iq2xxs ? &p36_vk.moe_gate_up_gemm_iq2s_iq2xxs :
                                                 (gu_iq2s ? &p36_vk.moe_gate_up_gemm_iq2s : &p36_vk.moe_gate_up_gemm)),
                                                gb, &gpush, sizeof(gpush),
                                                (mid_dim + 63u) / 64u, wave_tile_count, 1)
                          : p36_vk_run_unlocked(identity ? (gu_q3k ? "moe_q3k_gate_up_decode" :
                                                              gu_iq3xxs ? "moe_iq3xxs_gate_up_decode" :
                                                               (gu_iq2s_iq2xxs ? "moe_iq2s_iq2xxs_gate_up_decode" :
                                                                (gu_iq2s ? "moe_iq2s_gate_up_decode" : "moe_iq2_gate_up_decode")))
                                                         : (gu_iq2s ? "moe_iq2s_gate_up_f32b" : "moe_iq2_gate_up"),
                                                identity ? (gu_q3k ? &p36_vk.moe_gate_up_decode_q3k :
                                                            gu_iq3xxs ? &p36_vk.moe_gate_up_decode_iq3xxs :
                                                            (gu_iq2s_iq2xxs ? &p36_vk.moe_gate_up_decode_iq2s_iq2xxs :
                                                             (gu_iq2s ? &p36_vk.moe_gate_up_decode_iq2s
                                                                      : &p36_vk.moe_gate_up_decode)))
                                                         : (gu_iq2s ? &p36_vk.moe_gate_up_f32b_iq2s
                                                                     : &p36_vk.moe_gate_up_f32b),
                                                gb, &gpush, sizeof(gpush), mid_dim, wave_tile_count, 1);
            }
        }
        if (ok) {
            struct {
                uint32_t out_dim;
                uint32_t in_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_scales;
                uint32_t identity;
            } dpush = { out_dim, mid_dim, mid_dim / P36_VK_QK_K, n_used, down->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const p36_gpu_tensor *schedule = identity ? (sel_slots ? sel_slots : selected) : wave_tiles;
            if (down_sum_decode) {
                const p36_gpu_tensor *db[7] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    weights, out, tables,
                };
                const char *dname = q4k ? "moe_q4k_down_sum_decode" :
                                    down_q3k ? "moe_q3k_down_sum_decode" :
                                    down_iq3xxs ? "moe_iq3xxs_down_sum_decode" :
                                    down_iq3s ? "moe_iq3s_down_sum_decode" :
                                    down_iq2s ? "moe_iq2s_down_sum_decode" : "moe_q2k_down_sum_decode";
                p36_vk_kernel *dkernel = q4k ? &p36_vk.moe_down_q4k_sum_decode :
                                          down_q3k ? &p36_vk.moe_down_q2k_sum_decode_q3k :
                                          down_iq3xxs ? &p36_vk.moe_down_q2k_sum_decode_iq3xxs :
                                          down_iq3s ? &p36_vk.moe_down_q2k_sum_decode_iq3s :
                                          down_iq2s ? &p36_vk.moe_down_q2k_sum_decode_iq2s :
                                                       &p36_vk.moe_down_q2k_sum_decode;
                ok = p36_vk_run_unlocked(dname, dkernel,
                                         db, &dpush, sizeof(dpush),
                                         out_dim, 1, 1);
            } else {
                const p36_gpu_tensor *db[6] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    down8, tables,
                };
                const char *dname = down_q3k ? "moe_q3k_down_gemm" :
                                    down_iq3xxs ? "moe_iq3xxs_down_gemm" :
                                    down_iq3s ? "moe_iq3s_down_gemm" :
                                    down_iq2s ? "moe_iq2s_down_gemm" : "moe_q2k_down_gemm";
                p36_vk_kernel *dkernel = down_q3k ? &p36_vk.moe_down_gemm_q3k :
                                          down_iq3xxs ? &p36_vk.moe_down_gemm_iq3xxs :
                                          down_iq3s ? &p36_vk.moe_down_gemm_iq3s :
                                          down_iq2s ? &p36_vk.moe_down_gemm_iq2s : &p36_vk.moe_down_gemm;
                ok = gemm ? p36_vk_run_unlocked(dname, dkernel, db, &dpush, sizeof(dpush),
                                                (out_dim + 63u) / 64u, wave_tile_count, 1)
                          : p36_vk_run_unlocked(down_iq2s ? "moe_iq2s_down_f32b" : "moe_q2k_down",
                                                down_iq2s ? &p36_vk.moe_down_q2k_f32b_iq2s
                                                          : &p36_vk.moe_down_q2k_f32b,
                                                db, &dpush, sizeof(dpush),
                                                out_dim, wave_tile_count, 1);
            }
        }
        pthread_mutex_unlock(&p36_vk_mu);
        if (overflow) p36_gpu_tensor_free(wave_tiles);
        wave += wave_experts;
    } while (ok && overflow && wave < unique);

    if (ok && !down_sum_decode) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } rpush = { out_dim, n_used };
        const p36_gpu_tensor *rb[3] = { down8, weights, out };
        pthread_mutex_lock(&p36_vk_mu);
        ok = p36_vk_run_unlocked("moe_reduce", &p36_vk.moe_reduce, rb, &rpush, sizeof(rpush),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&p36_vk_mu);
    }

done:
    p36_gpu_tensor_free(down8);
    p36_gpu_tensor_free(mid8);
    p36_gpu_tensor_free(tiles);
    p36_gpu_tensor_free(sel_slots);
    return ok;
}

int p36_gpu_moe_ffn_tensor(p36_gpu_tensor *out,
                           const void *model_map,
                           uint64_t model_size,
                           const p36_gpu_moe_weight *gate,
                           const p36_gpu_moe_weight *up,
                           const p36_gpu_moe_weight *down,
                           const p36_gpu_tensor *selected,
                           const p36_gpu_tensor *weights,
                           uint32_t layer,
                           uint32_t n_used,
                           const p36_gpu_tensor *x,
                           uint32_t n_tok,
                           uint32_t in_dim,
                           uint32_t mid_dim,
                           uint32_t out_dim,
                           uint32_t n_expert) {
    p36_gpu_tensor *qx = p36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!qx) return 0;
    int ok = p36_gpu_moe_ffn_q8_tensor(out, model_map, model_size,
                                       gate, up, down,
                                       selected, weights, layer, n_used,
                                       qx, n_tok, in_dim, mid_dim, out_dim, n_expert);
    p36_gpu_tensor_free(qx);
    return ok;
}
