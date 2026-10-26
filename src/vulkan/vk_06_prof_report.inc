static void p36_vk_prof_report(void) {
    fprintf(stderr,
            "p36: Vulkan prof dispatches=%" PRIu64 " record_ms=%.1f flushes=%" PRIu64 " submit_wait_ms=%.1f\n",
            p36_vk_prof_dispatches,
            (double)p36_vk_prof_record_ns / 1e6,
            p36_vk_prof_flushes,
            (double)p36_vk_prof_submit_ns / 1e6);
    fprintf(stderr,
            "p36: Vulkan record split desc_ms=%.1f cmd_ms=%.1f book_ms=%.1f"
            " (per dispatch us: %.2f / %.2f / %.2f)\n",
            (double)p36_vk_prof_desc_ns / 1e6,
            (double)p36_vk_prof_cmd_ns / 1e6,
            (double)p36_vk_prof_book_ns / 1e6,
            p36_vk_prof_dispatches ? (double)p36_vk_prof_desc_ns / 1e3 / (double)p36_vk_prof_dispatches : 0.0,
            p36_vk_prof_dispatches ? (double)p36_vk_prof_cmd_ns / 1e3 / (double)p36_vk_prof_dispatches : 0.0,
            p36_vk_prof_dispatches ? (double)p36_vk_prof_book_ns / 1e3 / (double)p36_vk_prof_dispatches : 0.0);
    if (p36_vk_prof_op_n != 0) {
        fprintf(stderr, "p36: Vulkan op profile\n");
        for (uint32_t i = 0; i < p36_vk_prof_op_n; i++) {
            const p36_vk_prof_op_row *r = &p36_vk_prof_op_rows[i];
            fprintf(stderr,
                    "p36:   op %-28s dispatches=%" PRIu64 " flushes=%" PRIu64 " barriers=%" PRIu64
                    " groups=%" PRIu64 " record_ms=%.3f submit_wait_ms=%.3f gpu_ms=%.3f wall_ms=%.3f\n",
                    r->name,
                    r->dispatches,
                    r->flushes,
                    r->barriers,
                    r->groups,
                    (double)r->record_ns / 1e6,
                    (double)r->submit_ns / 1e6,
                    (double)r->gpu_ns / 1e6,
                    (double)r->wall_ns / 1e6);
        }
    }
    if (p36_vk_prof_kernel_n != 0) {
        uint64_t total_ns = 0;
        for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) total_ns += p36_vk_prof_kernel_rows[i].gpu_ns;
        fprintf(stderr,
                "p36: Vulkan kernel GPU time total_ms=%.3f barriers=%" PRIu64
                " fence_ready=%" PRIu64 " fence_blocked=%" PRIu64
                " spin_ms=%.3f block_ms=%.3f\n",
                (double)total_ns / 1e6, p36_vk_hazard_barrier_count,
                p36_vk_wait_ready, p36_vk_wait_blocked,
                (double)p36_vk_wait_spin_ns / 1e6, (double)p36_vk_wait_block_ns / 1e6);
        if (getenv("P36_VK_HAZARD_STATS")) {
            fprintf(stderr,
                    "p36: hazard classes raw=%" PRIu64 " waw=%" PRIu64
                    " war=%" PRIu64 " pure_war=%" PRIu64 "\n",
                    p36_vk_hazard_raw_count, p36_vk_hazard_waw_count,
                    p36_vk_hazard_war_count, p36_vk_hazard_pure_war_count);
        }
        for (uint32_t i = 0; i < p36_vk_prof_kernel_n; i++) {
            const p36_vk_prof_kernel_row *r = &p36_vk_prof_kernel_rows[i];
            const double pct = total_ns ? 100.0 * (double)r->gpu_ns / (double)total_ns : 0.0;
            fprintf(stderr,
                    "p36:   %-32s dispatches=%" PRIu64 " groups=%" PRIu64 " gpu_ms=%.3f pct=%.1f\n",
                    r->path,
                    r->dispatches,
                    r->groups,
                    (double)r->gpu_ns / 1e6,
                    pct);
        }
    }
    if (p36_vk_prof_timeline_fp) {
        fclose(p36_vk_prof_timeline_fp);
        p36_vk_prof_timeline_fp = NULL;
    }
}

void p36_gpu_cleanup(void) {
    pthread_mutex_lock(&p36_vk_mu);
    /* Teardown after device loss must not wait on a queue that will never
     * drain; the objects are still destroyable. */
    if (p36_vk.device && !p36_vk_device_lost) vkDeviceWaitIdle(p36_vk.device);
    p36_vk_stream_profile_report();
    p36_vk_batch_recording = false;
    p36_vk_batch_count = 0;
    p36_vk_prof_query_count = 0;
    p36_vk_batch_written_n = 0;
    p36_vk_batch_dirty_all = false;
    p36_vk_submit_seq = 0;
    p36_vk_completed_seq = 0;
    p36_vk_hazard_reset();
    p36_vk_b16_ring_destroy_unlocked();
    p36_vk_tensor_free_unlocked(p36_vk_vision_weight);
    p36_vk_vision_weight = NULL;
    while (p36_vk_retired_n) p36_vk_tensor_free_unlocked(p36_vk_retired[--p36_vk_retired_n]);
    while (p36_vk_pool_n) p36_vk_tensor_free_unlocked(p36_vk_pool[--p36_vk_pool_n]);
    while (p36_vk_private_pool_n)
        p36_vk_tensor_free_unlocked(p36_vk_private_pool[--p36_vk_private_pool_n]);
    p36_vk_weight_cache_free_unlocked();
    p36_vk_tensor_free_unlocked(p36_vk.iq_tables);
    p36_vk.iq_tables = NULL;
    p36_vk_tensor_free_unlocked(p36_vk.attn_part);
    p36_vk.attn_part = NULL;
    p36_vk.attn_part_bytes = 0;
    p36_vk_tensor_free_unlocked(p36_vk.top2_part);
    p36_vk.top2_part = NULL;
    p36_vk.top2_part_bytes = 0;
    p36_vk_tensor_free_unlocked(p36_vk.argmax_rows_part);
    p36_vk.argmax_rows_part = NULL;
    p36_vk.argmax_rows_part_bytes = 0;
    p36_vk_tensor_free_unlocked(p36_vk.topk8_part);
    p36_vk.topk8_part = NULL;
    p36_vk.topk8_part_bytes = 0;
    p36_vk_kernel_destroy(&p36_vk.ffn_tail);
    p36_vk_kernel_destroy(&p36_vk.top2);
    p36_vk_kernel_destroy(&p36_vk.topk8);
    p36_vk_kernel_destroy(&p36_vk.logits_minp);
    p36_vk_kernel_destroy(&p36_gdn_front);
    p36_vk_kernel_destroy(&p36_vk.recur_norm_gate);
    p36_vk_kernel_destroy(&p36_vk.recur_norm_gate_q8_k);
    p36_vk_kernel_destroy(&p36_vk.moe_reduce);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_bm64_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq1_m);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_mmq_iq2s);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_small_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_small_pq2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_q2_0);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2s_pair);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk.dense_extra_decode_iq2xs);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs_rows);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_iq4_xs);
    p36_vk_kernel_destroy(&p36_vk.dense_kquant_mmq);
    p36_vk_kernel_destroy(&p36_vk.predequant_b16);
    p36_vk_kernel_destroy(&p36_vk.dense_kquant_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q4k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q5k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q3k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q2k_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_q3k_rows);
    p36_vk_kernel_destroy(&p36_vk.dense_q4k_rows);
    p36_vk_kernel_destroy(&p36_vk.dense_q5k_rows);
    p36_vk_kernel_destroy(&p36_vk.dense_q6k_rows);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_r1);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode_full);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_decode);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_s_mmq_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_mmq);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_mmq_pair);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode_r4);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode_nx);
    p36_vk_kernel_destroy(&p36_vk.dense_iq3_xxs_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_matvec_fast);
    p36_vk_kernel_destroy(&p36_vk.moe_matvec);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_iq2s);
    p36_vk_kernel_destroy(&p36_vk.router_topk);
    p36_vk_kernel_destroy(&p36_vk.moe_tiles);
    p36_vk_kernel_destroy(&p36_vk.kv_store);
    p36_vk_kernel_destroy(&p36_vk.kv_store_quant);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope_kv_quant);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_f32b_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_decode_iq2s_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_f32b_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq3s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q2k_sum_decode_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_q4k_f32b);
    p36_vk_kernel_destroy(&p36_vk.moe_down_q4k_sum_decode);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_q3k);
    p36_vk_kernel_destroy(&p36_vk.moe_gate_up_gemm_iq2s_iq2xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq2s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq3s);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_iq3xxs);
    p36_vk_kernel_destroy(&p36_vk.moe_down_gemm_q3k);
    p36_vk_kernel_destroy(&p36_vk.attn_decode_fused);
    p36_vk_kernel_destroy(&p36_vk.attn_decode_split);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile2);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_qtile2_gqa6);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_fa_gqa6);
    p36_vk_kernel_destroy(&p36_vk.attn_prefill_fa_gqa8);
    p36_vk_kernel_destroy(&p36_vk.attn_combine);
    p36_vk_kernel_destroy(&p36_vk.attn_reduce);
    p36_vk_kernel_destroy(&p36_vk.attn_post);
    p36_vk_kernel_destroy(&p36_vk.attn_scores);
    p36_vk_kernel_destroy(&p36_vk.delta_net_cols);
    p36_vk_kernel_destroy(&p36_vk.delta_net_cols_f16);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode_reg);
    p36_vk_kernel_destroy(&p36_vk.delta_net_decode_reg_f16);
    p36_vk_kernel_destroy(&p36_vk.delta_net_fast);
    p36_vk_kernel_destroy(&p36_vk.delta_net);
    p36_vk_kernel_destroy(&p36_vk.delta_gates);
    p36_vk_kernel_destroy(&p36_vk.delta_qk);
    p36_vk_kernel_destroy(&p36_vk.delta_qkv);
    p36_vk_kernel_destroy(&p36_vk.recur_conv_silu_decode);
    p36_vk_kernel_destroy(&p36_vk.conv_silu);
    p36_vk_kernel_destroy(&p36_vk.recur_window);
    p36_vk_kernel_destroy(&p36_vk.copy_rows);
    p36_vk_kernel_destroy(&p36_vk.embed_row);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope_kv);
    p36_vk_kernel_destroy(&p36_vk.rms_norm_rope);
    p36_vk_kernel_destroy(&p36_vk.rope_mrope);
    p36_vk_kernel_destroy(&p36_vk.rope);
    p36_vk_kernel_destroy(&p36_vk.swiglu);
    p36_vk_kernel_destroy(&p36_vk.swiglu_q8_k);
    p36_vk_kernel_destroy(&p36_vk.rms_norm);
    p36_vk_kernel_destroy(&p36_vk.add_rms_norm);
    p36_vk_kernel_destroy(&p36_vk.add_rms_norm_q8_k);
    p36_vk_kernel_destroy(&p36_vk.quantize_q8_0);
    p36_vk_kernel_destroy(&p36_vk.matmul_q6k_mmq_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_q5k_mmq_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_q6k_mmq);
    p36_vk_kernel_destroy(&p36_vk.matmul_q5k_mmq);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode_b64);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode_p36);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_decode);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b_pair);
    p36_vk_kernel_destroy(&p36_vk.shared_down_tail_decode);
    p36_vk_kernel_destroy(&p36_vk.shared_gate_up_decode);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm_f16);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_mm_f16_out32);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_f32b_nx);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0_p36);
    p36_vk_kernel_destroy(&p36_vk.matmul_q8_0);
    p36_vk_kernel_destroy(&p36_vk.matmul_kquant);
    p36_vk_kernel_destroy(&p36_vk.quantize_q8_k);
    p36_vk_kernel_destroy(&p36_vk.directional_steering);
    p36_vk_kernel_destroy(&p36_vk.add);
    p36_vk_kernel_destroy(&p36_vk.fwht);
    p36_vk_kernel_destroy(&p36_vk.signs_mul);
    p36_vk_kernel_destroy(&p36_vk.v_grouped_permute);
    p36_vk_kernel_destroy(&p36_vk.hadamard_prepare);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32_fast);
    p36_vk_kernel_destroy(&p36_vk.matmul_f32_fast_w256);
    p36_vk_kernel_destroy(&p36_vk.matmul_f16);
    p36_vk_kernel_destroy(&p36_vk.matmul_bf16);
    p36_vk_kernel_destroy(&p36_vk.vision_matmul_f16);
    p36_vk_kernel_destroy(&p36_vk.vision_attention);
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pool) vkDestroyDescriptorPool(p36_vk.device, p36_vk_ring[i].pool, NULL);
        if (p36_vk_ring[i].fence) vkDestroyFence(p36_vk.device, p36_vk_ring[i].fence, NULL);
    }
    memset(p36_vk_ring, 0, sizeof(p36_vk_ring));
    p36_vk_ring_cur = 0;
    if (p36_vk.query_pool) vkDestroyQueryPool(p36_vk.device, p36_vk.query_pool, NULL);
    if (p36_vk.command_pool) vkDestroyCommandPool(p36_vk.device, p36_vk.command_pool, NULL);
    if (p36_vk.device) vkDestroyDevice(p36_vk.device, NULL);
    if (p36_vk.instance) vkDestroyInstance(p36_vk.instance, NULL);
    memset(&p36_vk, 0, sizeof(p36_vk));
    p36_gpu_model_map = NULL;
    p36_gpu_model_size = 0;
    p36_gpu_model_fd = -1;
    p36_gpu_quality = false;
    p36_gpu_commands_open = false;
    p36_vk_private_bytes = 0;
    p36_vk_private_peak = 0;
    p36_vk_moe_bank_cache = true;
    pthread_mutex_unlock(&p36_vk_mu);
    p36_vk_host_threads_shutdown();
}

static p36_gpu_tensor *p36_gpu_tensor_alloc_persistent(uint64_t bytes, bool clear) {
    if (!p36_gpu_init()) return NULL;
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tensor = NULL;
    const uint64_t alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;
    for (uint32_t i = 0; i < p36_vk_pool_n; i++) {
        if (p36_vk_pool[i]->alloc_bytes != alloc_bytes) continue;
        tensor = p36_vk_pool[i];
        p36_vk_pool[i] = p36_vk_pool[--p36_vk_pool_n];
        tensor->bytes = bytes;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        if (clear) memset(tensor->data, 0, (size_t)alloc_bytes);
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        break;
    }
    if (!tensor) tensor = p36_vk_tensor_alloc_kind_unlocked(bytes, false, clear);
    pthread_mutex_unlock(&p36_vk_mu);
    return tensor;
}

p36_gpu_tensor *p36_gpu_tensor_alloc(uint64_t bytes) {
    return p36_gpu_tensor_alloc_persistent(bytes, true);
}

p36_gpu_tensor *p36_gpu_tensor_alloc_uninitialized(uint64_t bytes) {
    return p36_gpu_tensor_alloc_persistent(bytes, false);
}

p36_gpu_tensor *p36_gpu_tensor_alloc_scratch(uint64_t bytes) {
    if (!p36_gpu_init()) return NULL;
    pthread_mutex_lock(&p36_vk_mu);
    p36_gpu_tensor *tensor = p36_vk_tensor_alloc_scratch_unlocked(bytes);
    pthread_mutex_unlock(&p36_vk_mu);
    return tensor;
}

p36_gpu_tensor *p36_gpu_tensor_view(const p36_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!p36_gpu_tensor_range_ok(base, offset, bytes)) return NULL;
    p36_gpu_tensor *view = calloc(1, sizeof(*view));
    if (!view) return NULL;
    view->bytes = bytes;
    view->offset = offset;
    view->owner = (p36_gpu_tensor *)base;
    return view;
}

void p36_gpu_tensor_free(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    pthread_mutex_lock(&p36_vk_mu);
    if (tensor->owner) {
        p36_vk_tensor_free_unlocked(tensor);
    } else {
        p36_vk_tensor_release_unlocked(tensor);
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

uint64_t p36_gpu_tensor_bytes(const p36_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

/* The lazy-submission sync point: any host access to tensor bytes lands
 * here first, so flushing the open batch makes every host-side op see
 * completed GPU results without explicit synchronization at call sites. */
void *p36_gpu_tensor_contents(p36_gpu_tensor *tensor) {
    uint64_t offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
    if (!root || !root->data || offset > root->bytes) return NULL;
    /* After device loss the bytes behind this pointer are whatever the
     * dead batch left there; refuse rather than hand back stale results. */
    if (p36_vk_device_lost) return NULL;
    bool flush = false;
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk_gpu_busy_unlocked()) {
        if (root->gpu_written) {
            /* dirty_all only means the written set overflowed; a root no
             * dispatch ever wrote cannot have pending GPU writes. */
            flush = p36_vk_batch_dirty_all;
            for (uint32_t i = 0; !flush && i < p36_vk_batch_written_n; i++) {
                flush = p36_vk_batch_written[i] == root;
            }
        } else {
            /* Host-written staging the GPU only reads (embed stage, conv
             * window): the caller may overwrite it, so wait out a pending
             * read; with double-buffered stages this is almost never hit. */
            flush = root->last_use_seq > p36_vk_completed_seq;
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
    if (flush) {
        pthread_mutex_lock(&p36_vk_mu);
        int ok = p36_vk_flush_reason_unlocked(p36_vk_tensor_contents_reason);
        pthread_mutex_unlock(&p36_vk_mu);
        if (!ok) return NULL;
    }
    return root->data + offset;
}

static void *p36_vk_tensor_contents_labeled(p36_gpu_tensor *tensor, const char *reason) {
    const char *prev = p36_vk_tensor_contents_reason;
    p36_vk_tensor_contents_reason = reason ? reason : "submit_wait_tensor_contents";
    void *p = p36_gpu_tensor_contents(tensor);
    p36_vk_tensor_contents_reason = prev;
    return p;
}

void *p36_gpu_tensor_contents_named(p36_gpu_tensor *tensor, const char *reason) {
    return p36_vk_tensor_contents_labeled(tensor, reason);
}

int p36_gpu_tensor_write(p36_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    unsigned char *dst = p36_vk_tensor_contents_labeled(tensor, "submit_wait_tensor_write");
    if (!dst || !data || !p36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(dst + offset, data, (size_t)bytes);
    return 1;
}

int p36_gpu_tensor_read(const p36_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    unsigned char *src = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)tensor, "submit_wait_tensor_read");
    if (!src || !data || !p36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(data, src + offset, (size_t)bytes);
    p36_vk_read_bytes_total += bytes;
    return 1;
}

uint64_t p36_gpu_read_bytes(void) { return p36_vk_read_bytes_total; }
uint64_t p36_gpu_read_flushes(void) { return p36_vk_flush_total; }

int p36_gpu_tensor_copy(p36_gpu_tensor *dst, uint64_t dst_offset,
                        const p36_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes) {
    if (!bytes) return 1;
    if (!p36_gpu_tensor_range_ok(dst, dst_offset, bytes) ||
        !p36_gpu_tensor_range_ok(src, src_offset, bytes)) return 0;
    if (p36_vk.ready && dst && src) {
        uint64_t root_dst_offset = 0;
        uint64_t root_src_offset = 0;
        p36_gpu_tensor *dst_root = p36_gpu_tensor_root(dst, &root_dst_offset);
        p36_gpu_tensor *src_root = p36_gpu_tensor_root(src, &root_src_offset);
        root_dst_offset += dst_offset;
        root_src_offset += src_offset;
        bool overlap = dst_root == src_root &&
                       root_dst_offset < root_src_offset + bytes &&
                       root_src_offset < root_dst_offset + bytes;
        if (dst_root && src_root && dst_root->buffer && src_root->buffer && !overlap) {
            pthread_mutex_lock(&p36_vk_mu);
            VkCommandBuffer cmd;
            int ok = p36_vk_begin_batch_unlocked(&cmd);
            if (ok) {
                bool need_barrier = p36_vk_hazard_all ||
                    p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, src_root) ||
                    p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, dst_root) ||
                    p36_vk_hazard_in(p36_vk_hazard_read, p36_vk_hazard_read_n, dst_root);
                if (need_barrier) {
                    VkMemoryBarrier barrier = {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    };
                    vkCmdPipelineBarrier(cmd,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &barrier, 0, NULL, 0, NULL);
                    p36_vk_hazard_reset();
                }
            }
            if (ok) {
                VkBufferCopy copy = {
                    .srcOffset = root_src_offset,
                    .dstOffset = root_dst_offset,
                    .size = bytes,
                };
                vkCmdCopyBuffer(cmd, src_root->buffer, dst_root->buffer, 1, &copy);
                src_root->last_use_seq = p36_vk_submit_seq + 1;
                dst_root->last_use_seq = p36_vk_submit_seq + 1;
                dst_root->gpu_written = true;
                p36_vk_hazard_add(p36_vk_hazard_read, &p36_vk_hazard_read_n, src_root);
                p36_vk_hazard_add(p36_vk_hazard_written, &p36_vk_hazard_written_n, dst_root);
                p36_vk_note_written_tensor(dst_root);
                p36_vk_prof_dispatches++;
                p36_vk_prof_op_add("tensor_copy", 1, 0, 1, 0, 0, 0, 0);
                if (++p36_vk_batch_count >= P36_VK_BATCH_FLUSH_AT) ok = p36_vk_submit_eager_unlocked();
            }
            pthread_mutex_unlock(&p36_vk_mu);
            return ok;
        }
    }
    unsigned char *dstp = p36_vk_tensor_contents_labeled(dst, "submit_wait_tensor_copy_dst");
    unsigned char *srcp = p36_vk_tensor_contents_labeled((p36_gpu_tensor *)src, "submit_wait_tensor_copy_src");
    if (!dstp || !srcp) return 0;
    memmove(dstp + dst_offset, srcp + src_offset, (size_t)bytes);
    return 1;
}

int p36_gpu_begin_commands(void) {
    p36_gpu_commands_open = true;
    return 1;
}

int p36_gpu_flush_commands(void) {
    return p36_gpu_commands_open ? 1 : 0;
}

int p36_gpu_end_commands(void) {
    if (!p36_gpu_commands_open) return 0;
    p36_gpu_commands_open = false;
    return 1;
}

int p36_gpu_synchronize(void) {
    if (!p36_vk.ready) return 1;
    pthread_mutex_lock(&p36_vk_mu);
    int ok = p36_vk_flush_reason_unlocked("submit_wait_explicit");
    pthread_mutex_unlock(&p36_vk_mu);
    if (!ok || p36_vk_device_lost) return 0;
    return p36_vk_ok(vkDeviceWaitIdle(p36_vk.device), "vkDeviceWaitIdle");
}

void p36_gpu_trim_scratch(void) {
    if (!p36_vk.ready || !p36_gpu_synchronize()) return;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_b16_ring_destroy_unlocked();
    p36_vk_retired_sweep_unlocked();
    p36_vk_pools_drain_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
}

int p36_gpu_device_lost(void) {
    pthread_mutex_lock(&p36_vk_mu);
    int lost = p36_vk_device_lost_unlocked();
    pthread_mutex_unlock(&p36_vk_mu);
    return lost;
}

int p36_gpu_device_is_gfx1013(void) {
    /* The BC-250 (1002:13fe) is the only board this backend detects, and
     * its GPU is GFX10.1/gfx1013.  p36_gpu_init() is idempotent and the
     * engine already calls it at open; calling it again keeps this correct
     * if that order ever changes, and returns 0 if the GPU is unavailable. */
    if (!p36_gpu_init()) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    int is_gfx1013 = p36_vk.bc250 ? 1 : 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return is_gfx1013;
}

int p36_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = model_map;
    p36_gpu_model_size = model_size;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_set_model_fd(int fd) {
    p36_gpu_model_fd = fd;
#if defined(POSIX_FADV_RANDOM)
    if (fd >= 0 && p36_vk_stream.enabled &&
        getenv("P36_VK_STREAM_READAHEAD") == NULL) {
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
#endif
    return fd >= 0;
}

int p36_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_offset > model_size || map_size > model_size - map_offset) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = (const unsigned char *)model_map + map_offset;
    p36_gpu_model_size = map_size;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || (!offsets && count) || (!sizes && count)) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] > model_size - offsets[i]) return 0;
    }
    if (!p36_vk_env_default_on("P36_VK_PLAN_ARENA"))
        return p36_gpu_set_model_map(model_map, model_size);
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_cache_free_unlocked();
    p36_gpu_model_map = model_map;
    p36_gpu_model_size = model_size;
    (void)p36_vk_arena_plan_unlocked(sizes, count);
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    (void)label;
    const unsigned char *source = p36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    int ok;
    if (!source) return 0;
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_weight_copy_parallel = true;
    ok = p36_vk_weight_get_unlocked(source, bytes) != NULL;
    p36_vk_weight_copy_parallel = false;
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

int p36_gpu_finish_model_cache(void) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk.arena_plan = NULL;
    free(p36_vk.arena_plan_sizes);
    p36_vk.arena_plan_sizes = NULL;
    p36_vk.arena_plan_count = 0;
    p36_vk.arena_plan_next = 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;
}

int p36_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    (void)in_dim;
    (void)out_dim;
    return p36_gpu_cache_model_range(model_map, model_size, offset, bytes, label);
}

void p36_gpu_set_quality(bool quality) {
    p36_gpu_quality = quality;
}

static void p36_vk_prepare_dense_kernels(void) {
    /* RADV pipeline creation can take seconds.  Do it while the dense model
     * opens instead of charging the first prompt that reaches each shape. */
    p36_vk_kernel *kernels[] = {
        &p36_vk.dense_iq3_s_mmq,
        &p36_vk.dense_iq3_s_mmq_r4,
        &p36_vk.dense_iq3_xxs_mmq,
        &p36_vk.dense_iq3_xxs_mmq_pair,
        &p36_vk.dense_kquant_mmq,
        &p36_vk.predequant_b16,
        &p36_vk.dense_iq3_s_decode,
        &p36_vk.dense_iq3_s_decode_r1,
        &p36_vk.dense_iq3_s_decode_r4,
        &p36_vk.dense_iq3_xxs_decode_r4,
        &p36_vk.dense_q4k_decode,
        &p36_vk.dense_kquant_decode,
        &p36_vk.dense_extra_decode_iq2s,
        &p36_vk.dense_extra_decode_iq2xxs,
        &p36_vk.dense_extra_decode_iq2xs,
        &p36_vk.dense_extra_mmq_iq2s,
        &p36_vk.dense_extra_decode_iq2s_pair,
        &p36_vk.dense_extra_mmq,
    };
    if (!p36_vk_env_default_on("P36_VK_PREPARE_DENSE")) return;
    pthread_mutex_lock(&p36_vk_mu);
    for (uint32_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++) {
        if (!p36_vk_kernel_init(kernels[i])) break;
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

