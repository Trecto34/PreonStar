static const p36_gpu_tensor *p36_vk_stream_weight_tensor_unlocked(p36_vk_stream_part part) {
    if (p36_vk_stream.cap == 0) return NULL;
    if (part == P36_VK_STREAM_PART_GATE) return p36_vk_stream.gate;
    if (part == P36_VK_STREAM_PART_UP) return p36_vk_stream.up;
    if (part == P36_VK_STREAM_PART_DOWN) return p36_vk_stream.down;
    return NULL;
}

static const p36_gpu_tensor *p36_vk_stream_scale_tensor_unlocked(p36_vk_stream_part part) {
    if (p36_vk_stream.cap == 0) return NULL;
    if (part == P36_VK_STREAM_PART_GATE) return p36_vk_stream.gate_scales;
    if (part == P36_VK_STREAM_PART_UP) return p36_vk_stream.up_scales;
    if (part == P36_VK_STREAM_PART_DOWN) return p36_vk_stream.down_scales;
    return NULL;
}

static int p36_vk_descriptor_for_tensor(const p36_gpu_tensor *tensor, VkDescriptorBufferInfo *info) {
    uint64_t root_offset = 0;
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &root_offset);
    if (!root || !root->buffer) return 0;
    uint64_t range = tensor->bytes ? p36_round_up_u64(tensor->bytes, 4) : 1;
    if (root_offset > root->alloc_bytes) return 0;
    if (range > root->alloc_bytes - root_offset) range = root->alloc_bytes - root_offset;
    if (range == 0) range = 1;
    *info = (VkDescriptorBufferInfo){
        .buffer = root->buffer,
        .offset = root_offset,
        .range = range,
    };
    return 1;
}

static int p36_vk_alloc_descriptor_unlocked(p36_vk_kernel *k, VkDescriptorSet *set) {
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p36_vk.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->set_layout,
    };
    VkResult rc = vkAllocateDescriptorSets(p36_vk.device, &ai, set);
    if (rc == VK_SUCCESS) return 1;
    if (rc == VK_ERROR_DEVICE_LOST) return p36_vk_ok(rc, "vkAllocateDescriptorSets");
    /* The pool's live sets belong to the open batch; rotate to the next
     * ring slot (its pool was reset when its fence completed). */
    if (!p36_vk_submit_eager_unlocked()) return 0;
    ai.descriptorPool = p36_vk.descriptor_pool;
    rc = vkAllocateDescriptorSets(p36_vk.device, &ai, set);
    if (rc == VK_ERROR_DEVICE_LOST) return p36_vk_ok(rc, "vkAllocateDescriptorSets");
    return rc == VK_SUCCESS;
}

static int p36_vk_run_unlocked(
        const char                 *op,
        p36_vk_kernel              *kernel,
        const p36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z) {
    if (p36_vk_device_lost) return 0;
    const char *prof_op = p36_vk_prof_op_name(op);
    uint64_t prof_t0 = p36_vk_now_ns();
    uint64_t groups_total = (uint64_t)groups_x * groups_y * groups_z;
    if (!p36_vk_kernel_init(kernel)) return 0;
    if (push_bytes != kernel->push_bytes) return 0;
    if (p36_vk.prof_kernel && p36_vk.query_pool && p36_vk_prof_query_count + 2u > P36_VK_PROF_QUERY_CAP) {
        if (!p36_vk_flush_reason_unlocked("submit_wait_query_pool")) return 0;
    }

    VkDescriptorBufferInfo infos[16];
    VkWriteDescriptorSet writes[16];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        if (!p36_vk_descriptor_for_tensor(bindings[i], &infos[i])) return 0;
    }

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!p36_vk_alloc_descriptor_unlocked(kernel, &set)) return 0;
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i],
        };
    }
    vkUpdateDescriptorSets(p36_vk.device, kernel->bindings, writes, 0, NULL);
    uint64_t prof_ph = 0;
    if (p36_vk.prof_ops) {
        prof_ph = p36_vk_now_ns();
        p36_vk_prof_desc_ns += prof_ph - prof_t0;
    }

    /* Record into the open batch; submission happens at the next host read
     * (p36_vk_flush_unlocked).  One reusable command buffer and fence are
     * enough because the flush always waits before recording resumes. */
    VkCommandBuffer cmd;
    if (!p36_vk_begin_batch_unlocked(&cmd)) return 0;
    p36_gpu_tensor *roots[16];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i] = p36_gpu_tensor_root(bindings[i], NULL);
    }
    {
        /* Barrier only on a real hazard against work recorded since the
         * last barrier - including work in earlier eager-submitted buffers,
         * which executes in submission order and is still covered by a
         * barrier recorded here. Independent dispatches overlap. */
        bool has_raw = p36_vk_hazard_all;
        bool has_waw = p36_vk_hazard_all;
        bool has_war = p36_vk_hazard_all;
        for (uint32_t i = 0; i < kernel->bindings; i++) {
            if (roots[i]->weight_only) continue;
            bool written_before = p36_vk_hazard_in(p36_vk_hazard_written, p36_vk_hazard_written_n, roots[i]);
            bool read_before = p36_vk_hazard_in(p36_vk_hazard_read, p36_vk_hazard_read_n, roots[i]);
            bool writes_here = ((kernel->write_mask >> i) & 1u) != 0;
            if (written_before && writes_here) has_waw = true;
            else if (written_before) has_raw = true;
            else if (writes_here && read_before) has_war = true;
        }
        bool need_barrier = has_raw || has_waw || has_war;
        if (need_barrier) {
            if (has_raw) p36_vk_hazard_raw_count++;
            if (has_waw) p36_vk_hazard_waw_count++;
            if (has_war) p36_vk_hazard_war_count++;
            if (has_war && !has_raw && !has_waw) p36_vk_hazard_pure_war_count++;
        }
        if (need_barrier && !p36_vk_skip_barriers()) {
            p36_vk_hazard_barrier_count++;
            p36_vk_prof_op_barrier(prof_op);
            VkMemoryBarrier compute_barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &compute_barrier, 0, NULL, 0, NULL);
            p36_vk_hazard_reset();
        }
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout, 0, 1, &set, 0, NULL);
    if (push_bytes) vkCmdPushConstants(cmd, kernel->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    /* Bracketing is per op-group, not per dispatch: 756 decode dispatches per
     * token would each force a BOTTOM_OF_PIPE flush and change the thing being
     * measured.  A group opens on the first dispatch of a kernel path, stays
     * open across consecutive dispatches of that path and closes when the path
     * changes (or at vkEndCommandBuffer). */
    if (p36_vk.prof_kernel && p36_vk.query_pool) {
        if (p36_vk_prof_skip_left > 0) {
            p36_vk_prof_skip_left--;
            p36_vk_prof_group_close_locked();
        } else if (p36_vk_prof_grp_path != kernel->path) {
            p36_vk_prof_group_close_locked();
            if (p36_vk_prof_query_count + 2u <= P36_VK_PROF_QUERY_CAP) {
                uint32_t q0 = p36_vk_prof_query_count;
                p36_vk_prof_query_count += 2u;
                p36_vk_prof_query_path[q0 / 2u] = kernel->path;
                p36_vk_prof_query_op[q0 / 2u] = prof_op;
                p36_vk_prof_query_groups[q0 / 2u] = 0;
                p36_vk_prof_query_host_ns[q0 / 2u] = p36_vk_now_ns();
                /* Reset exactly the pair about to be written.  Resetting the
                 * whole pool once per command buffer hangs the RADV device
                 * (pending queries from an in-flight submission are clobbered). */
                vkCmdResetQueryPool(cmd, p36_vk.query_pool, q0, 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, p36_vk.query_pool, q0);
                p36_vk_prof_grp_base = q0;
                p36_vk_prof_grp_path = kernel->path;
            }
        }
        if (p36_vk_prof_grp_base != UINT32_MAX)
            p36_vk_prof_query_groups[p36_vk_prof_grp_base / 2u] += groups_total;
    }
    p36_vk_prof_wg_hist_add(prof_op, groups_total);
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    if (p36_vk.prof_ops) {
        uint64_t now = p36_vk_now_ns();
        p36_vk_prof_cmd_ns += now - prof_ph;
        prof_ph = now;
    }
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i]->last_use_seq = p36_vk_submit_seq + 1;
        if ((kernel->write_mask >> i) & 1u) {
            roots[i]->gpu_written = true;
            roots[i]->write_gen++;
            p36_vk_hazard_add(p36_vk_hazard_written, &p36_vk_hazard_written_n, roots[i]);
            p36_vk_note_written_tensor((p36_gpu_tensor *)bindings[i]);
        } else {
            p36_vk_hazard_add(p36_vk_hazard_read, &p36_vk_hazard_read_n, roots[i]);
        }
    }
    if (p36_vk.prof_ops) p36_vk_prof_book_ns += p36_vk_now_ns() - prof_ph;
    p36_vk_prof_dispatches++;
    {
        uint64_t record_ns = p36_vk_now_ns() - prof_t0;
        p36_vk_prof_record_ns += record_ns;
        p36_vk_prof_op_add(prof_op, 1, 0, groups_total, record_ns, 0, 0, 0);
    }
    if (++p36_vk_batch_count >= P36_VK_BATCH_FLUSH_AT) return p36_vk_submit_eager_unlocked();
    return 1;
}

int p36_gpu_init(void) {
    pthread_mutex_lock(&p36_vk_mu);
    if (p36_vk.ready) {
        pthread_mutex_unlock(&p36_vk_mu);
        return 1;
    }

    memset(&p36_vk, 0, sizeof(p36_vk));
    if (!getcwd(p36_vk.shader_root, sizeof(p36_vk.shader_root)))
        p36_vk.shader_root[0] = '\0';
    p36_vk.prof_ops = getenv("P36_VK_PROF_OP") || getenv("P36_VK_PROF") || getenv("P36_VK_PROF_KERNEL");
    {
        const char *env = getenv("P36_VK_FAULT_INJECT_LOST");
        p36_vk_fault_inject_lost = env ? strtoull(env, NULL, 10) : 0;
        p36_vk_submit_calls = 0;
    }
    /* write_mask marks the bindings each shader writes; the lazy-flush
     * tracker uses it so host reads of input-only tensors do not submit
     * the open batch. */
    p36_vk.matmul_f16 = P36_VK_KERNEL("vulkan/matmul_f16.spv", 3, 16, 1u << 2);
    p36_vk.matmul_bf16 = P36_VK_KERNEL("vulkan/matmul_bf16.spv", 3, 16, 1u << 2);
    p36_gdn_front = P36_VK_KERNEL("vulkan/gdn_front.spv", 12, 24, 1u | (15u << 8));
    p36_vk.vision_matmul_f16 = P36_VK_KERNEL("vulkan/vision_matmul_f16.spv", 3, 16, 1u << 2);
    p36_vk.vision_attention = P36_VK_KERNEL("vulkan/vision_attention.spv", 2, 4, 1u << 1);
    p36_vk.matmul_f32 = P36_VK_KERNEL("vulkan/matmul_f32.spv", 3, 16, 1u << 2);
    p36_vk.matmul_f32_fast = P36_VK_KERNEL("vulkan/matmul_f32_fast.spv", 3, 16, 1u << 2);
    p36_vk.matmul_f32_fast_w256 = P36_VK_KERNEL("vulkan/matmul_f32_fast_w256.spv", 3, 16, 1u << 2);
    p36_vk.add = P36_VK_KERNEL("vulkan/add.spv", 3, 4, 1u << 2);
    p36_vk.fwht = P36_VK_KERNEL("vulkan/fwht.spv", 2, 8, 1u << 1);
    p36_vk.signs_mul = P36_VK_KERNEL("vulkan/signs_mul.spv", 3, 8, 1u << 2);
    p36_vk.v_grouped_permute = P36_VK_KERNEL("vulkan/v_grouped_permute.spv", 2, 20, 1u << 1);
    p36_vk.hadamard_prepare = P36_VK_KERNEL("vulkan/hadamard_prepare.spv", 3, 24, 1u << 2);
    p36_vk.directional_steering = P36_VK_KERNEL("vulkan/directional_steering.spv", 2, 16, 1u << 0);
    p36_vk.rms_norm = P36_VK_KERNEL("vulkan/rms_norm.spv", 3, 12, 1u << 2);
    p36_vk.add_rms_norm = P36_VK_KERNEL("vulkan/add_rms_norm.spv", 5, 8, (1u << 3) | (1u << 4));
    p36_vk.add_rms_norm_q8_k = P36_VK_KERNEL("vulkan/add_rms_norm_q8_k.spv", 8, 24,
                                                 (1u << 3) | (1u << 4) | (1u << 5) | (1u << 7));
    p36_vk.swiglu = P36_VK_KERNEL("vulkan/swiglu.spv", 3, 12, 1u << 2);
    p36_vk.swiglu_q8_k = P36_VK_KERNEL("vulkan/swiglu_q8_k.spv", 4, 20,
                                        (1u << 2) | (1u << 3));
    p36_vk.rope = P36_VK_KERNEL("vulkan/rope_qwen.spv", 1, 16, 1u << 0);
    p36_vk.rope_mrope = P36_VK_KERNEL("vulkan/rope_qwen_mrope.spv", 2, 12, 1u << 0);
    p36_vk.rms_norm_rope = P36_VK_KERNEL("vulkan/rms_norm_rope_qwen.spv", 3, 24, 1u << 2);
    p36_vk.rms_norm_rope_kv = P36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen.spv", 5, 24,
                                            (1u << 2) | (1u << 4));
    p36_vk.top2 = P36_VK_KERNEL("vulkan/top2.spv", 3, 12, (1u << 1) | (1u << 2));
    p36_vk.argmax_rows = P36_VK_KERNEL("vulkan/argmax_rows.spv", 3, 12,
                                       (1u << 1) | (1u << 2));
    p36_vk.topk8 = P36_VK_KERNEL("vulkan/topk8.spv", 3, 12, (1u << 1) | (1u << 2));
    p36_vk.logits_minp = P36_VK_KERNEL("vulkan/logits_minp_pack.spv", 2, 24, (1u << 1));
    p36_vk.copy_rows = P36_VK_KERNEL("vulkan/copy_rows.spv", 2, 20, 1u << 1);
    p36_vk.embed_row = P36_VK_KERNEL("vulkan/embed_row.spv", 3, 24, 1u << 2);
    p36_vk.recur_window = P36_VK_KERNEL("vulkan/recur_window.spv", 3, 12, (1u << 0) | (1u << 2));
    p36_vk.conv_silu = P36_VK_KERNEL("vulkan/conv_silu.spv", 3, 12, 1u << 2);
    p36_vk.recur_conv_silu_decode = P36_VK_KERNEL("vulkan/recur_conv_silu_decode.spv", 4, 12,
                                                   (1u << 0) | (1u << 3));
    p36_vk.recur_norm_gate = P36_VK_KERNEL("vulkan/recur_norm_gate.spv", 3, 12, 1u << 0);
    p36_vk.recur_norm_gate_q8_k = P36_VK_KERNEL(
        "vulkan/recur_norm_gate_q8_k.spv", 4, 8, (1u << 0) | (1u << 3));
    p36_vk.delta_qk = P36_VK_KERNEL("vulkan/delta_qk.spv", 3, 24, (1u << 1) | (1u << 2));
    p36_vk.delta_qkv = P36_VK_KERNEL("vulkan/delta_qkv.spv", 4, 24,
                                     (1u << 1) | (1u << 2) | (1u << 3));
    p36_vk.delta_gates = P36_VK_KERNEL("vulkan/delta_gates.spv", 5, 8, 1u << 4);
    p36_vk.quantize_q8_0 = P36_VK_KERNEL("vulkan/quantize_q8_0.spv", 2, 12, 1u << 1);
    p36_vk.quantize_q8_k = P36_VK_KERNEL("vulkan/quantize_q8_k.spv", 2, 12, 1u << 1);
    p36_vk.matmul_q8_0 = P36_VK_KERNEL("vulkan/matmul_q8_0.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_p36 = P36_VK_KERNEL("vulkan/matmul_q8_0_p36.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b_pair = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b_pair.spv", 5, 20,
                                                 (1u << 3) | (1u << 4));
    p36_vk.shared_gate_up_decode = P36_VK_KERNEL("vulkan/shared_gate_up_decode.spv", 4, 16, 1u << 3);
    p36_vk.shared_down_tail_decode = P36_VK_KERNEL("vulkan/shared_down_tail_decode.spv", 4, 12, 1u << 3);
    p36_vk.matmul_q8_0_mm = P36_VK_KERNEL("vulkan/matmul_q8_0_mm.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_mm_f16 = P36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_mm_f16_out32 = P36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16_out32.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_f32b_nx = P36_VK_KERNEL("vulkan/matmul_q8_0_f32b_nx.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode = P36_VK_KERNEL("vulkan/matmul_q8_0_decode.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode_p36 = P36_VK_KERNEL("vulkan/matmul_q8_0_decode_p36.spv", 3, 20, 1u << 2);
    p36_vk.matmul_q8_0_decode_b64 = P36_VK_KERNEL("vulkan/matmul_q8_0_decode_b64.spv", 3, 20, 1u << 2);
    p36_vk.matmul_kquant = P36_VK_KERNEL("vulkan/matmul_kquant.spv", 3, 28, 1u << 2);
    p36_vk.matmul_q5k_mmq = P36_VK_KERNEL("vulkan/matmul_q5k_mmq.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q6k_mmq = P36_VK_KERNEL("vulkan/matmul_q6k_mmq.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q5k_mmq_fast = P36_VK_KERNEL("vulkan/matmul_q5k_mmq_fast.spv", 3, 16, 1u << 2);
    p36_vk.matmul_q6k_mmq_fast = P36_VK_KERNEL("vulkan/matmul_q6k_mmq_fast.spv", 3, 16, 1u << 2);
    p36_vk.delta_net = P36_VK_KERNEL("vulkan/delta_net.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_fast = P36_VK_KERNEL("vulkan/delta_net_fast.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_cols = P36_VK_KERNEL("vulkan/delta_net_cols.spv", 6, 16, (1u << 0) | (1u << 5));
    p36_vk.delta_net_cols_f16 = P36_VK_KERNEL("vulkan/delta_net_cols_f16.spv", 6, 16, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode = P36_VK_KERNEL("vulkan/delta_net_decode.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode_reg = P36_VK_KERNEL("vulkan/delta_net_decode_reg.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.delta_net_decode_reg_f16 = P36_VK_KERNEL("vulkan/delta_net_decode_reg_f16.spv", 6, 12, (1u << 0) | (1u << 5));
    p36_vk.attn_scores = P36_VK_KERNEL("vulkan/attn_scores.spv", 3, 28, 1u << 2);
    p36_vk.attn_post = P36_VK_KERNEL("vulkan/attn_post.spv", 4, 28, (1u << 0) | (1u << 3));
    p36_vk.attn_reduce = P36_VK_KERNEL("vulkan/attn_reduce.spv", 4, 28, 1u << 3);
    p36_vk.attn_decode_fused = P36_VK_KERNEL("vulkan/attn_decode_fused.spv", 6, 40, 1u << 5);
    p36_vk.attn_decode_split = P36_VK_KERNEL("vulkan/attn_decode_split.spv", 5, 44, 1u << 4);
    p36_vk.attn_prefill_qtile = P36_VK_KERNEL("vulkan/attn_prefill_qtile.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_qtile2 = P36_VK_KERNEL("vulkan/attn_prefill_qtile2.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_qtile2_gqa6 = P36_VK_KERNEL("vulkan/attn_prefill_qtile2_gqa6.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_fa_gqa6 = P36_VK_KERNEL("vulkan/attn_prefill_fa_gqa6.spv", 5, 48, 1u << 4);
    p36_vk.attn_prefill_fa_gqa8 = P36_VK_KERNEL("vulkan/attn_prefill_fa_gqa8.spv", 5, 48, 1u << 4);
    p36_vk.attn_combine = P36_VK_KERNEL("vulkan/attn_combine.spv", 4, 32, 1u << 3);
    p36_vk.moe_gate_up = P36_VK_KERNEL("vulkan/moe_gate_up.spv", 8, 28, 1u << 6);
    p36_vk.moe_gate_up_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_iq2s.spv", 8, 28, 1u << 6);
    p36_vk.router_topk = P36_VK_KERNEL("vulkan/router_topk.spv", 3, 16, (1u << 1) | (1u << 2));
    p36_vk.moe_tiles = P36_VK_KERNEL("vulkan/moe_tiles.spv", 3, 20, 1u << 1);
    p36_vk.kv_store = P36_VK_KERNEL("vulkan/kv_store.spv", 4, 16, (1u << 0) | (1u << 1));
    p36_vk.kv_store_quant = P36_VK_KERNEL("vulkan/kv_store_quant.spv", 2, 20, 1u << 0);
    p36_vk.rms_norm_rope_kv_quant = P36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen_quant.spv", 5, 32, (1u << 2) | (1u << 4));
    p36_vk.moe_gate_up_f32b = P36_VK_KERNEL("vulkan/moe_gate_up_f32b.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_f32b_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_f32b_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode = P36_VK_KERNEL("vulkan/moe_gate_up_decode.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq3xxs = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq3xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_q3k = P36_VK_KERNEL("vulkan/moe_gate_up_decode_q3k.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_decode_iq2s_iq2xxs = P36_VK_KERNEL("vulkan/moe_gate_up_decode_iq2s_iq2xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_down_q2k_f32b = P36_VK_KERNEL("vulkan/moe_down_q2k_f32b.spv", 5, 24, 1u << 4);
    p36_vk.moe_down_q2k_f32b_iq2s = P36_VK_KERNEL("vulkan/moe_down_q2k_f32b_iq2s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_q2k_sum_decode = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode.spv", 6, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq2s = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq2s.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq3s = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq3s.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_iq3xxs = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_iq3xxs.spv", 7, 24, 1u << 5);
    p36_vk.moe_down_q2k_sum_decode_q3k = P36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode_q3k.spv", 6, 24, 1u << 5);
    p36_vk.moe_gate_up_q4k_f32b = P36_VK_KERNEL("vulkan/moe_gate_up_q4k_f32b.spv", 7, 24, 1u << 6);
    p36_vk.moe_down_q4k_sum_decode = P36_VK_KERNEL("vulkan/moe_down_q4k_sum_decode.spv", 6, 24, 1u << 5);
    p36_vk.moe_gate_up_gemm = P36_VK_KERNEL("vulkan/moe_gate_up_gemm.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq3xxs = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq3xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_q3k = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_q3k.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s_iq2xxs = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s_iq2xxs.spv", 8, 24, 1u << 6);
    p36_vk.moe_down_gemm = P36_VK_KERNEL("vulkan/moe_down_gemm.spv", 5, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq2s = P36_VK_KERNEL("vulkan/moe_down_gemm_iq2s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3s = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3s.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3xxs = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3xxs.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_q3k = P36_VK_KERNEL("vulkan/moe_down_gemm_q3k.spv", 5, 24, 1u << 4);
    p36_vk.moe_gate_up_gemm_t16 = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_t16.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s_t16 = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s_t16.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq3xxs_t16 = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq3xxs_t16.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_q3k_t16 = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_q3k_t16.spv", 8, 24, 1u << 6);
    p36_vk.moe_gate_up_gemm_iq2s_iq2xxs_t16 = P36_VK_KERNEL("vulkan/moe_gate_up_gemm_iq2s_iq2xxs_t16.spv", 8, 24, 1u << 6);
    p36_vk.moe_down_gemm_t16 = P36_VK_KERNEL("vulkan/moe_down_gemm_t16.spv", 5, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq2s_t16 = P36_VK_KERNEL("vulkan/moe_down_gemm_iq2s_t16.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3s_t16 = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3s_t16.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_iq3xxs_t16 = P36_VK_KERNEL("vulkan/moe_down_gemm_iq3xxs_t16.spv", 6, 24, 1u << 4);
    p36_vk.moe_down_gemm_q3k_t16 = P36_VK_KERNEL("vulkan/moe_down_gemm_q3k_t16.spv", 5, 24, 1u << 4);
    p36_vk.moe_matvec = P36_VK_KERNEL("vulkan/moe_matvec.spv", 6, 36, 1u << 4);
    p36_vk.moe_matvec_fast = P36_VK_KERNEL("vulkan/moe_matvec_fast.spv", 6, 36, 1u << 4);
    p36_vk.dense_iq3_xxs_decode = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_xxs_decode_r4 = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode_r4.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_xxs_decode_nx = P36_VK_KERNEL("vulkan/dense_iq3_xxs_decode_nx.spv", 4, 20, 1u << 2);
    p36_vk.dense_iq3_xxs_mmq = P36_VK_KERNEL("vulkan/dense_iq3_xxs_mmq.spv", 5, 20, 1u << 2);
    /* Bindings: gate weights, up weights, gate out, up out, tables, b16. */
    p36_vk.dense_iq3_xxs_mmq_pair = P36_VK_KERNEL("vulkan/dense_iq3_xxs_mmq_pair.spv", 6, 20,
                                                  (1u << 2) | (1u << 3));
    p36_vk.dense_iq3_s_decode = P36_VK_KERNEL("vulkan/dense_iq3_s_decode.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_full = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_full.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_r4 = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_r4.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_decode_r1 = P36_VK_KERNEL("vulkan/dense_iq3_s_decode_r1.spv", 4, 16, 1u << 2);
    p36_vk.dense_iq3_s_mmq = P36_VK_KERNEL("vulkan/dense_iq3_s_mmq.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq3_s_bm64_mmq = P36_VK_KERNEL("vulkan/dense_iq3_s_bm64_mmq.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq3_s_mmq_r4 = P36_VK_KERNEL("vulkan/dense_iq3_s_mmq_r4.spv", 5, 20, 1u << 2);
    p36_vk.dense_iq4_xs = P36_VK_KERNEL("vulkan/dense_iq4_xs.spv", 3, 20, 1u << 2);
    p36_vk.dense_iq4_xs_decode = P36_VK_KERNEL("vulkan/dense_iq4_xs_decode.spv", 3, 16, 1u << 2);
    p36_vk.dense_iq4_xs_rows = P36_VK_KERNEL("vulkan/dense_iq4_xs_rows.spv", 3, 20, 1u << 2);
    p36_vk.dense_iq4_xs_mmq = P36_VK_KERNEL("vulkan/dense_iq4_xs_mmq.spv", 3, 20, 1u << 2);
    p36_vk.dense_iq1_m = P36_VK_KERNEL("vulkan/dense_iq1_m.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode = P36_VK_KERNEL("vulkan/dense_extra_decode.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_q2_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_pq2_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_small_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_small_q2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_small_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_small_pq2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_decode_ptq1_0 = P36_VK_KERNEL("vulkan/dense_extra_decode_ptq1_0.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_ptq1_0_nx = P36_VK_KERNEL("vulkan/dense_extra_decode_ptq1_0_nx.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_decode_iq2s_pair = P36_VK_KERNEL(
        "vulkan/dense_extra_decode_iq2s_pair.spv", 6, 24, (1u << 4) | (1u << 5));
    p36_vk.dense_extra_decode_iq2s = P36_VK_KERNEL("vulkan/dense_extra_decode_iq2s.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_iq2xxs = P36_VK_KERNEL("vulkan/dense_extra_decode_iq2xxs.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_decode_iq2xs = P36_VK_KERNEL("vulkan/dense_extra_decode_iq2xs.spv", 4, 20, 1u << 2);
    p36_vk.dense_extra_mmq = P36_VK_KERNEL("vulkan/dense_extra_mmq.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_q2_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_q2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_pq2_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_pq2_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_ptq1_0 = P36_VK_KERNEL("vulkan/dense_extra_mmq_ptq1_0.spv", 4, 24, 1u << 2);
    p36_vk.dense_extra_mmq_iq2s = P36_VK_KERNEL("vulkan/dense_extra_mmq_iq2s.spv", 4, 24, 1u << 2);
    p36_vk.dense_kquant_mmq = P36_VK_KERNEL("vulkan/dense_kquant_mmq.spv", 4, 28, 1u << 2);
    p36_vk.predequant_b16 = P36_VK_KERNEL("vulkan/predequant_b16.spv", 2, 4, 1u << 1);
    p36_vk.dense_kquant_decode = P36_VK_KERNEL("vulkan/dense_kquant_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q4k_decode = P36_VK_KERNEL("vulkan/dense_q4k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q5k_decode = P36_VK_KERNEL("vulkan/dense_q5k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q3k_decode = P36_VK_KERNEL("vulkan/dense_q3k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q2k_decode = P36_VK_KERNEL("vulkan/dense_q2k_decode.spv", 3, 28, 1u << 2);
    p36_vk.dense_q3k_rows = P36_VK_KERNEL("vulkan/dense_q3k_rows.spv", 3, 28, 1u << 2);
    p36_vk.dense_q4k_rows = P36_VK_KERNEL("vulkan/dense_q4k_rows.spv", 3, 28, 1u << 2);
    p36_vk.dense_q5k_rows = P36_VK_KERNEL("vulkan/dense_q5k_rows.spv", 3, 28, 1u << 2);
    p36_vk.dense_q6k_rows = P36_VK_KERNEL("vulkan/dense_q6k_rows.spv", 3, 28, 1u << 2);
    p36_vk.dense_q3k_rows2 = P36_VK_KERNEL("vulkan/dense_q3k_rows2.spv", 3, 28, 1u << 2);
    p36_vk.dense_q4k_rows2 = P36_VK_KERNEL("vulkan/dense_q4k_rows2.spv", 3, 28, 1u << 2);
    p36_vk.dense_q5k_rows2 = P36_VK_KERNEL("vulkan/dense_q5k_rows2.spv", 3, 28, 1u << 2);
    p36_vk.dense_q6k_rows2 = P36_VK_KERNEL("vulkan/dense_q6k_rows2.spv", 3, 28, 1u << 2);
    p36_vk.moe_reduce = P36_VK_KERNEL("vulkan/moe_reduce.spv", 3, 8, 1u << 2);
    p36_vk.moe_down_route_reduce = P36_VK_KERNEL("vulkan/moe_down_route_reduce.spv", 3, 8, 1u << 2);
    p36_vk.ffn_tail = P36_VK_KERNEL("vulkan/ffn_tail.spv", 3, 8, 1u << 0);

    p36_vk.api_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_instance_version =
        (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enumerate_instance_version) (void)enumerate_instance_version(&p36_vk.api_version);

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "p36",
        .applicationVersion = 1,
        .pEngineName = "p36_vulkan",
        .engineVersion = 1,
        .apiVersion = p36_vk.api_version,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &p36_vk.instance) != VK_SUCCESS) goto fail;

    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(p36_vk.instance, &ndev, NULL) != VK_SUCCESS || ndev == 0) goto fail;
    VkPhysicalDevice *devs = calloc(ndev, sizeof(devs[0]));
    if (!devs) goto fail;
    if (vkEnumeratePhysicalDevices(p36_vk.instance, &ndev, devs) != VK_SUCCESS) {
        free(devs);
        goto fail;
    }

    for (uint32_t di = 0; di < ndev && !p36_vk.physical; di++) {
        VkPhysicalDeviceProperties candidate_props;
        VkPhysicalDeviceFeatures candidate_features;
        vkGetPhysicalDeviceProperties(devs[di], &candidate_props);
        vkGetPhysicalDeviceFeatures(devs[di], &candidate_features);
        if (candidate_props.apiVersion < VK_API_VERSION_1_1 ||
            !candidate_features.shaderFloat64 || !candidate_features.shaderInt64) continue;
#ifdef P36_VULKAN_REQUIRE_BC250
        if (candidate_props.vendorID != 0x1002u || candidate_props.deviceID != 0x13feu) continue;
#endif
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, NULL);
        VkQueueFamilyProperties *qprops = calloc(nq, sizeof(qprops[0]));
        if (!qprops) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, qprops);
        uint32_t compute = UINT32_MAX;
        uint32_t ace = UINT32_MAX;
        for (uint32_t qi = 0; qi < nq; qi++) {
            if ((qprops[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            if (compute == UINT32_MAX) compute = qi;
            if ((qprops[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 &&
                ace == UINT32_MAX) ace = qi;
        }
        if (compute != UINT32_MAX) {
            const char *ace_env = getenv("P36_VK_ACE");
            bool bc250 = candidate_props.vendorID == 0x1002u &&
                         candidate_props.deviceID == 0x13feu;
            bool use_ace = bc250 && ace != UINT32_MAX &&
                           (!ace_env || !ace_env[0] || ace_env[0] != '0');
            uint32_t qi = use_ace ? ace : compute;
            p36_vk.physical = devs[di];
            p36_vk.queue_family = qi;
            p36_vk.queue_count = qprops[qi].queueCount;
            p36_vk.timestamp_valid_bits = qprops[qi].timestampValidBits;
            p36_vk.ace_queue = use_ace;
        }
        free(qprops);
    }
    free(devs);
    if (!p36_vk.physical) {
        fprintf(stderr, "p36: no compatible Vulkan compute device found\n");
        goto fail;
    }

    vkGetPhysicalDeviceProperties(p36_vk.physical, &p36_vk.props);
    if (p36_vk.api_version < VK_API_VERSION_1_1 ||
        p36_vk.props.apiVersion < VK_API_VERSION_1_1) {
        fprintf(stderr, "p36: Vulkan 1.1 or newer is required\n");
        goto fail;
    }
    VkPhysicalDeviceSubgroupProperties subgroup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    };
    if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
        p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
        subgroup.pNext = &subgroup_size_props;
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup,
    };
    vkGetPhysicalDeviceProperties2(p36_vk.physical, &props2);
    p36_vk.props = props2.properties;
    p36_vk.subgroup_size = subgroup.subgroupSize;
    p36_vk.subgroup_arithmetic =
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    p36_vk.subgroup_clustered =
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0;
    p36_vk.bc250 = p36_vk.props.vendorID == 0x1002u && p36_vk.props.deviceID == 0x13feu;
    /* Timestamp capability first: every GPU-ms number below divides by
     * timestampPeriod, so log it (and whether the chosen queue supports
     * timestamps at all) before any measurement can be trusted. */
    fprintf(stderr,
            "p36: timestamps period_ns=%.6f computeAndGraphics=%u valid_bits=%u "
            "queue_family=%u (%s) ace=%d\n",
            (double)p36_vk.props.limits.timestampPeriod,
            (unsigned)p36_vk.props.limits.timestampComputeAndGraphics,
            p36_vk.timestamp_valid_bits,
            p36_vk.queue_family,
            p36_vk.ace_queue ? "ACE compute" : "graphics fallback",
            (int)p36_vk.ace_queue);
#ifdef P36_VULKAN_REQUIRE_BC250
    if (!p36_vk.bc250) {
        fprintf(stderr, "p36: this vulkan-bc250 build requires an AMD BC-250 (1002:13fe)\n");
        goto fail;
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(p36_vk.physical, &p36_vk.mem_props);

    uint64_t device_bytes = p36_vk_memory_total();
    if (device_bytes) {
        fprintf(stderr, "p36: Vulkan device %s, %.2f GiB%s\n",
                p36_vk.props.deviceName, (double)device_bytes / 1073741824.0,
                p36_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? " UMA" : " memory");
    } else {
        fprintf(stderr, "p36: Vulkan device %s\n", p36_vk.props.deviceName);
    }
    fprintf(stderr, "p36: Vulkan generic backend%s, subgroup %u\n",
            p36_vk.bc250 ? ", BC-250 fast path" : "", p36_vk.subgroup_size);
    if (p36_vk.bc250) {
        fprintf(stderr, "p36: Vulkan queue family %u: %s (%u queue%s exposed)\n",
                p36_vk.queue_family,
                p36_vk.ace_queue ? "ACE compute" : "graphics fallback",
                p36_vk.queue_count, p36_vk.queue_count == 1 ? "" : "s");
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = p36_vk.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    /* The matmul / attention / delta-net kernels accumulate in fp64 and
     * emulate correctly rounded f32 fma via int64 bit ops to stay bit-exact
     * against the CPU reference engine, so shaderFloat64 and shaderInt64 are
     * hard requirements (RADV exposes both on the BC-250 target). */
    VkPhysicalDevice16BitStorageFeatures storage16_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    };
    VkPhysicalDevice8BitStorageFeatures storage8_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    };
    VkPhysicalDeviceShaderFloat16Int8Features f16_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures int_dot_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_query = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    };
    if (p36_vk.api_version >= VK_API_VERSION_1_2 &&
        p36_vk.props.apiVersion >= VK_API_VERSION_1_2) {
        storage16_query.pNext = &storage8_query;
        storage8_query.pNext = &f16_query;
        if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
            p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
            f16_query.pNext = &int_dot_query;
        if (p36_vk.api_version >= VK_API_VERSION_1_3 &&
            p36_vk.props.apiVersion >= VK_API_VERSION_1_3)
            int_dot_query.pNext = &subgroup_size_query;
    }
    VkPhysicalDeviceFeatures2 feat2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &storage16_query,
    };
    vkGetPhysicalDeviceFeatures2(p36_vk.physical, &feat2);
    if (!feat2.features.shaderFloat64 || !feat2.features.shaderInt64) {
        fprintf(stderr, "p36: Vulkan shaderFloat64 and shaderInt64 are required\n");
        goto fail;
    }
    p36_vk.have_storage16 = storage16_query.storageBuffer16BitAccess != 0;
    p36_vk.have_storage8 = storage8_query.storageBuffer8BitAccess != 0;
    p36_vk.have_f16 = p36_vk.have_storage16 && f16_query.shaderFloat16;
    p36_vk.have_int_dot = int_dot_query.shaderIntegerDotProduct != 0;
    p36_vk.subgroup_size_control = subgroup_size_query.subgroupSizeControl &&
        subgroup_size_query.computeFullSubgroups &&
        subgroup_size_props.minSubgroupSize <= 32u &&
        subgroup_size_props.maxSubgroupSize >= 32u;
    VkPhysicalDeviceFeatures enabled = { .shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE };
    VkPhysicalDevice16BitStorageFeatures storage16_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .storageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDevice8BitStorageFeatures storage8_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        .storageBuffer8BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceShaderFloat16Int8Features f16_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16 = VK_TRUE,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures int_dot_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        .shaderIntegerDotProduct = VK_TRUE,
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .pEnabledFeatures = &enabled,
    };
    if (p36_vk.have_storage16) {
        dci.pNext = &storage16_enable;
        void **storage_tail = &storage16_enable.pNext;
        if (p36_vk.have_storage8) {
            storage16_enable.pNext = &storage8_enable;
            storage_tail = &storage8_enable.pNext;
        }
        if (p36_vk.have_f16) {
            *storage_tail = &f16_enable;
            if (p36_vk.have_int_dot) {
                f16_enable.pNext = &int_dot_enable;
                if (p36_vk.subgroup_size_control)
                    int_dot_enable.pNext = &subgroup_size_enable;
            } else if (p36_vk.subgroup_size_control) {
                f16_enable.pNext = &subgroup_size_enable;
            }
        } else if (p36_vk.have_int_dot) {
            *storage_tail = &int_dot_enable;
            if (p36_vk.subgroup_size_control)
                int_dot_enable.pNext = &subgroup_size_enable;
        } else if (p36_vk.subgroup_size_control) {
            *storage_tail = &subgroup_size_enable;
        }
    } else if (p36_vk.have_int_dot) {
        dci.pNext = &int_dot_enable;
        if (p36_vk.subgroup_size_control)
            int_dot_enable.pNext = &subgroup_size_enable;
    } else if (p36_vk.subgroup_size_control) {
        dci.pNext = &subgroup_size_enable;
    }
    if (vkCreateDevice(p36_vk.physical, &dci, NULL, &p36_vk.device) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(p36_vk.device, p36_vk.queue_family, 0, &p36_vk.queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = p36_vk.queue_family,
    };
    if (vkCreateCommandPool(p36_vk.device, &cpci, NULL, &p36_vk.command_pool) != VK_SUCCESS) goto fail;

    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16384,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (vkCreateDescriptorPool(p36_vk.device, &dpci, NULL, &p36_vk_ring[i].pool) != VK_SUCCESS) goto fail;
    }
    p36_vk_ring_activate_unlocked(0);

    if (getenv("P36_VK_PROF_KERNEL") && p36_vk.timestamp_valid_bits != 0) {
        /* P36_VK_PROF_SKIP=N leaves the first N dispatches untimed so the report
         * can describe a decode window instead of the prefill burst. */
        const char *skip = getenv("P36_VK_PROF_SKIP");
        if (skip && *skip) p36_vk_prof_skip_left = strtoull(skip, NULL, 10);
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = P36_VK_PROF_QUERY_CAP,
        };
        if (vkCreateQueryPool(p36_vk.device, &qpci, NULL, &p36_vk.query_pool) == VK_SUCCESS) {
            p36_vk.prof_kernel = true;
        }
    }

    p36_vk.ready = true;
    p36_vk_stream_profile_init();
    if (getenv("P36_VK_PROF") || getenv("P36_VK_PROF_KERNEL") || getenv("P36_VK_PROF_OP") ||
        getenv("P36_VK_PROF_WG_HIST")) atexit(p36_vk_prof_report);
    pthread_mutex_unlock(&p36_vk_mu);
    return 1;

fail:
    if (p36_vk.device) vkDeviceWaitIdle(p36_vk.device);
    if (p36_vk.query_pool) vkDestroyQueryPool(p36_vk.device, p36_vk.query_pool, NULL);
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pool) vkDestroyDescriptorPool(p36_vk.device, p36_vk_ring[i].pool, NULL);
        if (p36_vk_ring[i].fence) vkDestroyFence(p36_vk.device, p36_vk_ring[i].fence, NULL);
    }
    memset(p36_vk_ring, 0, sizeof(p36_vk_ring));
    p36_vk_ring_cur = 0;
    if (p36_vk.command_pool) vkDestroyCommandPool(p36_vk.device, p36_vk.command_pool, NULL);
    if (p36_vk.device) vkDestroyDevice(p36_vk.device, NULL);
    if (p36_vk.instance) vkDestroyInstance(p36_vk.instance, NULL);
    memset(&p36_vk, 0, sizeof(p36_vk));
    pthread_mutex_unlock(&p36_vk_mu);
    return 0;
}

