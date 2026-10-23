static p36_gpu_tensor *p36_vk_tensor_alloc_kind_unlocked(uint64_t bytes, bool device_pref,
                                                         bool clear) {
    p36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->bytes = bytes;
    tensor->alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tensor->alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(p36_vk.device, &bci, NULL, &tensor->buffer) != VK_SUCCESS) {
        free(tensor);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p36_vk.device, tensor->buffer, &req);
    /* Prefer cached (snooped) host-visible memory: on the BC-250 the
     * write-combined GTT type is not reliably coherent for device writes
     * followed by host reads, and every host op reads GPU output.
     * Weights are the exception (device_pref): written once by memcpy and
     * only ever read by shaders, so they go to the device-local heap, which
     * is several times larger than the host-cached one and must hold the
     * routed expert banks. */
    uint32_t mt = UINT32_MAX;
    if (device_pref) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        tensor->device_local = mt != UINT32_MAX;
    }
    if (mt == UINT32_MAX) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    if (mt == UINT32_MAX) {
        mt = p36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
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
    if (vkBindBufferMemory(p36_vk.device, tensor->buffer, tensor->memory, 0) != VK_SUCCESS ||
        vkMapMemory(p36_vk.device, tensor->memory, 0, req.size, 0, (void **)&tensor->data) != VK_SUCCESS)
    {
        vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (clear) memset(tensor->data, 0, (size_t)tensor->alloc_bytes);

    p36_gpu_live_bytes += bytes;
    if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
    return tensor;
}

static void p36_vk_tensor_free_unlocked(p36_gpu_tensor *tensor);
static void p36_vk_b16_ring_destroy_unlocked(void);

/* Pooled scratch keeps its VRAM; free it all when an allocation fails so a
 * long-context prefill retries instead of dying on a full heap. */
static bool p36_vk_pools_drain_unlocked(void) {
    bool any = p36_vk_pool_n || p36_vk_private_pool_n;
    while (p36_vk_pool_n) p36_vk_tensor_free_unlocked(p36_vk_pool[--p36_vk_pool_n]);
    while (p36_vk_private_pool_n)
        p36_vk_tensor_free_unlocked(p36_vk_private_pool[--p36_vk_private_pool_n]);
    return any;
}

static p36_gpu_tensor *p36_vk_tensor_alloc_scratch_unlocked(uint64_t bytes) {
    const uint64_t alloc_bytes = bytes ? p36_round_up_u64(bytes, 4) : 4;
    if (p36_gpu_quality) return p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    for (uint32_t i = 0; i < p36_vk_private_pool_n; i++) {
        p36_gpu_tensor *tensor = p36_vk_private_pool[i];
        if (tensor->alloc_bytes != alloc_bytes) continue;
        p36_vk_private_pool[i] = p36_vk_private_pool[--p36_vk_private_pool_n];
        tensor->bytes = bytes;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        return tensor;
    }
    if (p36_vk_private_bytes <= P36_VK_PRIVATE_LIMIT &&
        alloc_bytes <= P36_VK_PRIVATE_LIMIT - p36_vk_private_bytes) {
        p36_gpu_tensor *tensor = p36_vk_tensor_alloc_dl_unlocked(bytes);
        if (tensor && tensor->private_charge <=
                      P36_VK_PRIVATE_LIMIT - p36_vk_private_bytes) {
            tensor->private_scratch = true;
            p36_vk_private_bytes += tensor->private_charge;
            if (p36_vk_private_bytes > p36_vk_private_peak)
                p36_vk_private_peak = p36_vk_private_bytes;
            return tensor;
        }
        p36_vk_tensor_free_unlocked(tensor);
    }
    p36_gpu_tensor *t = p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    if (!t && p36_vk_pools_drain_unlocked())
        t = p36_vk_tensor_alloc_kind_unlocked(bytes, false, true);
    return t;
}

static void p36_vk_tensor_free_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (!tensor->owner) {
        p36_gpu_live_bytes -= tensor->bytes;
        if (tensor->private_scratch) p36_vk_private_bytes -= tensor->private_charge;
        if (tensor->data) vkUnmapMemory(p36_vk.device, tensor->memory);
        if (tensor->memory) vkFreeMemory(p36_vk.device, tensor->memory, NULL);
        if (tensor->buffer) vkDestroyBuffer(p36_vk.device, tensor->buffer, NULL);
    }
    free(tensor);
}

/* Pool-or-free for a tensor with no pending GPU references. A full pool
 * evicts round-robin instead of rejecting: otherwise a prefill chunk fills
 * every slot with chunk-sized scratch and decode-sized buffers can never
 * enter, turning each decode step into fresh vkAllocateMemory calls. */
static uint32_t p36_vk_pool_rr;
static uint32_t p36_vk_private_pool_rr;
static void p36_vk_tensor_retire_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor->owner && tensor->private_scratch) {
        p36_gpu_live_bytes -= tensor->bytes;
        tensor->bytes = 0;
        if (p36_vk_private_pool_n < P36_VK_PRIVATE_POOL_CAP) {
            p36_vk_private_pool[p36_vk_private_pool_n++] = tensor;
        } else {
            p36_vk_private_pool[p36_vk_private_pool_rr]->bytes = 0;
            p36_vk_tensor_free_unlocked(p36_vk_private_pool[p36_vk_private_pool_rr]);
            p36_vk_private_pool[p36_vk_private_pool_rr] = tensor;
            p36_vk_private_pool_rr =
                (p36_vk_private_pool_rr + 1u) % P36_VK_PRIVATE_POOL_CAP;
        }
    } else if (!tensor->owner && !tensor->device_local && tensor->data &&
        tensor->alloc_bytes <= P36_VK_POOL_MAX_BYTES) {
        p36_gpu_live_bytes -= tensor->bytes;
        if (p36_vk_pool_n < P36_VK_POOL_CAP) {
            p36_vk_pool[p36_vk_pool_n++] = tensor;
        } else {
            /* Pooled entries already left the live-byte count at retire. */
            p36_vk_pool[p36_vk_pool_rr]->bytes = 0;
            p36_vk_tensor_free_unlocked(p36_vk_pool[p36_vk_pool_rr]);
            p36_vk_pool[p36_vk_pool_rr] = tensor;
            p36_vk_pool_rr = (p36_vk_pool_rr + 1u) % P36_VK_POOL_CAP;
        }
    } else {
        p36_vk_tensor_free_unlocked(tensor);
    }
}

/* Retire now if the tensor's last GPU reference has completed, else park
 * it on the retired queue for the next sweep. */
static void p36_vk_tensor_release_unlocked(p36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->last_use_seq > p36_vk_completed_seq) {
        if (p36_vk_retired_n == P36_VK_RETIRE_CAP) p36_vk_flush_reason_unlocked("submit_wait_retire_cap");
        if (tensor->last_use_seq > p36_vk_completed_seq) {
            p36_vk_retired[p36_vk_retired_n++] = tensor;
            return;
        }
    }
    p36_vk_tensor_retire_unlocked(tensor);
}

static void p36_vk_stream_cache_release_unlocked(bool keep_config) {
    bool enabled = p36_vk_stream.enabled;
    uint32_t budget = p36_vk_stream.budget;
    uint32_t full_layers = p36_vk_stream.full_layers;
    uint32_t runtime_cap = p36_vk_stream.runtime_cap;
    uint32_t allocation_failures = p36_vk_stream.allocation_failures;
    uint64_t expert_bytes = p36_vk_stream.configured_expert_bytes;

    p36_vk_tensor_free_unlocked(p36_vk_stream.gate);
    p36_vk_tensor_free_unlocked(p36_vk_stream.up);
    p36_vk_tensor_free_unlocked(p36_vk_stream.down);
    p36_vk_tensor_free_unlocked(p36_vk_stream.gate_scales);
    p36_vk_tensor_free_unlocked(p36_vk_stream.up_scales);
    p36_vk_tensor_free_unlocked(p36_vk_stream.down_scales);
    free(p36_vk_stream.entry);
    memset(&p36_vk_stream, 0, sizeof(p36_vk_stream));
    if (keep_config) {
        p36_vk_stream.enabled = enabled;
        p36_vk_stream.budget = budget;
        p36_vk_stream.full_layers = full_layers;
        p36_vk_stream.runtime_cap = runtime_cap;
        p36_vk_stream.allocation_failures = allocation_failures;
        p36_vk_stream.configured_expert_bytes = expert_bytes;
    }
}

static bool p36_vk_stream_copy_model_range(const void *model_map,
                                           uint64_t model_size,
                                           uint64_t offset,
                                           void *dst,
                                           uint64_t bytes) {
    if (bytes == 0) return true;
    if (!dst || bytes > (uint64_t)SIZE_MAX) return false;
    if (p36_gpu_model_fd >= 0) {
        uint64_t done = 0;
        while (done < bytes) {
            size_t n = (size_t)(bytes - done);
            if (n > (size_t)(256u << 20)) n = (size_t)(256u << 20);
            ssize_t got = pread(p36_gpu_model_fd,
                                (unsigned char *)dst + done,
                                n,
                                (off_t)(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;
            done += (uint64_t)got;
        }
        if (done == bytes) return true;
    }

    const unsigned char *src = p36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    if (!src) return false;
    memcpy(dst, src, (size_t)bytes);
    return true;
}

/* Submit the open batch, wait for it, then settle everything that was
 * deferred while it was recording: retired tensors and the descriptor
 * pool (all its sets belong to the finished batch). */
/* Command-buffer ring: the batch cap submits eagerly WITHOUT waiting, so
 * the GPU chews on earlier work while the host records ahead; only a host
 * read waits for everything. Hazard sets survive eager submits because the
 * pending work still executes in submission order. */
enum { P36_VK_CB_RING = 4 };
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorPool pool;
    bool pending;
    uint64_t seq;
} p36_vk_cb_slot;
static p36_vk_cb_slot p36_vk_ring[P36_VK_CB_RING];
static uint32_t p36_vk_ring_cur;

/* Device-lost latch.  A gfx ring timeout (amdgpu's 10 s TDR) destroys the
 * context: every subsequent Vulkan call against this device fails, and
 * RADV's command buffer handle becomes unusable, so recording into it
 * segfaults inside vkEndCommandBuffer rather than returning an error.
 * Once any submit-path call reports VK_ERROR_DEVICE_LOST we latch here and
 * fail every later batch operation at its entry point.  The latch is
 * deliberately one-way: there is no recover-and-continue path, because the
 * dispatches that were in flight produced no results and silently resuming
 * would corrupt the forward pass.  p36_session_sync() turns the failed pass
 * into an error string for the caller. */
static bool p36_vk_device_lost;
/* P36_VK_FAULT_INJECT_LOST=<n>: force the n-th vkQueueSubmit to report
 * device loss, so the latch and its unwind can be tested without provoking
 * a real TDR.  Diagnostic only; 0 (unset) disables it. */
static uint64_t p36_vk_fault_inject_lost;
static uint64_t p36_vk_submit_calls;

static int p36_vk_ok(VkResult rc, const char *what) {
    if (rc == VK_SUCCESS) return 1;
    if (rc == VK_ERROR_DEVICE_LOST && !p36_vk_device_lost) {
        p36_vk_device_lost = true;
        fprintf(stderr,
                "p36: Vulkan device lost during %s -- the GPU context was destroyed "
                "(usually an amdgpu ring timeout; check `journalctl -k` for "
                "\"ring gfx_0.0.0 timeout\").  All further GPU work is refused; "
                "restart the process.  Lower --prefill-chunk if this reproduces.\n",
                what);
    }
    return 0;
}

static bool p36_vk_device_lost_unlocked(void) { return p36_vk_device_lost; }

static bool p36_vk_gpu_busy_unlocked(void) {
    if (p36_vk_batch_recording) return true;
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        if (p36_vk_ring[i].pending) return true;
    }
    return false;
}

static int p36_vk_slot_wait_unlocked(p36_vk_cb_slot *s) {
    VkResult rc = VK_SUCCESS;
    if (!s->pending) return 1;
    uint64_t w0 = p36_vk.prof_kernel ? p36_vk_now_ns() : 0;
    if (p36_vk_device_lost) { s->pending = false; return 0; }
    /* Decode batches finish in tens of microseconds; spinning on the fence
     * status dodges the syscall + thread wakeup of a blocking wait. */
    for (uint32_t spin = 0; spin < 4096; spin++) {
        rc = vkGetFenceStatus(p36_vk.device, s->fence);
        if (rc != VK_NOT_READY) break;
    }
    if (rc == VK_NOT_READY) rc = vkWaitForFences(p36_vk.device, 1, &s->fence, VK_TRUE, UINT64_MAX);
    if (rc != VK_SUCCESS) {
        /* A lost device never signals this fence; drop the slot so callers
         * unwind instead of waiting on it again. */
        s->pending = false;
        return p36_vk_ok(rc, "fence wait");
    }
    if (p36_vk.prof_kernel)
        p36_vk_prof_op_add("vk_wait_fence", 0, 0, 0, 0, p36_vk_now_ns() - w0, 0, 0);
    vkResetFences(p36_vk.device, 1, &s->fence);
    if (s->pool) vkResetDescriptorPool(p36_vk.device, s->pool, 0);
    s->pending = false;
    if (s->seq > p36_vk_completed_seq) p36_vk_completed_seq = s->seq;
    return 1;
}

static int p36_vk_submit_current_unlocked(bool host_read) {
    p36_vk_cb_slot *s = &p36_vk_ring[p36_vk_ring_cur];
    if (!p36_vk_batch_recording) return 1;
    if (p36_vk_device_lost) return 0;
    if (host_read) {
        /* Host ops read these buffers right after the fence.  The fence's
         * implicit host visibility is not reliable on the BC-250's RADV
         * (intermittent stale reads), so flush shader writes to host
         * explicitly. */
        VkMemoryBarrier host_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(p36_vk.command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &host_barrier, 0, NULL, 0, NULL);
    }
    {
        /* P36_VK_PROF_KERNEL splits the submit path, which no other counter
         * covers: driver-side command buffer teardown versus queue submit. */
        uint64_t e0 = p36_vk.prof_kernel ? p36_vk_now_ns() : 0;
        VkResult rc = vkEndCommandBuffer(p36_vk.command_buffer);
        if (p36_vk.prof_kernel)
            p36_vk_prof_op_add("vk_end_cb", 0, 0, 0, 0, p36_vk_now_ns() - e0, 0, 0);
        if (rc != VK_SUCCESS) {
            p36_vk_batch_recording = false;
            return p36_vk_ok(rc, "vkEndCommandBuffer");
        }
    }
    {
        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &p36_vk.command_buffer,
        };
        uint64_t s0 = p36_vk.prof_kernel ? p36_vk_now_ns() : 0;
        VkResult rc = vkQueueSubmit(p36_vk.queue, 1, &si, s->fence);
        if (p36_vk.prof_kernel)
            p36_vk_prof_op_add("vk_queue_submit", 0, 0, 0, 0, p36_vk_now_ns() - s0, 0, 0);
        if (rc == VK_SUCCESS && p36_vk_fault_inject_lost &&
            ++p36_vk_submit_calls == p36_vk_fault_inject_lost) {
            rc = VK_ERROR_DEVICE_LOST;
        }
        if (rc != VK_SUCCESS) {
            p36_vk_batch_recording = false;
            return p36_vk_ok(rc, "vkQueueSubmit");
        }
    }
    s->pending = true;
    s->seq = ++p36_vk_submit_seq;
    p36_vk_batch_recording = false;
    p36_vk_batch_count = 0;
    return 1;
}

static void p36_vk_ring_activate_unlocked(uint32_t idx) {
    p36_vk_ring_cur = idx;
    p36_vk.command_buffer = p36_vk_ring[idx].cb;
    p36_vk.fence = p36_vk_ring[idx].fence;
    p36_vk.descriptor_pool = p36_vk_ring[idx].pool;
}

/* Retire every parked tensor whose last GPU reference has completed.
 * Runs on each eager-submit fence wait so scratch memory drains as the
 * ring advances instead of piling up until the next full flush. */
static void p36_vk_retired_sweep_unlocked(void) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < p36_vk_retired_n; i++) {
        p36_gpu_tensor *t = p36_vk_retired[i];
        if (t->last_use_seq > p36_vk_completed_seq) p36_vk_retired[kept++] = t;
        else p36_vk_tensor_retire_unlocked(t);
    }
    p36_vk_retired_n = kept;
}

static int p36_vk_submit_eager_unlocked(void) {
    uint64_t t0 = p36_vk_now_ns();
    int ok = p36_vk_submit_current_unlocked(false);
    if (ok) {
        uint32_t next = (p36_vk_ring_cur + 1u) % P36_VK_CB_RING;
        ok = p36_vk_slot_wait_unlocked(&p36_vk_ring[next]);
        p36_vk_ring_activate_unlocked(next);
        p36_vk_retired_sweep_unlocked();
    }
    p36_vk_prof_op_add("submit_eager", 0, 1, 0, 0, p36_vk_now_ns() - t0, 0, 0);
    return ok;
}

static int p36_vk_flush_unlocked(void) {
    bool had_work = p36_vk_batch_recording;
    int ok = p36_vk_submit_current_unlocked(true);
    uint64_t t0 = p36_vk_now_ns();
    for (uint32_t i = 0; i < P36_VK_CB_RING; i++) {
        had_work = had_work || p36_vk_ring[i].pending;
        if (!p36_vk_slot_wait_unlocked(&p36_vk_ring[i])) ok = 0;
    }
    if (had_work) {
        p36_vk_prof_flushes++;
        p36_vk_flush_total++;
        {
            uint64_t submit_ns = p36_vk_now_ns() - t0;
            p36_vk_prof_submit_ns += submit_ns;
            p36_vk_prof_op_add(p36_vk_flush_reason, 0, 1, 0, 0, submit_ns, 0, 0);
        }
        if (ok) p36_vk_prof_kernel_collect_unlocked();
        p36_vk_prof_query_count = 0;
        p36_vk_batch_written_n = 0;
        p36_vk_batch_dirty_all = false;
        p36_vk_hazard_reset();
    }
    p36_vk_retired_sweep_unlocked();
    return ok;
}

static int p36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out) {
    p36_vk_cb_slot *slot = &p36_vk_ring[p36_vk_ring_cur];
    if (p36_vk_device_lost) return 0;
    if (!slot->cb) {
        VkCommandBufferAllocateInfo cai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = p36_vk.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (vkAllocateCommandBuffers(p36_vk.device, &cai, &slot->cb) != VK_SUCCESS) return 0;
    }
    if (!slot->fence) {
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(p36_vk.device, &fci, NULL, &slot->fence) != VK_SUCCESS) return 0;
    }
    p36_vk.command_buffer = slot->cb;
    p36_vk.fence = slot->fence;
    if (!p36_vk_batch_recording) {
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult rc = vkBeginCommandBuffer(p36_vk.command_buffer, &bi);
        if (rc != VK_SUCCESS) return p36_vk_ok(rc, "vkBeginCommandBuffer");
        p36_vk_batch_recording = true;
    }
    if (cmd_out) *cmd_out = p36_vk.command_buffer;
    return 1;
}

static void p36_vk_note_written_tensor(p36_gpu_tensor *tensor) {
    p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, NULL);
    if (!root) return;
    uint32_t i = 0;
    while (i < p36_vk_batch_written_n && p36_vk_batch_written[i] != root) i++;
    if (i < p36_vk_batch_written_n) return;
    if (p36_vk_batch_written_n == P36_VK_WRITTEN_CAP) p36_vk_batch_dirty_all = true;
    else p36_vk_batch_written[p36_vk_batch_written_n++] = root;
}

static int p36_vk_flush_reason_unlocked(const char *reason) {
    const char *prev = p36_vk_flush_reason;
    p36_vk_flush_reason = reason ? reason : "submit_wait_other";
    int ok = p36_vk_flush_unlocked();
    p36_vk_flush_reason = prev;
    return ok;
}

static void p36_vk_weight_cache_free_unlocked(void) {
    p36_vk_weight *w = p36_vk.weights;
    p36_vk_packed_weight *pw = p36_vk.packed_weights;
    p36_vk_arena_block *b = p36_vk.arena;
    p36_vk_stream_cache_release_unlocked(true);
    while (w) {
        p36_vk_weight *next = w->next;
        p36_vk_tensor_free_unlocked(w->tensor);
        free(w);
        w = next;
    }
    p36_vk.weights = NULL;
    while (pw) {
        p36_vk_packed_weight *next = pw->next;
        p36_vk_tensor_free_unlocked(pw->tensor);
        free(pw);
        pw = next;
    }
    p36_vk.packed_weights = NULL;
    while (b) {
        p36_vk_arena_block *next = b->next;
        p36_vk_tensor_free_unlocked(b->base);
        free(b);
        b = next;
    }
    while (p36_vk_dense_recycled) {
        p36_gpu_tensor *next = p36_vk_dense_recycled->recycled_next;
        p36_vk_tensor_free_unlocked(p36_vk_dense_recycled);
        p36_vk_dense_recycled = next;
    }
    p36_vk.arena = NULL;
    p36_vk.arena_plan = NULL;
    free(p36_vk.arena_plan_sizes);
    p36_vk.arena_plan_sizes = NULL;
    p36_vk.arena_plan_count = 0;
    p36_vk.arena_plan_next = 0;
}

/* Dense Qwen3.8-27B IQ4_XS is larger than the BC-250's practical resident
 * budget. Keep the normal cache for smaller models, but allow the launcher
 * to bound it for oversized dense models. Eviction happens only after a
 * full queue drain, because cached weights may still be referenced by the
 * command buffer even when the host has finished recording the operation. */
static uint64_t p36_vk_env_gib(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long gib = strtoull(env, &end, 10);
    if (end == env || *end != '\0' || gib == 0 ||
        gib > (unsigned long long)(UINT64_MAX >> 30)) return 0;
    return (uint64_t)gib << 30;
}

static uint64_t p36_vk_dense_weight_cache_limit_unlocked(void) {
    if (!p36_gpu_dense_model) return 0;
    return p36_vk_env_gib("P36_VK_DENSE_WEIGHT_CACHE_GIB");
}

static uint64_t p36_vk_dense_weight_pin_limit_unlocked(uint64_t cache_limit) {
    uint64_t configured = p36_vk_env_gib("P36_VK_DENSE_WEIGHT_PIN_GIB");
    if (configured) return configured < cache_limit ? configured : cache_limit;
    return cache_limit > (2ull << 30) ? cache_limit - (2ull << 30) : 0;
}

static uint64_t p36_vk_weight_cache_bytes_unlocked(void) {
    uint64_t total = 0;
    for (p36_vk_weight *w = p36_vk.weights; w; w = w->next) {
        if (UINT64_MAX - total < w->bytes) return UINT64_MAX;
        total += w->bytes;
    }
    for (p36_vk_packed_weight *w = p36_vk.packed_weights; w; w = w->next) {
        uint64_t bytes = w->tensor ? w->tensor->bytes : 0;
        if (UINT64_MAX - total < bytes) return UINT64_MAX;
        total += bytes;
    }
    return total;
}

static uint64_t p36_vk_weight_cache_pinned_bytes_unlocked(void) {
    uint64_t total = 0;
    for (p36_vk_weight *w = p36_vk.weights; w; w = w->next) {
        if (!w->pinned) continue;
        if (UINT64_MAX - total < w->bytes) return UINT64_MAX;
        total += w->bytes;
    }
    return total;
}

/* Remove the least-recently-used evictable weight. Bounded weights use their
 * own allocations, so unlike arena views these frees return memory that can
 * be reused by a later layer without rebuilding the retained prefix. */
static bool p36_vk_weight_cache_evict_one_unlocked(void) {
    p36_vk_weight *victim = NULL;
    p36_vk_weight *victim_prev = NULL;
    p36_vk_weight *prev = NULL;
    for (p36_vk_weight *w = p36_vk.weights; w; prev = w, w = w->next) {
        if (!w->pinned) {
            victim = w;
            victim_prev = prev;
        }
    }
    if (!victim) return false;
    if (victim_prev) victim_prev->next = victim->next;
    else p36_vk.weights = victim->next;
    p36_gpu_tensor *tensor = victim->tensor;
    if (tensor && !tensor->owner) {
        p36_gpu_live_bytes -= tensor->bytes;
        tensor->bytes = 0;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        tensor->write_gen = 0;
        tensor->recycled_next = p36_vk_dense_recycled;
        p36_vk_dense_recycled = tensor;
    } else {
        p36_vk_tensor_free_unlocked(tensor);
    }
    free(victim);
    return true;
}

/* Reuse the Vulkan allocations for streamed weights. Repeated vkAllocateMemory
 * and vkFreeMemory calls were measurable once the cache began rotating every
 * decode pass; copied bytes are overwritten completely on reuse. */
static p36_gpu_tensor *p36_vk_dense_stream_alloc_unlocked(uint64_t bytes) {
    const uint64_t need = p36_round_up_u64(bytes ? bytes : 4u, 4u);
    p36_gpu_tensor *best = NULL;
    p36_gpu_tensor *best_prev = NULL;
    p36_gpu_tensor *prev = NULL;
    for (p36_gpu_tensor *t = p36_vk_dense_recycled; t; prev = t, t = t->recycled_next) {
        if (t->alloc_bytes < need) continue;
        if (!best || t->alloc_bytes < best->alloc_bytes) {
            best = t;
            best_prev = prev;
        }
    }
    if (best) {
        if (best_prev) best_prev->recycled_next = best->recycled_next;
        else p36_vk_dense_recycled = best->recycled_next;
        best->recycled_next = NULL;
        best->bytes = bytes;
        best->owner = NULL;
        best->offset = 0;
        best->gpu_written = false;
        best->last_use_seq = 0;
        best->write_gen = 0;
        p36_gpu_live_bytes += bytes;
        if (p36_gpu_live_bytes > p36_gpu_peak_bytes) p36_gpu_peak_bytes = p36_gpu_live_bytes;
        return best;
    }
    return p36_vk_tensor_alloc_kind_unlocked(bytes, true, false);
}

/* Suballocate weight bytes from the arena; new blocks prefer the
 * device-local heap and fall back to host-cached memory when it is full,
 * like the old per-weight allocations did.  256-byte alignment satisfies
 * any minStorageBufferOffsetAlignment the spec allows. */
enum { P36_VK_ARENA_BLOCK_BYTES = 512u << 20 };

