static bool p36_vk_arena_plan_unlocked(const uint64_t *sizes, uint32_t count) {
    const uint64_t cap = 2ull << 30;
    uint64_t *blocks = calloc(count ? count : 1u, sizeof(*blocks));
    uint32_t nblocks = 0;
    uint64_t used = 0;

    if (!blocks) return false;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t need = p36_round_up_u64(sizes[i] ? sizes[i] : 4u, 256u);
        if (used && (need > cap || used > cap - need)) {
            blocks[nblocks++] = used;
            used = 0;
        }
        if (need > cap) blocks[nblocks++] = need;
        else used += need;
    }
    if (used) blocks[nblocks++] = used;

    p36_vk.arena_plan_sizes = blocks;
    p36_vk.arena_plan_count = nblocks;
    p36_vk.arena_plan_next = 0;
    return true;
}

static p36_gpu_tensor *p36_vk_arena_alloc_unlocked(uint64_t bytes) {
    const uint64_t need = p36_round_up_u64(bytes ? bytes : 4, 256);
    p36_vk_arena_block *block = NULL;
    uint64_t best_left = UINT64_MAX;
    while (p36_vk.arena_plan &&
           p36_vk.arena_plan->base->alloc_bytes - p36_vk.arena_plan->used < need) {
        p36_vk.arena_plan = NULL;
    }
    if (!p36_vk.arena_plan && p36_vk.arena_plan_next < p36_vk.arena_plan_count) {
        uint64_t block_bytes = p36_vk.arena_plan_sizes[p36_vk.arena_plan_next++];
        p36_gpu_tensor *base = p36_vk_tensor_alloc_kind_unlocked(block_bytes, true, false);
        p36_vk_arena_block *planned = base ? calloc(1, sizeof(*planned)) : NULL;
        if (planned) {
            base->weight_only = true;
            planned->base = base;
            planned->next = p36_vk.arena;
            p36_vk.arena = planned;
            p36_vk.arena_plan = planned;
        } else {
            p36_vk_tensor_free_unlocked(base);
            free(p36_vk.arena_plan_sizes);
            p36_vk.arena_plan_sizes = NULL;
            p36_vk.arena_plan_count = 0;
            p36_vk.arena_plan_next = 0;
        }
    }
    if (p36_vk.arena_plan) block = p36_vk.arena_plan;
    if (!block) {
        for (p36_vk_arena_block *b = p36_vk.arena; b; b = b->next) {
            uint64_t free_bytes = b->base->alloc_bytes - b->used;
            if (free_bytes >= need && free_bytes - need < best_left) {
                block = b;
                best_left = free_bytes - need;
            }
        }
    }
    if (!block) {
        uint64_t block_bytes = need > P36_VK_ARENA_BLOCK_BYTES ? need : P36_VK_ARENA_BLOCK_BYTES;
        p36_gpu_tensor *base = p36_vk_tensor_alloc_kind_unlocked(block_bytes, true, false);
        if (!base && block_bytes > need) base = p36_vk_tensor_alloc_kind_unlocked(need, true, false);
        if (!base) base = p36_vk_tensor_alloc_kind_unlocked(need, false, false);
        if (!base) return NULL;
        block = calloc(1, sizeof(*block));
        if (!block) {
            p36_vk_tensor_free_unlocked(base);
            return NULL;
        }
        base->weight_only = true;
        block->base = base;
        block->next = p36_vk.arena;
        p36_vk.arena = block;
    }
    {
        p36_gpu_tensor *view = p36_gpu_tensor_view(block->base, block->used, bytes);
        if (view) block->used += need;
        return view;
    }
}

/* Writes into the host-visible weight arena go to write-combined memory and
 * top out well below the SSD on a single thread, so bulk staging (prewarm)
 * splits the copy across the host pool.  Inference-time lazy staging keeps
 * the plain memcpy: it runs under p36_vk_mu in paths where the pool may be
 * busy with host compute. */
static bool p36_vk_weight_copy_parallel = false;

typedef struct {
    unsigned char *dst;
    const unsigned char *src;
    uint64_t bytes;
} p36_vk_weight_copy_ctx;

static void p36_vk_weight_copy_rows(void *ctx, uint64_t row0, uint64_t row1) {
    const p36_vk_weight_copy_ctx *c = ctx;
    uint64_t off0 = row0 << 20, off1 = row1 << 20;
    if (off1 > c->bytes) off1 = c->bytes;
    if (off0 < off1) memcpy(c->dst + off0, c->src + off0, (size_t)(off1 - off0));
}

static p36_gpu_tensor *p36_vk_weight_get_unlocked(const void *source, uint64_t bytes) {
    /* Move hits to the head: the cache is a linear list scanned per matvec
     * and a token touches the same few hundred weights in layer order. */
    for (p36_vk_weight *w = p36_vk.weights, *prev = NULL; w; prev = w, w = w->next) {
        if (w->source == source && w->bytes == bytes) {
            if (prev) {
                prev->next = w->next;
                w->next = p36_vk.weights;
                p36_vk.weights = w;
            }
            return w->tensor;
        }
    }

    const uint64_t limit = p36_vk_dense_weight_cache_limit_unlocked();
    bool pinned = false;
    if (limit) {
        const uint64_t pin_limit = p36_vk_dense_weight_pin_limit_unlocked(limit);
        const uint64_t pinned_bytes = p36_vk_weight_cache_pinned_bytes_unlocked();
        pinned = pinned_bytes <= pin_limit && bytes <= pin_limit - pinned_bytes;
        if (!pinned) {
            if (bytes > limit) return NULL;
            uint64_t cached = p36_vk_weight_cache_bytes_unlocked();
            if (cached > limit - bytes) {
                if (!p36_vk_flush_reason_unlocked("dense_weight_cache_evict")) return NULL;
                do {
                    if (!p36_vk_weight_cache_evict_one_unlocked()) return NULL;
                    cached = p36_vk_weight_cache_bytes_unlocked();
                } while (cached > limit - bytes);
            }
        }
    }

    /* Retained prefix weights use the arena's low-allocation path. Streamed
     * weights are individually reclaimable so the prefix survives eviction. */
    p36_gpu_tensor *tensor = pinned ? p36_vk_arena_alloc_unlocked(bytes) :
                                      (limit ? p36_vk_dense_stream_alloc_unlocked(bytes) :
                                               p36_vk_arena_alloc_unlocked(bytes));
    if (!tensor) return NULL;
    {
        uint64_t offset = 0;
        p36_gpu_tensor *root = p36_gpu_tensor_root(tensor, &offset);
        bool copy_parallel = p36_vk_weight_copy_parallel ||
                             (limit && bytes >= (1u << 20) &&
                              p36_vk_env_default_on("P36_VK_DENSE_WEIGHT_COPY_PARALLEL"));
        if (copy_parallel) {
            p36_vk_weight_copy_ctx c = { root->data + offset, source, bytes };
            p36_gpu_parallel_for_rows((bytes + ((1u << 20) - 1)) >> 20, 8, p36_vk_weight_copy_rows, &c);
        } else {
            memcpy(root->data + offset, source, (size_t)bytes);
        }
    }

    p36_vk_weight *w = calloc(1, sizeof(*w));
    if (!w) {
        p36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    w->source = source;
    w->bytes = bytes;
    w->tensor = tensor;
    w->pinned = pinned;
    w->next = p36_vk.weights;
    p36_vk.weights = w;
    return tensor;
}

typedef enum {
    P36_VK_STREAM_PART_NONE = 0,
    P36_VK_STREAM_PART_GATE,
    P36_VK_STREAM_PART_UP,
    P36_VK_STREAM_PART_DOWN,
} p36_vk_stream_part;

static bool p36_vk_stream_table_valid(const p36_gpu_stream_expert_table *t) {
    return t && t->model_map && t->model_size != 0 &&
           t->n_total_expert != 0 && t->n_total_expert <= 256 &&
           t->gate_expert_bytes != 0 &&
           t->up_expert_bytes != 0 &&
           t->down_expert_bytes != 0;
}

static uint64_t p36_vk_stream_table_expert_bytes(const p36_gpu_stream_expert_table *t) {
    if (!t) return 0;
    if (t->gate_expert_bytes > UINT64_MAX - t->up_expert_bytes) return 0;
    uint64_t sum = t->gate_expert_bytes + t->up_expert_bytes;
    if (sum > UINT64_MAX - t->down_expert_bytes) return 0;
    return sum + t->down_expert_bytes;
}

static bool p36_vk_stream_layout_matches_unlocked(const p36_gpu_stream_expert_table *t) {
    if (p36_vk_stream.cap == 0) return true;
    return p36_vk_stream.gate_expert_bytes == t->gate_expert_bytes &&
           p36_vk_stream.up_expert_bytes == t->up_expert_bytes &&
           p36_vk_stream.down_expert_bytes == t->down_expert_bytes &&
           p36_vk_stream.gate_type == t->gate_type &&
           p36_vk_stream.up_type == t->up_type &&
           p36_vk_stream.down_type == t->down_type &&
           p36_vk_stream.has_gate_scales == t->has_gate_scales &&
           p36_vk_stream.has_up_scales == t->has_up_scales &&
           p36_vk_stream.has_down_scales == t->has_down_scales;
}

static bool p36_vk_stream_cache_init_unlocked(const p36_gpu_stream_expert_table *t) {
    if (!p36_vk_stream.enabled || p36_vk_stream.budget == 0 || !p36_vk_stream_table_valid(t)) return false;
    uint64_t expert_bytes = p36_vk_stream_table_expert_bytes(t);
    if (expert_bytes == 0) return false;
    if (p36_vk_stream.configured_expert_bytes != 0 &&
        p36_vk_stream.configured_expert_bytes != expert_bytes) {
        return false;
    }
    if (p36_vk_stream.cap != 0) return p36_vk_stream_layout_matches_unlocked(t);

    uint32_t cap = p36_vk_stream.runtime_cap ? p36_vk_stream.runtime_cap : p36_vk_stream.budget;
    while (cap != 0) {
        uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0, scale_bytes = 0;
        bool sizes_ok = p36_u64_mul_ok(cap, t->gate_expert_bytes, &gate_bytes) &&
                        p36_u64_mul_ok(cap, t->up_expert_bytes, &up_bytes) &&
                        p36_u64_mul_ok(cap, t->down_expert_bytes, &down_bytes) &&
                        p36_u64_mul_ok(cap, sizeof(float), &scale_bytes);
        if (sizes_ok) {
            p36_vk_stream.entry = calloc(cap, sizeof(p36_vk_stream.entry[0]));
            if (p36_vk_stream.entry)
                p36_vk_stream.gate = p36_vk_tensor_alloc_kind_unlocked(gate_bytes, true, true);
            if (p36_vk_stream.gate)
                p36_vk_stream.up = p36_vk_tensor_alloc_kind_unlocked(up_bytes, true, true);
            if (p36_vk_stream.up)
                p36_vk_stream.down = p36_vk_tensor_alloc_kind_unlocked(down_bytes, true, true);
            if (p36_vk_stream.down && t->has_gate_scales)
                p36_vk_stream.gate_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
            if (p36_vk_stream.down && (!t->has_gate_scales || p36_vk_stream.gate_scales) && t->has_up_scales)
                p36_vk_stream.up_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
            if (p36_vk_stream.down && (!t->has_gate_scales || p36_vk_stream.gate_scales) &&
                (!t->has_up_scales || p36_vk_stream.up_scales) && t->has_down_scales)
                p36_vk_stream.down_scales = p36_vk_tensor_alloc_kind_unlocked(scale_bytes, true, true);
        }
        if (sizes_ok && p36_vk_stream.entry && p36_vk_stream.gate &&
            p36_vk_stream.up && p36_vk_stream.down &&
            (!t->has_gate_scales || p36_vk_stream.gate_scales) &&
            (!t->has_up_scales || p36_vk_stream.up_scales) &&
            (!t->has_down_scales || p36_vk_stream.down_scales)) {
            break;
        }

        p36_vk_stream_cache_release_unlocked(true);
        uint32_t next = p36_ssd_shrink_cache_experts(cap);
        p36_vk_stream.runtime_cap = next;
        p36_vk_stream.allocation_failures++;
        fprintf(stderr,
                "p36: Vulkan streaming expert cache allocation failed at %u experts / %.2f GiB\n",
                cap, (double)cap * (double)expert_bytes / 1073741824.0);
        if (next != 0) {
            fprintf(stderr,
                    "p36:   shrinking resident cache margin by %u experts / %.2f GiB; runtime cache cap now %u experts\n",
                    cap - next,
                    (double)(cap - next) * (double)expert_bytes / 1073741824.0,
                    next);
        }
        cap = next;
    }
    if (cap == 0) return false;

    p36_vk_stream.cap = cap;
    p36_vk_stream.gate_expert_bytes = t->gate_expert_bytes;
    p36_vk_stream.up_expert_bytes = t->up_expert_bytes;
    p36_vk_stream.down_expert_bytes = t->down_expert_bytes;
    p36_vk_stream.gate_type = t->gate_type;
    p36_vk_stream.up_type = t->up_type;
    p36_vk_stream.down_type = t->down_type;
    p36_vk_stream.has_gate_scales = t->has_gate_scales;
    p36_vk_stream.has_up_scales = t->has_up_scales;
    p36_vk_stream.has_down_scales = t->has_down_scales;
    {
        const char *env = getenv("P36_VK_STREAM_CACHE_INDEX");
        p36_vk_stream.indexed = !env || !env[0] || env[0] != '0';
    }
    return true;
}

static bool p36_vk_stream_cache_entry_matches(const p36_vk_stream_entry *e,
                                               const p36_gpu_stream_expert_table *t,
                                               uint32_t expert) {
    return e && e->valid && e->layer == t->layer && e->expert == expert &&
           e->gate_offset == t->gate_offset &&
           e->up_offset == t->up_offset &&
           e->down_offset == t->down_offset;
}

static int p36_vk_stream_cache_find_unlocked(const p36_gpu_stream_expert_table *t, uint32_t expert) {
    if (p36_vk_stream.indexed && t->layer < P36_VK_STREAM_MAX_LAYERS &&
        expert < P36_VK_STREAM_MAX_EXPERTS) {
        uint32_t mapped = p36_vk_stream.slot_of[t->layer][expert];
        p36_vk_stream.lookup_steps++;
        if (mapped == 0) return -1;
        uint32_t slot = mapped - 1u;
        if (slot < p36_vk_stream.cap &&
            p36_vk_stream_cache_entry_matches(&p36_vk_stream.entry[slot], t, expert)) {
            return (int)slot;
        }
        p36_vk_stream.slot_of[t->layer][expert] = 0;
        return -1;
    }
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
        p36_vk_stream_entry *e = &p36_vk_stream.entry[i];
        p36_vk_stream.lookup_steps++;
        if (p36_vk_stream_cache_entry_matches(e, t, expert)) return (int)i;
    }
    return -1;
}

static void p36_vk_stream_cache_unmap_slot_unlocked(uint32_t slot) {
    if (!p36_vk_stream.indexed || slot >= p36_vk_stream.cap) return;
    p36_vk_stream_entry *e = &p36_vk_stream.entry[slot];
    if (!e->valid || e->layer >= P36_VK_STREAM_MAX_LAYERS ||
        e->expert >= P36_VK_STREAM_MAX_EXPERTS) return;
    uint32_t *mapped = &p36_vk_stream.slot_of[e->layer][e->expert];
    if (*mapped == slot + 1u) *mapped = 0;
}

static uint32_t p36_vk_stream_route_hotness_unlocked(uint32_t layer, uint32_t expert) {
    if (layer >= P36_VK_STREAM_MAX_LAYERS || expert >= P36_VK_STREAM_MAX_EXPERTS) return 0;
    uint32_t hotness = p36_vk_stream.route_hotness[layer][expert];
    uint32_t bias = p36_vk_stream.route_bias[layer][expert];
    return hotness > UINT32_MAX - bias ? UINT32_MAX : hotness + bias;
}

static void p36_vk_stream_note_hotness_unlocked(uint32_t layer, uint32_t expert,
                                                uint32_t amount) {
    if (layer >= P36_VK_STREAM_MAX_LAYERS || expert >= P36_VK_STREAM_MAX_EXPERTS || amount == 0) return;
    uint32_t *hotness = &p36_vk_stream.route_hotness[layer][expert];
    *hotness = *hotness > UINT32_MAX - amount ? UINT32_MAX : *hotness + amount;
}

static void p36_vk_stream_decay_hotness_unlocked(void) {
    while (p36_vk_stream.route_tokens - p36_vk_stream.hotness_decay_token >= 16u) {
        for (uint32_t layer = 0; layer < P36_VK_STREAM_MAX_LAYERS; layer++) {
            for (uint32_t expert = 0; expert < P36_VK_STREAM_MAX_EXPERTS; expert++) {
                p36_vk_stream.route_hotness[layer][expert] >>= 1;
            }
        }
        p36_vk_stream.hotness_decay_token += 16u;
    }
}

static bool p36_vk_stream_protect_selected(void) {
    return p36_vk_env_default_on("P36_VK_STREAM_PROTECT_SELECTED");
}

static bool p36_vk_stream_evict_past_layers_first(void) {
    const char *env = getenv("P36_VK_STREAM_EVICT_PAST_LAYERS_FIRST");
    if (env) return !env[0] || env[0] != '0';
    return p36_vk_stream.has_route_bias;
}

static bool p36_vk_stream_victim_better(uint32_t slot, uint32_t best) {
    uint32_t hotness = p36_vk_stream_route_hotness_unlocked(
        p36_vk_stream.entry[slot].layer, p36_vk_stream.entry[slot].expert);
    uint32_t best_hotness = best == UINT32_MAX ? UINT32_MAX :
        p36_vk_stream_route_hotness_unlocked(
            p36_vk_stream.entry[best].layer, p36_vk_stream.entry[best].expert);
    return best == UINT32_MAX || hotness < best_hotness ||
           (hotness == best_hotness &&
            p36_vk_stream.entry[slot].last_used <
            p36_vk_stream.entry[best].last_used);
}

static uint32_t p36_vk_stream_cache_pick_slot_unlocked(uint32_t protect_layer,
                                                       const bool *protect) {
    uint32_t best = UINT32_MAX;
    /* Prefill will not revisit past layers. Decode will, so it keeps the
     * global frequency/LRU policy. */
    bool prefer_past = p36_vk_stream.phase == 0 &&
                       p36_vk_stream_evict_past_layers_first();
    for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[i];
        if (entry->loading) continue;
        if (!entry->valid) return i;
        if (protect && entry->layer == protect_layer &&
            entry->expert < P36_VK_STREAM_MAX_EXPERTS &&
            protect[entry->expert]) {
            continue;
        }
        if (prefer_past && entry->layer > protect_layer) continue;
        if (p36_vk_stream_victim_better(i, best)) best = i;
    }
    if (best == UINT32_MAX && prefer_past) {
        for (uint32_t i = 0; i < p36_vk_stream.cap; i++) {
            p36_vk_stream_entry *entry = &p36_vk_stream.entry[i];
            if (entry->loading || !entry->valid) continue;
            if (protect && entry->layer == protect_layer &&
                entry->expert < P36_VK_STREAM_MAX_EXPERTS &&
                protect[entry->expert]) {
                continue;
            }
            if (p36_vk_stream_victim_better(i, best)) best = i;
        }
    }
    if (best != UINT32_MAX) p36_vk_stream.evictions++;
    return best;
}

static bool p36_vk_stream_cache_copy_scale_unlocked(p36_gpu_tensor *dst,
                                                    uint32_t slot,
                                                    const p36_gpu_stream_expert_table *t,
                                                    uint64_t scale_offset,
                                                    uint32_t expert) {
    if (!dst) return true;
    float value = 1.0f;
    if (!p36_vk_stream_copy_model_range(t->model_map,
                                        t->model_size,
                                        scale_offset + (uint64_t)expert * sizeof(float),
                                        &value,
                                        sizeof(value))) {
        return false;
    }
    memcpy(dst->data + (uint64_t)slot * sizeof(float), &value, sizeof(value));
    return true;
}

/* Copy one expert's gate/up/down (+scales) into its cache slot.  Pure data
 * movement into disjoint slot ranges, so the miss batch fans out across the
 * host pool while the prepare pass holds p36_vk_mu: parallel preads keep the
 * SSD queue full where one blocking read per expert leaves it idle. */
static bool p36_vk_stream_slot_fetch(uint32_t slot,
                                     const p36_gpu_stream_expert_table *t,
                                     uint32_t expert) {
    unsigned char *gate_dst = p36_vk_stream.gate->data + (uint64_t)slot * p36_vk_stream.gate_expert_bytes;
    unsigned char *up_dst = p36_vk_stream.up->data + (uint64_t)slot * p36_vk_stream.up_expert_bytes;
    unsigned char *down_dst = p36_vk_stream.down->data + (uint64_t)slot * p36_vk_stream.down_expert_bytes;
    return p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->gate_offset + (uint64_t)expert * t->gate_expert_bytes,
                                          gate_dst,
                                          t->gate_expert_bytes) &&
           p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->up_offset + (uint64_t)expert * t->up_expert_bytes,
                                          up_dst,
                                          t->up_expert_bytes) &&
           p36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->down_offset + (uint64_t)expert * t->down_expert_bytes,
                                          down_dst,
                                          t->down_expert_bytes) &&
           (!t->has_gate_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.gate_scales, slot, t, t->gate_scales_offset, expert)) &&
           (!t->has_up_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.up_scales, slot, t, t->up_scales_offset, expert)) &&
           (!t->has_down_scales || p36_vk_stream_cache_copy_scale_unlocked(p36_vk_stream.down_scales, slot, t, t->down_scales_offset, expert));
}

typedef struct {
    uint32_t slot;
    uint32_t expert;
    bool ok;
} p36_vk_stream_miss;

typedef struct {
    const p36_gpu_stream_expert_table *t;
    p36_vk_stream_miss *miss;
} p36_vk_stream_fetch_ctx;

static void p36_vk_stream_fetch_rows(void *ctx, uint64_t row0, uint64_t row1) {
    p36_vk_stream_fetch_ctx *c = ctx;
    for (uint64_t i = row0; i < row1; i++) {
        c->miss[i].ok = p36_vk_stream_slot_fetch(c->miss[i].slot, c->t, c->miss[i].expert);
    }
}

/* Hit-or-reserve: hits bump recency, misses claim a slot and record the
 * pending fetch.  The reserved entry carries its final identity immediately
 * so duplicate selections in the same list hit it. */
static bool p36_vk_stream_cache_prepare_one_unlocked(const p36_gpu_stream_expert_table *t,
                                                     uint32_t expert,
                                                     uint32_t priority,
                                                     const bool *protect,
                                                     uint32_t *slot_map,
                                                     p36_vk_stream_miss *miss,
                                                     uint32_t *n_miss,
                                                     bool *need_flush) {
    if (expert >= t->n_total_expert) return false;
    p36_vk_stream_note_hotness_unlocked(t->layer, expert, priority ? priority : 1u);
    int slot = p36_vk_stream_cache_find_unlocked(t, expert);
    if (slot >= 0) {
        p36_vk_stream.hits++;
    } else {
        if (*n_miss >= 256u) return false;
        p36_vk_stream.misses++;
        uint32_t pick = p36_vk_stream_cache_pick_slot_unlocked(t->layer, protect);
        if (pick == UINT32_MAX) return false;
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[pick];
        if (entry->valid) *need_flush = true;
        p36_vk_stream_cache_unmap_slot_unlocked(pick);
        entry->valid = true;
        entry->loading = true;
        entry->layer = t->layer;
        entry->expert = expert;
        entry->gate_offset = t->gate_offset;
        entry->up_offset = t->up_offset;
        entry->down_offset = t->down_offset;
        if (p36_vk_stream.indexed && t->layer < P36_VK_STREAM_MAX_LAYERS &&
            expert < P36_VK_STREAM_MAX_EXPERTS) {
            p36_vk_stream.slot_of[t->layer][expert] = pick + 1u;
        }
        miss[*n_miss] = (p36_vk_stream_miss) { pick, expert, false };
        (*n_miss)++;
        slot = (int)pick;
    }

    p36_vk_stream_entry *entry = &p36_vk_stream.entry[slot];
    entry->last_used = ++p36_vk_stream.clock;
    if (slot_map) slot_map[expert] = (uint32_t)slot;
    return true;
}

/* Flush once if any reserved slot evicts live data, then fetch the whole
 * miss batch in parallel and finalize the entries. */
static bool p36_vk_stream_cache_finish_unlocked(const p36_gpu_stream_expert_table *t,
                                                p36_vk_stream_miss *miss,
                                                uint32_t n_miss,
                                                bool need_flush,
                                                bool prepared) {
    bool ok = prepared;
    if (n_miss == 0) return ok;
    if (ok && need_flush && !p36_vk_flush_reason_unlocked("submit_wait_stream_cache_evict")) ok = false;
    if (ok) {
        p36_vk_stream_fetch_ctx ctx = { t, miss };
        p36_gpu_parallel_for_rows(n_miss, 2, p36_vk_stream_fetch_rows, &ctx);
    }
    for (uint32_t i = 0; i < n_miss; i++) {
        p36_vk_stream_entry *entry = &p36_vk_stream.entry[miss[i].slot];
        entry->loading = false;
        if (ok && miss[i].ok) {
            p36_vk_stream.loads++;
        } else {
            p36_vk_stream_cache_unmap_slot_unlocked(miss[i].slot);
            entry->valid = false;
            ok = false;
        }
    }
    return ok;
}

static bool p36_vk_stream_cache_prepare_u32(const p36_gpu_stream_expert_table *t,
                                            const uint32_t *selected,
                                            uint64_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || !slot_map || n_selected == 0) return false;
    for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;

    bool seen[256] = {0};
    uint32_t unique = 0;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= 256) return false;
        if (!seen[expert]) {
            seen[expert] = true;
            unique++;
        }
    }
    pthread_mutex_lock(&p36_vk_mu);
    bool ok = p36_vk_stream_cache_init_unlocked(t);
    if (ok && unique > p36_vk_stream.cap) ok = false;
    if (ok) p36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    p36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    const bool *protect = p36_vk_stream_protect_selected() ? seen : NULL;
    for (uint64_t i = 0; ok && i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= t->n_total_expert) {
            ok = false;
            break;
        }
        ok = p36_vk_stream_cache_prepare_one_unlocked(t, expert, 1, protect, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = p36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

static uint32_t p36_vk_stream_cache_cap(const p36_gpu_stream_expert_table *t) {
    pthread_mutex_lock(&p36_vk_mu);
    uint32_t cap = p36_vk_stream_cache_init_unlocked(t) ? p36_vk_stream.cap : 0;
    pthread_mutex_unlock(&p36_vk_mu);
    return cap;
}

static void p36_vk_stream_note_routes(const p36_gpu_stream_expert_table *t,
                                      const uint32_t *selected,
                                      uint64_t n_selected) {
    pthread_mutex_lock(&p36_vk_mu);
    p36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    for (uint64_t i = 0; i < n_selected; i++) {
        if (selected[i] < t->n_total_expert) {
            p36_vk_stream_note_hotness_unlocked(t->layer, selected[i], 1u);
        }
    }
    pthread_mutex_unlock(&p36_vk_mu);
}

static bool p36_vk_stream_cache_prepare_i32(const p36_gpu_stream_expert_table *t,
                                            const int32_t *selected,
                                            const uint32_t *priorities,
                                            uint32_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || n_selected == 0) return false;
    if (slot_map) for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;
    bool selected_set[256] = {0};
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= t->n_total_expert)
            return false;
        selected_set[(uint32_t)selected[i]] = true;
    }

    pthread_mutex_lock(&p36_vk_mu);
    bool ok = p36_vk_stream_cache_init_unlocked(t);
    p36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    const bool *protect = p36_vk_stream_protect_selected() ? selected_set : NULL;
    for (uint32_t i = 0; ok && i < n_selected; i++) {
        uint32_t priority = priorities ? priorities[i] : 1u;
        ok = p36_vk_stream_cache_prepare_one_unlocked(t, (uint32_t)selected[i], priority,
                                                      protect, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = p36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&p36_vk_mu);
    return ok;
}

