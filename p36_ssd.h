#ifndef P36_SSD_H
#define P36_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} p36_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} p36_ssd_cache_plan;

bool p36_parse_gib_arg(const char *s, uint64_t *bytes);
bool p36_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t p36_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
uint32_t p36_ssd_shrink_cache_experts(uint32_t experts);
bool p36_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint32_t            percent,
                             uint64_t            model_limit_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             p36_ssd_cache_plan *out);

bool p36_ssd_memory_lock_acquire(p36_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void p36_ssd_memory_lock_release(p36_ssd_memory_lock *lock);

#endif
