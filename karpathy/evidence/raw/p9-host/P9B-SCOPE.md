# P9(b) — replayable decode command buffer: scope, blockers, prototype plan

2026-09-24, audit §P9(b).  Design/scoping only; nothing here is built.  Read
against `q36_vulkan.c` at `4c13bf4`.

## What the host does per token today

Every dispatch is a fresh record through `q36_vk_run_unlocked`
(`q36_vulkan.c:3118`):

- `q36_vk_alloc_descriptor_unlocked` (`:3088`) allocates a `VkDescriptorSet`
  from a 4096-set / 16384-storage-descriptor pool, then
  `vkUpdateDescriptorSets` (`:3143`) writes one `VkDescriptorBufferInfo` per
  binding (up to 6 for the MoE kernels).
- `vkCmdBindDescriptorSets`, `vkCmdPushConstants`, `vkCmdDispatch` (`:3180`).
- The per-token scalars — position, KV length, `n_tok` — are *inside* the push
  constant block, so the push is part of the record.
- The hazard barrier is decided per dispatch from tensor `write_gen`s
  (`:3146-3172`); independent dispatches are deliberately left to overlap.
- Command buffers are a 4-slot ring (`Q36_VK_CB_RING`, `:2031`) submitted
  eagerly; a slot's descriptor pool is reset only when its fence completes
  (`:2224`, `:3622-3637`).  So the host *already* re-records the whole step
  every token.

`Q36_VK_CB_RING = 4` is therefore a lookahead depth, not a cache: the recorded
content is never reused.

## Scope (what "replayable" has to mean here)

1. **Param buffer.**  Move the per-token scalars out of the push constant block
   into one host-mapped SSBO (`q36_vk_params`, a few 16-byte slots per layer).
   Kernels read `pos` / `kv_len` / `n_tok` from it.  The host writes 16-64 B per
   token instead of re-recording.
2. **Baked descriptors.**  Allocate the descriptor set once per
   (kernel, tensor-role) tuple and keep it in the kernel/tensor state; rebind
   only when a root `VkBuffer` handle or a range changes.
3. **Record once per shape.**  With (1) + (2) and a barrier plan derived from the
   static producer/consumer graph, one command buffer per token *shape* is
   recorded and replayed; the host only writes the param buffer and
   `vkQueueSubmit`s.

## Blockers found by reading the current code

- **MoE per-call staging is host-driven.**  `selected_host` is pack-indexed on
  the host and the expert weights are `memcpy`-ed into `packed_weights` every
  call (`:8362-8394`), so the packed `VkBuffer` is stable only after the pack.
  The resident expert-bank path (GPU-built tiles, `q36_vk_moe_build_tiles`) is
  the shape that makes a baked set possible; the per-call path would need
  `VK_EXT_descriptor_buffer` or an indirect/bindless expert gather.
- **Tensor reallocation invalidates a baked set.**  Scratch is retired and
  reallocated, so replay needs a generation guard (the existing `write_gen` /
  `last_use_seq` pair is the right hook) with re-record on mismatch.
- **Barriers.**  The dynamic check only ever *adds* a barrier, so a baked plan is
  free to be conservative; the risk is the opposite — baking the current
  per-shape plan loses the "independent dispatches overlap" property if the plan
  is built from a coarse rule.  The plan must be derived from the same
  producer/consumer relation, per token shape.
- **Push-constant layout.**  Kernels whose push block mixes static geometry with
  per-token scalars need the static half hoisted (second push range, uniform, or
  the param SSBO) or the push still has to be re-recorded and replay buys
  nothing.

## Prototype plan (in this order, each step measurable)

1. Add the param SSBO and convert **one** kernel (`attn_decode_split`) to read
   `pos` / `kv_len` from it.  Parity: frontier-513 dump, `cmp_logits.py`,
   `max_abs_diff = 0`.
2. Hoist descriptor allocation out of `q36_vk_run_unlocked` for that one kernel
   only; add the generation guard and re-record on mismatch.  Parity as above.
3. Measure the per-token wall-vs-GPU gap before and after — the same measurement
   P9(a) needs — and only then decide whether the remaining ~100 dispatches per
   token are worth converting.

## Why not start with the full decoder

The ceiling is the host-side per-token gap.  If the measured gap is a small
fraction of the 46.7 ms/tok decode, a replayed command buffer cannot pay for the
rework; the step-3 measurement decides that, and step 1 is the cheapest way to
get it.  Do not convert the MoE dispatch path first: it carries the two hardest
blockers (host-side expert packing, unstable packed buffers).
