# Splash / Qwen3.6-Qwen3.8 Investigation — HANDOFF

## Context

Track B (PTQ1_0/PQ2_0 ternary kernel optimization, branch `trackB-ptq1_0`,
worktree `/home/server/q36-opt-27b`) is PAUSED as of 2026-09-19: diminishing
returns, PQ2_0 already faster than PTQ1_0. Full history in that worktree's
`logs/bc250-sustained-20260918/{ptq1_0,pq2_0}-validation/HANDOFF.md` and the
Maestri canvas note "Track B Coordination". Closing state also persisted to
mem0 (user_id `q36-opt-27b`), memory ids:
- `99024eda-9584-4f8e-a836-401bd42dcacd` — pause/pivot decision
- perf win (commit 43dcf03) and negative-results (commit 8d527f4) entries also
  written the same session.

Active focus moves to Qwen3.6/Qwen3.8 quantized-model optimization, informed
by techniques from https://github.com/incoai/splash.

## Splash — what we know so far (README/metadata only, no source read yet)

- Apple-silicon-only local inference engine, Apache-2.0.
- Per-model specialized kernels (not a generic ggml-style dispatcher).
- Ships a draft model alongside the target model — implies speculative
  decoding is core to its throughput story, not incidental.
- Has an explicit "memory plan" concept (unclear yet what this controls —
  KV cache layout? weight residency? needs source read).
- Claims 2x next-fastest engine decoding Qwen3.8-27B on a 48GB M5 Pro; 282ms
  TTFT at 32K cached context.
- OpenAI/Anthropic-compatible API server: `/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, `/tokenize`, `/apply-template`.
- Install: `brew install incoai/tap/splash`; serve:
  `splash serve --model incoai/Qwen3.8-27B-Splash`.

Note there is an existing separate worktree/branch for Qwen3.8 work already:
`/home/server/q36-opt-qwen38` (`feat/qwen38-opt`, marked prunable) — check
that before duplicating effort; this investigation may fold into it rather
than stay standalone.

## Team / workflow carried forward from Track B

Maestri canvas roster: Builder, Mcode, Reviewer, Grendel (ruthless Principal
Engineer judge), Big Pickle (mem0 memory archivist). Same discipline applies:
independent review before any GPU test/commit, never trust an agent's
self-reported benchmark numbers without reading the actual log file, profile
before optimizing, one GPU/benchmark job running at a time (this box OOM'd
twice under concurrent q36-bench + 5 agent CLIs — 14GB RAM), and favor fast
capped-scope results over long thorough runs.

## Status

Source cloned read-only to `splash-src/` (git clone, HEAD, not built — Metal-only,
this box has no Xcode/Metal toolchain and AGENT.md forbids editing q36 kernels
in this pass anyway). Findings below are from reading `splash-src/runtime/`,
`splash-src/dev/tuning/`, and `splash-src/runtime/model/` directly; every claim
cites file:line in that checkout as of this commit (shallow clone, HEAD at
clone time 2026-09-19).

## Finding 1 — "Per-model specialized kernels": hand-written + heuristic dispatch + offline auto-tuner, not codegen

There is no code generation and no generic ggml-style op graph. Two model
families each get their own hand-written C++ class:
`runtime/model/Qwen3_8.{hpp,cpp}` and `runtime/model/Qwen3_6Moe.{hpp,cpp}`,
selected at load time by a string tag in the package manifest
(`splash-packed-q4` vs `splash-packed-q4-moe`,
`runtime/model/ModelDescriptor.mm:386-392`). The manifest also carries an
`execution_geometry` block that must match the runtime's compiled constants
field-for-field before the model loads at all
(`runtime/model/ModelDescriptor.mm:23-31,135-145`) — kernel/geometry versioning
is enforced by a hard equality check, not a compatibility shim.

Within one model family, kernel *variant* selection is a two-layer scheme:

1. Every linear (Q4 matmul) operator call goes through `Q4Linear::plan()`
   (`runtime/ops/Linear.cpp:296-301`), which looks up a workload key
   (matrix shape, phase, epilogue, row count) in a sorted `choices_` table
   installed at startup; on a miss it falls back to `Q4Linear::baseline()`
   (`runtime/ops/Linear.cpp:253-294`). `baseline()` is pure heuristic: it
   branches on `appleGpuFamily_` (9 vs ≥10) and `gpuCores_` to pick a metal
   pipeline string (tile width N128/N256/Paired128, simdgroup count 4 vs 8,
   persistent threadgroup-group count via `decodeGroups()`,
   `runtime/ops/Linear.cpp:216-234`). The actual pipeline name strings are
   assembled in `LinearPlan::LinearPlan()` (`runtime/ops/Linear.cpp:104-176`,
   e.g. `"decode_linear_q4_n128_m24_sg4"`) — one string per hand-written
   `.metal` kernel entry point, not a generated permutation.
2. `Q4Linear::candidates()` (`runtime/ops/Linear.cpp:317-350`) enumerates the
   legal tile/simdgroup/group-count combinations for a given workload shape.
   `dev/tuning/LinearTuning.cpp:120-307` (`tuneLinear`) is an *offline* A/B
   harness: it allocates real device buffers, runs every candidate against a
   correctness oracle (byte-exact match against the baseline's output,
   `LinearTuning.cpp:218-230`), takes 12–64 paired GPU-timing samples with
   alternating baseline/candidate order to cancel drift
   (`dev/tuning/Tuning.hpp:26-53`), and only accepts a candidate if both GPU-time
   and wall-time statistics agree it's a real, non-noisy win
   (`Tuning.hpp:117-139`, `LinearTuning.cpp:297-301`). The winning
   `LinearChoice` table is what ends up installed via `Q4Linear::setChoices()`
   (`Linear.hpp:129`, `Linear.cpp:305-315`) — i.e., the "kernel specialization"
   is: hand-written kernel zoo → heuristic device-family dispatch → optional
   per-shape override table populated by a paired-timing statistical
   auto-tuner run once per device class, not at request time.

**Portability**: the *architecture* — hand-written kernel variants keyed by
workload shape, a heuristic baseline chooser keyed on device tier, and an
offline paired-timing auto-tuner with a correctness oracle and statistical
acceptance gate — is fully portable to GLSL/Vulkan and has nothing
Metal-specific in it (it's just "shader permutations + a benchmark harness
that promotes winners"); q36 currently has no comparable auto-tuner for its
`vulkan/*.comp` kernels and BC-250-specific heuristics are hand-picked, so
`dev/tuning/{Tuning,LinearTuning,Measurement}.{hpp,cpp}`'s *design* (paired
samples, alternating order, byte-exact oracle, conservative-gain selection)
is a concrete, low-risk thing to copy structurally. The specific tuning
knobs (simdgroup count, N128/N256 tile, Apple GPU family thresholds) are
Metal-SIMD-group-specific and do not transfer; RDNA's wavefront size (32 on
RDNA1/2) and subgroup ops would need their own baseline() equivalent.

## Finding 2 — Speculative decoding: DFlash — a graph-structured draft proposer with hidden-state sharing, not simple linear draft chaining

The draft model is a full 6-layer transformer (`DFlashDraftLayout`,
`runtime/model/DFlashDraft.hpp:54-79`) that reads the *target model's captured
hidden states* directly (`contextProjection` from `targetHiddenSize` 16384/25600
down to the draft's own `hiddenSize`, `DFlashDraft.hpp:105-116`,
`DFlashDraft.cpp:95-141` `addContextPrefill`) rather than re-embedding raw
tokens — this avoids the draft doing its own redundant embedding/attention
pass over history the target already processed.

Proposal generation is not "sample one token, repeat": it uses two learned
"codebooks" (`predecessorCodebook`, `successorCodebook`,
`DFlashDraft.hpp:171-172`, loaded `DFlashDraft.cpp:339-346`) and three
kernels chained in `Sampling::addDraftSelector`
(`runtime/ops/Sampling.cpp:176-209`):
`draft_select_top16_sharded` → `draft_select_edges` → `draft_select_dflash`.
`draft_select_edges` builds a 16×16 "edge table" per position
(`Sampling.cpp:32,198-203`, workspace comment `Sampling.hpp:34-40`
"candidates the followed by the position's 16x16 edge table") scoring
predecessor→successor token-pair compatibility via the codebooks, then
`draft_select_dflash` walks that structure to emit `proposalTokens`
candidate tokens with probabilities in one dispatch
(`Sampling.cpp:204-208`). This is a small, learned bigram-transition graph
over the draft's own top-16 candidates per step, not a straight-line
token-by-token draft — it's proposing structured, position-pair-scored
candidates that the target then verifies in a batch.

Verification/acceptance is `Sampling::addAcceptance` →
`decode_accept_dflash` (`Sampling.cpp:211-237`): it compares
`proposedTokens`/`proposalProbabilities` (from the draft) against
`targetTopIds`/`targetTopProbabilities` (the target's own top-32 sampling
output for the same verify rows) and writes `outputTokens`,
`retainedCounts`, `nextAnchors`, `acceptedCounts` per lane in a single
kernel — rejection sampling against the full top-k target distribution
happens on-GPU, not a CPU round-trip. Batch width and row budgets are fixed
compile-time constants: `draftProposalTokens = 7`, `draftQueryRows =
targetVerifyRows = 8`, `maximumBatchWidth = 4`
(`runtime/model/Model.hpp:249-260`), so one speculative step verifies at most
7 drafted tokens + 1 terminal token per lane, 4 lanes at a time.
Context management for the draft is a *separate* fixed-size sliding-window
ring (`DFlashDraftRing`, 2048-token window, `DFlashDraft.hpp:31-52,54-71`,
allocated in `DFlashDraft.cpp:34-61`) distinct from the target's paged KV —
the draft never needs the target's full context window, just a bounded
recent slice, saving KV memory for the (cheap) draft model.

**Portability**: the core mechanism — draft reads target hidden states
instead of re-tokenizing (`addContextPrefill`), draft proposes a *small
graph* of candidates scored by a learned compatibility table instead of one
greedy chain, verification is a single fused accept/reject kernel against
target top-k, and the draft keeps its own small bounded KV window separate
from the target's — is entirely backend-agnostic; none of it depends on
Metal SIMD-groups, unified memory, or MPS. It ports directly to GLSL compute:
`draft_select_edges`/`draft_select_dflash` are just small fixed-size
GEMM/argmax-ish shaders (16×16 edge table, top-16 candidates) that would
compile fine as Vulkan compute shaders with subgroup-size-portable logic (no
Metal-specific simdgroup width assumptions in the *algorithm*, only in the
current *tuning*, per Finding 1). The one caveat: q36 today has no draft
model or DFlash-style codebook infrastructure at all, so this is new
capability to build, not a kernel swap — but it's the single most
directly-applicable idea in the whole investigation (see summary below).

## Finding 3 — "Memory plan": elastic byte budget + Q8 paged KV cache over sparse GPU virtual memory

`EngineMemoryPlan`/`evaluateEngineMemoryPlan`
(`runtime/engine/MemoryPlan.hpp:158-201`, `MemoryPlan.cpp:257-365`) computes a
single `hardBudgetBytes` from the device's `recommendedMaxWorkingSetBytes`
minus a margin (max of 1 GiB or 2%, `MemoryPlan.hpp:50-61`, optionally
further clamped by a user-configured ceiling, `MemoryPlan.hpp:63-73`). It then
subtracts every *fixed* cost — target/draft/vision weights, one active
state cell, shared prefill/decode scratch, a pipeline reserve, a runtime
overhead reserve (`ModelMemoryFootprint`, `MemoryPlan.hpp:23-32`, summed in
`ModelMemoryProfile::fixedRuntimeBytes`, `MemoryPlan.cpp:94-106`) — leaving one
`dynamicBudgetBytes` (`MemoryPlan.cpp:301-304`) that is the *only* pool active
per-request state cells and the KV cache both draw from
(`EngineMemoryBreakdown` comment, `MemoryPlan.hpp:119-122`: "All active state
cells, cached composite states, and physical KV extents grow from this one
governor-controlled byte budget. None is preallocated merely because the
address space exists."). If that budget can't fit even one active state cell
plus one physical KV extent, the plan fails outright with
`BudgetErrorCode::KvPoolDoesNotFit` (`MemoryPlan.cpp:345-359`) rather than
silently degrading.

The KV cache itself (`runtime/ops/PagedKv.hpp`) is Q8 (INT8) symmetric
quantization with one FP32 scale per (token, head) — `Q8Layout`
(`PagedKv.hpp:87-150`), pages of fixed `SPLASH_TARGET_KV_BLOCK_TOKENS`
tokens, physically backed by `Q8PageStorage`
(`runtime/ops/Q8PageStorage.hpp:16-75`), which implements the `Backing`
interface (`PagedKv.hpp:15-31`: `ensureResident`/`releaseBackingForPage`/
`extentFirstPage`/`extentPageCount`) over Metal's *placement-sparse* buffers
— `DeviceCapabilities::supportsPlacementSparse`
(`runtime/metal/DeviceCapabilities.hpp:34-35`, requires macOS ≥ 26.4,
`DeviceCapabilities.hpp:12-14`). This exposes one huge virtual KV address
space up front but only commits physical backing for pages actually
touched, mapped/unmapped in 64 KiB-tile-aligned batches
(`kSparseMappingAlignmentBytes`, `PagedKv.hpp:48`, batch-size math
`sparseMappingBatchPages()`, `PagedKv.hpp:118-126`), and release is async and
paced so a page extent isn't reused until the kernel actually finishes
tearing down the old mapping (`Q8PageStorage.hpp:44-51`,
`releaseReady()`/`awaitRelease()`). `EngineMemoryPlan::maximumContextTokens()`
(`MemoryPlan.cpp:237-245`) derives the actual context ceiling from
`kvVirtualTokens` (virtual pages × page tokens) minus a small reserve for
in-flight speculative rows, not from the model's nominal max-context — the
context length you actually get is whatever the sparse virtual KV pool can
back, clamped to the model's stated ceiling.

**Portability — this is the strongest direct hit of the three findings, and
I checked it against actual hardware, not just docs.** Vulkan's sparse
resource binding (`VK_KHR_sparse_binding`, `sparseResidencyBuffer`) is the
exact analog of Metal's placement-sparse buffers, and **this BC-250 already
reports both as supported**: `vulkaninfo` on this box shows
`sparseBinding = true`, `sparseResidencyBuffer = true`,
`sparseResidencyImage2D = true` (checked live in this session,
`deviceName = AMD BC-250 (RADV GFX1013)`). That means the whole
over-commit-then-page-in-on-demand KV design — expose a huge virtual KV
region, physically bind only the pages a request is actually using, unbind
in aligned batches when a request's context shrinks or ends — could be built
on RADV's sparse-buffer support today, no driver gap to work around. The
INT8 symmetric per-(token,head) KV quantization is backend-agnostic math and
already has a spirit-of-the-idea analog in AGENT.md's existing "Q2
routed-expert quant only" tight-memory posture; adopting Q8-for-KV
specifically (as opposed to weights) is a new, separately portable idea
q36 doesn't currently do. The one part that's Metal/macOS-specific is the
*margin arithmetic* (`recommendedMaxWorkingSetBytes`, macOS-version-gated
sparse support) — Vulkan has no equivalent single "recommended working set"
query, so q36 would need its own headroom heuristic against
`VkPhysicalDeviceMemoryProperties`/`VK_EXT_memory_budget`, but the *shape* of
the plan (fixed costs subtracted first, one elastic dynamic pool shared by
active-state and KV, hard failure instead of degrade-silently) transfers
directly.

## Most promising idea to hand off

Finding 3 (Q8 sparse-virtual-memory paged KV cache) is the one with a
verified, zero-gap hardware path on this exact box (`sparseResidencyBuffer =
true` on the BC-250 today) and would let q36 stop pre-committing full KV
capacity per session. Finding 2 (DFlash-style draft proposer reading target
hidden states + graph-scored multi-candidate proposals + fused GPU
accept/reject) is the higher-effort but higher-payoff idea since q36 has no
speculative decoding at all yet. Both are algorithm/architecture ports, not
code ports — no splash source was copied and no q36 kernel code was
touched in this investigation.

## Design: Converting q36 Vulkan KV cache to sparse‑binding paging (VK_KHR_sparse_binding)

### Current allocation (incremental growth, not full pre‑allocation)
* `q36_vulkan_runtime_create` determines the initial KV capacity with `kv_cap = q36_kv_initial_cap(ctx_size)`【q36.c:6481-6497】.
* The actual GPU buffers are allocated per‑layer here:
  * `rt->full[il].k = q36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * k_row_bytes)`
  * `rt->full[il].v = q36_gpu_tensor_alloc_uninitialized((uint64_t)kv_cap * v_row_bytes)`【q36.c:6820-6821】.
* When more tokens are needed, `q36_vulkan_runtime_reserve_kv` (`q36.c:6874`) invokes `q36_vulkan_cache_reserve` (`q36.c:6851-6870`), which allocates new, bigger K/V buffers, GPU-copies the live rows forward with `q36_gpu_tensor_copy` + `q36_gpu_synchronize`, then frees the old buffers — confirmed: this is a synchronous realloc-and-copy, not in-place growth.
* `q36_kv_initial_cap` starts at `min(ctx, Q36_KV_INITIAL_CAP)`【q36.c:5998-6000】 and `q36_kv_next_cap` grows by ~1.25× steps (next = cap + cap/4) up to the user‑requested limit【q36.c:6002-6009】.
* Therefore q36 already avoids allocating the full `--ctx` buffer up front; the benefit of sparse‑binding is to **eliminate the copy‑on‑grow reallocations** and to enable **true release of pages when a session ends or the context shrinks**, which the current grow‑and‑copy approach cannot do.

### Sparse‑binding design

| Aspect | Decision | Rationale / reference |
|--------|----------|-----------------------|
| **Virtual buffer size** | `max_tokens = ctx_size` (the user‑requested `--ctx`). Allocate one **sparse** `VkBuffer` per layer per K and V with `VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT` and size = `max_tokens * row_bytes`. | Matches Splash’s “expose huge virtual KV address space”【HANDOFF.md:225-229】. |
| **Page granularity** | 32 tokens = `SPLASH_TARGET_KV_BLOCK_TOKENS` (found in Splash) → page bytes = `32 * row_bytes`. 32‑token pages align with the 64 KiB sparse‑mapping tile used by Splash (32 tok × 256 B ≈ 8 KiB per row, 8 pages ≈ 64 KiB)【splash‑src/runtime/metal/abi/ExecutionGeometry.h:16】. | Keeps bind/unbind granularity identical to Splash; avoids fragmenting VRAM. |
| **Buffer creation time** | Replace the current `q36_gpu_tensor_alloc_uninitialized` calls with a new helper `q36_vk_tensor_alloc_sparse(bytes, page_bytes)` that:
  1. Calls `vkCreateBuffer` with `flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT`.
  2. Queries memory requirements via `vkGetBufferMemoryRequirements2`.
  3. Allocates memory with `VkMemoryAllocateFlagsInfoKHR(flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR | VK_MEMORY_ALLOCATE_SPARSE_BIT_KHR)`.
  4. Defers actual page binding to later via `vkQueueBindSparse`.
| Vulkan sparse buffers require the creation flags above and later a `VkSparseBufferMemoryBindInfo` per bind【Vulkan spec】. |
| **Per‑step binding** | In the decode/prefill loop (where `q36_vulkan_runtime_reserve_kv` is invoked) compute `need_pages = ceil(live_tokens / 32)`. For each newly required page, submit a `VkBindSparseInfo` with a `VkSparseBufferMemoryBindInfo` pointing at the appropriate offset (`page_index * page_bytes`). Track bound pages per‑layer to avoid duplicate work. | Mirrors Splash’s on‑demand page mapping【HANDOFF.md:207-212】. |
| **Release / shrink** | When a session ends or the sliding context window moves, submit unbind operations for pages no longer needed and synchronize with a fence or timeline semaphore before the memory is reused. | Mirrors Splash’s async release (`releaseReady` / `awaitRelease`)【HANDOFF.md:209-212】. |
| **Memory budget** | Virtual size = `ctx_size * (k_row_bytes + v_row_bytes) * full_layers`. Physical memory is only allocated for bound pages. Use `VK_EXT_memory_budget` to enforce a headroom heuristic similar to Splash’s `hardBudgetBytes`【HANDOFF.md:238-242】. | Prevents over‑commit on limited VRAM. |
| **Error handling** | If `vkQueueBindSparse` fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`, fall back to truncating the context or abort with a clear error, replicating Splash’s `BudgetErrorCode::KvPoolDoesNotFit` behaviour【HANDOFF.md:192-194】. |

### Files to modify (design only – no code yet)
* `q36_vulkan.c` – add `q36_vk_tensor_alloc_sparse`, `q36_vk_sparse_bind_pages`, `q36_vk_sparse_unbind_pages`.
* `q36.c` – in `q36_vulkan_runtime_create` replace the K/V `q36_gpu_tensor_alloc_uninitialized` calls with sparse buffers; in `q36_vulkan_runtime_reserve_kv` (and the decode path) invoke the bind helpers instead of realloc‑and‑copy.
* No shader changes required; existing `kv_store` / `kv_store_quant` kernels already index by token offset, which works on a sparsely‑backed buffer because the virtual address space is contiguous.

### Single most important design decision & risk
**Decision:** Use **32‑token pages (SPLASH_TARGET_KV_BLOCK_TOKENS)** as the sparse‑binding granularity, matching Splash’s page size and the 64 KiB tile alignment required by the hardware.

**Risk:** RADV’s sparse‑buffer implementation may have higher bind/unbind latency than Metal’s placement‑sparse buffers. If bind latency per decode step becomes a bottleneck, the design would need to batch multiple page binds into a single `vkQueueBindSparse` submission or pre‑bind a larger window (e.g., next N pages) to amortize cost.
