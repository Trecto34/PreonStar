# q36 Vulkan optimization log — BC-250 / GFX1013

Working log for the roadmap at `~/.claude/plans/you-are-a-gpu-partitioned-zebra.md`.
Branch `feat/vulkan-opt-phase1`, worktree `/home/server/q36-opt`.

Every entry records what changed, how it was gated, and what was measured. Claims are labelled
`MEASURED_ON_BC250`, `CODE_ANALYSIS_ONLY`, or `UNVERIFIED_EXTRAPOLATION`.

---

## Setup

### Build

```bash
make all VULKAN_CFLAGS=-DQ36_VULKAN_REQUIRE_BC250 -j8
```

Note: the documented `make vulkan-bc250` target runs `$(MAKE) -B all`, which force-rebuilds all
100 shaders on every invocation. For iteration the incremental form above is equivalent and much
faster — `-B` was only needed for the first build.

### Model

The roadmap names `RavenX-35B-Q36-IQ2XXS.gguf`; the file actually present on this box is
`gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`. Confirmed to be the same shape the
baseline was measured on (`MEASURED_ON_BC250`):

```
arch: qwen35moe, 40 layers, heads=16 kv_heads=2 head_dim=256 (GQA 8), full_interval=4
experts: count=256 used=8 ff=512 shared_ff=512
10.93 GiB, 34.66 B logical parameters
```

All work below uses this file. Do **not** use the `q36moe.gguf` symlink — it was repointed at the
dense 27B during the benchmark handoff.

### Parity gate

```bash
./q36-bench -m gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf --vulkan \
  --prompt-file tests/long_context_story_prompt.txt --prefill-chunk 1024 \
  -ctk q8_0 -ctv q4_0 --ctx-start 2048 --ctx-max 2064 --step-incr 1 --gen-tokens 0 \
  --dump-frontier-logits-dir <DIR>
```

17 frontiers, 51 MB. Frontier 002048 gates the prefill kernels; 002049–002064 are single-token
forward passes and gate the decode kernels. Candidate passes only if every file is `cmp`-identical
to the baseline.

**The gate was validated before use.** Two baseline captures from the same unmodified binary are
byte-identical, so a diff means a real numerical change and not run-to-run nondeterminism
(`MEASURED_ON_BC250`). This matters: without that check a "parity pass" proves nothing.

Baseline captured at HEAD `10c6965` before any edit. Decode throughput on the capture run was
84.55–84.63 t/s, consistent with the documented 84.3 t/s baseline.

---

## 1.1 — Sticky device-lost latch

**Status: done. Parity clean. Defect fixed and verified by fault injection.**

### The defect

A gfx ring timeout destroys the GPU context. `q36_vulkan.c` had **zero** `VK_ERROR_DEVICE_LOST`
handling: every submit path collapsed failure to `return 0` and kept recording into a dead device,
so the process died with SIGSEGV inside `vkEndCommandBuffer` instead of reporting an error.
Reproduced twice from `--prefill-chunk 4096` (`MEASURED_ON_BC250`).

### The change

A file-scope one-way latch plus a `q36_vk_ok(VkResult, const char *)` helper, declared beside the
command-buffer ring.

- `q36_vk_ok()` latches on `VK_ERROR_DEVICE_LOST`, prints one diagnostic naming the failing call,
  and returns 0 for any non-success result.
- Wrapped call sites: `vkEndCommandBuffer`, `vkQueueSubmit`, `vkBeginCommandBuffer`,
  `vkAllocateDescriptorSets` (both the first attempt and the post-rotate retry), the fence
  wait in `q36_vk_slot_wait_unlocked`, and `vkDeviceWaitIdle` in `q36_gpu_synchronize`.
- Guarded chokepoints, which between them cover all 93 `q36_gpu_*` entry points without touching
  any of them: `q36_vk_run_unlocked`, `q36_vk_begin_batch_unlocked`,
  `q36_vk_submit_current_unlocked`, `q36_gpu_tensor_contents`.
- `q36_gpu_cleanup` no longer calls `vkDeviceWaitIdle` on a lost device — that queue never drains.
- New public query `q36_gpu_device_lost()` in `q36_gpu.h`, so a caller can distinguish "the device
  died" from an ordinary op failure.

Two decisions worth recording:

1. **`q36_vk_slot_wait_unlocked` clears `pending` on failure.** A lost device never signals the
   fence, so leaving the slot pending would let a later flush wait on it again.
2. **`q36_gpu_tensor_contents` returns NULL once latched** rather than handing back the bytes the
   dead batch left behind. Those are stale, and returning them would let a corrupted forward pass
   look successful — the exact failure the latch exists to prevent.

The latch never recovers and continues. Dispatches in flight at the timeout produced no results.

### Fault injection

`Q36_VK_FAULT_INJECT_LOST=<n>` forces the n-th `vkQueueSubmit` to report device loss. Diagnostic
only, off when unset. AGENT.md permits diagnostic switches that validate the one release path, and
this is otherwise untestable without provoking a real TDR.

### Verification (`MEASURED_ON_BC250`)

| check | before | after |
|---|---|---|
| Frontier parity, all 17 files | — | **bit-identical** |
| Injected loss at submit 3 (pre-prefill) | SIGSEGV | clean error, **exit 1** |
| Injected loss at submit 30 (mid-prefill) | SIGSEGV | clean error, **exit 1** |
| Diagnostic printed | — | exactly **once** per process |
| Kernel log during injection runs | — | zero amdgpu timeout/reset events (synthetic fault, as intended) |
| Control run, no injection | 620.26 prefill / 84.3 gen | **620.26 prefill**, unchanged |

Delta: **0 t/s**, as predicted. This item buys a clean failure, not speed.

---

## 2.1 — Split the f32 profiler row

**Status: done. Diagnostic naming only, no dispatch change. Produced a finding that revises the
roadmap.**

### The change

`q36_vk_matmul_dense()` reported every f32 dispatch under one hardcoded op name, `dense_f32`, so
the router gate and the shared-expert gate — which differ by more than two orders of magnitude in
launch count — were indistinguishable in the profile.

Added `q36_vk_f32_op_name(in_dim, out_dim, n_tok, fast)`, mirroring the existing
`q36_vk_q8_0_op_name()`: literal strings, `_d_`/`_p_` for decode/prefill, and an `f32f` marker
separating the fast vec4 kernel from the exact-accumulation `matmul_f32`, which `--quality` and
`Q36_VK_F32_FAST=0` select. Names are only built when profiling is enabled; the non-profiling path
still passes the constant `"dense_f32"`.

### Finding: the roadmap's rank-#1 item is roughly 2.6x smaller than recorded

`MEASURED_ON_BC250`, ctx 2048, 128 decode tokens, `Q36_VK_PROF_KERNEL=1`:

| op | dispatches | groups | gpu_ms |
|---|---:|---:|---:|
| `dense_f32f_p_256x2048` (prefill) | 160 | 41,943,040 | 213.320 |
| `dense_f32f_p_1xN` (prefill) | 160 | 163,840 | 6.993 |
| `dense_f32f_d_256x2048` (decode) | 5,120 | 1,310,720 | **56.189** |
| `dense_f32f_d_1xN` (decode) | 5,120 | 5,120 | **20.242** |

Decode f32 totals **76.43 ms / 128 tokens**, not the **197 ms** the roadmap's kernel table
attributes to `matmul_f32_fast`. 5,120 dispatches = 128 tokens x 40 layers, confirming these are
the per-layer router and shared-expert gates and nothing else.

The 197 ms figure evidently pooled prefill and decode dispatches into one row — exactly the defect
this item fixes. **§3.1's projected +7 to +9% is therefore overstated and must be re-derived**
against a correctly separated decode budget before that work is scheduled.

What the split does confirm is the *shape* of the §3.1(a) argument: `dense_f32f_d_1xN` spends
20.242 ms across 5,120 single-workgroup dispatches, i.e. **~3.95 us per dispatch to move 8 KB**.
That is pure launch latency and is what the pair-fuse would remove. The `_256x2048` row at
56.189 ms over 1,310,720 groups is real work by comparison.

