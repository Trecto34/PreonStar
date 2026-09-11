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


---

## 1.3 — GFX1013 prefill-chunk clamp

**Status: done. Parity clean. Removes the memory footgun and makes the crashing invocation safe.**

### The change

New `q36_gpu_device_is_gfx1013()` (`q36_gpu.h`, implemented in `q36_vulkan.c` against the existing
BC-250 device-id test, with a constant-0 stub added to `q36_metal.m` so the shared engine code can
call it unconditionally).

`q36_engine_clamp_prefill_cap()` caps the chunk at **1024**, or **256** when GQA != 8, and is
applied **after** the `--prefill-chunk` override — the override is precisely the input that
crashed the engine, so clamping before it would have been useless.
`Q36_VK_PREFILL_CHUNK_UNSAFE=1` restores the raw value. The warning prints once per process.

### Finding: clamping the runtime alone was not enough

The first working version clamped `q36_engine_gpu_prefill_cap()` only. Testing showed the run
still reserved **960.30 MiB** and still reported `prefill_chunk=4096`:

```
q36-bench: context buffers 960.30 MiB (scratch=880.97 MiB, ..., prefill_chunk=4096)
q36: clamping prefill chunk 4096 -> 1024 on GFX1013 (GQA 8)
```

`q36_context_memory_estimate_configured()` takes the chunk straight from the caller and never
consults the engine cap, so `q36-bench`, `q36-server` and the engine each sized scratch from the
unclamped request. **The +634 MB footgun this item exists to remove was still fully present**, and
the clamp message was actively misleading — it announced a cap that the allocator ignored.

Fixed by clamping inside the estimator itself, the one chokepoint all three callers share.

### Verification (`MEASURED_ON_BC250`)

`--prefill-chunk 4096`, ctx 2048:

| | before | after |
|---|---:|---:|
| context buffers | 960.30 MiB | **300.28 MiB** |
| scratch | 880.97 MiB | **220.96 MiB** |
| reported chunk | 4096 | **1024** |
| prefill t/s | — | 608.70 |
| gen t/s | — | 81.35 |

**660 MiB reclaimed**, matching the ~634 MB the roadmap predicted from ~206 KB per chunk token.

Independent confirmation of the roadmap's throughput claim, via the escape hatch
(`Q36_VK_PREFILL_CHUNK_UNSAFE=1`, genuinely running chunk 4096): **188.50 t/s prefill against
608.70 clamped**, a 3.2x penalty. The roadmap recorded 202 vs 616 from an earlier session. So the
clamp costs nothing anyone wants — the unclamped path is both slower and 660 MiB heavier.

No amdgpu timeout, reset or wedged events were logged during the unclamped run (it survives at
ctx 2048; the original crash needed a longer context).

---

## 2.2 — rope_qwen / quantize_q8_0 forced wave32

**Status: done. Bit-exactness is provable here, not merely argued.**

Verified before editing, in each `.comp` source (`MEASURED_ON_BC250`):

| shader | local_size_x | cross-lane builtin uses |
|---|---:|---:|
| `rope_qwen.comp` | 32 | **0** |
| `rope_qwen_mrope.comp` | 32 | **0** |
| `quantize_q8_0.comp` | 32 | **0** |

With `local_size_x = 32` and no lane-crossing operation anywhere in the shader, wave width cannot
influence a result bit — on wave64 the upper 32 lanes are simply inactive. This is the distinction
the roadmap draws against §3.2(b), where `subgroupAdd` over wave64 makes the same change
argued-but-unproven.

Replaced the ad-hoc condition in the pipeline-creation path with an explicit table, and wrote down
the precondition so nothing gets added to it without the check:

```c
static const char *const q36_vk_force_wave32[] = {
    "delta_net_cols.spv", "rope_qwen.spv", "rope_qwen_mrope.spv", "quantize_q8_0.spv",
};
```

`dense_*_mmq.spv` keeps its existing pattern match.

---

## Re-derived decode budget (supersedes the roadmap's kernel table)

`MEASURED_ON_BC250`, ctx 2048, 128 decode tokens, `Q36_VK_PROF_KERNEL=1`, single run.
Decode rows only: op rows tagged `_d_`, plus shaders that exist only on the decode path.

| row | gpu_ms /128tok | % of decode | roadmap said |
|---|---:|---:|---|
| `dense_q8_0_d_pair` | 283.7 | 15.5% | 285 — matches, at roofline |
| `delta_net_decode_reg_f16` | 191.0 | 10.4% | 127, in the "measure first" bucket |
| `dense_q8_0_d_vocabx2048` (output head) | 186.7 | 10.2% | folded into the 392 q8_0 row |
| `moe_down_q2k_sum_decode` | 164.1 | 8.9% | 165 — matches |
| `moe_gate_up_decode` | 143.9 | 7.8% | 145 — matches |
| `dense_q8_0_d_2048x4096` | 131.8 | 7.2% | folded into the 392 q8_0 row |
| `attn_decode_split` | 127.8 | 7.0% | 128 — matches |
| `add_rms_norm` (decode share) | ~127 | 6.9% | 112 |
| `recur_conv_silu_decode` | 105.2 | 5.7% | 57 |
| `dense_q8_0_d_8192x2048` | 62.4 | 3.4% | folded into the 392 q8_0 row |
| **`matmul_f32_fast` (both shapes)** | **72.5** | **3.9%** | **197 — wrong by 2.7x** |
| `router_topk` (decode share) | ~55 | 3.0% | 55 — matches |
| `shared_gate_up_decode` | 37.5 | 2.0% | 65 for both shared kernels |
| `shared_down_tail_decode` | 25.2 | 1.4% | (as above) |
| `dense_q8_0_d_512x2048` | 10.1 | 0.6% | — |

Decode total reconciles to **~1836 ms**, against the roadmap's 1814 ms. The *total* was right; the
*attribution* was not. Cross-check on the one row I split: prefill 220.3 + decode 72.5 = 292.8 vs
the 296.7 ms shader-level total for `matmul_f32_fast` — consistent to 1.3%.

### Consequences for Phase 3

**§3.1 (`matmul_f32_fast`) drops from rank #1 to roughly rank #11.** The row is 3.9% of decode, so
deleting it entirely would yield at most **+4.1%**. The realistic change — fusing the 5,120
single-workgroup `_1xN` dispatches (19.5 ms) and unrolling the `_256x2048` loop (53.0 ms, perhaps
30% recoverable) — is about 35 ms, i.e. **+1.9%**, against the roadmap's projected **+7 to +9%**.
That is a ~4x overstatement, and the 1.5 d estimate makes it the worst effort-to-return item in
Phase 3. **Demoted.**

**The corrected ranking for bit-exact decode work:**

| new rank | row | ms | why |
|---|---|---:|---|
| 1 | `moe_down_q2k_sum_decode` | 164.1 | 8 serialised round-trips + `subgroupAdd` per output row; roadmap's §3.2 hoist applies unchanged |
| 2 | `delta_net_decode_reg_f16` | 191.0 | largest addressable row; roadmap deferred it unmeasured |
| 3 | `moe_gate_up_decode` | 143.9 | §3.3 ROWS=4 |
| 4 | `add_rms_norm` | ~127 | fusion candidate, 10,560 dispatches |
| — | `dense_q8_0_d_*` (674.6 total) | | at ~360 GB/s roofline; bytes-only, Phase 4 |
| — | `matmul_f32_fast` | 72.5 | demoted from #1 |

`delta_net_decode_reg_f16` is the notable one: at 191.0 ms it is the single largest row that is not
already at the memory roofline, and the roadmap put it in the "measure before cutting" bucket at a
recorded 127 ms. It runs 3,960 dispatches over the 30 recurrent Gated-DeltaNet layers.

---

## Methodology finding: sequential runs drift thermally, and it is large enough to fake a result

Measuring §3.2(a) exposed a problem that invalidates naive before/after comparison on this box.

Four consecutive `Q36_VK_PROF_KERNEL=1` runs of the **same binary**, back to back
(`MEASURED_ON_BC250`):

| run | `moe_down_q2k_sum_decode` ms | `moe_gate_up_decode` ms (untouched) | gen t/s |
|---|---:|---:|---:|
| 1 | 167.0 | 141.8 | — |
| 2 | 168.7 | 143.0 | 74.66 |
| 3 | 170.8 | 144.7 | 74.26 |
| 4 | 175.3 | 148.8 | 73.00 |

Every row drifts monotonically upward, including kernels no edit touched, and decode falls from
the 84.3 t/s baseline to **73.0 t/s — a 13% loss with no code change at all**. GPU idle
temperature was 44 C at the start of the session and 63 C immediately after this block.

**Consequences:**

1. A sequential A-then-B comparison on this hardware can manufacture a regression of >10%, or hide
   an improvement of the same size, purely from run ordering. The roadmap's median-of-5 protocol
   does not protect against this, because it assumes noise is random rather than monotonic drift.
2. **My first reading of §3.2(a) was wrong.** I recorded 164.054 -> 167.033 ms and called the hoist
   a small regression. That delta is smaller than the drift measured above across identical
   binaries, so it was not a measurement of the change.
3. This is the same failure mode I flagged in `BENCH_27B_RESULTS.md`, where sustained clocks sat at
   580-680 MHz against a 1580-1760 MHz reference band. It is a property of the box, not of that
   benchmark.

**Protocol adopted for all A/B work from here:**

- **Interleave** variants (A,B,A,B,...) rather than running blocks, so drift is shared equally
  instead of loading onto whichever variant ran second.
- **Cool to a fixed gate** (GPU package <= 52 C, sampled from `hwmon1/temp1_input`) before every
  measured run, with the entry temperature recorded alongside the result.
- Swap the `.spv` file rather than rebuilding: shaders are loaded from `vulkan/` at runtime, so
  both variants run on a byte-identical binary and the comparison cannot pick up a compiler or
  link difference.
- Record start and end temperature per run and reject any pair whose entry temperatures differ by
  more than a few degrees.

---

## 3.2(a) — moe_down reduction hoist: REVERTED, the roadmap's +5-6% was negative

Implemented as specified: hold eight per-expert accumulators, reduce after the loop. Parity passed
17/17, so the change was correct — it just was not faster.

Interleaved A/B, cooled to a matched entry temperature before each run, `.spv` swapped on a
byte-identical binary (`MEASURED_ON_BC250`):

| variant | entry C | `moe_down_q2k_sum_decode` ms | gen t/s |
|---|---:|---:|---:|
| base | 56 | 156.738 | 76.32 |
| hoist | 56 | 160.081 | 76.77 |
| base | 55 | 154.136 | 76.95 |
| hoist | 56 | 159.272 | 76.76 |

Base ~155.4 ms, hoist ~159.7 ms: the hoist is **~2.8% slower** on the kernel row, consistently, in
both replicates. End-to-end decode is flat (76.3-77.0, within noise). Reverted.

Note the baseline here is ~155 ms against the 164.054 ms first recorded for the same code — the
original figure was heat-inflated by ~6%, which is why the cooled interleaved protocol was needed
to see the real sign of the effect.

**Why the premise was wrong:** the roadmap argued that eight `subgroupAdd` calls serialise eight
memory round-trips. But nothing in expert *u+1*'s loads depends on expert *u*'s reduction — only
lane 0's running `sum` does — so ACO was already free to overlap them. The hoist added an 8-float
accumulator array and register pressure, and bought no parallelism that was not already available.

---

## Finding: decode is dispatch-bound, not bandwidth-bound, in the rows the roadmap deferred

Per-dispatch costs from the profile (`MEASURED_ON_BC250`). This board has **40 CUs**.

| kernel | dispatches | groups/dispatch | us/dispatch | ms |
|---|---:|---:|---:|---:|
| `add_rms_norm` | 10,560 | **32** | 12.4 | 130.9 |
| `attn_decode_split` | 1,280 | **80** | 99.8 | 127.8 |
| `recur_conv_silu_decode` | 3,960 | **32** | 26.6 | 105.2 |
| `delta_net_decode_reg_f16` | 3,960 | 512 | 48.2 | 191.0 |

A 32-workgroup dispatch uses **less than one workgroup per CU**. `attn_decode_split` runs two per
CU. Together these rows are **364 ms, ~20% of decode**, executed on a nearly idle GPU.

`attn_decode_split` moves roughly 17 MB/token of KV cache, i.e. ~17 GB/s against the ~360 GB/s
roofline — 21x below it. It is unambiguously occupancy-bound.

Barrier elision is already hazard-based (`q36_vulkan.c:2803`), so these dispatches are separated by
genuine data dependencies; there is no free win from dropping barriers. The lever is fewer, wider
dispatches — fusion for the norm/conv rows, and a narrower split-K span for attention.

---

## 2.3 — glslc shim documented

`glslc:10` accepts and discards `-O`. Documented in the script why that is correct rather than a
bug: `glslangValidator` has no `-O` (only `-Os`/`-Od`), `spirv-opt -O` output was byte-identical to
the current `-Os` output on six of the eight hot shaders, and RADV's ACO backend re-schedules from
SPIR-V anyway. Explicit warning added against "fixing" it by switching to `/usr/bin/glslc`, which
would rewrite ~100 `.spv` blobs and put every one back through the parity gate for no measured gain.
Delta: 0, as the roadmap predicted.

---

## NEW — attention split-K span (not in the roadmap)

The largest *measured* win of the session, and it came from the dispatch-cost table rather than
from the roadmap.

`attn_decode_split` hardcoded a 512-key span. Decode dispatches `n_head * n_spans` workgroups, so
at ctx 2048 that is 16 * 5 = **80 workgroups on a 40-CU board** — two per CU. The kernel moves
~17 MB/token of KV cache, about **17 GB/s against a ~360 GB/s roofline**, i.e. 21x below it. It is
occupancy-bound, not bandwidth-bound.

Span is now a push constant (`Q36_VK_ATTN_SPAN`, default 512). Default path verified bit-exact,
17/17 frontiers.

Interleaved, temperature-gated at 55-56 C entry, ctx 2048 (`MEASURED_ON_BC250`):

| span | split ms | combine ms | attention total | gen t/s |
|---:|---:|---:|---:|---:|
| 512 | 119.939 | 10.302 | 130.24 | 77.37 |
| 256 | 90.046 | 11.888 | 101.93 | 78.24 |
| 128 | 76.067 | 14.870 | 90.94 | 79.07 |
| 512 | 121.563 | 10.384 | 131.95 | 76.87 |
| 256 | 90.043 | 11.924 | 101.97 | 78.00 |

Reproducible to three decimals (90.046 / 90.043). **+2.2% end-to-end at ctx 2048.**

**The important caveat is visible in the combine column:** it *rises* as the span narrows
(10.3 -> 14.9 ms), because more spans mean more partials to reduce. 4x the parallelism buys 1.43x
on the split, not 4x. Occupancy fixes on this board have a hard floor, and any estimate that
assumes linear scaling from added parallelism is wrong.

Not bit-exact: `attn_combine` reduces partials in span order, so a different span count regroups
that sum. This is a **reassociation, not a precision loss** — the same values added in a different
order. Left at 512 by default pending a decision on the parity bar.

---

## Failed hypotheses (recorded so they are not retried)

Four hypotheses were tested this session; **one held**. Recording the failures, because each was
plausible and each cost a measurement cycle.

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | `moe_down` reduction hoist gives +5-6% (roadmap §3.2) | **FALSE** | -2.8%, interleaved A/B |
| 2 | Decode is launch-latency bound; 32-group dispatches waste the GPU | **FALSE** | `delta_gates` does 32 groups in 1.11 us vs `add_rms_norm`'s 12.40 us -- same shape, 11x the time, so launch cost is ~1.1 us and fusion would save ~11.6 ms (+0.6%), not ~78 ms |
| 3 | `add_rms_norm` is slow because of `double` accumulation on a chip with poor FP64 | **FALSE** | base 129.414 / 127.508 ms vs fp32 128.394 / 125.897 ms — ~1%, within noise |
| 4 | `attn_decode_split` is occupancy-bound | **TRUE** | +2.2%, reproducible |

**The common error in 2 and 3:** reasoning from dispatch *counts* and source *inspection* instead
of measuring what the dispatch actually costs. Hypothesis 2 was additionally built on a pooling
artifact — `add_rms_norm`'s "32 groups/dispatch" is the average of prefill dispatches (1024 rows)
and decode dispatches (**1 row**), the exact same defect item 2.1 fixed for the f32 row.

**What is actually true about `add_rms_norm`:** decode runs it with `rows = 1`, and the shader is
`local_size_x = 256`, one workgroup per row. So each of ~10,240 decode calls occupies **one
workgroup on one CU out of 40**, moving ~40 KB in 12.4 us — about **3.2 GB/s**. The cost is the
shape, not the arithmetic. The fix is a multi-workgroup two-pass reduction, not fusion and not
lower precision.

---

## Why the engine sits at ~56% of the memory roofline

At 84.3 t/s and 2404 MB/token the engine achieves **~203 GB/s** against a ~360 GB/s roofline. That
average hides two very different populations (`MEASURED_ON_BC250` + arithmetic from the byte model):

| | share of decode time | achieved bandwidth |
|---|---:|---:|
| Q8_0 weight matvecs (79% of bytes) | ~44% | **~360 GB/s — at roofline** |
| everything else (21% of bytes) | ~56% | **~77 GB/s — 21% of roofline** |

This is the single most useful framing of the project:

- **~44% of decode time is already perfect** and cannot be improved by any engine work. Only
  reducing bytes moves it — that is the entire justification for Phase 4.
- **~56% of decode time runs far below roofline** because those kernels are not bandwidth-limited
  at all. That is the pool Phase 3 can address.

The two tracks are disjoint. Neither can reach the other's half.

---

## Phase 4 blocker discovered: the quantizer cannot re-type an existing quantized file

`gguf-tools/qwen36-quantize.c:948` rejects any source tensor that is not BF16/F16/F32, and
`classify_tensor` (`:874`) only assigns Q8_0 when the source is BF16/F16 — otherwise it preserves
the source type. So Recipe A **cannot** be produced from the current
`Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`; it requires the original full-precision
Qwen3.6-35B (~70 GB at BF16).

Free space on this box is **67 GB**, which is not enough for a 70 GB source plus an ~11 GB output.
Phase 4 therefore needs external storage before any of its code changes matter. This should have
been checked before Phase 4 was scheduled.

---

## add_rms_norm: 1024-thread workgroup — 42.5 ms saved, bit-exact

The largest measured win of the session.

Decode dispatches `add_rms_norm` with `rows = 1`, and the shader is one workgroup per row, so a
single workgroup owned the entire 2048-element row on **one CU out of 40** — about 10,240 times per
128 tokens, moving ~40 KB per call at roughly **3.2 GB/s** against a ~360 GB/s roofline.

At `local_size_x = 256` that workgroup is four wave64s, far too few to cover DRAM latency. Widening
to 1024 gives sixteen waves on the same CU.

Interleaved A/B, thermally gated (`MEASURED_ON_BC250`):

| threads | entry C | `add_rms_norm` ms | gen t/s |
|---:|---:|---:|---:|
| 256 | 59 | 129.048 | 75.62 |
| 1024 | 61 | **87.165** | 76.28 |
| 256 | 61 | 131.568 | 73.95 |
| 1024 | 61 | **88.365** | 75.42 |

Mean 130.3 -> 87.8 ms: **-32.6%, 42.5 ms off the decode path**, consistent across both replicates.

**Bit-exact in practice: 17/17 frontier logit dumps byte-identical.** Deepening the reduction tree
from 256 to 1024 leaves reassociates the partial sums, which is not bit-exact by construction, but
the accumulator is fp64 and only the final scale is rounded to fp32, so the difference lands far
below fp32 resolution. The fp64 accumulator is load-bearing for this and must not be "optimised"
to fp32 — the earlier fp32 experiment measured no speed gain anyway (129.4/127.5 vs 128.4/125.9 ms).

### Why the earlier hypotheses missed this

The correct diagnosis needed two corrections: the "32 groups/dispatch" figure was a prefill/decode
pooling artifact (decode is **1** group), and the cost was neither launch overhead (~1.1 us, from
`delta_gates`) nor fp64 arithmetic (measured flat). It was wave occupancy inside a single
workgroup, which none of the three earlier hypotheses named.

### Generalisation, and its limit

The same "too few waves" question applies to any decode kernel dispatching one workgroup. But the
fix direction depends on the dispatch shape, and is **not** "always widen":

| kernel | local_size | decode groups | right direction |
|---|---:|---:|---|
| `add_rms_norm` | 256 -> **1024** | 1 | widen: more waves on the one CU |
| `recur_conv_silu_decode` | 256 | ~32 | **narrow**: 32 workgroups leaves 8 of 40 CUs idle |
| `router_topk` | 256 | 1 per token | neither: top-k over exactly 256 experts, 1 thread each |

`recur_conv_silu_decode` has no shared memory and no barriers -- every thread maps to one channel
via `gl_GlobalInvocationID` -- so any workgroup width is bit-exact by construction there.

---

## recur_conv_silu_decode: workgroup width — NO WIN, rejected

Tested 64 / 256 / 1024 threads, interleaved and gated (`MEASURED_ON_BC250`):

| threads | rep 1 | rep 2 |
|---:|---:|---:|
| 256 (stock) | 105.6 | **137.7** |
| 64 | 105.9 | 105.0 |
| 1024 | 107.3 | 125.1 |

No width is faster; 64 is merely the most *consistent*. Unlike `add_rms_norm` this kernel already
dispatches ~32 workgroups of 4 waves (128 waves total), which is enough to keep the machine busy,
and it has no shared memory or barriers — each thread maps one channel via `gl_GlobalInvocationID`.
Its 105 ms is real work (the `exp()` in `sigmoid_stable`, the tap loop), not idle silicon. Kept at
256.

Note the 30% spread on the 256 and 1024 replicates: the 61 C thermal gate is no longer controlling
well, because the board's idle floor drifted from 44 C at session start to 58-60 C. Single numbers
from late in the session are worth less than early ones.

---

## delta_net_decode_reg: COLS 8 -> 32 — 12.5 ms, bit-exact, below end-to-end resolution

`subgroupClusteredAdd(sk, LANES)` forces `lane` to be the fast-varying index, so the state access
`s[mat + (r*LANES + lane)*N + j]` is contiguous in `j` but strided by `N` floats in `lane`. With
`COLS = 8`, a workgroup spans only 8 `j` values = 32 bytes of every 128-byte cache line — about 25%
memory efficiency, consistent with the ~40 GB/s measured against a ~360 GB/s roofline.

`COLS = 32` makes a workgroup consume whole cache lines. `local_size_x` becomes `8 * COLS = 256`
and the dispatch grid drops from `state_dim/8 = 16` to `state_dim/32 = 4` in y.

| COLS | entry C | `delta_net_decode_reg_f16` ms | gen t/s |
|---:|---:|---:|---:|
| 8 | 60 | 200.529 | 76.35 |
| 32 | 61 | **186.326** | 75.65 |
| 8 | 61 | 195.404 | 74.86 |
| 32 | 61 | **184.505** | 75.55 |

Mean 198.0 -> 185.4 ms: **-6.3%, 12.5 ms**, same direction in both replicates. Bit-exact, 17/17,
both hand-swapped and as built by default — each output element's arithmetic is unchanged, only
which thread computes it.

**Honest caveat: end-to-end throughput did not move** (75.61 vs 75.60 t/s). 12.5 ms of a ~1836 ms
budget predicts +0.68%, which is below what this board can resolve. Kept because the kernel-level
result is consistent and the change is free, but it should not be counted as measured throughput.

**Why the win is smaller than the coalescing argument suggests:** the state is loaded into registers
once per dispatch and the token loop then runs from registers, so the strided access is paid twice
per dispatch rather than per token. The inner loop was never the thing being fixed.

---

## END-TO-END RESULT: +1.8% to +2.3% decode, bit-exact

The only number that counts. Untouched `10c6965` binary in `/home/server/q36` against this branch's
build, alternating, thermally gated, three replicates each, ctx 2048 (`MEASURED_ON_BC250`):

| build | gen t/s | median | prefill t/s |
|---|---|---:|---|
| baseline `10c6965` | 81.58 / 80.39 / 80.32 | **80.39** | 609.19 / 559.29 / 611.03 |
| this branch | 82.34 / 82.23 / 82.05 | **82.23** | 595.09 / 608.29 / 612.35 |

**+1.80% on means, +2.29% on medians.** Passes the roadmap's strict acceptance rule: the
candidate's minimum (82.05) beats the baseline's maximum (81.58). Prefill is unchanged, as
intended — every change targets decode.

All of it is **bit-exact**: 17/17 frontier logits identical. No quality decision is required to
ship this.

Both figures sit below this morning's 84.3 t/s baseline because the board's idle floor drifted from
44 C to ~60 C over the session. Absolute numbers from different times of day are not comparable;
the interleaved design is what makes this A/B valid.

### Predicted vs measured, and why they differ

Per-kernel savings totalled 42.5 + 12.5 = 55 ms of a ~1836 ms budget, predicting **+3.1%**.
Measured **+1.8 to +2.3%**. The gap is expected: the profiler's `gpu_ms` figures sum to ~5.6% more
than wall-clock because dispatches overlap, so part of the saved kernel time was already hidden
behind other work. **Summed per-kernel deltas systematically overstate end-to-end gain** — a
reason to distrust any roadmap estimate built by adding kernel rows together, including the
original +25-30% projection.

### Available but not enabled

The attention split-K span (`Q36_VK_ATTN_SPAN=128`) adds a further **+2.2%** measured, for roughly
**+4.5% total**. It is left off because it is a reassociation that changes output text, and the
12-case eval run to check it was inconclusive (4 passed vs 4 passed, but 7 of 12 cases exhausted
their token budget — my budget was 3000 against a 16000 default, so the test mostly measured
whether cases finished). Settling it needs the full 92-case suite at the real budget.

---

## Remaining addressable pool (measured, not estimated)

Recovered so far: **55 ms of roughly 900 ms** of measured below-roofline decode time — about 6% of
the pool. The pool is not exhausted; these bandwidth figures are measured, and the gap to roofline
is real.

| kernel | ms | achieved GB/s | % of roofline | ms if at roofline |
|---|---:|---:|---:|---:|
| `moe_down_q2k_sum_decode` | 155 | 85 | **24%** | ~37 |
| `moe_gate_up_decode` | 144 | 153 | **43%** | ~61 |
| `matmul_f32_fast` | 72 | 54 | **15%** | ~11 |
| `attn_decode_split` (post-span) | 91 | ~25 | 7% | — |
| `recur_conv_silu_decode` | 105 | compute-bound | — | width tested, no win |
| `router_topk` | 57 | not examined | — | — |

The first three total **262 ms of a ~1836 ms budget (14%)**; full recovery would be **+17%**, half
recovery **+8%**. Neither is a prediction of what the work will yield — it is the size of the gap.

**Method note for the next round.** Six hypotheses were tested this session and two landed. Both
winners came from a *measured mechanism* with a number attached — wave count per CU for
`add_rms_norm`, cache-line utilisation for `delta_net`. All four failures came from reading source
or dispatch counts and reasoning about what *ought* to be slow. Start from the profiler's
per-dispatch numbers, not from the shader.

Next target: `moe_down_q2k_sum_decode`, 155 ms at 24% of roofline, 2048 workgroups per dispatch.
Note the §3.2 reduction hoist was already tried here and measured -2.8%, so the cost is elsewhere.

---

## moe_down wave32 forcing (roadmap §3.2b) — NO WIN, reverted

`moe_down_q2k_sum_decode` declares `local_size_x = 32` on a subgroup-64 device, so every workgroup
occupies a wave64 with half its lanes permanently inactive. The roadmap proposed forcing wave32
and flagged bit-exactness as "argued, not proven".

**Bit-exactness: proven, not argued — 17/17 frontier logits identical.** The `-0.0` concern about
`subgroupAdd` over inactive lanes does not materialise on this workload.

**Speed: nothing.** Interleaved A/B on one binary via a `Q36_VK_WAVE32` switch
(`MEASURED_ON_BC250`):

| wave32 | entry C | `moe_down` ms | gen t/s |
|---:|---:|---:|---:|
| on | 56 | 156.109 | 78.16 |
| off | 60 | 161.171 | 76.99 |
| on | 61 | 163.630 | 76.35 |
| off | 61 | 164.551 | 76.25 |

Only the last two share an entry temperature: **163.630 vs 164.551, a 0.6% difference — noise.**
The attractive 156.1 came from the one run that started 5 C cooler, which is precisely the confound
the thermal gate exists to expose. Reverted.

**Why the mechanism was wrong:** idle lanes cost ALU throughput, and this kernel is not ALU-bound.
The 32 active lanes issue the same memory requests either way, so halving the wave width changes
nothing about the traffic.

### So what does limit moe_down to 85 GB/s?

Not occupancy: 2048 workgroups of one wave each is ~51 waves per CU across 40 CUs. Not lane
utilisation, per above. Not the reduction structure — the §3.2(a) hoist was already measured at
**-2.8%**.

The remaining candidate is **access locality**: each workgroup gathers eight *routed* expert rows,
selected per token, at scattered offsets in the weight buffer (~164 bytes each at `base_u16 + i *
Q2K_U16`). Scattered short reads is the defining access pattern of a routed MoE and is not
obviously fixable at the kernel level. Three of its four plausible causes are now eliminated by
measurement; treat 85 GB/s as possibly near this kernel's practical ceiling rather than as 4x of
available headroom.

---

## matmul_f32_fast 64 -> 256 threads — +6.1% end-to-end, but NOT bit-exact. Opt-in.

The largest single speed result of the session, and the one that must not ship on my judgment alone.

Same shape as `add_rms_norm`: the `_1xN` router/shared-expert gate dispatches **one workgroup of 64
threads — a single wave64** — 5,120 times per 128 tokens. `n4 = in_dim/4 = 512`, so at 64 threads
each thread walks 8 iterations with nothing else in flight to cover the latency.

Interleaved A/B, thermally gated (`MEASURED_ON_BC250`):

| threads | entry C | `matmul_f32_fast` ms | gen t/s |
|---:|---:|---:|---:|
| 64 (stock) | 60 | 295.217 | 76.36 |
| **256** | 61 | **249.983** | **80.58** |
| 512 | 61 | 256.908 | 79.85 |
| 64 (stock) | 61 | 300.714 | 75.34 |
| **256** | 61 | **246.954** | **80.36** |
| 512 | 61 | 260.574 | 79.77 |

Mean 298.0 -> 248.5 ms (**-49.5 ms, -16.6%**) and gen **75.85 -> 80.47 t/s (+6.1%)**. 512 is worse
than 256, so 256 is the optimum, not merely "wider is better". Both 256 runs entered at equal or
hotter temperatures than the 64 runs, so thermal drift works *against* this result.

### Why it is gated

**Not bit-exact: 17/17 frontier logits differ.** At 64 threads `gl_NumSubgroups == 1` and the
shared-memory combine is skipped entirely; at 256 four subgroup partials are summed through
`sh_sum` in a different order. Unlike `add_rms_norm` and `delta_net`, there is no fp64 accumulator
to absorb the reassociation — this kernel accumulates in fp32.

**And it computes the MoE router gate.** Per the roadmap, `ffn_gate_inp` is the most
quality-sensitive tensor in an 8-of-256 MoE: a rounding flip changes *which experts a token is
routed to*, not the value of a logit. That is a categorically different risk from the other
non-exact change (the attention span), and it is why llama.cpp keeps this tensor in F32.

### How it is wired

`Q36_MFF_LOCAL` is a compile-time define; the Makefile builds both
`vulkan/matmul_f32_fast.spv` (64, default) and `vulkan/matmul_f32_fast_w256.spv` (256). The host
registers both kernels and selects the wide one only under `Q36_VK_F32_FAST_WIDE=1`.

Verified: default build **0/17 differ** (shipping path unchanged), opt-in **17/17 differ** (the
switch really does select the wide kernel — checked because an inert switch would look like a pass).

**Before enabling this, run the full 92-case eval at the default 16000-token budget, several
replicates, and compare pass rates — not logits.** The 12-case run used earlier in this session was
inconclusive and must not be used to clear it.

---

## FINAL RESULT

Three-way interleaved sweep, thermally gated, three replicates, ctx 2048 (`MEASURED_ON_BC250`).
Baseline is the untouched `10c6965` binary in `/home/server/q36`.

| config | gen t/s | median | prefill t/s | median |
|---|---|---:|---|---:|
| baseline `10c6965` | 81.39 / 80.31 / 80.16 | **80.31** | 607.83 / 605.98 / 591.93 | **605.98** |
| this branch, default | 82.32 / 81.84 / 82.15 | **82.15** | 564.82 / 605.22 / 613.75 | 605.22 |
| this branch, opt-ins on | 86.11 / 85.88 / 86.52 | **86.11** | 708.85 / 703.90 / 711.97 | **708.85** |

- **Default (bit-exact, ships as-is): +2.3% decode**, prefill unchanged. 17/17 frontier logits
  identical. No quality decision needed.
- **Opt-ins on** (`Q36_VK_F32_FAST_WIDE=1 Q36_VK_ATTN_SPAN=128`): **+7.2% decode and +17.0%
  prefill.** Both change numerics and need the eval.

### The prefill gain was not anticipated

`matmul_f32_fast` serves the prefill path too — `dense_f32f_p_256x2048` was 213.3 ms of it — so
widening the workgroup helps prefill far more than decode. **+17% prefill** matters
disproportionately for agentic workloads, which are prefill-dominated (long context re-read per
turn), and none of the roadmap's analysis considered prefill at all: every projection in it targets
decode. Prefill headroom is a genuinely unexplored track, and the one measurement that touched it
returned the largest single number of the session.

### Session tally

Seven hypotheses tested, **three landed**:

| hypothesis | result |
|---|---|
| `attn_decode_split` occupancy-bound | **+2.2%** (opt-in) |
| `add_rms_norm` too few waves per CU | **+42.5 ms, bit-exact** |
| `matmul_f32_fast` too few waves per CU | **+6.1% decode, +17% prefill** (opt-in) |
| `delta_net` cache-line utilisation | +12.5 ms kernel, below end-to-end resolution |
| roadmap §3.2a `moe_down` reduction hoist | **-2.8%**, reverted |
| roadmap §3.2b `moe_down` wave32 | noise at matched temperature, reverted |
| `add_rms_norm` fp64-bound | flat, rejected |
| `recur_conv_silu` width | no width wins, rejected |

**The pattern that predicts success:** every win came from a *measured* mechanism with a number
attached — waves per CU, cache-line bytes. Every failure came from reading source or dispatch
counts and reasoning about what ought to be slow. Two of the three wins are the same finding: a
decode kernel dispatching one workgroup whose `local_size` is too small to cover memory latency.
**That pattern is worth sweeping for systematically** — check every hot kernel's decode-path
workgroup count and width before trying anything cleverer.

---

## moe_gate_up_decode 64 -> 128 threads — NO WIN (-17%), reverted. And the rule that explains it.

| threads | entry C | `moe_gate_up_decode` ms | gen t/s |
|---:|---:|---:|---:|
| 64 | 56 | 138.172 | 78.07 |
| 128 | 60 | 167.149 | 75.64 |
| 64 | 61 | 144.579 | 76.08 |
| 128 | 61 | **169.860** | 74.80 |

At matched entry temperature, 128 threads is **17% slower**. Reverted.

### The refined rule

Widening a workgroup helps only when the dispatch has almost no workgroups:

| kernel | groups/dispatch | widening result |
|---|---:|---|
| `add_rms_norm` | **1** | **-33%** |
| `matmul_f32_fast` `_1xN` | **1** | **-16%**, +6.1% end-to-end |
| `delta_net_decode_reg` | 512 | -6% (via COLS, cache lines — different mechanism) |
| `recur_conv_silu_decode` | 32 | no width wins |
| `moe_gate_up_decode` | **4096** | **+17% slower** |

With 4096 workgroups the machine is already full, so a second subgroup buys no parallelism and
costs a cross-subgroup `sh_gate` combine plus an extra `barrier()` in **every** workgroup. The
overhead scales with workgroup count, which is exactly why it is catastrophic here and free where
there is only one workgroup.

**Corollary: the "too few waves" vein is mined out for decode.** The only genuinely starved decode
kernels were the two dispatching a single workgroup, and both are now fixed. Note also that the
pooled `groups/dispatch` column is only trustworthy for `*_decode` shaders — for any kernel used in
both phases it averages prefill and decode, the same artifact that hid `add_rms_norm`'s true shape.

Also worth recording: `moe_gate_up_decode`'s `shared float sh_gate[2]` with `acc += sh_gate[1]` is
hardcoded for exactly two subgroups. Any widening past 128 threads would **silently drop subgroups
2 and 3** and produce wrong results, not merely slow ones.

---

## Where the remaining headroom actually is

Decode occupancy work is done. What is left, in order of size:

1. **Prefill — entirely unexplored.** Every one of the roadmap's thirteen items targets decode, and
   no prefill kernel has been examined. Yet the two largest kernels in the whole profile live
   there: `matmul_q8_0_mm_f16` (**2173 ms**) and `moe_gate_up_gemm` (**1881 ms**), with
   `attn_prefill_qtile2` (1108 ms) and `moe_down_gemm` (1092 ms) behind them. The single change
   that touched prefill by accident returned **+17%**, the largest result of the session. For
   agentic workloads, which re-read long context every turn, this is the track that matters most.
2. **`moe_down_q2k_sum_decode` restructured to 64 threads.** It is the only hot kernel running
   *half* a wave (32 threads on a 64-wide machine). Forcing wave32 did nothing, but genuinely
   reshaping the algorithm to a full wave is a different change and was not attempted.
3. **Phase 4 requant** — still blocked on disk space for the ~70 GB source model.

---

## Prefill investigation: chunk 1024 is the measured optimum, and the 2048 cliff is explained

### Chunk size is already optimal (zero-code, bit-exact by construction)

ctx 4096, interleaved, gated (`MEASURED_ON_BC250`):

| chunk | prefill t/s |
|---:|---:|
| 512 | 450.19 / 420.75 |
| **1024** | **495.82 / 493.16** |
| 2048 | 179.85 / 177.04 |

**1024 is the optimum, not merely a safe value** — which independently validates the clamp shipped
this morning. 512 is worse, and 2048 collapses by 2.8x.

### The 2048 cliff: a host readback, not memory and not the kernels

GPU kernel time at 2048 is only 12-15% worse (`matmul_q8_0_mm_f16` 5039 -> 5625 ms,
`attn_prefill_qtile2` 5224 -> 6023 ms) while wall-clock prefill is 2.8x worse. The loss is host-side:

| | chunk 1024 | chunk 2048 |
|---|---:|---:|
| `submit_wait_moe_gate_up_selected` | — | **50,646 ms** |
| `submit_eager` | 15,533 ms | 79 ms |
| flushes | **16** | **170** |

`q36_gpu_matmul_iq_quant_scaled_tensor` and `q36_vk_moe_matvec` build MoE tiles on the GPU only
when `n_slot <= 4096`; otherwise they pull the expert selection back to the host, which forces a
flush and a fence wait. `n_slot = tokens * 8`, so chunk 2048 gives 16384 and always falls back.

### Two hypotheses tested against it, both refuted

**1. Allocator fallback past `Q36_VK_PRIVATE_LIMIT` (512 MB).** At chunk 1024 scratch sits within
3 MB of the cap, so this looked compelling. Made the limit tunable and tested:

| chunk | limit | prefill t/s | scratch/limit |
|---:|---:|---:|---|
| 1024 | 512 MB | 493.20 | 533/536 MB |
| 2048 | 512 MB | 177.24 | 438/536 MB |
| 2048 | 1536 MB | 188.96 | 572/1610 MB |
| 1024 | 1536 MB | 491.59 | 617/1610 MB |

The limit demonstrably took effect (scratch rose above the old cap) but bought only 177 -> 189.
**The control refutes it outright:** at chunk 1024, raising the limit moved 84 MB from arena to
private and throughput did not move (493.20 -> 491.59). The arena path is not slow. Reverted.

**2. Call-site guard too tight.** `q36_vk_moe_tiles_gpu()` validates `n_slot <= 8192` internally
while both call sites guarded at 4096 — half the supported range. Raised both to 8192, bit-exact
(17/17), and measured: chunk 1024 unchanged (that path was never taken there — stall row absent
both before and after), chunk 2048 unchanged at 177.98/176.38 with the 45,437 ms stall still
present, because 16384 exceeds the helper's own hard limit. Reverted.

### Conclusion: this direction is closed

Fixing the 2048 stall requires extending the GPU tile builder past 8192 slots — a real change, not
a constant. And it would not pay: GPU kernel time at 2048 is already 12-15% worse than at 1024, so
even with the stall fully removed, chunk 2048 would not beat chunk 1024. **Chunk 1024 stands as the
optimum and the prefill chunk track is finished.**

Remaining prefill headroom is inside the GEMMs themselves — `matmul_q8_0_mm_f16` (2173 ms) and
`moe_gate_up_gemm` (1881 ms) — which are properly tiled shared-memory kernels with hardcoded
BM/BN/BK. That is genuine GEMM tuning work, and `matmul_q8_0_mm_f16.comp:70` already records that
register prefetch was tried there and measured flat.

---

## CLI flags for the two opt-in fast paths

Both opt-ins now have `--` flags on every front end, resolving to the environment variables the
Vulkan backend already reads, so there is one implementation of the behaviour
(`q36_set_gpu_fast_path_env()` in `q36.c`, declared in `q36.h`).

| flag | effect | measured | exact? |
|---|---|---|---|
| `--f32-fast-wide` | 256-thread `matmul_f32_fast` | **+6.1% decode, +17% prefill** | **no** — reassociates the MoE router gate, can change expert selection |
| `--attn-span N` | split-K span in keys (default 512) | 128 gives **+2.2% decode** | **no** — regroups an ordered sum only |

Verified on `q36-bench` (`MEASURED_ON_BC250`):

- `--f32-fast-wide`: **17/17** frontier logits differ — the flag reaches the kernel selection.
- `--attn-span 128`: **16/17** differ, with frontier `002048` matching. That is the correct
  signature: the span only affects the decode split-K path, so the prefill frontier must be
  untouched.
- no flags: **0/17** differ — the default path is unchanged by the CLI work.
- `--attn-span` with no argument errors rather than silently defaulting.

Help text is in `q36_help.c` (shared by `q36`, `q36-server`, `q36-eval`, `q36-agent`) and in
`q36_bench.c`'s own usage block.

Note on `AGENT.md`: it says not to add permanent semantic variants behind flags. These are
deliberately *not* the release path — both default to off, the default build stays bit-exact, and
they exist so the 92-case eval can be run against them without setting environment variables by
hand. If the eval clears `--f32-fast-wide`, the right follow-up is to make it the default and
delete the flag rather than keep both paths.
