# A/B — host query-pool drain removal (`experiment/w6-hostdrain`)

**Date:** 2026-09-17 · **Verdict: REJECT as a speed lever** (inert in production;
reclassified as a profiling-tooling fix)
**Raw:** `raw/ab-drain-swift.csv`, `raw/ab-drain-moe.csv`, `raw/ab-drain-verdicts.txt`

- **A** = `/home/server/q36-opt-27b/q36-bench` built from `11bb345` (tree clean).
- **B** = `/home/server/q36-wt/w6-drain/q36-bench`, one hunk off the same commit.
- Caveat: **9 other `.spv` and our four other binaries are all default `make`** — same flavor on both sides.

## The change

`q36_vulkan.c:2981-2983` (A) removed in B — the mid-dispatch query-pool drain:

```c
if (q36_vk.prof_kernel && q36_vk.query_pool && q36_vk_prof_query_count + 2u > Q36_VK_PROF_QUERY_CAP) {
    if (!q36_vk_flush_reason_unlocked("submit_wait_query_pool")) return 0;
}
```

Replaced by a comment. Safety was verified in source **before** building: the
timestamp write further down self-limits (`+ 2u <= Q36_VK_PROF_QUERY_CAP`, and the
counter only advances when a pair is really emitted), so deleting the drain
cannot overflow the pool. The diff is exactly that one hunk (verified with
`diff -u` against A's source; `q36_vulkan.o` 17:58 → binary 18:05).

Why it looked like a win: the per-kernel profile showed `dense_q4k_decode` with
**55.751 ms** of record time for 592 dispatches ≈ **94 µs/dispatch**, when
`record_ms` for the kernel's own work is ~2.6 ms. The rest was this synchronous
4-fence drain (`q36_vk_flush_unlocked:2049` → ring-slot waits `:2053-2056`).

## Result — two models, both FAIL

```
tests/bench_ab.sh <A> <B> <MODEL> 7      # interleaved, ctx 1024, greedy gen 16
```

**Dense target — Swift-Qwen3.8-27B-IQ3_XXS:**

| arm | n | prefill median | MAD | decode median | MAD |
|---|---|---|---|---|---|
| A baseline `11bb345` | 7 | 170.91 | 0.650 | 18.60 | 0.100 |
| B drain-removed | 7 | 171.68 | 0.550 | 18.68 | 0.110 |

**B vs A: prefill +0.45%, decode +0.43% (gate +1.50%) → FAIL.**
+0.45% sits inside A's own 0.65 MAD. Nothing to accept.

**No-regression guard — Huihui Qwen3.6-35B-A3B (MoE), same protocol:**

| arm | n | prefill median | MAD | decode median | MAD |
|---|---|---|---|---|---|
| A baseline `11bb345` | 7 | 712.75 | 6.660 | 88.13 | 0.290 |
| B drain-removed | 7 | 711.95 | 9.330 | 87.91 | 0.180 |

**B vs A: prefill −0.11%, decode −0.25% → FAIL.** Both arms inside the MoE
prefill MAD of 6.7–9.3 t/s, consistent with the campaign's 3.4% MoE noise band.

## Why this is a null *by construction*, not a small win

`q36_vk.prof_kernel` is set **only** when `Q36_VK_PROF_KERNEL` is present in the
environment (`q36_vulkan.c:3465-3474`), and the removed block was gated on
exactly that flag. `tests/bench_ab.sh` sets no profiler env, so in **both arms of
this A/B the drained block could never execute** — the two binaries are
behaviourally identical in the configuration that was measured. The experiment as
designed cannot observe the change; it only re-measured the noise floor on two
models (incidentally confirming it twice: +0.45% and −0.11% ≈ one MAD).

Consequence: **the change is inert in production.** Its only effect is on timings
produced *by the profiler itself*. Correct classification is a
profiling-tooling fix, not a speed lever — and per the campaign rule (accept only
measured end-to-end gains), it does not get committed as a speed change. It is
recorded here so a future session does not rediscover the "94 µs/dispatch"
anomaly and spend a build + A/B cycle on it again.

Lesson for the harness: **check which build/runtime flags gate a code path before
designing its A/B.** A change gated on a profiler flag must be A/B'd with that
flag on, or not A/B'd for speed at all.

## Harness bug found and fixed in the same pass

`tests/bench_ab.sh`'s verdict `awk` passed the SUBSEP-joined multipart arrays
(`p[lab,k]`, `d[lab,k]`) straight into `med()`, which indexes its argument with
plain integers (`arr[i]`) — so every read was of an uninitialized element and
`med()` returned `0`. The summary therefore printed `prefill_med 0.00`,
`decode_med 0.00` and a fake `+0.00%` for **every run**, i.e. `verdict: FAIL
(prefill)` regardless of the data. MAD looked correct only because `mad_of()`
builds its own flat array.

Fixed by flattening into the copy arrays before calling `med()` (line 114-126).
Verified by re-running the patched block over the raw CSV of **both** runs: it
now prints 170.91/171.68/+0.45% and 712.75/711.95/−0.11%, identical to
independent `statistics.median` computations over the same files. Any A/B verdict
produced by this script before this fix is **unusable** and must be recomputed
from its CSV.

### Two traps this experiment set off — check for them before trusting a summary

1. **A patched script does not patch a running one.** The MoE run's own
   `/tmp/ab-drain-moe.summary` printed `prefill_med 0.00` for both arms because
   the `med()` fix was written to `tests/bench_ab.sh` at **18:17:33, while that
   A/B was still running**, and a `bash` process keeps reading the script it
   launched. Its summary was discarded; the numbers above come from re-running
   the fixed block over `raw/ab-drain-moe.csv`. Compare the script's mtime
   against the run's start before trusting any summary file.
2. **The broken code is recognisable by shape:** `prefill_med 0.00` with a
   plausible number in the MAD column. That is `mad_of(arr, n, 0)`, which
   degenerates into the median of the raw values — a real MAD here is always
   small (0.1–10 t/s).
