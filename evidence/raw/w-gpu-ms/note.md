# Real GPU ms on APEX-I-NanoPlus (BC-250 / RADV gfx1013) — 2026-10-27

Models: `Qwen3.6-35B-A3B.APEX-I-NanoPlus.gguf` (13.03 GiB),
`Cyber-Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf` (12.55 GiB). Both run on the
14 GiB board **without** an MTP support head. ctx 2048, 192 decode tokens,
prompt `prompt.txt` (4920 words), 3 interleaved pairs per arm.

## 0. Unlock state (gate for everything below)

| check | result | how |
|---|---|---|
| `bc250_cc_write_mode` | **3** | read directly from sysfs |
| active CUs | **40** (`num_se 2`, `num_cu 40`) | `RADV_DEBUG=info` (AGENT.md's sanctioned check; `journalctl` needs sudo, unavailable — approval prompts are disabled in this session) |
| co-scheduling knee | **exactly 40 WG** | `/tmp/cuprobe/cu_scale`, re-run on an idle GPU: 36→51.47, 38→48.47, 39→48.60, **40→48.65 ms flat**, 41→65.89 (+35.4%), 42/44/46/48 ≈ 66 ms |

Second-wave step at 41 WGs, so 40 CUs are co-schedulable. Nothing below is
gated by CU count.

## 1. Timestamp capability (logged before any measurement)

```
p36: timestamps period_ns=10.000000 computeAndGraphics=1 valid_bits=64
     queue_family=0 (graphics fallback) ace=0
```

`VK_QUERY_TYPE_TIMESTAMP`, results read with `VK_QUERY_RESULT_64_BIT`, pool
reset per bracket pair with `vkCmdResetQueryPool` (whole-pool reset hangs RADV,
see comment in `vk_05`). Brackets are per op-group (one pair per run of
consecutive dispatches of the same kernel), never per dispatch. In decode the
run lengths collapse to ~1, so in practice there are ~1081 brackets/token.

gpu_ms used to read 0.000 because the path is gated on `P36_VK_PROF_KERNEL=1`
and the capability was never logged. Added: capability log line, host-side WG
histogram (`P36_VK_PROF_WG_HIST=1`), report trigger for the histogram.

## 2. Instrumentation A/B (timestamps ON vs OFF, interleaved)

| arm | CyberTiel gen t/s | Qwen3.6 gen t/s |
|---|---|---|
| OFF (baseline) | 77.79 / 77.88 / 77.27 / 77.28 / 77.63 → **median 77.63** | 77.72 / 77.59 / 78.24 → **median 77.72** |
| ON (timestamps) | 66.61 / 66.81 / 66.93 / 66.76 → **median 66.79** | 66.40 / 66.72 / 66.76 → **median 66.72** |
| delta | **-14.0%** | **-14.2%** |

Attribution arms (CyberTiel, 2 reps each, disjoint ranges):

| env | gen t/s | verdict |
|---|---|---|
| none | 77.28, 77.63 | baseline |
| `P36_VK_PROF=1` (host clocks only) | 77.71, 77.89 | free |
| `P36_VK_PROF_WG_HIST=1` (host buckets only) | 77.98, 77.62 | free |
| `P36_VK_PROF_KERNEL=1` (timestamp queries) | 66.93, 66.76 | **costs 14%** |
| `P36_VK_PROF_KERNEL=1 P36_VK_PROF_SKIP=2600` (decode-only timing) | 66.84 | same 14% |

**Flagged: timestamp instrumentation contaminates the benchmark by -14%.**
Both numbers reported. The 14% is the per-bracket pipeline drain: 2.1 ms/token
over ~1081 brackets = **~1.9 us per serialization point**, which matches the
independently measured 2.3 us one-workgroup dispatch floor. All op-ms below come
from ON runs and are therefore an upper bound; shares are the usable output.

## 3. Decode GPU ms per op-group (CyberTiel; Qwen3.6 identical to 1%)

Decode-classified bracket sum **15.6 ms/token**, vs 15.0 ms/token wall for the
same run (1.04x over-count) and 12.87 ms/token for the uncontaminated baseline.
Prefill-classified rows sum to 0.84x of prefill wall because prefill's few huge
dispatches leave gaps between brackets.

| op | disp/tok | ms/tok | us/disp | % of decode |
|---|---:|---:|---:|---:|
| dense_q4k_decode | 240.0 | 3.15 | 13.2 | 20.2% |
| moe_iq3xxs_down_sum_decode | 36.0 | 1.95 | 54.5 | 12.5% |
| dense_q6k_decode | 11.0 | 1.39 | 126.5 | 8.9% |
| delta_net_decode | 30.6 | 1.07 | 35.1 | 6.9% |
| add_rms_norm_q8_k | 80.6 | 0.96 | 11.9 | 6.2% |
| attn_decode_split | 10.0 | 0.88 | 87.8 | 5.6% |
| moe_iq2s_iq2xxs_gate_up_decode | 22.0 | 0.78 | 36.2 | 5.0% |
| dense_q8_0_d_4096x2048 | 30.0 | 0.78 | 26.1 | 5.0% |
| q8_k_quant | 125.2 | 0.57 | 4.5 | 3.6% |
| recurrent_conv_silu_decode | 30.6 | 0.51 | 17.1 | 3.3% |
| moe_iq2s_gate_up_decode | 14.0 | 0.50 | 35.9 | 3.2% |
| dense_f32f_d_256x2048 | 40.0 | 0.48 | 12.0 | 3.1% |
| moe_q3k_down_sum_decode | 4.0 | 0.32 | 80.1 | 2.1% |
| delta_qkv_l2 | 30.6 | 0.25 | 8.2 | 1.6% |
| router_topk | 40.8 | 0.24 | 5.8 | 1.5% |
| moe_reduce | 0.8 | 0.24 | 282.8 | 1.5% |
| dense_f32f_d_1xN | 40.0 | 0.18 | 4.6 | 1.2% |
| moe_iq3xxs_gate_up_decode | 4.0 | 0.18 | 44.4 | 1.1% |
| recurrent_norm_gate_q8_k | 30.0 | 0.15 | 5.1 | 1.0% |
| moe_tail | 40.8 | 0.14 | 3.4 | 0.9% |

**MoE decode total: 3.76 ms/token = 24.1% of decode GPU time, in 72
dispatches/token.** Prefill profile (2048 tok, 849 t/s, 2574 dispatches = 1.26
disp/token) is 26.4% `dense_kquant_mmq_q4k` + 41.5% MoE `*_gemm` rows; nothing
to fix there.

## 4. Dispatches per token

| phase | dispatches/token | ms/token wall |
|---|---:|---:|
| prefill | **1.26** (2574 / 2048) | 2.41 (849 t/s) |
| decode | **1001.8** (192,352 / 192) | 12.87 (77.63 t/s) |

794x asymmetry between phases, same machine, same kernels.

## 5. Decode workgroup histogram (192,352 dispatches = 192 tokens)

| bucket | dispatches | per token | share |
|---|---:|---:|---:|
| <5 WG | 54,306 | 282.8 | 28.3% |
| 5–20 WG | 38,978 | 203.0 | 20.3% |
| 20–40 WG | 17,400 | 90.6 | 9.1% |
| >40 WG | 81,276 | 423.3 | 42.3% |

**48.6% of decode dispatches use ≤20 of 40 CUs; 57.7% use ≤40.** Ops that are
majority ≤20 WG account for 486 dispatches/token but only **2.72 ms/token
(17%)** of GPU time — they cost ~5.9 us each, close to the 1.9 us serialization
floor plus a short body.

## 6. Method files

`run_ab.sh` (A/B + prefill), `run_arms.sh` (contamination attribution),
`summ.py` / `parse.py` (op rows, buckets), `tiel_*.log|csv`, `qwen_*.log|csv`,
`tiel_timeline.txt` (`P36_VK_PROF_TIMELINE`).

## 7. Shape pass (`P36_VK_PROF_SHAPE=1`), and the down_sum anomaly (2026-10-27)

Shape names encode `in x out x n_tok`, so WG count per dispatch is exact
(`groups/dispatches`). Decode rows only:

| shape | disp/tok | WG/disp | µs/disp | ms/tok |
|---|---:|---:|---:|---:|
| moe_iq3xxs_down_sum_decode | 36.0 | 2048 | 54.5 | 1.96 |
| moe_iq2s_iq2xxs_gate_up_decode | 22.0 | 4096 | 36.8 | 0.80 |
| moe_iq2s_gate_up_decode | 14.0 | 4096 | 36.3 | 0.50 |
| moe_iq3xxs_gate_up_decode | 4.0 | 4096 | 44.6 | 0.18 |
| dense_q4k_decode `2048x8192` | 40.0 | 2048 | 37.8 | 1.51 |
| dense_q4k_decode `4096x2048` | 30.0 | 512 | 26.3 | 0.79 |
| dense_q4k_decode `2048x512` | 100.0 | 128 | 4.6 | 0.46 |
| dense_q4k_decode `512x2048` | 40.0 | 512 | 6.7 | 0.27 |
| dense_q4k_decode `2048x32` | 30.0 | **8** | 5.7 | 0.17 |
| delta_net_decode | 30.6 | 128 | 35.1 | 1.07 |
| recurrent_conv_silu_decode | 30.6 | **32** | 17.1 | 0.53 |
| add_rms_norm_q8_k | 80.6 | 147 | 11.9 | 0.96 |
| q8_k_quant | 125.2 | 1498 | 4.5 | 0.57 |
| dense_f32f_d_1xN | 40.0 | **1** | 4.6 | 0.19 |

### The anomaly is a throughput gap, not a grid gap

`down_sum` launches **2048 WGs** (51 per CU — not starved) and still runs at
**58 GB/s** (3.06 MiB per dispatch / 54.5 µs), while the IQ3_XXS *gate_up*
build — same quant format, same table lookups, same expert-8 batch — moves
**6.42 MiB in 44.6 µs = 144 GB/s**. Fewer bytes, more time: 2.5x worse.

What differs, from the source:

| | down_sum | gate_up |
|---|---|---|
| local_size | **32** (half of the 64-wide wave, or wave32 with 32 lanes) | 64 |
| grid | 1D, one WG per output row (2048) | 2D, one WG per (row, expert) (4096) |
| experts per WG | **8, serial**, each ending in `subgroupAdd` | 1 |
| blocks per thread | 2 (in_dim 512) | 8 (in_dim 2048) |
| threads per dispatch | 65,536 | 262,144 |

Two experiments, both bit-exact (logits md5 `e7c953fc…` unchanged):

| change | result | verdict |
|---|---|---|
| force wave32 on `moe_down_q2k_sum_decode` (`P36_VK_WAVE32_MOE_DOWN=1`) | 77.57 → 77.75 t/s median, +0.2%, ranges overlap | **null**; env-gated and left OFF |
| hoist the 8 `subgroupAdd` out of the chain into `part[8]`, apply route after | 77.75 → 76.68 t/s median, **-1.4%**, ranges disjoint | **reverted** (the 8-slot array spills to scratch) |

So the wave width is not the limiter and the reduction chain is not the
limiter. What is left is thread/wave count: 4x fewer lanes in flight than the
sibling that runs 2.5x faster. A 62,080-WG 32-thread kernel on this board
(`dense_q6k_decode_2048x248320_n1`) sustains 348 GB/s, so the format and the
32-thread width are both capable of far more than 58 GB/s.

### Proposed change (not implemented — breaks bit-exactness, needs its own go/no-go)

Split the expert loop across the grid, the way the **prefill** path already
does (`moe_iq3xxs_down_gemm` runs 16,384 WGs = 2048 rows x 8 experts, then
`moe_reduce`):

* `moe_down_q2k_sum_decode`: 2D grid `(out_dim, n_used)`, each WG owns one
  (row, expert), writes `acc * scales[expert]` to a per-expert partial buffer.
  Threads per dispatch 65,536 -> 524,288, no serial chain, no per-expert
  reduction.
* one `moe_reduce`-style pass applies `route[u]` and sums 8 partials
  (+1 dispatch per layer = +36/token, ~0.07 ms/token).

Ceiling: at the 144-232 GB/s its siblings reach, the 3.06 MiB/dispatch becomes
21-14 µs instead of 54.5, i.e. **1.96 -> 0.74-0.5 ms/token, 6.5-9% of the
12.9 ms decode token**. Cost: reduction order changes (logits drift, not
bit-exact), extra 64 KB of partial traffic per layer, one more dispatch.
`reconsider_if`: someone accepts a non-bit-exact MoE decode sum, or the
partials are reduced in expert order by a kernel that applies `route[]` with
the same `fma32` sequence (then it can still be bit-exact).

## 8. Split expert grid + ordered reduce — built, bit-exact, +3.1%, reverted (2026-10-27)

Per the decision to try the ordered reduce: `moe_down_q2k_sum_decode` gained a
`P36_MOE_DOWN_SPLIT` compile-time variant (grid `(out_dim, n_used)`, one
(row, expert) per WG, writes `acc*scales[expert]` to a per-expert partial
buffer on binding 5) plus `moe_down_route_reduce.comp`, which replays the exact
fused accumulation `sum = fma32(part[u], route[u], sum)` for u = 0..n_used-1
with the identical software `fma32`.

**Parity: BIT-EXACT.** `frontier_002048.logits.json` md5 `e7c953fccb64ac6285b5eb4aea48604a`
for split ON, split OFF and the pre-change binary.

A/B, 3 interleaved pairs, ctx 2048, 192 tokens:

| arm | gen t/s | median |
|---|---|---:|
| fused (default) | 76.52 / 77.46 / 77.36 | 77.36 |
| split + reduce | 79.11 / 79.82 / 79.79 | **79.79** |

**+3.1%**, ranges disjoint. Cost accounting (instrumented, 24 tok):

| | disp/tok | µs/disp | ms/tok | achieved |
|---|---:|---:|---:|---:|
| fused `moe_iq3xxs_down_sum_decode` | 36 | 54.5 | 1.96 | 58 GB/s |
| split `moe_iq3xxs_down_split_decode` | 36 | 38.3 | 1.379 | **84 GB/s** |
| `moe_down_route_reduce` | 40 | 2.9 | 0.118 | — |
| pair | | | **1.497** | 3.6% of the 12.9 ms token |

Verdict: below the 5-6% bar, so the variant was **removed** (AGENT.md forbids
permanent semantic variants behind flags). Default path untouched; tree back
to baseline (77.59 t/s confirmed after revert).

Why it stopped at 84 GB/s and not the sibling's 144: splitting by expert makes
each workgroup read a single 196 B burst (2 blocks x 98 B), eight times more
workgroups with one eighth the bytes each, so it trades the serial chain for
short-stream latency. The 64-thread gate/up sibling pulls 1.2 KB contiguous per
workgroup. Next variant to try, if this line is reopened: 4 rows per workgroup
(784 B contiguous, 4096 workgroups) on top of the split grid.

## 9. 4-row split grid — bit-exact, +2.0%, bandwidth flat, reverted (2026-10-27)

Same ordered-reduce path as §8, but grid.x packs `P36_MOE_DOWN_ROWS = 4`
consecutive output rows per workgroup: 512 x 8 = **4096 workgroups** instead of
16384, 784 B contiguous per (tile, expert).

**Parity: BIT-EXACT.** md5 `e7c953fccb64ac6285b5eb4aea48604a` for split ON,
OFF and the pre-change binary.

| arm | gen t/s | median |
|---|---|---:|
| fused (default) | 76.09 / 78.10 / 77.52 | 77.52 |
| split 4-row + reduce | 79.41 / 79.08 / 78.51 | **79.08** |

**+2.0%** (1-row split was +3.1%). Instrumented, 24 tok:

| | WG/disp | µs/disp | ms/tok | GB/s |
|---|---:|---:|---:|---:|
| fused `moe_iq3xxs_down_sum_decode` | 2048 | 55.4 | 1.993 | 58 |
| split 1-row (§8) | 16384 | 38.3 | 1.379 | 84 |
| split 4-row | **4096** | 40.0 | 1.440 | **80** |
| `moe_down_route_reduce` | — | 3.0 | 0.119 | — |
| pair total | | | 1.559 | |

Row packing did **not** raise bandwidth (80 vs 84 GB/s, slightly worse), so the
196 B burst was not the limiter after all. Both split tilings stall at ~80-84
GB/s while the 64-thread gate/up sibling reaches 144, which points at a
kernel-internal limit in the 512-wide IQ3_XXS down row (2 blocks of work per
workgroup, i.e. the per-workgroup fixed cost - selected[]/scales load, pointer
setup, subgroupAdd, partial store - is amortised over 8x fewer elements than
gate/up's 2 x 2048). Below the bar, reverted; default fused path unchanged
(post-revert baseline 77.26 t/s, no split references left in the tree).
