# R5 — decode attention at long context: occupancy, not redundancy

Diagnosis item.  No semantic source change; the only knob exercised is the
host-side `Q36_VK_ATTN_SPAN` (stages B/C).  Swift = `Swift-Qwen3.8-27B-IQ3_XXS.gguf`
(64 layers, 16 full-attention with 24 q heads / 4 kv heads, ratio 6, head_dim
256, K Q8_0 / V Q4_0), guard = `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`
(10 full-attention layers of 40, 16 q / 2 kv, ratio 8).  Fresh process per
(model, ctx), `tests/long_context_story_prompt.txt`, `--prefill-chunk 256`,
`--gen-tokens 64 --mtp-margin 0`, `Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1`.

## Method (why the profile is usable without a prefill subtraction)

`attn_decode_split` runs exactly once per full-attention layer per decode token,
and prefill uses `attn_prefill_fa_gqa{6,8}` instead, so its dispatch count *is*
the decode count: `L = split_disp / 64` (16 Swift, 10 guard).  `attn_combine`
carries `2 x chunks x L` prefill dispatches plus `1 x L` per decode token, so
`combine_ms/token = combine_gpu_ms * split_disp / combine_disp / 64`.  No
prefill subtraction is needed, which sidesteps the round-1 caveat that
decode-after-long-prefill tok/s swings 20-32 t/s with an identical binary.
`split_ms/call` is quoted as the primary quantity: it reproduces to 0.3-1.4%
where tok/s does not.

## Stage A — ctx 8192 / 16384 / 32768, span 512

`req GB/s` counts every query head's own reads; `uniq GB/s` counts the KV bytes
that are actually unique (6 or 8 heads share a row).

arm                  span   ctx   t/s  tok_ms  L  split/tok comb/tok attn%  (split% comb%)  spans   wg  split_ms/call comb_ms/call  req GB/s  uniq GB/s
swift-8192-r1          512  8192 20.15   49.63 16      4.56     0.63  10.4  ( 9.2  1.3)    17  408        0.2847       0.0394     266.2       44.4
swift-8192-r2          512  8192 19.97   50.08 16      4.59     0.63  10.4  ( 9.2  1.3)    17  408        0.2870       0.0396     264.1       44.0
swift-16384-r1         512 16384 18.48   54.11 16      8.53     1.02  17.7  (15.8  1.9)    33  792        0.5334       0.0639     283.7       47.3
swift-16384-r2         512 16384 18.41   54.32 16      8.58     1.02  17.7  (15.8  1.9)    33  792        0.5365       0.0638     282.0       47.0
swift-32768-r1         512 32768 15.40   64.94 16     16.35     1.67  27.7  (25.2  2.6)    65 1560        1.0221       0.1041     295.8       49.3
swift-32768-r2         512 32768 14.63   68.35 16     17.51     1.67  28.1  (25.6  2.4)    65 1560        1.0943       0.1041     276.2       46.0
guard-8192-r1          512  8192 72.39   13.81 10      2.03     0.27  16.6  (14.7  2.0)    17  272        0.2028       0.0270     249.2       31.1
guard-8192-r2          512  8192 73.94   13.52 10      1.98     0.27  16.7  (14.7  2.0)    17  272        0.1984       0.0269     254.7       31.8
guard-16384-r1         512 16384 64.09   15.60 10      3.59     0.44  25.8  (23.0  2.8)    33  528        0.3588       0.0438     281.1       35.1
guard-16384-r2         512 16384 64.57   15.49 10      3.59     0.44  26.0  (23.2  2.8)    33  528        0.3592       0.0435     280.8       35.1
guard-32768-r1         512 32768 52.26   19.14 10      6.59     0.70  38.1  (34.5  3.7)    65 1040        0.6592       0.0701     305.7       38.2
guard-32768-r2         512 32768 52.58   19.02 10      6.50     0.71  37.9  (34.2  3.7)    65 1040        0.6502       0.0707     309.9       38.7

Reproducibility: 0.3-1.4% on `split_ms/call` at 8K/16K and on the guard at 32K;
**Swift at 32K is the one arm that disagrees with itself** (1.0221 vs 1.0943
ms/call = 7%, 15.40 vs 14.63 t/s = 5%), which is the enter-temperature caution
round 1 recorded, now localized to the longest arm.  The *share* is still stable
(27.7 vs 28.1%) because both terms move together.

## Stages B/C — span sweep at ctx 16384 (the cheap experiment)

`Q36_VK_ATTN_SPAN` only changes how the same key range is cut into partials
(`n_head * n_spans` workgroups) and regrouped by `attn_combine`.  1 rep per arm,
fresh process, everything else identical to stage A; 512 is stage A's 2 reps.
`attn ms/token = L x split_ms/call + combine_ms/token`.

| model | span | spans | workgroups | split ms/call | vs 512 | ns per (wg,key) | combine ms/call | attn ms/token | vs 512 | decode t/s |
|-------|------|-------|-----------|---------------|--------|-----------------|-----------------|---------------|--------|-----------|
| swift |   128 |   129 |  3096 |   0.4961 |  -7.0% |  1.2519 |          0.0735 |    9.113 |  -4.6% | 18.61 |
| swift |   256 |    65 |  1560 |   0.5110 |  -4.2% |  1.2796 |          0.0670 |    9.249 |  -3.2% | 18.58 |
| swift |   512 |    33 |   792 |   0.5334 |        |  1.3153 |          0.0639 |    9.556 |        | 18.48 |
| swift |   512 |    33 |   792 |   0.5365 |  +0.6% |  1.3230 |          0.0638 |    9.604 |  +0.5% | 18.41 |
| swift |  1024 |    17 |   408 |   0.5710 |  +7.1% |  1.3667 |          0.0620 |   10.128 |  +6.0% | 18.35 |
| swift |  2048 |     9 |   216 |   0.6834 | +28.1% |  1.5448 |          0.0613 |   11.915 | +24.7% | 16.63 |
| swift |  4096 |     5 |   120 |   0.9095 | +70.5% |  1.8504 |          0.0608 |   15.524 | +62.5% | 15.91 |
| guard |   128 |   129 |  2064 |   0.3232 |  -9.9% |  1.2234 |          0.0533 |    3.765 |  -6.5% | 65.83 |
| guard |   256 |    65 |  1040 |   0.3731 |  +4.0% |  1.4015 |          0.0467 |    4.199 |  +4.3% | 61.97 |
| guard |   512 |    33 |   528 |   0.3588 |        |  1.3271 |          0.0438 |    4.025 |        | 64.09 |
| guard |   512 |    33 |   528 |   0.3592 |  +0.1% |  1.3288 |          0.0435 |    4.027 |  +0.0% | 64.57 |
| guard |  1024 |    17 |   272 |   0.4028 | +12.3% |  1.4461 |          0.0417 |    4.445 | +10.4% | 63.80 |
| guard |  2048 |     9 |   144 |   0.5002 | +39.4% |  1.6960 |          0.0410 |    5.412 | +34.4% | 59.68 |
| guard |  4096 |     5 |    80 |   0.7051 | +96.5% |  2.1517 |          0.0404 |    7.455 | +85.2% | 53.32 |

1. **Widening starves the grid monotonically.**  Per split call, 512 -> 1024 ->
   2048 -> 4096 spans costs +7.1 / +28.1 / +70.5% (Swift) and +12.3 / +39.4 /
   +96.5% (guard); per-key rate degrades 1.32 -> 1.85 ns (Swift) and 1.33 ->
   2.15 ns (guard).  792 Swift / 528 guard workgroups is at or above the knee;
   the 8K arm's 408 workgroups is ~4% below Swift's own.
2. **Narrowing buys a little, and only a little.**  Swift is monotone on the
   narrow side: 512 -> 256 -> 128 saves 4.2% / 7.0% of the split (per-key rate
   1.315 -> 1.280 -> 1.252 ns) = 3.2% / 4.6% of the attention term, which at a
   17.7% share is **+0.5% / +0.7% of the token** (t/s 18.48 -> 18.58 -> 18.61,
   inside the 0.3-1.4% noise band).  The guard saves 9.9% of the split at 128
   (6.5% of attention = ~1.7% of the token; t/s 64.09 -> 65.83 = +2.7%, one
   rep) while its **256 arm is an outlier**: it measures +4.0% on the split,
   where the trend from 128 and 512 puts it at ~-3%.  Single reps, so the
   narrow side is *not* established - and two independent things close it out
   beyond the single-rep caveat: it is not bit-exact (below), and **it trips the
   kernel gate**.  `Q36_VK_ATTN_SPAN=128 ./q36_test --vulkan-kernels` reports 3
   failures and `=256` reports 1, all at `tests/q36_test.c:4988` (the bitwise
   batch-vs-single-step-replay check), against 0 at 512, reproducibly.  The arms
   that fail are the ones whose causal range is 132 keys: at span 512 they are
   single-span and take the *fused* path, at 128 they become 2 spans and the
   split path is not `n_tok`-invariant for that shape.  So the shipped 512 is not
   merely the fastest setting measured - it is the setting the invariance
   property is verified at.  (FA arms unaffected: batch drift 9.1e-09 at span
   128/256 vs 1.0e-08 at 512.)
3. **`attn_combine` has a clean cost model, and it is the term that punishes
   narrow spans.**  Per call, over 5..129 spans:
   **Swift `0.06037 + 1.02e-4 x spans` ms**, **guard `0.04006 + 1.03e-4 x
   spans` ms** (max residual 0.00015 / 0.00030 ms; n=7 each).  The intercept
   scales with the head count (24 -> 0.060, 16 -> 0.040) and the slope is the
   same on both models, ~1 us per span.  So the combine is a fixed dispatch cost
   plus ~0.1 us per partial, exactly the term the stock comment predicted - it
   just is not large enough to change the verdict: at span 128 it costs 0.0735
   (Swift) / 0.0533 (guard) ms per call, +15% / +22% over span 512, which is why
   the net narrow-side gain is ~3-5% of the attention term instead of 7-10%.

## Diagnosis

1. **Not grid size, and not occupancy.**  At fixed span 512, `split_ms/call` is
   proportional to the workgroup count at a constant 25-28 us of CU time per
   workgroup while the count grows 408 -> 792 -> 1560 (1.94x, 1.97x Swift;
   272 -> 528 -> 1040 guard): the same marginal cost for workgroup 1 and
   workgroup 1560, i.e. 10 workgroups per CU (4 waves each) already fills the
   board at 8K.  The sweep brackets the knee from both sides (4% for 2x more
   workgroups on the narrow side, +7% for 2x fewer on the wide side).  The
   register/LDS probe agrees: `mmq_info vulkan/attn_decode_split.spv` = **SGPRs
   108, VGPRs 48, LDS 1536 B, Spilled 0, Subgroups per SIMD 20** - 48 VGPRs and
   1.5 KB of LDS cannot limit a 256-thread, 4-wave workgroup, and 20
   subgroups/SIMD is the driver's own ceiling.  `attn_combine.spv` = SGPRs 108,
   VGPRs 8, LDS 0, code 524 B, 40 subgroups/SIMD.  Round-1 P5's
   "occupancy-bound" reading of the 80-workgroup ctx-2048 case does not survive
   at 8K+.
2. **Not bandwidth.**  Requested bytes are 6x (Swift) / 8x (guard) the unique
   bytes, yet the requested rate is a flat **249-310 GB/s** across both models
   and all three contexts while the unique rate is **31-49 GB/s** and differs
   between the models at the same context (Swift 44-49, guard 31-39).  A
   DRAM-bound kernel would have to show the same unique rate for both; a
   request-bound kernel would have to show the 6-8x duplicate reads dominating.
   Round-1 P5 already established the re-read is served by L2 and that removing
   it (GQA grouping) wins only 1.01-1.14x past 8k, so this is consistent.
3. **What does: the per-element instruction stream of the tile loop - the R2
   lever again.**  Hand count for the live fast path (K Q8_0 / V Q4_0, TILE 64,
   256 threads) per key per workgroup: K `4 quads x k_q8_dot_pair` ~**580**
   (word-wide qs, 8 dwords + 8 shared reads per 64 dims), V `256 x v_q4` ~**2048**
   (per element: a dword load for the block scale, a dword load for one nibble,
   shift, and, int->float, mul, mad, add - `v_q4` at
   `vulkan/attn_decode_split.comp:158`), lane scan+exp ~**1500** (serial `tmax`
   over up to 64 keys per thread per tile, 13 `barrier()`s).  Total **4128
   thread-instr = 64.5 wave-instructions per key = 32.2 cycles of CU issue**
   (wave64 fp32 on a 4-SIMD CU) against **52.6 ns of CU time measured per key
   per workgroup** (40 CU x 0.5334 ms / (792 x 512) at Swift 16384) = 70 cycles
   at 1.34 GHz, 100 at the 1.9 GHz DPM state: **2.2-3.1x the bare issue bound**,
   i.e. an issue/stall-bound kernel carrying its own stalls, not a dependency
   hole (a latency-bound gather of this shape sits ~10x off).  ~86% of that
   stream is V extraction plus the lane scan; the K side already uses the
   wide-load form `k_q8_dot_pair` that the V side does not.
   `shader_isa attn_decode_split.spv`: 1697 static instructions, 135
   `s_waitcnt`, 133 `v_and_b32`, 79 `v_mac_f32`, 66 `buffer_load_dword`,
   64 `v_cvt_f32_i32_sdwa`, 47 `v_bfe_u32`, 34 `ds_read_b64`, 13 `s_barrier`.
4. **The combine is the one asymmetrically bad piece per unit of work.**  It is
   dispatched `n_head x n_tok x 1` = 24 (Swift) / 16 (guard) workgroups - under
   one per CU - with 524 B of code and a serial dependent-load loop (3 loads per
   span, no vector width, `exp`/`v_rcp_f32` per step).  It is 1.8-3.7% of the
   decode token, has the clean linear model above, and cannot recover more.

## Absolute cost and share of decode

| model | ctx | decode | split/token | combine/token | attention | share |
|-------|-----|--------|-------------|---------------|-----------|-------|
| Swift | 8192 | 49.6-50.1 ms (20.15/19.97 t/s) | 4.56-4.59 | 0.63 | 5.19 | **10.4%** |
| Swift | 16384 | 54.1-54.3 ms (18.48/18.41 t/s) | 8.53-8.58 | 1.02 | 9.55 | **17.7%** |
| Swift | 32768 | 64.9-68.4 ms (15.40/14.63 t/s) | 16.35-17.51 | 1.67 | 18.0-19.2 | **27.7-28.1%** |
| guard | 8192 | 13.5-13.8 ms (73.94/72.39 t/s) | 1.98-2.03 | 0.27 | 2.30 | **16.6-16.7%** |
| guard | 16384 | 15.5-15.6 ms (64.57/64.09 t/s) | 3.59 | 0.44 | 4.03 | **25.8-26.0%** |
| guard | 32768 | 19.0-19.1 ms (52.58/52.26 t/s) | 6.50-6.59 | 0.70 | 7.29 | **37.9-38.1%** |

The long-context decode slowdown is almost entirely attention.  From 8K to 32K
the Swift token grows 49.8 -> 66.6 ms (+16.8) and attention grows 5.19 -> 18.6
(+13.4) = **80% of it**; the guard grows 13.7 -> 19.1 ms (+5.4) and attention
2.30 -> 7.25 (+4.95) = **92% of it**.  A prefill-only view understates this:
prefill tok/s also falls with ctx (Swift 138.85 / 126.40 / 111.19, guard 569.39 /
488.47 / 391.98), so the whole long-context penalty is the same kernel twice.

Round-1's "guard at ctx 8192: 3.0 ms/tok for ~68 MB" reproduces here as
2.03 ms/tok for 63.4 MB unique bytes (31.1 GB/s).

## Verdict

**Diagnosed, no lever landed.**  The kernel is not latency-limited, not
occupancy-limited, not grid-limited and not bandwidth-limited: it is a
per-element instruction stream (V gathered one nibble at a time, plus a serial
per-key lane scan) issued at 2.2-3.1x its own minimum, and its total cost is
just keys x that per-key rate.  The one cheap experiment the item allowed - the
host-tunable `span_keys` - brackets the saturation knee and leaves **at most
~1% (Swift) / ~2-3% (guard, single rep) of the decode token** on the narrow side,
only by spending a non-bit-exact regrouping of the softmax reduction plus the
combine's ~1 us per extra partial - and it fails the `n_tok`-invariance check in
`./q36_test --vulkan-kernels` (3 failures at span 128, 1 at 256, 0 at 512).  Not
landed: 1-rep evidence, not bit-exact, and gated.  The rewrite the accounting names - vectorise the V extraction the way
`k_q8_dot_pair` already vectorises K, hoist `tmax` - is the same class R2
measured and **rejected** on `moe_iq2s_down_sum_decode` (fewer, wider loads
bought 0.06% on an instruction-issue-bound kernel), so it is not proposed.

Numeric note: `Q36_VK_ATTN_SPAN` is **not** bit-exact - it regroups the
online-softmax reduction in `attn_combine`, so the drift grows with the span
count - which is the other reason not to move the default on 1-rep evidence.
Any future use needs the NLL check, not the byte-identical one.

## Thermal/clock control

`thermal-samples.txt`.  The profiled arms run 3-10 min, so the throttling
question had to be answered: at 77-82 C the GPU sits at both 1.34 GHz and
1.9 GHz, 95 C is passed through at 1920 MHz, and idle drops to 500 MHz at 47 C
(vddgfx 856 mV, PPT 101-124 W), i.e. clock follows the DPM state for the
workload.  Kernel-side the longest arm is the healthiest: per-span split cost
0.01675 (8K) / 0.01616 (16K) / 0.01572 ms (32K, and 0.01683 for the r2 sample).
The long wall times are the profiler: `submit_eager` flushes 4904 / 7568 / 15696
and `submit_wait` 112 / 240 / 543 s, tracking ctx, with `gpu_ms` ~= wall
(99% GPU busy).

## Raw

`evidence/raw/r5-longctx-attn/` (`out/`, `outB/`, `diag/`, `scripts/`,
`ANALYSIS-stage*.txt`, `thermal-samples.txt`, this file)
