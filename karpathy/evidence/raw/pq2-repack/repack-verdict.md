# PQ2_0 physical repack — probe evidence and verdict

Session: BC-250 / RDNA2 Wave64, 40 CU. Branch `main` @ `6b75abb`, nothing committed.
Harness: `karpathy/evidence/raw/pq2-repack/probe_mv3.c` + `mv3.comp` (production decode
geometry: 4352 workgroups x 64 lanes x ROWS=4 per dispatch, window walked through a
DRAM-sized buffer). GB/s is always charged at `rows x blocks x 2 x 34` (the production
PQ2_0 stride) for every variant, SoA included.

## 1. What production actually streams (denominator, measured)

`gguf_cols.py` on `TERNARY-BONSAI-2-27B-DERISKED-PQ2_0.gguf` — byte-weighted row widths:

| cols (ne0) | units | row_bytes | MiB | share |
|---|---|---|---|---|
| 5120 | 20 | 1360 | 4851.64 | 70.71% |
| 17408 | 68 | 4624 | 1445.00 | 21.06% |
| 6144 | 24 | 1632 | 510.00 | 7.43% |

`token_embd.weight` (322.07 MiB) is gathered, not streamed. BF16 (45.00) + F32 (10.09) route to
`matmul_bf16` / other kernels, not to `dense_extra_decode_pq2_0`. Correct denominator:

    streamed per token = 6806.64 - 322.07 = 6484.57 MiB = 6799.6 MB

Live kernel, fresh, this session (`Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_OP=1`, ctx 512, c128):

    dense_extra_decode_pq2_0    51330 dispatches   2763.402 ms  ->  401.0 disp/tok, 21.589 ms/tok
    6799.6 MB / 0.021589 s = 315.0 GB/s
    ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps
    512,512,201.97,128,32.48

`51330/128 = 401.0` = 48 delta x 6 kernels + 16 full x 7 kernels + 1 output. Every
non-embedding PQ2_0 matvec at decode is this one kernel.

## 2. Clocks are NOT stable — absolutes do not transfer, ratios do

`/sys/class/drm/card0/device/pp_dpm_sclk` = `0: 1000Mhz / 1: 21Mhz * / 2: 2000Mhz`,
`pp_dpm_mclk` = `0: 450Mhz *`, `power_dpm_force_performance_level = auto`, no lock file.
For identical binary, geometry and variant the session measured:

    v0  272.4 / 228.5 (median) / 392.0 / 395.6 (best) GB/s   -> up to 2x spread across runs

Within a single run the variant ratios reproduce across three independent runs
(two different clock regimes), while the absolutes move by 70%:

| ratio | run A (med) | run C (best) | run D (best) |
|---|---|---|---|
| v1/v0 | 1.079 | 1.082 | 1.081 |
| v2/v0 | 1.083 | 1.090 | 1.083 |
| v4/v0 | 1.012 | 1.012 | 1.012 |
| v3/v0 | ~1.16 | 1.179 | 1.159 |

=> only within-run ratios are used below. This is the same wall the ledger records
(`AlreadyTried.md:1242-1250`, per-kernel `gpu_ms` not comparable across runs, +-12%;
here 2x).

## 3. Probe methodology checks

* **Per-dispatch-pair timestamps vs one span** — production brackets each dispatch with
  `COMPUTE_SHADER_BIT` -> `BOTTOM_OF_PIPE` (`q36_vulkan.c:3185-3190`). The probe now reports
  both. Span and production-convention pair-sum agree to 0.4% (228.5 vs 229.2 GB/s;
  295.6-span vs 296.4-pair). Inter-dispatch bubbles are NOT hiding in the probe number —
  the "probe absorbs launch gaps" hypothesis is refuted.
* **Probe v0 is production ISA, not SPIR-V.** `RADV_DEBUG=asm` on `mv3_v0.spv` shows
  `buffer_load_dwordx3 v[9:11]` + 2x `v_alignbit_b32` + 1x `buffer_load_dword` (scale) —
  exactly the fused form archived for production (`AlreadyTried.md:33-45`,
  `logs/bc250-sustained-20260918/isa/dense_extra_decode_q2_0.baseline.asm`). So the v0
  baseline is a faithful production-ISA model, not an artificially slow one.
* ISA diff, `mv3_v2.spv` (SoA, aligned): `buffer_load_dwordx4` (aligned 16 B), **0**
  `v_alignbit_b32`. Repack removes 2 ALU ops per lane-row and makes the weight load 16 B
  aligned.
* Small buffers lie: a 7 MB buffer is L2-resident and reported 453 GB/s. All numbers here
  use DRAM-sized aggregates.

## 4. Measured variants (production geometry, medians of 6 interleaved reps)

| variant | cols 5120 (69.3%) | cols 17408 (22.1%) | cols 6144 (7.8%) |
|---|---|---|---|
| v0 production 34 B, scale inline | 272.4 | 356.9 | 270.1 |
| v1 SoA payload 32 B, uvec2/lane | 294.0 | 359.2 | 291.7 |
| v2 SoA payload 32 B, uvec4/lane | 295.1 | 368.0 | 291.8 |
| v3 = v2 minus scale+activation loads (ablation) | 316.3 | 424.4 | 313.7 |
| v4 = v0 minus scale+activation loads (ablation) | 275.8 | 363.6 | 271.3 |
| v7/v8 in-row scales + pad to 32 B | 275.4 / 278.2 | - / 365.8 | 273.8 / 276.2 |

Byte-weighted (69.26 / 22.10 / 7.80%):

    v0  288.6 GB/s    baseline
    v1  305.7         +5.9%
    v2  308.5         +6.9%   <- best implementable layout
    v3  337.3        +16.9%   <- ceiling, NOT implementable
    v8  214.2        -25.8%   <- in-row scales + padding loses badly

## 5. Why this is already the end of the road on layout

For the dominant 1360 B row, v2 with its mandatory scale load is within **0.8%** of the
byte-limited ceiling implied by its own ablation:

    payload 1280 B/row is 94.1% of the 1360 B/row that must move
    ceiling-implied v2 = v3 / (1360/1280) = 316.3 / 1.0625 = 297.7 GB/s
    measured v2                                                    = 295.1 GB/s

A dense SoA pair of planes (payload 1280 B rows = exactly 20 cachelines, scale plane =
contiguous 80 B rows) has zero partial-line waste; the only remaining cost is the scale
bytes themselves, which are charged and are irreducible without requantizing. The same
check at cols 6144: ceiling 313.7/1.0625 = 295.2 vs measured 291.8 (98.8%).
=> SoA is traffic-optimal for the 34 B/block format. There is no layout headroom left.

## 6. Verdict

**REJECT — do not integrate. The layout can beat the current one, but not by enough.**

1. **Can physical repacking beat the current layout?** Yes, by a measurable but small margin:
   +8.3% at the dominant 1360 B rows, +3.1% at 4624 B rows, +8.0% at 1632 B rows;
   byte-weighted **+6.9%**. It cannot reach the ≥360 GB/s integration gate.
2. **Winning layout:** SoA, two dense planes per tensor — payload 32 B aligned blocks
   (uvec2/uvec4 per lane) + a separate contiguous fp16 scale plane. Zero padding, zero
   extra bytes. In-row scales (v7/v8) lose 6-26% to partial-line waste and are rejected.
3. **Decode:** 32.48 t/s -> ~34.0 t/s (kernel 21.59 -> 20.19 ms/token). Not the 37-39 target.
4. **Prefill:** unchanged (nothing integrated). Baseline re-measured 201.97 t/s. The mmq
   kernel is 4780 ms / 54% of prefill GPU time and would need the same two-plane addressing
   for no upside.
5. **Effective bandwidth:** 315.0 GB/s -> ~336.7 GB/s. Non-implementable ceiling 368 GB/s.
6. **Remaining bottleneck:** the format's own bytes — 34 B per 128 weights (2 bits/weight
   payload + 2 B fp16 scale), i.e. 1360 B per row for 69% of the stream. At the dominant
   width SoA is already within 0.8% of the byte-limited ceiling, so the loss is not
   address generation, not coalescing, not instruction count: it is bytes over a DRAM read
   path whose ceiling this probe itself showed can exceed 437 GB/s (455.6 measured on an
   ideal aligned pattern). The ledger's PTQ1_0 comparison points the same way: its 28 B per
   128 weights (-17.6% bytes vs PQ2_0) is what buys it ~34 t/s, not its layout.
7. **Is further PQ2_0 optimization worth continuing?** No. Layout is exhausted at +6.9%
   weighted and the ablation ceiling (deleting the mandatory scale reads) is only +16.9%.
   Further decode speedup requires fewer bytes (a smaller format) or less weight traffic
   (better speculative acceptance), not a different arrangement of the same 34 B blocks.

Residual uncertainty, stated plainly:
* Clock state is unforced (`auto`, 21/1000/2000 MHz) and swings the same measurement 2x;
  every claim above rests on within-run ratios, not absolute GB/s.
* The probe walks uniform-width windows, not production's 401-dispatches/token mix
  (average 16.96 MB/dispatch vs the probe's 23.67 MB), so the weighted model is an
  approximation of the real mix.
* The ACO fusion was verified for `mv3_v0.spv` and the archived Q2_0 build; both compile
  the same `load_u32` pattern, so the PQ2_0 build is inferred, not disassembled.

Not committed. Artifacts: `probe_mv3.c`, `mv3.comp`, `mv3_v*.spv`, `gguf_cols.py`,
`probe_mv3`, sweeps in this directory.
