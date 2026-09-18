# PQ2_0 (type 142) GPU validation on BC-250

Commit `c0e7c1a` plus the uncommitted Track A working tree (`Makefile`, `q36.c`,
`q36_vulkan.c`, `vulkan/dense_extra_decode.comp`, `vulkan/dense_extra_mmq.comp`).
Device: AMD BC-250 (RADV GFX1013), 40 CU, wave64, 15.35 GiB UMA, Mesa 26.2.2.
Kernel 7.2.6-1-cachyos. All runs serialized under `flock /tmp/q36-gpu.lock`.

This run establishes GPU-path sanity, numerical behaviour against g64, sustained
stability, and a paired performance result. **It is not a full reference
validation** -- the expensive `--gpu-cpu-parity` gate was deliberately skipped
and remains outstanding.

## Interface deviations from the directive

The directive named `./q36-generate` and `./ab_decode.sh --baseline/--candidate/--pairs`.
Neither exists. The real interfaces were used instead:

* generation: `./q36 --vulkan -m FILE -p TEXT --temp 0 --seed 1 --nothink`
* paired A/B: `logs/bc250-sustained-20260918/ab_model.sh PAIRS OUT.csv A.gguf B.gguf`
  (`ab_decode.sh` swaps two `.spv` on one model; this is a *format* change, so
  the model file is what must be swapped)

Frontier logits are emitted as `frontier_*.logits.json`, not `logs/frontier-512-*.bin`.

## 1. Execution -- PASS

`Q36_VK_SHADER_TRACE=1` shows the only dense-extra pipelines ever built:

    q36: building pipeline vulkan/dense_extra_mmq_pq2_0.spv
    q36: building pipeline vulkan/dense_extra_decode_pq2_0.spv

Pipelines are built lazily on first use, so the generic `dense_extra_decode.spv`
and the `dense_extra_decode_q2_0.spv` sibling were never instantiated.
`Q36_VK_PROF_KERNEL=1` confirms where the work landed (64-token run):

    vulkan/dense_extra_decode_pq2_0.spv dispatches=25665 gpu_ms=1358.019 pct=59.9
    vulkan/dense_extra_mmq_pq2_0.spv    dispatches=400   gpu_ms= 605.531 pct=26.7
    vulkan/hadamard_prepare.spv         dispatches=16705 gpu_ms=  94.129 pct= 4.2

No CPU fallback:

* statically, a failed GPU matmul on this path is `return false` propagated up
  `q36_gpu_tensor_matmul_dense_q8_scaled` (q36.c:8336, :8734) -- a hard abort of
  the forward pass, not a silent CPU detour;
* the type-142 host guard (q36_vulkan.c:8251) refuses loudly and returns 0 if
  int-dot / subgroup64 / subgroup-arithmetic are unavailable;
* measured CPU busy during the 508 s sustained decode: median 2.2 %, max 15.5 %
  (the max is model load). A host dequant+matmul of a 27B model cannot hide there.

Both models carry identical PRISM Hadamard metadata (`block=1024 widths=3
values=28672`), so the comparison is like-for-like; `hadamard_prepare.spv` runs
on the GPU in both arms.

## 2. Numerical behaviour -- bit-identical, for a reason that limits the claim

Frontier-512 logits, 248,320 elements (count derived from `output.weight` =
(5120, 248320) in both GGUFs, and cross-checked against each dump's own `vocab`
field -- not hard-coded):

    PQ2_0 vs g64:  top-1 same, top-5 5/5, KL 0.000e+00, max|d| 0.0000,
                   mean|d| 0.00000, rms 0.00000, p99.9 0.00000,
                   nan 0, inf 0, kl_skipped 0

200-token greedy generations from both models are byte-identical.

**Two different quantizations cannot produce identical logits, so this was
investigated rather than accepted.** Inspecting the files directly: in these two
GGUFs the g64 f16 scale is identical across every adjacent pair of 64-blocks, so
PQ2_0's one-scale-per-128 packing reproduces the same dequantized weights exactly.
Verified across all 402 quantized tensors, 8 random rows each, 181,888 128-blocks:

    unequal g64 adjacent scale pairs = 0
    pq2 scale != g64 scale           = 0
    code bytes differ                = 0

The 419,840,000-byte file-size difference is exactly 2 bytes per 128 weights --
the dropped duplicate scale -- over ~26.87e9 quantized weights.

**Therefore the correct claim is narrow: the PQ2_0 kernel is bit-exact against
the g64 path on a losslessly-repacked model.** It says nothing about a model
quantized natively at 128-weight granularity, where the two scales would genuinely
differ; that case exercises code paths this test cannot distinguish and is
unvalidated. This is also not "CPU parity" -- both arms are the same GPU backend.

### Control: the Track A patch did not move the existing g64 path

The pre-patch binary (`logs/bc250-sustained-20260918/baseline/q36-bench`, built
07:15, `strings | grep -c pq2_0` = 0) and the current binary produce **byte-identical**
g64 frontier logits: `max|d| = 0.0000`, KL 0.000e+00.

All four pre-existing shaders recompile bit-identically from pristine `HEAD`
(`git show HEAD:vulkan/*.comp` -> `glslc`):

    dense_extra_mmq         6697.. IDENTICAL   dense_extra_decode      IDENTICAL
    dense_extra_mmq_q2_0    IDENTICAL          dense_extra_decode_q2_0 IDENTICAL

(An earlier scratch artifact `chk_dense_extra_mmq.spv` initially looked like a
generic-mmq regression. It is a capture taken between the "unused const" bug and
its fix, not a real difference -- confirmed by the pristine recompile above.)

## 3. Stability -- PASS

Sustained deterministic decode, ctx 512 -> 8192, 192 tokens per frontier,
1728 generated tokens, 508 s, exit 0:

    ctx    prefill_tps  gen_tps
     512      204.39     35.62
    1536      160.46     34.15
    2560      134.30     33.37
    3584      117.78     25.61
    4608      104.86     23.57
    5632       99.60     22.71
    6656       93.31     22.00
    7680       87.19     21.29
    8192      112.18     22.22

No crash, no device loss, no device reset, no NaN/Inf, no early termination, no
fallback messages. Process RSS stable at ~5.48 GB with no growth. VRAM carveout
pinned at 500 MB (UMA -- weights are host-mapped, so the VRAM node is not a
meaningful residency signal on this box).

Thermals dominate the decay: peak 99 C, plateau 78-79 C, sclk swinging
500-1990 MHz and settling near 1030-1100 MHz. The throughput cliff at ctx 3584
(33.37 -> 25.61) tracks the clock drop, not context cost alone. **Unpaired
benchmarking on this machine is meaningless**, which is what the soak gate in the
A/B harness exists to handle.

Not done: a matched g64 soak. The stability claim is about PQ2_0 only; the decay
curve is not attributed to the format.

## 4. Performance -- +2.63 % decode median, prefill flat

`SOAK_TEMP=55 ab_model.sh 10 05-ab.csv <g64> <PQ2_0>`, A = g64 baseline,
B = PQ2_0 candidate, interleaved within each pair, order alternated, both arms
soak-gated to <=55 C entry.

    pair order  base_tps cand_tps  delta%   base_pp cand_pp
    1    A,B      32.47    33.35   +2.71    202.75  200.74
    2    B,A      32.60    33.40   +2.45    204.33  199.49
    3    A,B      32.44    33.60   +3.58    206.64  200.78
    4    B,A      32.44    33.27   +2.56    204.55  204.29
    5    A,B      32.54    33.60   +3.26    203.59  202.35
    6    B,A      32.64    33.54   +2.76    202.00  205.11
    7    A,B      32.69    33.31   +1.90    202.14  201.37
    8    B,A      32.66    33.37   +2.17    201.84  204.04
    9    A,B      32.62    33.45   +2.54    203.82  205.42
    10   B,A      32.61    33.62   +3.10    204.28  205.60

    decode baseline  median 32.61 tok/s  (32.44 .. 32.69)
    decode candidate median 33.42 tok/s  (33.27 .. 33.62)
    paired delta median +2.63 %  MAD 0.32  min +1.90 %  max +3.58 %
    pairs favouring candidate: 10/10

    pp512 baseline median 203.70, candidate 203.19, paired median -0.25 %
                                                  (min -2.84 %, max +1.54 %)

The two decode ranges do not overlap at all. 10/10 under a sign test is
p = 2^-10 ~ 0.001.

Drift and order controls: first-half median +2.71 % vs second-half +2.54 %;
baseline-first pairs +2.71 % vs candidate-first pairs +2.56 %; entry temp held in
54-57 C across all 20 arms. No evidence of thermal or order bias.

The prior expectation was +1.5 % to +4.4 %. The measurement landed inside it, but
that range was a target, not a prediction, and is not treated as confirmation.

### Harness audit (done before accepting the numbers)

* A and B use identical bench parameters, context, generation length, warmup and
  clock policy; only `-m` differs. Both arms run the same binary and the same
  shader set -- the kernel is chosen from the weight type at runtime.
* `pp512_tps` / `decode_tps` are timed inside `q36-bench` and exclude model load.
* Failed arms print `NO RESULT` and return non-zero; they are not silently dropped
  into the median. All 10 pairs completed.
* **Defect found and fixed:** the harness always ran A before B. Soak-gating
  equalizes entry temperature but not die state or page cache. Order alternation
  and an `order` column were added; the order effect measured afterwards is 0.15 pp.
* **Defect found, not fixed, affects telemetry only:** the harness samples clocks
  12 s into each arm. When model load runs longer than that, the sample lands at
  idle -- visible as `sclk 500 / ~35 W` in pairs 2A, 7B, 8A, 10A. The
  `sclk_mhz`/`power_w` columns are therefore unreliable on those rows.
  `decode_tps` and `pp512_tps` are unaffected.

## 5. Static ISA and runtime kernel counters

`mmq_info` (VK_KHR_pipeline_executable_properties):

    shader                        SGPR VGPR spill LDS   code(B) subgroups/SIMD scratch
    dense_extra_decode_q2_0        108   28    0   1024   3824        36          0
    dense_extra_decode_pq2_0       108   28    0   1024   3808        36          0
    dense_extra_mmq_q2_0           108  128    0  13312  18020         8          0
    dense_extra_mmq_pq2_0          108  128    0  13312  18020         8          0

Instruction counts from the ACO assembly (`shader_isa`), since RADV reports 0 for
the `Instructions`/`VALU`/`Branches`/`Latency` statistics in this build -- those
zeros are unpopulated fields, not measurements:

    decode_q2_0:  613 total, 421 VALU, 143 SALU, 15 VMEM (4x dwordx3, 5x dword),
                  9 SMEM, 34 LDS, 9 branches, 8 alignbit, 0 scratch
    decode_pq2_0: 610 total, 418 VALU, 143 SALU, 15 VMEM (identical mix),
                  9 SMEM, 34 LDS, 9 branches, 8 alignbit, 0 scratch

Occupancy, register pressure, LDS, spills and the memory-instruction mix are
**identical**. The only static difference is three fewer VALU ops of addressing
arithmetic. Static ISA therefore does not explain the speedup and no bandwidth,
cache-hit or latency claim is inferred from it.

Runtime counters do isolate it (`Q36_VK_PROF_KERNEL=1`, ctx 512, 128 tokens,
identical dispatch and group counts in both arms -- 51330 / 132850944):

    dense_extra_decode_q2_0.spv   gpu_ms 2918.083   (32.1 % of GPU time)
    dense_extra_decode_pq2_0.spv  gpu_ms 2803.798   (30.7 %)   -3.92 %

    dense_extra_mmq_q2_0.spv      gpu_ms 4797.763   (52.8 %)
    dense_extra_mmq_pq2_0.spv     gpu_ms 4925.894   (54.0 %)   +2.67 %

The decode kernel is 3.92 % faster and holds ~32 % of GPU time, which is
consistent in scale with the +2.63 % end-to-end decode median. The prefill kernel
is 2.67 % *slower*, consistent with the -0.25 % pp512 median. Profiling adds
timestamp overhead, so these runs are not throughput measurements -- only the
per-kernel ratio is used.

Format arithmetic, independent of the ISA: per 256 weights g64 reads 4x18 = 72 B
and PQ2_0 reads 2x34 = 68 B, 5.6 % less weight traffic. The observed 3.92 % kernel
gain is below that, as expected when weights are not the only decode traffic.

## What remains unverified

* Full `--gpu-cpu-parity` reference validation (deliberately skipped this run).
* `--dense-quant-model-rows` against CPU-dequantized GGUF rows.
* PQ2_0 correctness for a model quantized natively at 128 granularity, where the
  repack is *not* lossless. Every numerical result here rests on the two files
  containing the same underlying weights.
* PTQ1_0 (type 143) -- untouched by this run.
* Long-context quality (the tier-3 eval).
* A matched g64 sustained run.

PQ2_0 is not claimed production-ready on the strength of these tests.

---

# Addendum: `--dense-quant-model-rows`

Run after the report above, to close the one cheap gate still outstanding. It
compares the GPU dense-quant kernels against a CPU reference built from
`q36_quant_dequantize`, so unlike everything above it does **not** depend on
PQ2_0 and g64 containing the same weights.

## The gate was vacuous, then wrong, before it was useful

1. **Vacuous.** `./q36_test --dense-quant-model-rows --model <PQ2_0>` printed
   `dense-quant-model-rows: OK`. The test's `types[]` list was
   `{10..14, 16..23}`; the ternary models contain only types 0, 30 and 142, so
   every iteration hit `continue` and the test compared nothing. A pass that
   compared nothing is indistinguishable from a real pass in the output.

   Fixed: 42 and 142 added to `types[]`, plus a `tested` counter and
   `TEST_ASSERT(tested != 0)` so an empty run can never report OK again.

2. **Wrong reference.** With type 142 actually exercised, it failed at
   ~0.5 % relative error. This was *not* a PQ2_0 defect: running the same gate
   on the shipped g64 model (type 42) gave the same ~1 %, and on the Qwen
   IQ2XXS model Q2_K and IQ2_XXS also failed, at rel_rms 0.0019-0.0055. The gate
   was broken for every type and had been passing on ternary models only by
   testing nothing.

   Root cause: all of these kernels consume a **q8_K-quantized activation**,
   but the reference was built from full-precision `x_host`. The test was
   measuring the activation quantizer, not the weight kernel.

   Fixed: `x` is round-tripped through `test_quantize_q8_k` and the reference is
   built from that, isolating the weight kernel.

3. **Residual f16 floor at tokens>1 -- the reference now models it.** The
   `dense_extra_mmq` kernel stages weights and activations as `float16_t` by
   design, so no f32 reference can reach it. The first fix relaxed the tokens>1
   bound to `max_abs < 1e-1`, `rel_rms < 5e-2`, which passed but isolated
   nothing. That reference was then replaced with a CPU model of the kernel's
   own arithmetic: every `v_mul_f16`/`v_fma_f16` is one round-to-nearest-even
   to binary16 (`test_mmq_q2_family_ref`). The bound is one binary16 ulp
   (`2^-11 = 4.9e-04` relative), justified by the measured floor below rather
   than derived from the ulp in the abstract.

## Result -- PASS, and this one is independent of the repack

    model          type   tokens=1 max_abs  tokens=1 rel_rms  tokens 2/5/8 (q2 family: f16 model)
    PQ2_0          142    5.960464e-08     9.851645e-08    max_abs 0 / 0 / 0  rel_rms 0 / 0 / 0
    Q2_0-g64        42    4.470348e-08     7.966353e-08    max_abs 0 / 0 / 0  rel_rms 0 / 0 / 0
    Qwen Q2_K       10    7.566996e-09     1.007438e-07    f32 ref (UNMODELLED) rel_rms 0.0031 / 0.0027 / 0.0031
    Qwen IQ2_XXS    16    2.980232e-08     1.159542e-07    f32 ref (UNMODELLED) rel_rms 0.0061 / 0.0051 / 0.0050

All three models exit 0. Measured f16 floor for the two Q2-family types is
**exactly zero** -- 2 types across 2 models x 8 rows x 3 token counts
(tokens 2/5/8), every case bit-identical to the modelled f16 reference. The
4.9e-04 bound is pure headroom, not a measured residual.

**The tokens==1 row is the decode kernel**, and PQ2_0 agrees with the CPU
dequantizer to 5.96e-08 -- f32 rounding. This independently confirms the
shader's 34-byte block stride and 2-bit lane addressing for type 142, without
relying on PQ2_0 and g64 holding identical weights. It is still not full
`--gpu-cpu-parity`: it covers 8 rows of one tensor at four token counts, not a
whole-model forward pass.

Q2_K and IQ2_XXS still use a full-precision f32 reference and the loose 5e-02
bound with an explicit UNMODELLED note; they are not release-grade oracles.

Coverage is now asserted per type (eligible/tensors/rows/token-cases plus a
`tested != 0` guard), so a run that compared nothing can no longer print OK.

Files touched: `tests/q36_test.c` (`test_dense_quant_model_rows`,
`test_dense_quant_type_name`). The full `q36_test` suite was not re-run; all
edits are confined to that one test function.
