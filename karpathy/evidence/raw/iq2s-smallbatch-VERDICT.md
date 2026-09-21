# IQ2_S small-batch (2..127 token) fused MoE expert pair — ACCEPTED

Date: 2026-09-21 · branch `trackB-ptq1_0` · base HEAD `44143d8`
Model: `Qwen3.8-35B-A3B-IQ2_M.gguf` · guard `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`

Disclosure: the B tree was the shared main worktree, which at run time also
carried a concurrent instance's `dense_extra_decode.comp` /
`dense_extra_mmq.comp` IQ2_S specialization (their `Q36_VK_IQ2S_EXTRA`, default
on; committed after this run as `f36daf4`). So the B arm is *their change + this
one*, not this one alone. Their own 21-rep A/B measured that change at decode
+0.52% / prefill -2.02% (both inside the floors), and the `--prefill-chunk 256`
null control below independently bounds the pair's combined effect on an
untouched-route workload at <=3.23% with fully overlapping ranges. The 1g
attribution rests on the route-miss count, which is specific to this change and
was taken on the same B binary.

Closes HANDOFF-iq2s-20260920.md §5 item 2 ("small-batch 2..31-token IQ2_S
kernels"), and with it the last known q8-route residue on IQ2_M outside the
GEMM range.

## Root cause — measured, not guessed

`Q36_VK_MOE_ROUTE_DEBUG=1`, ctx 512, `--prefill-chunk 16`, `--gen-tokens 0`
(32 forwards of `n_tok=16`):

```
A (HEAD 44143d8): 1280 route misses
B (candidate):      96 route misses
q36: moe f32 route miss il=0 n_tok=16 gate=22 up=22 down=21
q36: moe f32 route miss il=1 n_tok=16 gate=22 up=22 down=21
q36: moe f32 route miss il=2 n_tok=16 gate=22 up=22 down=21
```

Types are `q36_vk` tensor ids: 22 = `IQ2_S`, 21 = `IQ3_S`. The 1280 misses are
40 layers × 32 forwards; the surviving 96 are exactly the three IQ3_S-down
layers (`il=0,1,2` — item 3 / entry 1c's territory, not this port's) × 32
forwards. So every IQ2_S-down layer at `n_tok=16` was leaving the fused f32
expert path for the q8 route, exactly as HANDOFF §5 item 2 predicted: the
admission predicate

```c
if (iq2s && !(gemm || (identity && down_sum_decode))) return 0;
```

admitted IQ2_S only for the prefill GEMM pair and the 1-token identity pair,
because `moe_gate_up_f32b` / `moe_down_q2k_f32b` read 82-byte blocks as Q2_K
garbage.

## Change

- `vulkan/moe_gate_up_f32b.comp`: new `-DQ36_MOE_IQ2S` build
  (`moe_gate_up_f32b_iq2s.spv`). The IQ2_S branch decodes the 82-byte
  superblock (`d` f16 at 0, `qs[64]` at 2, signs at 34, `qh[8]` at 66,
  `scales[8]` at 74), `ib32 = n>>5`, `lg = (n>>3)&3`, `j = n&7`, scale
  `d * 0.25 * (0.5 + nibble(lg))`, magnitude/sign per element, grid at table
  word 544. Math is verbatim-equal to the accepted `moe_gate_up_decode.comp`
  `#ifdef Q36_MOE_IQ2S` arm. The 1024-entry grid is 2048 words, so this path
  reads it from the global `tables` buffer instead of the 512-word shared grid
  the IQ2_XXS path stages — the shared-grid stage loop is `#ifndef`-ed out
  rather than widening `sh_grid` for every build.
- `vulkan/moe_down_q2k_f32b.comp`: same `-DQ36_MOE_IQ2S` treatment
  (`moe_down_q2k_f32b_iq2s.spv`), plus the `tables` binding (6 bindings vs 5)
  that HANDOFF §5 item 2 called out.
- `q36_vulkan.c`: two kernel fields + `Q36_VK_KERNEL` create/destroy entries;
  the non-GEMM dispatch sites pick the `_iq2s` variants when `gu_iq2s` /
  `down_iq2s`; admission widened to

  ```c
  if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;
  ```

- Coverage is **n_tok 2..127**, wider than HANDOFF's "2..31"
  (`Q36_VK_MOE_PAIR_TILE = 8`, `Q36_VK_MOE_GEMM_MIN` default 128; `gemm` is
  false below `Q36_VK_MOE_GEMM_MIN`, and `identity` is `n_tok == 1`). Scope is
  deliberately `down_iq2s`-only, so the IQ3_S-down layers (`il=0,1,2`) keep
  their q8 fallback outside the GEMM range — no wrong-kernel risk, unchanged
  behaviour.
- Makefile: the two `.spv` rules + `VULKAN_SHADERS` entries.

The IQ2_XXS/Q2_K builds stay byte-identical: all new code is inside
`#ifdef Q36_MOE_IQ2S` and the legacy rules are untouched.
`vulkan/moe_down_q2k_f32b.spv` = `0ce317941dbef148fd942b2271c0217280e12fcb7350131ef6b9301008769fb0`,
`vulkan/moe_gate_up_f32b.spv` = `9701f0330b5b5d73…`, both equal to the same
files built from clean `44143d8`.

**Bug caught before landing.** The first draft of the down kernel derived the
IQ2_S slotting by hand (`ib32 = tid>>2`, `lg = tid&3`, `unpack8` of two 8-byte
table words) and was wrong: measured `max_abs` 7.86 vs the q8 route with only
41/64 top64 overlap. Isolated at `n_tok=1` by forcing identity forwards
(`--prefill-chunk 1`, `--gen-tokens 0`, `Q36_VK_MOE_DOWN_SUM_DECODE=1` vs `=0`),
which pinned the defect to the down kernel rather than the pair. Replaced with
the accepted sum-decode IQ2_S loop copied verbatim
(`v_im = itid>>3`, `v_in = itid&7`, `y_off = 128*v_im + 2*v_in`,
`for (i = ix; i < blocks; i += 2)`, inner `p = 0..7`, `word = tables[TAB_IQ2S + entry*2 + (j>>2)]`, `vec2` magnitude + sign)
rather than debugging the custom layout: 7.86 → 0.858 against sum-decode and
0.788 against q8.

## Parity

Frontier dumps (`--dump-frontier-logits-dir`), ctx 512, `--prefill-chunk 16`,
`--gen-tokens 0`, frontier 512, 248,320 logits. Arms: `f32b` = candidate
default, `q8` = `Q36_VK_MOE_F32B=0`, `gemm16` = `Q36_VK_MOE_GEMM_MIN=16`
(the accepted GEMM pair, used as the numerics reference):

| comparison | max_abs (index) | top-1 | top64 |
|---|---|---|---|
| IQ2_M f32b(new) vs q8 | 0.779015 (115762) | 13 = 13 | 61/64 |
| IQ2_M gemm16(accepted) vs q8 | 1.15471 (3456) | 13 = 13 | 61/64 |
| IQ2_M f32b(new) vs gemm16(accepted) | 0.980383 (95945) | 13 = 13 | 62/64 |
| guard f32b vs q8 | 0.796737 (22961) | 13 = 13 | 63/64 |

The new pair is *closer* to q8 than the already-accepted GEMM arm (0.779 vs
1.155). 61/64 is the `n_tok>1` round-off floor this file already sits at — the
accepted arm scores the same — not a defect introduced here. The guard row is
the pre-existing IQ2_XXS f32b baseline (the guard file has no IQ2_S expert
tensors, so nothing in this change is reachable from it).

## Verdict

7-rep interleaved A/B (`tests/bench_ab.sh <HEAD-44143d8-build> ./q36-bench
<IQ2_M> 7`, ctx 512, gen 128, `Q36_AB_MIN_GAIN=3.4`):

| arm | prefill median (MAD) | decode median (MAD) | delta |
|---|---|---|---|
| A `44143d8`, `--prefill-chunk 16` | 72.92 (0.440) | 70.60 (1.310) | — |
| B candidate, `--prefill-chunk 16` | **116.21** (0.130) | 69.45 (0.040) | prefill **+59.37%**, decode −1.63% |
| A `44143d8`, `--prefill-chunk 256` | 547.59 (3.560) | 78.11 (0.330) | — |
| B candidate, `--prefill-chunk 256` | 565.26 (17.080) | 78.98 (0.170) | prefill +3.23%, decode +1.11% |

`--prefill-chunk 16` is the arm that exercises the new kernels: prefill-chunk
`N > 32` splits into smaller-chunk forwards inside one measured step, so chunk
16 forces `n_tok=16` dispatches, well inside the 2..127 range. Gate is MoE
prefill >= 3.4% -> **PASS**, cleared ~17x.

The `--prefill-chunk 256` arm is the null control: at chunk 256 (`n_tok=256`)
`gemm` is true, so the changed dispatch site is not on that path at all. It
reads +3.23% with B MAD 17.08 and fully overlapping ranges (A includes 588.94
and 577.07, B includes 547.32) — inside the documented 3.4% MoE prefill floor,
i.e. no signal either way, which is the correct result for an untouched path.

Decode: −1.63% at chunk 16 (A MAD 1.310) and +1.11% at chunk 256 — both inside
the documented 9.4% decode spread, and there is no mechanism for a decode
regression here: `Q36_VK_MOE_DOWN_SUM_DECODE` defaults on, so the 1-token
identity pair was already admitted before this change and its dispatch is
untouched.

Also run: `karpathy/compat_gate.sh` -> `COMPATIBILITY_GATE: PASS`
(Qwen3.6-35B-A3B IQ2XXS guard CPU/Vulkan parity OK; Swift IQ3_XXS smoke
172.71 tok/s prefill / 18.55 decode). All 121 `.spv` the HEAD binary loads are
byte-identical between the A and B trees, so the A arm is the true baseline.

**ACCEPTED, default ON.** No new env var: the IQ2_S variants are chosen by
tensor type inside the existing fused path. `Q36_VK_MOE_F32B=0` still reverts
the whole fused path to q8; `Q36_VK_MOE_GEMM_MIN` still moves the GEMM/f32
boundary.

`reconsider_if`: an IQ2_M-class file whose IQ2_S gate/up sits under an
**IQ3_S** down in a layer that is not already covered — that layer still takes
the q8 route outside the GEMM range and would need the same treatment on the
down side of the f32b pair; or a non-expert `in_dim % 256 != 0` file, which
neither kernel guards against.

Caveats: the realistic target for this change is a batched server workload
(concurrent requests, speculative-decode verify windows, chunk tails).
`tests/bench_ab.sh` has no concurrency knob, so `--prefill-chunk 16` is a proxy
for `n_tok ∈ 2..127` and not a measurement of real concurrent serving. The
chunk-256 control arm is too noisy at 5 reps to exclude a small real effect
there (one would not expect one).

Raw evidence: `ab-iq2s-smallbatch-chunk16.{csv,summary}`,
`ab-iq2s-smallbatch-chunk256.{csv,summary}`, `iq2s-smallbatch-parity.txt`,
`route-miss-A.txt`, `route-miss-B.txt`, `iq2s-smallbatch-ab3.sh`.
