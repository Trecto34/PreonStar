# f16 attention

**Status:** step A landed (`670febe`, default on); the rest open.

## Background

The prefill flash-attention tile (`vulkan/attn_prefill_fa.comp`) reads 32
K/V floats from LDS per key per lane against 16×RS FMAs each for QK and PV:
about 2 FMAs per LDS float on MoE (RS=2) and 3 on dense (RS=3). LDS return
and VALU were roughly co-limiting, which is why the MoE tile ran at ~54% of
the f32 peak while dense matmuls reach ~85%.

Step A (f16 K/V staging in LDS, f32 math via `v_fma_mix_f32`) confirmed it:
kernel −22% MoE / −10% dense; whole model at ctx 8192 +2.75% MoE prefill,
+0.78% dense; frontier NLL inside the f32 reorder control. Ledger:
`karpathy/AlreadyTried.md` (2026-11-17), evidence
`karpathy/evidence/raw/fa-f16-lds-20261117/`.

Long-context shares that bound the remaining work (2026-11-10 data):

| | 4-8k | 16-32k |
|---|---|---|
| Dense attention share of prefill | ~10% | ~36% |
| MoE attention share of prefill | — | ~42% |

MoE 8-16k ablation: QK FMAs 20%, PV FMAs 14% of prefill time.

## Open items

1. **Step B: packed-f16 PV.** Accumulate ~8 keys (one SUB step) in
   `f16vec2` with `v_pk_fma_f16`, flush into the f32 `acc`. Softmax weights
   are in [0,1], so this is the low-risk half. Ledger estimate ~+6% MoE
   prefill at 8-16k. Gate: `tests/test_attn_fa` (2e-3), `p36_test
   --vulkan-kernels`, frontier NLL vs the f32 control, 3-rep A/B.
2. **Step C: packed-f16 QK.** Riskier: score error goes straight into
   `exp()`. Only after B holds. Together ~+15% at 8-16k (ledger estimate).
3. **Dense VGPR relief.** gqa6 sits at 256 VGPRs. f16 Q and accumulators
   (part of B/C) free registers; check `mmq_info` for an occupancy step.
4. **f16 K/V staging in `attn_decode_fa`.** Same LDS staging as the prefill
   tile. At 32k MoE the kernel is ~15% of a decode token (0.24 ms × 10
   layers) and reads K/V at ~114 GB/s, a third of matvec bandwidth. Bigger
   key groups were already tried (equal at 32k). Cheap first experiment.
5. **Whole-model numbers at 16k+.** Dense only cleared +0.78% at 8k; the
   kernel gain should show at 16k+. One short run, not a sweep.
6. **Quality on other FA-path models.** Every GQA 6/8, head_dim 256,
   Q8_0 K / Q4_0 V model now runs f16 staging; only Axion and Huihui
   Coder390 were NLL-checked. `P36_VK_ATTN_FA_F16=0` is the rollback.

## Loose ends

- README long-context table predates f16 staging.
- `v_fma_mix_f32` emission is inferred from timing; the ISA dump through
  `VK_KHR_pipeline_executable_properties` crashes this RADV build.
- Double buffering is now possible at the same occupancy (16 KiB freed);
  low value while load/dequant time measures near zero.

## Closed

- K/V read-order rotation (llama.cpp PR #32407): p36 has no Stream-K, query
  tiles already read K/V in lockstep, and the tile is not L2-bound.
