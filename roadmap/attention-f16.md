# f16 attention

**Status:** step A landed (`670febe`, default on); step B done (`96a78f5`, default on,
inside the same f16 build); step C delayed; C.1 (`78bb5d6`, dense VGPR relief) done;
decode FA f16 staging done (`4328c72`, default on); the rest open.

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

Update 2026-11-17 (after steps A/B, profiled): dense attention is 13.1% of GPU
time for a 0..16k prefill; MoE attention is 13.6% over 0..16k (was 18.7%),
23.2% over 0..32k (was 30.5%), and 30.7% on the 16..32k slice (was 39.1%).

MoE 8-16k ablation: QK FMAs 20%, PV FMAs 14% of prefill time.

## Open items

1. **Step B: packed-f16 PV — done (2026-11-17).** 8-key f16 partials via
   `v_pk_fma_f16`, flushed into f32 `acc`. Kernel −10% (both shapes); ctx 8192
   MoE +1.45% (clean separation), dense +0.52% (neutral); NLL drift inside the
   f32 reorder control. Smaller than the ~+6% estimate because the 0..8k
   average carries little attention; 16k+ unmeasured. Ledger 2026-11-17,
   evidence `karpathy/evidence/raw/fa-f16-pv-20261117/`.
2. **Step C: packed-f16 QK — delayed (2026-11-17).** Tried and measured:
   kernel only −6..−8% vs step B, output error 1.4–2.7e-3, over the 2e-3
   gate (2–3× step B under peaked softmax). The error is the f16
   products/accumulation itself — f16 Q with f32 math adds none — so it
   cannot be traded away while partials accumulate in f16. *Why delayed, not
   dropped:* an instruction that multiplies f16 pairs but accumulates in f32
   (`v_dot2_f32_f16`) would keep the throughput without the error; whether
   gfx1013 has it and RADV emits it is unverified (ISA dump crashes). Also
   the ceiling is small (~+1% whole model at 8k, ~+3% at 32k), so it ranks
   below C.1 and the decode item. Revisit when: the dot2 question is
   answered, or C.1 changes the tile shape (re-measure then). Ledger
   2026-11-17, evidence `karpathy/evidence/raw/fa-f16-qk-20261117/`.
3. **Step C.1: dense VGPR relief — done (2026-11-17).** Narrower types never
   moved gqa6 off 256 VGPRs; the shape did: 2 rows per lane (like gqa8) gives
   128 VGPRs / 8 waves, 5 tokens per workgroup (one idle rowset). Kernel
   −11..−14% at 4-16k, −23% on small chunks, output bit-identical, gqa8
   untouched. Whole model only +0.04% (8k) / +0.41% (16k): profiled, dense
   attention is just 13% of GPU time averaged over 0..16k after steps A/B.
   Ledger 2026-11-17, evidence `karpathy/evidence/raw/fa-gqa6-rs2-20261117/`.
4. **f16 K/V staging in `attn_decode_fa` — done (2026-11-17).** Halves LDS
   (32 → 16 KiB, WGP workgroup residency 4 → 8) and on dense `gqa6` also drops
   VGPRs 256 → 128 (4 → 8 waves); math stays f32 (`FA_F16_PV` omitted: pushes
   `gqa6` back to 256 VGPRs and saves only ~2% on `gqa8`). In-model kernel GPU
   time: MoE 32k −18.7% (75.47 → 61.34 ms over 32 tokens), dense 16k −33.1%
   (140.34 → 93.87 ms); `n_tok=4` MTP verify −19% (MoE) / −34% (dense). Whole
   model MoE 16k decode +2.13% (75.00 → 76.60 t/s, clean separation); single-step
   NLL inside the f32 reorder control. Default on under `P36_VK_ATTN_FA_F16`
   (`P36_VK_ATTN_DECODE_FA_F16` overrides decode alone). Ledger 2026-11-17,
   evidence `karpathy/evidence/raw/decode-fa-f16-20261117/`.
5. **Whole-model numbers at 16k+ — done (2026-11-17).** Dense 0..16k only
   cleared +0.41% (step C.1; attention is 13.1% of GPU time there). MoE Axion
   (`P36_VK_ATTN_FA_F16=0` vs `1`, chunk 256, 3 interleaved reps): `0..16k`
   **+6.33%** (539.26 → 573.40 t/s, clean separation), `16k..32k` slice
   **+12.87%** (392.34 → 442.82 t/s, clean separation); profiled kernel GPU
   time −31.2% at 16k and −30.8% at 32k. Evidence
   `karpathy/evidence/raw/moe-prefill-16k-20261117/`.
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
