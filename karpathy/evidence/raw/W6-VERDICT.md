# W6 — merged Vulkan dense gate_up — ACCEPTED (2026-09-21)

Branch `trackB-ptq1_0`, base W5 commit `6b74da5`, model
`Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense, 5120 embed dim, 64 layers).
Plan: `karpathy/PLAN-implementation-2026-09-20.md` §W6.

## What changed

New shader `vulkan/dense_iq3_xxs_mmq_pair.comp` + one host entry point
`q36_gpu_matmul_iq3_xxs_pair_mmq_tensor()` in `q36_vulkan.c`, called from
`q36_forward_ffn_vulkan_model` (`q36.c`) in place of the two separate
`q36_gpu_tensor_matmul_q8_or_float_scaled` gate/up prefill dispatches.

Geometry is unchanged from `dense_iq3_xxs_mmq.comp` (BM 32 x BN 128 x BK 32,
local_size 128, wave32-forced by the existing `dense_" … "_mmq` suffix rule).
One workgroup now decodes **both** weight matrices' A tiles against **one**
staged `b16` activation tile and **one** LUT setup, i.e. the per-workgroup
activation staging + table load is amortized over 64 weight rows instead of 32.
Per-matrix accumulation order, A-tile feeding and the store mapping are byte-for-byte
the single-projection kernel's, so every output element is bit-identical to the
unfused path. Costs: `sum[32]` (16 more VGPRs) and +2176 B LDS for the second
A tile.

Gating: default ON; `Q36_VK_DENSE_IQ3_PAIR=0` disables. Only
`gate_type == IQ3_XXS && up_type == IQ3_XXS && n_tok > 1` takes the pair; mixed
type pairs, decode (`n_tok == 1`) and every refusal fall through to the existing
two dispatches. A failed pipeline init is loud (`failed to read Vulkan shader` /
`vkCreateComputePipelines failed`) and returns 0 → fallback.

## Coverage (why 94 and not 110)

110 gate/up layer-instances per prefill pass carry this shape; 94 (85.5%) are a
same-type IQ3_XXS pair and fuse. The 16 that do not are **genuinely mixed-type**
(IQ3_XXS gate with a non-XXS up, notably the early/MoE-shared layers) or il 63
IQ3_S/IQ4_XS, which a single decoder cannot serve. Those keep the fused-MoE
instance's files untouched and stay on the single-projection path.

## Measurement

Kernel attribution (same binary, `Q36_VK_PROF_KERNEL/PREFILL/SHAPE=1`, ctx 512,
chunk 256, gen 0), pair off vs on:

| | pair off | pair on |
|---|---|---|
| `dense_iq3_xxs_mmq.spv` | 510 disp / 1892.9 ms | 322 disp / 1089.5 ms |
| `dense_iq3_xxs_mmq_pair.spv` | — | 94 disp / 728.0 ms |
| gate/up shape (5120x17408) | 220 disp / 937.6 ms | 858.9 ms (= 130.9 leftover + 728.0 pair) |
| whole-run kernel total | 2792.7 ms | 2670.9 ms (**-4.36%**) |

The removed count closes exactly: 510 - 322 = 188 = 94 paired instances x 2, at
4.262 ms/instance = 801 ms of single-kernel time removed. The pair does that work
in 728.0 ms, so **gate/up is -8.4%** and mmq as a whole -4.0%.

The pair kernel's own dispatches are also folded into the 5120x17408 shape line
(126 = 94 pair + 32 real singles), so that shape line double-counts in the pair-on
arm; the spv totals do not.

Upper bound from the pre-implementation ablation (drop the b16 staging loads
outright): mmq -14.2%, prefill +10.6%. The pair recovers part of that
(-4.0% mmq) because it still stages B once per workgroup — it only stops paying
for it twice.

**7-rep interleaved `tests/bench_ab.sh`, ctx 1024, gen 16, chunk 256**
(wrappers that differ only in `Q36_VK_DENSE_IQ3_PAIR`; one binary):

- Arm A (pair off): prefill **169.25 tok/s** (MAD 0.140), decode **18.12 tok/s** (MAD 0.100)
- Arm B (pair on): prefill **173.01 tok/s** (MAD 0.440), decode **18.22 tok/s** (MAD 0.080)
- Delta: prefill **+2.22%**, decode **+0.55%** (inside the documented 9.4% decode spread)
- Arms do not overlap: max A 170.48 < min B 171.90. Gate is dense prefill ≥ +0.70% → **PASS (3.2x the gate)**.

## Quality

Frontier-513 GPU-vs-GPU oracle, pair off vs on, single binary:
`max_abs_diff = 0`, argmax 5316 (19.820942) both arms, top-64 overlap 64/64,
all 248,320 logits. **Bit-exact**, as the shader construction predicts — not
merely within noise.

## Guard

No existing `.comp` was touched, so the IQ2_S guard is untouched by construction;
re-verified anyway: `vulkan/moe_gate_up_gemm.spv` = `b46fa81a…`,
`vulkan/moe_down_gemm.spv` = `93a38508…`, `vulkan/moe_gate_up_decode.spv` =
`946097d8…`, `vulkan/moe_down_q2k_sum_decode.spv` = `3f316c16…` (sha256, unchanged).
`vulkan/dense_iq3_xxs_mmq.spv` md5 `8af2502ca5e652113f2cd37042291190`, unchanged
(`./glslc` reproduces it byte-identically).

## Verdict

**ACCEPTED, default ON.** Prefill +2.22% > the 0.7% dense gate; decode neutral
with MAD reported; bit-exact parity; guard untouched.

`reconsider_if`: the 16 mixed-type instances are ever converted to same-type
(that would raise coverage above 85.5% and the win with it — scaling the measured
+2.22% by 1/0.855 implies ≈ +2.6% at full coverage), or the second A tile pushes
LDS/occupancy over a cliff on a future part while a smaller BN does not.

Raw evidence: `karpathy/evidence/raw/ab-w6-pair.csv`,
`ab-w6-pair-summary.txt`, `ab-w6-pair-parity.txt`, `ab-w6-pair-profile.txt`,
`w6-baseline-profile.txt`, `w6-ablate-probe.txt`,
`w6-pairoff-profile-raw.txt`, `w6-pairon-profile-raw.txt`.

### Claude2 Audit Follow-up (Zero-Cost Correctness Fix)
During independent audit, Claude2 identified that in `load_u32_m`, the second word read `next = ...[(off >> 2u) + 1u]` was evaluated unconditionally, causing a 4-byte out-of-bounds read past the buffer nominal end on the final block/slice when `shift == 0u`. While discarded numerically (hence bit-exact), this was an undefined behavior risk in Vulkan without `robustBufferAccess`.
Restructured `load_u32_m` to return `word` immediately if `shift == 0u`, short-circuiting the `(off >> 2u) + 1u` read.
Re-verification:
- Frontier-513 oracle parity: `max_abs_diff = 0.0000`, top-1 agreement 1/1, vocab 248,320 (bit-exact).
- Interleaved A/B benchmark: Arm A median 170.51 t/s vs Arm B median 172.83 t/s (+1.36% prefill, decode +0.16% neutral, non-overlapping distributions).
- Hardware properties: 128 VGPRs, 108 SGPRs, 0 spills, 8 subgroups/SIMD.

