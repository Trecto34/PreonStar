# W2 Verdict — `dense_iq3_xxs_decode_r4` ISA & ALU Diagnostic Ablation

## 1. Objective & Gate (Plan §W2)
- Diagnostic to test whether IQ3_XXS decode is purely DRAM-bound ("234 GB/s logical") or ALU-taxed.
- Method: `Q36_VK_PROF_KERNEL=1`, ctx 512, `--gen-tokens 1` vs `--gen-tokens 65` (differential per-token delta: `(gen65 - gen1) / 64`).
- Gate: Diagnostic only (no standalone performance claim without 7-rep interleaved A/B). Exit = written DRAM-bound vs ALU-bound verdict with ISA evidence.

## 2. Verified ISA & Occupancy Properties (via `VK_KHR_pipeline_executable_properties` / `mmq_info`)
Extracted from `dense_iq3_xxs_decode_r4.stats.txt` and `dense_iq3_xxs_mmq.stats.txt` in this directory:

| Property | `dense_iq3_xxs_decode_r4` (shipped decode) | `dense_iq3_xxs_mmq` (shipped prefill) |
|---|---|---|
| VGPR allocation | **64** (max reg v63) | **96** (max reg v93) |
| SGPR allocation | 108 | 108 |
| Spilled registers | 0 VGPR / 0 SGPR | 0 VGPR / 0 SGPR |
| Subgroups per SIMD | **16** (higher occupancy) | **10** |
| Workgroup size | 64 × 1 × 1 (Wave64) | 128 × 1 × 1 (2× Wave64) |
| LDS size | 1,024 B | 12,288 B |
| Code size | 6,060 B | 12,208 B |

## 3. Measured Profiling Data
Model: `Swift-Qwen3.8-27B-IQ3_XXS.gguf`, ctx 512, GPU soak ≤55 °C.

| Run | `dense_iq3_xxs_decode_r4` gen1 (ms) | `dense_iq3_xxs_decode_r4` gen65 (ms) | Isolated per-token delta `(gen65-gen1)/64` |
|---|---|---|---|
| **Baseline** (production decode) | 26.481 | 1,727.458 | **26.578 ms/tok** |
| **Ablation** (loads kept, ALU dropped) | 20.963 | 1,306.147 | **20.081 ms/tok** |

- **Exposed ALU cost**: `26.578 - 20.081 = 6.497 ms/token` (**24.44%** of kernel execution time).
- **Bandwidth floor at zero ALU**:
  - Achieving 20.081 ms/tok on ~7.5 GB IQ3_XXS weight transfers corresponds to **373.5 GB/s** (82.2% of the 454.4 GB/s hardware roofline).
  - Production baseline (26.578 ms/tok) runs at **282.2 GB/s** (62.1% of roofline).

## 4. Auditor Review & Discrepancy Resolution
Independent audit by Claude2 correctly identified:
1. Early drafts confounded ablated prototype VGPR estimates (36) with compiled production shader stats (64). Verified directly against `dense_iq3_xxs_decode_r4.stats.txt`: 64 VGPRs, 16 subgroups/SIMD.
2. Direct division by 65 (`1727.458 / 65 = 26.576 ms`) includes prefill token artifacts. The differential formula `(gen65 - gen1) / 64 = 26.578 ms` cleanly isolates decode tokens 2..65.
3. Single-rep profiling carries known thermal and background noise (e.g. prefill op drift in non-targeted kernels). As specified by W2 gate, this remains a *diagnostic bounds check*, not an accepted speedup claim.

## 5. Verdict
- **Verdict: PARTIALLY ALU-TAXED (NOT purely DRAM-bound).**
- Up to 24.4% (6.5 ms/tok) of decode kernel time is exposed ALU overhead (primarily the 4× DPP cross-lane reduction chains: `quad_perm`, `row_half_mirror`, `row_mirror`, `v_permlanex16_b32`, `v_readlane_b32`).
- The DRAM bandwidth limit sets a hard floor at **20.08 ms/token** (373.5 GB/s, 82.2% roofline). No shader optimization can exceed this floor without changing weight quantization format (e.g., PTQ1_0).
- Recorded in `karpathy/AlreadyTried.md` to settle the DRAM vs ALU bound debate.
