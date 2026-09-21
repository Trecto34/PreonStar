# W4 Verdict — Wave32 Clean-Scan Shaders A/B

## 1. Context & Objective (Plan §W4)
- Test whether forcing `requiredSubgroupSize = 32` on clean-scan shaders (`karpathy/evidence/wave32-eligibility.md` §3b: `moe_matvec.spv`, `moe_matvec_fast.spv`, `matmul_q8_0_decode*.spv`, `add_rms_norm.spv`, `rms_norm*.spv`, `kv_store_quant.spv`) provides a throughput win on `Qwen3.8-35B-A3B-IQ2_M` or the guard model.
- Gate: MoE prefill ≥ 3.4% OR decode paired median with MAD; guard prefill no regression. Negative → ledger with `reconsider_if`.

## 2. Parity Verification (Frontier 512)
Raw artifacts: `karpathy/evidence/raw/ab-w4-wave32-parity.txt` and `ab-w4-wave32-guard.txt`.
- **IQ2_M**: `max_abs_diff = 0` over all 248,320 logits, top-1 identical (13/13), top-64 64/64 overlap.
- **Guard MoE**: `max_abs_diff = 0` over all 248,320 logits, top-1 identical (13/13), top-64 64/64 overlap.
- Result: **Bit-exact parity confirmed** on both MoE architectures.

## 3. Benchmark Results
Raw artifacts: `karpathy/evidence/raw/ab-w4-wave32-iq2m.csv` and `ab-w4-wave32-iq2m-summary.txt`.

### 3.1 Qwen3.8-35B-A3B-IQ2_M (7-rep interleaved A/B, ctx 512, gen 128)
- **Arm A (Stock baseline, native wave64)**:
  - Prefill median: **242.43 tok/s** (MAD 0.500)
  - Decode median: **46.62 tok/s** (MAD 0.410)
- **Arm B (Wave32 clean scan, `Q36_VK_WAVE32_CLEAN=1`)**:
  - Prefill median: **240.00 tok/s** (MAD 1.270)
  - Decode median: **46.62 tok/s** (MAD 0.170)
- **Delta (B vs A)**:
  - Prefill: **-1.00%** (inside the 3.4% MoE prefill noise floor; fails the +3.4% gate)
  - Decode: **+0.00%** (exact tie: 46.62 vs 46.62)
  - Verdict: **FAIL (prefill gate not met, decode flat)**

### 3.2 Guard MoE (ctx 512, unreplicated single-run diagnostic)
- Raw artifact: `karpathy/evidence/raw/ab-w4-wave32-guard.txt`.
- Stock baseline: **669.86 tok/s** prefill
- Wave32 candidate: **640.90 tok/s** prefill
- Delta: **-4.32% prefill** (unreplicated single-run diagnostic showing no indication of gain).

## 4. Verdict & Rationale
- **Verdict: REJECTED & CLOSED (Gate Not Met).**
- Forcing wave32 across clean-scan shaders is numerically bit-exact, but fails the performance gate on the primary target: prefill delta (-1.00%) is neutral inside the 3.4% noise floor and fails the ≥+3.4% gate, while decode is flat (+0.00%). Single-rep guard diagnostic similarly shows no positive signal.
- Structural rationale: BC-250 / GFX1013 executes Dual Compute Units (WGP) natively in Wave64 mode. For memory-bound / non-MMQ kernels with high register counts or simple loops, forcing wave32 halves lane occupancy per wave without improving memory coalescing or instruction latency, yielding no measurable throughput improvement.
- `reconsider_if`: A future compiler or hardware revision where wave32 enables double the active wave slots without doubling scheduling overhead.
- Evidence: `karpathy/evidence/raw/ab-w4-wave32-iq2m.csv`, `ab-w4-wave32-iq2m-summary.txt`, `ab-w4-wave32-parity.txt`, `ab-w4-wave32-guard.txt`.
