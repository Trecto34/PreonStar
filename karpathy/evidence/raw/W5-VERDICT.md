# W5 Verdict — RMSNorm → q8_K Producer Fusion

## 1. Context & Objective (Plan §W5)
- Test whether fusing activation quantization (`quantize_q8_k`) directly into residual add + RMSNorm (`add_rms_norm`) to eliminate the intermediate activation round-trip yields a throughput win on BC-250.
- Target model: `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense model, 5120 embedding dimension).
- Gate: dense prefill ≥ +0.70%, decode paired median with MAD, frontier logits parity bit-exact.

## 2. Parity Verification (Frontier 513 on Swift-27B)
Raw artifact: `karpathy/evidence/raw/ab-w5-fusedrms-parity.txt`.
- Evaluated frontier 513 across both prefill chunking (256-wide) and the single-token decode boundary (`n_tok == 1`).
- Reference: `env Q36_VK_FUSED_RMS_Q8=0 ./q36-bench ...`
- Candidate: `env Q36_VK_FUSED_RMS_Q8=1 ./q36-bench ...` (confirmed dispatching `vulkan/add_rms_norm_q8_k.spv`, 889 dispatches).
- Vocabulary tested: 248,320 logits.
- `max_abs_diff`: **0** across all 248,320 logits (`ref = -4.576600`, `new = -4.576600`).
- `top-1` agreement: **identical** (ref=5316, new=5316, logit 19.820942).
- `top-64` overlap: **64 / 64** (100% overlap).
- Result: **Bit-exact parity confirmed**.

## 3. Standard 7-Rep Interleaved A/B Benchmark (ctx 1024, gen 16, chunk 256)
Raw artifacts: `karpathy/evidence/raw/ab-w5-fusedrms.csv` and `ab-w5-fusedrms-summary.txt`.
Harness: `tests/bench_ab.sh` on `Swift-Qwen3.8-27B-IQ3_XXS.gguf`, 7 interleaved reps (14 runs total alternating A and B).

- **Arm A (Stock baseline, separate `add_rms_norm` and `quantize_q8_k` dispatches)**:
  - Prefill median: **170.38 tok/s** (MAD 0.610)
  - Decode median: **18.10 tok/s** (MAD 0.050)
- **Arm B (Candidate, fused `add_rms_norm_q8_k` kernel via `Q36_VK_FUSED_RMS_Q8=1`)**:
  - Prefill median: **169.83 tok/s** (MAD 0.140)
  - Decode median: **17.95 tok/s** (MAD 0.070)
- **Delta (B vs A)**:
  - Prefill: **-0.32%** (fails the dense prefill ≥+0.70% gate)
  - Decode: **-0.83%** (neutral; well within the documented 9.4% decode noise floor)
  - Verdict: **FAIL (prefill gate not met)**

## 4. Architectural Analysis & Hardware Evidence
Raw hardware artifact: `karpathy/evidence/raw/ab-w5-isa-properties.txt` (`VK_KHR_pipeline_executable_properties` on BC-250 / RADV GFX1013).

| Kernel | Workgroup Size | SGPRs | VGPRs | LDS Size | Max Subgroups / SIMD |
|---|---|---|---|---|---|
| `add_rms_norm.spv` | 1024 (16 waves) | 108 | 16 | 8,192 B | 40 |
| `quantize_q8_k.spv` | 64 (1 wave) | 108 | 20 | 1,536 B | 40 |
| `add_rms_norm_q8_k.spv` | 1024 (16 waves) | 108 | 28 | 25,600 B | 32 |

- **Workgroup Granularity Mismatch**:
  - Stock `quantize_q8_k` operates on independent 256-element blocks with 64 threads (1 wave) and minimal LDS (1,536 B). During prefill with chunk 256, 5,120 tiny workgroups distribute evenly across all 40 CUs with peak occupancy (40 subgroups/SIMD).
  - The fused kernel forces quantization into the single row-wide 1024-thread workgroup of `add_rms_norm`. For dimension 5120, this workgroup must serialize 20 blocks over 2 sequential rounds (16 blocks in round 0, 4 in round 1).
- **Barrier Overhead**:
  - Each round executes 10 workgroup-wide barriers: 1 after input staging, 6 during the 6-step reduction tree (`s = 32, 16, 8, 4, 2, 1`), 1 after `sum4`, 1 after `bsum`, and 1 at round boundary. Over 2 rounds, this is 20 barrier synchronizations across all 1024 threads.
- **Cache Locality**:
  - On BC-250, a 5120-float activation row is 20 KB. In separate dispatches, the intermediate norm output sits hot in L2 cache when `quantize_q8_k` reads it immediately afterward.
  - Avoiding an L2 read/write is completely eclipsed by the 20 barriers, the increase in LDS to 25.6 KB, and the occupancy drop from 40 to 32 subgroups/SIMD.

## 5. Scope, Plumbing & Safety
- Host changes in `q36.c` introduce `rt->inp_q8_valid` to track when `rt->inp_q8` has been populated by the fused norm.
- When `Q36_VK_FUSED_RMS_Q8=1` is not set (default), `q36_gpu_add_rms_norm_q8_k_tensor` returns `0` immediately via `q36_vk_use_fused_rms_q8()` (`env && env[0] == '1'`). `inp_q8_valid` remains `false`, and every call site falls back cleanly to the stock `q36_gpu_add_rms_norm_tensor` path with zero change in control flow or performance.
- Coverage disclosure: Bit-exact parity is verified on single-session incremental decode and prefill. Batched decode (`q36_sessions_eval_batch_vulkan`) and speculative MTP paths have not been independently tested under the candidate flag because the optimization is rejected and remains disabled by default.

## 6. Verdict & Status
- **Verdict: REJECTED & FROZEN (Gate Not Met).**
- Implementation retained behind opt-in environment gate `Q36_VK_FUSED_RMS_Q8=1`, but defaults to disabled (`env && env[0] == '1'`) to preserve stock peak throughput.
- `reconsider_if`: Target devices with very small L2 caches where intermediate activations spill to DRAM, or if a block-partitioned fused kernel architecture is developed that eliminates row-wide synchronization.
- Evidence files: `karpathy/evidence/raw/ab-w5-fusedrms.csv`, `ab-w5-fusedrms-summary.txt`, `ab-w5-fusedrms-parity.txt`, `ab-w5-isa-properties.txt`.
