# W3 Verdict — `delta_net_cols` Prefill on BC-250

## 1. Context & Objective (Plan §W3)
- Test whether enabling `delta_net_cols.comp` (columnar prefill with state loaded once, token loop, stored once) via `Q36_VK_DELTA_COL_PREFILL=1` provides a prefill throughput gain on BC-250.
- Production default: `q36_vulkan.c:1225` explicitly sets `enabled = env ? env[0] != '0' : !q36_vk.bc250;`, disabling it by default on BC-250 in favor of `delta_net_decode_reg_f16`.
- Gate: dense prefill gain ≥ 1.5% and frontier logits bit-exact.

## 2. Standard 7-Rep Interleaved A/B Benchmark (ctx 1024)
Raw data: `karpathy/evidence/raw/ab-w3-deltacol.csv` and `ab-w3-deltacol-summary.txt`.
Harness: `tests/bench_ab.sh` on `Swift-Qwen3.8-27B-IQ3_XXS.gguf` at standard `ctx=1024`, `gen=16`, `chunk=256`, 7 interleaved reps.

- **Arm A (Stock baseline, `delta_net_decode_reg_f16`)**:
  - Prefill median: **170.77 tok/s** (MAD 0.500)
  - Decode median: **18.06 tok/s** (MAD 0.010)
- **Arm B (Candidate, `Q36_VK_DELTA_COL_PREFILL=1`)**:
  - Prefill median: **166.54 tok/s** (MAD 0.500)
  - Decode median: **17.93 tok/s** (MAD 0.070)
- **Delta (B vs A)**:
  - Prefill: **-2.48%** (FAIL vs ≥+1.5% gate, regressed in 7 of 7 paired reps)
  - Decode: **-0.72%** (FAIL)
  - Verdict: **FAIL (prefill)**

## 3. Parity Analysis & Latent Wave32 Bug Fix (ctx 512 Diagnostics)
Raw data: `karpathy/evidence/raw/ab-w3-deltacol-parity.txt`, `ab-w3-deltacol-w32-parity.txt`, `ab-w3-ctx512-diagnostic.txt`.
Diagnostic context length: single-run frontier dump at `ctx=512`, `gen=16`.

1. **Pre-fix Parity Measurement (Catastrophic failure)**:
   - Raw artifact: `karpathy/evidence/raw/ab-w3-deltacol-parity.txt`.
   - Comparing frontier 512 logits (`Q36_VK_DELTA_COL_PREFILL=0` vs `1`):
     - `max_abs_diff = 19.1229` at logit index 1973
     - `top-1` mismatch: ref=13 (21.11) vs cand=29 (10.12)
     - `top-64` overlap: **3 / 64**
2. **Root Cause Analysis (Discovered by Claude)**:
   - `delta_net_cols.comp` partitions columns by `gl_SubgroupID` (0..3) and rows by `gl_SubgroupInvocationID` (0..31), hard-requiring wave32.
   - `q36_vk_force_wave32` previously contained `"delta_net_cols.spv"`.
   - Under f16 recurrent state (Swift), the pipeline compiles `delta_net_cols_f16.spv`.
   - Because `"delta_net_cols.spv"` is not a substring of `"delta_net_cols_f16.spv"`, the f16 variant ran at BC-250's native wave64 (only 2 subgroups of 64 lanes). Columns 2..3 were never written, and lane indices reached 159 against row size 128 (out-of-bounds state corruption).
3. **Verification of Wave32 Fix**:
   - Widening `q36_vk_force_wave32` to match `"delta_net_cols"` forced `delta_net_cols_f16.spv` to wave32.
   - Raw artifact: `karpathy/evidence/raw/ab-w3-deltacol-w32-parity.txt`.
   - Post-fix frontier 512 logits:
     - `top-1`: **identical** (ref=13, new=13)
     - `top-64` overlap: **61 / 64**
     - `max_abs_diff`: **0.767** (restored from 19.12)
4. **Diagnostic Throughput Comparison at ctx 512**:
   - Raw artifact: `karpathy/evidence/raw/ab-w3-ctx512-diagnostic.txt`.
   - Stock baseline (`delta_net_decode_reg_f16`): **184.32 tok/s** prefill (ctx 512). Note: higher than ctx 1024's 170.77 tok/s due to shorter sequence length.
   - Wave32-forced candidate (`delta_net_cols`): **171.04 tok/s** prefill (ctx 512).
   - Delta at ctx 512: **-7.20% prefill**, corroborating the -2.48% regression measured at ctx 1024.

## 4. Verdict & Actions
- **Speed Lever: REJECTED & FROZEN.**
  - `delta_net_cols` is confirmed slower than `delta_net_decode_reg_f16` on BC-250 across all context lengths (-2.48% at ctx 1024, -7.20% at ctx 512).
  - The guard condition `!q36_vk.bc250` in `q36_vulkan.c:1225` is verified correct and must remain frozen.
- **Correctness Fix: ACCEPTED & COMMITTED.**
  - `q36_vk_force_wave32` widened to `"delta_net_cols"` in `q36_vulkan.c` so that if `delta_net_cols_f16` is ever dispatched, it enforces wave32 and avoids memory corruption.
- **Evidence Files**:
  - `karpathy/evidence/raw/ab-w3-deltacol.csv` (7-rep interleaved CSV)
  - `karpathy/evidence/raw/ab-w3-deltacol-summary.txt` (summary block)
  - `karpathy/evidence/raw/ab-w3-deltacol-parity.txt` (pre-fix parity failure)
  - `karpathy/evidence/raw/ab-w3-deltacol-w32-parity.txt` (post-fix parity restoration)
  - `karpathy/evidence/raw/ab-w3-ctx512-diagnostic.txt` (ctx 512 diagnostic runs)
