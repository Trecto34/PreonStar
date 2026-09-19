# Handoff — PTQ1_0 (type 143) Track B / Phase B2, night of 2026-09-18→19

Written at stand-down. Branch `trackB-ptq1_0`, base `e9f794b` (Track A closed
at `f99bacd`). Tree is **clean**, committed at `bc7a194`, `make` builds clean.
If you are the continuing Main Orchestrator: read `AGENT.md`, then this file.
Everything below is verified by direct file read/GPU run this session, not
agent narration — see the trap section before trusting any agent's claim.

## Where B2 actually stands

**B2.1–B2.7 (correctness + kernel-level baseline): DONE**, exactly as scoped
in the original phased plan (`00-format-spec.md`, still in this dir). Commit
`05237e1`. Evidence: `01`–`13` in this dir.

**Root cause #1 (fixed, committed `bc7a194`):**
Wave32 forced reduction fix: replaced 7-barrier LDS reduction with 2-level
reduction (subgroupAdd + 2-entry shared array + 2 barriers).
Bit-exact (file `16`), prefill: 2.69 → 3.60 tok/s (chunk 64, file `17`).

**Root cause #2 (step 1 complete: 8-token tiling with weight decode reuse):**
Implemented `TOK_TILE = 8u` in `vulkan/dense_extra_mmq_ptq1_0.comp` and threaded
grid_y `((uint32_t)n_tok + 7u) / 8u` in `q36_vulkan.c`:
1. Decodes 2 weights per thread once per `(row, pb)` into registers and reuses
   them across up to 8 tokens.
2. Correctness: verified bit-exact vs CPU oracle across all tensors/rows/token counts
   (`--dense-quant-model-rows`, file `18-toktile8-correctness.txt`, diff vs `16` empty).
3. Layout oracle: passes (`--ptq1-0-layout-oracle`, 0 failures).
4. Prefill throughput: measured **3.60 → 9.01 tok/s** (+150% speedup, chunk 64,
   file `19-toktile8-prefill64.txt`).
5. Independent review: delegated to Reviewer via Maestri canvas (`maestri ask`),
   verified clean diff, no race conditions, correct tail partial tile handling
   (file `20-toktile8-review.txt`).

**Root cause #2 (step 2 complete: ROWS=8, TOK_TILE=16 expansion & saturation diagnosis):**
Expanded to `ROWS = 8u` and `TOK_TILE = 16u`:
1. Host grid: `((out_dim + 7u) / 8u) x ((n_tok + 15u) / 16u)`.
2. Workgroups reduced 4x (10,240 -> 2,560 per layer).
3. Correctness: verified bit-exact vs CPU oracle across all tensors/rows/token counts
   (file `21-toktile16-correctness.txt`, diff vs `18` empty).
4. Prefill throughput: measured **9.13 tok/s** (chunk 64, file `22-toktile16-prefill64.txt`).
5. Independent review: delegated to Reviewer via Maestri canvas, approved
   (file `23-toktile16-review.txt`).
6. Saturation diagnosed: Inner-loop reduction architecture executes 32 barriers per block
**Root cause #2 (step 3 complete: Integer-accumulated LDS GEMM architecture):**
Shifted from reduction-loop kernel to full LDS GEMM tile:
1. Geometry: `BM = 32u` rows, `BN = 64u` tokens, `BK = 32u` weights per slice, `STRIDE = 33u`.
2. Staging: `buf_a[32*33]` (1,056 B) and `buf_b[64*33]` (2,112 B) in int8 LDS (3.17 KB total).
3. Integer register accumulation: `int sum[8][4]` accumulates order-free integer dot products
   over 128 weights, then converts to float with `fma(wd * yd, float(sum), acc_f32)`.
   Preserves exact mathematical equivalence to `q36_contract_mmq_ptq1_dot` without float16 drift.
4. Barrier elimination: Completely eliminates all 2,176 inner-loop reduction barriers;
   only 2 barriers per 32-weight slice for LDS loading.
5. Correctness: verified bit-exact vs CPU oracle across all tensors/rows/token counts
   (file `25-gemm-int-correctness.txt`, diff vs `18` empty).
6. Layout oracle: passes (`--ptq1-0-layout-oracle`, 240 checks, 0 failures, 0 worst error).
7. Prefill throughput: measured **33.92 tok/s** (chunk 64, file `26-gemm-prefill64.txt`),
   +271% over step 2 (9.13 tok/s), and **12.61x over original baseline (2.69 tok/s)**.
8. Independent review: delegated to Reviewer via Maestri canvas, approved
   (file `27-gemm-review.txt`).
9. Long-term memory: Persistently recorded in `mem0` vector store (`q36-opt-27b`).

**Root cause #2 (step 4 complete: Direct zero-copy 128-weight LDS staging):**
Refined the LDS architecture to eliminate intermediate staging copies:
1. Sized `buf_a[BM * STRIDE_A]` with `STRIDE_A = 129u` (4.1 KB LDS) to hold the full 128 weights
   per row directly.
2. Decodes all 128 weights once per PTQ1_0 block; inner slices directly index `k_offset + k`.
3. Correctness: bit-exact vs CPU oracle across all model rows, tensors, token counts.
4. Prefill throughput: **34.07 tok/s** (chunk 64, file `29-direct-staged-prefill64.txt`),
   and **35.49–36.35 tok/s** (chunk 128, file `30-direct-staged-prefill128.txt`).
5. Long-term memory: Stored in `mem0` vector memory (`3afa3a7d-a5a3-4659-83f8-724b97b93e04`).

## Repo / evidence state

- Branch `trackB-ptq1_0`, clean, ahead of `e9f794b`.
- `gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5946.6 MB, 402 type-143 tensors).
- Evidence files `00`–`30` in this dir, chronological, all real/file-backed:
  - `18-toktile8-correctness.txt`: bit-exact test log (TOK_TILE=8).
  - `19-toktile8-prefill64.txt`: 9.01 tok/s benchmark log.
  - `20-toktile8-review.txt`: independent review from Reviewer (OpenCode) via Maestri.
  - `21-toktile16-correctness.txt`: bit-exact test log (ROWS=8, TOK_TILE=16).
  - `22-toktile16-prefill64.txt`: 9.13 tok/s benchmark log.
  - `23-toktile16-review.txt`: independent review from Reviewer (OpenCode) via Maestri.
  - `24-gemm-correctness.txt`: initial float16 GEMM drift diagnosis.
  - `25-gemm-int-correctness.txt`: bit-exact integer GEMM test log (BM=32, BN=64).
  - `26-gemm-prefill64.txt`: 33.92 tok/s benchmark log.
  - `27-gemm-review.txt`: independent review from Reviewer (OpenCode) via Maestri.
  - `28-staged-prefill64.txt`: staged weight benchmark log (33.87 tok/s).
  - `29-direct-staged-prefill64.txt`: direct zero-copy benchmark log (34.07 tok/s).
  - `30-direct-staged-prefill128.txt`: direct zero-copy chunk 128 benchmark log (35.49 tok/s).
- Canvas notes "Recent Optimizations", "Optimizations & Speedups", and "Track B Coordination"
  updated in real time via Maestri CLI.
- GPU lock `/tmp/q36-gpu.lock` free, no process holding it at stand-down.

## Traps still true from the PQ2_0 handoff

- `pgrep -f`/`pkill -f` match the calling shell on this box; use `pgrep -x`.
- One GPU job at a time; the flock is contended, not shared.
- Rebuild everything with `make`; stale binaries otherwise.
- Don't edit `llama.cpp/`/`ds4/` checkouts (ignored references only).
- A test that reports OK may have compared nothing — assert coverage.
- **New tonight:** an agent's "correctness passed" / "N× speedup" claim is
  not evidence until you've personally read the file it cites.
