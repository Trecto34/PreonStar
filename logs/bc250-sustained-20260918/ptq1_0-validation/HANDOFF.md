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

**Root cause #2 (step 5 complete: Vectorized 5-trit byte unpacking):**
Replaced scalar `weight_at` (64 global byte loads, modulos, divisions per thread) with a
partitioned 13-byte direct unpack into `buf_a` across `kw` in `{0, 1}`:
1. Thread `kw=0` unpacks bytes [0..7], [16..19], 24; thread `kw=1` unpacks bytes [8..15], [20..23], 25.
2. Completely eliminated all runtime division, modulo, and per-trit branch divergence.
3. Correctness: bit-exact vs CPU reference contract (`--dense-quant-model-rows`, file `31-vec-unpack-correctness.txt`, 0 diff; layout oracle: 240/240 passed).
4. Prefill throughput: **36.81 tok/s** (chunk 64, file `32-vec-unpack-prefill64.txt`) and **38.41 tok/s** (chunk 128, file `33-vec-unpack-prefill128.txt`).
5. Independent review: delegated to Reviewer via Maestri canvas, approved (file `34-vec-unpack-review.txt`).
6. Long-term memory: Persistently recorded in `mem0` vector store (`f898e309-426c-451b-9193-8555ad2af369`).
7. Commit: `1c38d4c`.

**Root cause #2 (step 6 complete: Double-buffered activation prefetching):**
Implemented K-dimension double-buffering in LDS (`shared int8_t buf_b[2][BN * STRIDE_B]` = 4.2 KB LDS):
1. Overlaps slice $k+1$ activation loading from global memory with VALU GEMM computation of slice $k$.
2. Slashed workgroup barriers from 9 to 4 per 128-weight block (55% reduction).
3. Correctness: bit-exact vs CPU reference contract (`38-doublebuf-correctness.txt`, 0 diff; layout oracle: 240/240 passed).
4. Prefill throughput: **38.06 tok/s** (chunk 64, file `39-doublebuf-prefill64.txt`) and **39.59 tok/s** (chunk 128, file `40-doublebuf-prefill128.txt`), achieving **14.72x speedup over original 2.69 tok/s baseline**.
5. Independent review: delegated to Reviewer via Maestri canvas, approved (file `41-doublebuf-review.txt`).
6. Long-term memory: Persistently recorded in `mem0` vector store (`b9cddfb3-e62a-4cc8-b5d1-21237092e175`).
7. Commit: `b6903a3`.

**Root cause #2 (step 7 architectural evaluations & negative results):**
1. **$BN=128$ Macro-Tile Expansion**: Sizing $BN=128$ caused a 46% throughput regression on batch 64 (19.82 tok/s, file `36`) due to 50% idle threads on small prompt frontiers, while yielding no throughput gain on batch 128/256 (38.38 tok/s, file `37`). Reverted to optimal $BM=32, BN=64$.
2. **`i8vec4` Staging & Vectorized Reads**: Loading $a[8][4]$ into registers increased VGPR pressure from 64 to 96 VGPRs, reducing hardware occupancy from 100% (32 waves/CU) to 66% (20 waves/CU), resulting in 37.13 tok/s (c64) and 38.80 tok/s (c128) (files `42`–`44`). Preserved commit `b6903a3` (100% occupancy) as primary.
**Root cause #2 (step 8 complete: Packed 16-bit integer vectorization & 2-way arithmetic):**
Implemented packed 16-bit integer vectorization (`i16vec2`) in `vulkan/dense_extra_mmq_ptq1_0.comp`:
1. Shared memory packed storage: `shared i16vec2 buf_a[BM * STRIDE_A]` (STRIDE_A=65u, 260 B, skew 1 bank) and `shared i16vec2 buf_b[2][BN * STRIDE_B]` (STRIDE_B=17u, 68 B, skew 17 banks) eliminating LDS bank conflicts while reducing total LDS load instructions by 50%.
2. Hardware 2-way arithmetic: Accumulates in `i16vec2 sum[8][4]`, leveraging RDNA1 dual-packed integer math (`v_pk_mul_lo_u16` and `v_pk_add_i16`), halving inner-loop iterations (`BK_PAIRS = 16u`).
3. Mathematical equivalence & overflow headroom: Max accumulation per 16-bit lane is $64 \times 127 = 8,128 \ll 32,767$, guaranteeing 100% overflow safety. Reconstituted with `int s = int(sum[r][t].x) + int(sum[r][t].y)` before single-precision `fma`.
4. Correctness: Bit-exact vs CPU reference contract (`--dense-quant-model-rows`, file `45-i16vec2-correctness.txt`, 0 diff; layout oracle: 240/240 passed, 0 failures, 0 abs error).
5. Prefill throughput: **95.90 tok/s** (chunk 64, file `46-i16vec2-prefill64.txt`), **103.01 tok/s** (chunk 128, file `47-i16vec2-prefill128.txt`), and **102.18 tok/s** (chunk 256, file `48-i16vec2-prefill256.txt`).
   - Speedup vs step 6: **+152% on chunk 64, +160% on chunk 128**.
   - Speedup vs original baseline (2.69 tok/s): **35.65x - 38.29x**.
6. Independent review: delegated to Reviewer via Maestri canvas, approved (`49-i16vec2-review.txt`).
7. Long-term memory: Persistently recorded in `mem0` vector store.

## Repo / evidence state

- Branch `trackB-ptq1_0`, clean, ahead of `e9f794b`.
- `gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5946.6 MB, 402 type-143 tensors).
- Evidence files `00`–`49` in this dir, chronological, all real/file-backed:
  - `45-i16vec2-correctness.txt`: bit-exact test log for i16vec2 kernel (0 diff vs reference).
  - `46-i16vec2-prefill64.txt`: 95.90 tok/s benchmark log (chunk 64).
  - `47-i16vec2-prefill128.txt`: 103.01 tok/s benchmark log (chunk 128).
  - `48-i16vec2-prefill256.txt`: 102.18 tok/s benchmark log (chunk 256).
  - `49-i16vec2-review.txt`: independent review from Reviewer (OpenCode) via Maestri (APPROVED).
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
