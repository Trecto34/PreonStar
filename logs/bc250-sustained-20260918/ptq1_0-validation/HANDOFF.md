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
   and 256 serialized subgroupAdd operations per workgroup. Further scaling within
   this reduction-based loop is saturated. Reaching >100 tok/s requires full LDS-staged GEMM.

## Repo / evidence state

- Branch `trackB-ptq1_0`, clean, ahead of `e9f794b`.
- `gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5946.6 MB, 402 type-143 tensors).
- Evidence files `00`–`23` in this dir, chronological, all real/file-backed:
  - `18-toktile8-correctness.txt`: bit-exact test log (TOK_TILE=8).
  - `19-toktile8-prefill64.txt`: 9.01 tok/s benchmark log.
  - `20-toktile8-review.txt`: independent review from Reviewer (OpenCode) via Maestri.
  - `21-toktile16-correctness.txt`: bit-exact test log (ROWS=8, TOK_TILE=16).
  - `22-toktile16-prefill64.txt`: 9.13 tok/s benchmark log.
  - `23-toktile16-review.txt`: independent review from Reviewer (OpenCode) via Maestri.
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
