# Handoff — PQ2_0 (type 142) validation, BC-250

Written at the point work was stopped. Base commit `c0e7c1a`, branch
`experiment/bc250-sustained-20260918`. **Nothing below is committed.**

## Resume here (updated: build + run done)

`tests/q36_test.c`'s rewritten `test_dense_quant_model_rows` now builds clean
and has been run on all three models. `mmq_tol_f16` was measured, not assumed:
the f16-modelled reference matches the GPU **bit-identically** (max_abs 0,
rel_rms 0) for the two Q2-family types across 2 models x 8 rows x 3 token
counts, so 4.9e-4 is pure one-ulp headroom. Q2_K / IQ2_XXS stay on the loose
5e-2 UNMODELLED f32 bound.

    make q36_test
    flock -w 900 /tmp/q36-gpu.lock ./q36_test --dense-quant-model-rows \
      --model gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf
    # also clean on Q2_0-g64 and on the Qwen IQ2XXS/Q2K model

Also re-ran and passing: `--q8k-activation-parity` (0 mismatches) and
`--pq2-0-layout-oracle` (232 checks, worst abs err 0).

Still open: full `./q36_test --all` and the provenance freeze below.

The last known-good copy of the file, with the two new oracles but the OLD
dense-quant function, is at
`/home/server/.claude/jobs/f4739b9d/tmp/q36_test.c.good` (job-scoped; copy it
somewhere durable if it is still needed).

## Tree state

Modified, uncommitted:

    Makefile                        PQ2_0 shader build targets
    q36.c                           PQ2_0 plumbing; + q36_engine_debug_count_tensors_of_type
    q36.h                           declaration for the above
    q36_vulkan.c                    type-142 kernel selection, host guard
    vulkan/dense_extra_decode.comp  Q2_BLK_* geometry defines
    vulkan/dense_extra_mmq.comp     same
    tests/q36_test.c                two new oracles + UNBUILT dense-quant rewrite

Untracked: `logs/bc250-sustained-20260918/{ab_model.sh,logit_divergence.py,
pq2_0-validation/}` and several prebuilt tool binaries.

Provenance was **not** frozen (directive item 8). No commits were made, so every
result below was produced on a dirty, unnamed tree. That is the single biggest
gap: re-run the critical gates from a committed tree before quoting them.

## What is established

Artifacts in this directory; `REPORT.md` has the full write-up.

* **Execution.** `Q36_VK_SHADER_TRACE=1` shows only `dense_extra_decode_pq2_0.spv`
  and `dense_extra_mmq_pq2_0.spv` were ever built (pipelines are lazy, so an
  absent line proves a shader never ran). `Q36_VK_PROF_KERNEL=1`: 25,665 decode
  dispatches, 59.9 % of GPU time. CPU ~2 % median over a 508 s decode.
* **PQ2_0 is a lossless repack of Q2_0-g64** in these two files — the g64 scale
  is identical across every adjacent pair of 64-blocks. Verified over all 402
  quantized tensors, 181,888 blocks, zero mismatches. This is why the two models
  give bit-identical logits, and it is why "PQ2_0 is numerically correct" does
  **not** follow from that comparison.
* **Layout oracle (new, passing).** `./q36_test --pq2-0-layout-oracle`:
  28 probes x 8 rows + a dense probe, 232 checks, **worst abs err 0**, expected
  values derived from the format spec in-test, not from any production helper.
  Proven falsifiable: injecting a one-lane read shift made 224/232 checks fail
  with the full diagnostic. This is the strongest PQ2_0 evidence held.
* **q8_K activation parity (new, passing).** `./q36_test --q8k-activation-parity`:
  39 blocks x 13 adversarial patterns, GPU vs production CPU vs test duplicate,
  **0 mismatches**, including exact round-half ties and both extrema tie-break
  orders.
* **Control.** Pre-patch binary vs current on g64: `max|d| = 0`. All four
  pre-existing `.spv` recompile bit-identically from pristine `HEAD`.
* **Performance.** 10 soak-gated interleaved pairs, order alternated:
  decode median 32.61 -> 33.42 tok/s, **+2.63 %** (MAD 0.32, 10/10 pairs,
  non-overlapping ranges). pp512 -0.25 %. Runtime counters: decode kernel
  -3.92 %, mmq kernel +2.67 %.

## Findings that changed the picture

1. **`--dense-quant-model-rows` reported OK while testing nothing.** Its
   `types[]` omitted 42/142 and the ternary models contain no other listed type.
2. **Its reference used full-precision activations** though every kernel consumes
   q8_K. Broken for *all* types — Q2_K and IQ2_XXS on the Qwen model failed too.
3. **The CPU and GPU q8_K block layouts differ**: `q36_block_q8_k` is 292 bytes
   with no `dmin`; the GPU/test block is 296 with one. A `memcmp` across the two
   reads `qs` bytes as a float. Anyone treating them as interchangeable corrupts
   memory.
4. `test_quantize_q8_k` is an **independent test-only duplicate**, a third
   implementation — now proven equivalent, but it is not shared code.

## Directive items 1-9: status

| # | Item | Status |
|---|---|---|
| 1 | Per-type coverage accounting | code written, **UNBUILT/UNRUN** |
| 2 | q8_K reference provenance | **DONE** — duplicate, proven equivalent |
| 3 | Independent PQ2_0 layout oracle | **DONE**, passing, proven falsifiable |
| 4 | Replace loose MMQ oracle | **DONE** — f16 reference built, measured floor 0, bound justified |
| 5 | Decode error reporting | **DONE** — worst token/row/value/abs/ulp printed, running |
| 6 | Do not overinterpret cross-format MMQ | **DONE** — stale "slightly lower" claim removed; both are bit-exact |
| 7 | Full `q36_test` suite | **PARTIAL** — non-parity tests pass individually; `--all` blocks on a pre-existing `--server` hang (see Traps) |
| 8 | Freeze provenance / commits | **NOT DONE** — needs explicit go-ahead; no commit made |
| 9 | Resume Track A gates | not started, correctly blocked |

## Traps worth carrying forward

* Running a GPU test before `--server` in one `q36_test` process hangs:
  `test_gguf_counts_are_rejected_before_allocation` (q36_server.c:12777)
  `fork()`s after Vulkan threads exist and `waitpid` never returns. Confirmed
  pre-existing on `c0e7c1a` with a clean HEAD worktree build, not a Track A
  regression. `--server` alone passes; so do `--vulkan-kernels`,
  `--quant-primitives`, `--ssd-cache-shrink`, `--qwen-tool-call-format`,
  `--vector-fixtures`, and the three PQ2_0 oracles, each in its own process.
  Full `--all` will hang at `--server` for the same reason.
* `pgrep -f` / `pkill -f` match the calling shell on this box. Four incidents
  this session, one self-killed shell. Use `pgrep -x` or an explicit PID.
* Editing an MCP server's source does nothing until the process restarts.
* Rebuilding only some targets leaves stale binaries; `make` everything.
* A test that reports OK may have compared nothing. Assert coverage, not just
  absence of failure.
