# Handoff — PQ2_0 (type 142) validation, BC-250

Written at the point work was stopped. Base commit `c0e7c1a`, branch
`experiment/bc250-sustained-20260918`. Committed history: `598eb58` (PQ2_0
sources + validation), `60b67b2` (handoff provenance). **Everything after that
is uncommitted** — see "Tree state (uncommitted session)" below. The top
sections were rewritten at the 2026-09-18 evening stop; the historical
"Directive items" table and "What is established" retain older wording where it
is still true but read the CORRECTION first.

## Status at a glance

* **Track A kernel validation: PASS.** GPU decode/MMQ for types 42/142 are
  bit-exact against the shared f16 contract at every distinct width, including
  synthetic independent-scale native-128 blocks.
* **The shipped PQ2_0 model is native-128** (proven against the publisher F16
  source); no packer was needed. The old "repack" claim is retracted.
* **Whole-model CPU↔GPU parity: closed as low priority.** GPU-first policy;
  do not resume it.
* **Only open item: tier-3 quality** (`hard-smoke`, 3 pass / 4 incomplete so
  far, stopped mid-run). Resume command below.
* All 2026-09-18 evening changes are **uncommitted**.

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

**Policy change — see REPORT "Policy change: q36 is GPU-first".** Whole-model
CPU↔GPU parity is no longer a release blocker; do not resume the CPU-parity
investigation or build per-layer dump infrastructure. Track A continues with
GPU-only gates.

## Whole-model parity: closed out as a low-priority reference issue

`--gpu-cpu-parity` initially failed identically on g64 and PQ2_0 (0/5 top-5,
max_abs ~19) while the Qwen models pass. Root cause: the CPU reference never
implemented the forward PRISM Hadamard transform that `e50a3e1` added to the
Vulkan path; Ternary-Bonsai has `prism.hadamard`, the Qwen files do not. Fixed
(`q36_hadamard_forward_host`, `..._grouped_host`, per-tensor
`hadamard_grouped`, rotation at every CPU matmul site): top-1 matches all steps,
max_abs ~19 -> ~4.

The remaining gap was localized, not fixed, and is **not** PQ2_0:

* g64 and PQ2_0 fail byte-identically;
* targeted 42/142 decode+MMQ oracles are bit-exact at every input width;
* GPU-vs-GPU (`Q36_TEST_PARITY_REF=vulkan`) is bit-exact (rms 0);
* contract-aware f16 MMQ (`q36_mmq_contract.h`, `Q36_CPU_MMQ_CONTRACT=1`)
  barely moved the residual; so did f32 SSM state + matched KV types.

It is generic third-implementation CPU-f32-vs-GPU divergence over 64 layers
(top-1 correct everywhere, `top20_max_abs` under bound, two rank-overlap
thresholds missed). Documented as low priority; do not chase further.

The targeted oracles remain useful and allowed: `--pq2-0-layout-oracle`
(232 checks, err 0), `--q8k-activation-parity` (0 mismatches),
`--dense-quant-model-rows` (bit-exact vs the shared f16 contract, plus explicit
f16-vs-f32 drift reporting).

## CORRECTION: the shipped PQ2_0 model is native-128; no packer needed

Earlier notes called `Ternary-Bonsai-2-27B-PQ2_0.gguf` a lossless repack of a
g64 model. **That was wrong and circular.** The publisher repo
`prism-ml/Ternary-Bonsai-2-27B-gguf` ships an **F16** source and PQ2_0, and no
g64; the local `Q2_0-g64.gguf` was derived *from* PQ2_0 by duplicating each
128-block scale into two 64-block scales, so its adjacent scales were equal by
construction.

`verify_native_128.py` proves the shipped file is native-128 by range-fetching
the F16 source and applying the publisher rule (`ggml-quants.c:113`, `d=amax`
per 128, `q=clamp(round(x/d),-1,2)`): 224 blocks over in_dim 5120/6144/17408,
**0 scale and 0 code mismatches**, and 777/780 adjacent 128-block scales differ.
So the PQ2_0-vs-g64 bit-identity means both files encode the *same native-128
weights*.

No offline quantizer/packer is required. The whole-model gates run directly on
the shipped PQ2_0 file.

## Native-128 PQ2_0 — kernel layer PASS

New `--pq2-0-native-128`: synthetic rows with independent signed 128-block
scales and arbitrary codes; GPU decode max_abs 2.4e-7, MMQ tokens 2/5/8
bit-exact vs the f16 contract, and a forced-equal-scale injection moves the
result by 3.55 (falsifiability). Independent of the shipped file, and the
`--dense-quant-model-rows` oracle is bit-exact on the shipped native blocks.

The last known-good copy of the file, with the two new oracles but the OLD
dense-quant function, is at
`/home/server/.claude/jobs/f4739b9d/tmp/q36_test.c.good` (job-scoped; copy it
somewhere durable if it is still needed).

## Resume here: tier-3 quality (in progress)

The only Track A gate not yet complete. Command (GPU, native-128 model):

    flock -w 7200 /tmp/q36-gpu.lock ./q36-eval --vulkan \
      -m gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf --suite hard-smoke --plain \
      --trace /tmp/opencode/eval_hardsmoke.trace

`hard-smoke` = 12 cases. The run was stopped mid-case 8 at ~30 min. Partial
results in `/tmp/opencode/eval_hardsmoke.log` (trace `...trace`):

    1 PASSED J
    2 INCOMPLETE ? expected C  (4096 tok)
    3 PASSED F
    4 INCOMPLETE ? expected B  (4096 tok)
    5 INCOMPLETE ? expected C  (4096 tok)
    6 PASSED H
    7 INCOMPLETE ? expected 2  (16000 tok)
    -> 3 PASSED / 4 INCOMPLETE so far

The INCOMPLETEs are the model hitting its reply budget while still reasoning,
not a GPU-path fault (no NaN/crash/fallback on any case). Resume with the same
command; the trace can be regraded offline with `--regrade-trace`. A full
`hard` suite is 50 cases at up to 16k tokens each (~hours) and was not run.

Important: the quality *comparison* baseline is `g64`, and PQ2_0/g64 are the
same native-128 weights with bit-identical logits, so the paired quality delta
is exactly zero by construction. The absolute score is the only new number.
An F16 baseline cannot be run on BC-250 (54 GB > 15 GB UMA).

The eval process was killed cleanly (`pkill -x q36-eval`); no GPU lock holder.

## Uncommitted session changes (2026-09-18 evening)

Modified, not committed:

    q36.c        + q36_hadamard_forward_host / _grouped_host / _rows; per-tensor
                 hadamard_grouped; rotation wired into every CPU matmul site;
                 shared f16-MMQ contract path (Q36_CPU_MMQ_CONTRACT) + engine flag;
                 q36_engine_debug_tensor_of_type_at
    q36.h        declaration for the above
    tests/q36_test.c  --pq2-0-native-128; dense-quant now walks distinct input
                 widths and reports f16-vs-f32 drift; parity worst-coordinate
                 diagnostics; coverage accounting updates
    q36_mmq_contract.h  NEW: single shared f16-MMQ contract implementation
    logs/.../REPORT.md, HANDOFF.md  policy change + native-128 correction
    logs/.../verify_native_128.py   NEW: reproducible native-128 proof

Not committed. `make q36_test` and `make q36-eval` are clean. Suggested commit
message: `gpu: native-128 PQ2_0 validation; fix CPU Hadamard reference;
GPU-first policy`.

## What to do next

Per the GPU-first policy (REPORT "Policy change"):

1. Finish tier-3 (`hard-smoke`) on the native PQ2_0 model, record the score.
2. Re-run the data-independent GPU gates on the current tree from a committed
   point: execution trace (`Q36_VK_SHADER_TRACE`, `Q36_VK_PROF_KERNEL`),
   sustained decode/context-growth, and the controlled paired benchmark vs g64.
   The pre-existing numbers (REPORT sections 1-5) already passed and the weights
   are unchanged, so this is a re-confirmation, not new work.
3. Close Track A if (1)+(2) hold. Then PTQ1_0 (Track B) under the same rule.
4. Do **not** resume whole-model CPU↔GPU parity or build per-layer dumps.

## Tree state (committed at `598eb58`; superseded by the uncommitted section above)

Committed at `598eb58` (directive item 8):

    Makefile                        PQ2_0 shader build targets
    q36.c                           PQ2_0 plumbing; + q36_engine_debug_count_tensors_of_type
    q36.h                           declaration for the above
    q36_vulkan.c                    type-142 kernel selection, host guard
    vulkan/dense_extra_decode.comp  Q2_BLK_* geometry defines
    vulkan/dense_extra_mmq.comp     same
    tests/q36_test.c                two new oracles + rewritten dense-quant test
    logs/.../{ab_model.sh, logit_divergence.py, pq2_0-validation/}

Prebuilt tool binaries (`attn_parity`, `hadamard_parity`, `mmq_info`,
`read_copy_bw`, `swiglu_hadamard_parity`, `baseline/q36-bench`) and the raw
`runs/` traces are untracked by design.

The original performance/soak numbers were produced on the pre-commit dirty
tree; only the dense-quant results were re-run after the freeze and are
unchanged. Re-run the critical gates from `598eb58` before quoting the report.

## What is established

Artifacts in this directory; `REPORT.md` has the full write-up.

* **Execution.** `Q36_VK_SHADER_TRACE=1` shows only `dense_extra_decode_pq2_0.spv`
  and `dense_extra_mmq_pq2_0.spv` were ever built (pipelines are lazy, so an
  absent line proves a shader never ran). `Q36_VK_PROF_KERNEL=1`: 25,665 decode
  dispatches, 59.9 % of GPU time. CPU ~2 % median over a 508 s decode.
* ~~**PQ2_0 is a lossless repack of Q2_0-g64**~~ — **RETRACTED, see CORRECTION
  above.** The shipped PQ2_0 file is native-128 and the local g64 is the derived
  file. The bit-identical PQ2_0/g64 logits mean both encode the same native-128
  weights.
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
| 8 | Freeze provenance / commits | **DONE** for the first batch (`598eb58`); the 2026-09-18 evening session is **uncommitted** |
| 9 | Resume Track A gates | **done for the kernel layer** — native-128 is proven and no packer is needed; only tier-3 quality remains (see Resume here) |

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
