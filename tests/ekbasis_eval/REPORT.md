# POST /v1/systemone on q36-server: Ekbasis-27B Q3_K_S on AMD BC-250 (round 2)

Branch `worktree-agent-a4129648c912da856` (not merged). Model `Ekbasis-27B.Q3_K_S.gguf` (sha256 `9d6dfe2c…8a8e47`).
Contract and prompt follow the official Ekbasis `serve.py` / `decision_core.py` (letter-v1-semif, T=1, thinking off).
Raw outputs: `results/baseline` = `results/accuracy`, `results/bench_cold`, `results/bench_warm` (round 1, commit 608f304);
`results/final/*` (this round); `results/prefix_reuse_ab`.

## Commits
| Commit | What |
|---|---|
| 608f304 | baseline endpoint + eval harness + preserved round-1 results |
| 10905f2 | slot hygiene (evict store, pending/vision state, slot choice) + shared-prefix reuse |
| 2bce371 | git choice fixtures (stash/content/branch, 396 rows) + eval_git.py |
| b7f41c1 | batched-mode prefill sliced on the chunk grid + concurrency check |
| 6b2e6be | official serve.py contract (422, string errors, defaults, latency_s) |
| 0ff2869 | disjoint calibration set + calibrate.py |
| 8a30d46 | final accuracy/calibration/git results; p95 + tok/s in bench |
| (this) | final benchmarks, regression logs, this report |

## What changed
1. Cache/concurrency. `sys_run_job` now mirrors `generate_job`: clears stale pending-tool state, drops a vision
   checkpoint, stores a warm checkpoint to the disk KV store before replacing it, and `job_slot_score` places a
   systemone job on the slot its prompt extends, else the least-warm slot. Found and fixed by test: in
   `--batched-session` mode the stock scheduler slices prefill at 128 tokens while another slot generates, which shifts the
   256-token chunk grid and moved probabilities by up to 0.03 under concurrent traffic. systemone now slices on whole
   chunks (`sys_session_sync`), so concurrent and sequential answers are bit-identical.
2. Latency. With several independent questions the shared leading prompt is prefilled once, staged
   (`q36_session_stage_payload`, private temp file, always removed) and restored per question, so each question prefills
   only its suffix. The prefix is rounded down to the prefill-chunk grid so chunk boundaries equal a full prefill.
   `Q36_SYSTEMONE_PREFIX_REUSE=0` disables it.
3. Contract (vs official serve.py): HTTP 422 with `{"error": "<message>"}` for validation errors, `latency_s` returned,
   `state` defaults to "", `questions` to {} (200, no answers), missing/unknown `type` = choice, `read_once` by Python
   truthiness, over-long prompt = 422. `images` and `\u0000` are rejected explicitly (not silently dropped/truncated).
4. Fixtures: git choice sample (stash count, file content, branch; 66 rows per question type x split).

## Tests
| Test | Result |
|---|---|
| C unit tests (`./q36_reference_test --server`), incl. goldens, validation, official defaults, shared-prefix length | pass |
| `tests/systemone_live.py` (contract, errors, alias, read_once agreement) | pass |
| Token counts vs HF tokenizer, 21 golden prompts (incl. 3,689-token 160-option, both read_once layouts) | all equal |
| Prefix reuse A/B (`ab_prefix.py`), 27 requests, reuse on vs off | all answers + usage bit-identical |
| `concurrency_check.py`: 6 concurrent systemone + 2 chat, 3 rounds, single and `--batched-session 2` | bit-identical to sequential; chat unchanged around systemone calls |
| 200-row guard eval re-run on the final binary vs preserved baseline | 200/200 identical answers and usage |
| Repo `tests/server_live_smoke.sh` and `tests/server_batching.sh` (Ornith-1.5-35B MoE) | both ok |
| `python3 tests/ekbasis_eval/test_offline.py` (fake server) | pass |

## Latency, cold, median / p95 seconds (5 reps; baseline = round 1, 3 reps)
Independent mode (one prompt per question). State = state-text tokens; each question adds about 110 prompt tokens.

| State tokens | Questions | Before (608f304) median | After median | After p95 | Speed-up |
|---|---|---|---|---|---|
| 128 | 3 / 5 | 7.04 / 12.10 | 6.86 / 11.81 | 6.86 / 11.81 | 1.0x (shared prefix < one chunk: no reuse) |
| 256 | 3 / 5 | 11.97 / 19.30 | 7.08 / 11.69 | 7.37 / 11.74 | 1.7x / 1.7x |
| 512 | 3 / 5 | 16.07 / 28.42 | 6.81 / 9.52 | 7.15 / 9.98 | 2.4x / 3.0x |
| 1024 | 3 / 5 | 31.23 / 50.29 | 11.72 / 14.70 | 11.75 / 14.71 | 2.7x / 3.4x |
| 128-1024 | 1 | 1.9 / 3.8 / 6.3 / 10.7 | 2.1 / 4.6 / 5.9 / 11.3 | up to 11.4 | same path (see variance note) |

`read_once` is unchanged (it already shared the evidence): 1024 tokens x 5 questions 13.9 s before, 13.8 s after.
Cost is per 256-token chunk, not per token (a partial chunk costs about as much as a full one, roughly 2.2 s/chunk), so
reuse saves whole chunks. Full tables: `results/final/bench_cold/results.md`.
Warm (exact repeat of the same request, `results/final/bench_warm`): 1 question 1-2 ms (no-op sync on the live
checkpoint, valid because the logits are already there; NOT a cold latency); 3-5 questions equal cold, since the live
checkpoint ends in the last question and the next request re-prefills (cross-request reuse is not implemented).
Variance: 1-question rows are bimodal across reps (for example 256 tokens: min 3.1 s, p95 4.6 s), consistent with GPU
power-state changes; medians move by up to about 20 percent between runs, so do not read small differences as real.

Throughput: about 85-116 input tok/s for single-question prompts (prefill-bound, no decode); "effective" tok/s in the
table counts the redundant prompt tokens reuse avoids, so it is not a GPU rate.

## Memory
Peak GTT 11,303 MiB of 15.9 GB at ctx 8192 (about 3.8 GiB headroom); dedicated VRAM about 0.17 GiB; MemAvailable never
below 2.57 GiB in the final sweep; the staged prefix payload is a temp file in `$TMPDIR` (`/tmp` is tmpfs here, so a
payload costs RAM for the duration of one request). No OOM, no CPU fallback.

## Accuracy and calibration (preserved baseline vs final: identical)
Guard `lost` (200 rows, `floor` state): 92.5% overall, 95.0% destructive recall, 90.0% on safe, ECE 0.048, Brier 0.067.
Git choice (396 rows, 66 per type x split): 96.7% overall (stash 99.2%, content 94.7%, branch 96.2%; known 97.5%,
held 96.0%), ECE 0.029, gold-term Brier 0.031. The published git v3 figure is 96.6%; the fixture is not the release
set, so treat that as consistent, not a replication.

Error analysis of the 15 baseline errors (5 destructive missed, 10 safe flagged; `results/accuracy` + this round's
analysis): 13 of 15 have confidence >= 0.9; held-out safe is the weakest cell (7/50 wrong); `restore --source`,
`git rm` and failed commands recur among false positives (the model assumes a command succeeds or that a worktree-only
restore loses the staged version); two destructive misses are staging-area overwrites. Two gold labels depend on
simulator semantics (reset --merge; "any branch" meaning tip vs history). Descriptive, n is small, nothing tuned on it.

Calibration (fit on the disjoint 200-row calibration set only, evaluation set scored once, `calibrate.py`):
| Setting | Eval-set result |
|---|---|
| baseline T=1, threshold 0.5 | acc 92.5%, recall 95.0%, spec 90.0%, NLL 0.252, Brier 0.0673, ECE 0.0481 |
| T=1.46 (fitted), 0.5 | same accuracy, NLL 0.230, Brier 0.0636, ECE 0.0489 |
| T=1.46, threshold 0.20 (fitted for >=95% recall) | recall 97.0%, spec 88.0%, acc 92.5% |
| T=1.46, threshold 0.13 (fitted for >=98% recall) | recall 98.0%, spec 84.0%, acc 91.0% |
The calibration set itself scored 90.5%, so sample noise is about +/-3 points. A temperature helps NLL/Brier but not ECE;
a lower threshold buys destructive recall at the cost of safe-class precision. The server stays at T=1 / 0.5 like the
official one; clients can apply `sigmoid(logit(p)/T)` and a threshold themselves. Q3_K_S contribution to the gap to the
published full-precision numbers is still not isolated (no BF16 reference possible here).

## Vulkan / model regression
Engine, kernel and shader files are byte-identical to the base commit (`git diff a246c6b -- q36.c q36_vulkan.c
q36_gpu_core.c vulkan q36_bench.c` is empty); only `q36_server.c` and tests changed. q36-bench on Qwen3.8-27B IQ3_S,
ctx 512, 128 decode tokens, 3 runs: branch 132.8-133.4 prefill / 20.50-20.60 decode tok/s; main's existing binary run
back to back: 134.3-135.0 / 20.63-20.69 (about 1 percent apart, same sources, so build/noise, not the change). The
pre-reboot figures (134.0 / 21.04) were measured on a differently clocked machine state. Ornith MoE passes the repo's
server suites; Ekbasis Q3_K_S and IQ3_S load and run.

## Remaining limitations
- No cross-request prefix reuse: the live checkpoint ends in the last question, so a second request on the same state
  re-prefills (an official-style `/v1/sessions` API, or keeping the staged payload keyed by prefix, would fix it).
- States whose shared prefix is under 256 tokens get no reuse (exactness needs the chunk grid); they are cheap anyway.
- 160 options / 64 questions per request; no tournament for more; no images; no `/v1/sessions`.
- Known divergences: `\u0000` and duplicate keys inside `state`, `null` instructions, nested option descriptions print as
  JSON not Python `str()`; `read_once` is not gated on the model's training.
- Batched mode: a systemone job still occupies one slot for its whole duration; other slots keep decoding between
  chunk groups, but a long job delays them by about one chunk (about 2 s) per turn.
- Accuracy: 92.5% on the guard `lost` question at n=200 (+/-4 points), overconfident on safe commands; the gap to the
  published 98.1%/95.3% is unexplained. No BF16 comparison.

## Production readiness
Suitable for single-user, low-QPS local use of the consequence readout: deterministic, contract-compatible with the
official server for the supported fields, tested for concurrency and cache isolation, memory headroom about 3.8 GiB.
Not ready as a multi-tenant or latency-sensitive service: single GPU, 6-15 s per multi-question request, no
cross-request cache, no authentication or rate limiting (inherited from q36-server), accuracy below the published figure.
