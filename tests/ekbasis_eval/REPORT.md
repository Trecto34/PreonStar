# POST /v1/systemone on q36-server: Ekbasis-27B Q3_K_S, AMD BC-250

Model: `mradermacher/Ekbasis-27B-GGUF` `Ekbasis-27B.Q3_K_S.gguf` (12,073,955,264 B, sha256 `9d6dfe2c…8a8e47`, matches HF).
Server: `./q36-server --vulkan -m <gguf> -c 8192 --port 8123`, one request at a time. Spec/contract: official Ekbasis
`serve.py` / `decision_core.py` (MIT), prompt `letter-v1-semif`, `enable_thinking=False`, softmax over option-letter
logits, T=1 (`calib.json` is the identity).

## Reproduce
```
make q36-server q36_reference_test && ./q36_reference_test --server        # CPU unit tests (goldens, validation)
./q36-server --vulkan -m gguf/Ekbasis-27B.Q3_K_S.gguf -c 8192 --port 8123
python3 -I tests/systemone_live.py --server http://127.0.0.1:8123           # contract + error cases
python3 -I tests/ekbasis_eval/eval.py  --url http://127.0.0.1:8123          # accuracy / ECE / Brier (200 rows)
python3 -I tests/ekbasis_eval/bench.py --url http://127.0.0.1:8123 --reps 3 --cold   # latency + GTT/RAM
```
Raw outputs: `results/accuracy/` (every raw response), `results/bench_cold/` (headline), `results/bench_warm/` (see caveat).

## Correctness checks
- Server `usage.input_tokens` equals the HF-tokenizer token count on all 21 golden prompts (the 3,689-token 160-option
  case, both read_once layouts: 247 and 1,079). So no BOS is added and special tokens parse as in the reference.
- Identical requests repeated back-to-back return byte-identical answers (no stale-logit read).
- Live suite passes: contract shapes, `/v1/evaluate` alias, errors (missing state/questions, 1 option, unknown type,
  161 options, 65 questions, malformed JSON), read_once vs independent max |dp| = 0.030 on the live fixture.

## Accuracy and calibration (200 guard questions, Git; `floor` = state before the commands)
Question: "Do these commands permanently lose any uncommitted work…?" (yes = destructive). 100 destructive / 100 safe,
100 known / 100 held-out command types, seeded sample (`fetch_fixtures.py`). Run on the build before the review
fixes (they do not touch the compute path).

| Group | n | Accuracy | Mean conf | ECE (15 bins) | Brier |
|---|---|---|---|---|---|
| overall | 200 | 92.5% | 0.968 | 0.048 | 0.067 |
| destructive | 100 | 95.0% | 0.962 | 0.015 | 0.042 |
| safe | 100 | 90.0% | 0.973 | 0.085 | 0.092 |
| known types | 100 | 94.0% | 0.967 | 0.039 | 0.056 |
| held-out types | 100 | 91.0% | 0.969 | 0.066 | 0.079 |

Latency per request on these prompts: p50 3.26 s, p95 3.71 s. 200/200 answered, 0 contract violations.
Reliability: 182/200 answers at confidence >= 0.93 with 96.2% right; the 0.80-0.93 bins (14 answers) are right 50%.

Published (full precision, all guard questions, different sample): 98.1% seen / 95.3% never-seen, ECE 0.024.
Not comparable one-to-one (n = 100 per group is about +/-5 points, one question type only), and no BF16 reference
run was possible here (54 GB), so the Q3_K_S contribution to the gap is NOT isolated.

## Latency (cold; every request carries a unique prefix so no live-prefix reuse), 3 reps, median seconds
State size = state-text tokens (+ about 103 wrapper tokens + question text in `input_tokens`).

| State tokens | 1 q | 3 q independent | 3 q read_once | 5 q independent | 5 q read_once |
|---|---|---|---|---|---|
| 128 | 1.9 | 7.0 | 4.0 | 12.1 | 6.0 |
| 256 | 3.8 | 12.0 | 6.5 | 19.3 | 8.5 |
| 512 | 6.3 | 16.1 | 7.8 | 28.4 | 9.4 |
| 1024 | 10.7 | 31.2 | 11.2 | 50.3 | 13.9 |

Prefill-bound (no decode: 0 output tokens): about 107-130 tok/s overall (5,720 input tokens in 50.3 s). Independent
mode costs N full prefills; read_once costs one prefill plus about 0.3-1.5 s per extra question.
`bench_warm/` was run without `--cold`: its 1-question rows (about 1 ms) are a no-op sync on an already-evaluated
prompt and its independent rows are partly cache hits. Do not quote it as latency.

## Memory (peak over the whole cold sweep, ctx 8192)
GTT used 11,303 MiB (11.0 GiB) of 15.9 GB (about 3.8 GiB headroom); dedicated VRAM about 0.17 GiB; MemAvailable never
below 2.6 GiB (2.4 GiB in the warm run). Context buffers 451 MiB at ctx 8192. No OOM, no CPU fallback seen
(`backend=vulkan`; Q3_K and Q6_K dispatch to `dense_kquant_*`). Swap use of 300-500 MiB was also present in the
IQ3_S baseline. Compare: IQ3_S (Qwen3.8-27B-GSQ-RCO) 10.70 GiB GTT at ctx 641.

## Behavioural differences
Q3_K_S vs the reference: probabilities come from Q3_K weights (F32 logits, double softmax), so they are not bit-equal
to the BF16 torch path; accuracy/ECE above show a real but not isolated gap and overconfidence on the safe class.
Server vs `serve.py` (known, by design or open):
- validation errors are HTTP 400 (reference 422); missing/unknown `type`, missing `state`, empty `questions`,
  duplicate option names are rejected (reference defaults or accepts); `latency_s` is not returned.
- more than 160 options or 64 questions per request are rejected (the reference tournament is not implemented).
- `\u0000` in strings truncates the string, duplicate keys inside `state` are kept in document order, nested
  dict/list option descriptions print as JSON, not Python `str()`.
- `read_once` is allowed for any model (reference requires `decision_config.json` `read_once: turns`).
- A /v1/systemone call replaces the slot's live checkpoint without the disk-KV "evict" store, and in batched mode it
  always lands on slot 0; either can cost the next chat turn a re-prefill.
- Not done: shared-prefix reuse in independent mode (needs the disk payload round trip), git2/git3 choice fixtures,
  a BF16 reference comparison.
