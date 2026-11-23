# Prompt lookup on Inkling-2-Mini (2026-11-23) — REJECTED

Branch `experiment/ink-lookup`, MiniPlus-R2, BC-250, `p36-server --ctx 8192`.
`client.py` sends four temp-0 chat requests (700 tokens each, streamed,
decode t/s from `usage.completion_tokens` after the first token); `run.sh`
starts a server, runs the client, stops it. The `edit` prompt embeds
`edit-prompt-source.inc` (frozen copy; it is 876 tokens).

## Rollback correctness (the kept part)
A temporary diagnostic (not committed) forced every speculative step to
commit only its first row, so each cycle ran snapshot -> verify -> restore ->
1-row replay. Output was byte-identical to plain decode on all four prompts:
non-folded cycle 654 rollbacks, folded cycle 1473 rollbacks, positions up to
1513 (past the 1024-slot SWA ring wrap).

## Speed (decode t/s, plain vs `--lookup-k 8`)
| request  | plain | lookup, decode-then-verify | lookup, one-forward + adaptive |
|----------|------:|---------------------------:|-------------------------------:|
| edit     | 74.2  | 67.6 / 68.0                | 50.6 (think) / 48.3 (no think) |
| newcode  | 75.7  | 64.8 / 65.1                | 64.9 / 62.1                    |
| prose    | 75.6  | 74.8 / 75.1                | 75.0 / 70.9                    |
| tooljson | 74.4  | 62.8 / 62.9                | 55.0 / 51.3                    |
(The first lookup column used an older 816-token edit prompt.)

## Why (fold8-*.txt)
Cycle time on `edit`, folded (`fold8-edit-cycle-timing.txt`):
snapshot 0.4 ms; verify 3 / 5 / 9 rows = 36 / 53 / 106 ms; partial-accept
replay 20-30 ms. Acceptance: 2.97 tokens per cycle on edit, 3.53 on tooljson.
* 9 rows exceed the 2..8-row Q8_0 nx route and fall back to the GEMM tile.
* `inkling2_attn_prefill` for 2..8 rows grows with context: per 4-row
  forward 6.8 ms of attention at ctx 2048 and 11.7 ms at 8192, against
  0.7 / 1.0 ms for one decode step.
* Each verify row reads its own 8 of 128 experts: ~0.5 of a decode step.
