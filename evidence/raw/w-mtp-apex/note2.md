# One-forward MTP cycle (2026-10-26, second pass)

## Change

`p36_session_eval_speculative_argmax` now follows the upstream cycle shape:

* draft from the raw residual hidden of the last committed row (`last_h_raw`,
  new buffer) instead of the final-norm hidden, then recurse on the head's own
  hidden;
* verify `[pending, drafts...]` in ONE target forward, so a cycle costs one
  target forward plus the draft head, not a target forward plus a verify pass;
* commit compares `row_tops[i-1]` with `verify[i]`, adopts the frontier on a
  full accept, replays the accepted prefix otherwise (`p36_spec_commit_pending`).

Deleted the old piggyback path (`p36_session_eval_with_draft`, `mtp_backoff`).

## Measurements (ctx 512, temp 0, two reps, IQ2XXS-Q2K model)

| config | t/s |
|---|---:|
| plain | 95.52 |
| `--mtp-draft 3 --mtp-margin 1` | **97.52** |
| `--mtp-draft 4 --mtp-margin 1` | 90.80 |
| `--mtp-draft 4 --mtp-margin 2` | 93.77 |

Token parity with plain decode holds on both models, and acceptance is 96.8% at
draft 3 (30/33 cycles full accept).

`P36_MTP_TIMING` at draft 8 shows where the remaining time goes: a 7-row verify
costs 34.8 ms (~4.5 ms/row past the first) and a partial accept replays the
accepted prefix for 16-26 ms. Both are upstream's cheap cases: they replay only
the recurrent layers from captured per-layer inputs, and their multi-row kernels
amortise the weight read across rows (`dense_*_rows` here stops at 4 rows).

The extraction tool now takes `--expert-quant q4_k|q8_0`; requantisation is not
what limits acceptance (an IQ2_XXS head requantised to Q4_K still accepts 96.8%).
The APEX head itself only reaches ~50% acceptance at draft 3 (73 t/s vs 85.7
plain) and needs its own look.
