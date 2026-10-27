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

## Speed pass 2 (same day)

* `p36_gpu_moe_ffn_f32` row-pair path now covers the native IQ2_XXS/Q2_K recipe
  (`gu_iq2xxs && down_q2k`). Before, MoE fell off the fast path for 2..31 tokens
  and only the verify batch paid for it: 99.1 -> 106.5 t/s at draft 3, margin 1.
  Token parity with plain decode still exact.
* The MTP draft no longer pulls the 1 MB logits row back per draft: top-2 comes
  from the device reduction and the host reads 8 bytes of ids plus, when the
  margin gate wants them, two 4-byte logit values (`p36_mtp_pick`).
* The APEX head file in `/home/server/models/llm/` was stale (built before the
  tensor-offset fix), which is why its acceptance was stuck at 50%. Re-extracted
  it: acceptance 92.9%, but APEX is still level with plain (80.4 vs 84.4 t/s)
  because a verify row costs ~4.2 ms there.

Three-rep, 128 tokens, temp 0:

| config | t/s |
|---|---:|
| plain | 96.05 |
| `--mtp-draft 3 --mtp-margin 1` | 101.43 |
| `--mtp-draft 4 --mtp-margin 2` | 101.93 |
| `--mtp-draft 3 --mtp-margin 0` | 92.86 |
| `--mtp-draft 5 --mtp-margin 2` | 97.42 |
