# MTP on our own models (2026-10-26)

## What was broken

`--mtp` could not run at all on the APEX/Q4_0-expert files, and even with a
working head the cycle never beat plain decode.

1. The nextn block of the APEX-style files stores its routed experts as Q4_0.
   No MoE kernel dispatches Q4_0, and `mtp_weights_validate_layout` also
   rejected the Q4_K shared experts and the bf16 router.
2. The speculative cycle gated every draft on `mtp_draft_token ==
   p36_session_argmax(s)`, so nearly every cycle reported `gate-reject` and
   paid a full target forward for nothing.
3. The draft head itself produced all-NaN logits whenever the head came from an
   extracted support file (see the alignment bug below).

## Landed

* `gguf-tools/mtp-head-extract.c` — pulls the `blk.<n>` nextn block out of a
  target GGUF into a standalone support file (466 MB for APEX, 328 MB for the
  IQ2XXS file) and requantizes Q4_0 experts to Q4_K with the engine's own
  quantizer, plus bf16/f16 -> f32 for the small head vectors.
  First version wrote tensor data at pointer-rounded offsets; `calloc` is only
  16-byte aligned, so every tensor after the first was shifted and the head
  produced NaN logits. Data offsets are now tracked as integers.
* `mtp_weights_validate_layout` accepts any dense quant for the shared experts
  (`tensor_expect_dense_quant`) instead of q5_k/q6_k/q8_0 only.
* Removed the argmax pre-gate; only the head's own margin gates a draft.

## Measurements (ctx 512, 64 tokens, temp 0, one-shot)

| config | plain | MTP | acceptance |
|---|---:|---:|---:|
| Qwen3.6 APEX-I + extracted head (draft 3, margin 0) | 85.50 | 44.17 | 50.8% |
| Qwen3.6 IQ2XXS-Q2K + extracted head (draft 4, margin 0) | 96.36 | 84.44 | 89.2% |

Output parity: the IQ2XXS model with MTP is coherent and matches plain; the
APEX head injects stray `!` tokens, so its requantized head is not yet trusted.

## Why it is still slower than plain

p36 runs the pending token as its own target forward and *then* the verify
forward, and a partial accept replays the accepted prefix as a third forward.
Upstream q36 drafts from `last_h` before the target forward, verifies
`[pending, drafts...]` in that one forward, and on a partial accept replays only
the recurrent layers from captured per-layer inputs (`spec_inputs`) — no second
full forward, no full replay. Upstream reaches 159 t/s (1.74x) on this board
with that shape; the remaining gap is that cycle, not the head.
