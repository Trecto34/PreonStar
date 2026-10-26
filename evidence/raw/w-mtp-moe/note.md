# MoE MTP: enabled, measured, still a loss

Model: `Cyber-Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf` (12.55 GiB, vocab
248320) + `Qwen3.6-35B-A3B-MTP-Q4K-Q8_0-F32.gguf` (1.59 GiB), ctx 512,
prompt "The quick brown fox", `--temp 0`.

The Qwen3.6 APEX main model (13.03 GiB) cannot host the head on this 14 GiB
board: device allocation fails (radv 540 MB buffer) and later runs are
OOM-killed. CyberTiel has the same vocab and is 0.5 GiB smaller, so it fits.

| arm | decode median (3 interleaved) |
|---|---:|
| plain | 85.51 t/s |
| MTP draft 2, margin 0 | 78.60 t/s |

0.92x, and greedy output is byte-identical to plain decode (speculation is
exact; `diff` clean over 48 tokens).

Cycle accounting from `P36_MTP_TIMING=1` (draft 2, margin 0):

```
full-accept  fwd 13.65 + draft 0.21 + verify 11.72 = 25.67 ms -> 2 tokens
gate-reject  fwd 13.50                              = 13.50 ms -> 1 token
```

Verify costs 0.86 of a full forward, so an accepted cycle buys 2 tokens for
the price of 1.86 forwards: 77.8 t/s, below the 85.5 t/s plain rate, and a
rejected cycle is 74 t/s. Budget sweep: d2/m0 77.98, d3/m3 76.63, d4/m3
75.64, d3/m0 70.90 - every configuration loses.

Enabling changes that made this run possible: the head now binds the token
table and output projection from the target model (a support GGUF's Q8_0
copies are 1 GiB of dead weight), the MTP prewarm skips those three tensors,
and the MoE MTP layout validator accepts any dense matrix type the main model
already runs instead of requiring Q8_0.

Conclusion unchanged from the upstream review: p36 pays a separate pending
forward plus a verify forward per accepted group. Folding the pending token
into verify row 0 is the only structural fix that can turn this positive.
