# Router top-k: subgroup selection vs 256-lane tree

Model: `gguf/MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf`, ctx 512,
48 greedy tokens, prompt "The quick brown fox", `--temp 0`.
5 interleaved pairs, shader binary swapped between runs (no rebuild), RADV
GFX1013, one process at a time.

| arm | decode median | MAD |
|---|---:|---:|
| base (256-lane tree, 1+11k barriers) | 84.60 t/s | 0.30 |
| new (subgroup max + index min, 1+2k barriers) | 85.65 t/s | 0.07 |

+1.24% decode, all five pairs in the same direction. Prefill numbers are
noise (that path mostly uses the host-row router at >=64 rows) and are kept
only for completeness.

Parity: 6 prompts (4x64 and 2x48 greedy tokens) byte-identical against the
old shader. `original.comp` is the pre-change source, `router_topk.comp` the
landed one, `subgroup.spv` the landed binary.
