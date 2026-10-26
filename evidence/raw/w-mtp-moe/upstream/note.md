# Upstream Q36 on this BC-250 (2026-10-26)

Cloned https://github.com/Ninnix/q36 to /home/server/q36-ninnix, built it,
downloaded `Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf` (12.07 GB, embedded 333 MB
MTP head) with its own `download_model.sh q2`.

Their harness (`q36-bench`, story prompt, 128 gen tokens, q8_0/q4_0 KV):

| ctx | prefill | plain decode | MTP decode (draft 8, margin 0) |
|---:|---:|---:|---:|
| 2048 | 716.0 | 91.59 | **159.27** |
| 4096 | 662.0 | 90.98 | 135.32 |

One-shot chat prompt (`-p "The quick brown fox"`, ctx 2048, 128 tokens,
3 interleaved reps): plain 95.79 / 95.03 / 94.77, MTP 136.37 / 136.82 /
136.93. Acceptance report: drafted 105, accepted 79, full 32, accept 75.2%,
recursive depths 2=18 3=6 4=25.

The APEX-I file cannot be loaded by upstream: `unsupported tensor type 2 for
blk.40.ffn_down_exps.weight` (Q4_0 MTP experts), same wall p36 hits.

p36 on the same q2 model: plain 96.06 t/s (one-shot), MTP 84.77 t/s (0.88x).
So p36's plain decode is on par with upstream; the entire gap is the MTP
implementation (8-row verify, shallow-prefix replay snapshots, opportunistic
block sampling, adaptive depth).

Do not run two of these at once on this box: a 12 GB model plus a second
process kills the desktop session.
