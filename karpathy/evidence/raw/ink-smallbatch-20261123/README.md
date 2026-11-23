# Inkling-2-Mini 2..16-row micro-batch forward (2026-11-23)

A = `main` @ 45ac3c8 (`/home/server/q36`), B = this branch. BC-250, 40 CUs.
Prompt: first 6000 bytes of `README.md` (`prompt.txt`), frontier 512.

* `rows-curve.csv` — `P36_VK_MICRO_ALL=1 p36-bench --prefill-chunk N
  --gen-tokens 0`, MiniPlus-R2. `ms_per_forward = N * 1000 / prefill_tps`.
  A ignores the variable, so its rows are the old GEMM-tile route.
* `smaller-quants.csv` — same curve for B on NanoPlus-V2.1 and UD-IQ3_XXS,
  plus plain decode `gen_tps` (96 tokens).
* `parity.txt` — `karpathy/tools/cmp_logits.py` on frontier logits 256/512:
  A vs B without the variable (chunk 1 and 8: max_abs_diff 0), decode vs
  B micro-batch 8 rows, decode vs A GEMM 8 rows.
* `decode-ab.csv` — 4 interleaved reps, 96-token decode, MiniPlus-R2.
