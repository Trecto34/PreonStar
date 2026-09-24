# R3 — what the P6 nx small-batch kernel actually buys: `--batched-session`

Measurement item (no source change in the measured binary; the R4 commit that
later added `Q36_MTP_TIMING` counters is inert for all of this).

Model: `/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf`, `--vulkan
--ctx 1024 --prefill-chunk 256 --batched-session N`, 4 fixed greedy streaming
clients (one prompt each, ~430 tokens), 64 generated tokens per stream,
`Q36_VK_DENSE_IQ3_NX` 0 vs 1, 2 interleaved reps (arm order flipped in rep 2).

## Aggregate / per-stream / decode-only

`aggregate_tps` = total tokens / wall (includes TTFT).  `per-stream gen` and
`decode-only agg` exclude each stream's own TTFT, i.e. they are the honest
decode rates.  Medians of the 2 reps.

| N | nx | agg t/s (rep1, rep2) | per-stream gen t/s | decode-only agg t/s | TTFT s |
|---|----|----------------------|--------------------|---------------------|--------|
| 1 | 0  | 11.782, 11.777 → **11.779** | 23.223 | 23.223 | 2.68 |
| 1 | 1  | 11.873, 11.481 → **11.677** | 22.687 | 22.687 | 2.66 |
| 2 | 0  | 2.534, 2.450 → **2.492** | 1.410 | 2.820 | 5.92 |
| 2 | 1  | 3.732, 3.706 → **3.719** | 2.237 | 4.474 | 5.75 |
| 4 | 0  | 4.222, 4.152 → **4.187** | 1.261 | 5.090 | 10.13 |
| 4 | 1  | 5.736, 5.812 → **5.774** | 1.864 | 7.611 | 10.57 |
| 8 | 0  | 5.829, 5.902 → **5.865** | 1.202 | 9.144 | 26.18 |
| 8 | 1  | 7.253, 7.884 → **7.569** | 1.489 | 11.719 | 20.34 |

nx ON/OFF aggregate ratio: N=1 **0.991** (inert; the kernel needs
`n_tok >= 2`, and the nx1 rep spread is 0.39 t/s), N=2 **+49.2%**, N=4
**+37.9%**, N=8 **+29.1%**.

**The headline is not the nx win, it is that batching loses.**  Same binary,
same prompts, same ctx: one stream decodes at **23.22 t/s**, while 8 concurrent
streams aggregate **11.72 t/s** (nx ON) = **0.50x** of a single stream;
N=2 is 0.19x, N=4 is 0.33x.  A `--batched-session 8` server is a throughput
loss versus serialising the same eight requests.

## Per-batched-step cost (server `decode batch count=N elapsed=… ms`)

| count | nx OFF | nx ON | delta |
|-------|--------|-------|-------|
| 1 | 43 ms | 43 ms | 0 |
| 2 | 725 ms | 462 ms | −263 ms |
| 4 | 770 ms | 511 ms | −259 ms |
| 8 | 823 ms | 617 ms | −206 ms |

The cost is **flat in the row count** from 2 to 8 (725 → 823 ms for the MMQ
arm) while a single-row step is 43 ms: at `n_tok >= 2` the whole dense trunk
pays a fixed 128-row-tile price.  nx removes 260 ms of that fixed price, not a
per-row term.

## Kernel attribution (profiled arms, identical prefill, nx toggled)

`dense_iq3_xxs_mmq` `gpu_ms` **14192.131 → 4517.989** (−9674 ms) and
`dense_iq3_xxs_decode_nx` 0 → **1430.415** (+1430 ms).  Every other dense row
is unchanged within noise when nx toggles: `dense_iq3_xxs_mmq_pair` 9811.152 →
9729.182, `dense_kquant_mmq` 6311.880 → 6273.131, `dense_iq4_xs_mmq`
3806.981 → 3787.185, `attn_decode_fused` 167.794 → 168.451.  Audit's own caveat
applies (the profile also contains prefill), so only the *differential* is
used: over the 32 decode steps of that arm the swap is −8244 ms / 32 =
**−258 ms/step**, which reproduces the server step medians exactly.

## Which dense types still take the 128-row MMQ tile at n_tok 2..8

From the same nx-ON profiled arm (dispatch counts for the whole run, 35 decode
steps): `dense_iq3_xxs_mmq_pair` 1927 disp / 9729 ms, `dense_kquant_mmq`
4174 / 6273 ms, `dense_iq4_xs_mmq` 1804 / 3787 ms, residual
`dense_iq3_xxs_mmq` 1288 / 4518 ms, `dense_iq3_xxs_decode_nx` 5313 / 1430 ms.
So of the four dense trunk types, nx covers exactly one; the IQ3_XXS **pair**
(fused gate+up), the K-quants and IQ4_XS still take the tile.  This is also the
kernel set that makes the R4 two-row MTP verify cost 424 ms.

No single one dominates the batched step (pair 27%, kquant 18%, iq4xs 11%,
residual iq3 13%) — the *class* does (69%).

## Probe: is the 128-row tile itself the defect?

The one existing switch that reroutes `n_tok > 1` K-quant work is
`Q36_VK_DENSE_KQUANT_MMQ=0` (it falls through to the generic 8x8
`matmul_kquant`).  N=4, nx ON, 2 reps each:

| arm | agg t/s | TTFT s |
|-----|---------|--------|
| kquant MMQ tile ON  | 5.743, 5.744 | 9.71 |
| kquant MMQ tile OFF | 4.228, 4.201 | 23.35, 26.60 |

**Worse both ways** (−26% aggregate, 2.4x worse TTFT, because the same flag also
drops the prefill path off its tile).  So the tile is not a defect that a naive
fallback fixes: the only thing that beat it for IQ3_XXS was a purpose-built
`n_tok 2..8` kernel (4.7x on that tensor).  Building the same nx-style variant
for `dense_iq3_xxs_mmq_pair` (the largest remaining tile, 27% of the batched
step) / `dense_kquant_mmq` / `dense_iq4_xs_mmq` is the follow-up lever; it is
**not built in this round**.

## Raw

`evidence/raw/r3-batched-nx/` — `results.jsonl`, `results-kmq.jsonl`,
`results-prefillonly.jsonl`, `R3-ANALYSIS.txt`, `srv/srv-*.log` (server logs
with the per-step `decode batch count=N elapsed` lines), `prof/` (op+kernel
profiles), `scripts/` (`run-r3.s`, `run-r3b.s`, `batchbench.py`, `analyze.py`).
