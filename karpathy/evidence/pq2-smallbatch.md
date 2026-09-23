# PQ2_0/Q2_0 small-batch matmul — `TERNARY-BONSAI-2-27B-DERISKED-PQ2_0` (2026-09-23)

Worktree `/home/server/q36-wt/pq2-persist`, branch `experiment/pq2-smallbatch`
(from `5e3c440`). Uncommitted by instruction. Verdicts are in
`karpathy/AlreadyTried.md`; this is the running record with the raw pointers.

## 0. Why

The DERISKED re-release of the ternary model loads (type-142 PQ2_0, 402 tensors,
dense 27B + MTP head). Decode on this board is DRAM-bound: 7.2 GB read per token
against the measured 437 GB/s streaming ceiling. Everything below is either (a)
that model's decode path or (b) the cost of a *verify step*, which is what a
2-token MTP draft needs.

## 1. The finding that mattered: 2..127 tokens paid a 128-token tile

`dense_extra_mmq_pq2_0` serves any batch by tiling 128 tokens, so a 2-token
verify step or a 12-token prompt tail cost the same as 128 tokens:
`--prefill-chunk 2` measured **2.95 t/s** (a 2-token step ≈ 26× a decode step).
Chunk 4 and 8 were the same story (6.07 / ~13.8 t/s).

## 2. Accepted: `vulkan/dense_extra_small_q2.comp` (bit-exact)

One wave64 per 2 output rows streams each row once, expands it through the decode
kernel's LUT **once**, and dots it against up to 8 tokens. Per token the integer
sums, fma order and `subgroupAdd` are the decode kernel's, so the batched result
must reproduce n one-token calls bit for bit — and does.

Host takes it when `n_tok <= Q36_VK_Q2_SMALL_MAX` (default 16), and for the
ragged tail (`n_tok % 128 <= 16`) after an mmq pass over the aligned head, which
is what fixes prompt tails rather than only tiny batches. `Q36_VK_Q2_SMALL_MAX=0`
disables it.

| check | result |
|---|---|
| `tests/test_pq2_small` | 27/27 PASS, 0 mismatching bits (5120×17408, 17408×5120, 5120×1001; n_tok 1..16); the 24..64 rows print `mmq` and are timed only |
| greedy CLI, both binaries | identical output |
| 7-rep interleaved A/B, ctx 1024, gen 16 | prefill −0.03%, decode +0.00% (inert where it does not dispatch) |

| workload | A (main) | B (this) | delta |
|---|---|---|---|
| ctx 64, `--prefill-chunk 2` | 2.95 t/s | **54.60 t/s** | 18.5× |
| ctx 64, chunk 4 | 6.07 | 70.41 | 11.6× |
| ctx 64, chunk 8 | ~13.8 | 79.74 | ~5.8× |
| ctx 64, chunk 16 | – | 82.03 | – |
| pp130 (128 mmq + 2 small) | 110.35 | **189.51** | +71.7% |
| pp140 (128 + 12 small) | 118.74 | **172.69** | +45.4% |
| pp256 (chunk 256, no small batch) | 211.51 | 214.78 | +1.5% (control, noise) |
| 2-token batch vs one-token call | – | 1.38–1.6× | the MTP-verify regime |

Raw: `raw/pq2-smallbatch-chunk-sweep.txt` (regenerated; the first in-session
sweep was not archived and read 2.96/6.0/13.8 → 55.6/71.1/79.2 and pp130
122.9→181.1, pp140 130.0→165.9 — same direction, same verdict),
`raw/pq2-smallbatch-parity.txt`, `raw/ab-pq2-smallbatch-ctx1024.{csv,summary.txt}`.

`reconsider_if`: a ragged tail of 17..127 tokens matters — then raise the
threshold per shape (17408-in shapes still win vs mmq to ~48 tokens, 5120-in ones
cross over near 16).

## 3. Rejected: PQ2_0 decode matvec dispatch shapes

The matvec reads at ~318 GB/s against a **437 GB/s** measured ceiling, so it was
the obvious target; dispatch shape is not the cause. Full detail in the ledger.
Summary: workgroup caps 320/640/1280/2560 → +43/+11/+5/+2% slower; ROWS 1/2/8/16
vs 4 → +7.5/−1.8/+11.5/+31%; four 4-row wave64s in one 256-thread workgroup →
1385 → 1381 ms (neutral). The probe result that motivated the last one
(251 vs 443 GB/s for 64- vs 256-thread workgroups) **did not reproduce** — the
archived harness gives ~298 GB/s for both. The matvec's own row-granular access
pattern caps at ~273 GB/s even with aligned code-only loads and no dequant math.

Raw: `raw/pq2-shape-rejected/` (probe sources with the exact rebuild/run lines in
`probe-rerun-20260923.txt`, profiles, span sweep). The in-engine shape-sweep
output itself was never archived and the variants are reverted — the ROWS ones
were a one-line `#define ROWS` in `dense_extra_decode.comp`, and their SPIR-V is
working-tree-only (`*.spv` is ignored), so do not quote those deltas as
re-derivable without rebuilding them.

## 4. Rejected: GQA-grouped split-K decode attention

One workgroup per (kv head, token, span) shares each K/V read across the query
heads that share a kv head — bit-exact (18/18 cases, 600..65535 keys), but
0.3–0.7× at 600–4k keys and only ~1.0–1.17× at 16k–64k, net negative end to end
(1k −8.45%, MAD 0.02; 1.5k–4k −2..−9%). The apparent wins of the first sweeps
(+11% at ctx 8k, +23.6% at ctx 2k) did not survive re-measurement — they were
decode variance (§5), not the kernel.

Raw: `raw/ab-attn-decode-gqa-ctx{1k,2k,8k}*`, `raw/attn-decode-gqa-rejected.patch`,
`raw/attn-decode-gqa-rejected/` (shader + harness). Reverted; nothing depends on
it. `reconsider_if`: contexts well past 32k become the workload, and then with a
narrower span for the grouped kernel only (not bit-exact vs span 512).

## 5. Pitfall found on the way: post-prefill decode does not repeat

Same binary, same settings, ctx 4096: 30.9 / 21.8 / 25.2 t/s. Per-kernel `gpu_ms`
moves too (one prefill mmq changed 12% between runs with identical prefill t/s).
Decode at ctx ≤ 1024 is stable to MAD 0.02. Judge decode-only changes at short
context, in a kernel harness, or with ≥7 interleaved reps that agree on
direction. This cost two false positives in §4 before it was recognised.

## 6. State and next levers

Landed in the worktree: `vulkan/dense_extra_small_q2.comp`, its two build rules,
the kernel slot/registration/wave64 pin, the dispatch split in `q36_vulkan.c`,
`tests/test_pq2_small.c` (+ Makefile target), and the two doc-layer files. Skips
worth recording: the `ssm_alpha`/`ssm_beta` fusion was **not attempted** — the
IQ2_S precedent for that exact fusion measured +1.42% on a MoE file, and this
dense file has the same two tiny per-layer tensors, so its ceiling here is under
1%; the in-file MTP head was parked (needs a speculative-core rework, breaks even
only near 0.7 acceptance by the two schemes' algebra).

Next lever on this file, not attempted: the PQ2_0 decode matvec is still at ~318
of 437 GB/s and the remaining headroom is a load layout with wide, aligned
per-lane loads that needs no offline repack — rows are 16 B aligned at 1360 B,
blocks are not.
