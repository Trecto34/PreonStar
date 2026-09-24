# R4 — MTP loop diagnosis: it is a net loss, and the ceiling is gated on the R3 nx work

Step 1 (diagnosis) complete; step 2 (restructure) not landed — the ceiling math
below says the win is small and conditional on a kernel set that does not exist
yet, which the item allows to be reported instead of built.

Model `Swift-Qwen3.8-27B-IQ3_XXS.gguf`, greedy, ctx 512, `--mtp-margin 0`,
story prompt, same binary for every arm. New env-gated phase timers
(`Q36_MTP_TIMING`) plus a host-readback/flush counter pair
(`q36_gpu_read_bytes` / `q36_gpu_read_flushes`) produce one line per cycle.

## Absolute rates on the same binary (plain decode alongside)

| arm | gen t/s | ms/token | vs plain |
|-----|---------|----------|----------|
| plain (MTP disabled), 32 tok | **22.68** | 44.1 | 1.00x |
| plain (MTP disabled), 256 tok | **23.04** | 43.4 | 1.00x |
| MTP `--mtp-draft 2` | **21.17** | 47.2 | **0.92x** |
| MTP `--mtp-draft 3`, nx ON | **6.97** | 143 | **0.30x** |
| MTP `--mtp-draft 3`, nx OFF | **4.10** | 244 | **0.18x** |

Round-1's 6.88 t/s reproduces exactly (6.97 here). Both MTP depths are net
losses on this hardware.

## Per-cycle phase breakdown (`Q36_MTP_TIMING`, d3, nx ON)

97-cycle run (gen 256, `--mtp-margin 0`, `accept=75.8%`, exits 37 gate-reject /
29 partial / 31 full-accept, 1.94 tokens/cycle):

| phase | median | min | max |
|-------|--------|-----|-----|
| fwd | 51.70 | 46.58 | 56.54 |
| draft | 4.84 | 0.00 | 47.88 |
| snap | 0.13 | 0.00 | 0.19 |
| **verify** | **391.03** | 0.00 | 482.21 |
| commit | 0.01 | 0.00 | 51.55 |
| total | 462.69 | 46.58 | 595.04 |

`read_kb` median 4850, `drains` median 3 per cycle.  The 16-cycle arms agree
(`verify` 370.47, `commit` max 46.96, `total` 464.16; `read_kb=4850 drains=4`
on a partial accept, 1940/1 on a gate-reject cycle).

- `fwd` = one 1-token target forward + the first MTP draft; inside it
  `eval fwd=38.5-39.1 mtp=11.9-12.0 read=0.13 argmax=0.3 ms`, so a bare
  1-token forward is ~39 ms against plain's 43.4 ms/token.
- `draft` ~4.8 ms = one recursive draft (MTP block + a 993 KB `mtp_logits`
  readback + host argmax).  `snap` 0.13 ms.  `commit` = restore + replay
  (42.5-51.6 ms) on a partial accept, 0.01 ms on a full accept.
- **`verify` 391 ms (median) for a two-row forward** against 39 ms for a one-row
  forward: **10x**.  With nx OFF the same phase is **638 ms** (4.10 t/s).  This
  single phase is what makes d3 a 3.3x loss.
- `read_kb=4850` per full cycle = five 993 KB logits readbacks
  (`Q36_N_VOCAB * 4`); `drains` = open-batch flushes, 3-4 per full cycle vs 1 on
  a one-token cycle.  Profile: `submit_wait_ms` **563.0 (d3) vs 167.4 (plain)**,
  i.e. +395.7 ms over the 7 verify cycles = **+57 ms/cycle of pure queue drain**
  on top of the GPU work.

## Where the 410 ms verify goes (kernel differential, same 16 gen tokens)

`gpu_ms(mtp3) - gpu_ms(plain)`, split over the 7 two-row verify cycles:

| kernel | diff ms | ms/cycle | n_tok |
|--------|---------|----------|-------|
| `dense_iq3_xxs_mmq_pair` | +726.9 | 104 | 2 (128-row tile) |
| `dense_kquant_mmq` | +486.9 | 70 | 2 (128-row tile) |
| `dense_iq4_xs_mmq` | +277.4 | 40 | 2 (128-row tile) |
| `dense_iq3_xxs_mmq` (residual) | +207.9 | 30 | 2 |
| `dense_iq3_xxs_decode_nx` | +67.1 | 10 | 2 (the one type nx covers) |
| `attn_prefill_fa_gqa6` | +19.6 | 3 | 2 (verify takes the FA path) |
| `dense_q5k_decode` (lm_head) | +33.1 | 5 | 2 |
| `predequant_b16` / `quantize_q8_k` | +0.4 / +3.1 | — | 457 extra dispatches |

So the two-row verify leaves the decode kernels for the **fixed 128-row MMQ
tile** on three of the four dense trunk types — the same wall R3 measured for
`--batched-session` (step cost flat at 725-823 ms for any row count 2..8).

**The Q8_0 host-drain suspicion is refuted.**  The MTP block's Q8_0 matmuls
appear as `matmul_q8_0_f32b` (0 → 88 disp / 14.2 ms in d3, 64 / 10.5 in d2);
`submit_wait_q8_0_quant_x` never shows up in any profile row.  The MTP block is
not the cost: `draft` is 5 ms and `mtp` inside `fwd` is 12 ms.

## Acceptance (measurement)

`Q36_MTP_STATS`: d3 `calls=14 drafted=16 accepted=10 full=2 accept=62.5%`
(round-1 with nx ON: 75.8%); d2 `calls=15 drafted=9 accepted=9 full=9` — 100%
**by construction**, because `draft_cap = N-1 = 1` makes `verify_n = 1`, the
`row_tops[i-1]` comparison loop never runs, and `commit_n == verify_n == 1` is
always "full accept".  So d2 is not speculation at all: it is one extra forward
per token plus a wasted draft head, which is exactly its 0.92x.

Tokens per cycle: 1.71 (24 tokens over 14 cycles at d3).

## Ceiling math

Measured d3 cycle: 466-531 ms for 1.71 tokens = **273-310 ms/token = 0.15x
plain**.  Even with the verify phase free, `fwd + commit` = 97 ms per 1.71
tokens is 57 ms/token = **0.76x plain**: the shipped shape is structurally a
loss because it runs **two target forwards per cycle** (the committed token's
own forward, then the verify rows) plus a replay on every partial accept.

Standard shape (one forward over [committed, draft...], per-row argmax on the
GPU, adopt-on-partial-accept so there is no replay):
`52 ms (2-row nx step, 1.2x a 43 ms step) + 12 ms MTP + 5 ms draft = 69 ms` per
1.71 tokens = **40 ms/token = 25 t/s**, i.e. **+9% over plain**; at round-1's
76% acceptance it is +17%.  Every term in that sum assumes the n_tok 2..8
**nx trunk covers pair/kquant/iq4xs**, which the R3 measurement shows it does
not — otherwise that "52 ms" step is 725 ms.  The restructure is therefore a
strict dependency on the deferred R3 kernel work, and its upside is ~+10%.

Verdict: **diagnosed; reported as a net loss.**  No MTP code change was landed
(the only code added is the env-gated diagnostic above).  Do not restructure
until the n_tok 2..8 dense trunk has nx coverage for the pair/K-quant/IQ4_XS
tiles; with today's kernels the best possible shape is still a loss.

## Raw

`evidence/raw/r4-mtp-diagnosis/` — `plain32.log`, `plain256.log`, `mtp2-32.log`,
`mtp3-32.log`, `mtp3-256.log`, `mtp3-32-nxoff.log`, `prof-{plain,mtp2,mtp3,mtp3-nxoff}.log`,
`run-r3c-r4.s`, `run-r4b.s`.
