# Qwen3.8 headroom probe — 2026-09-20

Box: BC-250, RADV GFX1013, 15.35 GiB UMA. Engine HEAD `bc7c31e` (branch
`trackB-ptq1_0`, tree clean, `q36-bench` rebuilt from it). One GPU process at a
time; entry temp 46-66 C. Model: `/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf`.

## 1. Standing (ctx frontier, `--prefill-chunk 256`)

| ctx | prefill t/s | decode t/s | gen tokens | raw |
|---|---|---|---|---|
| 512 | 182.68 | 21.72 | 128 | `qwen38-dec128.txt` |
| 1024 | 170.83 | 22.53 | 16 | `qwen38-clean.txt` |
| 4096 | 132.99 | 22.07 | 64 | `qwen38-ctx4096.txt` |
| 8192 | 100.84 | 20.54 | 64 | `qwen38-ctx8192.txt` |

Campaign baseline (2026-09-17, `11bb345`) was 171.09 / 23.19 at ctx 1024. Decode
is flat to 4K and only -9% at 8K, so the KV scan is not the current limiter.

## 2. Decode: where the 48.5 ms/token goes (ctx 512, differential gen1 vs gen65)

`diffprof.out` (op and .spv rows identical, so only SPIR-V rows listed).

| kernel | ms/token | share | format bytes |
|---|---|---|---|
| `dense_iq3_xxs_decode_r4` | 27.44 | 56.6 % | IQ3_XXS 7.263 GB |
| `dense_q5k_decode` | 5.81 | 12.0 % | Q5_K 1.642 GB (874 MB = lm_head) |
| `dense_iq4_xs_decode` | 4.16 | 8.6 % | IQ4_XS 1.498 GB |
| `dense_q4k_decode` | 2.87 | 5.9 % | Q4_K 0.687 GB excl. embed |
| `matmul_f32_fast` + `dense_f32f_d_other` | 1.88 | 3.9 % | F32 0.105 GB |
| `attn_decode_split` | 1.47 | 3.0 % | — |
| `add_rms_norm` | 1.11 | 2.3 % | — |
| `delta_net_decode` | 0.77 | 1.6 % | — |
| all other (q8 quant, recur, rope, swiglu, sampling) | ~1.15 | 2.4 % | — |
| `dense_iq3_mmq` (phase-pooled row) | 1.83 | 3.8 % | — |
| **GPU sum** | **48.49** | 100 % | wall 49.4 ms -> 0.9 ms host/dispatch |

## 3. Bytes/token ledger (GGUF header accounting, `gguf_types.out`, `gguf_names.out`)

File = 12.522 GB. Not streamed per token: `token_embd.weight` 0.715 GB (Q4_K,
gathered row) and the whole blk.64 nextn/MTP head 0.451 GB (Q8_0, unused at
draft=1). Streamed per decode token ~= **11.36 GB** -> **234 GB/s logical** at
48.49 ms. Per-format logical rates: IQ4_XS 360, Q5_K 282, IQ3_XXS 265, Q4_K
240, Q6_K 235 GB/s (1.5x spread). The box's own synthetic 512 MiB shader-read
test measured 236.5 GB/s useful (`reports/bc250_sustained_20260918.md` §75), and
that test is explicitly not a ceiling; the nominal 454.4 GB/s GDDR6 figure has
no DRAM-counter backing.

## 4. Prefill: compute-bound, not bandwidth-bound

11.36 GB per 256-token chunk = 44 MB/token; at 183 t/s that is 8.1 GB/s of
weight traffic. `dense_iq3_xxs_mmq` = 63.9 % of whole-run GPU time (8248 of
12901 ms), `dense_kquant_mmq` 12.2 %, `dense_iq4_xs_mmq` 9.4 %,
`attn_prefill_qtile2_gqa6` 3.1 %. Campaign ISA work already put
`dense_iq3_xxs_mmq` at ~39 % of packed-f16 peak, i.e. instruction/latency bound,
and it already emits packed f16.

## 5. Verdict

- **Decode: effectively no room on this quant.** Weight streaming is the whole
  budget; dispatch/host gap is 0.9 ms (2 %), non-GEMM is 4.6 ms (9.5 %). Kernel
  ceiling if every format matched IQ4_XS's 360 GB/s: 31.6 ms -> 31.7 t/s, i.e.
  +40 % theoretical, but the dominant kernel already rejected rows8, integer
  sign-mask, f16-grid LDS and global-LUT variants.
- **Speculative decode is closed, not open.** MTP audit session (worktree
  `/home/server/q36-wt/mtp-audit`, branch `perf/mtp-iq3xxs-microbatch-kernel`,
  closed 2026-09-20): 20.40 tok/s plain vs 6.80-9.61 tok/s MTP with the new
  IQ3_XXS/IQ4_XS n_tok 2-8 micro-batch kernels; bottleneck is one GPU round trip
  per causally dependent draft token. Reopen only with a parallel drafter,
  GPU-resident draft loop, or external drafter.
- **Prefill: theoretical ~2.5x, no known lever.** Every structural attempt is in
  the ledger as rejected (tiles 64/256, BN 256, BK 64/16, staging/barrier
  restructure, XOR swizzle, f16 LUT, dynamic-k, wave32 env). Untouched in-tree
  item: the wave32-eligibility clean-scan set (`evidence/wave32-eligibility.md`
  §3b) never got its one A/B.
- **The real remaining gap on this box is not the dense 27B.** Qwen3.8-35B-A3B
  IQ2_M at HEAD: prefill 95.70 / decode 24.49 (`qwen38-moe-iq2m.txt`) against
  upstream 529.90 / 91.53, with `moe_matvec` still taking 2880 dispatches and
  `dense_kquant` 978 - the kernel-selection item (CAMPAIGN §6.1), not a
  micro-optimization.

## 6. Only byte-level levers left for dense decode

- lm_head is 874 MB of Q5_K (7.7 % of the stream); requantizing only
  `output.weight` to Q4_K saves ~370 MB/token = ~+3 % decode, at some logit
  quality cost. `gguf-tools/quants.c` is in-tree.
- A smaller body quant (IQ2-class) cuts ~30 % of bytes/token; that is the only
  path to a large decode gain that does not need new kernel science.
