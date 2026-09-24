# Round-2 summary table — Qwen3.8 audit follow-up (R1–R5)

BC-250 / RADV GFX1013, 40 CU, 16 GB UMA.  Swift = `Swift-Qwen3.8-27B-IQ3_XXS.gguf`,
MoE = `Qwen3.8-35B-A3B-IQ2_M.gguf`, guard = `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`.
All tok/s are absolute and plain (non-MTP) decode on the same binary unless labelled otherwise.
Verdicts follow `karpathy/CAMPAIGN.md` §4; each item's raw data is in `evidence/raw/<item>/`.

| item | probe result | verdict | absolute tok/s before → after (plain decode alongside) | parity | commit |
|------|--------------|---------|--------------------------------------------------------|--------|--------|
| R1 | bisect: first bad commit `9d54dc7` (FA prefill GQA 8). 11137/12288 floats differ by ≤ 1.02e-08; FA is 1.184e-08 from the f64 reference, the old kernel 1.624e-08 — FA is the *closer* one | **stale test expectation, not a kernel bug.** Test widened (`drift <= 4.0e-5f`, FA-aware predicate, `n_head` arm plumbing); kernel untouched | no perf change (test + comment only) | `./q36_test --vulkan-kernels` PASS with FA on and `Q36_VK_ATTN_FA=0`; `compat_gate.sh` PASS; guard greedy byte-identical FA1 vs FA0 | `e77871d` |
| R2 | LDS-staged wide-load rewrite of the IQ2_S MoE down sum-decode: `buffer_load_ushort` 33→3, hot block 549→529 instructions, bit-exact | **NEGATIVE — built, bit-exact, measures nothing**, reverted. Kernel is instruction-issue bound (1.07 warp-instr/weight), not load-width bound | IQ2_M prefill 617.06 (MAD 8.30) → 615.01 (MAD 7.36) = −0.33%; decode 82.15 (MAD 0.150) → 82.32 (MAD 0.290) = +0.21%, MADs overlap; kernel `gpu_ms` 316.383 → 316.571 (+0.06%) | frontier 512/520 `max_abs_diff = 0`, top-1 same, top-64 64/64; `compat_gate.sh` PASS | `dcec7ad` |
| R3 | nx kernel on `--batched-session` 1/2/4/8: agg +49.2% (N=2), +37.9% (N=4), +29.1% (N=8), inert at N=1 (0.991x); batched step fixed at 725–823 ms for 2–8 rows (128-row tile) | **nx real but insufficient — the finding is that batching loses**: 8 streams aggregate 11.72 t/s vs one stream 23.22 t/s = 0.50x. `Q36_VK_DENSE_KQUANT_MMQ=0` at N=4 = −26%. Pair/K-quant/IQ4_XS nx variants not built | Swift batched agg 5.865 (N=8, nx OFF) → 7.569 t/s (nx ON); single stream 23.22 t/s plain | no source change (measurement only) | `dabb78b` |
| R4 | `Q36_MTP_TIMING` per-cycle phases: `verify` **391.03 ms for a 2-row forward vs ~39 ms for 1 row = 10x**; `submit_wait` 563.0 vs 167.4 ms plain; pair/K-quant/IQ4_XS still on the 128-row MMQ tile; Q8_0 host-drain suspicion **refuted** | **diagnosed, reported as a net loss, no restructure.** Two target forwards per cycle, one of them 10x overpriced; the standard shape is +9% and strictly depends on R3's deferred kernels | Swift plain 23.04 t/s vs `--mtp-draft 3` 6.97 (0.30x), `--mtp-draft 2` 21.17 (0.92x); plain 32-token 22.68. Deeper: nx ON d3 6.97 vs nx OFF 4.10 | no source change except the env-gated diagnostic | `43dc22e` |
| R5 | 8K/16K/32K on Swift + guard: split cost exactly proportional to workgroup count at a constant 25–28 us CU/workgroup; requested rate flat 249–310 GB/s vs 31–49 GB/s unique; `attn_decode_split.spv` VGPR 48 / LDS 1536 B / 0 spills / 20 subgroups/SIMD; combine latency-bound with `n_head` workgroups | **diagnosed: issue-bound tile loop + latency-bound combine, grid already saturated — not bandwidth, not redundancy.** Span 512 unchanged; narrowing is single-rep, not bit-exact, fails the `n_tok`-invariance kernel gate, nets ~1–3% of the token | Swift decode 20.15 / 18.48 / 15.40 t/s at ctx 8192 / 16384 / 32768 (−8.3%, −23.6%); guard 72.39 / 64.09 / 52.26 (−11.5%, −27.8%). Attention share 10.4 → 17.7 → 27.7% (Swift), 16.6 → 26.0 → 38.1% (guard) = 80% / 92% of the 8K→32K loss. Span 512→128 gives 18.48 → 18.61 t/s (single rep, gate fails) | no semantic change (comment-only correction of `q36_vk_attn_span()`, +23 −2). `./q36_test --vulkan-kernels` PASS at span 512; `compat_gate.sh` = COMPATIBILITY_GATE: PASS | `1b9be3f` |

## What round 2 actually established

1. **R1** — the red gate was a stale expectation from `9d54dc7`, not a kernel
   regression; FA is numerically closer to the f64 reference than the kernel it
   replaced. All later parity results in this round rest on that.
2. **R2** — the largest lever round 1 named (byte-granular loads in
   `moe_iq2s_down_sum_decode`) is not a load-bound kernel: swapping 33 ushort
   loads for 3 changes nothing. Class of fix rejected; do not reopen load-width
   rewrites without a new instruction-count argument.
3. **R3** — the nx kernel does what P6 claimed at n_tok 2..8, but the
   user-facing feature it was built for (`--batched-session`) is a 2x throughput
   loss regardless, because the batched step pays a fixed 128-row-MMQ tile.
   Batching is not a lever on this hardware until the pair/K-quant/IQ4_XS tiles
   are ported.
4. **R4** — MTP is a net loss (0.30x at draft 3, 0.92x at draft 2) and the cost
   is localized: a 2-row verify forward costs 10x a 1-row forward, i.e. 391 ms
   of a 463 ms cycle. Ceiling math (acceptance × nx cost) already says it cannot
   beat plain decode before the nx second-token cost drops toward ≤1.2x.
5. **R5** — long-context decode attention is instruction-issue bound, not
   occupancy- or bandwidth-bound: the grid saturates at 10 workgroups/CU, the
   kernel has no register/LDS limit, and the per-key instruction stream (V
   gathered one nibble at a time, serial `tmax`) is 2.2–3.1x its own issue
   bound. The one cheap knob (`Q36_VK_ATTN_SPAN`) moves the attention term by at
   most ~4.6% and is not bit-exact, so 512 stays.

Net across the round: one stale test fixed (R1), three negatives with the
mechanism named (R2, R4, R5), one measurement that kills a feature direction
(R3). No decode lever landed; the surviving cheap work is R3's pair/K-quant/
IQ4_XS nx tiles, which R4 needs for ~+10%.
