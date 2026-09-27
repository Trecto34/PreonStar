# Swift-1.5 GSQ-RCO IQ3_S — IQ1_M prefill on the dense_extra mmq tile (2026-09-26)

Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` (qwen35 dense, 64 blocks).
Despite the name it is a per-tensor mixed quant (`model-scan.txt`, GB):
IQ3_S 3.64, IQ4_XS 3.02, IQ3_XXS 2.08, Q4_K 1.58, IQ2_S 0.80, Q2_K 0.24, plus
IQ2_XS x9, IQ2_XXS x5 and one IQ1_M tensor (`blk.13.ffn_gate`, 5120 -> 17408, 19 MB).

## Baseline (HEAD 6dd3c89, ctx 1024, chunk 256, gen 16, 3 runs)
prefill 169.22-170.38 t/s, decode 18.81-19.92 t/s (`baseline-*.csv`).
Per-kernel profile, ctx 512, gen 32 (`prof.log`, `Q36_VK_PROF_KERNEL=1`):
`dense_iq1_m` = 263.6 ms, **3.5% of GPU time for 0.16% of the weights**. The naive
kernel dispatches one workgroup per (row, token), so prefill re-reads the whole
tensor once per token. Every other type already routes to a tuned kernel.

## Change
- `vulkan/dense_extra_mmq.comp`: IQ1_M arm in the generic main (same grid and
  element mapping as IQ1_S; block scale assembled from the four scale-word nibbles).
- `q36_vulkan.c`: IQ1_M with `n_tok > 8`, not `--quality`, and
  `Q36_VK_DENSE_IQ1M_MMQ` not 0 -> `dense_extra_mmq`. Decode and 2..8-row
  batches stay on `dense_iq1_m`. `dense_extra_mmq` added to the prewarm list.

## Correctness
- `tests/test_dense_iq1m_mmq` (`unit-iq1m.txt`): mmq vs one-token `dense_iq1_m`
  (f32), scale 0.37, rms_rel 0.0103-0.0104 at 17408x256, 17408x9, 1000x131. The
  shipped IQ1_S arm on the same harness: 0.0104 (the f16 staging floor).
  Fault injection (delta sign flipped): 0.32 FAIL. Route off: exactly 0.
- Swift frontier logits 512/768, route **off** vs pre-change: **byte-identical**
  (`frontier-sha256.txt`). The IQ2_XS/IQ2_XXS tensors on the rebuilt SPIR-V are
  therefore bit-exact, and all drift comes from the one IQ1_M tensor.
- KL(ref || x) against the f32 per-token reference (`--prefill-chunk 1`),
  `kl-vs-f32ref.txt`: frontier 512 pre-change 0.0019 / new 0.0031 nats, 768
  0.0007 / 0.0014. Top-1 matches the reference in both arms.
- Retracted: the first NLL/top-1 claim (`nll/p3stats.txt`) used misaligned targets
  (A's NLLs were 6-27 nats on confident predictions). Do not cite it.

## No-breakage (other models)
- SPIR-V: only `dense_extra_mmq.spv` changed. The `_iq2s`, `_q2_0` and `_pq2_0`
  builds are byte-identical. `RADV_DEBUG=shaderstats` (`stats-A/B.txt`): VGPR
  88/88, SGPR 108/108, LDS 11776/11776, 0 spills, code 16908 -> 18324 B.
- Model scan over gguf/, ~/models/llm and /mnt/external/models/llm
  (`model-scan.txt`): only Penjing-27B-IQ2_XXS and the two GSQ-RCO IQ3_S files
  reach generic `dense_extra_mmq`. Only the GSQ-RCO files have IQ1_M.
- Penjing frontier logits 512/768: byte-identical (`penjing/sha256.txt`).
  Prefill +3.55% / decode 0.00% (`ab-penjing.*`, 3 reps). With identical logits
  this comes from the prewarm moving a lazy pipeline compile out of the timed
  first prefill, not from any kernel change.
- `karpathy/compat_gate.sh`: **not applicable / infra timeout, not a pass**. Exit 124 (180 s timeout) in the
  MoE CPU-reference capture of a model read from the external HDD, before any GPU
  comparison. Not retried (AGENT.md: avoid long CPU inference runs). The model
  has no tensors of the affected types.

## Speed (`tests/bench_ab.sh`, interleaved)
| arms | reps | ctx/gen | prefill | decode |
|---|---|---|---|---|
| route off vs on (both prewarmed) | 7 | 1024/16 | 168.54 -> 176.98 **+5.01%** (MAD 0.38/0.18, no overlap) | bimodal, -9.7% median (noise, see below) |
| route off vs on | 3 | 512/64 | +5.72% | **+0.54%** (MAD <=0.02) |
| pre-change vs new (first build) | 5 | 1024/16 | +4.88% | -3.10% |
| pre-change vs new (first build) | 3 | 512/64 | +3.37% | +0.35% |

Decode at gen 16 falls into 17.6-17.9 and 19.5-20.6 t/s clusters in **both** arms
(`ab-iq1m-gate7.csv`). The one-token path is unchanged, and the 64-token runs are
flat, so the -9.7% is sampling noise.

## Verdict
**ACCEPTED** (adversarial review: REJECT, then ACCEPT-WITH-FIXES after the fixes above) — **+3.4-5% prefill** on the GSQ-RCO IQ3_S files, decode unchanged, other
models bit-identical. The +5.01% implies ~0.29 s saved per 1024 tokens, more than the
0.256 s the serialized profile gave the old kernel; that gap is unexplained (profiler
serialization or neighbour interference), hence the range. `dense_iq1_m` is still
not prewarmed (pre-existing). `reconsider_if`: an IQ1_M `dense_extra_decode` arm would
move this tensor's decode off the naive kernel (its decode share was not measured separately).
