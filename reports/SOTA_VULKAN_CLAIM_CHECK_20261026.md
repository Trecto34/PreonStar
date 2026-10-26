# Claim check: "State of the art: Vulkan inference" vs p36

Checked 2026-10-26 against this checkout (`PreonStar`, fork of `Ninnix/q36`),
p36 docs/ledger, and the cited external sources (arXiv, llama.cpp API, Vulkan
registry). No builds, no benchmarks run; every number below is either quoted
from the repo or fetched live.

## Verdict

The external half of the document holds up: the ExecuTorch paper, the WebGPU
dispatch study, the llama.cpp Intel attention PR, the A770 measurements, and
the `GL_NV_cooperative_matrix_decode_vector` work all exist with the stated
content. The engine half does not describe p36. It describes **upstream q36**,
and even there it describes a slower engine than the one in this repo.

Concretely:

1. All six tok/s figures and the "October 25, 2026" date are upstream q36
   README values. None appear anywhere in this repo's docs.
2. p36's own MTP result is the **opposite sign**: MTP is a measured loss here.
3. About ten "research directions" are already implemented, and six more are
   already in the rejection ledger with measurements.
4. Two citation defects: two different arXiv papers are cited as one study,
   and the "Intel PR initially supported F16 KV" chronology is not supported
   by the merged shader.

## 1. Wrong for this repo

| Claim in document | Reality in p36 | Evidence |
|---|---|---|
| Qwen3.6-35B-A3B: 90.63 → 151.22 tok/s @2048 | p36 publishes 640 prefill / **82 decode** @2048, same harness (greedy, 128 tokens, Q8_0 K/Q4_0 V) | `docs/benchmarks.md:95-105` |
| Qwen3.8-27B IQ3_XXS: 27.07 → 36.44 @2048, 20.18 → 25.32 @65536 | 27B is only published at ctx 1024: 171.09 prefill / **23.19 decode**. No 2k or 64k row exists. | `karpathy/CAMPAIGN.md:66,1053` |
| MTP gives 1.25-1.67x on BC-250 | MTP is a net loss: plain 22.68/23.04 t/s, `--mtp-draft 2` 21.17 (0.92x), draft 3 6.97 (0.30x); 6 of 6 pairs slower | `karpathy/CAMPAIGN.md:938-942`, `karpathy/AlreadyTried.md:1334` |
| "MTP remains experimental" | True here too, but docs call it "at most a slight speedup" | `docs/models.md:123`, `docs/configuration.md:31` |
| "Opportunistic" positive-temperature MTP mode | Does not exist in p36. MTP runs only when `temperature <= 0` and all penalties are neutral; acceptance is strict argmax equality. | `src/server/server_15_generate_job.inc:324-329`, `src/core/core_19_mtp_timing_report.inc:32-34` |
| Date "October 25, 2026" | Newest p36 ledger entry is 2026-10-24; CAMPAIGN last updated 2026-09-26 | `karpathy/AlreadyTried.md:2751`, `karpathy/CAMPAIGN.md:7` |

The q36 numbers are not fabrications; they are in the upstream README at
`Ninnix/q36` (verified: 90.63/151.22, 27.07/36.44, 20.18/25.32, plus Q8_0 K /
Q4_0 V, 128 tokens, `speed-bench/bc250-20261025`). They are simply not this
engine's results, and the MTP direction does not survive the fork.

## 2. Recommendations already implemented in p36

| Document says | p36 already has | Evidence |
|---|---|---|
| "Candidate: quantized matrix-vector kernels" | Per-format decode kernels: IQ2_S/XXS, IQ3_XXS/S, Q3_K, Q4_K, Q2_K, Q8_0; fused expert pairs; r4 dense variants | `src/vulkan/vk_05_stream_weight_tensor_unlocke.inc:280-330`, `src/vulkan/vk_08_q8_0_quant_wait_name.inc:462` |
| "Candidate: small-M kernels for verification" | Two-row dense IQ3_XXS +62.6%; MoE row-pair for 2-4 rows; GEMM path at >=128 rows; ordered serial fallback above 4 rows | `karpathy/AlreadyTried.md:1551`, `src/vulkan/vk_13_moe_build_tiles.inc:1277`, `src/vulkan/vk_01_prof_reset.inc:608` |
| "Candidate: split-KV decode attention" | Both `attn_decode_fused` and `attn_decode_split` + `attn_combine`; span tuned (default 512) with context-dependent tradeoff documented | `src/vulkan/vk_02_use_attn_qtile2.inc:57-86`, `src/vulkan/vk_12_attn_host_rows.inc:242,248` |
| "Candidate: consume quantized K/V directly, no expanded cache" | Inline q8_0/q4_0 dequant from the packed cache; no temp float cache | `vulkan/attn_decode_fused.comp:61-62,97-135` |
| "Candidate: GPU-resident control paths / avoid readbacks" | GPU top-2 token selection, default on (`P36_VK_ARGMAX` unset = enabled); GPU argmax rows for batched verify | `src/vulkan/vk_12_attn_host_rows.inc:563,613`, `src/vulkan/vk_01_prof_reset.inc:591-594` |
| "Candidate: command-buffer reuse, persistent descriptors" | 4-deep command-buffer ring, re-recorded per batch, one `vkQueueSubmit` + fence per batch (32-dispatch flush); descriptor pool reset per slot; grow-only scratch | `src/vulkan/vk_03_tensor_alloc_kind_unlocked.inc:247,384-435`, `src/vulkan/vk_00_now_ns.inc:514` |
| "Candidate: per-shape/regime dispatch" | GEMM vs matvec threshold, MoE GEMM min, attention quantile switches keyed on head_dim/GQA/KV types | `src/vulkan/vk_02_use_attn_qtile2.inc:2-30`, `src/vulkan/vk_01_prof_reset.inc:605-626` |
| "Prefix reuse / session batching" | Exact prefix check, disk KV (`<sha1>.kv`), row-batched sessions with ordered fallback | `docs/configuration.md:77,95,99`, `docs/server.md:57-61` |
| "Subgroup width as per-kernel parameter" | Already per-pipeline: 32 for allowlisted shaders, 64 for ptq1_0/q2_0/small_, device default otherwise | `src/vulkan/vk_02_use_attn_qtile2.inc:448-521` |

## 3. Recommendations already measured and rejected in p36

| Document says | p36 ledger says | Where |
|---|---|---|
| "Investigate integer dot product for low-bit weights" | Bit-exact packed signed dot rejected: ACO emits **zero** `v_dot4_i32_i8` on GFX1013, so `dotPacked4x8EXT` compiles to scalar emulation. Int dot remains only in 3 dense-extra shaders. | `karpathy/AlreadyTried.md:1379-1396`; `vulkan/dense_extra_decode.comp:116` |
| "Tune split-KV partition count" | Split-K decode rejected: bit-exact but 0.3-0.7x at 600-4k keys, ~1.0-1.17x only at 16k-64k | `karpathy/AlreadyTried.md:1222` |
| "Reduce dispatch/descriptor overhead" | Descriptor-cleanup variant: +0.34% against a 1.5% gate, rejected. Record overhead measured ~0.33 ms/token, judged not a factor. | `karpathy/AlreadyTried.md:42-60`, `reports/decode_overhead_breakdown.md:62` |
| "Tune subgroup width locally" | Wave32 A/B neutral: prefill 242.43 -> 240.00 (-1.00%), decode +0.00%, rejected | `karpathy/AlreadyTried.md:582` |
| "Kernel fusion to cut intermediates" | RMSNorm->q8_K fusion rejected (gate not met); MoE mid-buffer q8 fusion neutral, reverted | `karpathy/AlreadyTried.md:606,2680` |
| "GPU sampling / fewer readbacks" | Deferred top-2 readback measured 2.1% **slower**, rejected; draft-argmax-on-GPU priced at <=7.5% of a spec call, not built | `karpathy/AlreadyTried.md:2353,1855-1861` |
| "Wider multi-row kernels" | 8 rows/workgroup 45% slower (3.13 -> 4.55 ms/token), rejected; 3-row verify probe negative | `karpathy/AlreadyTried.md:2335,1780` |
| "Async compute / overlap" | No second queue, no transfer staging, everything host-visible coherent; only queue-family choice. Not tried, and no independent work identified. | `src/vulkan/vk_03_tensor_alloc_kind_unlocked.inc:32-45,385` |
| "Cooperative matrices / coopmat2" | No cooperative-matrix code anywhere in `src/` or `vulkan/`; BC-250 (GFX1013) exposes none. The document's "reference dataflow, not an available path" framing is correct. | `grep -rniE "cooperative_matrix|coopmat" src/ vulkan/` -> 0 hits |

## 4. External claims that check out

- **ExecuTorch paper**: `arXiv:2605.08195`, submitted 2026-05-05, real. Vulkan backend does per-linear-layer dynamic activation quantization; Appendix A.2 comparator is llama.cpp **OpenCL**, as the document says.
- **WebGPU study**: 876 -> 564 dispatches, +53% throughput, "same WGSL shaders, negligible memory traffic" — exact quote from `arXiv:2608.08730`.
- **llama.cpp PR 24406**: merged 2026-09-22, "vulkan: add Intel Xe flash attention optimization kernels". Two-phase split prefill/decode, pipelines keyed by `(head_dim, gqa_ratio)`, non-power-of-two GQA support. A770 numbers 44.39 -> 49.17 and 12.01 -> 15.11 tok/s (`tg128 @ d16384`) are in the PR comments, as cited. The 16x16x16 coopmat mismatch is a real reviewer concern in that thread.
- **Shaders**: `flash_attn_cm2.comp` exists at `5cdadfc3`, uses `GL_KHR_cooperative_matrix` + `GL_NV_cooperative_matrix2`, and decodes Q4_0/Q8_0 K/V inline.
- **`GL_NV_cooperative_matrix_decode_vector`**: committed in llama.cpp #23541 (2026-05-27) and #23991 (2026-06-08); issue #26299 is exactly a request to bump the Windows Vulkan SDK (1.4.313.2 -> 1.4.357) to get it. Issue #29373 (open) is the "glslc lacks GL_KHR_cooperative_matrix" build break, also as described.
- **Upstream q36 docs**: adaptive draft length (margin gate, 1..4 rows, tails to 8), multi-row verify, rollback, GPU token selection, Q8 activation-load sharing, opportunistic positive-temperature mode, prefix reuse, disk KV — all present in `Ninnix/q36` README.

## 5. Citation defects

1. **Two papers conflated.** The runtime section cites `2608.08730` ("Measuring and Reducing WebGPU Dispatch Overhead", Aug 2026) and then `2604.02344` ("Characterizing WebGPU Dispatch Overhead ... Four GPU Vendors ...", Feb 2026) as if they were the same study. They are different papers.
2. **"Intel PR initially supported F16 KV; a later follow-up targeted Q8/Q4 KV."** Not supported. The merged cooperative-matrix attention shader decodes Q4_0/Q8_0 K/V inline at the cited commit. What is true is narrower: the new Intel `flash_attn_hdim*` kernels are a separate path from the generic quantized-KV FA path, and other llama.cpp work on quantized-KV FA exists separately (#20797 DP4A, #30738 open Q4_0 dequant+transpose). Keep the "support != fast path" caution; drop the chronology.
3. **"ExecuTorch evaluation shows prefill and decode can favor different implementations."** The paper's actual finding is narrower: ExecuTorch's attention is much faster at prefill and much slower at decode than the comparator. "Different implementations per regime" is the document's extrapolation, not the paper's result.
4. **"Final merged work focused on split-phase decode, prefill tracked separately."** Partly right: the PR merged single-phase prefill shaders too; a comment says the consolidated prefill shader was the part still in progress.

## 6. What the document misses (the actual p36 frontier)

1. **MTP verify cost is the open problem, not MTP kernel throughput.** One measured cycle: `fwd` 51.70 ms, `draft` 4.84, `verify` 391.03 (max 482), `commit` 0.01 (max 51.55 on partial accept), 4850 KB read — verify dominates (`karpathy/CAMPAIGN.md:942`). Upstream q36 claims newer MTP kernels worth +19% MTP throughput (Q8 activation-load sharing across rows, shared-expert multi-token verify, 8-head K/V reuse). Checking whether this fork carries them is the one genuinely useful "port from upstream" action left.
2. **W8 KV-dequant scratch cache** is the only performance item the ledger lists as identified-but-unstarted (`karpathy/AlreadyTried.md:808-818`); QT-widening was rejected on a VGPR-ceiling probe.
3. **MTP is disabled in batched mode** (`docs/server.md:49`), so the document's implicit combination of speculative verification with multi-session row batching is not available here.
4. **Test gaps** the document's validation list correctly targets: no NaN/Inf injected into activations (only sampling-level), no q8_0/q8_0 or mixed f16-K + quant-V pairs, GQA 1/2/4 untested (only 6 and 8), no explicit recurrent-state rollback assertion, no sliding-window masks (engine has no SWA).
5. **Baseline context**: p36 leads upstream llama.cpp on every file it can load — MoE 909.49/87.37 vs 615.08/78.72, dense 171.09/23.19 vs 105.43/22.13 at ctx 1024 (`karpathy/CAMPAIGN.md:63-65`). A llama.cpp-derived "SOTA" survey is therefore one generation behind this engine's kernel layer, which weakens the framing more than any individual claim.

## 7. Minimal edits to the document

1. Retitle the benchmark table "upstream q36 (`Ninnix/q36`, README)" and add p36's own rows (82 t/s decode @2048; 23.19 t/s @1024 for 27B).
2. Delete the "October 25, 2026" date or attribute it to upstream.
3. Replace the MTP section's implied gains with p36's measured loss and the verify-cost breakdown.
4. Move "split-KV", "integer dot", "subgroup tuning", "descriptor cleanup", "command-buffer reuse", "small-M kernels", "quantized-KV direct consumption" from "candidate" to "already implemented or already rejected", with the ledger numbers.
5. Fix the 2604.02344/2608.08730 conflation and the F16-KV chronology.
6. Keep: the support-vs-acceleration distinction, the traffic model, the WebGPU-is-not-native caution, the exactness taxonomy, and the validation/test-gap list.
