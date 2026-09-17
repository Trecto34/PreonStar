# Local Swift campaign ledger

This ledger starts from the completed Swift bring-up in `changelog.txt`.

| Result | Finding |
|---|---|
| IQ4_XS rejected | Bounded streaming ran, but dense-weight turnover was about 0.11 tok/s; resident loading triggered the host OOM/SIGKILL path. |
| IQ3_XXS accepted as baseline | Completed 11.67 GiB model; short protected smoke passed; q36-bench measured about 180 tok/s prefill and 23 tok/s generation at context 1024. |
| MTP draft=3 rejected as default | Slower than draft=1 on the measured workload. |
| attention span=128 rejected as default | Helped at context 1024 but regressed at context 2048; span 512 remains the stable default. |

Do not repeat these hypotheses without a new hardware or runtime reason.
Preserve MoE routing parity for the Qwen3.5/Qwen3.6 reference on every
accepted change, even when the primary target is dense Swift Qwen3.8.

## Interrupted experiment: 2026-09-17 12:00 (DeepSeek worker)

- **IQ3_XXS dense MMQ lane split — REJECTED/INTERRUPTED:** split the
  `dense_iq3_xxs_mmq` dequant staging across all 128 lanes. The first build
  failed because `half` is a GLSL reserved word; after renaming it, the
  candidate was bit-exact but regressed the down-projection microbenchmark by
  about 2.7% and did not establish an end-to-end win. The compatibility gate
  passed, but the worker was terminated before its required final report.
  Do not retry this lane-split mechanism without a measured explanation for
  the down-projection regression.

## BC-250 transfer audit: 2026-09-17

- **512-thread add/RMS — REJECTED:** Swift kernel 23.64 -> 30.13 ms.
- **Integer sign-mask IQ3_XXS decode — REJECTED:** Swift decode kernel
  373.83 -> 407.97 ms over 16 generation tokens.
- **XOR-swizzled IQ3_XXS MMQ staging — REJECTED:** Swift prefill MMQ kernel
  940.28 -> 1309.31 ms at 128-token context. The existing 17-element padded
  stride is faster than the attempted 16-element XOR layout on BC-250.
- **Global IQ2 grid lookup in MoE gate/up GEMM — REJECTED:** 128-token kernel
  time improved slightly (197.71 -> 194.26 ms), but 1024-token whole-model
  prefill was not reliably faster. Shared staging remains the default.
- **Smaller prefill chunks — REJECTED:** on 1024-token prompts, the existing
  watchdog-safe caps (Swift 256, Qwen3.6 1024) were fastest among tested
  chunk sizes. Do not raise Swift above 256 without a watchdog-safety proof.
- **IQ3_XXS MMQ token tiles 64 or 256 — REJECTED:** with matching dispatch
  divisors, they produced 161.07 and 166.87 prefill tok/s respectively at
  context 1024, versus 172.41 tok/s for the existing 128-token tile.
- **IQ3_XXS MMQ dynamic k loop — REJECTED:** removing `[[unroll]]` raised
  128-token MMQ GPU time from about 940 to 960 ms.
- **IQ3_XXS decode rows 8 — REJECTED:** 128-token generation throughput
  matched the existing four-row shader at 24.14 tok/s; no speedup.
- **Swift MTP draft depth 2 — REJECTED:** 23.28 tok/s over 128 generated
  tokens against 24.14 tok/s for plain depth-1 decode.

## Peer BC-250 cluster ledger: 2026-09-17

External results from a two-node BC-250 pair (gfx1013, RADV/Vulkan) running a
tq-patched llama.cpp fork. Recorded so the same dead ends are not reopened
here. Their stack is not ours — llama.cpp + GGML shaders, where ours is a
bespoke Vulkan engine — so their kernel-level wins do not port directly.
The NULLS do, and those are what this section exists for.

- **MOVNTDQA streaming loads** for the readback memcpy — measured SLOWER than
  plain loads; the mapping behaves UNCACHED, not write-combined.
- **Warptile BK 32->64** (halve barrier count) — cost -54% prefill; occupancy
  fell from 3 to 2 waves/SIMD. Barrier count is not the prefill limiter.
- **RADV_PERFTEST=nogttspill** — DISPROVEN as a 2x lever: 19.45 vs 19.47 t/s
  on/off. The old "18 -> 9 t/s" claim was refuted.
- **shader_core_count 24->40 on diffusion workloads** (image/video gen) —
  NO-OP, 0.16% wall clock; split-K never engages on large diffusion matmuls.
  Their CU-count win is LLM-decode-specific, not a blanket GPU win.
- **Decode-ubatch pipelining np=16/ub=8** — the "+11.3% overlap" was an
  artifact of staying under the mul_mat_vec_max_cols=8 kernel-path cliff, not
  real overlap; np=16 is -44.9% vs np=8 overall. Closed as a net loss.
- **L2 lever (LLAMA_OUTPUT_CPU_BUF=1)** — NO WIN / inconsistent vs control
  once actually A/B'd, despite a +25% prediction for MTP verify.
- **MTU 1500 vs 9000** on the RPC link, real-inference rerun — NULL; all
  deltas fell inside 4-6% drift. Stays at 1500.
- **Split-heap / app-block readback trick under MTP + q4_0 KV** — no win,
  slight loss (29.99 vs 30.78 t/s); does not generalize when MTP is on.
- **B2-1 change 2** (per-block norm/sign hoist) — DISPROVEN, neutral to
  negative. **B2-1 change 3** (two-level f16 P.V rewrite) — scoped, judged
  not worth the build cost. Both off the table.
- **GEMM_FORCE_L_PERTYPE** — RETRACTED: the original +20.8% prefill was a
  24-CU-starved-machine artifact; INCONCLUSIVE (-0.2%) once re-measured at
  true 40/40 CUs.
- **tq fork vs mainline** — CONFOUNDED (build and KV type moved together);
  the TG residual (~17 pts) is an open unknown, not a usable comparison.

### Applicability audit against this repo (2026-09-17)

- **HOST_CACHED readback lever — NOT APPLICABLE, already implemented.** They
  needed `HOME=/tmp/emptyhome` to bypass a driconf unified-heap setting in
  order to expose a HOST_CACHED GTT type, and measured +20.3% dense decode.
  On this box `vulkaninfo` already reports HOST_CACHED types [5], [6] and [10]
  with unified heap ON (`/usr/share/drirc.d/00-radv-defaults.conf:135`), and
  `q36_vulkan.c:1693-1698` already selects
  HOST_VISIBLE|HOST_COHERENT|HOST_CACHED for every non-weight buffer, exactly
  as its own comment intends. Nothing to port, no gain available here.
- **shader_core_count 24->40 split-K lever — NOT APPLICABLE.** This engine has
  no shader-core-count query at all, so there is no scheduling input to
  correct. Their effect was correcting a driver-reported split-K sizing input.
- **Wave32 — DEPRIORITIZED (evidence added 2026-09-17).** See
  `q36_vulkan.c:1569`; the source-level force covers prefill MMQ only. But
  llama.cpp's Vulkan backend on this same device reports `warp size: 32` out of
  the box, i.e. the driver's default compute subgroup is already 32-wide, so
  the peer's `RADV_PERFTEST=cswave32` +1.3% has no premise in a backend that
  already runs 32. Re-verify whether q36's decode really dispatches at 64
  before spending a build on it; `tests/bench_cswave32.sh` stays as the
  harness if it does.
- **f16-grid LDS staging (their iq3_s GEMV port, +14.0% on-shape)** — the one
  peer kernel technique plausibly applicable here, same family as their
  IQ3_XXS dequant VGPR fix. Candidate for the dense_iq3_xxs_decode 269 GB/s
  vs sibling iq4_xs 333 GB/s gap. Not yet tried.

### llama.cpp head-to-head + the integer-dot dead end (2026-09-17)

The peer wins begged a fair question: is the q36 work worth anything next to
upstream? Measured it. Same box, same GGUF
(`/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf`), same ctx 1024,
`-ngl 99`, Vulkan, `llama-bench -p 1024 -n 16 -r 3`.

| engine | prefill t/s | decode t/s |
|---|---|---|
| upstream llama.cpp `0cae430` (built Sep 7) | 107.81 (+/-4.84) | 21.47 (+/-0.61) |
| upstream llama.cpp `972d231` (pulled + rebuilt) | 105.43 (+/-4.46) | 22.13 (+/-0.65) |
| **this repo (q36)** | **171.09** | **23.19** |

- **q36 is +58.7% prefill / +8.0% decode over stock upstream.** Not useless.
  The decode margin understates it: llama-bench `tg` is greedy, while q36's
  23.19 carries the full temp-0.6 / top-k / top-p / min-p sampling chain.
- **The upstream update bought nothing.** `0cae430` -> `972d231` plus a full
  reconfigure and rebuild moved prefill -2.2% and decode +3.1%, both inside the
  prefill stddev of 4.4. Do not re-litigate this: version freshness is not a
  lever here. The peer's GET_ROWS -> `vkCmdCopyBuffer` rewrite (+0.4% TG) is
  already in-tree at both commits.
- **`int dot: 0` is a hardware truth, NOT a build bug. RETRACTED CLAIM:** I
  first reported it as "the same failure shape as the peer's, one reconfigure
  away." Wrong, twice over. The CMake gate
  (`ggml-vulkan/CMakeLists.txt:38-57`, a plain `execute_process` of `glslc`)
  does pass with glslc 2026.3, the define
  `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` does reach
  `build/.../ggml-vulkan.dir/flags.make`, and the shaders do carry
  `dotPacked4x8EXT` (40 hits in `mul_mat_vecq_funcs.glsl`, 14 in
  `mul_mmq_funcs.glsl`) — and the device still reports `int dot: 0`, because
  `ggml-vulkan.cpp:4019` additionally demands
  `shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated`.
  On GFX1013 **every** accelerated variant is false
  (`integerDotProduct8BitSignedAccelerated`,
  `4x8BitPackedSigned/UnsignedAccelerated`,
  `16BitMixedSignednessAccelerated` — all `false`). RADV advertises the
  extension but no accelerated path. No flag, no pull, no shader edit fixes it.
- **Why the peer's wins still transfer.** Their headline +17-24% prefill came
  from `VK_VALVE_shader_mixed_float_dot_product` being absent, so f16 dots fell
  back to serial scalar FMAs. Their fix was NOT an extension: they restructured
  `mul_mm.comp` so both row accumulators are independent and ACO emits packed
  `v_pk_fma_f16` (bit-identical output). Same story for the iq3_s GEMV port
  (+14.0%) — pre-convert the grid to `f16vec4` in LDS so the math runs packed.
  **General rule for this driver: dot-product intrinsics are unavailable, so
  packed-native restructuring is the only lever.** Both llama.cpp and q36 run
  under that handicap, which is what makes q36's 58.7% an engine result rather
  than a build accident.
- **Two corrections to this repo's own notes.** (a) The earlier "RADV exposes
  no shader-core-properties on GFX1013" was stale — all three arch-gate
  extensions (`VK_AMD_shader_core_properties2` rev 1,
  `VK_KHR_shader_integer_dot_product` rev 1, `VK_EXT_subgroup_size_control`
  rev 2, min 32 / max 64) are exposed and the RDNA gate at line 424 passes.
  (b) `bf16: 0` is likewise correct and permanent: `VK_KHR_shader_bfloat16` and
  `VK_VALVE_shader_mixed_float_dot_product` are genuinely absent (grep = 0), so
  the peer's packed-FMA fix must be a shader patch here, never a flag.
- **`GGML_VK_SHADER_CORE_COUNT` does not exist upstream** — the complete
  `getenv` inventory is 23 `GGML_VK_*` variables and it is not among them. That
  is a stable-diffusion.cpp variable, consistent with the peer's own finding
  that its CU lever is a no-op for diffusion and LLM-decode-specific.
