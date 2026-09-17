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
- **FA LDS staging / `FA_SHMEM_STAGING` on `attn_prefill_qtile2` — REJECTED, NEUTRAL (2026-09-17).** codex +20/-5; A/B 7 interleaved reps: prefill 171.47 (MAD 0.660) -> 170.75 (MAD 0.480) = **-0.42%**, decode +0.00%, gate +1.50% -> FAIL. Inside the 0.7% dense floor -> drift, not a win. The peer reported PP +5.4% / TG +12.4% for llama.cpp; unconfirmed here rather than disproven (attention is not top-3 in this model's prefill profile, so an e2e A/B is blunt for this kernel). Evidence: `evidence/raw/ab-w10-fa.csv`, patch `evidence/raw/w10-fa.diff`.
- **Restaged `dense_iq3_xxs_mmq` (double-buffered A, register-built B) — REJECTED, NEUTRAL (2026-09-17).** codex +71/-63: A double-buffered, `buf_b` deleted (B operand built via `subgroupShuffle`), trailing barrier conditional. Prefill 170.93 (MAD 0.080) -> 170.75 (MAD 0.660) = -0.11%, decode +0.05%, FAIL. Not staging-bound: it already packs f16. Evidence: `evidence/raw/ab-w8-mmq.csv`, patch `evidence/raw/w8-iq3mmq.diff`.
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
- **Wave32 for the non-mmq paths — RUN, MEASURED NEGATIVE (2026-09-17).** The
  earlier note in this entry was wrong on one point and it cost a re-check: it
  reasoned from *llama.cpp's* `warp size: 32` that the driver default is already
  32-wide, so the peer's `RADV_PERFTEST=cswave32` had "no premise". That is
  llama.cpp's backend, not this engine — q36 prints `subgroup 64` on the same
  device (`q36_vulkan.c:1569` forces only `delta_net_cols`, `rope_qwen`,
  `rope_qwen_mrope`, `quantize_q8_0` and the `dense_*_mmq` family). So the
  premise WAS live here and the lever earned one run. It got it: Swift 27B,
  7 interleaved reps, ctx 1024, `tests/bench_cswave32.sh` — prefill **171.14 ->
  169.88 = -0.74%** (MAD 0.600 / 1.540). Decode 19.69 -> 18.67 raw looks like
  -5%, but that is an artifact of A's own cold start (A reps 1-2 ran 23.66/21.79
  before the die settled at 66-68 C; steady-state A3-A7 median 18.58 vs B 18.67
  is a **tie**). Verdict: **no gain**, prefill negative in all seven pairs.
  Structural reason, from the shader census: the hot kernels are *already*
  wave32-forced (`dense_*_mmq`, `q36_vulkan.c:1575`), so the flag only moved
  kernels off the critical path. A per-kernel source widening of the
  one-cross-lane-op decode kernels is still possible but the measured ceiling is
  now ~0, so it is no longer ranked. Evidence: `evidence/raw/ab-cswave32.csv`.
- **`dense_iq3_s_mmq_r4` escaped the wave32 force predicate — CORRECTNESS BUG,
  found by the wave32 eligibility audit, FIXED (2026-09-17).** The force
  predicate is a *suffix* test:
  `strstr(k->path,"dense_") && strstr(k->path,"_mmq.spv")`.
  `dense_iq3_s_mmq_r4.spv` (built at `Makefile:198-202` with
  `-DQ36_BM=4 -DQ36_BK=64`) ends `_mmq_r4.spv`, so it never matched and the
  shader ran at wave64. That shader partitions its token tile by
  `gl_SubgroupID`: `tok = tok0 + warp_tok*32 + (tt>>1)*16 + thread_tok*2 +
  (tt&1)`, `thread_tok = lane>>2`, and the store covers a 128-token tile. At
  wave32 (4 subgroups) the formula reaches **127**; at wave64 (2 subgroups) it
  peaks at **79**, so **tokens 80-127 of every tile were written by nobody**.
  Reachable from the `small_rows` selection site (`q36_vulkan.c:8160-8170`:
  IQ3_S weights, `out_dim == 48 && n_tok <= 128`). Fix: predicate widened to
  `_mmq` and the eligibility comment now names both traps (`gl_Subgroup*` reads
  count as cross-lane use; a `gl_SubgroupID`-partitioned tile makes wave32 a
  *correctness* requirement, not a preference). Predicate delta is exactly one
  pipeline — `dense_iq3_s_mmq_r4.spv` 0 -> 1, the other six `dense_*mmq*.spv`
  unchanged. A/B, 7 interleaved reps, Swift IQ3_XXS ctx 1024: prefill **170.92
  (MAD 0.290) -> 170.87 (MAD 0.320) = -0.03%**, decode **18.59 -> 18.58** =
  -0.05% = **no change, as expected**: none of the three local models selects
  the touched pipeline (they are IQ3_XXS / IQ2_XXS / IQ2_M), so the bug is
  **latent** and this is a landmine closed, not a speed win. Compatibility gate
  PASS on the patched binaries. Merged as `6e999eb`. Evidence:
  `evidence/raw/w11-wave32-r4-fix.txt`, `evidence/raw/w11-wave32-r4-ab.csv`,
  `evidence/raw/w11-wave32-r4-ab-summary.txt`.
- **f16-grid LDS staging (their iq3_s GEMV port, +14.0% on-shape) — TRIED,
  MEASURED NEUTRAL, rejected (2026-09-17).** Ported to
  `vulkan/dense_iq3_xxs_decode.comp` (+21/-22): `grid_lut` staged as `f16vec4`
  in LDS (1 KiB -> 2 KiB), all 256 entries converted once per workgroup instead
  of per-lane per row, signs applied packed in half, promoted back to float at
  the *unchanged* FMA boundary (nesting and order untouched, so it should be
  exact: grid components are integers <= 62, representable in binary16, x +-1 is
  exact in half, promotion is exact). Built clean and **both**
  `dense_iq3_xxs_decode.spv` and `dense_iq3_xxs_decode_r4.spv` rebuilt, so the
  edit did reach the variant the profile names. Measured, 7 interleaved reps:
  prefill **170.67 -> 170.65 = -0.01%**, decode **18.48 -> 18.55 = +0.38%**
  (MAD 0.100 / 0.020), steady-state decode 18.48 vs 18.48 = **0.00%**. Both
  inside the noise floor => **no gain**. The 269 vs 333 GB/s gap is therefore
  NOT grid-LUT conversion cost; at this occupancy the kernel is already at its
  practical limit. Do not re-port this technique to this kernel.
  Evidence: `evidence/raw/ab-w9-decode.csv` + `ab-w9-decode-summary.txt`.
- **Thermal margin is now a measured hazard to a single rep,** not a theory: in
  the w9 A/B, rep A,5 hit 69 C and collapsed to 147.52 prefill / 9.89 decode
  (-13% / -46%) while every other rep held 169.6-172.2. Only the median is a
  valid number on this box; a single-rep decode comparison is meaningless.

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

## Cross-engine MoE matrix + the IQ2_M loader wall (2026-09-17)

Full table, asymmetries and raw CSVs: `evidence/cross-engine-matrix.md`.
Same box, same GGUFs, ctx 1024, `-ngl 99`, `llama-bench -p 1024 -n 16 -r 3`.

| model | q36 prefill / decode | llama.cpp `972d231` | delta |
|---|---|---|---|
| Qwen3.6-35B-A3B IQ2XXS (MoE guard) | 909.49 / 87.37 | 615.08 / 78.72 | **+47.9% / +11.0%** |
| RavenX-35B-Q36-IQ2XXS | 717.36 / 89.47 | 545.46 / 77.96 | **+31.5% / +14.8%** |
| Qwen3.8-35B-A3B-IQ2_M | refuses to load | 529.90 / **91.53** | n/a |

- **`Qwen3.8-35B-A3B-IQ2_M` is a q36 loader limitation, not a bad file.** Upstream
  loads it (35.5B params, 2.7 bpw, 99/99 layers on GPU) and posts the fastest MoE
  decode measured on this box — 91.53 +/- 2.11, above q36's best (RavenX 89.47).
  q36 stops it at `q36: expected qwen35moe.block_count=40, got 41` (3/3 reps,
  exit 1). This is the highest-value open item in the repo and it is a loader
  condition, not a kernel.

  **Root cause located (2026-09-17, do not re-derive):** the guard is
  `q36.c:5377-5381` in `config_validate_model`, and it *already contains the
  escape hatch this file needs* — but it is gated on `Q36_MODEL_DENSE`:

  ```c
  if (block_count != Q36_N_LAYER &&
      !(Q36_MODEL_DENSE && nextn_layers == 1 && block_count == Q36_N_LAYER + 1)) {
  ```

  Read from the GGUF itself (`gguf-py`): IQ2_M is `block_count=41`,
  `nextn_predict_layers=1`, blk 0..40, 753 tensors, and **`blk.40.*` is a
  complete MoE block** (attn_q/k/v/output + norms, ffn_gate/up/down_exps,
  ffn_gate_inp, shared-expert tensors). The guard has 40 blocks / 733 tensors and
  no `nextn` key; RavenX likewise. So IQ2_M is a 40-block model **plus one
  NextN/MTP block** — structurally the shape the *dense* path already accepts.
  A second dense-gated check at `q36.c:5419-5420` allows
  `Q36_TENSOR_COUNT + 15` (dense NextN = +15 tensors; MoE NextN = +20 by
  arithmetic on 753 vs 733). The engine has a non-dense MTP binder
  (`mtp_weights_bind`, `q36.c:4454`) but the embedded-MTP *skip* is dense-only:
  `q36_tensor_is_disabled_embedded_mtp` returns false when `!Q36_MODEL_DENSE`
  (`q36.c:5177`).

  ⇒ **Reconsider_if:** extend both gates to the MoE shape *and* route block 40
  through the embedded-MTP skip. **RESOLVED 2026-09-17 — implemented, measured
  and accepted as `f9d537d`; see "w7: the 41-block loader fix" below.** Relaxing the check alone is NOT the fix: the
  extra block's weights (~800 MB on a 15.35 GiB unified pool) would load into a
  40-block graph. Verify by loading IQ2_M, then A/B against the guard for
  no-regression.
- **MoE prefill spread is 3.4%, not the 27B's 0.7%.** Identical code produced
  713.98 -> 738.81 t/s on RavenX. Do not report a MoE prefill delta under ~3%;
  the MoE guard's acceptance band is 3.4%, not 1.5%.
- **RavenX out-decodes the guard (89.47 vs 87.37) while its prefill is 21%
  lower**, despite the two files differing by 128 bytes in length. That is quant
  mix, not layout — no single "MoE ≈ 900 t/s prefill" figure covers both files.

## Host query-pool drain removal — REJECTED as a speed lever (2026-09-17)

- **Removing the mid-dispatch query-pool drain at `q36_vulkan.c:2981-2983`** —
  the thing that produced the "94 us/dispatch" `dense_q4k_decode` anomaly — is
  **INERT in production**. The block was gated on `q36_vk.prof_kernel`, and that
  flag is only set when `Q36_VK_PROF_KERNEL` is present in the environment
  (`q36_vulkan.c:3465-3474`). Interleaved 7-rep A/B on Swift at ctx 1024:
  A 170.91 (MAD 0.650) vs B 171.68 (MAD 0.550) → **+0.45% prefill, +0.43%
  decode, FAIL** against the +1.5% gate. The MoE guard run agrees: **712.75
  (MAD 6.660) vs 711.95 (MAD 9.330) → −0.11% prefill, −0.25% decode, FAIL**,
  both arms inside the MoE noise band. That is one MAD of noise on each model,
  measured on two binaries that *cannot* differ in the configuration
  bench_ab.sh runs. Do not re-open the 94 us/dispatch anomaly as a speed lever —
  it is profiler overhead, not a stall. Breakdown and both raw CSVs:
  `evidence/ab-hostdrain.md`, `evidence/raw/ab-drain-verdicts.txt`.
- **Rule learned: a change gated on a runtime flag must be A/B'd with that flag
  set, or not A/B'd for speed at all.** Check the gate before designing the
  experiment; otherwise the null result is guaranteed and reads like a finding.
- **`tests/bench_ab.sh`'s verdict `awk` was broken and is now fixed.** It passed
  the SUBSEP-joined multipart arrays (`p[lab,k]`, `d[lab,k]`) straight into
  `med()`, which indexes its argument with plain integers — so every read hit an
  uninitialized element and `med()` returned 0. Every run therefore printed
  `prefill_med 0.00` / `decode_med 0.00` and a fake `+0.00%`, i.e.
  `verdict: FAIL (prefill)` regardless of the data. Fixed by flattening into the
  copy arrays first, and verified by re-running the patched block over the same
  raw CSV (170.91 / 171.68 / +0.45%, identical to an independent Python median).
  **Any A/B verdict this script produced before this fix is unusable — recompute
  it from the CSV.**

## w7: the 41-block loader fix — ACCEPTED as a capability fix, NOT a speed win (2026-09-17)

Commit `f9d537d` on `experiment/w7-moe-nextn` (off `56126ca`), `q36.c` only,
+66/−22. All three dense-gated conditions identified above were extended to the
MoE shape:

- `config_validate_model`: `embedded_nextn_block = nextn_layers == 1 &&
  block_count == Q36_N_LAYER + 1`, now accepted for any model kind; the
  tensor-count gate is `+15` dense / **`+20` MoE** (753 − 733).
- `q36_tensor_is_disabled_embedded_mtp`: `!Q36_MODEL_DENSE` dropped, so the
  `blk.<Q36_N_LAYER>.` prefix skip applies to MoE too. `e->mtp_ready` still
  short-circuits, so sidecar-MTP runs are unaffected; a 40-block file has no
  `blk.40`, so the guard is unchanged by construction.
- `weights_validate_layout(w, mixed_moe_quants)`: new `tensor_expect_moe_matrix()`
  helper. Under the predicate `!Q36_MODEL_DENSE && n_tensors == Q36_TENSOR_COUNT + 20`
  (exactly the 753-tensor IQ2_M shape) the trunk/non-expert matrix types are
  accepted with dims still validated (`tensor_expect_cpu_matrix`) instead of the
  Q4_K/Q5_K/Q6_K/Q8_0 whitelist. **The routed experts keep their original strict
  `tensor_expect_routed_*` validation.** This is a type-whitelist widening on the
  trunk, not a check removal, and the predicate cannot reach the guard or any
  dense file. Describing it as "a validation relaxation" overstates it; describing
  it as "loads" alone understates nothing but proves nothing either.

Measured:

- **Loads and runs:** `116.51 / 29.29` tps (profiled). The 3/3 refusal is gone.
- **Quality, specifically targeting the relaxed check:** `q36_test --gpu-cpu-parity`
  **OK** on IQ2_M — top1 `ref == cand` at all 3 steps (16/16, 21/21, 248046/248046),
  rms 0.263/0.208/0.272, top15 15/15, 14/15, 13/15. Guard parity also **OK**
  (top1 16/21/248046 match, rms 0.31/0.24/0.32). A mis-dispatched quant type would
  break top1 agreement; it does not.
- **No regression on the guard:** 7-rep interleaved A/B, ctx 1024 —
  A `713.80` prefill / `88.98` decode (MAD 6.180 / 0.110) vs
  B `713.85` / `89.12` (MAD 10.570 / 0.080) = **+0.01% / +0.16%** —
  indistinguishable. `bench_ab.sh`'s `verdict: FAIL (prefill)` is its *speed-gain*
  semantics, not a regression; read the medians, not the label. A's own rep spread
  (704.62 → 737.83) also reclassifies the earlier single-run reading of `686.84`
  as cold-run noise, not a signal.
- **Merged tree re-verified after the merge + rebuild of the main checkout**
  (main `q36-bench`, 19:25): IQ2_M `117.98 / 32.36` exit 0, guard `715.16 / 88.63`
  exit 0 — the committed source is what produced the accepted numbers, not just
  the worktree build.
- Raw: `evidence/raw/ab-w7-guard.csv`, `w7-parity.txt`, `w7-iq2m-prof.txt`,
  `w7-guard-patched-prof.txt`, `w7-verify-main.txt`. Binary sha256:
  A `ffd2d262…8883`, B `3264f5dc…afbb`.
- **Accepted because** it makes a file the engine could not open load *and* decode
  with verified parity, at zero measured cost to the guard. **Reconsider_if:** a
  future MoE GGUF of the same 753-tensor shape ships non-trunk types the generic
  dispatch does not actually handle — the predicate is by tensor count, so it would
  not re-check the whitelist. Run `--gpu-cpu-parity` before trusting such a file.

## IQ2_M's slowness is kernel SELECTION, not the loader (2026-09-17, decisive)

**Do not look for another loader fix, and do not read "it loads" as a win.**
IQ2_M under q36 is 4.5×/3.1× slower than upstream on the identical file
(116.51 / 29.29 vs 529.90 ± 11.48 / 91.53 ± 2.11).

- **Quant inventory** (`gguf-py` `GGUFReader`): IQ2_M's trunk is **375× `IQ2_S`
  (10.3 GB)**; the guard is **`IQ2_XXS` + `Q2_K`**. Different quant family — and
  every tuned prefill path in this engine is quant-specific.
- **Profiled IQ2_M:** `moe_matvec` **11,845 ms = 67% of all GPU time** (generic
  per-expert matvec) plus `dense_kquant` **4,662 ms**, versus the guard's
  `moe_iq2_gate_up_gemm` 878.7 + `moe_q2k_down_gemm` 432.6 and `dense_q8_0_p_*`
  GEMMs (~950 ms) on the *same* binary.
- `IQ2_S` **is** present in-tree (`moe_matvec.comp`, `moe_matvec_fast.comp`,
  `dense_extra_*`) — so the lever is **quant-aware dispatch into the per-quant
  prefill GEMM family**, i.e. in-tree prior art, not new shader work.
- Corollary, recorded so it is not re-derived: the "IQ2_M is the fastest MoE
  decode on this box" thesis **does not transfer to q36** until that dispatch
  exists. Ranking the loader fix as a *speed* item was this campaign's error.

