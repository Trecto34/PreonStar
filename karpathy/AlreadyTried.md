# Local Swift campaign ledger

This ledger starts from the completed Swift bring-up in `changelog.txt`.

| Result | Finding |
|---|---|
| IQ4_XS rejected | Bounded streaming ran, but dense-weight turnover was about 0.11 tok/s; resident loading triggered the host OOM/SIGKILL path. |
| IQ3_XXS accepted as baseline | Completed 11.67 GiB model; short protected smoke passed; p36-bench measured about 180 tok/s prefill and 23 tok/s generation at context 1024. |
| MTP draft=3 rejected as default | Slower than draft=1 on the measured workload. |
| attention span=128 rejected as default | Helped at context 1024 but regressed at context 2048; span 512 remains the stable default. |

Do not repeat these hypotheses without a new hardware or runtime reason.
Preserve MoE routing parity for the Qwen3.5/Qwen3.6 reference on every
accepted change, even when the primary target is dense Swift Qwen3.8.

## Q2_0 decode inner-loop coalescing + latency audit — REJECTED, FREEZE (2026-09-18)

Branch `experiment/bc250-sustained-20260918`, HEAD `c133a09`, model
`Ternary-Bonsai-2-27B-Q2_0-g64.gguf`. Full write-up:
`reports/decode_coalescing_audit.md`.

- **The "ACO IR query crashes this RADV build" blocker is FALSE — deleted.**
  `reports/compiler_and_occupancy_audit.md` and the main report both record that
  disassembly is unobtainable here. It is obtainable: build the pipeline with
  `VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR` and run the
  three-pass `vkGetPipelineExecutableInternalRepresentationsKHR` query
  (count -> sizes -> data). RADV returns NIR + ACO IR + **Assembly**,
  unprivileged, no `RADV_DEBUG`, no wedge. Tool:
  `logs/bc250-sustained-20260918/shader_isa.c`; output archived under
  `logs/bc250-sustained-20260918/isa/`. **Future kernel work should read the
  assembly instead of inferring from timings.**
- **Loads are already optimal width — not a lever.** ACO merges the two
  unaligned `load_u32()` calls into one `buffer_load_dwordx3` plus two
  `v_alignbit_b32`. Three dwords is the minimum covering 8 B at an arbitrary
  byte offset; `uvec2`/`uvec4` are impossible because the 18-byte Q2_0 block
  stride makes the data 2-byte aligned by construction.
- **Transactions are already coalesced — not a lever.** One load instruction
  from a wave64 spans a contiguous `[row_base, row_base+580)` region. No
  striding, no split transactions. (L0 over-request is 1.78x from overlapping
  12-byte windows, but those hit the same cache lines, so DRAM traffic is 1.0x.)
- **Clamp-instead-of-branch (the one real gap) — REJECTED.** The per-row
  `if (row >= pc.out_dim) continue;` is wave-uniform and dead at runtime (every
  Q2_0 `out_dim` is a multiple of 64, `ROWS=4`), yet it splits the four rows
  into separate basic blocks: the weight descriptor is reloaded 4x per iteration
  and each `buffer_load_dwordx3` is waited on 5 instructions after issue.
  Replacing it with `min(first_row + r, pc.out_dim - 1u)` merged the blocks
  (21 -> 13), cut descriptor reloads (9 -> 4) and cost registers
  (28 -> 36 VGPRs, 36 -> 28 subgroups/SIMD). Bit exact: frontier-512 logits
  `max_abs_diff = 0` over 248,320 entries, greedy generation byte-identical
  (5,901 B). 10 soak-gated interleaved pairs: decode **median paired +0.34 %**
  (6/10 positive, A 32.66 MAD 0.16 -> B 32.70 MAD 0.07), aggregate-of-medians
  +0.15 %, pp512 -0.90 %. Gate >= +1.5 % -> **FAIL**. Patch
  `logs/bc250-sustained-20260918/decode-clamp-rejected.patch`, raw
  `runs/ab-decode-clamp.csv`.
- **Two premises fall out of that number.** (a) The 28-VGPR ceiling is not a
  cliff: occupancy fell 36 -> 28 subgroups/SIMD with *no* decode regression, so
  "dropping occupancy degrades throughput" is not supported on this kernel.
  (b) The kernel is **not latency-bound** — removing the stall, the branches and
  five descriptor reloads bought +0.34 %, inside the run's own spread.
- **Verdict: the inner loop is memory-bound and near-optimal. Freeze it.**
  Do not re-litigate load vectorization, transaction coalescing, software
  prefetch/double-buffering, or the row guard on this kernel.
  `reconsider_if`: a hardware DRAM byte counter becomes available and shows the
  kernel well under bus saturation, or the weight *format* changes (below).
- **Where the headroom actually is — PTQ1_0 (not tried, ranked next).** ggml type
  id **143**: `QK=128`, block = 24 B base-3 trits (5 trits/byte) + 2 B qh + 2 B
  f16 scale = 28 B per 128 weights = **1.75 bpw**, against the current
  Q2_0-g64's 18 B/64 = **2.25 bpw**. That is **~22 % less weight traffic per
  token** on a kernel this audit just proved is limited by that traffic.
  `prism-ml/Ternary-Bonsai-2-27B-gguf` publishes
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB vs the local 7.63 GB, matching the
  bpw ratio), and `prism-llama.cpp` already has Vulkan shaders
  (`ptq1_0.glsl`, `dequant_ptq1_0.comp`) to port. Blockers: p36 has no type-143
  loader entry and no trit unpack — the current 256-entry `q2tab` LUT does not
  carry over, because one byte holds 5 trits, not 4 codes. Note this changes the
  quantization, so it is a new baseline needing a quality eval, **not** a
  bit-exact parity check.

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
  `p36_vulkan.c:1693-1698` already selects
  HOST_VISIBLE|HOST_COHERENT|HOST_CACHED for every non-weight buffer, exactly
  as its own comment intends. Nothing to port, no gain available here.
- **shader_core_count 24->40 split-K lever — NOT APPLICABLE.** This engine has
  no shader-core-count query at all, so there is no scheduling input to
  correct. Their effect was correcting a driver-reported split-K sizing input.
- **Wave32 for the non-mmq paths — RUN, MEASURED NEGATIVE (2026-09-17).** The
  earlier note in this entry was wrong on one point and it cost a re-check: it
  reasoned from *llama.cpp's* `warp size: 32` that the driver default is already
  32-wide, so the peer's `RADV_PERFTEST=cswave32` had "no premise". That is
  llama.cpp's backend, not this engine — p36 prints `subgroup 64` on the same
  device (`p36_vulkan.c:1569` forces only `delta_net_cols`, `rope_qwen`,
  `rope_qwen_mrope`, `quantize_q8_0` and the `dense_*_mmq` family). So the
  premise WAS live here and the lever earned one run. It got it: Swift 27B,
  7 interleaved reps, ctx 1024, `tests/bench_cswave32.sh` — prefill **171.14 ->
  169.88 = -0.74%** (MAD 0.600 / 1.540). Decode 19.69 -> 18.67 raw looks like
  -5%, but that is an artifact of A's own cold start (A reps 1-2 ran 23.66/21.79
  before the die settled at 66-68 C; steady-state A3-A7 median 18.58 vs B 18.67
  is a **tie**). Verdict: **no gain**, prefill negative in all seven pairs.
  Structural reason, from the shader census: the hot kernels are *already*
  wave32-forced (`dense_*_mmq`, `p36_vulkan.c:1575`), so the flag only moved
  kernels off the critical path. A per-kernel source widening of the
  one-cross-lane-op decode kernels is still possible but the measured ceiling is
  now ~0, so it is no longer ranked. Evidence: `evidence/raw/ab-cswave32.csv`.
- **`dense_iq3_s_mmq_r4` escaped the wave32 force predicate — CORRECTNESS BUG,
  found by the wave32 eligibility audit, FIXED (2026-09-17).** The force
  predicate is a *suffix* test:
  `strstr(k->path,"dense_") && strstr(k->path,"_mmq.spv")`.
  `dense_iq3_s_mmq_r4.spv` (built at `Makefile:198-202` with
  `-DP36_BM=4 -DP36_BK=64`) ends `_mmq_r4.spv`, so it never matched and the
  shader ran at wave64. That shader partitions its token tile by
  `gl_SubgroupID`: `tok = tok0 + warp_tok*32 + (tt>>1)*16 + thread_tok*2 +
  (tt&1)`, `thread_tok = lane>>2`, and the store covers a 128-token tile. At
  wave32 (4 subgroups) the formula reaches **127**; at wave64 (2 subgroups) it
  peaks at **79**, so **tokens 80-127 of every tile were written by nobody**.
  Reachable from the `small_rows` selection site (`p36_vulkan.c:8160-8170`:
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

The peer wins begged a fair question: is the p36 work worth anything next to
upstream? Measured it. Same box, same GGUF
(`/home/server/p36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf`), same ctx 1024,
`-ngl 99`, Vulkan, `llama-bench -p 1024 -n 16 -r 3`.

| engine | prefill t/s | decode t/s |
|---|---|---|
| upstream llama.cpp `0cae430` (built Sep 7) | 107.81 (+/-4.84) | 21.47 (+/-0.61) |
| upstream llama.cpp `972d231` (pulled + rebuilt) | 105.43 (+/-4.46) | 22.13 (+/-0.65) |
| **this repo (p36)** | **171.09** | **23.19** |

- **p36 is +58.7% prefill / +8.0% decode over stock upstream.** Not useless.
  The decode margin understates it: llama-bench `tg` is greedy, while p36's
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
  packed-native restructuring is the only lever.** Both llama.cpp and p36 run
  under that handicap, which is what makes p36's 58.7% an engine result rather
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

| model | p36 prefill / decode | llama.cpp `972d231` | delta |
|---|---|---|---|
| Qwen3.6-35B-A3B IQ2XXS (MoE guard) | 909.49 / 87.37 | 615.08 / 78.72 | **+47.9% / +11.0%** |
| RavenX-35B-P36-IQ2XXS | 717.36 / 89.47 | 545.46 / 77.96 | **+31.5% / +14.8%** |
| Qwen3.8-35B-A3B-IQ2_M | refuses to load | 529.90 / **91.53** | n/a |

- **`Qwen3.8-35B-A3B-IQ2_M` is a p36 loader limitation, not a bad file.** Upstream
  loads it (35.5B params, 2.7 bpw, 99/99 layers on GPU) and posts the fastest MoE
  decode measured on this box — 91.53 +/- 2.11, above p36's best (RavenX 89.47).
  p36 stops it at `p36: expected qwen35moe.block_count=40, got 41` (3/3 reps,
  exit 1). This is the highest-value open item in the repo and it is a loader
  condition, not a kernel.

  **Root cause located (2026-09-17, do not re-derive):** the guard is
  `p36.c:5377-5381` in `config_validate_model`, and it *already contains the
  escape hatch this file needs* — but it is gated on `P36_MODEL_DENSE`:

  ```c
  if (block_count != P36_N_LAYER &&
      !(P36_MODEL_DENSE && nextn_layers == 1 && block_count == P36_N_LAYER + 1)) {
  ```

  Read from the GGUF itself (`gguf-py`): IQ2_M is `block_count=41`,
  `nextn_predict_layers=1`, blk 0..40, 753 tensors, and **`blk.40.*` is a
  complete MoE block** (attn_q/k/v/output + norms, ffn_gate/up/down_exps,
  ffn_gate_inp, shared-expert tensors). The guard has 40 blocks / 733 tensors and
  no `nextn` key; RavenX likewise. So IQ2_M is a 40-block model **plus one
  NextN/MTP block** — structurally the shape the *dense* path already accepts.
  A second dense-gated check at `p36.c:5419-5420` allows
  `P36_TENSOR_COUNT + 15` (dense NextN = +15 tensors; MoE NextN = +20 by
  arithmetic on 753 vs 733). The engine has a non-dense MTP binder
  (`mtp_weights_bind`, `p36.c:4454`) but the embedded-MTP *skip* is dense-only:
  `p36_tensor_is_disabled_embedded_mtp` returns false when `!P36_MODEL_DENSE`
  (`p36.c:5177`).

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

- **Removing the mid-dispatch query-pool drain at `p36_vulkan.c:2981-2983`** —
  the thing that produced the "94 us/dispatch" `dense_q4k_decode` anomaly — is
  **INERT in production**. The block was gated on `p36_vk.prof_kernel`, and that
  flag is only set when `P36_VK_PROF_KERNEL` is present in the environment
  (`p36_vulkan.c:3465-3474`). Interleaved 7-rep A/B on Swift at ctx 1024:
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

Commit `f9d537d` on `experiment/w7-moe-nextn` (off `56126ca`), `p36.c` only,
+66/−22. All three dense-gated conditions identified above were extended to the
MoE shape:

- `config_validate_model`: `embedded_nextn_block = nextn_layers == 1 &&
  block_count == P36_N_LAYER + 1`, now accepted for any model kind; the
  tensor-count gate is `+15` dense / **`+20` MoE** (753 − 733).
- `p36_tensor_is_disabled_embedded_mtp`: `!P36_MODEL_DENSE` dropped, so the
  `blk.<P36_N_LAYER>.` prefix skip applies to MoE too. `e->mtp_ready` still
  short-circuits, so sidecar-MTP runs are unaffected; a 40-block file has no
  `blk.40`, so the guard is unchanged by construction.
- `weights_validate_layout(w, mixed_moe_quants)`: new `tensor_expect_moe_matrix()`
  helper. Under the predicate `!P36_MODEL_DENSE && n_tensors == P36_TENSOR_COUNT + 20`
  (exactly the 753-tensor IQ2_M shape) the trunk/non-expert matrix types are
  accepted with dims still validated (`tensor_expect_cpu_matrix`) instead of the
  Q4_K/Q5_K/Q6_K/Q8_0 whitelist. **The routed experts keep their original strict
  `tensor_expect_routed_*` validation.** This is a type-whitelist widening on the
  trunk, not a check removal, and the predicate cannot reach the guard or any
  dense file. Describing it as "a validation relaxation" overstates it; describing
  it as "loads" alone understates nothing but proves nothing either.

Measured:

- **Loads and runs:** `116.51 / 29.29` tps (profiled). The 3/3 refusal is gone.
- **Quality, specifically targeting the relaxed check:** `p36_test --gpu-cpu-parity`
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
  (main `p36-bench`, 19:25): IQ2_M `117.98 / 32.36` exit 0, guard `715.16 / 88.63`
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
IQ2_M under p36 is 4.5×/3.1× slower than upstream on the identical file
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
  decode on this box" thesis **does not transfer to p36** until that dispatch
  exists. Ranking the loader fix as a *speed* item was this campaign's error.


## IQ2_M experts land on the fused IQ2_S path (2026-09-20, accepted)

Closes the 2026-09-17 dispatch item above: the per-quant prefill GEMM family now
has IQ2_S members, so `Qwen3.8-35B-A3B-IQ2_M` stops falling into `moe_matvec`.

- **What changed.** `moe_gate_up.comp` (q8_K activations), `moe_gate_up_gemm.comp`
  and `moe_down_gemm.comp` gain an IQ2_S decode, built from the *same* sources with
  `-DP36_MOE_IQ2S` into `moe_gate_up_iq2s.spv` / `moe_gate_up_gemm_iq2s.spv` /
  `moe_down_gemm_iq2s.spv`. Host side: `p36_vk_moe_gate_up_iq2_swiglu` accepts
  IQ2_S pairs at 82 bytes/block, and `p36_gpu_moe_ffn_f32_tensor` accepts IQ2_S
  gate/up/down with 82-byte `gu_stride`/`down_stride` plus a sixth `tables`
  binding for the down GEMM. The IQ2_XXS/Q2_K builds are **byte-identical** to the
  pre-change binaries (`b46fa81a…` gate/up GEMM, `93a38508…` down GEMM,
  `601d56a9…` q8 gate/up), so the guard is untouched *by construction*, not by
  measurement.
- **Measured (7-rep interleaved, soak-gated entry ≤55 °C, actual 52-56 °C, sclk
  500 MHz, ctx 512, gen 128, `P36_VK_MOE_GEMM=0 P36_VK_MOE_GATE_UP=0` as arm A):**
  prefill **108.21 → 241.20 t/s = 2.23×** (paired per-pair 2.188-2.254, MAD
  0.49 → 1.40), decode **29.52 → 34.81 t/s = 1.18×** (MAD 0.87 → 0.07).
- **Profile at ctx 512:** `moe_matvec` 6355 ms (arm A) → `moe_iq2s_gate_up_gemm`
  637 ms + `moe_iq2s_down_gemm` 253 ms (arm B). Decode still runs
  `moe_iq2s_gate_up` 276 ms + a 218 ms matvec for the down.
- **Parity.** Against the matvec arm at a 512-token frontier: top-1 identical,
  top-64 64/64, max_abs 0.82. Calibration for "is that small enough": the *shipped*
  IQ2_XXS+Q2_K GEMM, toggled the same way on the guard, is 64/64 with max_abs 0.42 —
  same class, and IQ2_S carries 25% more bits per weight.
- **The calibration caught a real bug, which is why it was run.** The first port
  folded the 32-element group's low nibble scale into the wrong kk half for rows
  whose LDS swizzle has bit 2 set — 50% of rows, silently. It read as top-64 59/64 /
  max_abs 1.89, not as garbage; the fix is `h == (swz_ry >> 2u)` in the fold.
  Compare `iq2s-gemm-parity-prefix.txt` against `iq2s-gemm-parity.txt`.
- **Do not trust `--vulkan-fusion-parity` as the gate here.** Its env list toggles
  `P36_VK_MOE_GEMM` but **not** `P36_VK_MOE_GATE_UP`, so it can pass while comparing
  the new kernel against itself; with a 6-token prompt `gemm` is off anyway. The
  gate used was the frontier logits dump with `P36_VK_MOE_GEMM=0
  P36_VK_MOE_GATE_UP=0`.
- Raw: `evidence/raw/iq2s-gemm-ab-iq2m.csv`, `iq2s-gemm-ab.sh`,
  `iq2s-gemm-matvec-ref.txt`, `iq2s-gemm-new-prof.txt`, `iq2s-gemm-parity.txt`,
  `iq2s-gemm-parity-prefix.txt`, `iq2s-gemm-guard-calibration.txt`,
  `iq2s-gemm-xxs-identical.txt`.
- **Accepted because** it is a 2.2× prefill / 1.18× decode gain on a file the engine
  could load but not schedule, with zero risk to the guard. **Reconsider_if:** the
  decode-side f32 kernels (`moe_gate_up_decode`, `moe_down_q2k_sum_decode`) are
  still IQ2_XXS-only — porting them should beat the current 1.18× decode, since the
  guard's decode runs exactly those. Upstream's 91.5 t/s decode stays out of reach
  while 10.06 GB of IQ2_S experts stream per token.

## IQ2_M decode takes the f32 identity pair (2026-09-20, accepted)

Continues the entry above. With IQ2_S in the prefill GEMM pair, IQ2_M's decode still
ran the Q8_K-activation kernels plus a generic `moe_matvec` down; the f32 identity
pair the guard's own decode uses (`moe_gate_up_decode` + `moe_down_q2k_sum_decode`)
was IQ2_XXS/Q2_K-only, so the host guard dropped IQ2_S back to the q8 route.

- **What changed.** Both shaders gain an IQ2_S decode built from the same sources
  with `-DP36_MOE_IQ2S` (`moe_gate_up_decode_iq2s.spv`,
  `moe_down_q2k_sum_decode_iq2s.spv`; the down one gains a 7th `tables` binding).
  The host predicate is now `iq2s && !(gemm || (identity && down_sum_decode))` — the
  2..31-token small-batch kernels (`moe_gate_up_f32b`, `moe_down_q2k_f32b`) are still
  IQ2_XXS/Q2_K-only and must not see 82-byte blocks.
- **Measured (7-rep interleaved, soak-gated entry ≤55 °C, ctx 512, gen 128; arm A =
  `P36_VK_MOE_DOWN_SUM_DECODE=0`, i.e. the previous default posture on the same
  binary):** decode **34.97 → 47.14 t/s = 1.34×** (paired 1.273-1.356, MAD 0.28 →
  0.16). Prefill is the null control and it is flat: 241.56 → 241.27 (**1.002**),
  which is what proves the two arms differ only on the decode path.
- **Cumulative against the all-matvec arm:** prefill 108.21 → 241.27 (**2.23×**),
  decode 29.52 → 47.14 (**1.60×**).
- **Parity, frontier 513** (that forward pass is a 1-token identity chunk, i.e.
  exactly the two new kernels, on top of a 512-token GEMM prefill): vs the previous
  default posture top-1 identical, top-64 62/64, max_abs 0.907; vs the pure-matvec
  arm 60/64, max_abs 1.245. The accepted GEMM pair alone measured 64/64 / 0.82, so
  the decode pair adds a same-class delta, not a new one.
- **The bug worth remembering.** The first version applied one grid magnitude to
  both mid elements of each `vec2`: in IQ2_S the *scale* is per 8-group but the
  *magnitude and sign* are per element. It read as top-64 37/64 with a top-1 flip
  and max_abs 7.19 — a hard failure, but only because the decode path was compared
  at a frontier that actually exercises it. Prefill-only frontiers would have
  shipped it.
- **Guard:** the IQ2_XXS/Q2_K builds of both touched shaders are byte-identical to
  the pre-change sources (`946097d8…` gate/up decode, `3f316c16…` down sum-decode).
- Raw: `evidence/raw/iq2s-decode-ab-iq2m.csv`, `iq2s-decode-ab.sh`,
  `iq2s-decode-parity.txt`, `iq2s-decode-parity-prefix.txt`,
  `iq2s-decode-xxs-identical.txt`, `iq2s-decode-diffprof.txt`.
- **Accepted because** the pair the guard already trusts now covers IQ2_S too, at
  +34% decode with the guard byte-identical. **Reconsider_if:** the small-batch
  (2..31 token) f32 kernels are still unported, so a batched server workload on
  IQ2_M still falls to the q8 route; and the per-token decode profile is now
  dominated by `dense_kquant` (9.89 ms/tok = 36%) and `moe_tiles` (1.71 ms/tok =
  6% of dispatch overhead) — the experts are no longer the leading decode cost.

## W2 — IQ3_XXS ISA + ALU diagnostic ablation — SETTLED, NOT PURELY DRAM-BOUND (2026-09-20)

Branch `trackB-ptq1_0`, model `Swift-Qwen3.8-27B-IQ3_XXS.gguf`. Full write-up:
`karpathy/evidence/raw/iq3xxs-isa/W2-VERDICT.md`.

- **Diagnostic question settled:** "234 GB/s logical = DRAM-bound" is NOT a
  complete description. `dense_iq3_xxs_decode_r4` was tested under
  `P36_IQ3XXS_ABLATE` (keep loads, drop decode ALU): baseline production decode
  26.578 ms/token (282.2 GB/s, 62.1% of roofline) vs ablated 20.081 ms/token
  (373.5 GB/s, 82.2% of the 454.4 GB/s hardware roofline).
- **Exposed ALU cost is 6.497 ms/token (24.44% of kernel execution time).** The
  kernel is partially ALU-taxed, primarily by the 4× DPP cross-lane reduction
  chains (`quad_perm`, `row_half_mirror`, `row_mirror`, `v_permlanex16_b32`,
  `v_readlane_b32`).
- **Occupancy verified via `mmq_info` (`VK_KHR_pipeline_executable_properties`):**
  `dense_iq3_xxs_decode_r4` compiles to 64 VGPRs, 108 SGPRs, 0 spills, 16
  subgroups/SIMD, 1024 B LDS (vs `dense_iq3_xxs_mmq` at 96 VGPRs, 10
  subgroups/SIMD, 12288 B LDS).
- **Ceiling:** the hard DRAM roofline floor on this weight format is 20.08 ms/token
  (373.5 GB/s). ALU optimizations cannot exceed this floor.
- Raw evidence: `karpathy/evidence/raw/iq3xxs-isa/` (`base_gen*.txt`,
  `ablate_gen*.txt`, `*.stats.txt`, `*.isa.txt`, `W2-VERDICT.md`).

## W3 — delta_net_cols prefill on BC-250 — REJECTED, FREEZE; WAVE32 BUG FIXED (2026-09-20)

Branch `trackB-ptq1_0`, model `Swift-Qwen3.8-27B-IQ3_XXS.gguf`. Full write-up:
`karpathy/evidence/raw/W3-VERDICT.md`.

- **Tested via `P36_VK_DELTA_COL_PREFILL=1`**: 7-rep interleaved A/B at ctx 1024,
  gen 16. Prefill **170.77 (MAD 0.500) → 166.54 (MAD 0.500) = -2.48%** (FAIL vs
  ≥+1.5% gate, negative in all 7 pairs), decode **18.06 → 17.93 = -0.72%**.
  Ctx 512 diagnostic corroborates: **184.32 → 171.04 = -7.20%**.
- **Latent correctness bug found and fixed:** `delta_net_cols` hard-assumes
  wave32 (`gl_SubgroupID` 0..3 for 4 columns, `gl_SubgroupInvocationID` 0..31 for
  rows), but `p36_vk_force_wave32` previously matched `"delta_net_cols.spv"`.
  When Swift runs f16 recurrent state, the dispatched pipeline is
  `delta_net_cols_f16.spv`, which missed the pattern and ran at BC-250's native
  wave64 (only 2 subgroups), skipping half the state columns and writing OOB up
  to row 159 (row size 128), producing catastrophic parity failure (`max_abs`
  19.12, top-1 flip 13 vs 29, top-64 overlap 3/64). Widening the pattern in
  `p36_vulkan.c` to `"delta_net_cols"` restored parity (`max_abs` 0.767, top-1
  identical 13/13, top-64 61/64).
- **Verdict: REJECTED & FROZEN.** `delta_net_cols` is slower on BC-250 than the
  native wave64 register recurrence (`delta_net_decode_reg_f16`). The default
  guard `!p36_vk.bc250` at `p36_vulkan.c:1225` is verified correct and remains
  frozen. The wave32 force fix is committed as a latent correctness fix.
- Raw evidence: `karpathy/evidence/raw/ab-w3-deltacol.csv`,
  `ab-w3-deltacol-summary.txt`, `ab-w3-deltacol-parity.txt`,
  `ab-w3-deltacol-w32-parity.txt`, `ab-w3-ctx512-diagnostic.txt`.

## W4 — Wave32 clean-scan shaders A/B — REJECTED, NEUTRAL/NO WIN (2026-09-20)

Branch `trackB-ptq1_0`, models `Qwen3.8-35B-A3B-IQ2_M.gguf` and Guard MoE.
Full write-up: `karpathy/evidence/raw/W4-VERDICT.md`.

- **Tested via `P36_VK_WAVE32_CLEAN=1`**: forcing wave32 across the clean-scan
  shader set (`karpathy/evidence/wave32-eligibility.md` §3b: `moe_matvec.spv`,
  `moe_matvec_fast.spv`, `matmul_q8_0_decode*.spv`, `add_rms_norm.spv`,
  `rms_norm*.spv`, `kv_store_quant.spv`).
- **Parity is bit-exact**: `max_abs_diff = 0` across all 248,320 logits on both
  IQ2_M and the Guard model.
- **7-rep interleaved A/B on IQ2_M (ctx 512, gen 128)**: prefill **242.43
  (MAD 0.500) → 240.00 (MAD 1.270) = -1.00%** (inside the 3.4% MoE prefill noise
  floor; fails the ≥+3.4% gate); decode **46.62 → 46.62 = +0.00%** (exact tie).
  Single-run diagnostic on Guard MoE: **669.86 → 640.90 (-4.32%)**, showing no
  positive signal.
- **Verdict: REJECTED & CLOSED.** The unrun test from the wave32 audit is now
  measured: native wave64 remains optimal for these non-mmq shaders on GFX1013.
  `reconsider_if`: A future compiler or hardware revision where wave32 enables
  double active waves without doubling scheduling overhead.
- Raw evidence: `karpathy/evidence/raw/ab-w4-wave32-iq2m.csv`,
  `ab-w4-wave32-iq2m-summary.txt`, `ab-w4-wave32-parity.txt`,
  `ab-w4-wave32-guard.txt`.

## W5 — RMSNorm → q8_K producer fusion — REJECTED, GATE NOT MET (2026-09-21)

Branch `trackB-ptq1_0`, model `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense model, 5120 embedding dimension).
Full write-up: `karpathy/evidence/raw/W5-VERDICT.md`.

- **Hypothesis**: Fusing activation quantization (`quantize_q8_k`) directly into residual add + RMSNorm (`add_rms_norm`) removes an activation write/read round-trip and improves prefill/decode throughput.
- **Parity is bit-exact**: `max_abs_diff = 0` across all 248,320 vocabulary logits at frontier 513 on Swift-27B (`top-1` identical 5316 vs 5316, `top-64` overlap 64/64).
- **7-rep interleaved A/B on Swift 27B (ctx 1024, gen 16, chunk 256)**:
  - Arm A (stock baseline, separate dispatches): prefill **170.38 tok/s** (MAD 0.610), decode **18.10 tok/s** (MAD 0.050).
  - Arm B (candidate, fused `add_rms_norm_q8_k` kernel): prefill **169.83 tok/s** (MAD 0.140), decode **17.95 tok/s** (MAD 0.070).
  - Delta: prefill **-0.32%** (fails the dense prefill ≥+0.70% gate), decode **-0.83%** (neutral; well within the documented 9.4% decode noise floor).
- **Architectural Reason**:
  - `add_rms_norm` operates row-wide with 1024 threads (1 workgroup per token row).
  - `quantize_q8_k` operates on 256-element blocks with 64 threads (20 workgroups per token row, distributed as 5,120 independent workgroups across 40 CUs with 1,536 B LDS and 40 subgroups/SIMD).
  - Fusing them inside a 1024-thread workgroup serializes the 20 blocks across 2 rounds with 10 workgroup-wide barriers per round (20 barriers total), requires 25,600 B LDS (vs 8,192 B for stock norm), and drops subgroup occupancy from 40 to 32.
  - On BC-250, the 20 KB activation sits in L2 cache between separate dispatches; avoiding the L2 round-trip is outweighed by the 20-barrier serialization cost and occupancy drop.
- **Verdict: REJECTED & FROZEN.** Default remains separate dispatches (`P36_VK_FUSED_RMS_Q8=0`, gated on `env && env[0] == '1'`). Scaffolding retained behind opt-in flag.
- Raw evidence: `karpathy/evidence/raw/ab-w5-fusedrms.csv`, `ab-w5-fusedrms-summary.txt`, `ab-w5-fusedrms-parity.txt`, `ab-w5-isa-properties.txt`.




## W6 — merged Vulkan dense gate_up (one dispatch for the gate/up pair) — ACCEPTED (2026-09-21)

Branch `trackB-ptq1_0`, base `6b74da5`, model `Swift-Qwen3.8-27B-IQ3_XXS.gguf`
(dense, 5120 embed dim). Full write-up: `karpathy/evidence/raw/W6-VERDICT.md`.

- **Hypothesis**: the dense FFN prefill spends its per-workgroup activation
  staging (`b16`) and LUT setup twice — once for `gate` (5120x17408) and again
  for `up` (same shape, same activation). One workgroup serving both matrices
  amortizes that over 64 weight rows instead of 32.
- **Probe before building**: temporarily dropping the b16 staging loads from
  `dense_iq3_xxs_mmq.comp` moved mmq 1893.6 -> 1625.1 ms (-14.2%) and prefill
  178.56 -> 197.42 t/s (+10.6%) — the ceiling, and the reason to implement
  rather than reject. Probe reverted; `dense_iq3_xxs_mmq.spv` md5 unchanged.
- **Implemented**: new `vulkan/dense_iq3_xxs_mmq_pair.comp` (bindings: gate w,
  up w, gate out, up out, tables, b16) + `p36_gpu_matmul_iq3_xxs_pair_mmq_tensor()`
  in `p36_vulkan.c`, called from `p36_forward_ffn_vulkan_model` (`p36.c`).
  BM/BN/BK, per-matrix accumulation order and store mapping are the unfused
  kernel's, so each element is bit-identical. `sum[32]` costs 16 VGPRs, +2176 B LDS.
- **Coverage 94/110 layer-instances (85.5%)**: the 16 that stay on the single path
  are genuinely mixed-type (IQ3_XXS gate with a non-XXS up) or IQ3_S/IQ4_XS, which
  one decoder cannot serve.
- **Parity is bit-exact**: frontier 513 on Swift-27B, pair off vs on, single
  binary — `max_abs_diff = 0` over all 248,320 logits, argmax 5316 (19.820942)
  both arms, top-64 overlap 64/64.
- **7-rep interleaved A/B on Swift 27B (ctx 1024, gen 16, chunk 256)**, one
  binary, wrappers differing only in `P36_VK_DENSE_IQ3_PAIR`:
  - Arm A (pair off): prefill **169.25 tok/s** (MAD 0.140), decode **18.12 tok/s** (MAD 0.100).
  - Arm B (pair on): prefill **173.01 tok/s** (MAD 0.440), decode **18.22 tok/s** (MAD 0.080).
  - Delta: prefill **+2.22%** (PASSES the dense ≥ +0.70% gate, 3.2x it), decode
    **+0.55%** (inside the 9.4% decode spread — neutral). Arms do not overlap:
    max A 170.48 < min B 171.90.
- **Attribution** (ctx 512, gen 0, `P36_VK_PROF_*`): gate/up 937.6 -> 858.9 ms
  (**-8.4%**), mmq 1892.9 -> 1089.5 + 728.0 = 1817.5 ms (-4.0%), whole-run kernel
  total 2792.7 -> 2670.9 ms (-4.36%). The spv dispatch drop 510 -> 322 is exactly
  the 188 removed singles (94 pairs x 2).
- **Verdict: ACCEPTED, default ON** (`P36_VK_DENSE_IQ3_PAIR=0` disables).
  The earlier "staging/barrier/LDS is CLOSED for this kernel" note (item 3b,
  codex double-buffered restructure, -0.11%) still holds *within one projection*;
  what pays is sharing the stage **across the gate/up pair**, not restaging it
  faster. `reconsider_if`: the 16 mixed-type instances become same-type
  (measured +2.22% at 85.5% coverage implies ~+2.6% at full), or a future part
  where the second A tile costs occupancy.
- Raw evidence: `karpathy/evidence/raw/ab-w6-pair.csv`,
  `ab-w6-pair-summary.txt`, `ab-w6-pair-parity.txt`, `ab-w6-pair-profile.txt`,
  `W6-VERDICT.md`.


## W7 — Stream-K geometry audit — REJECTED (NOT A LEVER) (2026-09-21)

Plan: `karpathy/PLAN-implementation-2026-09-20.md` §W7.
Hardware: AMD BC-250 (40 CUs, RADV GFX1013). Model: `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (prefill chunk 256).
Full audit report: `karpathy/evidence/raw/W7-VERDICT.md`, audit tool `karpathy/evidence/raw/w7_streamk_audit.py`.

- **Hypothesis**: Stream-K dynamically partitions workgroup grids across compute units to eliminate wave-quantization tail waste in GEMM/MMQ kernels.
- **Audit Steps**: For every MMQ shape executed in Swift-27B prefill, computed `grid / (40 CU * resident WGs/CU)` across 40 CUs for both realistic occupancy models ($R=2$, 80 GPU slots; $R=4$, 160 GPU slots). Measured actual GPU time per shape via `P36_VK_PROF_SHAPE`.
- **Findings**:
  1. Down-projection (`17408x5120`, 575 ms, 23.4% MMQ share): Grid is exactly 320 WGs. $320/80 = 4.00$ waves ($R=2$), $320/160 = 2.00$ waves ($R=4$). **Tail loss is exactly 0.00%**.
  2. QKV-projection (`5120x10240`, 176 ms, 7.1% MMQ share): Grid is exactly 640 WGs. $640/80 = 8.00$ waves ($R=2$), $640/160 = 4.00$ waves ($R=4$). **Tail loss is exactly 0.00%**.
  3. Gate/Up-projection (`5120x17408`, 859 ms, 34.9% MMQ share): Grid is 1088 WGs. 13.60 waves ($R=2$, 0.40 wave tail = 2.86%), 6.80 waves ($R=4$, 0.20 wave tail = 2.86%). **Tail loss is 2.86%**.
  4. IQ4_XS de-homogenization: Deconstructing the 88 dispatches into their 5 actual tensor geometries (ffn_up, ffn_down, attn_qkv, attn_gate, attn_q) reveals that 84.1% of IQ4_XS work has <= 2.86% tail (0.0% on down and qkv), resulting in a true blended tail of 2.23% ($R=2$) and 4.39% ($R=4$), disproving the 11.0% artifact from averaging heterogeneous grids.
  5. Aggregate whole-model wave tail: Weighted across all shapes, partial wave tail accounts for **1.93%** of MMQ time (**1.78%** of prefill kernel time) under the measured hardware occupancy model ($R=2$, 80 slots), and **3.61%** of MMQ time (**3.32%** of prefill kernel time) under theoretical upper-bound concurrency ($R=4$, 160 slots).
  6. Timeline analysis (`timeline-analysis.md`) confirms 0.5% GPU idle across prefill dispatches; shapes execute at uniform per-MAC throughput, proving the bottleneck is per-workgroup LDS/barrier latency, not scheduling tail.
- **Decision**: Gate required tail > 3.0% of kernel time for primary compute paths. Hardware occupancy measures 8 subgroups/SIMD (pair, 128 VGPRs) to 10 subgroups/SIMD (stock, 96 VGPRs), bracketing active residency between $R=2$ (80 slots, tail 1.78% < 3.0%) and $R=4$ (160 slots, tail 3.32%). Even under the upper-bound $R=4$ model (3.32%), the primary compute paths (>80% of MMQ work) have tail <= 2.86% (0.0% on down/qkv). Stream-K introduces atomic K-reduction across CUs, partial-sum buffers, split-K barriers, and extra synchronization overhead for < 1.8% theoretical headroom.
- **Verdict**: **REJECTED (NOT A LEVER)**. Closed without code churn.
- `reconsider_if`: Batch size or token chunk changes to an irregular dimension that produces small grids (< 80 workgroups) with severe partial-wave cliffs, or targeting a GPU with a non-divisible CU count (e.g. 36 or 68 CUs).

## Dense K-quant fast path widened to MoE models — ACCEPTED (2026-09-21)

Branch `trackB-ptq1_0`, model `Qwen3.8-35B-A3B-IQ2_M.gguf`. Guard:
`Huihui-Qwen3.6-35B-A3B-Abliterated-P36-IQ2XXS.gguf`.

- **HANDOFF-iq2s-20260920.md's ranked item #1 was misdiagnosed and this
  supersedes it.** It attributed the `dense_kquant`/`matmul_kquant.spv`
  hotspot (48.2% of IQ2_M decode, 10.7 ms/tok at the time) to "non-expert
  IQ2_S tensors" and proposed writing a new IQ2_S decode shader. Reproducing
  the differential profile on current HEAD (`P36_VK_PROF_KERNEL=1`, gen 1 vs
  gen 65, ctx 512) still showed **9.453 ms/tok, 41 dispatches/tok** — the
  figure hadn't moved across W2-W7. Parsing the GGUF header directly
  (`gguf-tools`-style tensor listing) instead of trusting the label showed
  the 41 tensors are **40x `Q4_K` `blk.N.attn_qkv.weight` + 1x `Q5_K`
  `output.weight`** — real K-quants, not IQ2_S at all.
- **Root cause**: `p36_gpu_matmul_k_quant_q8_scaled_tensor()` (`p36_vulkan.c`)
  has tuned decode (`n_tok==1`) and mmq (`n_tok>1`) branches that dispatch
  `dense_kquant_decode`/`dense_q4k_decode`/`dense_q5k_decode`/
  `dense_kquant_mmq` — kernels that read only the one tensor's own
  `in_dim`/`out_dim`/`blocks`/`row_bytes`/`type`/`scale` from push constants,
  nothing model-wide. Both branches were gated on `p36_gpu_dense_model`, true
  only for the dense Swift-27B/UD-IQ3_S launches (`p36_gpu_set_dense_model`
  is called once per engine open with `P36_MODEL_DENSE`). Nobody widened it
  when a MoE model started shipping its own K-quant trunk tensors, so IQ2_M's
  Q4_K/Q5_K rows fell through to the generic AVX2-style 8-rows-x-8-lanes
  `matmul_kquant` fallback unconditionally, on both prefill and decode.
- **Fix**: `p36_vulkan.c`, two-line change — drop `p36_gpu_dense_model &&`
  from both branch conditions. Also widened `p36_gpu_set_dense_model()` to
  always call `p36_vk_prepare_dense_kernels()` (was `if (dense)` only), so
  MoE models get the same pipeline prewarm at model-open instead of paying a
  multi-second RADV pipeline-compile stall on the first live decode/prefill
  token — caught by an independent review (Claude2, since Codex was over its
  usage cap) before this landed; the review's other finding (a comment citing
  a not-yet-written ledger entry) was also fixed.
- **Parity**: guard frontier-513 logits byte-identical (`max_abs_diff=0`,
  top-1 identical) — expected, the guard has zero K-quant trunk tensors so
  this is a pure no-op-on-guard check. IQ2_M frontier-513: top-1 preserved in
  every configuration tested (combined change, mmq-path-only, decode-path-
  only, frontier-512 mmq-only-no-tail, and the candidate binary run twice for
  self-consistency — the last is `max_abs_diff=0`, fully deterministic).
  Combined max_abs=1.523, mean_abs=0.190, top-64 overlap 60/64. This is the
  same order of magnitude and *shape* as HANDOFF's own accepted W1 calibration
  (accepted-arm-vs-previous max_abs 0.907/top-64 62/64; cruder all-matvec-arm
  max_abs 1.245/top-64 60/64): differences spread near-uniformly across all
  248,320 logits (mean 0.19 on a ~29-wide logit range), consistent with
  `output.weight` directly producing every logit so any kernel-order
  floating-point reassociation shows up everywhere, not concentrated on a
  handful of tokens the way a real bug would look. `./p36_test
  --vulkan-kernels` (CPU-reference dot-product oracle, tolerance 2e-3) also
  passes and, for the first time, actually exercises this decode path — it
  was dead code in that test before this change since `p36_gpu_dense_model`
  was never set true there.
- **7-rep interleaved A/B** (`tests/bench_ab.sh`, ctx 512, gen 128, matching
  HANDOFF's own IQ2_M convention): prefill **244.35 (MAD 2.360) -> 507.87
  (MAD 9.130) = +107.85%**, decode **47.34 (MAD 0.410) -> 75.56 (MAD 0.130) =
  +59.61%**. Gate is MoE prefill >=3.4%; cleared by ~32x. Both arms show
  tight spread and no overlap.
- **Verdict: ACCEPTED, default ON** (same env vars as before,
  `P36_VK_DENSE_KQUANT_DECODE=0` / `P36_VK_DENSE_KQUANT_MMQ=0` still disable
  each branch individually if needed — nothing new to opt into).
  `reconsider_if`: a future MoE model whose non-expert K-quant tensor shapes
  don't hit the `out_dim % 4 == 0` "full" variant (would still take the
  bounds-checked generic `dense_kquant_decode`/`dense_kquant_mmq`, just
  without that extra edge), or new evidence that the fast kernel's numerics
  are wrong rather than merely reassociated (would need to show up as
  *localized*, not uniform, logit deltas).
- Raw evidence: `karpathy/evidence/raw/ab-kquant-moe-decode.csv`,
  `ab-kquant-moe-decode-summary.txt`, `kquant-moe-diffprof-gen{1,65}.txt`,
  `kquant-moe-parity-summary.txt`, `kquant-moe-unit-test.txt`.

## W8 — long-context KV dequant redundancy — go/no-go MEASURED POSITIVE,
## QT-widening mechanism REJECTED (VGPR ceiling), scratch-cache UNEXPLORED (2026-09-21)

Branch `trackB-ptq1_0`. Models: guard `Huihui-Qwen3.6-35B-A3B-Abliterated-
P36-IQ2XXS.gguf` (MoE), `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense).

- **Step 1, per the plan (`PLAN-implementation-2026-09-20.md` §W8): measure
  attention's prefill share at ctx 4096/8192 before touching any shader.**
  `P36_VK_PROF_KERNEL=1 ./p36-bench --vulkan -m <model> --prompt-file
  tests/long_context_story_prompt.txt --ctx-start N --ctx-max N
  --prefill-chunk 256 --gen-tokens 0`, single-frontier runs (no sweep, no
  decode), attention kernel's `pct` from the per-kernel GPU-time report:

  | model | shader | ctx 1024 (plan's own baseline) | ctx 4096 | ctx 8192 |
  |---|---|---|---|---|
  | guard (MoE) | `attn_prefill_qtile2.spv` | ~3.1% | 20.7% (1491.6/7188.8 ms) | 31.5% (5430.1/17264.9 ms) |
  | Swift-27B (dense) | `attn_prefill_qtile2_gqa6.spv` | — | 12.6% (3931.1/31210.2 ms) | 20.7% (15984.3/77044.6 ms) |

  Both climb sharply with context and clear the informal go/no-go (a partial
  win on this kernel would easily clear the formal ≥1.5%-prefill gate at
  these shares) — proceed to design a fix.
- **Root cause, read from `vulkan/attn_prefill_qtile2{,_gqa6}.comp`:** each
  workgroup already shares K/V dequant across its own `QT=2` query tokens and
  all `HQ` GQA heads for one `kvh` (dequant happens once per key, feeds all
  `ROWS=QT*HQ` rows via `sh_w`). The redundancy is *across* workgroups: at
  ctx 8192 there are `n_tok/QT` query-tile workgroups per `kvh`, and causal
  attention means nearly every one re-dequantizes the same early KV spans
  independently.
- **Mechanism 1 tried: widen `QT` (more query rows sharing each dequant).**
  Cheap probe before any real build (matching the W6 LUT-staging probe
  precedent): scratch copy of `attn_prefill_qtile2.comp` with only
  `QT 2u->4u` / `ROWS 8u->16u` changed (compiles, not correct — this is a
  register/LDS probe only, no repo files touched), built with the Makefile's
  own `glslc -O --target-env=vulkan1.1`, inspected with `./mmq_info`
  (`VK_KHR_pipeline_executable_properties`):
  - QT=2 (current): VGPRs=168, LDS=21504 B, **6 subgroups/SIMD**.
  - QT=4 (probe): VGPRs=**256** (RDNA's hard per-wave ceiling, zero
    headroom), LDS=43008 B, **2 subgroups/SIMD**.
  - **Verdict on this mechanism: REJECTED without a build or A/B.** Doubling
    `QT` at best halves per-key dequant work, but a 3x occupancy cut
    (6->2 subgroups/SIMD) is very likely to erase or reverse that — the same
    register/LDS trap this campaign hit in W4 and the "packed-native shader
    restructuring" items. The probe is the evidence; no throughput run was
    needed to reject it.
- **Mechanism 2, not attempted this session: a KV-dequant-once scratch
  cache** (materialize each `(kvh, span)`'s dequantized K/V once, shared
  across query-tile dispatches, instead of growing per-lane state — no VGPR
  risk). No cheap compile-time probe exists for this one (it's a bandwidth
  question, not a register-allocation one), and the plan's own note is
  skeptical of a related technique ("materializing fp16 KV to scratch adds
  write+read traffic on top of a compact quantized cache — the coopmat1
  result does not transfer"). Explicitly **left unexplored** rather than
  built speculatively.
- **Verdict: item closed for this session.** The underlying hotspot is real
  and confirmed growing (Step 1), the first mechanism tried is dead by direct
  measurement, and the second has a real unresolved bandwidth risk with no
  cheap way to bound it the way the VGPR probe bounded the first. Rather than
  sink a full build+A/B into an unmeasured design, stopping here.
- `reconsider_if`: (a) someone designs and measures the KV-scratch-cache
  mechanism (build it, A/B at ctx>=4096, check parity — this is real,
  unstarted work, not a dead end); (b) per-lane state in the `QT=2` kernel
  can be shrunk enough (e.g. f16 accumulators instead of f32) that `QT=4`
  fits under the VGPR ceiling without the occupancy cliff; (c) a driver/HW
  change raises the VGPR-per-wave ceiling on this target.
- Raw evidence: `karpathy/evidence/raw/w8-attn-share-{guard,swift}-ctx{4096,8192}.txt`,
  `karpathy/evidence/raw/w8-vgpr-probe.txt`.


## IQ2_M's q8-route residue closed: IQ3_S down projection added to the fused f32 expert path — ACCEPTED (2026-09-21)

Branch `trackB-ptq1_0`, model `Qwen3.8-35B-A3B-IQ2_M.gguf`, base HEAD `e6c15ba`.
Closes `HANDOFF-iq2s-20260920.md` §5 item 3 for this file. Full write-up:
`karpathy/evidence/raw/iq3s-down-VERDICT.md`.

- **Root cause (measured with a diagnostic, not inferred).** The new
  `P36_VK_MOE_ROUTE_DEBUG=1` print in `p36.c` fires on every fused-f32-path
  miss and pins the residue exactly: layers **0, 1, 2 only**, on both prefill
  and decode, with `gate=22 up=22 down=21` (22 = `IQ2_S`, 21 = `IQ3_S`).
  `p36_gpu_moe_ffn_f32_tensor()` required all three expert tensors to share one
  quant, and those three layers ship an **IQ3_S down under IQ2_S gate/up**
  (GGUF tensor table: `ffn_down_exps` IQ2_S x37, IQ3_S x3 at il 0-2, Q8_0 x1 at
  il 40 = the MTP block). They fell to the q8_K route, whose
  `moe_iq2s_gate_up` + `moe_matvec` + `moe_tiles` cost 19.6% of profiled
  prefill GPU and 6.8% of decode per token.
- **Fix**: `-DP36_MOE_IQ3S` builds of `moe_down_q2k_sum_decode.comp` and
  `moe_down_gemm.comp` (`moe_down_q2k_sum_decode_iq3s.spv`,
  `moe_down_gemm_iq3s.spv`) + host wiring: `iq2s` split into `gu_iq2s` /
  `down_iq2s` / `down_iq3s`, `down_stride` 82 or 110 bytes, the down dispatch
  site chooses the variant. IQ3_S block `[d f16][64 qs][8 qh][32 signs][4 scales]`
  = 110 B, 9-bit grid index (8 qs bits + 1 qh bit per 4 weights) against the
  IQ3_S grid already in the shared IQ tables buffer at word 2592, scale
  `d*(2*nib+1)` — same convention as `dense_iq3_s_decode.comp`. All new code is
  inside `#ifdef P36_MOE_IQ3S`; IQ2_S/legacy builds stay byte-identical.
- **Parity** (frontier dumps vs the `e6c15ba` binary, 248,320 logits): IQ2_M
  frontier 512 (`max_abs` 0.762, mean 0.112, top-1 match, top64 62/64);
  frontier 513 (0.865, mean 0.138, top-1 match, top64 61/64); guard frontier
  513 **sha256-identical, `max_abs` 0.0** (the guard has no IQ3_S experts, so
  this is a pure no-op for it). Deltas are the same order and *shape* as W1's
  own accepted calibration (0.907/62-64) and smaller than it; spread across
  ~all logits (7.7%/15.0% of logits above 0.25, max ~2x p99), not localized.
- **7-rep interleaved A/B** (`tests/bench_ab.sh`, ctx 512, gen 128,
  `P36_AB_MIN_GAIN=3.4`): prefill **497.00 (MAD 15.150) -> 556.21 (MAD 12.400)
  = +11.91%**, decode **75.26 (MAD 0.180) -> 78.40 (MAD 0.100) = +4.17%**
  (decode separates rep-by-rep, no overlap). Gate is MoE prefill >= 3.4% ->
  PASS, ~3.5x. `./p36_test --vulkan-kernels` OK; `compat_gate.sh` PASS.
- **Verdict: ACCEPTED, default ON** (no new env var; type-driven inside the
  existing fused path, `P36_VK_MOE_F32B=0` still disables it).
  `reconsider_if`: a file mixes the other way (IQ3_S gate/up under an IQ2_S or
  IQ3_S down) — that needs the gate/up side ported and is still rejected into
  the q8 route (safe, unchanged); or an IQ3_S expert tensor whose
  `in_dim % 256 != 0`.
- Raw evidence: `karpathy/evidence/raw/ab-iq3s-down.csv`,
  `ab-iq3s-down-summary.txt`, `iq3s-down-parity.txt`,
  `iq3s-down-parity-stats.txt`, `iq2m-prof-postkquant-{decode,prefill}.txt`.

## IQ2_S `ssm_alpha`/`ssm_beta` pair-fusion — ACCEPTED, small win (2026-09-21)

Branch `trackB-ptq1_0`, model `Qwen3.8-35B-A3B-IQ2_M.gguf`. Full diagnosis:
`karpathy/evidence/raw/iq2m-reprofile-2026-09-21.md` (item 1d in
`CAMPAIGN.md`). First of the two-part plan from that diagnosis (pair-fusion
for the overhead-bound small tensors; a tuned decode/mmq kernel for the
ALU-bound big tensors — `attn_gate`/`ssm_out`/shared-expert — is separate,
unstarted work).

- **What it is**: `ssm_alpha`/`ssm_beta` (recurrent-layer projections,
  2048x32, ~20.5 KB each) are IQ2_S on this file and fall through the
  generic `dense_extra_decode` path individually — 2 dispatches, 2
  independent Q8_K quantizations of the same activation. New
  `vulkan/dense_extra_decode_iq2s_pair.comp` +
  `p36_gpu_matmul_iq2s_pair_scaled_tensor()` (`p36_vulkan.c`) fuse both into
  one dispatch sharing one Q8_K read, mirroring the existing
  `p36_gpu_matmul_q8_0_pair_scaled_tensor` pattern (same shape, IQ2_S
  instead of Q8_0). Wired into `p36_forward_recurrent_vulkan`'s existing
  `pair_projected` fallback chain (`p36.c`), gated on both tensors being
  exactly `IQ2_S` — inert by construction for any model where that's not
  the type (e.g. the guard, which is IQ2_XXS-based; not separately
  A/B'd for that reason). Default on, `P36_VK_IQ2S_PAIR=0` disables.
- **Parity: bit-exact.** Frontier-512 first-decode-token logits,
  `P36_VK_IQ2S_PAIR=1` vs `=0`, same binary: `max_abs_diff = 0.0` across all
  248,320 logits, argmax and top-64 identical. Expected — the dequant math
  is copied verbatim from `dense_extra_decode.comp`'s IQ2_S branch of
  `block_dot()`, just evaluated twice per Q8_K block instead of once per
  dispatch; same accumulation order per weight matrix.
- **7-rep interleaved A/B (ctx 512, gen 128) was inconclusive on its own**:
  prefill +0.66%, decode +1.11%, overlapping rep ranges. **21-rep re-run
  clarified it**: prefill 555.39→544.20 = -2.01% (noise — this fix is
  `n_tok==1`-gated, never touches prefill, so A and B run identical prefill
  code; -2.01% is well inside the campaign's documented 3.4% MoE prefill
  floor), decode **78.15→79.26 = +1.42%** (MAD 0.190/0.270, right at the
  formal ≥1.50% gate — 19 of 21 reps per arm fall in disjoint bands
  [77.75-78.45] vs [78.62-79.88], only the extreme tails touch).
  `bench_ab.sh`'s auto-verdict says `FAIL (prefill)`; that's the harness's
  generic prefill-gain semantics and doesn't apply to a decode-only fix —
  read the medians, not the label, same as prior sessions' notes on this
  harness.
- **Binaries built in isolated worktrees** (`/home/server/p36-wt/
  iq2s-ssm-pair{,-base}`) from a clean `b738610` checkout, not the shared
  main worktree — a second instance had uncommitted, unrelated changes
  (small-batch IQ2_S kernels, HANDOFF §5 item 2) mid-flight in the same
  working directory at the time; A/B'd only this change.
- **Honest framing**: this is a smaller, noisier result than 1b/1c/1d — the
  win is real (bit-exact, mechanistically sound, ~1.2 MB / 60-of-258
  dispatches is a small slice of the 11.9 ms/tok decode budget so a ~1%
  decode gain is the expected order of magnitude, not a surprise) but sits
  at the edge of the formal gate rather than clearing it by a wide margin.
  Accepted on the strength of the parity + mechanism + 21-rep separation,
  not a single clean A/B pass.
- `reconsider_if`: the bigger, unstarted half of the plan (tuned IQ2_S
  decode/mmq kernel for `attn_gate`/`ssm_out`/shared-expert, ~79% of the
  194.6 MB/tok this residue reads) is where the real remaining win is —
  revisit this pairing approach if that kernel's shape makes fusing
  `attn_gate` with something else attractive too.
- Raw evidence: `karpathy/evidence/raw/ab-iq2s-ssm-pair.csv` (21 reps),
  `ab-iq2s-ssm-pair-7rep.csv`, `ab-iq2s-ssm-pair-summary.txt`,
  `iq2s-ssm-pair-parity.txt`.

## IQ2_S tuned decode/mmq kernel (`attn_gate`/`ssm_out`/shared-expert) — ACCEPTED, real but small (2026-09-21)

Branch `trackB-ptq1_0`, model `Qwen3.8-35B-A3B-IQ2_M.gguf`. Second and last
piece of the 1d/1e two-part plan — the ALU-bound big-tensor half (~79% of
the 194.6 MB/tok the IQ2_S residue reads: `attn_gate`, `ssm_out`,
shared-expert gate/up/down), vs 1e's overhead-bound small-tensor half
(`ssm_alpha`/`ssm_beta`).

- **What it is.** `dense_extra_decode.comp` / `dense_extra_mmq.comp` are
  generic kernels that runtime-branch on `pc.type` to serve
  IQ2_XXS/IQ2_XS/IQ2_S/IQ1_S/IQ4_NL/Q2_0 from one shader (Q2_0/PQ2_0/PTQ1_0
  already get their own compile-time-specialized variant via
  `P36_Q2_0_ONLY`, precedent cited in the source comments: "-19..28% on
  dense_extra_decode_q2_0" from a LUT). Added the same treatment for IQ2_S:
  a `P36_IQ2S_ONLY` `#ifdef` in both files producing
  `dense_extra_decode_iq2s.spv` / `dense_extra_mmq_iq2s.spv`, math copied
  verbatim from the existing `pc.type == IQ2_S` arm (same block layout,
  same `signed_grid`/`tables` lookups, same accumulation order), just
  without the runtime type-branch chain. Wired into
  `p36_gpu_matmul_iq_quant_q8_scaled_tensor`'s decode/mmq kernel selection,
  gated on `weight_type == IQ2_S` and `P36_VK_IQ2S_EXTRA` (default on).
  Both new kernels added to `p36_vk_prepare_dense_kernels()`'s prewarm list
  (along with 1e's `dense_extra_decode_iq2s_pair`, missed there originally)
  so the RADV pipeline-compile cost lands at model-open, not the first live
  token — same class of fix 1b needed for its kernels.
- **A real bug caught and fixed before landing**: the first draft of the
  specialized mmq `main()` had `barrier()` moved inside the `BK/2`
  accumulation loop instead of after it (a copy-paste slip while manually
  reconstructing the function body) — not a silent-corruption bug since all
  threads in a workgroup still hit the same barrier count, but it would
  have added 16 barriers per slice instead of 1, quietly eating most of the
  kernel's intended win. Caught by re-reading the specialized function
  against the generic one side by side before compiling, not by a failed
  test. This is exactly the class of risk flagged going in (W3 latent-bug
  precedent) — found by review, not by luck.
- **Parity: bit-exact**, checked three ways on the exact binaries A/B'd
  (isolated worktrees, see below): combined prefill+1-decode frontier
  (`max_abs_diff=0.0`, 248,320 logits, top-1/top-64 identical),
  prefill-only isolation (`--gen-tokens 0`), and the decode-only diff used
  for 1e — all three bit-exact, expected since the dequant math is
  unchanged, only the branch chain is removed.
- **Shader byte-identity**: `dense_extra_decode.spv` / `_q2_0` / `_pq2_0`
  and `dense_extra_mmq.spv` / `_q2_0` / `_pq2_0` (the builds the guard model
  and any Q2_0/PQ2_0/PTQ1_0-carrying file actually use) are sha256-identical
  before and after — the new `P36_IQ2S_ONLY` branch is additive, the
  existing branches are untouched. Guard not separately A/B'd for this
  reason, same as 1e.
- **ISA, checked before building anything real** (`mmq_info`,
  `VK_KHR_pipeline_executable_properties`, matching the W8 probe
  methodology): decode VGPRs 64->56, occupancy 16->18 subgroups/SIMD (both
  *improved*, no cliff — unlike W8's QT-widening probe); mmq VGPRs 88->84,
  LDS/occupancy unchanged (11776 B, 10 subgroups/SIMD). No register-pressure
  risk anywhere.
- **Whole-model 21-rep interleaved A/B was weak and ambiguous on its own**
  (ctx 512, gen 128; 7-rep even weaker): decode 78.75->79.16 = **+0.52%**,
  heavy overlap between arms (not the clean separation 1e's 21-rep run
  showed); prefill 556.77->545.53 = **-2.02%**, inside the 3.4% MoE floor
  but this fix *does* touch prefill (unlike 1e), so "probably noise" wasn't
  good enough on its own.
- **Resolved by kernel-level profiling instead of more reps** — a cleaner
  instrument than fighting session noise with rep count. Differential
  profile (`P36_VK_PROF_KERNEL=1`, gen 1 vs gen 65 delta/64) isolates the
  kernel directly: decode op cost **1.891 -> 1.822 ms/tok (-3.7%)**, which
  is ~0.58% of the 11.9-12.0 ms/tok whole-model decode budget — matching
  the noisy end-to-end +0.52% almost exactly, not a coincidence. Single-shot
  prefill profile (`--gen-tokens 0`) settles the prefill question directly:
  `dense_extra_mmq_iq2s` **132.151 ms vs dense_extra_mmq 138.320 ms
  (-4.46%)**, and *whole-prefill* GPU time **808.025 ms vs 825.393 ms
  (-2.10%)** — prefill is measurably faster with this change, which
  directly contradicts the throughput A/B's -2.02% reading and confirms it
  was session noise, not a regression.
- **Why the win is smaller than the ISA numbers suggested going in**: worth
  recording so a future session doesn't re-derive it. `pc.type` is a push
  constant — the same value for every thread in a dispatch — so the
  "runtime branch chain" this removes was already a *uniform* (non-divergent)
  branch, which GPUs execute cheaply via predication/scalar jumps, not the
  per-thread-divergent branching that's actually expensive on SIMD hardware.
  The VGPR/occupancy improvement is real (fewer live temporaries from dead
  branches), but the dominant remaining cost is the dequant arithmetic
  itself — multiple `load_u8` calls, `signed_grid`/table lookups, bitfield
  extracts — which this change deliberately left untouched to guarantee
  bit-exactness. That arithmetic is the next lever if anyone wants to keep
  pushing this specific kernel; not attempted here.
- **Verdict: ACCEPTED.** Small (2-4% on the specific kernels, <1% decode /
  ~2% prefill end-to-end) but real by two independent measurements
  (kernel-level profile and, for prefill, a controlled single-shot total),
  bit-exact, zero measured downside anywhere, and the one real bug found
  during construction was caught by review before it ever reached a
  benchmark. Default on, `P36_VK_IQ2S_EXTRA=0` disables.
- A/B'd in isolated worktrees (`p36-wt/iq2s-tuned-kernel{,-base}`) built
  from clean `44143d8`, not the shared main worktree — the same concurrent
  instance from 1e was still mid-flight on HANDOFF item 2 (small-batch
  IQ2_S kernels) in the main worktree the whole time this was built; two
  benchmark runs had to queue behind their GPU usage via `flock` rather
  than racing `bench_ab.sh`'s bare `pgrep` check.
- Raw evidence: `karpathy/evidence/raw/ab-iq2s-tuned-kernel.csv` (21 reps),
  `ab-iq2s-tuned-kernel-7rep.csv`, `ab-iq2s-tuned-kernel-summary.txt`,
  `iq2s-tuned-kernel-parity.txt`, `iq2s-tuned-kernel-shader-hashes.txt`,
  `iq2s-tuned-prof-{on,off}-{gen1,gen65,prefill}.txt`.

## IQ2_S in the small-batch (2..127 token) fused MoE expert pair — ACCEPTED (2026-09-21)

Branch `trackB-ptq1_0`, base HEAD `44143d8`, model `Qwen3.8-35B-A3B-IQ2_M.gguf`.
HANDOFF-iq2s-20260920.md §5 item 2 — the last known q8-route residue on this
file outside the GEMM range. Full write-up:
`evidence/raw/iq2s-smallbatch-VERDICT.md`.

- **What it is.** `moe_gate_up_f32b` / `moe_down_q2k_f32b` were IQ2_XXS/Q2_K-only
  builds, so the admission predicate
  `if (iq2s && !(gemm || (identity && down_sum_decode))) return 0;` rejected
  IQ2_S for every `n_tok` between the 1-token identity pair and the GEMM range,
  and the whole MoE FFN took the q8 route. Ported the same IQ2_S element
  mapping the decode and GEMM ports already use (82-byte block, `d` f16 at 0,
  `qs[64]` at 2, signs at 34, `qh[8]` at 66, `scales[8]` at 74, grid at table
  word 544) into both kernels behind `#ifdef P36_MOE_IQ2S`, added the `tables`
  binding the down kernel needed (6 vs 5), and widened the predicate to
  `if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;`.
  Coverage is `n_tok` 2..127, wider than HANDOFF's stated 2..31
  (`P36_VK_MOE_PAIR_TILE = 8`, `P36_VK_MOE_GEMM_MIN` default 128).
- **Scope is `down_iq2s`-only on purpose**: the IQ3_S-down layers (`il=0,1,2`,
  entry 1c) still fall back to q8 outside the GEMM range, so this change cannot
  put an IQ2_S kernel in front of a 110-byte block.
- **Route-miss evidence that the change is live, not the A/B's assumption**
  (`P36_VK_MOE_ROUTE_DEBUG=1`, ctx 512, `--prefill-chunk 16`, `--gen-tokens 0`):
  A `44143d8` = **1280** misses, B = **96**, and the surviving 96 are exactly
  `il=0,1,2 × 32 forwards` — i.e. 37 IQ2_S-down layers × 32 forwards moved onto
  the new kernels. Remaining misses belong to 1c's scope, not this one.
- **A bug was caught before the benchmark, by measurement not by review.** The
  first draft hand-derived the IQ2_S slotting and was wrong — `max_abs` 7.86 vs
  the q8 route, top64 41/64. Isolating it at `n_tok=1` by forcing identity
  forwards (`P36_VK_MOE_DOWN_SUM_DECODE=1` vs `=0`) pinned the fault to the down
  kernel. Replaced the custom layout with the accepted sum-decode IQ2_S loop
  copied verbatim (`y_off = 128*(itid>>3) + 2*(itid&7)`, `i += 2`, inner
  `p = 0..7`): 7.86 -> 0.858 vs sum-decode, 0.788 vs q8. Copying an accepted
  loop beat debugging a novel one.
- **Parity** (frontier 512, 248,320 logits, three arms): f32b(new) vs q8
  `max_abs` 0.779015 / top-1 match (13) / top64 61/64; the accepted GEMM arm vs
  q8 1.15471 / 61/64; f32b vs gemm16 0.980383 / 62/64; guard vs q8 0.796737 /
  63/64. The new pair is *closer* to q8 than the already-accepted GEMM arm, and
  61/64 is this file's existing `n_tok>1` round-off floor — same score as the
  accepted arm, not a new defect.
- **Verdict: ACCEPTED.** 7-rep interleaved A/B (ctx 512, gen 128, gate 3.4%):
  at `--prefill-chunk 16` — the arm that actually dispatches `n_tok=16` —
  prefill **72.92 -> 116.21 t/s = +59.37%** (MAD 0.440/0.130), decode 70.60 ->
  69.45 = -1.63% (A MAD 1.310). The `--prefill-chunk 256` null control
  (`n_tok=256` -> `gemm`, changed dispatch site not on the path) reads +3.23%
  with B MAD 17.08 and fully overlapping ranges: inside the 3.4% MoE floor, no
  signal, which is the correct result for an untouched path. Decode has no
  mechanism for a regression (`P36_VK_MOE_DOWN_SUM_DECODE` defaults on, so the
  1-token pair was already admitted) and both deltas sit inside the 9.4% spread.
- Base builds byte-identical: `vulkan/moe_down_q2k_f32b.spv`
  `0ce317941dbef148…` and `vulkan/moe_gate_up_f32b.spv` `9701f0330b5b5d73…`,
  equal to the same files built from clean `44143d8`, so every non-IQ2_S build
  variant is bit-for-bit unchanged. `compat_gate.sh` -> PASS.
- **Method notes worth reusing.** (1) `--prefill-chunk 16` is the only way the
  existing harness can be made to dispatch `n_tok ∈ 2..31` at all; a chunk-256
  A/B would have measured nothing and looked like a null result for a correct
  change. (2) The baseline must be a worktree at the *candidate's* HEAD, not a
  stale `/tmp/p36-*` copy: the first attempt used a `b738610` baseline and would
  have folded the other session's -2.01% `ssm-pair` prefill into this delta.
  (3) `tests/bench_ab.sh` has no `flock`, so its bare `pgrep` check bounces
  whenever a *parallel instance* runs a benchmark — two attempts failed that way
  before the run was queued behind `/tmp/p36-gpu.lock`.
- Caveat: the target workload (a batched server with concurrent requests) cannot
  be measured here — `bench_ab.sh` has no concurrency knob — so chunk 16 is a
  proxy for the routing, not a serving measurement.
- **Attribution caveat, disclosed rather than buried.** The B arm was the shared
  main worktree, which at run time also carried the concurrent instance's
  `dense_extra_*` IQ2_S specialization (default on, committed afterwards as
  `f36daf4`). The +59.37% is therefore "their change + this one" versus
  `44143d8`, not this change alone. Two independent reasons it still stands:
  their own 21-rep A/B measured that peer change at decode +0.52% / prefill
  -2.02% (both inside the noise floors), and the `--prefill-chunk 256` null
  control below — where *this* change's dispatch site is never reached — bounds
  the two combined at <=3.23% with fully overlapping ranges. The route-miss
  count (1280 -> 96) is specific to this change, was taken on the same B binary,
  and is what the verdict actually rests on.
- Raw evidence: `karpathy/evidence/raw/iq2s-smallbatch-VERDICT.md`,
  `ab-iq2s-smallbatch-chunk{16,256}.{csv,summary}`,
  `iq2s-smallbatch-parity.txt`, `route-miss-A.txt`, `route-miss-B.txt`,
  `iq2s-smallbatch-ab3.sh`.

## Flash-attention prefill (`attn_prefill_fa.comp`, GQA 6 + GQA 8) — ACCEPTED (2026-09-22)

- **What changed.** `attn_prefill_qtile2{,_gqa6}` (2 tokens per workgroup, six Q
  rows re-read from LDS per K vector, ~1 FMA per LDS float, ~0.8 TFLOP/s at ctx
  8192) replaced by one parametric flash-attention kernel: 8 tokens x 6 heads
  (dense) or 4 tokens x 8 heads (MoE) per workgroup, K/V dequantized once per
  16-key tile into LDS as f32 (exact for Q8_0/Q4_0), Q in registers, 16-lane
  clustered score reduction. Same per-4096-key-group partials, `attn_combine`
  untouched. `P36_VK_ATTN_FA=0` falls back. Branch `perf/attn-prefill-fa`.
- **Measured.** Kernel: 2.4–2.6x (GQA 6), 3.0–3.5x (GQA 8) at pos0 >= 3968,
  18/18 parity cases PASS (max_abs <= 8e-5 even with very peaked softmax).
  Whole model, ctx 8192, 5 interleaved reps: Swift 27B prefill **97.98 ->
  107.28 t/s (+9.49%)**, Qwen3.6-35B-A3B guard **386.47 -> 491.38 t/s
  (+27.15%)**, reps disjoint on both. Grows with context (attention is the
  quadratic term); ctx 1024 impact is small by construction.
- **Quality gate used, and why.** Long-context frontier logits are chaotic under
  any f32 reorder (baseline chunk 256 vs 128 flips top-1 at 8k), so max_abs is
  not a gate there. Teacher-forced NLL over 16 frontiers (7952..8192): FA's mean
  |dNLL| vs baseline is 3.08 (Swift) / 2.01 (guard), below the chunk-128
  reference's 3.58 / 2.60. Guard frontier-8192 top-1 same, top64 59/64. Guard
  is no longer byte-identical: this is an intentional f32 reassociation.
- **Not a regression:** guard A/B decode −4.92% with no GPU mechanism —
  differential decode profile 15.572 vs 15.580 ms/tok, no FA dispatch at n_tok=1.
- `reconsider_if`: a real long-context task eval regresses beyond what the
  chunk-size reference also shows. Next lever on the same kernel: the GQA-6
  build sits at 256 VGPRs / 4 subgroups vs GQA-8's 128 / 8 — occupancy.
- Evidence: `karpathy/evidence/attn-fa-prefill.md`, `raw/ab-fa-*`, `raw/fa-*`,
  `tests/test_attn_fa.c`, `karpathy/tools/{cmp_logits,frontier_nll}.py`.

## PQ2_0/Q2_0 small-batch matmul (`dense_extra_small_q2.comp`) — ACCEPTED (2026-09-23)

Model `TERNARY-BONSAI-2-27B-DERISKED-PQ2_0.gguf` (402 PQ2_0 tensors, dense 27B).

- **Problem.** Any PQ2_0 batch of 2..127 tokens went to `dense_extra_mmq_pq2_0`,
  whose 128-token tile costs the same for 2 tokens as for 128: `--prefill-chunk 2`
  measured 2.96 t/s (a 2-token step = 26x a decode step). Hit by MTP verify, the
  ragged tail of every prompt, short incremental (KV-reuse) prefills and batched
  server decode.
- **Change.** New kernel: one wave64 per 2 rows streams each row once, expands
  it through the decode kernel's LUT once, and dots it against up to 8 tokens.
  Host takes it for `n_tok <= P36_VK_Q2_SMALL_MAX` (default 16), and for a
  ragged tail (`n_tok % 128 <= 16`) after an mmq over the aligned head.
- **Parity: bit-exact** against n one-token decode calls — per token the integer
  sums, fma order and subgroupAdd are the decode kernel's.
  `tests/test_pq2_small`: 27/27 PASS (5120x17408, 17408x5120, 5120x1001;
  n_tok 1..16), 0 mismatching bits; the 24/32/48/64-token rows print `mmq` and
  are timed only, since above the threshold they take the f16-accumulating tile.
  Greedy CLI output identical to main. Raw: `raw/pq2-smallbatch-parity.txt`.
- **Measured** (re-run and archived; the first in-session sweep was not saved and
  read 2.96/6.0/13.8 -> 55.6/71.1/79.2, same direction and verdict):
  ctx-64 prefill by chunk, A=main vs B=this kernel:
  2 → **2.95 → 54.60 t/s (18.5x)**, 4 → 6.07 → 70.41, 8 → B 79.74,
  16 → B 82.03. Prompt tails (mmq head + small ragged tail):
  pp130 **110.35 → 189.51 (+71.7%)**, pp140 **118.74 → 172.69 (+45.4%)**,
  pp256 211.51 → 214.78 (+1.5%, control — no small batch there).
  A 2-token batch costs 1.38-1.6x a one-token call.
  No regression where it is inert: 7-rep interleaved ctx 1024 A/B, prefill
  -0.03%, decode +0.00% (`raw/ab-pq2-smallbatch-ctx1024*`).
  Raw: `raw/pq2-smallbatch-chunk-sweep.txt`.
- `reconsider_if`: a ragged tail of 17..127 tokens matters (raise the threshold
  per shape: the 17408-in shapes still win vs mmq to ~48 tokens, the 5120-in
  ones cross over near 16).
- Evidence: `evidence/pq2-smallbatch.md` (running record),
  `raw/pq2-smallbatch-{parity,chunk-sweep}.txt`, `raw/pq2-bonsai-workload.txt`,
  `raw/ab-pq2-smallbatch-ctx1024.{csv,summary.txt}`, `tests/test_pq2_small.c`,
  `vulkan/dense_extra_small_q2.comp`. Campaign map: §6.7, §8.

## PQ2_0 decode matvec workgroup shape — REJECTED (2026-09-23)

- **GPU streaming ceiling measured: ~437 GB/s** (standalone Vulkan probe,
  `raw/probe.c` + `raw/stream.comp`, identical for memory types 0/3/5), re-run
  and archived: `raw/pq2-shape-rejected/probe-rerun-20260923.txt`. The PQ2_0
  decode matvec runs at ~318 GB/s, so the kernel is ~27% under the bus.
- **RETRACTED sub-claim:** "a 64-thread one-wave workgroup reading 4 KB exits at
  ~251 GB/s while the same 4 KB in 256-thread workgroups reaches ~443 GB/s". The
  archived chunk probe re-run gives **~298 GB/s for both WG=64 and WG=256**
  (same file, command lines spelled out), so the launch-limit story that motivated
  the wave-packing experiment is unconfirmed. The experiment's verdict does not
  rest on it — it was rejected on its own in-kernel measurement (below). Do not
  cite the 251/443 pair again.
- Probes were a dead end for the matvec. What the kernel actually did, all
  in-engine, all neutral-or-worse, and **no raw output archived** (the variants
  were reverted; re-deriving them means rebuilding them):
  - grid-stride LUT amortization (fewer workgroups): cap 320/640/1280/2560 →
    kernel +43/+11/+5/+2% slower;
  - ROWS 1/2/8/16 vs 4: +7.5% / -1.8% / +11.5% / +31% kernel time;
  - four 4-row wave64s per 256-thread workgroup: 1385 → 1381 ms, neutral.
  The matvec's own access pattern (4 rows x 544 B per wave, `mv2.comp`) still
  caps at ~273 GB/s with aligned code-only loads, no scale and no q8 loads
  (`raw/pq2-shape-rejected/probe-rerun-20260923.txt`), i.e. the ceiling is the
  row-granular layout, not the dispatch shape.
- Artifacts: `raw/pq2-shape-rejected/` (probe sources + `probe-rerun-20260923.txt`
  with the exact rebuild/run lines, the two 2k/4k decode logs, `prof*.txt`,
  `span8k.txt`). The ROWS variants were a one-line `#define ROWS` change in
  `vulkan/dense_extra_decode.comp` (committed in its ROWS=4 form); their SPIR-V
  exists only in the worktree because the repo ignores `*.spv`, so re-deriving
  them means re-applying that one-line edit.
- `reconsider_if`: a load layout with wide aligned per-lane loads that does not
  need an offline repack (rows are 16 B aligned at 1360 B, blocks are not).

## GQA-grouped split-K decode attention — REJECTED (2026-09-23)

- **Change tried.** `attn_decode_split_gqa{6,8}`: one workgroup per (kv head,
  token, span) loads each K word / V value once for all HQ heads. **Bit-exact**
  vs `attn_decode_split` (18/18 cases, 0 mismatching bits, 600..65535 keys).
- **Measured.** Kernel alone: 0.3-0.7x at 600-4k keys (fewer workgroups),
  ~1.0-1.17x at 16k-64k. End to end (same binary, env toggle, gen 64): 1k
  -8.45% (MAD 0.02); 1.5k/2k/4k -2..-9% in a single sweep. Two 5-rep A/Bs showed
  "+11% at 8k" and "+23% at 2k", both artifacts — see next entry.
- End-to-end numbers: `raw/ab-attn-decode-gqa-ctx{1k,2k,8k}.{csv,summary}`.
  Kernel parity/timing is `tests/test_attn_decode_gqa.c` (the per-call env gate
  was `P36_VK_ATTN_DECODE_GQA`; only `n_tok == 1` reaches the split-K path, so
  batches of 2+ would have tested nothing).
- Patch (shader + host wiring + test, unapplied), shader and test source:
  `raw/attn-decode-gqa-rejected.patch` and `raw/attn-decode-gqa-rejected/`.
  The host wiring was reverted out of the worktree; nothing in the tree depends
  on these files.
- `reconsider_if`: contexts well past 32k become the target workload; then add
  a narrower span for the grouped kernel only (not bit-exact vs span 512).

## Pitfall: decode t/s after a long prefill is thermal noise (2026-09-23)

- On this board, decode measured right after a >=1.5k-token prefill swings
  20-32 t/s run to run with identical binary and settings (ctx 4096: 30.9, 21.8,
  25.2). Per-kernel `gpu_ms` is not comparable across runs either (the prefill
  mmq moved 12% between two runs with identical prefill t/s). Short-prefill
  decode (ctx <= 1024) is stable to MAD 0.02.
- Judge decode-only changes at ctx <= 1024, or in a kernel harness, or with
  >=7 interleaved reps and all pairs in one direction.

## MoE IQ2_S down sum-decode: fp64 and wave64 dead, field-load shape is the lever (2026-09-24)

Reopens the audit's 2026-09-24 P1 items and, for the wave32 half, the exclusion
in `evidence/wave32-eligibility.md:146` (`moe_down_q2k_sum_decode.comp` marked
CROSS-LANE-HEAVY on the strength of its `subgroupAdd`, never measured). New
evidence: a per-kernel decode profile plus two in-kernel floor probes that
decompose the kernel's 313.2 ms into DRAM floor, field-load, and ALU cost.

- **Measured baseline** (`moe_iq2s_down_sum_decode`, IQ2_M, ctx 512, gen 128):
  `dispatches=4736 groups=9699328 gpu_ms=313.156` = 37 calls/tok x 2048 rows x
  8 experts x 164 B = **99.4 MB/tok unique** = 12.72 GB/run at **40.6 GB/s**;
  21.6% of decode GPU, 2.46 ms of a 13.42 ms/tok decode.  The audit's ~40 GB/s
  figure is confirmed, and it is unique traffic: the 16 lanes of a half-wave
  share one 82 B superblock, so per-lane redundancy is L1 broadcast.
- **Probe A — fp64 (`fma32`, audit item c) — NEGATIVE.** `fma32` already
  compiles to **one `v_fma_f64` per expert** (plus 3 `v_cvt_f64_f32`, 1
  `v_cvt_f32_f64`), all inside the `tid == 0` tail, i.e. once per expert per
  workgroup, not per weight: there is no 7-op fp64 sequence to remove.  The
  profile-only `fma32 -> fma` swap moved the kernel 315.18/314.65 ->
  314.65 ms, inside noise.  Raw `evidence/raw/p1-sumdecode/isa-f64-counts.txt`,
  `ab-p1-fma.csv`.  Do not reuse the "fp64 emulation is 1/16 rate on every MAC"
  framing.
- **Probe B — wave32 (audit item a) — PREMISE FALSE.** `mmq_info` reports
  `subgroup=32` for `moe_down_q2k_sum_decode{,_iq2s,_iq3s}.spv` (VGPR 48, code
  4004/4112, 20 subgroups/SIMD), so the dispatched blobs already run wave32 and
  no half-wave idles.  Only the non-IQ2S/Q2_K build reports VGPR 64 /
  16 subgroups.  Nothing to force, nothing to fix.
- **Probe C — "experts in parallel" — NEGATIVE.** 3 interleaved env pairs of
  `P36_VK_MOE_DOWN_SUM_DECODE=0` (falls back to the per-expert parallel down
  kernels) vs default: decode **70.65/71.07/70.95** vs **74.46/74.42/74.39 t/s**,
  prefill unchanged.  The serial 8-expert combine with one `subgroupAdd` per
  expert already wins by 4.8%; the audit's "experts in parallel" suggestion is
  dead in that form.
- **Probe D — load shape — POSITIVE, 3.23x.** A diagnostic IQ2_S body that
  keeps the workgroup/block/expert mapping but reads the whole 82 B superblock
  with **3 lane-strided 2-byte loads** (and does no dequant, no mid, no
  tables - `P36_P1_LOADFLOOR`, `diag.comp` + `p1-diag-probes.diff`) runs
  **97.10/97.74/97.69 ms** against the stock **314.88/315.51/315.52 ms** for the
  *same 12.72 GB*: 131 GB/s achievable vs 40.6 GB/s delivered.  End to end the
  run drops 1465 -> 1246 ms of decode GPU, decode 74.4 -> 85.0 t/s.
- **Probe E — ALU is only 51 ms.** The same diagnostic with the stock ~40
  scattered field loads kept verbatim and the dequant arithmetic collapsed to an
  add (`P36_P1_ALUFLOOR`) runs **263.93/264.40/264.53 ms**, so
  `315.4 = 97.0 floor + 167.2 field loads + 51.2 ALU`.  The kernel is
  load-shape-bound, not ALU-bound and not fp64-bound.
- **Why it does not beat the Q2_K branch:** both experts' byte footprints are
  ~identical per block (82 B vs 84 B), but the IQ2_S branch fetches its scale /
  9-bit grid index / qh / sign fields with ~40 independent 2-byte loads at
  scattered offsets per lane per block, where the Q2_K branch of the same file
  needs ~7 loads per block.  That is the whole 40.6 vs ~62 GB/s gap.
- **Not built this session.** The fix (fetch the superblock with coalesced
  lane-strided loads and redistribute the field bytes with `subgroupShuffle` or
  LDS, keeping the per-element fma order so the result stays bit-exact) is a
  real rewrite of the IQ2_S branch; it was left unbuilt rather than starting a
  build the remaining audit items cannot be traded against.  Probes D/E are the
  evidence that it is worth it.  Raw:
  `evidence/raw/p1-sumdecode/P1-VERDICT.md`, `ab-p1-loadfloor.csv`,
  `ab-p1-alufree.csv`, `summary.csv`, `iq2m-decode-base.txt`, `loadfloor.spv`,
  `alufree.spv`.
- `reconsider_if`: the IQ2_M decode kernel is on the critical path again (it is
  #1 at 21.6% of decode) and someone builds the coalesced-fetch version; target
  315 -> 200-230 ms kernel and 74.4 -> ~80 t/s decode, parity `max_abs_diff = 0`
  on the frontier-513 dump because only the fetch changes.

## MTP draft acceptance measured, and the MTP loop is a net loss as shipped (2026-09-24)

Reopens nothing; this is the audit's P2 measurement, and it **corrects** an
earlier reading of the same harness.  Swift, in-file MTP head (`blk.64 nextn`,
`draft=N`), greedy, ctx 512, 512 generated tokens, three prompts (prose / code /
agent transcript).  Raw: `evidence/raw/p2-mtp/*.log`.

- **Draft acceptance, when a draft is actually carried: 100% at depth 1, >= 69%
  at depth 2.**  Depth 1: 172/172 (story, `--mtp-margin 0`), 60/60 (story,
  margin 3), 91/91 (code), 77/77 (agent).  Depth 2 (`--mtp-draft 3`,
  `draft_cap = 2`): 100 calls committed both drafts out of at most 144 two-draft
  calls, so the second draft is accepted at least 69.4% of the time (the
  `P36_MTP_STATS` `drafted` count gives the exact denominator).  The audit's
  >= 60% P7 gate is met.
- **CORRECTION — the `p36: MTP stats ... accept=100.0%` line is not an
  acceptance rate at the default setting.**  `p36_mtp_stats_add(verify_n,
  commit_n, ...)` is only called on calls that carry a draft, and
  `--mtp-draft N` sets `draft_cap = N - 1` (`p36.c:12366`), so at the published
  default of 2 `verify_n` is always 1 and `commit_n` is always 1: the line is
  100% by construction (earlier runs read `drafted=60 accepted=60 full=60`, and
  so on).  Set the margin to 0 and/or draft to 3+ before quoting acceptance.
- **MTP makes decode SLOWER at every setting measured — 6 of 6 prompt/flag
  pairs, 0.4-7%.**  ctx 512, gen 512, t/s (plain -> MTP): story 22.63 -> 21.08
  (margin 3) / 21.14 (margin 0) / 22.24 (earlier run, margin 3); code 21.89 ->
  20.92 / 21.81; agent 21.87 -> 20.98 / 20.99.  Prefill is untouched
  (186-191 t/s in every arm).
- **Why, and it is structural: at `draft_cap = 1` the verify row is a new
  position, so it costs a whole extra forward.**  `histogram=k` counts *tokens
  committed per call* (`1 + commit_n`, `p36.c:12483`), so solving the two
  `--mtp-margin 0` arms (340 calls: 168 committing 1, 172 committing 2, 24.22 s;
  452 calls: 392 / 60, 24.29 s) gives **47.5 ms per one-token spec call and
  94.4 ms per two-token spec call = 47.2 ms/token, against 44.19 ms for a plain
  step**.  The piggybacked draft head is only +3.3 ms (7.5%); the loss is that
  two forwards buy two tokens.  Depth 1 is therefore break-even at best and a
  loss after the draft head — it cannot be tuned into a win, only kept from
  hurting.
- **The default margin gate is pure cost.**  `--mtp-margin 0` beats the shipped
  margin 3 on tokens/call by 33% with no measurable decode change; the
  `s->mtp_draft_token != p36_session_argmax(s)` check that cannot be turned off
  is already the free, exact filter.  Backoff then spends 1-4 draftless tokens
  after each miss, which is why only ~47% of calls carry a draft at margin 0
  (172/340) and ~13% at margin 3 (60/452).
- **Depth 2+ collapses: `--mtp-draft 3` (draft_cap 2) = 4.17 t/s and
  `--mtp-draft 4` (draft_cap 3) = 4.44 t/s** against 22.63 plain, while still
  committing 1.910 / 1.673 tokens per call and with prefill unchanged at
  ~187 t/s.  The same algebra puts a two-draft call at ~1.1 s, i.e. ~0.4 s per
  committed token, 8-9x a plain step; the direct evidence is the per-kernel
  decode profile in `evidence/raw/p6-smallbatch/`.  Mechanism: for IQ3_XXS only
  `n_tok == 1` is given a decode kernel (`p36_vulkan.c:8687-8730`), so the
  moment the verify carries a *second* row it takes `dense_iq3_xxs_mmq` (a
  `ceil(n_tok/128)` grid) plus a `predequant_b16` pass — the 128-token prefill
  tile.  `p36_vk_micro_batch` exists but is only wired for Q8_0
  (`p36_vulkan.c:5173`).  This is the audit's P6, measured from the outside.
- `reconsider_if`: the small-batch (n_tok 2..8) decode kernel from P6 lands — at
  draft depth 1 there is nothing to fix (one verify row is one row), so the win
  has to come from depth 2, which is exactly the case that is broken today.  The
  break-even is arithmetic: at the measured 1.469 tokens/call a call may cost at
  most `1.469 x 44.19 = 64.9 ms` and costs 94.4; with two drafts accepted at
  100% / >= 69.4% the same call commits 1 + 1 + 0.694 = 2.694 tokens, so a
  two-row verify that costs ~1.2 plain steps projects ~50 t/s (2.2x).  A kernel
  that loads each weight block once for both rows (the MoE sum-decode shape, or
  `dense_extra_small_q2`) is the only way there.

## Bit-exact integer IQ3_XXS decode — REJECTED: no packed signed dot on GFX1013 (2026-09-24)

Reopens **"Integer sign-mask IQ3_XXS decode — REJECTED"** (2026-09-17 §"BC-250
transfer audit": 373.83 -> 407.97 ms over 16 generation tokens) and §7's closed
line **"`int dot: 0` is a hardware truth on GFX1013"**.  The reopening case was
legitimate: that 2026-09-17 variant applied the sign *per weight* and still paid
the per-weight convert + fma, so it never tested the audit's actual proposal (a
packed signed integer dot).  New evidence reproduced here: the shipped
`dense_iq3_xxs_decode_r4` loop is 5.4 VALU per weight MACC (690 VALU / 128 MACCs
per lane, `evidence/raw/iq3xxs-isa/dense_iq3_xxs_decode_r4.isa.txt`) against a
20.081 ms/tok loads-only floor and a 26.578 ms/tok baseline, i.e. 6.497 ms/tok
(24.4%) of exposed ALU (`evidence/raw/iq3xxs-isa/W2-VERDICT.md`).  The audit's
arithmetic premise also holds: grid magnitudes <= 62, q8 <= 127, 32-term partial
sums < 2^24, so the shipped f32 loop *is* exact integer math and an int32 inner
product would be bit-identical (`max_abs_diff = 0`).

- **Probe — does this chip have a packed signed 4x8 dot? NEGATIVE.** A minimal
  `dotPacked4x8EXT(int, int)` shader over two SSBOs (`dot4.comp`) is accepted by
  `./glslc --target-env=vulkan1.1 -O`, but the ACO dump shows **zero**
  `v_dot4_i32_i8`.  ACO lowers it to 4x
  `v_mul_i32_i24_sdwa ... src0_sel:BYTE_n src1_sel:BYTE_n` + 2 `v_add_nc_u32`/
  `v_add3_u32` + 1 accumulate = **8 VALU per 4 weights = 2 VALU/weight for the
  MAC alone**, with the byte select eating SDWA slots.  Raw: `dot4.comp`,
  `dot4.isa.txt`, `dot4.stats.txt`.
- **Op accounting.**  shipped f32 5.4-6.0 VALU/weight; int with the signs from a
  *second* packed dot ~5.0; int with signs pre-folded into the weight word ~3.0;
  int with signs pre-folded + SWAR negation ~4.75 (the byte-wise `~w + 1` carry
  corrupts a zero magnitude).  The <= 2.5 target needs the sign-free 3.0 row,
  which needs the grid word to *already be signed* — 16 sign variants of the
  512-entry table, 32 KB of LDS built per workgroup, over budget and a
  per-dispatch build cost.  The two reachable layouts (5.0, 4.75) buy
  0.3-1.0 ms/tok = 1-2% of Swift's 46.7 ms/tok decode, below the dense gate and
  far below the audit's 13-15% estimate, which assumed a packed dot that this
  hardware does not have.
- **Not built.**  `device-query.txt:165` `has_accelerated_dot_product = 0` and
  the §7 closed line stand.  Caveat recorded for honesty: the ISA dump was made
  by a tool that never enables
  `VkPhysicalDeviceShaderIntegerDotProductFeatures`, so it shows ACO's fallback
  lowering; the hardware claim rests on RADV's own feature query, not on the
  ISAs.  That does not change the verdict — a capability the driver refuses is a
  capability the shipped pipeline cannot use.
- **The Q4_K/Q5_K half of the audit item does not inherit this verdict, but its
  only remaining lever is a *load* lever, not an integer one.**  Read
  `vulkan/dense_kquant_decode.comp:229-234`: for `QTYPE == Q4_K` the dword is
  fetched at `base + 16 + (group >> 1) * 32 + (idx & 31)` with
  `group = itid >> 1` and `idx = 16 * itid + 4 * i`, so lanes `4k` and `4k + 2`
  read the **same** dword row and select opposite nibbles (`group & 1`), and
  lanes `4k + 1` / `4k + 3` likewise — every 32-value group's bytes are loaded
  twice from L1.  The audit's "lane pairs reload the same dword for the other
  nibble" is confirmed.  No packed dot is needed to fix it (one lane unpacks both
  nibbles of a dword row), but that is a rebuild of the loop's lane mapping, not
  a probe; left unbuilt and listed under `reconsider_if`.
- Raw: `evidence/raw/p4-iq3xxs/P4-VERDICT.md`, `dot4.comp`, `dot4.isa.txt`,
  `dot4.stats.txt`, plus the pre-existing `evidence/raw/iq3xxs-isa/`.
- `reconsider_if`: hardware with a real `V_DOT4_I32_I8` (RDNA3+/gfx11), where the
  MAC drops to 1 op per 4 weights and the 2.5 target becomes reachable; or a
  repacked IQ3_XXS layout that stores the magnitudes already signed, which makes
  the 3.0 VALU/weight row reachable with no per-4-weight sign work; or for the
  Q4_K/Q5_K half, anyone who builds the one-lane-per-dword-row load mapping
  (lm_head Q5_K is 5.8 ms/tok at ~282 GB/s vs IQ4_XS 360).

## Grouped-GQA decode attention re-probe — still REJECTED, the premise was bandwidth and the kernel is not bandwidth-bound (2026-09-24)

Reopens **"GQA-grouped split-K decode attention — REJECTED (2026-09-23)"**.  The
new case is the audit's P5 premise, and it is *bandwidth*, not occupancy: one
workgroup per query head re-reads and re-dequantizes every KV row 6x (Swift) /
8x (MoE), and the guard at ctx 8192 spends 3.0 ms/tok on ~68 MB of unique KV
(~22 GB/s), so the kernel looked bandwidth-starved — something the 2026-09-23
probe (1k/2k/8k end-to-end only) could not see.  The new evidence is a per-`pos0`
kernel curve for both model shapes, from the harness the rejection stored: shader
and test rebuilt from `evidence/raw/attn-decode-gqa-rejected/` and verified
byte-identical to it (`diff`), rewired behind the same per-call env gate
(`P36_VK_ATTN_DECODE_GQA`), wiring reverted out of the tree again afterwards.

- **Parity holds, exactly as before.**  18/18 cases bit-exact
  (`mismatches=0 max_abs=0`): dense 24 q / 4 kv and MoE 16 q / 2 kv, head_dim
  256, q8_0 K + q4_0 V, attention sinks on and off, pos 600..65535, ragged last
  tile and the exact two-span boundary included.
- **The curve is the new information, and it inverts the premise.**  Ratio is
  per-head/grouped, so **< 1 means the grouped kernel is slower**:

  | pos0 | dense 24/4 per-head | grouped | ratio | MoE 16/2 per-head | grouped | ratio |
  |---|---|---|---|---|---|---|
  | 1023 | 0.123 ms | 0.263 ms | 0.47x | 0.117 ms | 0.327 ms | 0.36x |
  | 4095 | 0.184 | 0.257 | 0.72x | 0.164 | 0.330 | 0.50x |
  | 8191 | 0.312 | 0.293 | **1.07x** | 0.248 | 0.336 | 0.74x |
  | 16383 | 0.582 | 0.579 | 1.01x | 0.405 | 0.388 | 1.04x |
  | 32767 | 1.076 | 1.032 | 1.04x | 0.746 | 0.735 | 1.02x |
  | 65535 | 2.061 | 1.810 | 1.14x | 1.494 | 1.397 | 1.07x |

  Grouping is a **2.1x (dense) / 2.8x (MoE) loss** at 1k keys, first wins at 8k
  (dense) / 16k (MoE), and where it wins it is 1.01-1.14x of *one attention
  dispatch*.
- **Why the bandwidth premise does not cash.**  The re-read is real at L1/L2 but
  not at DRAM: per-head at 32767 keys moves 54.5 MB of unique KV (32768 rows x
  1664 B; 4 kv heads x 256 dims, q8_0 K + q4_0 V) in 1.076 ms = **50 GB/s, 11%
  of the 454.4 GB/s roofline**.  One span is 512 x 1664 B = 852 KB and the 6
  query-head workgroups sharing a kv head walk the same span, so the 6x re-read
  is served inside L2: removing it saves L2 traffic and redundant dequant work,
  not DRAM.  The kernel is latency/occupancy-bound at the contexts this board can
  measure, which is why cutting the workgroup count 6x costs 2x at 1k.
- End-to-end the 2026-09-23 numbers stand and were **not re-run** at 8k/16k/32k:
  1k -8.45% (MAD 0.02 — the ledger's own "judge decode at ctx <= 1024" rule), and
  the earlier "+11% at 8k" sits inside the 20-32 t/s run-to-run swing of *decode
  after a long prefill* (entry above).  A 1.01-1.07x change in one dispatch of a
  33-36 ms/token decode is not measurable under that pitfall, so the kernel curve
  is the verdict, not another noisy A/B.
- Raw: `evidence/raw/p5-gqa-decode/gqa-kernel-repro-20260924.txt` and
  `host-wiring-20260924.patch` (this run), plus the pre-existing
  `evidence/raw/attn-decode-gqa-rejected{,.patch}` and
  `evidence/raw/ab-attn-decode-gqa-ctx{1k,2k,8k}.{csv,summary}`.
- `reconsider_if`: unchanged, now with a shape attached — group only *past* 8k
  and keep the workgroup count up by halving the grouped build's span
  (512 -> 256), which gives up bit-exactness against span 512; or hardware where
  the attention dispatch is genuinely DRAM-bound at ctx <= 4k.

## Prefill f16 accumulator and f16 DeltaNet state — NLL check, no measurable regression (2026-09-24)

Audit §P3 ("measurement only").  Not a rejection of an earlier attempt: the
ledger has no entry for either dtype, and `P36_VK_RECURRENT_STATE_F16` was
landed on throughput with no quality probe behind it.  Mechanism under test:
`vulkan/dense_iq3_xxs_mmq.comp` accumulates the whole K dimension (up to 17408,
68 blocks) in `f16vec2 sum[16]` while the decode path accumulates in f32, and
the DeltaNet recurrent state can be stored f16 or f32.

- **Method.** `p36-bench --gen-tokens 0 --dump-frontier-logits-dir` walks
  `tests/long_context_story_prompt.txt` one frontier at a time (each frontier
  teacher-forces one more token and dumps full-vocab logits); `p3stats.py`
  computes per-frontier NLL of the actual next token and the *paired* signed
  difference between arms, with top-1 agreement, sign split and MAD.  The
  sequence is identical in both arms (same prompt, greedy, no sampling), so the
  paired difference is meaningful without a sampler in the loop.
- **Determinism control first.**  Chunk 256 run twice in two processes:
  `mean_nll=8.6713` both, `dNLL mean=med=MAD=mean|d|=0.0000`, `top1_same=25/25`.
  Nothing below is run-to-run noise.
- **Arm A, prefill chunk 256 (f16 accumulator) vs chunk 1 (decode path, f32):**
  ctx 512..2048 n=25 -> mean NLL 8.6713 vs 8.7357, **dNLL median -0.0000**,
  MAD 0.2753, mean|d| 0.9698, worse/better 10/14, top-1 **25/25**; ctx
  2048..4096 n=33 -> 13.1309 vs 12.5390, **median +0.0005**, MAD 0.4474,
  mean|d| 1.7924, worse/better 17/15, top-1 **32/33**.  Median ~0 and a ~50/50
  sign split at both windows: no systematic penalty.  The means (+0.06, -0.59)
  are set by a handful of frontiers where the distribution is nearly flat
  (frontier 832 +6.53, 1408 -3.00, 1664 +2.92; the 2048..4096 window sits at
  NLL ~13 nats).
- **Arm B, f16 vs f32 recurrent state (`P36_VK_RECURRENT_STATE_F16=0`):** the
  usable sample is ctx 8192..16384 n=33 -> mean NLL 15.0033 (f16) vs 15.3388
  (f32), **median -0.0053**, MAD 1.3570, mean|d| 3.0220, worse/better 14/19,
  top-1 **30/33**.  The two small windows agree in direction and add nothing:
  ctx 2048..8192 n=4 median -1.1968 top-1 3/4; ctx 16384..32768 n=2 median
  -1.4499 top-1 2/2.  Scatter is large at 15 nats of NLL because top-1 there is
  a near-tie decision.
- **Non-finite check.**  Every frontier dump in all six arms (25+33+4+33+2+2
  files, full 248,320-logit vocab each) scanned: no NaN, no +-inf.  Largest
  `max|dlogit|` seen anywhere was 28.3 (ctx 16384, state dtype arm).
- **Verdict: no change.**  The audit made the f32 flush conditional on
  "prefill NLL measurably worse"; it is not, so no flush was added and the f16
  state default stays.
- Honest limitation: chunk 256 vs chunk 1 are different *kernels* (MMQ +
  `predequant_b16` vs `dense_iq3_xxs_decode_r4`), so arm A bounds the whole
  "different prefill path" error rather than the accumulation dtype alone.
  Isolating the dtype needs a diagnostic MMQ build with `vec2 sum[16]`, which
  was not built because there is no signal to chase.
- Raw: `evidence/raw/p3-f16-nll/` (`VERDICT.md`, `p3stats.py`, the five paired
  statistic files, the eleven `p36-bench` logs, and the sweep scripts/outputs
  `run4.s|out`, `run5.s|out`).
- `reconsider_if`: a criterion that is sensitive to a *flat* distribution (an
  exact-match rate over many short completions, paired per token, n >= 64)
  rather than frontier NLL, or hardware whose f16 accumulate path rounds
  differently (GFX11+).  Also relevant: if a future change makes the decode
  path's reduction order match the MMQ tile's, arm A's reference disappears and
  this check has to be re-based on an f32-accumulate MMQ build.

## Two-row dense IQ3_XXS decode (`dense_iq3_xxs_decode_nx.comp`) — ACCEPTED, +62.6% on MTP draft 3 (2026-09-24)

Audit §P6 ("build a dequantize-once / n-FMA decode variant for small batches").
Reopens **P2**'s cost model (`## MTP draft acceptance measured, and the MTP loop
is a net loss as shipped`), whose closing note is that `--mtp-draft 3/4`
collapses to 4.17/4.44 t/s "the moment the verify carries a second row": the
new evidence is that the collapse is one dispatch wide, not a property of small
batches, and it is worth 1.63x once that dispatch has a decode kernel.  The
precedent is the accepted `## IQ2_S in the small-batch (2..127 token) fused MoE
expert pair` (2026-09-21) — same shape of fix, different format; what was *not*
reopened is any of the rejected IQ3_XXS decode variants (sign mask, f16 grid,
rows 8, global LUT), all of which keep the per-weight cvt+fma that P4 killed.

- **Mechanism.** `p36_vulkan.c:8774-8813` gives `n_tok == 1` a purpose-built
  decode kernel and sends everything else to the 128-row MMQ tile
  (`dense_iq3_xxs_mmq` + `predequant_b16`).  A 2-row step therefore pays a whole
  128-row tile.  The new kernel unpacks each weight word once and walks the two
  activation rows through it: the loads and the grid/sign work are shared, only
  the FMA chain is duplicated.  Per-row operand order is identical to
  `dense_iq3_xxs_decode_r4`, so each row is *bit-identical* to a separate
  one-token dispatch — that is the parity gate, not a max_abs tolerance.
- **Landed.** `vulkan/dense_iq3_xxs_decode_nx.comp` (local_size 64, ROWS 4,
  NTOK 2), `tests/test_dense_iq3xxs_nx.c`, and a `p36_vulkan.c:8814-8835`
  branch gated on `weight_type == IQ3_XXS && out_dim % 4 == 0 && n_tok == 2 &&
  P36_VK_DENSE_IQ3_NX` (default on; `=0` restores the MMQ tile).
- **Kernel cost, real weights 5120x17408 x 68 blocks (the `attn_qkv` shape), 200
  iters, best-of:** two one-token dispatches **0.242 ms** vs one two-token
  dispatch **0.167 ms** = **1.44x**, i.e. 0.121 vs **0.084 ms/token = 0.70x the
  per-token cost**.  The audit's gate was a 2-token step at <= 1.3x a 1-token
  step; it is met with room.  Parity on the same call: `mismatches=0 max_abs=0`.
- **Against the path it replaces:** the MMQ 2-row tile is **2.086 ms**, 12.5x the
  nx dispatch and 8.6x two one-token decode dispatches — that is the whole P2
  collapse.  It is **not** bit-identical to nx (`bit-diff 10240/10240`), which
  matters below.
- **End-to-end, Swift, ctx 512, gen 256, 7 interleaved reps (median + MAD):**
  `--mtp-draft 3` nx OFF **4.220 t/s (MAD 0.020)** vs nx ON **6.860 t/s (MAD
  0.030) = 1.626x**; worst-case pairing (min ON / max OFF) still **1.569x**.
  This is the exact configuration P2 recorded at 4.17/4.44 t/s.  Profile at gen
  16: total kernel time 4022.954 -> 1960.724 ms, with 483 `nx` dispatches /
  54.049 ms replacing `dense_iq3_xxs_mmq` 1374.146 ms + `dense_iq3_xxs_mmq_pair`
  948.838 ms + 1210 `predequant_b16` dispatches.
- **`--mtp-draft 2` is neutral, and that is the parity proof.**  With draft 2 the
  verify itself is the 2-row step, so nx is on the verify path: 21.00/22.06/21.94
  t/s vs plain 21.88/22.86/21.90, and the d2 output is **byte-identical to MTP
  off**.  At draft 3 the verify is 3 rows and nx only fires on the accepted-prefix
  replay (`commit_n == 2`); d3 nx OFF and nx ON are byte-identical to each other
  over 64 greedy tokens, and both differ from MTP off for the pre-existing
  MTP-side reason (P2).
- **Known, and why it does not inflate the number.**  The MTP accept counters are
  *deterministic* and *different* between the arms — identical in all 7 reps per
  arm, OFF `calls=100 drafted=140 accepted=113 full=43 (80.7%)` vs ON
  `calls=97 drafted=120 accepted=91 full=31 (75.8%)`.  Cause: nx (f32 accumulate)
  and the MMQ tile it replaces (f16 accumulate, see the P3 entry) do not agree to
  the last bit, and the MTP gate compares a draft argmax against the target
  argmax, so the backoff trips on a different set of positions.  The direction
  matters: the faster arm accepts *less*, so 1.626x is a floor on the kernel's
  contribution, not a mix of a kernel win with a lucky acceptance pattern.
- **Verdict: landed, default on.**  It does not make MTP pay on its own — 6.86
  t/s against 21.9 t/s plain, because a 3-row verify still takes the MMQ tile and
  the replay is only 2 rows.  It removes the specific collapse P2 recorded, which
  is what makes the rest of P7 measurable at all.
- Honest limitations: (a) nx is bit-exact with the *one-token decode path*, not
  with the MMQ tile it replaces — anyone who needs the shipped d3 token stream
  byte-for-byte has to keep `P36_VK_DENSE_IQ3_NX=0`; (b) only `n_tok == 2` was
  built.  The kernel loops over `t0 < n_tok` in steps of 2, so 4/6/8 rows is a
  dispatch-condition change, deliberately not made (audit scope was 2..8; the
  measured target was the single 2-row step); (c) the guard is untouched by
  construction — the branch is P36_VK_TENSOR_IQ3_XXS-only and the guard's tensors
  are IQ2XXS, confirmed by `dense_iq3_xxs_decode_nx` never appearing in its
  profile.
- **Gates.** `karpathy/compat_gate.sh` **PASS** (Swift 1024/177.23 prefill tps +
  guard).  `./p36_test --vulkan-kernels` fails, but **on an attention test:**
  `tests/p36_test.c:4958` (`memcmp(out_host, out_single)` after replaying each
  batch row of `p36_gpu_attn_decode_tensor` as a decode step).  Reproduced on
  unmodified HEAD by stashing the whole change and rebuilding — **pre-existing,
  not caused by this kernel**, and out of P6's scope (`git stash` / `pop` used;
  the routed-gemm oracle lines `mv_max_rel=3.93391e-06 gemm_max_rel=0.00894794`
  are identical before and after).  Guard inertness measured, not argued: guard
  profile at ctx 512 / gen 32 runs 82.11 t/s with **0 `dense_iq3_xxs_decode_nx`
  dispatches** (IQ2XXS model, branch is IQ3_XXS-only).
- Raw: `evidence/raw/p6-nx2/` (`nx2-test.log`, `p6gate.py` + `p6gate.txt`, the 14
  `out9` logs, `prof-d3{off,on}.log`, the four greedy parity outputs, `run6.s`,
  `run9.s`, `run11.s` + `compat.log` / `vkern.log` / `guard-prof.log`).
- `reconsider_if`: MTP verify moves to 3+ rows (then build the `n_tok 2..8`
  dispatch, which is a condition change, not new kernel work); the flat `n_tok ==
  2` check is replaced by a general small-batch path; or an f32-accumulate MMQ
  replacement lands, which would make the d3 OFF/ON accept counters agree and let
  the 1.626x be quoted without the acceptance caveat.

## IQ2_M shared expert folded into the fused expert pair — premise measured, prize below the MoE bar, NOT built (2026-09-24)

Audit §P8 ("add the shared expert as a 9th slot in `moe_gate_up_decode` /
`moe_down_q2k_sum_decode` IQ2_S builds").  Not a rejection of an earlier
attempt: the ledger has no entry for the shared expert, and the audit's argument
was a *dispatch count* (`ffn_*_shexp` are IQ2_S so they miss the Q8_0-only fused
path at `p36.c:8278-8320` and fall through to `p36.c:8337-8361`, three matvecs +
swiglu + one mid `q8_k_quant`).  What was measured is that the dispatch count is
right and the cost model behind it is not.

- **Method.** `P36_VK_PROF=1 P36_VK_PROF_SHAPE=1` on IQ2_M, ctx 512, gen 128 —
  per-shape op rows, and `p36_vk_prof_iq3_shape` (`p36_vulkan.c:700-733`) names a
  `n_tok == 1` matvec `dense_<kind>_decode_<in_dim>x<out_dim>_n1`, so the shared
  expert's two shapes are identifiable: `2048x512` (gate and up, 2048 -> 512) and
  `512x2048` (down, 512 -> 2048).  Byte model: IQ2_S = 2.5625 bpw = 0.3203 B per
  weight.  Cross-check: the model predicts 40.3 GB/s for
  `moe_iq2s_down_sum_decode`, which reproduces P1's independently measured 40.6
  GB/s (`evidence/raw/p1-sumdecode/`).
- **Shared-expert matvecs are 4.55% of decode, at 60-80 GB/s:**
  `dense_iq2_s_decode_2048x512_n1` 11520 disp (90.0/tok, 2.25/layer) 48.565 ms,
  4.22 us/disp, 0.336 MB/disp -> **79.7 GB/s**;
  `dense_iq2_s_decode_512x2048_n1` 4864 disp (38.0/tok, 0.95/layer) 26.872 ms,
  5.52 us/disp -> **60.8 GB/s**.  Total **75.437 ms = 0.589 ms/tok = 4.55%** of
  the decode-row GPU total (1657.702 ms over 128 tokens = 12.951 ms/tok); ~5.1%
  including the swiglu and the mid `q8_k_quant`.
- **Why the fold does not capture the audit's number.**  Same model, same format,
  same run: `moe_iq2s_gate_up_decode` (8 experts x gate+up) runs at **151.0
  GB/s** — so there is real headroom (~2x) on the shexp bytes — but the shexp
  matvecs are **efficiency-bound, not dispatch-latency-bound**: 0.336 MB per
  dispatch in 4.2-5.5 us puts them in the same bytes-per-microsecond class as the
  kernels they would be folded into, and the audit's 128 disp/token is a
  consequence of the same small shape, not a separate overhead to delete.
  Realizable prize: 75.4 ms -> ~25-35 ms, i.e. **~0.3-0.4 ms/tok = 2.5-3% of
  IQ2_M decode**, at or under the audit's 3.4% MoE bar, for two new shader paths
  plus host wiring plus a new parity surface.  The 5 fewer dispatches per layer
  are irrelevant precisely because the path is not dispatch-bound.
- **Cheaper capture point identified, not taken.**  `p36_gpu_matmul_iq2s_pair_scaled_tensor`
  (`p36_vulkan.c:8429`) already pairs two IQ2_S projections off one Q8_K input,
  and is already used for `ssm_beta`/`ssm_alpha` (`p36.c:8722`).  The shexp
  gate/up is the identical call shape (two 2048->512 projections, one input
  quant) and would save ~1 dispatch/layer, ~1% — the same order as the fold's
  dispatch saving and a fraction of its cost.  Recorded as the first thing to try
  if anyone wants this prize; deliberately not in this pass.
- **Verdict: not built.**  The premise is confirmed (the path exists, is IQ2_S,
  misses the Q8_0 fast path, and is 128 of the ~713 decodes/token) but the
  prize is bounded at 2.5-3% of IQ2_M decode, under the audit's own MoE bar.
  Guard unchanged by construction (its `ffn_*_shexp` are Q5_K/Q6_K, so the IQ2_S
  branch cannot fire).
- Raw: `evidence/raw/p8-shexp/` (`p8-shape.log`, `p8probe.py`, `p8probe.txt`,
  `run10.s`).
- `reconsider_if`: the fold is bundled with a general fix for small
  single-projection IQ2_S matvecs (the 60-80 GB/s is a shared property of the
  `dense_iq2_s_decode_*` shapes, and both shexp legs plus the delta/ssm
  projections ride on it — that lever is worth more than the fold); or a target
  where 2.5-3% of IQ2_M decode decides something.

## GPU min-p prefilter for the default sampler — BUILT, +2.04% Swift decode, +8.21% MoE decode (2026-09-24)

Audit §P9(a). Not a reopening of a closed line, but it does finally *price* a
sentence that had been asserted for months: `AlreadyTried.md` "engine comparison"
claims p36's 23.19 t/s "carries the full temp-0.6 / top-k / top-p / min-p sampling
chain" against llama-bench's greedy `tg`, i.e. that the sampler costs something —
never measured. The two closed-line entries that touch readback
(`MOVNTDQA streaming loads`, `HOST_CACHED readback lever`) are about the readback
*buffer type*, not about the two scalar full-vocab passes, and are untouched here.

- **The premise was first read the other way and the reading was wrong.** The
  first pass argued "the filter touches the same 608 KB as the existing readback,
  same memory-traffic shape, nothing to win" and was headed for a negative
  write-up. Measured instead, on the CPU, on both sides of the argument:
  `sampler_bench3.c` reproduces `p36_sample_full_vocab`'s `top_p >= 1.0f` branch
  on real Swift logits — **full-vocab two loops 0.612 ms/token, survivor-only
  loops 0.000 ms, removable 0.612 ms (100%)**, of which the max+filter scan alone
  is 0.022 ms. `bw_bench.c` then streams the same 608 KB single-core: `sum` 0.131
  ms (4.6 GB/s), `max` 0.175 ms (3.5 GB/s), `store` 0.012 ms. So the two loops
  run at ~1 GB/s over a vector that a bare add loop moves at 4.6 GB/s: the cost is
  the scalar loop body (`isfinite`, subtract, divide, compare, `expf` per element
  over 151936 elements), not DRAM. Only the handful of survivors need an `expf`;
  everything else is a max and one comparison, which a GPU does in one pass.
- **Build.** New `vulkan/logits_minp_pack.comp` (109 lines, 256 lanes, three
  passes in one entry point): pass 0 per-block max over 256 contiguous logits
  with the Radeon order-preserving float->uint trick, pass 1 single-workgroup
  reduce, pass 2 filter `scaled > reject_scaled` + order-preserving compaction
  into fixed-capacity per-block buckets. The order-preserving uint is what lets
  the max travel without a shared-float reset between tokens, and the
  block-indexed buckets are what remove the global prefix sum. Host:
  `p36_session_sample_packed` (`p36.c:11932`) hooked into `p36_session_sample`
  before `p36_session_ensure_logits_host` (`p36.c:12043`), buffer
  `P36_GPU_MINP_BYTES(P36_N_VOCAB)` (80.8 KB at vocab 151936), gated by
  `P36_VK_MINP_PACK` (default ON).
- **Bit-identity is structural, not lucky.** `expf`, the sum and the draw stay on
  the CPU — a GPU `expf` is not the CPU's `expf` and one ulp in the sum moves the
  drawn token — and the packed list is emitted in ascending token order (blocks
  in index order, order-preserving compaction inside a block), which is exactly
  `p36_sample_full_vocab`'s scan order at `p36.c:10481-10499`. Confirmed
  byte-identical on four configurations, seed 42, 64 tokens each:
  Swift temp 0.8 / min_p 0.05 328/328 bytes, MoE 0.8 / 0.05 299/299, MoE temp
  1.0 / 0.05 333/333, Swift temp 0.8 / min_p 0.2 334/334, pack ON vs OFF.
- **Fallback is the correctness boundary, not the bucket size.** The packed path
  returns -1 — and the scalar path runs unchanged — on `top_k != 0`,
  `top_p < 1.0`, `min_p` outside `(0, 1]`, a non-Vulkan runtime, `have_reject`
  unfound, **any block whose recorded count exceeds `P36_GPU_MINP_CAP` (16)**, or
  zero survivors. Measured survivors on real logits: 1 at temp 0.8, 3 at temp
  1.0, so cap 16 is ~5x headroom; the overflow check means a wrong cap can only
  cost a fallback, never a wrong token.
- **A/B, decode-only change, judged on the decode median.** 7 interleaved reps,
  ctx 1400 Swift / 1024 MoE, `--temp 0.8 --min-p 0.05`, all 28 arms rc=0:
  Swift OFF 23.070 t/s (MAD 0.070) -> ON **23.540** (MAD 0.100) = **+2.04%**
  (43.346 -> 42.481 ms/token, worst rep pairing +1.37%);
  MoE OFF 79.340 t/s (MAD 0.190) -> ON **85.850** (MAD 0.160) = **+8.21%**
  (12.604 -> 11.648 ms/token, worst rep pairing +7.45%). Every rep moves the same
  way and the delta is 4-7x the MAD. The MoE win is larger than the 0.61 ms/token
  the CPU probe predicts because the packed path also skips the full 608 KB
  logits readback (`p36_session_ensure_logits_host` is not reached) and the
  sampler is on the critical path behind the last GPU flush.
- **Ceiling cross-check.** Sweep 13, greedy vs sampled, same prompt and ctx:
  Swift 23.74/23.45/23.50 greedy vs 23.21/23.02/23.13 sampled (-1.7%); MoE
  86.65/86.68/86.66 vs 79.70/79.47/79.38 (-8.3%). With the pack ON the sampled
  arms recover essentially the whole gap on Swift (23.54 vs greedy 23.5) and most
  of it on MoE (85.85 vs 86.67), which bounds what any further sampler work can
  still buy at ~0.2 ms/token Swift / ~0.8 ms/token MoE, `expf` and the draw
  included.
- **Prefill is not a gate here and is neutral:** medians 52.510 (OFF) / 51.710
  (ON) Swift, 125.820 / 128.860 MoE, with per-rep spread of 117-150 t/s on MoE
  (chunked prefilling). The change is downstream of the logits.
- **Gates:** `./karpathy/compat_gate.sh` PASS (including the CPU/GPU logits parity
  step, which never sees the sampler); `./p36_test --vulkan-kernels` still fails at
  `tests/p36_test.c:4958` (`attn decode n_tok invariance` memcmp) exactly as it does
  on unmodified HEAD — reproduced this pass, unchanged, out of scope. Dispatch
  confirmed in the profile: `logits_minp_max` 16 disp / `logits_minp_reduce` 16 disp
  for 16 generated tokens, 15520 groups, no extra flush.
- Raw: `evidence/raw/p9-minp/` (`ab-decode.txt`, `ab-reps.csv`, `parity.txt`, the
  8 parity logs, `prof-on.txt`, `cpu-probes.txt`, `sampler_bench3.c`, `bw_bench.c`,
  `vulkan-kernels.txt`).
- `reconsider_if`: someone wants the sampler off the host entirely (then the
  precision argument above is the thing to beat, not the speed); `P36_GPU_MINP_CAP`
  overflows on a real prompt family (the fallback makes this a no-op, but the cap
  is one line); or the top-k / top_p < 1 path is ever made the default, which this
  path deliberately does not cover.

## MTP loop (P7) — probe NEGATIVE: the 3-row verify does not exist, nx already serves every multi-row step (2026-09-24)

> **Corrected 2026-09-25 (round 2, R4 — see the entry below).** Two claims here
> are too strong and the numbers behind them are now measured directly.  (i) "nx
> already serves every multi-row step" holds only for IQ3_XXS: the 2-row verify
> still takes the 128-row MMQ tile for `dense_iq3_xxs_mmq_pair`, `dense_kquant_mmq`
> and `dense_iq4_xs_mmq`, which is why `verify` costs **391 ms for 2 rows vs 39 ms
> for 1 row** (nx OFF: 638 ms).  (ii) "the remaining d3 cost is structural, not a
> kernel" is wrong about the cost: the structure costs that because of those
> kernels - the kernel differential over the 7 verify cycles attributes
> +104/+70/+40/+30 ms per cycle to pair/kquant/iq4xs/residual-iq3 and only +10 ms
> to nx.  The shape conclusion (two target forwards per cycle; even with `verify`
> free, `fwd + commit` is 0.76x plain) stands and is now measured per phase.

Reopens **P2** (`## MTP draft acceptance measured, and the MTP loop is a net loss
as shipped`) and **corrects one sentence of P6** (`## Two-row dense IQ3_XXS
decode...`).  Audit §P7's cheapest listed fix is "use P6's kernel for verify",
and P6's own `reconsider_if` is "MTP verify moves to 3+ rows (then build the
`n_tok 2..8` dispatch, which is a condition change, not new kernel work)".  Both
rest on the same premise, and P6 states it outright: "at draft 3 the verify is 3
rows and nx only fires on the accepted-prefix replay (`commit_n == 2`)".  That
premise is false.

- **What the MTP loop actually dispatches.**  An env-gated diagnostic
  (`P36_VK_DBG_IQ3N`, kept — it is the instrument, not a semantic variant) prints
  every `dense_iq3_xxs` dispatch as `n_tok in out branch`.  Swift, ctx 512,
  `--mtp-margin 0`, whole run including the 512-token prefill:
  `--mtp-draft 3`: n_tok=1 (r4) 1530 disp, **n_tok=2 (nx) 161 disp**, n_tok=256
  (MMQ, prefill tile) 644 disp, and **nothing else**.
  `--mtp-draft 4`: n_tok=1 1144, n_tok=256 644, no n_tok=2 at all.
  `--mtp-draft 5`: same shape as draft 4.
  There is no 3-row, 4-row or 8-row IQ3_XXS step in any reachable MTP
  configuration.  The 2-row step P6's kernel serves *is* the verify; the
  "3-row verify" is a misreading of `draft_cap = N - 1` and does not appear on
  the wire.  Raw: `evidence/raw/p7-mtp/ntok-reached.txt`.
- **The condition change P6 named was made anyway, and measures inert.**
  `vulkan/dense_iq3_xxs_decode_nx.comp` now guards the partial tail chunk with
  `tn = min(NTOK, n_tok - t0)` (the per-row FMA chains are independent, so the
  rows a partial chunk *does* compute keep their exact operand order) and the
  dispatch at `p36_vulkan.c:8858` is `n_tok >= 2u && n_tok <= 8u` instead of
  `== 2u`.  `tests/test_dense_iq3xxs_nx.c` now takes `n_tok` and checks every row
  against its own one-token `dense_iq3_xxs_decode_r4` dispatch:
  **mismatches=0 max_abs=0 at n_tok 2, 3, 4 and 8** (real weights, attn_qkv shape
  5120x17408, 68 blocks).  Cost on the same call: 3 rows 0.344 ms vs the 3-row
  MMQ tile **2.089 ms (6.1x)**, 4 rows 0.347 vs 2.100, 8 rows 0.654 vs 2.103;
  the 3-row figure repeats at 0.339/0.344, but its *first* dispatch after
  process start costs 0.679, so a single cold sample reads 2.7x, not 6.1x;
  the MMQ arms differ in the last bits (bit-diff = every element, `max_abs` 121-170
  on random-byte weights), which is the f16-vs-f32 accumulation from P3 and is
  why P6 saw the accept counters drift.
- **End-to-end this changes nothing, exactly as the diagnostic predicts.**
  7 interleaved reps, Swift, ctx 512, gen 256, `--mtp-margin 0 --mtp-draft 3`:
  nx OFF **4.330 t/s (MAD 0.050)** -> nx ON **6.880 t/s (MAD 0.100)** = **+58.9%**,
  worst pairing +52.7% — against P6's **+62.6%** on the same configuration, i.e.
  the difference is run drift, not the widening.  The accept counters are
  bit-identical to P6's arms (OFF `100/140/113/43` = 80.7%, ON `97/120/91/31`
  = 75.8%, identical in all 7 reps), and all four greedy parity outputs
  (MTP-off, d2, d3 nx-OFF, d3 nx-ON; seed 42, 64 tokens, ctx 1400) are
  **byte-identical to P6's saved outputs**, including d2 vs P6's d2 — so the
  `n_tok == 2` path is unchanged and no token moved anywhere.  Evidence:
  `evidence/raw/p7-mtp/parity.txt`, `p7gate.txt`.
- **Attribution, with the audit's own profiling caveat applied.**  The prof
  `label=decode` block leaks prefill; subtracting a `--gen-tokens 1` run of the
  same ctx (both prefill 512) gives, per generated token, nx 4.40 -> 0.00 and
  `dense_iq3_xxs_decode` (r4) 26.35 -> 19.49, while **every IQ3_XXS MMQ row
  subtracts negative** (`dense_iq3_mmq` 322 disp ON vs 1127 OFF in the decode
  block, yet -83.7 ms/tok after subtraction) — consistent with the diagnostic
  showing MMQ only at n_tok=256.  Decode-block kernel totals 313.7 ms/tok (ON)
  and 475.4 (OFF) against 5805.3 ms for the prefill-mostly run, which is larger
  than either decode block and is why the subtraction is only good to ~20 ms per
  row.  Raw: `evidence/raw/p7-mtp/attribution.txt` + `p7attr.py`.
- **So the remaining d3 cost is structural, not a kernel.**  (The "not a kernel"
  half is corrected above: R4 measures the 2-row verify at 391 ms, and three of
  the four dense types pay the 128-row tile for it.)  After P6, a spec
  call is one draft forward (1 row) plus one verify forward (2 rows, already nx)
  and commits 1.94 tokens/call (97 calls, 91 accepted), i.e. roughly two forwards
  per 1.9 tokens: 6.88 t/s against 21.9 t/s plain.  The audit's other three P7
  fixes — draft argmax on the GPU instead of a 1 MB logits readback, a
  reduced-vocab draft head, per-verify-row DeltaNet state capture — all sit inside
  the draft head, which P2 measured at **+3.3 ms of a 47.5 ms one-token spec call
  (7.5%)**.  <=7.5% of a spec call cannot close a 3.2x gap to plain decode, and
  the per-row state capture only addresses the 29/120 rejected drafts, not the
  second forward.  Verdict: **not built**; the audit's P7 lever, as written,
  does not exist.
- **Gates:** `karpathy/compat_gate.sh` **PASS** (Swift 1024/177.96 prefill tps +
  guard).  `./p36_test --vulkan-kernels` fails at `tests/p36_test.c:4958`
  (`attn decode n_tok invariance` memcmp) exactly as on unmodified HEAD —
  reproduced this pass, unchanged, out of scope.  Guard untouched by
  construction (the branch is `P36_VK_TENSOR_IQ3_XXS`-only; the guard's tensors
  are IQ2XXS).
- Honest limitation: the widened dispatch range is **unreachable today**.  It is
  kept because it is the condition P6 recorded, it is covered by a bit-exact
  test at 3/4/8 rows, and it is strictly better than the MMQ tile if any future
  path does dispatch 3-8 rows — but do not count it as a P7 win.
- Raw: `evidence/raw/p7-mtp/` (`ntok-reached.txt`, `nx-rows.txt`, `parity.txt`,
  `p7gate.py` + `p7gate.txt`, `p7attr.py` + `attribution.txt`, the four parity
  outputs, `prof-d3{off,on}.log`, `prof-pre.txt`, `shape-d3on.log`, `run16.s`,
  `compat.log`, `vkern.log`).
- `reconsider_if`: the MTP loop starts carrying 3+ verify rows (the dispatch is
  already right and already bit-exact for them); or a depth-2 path appears that
  does *not* spend 1-4 draftless backoff tokens after each miss — P2 measured
  that as the reason only ~47% of calls carry a draft at `--mtp-margin 0`, and
  committed-tokens-per-forward has to come from there, not from the draft head;
  or a draft head that is a much larger share of a spec call than 7.5%.

## Flash-attention prefill vs the `n_tok`-invariance test — stale expectation, fixed the test, not the kernel (2026-09-25)

Closes the paragraph P7 left open ("`p36_test --vulkan-kernels` fails at
`tests/p36_test.c:4958` exactly as on unmodified HEAD — reproduced this pass,
unchanged, out of scope").  Round-2 R1.

- **Bisect.**  `git bisect run` between `8e8e788` (last commit CAMPAIGN §1c
  records as PASS) and `e6a442b`, harness = rebuild `p36_test`, exit 1 iff
  `p36_test.c:4958` appears in the log.  First bad commit: **`9d54dc7`**
  ("flash-attention prefill for GQA 8 MoE too").  Raw: `evidence/raw/r1-fa-gate/`
  (`bisect.log`, `bisect-check.sh`).
- **Which arm, and how far off.**  The failing assertion is `test_vulkan_attn_case`'s
  "a batch row must reproduce the per-row decode step bit for bit" memcmp.  A
  temporary diagnostic print (removed) pinned it to one shape: `pos0=129 n_tok=3`
  with the production KV pair `k=P36_KV_CACHE_Q8_0 / v=P36_KV_CACHE_Q4_0`, i.e.
  exactly the arm `attn_prefill_fa.comp` takes over.  FA on: **11137 of 12288
  floats differ, all by <= 1.02445e-08**, and that same batch output sits
  **1.18406e-08** from the f64 CPU reference the test already computes; FA off
  (qtile2): 0/12288 differ and **1.62379e-08** from the same reference.  So FA is
  *closer* to the reference than the kernel it replaced; the old assertion only
  held because qtile2 happened to reuse the decode reduction order for its V
  accumulation and FA's LDS-staged tile order does not.  Reassociation, not a bug.
- **Inert end to end.**  Greedy, temp 0, seed 42, 64 tokens, ctx 1400, FA on vs
  off: **guard (ratio 8) byte-identical**; Swift (ratio 6) diverges at token ~35.
  Frontier dumps at ctx 512/520 (chunk 256): guard `max_abs` 0.650/0.615, top-1
  same, top64 64/64 and 62/64; Swift `max_abs` 0.668/0.855, top-1 same, top64
  61/64 and 63/64.  A 5-arm x 2-run determinism control shows every arm
  reproducing its own token text exactly (only the timing line moves), so Swift's
  token-35 divergence is a real FA effect, not run noise — and it is inside the
  drift the FA landing itself accepted: `evidence/raw/fa-parity.txt` measures FA's
  teacher-forced NLL moving *less* than a plain prefill-chunk 256 -> 128 change
  (Swift 3.08 vs 3.58, guard 2.01 vs 2.60 mean |dNLL|), and at ctx 8192 the
  chunk-128 reference arm is the one that flips top-1, not FA.
- **Fix = make the test FA-aware.**  `test_vulkan_attn_case` gains an `n_head`
  parameter (ratio 6 and 8 are now both reachable) and a `fa_batch` predicate that
  mirrors the dispatcher (head_dim 256, K q8_0, V q4_0, ratio 6 or 8, FA not
  disabled; `p36_vk_use_attn_fa()` is static to `p36_vulkan.o` so the env default
  is repeated).  FA arms assert `drift <= 4.0e-5f` — ~4000x the measured drift,
  four orders inside the test's own 2e-3 quality bound, so a broken FA reduction
  still trips it — and print the drift.  Every other arm keeps the bitwise memcmp.
  Coverage widened with three production-KV arms: `129/3` and `2100/12` at ratio
  6, `2100/12` at ratio 8.  `attn_prefill_fa.comp`'s header now says outright that
  it is not bit-identical to `attn_decode_split/fused` and by how much.
- **Gates:** `./p36_test --vulkan-kernels` **exit 0, "vulkan-kernels: OK"** with FA
  on and with `P36_VK_ATTN_FA=0`; drift printed 1.02445e-08 at `129/3` and
  1.00117e-08 at `2100/12`, identical at both ratios.
  `karpathy/compat_gate.sh` **PASS** (Swift 1024/177.46 prefill tps).  No
  performance delta: test + comment only, no kernel touched.
- Raw: `evidence/raw/r1-fa-gate/` (`R1-VERDICT.md`, `bisect.log`,
  `bisect-check.sh`, `gate-fa-{on,off}.log`, `sweep.log`, `run-r1.s`, `run-ctl.s`,
  `e2e/` incl. `frontier-cmp.txt` and the 8 frontier dumps, `ctl/` 10 outputs).
- `reconsider_if`: the batch-vs-decode drift ever exceeds 4e-5 (then it is a
  reduction bug, not reassociation); or a real-task eval shows long-context
  regression that a prefill-chunk 256 -> 128 change does not also show.

## IQ2_S MoE down sum-decode wide-load rewrite — built, bit-exact, NEGATIVE (2026-09-25)

Reopens round-1 **P1** (`## MoE IQ2_S down sum-decode — audited levers NEGATIVE, load
shape CONFIRMED as the lever, not built`), whose `reconsider_if` is "someone builds
the coalesced fetch and it measures".  Round-2 R2.  It measures, and it is a no-op.

- **What was built.**  `vulkan/moe_down_q2k_sum_decode.comp` `P36_MOE_IQ2S` branch
  only: the per-element `wb_u8()` field reads are replaced by a staged LDS copy of
  the 82-byte superblock -- the 16 lanes of one `ix` half fetch its 41 words in three
  consecutive rounds (`k = itid; k < 41; k += 16`, so each round covers one contiguous
  32-byte span, third round lanes 0..8), `barrier()` around the refill, all field
  reads (`d`, `scb`, `entry`, `signs`) via LDS.  Element slotting, per-element fma
  order, the serial expert loop, `subgroupAdd(acc)` and the `tid == 0` weighted
  `fma32` combine are untouched, so the output is bit-exact by construction.
- **ISA confirms the intended change happened** (`isa-old.asm.txt` /
  `isa-wideload.asm.txt`): `buffer_load_ushort` 33 -> **3** (staging only), total
  `buffer_load_*` per workgroup-iteration 52 -> 22, `ds_read_b32` 0 -> 33,
  `ds_write_b32` 0 -> 3, LDS 0 -> 384 B, VALU 478 -> 476, hot basic block 549 -> 529
  instructions.
- **It measures nothing.**  Kernel, same 4736 dispatches / 9699328 groups:
  `gpu_ms` **316.383 -> 316.571** (+0.06%), 21.5% -> 21.6% of IQ2_M decode.  End to
  end (`tests/bench_ab.sh`, 7 interleaved reps, same binary, only the `.spv` swapped
  per run, IQ2_M ctx 1024 chunk 256 gen 128, plain decode): prefill median 617.06
  (MAD 8.30) -> 615.01 (MAD 7.36) = **-0.33%**; decode median **82.15 t/s
  (MAD 0.150) -> 82.32 t/s (MAD 0.290) = +0.21%** with overlapping MADs.  Parity is
  clean: frontier 512/520 dumps on IQ2_M and the guard, 2 reps per arm,
  `max_abs_diff = 0` / top-1 same / top64 64/64 in every pairing, and a shipped-vs-
  shipped re-run also 0.  Bit-exact and useless -> **rejected, reverted**; the
  variant survives only as `wideload-rejected.patch`.
- **Why, and where P1's attribution went wrong.**  P1 decomposed the kernel as
  `315.4 ms = 97.0 coalesced-load floor + 167.2 scattered field loads + 51.2 ALU` and
  named the 167.2 ms component the lever.  Deleting that component entirely changes
  the kernel by 0.06%, so it was miscounted: the field loads were never the stall.
  There is also no overfetch to fix -- the loop already moves only 99.4 MB/token of
  unique bytes (12.72 GB per 128-token run) in 316 ms = **40.6 GB/s**, the same bytes
  at the same rate after staging.  P1 probe D's 131 GB/s floor is real but a load-only
  kernel has no dependent consumers, so it does not follow that the shipped kernel
  waits on memory.  The kernel is **instruction-issue bound**: the hot basic block is
  549 instructions and runs once per expert per row (2048 rows x 8 experts = 16384
  times per dispatch), i.e. **9.0M warp-instructions for 8.39M weights = 1.07
  warp-instructions per weight**; against 40 CUs x 4 issue slots at ~1.5 GHz that is
  ~37.5 us of pure issue against 66.8 us measured (~56% of peak issue).  Only ~89 of
  the 549 instructions are arithmetic (24 cvt, 16 mul, 16 mac, 8 add, 16 cndmask,
  16 cmp, 1 fma_mix); ~348 are index/address math for the 16 weights one lane owns
  per block, i.e. **34 instructions per weight per lane, 63% index arithmetic**
  (`opmix-main-loop.txt`).  The lever is instructions per weight, not load width.
- **Not built, deliberately.**  The same restructure for the Q2_K branch (the guard's
  `moe_q2k_down_sum_decode`, 62 GB/s) and for `moe_iq2s_gate_up_decode` (~150 GB/s,
  and not a sum-decode kernel at all).  The premise is refuted on the kernel where it
  was strongest, so porting it buys the same nothing for the same work.
- **Gates:** `karpathy/compat_gate.sh` **PASS**; `./p36_test --vulkan-kernels` **OK**
  (R1 fix, `e77871d`).  Shipped blob restored byte-identically (`md5 2603136d…`).
- Raw: `evidence/raw/r2-iq2s-wideload/` (`R2-VERDICT.md`, `wideload-rejected.patch`,
  `isa-{old,wideload}.asm.txt`, `opmix-main-loop.txt`,
  `iq2s-down-{old,wideload}.spv`, `parity/` incl. `cmp.txt`, `ab/` incl.
  `ab-iq2m.csv` + `ab-iq2m.summary` + `prof-{old,new}.log`, `run-ab.s`, `parity.s`,
  `arm-{old,new}.sh`).
- `reconsider_if`: a variant that changes the **instruction count per weight** (index
  arithmetic or dequant extraction) rather than the load width; a mapping that gives
  one lane 64 contiguous weights instead of 16 interleaved; or evidence that
  occupancy rather than issue is the limiter (then the fix is more rows or waves per
  workgroup, not fewer loads).

## The P6 nx kernel's user-facing payoff: `--batched-session` is a throughput LOSS, and nx only covers 1 of 4 dense trunk types (2026-09-25)

Reopens round-1 **P6** (`## Two-row dense IQ3_XXS decode kernel — BUILT and
LANDED, +62.6% on MTP draft 3`) only to ask the question P6 never asked: the nx
kernel has exactly one remaining multi-row user, `--batched-session N`, and it
had never been measured.  Round-2 R3.  Measurement only, no source change.

- **Setup.**  Swift (`IQ3_XXS`, 24/4 heads), `p36-server --batched-session N
  --ctx 1024 --prefill-chunk 256`, 4 fixed greedy streaming clients (one prompt
  each, ~430 tokens) with 64 generated tokens per stream, `P36_VK_DENSE_IQ3_NX`
  0 vs 1, 2 interleaved reps (arm order flipped in rep 2).  `aggregate_tps`
  includes TTFT; `decode-only agg` excludes each stream's own TTFT.
- **nx pays on aggregate, and shrinks as N grows.**  N=1 **11.779 -> 11.677
  t/s = 0.991x** (inert, as expected: the kernel needs `n_tok >= 2`, and the nx1
  rep spread is 0.39 t/s); N=2 2.492 -> 3.719 = **+49.2%**; N=4 4.187 -> 5.774 =
  **+37.9%**; N=8 5.865 -> 7.569 = **+29.1%**.
- **The headline is that batching loses.**  Same binary, same prompts, same ctx:
  one stream decodes **23.22 t/s**; 8 concurrent streams aggregate **11.72 t/s**
  (nx ON) = **0.50x**; N=2 is 0.19x, N=4 0.33x.  A `--batched-session 8` server
  is a throughput loss against serialising the same eight requests.  nx moves
  this from 0.39x to 0.50x, i.e. it does not change the sign.
- **The batched step is a fixed 128-row-tile price, not a per-row cost.**  Server
  `decode batch count=N elapsed=… ms`: count 1 = 43 ms either way; count 2 =
  **725 -> 462 ms** with nx; 4 = 770 -> 511; 8 = 823 -> 617.  **Flat in the row
  count from 2 to 8** while a single-row step is 43 ms.  nx removes ~260 ms of
  that fixed price, not a per-row term.
- **Attribution (identical prefill, nx toggled):** `dense_iq3_xxs_mmq`
  `gpu_ms` **14192.131 -> 4517.989** and `dense_iq3_xxs_decode_nx` 0 ->
  **1430.415**; every other dense row is unchanged within noise
  (`dense_iq3_xxs_mmq_pair` 9811 -> 9729, `dense_kquant_mmq` 6312 -> 6273,
  `dense_iq4_xs_mmq` 3806 -> 3787).  The audit's prefill-leak caveat is handled
  by using only the differential: -8244 ms over the 32 decode steps of that arm
  = **-258 ms/step**, which reproduces the server medians exactly.
- **nx covers one of the four dense trunk types.**  In the nx-ON arm:
  `dense_iq3_xxs_mmq_pair` 1927 disp / 9729 ms, `dense_kquant_mmq` 4174 / 6273,
  `dense_iq4_xs_mmq` 1804 / 3787, residual `dense_iq3_xxs_mmq` 1288 / 4518,
  `dense_iq3_xxs_decode_nx` 5313 / 1430.  Of the dense types still taking the
  128-row tile at n_tok 2..8 no single one dominates (pair 27%, kquant 18%,
  iq4xs 11%, residual iq3 13%) - the *class* does (69%).  This is the same
  kernel set that makes R4's two-row MTP verify cost 391 ms.
- **Probe: the tile is not a defect a naive fallback fixes.**
  `P36_VK_DENSE_KQUANT_MMQ=0` (the one switch that reroutes `n_tok > 1` K-quant
  work to the generic 8x8 `matmul_kquant`) at N=4, nx ON: **4.228/4.201 t/s vs
  5.743/5.744 = -26%**, TTFT 23.4/26.6 s vs 9.7 s (the flag also drops prefill
  off its tile).  The only thing that ever beat the tile for this shape is a
  purpose-built `n_tok 2..8` kernel (4.7x for IQ3_XXS).  So the deferred lever is
  an nx-style variant for `dense_iq3_xxs_mmq_pair` (largest remaining tile,
  27% of the batched step), `dense_kquant_mmq` and `dense_iq4_xs_mmq` -
  **not built this round**, and R4 shows the MTP restructure is blocked on it.
- Raw: `evidence/raw/r3-batched-nx/` (`R3-VERDICT.md`, `R3-ANALYSIS.txt`,
  `results.jsonl`, `results-kmq.jsonl`, `results-prefillonly.jsonl`, `srv/`,
  `prof/`, `scripts/` incl. `run-r3.s`, `run-r3b.s`, `batchbench.py`,
  `analyze.py`).
- `reconsider_if`: an nx-style kernel exists for the pair/K-quant/IQ4_XS tiles
  (then re-run this sweep - the fixed 260 ms is only part of the 680-780 ms
  fixed price); or a serving workload that must hold N sessions resident for
  reasons other than throughput (latency spread, KV memory), in which case the
  0.50x is a documented cost of that choice and not a defect.

## MTP loop diagnosed: two target forwards per cycle, `verify` is 10x a one-row forward, net loss (2026-09-25)

Corrects round-1 **P7** (`## MTP loop (P7) — probe NEGATIVE: the 3-row verify does
not exist, nx already serves every multi-row step`), which concluded "the
remaining d3 cost is structural, not a kernel".  That conclusion was right about
the *shape* and wrong about the *cost*: the shape costs 391 ms per cycle because
the two-row verify is **not** served by nx for three of the four dense trunk
types.  Round-2 R4.

- **Diagnostic added (env-gated, kept - it is the instrument).**
  `P36_MTP_TIMING` prints one line per speculative cycle from
  `p36_session_eval_speculative_argmax` (`fwd`, each draft, `snap`, `verify`,
  `commit`, total, `read_kb`, `drains`), plus `p36_gpu_read_bytes()` /
  `p36_gpu_read_flushes()` counters in `p36_vulkan.c` / `p36_gpu.h`.  No
  semantic change; inert when unset.
- **Absolute rates, same binary, Swift ctx 512 margin 0, plain alongside:**
  plain 32 tok **22.68 t/s**, plain 256 tok **23.04**; `--mtp-draft 2`
  **21.17 (0.92x)**; `--mtp-draft 3` nx ON **6.97 (0.30x)**, nx OFF
  **4.10 (0.18x)**.  Round-1's 6.88 reproduces.  Both depths are net losses.
- **Per cycle (97 cycles, d3 nx ON, 1.94 tokens/cycle, accept 75.8%, exits 37
  gate-reject / 29 partial / 31 full-accept):** `fwd` 51.70 ms median,
  `draft` 4.84, `snap` 0.13, **`verify` 391.03** (max 482.21), `commit` 0.01
  (max 51.55 on a partial accept = restore + replay), total 462.69.  `read_kb`
  median 4850 = five 993 KB logits readbacks per full cycle; `drains` 3.
  `submit_wait_ms` **563.0 d3 vs 167.4 plain** = +57 ms/cycle of queue drain on
  top of the GPU work.
- **`verify` is 391 ms for 2 rows against ~39 ms for 1 row = 10x** (nx OFF: 638
  ms).  Kernel differential over the 7 verify cycles: `dense_iq3_xxs_mmq_pair`
  +104 ms/cycle, `dense_kquant_mmq` +70, `dense_iq4_xs_mmq` +40, residual
  `dense_iq3_xxs_mmq` +30, `dense_iq3_xxs_decode_nx` +10 (the one type nx
  covers), `attn_prefill_fa_gqa6` +3, lm_head `dense_q5k_decode` +5.  So the
  two-row verify leaves the decode kernels for the **128-row tile** on three of
  four types - R3's wall, in the MTP path.
- **The Q8_0 host-drain suspicion is refuted.**  The MTP block's matmuls show up
  as `matmul_q8_0_f32b` (0 -> 88 disp / 14.2 ms); `submit_wait_q8_0_quant_x`
  never appears in any profile.  The draft head is 5 ms; `mtp` inside `fwd` is
  12 ms.  P2's 7.5% figure held.
- **d2 is not speculation at all.**  `draft_cap = N - 1 = 1` makes
  `verify_n == commit_n == 1`, so the `row_tops` comparison loop never runs and
  the 100% accept counter is **by construction**.  d2 = one extra forward per
  token plus a wasted draft head = its 0.92x.
- **Ceiling.**  Measured d3 cycle 466-531 ms for 1.71 tokens = **273-310
  ms/token = 0.15x plain**; even with `verify` free, `fwd + commit` = 57
  ms/token = **0.76x plain**, because the shipped shape runs two target forwards
  per cycle plus a replay on every partial accept.  The standard shape (one
  forward over [committed, draft...], per-row argmax on GPU, adopt-on-partial
  accept so there is no replay) is `52 + 12 + 5 = 69 ms` per 1.71 tokens =
  **25 t/s = +9% over plain** (+17% at round-1's 76% accept) - and that `52 ms`
  assumes the n_tok 2..8 trunk covers pair/kquant/iq4xs.  **Verdict: diagnosed,
  reported as a net loss, no restructure landed.**  The restructure is a strict
  dependency on the deferred R3 kernel work for ~+10%.
- Raw: `evidence/raw/r4-mtp-diagnosis/` (`R4-VERDICT.md`, `R4-PHASES.txt`,
  `plain32.log`, `plain256.log`, `mtp2-32.log`, `mtp3-32.log`, `mtp3-256.log`,
  `mtp3-32-nxoff.log`, `prof-{plain,mtp2,mtp3,mtp3-nxoff}.log`, `run-r3c-r4.s`,
  `run-r4b.s`).
- `reconsider_if`: the n_tok 2..8 nx trunk covers the pair/K-quant/IQ4_XS tiles
  (then build the standard shape - the +9% is real once the 725 ms step becomes
  52 ms); or a per-verify-row DeltaNet state capture lands, which removes the
  `commit` replay (max 51.55 ms/cycle) but not the second forward.

## Long-context decode attention (R5) — diagnosed: instruction-bound tile loop, combine-cost model, grid knee bracketed (2026-09-25)

Takes up round-1 **P5**'s open end ("16K/32K were not re-run because
decode-after-long-prefill swings 20-32 t/s with an identical binary").  That
swing is why this pass reports **per-dispatch GPU time and dispatch counts**, not
end-to-end tok/s: `attn_decode_split` runs exactly once per full-attention layer
per *decode* token (prefill uses `attn_prefill_fa_gqa{6,8}`), so
`split_disp / gen_tokens` gives the layer count (16 Swift, 10 guard) and the
split's ms/token needs no prefill subtraction.  `attn_combine` carries
`2 x chunks x L` prefill dispatches plus `1 x L` per decode token, so its decode
share is recovered from the combine/split dispatch ratio.  Round-2 R5.

- **Setup.**  Fresh process per (model, ctx), Swift and the guard at ctx
  8192/16384/32768 (2/2/3 reps), `--prefill-chunk 256 --gen-tokens 64
  --mtp-margin 0`, `P36_VK_PROF_OP=1 P36_VK_PROF_KERNEL=1`.  Stages B/C sweep
  `P36_VK_ATTN_SPAN` 128/256/1024/2048/4096 at ctx 16384, 1 rep each (512 is
  stage A's 2 reps).
- **Reproducibility.**  0.3-1.4% on `split_ms/call` everywhere except Swift at
  ctx 32768, which disagrees with itself by 7% (1.0221 vs 1.0943 ms/call, 15.40
  vs 14.63 t/s) - the enter-temperature caution, localized to the longest arm;
  the attention *share* is stable there anyway (27.7 vs 28.1%).  Per-span split
  cost 0.01675 / 0.01616 / 0.01572 ms at 8K/16K/32K, i.e. the 32K arm is the
  *fastest* per key, so the long wall time is the profiler's flush count
  (`submit_eager` 4904 / 7568 / 15696), not throttling.
- **Grid: the knee is bracketed, and 512 is on the good side of it.**  At fixed
  span 512 the split costs a constant 25-28 us of CU time per workgroup while
  the count grows 408 -> 792 -> 1560 (1.94x, 1.97x Swift) and 272 -> 528 -> 1040
  (guard), so 10 workgroups per CU (4 waves each) already fills the board at 8K.
  The sweep says the same from both sides: **widening** 512 -> 1024 -> 2048 ->
  4096 costs +7.1 / +28.1 / +70.5% per split call (Swift) and +12.3 / +39.4 /
  +96.5% (guard), per-key rate 1.32 -> 1.85 ns / 1.33 -> 2.15 ns; **narrowing**
  to 256/128 saves 4.2 / 7.0% of the split on Swift (per-key 1.315 -> 1.280 ->
  1.252 ns) and 9.9% on the guard at 128, but the **guard's 256 arm is an
  outlier** (+4.0% where the 128/512 trend puts it at ~-3%).  The narrow side is
  closed anyway: single-rep, not bit-exact, and it **fails the kernel gate** -
  `P36_VK_ATTN_SPAN=128 ./p36_test --vulkan-kernels` gives 3 failures and `=256`
  gives 1, all at `tests/p36_test.c:4988` (bitwise batch-vs-step-replay), 0 at
  512, reproducibly.  The failing arms are the 132-key ones: single-span (fused)
  at 512, 2 spans at 128 - so the split path is not `n_tok`-invariant for that
  shape, and 512 is the setting that property is verified at.  FA arms unaffected
  (batch drift 9.1e-09 at 128/256 vs 1.0e-08 at 512).  Round-1 P5's "occupancy-bound" reading of the
  80-workgroup ctx-2048 case does not survive at 8K+.
- **Occupancy probe (compile-free, `mmq_info`):**
  `attn_decode_split.spv` = SGPRs 108, **VGPRs 48**, LDS 1536 B, Spilled 0,
  **Subgroups per SIMD 20** (the driver's own ceiling); `attn_combine.spv` =
  SGPRs 108, VGPRs 8, LDS 0, code 524 B, subgroups/SIMD 40.  Nothing about the
  256-thread, 4-wave workgroup is register- or LDS-limited.
- **Not bandwidth.**  Requested bytes (every query head reads its own copy) are
  6x (Swift) / 8x (guard) the unique bytes, yet the requested rate is a flat
  **249-310 GB/s** across both models and all three contexts while the unique
  rate is 31-49 GB/s and *differs by model* at the same ctx (Swift 44-49, guard
  31-39).  A DRAM-bound kernel would have to show the same unique rate for both;
  a request-bound one would show the 6-8x duplicate reads dominating.  P5 already
  showed the re-read is an L2 effect (GQA grouping buys only 1.01-1.14x past 8k).
- **What does: per-element instructions in the tile loop - the R2 lever again.**
  Hand count of the live fast path (K q8_0 / V q4_0, TILE 64, 256 threads) per
  key per workgroup: K `4 quads x k_q8_dot_pair` ~580 (word-wide qs, 8 dwords +
  8 shared reads per 64 dims), V `256 x v_q4` ~2048 (per element: a dword load
  for the block scale, a dword load for one nibble, shift, and, int->float, mul,
  mad, add - `v_q4` at `vulkan/attn_decode_split.comp:158`), lane scan+exp ~1500
  (serial `tmax` over up to 64 keys per thread per tile, 13 `barrier()`s).
  4128 thread-instr = 64.5 wave-instructions per key = 32.2 cycles of CU issue
  (wave64 fp32, 4-SIMD CU) against **52.6 ns of CU time measured per key per
  workgroup** (40 CU x 0.5334 ms / (792 x 512) at Swift 16384) = 70 cycles at
  1.34 GHz, 100 at 1.9 GHz: the kernel runs its instruction stream at
  **2.2-3.1x its own issue bound**, i.e. issue/stall-bound, not a dependency hole
  (a latency-bound gather of this shape sits ~10x off).  ~86% of that stream is
  V extraction plus the lane scan; the K side already uses the wide-load form
  `k_q8_dot_pair` that the V side does not.
- **`attn_combine` has a clean cost model and it is what punishes narrowing.**
  24 (Swift) / 16 (guard) workgroups - under one per CU - 524 B of code, serial
  dependent-load loop (3 loads per span, no vector width, `exp`/`v_rcp_f32` per
  step).  Per call over 5..129 spans: **Swift `0.06037 + 1.02e-4 x spans` ms**,
  **guard `0.04006 + 1.03e-4 x spans` ms** (max residual 0.00015 / 0.00030 ms,
  n=7): a fixed cost that scales with the head count plus ~1 us per partial -
  the stock comment's prediction, now quantified.  At span 128 that is +15% /
  +22% over span 512, which is why the narrow side nets out small.  The whole
  term is 1.8-3.7% of the decode token.
- **Absolute share of decode** (same binary, plain decode, gen 64):

  | model | ctx | decode | split/tok | comb/tok | attn | share |
  |-------|-----|--------|-----------|----------|------|-------|
  | Swift | 8192 | 49.6 ms (20.15 t/s) | 4.56 | 0.63 | 5.19 | **10.4%** |
  | Swift | 16384 | 54.1 ms (18.48 t/s) | 8.53 | 1.02 | 9.55 | **17.7%** |
  | Swift | 32768 | 66.6 ms (15.40/14.63 t/s) | 16.35 | 1.67 | 18.0 | **27.7-28.1%** |
  | guard | 8192 | 13.8 ms (72.39 t/s) | 2.03 | 0.27 | 2.30 | **16.6%** |
  | guard | 16384 | 15.6 ms (64.09 t/s) | 3.59 | 0.44 | 4.03 | **25.8%** |
  | guard | 32768 | 19.1 ms (52.26 t/s) | 6.50 | 0.70 | 7.20 | **37.9-38.1%** |

  From 8K to 32K the Swift token grows 49.8 -> 66.6 ms and attention grows
  5.19 -> 18.6 = **80% of the slowdown**; guard 13.7 -> 19.1 ms, attention
  2.30 -> 7.25 = **92%**.  Prefill tok/s falls with ctx too (Swift 138.85 /
  126.40 / 111.19, guard 569.39 / 488.47 / 391.98), so the long-context penalty
  is the same kernel twice.  Round-1's "guard ctx 8192: 3.0 ms/tok for ~68 MB"
  reproduces as 2.03 ms/tok for 63.4 MB unique (31.1 GB/s).
- **Verdict: diagnosed, no lever landed.**  Not latency-, occupancy-, grid- or
  bandwidth-limited; a per-element instruction stream (V gathered one nibble at a
  time + serial per-key lane scan) issued at 2.2-3.1x its minimum, whose total
  cost is keys x that per-key rate.  The only cheap experiment the item allowed
  (the host-tunable `span_keys`) is worth **at most ~1% of the decode token on
  Swift and ~2-3% on the guard** on the narrow side, single-rep, in exchange for
  a non-bit-exact regrouping of the softmax reduction that also fails the
  `n_tok`-invariance kernel check - so the default stays 512.
  The rewrite the accounting names - vectorise the V extraction as
  `k_q8_dot_pair` already does for K, hoist `tmax` - is the same class R2
  **rejected** on `moe_iq2s_down_sum_decode` (fewer/wider loads bought 0.06% on
  an instruction-issue-bound kernel), so it is not proposed.  The stale
  "narrower buys occupancy" claim in the `p36_vk_attn_span()` comment
  (`p36_vulkan.c:1282`) is corrected in place with these numbers.
- `P36_VK_ATTN_SPAN` is a **diagnostic switch, not a semantic variant**: it is
  not bit-exact, and any use needs the NLL check rather than the byte-identical
  one (the drift grows with the span count, which is also why 1-rep evidence is
  not enough to move the default).
- Raw: `evidence/raw/r5-longctx-attn/` (`R5-VERDICT.md`, `out/`, `outB/`,
  `diag/`, `scripts/`, `ANALYSIS-stage{A,B}.txt`, `thermal-samples.txt`).
- `reconsider_if`: an interleaved A/B plus an NLL check for span 256/128 at ctx
  16384 and 32768 (pays ~1-3% of the token, and only while attention stays this
  share of decode); or a V-side vectorised gather that keeps the per-key FMA
  order and beats 52.6 -> <45 ns per key per workgroup at ctx 16384 before being
  tested at ctx 32768.

## Swift-1.5 GSQ-RCO IQ3_S: IQ1_M prefill onto the dense_extra mmq tile — ACCEPTED, +3.4-5% prefill (2026-09-26)

`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` is a per-tensor mixed quant (IQ3_S,
IQ4_XS, IQ3_XXS, Q4_K, IQ2_S, Q2_K, IQ2_XS, IQ2_XXS, and one IQ1_M tensor).
Every type already reached a tuned kernel except IQ1_M. Its one tensor
(`blk.13.ffn_gate`, 0.16% of the weights) cost **3.5% of GPU time** on the naive
`dense_iq1_m` kernel, which dispatches one workgroup per (row, token).

- **Change.** An IQ1_M arm in the generic `dense_extra_mmq.comp` main. It uses
  the IQ1_S grid and mapping, with the block scale assembled from the scale-word
  nibbles. IQ1_M routes there only for `n_tok > 8`, not `--quality`, and not
  `P36_VK_DENSE_IQ1M_MMQ=0`. `dense_extra_mmq` joins the prewarm list.
- **Correctness.** `tests/test_dense_iq1m_mmq`: rms_rel 0.0104, equal to the shipped
  IQ1_S arm's f16 floor. A fault-injected delta flip gives 0.32. With the route
  off, Swift logits are byte-identical to the pre-change build. KL vs the f32
  per-token reference is 0.0019 -> 0.0031 / 0.0007 -> 0.0014 nats, with top-1 equal.
- **No-breakage.** Only `dense_extra_mmq.spv` changed. VGPR/LDS/spills are identical
  (88/11776/0). Only Penjing-27B-IQ2_XXS and the two GSQ-RCO files use it.
  Penjing logits are byte-identical.
- **Speed.** Route off vs on, 7 reps ctx 1024: prefill 168.54 -> 176.98
  (+5.01%, no overlap). Reported as +3.4-5% because the gain exceeds the
  profiled cost of the old kernel; that gap is unexplained. At 64 generated
  tokens decode is +0.54% (flat).
- **Mistakes caught by the adversarial review, kept here so they are not
  repeated.** (1) The P3 frontier-NLL recipe with `p36 --dump-tokens` targets
  gave 6-27 nat NLLs on confident predictions. The targets are misaligned, so do
  not use it for dNLL; use KL against a `--prefill-chunk 1` reference instead.
  (2) `env "A=x B=y" cmd` sets one variable; that invalidated a gate-off run.
  (3) `bench_ab.sh` inherits the cwd, and `shader_root` = cwd wins over the exe
  dir, so an A snapshot needs `P36_SHADER_DIR` in a wrapper.
- `compat_gate.sh`: infra timeout (exit 124) in the MoE CPU capture from the
  external HDD. Not applicable, since neither gate model has an affected tensor.
  Not a pass.
- Raw: `evidence/raw/swift15/` (`VERDICT.md`).
- `reconsider_if`: an IQ1_M `dense_extra_decode` arm (the decode share of the old
  kernel was not measured), or a mixed-type gate/up pair kernel. W6's pair covers
  only 12/64 layers on this file, and the same-type IQ3_S/IQ4_XS pairs (12 more
  layers) would be worth ~0.5-1% on paper, below the gate.

## VeriLoop-E2-GSQ-RCO-IQ3_S decode rounds 1-2 — NO LEVER, kernels at practical bandwidth limit (2026-10-15)
Baseline `a565aff`, BC-250, ctx 128 / 128 gen: 22.2 tok/s (ctx 4096: 20.35).  Swapped-.spv A/B, interleaved, 2 reps each:
- **Round 1 (all REJECTED):** IQ3_XXS LDS float-grid + 16/128-entry sign LUT 19.4 / 17.1 (-12% / -23%); IQ3_S LDS grid + sign LUT 18.7 (-16%), + u16 loads 18.7; IQ4_XS uvec2 loads 22.1 and byte-pair LDS LUT 22.1 (flat); Q4_K lane-pair single-dword mapping 21.0 (-5%, closes the unbuilt item in the dot4 entry above).  LDS tables lose on wave64 gfx1013.
- **Occupancy (`mmq_info`):** iq3_s r4 64 VGPR / 16 subgroups per SIMD, iq4_xs 64 / 16, iq3_xxs r4 64 / 16, q4k 56 / 18; 0 spills everywhere.  Not occupancy- or spill-limited.
- **IQ3_S ablation:** all grid/sign/fma math removed, every load kept: 22.2 -> 23.5 tok/s (+5.7%).  u16 loads replacing the 4x byte `load_u32` alone: 22.1 (flat).  The kernel is not ALU- or load-instruction-bound.
- **Achieved GB/s (GGUF bytes / profiled ms per tok):** IQ4_XS 362 (8.26 ms), IQ3_S 311 (11.29), IQ3_XXS 302 (6.82), Q4_K 279 (6.49), vs ~373 loads-only floor.  Best case all three laggards at floor saves ~4.4 ms of ~45 = ~24 tok/s, so 25 tok/s is out of reach by dominant-kernel work.
- **Fusion:** summed kernel GPU time (~49 ms with timestamps) is >= wall (45 ms): no CPU/submit gap to recover.
- `reconsider_if`: a repacked weight layout (changes the GGUF, out of scope) or hardware with more bandwidth.

## APEX-I MoE decode — type-specialized matvec + Q4_K packed loads — REJECTED, NEUTRAL (2026-10-23)

- **Target.** APEX-I-NanoPlus (MiniCPM-V-4.7-35B-A3B, qwen35moe, Q4_K experts) and
  APEX-I-MiniPlus (KAT-Coder-V2.5, Q5_K/Q3_K experts). Goal was to close a ~27%
  decode gap vs llama.cpp on these files.
- **Candidate 1 — type-specialized matvec.** `moe_matvec{,_fast}.comp` are one
  shader each with a runtime `if (pc.type == ...)` chain over nine ggml types.
  Built six `-DP36_TYPE_ONLY` variants and selected them in
  `p36_vk_moe_type_kernel()` at all three dispatch sites. Required a second pass:
  `P36_VK_MOE_FAST` defaults ON, so the first version only specialized the dead
  `moe_matvec` and was null by construction (CAMPAIGN §4 rule 5).
- **Candidate 2 — Q4_K packed loads.** The four consecutive Q4_K bytes one lane
  reads are a single aligned word (`wb[off>>2]`), as are its four q8_K
  activations: 8 loads collapsed to 2 per group.
- **Correctness.** Candidate 2 was **bit-identical**: 64 greedy tokens, md5
  `4915110d…` before and after (stash → rebuild → compare → pop).
- **Result.** Decode medians 71.8 t/s before, 71.8–71.9 t/s after, both candidates.
  No prefill change. 3 reps, below the §4 7-rep gate — recorded as observational.
- **Why it cannot work here.** Decode is **disk-read-bound**, not ALU-bound:
  p36 reads 10.3 MiB/token against a 745 MiB/s NVMe stream, and
  `bytes_per_token / disk_rate` accounts for the whole per-token time.
  llama.cpp shows the identical pattern at 7.9 MiB/token / 96.6 t/s. Speed ratio
  1.35 ≈ bytes ratio 1.30. See `evidence/decode-disk-bound-20261023.md`.
- **Disposition.** Both reverted, tree clean, baseline 71.47 t/s.
- **Reconsider if** the weights become resident (pinning, or a box with enough
  RAM for model + working set). Then decode is compute-bound and these levers
  become visible — re-rank them, do not assume they are still neutral.

## --ssd-streaming on the APEX-I MoE — REJECTED, 22× SLOWER (2026-10-23)

- **Tried.** `--ssd-streaming --ssd-streaming-cache-experts 8GB` on
  APEX-I-NanoPlus, hoping an explicit LRU cache in RAM would beat kernel
  page-cache thrash.
- **Result.** decode **3.18 t/s** vs **71.26 t/s** baseline. Cache held 796 of
  6853 experts, hit rate 97.8% (36068/36864), 796 loads, 0 evictions — the
  streaming path itself is ~22× slower per token regardless of hit rate.
- **Why it cannot win here.** Streaming swaps zero-copy mmap reads for staged
  copies through the cache buffer, and the per-load path dominates at these
  expert sizes. Page-cache thrash costs ~10 MiB/token; streaming costs far more.
- **Disposition.** Not a default, not a suggestion for this model set.
- **Reconsider if** the streaming path is rewritten to fuse the copy into the
  matvec, or if a model is used where the expert working set exceeds RAM by an
  order of magnitude.

## dense_q4k_decode — 8 rows per workgroup — REJECTED, 45% SLOWER (2026-10-23)

- **Tried.** `ROWS = 8` in `vulkan/dense_kquant_decode.comp` under
  `P36_Q4K_ONLY`, with the host dispatch changed to `(out_dim + 7) / 8` groups
  and a matching `out_dim % 8` guard. Motivation: `dense_q4k_decode` is 26% of
  decode GPU time and 225 matvecs per token.
- **Result.** Kernel cost **3.13 -> 4.55 ms/token (+45%)** measured by dispatch
  timestamps, over a 10-token window vs the 8-token baseline. Dispatch count per
  token is unchanged (225): `ROWS` sets rows per workgroup, not the number of
  matvec calls, so the launch-overhead saving the change was aiming at does not
  exist.
- **Why it cannot win.** Halving the group count halves the wave count on a
  small CU count while `acc[ROWS]` plus the four `vec4` activation registers
  push register pressure up. Fewer, fatter workgroups lose here.
- **Disposition.** Reverted; tree clean; baseline 71.83 t/s.
- **Reconsider if** occupancy data shows the 4-row kernel is wave-starved, in
  which case the answer is more parallelism, not fewer workgroups.

## Deferred top2 readback (sampling sync) — REJECTED, 2.1% SLOWER (2026-10-23)

- **Tried.** Decode showed a 630 us device gap per token between top2.spv and
  logits_minp_pack.spv, i.e. two full GPU->CPU drains per token (top2, then the
  packed min-p list). Made the top2 readback lazy: the forward only sets
  gpu_top2_pending, and p36_session_gpu_top2_resolve() reads it on demand from
  argmax, argmax_excluding and token_rank.
- **Result.** The sync gap collapsed 630 us -> 1 us per token, output stayed
  bit-identical (greedy md5 55f555660f6158501394ff16bde68faa), and the decode
  window went 14.43 -> 14.36 ms/token. End-to-end, however, five interleaved
  pairs all regressed: baseline 71.50/71.92/71.85/71.72/72.09 (median 71.85) vs
  lazy 70.37/70.86/70.16/69.84/70.55 (median 70.37), -2.1%.
- **Why it cannot win.** The 630 us of device-idle time was never on the critical
  path: the CPU is busy there doing the sampling work it has to do anyway. Moving
  the drain later just delayed the resubmission, so the GPU got its next command
  buffer later and the token took longer.
- **Lesson.** Device-timeline gaps are not automatically recoverable time. Only
  end-to-end interleaved A/B decides, as CAMPAIGN section 4 says.
- **Disposition.** Reverted; tree clean; baseline 71.77 t/s.

## moe_down_q2k_sum_decode_iq3xxs — aligned aux load — ACCEPTED, +4.9% decode (2026-10-23)

- **Tried.** The IQ3_XXS inner loop rebuilt its 32-bit aux word with four
  wb_u8() calls (each a 16-bit load plus shift and mask) per 256-element group.
  The word sits at an even offset, so one aligned wb_u32() replaces them.
- **Result.** Kernel time 2.91 -> 2.30 ms/token (-21%), decode span
  14.36 -> 13.70 ms/token, greedy output bit-identical
  (md5 55f555660f6158501394ff16bde68faa). End-to-end, five interleaved pairs:
  baseline 72.00/71.67/72.06/71.99/71.76 (median 71.99) vs 75.31/74.89/75.61/
  75.68/75.55 (median 75.55), **+4.9%**, ranges disjoint.
- **Why it works.** The loop is byte-unpack bound, and this was the last place
  still fetching one byte at a time from a 16-bit buffer.
- **Scope.** The wb_u32 helper only replaces the byte assembly; no arithmetic
  changed, which is why the token stream is unchanged.

## moe_down_q2k_sum_decode_iq3xxs — cache the aux word across the p-pair — ACCEPTED (2026-10-23)

- **Tried.** In the same inner loop, the 32-bit aux word was reloaded for p and
  p+1 even though n advances by 16, so two consecutive p share one g32 group and
  therefore one aux word. Cache it in caux/cg32 and reload only when g32 changes.
- **Note on the entry byte.** The same idea does *not* extend to
  wb_u8(bb + 2u + 8u*g32 + (m>>2u)): the two p of a group have m and m+16, so they
  read bytes exactly 4 apart inside the same 8-byte span. A word read replaces one
  byte load, not two, so there is nothing to coalesce there.
- **Result.** Kernel 2.30 -> 2.19 ms/token. Decode 76.38/76.07/76.50 t/s vs the
  previous 74.89-75.68 range (previous max 75.68, new min 76.07). Bit-identical
  (md5 55f555660f6158501394ff16bde68faa).
- **Caveat.** These three reps were not interleaved against the previous binary,
  so treat the margin as directional; the disjoint ranges are the argument.

## moe_matvec / moe_matvec_fast Q3_K unpack — word slices — ACCEPTED, +5.3% MiniPlus decode (2026-10-23)

- **Target.** APEX-I-MiniPlus (KAT-Coder-V2.5, Q3_K experts on all three banks).
  Before this entry `moe_matvec` held 32.0% and `moe_matvec_fast` 24.0% of decode
  GPU time; the exact kernel only runs the gate (down/up already take the fast
  path), so the exact kernel is 1/3 of the bytes for 32% of the time.
- **Tried.** `q3_quad`: each lane owns quants 4k..4k+3 of every 32-element
  sub-block, i.e. four *consecutive* bytes, and the sign byte at `base + 4k + j`
  is the same byte for all eight sub-blocks. So one `load_u32` for the sign word
  plus one per half-block quant window replaces 96 byte loads per block per
  thread, and `q3_scale` stays as-is. Sub-block index algebra: for sub-block g the
  2-bit field is at byte j of word `g<4 ? qlo4 : qhi4` shifted by 2*(g&3), the
  sign bit is `1 << (4*(g>>2) + (g&3))`.
- **Result.** Kernel GPU ms per profiled window (same dispatch counts before and
  after): `moe_matvec` 60.24 -> 55.78 (-7.4%), `moe_matvec_fast` 45.20 -> 37.15
  (-17.8%), 105.4 -> 92.9 ms of MoE matvec (-11.9%). Five interleaved pairs on
  MiniPlus: baseline 40.08/40.14/40.32/40.33/40.45 (median 40.32) vs
  42.46/42.84/42.96/41.92/41.79 (median 42.46), **+5.3%**, ranges disjoint.
  Greedy output bit-identical (md5 55f555660f6158501394ff16bde68faa, checked
  against a worktree built at HEAD, not against the known-good string alone).
- **What did *not* pay (keep it out of the ledger twice).** Slicing the q8 word
  once per (group,pair) — 256 shared-memory loads per block per thread down to 64
  — moved the kernels only 60.24 -> 59.79 and 45.20 -> 44.17. The loops are not
  LDS-throughput bound; ACO was already merging what it could.
- **The real gap between the two kernels is fp64.** Ablating `fma32` to a native
  `fma()` in the exact kernel took `moe_matvec` 60.24 -> 44.77 ms, i.e. the
  software correctly-rounded FMA is 15.5 ms = 25.7% of that kernel and ~8% of
  decode. The gate only stays exact because of it; changing it is a numerics
  decision (md5 gate), not a load cleanup, so it was measured and reverted.
- **Scope.** Q3_K paths only. Q4_K/QS_K/IQ paths in the same shader are untouched.

## moe_matvec gate — take the native-fma kernel — ACCEPTED, +6.3% MiniPlus decode (2026-10-23)

- **Tried.** The gate bank was the only one still forced onto the exact kernel
  while up and down already took `moe_matvec_fast`. Ablating `fma32()` to a native
  `fma()` in the exact kernel showed where the difference lives: `moe_matvec`
  60.24 -> 44.77 ms in the same profile window, i.e. the emulated correctly-rounded
  fp32 fma (fp64 mul/add plus a round-to-odd fixup) is 15.5 ms = 25.7% of that
  kernel and ~8% of decode. Everything else in the two kernels is identical.
- **Why the precision is worthless here.** Gate output never reaches the residual
  unquantized: swiglu multiplies it with up, then `quantize_q8_K` truncates the
  product to int8 with one scale per 256-element block. A 1-ulp difference dies in
  that quantization.
- **Result.** Five interleaved pairs, MiniPlus: exact gate
  43.43/43.52/42.50/42.99/42.88 (median 42.99) vs native 45.82/46.20/45.42/45.69/
  45.25 (median 45.69), **+6.3%**, ranges disjoint. Greedy output identical at 64
  and at 256 tokens (md5 55f555660f6158501394ff16bde68faa).
- **Honest limit of that evidence.** The md5 gate is *insensitive* to this kernel:
  leaving the exact gate on but replacing its fma with a plain `a*b+c` also keeps
  55f55566 over 64 tokens. So the guarantee is structural (int8 truncation one
  dispatch later, 256-token identical continuation), not bit-equality. A q8_K code
  sitting exactly on a rounding boundary could still split the two paths; nothing
  observed here does.
- **Scope.** Only when quality is off. `--quality` still picks the exact kernel, so
  CPU-parity runs keep the software fma chain.

## moe_tiles — division-free, then parallel expert scan — ACCEPTED, +5.8% MiniPlus decode (2026-10-23)

- **Target.** `moe_tiles` is dispatched twice per layer (gate/up schedule, down
  schedule) as a *single workgroup*, and held 10.6% of MiniPlus decode GPU time:
  942 dispatches, 21.04 us each (min 20.40, max 21.64 — a hard floor, not noise),
  for a job whose useful work is a few hundred bytes.
- **Candidate 1 — the division.** The per-expert loop carried
  `(counts[e] + pair_tile - 1) / pair_tile`, and `pair_tile` is a push constant,
  so GLSL emits an emulated integer division (~40 instructions) for each of 256
  iterations on one thread. `pair_tile` is always a power of two, so
  `findLSB` + shift replaces it, and the scatter's `rank / pair_tile` /
  `rank % pair_tile` become shift/mask. 19.82 -> 15.60 ms in the window
  (21.04 -> 16.6 us per dispatch, -21%).
- **Candidate 2 — the scan itself.** The remaining cost was the same 256-iteration
  loop as a dependent chain of shared-memory round trips on `tid == 0`. Replaced
  with a Hillis-Steele exclusive scan across the 256 slots (both the pair counts
  and the per-expert `ceil(counts/pair_tile)` tile counts, then one shift to
  exclusive). 15.60 -> 2.18 ms, 16.6 -> 2.31 us per dispatch, **9.1x the kernel**.
- **Result.** Five interleaved pairs, MiniPlus: serial 45.81/45.68/45.87/45.54/
  45.63 (median 45.68) vs parallel 48.35/48.46/47.93/48.34/48.43 (median 48.35),
  **+5.8%**, ranges disjoint. Tile bytes unchanged: greedy 256-token output
  identical (md5 55f555660f6158501394ff16bde68faa).
- **Dispatch count was never the problem.** Fusing the two per-layer dispatches
  would have saved at most 0.19 ms/token now that each is 2.3 us (~0.7% of
  decode); the 20 us bubbles were a serial loop wearing a launch-latency costume.
- **Ceiling.** 2.3 us for one workgroup is at the floor this GPU shows for any
  one-workgroup dispatch (`matmul_f32_fast` with 32 groups measures 4.3 us).

## moe_tiles — one schedule per layer instead of three — ACCEPTED, +4.3% MiniPlus decode (2026-10-23)

- **Finding.** Gate, up and down each built their own expert->pair schedule:
  three `moe_tiles` dispatches per layer, 120 per token, byte-identical whenever
  `stream_part == NONE` (the non-streaming path). The schedule depends only on
  `selected`, `n_slot` and `n_expert`, none of which change between the three.
- **Rejected first: fusing gate and up.** Measured to be worth 59 us/token =
  0.28%: the launch gap is a flat 1.48 us per dispatch and a fusion removes
  exactly 40 of 1317. The memory argument also fails — `qx` is 2.3 KB of Q8_K
  for one token and lives in L2. Not worth a second weight binding in the
  hottest kernel.
- **Change.** `p36_vk_moe_matvec` gained `tiles_handoff` / `tile_count_handoff`;
  the first of the three builds and transfers ownership to the caller, the other
  two borrow it. Offered only when not streaming (each streaming part has its
  own cache-slot map).
- **Result.** Five interleaved pairs, MiniPlus: 48.84/49.01/48.82/48.23/48.32
  (median 48.82) vs 51.49/50.91/50.82/51.06/50.74 (median 50.91), **+4.3%**.
  -0.84 ms/token, against the -0.3 ms predicted from dispatch bubbles plus
  kernel time; the redundant builds were also paying scratch allocation,
  descriptor writes and hazard bookkeeping.
- **Profiler.** `moe_tiles` 1200 -> 400 dispatches, 2.18 -> 0.70 ms; total
  dispatches 13170 -> 12370.

## embed — dequantize the row on the device — ACCEPTED, +5.1% MiniPlus decode (2026-10-23)

- **Finding.** `p36_gpu_embed_tokens` dequantized every decode row on the host
  (`p36_tensor_row_to_float` over the mmap) and wrote it into a staging buffer
  the GPU was about to read. Two host round trips per token for 1152 bytes.
- **Change.** `embed_row.spv` reads the row out of a device-resident copy of the
  table, Q4_K and Q8_0, row index from a push constant or from the top-2 buffer.
  Upload is once, during prefill: 272.8 MiB in 374 ms (write-combined GTT, the
  parallel row copy does not help; 434 ms when the table goes to cached host
  memory instead, so device-local stays).
- **Allocation.** Own exact-size buffer, *not* `p36_vk_weight_get_unlocked`:
  that path opens a 512 MiB arena plan block for a 272.8 MiB tensor and the
  allocation fails once the expert banks are resident, which cost 8-13 t/s
  while every token retried it.
- **Result.** Five interleaved pairs: 48.42/48.79/48.68/48.94/48.85 (median
  48.79) vs 51.36/51.69/51.07/51.27/50.84 (median 51.27), **+5.1%**. md5
  55f55566... unchanged. Host phases per token now forward=15.3 read=2.3
  caller=0.52 ms.

### fence wait is the GPU, not host latency — decode is tapped out (2026-10-23)

Split `vk_wait_fence` into "fence already signalled" versus "host had to
block", because only the second can be IRQ/wakeup latency worth chasing:

```
kernel GPU time  302.8 ms      8029 barriers
fence_ready        361 waits   spin_ms 319.2     (~0.88 ms each)
fence_blocked        5 waits   block_ms  2.9     (0.15 ms/token, all of it)
```

The 19.8 ms/token attributed to `vk_wait_fence` is **not** host wait-state:
the host arrives before the fence signals, polls, and sees it signal ~0.88 ms
later, 361 times out of 366. Only 5 waits ever block, for 2.9 ms across the
whole span. So that time is the GPU finishing its own work, observed through
a spin loop — a hybrid spin/block would reclaim CPU cycles, not t/s.

Which closes the 4 ms: 15.3 ms of kernels + ~3.9 ms of GPU-side barrier and
launch cost, with the host provably not the limiter (it is ahead of the GPU
in 361 of 366 waits). Decode on MiniPlus stands at 51.7 t/s.

### fused add + RMS norm + q8_K — ACCEPTED, ~0.6% (structural, 2026-10-23)

`add_rms_norm_q8_k.spv` and its host wrapper already existed but were parked
behind `P36_VK_FUSED_RMS_Q8`. Taking the switch out:

- md5 **55f55566... identical**, so the shader's `nearest_int` magic rounding
  and bsum words do match `p36_quant_q8_k` byte for byte. No arithmetic risk.
- Dispatch inventory, same 20-token span: `q8_k_quant` 1954 -> 1220,
  `add_rms_norm` 778 -> 10, new `add_rms_norm_q8_k` 790. Net **-722
  dispatches, ~36 per token** — the norm pass and the quantization pass merged
  into one, plus the q8 blocks come from values already in registers.
- Speed: 51.38 -> 51.71 t/s median over 5 interleaved pairs, i.e. ~+0.6%,
  which is at the edge of run noise. The reason to keep it is the dispatch
  count and the removed DRAM round trip, not the measured delta.
- The 1220 surviving quantizations are the FFN/MoE input norms, which do not
  go through `add_rms_norm` (they use `norm_rope` and friends). Extending the
  fusion there is the remaining ~60 dispatches/token.

### q8_K quantization dedupe — TRIED, REJECTED (0 hits in 1210 calls)

The premise (gate and up each quantize the same hidden state, so ~50 of the
98 `quantize_q8_k` dispatches per token are redundant) is false. Built the
two-slot cache keyed by `(root, write_gen, in_dim, n_tok)` and switched the
four `p36_vk_quantize_q8_k_alloc_tensor` call sites to it, with the cache
owning the tensor and a `_release()` that only frees uncached ones. md5
unchanged, and the counters read **0 hits / 1210 misses**: every call has a
genuinely distinct input. Gate and up already share one quantization inside
`p36_vk_moe_gate_up_iq2_swiglu`; the rest are per-layer hidden states and the
MoE mid buffer, all different bytes. Reverted.

Useful side finding: 1210 quantize *calls* produce 1954 dispatches, so
`p36_gpu_quantize_q8_k_tensor` splits a call across ~1.6 dispatches on
average. The dispatch count is chunking, not redundancy, which is why both
the dedupe (this) and the in-kernel fusion (rejected above) target the wrong
thing. Any further work on these 98 dispatches has to start at that split.

### Where the last 4 ms of a token goes — barriers, not bubbles (2026-10-23)

- Kernel time is 15.34 ms against a 19.5 ms token. Ruled *out* as the cause:
  host dispatch recording is 0.83 ms/token summed over every op, so launch
  gaps cannot be 4 ms; and raising `P36_VK_BATCH_FLUSH_AT` 32 -> 256 (four
  times fewer submits, four times fewer host gaps) is far worse, **51.2 ->
  39.8 t/s**, so submission overhead is not the limiter either. Small submits
  keep the GPU fed; the ring wants to stay shallow.
- The remaining unaccounted GPU work is the hazard barrier: a full
  compute->compute `VkMemoryBarrier` (SHADER_WRITE|TRANSFER_WRITE ->
  SHADER_READ|SHADER_WRITE, global scope) between dependent dispatches. Counted
  in the profile now (`barriers=`): **8459 over the profiled span, ~423 per
  token**, i.e. ~10 us each against the 4.2 ms gap.
- **Tested and falsified: buffer-scoped barriers are neutral.** Replacing the
  global `VkMemoryBarrier` with `VkBufferMemoryBarrier` entries over exactly
  the tensors in the hazard set (5 interleaved pairs against the global
  version: 51.67 vs 51.48 median, within noise) buys nothing, so RADV is not
  doing a cheaper flush for a range, or the barrier was not the cost.
- Which puts the ~4 ms back on per-dispatch overhead the kernel timestamps do
  not capture: ~1300 dispatches per token against ~3 us each. The lever is
  therefore fewer dispatches (fusing the per-layer chain), not barrier
  surgery. Barrier counting stayed in the profile as instrumentation.

### B+C — deferred top-2 read + lagged loop — TRIED, REJECTED (-3.7%)

Double-buffering alone landed and stays (`44806d9`, md5 unchanged, 51.4 t/s):
`rt->top2[2]` alternated per token, slot published on the session. The lagged
loop on top of it measured **51.21 -> 49.33 t/s median, 5 interleaved pairs**,
so it is reverted (stash `C: lagged decode loop`). What the run showed:

- The read stall does disappear: `P36_HOST_PROF` goes `read=2.3 -> 0.000 ms`,
  `forward` stays 15.3, and `caller_gap` rises 0.52 -> 2.45 ms. The 2.3 ms did
  not vanish, it moved into the resolve: by the time the host has recorded the
  next token the GPU still has ~2.4 ms of the *current* token left to run, so
  the wait is GPU-bound. There is nothing left to hide on the host side.
- That pins the limiter to the GPU, not the host: ~15.3 ms of kernel work per
  token against ~19.5 ms of wall means ~4 ms of bubbles inside the token.
- A deeper command ring (`P36_VK_CB_RING` 4 -> 8) to soften the seal's ring
  backpressure is much worse, 50.2 -> 44.9 t/s: at 4 slots the GPU never falls
  far enough behind for the resolve wait to grow, at 8 it does.

- **Written and measured.** `p36_gpu_tensor_read_begin()` seals the open batch
  and returns its submission seq; `p36_gpu_tensor_read_wait(seq)` waits only
  the ring slots at or below that seq instead of draining the queue. The
  forward pass publishes `s->gpu_top2_seq` and `p36_session_argmax` resolves it
  on demand. md5 stayed 55f55566...
- **Alone it is a 2 t/s regression** (49.30 vs 51.27): sealing early with
  nothing recorded behind it just moves the wait, it does not hide it.
- **C needs two pieces this did not have.** `rt->top2` has to become
  `rt->top2[2]` with a per-token parity flip, otherwise waiting on `seq_i`
  reads a buffer the GPU is already overwriting with `top2_{i+1}`. And the CLI
  loop has to call `eval_device(i+1)` — row index from `top2[i&1]`, token
  pushed as a placeholder and patched right after the fetch — *before*
  `argmax()` resolves `seq_i`; then token i+1's dispatches are already queued
  when top2_i lands, which is the 2.3 ms read plus the 0.5 ms caller gap.
- **Plumbing that works.** `p36_gpu_embed_row_tensor(..., ids, row)` already
  takes a device id; a `-1` sentinel through `p36_forward_tokens_vulkan_into()`
  selects it. EOS costs one wasted token of compute.


## Dense 27B: profile, and the barrier ceiling (2026-10-23)

`Swift-1.5-Qwen3.8-27B-Uncensored-VAL-APEX-I-NanoPlus-V3.gguf` (12.18 GB),
production server flags. Decode **21.9 t/s**, prefill **160-166 t/s** at 651
tokens. Interleaved against pre-session `643e1cf`: 22.04/22.06 vs 21.92/21.97,
i.e. **identical** — today's MoE wins (device embed, expert pair schedule, fused
q8_K norm) do not touch the dense path. Good control, no carry-over.

Generation-only numbers come from diffing `-n 1` and `-n 17` profiles, which
subtracts the shared 23-token prefill instead of guessing at it:

| model | dispatches/token | barriers/token | kernel ms/token | wall ms/token |
|---|---|---|---|---|
| dense 27B | 501 | 420 | 24.2 | 45.6 |
| Occamy MoE | 571 | 395 | 7.1 | 18.6 |

Half of every dense token, and 62% of every MoE token, is not kernel time.

**Barrier probe** (`P36_VK_SKIP_BARRIERS=1`, wrong numerics on purpose, timing
only, 64 tokens, 2 reps):

- dense 27B: 22.6/22.9 -> 25.7/25.8 t/s (**+12%**)
- Occamy MoE: 53.3/52.9 -> 108.5/105.0 t/s (**+100%**)

So the hazard barriers are *not* the dense limiter — that hypothesis is dead.
On MoE they cost ~9 ms of the 18.6 ms token, which means the "tapped out at
51.7 t/s" conclusion from earlier today was wrong about the cause: the GPU is
waiting on its own dependency chain, not on bandwidth. The 105 t/s figure is a
ceiling, not a target, because those dispatches really do depend on each other;
the recoverable part is whatever fusion removes from the chain (a
gate/up/silu/down expert chain collapsing 3 dispatches into 1 is the obvious
first candidate).


## MoE mid-buffer q8 fusion: measured neutral, reverted (2026-10-23)

Folded `q8_k_quant` into the MoE swiglu (the mid buffer is only ever consumed
through it) and A/B'd at `-n 128 --ctx 512`, 5 interleaved pairs against
`287ff53`:

| pair | prev | fused |
|---|---|---|
| 1 | 52.96 | 52.79 |
| 2 | 53.10 | 52.57 |
| 3 | 52.88 | 53.11 |
| 4 | 53.10 | 52.42 |
| 5 | 53.23 | 53.05 |
| median | **53.10** | **52.79** |

Structure did change as intended — `q8_k_quant` 61 -> 41 dispatches and
46 -> 26 barriers per token, `swiglu` 40 -> 20 with a new `swiglu_q8_k` at 20
and no barriers — so the fusion removes 20 dispatches and 20 barriers per
token. md5 stayed 55f55566... throughout.

It bought nothing. Two reasons, both worth remembering:

- The mid quantize was only 20 of the 61 `q8_k_quant` calls; the other 41 are
  shared-expert and attention inputs at sites that have no producer to fold
  into. The barrier saving is 20/395 = 5% of the serialization, ~0.4 ms.
- The fused workgroup does the amax reduction and b8 packing on top of the
  existent swiglu math, which costs back roughly what the removed dispatch
  saved.

Verdict: **rejected**, reverted. The 22%-of-barriers-for-0.42-ms observation is
real but the fix has to come from the *other* 41 calls or from removing whole
dependency steps, not from moving this one.



## (b) collapsed into a real win: the fused expert pair at decode (2026-10-23)

**The plan was wrong about the shape.** `moe_matvec` is not 60 expert
dispatches — it is 3 legs (gate, up, down) x 20 MoE layers, and each leg is
already one grid-over-experts launch. Merging gate+up (60 -> 40) removes zero
barriers, because the up leg never takes one: gate's barrier on `qx` already
covered it. The rejected swiglu fusion (20 dispatches + 20 barriers, net zero)
is the empirical ceiling for that class of change.

**What was actually there.** Occamy decode was on the q8 route while prefill
used the APEX f32 expert pair, purely because `apex_prefill = n_tok >=
P36_VK_MOE_GEMM_MIN` (default 128). The pair is the *identity-decode* schedule,
not a prefill GEMM: `moe_iq3xxs_gate_up_decode` plus `moe_iq3xxs_down_sum_decode`,
no q8_K staging at all. Forcing it on at n_tok 1:

| run | default (q8 route) | pair decode |
|---|---|---|
| -n 256, 3 pairs | 53.45 / 53.09 / 52.68 | 73.97 / 72.33 / 75.03 |
| flag on, 2 reps | 52.48 / 52.84 | 73.92 / 75.99 |

**+41% decode**, and dispatch count per token drops by two thirds of the MoE
front half. Prefill is unchanged (it already used the pair above 128 tokens).

**Parity is the catch.** md5 55f55566 holds on four of five gated prompts
(capital 64/256, haiku 128, sky 128), but `def fibonacci(n):` at 96 tokens
diverges at about token 30: same paragraph, then a heading and blank line
differ. Forcing the pair for prefill too gives a *third* hash (a51119c2...),
so this is inherent to f32 activations vs the q8 staging, not a
route-mixing artifact.

Landed **opt-in**: `P36_VK_MOE_PAIR_DECODE=1`. The q8 route stays the
bit-exact default and the md5 gate still reads 55f55566 with the flag unset.
Whether to flip the default is a numerics decision, not a performance one.



## Server telemetry: fused expert pair under agentic load (2026-10-24)

Validation run for `P36_VK_MOE_PAIR_DECODE=1` on the BC-250 with
`--prefill-chunk 512 --kv-disk-dir`, five agentic loops (Three.js generation
plus tool execution):

- **31,075 tokens** total: 15.5k prefill, 15.5k decode.
- Decode **46.0-50.8 t/s sustained** through one continuous 11,000-token
  generation. Short-context CLI decode on the same route is 74-76 t/s, so live
  11k context costs roughly a third of the rate.
- Prefill **~600 t/s average at 7.6k context**, peaking at **814 t/s** on
  512-token chunks.
- `--prefill-chunk 1024` still trips the RADV 64 MB scratch allocation
  failure; 512 scaled to **22,361 live context tokens** without a stall.
- Structured output held: JSON/schema adherence and stop-token precision across
  sequential `write`, `BrowserNavigate` and `ShellStart` calls, including a
  procedural voxel pagoda that rendered first try.

Verdict: the pair route is production-ready for the server, and stays behind
the flag so the q8 route remains the bit-exact default. The token-shift seen in
the fibonacci probe did not surface as a tool-call or schema failure in this
workload.


**Addendum (2026-10-24):** flipped to default-on after the agentic validation
above. `P36_VK_MOE_PAIR_DECODE=0` disables it and restores the q8 route; the
md5 gate reads 55f55566 either way on the capital-of-France prompt, while the
fibonacci probe still diverges between the two routes.


## The missing 30% was a stale `p36-server` binary (2026-10-24)

Chasing the gap between the server (46-51 t/s at 11k context) and the CLI
(68.3 t/s at 10,806 tokens), same board, same model, `--prefill-chunk 512`:

| path | route | decode at 10.8k |
|---|---|---:|
| CLI | fused pair | 68.3 t/s |
| server | fused pair (current build) | 68.1-68.7 t/s |
| server | q8 (`P36_VK_MOE_PAIR_DECODE=0`) | **49.2 t/s** |

The 49.2 reproduces the agentic telemetry exactly. Cause: `make p36` relinks
only the CLI, so `p36-server` was still the 21:38 build from before the pair
route existed — the env var was a no-op there, and the whole agentic suite ran
the q8 route. Rebuilt with `make -j16`, the server lands on the CLI number.

So there is no server-side harness tax at depth: SSE, sampling and the request
loop cost ~0.02 ms/token of caller gap. Attention decay at 10.8k is real but
small (-11% vs 1k, ~-0.9 t/s per 1k).

Disk KV: decode identical (68.08 vs 68.68 t/s), and \_ttft drops 15.88 s -> 1.98 s\_
on a repeated 10.8k prompt. It costs nothing; it only saves prefill.

Lesson: rebuild **all** binaries before benchmarking the server, and check
`ls -l p36-server` against the commit date when a route change is suspected.


## Expert router top-k: subgroup selection — ACCEPTED, +1.24% decode (2026-10-26)

The decode router kernel spent 42.8 ms of 1434 ms profiled GPU time (3.0%,
`evidence/raw/p1-sumdecode/iq2m-decode-base.txt`) and did it with a
256-lane shared-memory reduction: one tree pass of eight barriers plus three
barrier-separated publishes per selected expert, `1 + 11k` barriers for k
experts (`vulkan/router_topk.comp`, pre-change copy in
`evidence/raw/w-router-subgroup/original.comp`).

Replaced the tree with subgroup max plus a subgroup min over the tie-break
candidate lane, then a four-candidate workgroup reduce: `1 + 2k` barriers,
same selection order, same lowest-index tie-break, same FP64 normalization.
Selection arithmetic is untouched; only the reduction shape changed.

5 interleaved pairs on MiniCPM-V-4.7-35B-A3B APEX-I, ctx 512, 48 greedy
tokens, shader binary swapped between runs:

| arm | decode median | MAD |
|---|---:|---:|
| 256-lane tree | 84.60 t/s | 0.30 |
| subgroup selection | **85.65 t/s** | 0.07 |

All five pairs in the same direction; prefill unchanged within its own noise
(that path mostly uses the host-row router above 64 rows). Parity: six
prompts byte-identical to the old shader (4x64 and 2x48 greedy tokens).

Evidence: `evidence/raw/w-router-subgroup/` (`ab.csv`, `note.md`). The 3.0%
profile share bounds this at roughly +2%, and the measured +1.24% sits under
it; the barrier count was the cost, as the audit suspected.

Do not re-open router *projection* precision from this result: the FP32 gate
matmul and its reduction order are unchanged and were rejected separately
(`docs/backends.md`).

## War-only hazard barriers: census says there are none — CLOSED (2026-10-26)

The audit proposed narrowing pure write-after-read hazards to an
execution-only barrier (no memory masks). Classification needs every binding,
because the existing loop early-exits on the first hazard and cannot tell WAR
from RAW/WAW. Counters added in the dispatch path, printed with
`P36_VK_HAZARD_STATS=1`; inert otherwise.

48 greedy tokens, MiniCPM-V-4.7-35B-A3B APEX-I, ctx 512:

```
barriers=32962  raw=31443  waw=5439  war=1470  pure_war=0
```

Every WAR co-occurs with a RAW or WAW on the same dispatch, so an
execution-only barrier would change nothing on this workload. Stop here; do
not add a second barrier form without a profile that shows pure WAR first.

Cross-check on `Cyber-Tiel-Coder-35B-A3B.APEX-I-NanoPlus` (same 3-pair
interleave, same prompt): 84.33 -> 85.69 t/s median, +1.61%, all pairs in the
same direction (`evidence/raw/w-router-subgroup/ab-cybertiel.csv`).

## MoE MTP with a real head: 0.92x, structural (2026-10-26)

First run of MoE MTP in p36 with an actual support GGUF
(`Qwen3.6-35B-A3B-MTP-Q4K-Q8_0-F32.gguf`, 1.59 GiB) against
`Cyber-Tiel-Coder-35B-A3B.APEX-I-NanoPlus` (12.55 GiB, same 248320 vocab).
Qwen3.6 APEX itself (13.03 GiB) cannot host the head on this 14 GiB board:
radv buffer allocation fails, then the process is OOM-killed.

Three fixes were needed before it ran at all: bind the token table and output
projection from the target model instead of the support file (its Q8_0 copies
are 1 GiB of dead weight), skip those tensors in the MTP prewarm, and let the
MoE MTP validator accept the matrix types the main model already validated.
Greedy output stays byte-identical to plain decode, so the speculate/verify
path is exact.

3 interleaved pairs, ctx 512, 96-token cap, `--temp 0`:

| arm | decode median |
|---|---:|
| plain | 85.51 t/s |
| MTP draft 2, margin 0 | 78.60 t/s |

Cycle accounting: an accepted cycle is `fwd 13.65 + draft 0.21 + verify
11.72 = 25.67 ms` for 2 tokens (77.8 t/s); a gate-rejected cycle is 13.50 ms
for 1 token (74 t/s). Verify alone costs 0.86 of a forward, so speculation
cannot beat the 85.5 t/s plain rate at draft 2. Sweep: d2/m0 77.98, d3/m3
76.63, d4/m3 75.64, d3/m0 70.90 - all lose.

Do not tune draft count or margin further. The pending forward must fold into
verify row 0 (one forward per accepted group) before MTP can pay; that
restructure is the only remaining lever here. Evidence:
`evidence/raw/w-mtp-moe/`.

## One-forward MTP (draft before verify): exact, still slower (2026-10-26)

Restructured the speculative cycle so the head drafts from the last committed
hidden before the target forward, putting the pending token at row 0 of the
verify batch: one forward per accepted group instead of a forward plus a
verify.  Greedy parity held, drafts improved (full-accept 27/37 = 73% vs
27/42 = 64% with the old same-position pairing), and a full-accept cycle
costs 1.9 ms head + 18.5 ms two-row verify = 20.4 ms for 2 tokens
(98 t/s).  But a rejected first draft pays the verify forward *and* a 11.8 ms
replay, 32 ms for 1 token.

3 reps, ctx 512, 96 tokens, draft 2, margin 0:

| scheme | decode median |
|---|---:|
| plain | 85.5 t/s |
| old two-forward MTP | 78.6 t/s |
| one-forward MTP | 71.8 t/s |

Reverted: exact but slower.  The limiting cost is the second verify row:
one row 11.7 ms, two rows 18.5 ms, three 22.5, four 26.3.  Trying the 128-row
MMQ tile for the 2-row case instead of the nx kernels made it 71 ms - the nx
path stays.  Those extra rows do not amortize because the IQ3_XXS/Q2_K decode
kernels are ALU (unpack) bound on this GPU, not weight-bandwidth bound.

So 150 t/s via MTP needs 2-4 row forwards at ~1.05x the cost of one row; at
the current 1.37x the best possible is 2/1.37 = 1.46x *and only with 100%
acceptance*.  The kernel that would have to change first is the routed
gate/up and down matvec (moe_matvec_fast / moe_down_q2k_sum_decode_iq3xxs):
share the unpacked weight word across rows instead of re-unpacking per row.

## 2026-10-26 MTP on the APEX / Q4_0-expert files (works, still slower)

* `gguf-tools/mtp-head-extract` extracts the nextn block into a support GGUF and
  requantizes Q4_0 experts to Q4_K, so `--mtp` runs on our own models at all:
  APEX-I head 466 MB, IQ2XXS head 328 MB.
* Removed the `mtp_draft_token == argmax` pre-gate; it rejected nearly every
  cycle and made draft depth pointless.
* Draft head logits were NaN with an extracted file: the writer aligned tensor
  data by pointer instead of by byte offset (`calloc` is 16-byte aligned).
* Measured: APEX plain 85.5 vs MTP 44.2 (accept 50.8%); IQ2XXS plain 96.4 vs
  MTP 84.4 (accept 89.2%). Still a loss.
* Root cause of the loss is the cycle, not the head: p36 pays a separate target
  forward for the pending token plus a verify forward, and on partial accept a
  third full replay. Upstream does one target forward per cycle and replays only
  recurrent layers from captured inputs.
* `reconsider_if`: the cycle is rewritten to draft before the target forward and
  to replay recurrent layers only.

## 2026-10-26 MTP cycle rewrite (one target forward per cycle) - LANDS

* Draft before the target forward from the raw residual hidden (`last_h_raw`);
  verify `[pending, drafts...]` in one forward; commit via `row_tops`.
* IQ2XXS-Q2K model: plain 95.52 -> MTP 97.52 t/s (draft 3, margin 1), parity
  exact, 96.8% acceptance. First configuration in this campaign where MTP beats
  plain decode.
* Draft 4+ still loses: a 7-row verify costs 34.8 ms and a partial accept adds a
  16-26 ms full replay. `reconsider_if`: recurrent-only replay + row-amortised
  small-batch kernels (extend `dense_*_rows` past 4 rows).
* APEX-I head: only ~50% acceptance after Q4_0 -> Q4_K requant; the same
  treatment of the IQ2_XXS head keeps 96.8%, so the APEX head itself is suspect.

## 2026-10-26 MTP speed pass 2 - LANDS (1.06x)

* MoE row-pair path extended to the native IQ2_XXS/Q2_K recipe; the verify batch
  (2..31 rows) used to fall off the fast MoE path entirely. 99.1 -> 106.5 t/s
  (n=96); 3-rep n=128: plain 96.05, draft 3 margin 1 = 101.43, draft 4 margin 2
  = 101.93. Parity exact.
* Drafting reads top-2 from the device instead of the 1 MB logits row.
* The APEX support head must be re-extracted after any extractor change - the
  stale file explained its 50% acceptance (now 92.9%), but APEX stays at parity
  (80.4 vs 84.4) because a verify row costs ~4.2 ms and only ~0.3 extra tokens
  are accepted per row.
* `reconsider_if`: row cost under ~3.5 ms or per-row acceptance above 0.35.

## 2026-10-26 MTP tuning by content - LANDS (1.12x on code)

* Best flags: IQ2XXS-Q2K `--mtp-draft 3 --mtp-margin 2` (95.24 -> 106.77 t/s,
  3 reps, code prompt); APEX-I `--mtp-draft 4 --mtp-margin 2` (84.48 -> 88.05).
  Prose prompts give 1.06x and 1.00x - acceptance is content-dependent.
* 21% of cycles at the best setting are `no-draft` (gate rejects the first
  draft), costing one 1.6 ms head forward each.
* Break-even per verify row is ~0.36 accepted tokens; we measure 0.3-0.35. The
  1.7x upstream figure needs near-flat row cost, i.e. multi-row MoE kernels.

## 2026-10-26 Prompt lookup rides the MTP verifier - LANDS

* `--lookup-k N` blends with MTP: the chain is appended only when the first MTP
  draft equals the chain's first step, then verified in the same forward. Head
  KV for committed lookup rows is refilled by a projection-free head forward.
* Verbatim-repeat prompt: 95.21 plain -> 124.33 t/s (`--mtp-draft 3
  --mtp-margin 2 --lookup-k 8 --lookup-order 5`), i.e. +31%; lookup-only 108.8,
  MTP-only 107.6. Output byte-identical to MTP-only.
* Cost when it does not fire (code prose): ~1%. Opt-in, so defaults unchanged.
* Parity note: MTP can flip near-ties against plain serial decode (batched
  verify forward); strict mode is 0.37 t/s, so this is inherent, not a bug in
  the MoE row-pair path.


## 2026-10-27 PTQ1_0 small-batch matmul LANDS, in-file MTP REJECTED (Bonsai 27B)

Model `gguf/Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf` (qwen35 hybrid,
16 attention + 48 recurrent layers, all dense weights PTQ1_0 1.75 bpw, 5.95 GB,
in-file nextn head at `blk.64`). Evidence: `evidence/raw/w-bonsai-ptq1/`.
Baseline ctx 512, 128 gen: prefill 94-100 t/s, decode 29.4-32.1 t/s.

* **Fixed (ACCEPTED): in-file MTP head never worked for a multi-row verify.**
  `--mtp-draft 3` (>1 draft => mtp_cap >= 2 => 2 verify rows) aborted at
  `MTP draft failed at pos N`. `p36_mtp_forward_vulkan` projected the head's
  logits with `p36_gpu_tensor_matmul_scaled`, which ignores the prism Hadamard
  rotation that `output.weight` carries on every Prism quant; the target head
  has the rotation branch, the MTP head did not. Now branches on
  `p36_hadamard_rotate_enabled(weights.output)` and calls the
  rotate+q8_K+`matmul_q8_or_float_scaled` path. Greedy output with
  `--mtp-draft 3 --mtp-margin 0` is byte-identical to MTP off.
* **Landed: `vulkan/dense_extra_decode_ptq1_0_nx.comp`, PTQ1_0 small batches.**
  PTQ1_0 had no small-batch kernel, so every `n_tok != 1` forward used the
  64-token MMQ tile, which costs the same for 2 rows as for 64: a 2-row MTP
  verify measured ~550 ms/cycle (2.80 t/s decode). The new kernel streams each
  ternary block once for 2 tokens (ROWS=4, NT=2, wave64, same per-row int order
  and `subgroupAdd` as the one-token kernel, so row outputs are identical) and
  is selected for `n_tok <= 8` plus the ragged tail past whole 64-row tiles.
  2-row forward **~550 -> 57-60 ms**; `--mtp-draft 3` decode **2.80 -> 22.4
  t/s**. Greedy-identical to the 1-row path at chunk 2/4/8.
* **REJECTED: MTP on this model.** `--mtp-draft 3 --mtp-margin 0` = 20.3-22.4
  t/s vs 29.4-32.1 plain (0.66-0.70x); draft 4/5 = 15.9/12.4; draft 3 with the
  default margin gate = 28.2. Acceptance 1.54 tokens/cycle (1:38, 2:45 over 83
  cycles). Two independent blockers, both measured:
  1. A verify row is not cheap: marginal cost of an extra row in a batched
     PTQ1_0 forward is ~24 ms, a full weight pass (6.39 GB / 24 ms = 266 GB/s,
     against a 333 GB/s sibling). Chunk scaling: 1 row 35.3 ms, 2 rows 57-60,
     4 rows 103, 8 rows 198 -> the ternary kernel is token-linear, so the one
     structural lever a hybrid decode has is unavailable here. Break-even at
     1.54 tokens/cycle needs a 2-row verify under 35.5 ms, i.e. 13.5% *faster*
     than a 1-row forward.
  2. 48 of 64 layers are recurrent, so a partial accept cannot truncate state.
     The cycle restores the frontier snapshot and replays the accepted prefix
     (28.5 ms, a full forward); 46% of cycles are partial.
  `reconsider_if`: a trit kernel that adds a verify row for <2 ms, or per-row
  recurrent-state checkpoints inside the batched SSM kernels so a partial
  accept rolls back without replaying the forward.

## 2026-10-27 PTQ1_0 speed attempts that measured flat - REVERTED

Bonsai 27B PTQ1_0 (5.95 GB dense), BC-250/gfx1013. Baseline ctx 512:
prefill 98-100 t/s, decode 31.5-33.6 t/s. Evidence: `evidence/raw/w-bonsai-ptq1/`.

* **int8 dot4 tile kernel: 13% slower.** Full rewrite of
  `dense_extra_mmq_ptq1_0.comp` staging trits as int8 quads and using
  `dotPacked4x8EXT` (256 dots replacing 512 packed `i16vec2` madds per slice,
  same element pairing, 128 trits = 32 quads, sections 80/40/8). 85.7 t/s vs
  98.0. gfx1013 `v_dot4` gives no MAC-width win over `v_mad_i16`, so the i16
  kernel stays (it already sits at 5.4 of ~6.1 T-MAC/s packed-i16 peak, and the
  trit unpack is free: 100.03 t/s with the decode deleted).
* **One-token decode ROWS 8: flat.** 33.10 t/s both ways.
* Do not re-test either without new evidence that gfx1013 dot4 is full rate.
* Operational trap worth remembering: `p36*` load `vulkan/*.spv` from disk at
  run time, so re-running an older binary after editing a shader silently uses
  the new shader. Restore `.spv` for A/B text comparisons.
  Follow-up: a uvec2 (4-weights-per-8-byte-LDS) staging rewrite also went
  degenerate on the first try and was reverted unmeasured; both re-layouts
  failed the same way, so any future attempt must come with a kernel-level unit
  test (single tensor, tile kernel vs one-token kernel) instead of whole-model
  text diffs.
  Correction to the entry above: the tile kernel is not numerically broken -
  `--dump-logprobs` shows max |dlogp| = 0.106 vs the one-token kernel, and the
  text divergence is one borderline argmax flip 24 tokens in. Greedy text is
  too coarse to validate tile-kernel edits; build the single-tensor kernel unit
  test first.
* Oracle note: `./p36_test --dense-quant-model-rows --model <bonsai.gguf>` now
  covers the tile kernel (token_counts extended to 16/32 in
  `tests/p36_test.c:3684`); 18 PTQ1_0 cases pass at max_abs 2.1e-07. The tile
  kernel is CPU-exact, so remaining prefill work is pure throughput - use this
  test (~40 s) to check any staging rewrite instead of greedy text.

## 2026-10-27 Hazard-barrier scope narrowing - REVERTED (4% slower)

`P36_VK_SKIP_BARRIERS=1` gives decode 43.05 vs 31.63 t/s (+36%), but that upper
bound removes real RAW ordering. Narrowing the per-dispatch global
`VkMemoryBarrier` to buffer-scoped `VkBufferMemoryBarrier`s over the dispatch's
own hazard roots is correct yet 4% slower (31.42/30.43/29.88 vs
33.16/31.95/30.99, interleaved). The cost is pipeline drain on genuine
write->read edges in the per-token chain; only removing those edges (fuse
hadamard_prepare into the matmul, 254 dispatches/barriers per token, or
double-buffer) can reclaim it. reconsider_if: hadamard fusion or double-buffered
activations land.

## 2026-10-27 Hadamard prepare: ceiling measured, barrier shortcuts rejected

`P36_VK_SKIP_HADAMARD=1` (diagnostic stub in core_15, default off) removes the
254 prepare dispatches+barriers per decoded token: decode 37.5 vs 31.0 t/s
(+21%). Cheaper ways to get it:
* execution-only global barrier (access masks 0): **wrong output** - greedy text
  diverges, RADV needs the cache masks. Reverted.
* buffer-scoped barriers: -4% (see previous entry). Reverted.
Only fusion removes the edge; spec in evidence/raw/w-bonsai-ptq1/note.md.
reconsider_if: fused FWHT+q8 lands inside dense_extra_decode_ptq1_0.comp.

## 2026-10-27 Hadamard prepare: 40 CU proof, subgroup rewrite, latency floor (CLOSED)

Cu unlock is live: `bc250_cc_write_mode=3`, journal shows `active_cu_number 40`
(SE 2, SH 2, CU 10) and `bc250-40cu-enable: CC=0xfff80000->0xffe00000
SPI=0x7->0x1f`. Scaling probe (`/tmp/cuprobe/cu_scale.c`, 8 FMA chains, 256
thr/WG) is flat 24.3-24.4 ms from 1 to 40 workgroups, then 41 -> 33.10 ms.
Knee exactly at 40, so 40 CUs are co-schedulable. Nothing saturates at 24.

Decode does not use them: profiler dispatch-weighted occupancy 27.9/40 (70%),
43% of launches are under 40 WGs (`hadamard_prepare` 12.5, `add_rms_norm_q8_k`
1.5, `delta_gates` 1.5). Prefill is 39.9/40 (100%). Cause is dispatch size and
a 10-deep dependency chain, not the unlock.

Subgroup FWHT rewrite (stages h=4..32 via `subgroupShuffleXor`, h=64..512 stay
on LDS+barrier, 11 -> 5 dynamic barriers per WG): **bit-exact** (frontier_002048
logits md5 97ae3f86... unchanged) but **+0.5%** median over 6 interleaved reps
(28.84 -> 28.98). KEPT: equal speed, fewer explicit barriers. Reason it does
nothing: `subgroupShuffleXor` lowers to `ds_bpermute_b32`, the same LDS
crossbar plus an implicit wave sync - the critical path is unchanged.

Ablation decomposition, all wall-clock, same 192 tok / ctx 2048 A/B:
thread mapping 256->1024 = 0.0%; barriers 11->5 = +0.5%; nop body = +12.0%;
skip kernel entirely = +16.5%. So the body is load-to-use latency on 5
workgroups (width 5120/1024) at 12.5% occupancy, not bandwidth (9.2 KB per
dispatch) and not synchronisation. 157 dispatches/token x ~27 us each.
reconsider_if: rotation is folded into the consumer tile load so no separate
single-row pass exists. Fusion into add_rms_norm_q8_k was measured earlier at
-2.3% (serialises 5 CUs into 1) and is not the answer.

Profiling tools added/documented:
* `P36_VK_PROF_KERNEL=1` enables timestamp queries; gpu_ms is raw ticks x
  `timestampPeriod` (10.0 ns here, `timestampComputeAndGraphics` true, 64-bit
  results). Bracketing is per op-group (a run of consecutive same-path
  dispatches), not per dispatch.
* Cost of that instrumentation is 4-6% of gen t/s, so ON-run t/s are not
  comparable to OFF.
* Absolute per-op gpu_ms are unusable for long consecutive runs: a group
  interval absorbs host-recording gaps (dense_extra_mmq reported 641 ms for a
  prefill that takes 101 ms). Use the wall-clock ablations instead.
* Do NOT reset the whole query pool once per command buffer - it hangs RADV
  (pending query pairs from an in-flight submission get clobbered). Reset the
  exact pair being written.

## APEX-I-NanoPlus MoE decode: real GPU ms, and the merge point is the engine not the kernel (2026-10-27)

First per-op-group GPU-ms breakdown for the two 35B-A3B APEX-I-NanoPlus files
(`Qwen3.6-…` 13.03 GiB and `Cyber-Tiel-Coder-…` 12.55 GiB; both run headless of
MTP on the 14 GiB board). ctx 2048, 192 tokens, 3 interleaved pairs. Evidence:
`evidence/raw/w-gpu-ms/`.

- **Unlock OK.** `bc250_cc_write_mode=3`; `RADV_DEBUG=info` reports `num_se 2,
  num_cu 40` (journalctl needs sudo, unavailable this session);
  `/tmp/cuprobe/cu_scale` re-run on an idle GPU is flat 36..40 WG (48.5 ms) and
  steps to 65.9 ms at 41 WG — knee exactly at 40.
- **Timestamps cost 14%, not 4-6%.** ON 66.79 vs OFF 77.63 t/s (CyberTiel) and
  66.72 vs 77.72 (Qwen). Host-only `P36_VK_PROF` and `P36_VK_PROF_WG_HIST` are
  free; the cost is the per-bracket drain, 2.1 ms/token over ~1081 brackets =
  **~1.9 us per serialization point**. So the earlier "4-6%" note is a
  dense-model number; on a 1002-dispatch/token MoE token the drain-dominated
  cost is 14%. Decode bracket sum reconciles with wall (1.04x) once prefill rows
  are excluded; prefill rows still absorb gaps (0.84x of prefill wall).
- **Decode is 1001.8 dispatches/token; prefill is 1.26.** Same kernels, 794x
  asymmetry — the MoE decode is a latency/occupancy problem in the engine, not
  a CU problem.
- **48.6% of decode dispatches run on ≤20 of 40 CUs** (<5: 282.8/token,
  5-20: 203.0, 20-40: 90.6, >40: 423.3 per token). Ops that are majority ≤20 WG
  hold 486 dispatches/token but only 2.72 ms/token = 17% of decode GPU time.
- **The expert launch loop does not exist.** 8 experts x 40 layers = up to 320
  expert-FFN slots/token are served by **72 dispatches/token** (36
  `moe_*_gate_up_decode` + 36 `moe_*_down_sum_decode`, 36-55 us each). MoE
  decode = 3.76 ms/token = 24.1% of decode GPU time. Removing every one of those
  launches saves 72 x 1.9 us = 0.14 ms/token (0.9%), so "batched expert GEMM"
  as a launch-count lever is already shipped; the remaining MoE lever is inside
  the grouped kernels (`moe_iq3xxs_down_sum_decode` 1.95 ms/token, 12.5%).
- **Router is already fused and device-side**: `router_topk` = 1 dispatch/layer,
  5.8 us, 0.24 ms/token (1.5%); top-2 is read from the device buffer already.
- Ceiling for any dispatch-elimination plan: 1002 x 1.9 us = 1.9 ms/token
  (12-15%). `reconsider_if`: a fusion merges ≥3 of {q8_k_quant 125/token,
  add_rms_norm_q8_k 80, swiglu 40, router_topk 40, moe_tail 40,
  dense_f32f_d_1xN 40} per layer; single-kernel fusions buy ~0.5-1% and the
  `P36_VK_FUSED_HAD` -2.3% precedent applies.

## APEX-I-NanoPlus MoE down_sum: the gap is thread count, not the grid (2026-10-27)

Shape pass (`P36_VK_PROF_SHAPE=1`) on `moe_iq3xxs_down_sum_decode` vs the
IQ3_XXS gate/up build, same model/ctx/run:

| | down_sum | gate_up (IQ3_XXS) |
|---|---:|---:|
| workgroups | 2048 (51/CU) | 4096 |
| local_size | 32 | 64 |
| experts per WG | 8 serial + subgroupAdd each | 1 |
| threads/dispatch | 65,536 | 262,144 |
| bytes/dispatch | 3.06 MiB | 6.42 MiB |
| time | 54.5 us | 44.6 us |
| achieved | **58 GB/s** | **144 GB/s** |

2.5x the throughput for 2.1x the bytes, same quant format and table lookups.
Not the grid (2048 WGs already 51/CU), and not the format (a 62,080-WG
32-thread kernel, `dense_q6k_decode_2048x248320_n1`, sustains 348 GB/s).

Two bit-exact attempts, both measured on 3 interleaved pairs, logits md5
unchanged:
* force wave32 (`P36_VK_WAVE32_MOE_DOWN=1`, added to the wave32 forcing path):
  +0.2% (77.57 -> 77.75 median, overlapping ranges). NULL. Switch kept,
  default OFF.
* hoist the 8 per-expert `subgroupAdd` into `part[8]` and apply the route sum
  afterwards: -1.4% (77.75 -> 76.68, disjoint). REVERTED - the 8-slot array
  spills to scratch.

Remaining lever is the structure the prefill path already uses: 2D grid
`(out_dim, n_used)` = 16,384 WGs writing per-expert partials + a
`moe_reduce`-style sum. Ceiling from the measured sibling rates: 1.96 ->
0.74-0.5 ms/token (6.5-9% of the 12.9 ms decode token). Not implemented: it
changes the reduction order, so logits drift, and it needs an extra dispatch
per layer. Evidence: `evidence/raw/w-gpu-ms/`.

## MoE down sum: split expert grid + ordered reduce — bit-exact, +3.1%, reverted (2026-10-27)

Built the 2D-grid split (`(out_dim, n_used)`, one (row, expert) per WG, partial
`acc*scales[expert]` buffer) plus a separate ordered reduce that replays the
fused accumulation `sum = fma32(part[u], route[u], sum)` with the same software
`fma32`. Logits md5 `e7c953fc…` **identical** with the split on.

3 interleaved pairs, ctx 2048, 192 tok: fused 76.52/77.46/77.36 (median 77.36)
vs split 79.11/79.82/79.79 (median **79.79**) = **+3.1%**, disjoint.

Kernel accounting: down_sum 54.5 -> 38.3 us/dispatch (58 -> 84 GB/s), 1.96 ->
1.379 ms/token, plus a new 0.118 ms/token reduce (40 dispatches) = 1.497
ms/token, i.e. 3.6% of the 12.9 ms token. Below the 5-6% bar, so the variant
was removed rather than parked behind a flag (AGENT.md: no permanent semantic
variants). Default path unchanged, post-revert baseline re-measured at 77.59.

reconsider_if: rows-per-workgroup is raised on the split grid (4 rows = 784 B
contiguous per WG, 4096 WGs) instead of 1 row = 196 B; the split trades the
serial expert chain for short-stream latency and lands at 84 GB/s against the
144 GB/s its 1.2 KB-per-WG sibling reaches.

## MoE down split: 4-row workgroups do not add bandwidth (bit-exact, +2.0%, reverted) (2026-10-27)

Follow-up to the 1-row split (+3.1%). Same ordered reduce, but each workgroup
owns 4 consecutive output rows: 4096 workgroups instead of 16384, 784 B
contiguous per (tile, expert). Logits md5 `e7c953fc…` identical again.

3 pairs, ctx 2048, 192 tok: fused 76.09/78.10/77.52 (median 77.52) vs split
79.41/79.08/78.51 (median **79.08**) = +2.0%, i.e. *worse* than the 1-row
split. Instrumented: 4096 WG, 40.0 us/dispatch, 1.440 ms/token, **80 GB/s**
against the 1-row variant's 84 GB/s. Row packing moved nothing.

Conclusion: the 196 B burst was not the limiter. Both split tilings sit at
80-84 GB/s while the 64-thread IQ3_XXS gate/up sibling reaches 144 GB/s, and
the per-workgroup fixed cost (selected[]/scales load, pointer setup,
subgroupAdd, partial store) is spread over 8x fewer elements in a 512-wide down
row. Both variants reverted; default fused path unchanged, tree clean,
post-revert baseline 77.26 t/s.

reconsider_if: the split is retried only together with amortising that fixed
cost (e.g. 1 expert x 8-16 rows and two experts per workgroup), and only with a
prior measurement showing the fused kernel's us/dispatch floor is not itself
that fixed cost.

## Decode: reuse the norm's q8_K and record the K-quant shared expert first — ACCEPTED, +3.34% decode (2026-10-27)

Three bit-identical changes in `core_15_forward_tokens_cpu.inc`:

- **Bug:** the post-attention `add_rms_norm_q8_k` writes `rt->inp_q8`, but only
  the Hadamard variant set `inp_q8_valid`, so every MoE FFN quantized the same
  row again. Now `inp_q8_valid = norm_q8` / `= fused_next`.
- **MoE K-quant `ssm_alpha`/`ssm_beta`** read the live `rt->inp_q8` instead of
  the float route's internal requantize (`p36_recur_q8_reuse`).
- **Shared expert first:** a non-Q8_0 shared expert is recorded before the
  routed pass for `n_tok <= 4`, so its gate/up need no barrier after
  `router_topk` and overlap the expert kernels. At prefill size the same order
  cost ~3% (shared GEMMs contend with expert GEMMs), hence the gate.

Parity: frontier logits (chunk 256 and chunk 1) and 96-token greedy text
byte-identical vs `f220af7` on CyberTiel, Qwen3.6 NanoPlus, Occamy MiniPlus,
MiniCPM NanoPlus and Bonsai PTQ1_0 (`evidence/raw/w-decode-reorder/parity-split.txt`,
the split build includes this change). A/B CyberTiel ctx 1024, gen 128, 5
interleaved pairs: decode **78.52 -> 81.14 t/s (+3.34%)**, disjoint ranges;
prefill +0.86% (noise). `ab-tiel2.{csv,summary}`.

## Parallel token tiles for `recur_conv_silu_decode` at prefill — REJECTED (2026-10-27)

Grid (channels, n_tok/16) instead of one thread per channel looping all
tokens: bit-identical but the kernel went 21.9 -> 59-60 ms per 1024-token
prefill on CyberTiel, both with arrays and with the window in three
registers. The serial kernel already streams at ~92 GB/s and is only 1.4% of
prefill at ctx 1024; the 494 us/dispatch that motivated it was a
timestamp-contaminated ctx-2048 number. Not worth more work.

## MoE down split + ordered reduce, rebuilt as the default — REJECTED, decode halved (2026-10-27)

Rebuilt the §3212 experiment (grid.y = expert, partials, `moe_down_route_reduce`)
for all five sum_decode builds. Bit-identical on 5 models, but CyberTiel decode
**78.34 -> 36.74 t/s**: `moe_iq3xxs_down_sum_decode` 54 -> 383 us/dispatch.
Moving the partials from scratch to a dedicated device-local buffer did not
change it (36.74 again), so host-mapped memory is not the cause; the
2026-10-27 split that measured +3.1% differed in some way this rebuild did not
reproduce. `ab-split2.{csv,summary}`. Not pursued further.

## `add_rms_norm_q8_k`: subgroup shuffles for the in-wave reduction levels — REJECTED, neutral (2026-10-27)

The decode norm is one 1024-thread workgroup at ~9.5 us, 80x/token. Replaced
the in-subgroup levels of the fp64 sum-of-squares tree and of the 64-lane
argmax tree with `subgroupShuffleDown` in the same pairing (12 fewer
1024-thread barriers per call). Bit-identical (CyberTiel, Bonsai PTQ1_0:
chunk-256/chunk-1 frontier logits and greedy text), independently reviewed.
3 interleaved pairs vs `13eb0aa`, CyberTiel ctx 1024 gen 128: decode A
80.46/81.65/81.47 vs B 81.53/81.10/80.56 (median -0.45%, overlapping),
prefill +1.6% (inside the 3.4% floor). The barriers are not where this
kernel's time goes. `evidence/raw/w-decode-reorder/ab-norm-shuffle.csv`.

## Embedding table upload off the first prefill chunk — ACCEPTED, +15.8% cold prefill (2026-10-27)

Host profile of a 1024-token prefill (CyberTiel): first 256-token chunk 780 ms,
later chunks ~370 ms. 363 ms of the difference was `p36_gpu_embed_table_prepare`
copying the 272.8 MiB Q4_K token table to the device synchronously, at NVMe
speed (cold pages). Prefill embeds on the host and never reads that table, so
the first prefill now starts the copy on a background thread (`madvise
WILLNEED` + memcpy into the mapped buffer) and the first decode-side
`p36_gpu_embed_row_tensor` joins it. Same kernels and paths, byte-identical
output. 3 interleaved pairs vs `13eb0aa`, ctx 1024 gen 128: prefill
528.7 -> 612.4 t/s median (+15.8%, disjoint), decode 81.24 -> 81.06 (noise).
Scope: a process's first prefill (bench, one-shot CLI, first server request);
a warm server already had the table. `evidence/raw/w-decode-reorder/ab-embed-async.csv`.

## K-quant shared-expert fusion (D3) — ceiling probe says NOT WORTH BUILDING (2026-10-27)

Probe: skip the whole shared-expert block at n_tok 1 (wrong output, timing
only). CyberTiel ctx 1024 gen 128, 2 interleaved pairs: 81.21/80.76 ->
84.98/84.65 t/s, i.e. **+4.5% is the ceiling for removing it entirely**,
after `13eb0aa` already overlaps it with the routed pass. A fused 2-dispatch
path still has to read the ~70 MB of Q4_K shared weights, so the reachable
part is ~1-2%. Not built.

## MTP verify: 2-row build of the K-quant rows kernel — ACCEPTED, +8.4% MTP generation (2026-10-27)

Profile of `--mtp-draft 3 --mtp-margin 2` on CyberTiel + APEX MTP head: almost
every cycle verifies 2 rows, and a 2-row verify cost 19.2 ms vs 11.8 ms for a
1-row forward. `dense_q4k_rows` alone was ~5.1 ms per verify vs ~3.0 ms for the
1-row decode kernel: `dense_q*k_rows.spv` is built with `P36_ROWS_TOK=4` and
computes all four token slots for any n_tok 2..4 (inactive slots alias the
last row). Added `dense_q{3,4,5,6}k_rows2.spv` (`P36_ROWS_TOK=2`), selected
when n_tok == 2. Per-row arithmetic is unchanged. Greedy MTP text (200 tok,
code prompt) byte-identical in all 6 runs; 3 interleaved pairs vs `8409f3a`:
generation 89.01/89.02/89.47 -> 87.66/97.06/96.53 t/s (median +8.4%; B's
first run includes the new pipelines' first build). `evidence/raw/w-mtp-rows2/`.
n_tok 3 still runs the 4-slot build; a 3-row variant is the obvious follow-up
if deeper drafts become the default.

## MTP draft head over the vocab prefix (FR-Spec style) — REJECTED, content-dependent (2026-10-27)

Draft head output projection + top-2 over only the first V vocab rows (Qwen
BPE ids are roughly frequency-ordered; K-quant output rows are contiguous).
Pitfall found: a row-prefix matmul that imports `row_bytes * V` as its own
weight buffer *slows* the head (V=65536: 1.70 -> 3.21 ms) because it competes
with the full-vocab buffer; binding the whole tensor and dispatching V rows
fixes it (1.72 -> 1.29 ms at 65536, 1.08 at 32768, 0.54 at 16384).
Not lossless here: batched verify is not bit-identical to serial decode, so
different drafts flip different near-ties and the greedy text changes (it
already differs from plain greedy at byte 256 with the full-vocab head).
CyberTiel + APEX head, draft 3 margin 2, V=32768 vs full: code prompt
98.26/86.98/104.29 vs 96.75/97.51/97.93 (median +0.8%, high spread), prose
82.96 vs 87.36 (-5%). Rejected. `reconsider_if`: verify becomes bit-exact
with serial decode, or a longer A/B shows a consistent gain across prompts.

## APEX MoE: GEMM from 2 tokens for small prefill batches — ACCEPTED, +93% prefill at chunk 64 (2026-10-27)

Prefill batches of 2..127 tokens on APEX expert recipes (chunk tails, short
agent turns, small prompts) took neither the identity decode pair nor the GEMM
(`apex_prefill`/`gemm` required n_tok >= `P36_VK_MOE_GEMM_MIN` = 128) and fell
back to the q8 `moe_matvec` route; a 36-token prompt prefilled at ~84 t/s.
The q8 route's reason (decode-matching numerics) no longer holds since APEX
decode runs the f32 pair. Outside MTP micro-batches the APEX GEMM now starts
at 2 tokens. Env probe first, CyberTiel prefill t/s q8 -> GEMM: chunk 64
148 -> 293, 32 123 -> 189, 16 88 -> 119, 8 64 -> 74, 4 70 -> 79.
Numerics (frontier 512 vs the chunk-256 all-GEMM reference): top-1 preserved,
top-64 57/57 (chunk 64) and 55 vs 57 (chunk 16), max_abs 1.83 vs 1.46-1.62 for
the old route — chunking-level drift, same GEMM k-order as accepted >=128
prefill. A/B 3 pairs ctx 512 chunk 64 vs `7127dfe`: prefill
150.31 -> 290.66 (+93%), decode flat. Occamy MiniPlus: chunk-64 prefill
176 -> 277, route misses 640 -> 0, decode equal (77.8/78.3 vs 77.4/78.3).
`evidence/raw/w-apex-small-gemm/`.

## K-quant mmq: skip FMAs in waves past n_tok (small prefill chunks) — REJECTED, -1.8% kernel time (2026-10-27)

At chunk 64 `dense_kquant_mmq_q4k` costs ~2.4x per token vs chunk 256 (128-token
tile, 4 wave32 x 32 tokens, half the waves on zero padding). Skipping the FMA
loop in waves whose slice is entirely past n_tok (wave-uniform, barriers kept)
is bit-identical (frontier logits equal at chunk 64 and 256) but moved the
kernel only 392.4 -> 385.3 ms (CyberTiel ctx 512 chunk 64). The per-workgroup
A dequant + staging, shared by all waves, is the cost; a narrower tile has to
re-amortize that, it is not idle-wave ALU. Reverted.

## PITFALL: candidate binaries load the *caller's* shaders (2026-10-27)

`p36_read_file` resolves `vulkan/*.spv` against `getcwd()` (`p36_vk.shader_root`)
**before** the binary's own directory. Running a worktree binary from the main
checkout therefore runs the main checkout's SPIR-V for every shader file that
exists there; only brand-new shader files come from the worktree. Every speed
verdict on a *modified* shader measured that way compared old vs old shader
code. `tests/bench_ab.sh` now pins each arm with
`P36_SHADER_DIR=$(dirname binary)`; set it by hand for any manual run
(`P36_SHADER_DIR=/path/to/worktree ./wt/p36 ...`).
Verdicts this session that are **invalid** (shader edited in place, measured
from the main checkout) and need a re-measure: parallel conv tiles (prefill),
MoE down split rebuild ("decode halved" = old fused shader on an 8x grid),
`add_rms_norm_q8_k` subgroup shuffles ("neutral"), mmq dead-wave skip
("-1.8%"). Their parity checks ran inside the worktree and stand. Unaffected:
13eb0aa, 8409f3a, 7c7863d (new .spv files), the vocab-prefix verdict, aa7dc67.

## MTP verify: identity expert kernels over all verify rows at once — ACCEPTED (2026-10-27)

A 2..4-row MTP verify ran the routed experts as one identity pass per row
(`P36_VK_MOE_F32_ROWS` loop): 2 dispatches per row per layer, every expert
row read once per verify row. The identity gate/up kernel already indexed the
token by `pair / n_used`; `moe_down_q2k_sum_decode` now takes the token from
`gl_WorkGroupID.y`, and the host sets identity for the whole micro-batch
(`multi_rows`), so a layer is 2 dispatches regardless of rows. Per-slot math is
unchanged: an in-run check (per-row loop vs batched, same inputs) gave
maxdiff 0 on every layer, and greedy MTP text (200 tok) is byte-identical in
all runs. A token loop inside the workgroup (cache reuse of shared experts) was
slower than grid.y (93.8/87.0 vs 94.6/97.1) and dropped. MTP generation,
CyberTiel + APEX head, draft 3 margin 2, 3 interleaved pairs vs
`P36_VK_MOE_ROWS_SHARE=0`: 94.73/91.35/77.13 -> 99.06/92.52/99.35 (median
91.35 -> 99.06; one off-arm outlier, ~+4% against the clean off runs). Plain
decode/prefill vs `aa7dc67` unchanged (-0.12% / -0.42%, `evidence/raw/w-mtp-multirow/`).

## MTP verify: read each shared expert once across verify rows (gate/up dedupe) — REJECTED, neutral (2026-10-27)

Measured first: over a 100-token MTP run, 16.7% of verify expert slots are
duplicates of an expert another verify row already picked (28,526 slots,
23,754 unique; mostly 2-row verifies). Built on top of `d696e02`: in
`moe_gate_up_decode` the workgroup owning an expert's first slot dequantizes
each weight once and accumulates every row that picked it (per-row
accumulators, single-token fma order, static-index row arrays); duplicate
slots exit. Bit-identical (MTP text md5 equal in all runs). MTP generation
3 pairs vs `d696e02`: 96.00/93.55/95.09 -> 94.15/95.25/96.89 (median +0.2%),
plain decode -0.76% (80.71/80.41/80.40 -> 80.54/79.77/79.80). The saved reads
are spent on the owner workgroup doing the shared rows serially; reverted.
The batched identity pass (`d696e02`) is the part of this lever that pays.

## MoE down sum: one workgroup per (row, slot) + ordered route reduce — ACCEPTED, +2.94% decode (2026-10-27)

Re-measurement of the 2026-10-27 split (+3.1%, reverted for the 5% bar) and of
this session's "decode halved" rebuild, which was the shader-dir pitfall above.
`moe_down_q2k_sum_decode` (all five builds; Q4_K keeps its own shader) now runs
grid (out_dim, n_slot), one routed slot per workgroup, writing `acc*scale` to
a per-slot partial row; `moe_down_route_reduce` replays `fma32(part, route,
sum)` per token in expert order. Replaces the fused path (no flag). Covers the
batched MTP verify too (n_slot = rows x 8). Bit-identical: decode-path and
prefill frontier logits plus greedy text on CyberTiel and Occamy, MTP text on
CyberTiel. 3 interleaved pairs vs `d696e02` (harness pinned to each arm's
shaders): decode 80.67/80.70/81.02 -> 83.43/83.07/83.00 (+2.94%, disjoint),
prefill +1.49% (noise). `evidence/raw/w-down-split/`.

## K-quant mmq dead-wave skip — re-measured: ACCEPTED, +5.9% chunk-64 prefill (2026-10-27)

Supersedes the "-1.8%, rejected" entry above, which ran the main checkout's
shader (shader-dir pitfall). With each binary on its own shaders,
`dense_kquant_mmq_q4k` at ctx 512 chunk 64 went 385.5 -> 313.4 ms (-18.7%).
Bit-identical frontier logits at chunk 64 and 256. 3 pairs vs `d696e02`,
ctx 512 chunk 64: prefill 293.97/288.06/288.08 -> 298.73/310.88/305.13
(+5.92%). `evidence/raw/w-down-split/ab-mmq-deadwave.csv`.

## MoE GEMMs: skip the padded half of the 32-slot tile — ACCEPTED, +4.1% chunk-64 / +3.6% chunk-256 prefill (2026-10-27)

`moe_gate_up_gemm` / `moe_down_gemm` give each thread a 4-row x 2-slot tile
(slots sl and sl+16 of a 32-slot expert tile). An expert usually gets <= 16 of a
batch's routed slots (chunk 256 x 8 / 256 experts ~ 8 per expert), so the sl+16
half is padding for the whole workgroup. `hi_live` (any slot 16..31 live, read
once from `sh_slot` after the staging barrier) selects the original FMA loop
or a copy without the b1 loads / [1] accumulations. Uniform, bit-identical
(frontier logits equal at chunk 64 and 256). Expert GEMM GPU time at ctx 512
chunk 64: 1116 -> 1013 ms (down IQ3_XXS -18%, gate/up -7..8%). 3 pairs vs
`4f0ab41`: chunk 64 308.36 -> 321.04 (+4.11%), ctx 1024 chunk 256
598.21 -> 619.97 (+3.64%), decode flat. `evidence/raw/w-moe-gemm-half/`.

## Inter-token gap: chained decode with the token id kept on the device — RE-MEASURED, neutral; GPU sampling not built (2026-10-27)

Re-ran the stashed "C: lagged decode loop" (`stash@{0}`) on `46df8b5`, branch
`experiment/gpu-chain` (`p36-wt/chain`): token i+1 is recorded with its
embedding row read from token i's device top-2 buffer (`embed_row` ids
binding), the host seals the batch and later waits only for token i's
submission (`p36_gpu_tensor_read_begin/_wait`), so the GPU always has the next
token queued. Two fixes over the stash: `argmax_excluding` / `token_rank` now
resolve a pending read (they silently fell back to a 1 MB logits read), and
deferral is gated to the chained loop (`s->defer_top2`) because deferring in
the unchained bench loop cost -0.49%. Output identical (CLI greedy 256 tok md5
equal; bench parity, MTP text unchanged). CLI `--temp 0`, CyberTiel, 6
interleaved pairs vs `46df8b5`: A 85.59/84.59/83.70/85.49/83.00/85.48,
B 87.75/85.49/87.00/85.68/83.67/85.24 (pooled medians ~85.0 vs ~85.6,
+0.7%, inside noise); unchained bench -0.11%.
Verdict: with the next token always queued the GPU-idle round trip buys
nothing measurable here, so GPU-side sampling for temp > 0 (the same
dependency removal, plus a sampler port) is not worth building. Kept on the
experiment branch, not merged. `reconsider_if`: a profile without timestamp
brackets shows a real inter-token GPU gap (the 0.6-0.76 ms figure comes from
timestamp-instrumented runs, which add ~1.9 us per bracket).

## Narrower tiles for short prefill chunks — BUILT, REJECTED (2026-10-27)

- **Trunk, 64-token K-quant mmq tile** (`experiment/mmq-bn64`, `P36_MMQ_BN64`,
  64 threads = 2 wave32, same 4x8 per-thread tile, picked for n_tok <= 64):
  bit-identical (chunk-64 frontier logits equal) but `dense_kquant_mmq_q4k` at
  ctx 512 chunk 64 went 312.3 -> 346.0 ms (+10.8%) vs `2050825`. The weight
  tile staged per workgroup is shared by half the tokens; the dead-wave skip
  (`4f0ab41`) removes the padded FMAs without that cost.
- **Experts, 16-slot tile**: not built separately. For a tile with <= 16 live
  slots the shipped `hi_live` path (`46df8b5`) already runs exactly the
  16-slot computation; a 16-slot tile would only add a second staging of the
  same expert weights whenever an expert gets > 16 slots.

## MTP verify: per-expert share build re-tried with decode isolated — REJECTED (2026-10-27)

`experiment/mtp-share` (`a34f4fd`): the dedupe moved into separate
`moe_gate_up_share*.spv` builds used only for 2..4-row identity batches, so
decode keeps its kernel (decode +0.25%, bit-identical). MTP generation vs
`4f0ab41`, 3 pairs: 99.53/101.07/102.10 -> 101.64/99.54/99.52 (median -1.5%,
overlapping). Reading a shared expert once does not pay at a 16.7% duplicate
rate; the owner workgroup serializes the shared rows.

## Inter-token gap: GPU-side sampling for chained sampled decode — BUILT, neutral (2026-10-27)

`experiment/gpu-chain` (on top of the chained greedy loop): `minp_sample.comp`
(256 threads, per-thread block ranges + ordered scan) draws the next token from
the `logits_minp_pack` survivors with uniform draws the host precomputed from
the session RNG in host order, and writes the id into the top-2 buffer that
`embed_row` reads for the next token; `p36_session_gpu_sample_begin` arms it in
the forward pass and the CLI's default min-p sampler loop chains on
`p36_session_eval_device_next`. Cost ~21 us/token on the device (pack ~10 +
draw ~11). CyberTiel, `--temp 0.6 --top-k 0 --top-p 1 --min-p 0.05 --seed 42`,
256 tokens: text identical to the host sampler in all 6 runs; 3 interleaved
pairs host 85.92/81.94/84.50 vs device 84.02/83.95/86.23 (median -0.6%).
Same finding as the greedy chain: with the next token queued there is no
measurable inter-token GPU gap left on this board. Not merged.

## MoE GEMMs: 16-slot expert tiles for <= 64-token batches — ACCEPTED, +3.2% chunk-64 prefill (2026-10-27)

The narrow-tile half of the short-chunk item, for the expert GEMMs. `P36_SLOT_TILE`
in `moe_gate_up_gemm.comp` / `moe_down_gemm.comp` (store loop bounded by the
tile), ten `*_t16.spv` builds, and the host picks `pair_tile = 16` plus the t16
kernels when a GEMM batch has <= 64 tokens. All-16 experiment
(`experiment/moe-tile16`): chunk 64 +1.72%, chunk 256 -1.16% (an expert with
> 16 slots is staged twice) — hence the n_tok gate. Shipped selection vs
`2050825`, bit-identical frontier logits at chunk 64 and 256, 3 pairs:
ctx 512 chunk 64 313.83/318.96/320.24 -> 329.23/327.90/329.25 (+3.22%,
disjoint); ctx 1024 chunk 256 +0.82% (32-slot path, noise).
`evidence/raw/w-moe-tile16/`. The trunk's 64-token mmq tile stays rejected
(`experiment/mmq-bn64`, +10.8% kernel time).

## MTP verify: read each shared expert once in gate/up AND down — REJECTED, -8% (2026-10-27)

`experiment/mtp-share` (`b7dfa86` + next): share builds of both
`moe_gate_up_decode` and the split `moe_down_q2k_sum_decode`, used only for
2..4-row identity batches. The owner slot applies each dequantized weight to
every verify row that picked the expert and writes each slot's partial;
duplicates exit. Bit-identical (decode logits/text, MTP text), but MTP
generation 98.97/98.57/100.38 -> 90.95/92.70/81.86 (~-8%) vs `6c11c8c`. In the
split down kernel each workgroup is tiny (one row x one slot), so the
duplicate scan is a large fixed cost per workgroup; the 16.7% duplicate reads
it saves do not cover it. Every "read each shared expert once" variant tried
(gate only: -1.5%, gate+down: -8%) loses to plain batching (`d696e02`).

## Shader lookup now prefers the binary's directory (2026-10-27)

Follow-up to the shader-dir pitfall: `p36_read_file` resolves `vulkan/*.spv`
as `P36_SHADER_DIR` -> the binary's own directory -> the current directory ->
the raw path (the cwd used to win over the binary's directory). p36, p36-agent
and p36-server now always run the SPIR-V built with them. Checked from an
unrelated cwd: p36 CLI greedy 128 tok 83.39 t/s, p36-server 256-token greedy
chat completion 88.8 t/s (2883 ms decode), p36-agent non-interactive turn OK;
all three profiles show `moe_down_route_reduce` / the split down sum.

## 2026-10-28 Dense 27B (Qwen3.8 APEX-I-NanoPlus-V3): tiles + recurrent barriers

Model `Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.VAL-APEX-I-NanoPlus-V3.gguf`
(FFN IQ3_XXS/IQ4_XS, trunk Q3_K..Q5_K, ssm_beta Q8_0, ssm_alpha F32, in-file
MTP head). 1329-token prefill ctx 2048, 128-token decode ctx 1024, 3
interleaved reps, greedy text identical in every arm.

* **ACCEPTED: 64-row `dense_kquant_mmq` tile** (lane owns 8 rows, sum[32] as in
  the IQ3 pair kernel). Prefill 157.6 -> 163.9 (+4.0%). Bit-identical on
  dense 27B, CyberTiel, Occamy, Qwen3.6 NanoPlus (per-output FMA chain
  unchanged). 128 rows: 155, slower.
* **ACCEPTED: 64-row `dense_iq3_xxs_mmq` tile** (FFN down). 163.6 -> 166.4
  (+1.7%), bit-identical.
* **ACCEPTED: recurrent alpha/beta/gates off the barrier chain.** ssm_beta and
  ssm_alpha issue before the conv step (they read only `inp`), delta_gates
  right after conv, and on dense models recur_alpha/beta/gb get their own
  buffers instead of aliasing the ffn_shared_gate scratch root (the tracker
  saw WAW on the shared root). -144 barriers/token; decode 22.97 -> 23.35
  (+1.65%), bit-identical; CyberTiel identical, speed within noise.
* **MTP works here:** `--mtp-draft 3 --mtp-margin 2` = 27.5 t/s vs 23.35
  plain (+18%). draft 2 = no gain, draft 4 = 24.2.
* **REJECTED: Q3_K scales precomputed in LDS** - neutral (bit-identical).
  Ablating Q3_K dequant entirely only gains 2.5%: the "slow Q3_K mmq" in the
  profile is the barrier-free Q4_K attn_gate overlapping into its bucket.
* **REJECTED: narrow Q8_0 GEMM for out_dim < 64** (ssm_beta 5120x48 prefill,
  one lane per row, 4 tokens per WG, bit-identical) - neutral; its 274 ms
  profile bucket is overlap, not real cost.
* Not done: gdn_front for F32 ssm_alpha (would fold conv+l2+alpha+beta+gates
  into one dispatch; changes dot order, est. 1-2%).
