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
  (`ptq1_0.glsl`, `dequant_ptq1_0.comp`) to port. Blockers: q36 has no type-143
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


## IQ2_M experts land on the fused IQ2_S path (2026-09-20, accepted)

Closes the 2026-09-17 dispatch item above: the per-quant prefill GEMM family now
has IQ2_S members, so `Qwen3.8-35B-A3B-IQ2_M` stops falling into `moe_matvec`.

- **What changed.** `moe_gate_up.comp` (q8_K activations), `moe_gate_up_gemm.comp`
  and `moe_down_gemm.comp` gain an IQ2_S decode, built from the *same* sources with
  `-DQ36_MOE_IQ2S` into `moe_gate_up_iq2s.spv` / `moe_gate_up_gemm_iq2s.spv` /
  `moe_down_gemm_iq2s.spv`. Host side: `q36_vk_moe_gate_up_iq2_swiglu` accepts
  IQ2_S pairs at 82 bytes/block, and `q36_gpu_moe_ffn_f32_tensor` accepts IQ2_S
  gate/up/down with 82-byte `gu_stride`/`down_stride` plus a sixth `tables`
  binding for the down GEMM. The IQ2_XXS/Q2_K builds are **byte-identical** to the
  pre-change binaries (`b46fa81a…` gate/up GEMM, `93a38508…` down GEMM,
  `601d56a9…` q8 gate/up), so the guard is untouched *by construction*, not by
  measurement.
- **Measured (7-rep interleaved, soak-gated entry ≤55 °C, actual 52-56 °C, sclk
  500 MHz, ctx 512, gen 128, `Q36_VK_MOE_GEMM=0 Q36_VK_MOE_GATE_UP=0` as arm A):**
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
  `Q36_VK_MOE_GEMM` but **not** `Q36_VK_MOE_GATE_UP`, so it can pass while comparing
  the new kernel against itself; with a 6-token prompt `gemm` is off anyway. The
  gate used was the frontier logits dump with `Q36_VK_MOE_GEMM=0
  Q36_VK_MOE_GATE_UP=0`.
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
  with `-DQ36_MOE_IQ2S` (`moe_gate_up_decode_iq2s.spv`,
  `moe_down_q2k_sum_decode_iq2s.spv`; the down one gains a 7th `tables` binding).
  The host predicate is now `iq2s && !(gemm || (identity && down_sum_decode))` — the
  2..31-token small-batch kernels (`moe_gate_up_f32b`, `moe_down_q2k_f32b`) are still
  IQ2_XXS/Q2_K-only and must not see 82-byte blocks.
- **Measured (7-rep interleaved, soak-gated entry ≤55 °C, ctx 512, gen 128; arm A =
  `Q36_VK_MOE_DOWN_SUM_DECODE=0`, i.e. the previous default posture on the same
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
  `Q36_IQ3XXS_ABLATE` (keep loads, drop decode ALU): baseline production decode
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

- **Tested via `Q36_VK_DELTA_COL_PREFILL=1`**: 7-rep interleaved A/B at ctx 1024,
  gen 16. Prefill **170.77 (MAD 0.500) → 166.54 (MAD 0.500) = -2.48%** (FAIL vs
  ≥+1.5% gate, negative in all 7 pairs), decode **18.06 → 17.93 = -0.72%**.
  Ctx 512 diagnostic corroborates: **184.32 → 171.04 = -7.20%**.
- **Latent correctness bug found and fixed:** `delta_net_cols` hard-assumes
  wave32 (`gl_SubgroupID` 0..3 for 4 columns, `gl_SubgroupInvocationID` 0..31 for
  rows), but `q36_vk_force_wave32` previously matched `"delta_net_cols.spv"`.
  When Swift runs f16 recurrent state, the dispatched pipeline is
  `delta_net_cols_f16.spv`, which missed the pattern and ran at BC-250's native
  wave64 (only 2 subgroups), skipping half the state columns and writing OOB up
  to row 159 (row size 128), producing catastrophic parity failure (`max_abs`
  19.12, top-1 flip 13 vs 29, top-64 overlap 3/64). Widening the pattern in
  `q36_vulkan.c` to `"delta_net_cols"` restored parity (`max_abs` 0.767, top-1
  identical 13/13, top-64 61/64).
- **Verdict: REJECTED & FROZEN.** `delta_net_cols` is slower on BC-250 than the
  native wave64 register recurrence (`delta_net_decode_reg_f16`). The default
  guard `!q36_vk.bc250` at `q36_vulkan.c:1225` is verified correct and remains
  frozen. The wave32 force fix is committed as a latent correctness fix.
- Raw evidence: `karpathy/evidence/raw/ab-w3-deltacol.csv`,
  `ab-w3-deltacol-summary.txt`, `ab-w3-deltacol-parity.txt`,
  `ab-w3-deltacol-w32-parity.txt`, `ab-w3-ctx512-diagnostic.txt`.

## W4 — Wave32 clean-scan shaders A/B — REJECTED, NEUTRAL/NO WIN (2026-09-20)

Branch `trackB-ptq1_0`, models `Qwen3.8-35B-A3B-IQ2_M.gguf` and Guard MoE.
Full write-up: `karpathy/evidence/raw/W4-VERDICT.md`.

- **Tested via `Q36_VK_WAVE32_CLEAN=1`**: forcing wave32 across the clean-scan
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
- **Verdict: REJECTED & FROZEN.** Default remains separate dispatches (`Q36_VK_FUSED_RMS_Q8=0`, gated on `env && env[0] == '1'`). Scaffolding retained behind opt-in flag.
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
  up w, gate out, up out, tables, b16) + `q36_gpu_matmul_iq3_xxs_pair_mmq_tensor()`
  in `q36_vulkan.c`, called from `q36_forward_ffn_vulkan_model` (`q36.c`).
  BM/BN/BK, per-matrix accumulation order and store mapping are the unfused
  kernel's, so each element is bit-identical. `sum[32]` costs 16 VGPRs, +2176 B LDS.
- **Coverage 94/110 layer-instances (85.5%)**: the 16 that stay on the single path
  are genuinely mixed-type (IQ3_XXS gate with a non-XXS up) or IQ3_S/IQ4_XS, which
  one decoder cannot serve.
- **Parity is bit-exact**: frontier 513 on Swift-27B, pair off vs on, single
  binary — `max_abs_diff = 0` over all 248,320 logits, argmax 5316 (19.820942)
  both arms, top-64 overlap 64/64.
- **7-rep interleaved A/B on Swift 27B (ctx 1024, gen 16, chunk 256)**, one
  binary, wrappers differing only in `Q36_VK_DENSE_IQ3_PAIR`:
  - Arm A (pair off): prefill **169.25 tok/s** (MAD 0.140), decode **18.12 tok/s** (MAD 0.100).
  - Arm B (pair on): prefill **173.01 tok/s** (MAD 0.440), decode **18.22 tok/s** (MAD 0.080).
  - Delta: prefill **+2.22%** (PASSES the dense ≥ +0.70% gate, 3.2x it), decode
    **+0.55%** (inside the 9.4% decode spread — neutral). Arms do not overlap:
    max A 170.48 < min B 171.90.
- **Attribution** (ctx 512, gen 0, `Q36_VK_PROF_*`): gate/up 937.6 -> 858.9 ms
  (**-8.4%**), mmq 1892.9 -> 1089.5 + 728.0 = 1817.5 ms (-4.0%), whole-run kernel
  total 2792.7 -> 2670.9 ms (-4.36%). The spv dispatch drop 510 -> 322 is exactly
  the 188 removed singles (94 pairs x 2).
- **Verdict: ACCEPTED, default ON** (`Q36_VK_DENSE_IQ3_PAIR=0` disables).
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
- **Audit Steps**: For every MMQ shape executed in Swift-27B prefill, computed `grid / (40 CU * resident WGs/CU)` across 40 CUs for both realistic occupancy models ($R=2$, 80 GPU slots; $R=4$, 160 GPU slots). Measured actual GPU time per shape via `Q36_VK_PROF_SHAPE`.
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
`Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`.

- **HANDOFF-iq2s-20260920.md's ranked item #1 was misdiagnosed and this
  supersedes it.** It attributed the `dense_kquant`/`matmul_kquant.spv`
  hotspot (48.2% of IQ2_M decode, 10.7 ms/tok at the time) to "non-expert
  IQ2_S tensors" and proposed writing a new IQ2_S decode shader. Reproducing
  the differential profile on current HEAD (`Q36_VK_PROF_KERNEL=1`, gen 1 vs
  gen 65, ctx 512) still showed **9.453 ms/tok, 41 dispatches/tok** — the
  figure hadn't moved across W2-W7. Parsing the GGUF header directly
  (`gguf-tools`-style tensor listing) instead of trusting the label showed
  the 41 tensors are **40x `Q4_K` `blk.N.attn_qkv.weight` + 1x `Q5_K`
  `output.weight`** — real K-quants, not IQ2_S at all.
- **Root cause**: `q36_gpu_matmul_k_quant_q8_scaled_tensor()` (`q36_vulkan.c`)
  has tuned decode (`n_tok==1`) and mmq (`n_tok>1`) branches that dispatch
  `dense_kquant_decode`/`dense_q4k_decode`/`dense_q5k_decode`/
  `dense_kquant_mmq` — kernels that read only the one tensor's own
  `in_dim`/`out_dim`/`blocks`/`row_bytes`/`type`/`scale` from push constants,
  nothing model-wide. Both branches were gated on `q36_gpu_dense_model`, true
  only for the dense Swift-27B/UD-IQ3_S launches (`q36_gpu_set_dense_model`
  is called once per engine open with `Q36_MODEL_DENSE`). Nobody widened it
  when a MoE model started shipping its own K-quant trunk tensors, so IQ2_M's
  Q4_K/Q5_K rows fell through to the generic AVX2-style 8-rows-x-8-lanes
  `matmul_kquant` fallback unconditionally, on both prefill and decode.
- **Fix**: `q36_vulkan.c`, two-line change — drop `q36_gpu_dense_model &&`
  from both branch conditions. Also widened `q36_gpu_set_dense_model()` to
  always call `q36_vk_prepare_dense_kernels()` (was `if (dense)` only), so
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
  handful of tokens the way a real bug would look. `./q36_test
  --vulkan-kernels` (CPU-reference dot-product oracle, tolerance 2e-3) also
  passes and, for the first time, actually exercises this decode path — it
  was dead code in that test before this change since `q36_gpu_dense_model`
  was never set true there.
- **7-rep interleaved A/B** (`tests/bench_ab.sh`, ctx 512, gen 128, matching
  HANDOFF's own IQ2_M convention): prefill **244.35 (MAD 2.360) -> 507.87
  (MAD 9.130) = +107.85%**, decode **47.34 (MAD 0.410) -> 75.56 (MAD 0.130) =
  +59.61%**. Gate is MoE prefill >=3.4%; cleared by ~32x. Both arms show
  tight spread and no overlap.
- **Verdict: ACCEPTED, default ON** (same env vars as before,
  `Q36_VK_DENSE_KQUANT_DECODE=0` / `Q36_VK_DENSE_KQUANT_MMQ=0` still disable
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
Q36-IQ2XXS.gguf` (MoE), `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense).

- **Step 1, per the plan (`PLAN-implementation-2026-09-20.md` §W8): measure
  attention's prefill share at ctx 4096/8192 before touching any shader.**
  `Q36_VK_PROF_KERNEL=1 ./q36-bench --vulkan -m <model> --prompt-file
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
  `Q36_VK_MOE_ROUTE_DEBUG=1` print in `q36.c` fires on every fused-f32-path
  miss and pins the residue exactly: layers **0, 1, 2 only**, on both prefill
  and decode, with `gate=22 up=22 down=21` (22 = `IQ2_S`, 21 = `IQ3_S`).
  `q36_gpu_moe_ffn_f32_tensor()` required all three expert tensors to share one
  quant, and those three layers ship an **IQ3_S down under IQ2_S gate/up**
  (GGUF tensor table: `ffn_down_exps` IQ2_S x37, IQ3_S x3 at il 0-2, Q8_0 x1 at
  il 40 = the MTP block). They fell to the q8_K route, whose
  `moe_iq2s_gate_up` + `moe_matvec` + `moe_tiles` cost 19.6% of profiled
  prefill GPU and 6.8% of decode per token.
- **Fix**: `-DQ36_MOE_IQ3S` builds of `moe_down_q2k_sum_decode.comp` and
  `moe_down_gemm.comp` (`moe_down_q2k_sum_decode_iq3s.spv`,
  `moe_down_gemm_iq3s.spv`) + host wiring: `iq2s` split into `gu_iq2s` /
  `down_iq2s` / `down_iq3s`, `down_stride` 82 or 110 bytes, the down dispatch
  site chooses the variant. IQ3_S block `[d f16][64 qs][8 qh][32 signs][4 scales]`
  = 110 B, 9-bit grid index (8 qs bits + 1 qh bit per 4 weights) against the
  IQ3_S grid already in the shared IQ tables buffer at word 2592, scale
  `d*(2*nib+1)` — same convention as `dense_iq3_s_decode.comp`. All new code is
  inside `#ifdef Q36_MOE_IQ3S`; IQ2_S/legacy builds stay byte-identical.
- **Parity** (frontier dumps vs the `e6c15ba` binary, 248,320 logits): IQ2_M
  frontier 512 (`max_abs` 0.762, mean 0.112, top-1 match, top64 62/64);
  frontier 513 (0.865, mean 0.138, top-1 match, top64 61/64); guard frontier
  513 **sha256-identical, `max_abs` 0.0** (the guard has no IQ3_S experts, so
  this is a pure no-op for it). Deltas are the same order and *shape* as W1's
  own accepted calibration (0.907/62-64) and smaller than it; spread across
  ~all logits (7.7%/15.0% of logits above 0.25, max ~2x p99), not localized.
- **7-rep interleaved A/B** (`tests/bench_ab.sh`, ctx 512, gen 128,
  `Q36_AB_MIN_GAIN=3.4`): prefill **497.00 (MAD 15.150) -> 556.21 (MAD 12.400)
  = +11.91%**, decode **75.26 (MAD 0.180) -> 78.40 (MAD 0.100) = +4.17%**
  (decode separates rep-by-rep, no overlap). Gate is MoE prefill >= 3.4% ->
  PASS, ~3.5x. `./q36_test --vulkan-kernels` OK; `compat_gate.sh` PASS.
- **Verdict: ACCEPTED, default ON** (no new env var; type-driven inside the
  existing fused path, `Q36_VK_MOE_F32B=0` still disables it).
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
  `q36_gpu_matmul_iq2s_pair_scaled_tensor()` (`q36_vulkan.c`) fuse both into
  one dispatch sharing one Q8_K read, mirroring the existing
  `q36_gpu_matmul_q8_0_pair_scaled_tensor` pattern (same shape, IQ2_S
  instead of Q8_0). Wired into `q36_forward_recurrent_vulkan`'s existing
  `pair_projected` fallback chain (`q36.c`), gated on both tensors being
  exactly `IQ2_S` — inert by construction for any model where that's not
  the type (e.g. the guard, which is IQ2_XXS-based; not separately
  A/B'd for that reason). Default on, `Q36_VK_IQ2S_PAIR=0` disables.
- **Parity: bit-exact.** Frontier-512 first-decode-token logits,
  `Q36_VK_IQ2S_PAIR=1` vs `=0`, same binary: `max_abs_diff = 0.0` across all
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
- **Binaries built in isolated worktrees** (`/home/server/q36-wt/
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
  `Q36_Q2_0_ONLY`, precedent cited in the source comments: "-19..28% on
  dense_extra_decode_q2_0" from a LUT). Added the same treatment for IQ2_S:
  a `Q36_IQ2S_ONLY` `#ifdef` in both files producing
  `dense_extra_decode_iq2s.spv` / `dense_extra_mmq_iq2s.spv`, math copied
  verbatim from the existing `pc.type == IQ2_S` arm (same block layout,
  same `signed_grid`/`tables` lookups, same accumulation order), just
  without the runtime type-branch chain. Wired into
  `q36_gpu_matmul_iq_quant_q8_scaled_tensor`'s decode/mmq kernel selection,
  gated on `weight_type == IQ2_S` and `Q36_VK_IQ2S_EXTRA` (default on).
  Both new kernels added to `q36_vk_prepare_dense_kernels()`'s prewarm list
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
  before and after — the new `Q36_IQ2S_ONLY` branch is additive, the
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
  profile (`Q36_VK_PROF_KERNEL=1`, gen 1 vs gen 65 delta/64) isolates the
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
  benchmark. Default on, `Q36_VK_IQ2S_EXTRA=0` disables.
- A/B'd in isolated worktrees (`q36-wt/iq2s-tuned-kernel{,-base}`) built
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
  word 544) into both kernels behind `#ifdef Q36_MOE_IQ2S`, added the `tables`
  binding the down kernel needed (6 vs 5), and widened the predicate to
  `if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;`.
  Coverage is `n_tok` 2..127, wider than HANDOFF's stated 2..31
  (`Q36_VK_MOE_PAIR_TILE = 8`, `Q36_VK_MOE_GEMM_MIN` default 128).
- **Scope is `down_iq2s`-only on purpose**: the IQ3_S-down layers (`il=0,1,2`,
  entry 1c) still fall back to q8 outside the GEMM range, so this change cannot
  put an IQ2_S kernel in front of a 110-byte block.
- **Route-miss evidence that the change is live, not the A/B's assumption**
  (`Q36_VK_MOE_ROUTE_DEBUG=1`, ctx 512, `--prefill-chunk 16`, `--gen-tokens 0`):
  A `44143d8` = **1280** misses, B = **96**, and the surviving 96 are exactly
  `il=0,1,2 × 32 forwards` — i.e. 37 IQ2_S-down layers × 32 forwards moved onto
  the new kernels. Remaining misses belong to 1c's scope, not this one.
- **A bug was caught before the benchmark, by measurement not by review.** The
  first draft hand-derived the IQ2_S slotting and was wrong — `max_abs` 7.86 vs
  the q8 route, top64 41/64. Isolating it at `n_tok=1` by forcing identity
  forwards (`Q36_VK_MOE_DOWN_SUM_DECODE=1` vs `=0`) pinned the fault to the down
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
  mechanism for a regression (`Q36_VK_MOE_DOWN_SUM_DECODE` defaults on, so the
  1-token pair was already admitted) and both deltas sit inside the 9.4% spread.
- Base builds byte-identical: `vulkan/moe_down_q2k_f32b.spv`
  `0ce317941dbef148…` and `vulkan/moe_gate_up_f32b.spv` `9701f0330b5b5d73…`,
  equal to the same files built from clean `44143d8`, so every non-IQ2_S build
  variant is bit-for-bit unchanged. `compat_gate.sh` -> PASS.
- **Method notes worth reusing.** (1) `--prefill-chunk 16` is the only way the
  existing harness can be made to dispatch `n_tok ∈ 2..31` at all; a chunk-256
  A/B would have measured nothing and looked like a null result for a correct
  change. (2) The baseline must be a worktree at the *candidate's* HEAD, not a
  stale `/tmp/q36-*` copy: the first attempt used a `b738610` baseline and would
  have folded the other session's -2.01% `ssm-pair` prefill into this delta.
  (3) `tests/bench_ab.sh` has no `flock`, so its bare `pgrep` check bounces
  whenever a *parallel instance* runs a benchmark — two attempts failed that way
  before the run was queued behind `/tmp/q36-gpu.lock`.
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
  untouched. `Q36_VK_ATTN_FA=0` falls back. Branch `perf/attn-prefill-fa`.
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
  Host takes it for `n_tok <= Q36_VK_Q2_SMALL_MAX` (default 16), and for a
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
  was `Q36_VK_ATTN_DECODE_GQA`; only `n_tok == 1` reaches the split-K path, so
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
  `Q36_VK_MOE_DOWN_SUM_DECODE=0` (falls back to the per-expert parallel down
  kernels) vs default: decode **70.65/71.07/70.95** vs **74.46/74.42/74.39 t/s**,
  prefill unchanged.  The serial 8-expert combine with one `subgroupAdd` per
  expert already wins by 4.8%; the audit's "experts in parallel" suggestion is
  dead in that form.
- **Probe D — load shape — POSITIVE, 3.23x.** A diagnostic IQ2_S body that
  keeps the workgroup/block/expert mapping but reads the whole 82 B superblock
  with **3 lane-strided 2-byte loads** (and does no dequant, no mid, no
  tables - `Q36_P1_LOADFLOOR`, `diag.comp` + `p1-diag-probes.diff`) runs
  **97.10/97.74/97.69 ms** against the stock **314.88/315.51/315.52 ms** for the
  *same 12.72 GB*: 131 GB/s achievable vs 40.6 GB/s delivered.  End to end the
  run drops 1465 -> 1246 ms of decode GPU, decode 74.4 -> 85.0 t/s.
- **Probe E — ALU is only 51 ms.** The same diagnostic with the stock ~40
  scattered field loads kept verbatim and the dequant arithmetic collapsed to an
  add (`Q36_P1_ALUFLOOR`) runs **263.93/264.40/264.53 ms**, so
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
  `Q36_MTP_STATS` `drafted` count gives the exact denominator).  The audit's
  >= 60% P7 gate is met.
- **CORRECTION — the `q36: MTP stats ... accept=100.0%` line is not an
  acceptance rate at the default setting.**  `q36_mtp_stats_add(verify_n,
  commit_n, ...)` is only called on calls that carry a draft, and
  `--mtp-draft N` sets `draft_cap = N - 1` (`q36.c:12366`), so at the published
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
  committed per call* (`1 + commit_n`, `q36.c:12483`), so solving the two
  `--mtp-margin 0` arms (340 calls: 168 committing 1, 172 committing 2, 24.22 s;
  452 calls: 392 / 60, 24.29 s) gives **47.5 ms per one-token spec call and
  94.4 ms per two-token spec call = 47.2 ms/token, against 44.19 ms for a plain
  step**.  The piggybacked draft head is only +3.3 ms (7.5%); the loss is that
  two forwards buy two tokens.  Depth 1 is therefore break-even at best and a
  loss after the draft head — it cannot be tuned into a win, only kept from
  hurting.
- **The default margin gate is pure cost.**  `--mtp-margin 0` beats the shipped
  margin 3 on tokens/call by 33% with no measurable decode change; the
  `s->mtp_draft_token != q36_session_argmax(s)` check that cannot be turned off
  is already the free, exact filter.  Backoff then spends 1-4 draftless tokens
  after each miss, which is why only ~47% of calls carry a draft at margin 0
  (172/340) and ~13% at margin 3 (60/452).
- **Depth 2+ collapses: `--mtp-draft 3` (draft_cap 2) = 4.17 t/s and
  `--mtp-draft 4` (draft_cap 3) = 4.44 t/s** against 22.63 plain, while still
  committing 1.910 / 1.673 tokens per call and with prefill unchanged at
  ~187 t/s.  The same algebra puts a two-draft call at ~1.1 s, i.e. ~0.4 s per
  committed token, 8-9x a plain step; the direct evidence is the per-kernel
  decode profile in `evidence/raw/p6-smallbatch/`.  Mechanism: for IQ3_XXS only
  `n_tok == 1` is given a decode kernel (`q36_vulkan.c:8687-8730`), so the
  moment the verify carries a *second* row it takes `dense_iq3_xxs_mmq` (a
  `ceil(n_tok/128)` grid) plus a `predequant_b16` pass — the 128-token prefill
  tile.  `q36_vk_micro_batch` exists but is only wired for Q8_0
  (`q36_vulkan.c:5173`).  This is the audit's P6, measured from the outside.
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
(`Q36_VK_ATTN_DECODE_GQA`), wiring reverted out of the tree again afterwards.

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
ledger has no entry for either dtype, and `Q36_VK_RECURRENT_STATE_F16` was
landed on throughput with no quality probe behind it.  Mechanism under test:
`vulkan/dense_iq3_xxs_mmq.comp` accumulates the whole K dimension (up to 17408,
68 blocks) in `f16vec2 sum[16]` while the decode path accumulates in f32, and
the DeltaNet recurrent state can be stored f16 or f32.

- **Method.** `q36-bench --gen-tokens 0 --dump-frontier-logits-dir` walks
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
- **Arm B, f16 vs f32 recurrent state (`Q36_VK_RECURRENT_STATE_F16=0`):** the
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
  statistic files, the eleven `q36-bench` logs, and the sweep scripts/outputs
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

- **Mechanism.** `q36_vulkan.c:8774-8813` gives `n_tok == 1` a purpose-built
  decode kernel and sends everything else to the 128-row MMQ tile
  (`dense_iq3_xxs_mmq` + `predequant_b16`).  A 2-row step therefore pays a whole
  128-row tile.  The new kernel unpacks each weight word once and walks the two
  activation rows through it: the loads and the grid/sign work are shared, only
  the FMA chain is duplicated.  Per-row operand order is identical to
  `dense_iq3_xxs_decode_r4`, so each row is *bit-identical* to a separate
  one-token dispatch — that is the parity gate, not a max_abs tolerance.
- **Landed.** `vulkan/dense_iq3_xxs_decode_nx.comp` (local_size 64, ROWS 4,
  NTOK 2), `tests/test_dense_iq3xxs_nx.c`, and a `q36_vulkan.c:8814-8835`
  branch gated on `weight_type == IQ3_XXS && out_dim % 4 == 0 && n_tok == 2 &&
  Q36_VK_DENSE_IQ3_NX` (default on; `=0` restores the MMQ tile).
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
  byte-for-byte has to keep `Q36_VK_DENSE_IQ3_NX=0`; (b) only `n_tok == 2` was
  built.  The kernel loops over `t0 < n_tok` in steps of 2, so 4/6/8 rows is a
  dispatch-condition change, deliberately not made (audit scope was 2..8; the
  measured target was the single 2-row step); (c) the guard is untouched by
  construction — the branch is Q36_VK_TENSOR_IQ3_XXS-only and the guard's tensors
  are IQ2XXS, confirmed by `dense_iq3_xxs_decode_nx` never appearing in its
  profile.
- **Gates.** `karpathy/compat_gate.sh` **PASS** (Swift 1024/177.23 prefill tps +
  guard).  `./q36_test --vulkan-kernels` fails, but **on an attention test:**
  `tests/q36_test.c:4958` (`memcmp(out_host, out_single)` after replaying each
  batch row of `q36_gpu_attn_decode_tensor` as a decode step).  Reproduced on
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
path at `q36.c:8278-8320` and fall through to `q36.c:8337-8361`, three matvecs +
swiglu + one mid `q8_k_quant`).  What was measured is that the dispatch count is
right and the cost model behind it is not.

- **Method.** `Q36_VK_PROF=1 Q36_VK_PROF_SHAPE=1` on IQ2_M, ctx 512, gen 128 —
  per-shape op rows, and `q36_vk_prof_iq3_shape` (`q36_vulkan.c:700-733`) names a
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
- **Cheaper capture point identified, not taken.**  `q36_gpu_matmul_iq2s_pair_scaled_tensor`
  (`q36_vulkan.c:8429`) already pairs two IQ2_S projections off one Q8_K input,
  and is already used for `ssm_beta`/`ssm_alpha` (`q36.c:8722`).  The shexp
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
claims q36's 23.19 t/s "carries the full temp-0.6 / top-k / top-p / min-p sampling
chain" against llama-bench's greedy `tg`, i.e. that the sampler costs something —
never measured. The two closed-line entries that touch readback
(`MOVNTDQA streaming loads`, `HOST_CACHED readback lever`) are about the readback
*buffer type*, not about the two scalar full-vocab passes, and are untouched here.

- **The premise was first read the other way and the reading was wrong.** The
  first pass argued "the filter touches the same 608 KB as the existing readback,
  same memory-traffic shape, nothing to win" and was headed for a negative
  write-up. Measured instead, on the CPU, on both sides of the argument:
  `sampler_bench3.c` reproduces `q36_sample_full_vocab`'s `top_p >= 1.0f` branch
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
  `q36_session_sample_packed` (`q36.c:11932`) hooked into `q36_session_sample`
  before `q36_session_ensure_logits_host` (`q36.c:12043`), buffer
  `Q36_GPU_MINP_BYTES(Q36_N_VOCAB)` (80.8 KB at vocab 151936), gated by
  `Q36_VK_MINP_PACK` (default ON).
- **Bit-identity is structural, not lucky.** `expf`, the sum and the draw stay on
  the CPU — a GPU `expf` is not the CPU's `expf` and one ulp in the sum moves the
  drawn token — and the packed list is emitted in ascending token order (blocks
  in index order, order-preserving compaction inside a block), which is exactly
  `q36_sample_full_vocab`'s scan order at `q36.c:10481-10499`. Confirmed
  byte-identical on four configurations, seed 42, 64 tokens each:
  Swift temp 0.8 / min_p 0.05 328/328 bytes, MoE 0.8 / 0.05 299/299, MoE temp
  1.0 / 0.05 333/333, Swift temp 0.8 / min_p 0.2 334/334, pack ON vs OFF.
- **Fallback is the correctness boundary, not the bucket size.** The packed path
  returns -1 — and the scalar path runs unchanged — on `top_k != 0`,
  `top_p < 1.0`, `min_p` outside `(0, 1]`, a non-Vulkan runtime, `have_reject`
  unfound, **any block whose recorded count exceeds `Q36_GPU_MINP_CAP` (16)**, or
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
  logits readback (`q36_session_ensure_logits_host` is not reached) and the
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
  step, which never sees the sampler); `./q36_test --vulkan-kernels` still fails at
  `tests/q36_test.c:4958` (`attn decode n_tok invariance` memcmp) exactly as it does
  on unmodified HEAD — reproduced this pass, unchanged, out of scope. Dispatch
  confirmed in the profile: `logits_minp_max` 16 disp / `logits_minp_reduce` 16 disp
  for 16 generated tokens, 15520 groups, no extra flush.
- Raw: `evidence/raw/p9-minp/` (`ab-decode.txt`, `ab-reps.csv`, `parity.txt`, the
  8 parity logs, `prof-on.txt`, `cpu-probes.txt`, `sampler_bench3.c`, `bw_bench.c`,
  `vulkan-kernels.txt`).
- `reconsider_if`: someone wants the sampler off the host entirely (then the
  precision argument above is the thing to beat, not the speed); `Q36_GPU_MINP_CAP`
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
  (`Q36_VK_DBG_IQ3N`, kept — it is the instrument, not a semantic variant) prints
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
  dispatch at `q36_vulkan.c:8858` is `n_tok >= 2u && n_tok <= 8u` instead of
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
  guard).  `./q36_test --vulkan-kernels` fails at `tests/q36_test.c:4958`
  (`attn decode n_tok invariance` memcmp) exactly as on unmodified HEAD —
  reproduced this pass, unchanged, out of scope.  Guard untouched by
  construction (the branch is `Q36_VK_TENSOR_IQ3_XXS`-only; the guard's tensors
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

Closes the paragraph P7 left open ("`q36_test --vulkan-kernels` fails at
`tests/q36_test.c:4958` exactly as on unmodified HEAD — reproduced this pass,
unchanged, out of scope").  Round-2 R1.

- **Bisect.**  `git bisect run` between `8e8e788` (last commit CAMPAIGN §1c
  records as PASS) and `e6a442b`, harness = rebuild `q36_test`, exit 1 iff
  `q36_test.c:4958` appears in the log.  First bad commit: **`9d54dc7`**
  ("flash-attention prefill for GQA 8 MoE too").  Raw: `evidence/raw/r1-fa-gate/`
  (`bisect.log`, `bisect-check.sh`).
- **Which arm, and how far off.**  The failing assertion is `test_vulkan_attn_case`'s
  "a batch row must reproduce the per-row decode step bit for bit" memcmp.  A
  temporary diagnostic print (removed) pinned it to one shape: `pos0=129 n_tok=3`
  with the production KV pair `k=Q36_KV_CACHE_Q8_0 / v=Q36_KV_CACHE_Q4_0`, i.e.
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
  disabled; `q36_vk_use_attn_fa()` is static to `q36_vulkan.o` so the env default
  is repeated).  FA arms assert `drift <= 4.0e-5f` — ~4000x the measured drift,
  four orders inside the test's own 2e-3 quality bound, so a broken FA reduction
  still trips it — and print the drift.  Every other arm keeps the bitwise memcmp.
  Coverage widened with three production-KV arms: `129/3` and `2100/12` at ratio
  6, `2100/12` at ratio 8.  `attn_prefill_fa.comp`'s header now says outright that
  it is not bit-identical to `attn_decode_split/fused` and by how much.
- **Gates:** `./q36_test --vulkan-kernels` **exit 0, "vulkan-kernels: OK"** with FA
  on and with `Q36_VK_ATTN_FA=0`; drift printed 1.02445e-08 at `129/3` and
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

- **What was built.**  `vulkan/moe_down_q2k_sum_decode.comp` `Q36_MOE_IQ2S` branch
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
- **Gates:** `karpathy/compat_gate.sh` **PASS**; `./q36_test --vulkan-kernels` **OK**
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

- **Setup.**  Swift (`IQ3_XXS`, 24/4 heads), `q36-server --batched-session N
  --ctx 1024 --prefill-chunk 256`, 4 fixed greedy streaming clients (one prompt
  each, ~430 tokens) with 64 generated tokens per stream, `Q36_VK_DENSE_IQ3_NX`
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
  `Q36_VK_DENSE_KQUANT_MMQ=0` (the one switch that reroutes `n_tok > 1` K-quant
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
  `Q36_MTP_TIMING` prints one line per speculative cycle from
  `q36_session_eval_speculative_argmax` (`fwd`, each draft, `snap`, `verify`,
  `commit`, total, `read_kb`, `drains`), plus `q36_gpu_read_bytes()` /
  `q36_gpu_read_flushes()` counters in `q36_vulkan.c` / `q36_gpu.h`.  No
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
  --mtp-margin 0`, `Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1`.  Stages B/C sweep
  `Q36_VK_ATTN_SPAN` 128/256/1024/2048/4096 at ctx 16384, 1 rep each (512 is
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
  `Q36_VK_ATTN_SPAN=128 ./q36_test --vulkan-kernels` gives 3 failures and `=256`
  gives 1, all at `tests/q36_test.c:4988` (bitwise batch-vs-step-replay), 0 at
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
  "narrower buys occupancy" claim in the `q36_vk_attn_span()` comment
  (`q36_vulkan.c:1282`) is corrected in place with these numbers.
- `Q36_VK_ATTN_SPAN` is a **diagnostic switch, not a semantic variant**: it is
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
