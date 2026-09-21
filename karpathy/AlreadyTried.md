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



