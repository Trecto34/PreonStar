# Implementation plan — q36/BC-250 campaign, 2026-09-20

Base: branch `trackB-ptq1_0`, HEAD `6ef9ceb` (IQ2_S experts on the fused
gate/up + GEMM pair; tree clean). This is a *working* plan: fold each closed
item into `karpathy/CAMPAIGN.md` §6 and write its verdict into
`karpathy/AlreadyTried.md` as it lands. It is not a new design doc layer.

## 0. Concurrency contract (read first)

A second instance is actively porting **IQ2_S into the f32 decode kernels**
(`moe_gate_up_decode.comp`, `moe_down_q2k_sum_decode.comp`), the
`reconsider_if` named at the end of the `6ef9ceb` ledger entry. Treat these
as **owned by that instance; do not edit**:

- `vulkan/moe_gate_up.comp`, `vulkan/moe_gate_up_gemm.comp`, `vulkan/moe_down_gemm.comp`
- `vulkan/moe_gate_up_decode.comp`, `vulkan/moe_down_q2k_sum_decode.comp`
- Makefile rules for the `*_iq2s.spv` builds
- `q36_vulkan.c` IQ2_S host wiring (`q36_vk_moe_gate_up_iq2_swiglu`, `q36_gpu_moe_ffn_f32_tensor`)
- `logs/iq2s-gateup-20260920/`, `karpathy/evidence/raw/iq2s-gemm-*`

Hard rules for everything below:

1. **One GPU job at a time.** Run benches under the shared lock
   `flock -w 900 /tmp/q36-gpu.lock -c '<cmd>'`, and check
   `pgrep -x q36-bench || echo "GPU idle"` before any build or bench.
   Never build while a bench runs (a background `make -j16` corrupted a matrix
   run this campaign).
2. **Never A/B by running A fully then B.** `tests/bench_ab.sh` only; report
   **median + MAD**, never means.
3. **Noise floors:** dense 27B prefill 0.7%, **MoE prefill 3.4%**, decode up to
   9.4% spread. A MoE prefill delta <3% is noise.
4. **GPU-vs-GPU oracle, not CPU parity.** `--gpu-cpu-parity` decodes 10 GB of
   IQ2_S on CPU (minutes/case) — wrong instrument. `--vulkan-fusion-parity`
   does **not** toggle every gateway flag, so it can compare a kernel against
   itself (it missed `Q36_VK_MOE_GATE_UP`). For any kernel port, reference
   = the frontier-logits dump with the new path forced off, e.g.
   `Q36_VK_MOE_GEMM=0 Q36_VK_MOE_GATE_UP=0`. Calibrate "small enough" against
   the shipped equivalent (the IQ2_XXS guard measured 64/64 top-64, max_abs 0.42).
5. **Guard is byte-identity, not vibes.** Any shader touched must leave the
   IQ2_XXS/Q2_K `.spv` checksums unchanged (`b46fa81a` gate/up GEMM,
   `93a38508` down GEMM, `601d56a9` q8 gate/up).
6. **Raw CSV into `karpathy/evidence/raw/` before `/tmp` is cleared.** Numbers
   whose raw data is gone are not evidence.
7. **Rebase onto the other instance's commits** before each item; never edit a
   file they are mid-port on.

Standard command shapes:

```sh
cd /home/server/q36-opt-27b && pgrep -x q36-bench || echo "GPU idle"
git log --oneline -3 && git status --porcelain
tests/bench_ab.sh <A_BIN> <B_BIN> /home/server/q36/gguf/<MODEL>.gguf 7 \
  > /tmp/ab-<slug>.csv 2> /tmp/ab-<slug>.summary
cp /tmp/ab-<slug>.csv karpathy/evidence/raw/
```

Models: Swift `Swift-Qwen3.8-27B-IQ3_XXS.gguf` (dense, 0.7% floor),
IQ2_M `Qwen3.8-35B-A3B-IQ2_M.gguf`, guard
`Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf` (3.4% floor).

---

## W1 — IQ2_S f32 decode port (track only; owned by other instance)

- Goal: cover `n_tok == 1` in `moe_gate_up_decode` + `moe_down_q2k_sum_decode`,
  the pair the guard's own decode already runs, so it lifts the current 1.18× decode.
- Current verified state at `6ef9ceb`: prefill **108.21 → 241.20 t/s (2.23×)**,
  decode **29.52 → 34.81 t/s (1.18×)**; `moe_matvec` 6355 ms → gate/up GEMM 637 ms
  + down GEMM 253 ms.
- Our action: **wait.** Do not touch their files. After it commits, re-baseline
  W2–W8 against the new HEAD (IQ2_M numbers move, so re-run any pending A/B).

---

## W2 — IQ3_XXS ISA + ALU ablation (highest own-priority; diagnostic)

- Why: "234 GB/s logical = DRAM-bound" is a derived number, not a measurement.
  Per-format logical rates already span 235–360 GB/s; decide per kernel. Precedent:
  `dense_extra_decode_q2_0` was ALU-taxed (+31.7% from the LUT), not DRAM-bound.
- Files (touch, then revert the diagnostic):
  - `vulkan/dense_iq3_xxs_decode_r4.comp` — temporary `Q36_IQ3XXS_ABLATE` mode
    (keep loads, drop decode ALU), exactly as `reports/decode_overhead_breakdown.md`
    §4.1 did for Q2_0.
  - Tooling (already in tree): `logs/bc250-sustained-20260918/shader_isa.c`,
    `logs/bc250-sustained-20260918/mmq_info.c`.
- Steps:
  1. `cc -O2 -o /tmp/shader_isa logs/bc250-sustained-20260918/shader_isa.c -lvulkan`
     and same for `mmq_info`. Confirm `VK_KHR_pipeline_executable_properties: yes`.
  2. Dump `vulkan/dense_iq3_xxs_decode_r4.spv` and `dense_iq3_xxs_mmq.spv`:
     VGPR count, LDS/workgroup, subgroups/SIMD, load width, `s_waitcnt` placement,
     descriptor reloads. Archive asm under `karpathy/evidence/raw/iq3xxs-isa/`.
  3. Ablation run: `Q36_VK_PROF_KERNEL=1`, ctx 512, gen 1 vs gen 65, read the
     per-kernel `gpu_ms` delta for `dense_iq3_xxs_decode_r4`. Flat ⇒ memory-bound;
     large drop ⇒ ALU/issue-bound.
  4. Only if a real variant emerges: full 7-rep `bench_ab.sh` + frontier parity.
- Gate: diagnostic (no perf claim without the 7-rep A/B). Exit = written
  DRAM-bound / not verdict with the ISA table as raw evidence.
- Ledger: record the verdict either way (it settles an open debate).

## W3 — `delta_net_cols` on BC-250 (cheap; may be free prefill)

- Why: `vulkan/delta_net_cols.comp` already implements the "load state once →
  token loop → store once" pattern, but `q36_vulkan.c:1221`
  (`q36_vk_use_delta_col_prefill`) disables it when `q36_vk.bc250` is set, and
  this build **requires** BC-250 (`:3401`). The ctx-512 prefill profile shows
  `delta_net_decode_reg_f16` active instead.
- Files: `q36_vulkan.c` (env/gate, ~6 lines, small diff so it merges around the
  other instance), `vulkan/delta_net_cols.comp` (read only unless a fix is needed).
- Steps: `git log -p -- q36_vulkan.c | grep -A5 delta_col` + grep the ledger for
  why; if never tested on-target, A/B with `Q36_VK_DELTA_COL_PREFILL=1`.
- Gate: dense prefill **≥1.5%** and frontier logits bit-exact (the shader comment
  claims the chain order is identical to the decode kernel). Ceiling ≈ GDN's ~3%
  of prefill, so a clean ≤1% is plausible, not a win; a negative gets documented.
- Commands: `Q36_VK_DELTA_COL_PREFILL=1 ./q36-bench ...` then the standard A/B.

## W4 — wave32 clean-scan A/B (one unrun test; now also helps IQ2_M decode)

- Why: `karpathy/evidence/wave32-eligibility.md` §3b lists clean-scan shaders
  (incl. `moe_matvec`, `moe_matvec_fast`) blocked only by the force comment's
  letter; `moe_matvec` still runs the IQ2_M decode down projection (218 ms/token
  as of `6ef9ceb`). Never A/B'd.
- Files: `q36_vulkan.c:1628-1664` (`q36_vk_force_wave32` list / suffix predicate)
  — again a small isolated diff.
- Steps: add the §3b set behind the existing predicate (or a one-shot env gate),
  compile, verify `Q36_VK_SHADER_TRACE=1` selects it, A/B on IQ2_M and the guard.
- Gate: MoE prefill **≥3.4%** OR decode paired median with MAD; guard prefill no
  regression. Negative → ledger with `reconsider_if`.
- Risk: forcing wave32 on a `local_size_x=64` shader changes workgroup/subgroup
  mapping; parity dump required.

## W5 — RMSNorm → q8_K producer fusion

- Why: `add_rms_norm` (prefill 33.0 ms, decode ~1.11 ms/tok) and `q8_k_quant`
  (prefill 20.4 ms, decode ~0.11 ms/tok) are separate dispatches; `add_rms_norm.comp`
  already documents the 1024-wide fix. Fusing the quant into the norm removes one
  activation write/read. (`swiglu→q8` and `hadamard_prepare` are already fused.)
- Files: `vulkan/add_rms_norm.comp` (add a q8_K-emitting variant or a compile flag),
  `q36_vulkan.c` dispatch, `q36.c` call site. **Coordinate**: `q36_vulkan.c` is
  shared with W1 — keep to a few lines, rebase on their HEAD.
- Gate: dense prefill **≥0.7%**, decode paired median with MAD, frontier logits
  parity (the fp64 dual-accumulator reassociation is already documented as safe;
  keep that structure).
- Ceiling ~0.9% prefill / ~0.2% decode. Low, but clean.

## W6 — merged Vulkan dense gate_up

- Why: on Vulkan the dense FFN still calls gate and up as two separate
  `q36_gpu_tensor_matmul_q8_or_float_scaled` (`q36.c:8114-8124`); a fused pair
  exists only under `#ifdef Q36_METAL` for `n_tok==1` (`:8092-8112`).
  MoE gate/up fusion is already the other instance's work — this is dense only.
- Files: new pair-projection shaders for the dense IQ3_S/IQ3_XXS weights, port the
  host pair branch, `q36_vulkan.c`, `Makefile`.
- Gate: dense prefill ≥0.7% (tile efficiency), decode paired median with MAD.
  For a DRAM-bound decode the bytes are identical — expect the win in prefill only.
- Use the GPU-vs-GPU oracle (§0.4) before trusting the pair math.

## W7 — Stream-K geometry audit (measure before porting)

- Why: q36's grid is `ceil(out_dim/64) × ceil(n_tok/128)`. For 5120×17408 that is
  272×2 = 544 workgroups → ~6.8 waves over 40 CU, tail ≤1 wave. `timeline-analysis.md`
  shows every shape at the same per-MAC rate and 0 idle, i.e. per-workgroup
  (barrier/LDS) cost, not tail. Stream-K attacks K partitioning, not that.
- Files: none; measurement only (scratch script under `logs/`).
- Steps: per MMQ shape, compute `grid / (40 CU × resident WGs/CU)`; check partial
  waves via `Q36_VK_PROF_SHAPE`. Port only if the tail is **>3%** of kernel time.
- Gate: evidence; expected exit = "not a lever", written to the ledger.

## W8 — long-context KV dequant redundancy

- Why: `vulkan/attn_prefill_qtile2_gqa6.comp` dequantizes quantized KV inline
  (K 272 B/head, V 144 B/head) per lane now + P·V (lines 164-247), and the grid
  `(kvh, n_tok/QT=2, spans/8)` re-dequantizes the same keys for every query tile.
  Real, but attention is ~3.1% of ctx-1024 prefill and only grows at long ctx.
- Files: `vulkan/attn_prefill_qtile2_gqa6.comp` / `attn_prefill_qtile2.comp`.
- Steps: measure attention share at ctx 4096/8192 first; only then try a shared KV
  tile and A/B. Materializing fp16 KV to scratch adds write+read traffic on top of
  a compact quantized cache — the coopmat1 result (`200→282 / 99→166 t/s`) does
  **not** transfer.
- Gate: ctx ≥4096, ≥1.5% prefill, parity. Low priority unless sessions really run long.

## W9 — server track (separate; not t/s)

- Block-paged automatic prefix caching + hybrid full-attn/GDN state group reuse.
  q36 already has disk KV checkpoints + live reuse keyed by text prefix; the
  paged-block refinement changes TTFT for agentic workloads, not llama-bench.
  Open only when multi-user/agent throughput is the target.

---

## Sequencing

1. **W1** lands (other instance) → rebase, re-baseline.
2. **W2** IQ3_XXS ISA/ablation (diagnostic; settles the DRAM-bound question).
3. **W3** delta_net_cols gate (cheap, possibly free).
4. **W4** wave32 clean-scan A/B (one unrun test, now IQ2_M-relevant).
5. **W5** RMSNorm→q8 fusion, then **W6** dense gate_up (both small, shared-file).
6. **W7** Stream-K audit (likely negative), **W8** long-ctx (only if needed).
7. **W9** server track in parallel, separate owner.

## Do not reopen (ledger)

Stream-K port without the W7 measurement; FA LDS staging; MMQ staging/barrier
restructure; f16 LUT; tiles 64/256; BK 32/64; XOR swizzle; dynamic-k; 512-thread
add/RMS; `RADV_PERFTEST=cswave32` env lever; MTP draft depth 2/3; prefill chunk
>256; the drain patch as a speed lever; CPU-vs-GPU parity as a kernel-port oracle.

## Deliverables per item

Each closed item leaves: raw CSV in `karpathy/evidence/raw/`, a
`karpathy/AlreadyTried.md` entry (what changed, measured, why accepted/rejected,
`reconsider_if`), the code commit with the numbers in the message, and a
`karpathy/CAMPAIGN.md` §6 update.
