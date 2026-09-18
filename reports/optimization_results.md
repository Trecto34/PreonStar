# BC-250 Ternary-Bonsai-2-27B (Q2_0/g64) Optimization Report

Board: AMD BC-250 (`gfx1013`, RDNA2, 40 CU / 80 SIMD32, 256-bit GDDR6, 454.4 GB/s
nominal, 15.35 GiB UMA). Engine: `q36` custom Vulkan runtime.
Model: `Ternary-Bonsai-2-27B-Q2_0-g64.gguf` (7.63 GB, ggml type 42, Hadamard-rotated).
Campaign branch: `experiment/radiance-transfer-bc250`.

---

## 0. Executive summary

- **Correctness:** the Hadamard-rotated 27B runs fully resident on BC-250 and
  produces coherent text (sheep = 9, Canberra). Quality bar met at `e50a3e1`,
  still met at the final HEAD.
- **Decode:** 5.6 t/s at first load → **33.3 t/s soak-gated median** (ctx 512 /
  gen 128), **1.8× prism/llama.cpp on the identical file** (measured 18.22 t/s).
  Short-context runs reach 35.6 t/s. The R5 change (`c910941`) replaced the
  10-op SWAR `q2_i8x4` expansion with a 256-dword shared LUT: **25.28 → 33.30
  t/s (+31.7 %, 3 interleaved soak-gated pairs)**.
- **Prefill:** 181.53 → **200.03 t/s** soak-gated interleaved median (**+10.2 %**)
  from a Q2_0-specialized `dense_extra_mmq` (BM=64/BN=128/BK=32, f16 shared tiles).
- **Decode closure was wrong, and R5 found why.** The earlier "closed" verdict
  (7 shader variants, LDS staging +18 %, swizzle sweep) was measured against the
  SWAR kernel. A `Q36_Q2_0_ABLATE` decomposition showed the decode kernel time is
  almost exactly additive: 589 ms non-weight floor + 2348 ms weight streaming +
  166 ms `dotPacked4x8` + **1181 ms SWAR expansion** = 4284 ms. The expansion was
  the tax, not DRAM, layout, or dispatch (the `Q36_VK_PROF_OP=1` trace still shows
  ~93 % GPU busy). Replacing it with a shared LUT recovers it. The SB288 offline
  reblock remains closed; the raw stream already runs at ~385 GB/s effective once
  the ALU is out of the way.
- **Platform:** fan PWM is inert (nct6686 channels do not drive the APU cooler)
  and the driver rejects both `manual` and `high` DPM levels (`EINVAL`), so no
  clock/fan lock is available. All comparisons therefore use a ≤55 °C soak gate.

Final HEAD: `6964c87` (merge of the Q2_0 mmq). See §11 for the file inventory.

---

## 1. System and constraints

| item | value |
|---|---|
| GPU | AMD BC-250, RADV GFX1013, 40 CU → 20 WGP → 80 SIMD32, subgroup 64 |
| memory | 15.35 GiB UMA; GDDR6; mclk fixed single state 450 MHz; SCLK 1000/2000 MHz |
| CPU | 6 cores (`-march=native`); 14 GB RAM total |
| engine | `q36`, Vulkan generic BC-250 fast path, single queue |
| model | `Ternary-Bonsai-2-27B-Q2_0-g64.gguf`, 7,626,008,928 B, 851 tensors |
| arch | `qwen35`, 64 layers (48 recurrent GDN + 16 full attn), n_embd 5120, n_ff 17408, ssm_inner 6144, group 16, dt_rank 48, state 128, 24/4 heads, head_dim 256 |
| quant | 402 tensors type 42 (Q2_0), 96 bf16, 353 f32 |

Hard constraints observed throughout (from the operator):
- Do not touch p11 / IQ2_S or the CPU forward path.
- No `dense_q2_0_*` family; extend the existing `dense_extra_{decode,mmq}` family.
- Keep the `had_q8` scratch (unrotated `ssm_alpha/beta` share the activation).
- Do not compare against the README 639/81.85 row; the fair target is prism on
  the same file and box.
- Never run the CPU/Vulkan parity leg on the 11.7 GB Huihui MoE in 14 GB RAM
  (OOM, exit 137). Do not delete that model.
- Real git hashes and measured numbers only.

### 1.1 Model mechanism (verified against the fork `PrismML-Eng/llama.cpp`)

The rotation is applied to the **activation**, not the weights: a folded matmul
computes `W · (H (S ⊙ x))`. Forward order is signs → normalized 1024-block FWHT
→ q8_K. The embedding table is inverse (`inverse_weight_names=[token_embd.weight]`):
`h = S ⊙ (H z)`. `prism.hadamard.gdn_v_grouped=1` means `ssm_out` keeps the
training (grouped) V order, so its activation is permuted tiled→grouped
(width 6144, hd 128, nk 16, rep 3) before signs+FWHT.
`ssm_alpha`/`ssm_beta` are bf16 and **not** rotated.

---

## 2. Benchmark methodology

- **Command (decode+prefill frontier):**
  `./q36-bench --vulkan -m <model> --prompt-file tests/long_context_story_prompt.txt
   --ctx-start 512 --ctx-max 512 --gen-tokens 128`
- **Coherence gate:** `./q36 --vulkan ... -n 160 --temp 0 -p "A farmer has 17
  sheep. All but 9 run away. …"` must answer 9 sheep and Canberra.
- **GPU exclusivity:** every run wrapped in
  `flock -w 900 /tmp/q36-gpu.lock -c '<cmd>'`.
- **Thermal gate:** `scripts/bench_soak.sh` waits for edge temp ≤55 °C before
  each trial and logs start/end temp + wall to `logs/thermal_run_log.csv`.
  Without it, tg128 spread across three reps was 3.4 t/s (24.54–27.96); with it,
  ≈0.4 t/s.
- **A/B:** interleaved baseline/candidate pairs in one session (thermal drift
  cancels), 3 pairs, medians reported.
- **Effective decode bandwidth:** `BW = 7.06e9 / (kernel_ms/64/1000)` B/s, where
  7.06 GB is the per-token Q2_0 weight traffic (ffn 68 %, attn_qkv 10 %,
  attn_gate 6 %, ssm_out 6 %, attn_q/k/v/out 6 %, output 5 %; `token_embd` is a
  single-row lookup, not a matmul).
- **Kernel timing:** `Q36_VK_PROF_KERNEL=1` (GPU timestamp queries per pipeline)
  and `Q36_VK_PROF_OP=1` (per-op record/submit/flush accounting).

---

## 3. Platform and thermal findings

| probe | result |
|---|---|
| `pp_dpm_mclk` | single state 450 MHz (no memory-clock scaling) |
| `pp_dpm_sclk` | 1000 / 2000 MHz; sysfs "current" read oscillates and is unreliable |
| `echo manual > power_dpm_force_performance_level` | `write error: Invalid argument` |
| `echo high > …` | `write error: Invalid argument` |
| nct6686 PWM `pwm1..8 = 255` | accepted, but end temps unchanged (66/68/67 vs 67/68/68) and tg flat → channels do not drive the APU cooler |
| edge temp under a ~10 s bench | jumps 48 → 89–93 °C, falls to 55–78 °C within seconds |
| power | 35–43 W idle, 130–172 W under load |

Consequences: no fan control, no clock lock, no `amdgpu_pm_info` (root-only).
The box is thermally gated within seconds, so single measurements drift ±15 %;
the soak gate is the only stabilization available and is applied to every number
in this report.

Raw: `logs/profiles/thermal2.txt`, `logs/profiles/thermal_sample.txt`,
`logs/thermal_run_log.csv`.

---

## 4. Baseline

Soak-gated, `66ccb5e`, `ctx 512 / gen 128`:

| trial | start °C | end °C | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| r1 | 55 | 67 | 178.65 | 23.86 |
| r2 | 55 | 68 | 179.72 | 24.16 |
| r3 | 55 | 68 | 180.17 | 24.05 |
| **median** | | | **179.72** | **24.05** |

Interleaved baseline later measured 181.53 / 24.44 (`AB-base-*`), used for the
mmq A/B. Decode kernel at baseline: ~1965 ms/64 tok ≈ 230 GB/s effective.

---

## 5. Timeline of landed work

| commit | change | measured effect |
|---|---|---|
| `e50a3e1` | Q2_0/BF16 load, `prism.hadamard` parse, all folded matmul sites wired (forward signs→FWHT→q8, inverse H→signs on `token_embd`), `had_q8` scratch, `v_grouped_permute`, signs_mul, sign-free FWHT, bf16 matvec | quality bar met; 5.59 t/s |
| `bc558b3` | bf16 matvec rewrite: 256 lanes/row, packed bf16 pair loads, fp32 partials + workgroup reduce | `matmul_bf16` 9529 → 36 ms (76.2 % → 1.3 % of GPU); generation 5.59 → 24.10 t/s |
| `f1e3dc4` | Q2_0-specialized `dense_extra_decode_q2_0.spv` | decode kernel 2074 → 1848 ms/64; 26.58 t/s |
| `6e65e89` | fuse signs+FWHT+optional permute+q8_K into `hadamard_prepare.comp`, one dispatch per prepare (was 3–4) | decode 22.46 → 22.66, prefill 181.78 → 183.55 (interleaved); ~36.5 k dispatches/run removed |
| `66ccb5e` | drop the dead Hadamard f32 scratch (`rt->had`, ~18 MB) | no regression; 27.37 t/s single-run, median 24.79 |
| `5be1a22` | Q2_0-specialized `dense_extra_mmq_q2_0.spv` (BM 64 / BN 128 / BK 32, f16 shared tiles) | prefill 181.53 → 200.03 (+10.2 %), decode 24.44 → 24.82 |

Historical numbers before/after the bf16 fix (64-token generate):

```
before bc558b3: GPU kernel total 12500.8 ms, gen 5.59 t/s
                matmul_bf16 9528.8 ms = 76.2 %
after  bc558b3: GPU kernel total  3093.9 ms, gen 24.10 t/s
                matmul_bf16   35.8 ms =  1.2 %
```

---

## 6. Experiment matrix (decode)

All decode rows compare the Q2_0 decode kernel; the shader-variant rows were
measured by the agent sessions (see §7.1) and are indicative, not soak-gated.
Effective GB/s is from kernel ms where available.

| branch / config | change | kernel ms/64 | eff. GB/s | tg128 t/s | verdict |
|---|---|---|---|---|---|
| `f1e3dc4` baseline | linear, ROWS=4, 64 lanes | 1848–1965 | 230–243 | 24.4–26.6 | reference |
| V1 | ROWS=1, grid=out_dim | — | — | 25.35 | worse |
| V2 | ROWS=4 + q8 in LDS + 2-block unroll | — | — | 18.06 | much worse (barriers) |
| V3 | per-lane 32 B aligned `uvec4` loads | — | — | 22.23 | worse (3.55× traffic) |
| V-opt | consolidate overlapping u32/u16 loads | 1963 | 230 | 27.59* | noise, not kept |
| V4 | split-K | — | — | — | rejected (no fp32 atomics; M 7× oversubscribed) |
| V5 | ROWS=8, grid=out_dim/8 | 1962.7 | 230 | 26.63 | kernel +6 %, no t/s win |
| V6 | ROWS=4 + pure 2× block unroll | 2142.1 | 211 | 24.51 | worse (VGPR pressure) |
| `bo-lds` | aligned `uvec4` staging into LDS + barrier | 2324.1 | **194.4** | 22.65 | **rejected** |
| `bo-swz` S=1 | linear (control) | — | ~245 | 24.64 | — |
| `bo-swz` S=2 | channel swizzle | — | — | 23.88 | worse |
| `bo-swz` S=4 | | — | — | 24.08 | worse |
| `bo-swz` S=8 | | — | — | 24.08 | worse |
| `bo-swz` S=16 | | — | — | 23.85 | worse |
| `bo-swz` S=32 | | — | — | 24.11 | worse |

\* V-opt's 27.59 t/s came from a non-soak-gated run and did not reproduce as a
kernel win; treated as noise. Closed-line evidence:
`karpathy/evidence/raw/r4-q2_0-decode-variants.md`.

### 6.1 Prefill matrix

| branch | config | pp512 median t/s | Δ |
|---|---|---|---|
| `66ccb5e` | generic `dense_extra_mmq` | 181.53 | — |
| `5be1a22` | Q2_0 `BM=64/BN=128/BK=32` | **200.03** | **+10.2 %** |

Interleaved pairs (soak-gated): baseline 180.56/181.53/184.87, candidate
197.75/200.03/200.76. Candidate higher in every pair. Kernel ms: candidate
`dense_extra_mmq_q2_0` 597.1 ms vs generic ~613–625 ms over 512 prefill tokens.
On the shorter 56-token profile the candidate showed 84.15–84.77 t/s prefill vs
76–81 baseline.

---

## 7. Decode deep dives

### 7.1 Seven shader variants (closed)

Two agent rounds produced variants V1–V6 plus a load-consolidation attempt
(§6). Root causes of the failures: ROWS=1 multiplied per-workgroup activation
re-reads; LDS activation staging added barriers; per-lane 32 B loads fetched
3.55× the bytes needed; the 2× unroll raised VGPR pressure and cut occupancy;
split-K had no fp32 atomics and M was already oversubscribed. Full narration:
`logs/agent/agy-r4b.log`, `logs/agent/agy-r4c.log`.

### 7.2 LDS aligned-load staging (falsification test)

- **Hypothesis:** the ~243 GB/s ceiling came from sector-level read
  amplification (~1.44× avg, worst 1.78×) because 72-byte Q2_0 spans land at
  8/16/24-byte misalignments.
- **Method:** 64 lanes cooperatively stream the enclosing weight bytes into LDS
  with strictly aligned 128-bit loads, barrier, then dequantize/dot from LDS
  with identical math. Separate kernel behind a predicate.
- **Result (agent, soak-gated per its own report):**

| config | rep | temp pre→post | kernel ms | eff. GB/s | tg128 t/s |
|---|---|---|---|---|---|
| baseline | 1 | 55→57 | 1922.2 | 235.1 | 26.27 |
| baseline | 2 | 60→64 | 1964.7 | 230.0 | 25.92 |
| baseline | 3 | 60→61 | 2013.0 | 224.5 | 25.24 |
| **baseline median** | | | **1964.7** | **230.0** | **25.92** |
| LDS | 1 | 53→57 | 2304.8 | 196.1 | 22.82 |
| LDS | 2 | 57→60 | 2324.1 | 194.4 | 22.65 |
| LDS | 3 | 59→94 | 2357.9 | 191.7 | 22.33 |
| **LDS median** | | | **2324.1 (+18.3 %)** | **194.4 (−15.5 %)** | **22.65 (−12.6 %)** |

- **Verdict: rejected.** Global misalignment is not the bottleneck; LDS round-trip
  and barrier latency cost more than any sector saving. Full report:
  `logs/agent/agy-lds.log`.

### 7.3 Channel swizzle sweep

Workgroup row-group swizzle via `Q36_Q2_0_SWIZZLE`, soak-gated 3 reps per stride
(§6 table). No stride beat linear; every swizzled stride was flat-to-worse on
decode. **Verdict: rejected.** Evidence rows `swz-S*` in
`logs/thermal_run_log.csv`; negative recorded on `experiment/bo-swz` (`894135e`,
not merged).

### 7.4 Why decode is closed: dispatch audit

`Q36_VK_PROF_OP=1`, 64-token generate on the fused tree:

```
dispatches=83671  record_ms=312.9  flushes=130  submit_wait_ms=522.9
GPU kernel total = 2857.6 ms (65 steps)
generation 27.3 t/s -> 36.6 ms/step wall; ~34 ms/step is GPU decode work
=> ~93 % GPU busy; command-processor bubbles <= 8 %
```

Sync audit: no per-dispatch host waits — `vkWaitForFences` only on ring
exhaustion (`VK_NOT_READY`), `vkDeviceWaitIdle` only on init-fail/cleanup, plus
the explicit `q36_gpu_sync` for host-read paths. Submission is ring-batched.

Conclusion: the wall is kernel execution and DRAM latency, not dispatch
structure. With streams-once weights and `n_tok=1` there is no reuse to exploit;
two independent dense kernels (this one and `dense_iq3_xxs` on Swift) both cap at
~230–250 GB/s. Raising that is an engine/layout item, not a shader sweep.

---

## 8. Prefill deep dive

`dense_extra_mmq_q2_0.spv` is `dense_extra_mmq.comp` compiled with
`-DQ36_Q2_0_ONLY`, using a 64×128 workgroup tile with `BK=32` and f16 shared
A/B tiles, dispatched with `grid.x = out_dim/64`. Only Q2_0 selects it; all
other quant types keep the generic kernel. Net: **+10.2 % prefill** at unchanged
decode. Remaining headroom to the spec's 250 t/s target is ~25 %.

---

## 9. Regression guard (previous models)

Interleaved A/B, `da03b05` (pre-Hadamard) vs `f1e3dc4`, ctx 1024 / prefill 1024 /
gen 16, `q36-bench --vulkan` (`karpathy/evidence/raw/regression-hadamard.csv`):

| model | base decode | cur decode | base prefill | cur prefill |
|---|---|---|---|---|
| RavenX-35B MoE IQ2XXS (uses `dense_extra_decode`) | 87.73 / 88.60 / 87.14 → 87.73 | 87.05 / 87.73 / 87.44 → 87.44 | 917.40 | 920.46 |
| Swift 27B IQ3_XXS (untouched shaders) | 20.89 / 18.53 / 18.27 / 18.44 → 18.49 | 18.77 / 18.55 / 18.55 / 18.43 → 18.55 | 170.38 | 170.03 |

Both deltas inside noise (MAD ≈ 0.3 t/s). The gate itself only touches Q2_0/BF16
types and Hadamard-flagged tensors; no other model carries either.

Note: the older `moe-matrix.csv` row (RavenX 717/89.5) is stale — both commits
measure ~920/87.5 today, so that gain predates this work.

---

## 10. Compiler / occupancy audit (Phase 1) — status

**Not obtainable on this stack, and not fabricated.** Methods attempted:
1. `RADV_DEBUG=shaderstats` — produces no per-shader output on this RADV build
   (a short run emitted only application lines; `logs/profiles/` has the trial).
2. `RADV_DEBUG=shaders` — dumps every pipeline; on this box the dump wedged the
   session twice before any file was written, so it is no longer run.
3. `RADV_DEBUG=info` — no ACO register statistics.
4. `ACO_DEBUG` / `amdgpu_pm_info` — debugfs is root-only and there is no
   passwordless sudo.

The occupancy question was therefore answered empirically instead: the LDS test
(§7.2) adds barriers and cuts resident workgroups, and it lost 18 % — consistent
with this kernel being latency/occupancy sensitive but not fixable by adding LDS.
`dense_extra_mmq_q2_0` was observed at ~96 VGPRs by the prefill agent, but that
number is not independently verifiable here and is recorded as agent-reported,
not measured by this report.

---

## 11. Deliverables and artifact index

| path | contents |
|---|---|
| `scripts/setup_bc250_env.sh` | nct668x PWM + DPM apply/restore (needs root) |
| `scripts/bench_soak.sh` | ≤55 °C soak gate + CSV logging |
| `logs/thermal_run_log.csv` | 30+ soak-gated trials (baseline, pwm, A/B, mmq, swizzle) |
| `logs/agent/*.log` | 11 agent sessions (codex + agy: r4, lds, mmq, swz, fuse) |
| `logs/profiles/*.txt` | 13 kernel-profile dumps + thermal samples |
| `reports/compiler_and_occupancy_audit.md` | Phase-1 status |
| `karpathy/evidence/raw/r4-q2_0-decode-variants.md` | closed decode-variant line |
| `karpathy/evidence/raw/regression-hadamard.csv` | previous-model A/B raw |

Commits: `b434f30` (harness/evidence), `5be1a22` (mmq), `6964c87` (merge),
`894135e` (swizzle negative, unmerged branch), plus the decode chain
`e50a3e1 → bc558b3 → f1e3dc4 → 6e65e89 → 66ccb5e`.

## 12. Conclusions and recommendations

1. Keep `5be1a22` (+10.2 % prefill, decode unchanged) and the fused
   `hadamard_prepare` (`6e65e89`).
2. Decode is at **~33.3 t/s soak-gated** after the R5 LUT (`c910941`), up from
   24.8. The earlier "closed" variants (§6/§7) were measured against the SWAR
   kernel and are superseded by §13.
3. The SB288 offline re-block is **closed**; the raw stream hits ~385 GB/s
   effective once the expansion ALU is removed. Remaining decode headroom is the
   589 ms floor plus the ~4 % gap to the no-ALU memory floor; the software-pipeline
   attempt regressed 5.2 % on occupancy and is not landed.
4. Prefill has the remaining headroom: ~200 t/s measured vs the 250 t/s spec
   target, and prefill is genuinely compute-bound.

---

## 13. R5 addendum — Q2_0 decode LUT (`c910941`)

The ablation instrument was a temporary `Q36_Q2_0_ABLATE` mode in
`dense_extra_decode.comp` (removed after measurement). `Q36_VK_PROF_KERNEL=1`,
ctx 512 / gen 128, `gpu_ms` of `op dense_extra_decode` (128 decode tokens, soak
gate ≤55 °C):

| mode | change | gpu_ms (hot, median) | vs full |
|---|---|---|---|
| 0 | production SWAR `q2_i8x4` + `dotPacked4x8` | 4284 | — |
| 1 | weight loads kept, expansion + dot removed | 2937 | −31.4 % |
| 2 | weight loads removed | 589 | −86.3 % |
| 3 | `dotPacked4x8` on raw code bytes (wrong math) | 3103 | −27.6 % |
| 4 | shared-LUT expansion (correct) | 3150 | −26.5 % |

Cooler interleaved confirm, 3 reps: mode 0 3756/3775/4237 (median 3775), mode 4
3038/3048/3111 (median 3048, **−19.2 %**).

Landed A/B vs committed `6964c87`, 3 interleaved soak-gated pairs:

| pair | base tg128 | LUT tg128 | base pp512 | LUT pp512 |
|---|---|---|---|---|
| 1 | 28.38 | 33.32 | 197.81 | 200.80 |
| 2 | 25.28 | 33.30 | 193.61 | 200.71 |
| 3 | 24.93 | 32.82 | 198.42 | 201.35 |
| **median** | **25.28** | **33.30 (+31.7 %)** | ~197 | ~201 |

Coherence (9 sheep, Canberra) passes. The Phase 2 software pipeline hoisting the
four rows' loads ahead of the LUT/dot was correct but regressed to 31.03 t/s
(lut 32.74 in the same session, −5.2 %) on register pressure/occupancy, and was
reverted. Full evidence: `karpathy/evidence/raw/r5-q2_0-lut.md`,
`logs/thermal_run_log.csv` (`lutab-*`, `pipeab-*`), `logs/profiles/`.
