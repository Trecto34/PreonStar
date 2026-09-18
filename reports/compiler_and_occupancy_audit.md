# Compiler / Occupancy Audit (Phase 1) — BC-250 gfx1013

Status: **SUPERSEDED (2026-09-18) — the data IS obtainable.** Register/LDS stats
arrived via `VK_KHR_pipeline_executable_properties` (see the main report), and
full ACO **Assembly** via the same extension's internal-representation query:
build the pipeline with `VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR`
and run the three-pass query (count -> sizes -> data). Tool
`logs/bc250-sustained-20260918/shader_isa.c`, output
`logs/bc250-sustained-20260918/isa/`, analysis
`reports/decode_coalescing_audit.md`. The table below is retained as a record of
what does *not* work (`RADV_DEBUG`, `ACO_DEBUG`, `umr`), not as a statement that
disassembly is unavailable.

## Requested data
1. VGPR & SGPR per wave for `dense_extra_decode` / `matmul_bf16`.
2. LDS allocation per workgroup.
3. Active waves per SIMD.
4. Scratch/spill bytes.

## Methods attempted

| method | result |
|---|---|
| `RADV_DEBUG=shaderstats ./q36 …` | No per-shader statistics emitted on this RADV build; only application output. Trial: `logs/profiles/` (`shaderstats` run in `logs/agent/`). |
| `RADV_DEBUG=shaders ./q36 …` (to file) | Dumps every pipeline. Two attempts wedged the interactive session before a file was written. **Not re-attempted.** |
| `RADV_DEBUG=info` | Device info only; no ACO register stats. |
| `ACO_DEBUG=stats` | Not exposed for prebuilt ACO in this Mesa build. |
| `amdgpu_pm_info` / throttle flags | `/sys/kernel/debug` is root-only and there is **no passwordless sudo**; `amdgpu_pm_info` is absent. |
| `umr` register reads | Needs root; unavailable. |

## What can be said from measurement

- `dense_extra_decode_q2_0`: LDS was 0 B when this was written; since `c910941`
  the shared 256-entry dequant LUT makes it **1024 B/workgroup** (measured).
  The historical note below refers to the pre-LUT kernel. The `bo-lds` experiment (§7.2 of the main report)
  introduced LDS + a barrier and lost 18.3 % (1964.7 → 2324.1 ms/64), so the
  streaming decode is better with zero LDS and no barriers.
- The 2× block unroll (V6) and ROWS=8 (V5) variants regressed or were neutral,
  which is the signature of a VGPR/occupancy limit rather than an instruction
  limit — but the exact wave count cannot be measured here.
- `dense_extra_mmq_q2_0`: the prefill agent reported ~96 VGPRs and f16 shared
  A/B tiles; that is *agent-reported*, not independently verifiable on this box,
  and is quoted as such in the main report.

## Recommendation

If register/occupancy numbers are required, collect them where the tooling works:
run `RADV_DEBUG=shaders` / `umr` on a machine with root or in a CI container,
against the same shaders. On the BC-250 as configured, the soak-gated A/B is the
reliable instrument.
