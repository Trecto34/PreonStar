# P1 Verdict — `moe_iq2s_down_sum_decode` (IQ2_M #1 decode kernel)

2026-09-24, audit §P1. Model `gguf/Qwen3.8-35B-A3B-IQ2_M.gguf` (guard
`Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf` does not reach this kernel:
its expert down projections are Q2_K and dispatch the separate
`vulkan/moe_down_q2k_sum_decode.spv` blob).

Baseline (`Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1`, ctx 512, chunk 256,
gen 128, IQ2_M):

```
op moe_iq2s_down_sum_decode  dispatches=4736 groups=9699328 gpu_ms=313.156
op moe_iq3s_down_sum_decode  dispatches=384  groups=786432  gpu_ms=25.111
```

4736 dispatches / 128 tokens = 37 calls per token; 9699328 / 4736 = 2048
workgroups per dispatch (one per output row, `local_size_x = 32`).
Per token: 2048 rows x 37 calls x 8 experts x 164 B (2 blocks x 82 B) = **99.4 MB
unique weight bytes**, i.e. 12.72 GB per 128-token run.  In 313.2 ms that is
**40.6 GB/s** of DRAM traffic.  The audit's "~100 MB/tok at ~40 GB/s" is
confirmed and it is *unique* traffic (16 lanes share one 82 B superblock, so the
per-lane redundancy is served by L1 broadcast, not DRAM).

## Probes

| # | Probe | A (stock) | B | Verdict |
|---|---|---|---|---|
| A | `fma32()` -> `fma()` (profile only) | 315.18 / 314.65 / 313.16 ms | 314.65 ms | **negative**, inside noise |
| B | force wave32 | n/a | `mmq_info` says `subgroup=32` already | **premise false** |
| C | `Q36_VK_MOE_DOWN_SUM_DECODE=0` (per-expert parallel down kernels) | 74.46 / 74.42 / 74.39 t/s | 70.65 / 71.07 / 70.95 t/s | **negative**, -4.8% decode |
| D | whole-superblock coalesced read (3 lane-strided loads), no dequant/mid | 314.88 / 315.51 / 315.52 ms | 97.10 / 97.74 / 97.69 ms | **POSITIVE, 3.23x** |
| E | real field loads kept, dequant ALU collapsed to an add | 314.89 / 315.66 / 315.49 ms | 263.93 / 264.40 / 264.53 ms | **+51.2 ms = ALU cost** |

Raw: `ab-p1-fma.csv`, `summary.csv` (probe C, per-arm logs `[123][AB].dec`),
`ab-p1-loadfloor.csv` (D), `ab-p1-alufree.csv` (E), `iq2m-decode-base.txt`,
`isa-f64-counts.txt`, `p1-diag-probes.diff`, `diag.comp`, `loadfloor.spv`,
`alufree.spv`.

Probe A also dumped the ISA (`logs/bc250-sustained-20260918/shader_isa.c`):
`fma32()` is already compiled by ACO to exactly **one `v_fma_f64` per expert**
plus 3 `v_cvt_f64_f32` / 1 `v_cvt_f32_f64`, all in the `tid == 0` tail
(`isa-f64-counts.txt`).  There is no 7-op fp64 emulation sequence to remove, and
the one that exists runs once per expert per workgroup, not per weight.

Probe B: `mmq_info` reports `subgroup=32` (VGPR 48, code 4004/4112, 20 subgroups
per SIMD) for `moe_down_q2k_sum_decode{,_iq2s,_iq3s}.spv`, so the "wave64 with
half a wave idle" premise in the audit does not hold for the dispatched blobs;
only the non-IQ2S/Q2_K build reports VGPR 64 / 16 subgroups.

## Decomposition (per 128-token run, same bytes, same mapping)

```
315.4 ms  stock kernel
 97.0 ms  floor: 3 coalesced loads per lane, no dequant, no mid     (131 GB/s)
167.2 ms  the ~40 scattered 2-byte field loads per lane per block
 51.2 ms  the dequant ALU (grid unpack, sign select, dot, fma)
```

Per token: kernel 2.464 ms, floor 0.758 ms, field loads 1.306 ms, ALU 0.400 ms,
against a 13.42 ms/tok decode (74.4 t/s).  So the kernel is **not** fp64-bound,
**not** wave64-bound, and only 16% ALU-bound: it loses its time to the *shape* of
the IQ2_S field loads.  Each lane owns 16 elements of one 82 B superblock and
re-fetches its scale/grid-index/qh/sign bytes with ~40 independent 2-byte
`buffer_load_ushort`s at scattered offsets, where the Q2_K branch of the same
file needs ~7 wider loads per block and reaches ~62 GB/s.

Probe D therefore says a load-pattern restructure is worth up to
`167 ms` of a `315 ms` kernel on a 21.6%-of-decode dispatch: **+14% IQ2_M decode
if the field bytes can be fetched coalesced and redistributed** (subgroupShuffle
or LDS) while keeping the per-element fma order, which keeps the result
bit-exact.  It was **not built this session** (cost/benefit vs the remaining
audit items, all of which are still unprobed).

## Verdict

- Audit (a) wave32: **premise false**, nothing to fix.
- Audit (c) `fma32` fp64 emulation: **negative**, one f64 FMA per expert in the
  `tid == 0` tail; ACO already fused it.  Do not resurrect the "7-op fp64
  sequence" story.
- Audit (b) byte-granular loads: **confirmed as the lever**, 167 ms of 315 ms,
  quantified here for the first time.
- Serial 8-expert combine is already better than the parallel alternative
  (probe C), so the audit's "experts in parallel" suggestion is dead as-is.

`reconsider_if`: someone writes the coalesced-fetch (3 lane-strided loads +
`subgroupShuffle` redistribution, or LDS staging) version keeping the existing
per-element accumulation order, and A/Bs it on IQ2_M decode; the target is
315 -> ~200-230 ms kernel and decode 74.4 -> ~80 t/s, with `max_abs_diff = 0`
on the frontier-513 dump because only the fetch changes.
