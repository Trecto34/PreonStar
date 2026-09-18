# `dense_extra_decode` (Q2_0) inner-loop coalescing & latency audit

Branch `experiment/bc250-sustained-20260918`, HEAD `c133a09`. Model
`gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf`. Target: decode >= +1.5 % without
regressing pp512 and with a bit-exact numerical contract.

**The prior blocker is stale.** `reports/compiler_and_occupancy_audit.md` and the
main report both record that "the ACO internal-representation query crashes this
RADV build". It does not. `logs/bc250-sustained-20260918/shader_isa.c` builds the
pipeline with `VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR` and
does the three-pass `vkGetPipelineExecutableInternalRepresentationsKHR` query
(count -> sizes -> data); RADV returns NIR (54,343 B), ACO IR (37,947 B) and
**Assembly (55,554 B)** without faulting. It runs unprivileged, in a short-lived
process, with no `RADV_DEBUG`. Every claim below is read off that assembly, which
is archived in `logs/bc250-sustained-20260918/isa/`.

## 1. Load vectorization width

Not scalar. Per loop iteration the wave issues:

| what | instruction | bytes |
|---|---|---|
| activations `qv[0..7]` | `buffer_load_dwordx4` x2 (`offset:8`, `offset:24`) | 32 |
| `yd` activation scale | `buffer_load_dword` | 4 |
| **weight codes, per row x4** | **`buffer_load_dwordx3`** | 12 |
| weight f16 scale, per row x4 | `buffer_load_dword` | 4 |

The GLSL reads weights through two `load_u32()` calls at `code_base` and
`code_base + 4`. Q2_0 blocks are 18 B, so those addresses are never 4-byte
aligned and each `load_u32` is written as a two-dword read plus a shift-or. ACO
recognises the pair, merges all of it into **one `buffer_load_dwordx3`** and
extracts with two `v_alignbit_b32`:

```
buffer_load_dwordx3 v[21:23], v15, s[12:15], 0 offen
buffer_load_dword   v5,  v5,  s[12:15], 0 offen
s_waitcnt vmcnt(1)
v_alignbit_b32 v21, v22, v21, v20
v_alignbit_b32 v23, v23, v22, v20
```

Three dwords is the minimum that can cover 8 bytes at an arbitrary byte offset.
**There is no vectorization left to win here** — `uvec2`/`uvec4` cannot be used,
because the 18-byte block stride makes the data 2-byte aligned by construction.

## 2. Subgroup transaction alignment

`lid` 0..63 splits as `b = lid >> 3` (8 blocks in flight) and `lane8 = lid & 7`.
Within one Q8_K block (4 Q2_0 blocks = 72 B) the eight lanes land at:

| lane8 | `code_base` | dword-aligned `dwordx3` extent |
|---|---|---|
| 0 | +2 | [0, 12) |
| 1 | +10 | [8, 20) |
| 2 | +20 | [20, 32) |
| 3 | +28 | [28, 40) |
| 4 | +38 | [36, 48) |
| 5 | +46 | [44, 56) |
| 6 | +56 | [56, 68) |
| 7 | +64 | [64, 76) |

Union = `[0, 76)`, and across the full wave (`b` = 0..7) one load instruction
covers a **contiguous `[row_base, row_base + 580)` byte region**. So the wave
already reads one contiguous burst; there is no striding and no split
transaction.

Two second-order costs, both real but both off the DRAM path:

- **L0 over-request 1.78x.** The eight lanes request 8x12 B (codes) + 8x4 B
  (scales) = 128 B to consume 72 B of block data, because neighbouring lanes'
  12-byte windows overlap. This costs texture-unit request slots, *not* DRAM
  bytes — the overlapping windows hit the same cache lines.
- **Row base is 8-byte, not 128-byte, aligned.** `row_bytes = 72 * blocks`, which
  is never a multiple of 128, so each row's 580 B span straddles a cache line at
  both ends. A workgroup covers 4 consecutive rows, so the partial line at a row
  end is consumed by the next row of the same workgroup; the waste is confined to
  the workgroup boundary.

**Conclusion: the loads are coalesced.** Decision Gate 1's first condition is
met.

## 3. Latency hiding — this is where the gap is

Gate 1's second condition is **not** met. The four output rows do not share a
basic block:

```
BB6:   s_cmp_lt_u32 s9, s3
       s_cbranch_scc0 BB9        ; row 0 guard  -> own block
BB7:   s_load_dwordx4 s[12:15]   ; weight descriptor RELOADED
       buffer_load_dwordx3 ...   ; row 0 codes
       s_waitcnt vmcnt(1)        ; ...waited 5 instructions later
```

`if (row >= pc.out_dim) continue;` is **wave-uniform**, so ACO emits a scalar
branch and each row becomes its own block (`BB7`/`BB10`/`BB13`/`BB16`). Two
consequences, both visible in the assembly:

1. **The codes load is not latency-hidden.** Only 5 instructions separate
   `buffer_load_dwordx3` from its `s_waitcnt vmcnt(1)`. By contrast the f16 scale
   load *is* hidden — its `s_waitcnt vmcnt(0)` is ~49 instructions downstream.
   The four rows' loads are never in flight together, and nothing is prefetched
   across iterations of the `b` loop.
2. **The weight descriptor is re-fetched per row.** `s_load_dwordx4 s[12:15],
   s[0:1], null` appears at four separate points inside one iteration, each
   followed by `s_waitcnt lgkmcnt(0)`.

The guard is **dead at runtime**: every Q2_0 `out_dim` in this model is a
multiple of 64 and `ROWS = 4`, so `first_row + r < out_dim` always holds. It
costs scheduling freedom and buys nothing.

## 4. Resource utilization (measured, `VK_KHR_pipeline_executable_properties`)

| | baseline `c133a09` | clamp candidate |
|---|---|---|
| SGPRs | 108 | 108 |
| **VGPRs** | **28** | **36** |
| spilled SGPR / VGPR | 0 / 0 | 0 / 0 |
| LDS per workgroup | 1024 B | 1024 B |
| code size | 3824 B | 3652 B |
| **subgroups/SIMD** | **36** | **28** |
| basic blocks | 21 | 13 |
| descriptor reloads per iteration | 9 | 4 |

Dispatch is `out_dim/4` workgroups of 64 threads (one wave64, `ROWS = 4`).

## 5. Phase 2 candidate

Replace the dead uniform branch with a clamp, so the four rows share one block
and ACO can batch their loads and keep the descriptor live:

```glsl
uint row = min(first_row + r, pc.out_dim - 1u);   // was: if (row >= out_dim) continue;
```

A clamped row reads live memory and its accumulator is never stored (the store
loop still guards on `out_dim`), so every real row's arithmetic is unchanged.

It does what it was meant to do structurally — 21 -> 13 blocks, 9 -> 4 descriptor
loads — but **ACO spends the freedom on registers: 28 -> 36 VGPRs, and
subgroups/SIMD falls 36 -> 28.** Four rows' worth of `dwordx3` + scale held live
is 16 VGPRs where the branched form needed 4. That breaches the audit's stated
28-VGPR ceiling, so the candidate is measured, not assumed — see §6.

## 6. Verification

**Correctness — PASS, both gates.**

| check | result |
|---|---|
| frontier-512 logits, n = 248,320 | `max_abs_diff = 0.0` — bit identical |
| greedy generation, temp 0 | byte-identical, 5,901 B (sha256 `7f316c5c...10dc`) |

**Performance — FAIL.** 10 interleaved soak-gated pairs (entry edge 55-56 C),
same host binary, only `dense_extra_decode_q2_0.spv` swapped.

| | baseline A | clamp B |
|---|---|---|
| decode median | 32.66 t/s (MAD 0.16) | 32.70 t/s (MAD 0.07) |
| decode ms/token | 30.623 | 30.576 (-0.047) |
| **median paired change** | | **+0.34 %** (6/10 positive) |
| aggregate-of-medians | | +0.15 % |
| pp512 median | 206.10 t/s | 203.75 t/s (median paired -0.90 %) |

Per-pair decode change: -1.21, -0.03, +0.49, -0.21, +1.14, +0.34, +0.90, +0.34,
-0.31, +0.71 %.

Telemetry: SCLK median 1315 MHz (1220-1690), MCLK 450 MHz (DPM display only --
this box exposes no live memory-clock counter), entry edge 55-56 C, package
50-124 W.

Gate is >= +1.5 % median paired decode. **+0.34 % -> FAIL, rejected.**

## 7. Verdict — Decision Gate 1: freeze the inner-loop campaign

The interesting part is *why* it failed, because it falsifies two things at once.

1. **The audit's own VGPR premise is not supported.** Constraint 1 predicted that
   dropping occupancy "will degrade throughput". Occupancy fell 36 -> 28
   subgroups/SIMD and decode did **not** regress (+0.34 %, and 6/10 pairs
   positive). At this arithmetic intensity the kernel is nowhere near needing 36
   waves to cover its latency. So 28 VGPRs is not a cliff — but neither is
   spending registers a lever.
2. **The kernel is not latency-bound, so the latency gap found in §3 was not
   worth closing.** Batching four rows' loads, halving the basic blocks and
   removing five descriptor reloads per iteration bought +0.34 %, which is inside
   this run's own spread. If the codes load's 5-instruction wait had been
   costing real time, removing it would have shown more than a third of a percent.

Combined with §1 (loads already at the minimum width the 18-byte block stride
permits) and §2 (one contiguous 580 B burst per wave, no split transactions),
the conclusion is that `dense_extra_decode_q2_0` is **memory-bound and already
issuing near-optimal transactions**. There is no remaining inner-loop lever:
vectorization is capped by the format, coalescing is already achieved, and
latency hiding is demonstrably not the constraint.

**Recommendation: freeze the inner-loop campaign.** The remaining headroom is in
the *format*, not the loop — see `AlreadyTried.md` for the PTQ1_0 note: 1.75 bpw
vs the current 2.25 bpw is ~22 % less weight traffic per token on a kernel this
audit just showed is limited by exactly that traffic.

Raw data `logs/bc250-sustained-20260918/runs/ab-decode-clamp.csv`; rejected patch
`logs/bc250-sustained-20260918/decode-clamp-rejected.patch`; assembly
`logs/bc250-sustained-20260918/isa/`.
