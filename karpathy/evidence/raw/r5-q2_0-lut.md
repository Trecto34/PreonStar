# R5 — Q2_0 decode: SWAR expansion is the ALU tax; shared LUT recovers it

Branch: `experiment/radiance-transfer-bc250`. Baseline: committed `6964c87`
(== shader at report HEAD `4e5978f`).

## Method

The proposed stride ablation (halve the block count) scales weight loads and
ALU together and cannot separate them. Instead a `Q36_Q2_0_ABLATE` mode was
temporarily added to `dense_extra_decode.comp`, changing only the hot loop:

| mode | change |
|---|---|
| 0 | production: SWAR `q2_i8x4` + `dotPacked4x8` |
| 1 | weight loads kept, expansion **and** dot replaced by cheap int ops |
| 2 | weight loads removed (activation/loop floor) |
| 3 | `dotPacked4x8` on raw code bytes, no SWAR expansion (wrong math) |
| 4 | shared-LUT expansion (same values as `q2_i8x4`), correct math |

`Q36_VK_PROF_KERNEL=1`, `ctx 512 / gen 128`, soak gate ≤55 °C. The metric is
the `op dense_extra_decode` `gpu_ms` row (128 decode tokens).

## Results

Hot regime (start 55 °C), 3 reps each for modes 0/1/2:

| mode | gpu_ms (median) | vs full |
|---|---|---|
| 0 full | 4284 | — |
| 1 no ALU | 2937 | −31.4 % |
| 2 no weight loads | 589 | −86.3 % |
| 3 dot, no expansion | 3103 | −27.6 % |
| 4 LUT expansion (correct) | 3150 | −26.5 % |

Cooler interleaved confirm (start 39–55 °C), 3 reps:

| rep | mode 0 | mode 4 |
|---|---|---|
| 1 | 3756 | 3038 |
| 2 | 3775 | 3048 |
| 3 | 4237 | 3111 |
| median | 3775 | **3048 (−19.2 %)** |

Decomposition: non-weight floor 589 ms, weight streaming 2348 ms, dot 166 ms,
SWAR expansion ~1181 ms. The near-perfect additivity
(589+2348+166+1181 = 4284) shows the loop has essentially no memory/ALU
overlap; the SWAR expansion is the dominant removable ALU cost.

## Landed change

`vulkan/dense_extra_decode.comp`: the Q2_0 path builds a 256-dword (1 KiB)
`shared` table once per workgroup and replaces the 10-op SWAR spread with one
LDS read per 4 codes. The non-Q2_0 path and `q36_vulkan.c` are unchanged.

## Soak-gated interleaved A/B vs HEAD (3 pairs, ctx 512 / gen 128)

| pair | base tg128 | LUT tg128 | base pp512 | LUT pp512 |
|---|---|---|---|---|
| 1 | 28.38 | 33.32 | 197.81 | 200.80 |
| 2 | 25.28 | 33.30 | 193.61 | 200.71 |
| 3 | 24.93 | 32.82 | 198.42 | 201.35 |
| **median** | **25.28** | **33.30 (+31.7 %)** | ~197 | ~201 |

Coherence gate (sheep/Canberra) passes on the LUT kernel. Raw rows:
`logs/thermal_run_log.csv` (`lutab-*`); profiler dumps:
`logs/profiles/ablate-*.txt`, `logs/profiles/lutab-*.txt`.

## Phase 2: software-pipeline attempt (negative)

The spec asked for a cross-block ping-pong double buffer. Each lane's block
loop is only ~3 iterations (`blocks=20`, stride 8), so cross-block prefetch has
little to hide; the exposed serialization is inside the 4-row group. The
attempt therefore hoisted the four rows' `codes0/codes1/scale` loads ahead of
the LUT/dot (3×4 extra live registers), keeping correctness (sheep/Canberra
pass). Soak-gated interleaved A/B, 3 pairs:

| rep | lut tg128 | pipelined tg128 |
|---|---|---|
| 1 | 33.32 | 31.39 |
| 2 | 32.74 | 31.03 |
| 3 | 32.54 | 30.93 |
| median | **32.74** | **31.03 (−5.2 %)** |

The extra live registers cost more occupancy than the load hoisting recovered.
**Rejected**; the tree stays on the LUT kernel. Raw rows `pipeab-*` in
`logs/thermal_run_log.csv`, dumps in `logs/profiles/pipeab-*.txt`.

## Verdict

- Phase 1 gate (≥18 % kernel-time reduction, target ~28–29 t/s): **pass**, at
  33.3 t/s.
- Phase 2 gate (≥31 t/s): met by Phase 1; the explicit pipeline costs 5.2 % and
  is not landed. The LUT kernel is already within ~4 % of the no-ALU memory
  floor (mode 4 3048 ms vs mode 1 2937 ms), so there is little left to hide.
- SB288 reblock stays closed; the memory path is not the limiter for this win.
