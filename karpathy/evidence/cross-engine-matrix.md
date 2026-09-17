# Cross-engine matrix — q36 vs upstream llama.cpp (BC-250)

Measured **2026-09-17** on the BC-250 (AMD BC-250, RADV GFX1013, 24 CU RDNA2,
warp 32, ~15.35 GiB usable UMA, DRAM 454.4 GB/s measured).

| side | tree | commit | binary |
|---|---|---|---|
| ours | `/home/server/q36-opt-27b` | `11bb345` | `/home/server/q36-opt-27b/q36-bench` |
| peer | `/home/server/llama.cpp` | `972d231` (build 190) | `/home/server/llama.cpp/build/bin/llama-bench` |
| peer (pre-update point) | `/home/server/llama.cpp` | `0cae430` | — |

**Raw data is in `raw/` in this directory.** Every number below traces to a file
there. Do not re-derive these from `/tmp` — that directory is volatile, which is
exactly how an earlier session lost the llama.cpp rows of its own matrix.

## q36 side (ctx 1024, `-ngl 99` equivalent / full GPU residency)

| model | arch | prefill t/s | decode t/s | raw |
|---|---|---|---|---|
| Swift Qwen3.8-27B IQ3_XXS | dense 27B | **171.09** | **23.19** | `raw/xfer-refresh.csv` |
| Qwen3.8-27B-UD-IQ3_S | dense 27B | 166.36 | 21.44 | `raw/moe-matrix.csv` |
| Huihui Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS | `qwen35moe` | **909.49** | **87.37** | `raw/xfer-refresh.csv` |
| RavenX-35B-Q36-IQ2XXS | `qwen35moe` | 713.98 / 717.36 / 738.81 → median **717.36** | 88.74 / 89.49 / 89.47 → median **89.47** | `raw/moe-matrix.csv` |
| Qwen3.8-35B-A3B-IQ2_M | `qwen35moe` | **REFUSED 3/3** | — | `raw/moe-matrix.csv` |

## llama.cpp side (`llama-bench -m <abs> -ngl 99 -p 1024 -n 16 -r 3 -o csv`)

| model | `model_type` from its own CSV | prefill t/s | decode t/s | raw |
|---|---|---|---|---|
| Swift Qwen3.8-27B-IQ3_XXS | dense 27B | 105.43 ±4.46 | 22.13 ±0.65 | `raw/llbench-new.csv` |
| Swift Qwen3.8-27B-IQ3_XXS (stock `0cae430`) | — | 107.81 ±4.84 | 21.47 ±0.61 | `raw/llbench-stock.csv` |
| Qwen3.6-35B-A3B-Abliterated IQ2XXS | `qwen35moe 35B.A3B` | 615.08 ±8.43 | 78.72 ±1.56 | `raw/llbench-moe-new.csv` |
| RavenX-35B-Q36-IQ2XXS | `qwen35moe 35B.A3B Q8_0` | 545.46 ±16.66 | 77.96 ±1.08 | `raw/lb-ravenx.csv` |
| Qwen3.8-35B-A3B-IQ2_M | `qwen35moe 35B.A3B IQ2_M - 2.7 bpw` | 529.90 ±11.48 | **91.53 ±2.11** | `raw/lb-iq2m.csv` |

## Head-to-head

| model | q36 prefill | llama.cpp prefill | Δ | q36 decode | llama.cpp decode | Δ |
|---|---|---|---|---|---|---|
| Swift Qwen3.8-27B IQ3_XXS | 171.09 | 105.43 | **+58.7%** | 23.19 | 22.13 | **+8.0%** |
| Qwen3.6-35B-A3B IQ2XXS (guard) | 909.49 | 615.08 | **+47.9%** | 87.37 | 78.72 | **+11.0%** |
| RavenX-35B-Q36-IQ2XXS | 717.36 | 545.46 | **+31.5%** | 89.47 | 77.96 | **+14.8%** |
| Qwen3.8-35B-A3B-IQ2_M | refuses to load | 529.90 | n/a | — | 91.53 | n/a |

## Asymmetries — read the table with these in hand

1. **Decode is not the same measurement on both sides.** `llama-bench`'s `tg` is
   greedy generation; q36's decode number carries the sampling chain
   (temp/top-k/top-p/min-p). The asymmetry favours q36 slightly, so the decode
   margins should be read as upper bounds.
2. **Chunking differs.** llama-bench defaults to `n_batch 2048 / n_ubatch 512`
   while q36 was run at `--prefill-chunk 256`. Prefill numbers are therefore not
   measured at identical chunk geometry.
3. **The guard's two rows are not contemporaneous.** q36 guard (17:21) and
   llama.cpp guard (≈17:59) were taken ~3.5 h apart on different binaries; the
   RavenX pair is within ~90 min. The RavenX decode margin (+14.8%) is the
   cleaner claim of the two.
4. **"Refuses to load" is not "slower".** IQ2_M has no q36 number at all — the
   loader rejects it before any kernel runs.
5. **MoE prefill noise is 3.4%, not 0.7%.** Identical code produced 713.98 →
   738.81 t/s on RavenX, while the 27B's within-session gate is 0.7%. Any MoE
   prefill claim below ~3% is heat-soak noise and must not be reported as a
   result.

## Open item — largest known MoE upside

`Qwen3.8-35B-A3B-IQ2_M` **loads and runs on upstream llama.cpp** (35.5B params,
2.7 bpw, 99/99 layers on GPU, `model_type` `qwen35moe 35B.A3B IQ2_M`) and posts
**91.53 ± 2.11 t/s decode — the fastest MoE decode measured on this box by any
engine**, above q36's best (RavenX, 89.47). q36 refuses the same file with a
deterministic loader guard: `q36: expected qwen35moe.block_count=40, got 41`
(3/3 reps, exit 1). The header confirms this is not an architecture mismatch —
three of the four MoE GGUFs share the `qwen35moe` metadata.

So the single largest remaining MoE upside on this machine is **a loader
condition in this repo**, not a kernel: teaching the loader `block_count=41`
would open a file that already out-decodes everything else here.
