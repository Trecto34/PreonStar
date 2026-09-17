# w7 — the 41-block MoE loader fix (ACCEPTED as capability, NOT as a speed win)

Worked-experiment record. Template: `ab-hostdrain.md`. Raw data in
`evidence/raw/`. Commit: `f9d537d` (`q36.c`, +66/−22).

## 1. Question

`Qwen3.8-35B-A3B-IQ2_M` — a 41-block `qwen35moe` file (40 trunk blocks + one
embedded NextN/MTP block, `nextn_predict_layers=1`, 753 tensors) — was refused by
q36 3/3 with `q36: expected qwen35moe.block_count=40, got 41`, while upstream
llama.cpp loads the identical file and posts **91.53 ± 2.11 decode**, the fastest
MoE decode measured on this box.

Can q36 load it without regressing the MoE guard?

## 2. Root cause (pre-established, not re-derived here)

Three separate conditions were dense-gated: the block-count escape hatch, the
tensor-count gate, and the embedded-MTP prewarm skip
(`q36_tensor_is_disabled_embedded_mtp`). Mechanism walk and exact lines:
`AlreadyTried.md` §"Cross-engine MoE matrix + the IQ2_M loader wall".

## 3. Change

| condition | before | after |
|---|---|---|
| block count (`config_validate_model`) | escape hatch required `Q36_MODEL_DENSE` | `embedded_nextn_block = nextn_layers == 1 && block_count == Q36_N_LAYER + 1`, any model kind |
| tensor count | `+15`, dense-gated | `+15` dense / **`+20` MoE** (753 − 733) |
| prewarm skip | `if (!e \|\| !Q36_MODEL_DENSE \|\| e->mtp_ready \|\| !t) return false;` | `!Q36_MODEL_DENSE` dropped; `e->mtp_ready` short-circuit retained |
| trunk matrix types | Q4_K/Q5_K/Q6_K/Q8_0 whitelist | under `!Q36_MODEL_DENSE && n_tensors == Q36_TENSOR_COUNT + 20`: any type, **dims still validated** (`tensor_expect_cpu_matrix`) |

**Routed expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are
untouched** — they keep `tensor_expect_routed_gate_up` / `tensor_expect_routed_down`.
The whitelist widening reaches the trunk only, under a predicate that cannot match
the guard (733 tensors) or any dense file. The type predicate is exact, so this is
not "loosening a check" in general; it is naming the one shape that needs a wider
type set and still binding it to a dims check.

## 4. Evidence

### 4.1 It loads and runs
`116.51 / 29.29` tps, ctx 1024, profiled (`evidence/raw/w7-iq2m-prof.txt`).
The 3/3 refusal is gone.

### 4.2 Quality — the relaxed check is the thing being tested
`q36_test --gpu-cpu-parity --case short_reasoning_plain` (`evidence/raw/w7-parity.txt`):

| model | step | top1 ref/cand | top5 | top15 | top64 | rms |
|---|---|---|---|---|---|---|
| **IQ2_M** | 0 | 16 / 16 | 5/5 | 15/15 | 58/64 | 0.263 |
| **IQ2_M** | 1 | 21 / 21 | 5/5 | 14/15 | 59/64 | 0.208 |
| **IQ2_M** | 2 | 248046 / 248046 | 4/5 | 13/15 | 57/64 | 0.272 |
| guard | 0 | 16 / 16 | 5/5 | 14/15 | 58/64 | 0.306 |
| guard | 1 | 21 / 21 | 5/5 | 13/15 | 53/64 | 0.241 |
| guard | 2 | 248046 / 248046 | 4/5 | 14/15 | 60/64 | 0.322 |

Both `gpu-cpu-parity: OK`. Top-1 agreement at every step, rms ≈ 0.2–0.3 (normal
cross-backend float drift for this engine). **A mis-dispatched quant type would
break top-1 agreement against the CPU reference; it does not.** IQ2_M's overlap is
marginally *better* than the guard's.

### 4.3 No regression on the guard — 7-rep interleaved A/B
`tests/bench_ab.sh <A> <B> Huihui-…-IQ2XXS.gguf 7`, ctx 1024, `--gen-tokens 16`,
`--prefill-chunk 256` (`evidence/raw/ab-w7-guard.csv`):

| bin | n | prefill median | MAD | decode median | MAD | sha256 |
|---|---|---|---|---|---|---|
| A = `56126ca` build | 7 | **713.80** | 6.180 | **88.98** | 0.110 | `ffd2d262…8883` |
| B = w7 build | 7 | **713.85** | 10.570 | **89.12** | 0.080 | `3264f5dc…afbb` |

**+0.01% prefill / +0.16% decode — indistinguishable.** The harness prints
`verdict: FAIL (prefill)` because its question is "is B ≥1.5% *faster*"; the
question here was "is B slower", and it is not. Reading the label instead of the
medians would have rejected a correct fix.

Per-rep spread also reclassifies two earlier single readings: A ranged 704.62 →
737.83 prefill and B 701.84 → 734.02, so the orchestrator's one-shot `686.84` and
codex's `730.79` were both cold-run noise, not signals. **Single runs on this MoE
are worth nothing; only the interleaved median is.**

## 5. Verdict

**ACCEPTED** — as a *capability* fix: a file the engine could not open now loads
and decodes with verified parity, at zero measured cost to the guard.

**NOT a speed win.** Under q36, IQ2_M is 4.5×/3.1× slower than upstream
(116.51/29.29 vs 529.90/91.53). That gap is now characterized and is not the
loader's fault:

| | guard | IQ2_M |
|---|---|---|
| MoE expert compute | `moe_iq2_gate_up_gemm` 878.7 + `moe_q2k_down_gemm` 432.6 = **1,311 ms** | `moe_matvec` **11,845 ms** (67% of all GPU time) |
| trunk | `dense_q8_0_p_*` GEMMs ≈ **950 ms** | `dense_kquant` **4,662 ms** |
| result | 686.84 / 81.36 | 116.51 / 29.29 |

IQ2_M's trunk is **375× `IQ2_S` (10.3 GB)**; the guard is `IQ2_XXS` + `Q2_K`.
Every tuned prefill path here is quant-specific, so IQ2_S experts fall through to
the generic per-expert matvec and the K-quant trunk to the generic kquant matmul.
`IQ2_S` support already exists in-tree — the fix is **quant-aware dispatch into the
per-quant prefill GEMM family**: in-tree prior art, not new shader work. That is
now open item **#1** in `CAMPAIGN.md`.

**Reconsider_if:** a future MoE GGUF of the same 753-tensor shape ships non-trunk
types the generic dispatch does not actually handle — the mixed-quant predicate is
by tensor count and would not re-check the whitelist. Run `--gpu-cpu-parity`
before trusting any new file that takes that path.

## 6. Process notes worth keeping

- A worker's `+66/−22` summary is not a verdict: the patch had to be read, the
  expert-tensor validation checked, and the run measured on this box. The
  worker's own IQ2_M number (`118.26 / 32.43`) was accurate but measured the wrong
  question — "does it load", not "is it good".
- `pkill -f <pattern>` matched the orchestrator's own command line and killed the
  shell (exit −15). Kill workers **by PID**. The same self-match bug later hung an
  exit-watcher whose own command line contained the pattern it polled.
- Both arm binaries must be hashed into the record: the A binary is gitignored and
  is overwritten the moment anyone rebuilds the main checkout.
