# Long-context prefill for agentic use — dense Swift-Qwen3.8-27B (2026-09-22)

Branch `perf/attn-prefill-fa` (from `8e8e788`). Running record; newest section last.

## 0. Why

Agent sessions send 5k–30k-token prompts. Short-context gates (ctx 512/1024) hide
the cost that matters there. Target model: `Swift-Qwen3.8-27B-IQ3_XXS.gguf`
(dense, GQA 24/4 = 6, head_dim 256, Q8_0 K / Q4_0 V cache).

## 1. Baseline at agent-sized contexts

`Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 ./q36-bench --vulkan -m <swift>
--prompt-file tests/long_context_story_prompt.txt --ctx-start N --ctx-max N
--ctx-alloc N+256 --prefill-chunk 256 --gen-tokens 0`

| ctx | q36 prefill | llama.cpp `-fa 1` pp | attention share of GPU time |
|---|---|---|---|
| 2048 | 145.89 t/s | – | 6.9% (`attn_prefill_qtile2_gqa6` 951.7 ms) |
| 8192 | 104.96 t/s | 72.54 t/s | **20.8%** (16143.2 ms of 77588.7 ms) |

8192 ranking: `dense_iq3_mmq` 32.7%, `dense_iq3_xxs_pair_mmq` 22.5%,
`attn_prefill_qtile2_gqa6` 20.8%, `dense_kquant_mmq` 10.9%, `dense_iq4_xs_mmq` 8.7%.
Attention is the only term that grows with context (quadratic), so it is the
agentic lever; the matmuls are already at ~39% of packed-f16 peak and heavily
mined (`AlreadyTried.md`).

Attention efficiency: 4·24·256·8192²/2·16 layers ≈ 13.2 TFLOP in 16.1 s ≈
**0.82 TFLOP/s**, ~8% of the chip's f32 FMA rate. Source read: the kernel serves 2
tokens per workgroup and reads six Q rows from LDS for every K vector (~1 FMA per
LDS float), and each 2-token workgroup re-dequantizes the full causal K/V prefix.

Raw: `raw/fa-base-swift-prof-ctx{2048,8192}.txt`. Same shape on the MoE files
(`raw/fa-base-{iq2m,guard}-prof-ctx8192.txt`): attention 29% / #1 at ctx 8192.

## 2. Side findings (not code changes)

- **Heat soak is real for long runs.** The 2k profile rerun immediately after an
  8k run: 145.89 → 134.39 t/s (−7.9%), `dense_iq3_mmq` 5351 → 5807 ms (+8.5%).
  Per-token matmul cost is +19% inside the 8k run itself. Board hit 101 °C
  during llama-bench. This refines the campaign's "thermal is not a factor",
  which was measured on ~6 s runs. Owner is fitting a better fan; not pursued.
  Raw: `raw/fa-base-swift-prof-ctx2048-hot.txt`.
- **Prefix KV reuse works end to end** (agent-loop simulation, Swift, ~5.7k-token
  system prompt + tools, `raw/kvreuse-agent_turns.py`): cold turn 46.07 s;
  following turns 1.33 s / 1.92 s (29 new tokens each, `memory-token` hit);
  after a server restart, disk hit (5839 tokens, 410 ms load) → 2.24 s.
  Raw: `raw/kvreuse-srv{1,2}.txt`. So in a steady agent loop only the suffix pays
  prefill; the kernel work below targets cold prompts and prefix breaks.
- `tests/benchmark_prompt.sh` (untracked) forced `--ssd-streaming` on resident-
  sized MoE files: 3.49/1.64 t/s vs 175.94/80.08 resident. Removed.

## 3. Kernel: `attn_prefill_fa.comp` (built as `_gqa6` and `_gqa8`)

Flash-attention shape: one workgroup (256 threads) = one KV head × 8 tokens × 6
heads = 48 rows. K/V dequantized once per 16-key tile into LDS as f32 (Q8_0 /
Q4_0 dequant is exact in f32). Q in registers; a 16-lane rowset owns 3 rows,
lane j owns dims 64i+4j..+3, so K/V LDS reads are contiguous across the rowset
and broadcast across rowsets; scores finished with a 16-lane clustered add.
Same per-4096-key-group partials, so `attn_combine` is untouched. Gate:
`Q36_VK_ATTN_FA` (default on, `=0` → qtile2).

ISA (`./mmq_info`): VGPRs 256 (no spills), LDS 32768 B, 4 subgroups/SIMD —
vs qtile2 128 VGPRs / 16384 B / 8 subgroups.

Parity/timing harness: `tests/test_attn_fa.c` (both kernels through the real
dispatch, cases n_tok 2..256, pos0 0..16128, sinks on/off, group-edge straddle).

## 4. v1 result — kernel level (independent review: no bugs, "safe to GPU-test")

`tests/test_attn_fa` (best of N, includes readback; raw `raw/fa-v1-kernel-test.txt`):

| n_tok | pos0 | qtile2 ms | fa ms | speedup | max_abs |
|---|---|---|---|---|---|
| 256 | 0 | 3.25 | 1.69 | 1.93x | 1.4e-6 |
| 256 | 3968 | 21.73 | 8.98 | 2.42x | 5.7e-6 |
| 256 | 7936 | 37.31 | 15.19 | 2.46x | 4.2e-6 |
| 256 | 16128 | 73.07 | 27.88 | 2.62x | 3.6e-6 |
| 1024 (2 slices) | 7936 | 150.12 | 59.12 | 2.54x | 6.1e-6 |

max_abs is f32 reassociation size (outputs ~1.0). Review-driven test fixes: a
multi-slice case (tok_base != 0), clearing the QTILE/FUSED env overrides that
would make both arms identical, one sinks buffer for the run.

## 5. Whole-model verdicts (ctx 8192, 5 interleaved reps, gen 16, chunk 256)

Same binary both arms, wrappers differ only in `Q36_VK_ATTN_FA` (rule 5: flag ON in B).

| model | A qtile2 prefill (MAD) | B FA prefill (MAD) | delta | reps disjoint? |
|---|---|---|---|---|
| Swift 27B dense (GQA 6) | 97.98 (0.550) | **107.28** (1.880) | **+9.49%** | yes (min B 102.94 > max A 100.34) |
| Qwen3.6-35B-A3B guard (GQA 8) | 386.47 (1.650) | **491.38** (3.090) | **+27.15%** | yes (min B 486.94 > max A 417.35) |

Raw: `raw/ab-fa-ctx8192.{csv,summary}`, `raw/ab-fa-guard-ctx8192.{csv,summary}`.

Guard decode read −4.92% in the A/B with matching temperatures, so it was checked:
differential decode profile (gen 65 − gen 1, /64) gives GPU decode time **15.572 vs
15.580 ms/tok (+0.05%)** and no FA dispatch in decode (n_tok = 1 takes
`attn_decode_split`). No GPU mechanism; treated as clock/power state after a much
denser prefill. Raw: `raw/fa-guard-decodeprof-fa{0,1}-g{1,65}.txt`. Kernel test also
covers 2- and 16-token steps at pos0 ~8190 (MTP verify, tiny chunks): FA 2.4x
(GQA 6) / 3.4x (GQA 8) faster, parity ok.

## 6. GQA-8 variant (MoE)

Same source, `-DHQ=8`: 4 tokens x 8 heads, 2 rows per rowset (HQ=6's 3-row
shape already sits at 256 VGPRs). ISA: **128 VGPRs, 8 subgroups/SIMD**, LDS 32768 B.
Kernel test 3.0–3.5x vs `attn_prefill_qtile2` at pos0 >= 3968
(`raw/fa-v2-kernel-test.txt`, 18/18 PASS). The lower-VGPR shape is faster per
row than the GQA-6 build, which points at the next tuning step for the dense
kernel (occupancy, not math).

## 7. Quality

Frontier logits at ctx 8192 are chaotic on the dense model under **any** f32
reordering, so single-frontier max_abs is not a usable gate there:

| comparison (Swift) | 2k max_abs / top64 | 8k max_abs / top-1 / top64 |
|---|---|---|
| qtile2 vs qtile2 rerun | – | 0 / same / 64 (deterministic) |
| FA vs qtile2 | 1.38 / 63 | 13.48 / same / 38 |
| qtile2 chunk 256 vs 128 (legit reorder) | 2.21 / 59 | 8.07 / **flipped** / 40 |

So the gate used is teacher-forced NLL of the true next token at 16 frontiers
7952..8192 (`karpathy/tools/frontier_nll.py`, tokens from `q36 --dump-tokens`),
with a chunk-128 baseline as the noise reference:

| model | arm | mean NLL | top-1 | mean \|dNLL\| vs baseline |
|---|---|---|---|---|
| Swift | qtile2 | 15.35 | 3/16 | 0 |
| Swift | **FA** | 13.27 | 5/16 | **3.08** |
| Swift | qtile2 chunk 128 | 14.15 | 4/16 | 3.58 |
| guard | qtile2 | 1.87 | 12/16 | 0 |
| guard | **FA** | 2.91 | 12/16 | **2.01** |
| guard | qtile2 chunk 128 | 3.80 | 11/16 | 2.60 |

FA moves NLL less than a plain chunk-size change does on both models: inside
the envelope of legitimate f32 reassociation, not a defect. (Swift's high NLL
is the prompt — literal `<|im_start|>` text tokenized as characters — and is
identical across arms.) Guard frontier-8192: max_abs 1.79, top-1 same, top64
59/64. Raw: `raw/fa-parity.txt`.

`reconsider_if` (for anyone reverting): a real-task eval shows long-context
regression that the chunk-size reference does not also show.
