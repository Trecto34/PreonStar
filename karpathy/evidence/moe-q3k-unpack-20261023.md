# moe_matvec / moe_matvec_fast Q3_K unpack — word slices (2026-10-23)

Model: /home/server/models/llm/KAT-Coder-V2.5-Dev.APEX-I-MiniPlus-V2.1.gguf
(KAT-Coder-V2.5-Dev.APEX-I-MiniPlus-V2.1, qwen35moe, 40 layers, 256 experts,
8 used, ff 512, embed 2048, gate/up/down experts all Q3_K, 14 067 MiB).

## Per-kernel GPU time, decode window

Command (identical before/after; the window is ~11.8 decode tokens, same
dispatch counts in both arms, so ms are directly comparable):

    P36_VK_PROF_KERNEL=1 P36_VK_PROF_SKIP=3000 P36_VK_PROF_TIMELINE=/tmp/tl.txt \
      karpathy/tools/guard.sh MODEL -- ./p36 -m MODEL --vulkan -n 64 --ctx 512 \
      --nothink -p hello

| kernel | dispatches | before ms | after ms | delta |
|---|---:|---:|---:|---:|
| moe_matvec (exact, gate) | 474 | 60.243 | 55.775 | -7.4% |
| moe_matvec_fast (up+down) | 468 | 45.204 | 37.149 | -17.8% |
| both | 942 | 105.447 | 92.924 | -11.9% |

Baseline kernel ranking in the same window: moe_matvec 32.0%, moe_matvec_fast
24.0%, moe_tiles 10.6%, dense_q3k_decode 7.2%, dense_kquant_decode 5.8%,
dense_q5k_decode 3.5%, matmul_q8_0_f32b 3.1%, matmul_f32_fast 2.8%. GPU busy
86.9%, total 188.2 ms of dispatch time in a 216.4 ms span.

## End-to-end, five interleaved pairs

Each binary run from its own tree (own .spv set); A = worktree at HEAD 97136bb,
B = working tree with the Q3_K word slices.

    pair 1  base 40.08   new 42.46
    pair 2  base 40.14   new 42.84
    pair 3  base 40.32   new 42.96
    pair 4  base 40.33   new 41.92
    pair 5  base 40.45   new 41.79

median base 40.32, median new 42.46, **+5.3%**, ranges disjoint
(base max 40.45 < new min 41.79).

## Correctness

Greedy 64 tokens, prompt "The capital of France is", --temp 0:

    HEAD worktree shaders   55f555660f6158501394ff16bde68faa
    word-slice shaders      55f555660f6158501394ff16bde68faa

## Negative results kept here so they are not re-run

- **q8 word slicing alone is ~neutral.** One sh_q8 word read per (group,pair)
  instead of four (256 -> 64 shared loads per block per thread): moe_matvec
  60.24 -> 59.79, moe_matvec_fast 45.20 -> 44.17. Not LDS-bound.
- **fma32 ablation.** Replacing fma32() with native fma() in moe_matvec:
  60.243 -> 44.773 ms. The correctly-rounded software FMA is 15.5 ms (25.7% of
  that kernel, ~8% of decode). Reverted: it is a numerics decision, not a load
  cleanup.
