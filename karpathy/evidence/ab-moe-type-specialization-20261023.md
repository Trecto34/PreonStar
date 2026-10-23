# A/B — MoE matvec type specialization + Q4_K packed loads (APEX-I decode)

**Date:** 2026-10-23 · **Verdict: REJECT both — neutral within noise**
**Target:** `MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf` (qwen35moe, 11.69 GiB)
**Raw:** `raw/ab-moe-type-kernels.log`

Caveat up front: these arms were run **3 reps**, below §4's 7-rep gate, and are
recorded as *observational*. They are consistent across both arms and agree with
an independent explanation (see `decode-disk-bound-20261023.md`), so the
conclusion "no lever here" is safe even though the per-point numbers are not
gate-grade.

## Candidate 1 — type-specialized MoE matvec

The routed-expert matvec is one shader with a runtime `if (pc.type == ...)` chain
over nine ggml types. Candidate: compile the same source once per k-quant with
`-DP36_TYPE_ONLY=<id>` so the switch folds at compile time (the pattern already
used for `dense_q4k_decode` / `dense_q5k_decode`). Six new `.spv`
(`moe_matvec{,_fast}_{q3k,q4k,q5k}`), plus `p36_vk_moe_type_kernel()` in
`src/vulkan/vk_13_moe_build_tiles.inc` selecting them at all three dispatch sites.

**Gate check that matters:** `P36_VK_MOE_FAST` defaults ON
(`src/vulkan/vk_02_use_attn_qtile2.inc:132`), so the first version — which only
specialized `moe_matvec` — was **dead code by construction** (§4 rule 5). The
arm was rebuilt to specialize `moe_matvec_fast` as well before any verdict.

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| generic (P36_VK_MOE_TYPE=0) | 741.18, 710.00, 736.36 | 71.69, 71.48, 72.09 |
| typed | 694.10, 682.87, — | 71.38, 72.19, — |

Decode medians 71.8 vs 71.9 — **no change**.

## Candidate 2 — Q4_K packed loads in the fast matvec

`load_u8(off)` is already word-based (`wb[off >> 2]`), so a lane's four
consecutive Q4_K bytes are one aligned word; the four q8_K activations likewise
share one shared-memory word. Rewrote the group loop to 1 global + 1 shared load
per group instead of 4 + 4.

**Correctness: bit-identical.** 64 tokens, greedy, same prompt — md5
`4915110d…` for both the pre-change and post-change binaries
(`git stash` → rebuild → compare → `stash pop`).

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| before | 741.18, 710.00, 736.36 | 71.69, 71.48, 72.09 |
| after | 739.62, 715.82, 679.86 | 71.88, 71.59, 71.80 |

Decode medians 71.8 vs 71.8 — **no change**, as predicted by the I/O data: at
10.3 MiB/token against a 745 MiB/s disk, the ALU work in the matvec is not on the
critical path.

## Disposition

Both candidates reverted; tree clean; baseline reconfirmed at 71.47 t/s.
Do not re-litigate on this host without an I/O control or more RAM — the
measurement cannot see the difference.

## Incidental finding for the compatibility gate

Upstream `q36` (5895cdc) **cannot load either APEX-I file**:
`tensor token_embd.weight has dim[1]=248144, expected 248320`.
p36 accepts it. This is a p36-only capability; no upstream baseline exists.
