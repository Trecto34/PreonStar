# Karpathy Task — Swift Qwen3.8-27B IQ3_XXS

Optimize this checkout's Vulkan runtime and shaders for:

- **Primary:** `ukisai/Swift-Qwen3.8-27B-GGUF`, IQ3_XXS, on AMD BC-250/RADV.
- **Compatibility:** `/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`,
  the local Qwen3.5/Qwen3.6 MoE (`qwen35moe`) reference.

The primary model is the practical Swift tier on this machine. IQ4_XS was
measured as capacity/I/O limited and is intentionally not part of the default
loop.

## One-iteration objective

Find one measurable, source-controlled Vulkan throughput improvement. Build it,
run a focused measurement, then run the complete compatibility gate before
accepting it. Keep candidate changes in the isolated worktree; the experiment
manager snapshots them.

## Required acceptance evidence

An accepted report must include a positive, repeatable end-to-end Swift
measurement with no regression at the compared context frontier, the Swift
IQ3_XXS smoke benchmark, Qwen3.5/Qwen3.6 MoE CPU/Vulkan short-case parity, no
GPU crash/timeout/OOM/thermal violation, and the compatibility log at
`karpathy/evidence/compatibility-gate.log`.

## Guardrails

- Never use the 32768-token CLI default on this machine during an iteration.
- Never run two Vulkan model processes concurrently.
- Do not change fp32 accumulation or MoE router reductions without parity
  evidence; routing drift is a correctness failure even when tokens/s rises.
- Reject a candidate if its gain is below the experiment protocol threshold,
  is not repeatable, or only appears at one unvalidated attention span.
- Do not delete model assets or rewrite shared Karpathy records from a worker.
