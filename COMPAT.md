# Compatibility

Snapshot of what q36 loads and runs, taken from `README.md`, the loader in
`q36.c` (`weights_validate_layout`, `tensor_type_is_routed_*`) and the shader
set in `vulkan/`. Items marked *tested* were run on hardware; everything else is
read from the code and may lag behind it.

q36 is not a general GGUF loader. A file must match the expected tensor layout,
quantization mix and metadata for one of the models below.

## Models

| Model | Status | Notes |
| --- | --- | --- |
| Qwen3.6-35B-A3B (MoE) | Primary target, *tested* | Q2 routed experts (IQ2_XXS gate/up, Q2_K down, Q8_0 rest); mixed Q2 + Q4_K on layers 34-39 |
| Qwen3.8-27B (dense) | Supported, *tested* | `UD-IQ3_S`, Swift-1.5 / GSQ-RCO `IQ3_S` derivatives. IQ4_XS was dropped: too slow and triggered the host OOM killer |
| Ternary-Bonsai-2-27B (PQ2_0) | Supported, *tested* | PQ2_0 / PTQ1_0 have dedicated kernels |
| Ornith-1.5-35B-A3B and finetunes (e.g. CyberTiel) | Supported as a q36 re-quant, *tested* (smoke test, 2k-12k prefill sweep) | Same shape as Qwen3.6. Build it from a Q8_0/BF16 GGUF with `qwen36-quantize` (IQ2_XXS gate/up, Q2_K down, Q8_0 rest; `--strip-nextn` if the source carries the MTP block) and an imatrix, e.g. the one published by Tiel. Output quality was not evaluated. Ready-made Unsloth Dynamic quants (`UD-*`, e.g. Tiel-Coder) are **not** loadable: see Known gaps |
| Other GGUFs | No | |

## Quantization types

"Loader" = accepted by the layout validator. "Kernel" = a Vulkan shader exists
(judged from shader file names; a type may also be covered inside a generic
shader, so confirm in `q36_vulkan.c` before relying on a gap).

| Tensor class | Loader accepts | Kernels found |
| --- | --- | --- |
| Routed experts gate/up | Q4_K, IQ2_XXS, IQ2_S, IQ3_S, Q8_0 | `moe_gate_up` (IQ2_XXS), `_iq2s`, `q4k` |
| Routed experts down | Q2_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_S, Q8_0 | `moe_down` q2k, q4k, `_iq2s`, `_iq3s` |
| Dense / shared-expert matrices | Q8_0, Q2_K-Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, BF16, F16, PQ2_0, PTQ1_0 | Q8_0, Q5_K, Q6_K, IQ1_M, IQ3_S, IQ3_XXS, IQ4_XS, BF16, F16, PTQ1_0 |
| Norms, SSM state, router | F32 | n/a |

Known gaps:

- **IQ2_XS and IQ3_XXS routed experts, IQ3_XXS and IQ4_XS routed-down experts**
  are rejected by the loader and have no MoE kernels. This is what blocks
  Unsloth UD quants (e.g. `Tiel-Coder-35B-A3B-UD-Q2_K_XL`).
- **Mixed-quant mode** only turns on when `n_tensors == expected + 20`
  (`q36.c`, `weights_validate_layout` call). Outside it, `token_embd` and
  `output` must be Q4_K or Q8_0, so a Q5_K embedding fails with
  `tensor token_embd.weight has type q5_k, expected q4_k or q8_0`.

## Runtime features

| Feature | Status |
| --- | --- |
| Vulkan (AMD BC-250, RADV) | Primary, *tested*. Generic Vulkan 1.1+ build exists but is not validated on other GPUs |
| Metal (Apple Silicon M1+) | Supported, independent of the Vulkan build |
| CUDA, Windows | No |
| CPU backend | Reference/debug only, not a production path |
| Expert streaming from SSD | Yes (`--ssd-streaming`) for models larger than RAM |
| MTP (in-file draft head) | Draft depth 1 by default. Depth 2+ rejected; currently a net loss on BC-250 |
| External draft-model speculative decoding (DFlash, EAGLE, ...) | No |
| Vision (mmproj sidecar) | Yes, Qwen3.6 and Qwen3.8 |
| KV cache types | K and V: f16, q8_0, q4_0. Default K q8_0, V q4_0 |

## Hardware reference

AMD BC-250 (gfx1013, 24 CUs stock / 40 unlocked, 16 GB unified memory) is the
tuned device. About 10-14 GB is usable for weights after the OS and KV cache;
a model near 14 GiB resident can trigger the host OOM killer.
