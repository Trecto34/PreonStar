# Qwen3.6 target task

Target checkout: `/home/server/q36-opt`
Model: Qwen3.6-35B-A3B MoE, IQ2XXS routed experts
Hardware: AMD BC-250, RADV GFX1013, 16 GB unified memory
Baseline: commit `1a57ede`, approximately 53 tok/s decode at context 1024.

Find one measurable, correctness-preserving Vulkan optimization per pass.
Prioritize the measured Qwen3.6 hot path and use a focused microbenchmark
before an end-to-end claim. Preserve fp32 MoE router accumulation and native
Qwen tool-call behavior. Never use the 32768-token CLI default for tests.

The independent target gate is mandatory:

```sh
./karpathy/compat_gate.sh
```

It runs Qwen3.6 CPU/Vulkan parity serially and must pass before acceptance.
