# Speedup Rules

This file is a rulebook for implementing new Quarkstar speedups and kernel optimizations to improve prefill and decode tokens per second (t/s) for a target runtime (Vulkan, Metal) and a target model.

* A speedup must remain effective as the context size grows. Benchmark `q36_bench.c` with multiple context sizes, such as 128, 2K, and 4K tokens. In some cases, also test 8K and 16K.

* After a change brings a speedup, always verify that it is mathematically equivalent and produces logits consistent with the previous implementation. A good check is the parity test in `q36_test.c`:

  * CPU vs. llama: use 16 threads and short prompts only. This is required only if the `q36.c` CPU path was touched by the changes.
  * CPU vs. GPU (Vulkan or Metal): use short prompts and 16 CPU threads.
  * All parity thresholds from top-1 through top-64 must pass.

* RAM efficiency is very important for this tool. Even an additional 100 MB can be a problem. As a general rule, a speedup should not increase RAM usage. An exception can be made for a very good trade-off. For example, a speculative decoding layer that provides a 2x decode speedup at the cost of 500 MB of additional RAM can be a good fit.

* Check temperatures during benchmarks, since temperature can affect performance. Optimizations that reduce temperature or power usage while maintaining or improving performance are also valuable.

* **On a thermally constrained board, drift can be large enough to fabricate a result, so A/B by interleaving.** Small-form-factor and passively or marginally cooled boards lose throughput as they heat, and the loss is monotonic across consecutive runs rather than random noise, so median-of-N does not remove it. A sequential "measure A, then measure B" comparison attributes that drift to whichever variant ran second. Instead:

  * Interleave the variants (A, B, A, B, ...) so drift loads equally on both.
  * Gate every run on an entry temperature you have confirmed is *reachable* on the machine in front of you — read `hwmon*/temp1_input` and check the current idle floor first. A gate set below the floor silently times out and still yields mismatched temperatures.
  * Record the entry temperature alongside each result and discard any pair differing by more than a couple of degrees.
  * Prefer swapping the `.spv` file over rebuilding, so both arms execute on a byte-identical binary and no compiler or link difference leaks into the comparison.

  How large this effect is depends entirely on the individual machine's cooling, ambient temperature and power limits; measure it on yours before trusting any sequential comparison. It is not a small correction — on one marginally cooled board, consecutive runs of an unmodified binary fell 13% while the idle floor rose about 16 °C over a session.

* **Do not add per-kernel savings together to predict end-to-end gain.** The profiler's `gpu_ms` values sum to more than wall-clock because dispatches overlap, so part of any saved kernel time was already hidden behind other work. Confirm every change with an end-to-end interleaved A/B against an unmodified binary rather than with the sum of its kernel rows.

* **Watch for prefill/decode pooling in the profiler.** A kernel used in both phases reports a single row averaging the two, and its averaged `groups/dispatch` is meaningless for either phase — a kernel can show tens of workgroups on average while dispatching exactly one during decode. Give shape-tagged op names to any kernel whose time you are attributing before trusting its row.

Never launch two heavy inference processes at the same time. Do not run benchmarks in parallel. Models are often larger than 11 GB, while test machines may have only 16 GB of RAM.
