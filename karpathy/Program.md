# Qwen3.6 inner-worker profile

Perform exactly one hypothesis-driven optimization pass in the isolated
experiment worktree. Read the target `AlreadyTried.md`, shared
`/home/server/Karpathy/AlreadyTried.md`, `Task.md`, and the experiment protocol
before editing. Print `LEDGER_CHECK:` naming the nearest prior result or
explaining why the mechanism is genuinely new.

Use `make vulkan-generic -j2`, then a focused Qwen3.6 microbenchmark, then the
target compatibility gate. Keep CPU/Vulkan logits, routing, tool-call format,
KV behavior, memory limits, and long-context behavior intact. A candidate that
is faster only because it reassociates the MoE router is rejected.

Write the required JSON report to `$EXPERIMENT_RESULT`, print
`DECISION: ACCEPTED` or `DECISION: REJECTED`, and stop. Do not edit shared
Karpathy ledgers or commit from the experiment worktree.
