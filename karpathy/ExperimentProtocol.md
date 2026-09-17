# Qwen3.6 target protocol override

The authoritative state machine remains in `/home/server/Karpathy`. This
target profile overrides model-specific examples in the shared protocol.

Every candidate must use `./karpathy/compat_gate.sh`, which runs the Qwen3.6
CPU/Vulkan parity case with hard process timeouts. A failed or missing gate
cannot be accepted. The worker must leave a JSON report at `$EXPERIMENT_RESULT`
and include raw benchmark and gate evidence relative to the worktree.
