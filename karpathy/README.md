# Karpathy loops for the Swift Qwen3.8 target

This is the repo-local bilevel autoresearch setup for the BC-250 checkout.

- **Inner loop:** one isolated experiment worktree and one candidate change.
- **Outer loop:** serialized preflight, inner-loop execution, postflight, and
  periodic process audits.
- **Compatibility gate:** every candidate checks Swift IQ3_XXS and the local
  Qwen3.5/Qwen3.6 MoE reference (`qwen35moe`) in separate GPU processes.

The authoritative loop logic, persistent state machine, and experiment branch
manager remain in `~/Karpathy`. The scripts here are target wrappers plus the
Swift/Qwen compatibility gate and target guidance.

## Commands

```sh
# One isolated inner experiment (shared Karpathy logic)
./karpathy/run_inner.sh

# Infinite outer campaign, owned by ~/Karpathy (Ctrl-C stops its process group)
./karpathy/run_outer.sh

# Compatibility-only check
./karpathy/compat_gate.sh
```

The shared orchestrator never starts concurrent `q36`, `q36-bench`, or
`q36_test` jobs. The default context and generation lengths are deliberately
short enough for this 16 GB UMA machine. Override target/model paths with
`KARPATHY_TARGET_DIR`, `KARPATHY_SWIFT_MODEL`, and `KARPATHY_QWEN35_MODEL`.
Ctrl-C is handled by the shared orchestrator, which terminates the active gate,
agent, and child process group before returning.
