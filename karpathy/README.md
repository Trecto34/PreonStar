# Karpathy loops for the Swift Qwen3.8 target

This is the repo-local bilevel autoresearch setup for the BC-250 checkout.

- **Inner loop:** one isolated experiment worktree and one candidate change.
- **Outer loop:** serialized preflight, inner-loop execution, postflight, and
  periodic process audits.
- **Compatibility gate:** every candidate checks Swift IQ3_XXS and the local
  Qwen3.5/Qwen3.6 MoE reference (`qwen35moe`) in separate GPU processes.

The experiment branch manager is shared with the existing Karpathy install at
`/home/server/Karpathy/experiments.py`, but the target, task files, evidence,
and experiment store are local to this checkout.

## Commands

```sh
# One isolated inner experiment (safe default)
./karpathy/run_inner.sh

# One complete outer pass; increase explicitly for a campaign
KARPATHY_MAX_ITERATIONS=1 ./karpathy/run_outer.sh

# Compatibility-only check
./karpathy/compat_gate.sh
```

The loops never start concurrent `q36`, `q36-bench`, or `q36_test` jobs. The
default context and generation lengths are deliberately short enough for this
16 GB UMA machine. Override model paths with `KARPATHY_SWIFT_MODEL` and
`KARPATHY_QWEN35_MODEL` when testing another local quant.
