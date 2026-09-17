# Bilevel loop protocol

## Inner loop

The inner worker gets one isolated worktree and one hypothesis. It must read
`Task.md`, `AlreadyTried.md`, and `ExperimentProtocol.md`, inspect the current
diff, edit only the candidate worktree, and build with `make vulkan-generic
-j2`. Measure the narrow kernel first when possible. Then run:

```sh
./karpathy/compat_gate.sh
```

The worker writes its JSON report to `$EXPERIMENT_RESULT`. It must not commit,
revert, or edit the shared ledger. The harness snapshots the worktree.

## Outer loop

The outer loop owns campaign cadence and process safety. For every pass it runs
a compatibility preflight, starts exactly one inner experiment through the
serialized experiment store, runs a compatibility postflight, and periodically
asks the outer agent to update local task guidance.

The outer loop stops on a failed preflight/postflight and uses a finite one-pass
default. Long campaigns require an explicit `KARPATHY_MAX_ITERATIONS` value.

## Safety model

All model invocations are bounded by `timeout --kill-after`. GPU jobs are
serialized by both the experiment lock and the compatibility gate's executable
checks. A failed or timed-out gate never promotes a candidate. This prevents
the recurring SIGKILL cascade seen with oversized contexts and overlapping
Vulkan processes.
