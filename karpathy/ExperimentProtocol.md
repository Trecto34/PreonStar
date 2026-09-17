# Local experiment protocol

The canonical branch/worktree and report lifecycle is implemented by:

```sh
python3 /home/server/Karpathy/experiments.py
```

The local inner wrapper supplies `--target` as this checkout and stores records
under `.karpathy/experiments`, so the unrelated `/home/server/q36-opt` campaign
is not touched. Each trial gets a clean worktree, a report at
`$EXPERIMENT_RESULT`, and a snapshot commit. Accepted trials are fast-forwarded
only when the target is still unchanged.

For an accepted report the external manager requires positive measurements and
these gates: parity, routing, chunk invariance, memory, stability, thermal, and
baseline reverified. The local finalizer additionally requires both model legs
of `compat_gate.sh` to pass.
