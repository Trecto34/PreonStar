# Qwen3.6-35B-A3B optimization history

This is the target-specific ledger for `/home/server/q36-opt`.

- Baseline: commit `1a57ede`, Qwen3.6-35B-A3B IQ2XXS on AMD BC-250.
- Existing accepted Qwen3.6 work includes the qtile2 prefill Q/V addressing
  change and earlier MoE/matmul addressing, unrolling, and vector-load changes
  already recorded in `/home/server/Karpathy/AlreadyTried.md`.
- Existing rejected work includes the wide f32 router path and unvalidated
  attention-span variants. Do not retry them without changing the failure
  mechanism and proving routing/parity safety.
- Before every source edit, compare the proposed mechanism with both this file
  and `/home/server/Karpathy/AlreadyTried.md`. The shared harness injects both
  histories into the worker request and requires `LEDGER_CHECK:`.

Do not repeat a rejected hypothesis merely because it has a different tile
size or spelling. Record every completed or interrupted target attempt here.
