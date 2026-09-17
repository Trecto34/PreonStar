# Karpathy loops for Qwen3.6

The inner and outer entrypoints delegate state, experiment branching, and
process cleanup to `/home/server/Karpathy`. This target profile supplies the
Qwen3.6 model metadata, target ledger, and independent CPU/Vulkan parity gate.

```sh
./karpathy/run_inner.sh
./karpathy/run_outer.sh
./karpathy/compat_gate.sh
```

DeepSeek V4.1 Flash is the default agent provider and uses the protected key
file at `${HOME}/.config/deepseek/api-key`. The worker has a 600-second default
watchdog, a 32K context/4096-output model limit, and must emit `LEDGER_CHECK:`
before the harness will spend GPU time on validation. Use
`KARPATHY_INNER_TIMEOUT` only for a deliberately longer run.
The Qwen3.6 campaign keeps cadence state in
`${HOME}/Karpathy/state-q36-opt.json`, separate from the Swift campaign.
