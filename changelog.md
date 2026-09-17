# Changelog

## 2026-09-17

- Added a Qwen3.6-specific Karpathy target profile while keeping authoritative
  state and loop logic in `/home/server/Karpathy`.
- Added a serial Qwen3.6 CPU/Vulkan parity gate with process cleanup and hard
  timeouts.
- Added the direct DeepSeek V4.1 Flash OpenCode provider, protected-key-file
  loading, finite worker budget, mandatory previous-attempt check, and
  skip-GPU-on-failed-agent behavior to this target.
- Isolated Qwen3.6 cadence state in `state-q36-opt.json` so it cannot trigger or
  consume the Swift target's meta cadence.
- Validated the new Qwen3.6 gate on the BC-250: CPU/Vulkan
  `short_reasoning_plain` parity passed with matching top-1/top-k results.
- Added `--verbose`/`KARPATHY_VERBOSE=1` to both loop entrypoints, streaming the
  inner worker live while preserving its worker log.
