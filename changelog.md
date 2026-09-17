# Changelog

## 2026-09-17

- Added Swift Qwen3.8-27B download support for `ukisai/Swift-Qwen3.8-27B-GGUF`.
- Added `run_swift.sh` with BC-250 Vulkan fast-path settings, embedded MTP,
  and automatic model-sensitive context/cache defaults.
- Added `run_swift_iq4_xs.sh` for the IQ4_XS model.
- Added a bounded dense-weight cache controlled by `Q36_VK_DENSE_WEIGHT_CACHE_GIB`
  and `Q36_VK_DENSE_WEIGHT_PIN_GIB`. Cache eviction drains all Vulkan work first,
  preventing use-after-free while allowing the 14.6 GiB IQ4_XS model to run on
  the 16 GiB BC-250 memory budget.
- IQ4_XS launcher defaults: prewarming disabled, 7 GiB retained prefix plus a
  reclaimable 2 GiB streaming window. Stream-window Vulkan allocations are
  recycled instead of repeatedly allocated and freed,
  1024-token context, and the same optimized Vulkan kernels as the Swift launcher.
- Updated the README with Swift download and launch instructions.
- Added the `swift-iq3` downloader target and `run_swift_iq3_xxs.sh` launcher
  for the faster IQ3_XXS Swift tier.
- Resumed and completed `Swift-Qwen3.8-27B-IQ3_XXS.gguf` at 11.67 GiB, then
  permanently deleted the unused 14.6 GiB IQ4_XS file to reclaim disk space.
- Verified the IQ3_XXS launcher with Vulkan inspection and a protected text
  generation test: `READY`, 39.92 tok/s prefill, and 21.90 tok/s generation.
- Benchmarked IQ3_XXS with `q36-bench` at 1024 context: 180.27 tok/s prefill
  and 23.01 tok/s generation without MTP; three-position MTP measured 175.21
  and 21.73 tok/s, so the launcher now defaults to one draft position.
- Tested attention span 128 at 1024 and 2048 context; it measured 23.80 and
  19.10 tok/s versus 23.01 and 22.02 tok/s at span 512, so it was rejected as
  an unreliable default optimization.
- Verified `make vulkan-generic -j2`, shell syntax, IQ4_XS Vulkan inspection, and
  a bounded single-token generation smoke test returning `READY` without a
  SIGKILL. The IQ4_XS file was subsequently permanently deleted after the
  faster IQ3_XXS replacement was validated.
- Tested a resident IQ4_XS split-heap experiment, but rejected it: the kernel
  reported 14.6 GiB of active GPU memory and the host OOM killer terminated the
  Codex process. The launcher therefore remains on the tested bounded-cache
  profile, which prevents this recurring crash.
- Enabled parallel host copies for large bounded IQ4_XS weight uploads, using
  the existing worker pool without changing GPU residency or model numerics.
- Rebuilt and retested the bounded profile after this change; the smoke test
  still returned `READY` at about 0.11 tok/s with no SIGKILL. The copy setting
  did not materially change this workload, confirming that dense weight
  turnover—not single-threaded memcpy—is the dominant limit.
- Local reference measurements with the same 1024-token prompt: Qwen3.6
  35B-A3B IQ2XXS reached 53.39 tok/s, Qwen3.8-27B IQ3_S reached 18.93 tok/s,
  and bounded IQ4_XS reached 0.11 tok/s. The first two are the practical fast
  options on this machine; IQ4_XS is capacity/I/O limited by its 14.6 GiB size.

## Karpathy bilevel setup

- Added a versioned Swift target profile under `karpathy/`; its entrypoints now
  delegate inner/outer control, persistent state, and experiment branching to
  the authoritative `~/Karpathy` implementation.
- Added `karpathy/compat_gate.sh`, which serially runs Swift IQ3_XXS smoke and
  Qwen3.5/Qwen3.6 qwen35moe CPU/Vulkan parity with per-process timeouts.
- The inner report finalizer records the gate as evidence and prevents a failed
  compatibility run from being accepted. The outer loop performs preflight and
  postflight gates and defaults to one pass to avoid unattended crash loops.
- The compatibility gate was run successfully on the BC-250: Qwen3.5/Qwen3.6
  `qwen35moe` short CPU/Vulkan parity was `OK`, followed by Swift IQ3_XXS at
  177.84 prefill tok/s and 21.19 generation tok/s, with no lingering GPU
  process.
- Added process-group cleanup for Ctrl-C in the shared Karpathy orchestrator and
  target gate, terminating active model/agent children together instead of
  leaving an orphaned `q36_test` behind.
