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
- Verified `make vulkan-generic -j2`, shell syntax, IQ4_XS Vulkan inspection, and
  a bounded single-token generation smoke test returning `READY` without a
  SIGKILL. The IQ4_XS file is present at `gguf/Swift-Qwen3.8-27B-IQ4_XS.gguf`.
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
