# Changelog

## 2026-09-17

- Added Swift Qwen3.8-27B download support for `ukisai/Swift-Qwen3.8-27B-GGUF`.
- Added `run_swift.sh` with BC-250 Vulkan fast-path settings, embedded MTP,
  and automatic model-sensitive context/cache defaults.
- Added `run_swift_iq4_xs.sh` for the IQ4_XS model.
- Added a bounded dense-weight cache controlled by `Q36_VK_DENSE_WEIGHT_CACHE_GIB`.
  Cache eviction drains all Vulkan work first, preventing use-after-free while
  allowing the 14.6 GiB IQ4_XS model to run on the 16 GiB BC-250 memory budget.
- IQ4_XS launcher defaults: prewarming disabled, 2 GiB dense-weight working set,
  1024-token context, and the same optimized Vulkan kernels as the Swift launcher.
- Updated the README with Swift download and launch instructions.
- Verified `make vulkan-generic -j2`, shell syntax, IQ4_XS Vulkan inspection, and
  a bounded single-token generation smoke test returning `READY` without a
  SIGKILL. The IQ4_XS file is present at `gguf/Swift-Qwen3.8-27B-IQ4_XS.gguf`.
