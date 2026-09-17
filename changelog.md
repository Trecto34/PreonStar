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
- Reduced the shared meta-agent provider watchdog to 180 seconds per tier,
  added fallback-stage output, and exposed `KARPATHY_META_TIMEOUT` for faster
  failure when the provider is unavailable.
- Added `KARPATHY_SKIP_META=1` to bypass one scheduled meta audit and start the
  inner pass immediately when persistent cadence state lands on that audit.
- Added a project-local direct DeepSeek provider in `opencode.json`, using the
  current `deepseek-flash` API model (DeepSeek V4.1 Flash) at DeepSeek's
  official OpenAI-compatible base URL, and
  `DEEPSEEK_API_KEY`/`DEEPSEEK_API_KEY_FILE` without storing credentials.
- Added `KARPATHY_PROVIDER=deepseek` support to both shared Karpathy loops.
  Inner and outer runs use `deepseek-direct/deepseek-flash`, target the local
  OpenCode config explicitly, and stop on provider failure unless fallback is
  deliberately enabled outside this default mode.
- Added an outer-loop credential preflight so a missing DeepSeek key exits
  before any compatibility/GPU process is launched.
- Fixed the outer Python entrypoint to propagate preflight and loop failure
  statuses, and aligned the shared fallback regression fixture with the
  mandatory target compatibility-gate contract.
- The DeepSeek launcher now automatically loads the owner-only key file at
  `${HOME}/.config/deepseek/api-key`, so normal runs require no per-session
  export; explicit environment variables still override it.
- The target wrappers now select DeepSeek by default, so the persisted-key
  setup runs with `./karpathy/run_outer.sh` directly; `KARPATHY_PROVIDER=union`
  remains an explicit legacy override.
- Tightened the inner DeepSeek worker: the harness injects target/shared
  previous-attempt snapshots and requires a `LEDGER_CHECK` before source work;
  the direct model is capped at 32K context and 4096 output tokens, with a
  600-second default watchdog to stop open-ended exploratory passes.
- Failed or incomplete worker turns now skip the expensive compatibility gate;
  only a successful turn with a `LEDGER_CHECK` marker can consume GPU time on
  validation.
- Added Linux parent-death signals to the orchestrator and experiment worker;
  abrupt supervisor termination can no longer leave an orphaned DeepSeek agent
  running and consuming tokens.
- Added `--verbose`/`KARPATHY_VERBOSE=1` to both target loops; inner worker
  output now streams live to the terminal while remaining saved to its worker
  log.
- Closed and recorded the interrupted DeepSeek experiment
  `20260917T120056-9e9fa702`: its lane-split `dense_iq3_xxs_mmq` candidate was
  bit-exact but regressed down projection by about 2.7%, and the incomplete
  worker was terminated before its report. The rejected mechanism is now in
  the target `karpathy/AlreadyTried.md` ledger.

## 2026-09-17 — vllm-mxfp4 transfer audit on BC-250

- Created branch `experiment/radiance-transfer-bc250` in the clean Swift target;
  left the separate, dirty `/home/server/q36-opt` checkout untouched. Added
  `tests/benchmark_transfer.sh` to benchmark Swift IQ3_XXS and Qwen3.6 IQ2_XXS
  serially at context 1024, 16 generated tokens, safe per-model prefill chunks,
  and three repetitions by default. It kills an active benchmark on interrupt.
- Profiled both models and tested four shader ideas, reverting all source
  candidates that did not improve end-to-end throughput: 512-thread fused
  add/RMS (Swift kernel 23.64 -> 30.13 ms); integer sign-mask IQ3_XXS decode
  (Swift decode kernel 373.83 -> 407.97 ms); XOR-swizzled IQ3_XXS MMQ staging
  (Swift MMQ kernel 940.28 -> 1309.31 ms). Direct global-table reads in the
  Qwen3.6 MoE gate/up GEMM changed its 128-token kernel time 197.71 -> 194.26
  ms, but three full-context candidate runs were 932.91/913.41/953.56 prefill
  tok/s, not a reliable gain over the 951.38 tok/s single baseline point; it
  too was reverted. The model kernels remain unchanged.
- Safe prefill-chunk sweep at context 1024: Swift 64/128/256 yielded
  96.37/164.73/171.91 tok/s; Qwen3.6 256/512/1024 yielded
  720.58/858.86/951.38 tok/s. Existing defaults (256 and 1024) were best
  among tested safe options.
- Final unchanged-kernel three-run means from the new script: Swift 172.41
  prefill and 22.53 decode tok/s; Qwen3.6 923.48 prefill and 88.01 decode
  tok/s. **Accepted model-speed gain: 0% on both.** These are honest baseline
  measurements, not a claimed vLLM-MXFP4 port: neither model uses MXFP4 and
  the candidate kernel transfers failed the BC-250 speed gate. The existing
  Qwen3.6 CPU/Vulkan parity and Swift Vulkan smoke compatibility gate passed.

### Follow-up after throughput feedback

- Tried wider and narrower IQ3_XXS prefill tiles with matching Vulkan
  dispatch changes. At context 1024, the 256-token tile yielded 166.87
  prefill tok/s and the 64-token tile 161.07 tok/s, both below the 172.41
  tok/s three-run baseline for the 128-token tile. Both were reverted.
- Tested Swift's in-file MTP draft depth 2 over 128 generated tokens: 23.28
  tok/s versus 24.14 tok/s with the existing depth 1. No default change.
- Removing forced unrolling in the IQ3_XXS MMQ shader increased 128-token
  kernel time from roughly 940 to 960 ms; reverted. An eight-row IQ3_XXS
  decode variant matched the existing 24.14 tok/s generation rate over 128
  tokens, so it was also reverted. Restored the original shaders and build.
- No runtime speedup was accepted. The next meaningful step would require a
  larger change to the dominant IQ3_XXS prefill or Qwen3.6 MoE data path,
  with correctness and thermal-controlled end-to-end validation.
