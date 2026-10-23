# Testing and QA

The test suite is split by what it needs. Knowing which tier a test belongs to
tells you whether a failure means "the code is broken" or "this machine cannot
run that test".

## Tier 1 -- no model, no GPU

```sh
make check
```

Builds every binary and runs the test modes that need neither a GGUF model nor a
Vulkan device: quantization primitives, vector fixtures, the SSD cache shrink
test, the Qwen tool-call format, and the server request/response logic.

This is what CI runs on a hosted runner, and it is the fastest signal that a
change broke something. A red run here is always a real defect.

## Tier 2 -- model required

```sh
make test          # unit, CLI, agent and eval suites against a local model
make test-vulkan   # Vulkan kernel and parity coverage
make test-metal    # Metal kernel and parity coverage (macOS)
```

These need a downloaded model in `gguf/` and, for the GPU tiers, a working device.
Tests that cannot find a model report `SKIP (model not found)` rather than
silently passing.

Parity targets compare backends token-for-token on a short prompt:

```sh
make test-qwen27b-vulkan-parity
make test-qwen27b-metal-parity
```

## Tier 3 -- release

```sh
make test-all
```

plus the full matrix in [QA_BEFORE_RELEASES.md](../QA_BEFORE_RELEASES.md). Release
QA also covers throughput, so it needs the benchmark protocol in
[benchmarks.md](benchmarks.md) rather than a pass/fail test.

## Known harness limitation

A handful of GPU oracles (`vulkan-kernels`, the `pq2` and `ptq1` layout oracles,
`q8k-activation-parity`) do not probe for a Vulkan device. On a machine without a
GPU they report FAIL rather than SKIP, and the failure count varies between runs
because they validate host buffers that were never allocated. The other model-free
modes skip correctly. Treat those five as SKIP on GPU-less machines; they are
pre-existing behaviour, not a signal about your change.

## Continuous integration

[.github/workflows/ci.yml](../.github/workflows/ci.yml) installs the build
dependencies, runs `make -j$(nproc)` and then `make check`. It cannot test
throughput or the GPU paths -- a hosted runner has neither.
