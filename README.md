<p align="center">
  <img src="logo.svg" alt="PreonStar logo" width="220">
</p>

# PreonStar

**A small, self-contained inference engine that runs a 35B-parameter model on
hardware you already own** -- a laptop, a workstation, or a $150 mini-PC.

PreonStar is a native C99 engine for the Qwen3.6-35B-A3B and Qwen3.8-27B families.
It ships the whole stack: a CLI, an OpenAI/Anthropic-compatible HTTP server, a
coding agent, a benchmark harness, and an evaluation runner. No Python in the hot
path, no framework underneath, no build system beyond `make`.

It runs a mixture-of-experts model at interactive speed on a machine with 16 GB of
memory, and if the model does not fit, it streams experts from disk and still
answers.

[Quickstart](#quickstart) | [Documentation](docs/README.md) | [Benchmarks](docs/benchmarks.md) | [Status](#status)

## What you get

- **Run a capable model locally.** Qwen3.6-35B-A3B (35B total, `3B active) or the
  dense Qwen3.8-27B, in 2-4 bit quantizations, resident or streamed.
- **Talk to it four ways.** Interactive CLI, HTTP server with OpenAI and Anthropic
  wire formats, a native coding agent, and a batch evaluation harness.
- **Bring your own hardware.** Anything with a working Vulkan 1.1 driver runs.
  Apple Silicon runs natively through Metal. The AMD BC-250 is the board the fast
  path was tuned on -- a reference configuration, not a requirement.

## Supported runtimes

| Backend | Platform | Status |
|---|---|---|
| **Vulkan** | Linux, Vulkan 1.1+ | Primary GPU path. Generic; the BC-250 fast path is selected by device ID |
| **Metal** | macOS 11+, Apple Silicon M1 or newer | Native graph runtime, same CLI and server |
| **CPU** | Any C99 platform | Portable reference and debugging runtime, not the production path |

Windows is not a build target.

Both GPU backends are chosen **at build time** (`make` for Vulkan, `make metal` for
Metal) and produce the same six command names: `p36`, `p36-server`, `p36-agent`,
`p36-bench`, `p36-eval`, `p36_test`.

## Requirements

**Universal:**

- Linux or macOS, a C99 compiler, GNU `make`.
- 16 GB of memory for the default model resident; 8 GB is workable with SSD
  streaming.
- `glslang-tools` (or `glslc`) to compile the Vulkan shaders.

**Backend-specific:**

- **Vulkan**: a driver supporting shader `float64` and `int64`. Subgroup
  arithmetic, 16-bit storage and native `float16` are probed at runtime and the
  optimized kernels fall back when absent. See [backends.md](docs/backends.md).
- **Metal**: Xcode command line tools. No Vulkan loader needed.
- **BC-250**: needs a recent Mesa/RADV, a 512 MB UMA split, and a GPU governor to
  hold clocks. Step-by-step in [docs/hardware/bc250.md](docs/hardware/bc250.md).

## Quickstart

Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y build-essential glslang-tools libvulkan-dev
```

Arch and Fedora equivalents are in [docs/hardware/bc250.md](docs/hardware/bc250.md#1-install-linux-and-radv);
on macOS install the Xcode command line tools instead.

Build and fetch a model:

```sh
git clone https://github.com/Trecto34/PreonStar.git
cd PreonStar

make -j"$(nproc)"        # Vulkan, with runtime capability detection
make metal               # Apple Silicon instead

./download_model.sh q2-imatrix    # Qwen3.6-35B-A3B, `10 GB
```

Run it:

```sh
./p36 -p "Explain Redis streams in one paragraph."

./p36-server --ctx 32768 --kv-disk-dir `/.p36/server-kv --kv-disk-space-mb 8192

./p36-agent
```

Verify the build without a model or a GPU:

```sh
make check
```

## Configuration

A few flags cover most use; the rest are in [configuration.md](docs/configuration.md).

| Flag | Use |
|---|---|
| `-m, --model FILE` | Model to load. Defaults to the q2 imatrix download |
| `--ctx N` | Context window in tokens |
| `--ssd-streaming` | Stream experts from disk instead of full residency |
| `-ctk` / `-ctv` | KV cache types: `f16`, `q8_0`, `q4_0` (GPU default q8_0 / q4_0) |
| `--backend` | `vulkan`, `metal` or `cpu` |
| `--kv-disk-dir DIR` | Persist the KV cache so restarts skip prefill |

Platform-agnostic options behave the same everywhere. Hardware-specific flags
(prefill chunk width, cache sizing, CU occupancy hints) are documented next to the
backend they affect, not in the general flag list.

## Performance

Prefill and decode at 2048-32768 tokens of context, on the BC-250 reference board
with the default q2 model, greedy decoding, Q8_0 K / Q4_0 V cache:

| Context | Prefill t/s | Decode t/s |
|---:|---:|---:|
| 2048 | 640 | 82 |
| 8192 | 502 | 75 |
| 32768 | 244 | 51 |

These are reference points from one board, not a promise. **How to measure your
own machine, the exact harness invocation, the A/B protocol used to accept a
performance change, and the fields every result row must record are in
[benchmarks.md](docs/benchmarks.md).**

## Project layout

```text
p36.c            manifest -> src/core/     GGUF, kernels, forward pass, sampling
p36_server.c     manifest -> src/server/   HTTP, SSE, API parsing, tool schemas
p36_agent.c      manifest -> src/agent/    agent loop, tools, workspace jail
p36_vulkan.c     manifest -> src/vulkan/   device, pipelines, dispatch
p36_cli.c  p36_web.c  p36_eval.c  p36_kvstore.c  p36_ssd.c   support
src/*/*.inc      71 fragments, included in order into one translation unit
tests/           unit, kernel, server and parity tests
speed-bench/     CSV results and the plotting script
karpathy/        measurement campaign, evidence and the rejection ledger
```

The fragments are a single translation unit deliberately: the build has no LTO, so
separate `.c` files would force exports and lose inlining on the hot path. Why the
code is shaped this way, and what each module owns, is in
[architecture.md](docs/architecture.md).

## Documentation

- [docs/README.md](docs/README.md) -- index of everything below.
- [Configuration](docs/configuration.md) -- all flags, KV cache, migration from q36.
- [Server](docs/server.md) / [Agent](docs/agent.md) -- the two long-running entry points.
- [Backends](docs/backends.md) -- Vulkan, Metal and CPU capability requirements.
- [Benchmarks](docs/benchmarks.md) -- measure it yourself, and how to report it.
- [Streaming](docs/streaming.md) -- running models larger than memory.
- [Models](docs/models.md) -- supported families, quants, vision.
- [Troubleshooting](docs/troubleshooting.md) -- steering, debug switches, failure modes.
- [CONTRIBUTING.md](CONTRIBUTING.md) -- **read before sending a pull request.**
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) -- the release test matrix.
- [MODEL_CARD.md](MODEL_CARD.md) -- model details and limitations.
- [changelog.md](changelog.md) -- what changed and when.

## Status

Alpha. The software changes quickly and instabilities are possible. A full QA run
precedes each release; see [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md).

## Lineage and acknowledgements

PreonStar started as a fork of [**q36**](https://github.com/Ninnix/q36) (branded
QuarkStar) by Ninnix. Neither project would exist without **DwarfStar**,
**llama.cpp** and **GGML**. Thank you to Ninnix, Salvatore Sanfilippo (Antirez),
Georgi Gerganov, and every contributor to those projects.

Model support is intentionally opportunistic: the project follows the best open
weights for 16 GB machines and 24/32 GB workstations, and a model may be dropped
when a better replacement arrives.

## AI disclosure

Parts of this codebase and its documentation were written with AI assistance. All
of it is reviewed, built, tested and benchmarked by a human before release. Claims
in this repository are backed by code you can read and numbers you can reproduce;
where a claim is unverified, it says so.

## License

MIT. See [LICENSE](LICENSE).
