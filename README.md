> **Endorsed quants:** PreonStar approves and endorses the
> [IsValorum](https://huggingface.co/IsValorum) custom APEX-I quants,
> **APEX-I-MiniPlus** and **APEX-I-NanoPlus**, for their speed and quality.

<p align="center">
  <img src="logo.svg" alt="PreonStar logo" width="220">
</p>

# PreonStar

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)
![GPU: AMD BC-250 / Vulkan](https://img.shields.io/badge/GPU-BC--250%20Vulkan-red.svg)
![Language: C99](https://img.shields.io/badge/language-C99-555.svg)

**A small, self-contained inference engine that runs a 35B-parameter model on
a $150 AMD BC-250 mini-PC** (16 GB unified memory, Vulkan/RADV).

PreonStar is a native C99 engine for these model families:

| Model | What it is |
|---|---|
| **Sage** (e.g. Sage-A1.5) | Finetune of **Qwen3.6-35B-A3B** (MoE, 3B active) |
| **Axion-1.0-35B** | Finetune of **Qwen3.6-35B-A3B** (MoE, 3B active) |
| **Swift-1.5** | Finetune of **Qwen3.8-27B** (dense) |
| **Inkling-2-Mini** | Thinking Machines' own `inkling2_loop` model, *not* Qwen-based |

Sage and Axion are not separate architectures: they run on the Qwen3.6 35B-A3B
MoE code path, so anything that holds for Qwen3.6-35B-A3B holds for them.
It ships the whole stack: a CLI, an OpenAI/Anthropic-compatible HTTP server, a
coding agent, a benchmark harness, and an evaluation runner. No Python in the hot
path, no framework underneath, no build system beyond `make`.

It runs a mixture-of-experts model at interactive speed on a machine with 16 GB of
memory, and if the model does not fit, it streams experts from disk and still
answers.

**Speed on one BC-250** (2026-11-24, `p36-bench`, warm 2048-token prefill, greedy):

| Model | Prefill | Decode |
|---|---:|---:|
| Sage-A1.5 (Qwen3.6-35B-A3B finetune), APEX-I-NanoPlus | ~1,000 t/s | 118 t/s |
| Inkling-2-Mini, APEX-I-NanoPlus | ~750 t/s | 84 t/s |

**Why not llama.cpp?** On the same board, the `llama.cpp` Inkling-2 fork decodes
Inkling-2-Mini at 47.1 t/s; PreonStar measured 77.1 t/s under the same `tg128`
conditions (see [Performance](#performance)). PreonStar trades generality for
that: one board, a few model families, kernels tuned for each.

[Quickstart](#quickstart) | [Documentation](docs/README.md) | [Benchmarks](docs/benchmarks.md) | [Status](#status)

## What you get

- **Run a capable model locally.** Qwen3.6-35B-A3B (35B total, ~3B active),
  dense Qwen3.8-27B, or Thinking Machines' Inkling-2-Mini (`inkling2_loop`,
  24 physical / 48 looped logical layers), in 2-5 bit quantizations, resident
  or streamed.
- **Talk to it four ways.** Interactive CLI, HTTP server with OpenAI and Anthropic
  wire formats, a native coding agent, and a batch evaluation harness.
- **Built for the BC-250.** Every kernel is tuned and measured on the AMD BC-250
  (gfx1013, Vulkan/RADV). Other Vulkan 1.1 devices run through generic fallbacks
  but are not tuned or validated.

## Supported runtimes

| Backend | Platform | Status |
|---|---|---|
| **Vulkan** | Linux, Vulkan 1.1+ | Production path. The BC-250 fast path is selected by device ID |
| **CPU** | Any C99 platform | Portable reference and debugging runtime, not the production path |

Windows and macOS are not build targets.

`make` produces six command names: `p36`, `p36-server`, `p36-agent`, `p36-bench`,
`p36-eval`, `p36_test`.

## Requirements

**Universal:**

- Linux, a C99 compiler, GNU `make`.
- 16 GB of memory for the default model resident; 8 GB is workable with SSD
  streaming.
- `glslang-tools` (or `glslc`) to compile the Vulkan shaders.

**Backend-specific:**

- **Vulkan**: a driver supporting shader `float64` and `int64`. Subgroup
  arithmetic, 16-bit storage and native `float16` are probed at runtime and the
  optimized kernels fall back when absent. See [backends.md](docs/backends.md).
- **BC-250**: needs a recent Mesa/RADV, a 512 MB UMA split, and a GPU governor to
  hold clocks. Step-by-step in [docs/hardware/bc250.md](docs/hardware/bc250.md).

## Quickstart

Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y build-essential glslang-tools libvulkan-dev
```

Arch and Fedora equivalents are in [docs/hardware/bc250.md](docs/hardware/bc250.md#1-install-linux-and-radv).

Build and fetch a model:

```sh
git clone https://github.com/Trecto34/PreonStar.git
cd PreonStar

make -j"$(nproc)"        # Vulkan, with runtime capability detection
make vulkan-bc250        # same, plus a BC-250 device check

./download_model.sh q2-imatrix    # Qwen3.6-35B-A3B, ~10 GB
```

Run it:

```sh
./p36 -p "Explain Redis streams in one paragraph."

./p36-server --ctx 32768 --kv-disk-dir ~/.p36/server-kv --kv-disk-space-mb 8192

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
| `--backend` | `vulkan` or `cpu` |
| `--kv-disk-dir DIR` | Persist the KV cache so restarts skip prefill |

Platform-agnostic options behave the same everywhere. Hardware-specific flags
(prefill chunk width, cache sizing, CU occupancy hints) are documented next to the
backend they affect, not in the general flag list.

## Performance

**Headline, 2026-11-24** (`p36-bench`, 2048-token prefill chunks, 128 generated
tokens, greedy, governor active). Figures are from the 2048-token prefill that
takes the context from 2048 to 4096; the first frontier of a fresh process runs
while MoE expert pages are still faulting in from SSD and reads low (Sage: ~450-550 t/s).

| Model (APEX-I quant) | Base | Prefill (warm, 2048 tok) | Decode tg128 @ 4k |
|---|---|---:|---:|
| Sage-A1.5 NanoPlus-V2.1 | Qwen3.6-35B-A3B finetune | **985-1105 t/s** | **118.2 t/s** |
| Inkling-2-Mini NanoPlus-V2.1 | Inkling-2-Mini | **739-753 t/s** | **84.0 t/s** |
| Inkling-2-Mini MiniPlus-V2.1 | Inkling-2-Mini | **741-753 t/s** | **83.4 t/s** |

Older recorded results follow.

All numbers are from one BC-250 (16 GB UMA, RADV, GPU governor running), greedy
decoding, Q8_0 K / Q4_0 V cache unless stated. They need
`systemctl is-active cyan-skillfish-governor-smu` to print `active`: with the
governor stopped the same build reads 20-28% slower.

**Axion-1.0-35B (Qwen3.6-35B-A3B finetune), ctx 1024, gen 64:** decode **111.78 t/s**, up from
~103.9 at the start of the 2026-11-20 decode-kernel sprint; every step was
bit-identical (ROADMAP.md, kernel ledger in [karpathy/AlreadyTried.md](karpathy/AlreadyTried.md)).

### Inkling-2-Mini (`inkling2_loop`)

PreonStar supports Thinking Machines' `inkling2_loop` models (both APEX-I and
upstream `llama.cpp` PR #32695 loop-controller tensor layouts, including `Q5_0`
and `IQ4_NL` MoE down projections) with dedicated Vulkan compute shaders
(`vulkan/inkling2_*.comp`). Measured on BC-250 (16 GB UMA, RADV, greedy,
`p36-bench`, `n=128`, default `prefill_chunk=1024`):

| Model / Engine | GGUF Size | `pp1024` (warm) | `tg128 @ 512` | `tg128 @ 1k` | `tg128 @ 2k` | `6.5K` Needle *(pp / tg)* | Safe Max Context *(14 GiB RAM)* |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama-inkling2` `UD-IQ3_XXS` *(ref)* | — | 421.5 t/s | 47.1 t/s | — | — | — | — |
| **PreonStar NanoPlus-V2.1** | 10.23 GiB | **655.4 t/s** | **77.1 t/s** | **77.1 t/s** | **76.9 t/s** | — | **~64K–96K** *(32 KiB/tok)* |
| **PreonStar MiniPlus-V2.1** | 12.09 GiB | **634.3 t/s** | **75.4 t/s** | **75.4 t/s** | **75.1 t/s** | **582.6 / 66.5 t/s** | **~16K–32K** *(16K w/ LifeOS, 32K bare)* |

40 of the 48 logical layers use fixed-size sliding-window attention (window
1024, 160 MiB total), so only the 8 full-attention layers grow with context
(`32 KiB/tok` in F16, plus ~0.35 GiB fixed runtime buffers). Exact needle
retrieval (`COBALT-7749`) was verified at 6,495 tokens (`--ctx 8192`, 6× SWA
ring-buffer wrap, 0.22 GiB context buffers).

**Recommended models:** [IsValorum](https://huggingface.co/IsValorum) on Hugging
Face. PreonStar is built to run these quants well: the APEX-I NanoPlus files
(Axion 35B-A3B, Sage, the dense Qwen3.8-27B and Inkling) load as-is and land on
the engine's tuned Vulkan kernels, with no conversion step. If you are choosing weights for a 16 GB board, start there.

Later Inkling commits (not re-measured into the table above): MoE expert
re-layout at upload, +4.7% MiniPlus decode, bit-identical (5c08a39); long-context
attention prefill +4%, MiniPlus 16k->32k prefill 516.9 -> 542.0 t/s,
bit-identical (18648e8).

### 16 GB UMA: what actually fits

- Leave `--prefill-chunk` at its default with `--kv-disk-dir DIR`: on MoE it
  resolves to 1024, which fills the 32-slot expert tiles and prefills **~18%
  faster than 512** (Axion: 759 -> 896 t/s at 16k in `p36-bench`; a cold
  20.4k-token server prompt 709 -> 811 t/s end to end) for ~+125 MiB scratch.
  Measured 2026-11-17 to 32k (`p36-bench`) and 20.4k (`p36-server` with
  `--kv-disk-dir`) with no failure; the RADV 64 MB scratch-buffer failure
  earlier builds hit at 1024 no longer reproduces. Builds before the
  2026-11-17 scratch-pool fix decoded ~23% slower after a multi-chunk prefill
  at 1024 (Axion 76 vs 99 t/s); rebuild before switching a server to 1024.
  Dense (GQA 6) models stay
  clamped to 256 on this board: 512 measured +3% prefill for +93 MiB scratch.
- **`--kv-disk-dir` is free and worth it.** Decode is unchanged and a repeated
  prompt skips prefill (ttft 15.88 s -> 1.98 s at 10.8k, earlier build).
- Rebuild **every** binary after a route change (`make -j16`, not just `make
  p36`): a stale `p36-server` silently keeps the old route and reads ~30% slow.

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
- [Backends](docs/backends.md) -- Vulkan and CPU requirements.
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
