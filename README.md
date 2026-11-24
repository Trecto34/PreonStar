<p align="center">
  <img src="logo.svg" alt="PreonStar logo" width="220">
</p>

# PreonStar

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)
![GPU: AMD BC-250 / Vulkan](https://img.shields.io/badge/GPU-BC--250%20Vulkan-red.svg)
![Language: C99](https://img.shields.io/badge/language-C99-555.svg)

PreonStar is an inference engine in plain C99 that runs a 35B-parameter
mixture-of-experts model on a $150 AMD BC-250 mini-PC (16 GB unified memory,
Vulkan/RADV). When a model does not fit in memory, it streams experts from disk.

It runs these model families:

| Model | What it is |
|---|---|
| Sage (e.g. Sage-A1.5) | Finetune of **Qwen3.6-35B-A3B** (MoE, 3B active) |
| Axion-1.0-35B | Finetune of **Qwen3.6-35B-A3B** (MoE, 3B active) |
| Swift-1.5 | Finetune of **Qwen3.8-27B** (dense) |
| Inkling-2-Mini | Thinking Machines' `inkling2_loop` model, not based on Qwen |

Sage and Axion use the Qwen3.6-35B-A3B code path, so anything that holds for
Qwen3.6-35B-A3B holds for them too.

The repository builds a CLI, an HTTP server that speaks the OpenAI and Anthropic
APIs, a coding agent, a benchmark harness and an evaluation runner. The hot path
has no Python and no framework, and `make` is the whole build system.

## Speed on one BC-250

Measured 2026-11-24 with `p36-bench`: warm 2048-token prefill, 128 greedy
decode tokens, GPU governor running.

| Model | Prefill | Decode |
|---|---:|---:|
| Sage-A1.5 (Qwen3.6-35B-A3B finetune), APEX-I-NanoPlus | ~1,000 t/s | 118 t/s |
| Inkling-2-Mini, APEX-I-NanoPlus | ~750 t/s | 84 t/s |
| Inkling-2-Mini, APEX-I-MiniPlus | ~750 t/s | 83 t/s |

> **Endorsed quants:** PreonStar approves and endorses the
> [IsValorum](https://huggingface.co/IsValorum) custom APEX-I quants,
> **APEX-I-MiniPlus** and **APEX-I-NanoPlus**, for their speed and quality.
> Every number above was measured on them. They load as-is, with no conversion
> step, and use the engine's tuned Vulkan kernels. If you are choosing weights
> for a 16 GB board, start there.

On the same board, the `llama.cpp` Inkling-2 fork decodes Inkling-2-Mini at
47.1 t/s, and PreonStar measured 77.1 t/s under the same `tg128` conditions (see
[Performance](#performance)). The cost is generality: PreonStar targets one
board and a few model families, with kernels tuned for each.

[Quickstart](#quickstart) | [Documentation](docs/README.md) | [Benchmarks](docs/benchmarks.md) | [Status](#status)

## What you get

- Qwen3.6-35B-A3B (35B total, ~3B active), dense Qwen3.8-27B, or Inkling-2-Mini
  (`inkling2_loop`, 24 physical and 48 looped logical layers), in 2 to 5 bit
  quantizations, resident or streamed from disk.
- An interactive CLI, an HTTP server with OpenAI and Anthropic wire formats, a
  native coding agent, and a batch evaluation harness.
- Kernels tuned and measured on the AMD BC-250 (gfx1013, Vulkan/RADV). Other
  Vulkan 1.1 devices run through generic fallbacks, which are untuned and
  unvalidated.

## Supported runtimes

| Backend | Platform | Status |
|---|---|---|
| Vulkan | Linux, Vulkan 1.1+ | Production path. The BC-250 fast path is selected by device ID |
| CPU | Any C99 platform | Reference and debugging runtime |

Windows and macOS are not build targets.

`make` produces six binaries: `p36`, `p36-server`, `p36-agent`, `p36-bench`,
`p36-eval`, `p36_test`.

## Requirements

- Linux, a C99 compiler, GNU `make`.
- 16 GB of memory to keep the default model resident; 8 GB works with SSD
  streaming.
- `glslang-tools` (or `glslc`) to compile the Vulkan shaders.
- A Vulkan driver with shader `float64` and `int64`. The engine probes subgroup
  arithmetic, 16-bit storage and native `float16` at runtime and falls back
  when they are missing. See [backends.md](docs/backends.md).
- On the BC-250: a recent Mesa/RADV, a 512 MB UMA split, and a GPU governor to
  hold clocks. Setup steps are in [docs/hardware/bc250.md](docs/hardware/bc250.md).

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

Check the build without a model or a GPU:

```sh
make check
```

## Configuration

These flags cover most use. The rest are in [configuration.md](docs/configuration.md).

| Flag | Use |
|---|---|
| `-m, --model FILE` | Model to load. Defaults to the q2 imatrix download |
| `--ctx N` | Context window in tokens |
| `--ssd-streaming` | Stream experts from disk instead of keeping them resident |
| `-ctk` / `-ctv` | KV cache types: `f16`, `q8_0`, `q4_0` (GPU default q8_0 / q4_0) |
| `--backend` | `vulkan` or `cpu` |
| `--kv-disk-dir DIR` | Persist the KV cache so restarts skip prefill |

Hardware-specific flags (prefill chunk width, cache sizing, CU occupancy hints)
are documented with the backend they affect.

## Performance

All numbers come from one BC-250 (16 GB UMA, RADV) with greedy decoding and a
Q8_0 K / Q4_0 V cache unless stated. They need
`systemctl is-active cyan-skillfish-governor-smu` to print `active`; with the
governor stopped, the same build reads 20 to 28% slower.

Results from 2026-11-24 (`p36-bench`, 2048-token prefill chunks, 128 generated
tokens). The figures come from the 2048-token prefill that takes the context
from 2048 to 4096. In a fresh process the first prefill runs while MoE expert
pages are still loading from SSD, and on Sage it reads about 450 to 550 t/s.

| Model (APEX-I quant) | Base | Prefill (warm, 2048 tok) | Decode tg128 @ 4k |
|---|---|---:|---:|
| Sage-A1.5 NanoPlus-V2.1 | Qwen3.6-35B-A3B finetune | **985-1105 t/s** | **118.2 t/s** |
| Inkling-2-Mini NanoPlus-V2.1 | Inkling-2-Mini | **739-753 t/s** | **84.0 t/s** |
| Inkling-2-Mini MiniPlus-V2.1 | Inkling-2-Mini | **741-753 t/s** | **83.4 t/s** |

Axion-1.0-35B (Qwen3.6-35B-A3B finetune) at ctx 1024, 64 generated tokens,
decodes at 111.78 t/s, up from about 103.9 at the start of the 2026-11-20
decode-kernel sprint. Every step was bit-identical (ROADMAP.md; kernel ledger in
[karpathy/AlreadyTried.md](karpathy/AlreadyTried.md)).

### Inkling-2-Mini (`inkling2_loop`)

PreonStar runs Thinking Machines' `inkling2_loop` models with dedicated Vulkan
shaders (`vulkan/inkling2_*.comp`). It reads both the APEX-I and the upstream
`llama.cpp` PR #32695 loop-controller tensor layouts, including `Q5_0` and
`IQ4_NL` MoE down projections. Earlier measurements on the BC-250 (greedy,
`p36-bench`, `n=128`, default `prefill_chunk=1024`):

| Model / Engine | GGUF Size | `pp1024` (warm) | `tg128 @ 512` | `tg128 @ 1k` | `tg128 @ 2k` | `6.5K` Needle *(pp / tg)* | Safe Max Context *(14 GiB RAM)* |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama-inkling2` `UD-IQ3_XXS` *(ref)* | n/a | 421.5 t/s | 47.1 t/s | n/a | n/a | n/a | n/a |
| **PreonStar NanoPlus-V2.1** | 10.23 GiB | **655.4 t/s** | **77.1 t/s** | **77.1 t/s** | **76.9 t/s** | n/a | **~64K to 96K** *(32 KiB/tok)* |
| **PreonStar MiniPlus-V2.1** | 12.09 GiB | **634.3 t/s** | **75.4 t/s** | **75.4 t/s** | **75.1 t/s** | **582.6 / 66.5 t/s** | **~16K to 32K** *(16K w/ LifeOS, 32K bare)* |

Two later commits are not in this table: the MoE expert re-layout at upload
(+4.7% MiniPlus decode, bit-identical, 5c08a39) and faster long-context
attention prefill (+4%, MiniPlus 16k to 32k prefill 516.9 to 542.0 t/s,
bit-identical, 18648e8).

40 of the 48 logical layers use sliding-window attention with a fixed window of
1024 (160 MiB total), so only the 8 full-attention layers grow with context
(`32 KiB/tok` in F16, plus ~0.35 GiB of fixed runtime buffers). Exact needle
retrieval (`COBALT-7749`) passed at 6,495 tokens (`--ctx 8192`, six wraps of the
SWA ring buffer, 0.22 GiB context buffers).

### Fitting in 16 GB

- Keep `--prefill-chunk` at its default with `--kv-disk-dir DIR`. On MoE models
  it resolves to 1024, which fills the 32-slot expert tiles and prefills about
  18% faster than 512 (Axion at 16k in `p36-bench`: 759 to 896 t/s; a cold
  20.4k-token server prompt: 709 to 811 t/s end to end) for about 125 MiB more
  scratch. Tested on 2026-11-17 up to 32k (`p36-bench`) and 20.4k (`p36-server`
  with `--kv-disk-dir`) with no failures. The RADV 64 MB scratch-buffer failure
  that older builds hit at 1024 no longer reproduces. Builds before the
  2026-11-17 scratch-pool fix decoded about 23% slower after a multi-chunk
  prefill at 1024 (Axion 76 vs 99 t/s), so rebuild before switching a server to
  1024. Dense (GQA 6) models stay clamped to 256 on this board, where 512 gave
  +3% prefill for 93 MiB more scratch.
- `--kv-disk-dir` leaves decode speed unchanged and lets a repeated prompt skip
  prefill (time to first token 15.88 s to 1.98 s at 10.8k, on an earlier build).
- After a route change, rebuild every binary (`make -j16`, not only `make p36`).
  A stale `p36-server` keeps running the old route without an error and reads
  about 30% slower.

These numbers come from one board. [benchmarks.md](docs/benchmarks.md) explains
how to measure your own machine, gives the exact harness command and the A/B
protocol used to accept a performance change, and lists the fields every result
row must record.

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

The fragments form a single translation unit on purpose. The build has no LTO,
so separate `.c` files would need exported symbols and would lose inlining on
the hot path. [architecture.md](docs/architecture.md) covers why the code is
shaped this way and what each module owns.

## Documentation

- [docs/README.md](docs/README.md): index of everything below.
- [Configuration](docs/configuration.md): all flags, the KV cache, migrating from q36.
- [Server](docs/server.md) and [Agent](docs/agent.md): the two long-running programs.
- [Backends](docs/backends.md): Vulkan and CPU requirements.
- [Benchmarks](docs/benchmarks.md): how to measure and report results.
- [Streaming](docs/streaming.md): running models larger than memory.
- [Models](docs/models.md): supported families, quants, vision.
- [Troubleshooting](docs/troubleshooting.md): steering, debug switches, failure modes.
- [CONTRIBUTING.md](CONTRIBUTING.md): read this before sending a pull request.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md): the release test matrix.
- [MODEL_CARD.md](MODEL_CARD.md): model details and limitations.
- [changelog.md](changelog.md): what changed and when.

## Status

Alpha. The code changes quickly and may be unstable. Each release goes through a
full QA run; see [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md).

## Lineage and acknowledgements

PreonStar started as a fork of [**q36**](https://github.com/Ninnix/q36) (branded
QuarkStar) by Ninnix. Both projects build on **DwarfStar**, **llama.cpp** and
**GGML**. Thank you to Ninnix, Salvatore Sanfilippo (Antirez), Georgi Gerganov,
and every contributor to those projects.

The project follows the best open weights for 16 GB machines and 24/32 GB
workstations, and drops a model when a better replacement arrives.

## AI disclosure

Parts of this codebase and its documentation were written with AI assistance. A
human reviews, builds, tests and benchmarks all of it before release. Claims in
this repository are backed by code you can read and numbers you can reproduce,
and unverified claims are marked as such.

## License

MIT. See [LICENSE](LICENSE).
