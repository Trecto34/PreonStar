<p align="center">
  <img src="logo.svg" alt="PreonStar logo" width="220">
</p>

**PreonStar** is a small native inference engine for **Qwen3.6-35B-A3B**,
**Qwen3.8-27B**.
It is self-contained and deliberately narrow, not a general GGUF runner. 
The main paths are `qwen35moe`-specific Vulkan and Metal graph executors with
Q36-specific loading, prompt rendering, tool calls, KV state, HTTP server,
and coding agent. The repository also includes tools and data for GGUF,
imatrix, quality, and speed.

PreonStar started as a fork of [**q36**](https://github.com/Ninnix/q36)
(branded QuarkStar) by Ninnix, and neither it nor q36 would exist without
**DwarfStar**, **llama.cpp and GGML** — make sure to read the acknowledgements
section, a big thank you to Ninnix, Salvatore Sanfilippo aka Antirez, Georgi
Gerganov, and all the other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 16 GB machines and 24/32 GB
workstations. A model may be removed when a better replacement arrives.

Supported runtimes:

* **Metal** on Apple Silicon M1 or newer. A 16 GB Mac can run the default
  release model resident; 8 GB Macs are intended to use SSD streaming.
* **Vulkan** on Linux. The backend is generic; its current fast path was
  developed and tuned on the AMD BC-250 with RADV.
* **CPU** as a portable reference and debugging runtime, not the normal
  production path.

# So, what can I do with this software?

* You can run a quite capable model on your very cheap hardware, a $150 machine, 
the BC-250. Even if you don't have enough RAM, with SSD streaming, you can still 
run it at a decent speed.
* You can run the same CLI, agent, evaluation harness, benchmark, and server on
  Apple Silicon through the native Metal graph runtime.

## Requirements

QuarkStar has two native GPU backends:

* **Linux with Vulkan 1.1 or newer** and support for shader `float64` and
  `int64`. Subgroup arithmetic, 16-bit storage, and native `float16` are
  detected at runtime; optimized kernels fall back when an optional feature
  is unavailable.
* **AMD BC-250** (Cyan Skillfish, 24 CUs stock or 40 with the optional kernel
  unlock, 16 GB unified GDDR6) is the primary tested Vulkan device. The
  published benchmarks below were taken on an unlocked 40-CU board; confirm
  which your board reports with `RADV_DEBUG=info`, since occupancy tuning and
  any "workgroups per CU" reasoning depend on it. Its vendor/device ID selects the tuned fast path
  automatically. Follow [BC250.md](BC250.md) for the RADV, UMA, kernel-memory,
  governor, build, and optional 40-CU setup.
* **Apple Silicon M1 or newer** with macOS 11 or newer. Build with
  `make metal` and run with `./q36 --metal`.

Windows is not currently a build target. The Metal backend uses unified-memory,
mmap-backed model buffers and is independent of the Vulkan build.

## Motivations

Small open-weight models are already good, keep getting better, and fit on
normal personal machines — a $150 BC-250 (powerful GPU, fast unified memory,
decent SSD) is enough to run Qwen3.6-35B-A3B at good speed thanks to
aggressive routed-expert quantization, and compressed KV caches plus fast SSDs
make long contexts practical on that budget. This is an inference system
specialized for a few models rather than a general-purpose runner.

## AI full disclosure

Developed with strong assistance from GPT 5.5, 5.6, and Claude Fable, with
humans leading ideas, testing, and debugging. If you're not comfortable with
AI-assisted code, this project is not for you.

## Acknowledgements

QuarkStar is a port of [DwarfStar](https://github.com/antirez/ds4) by
Salvatore Sanfilippo (antirez), redesigned for Vulkan and retargeted at
Qwen3.6-35B-A3B; the Apple/Metal runtime also retains and specializes
DwarfStar's Metal kernel library. `q36.c` does not link against GGML, but
exists thanks to the path opened by [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and GGML — kernels, quantization formats, and the GGUF ecosystem. Some
source-level pieces (GGUF quant layouts/tables, CPU quant/dot logic, certain
kernels) are adapted under the MIT license, and we keep the GGML authors'
copyright notice in `LICENSE`. Thanks also to Georgi Gerganov and all other
llama.cpp/GGML and DwarfStar contributors.

## Status

The software is currently very fast changing. Consider it alpha quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

## More Documentation

The focused documents below cover development, release checks, model details,
and offline tooling. For normal usage, keep reading the next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [BC250.md](BC250.md): Linux, RADV, unified-memory, governor, and build setup
  for the primary Vulkan device.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md): the complete release test
  matrix.
- [MODEL_CARD.md](MODEL_CARD.md): the fixed Qwen3.6 architecture, tokenizer,
  quantization, and sampling assumptions used by Q36.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against OpenRouter continuations.
- [dir-steering/README.md](dir-steering/README.md): the preserved DS4 vector
  generation experiment that informed Q36's directional-steering runtime.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): tracked
  llama.cpp continuation vectors used for regression checks.

## Model Weights

This implementation works with Qwen3.6-35B-A3B and dense Qwen3.8-27B GGUFs.
It is not a general GGUF loader: arbitrary GGUF files will not have the
tensor layout, quantization mix, metadata, or optional MTP state the engine
expects. The 2-bit quants provided here are verified high quality — they
behave well and call tools reliably under coding agents.

Start dense Qwen3.8-27B explicitly:

```sh
./q36-server \
  -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vulkan
```

### Vision

Q36 accepts the matching Qwen3-VL projector as a disk-backed sidecar. It maps
the file without copying the model and streams one weight tensor at a time
through a reusable Vulkan buffer. The full 0.9 GiB projector is never retained
in RAM or VRAM, and streamed pages are released after each dispatch.

```sh
./q36 --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf
./q36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vision gguf/Qwen3.8-27B-mmproj-F16.gguf
```

At the interactive prompt, `/read photo.jpg` and `/read image.png` submit an
image turn, optionally with a prompt (`/read photo.jpg Describe this image`).
JPEG/PNG decoding is built in. Vision prompt execution currently requires
Vulkan. `q36-agent --vision FILE` exposes `view_image` for local files;
`q36-server --vision FILE` accepts OpenAI image data URIs, Responses image
blocks, and Anthropic base64 images (remote image URLs are rejected). The
server caches at most 32 MiB of decoded image embeddings; projector weights
stay streamed from disk. Vision sessions currently cannot be saved as agent
sessions or disk KV checkpoints.

The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality. The resulting
weight footprint is roughly 10-11 GB, which fits the BC-250 with room left
for KV cache and OS.

The higher-quality mixed model keeps that layout for layers 0..33 and uses
`Q4_K` routed gate, up, and down tensors in layers 34..39. Its roughly 13 GB
weight file is intended for SSD streaming on 16 GB machines.

Download one main model.

```sh
./download_model.sh q2-imatrix   # 16 GB unified memory machines
./download_model.sh q2-q4-imatrix # higher quality; stream on 16 GB machines
./download_model.sh 27b           # Dense Qwen3.8-27B IQ3_S
```

Choose one. The first target matches the compiled default path. Select the
second explicitly:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-Layers34-39Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-Q8Rest-imatrix.gguf \
  --ssd-streaming -p "Hello"
```

The script downloads Qwen3.6 models from
`https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf` and dense Qwen3.8-27B
from `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`. It stores files under
`./gguf/`, resumes partial downloads with `curl -C -`, and updates
`./q36moe.gguf` to point at the selected model for older scripts.

For the reasoning-efficient Swift derivative, the practical tier for this
16 GB BC-250 is IQ3_XXS, from
`ukisai/Swift-Qwen3.8-27B-GGUF`:

```sh
./download_model.sh swift-iq3
./run_swift_iq3_xxs.sh -p "Explain why the sky is blue in two sentences." --nothink
```

`run_swift.sh` uses a 4096-token default context and binds the embedded MTP
draft head, but defaults to one draft position because the BC-250 benchmark
was faster without speculative verification. Set `Q36_SWIFT_MTP_DRAFT=3` to
opt into three-position MTP. IQ4_XS had a dedicated bounded-memory
profile, but its 14.6 GiB file was removed because it was unusably slow on this
hardware and full residency triggered the host OOM killer.

```sh
./run_swift_iq3_xxs.sh -p "Explain why the sky is blue." --nothink
```

Then build for the target platform:

```sh
make                  # Linux / generic Vulkan, automatic BC-250 fast path
make vulkan-generic   # explicit generic Vulkan build
make vulkan-bc250     # Vulkan build that requires a BC-250 at runtime
make metal            # macOS / Apple Silicon
```

The normal and CPU-only builds are self-contained. They do not require a
llama.cpp checkout or GGML libraries; those are used only by explicit optional
reference targets under `make test-llama` and `make test-vectors-local`.

The vendored JPEG/PNG decoder in `third_party/iris` is Copyright (c) 2026
Salvatore Sanfilippo and distributed under its included MIT license.

`gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf` is
the default model path used by all runtime binaries. Pass `-m` to select another
supported GGUF from `./gguf/`. Run `./q36 --help` and
`./q36-server --help` for the full flag list.

If you want to regenerate GGUF files, quantize community fine-tunes / abliterated models
(via `--strip-nextn`), or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
Qwen3.6 model-building work. The native quantizer accepts Q8, F16, or BF16
inputs; imatrix collection still uses the optional llama.cpp tooling.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF for Qwen 3.6 MoE. It can be used with the `q2-imatrix` and
`q2-q4-imatrix` main models, but must be enabled explicitly with `--mtp`. The
current MTP/speculative decoding path is still experimental: it is
correctness-gated and currently provides at most a slight speedup, not a
meaningful generation-speed win.

## Speed

Benchmarks on a **BC-250** (40 CUs unlocked, Cyan-Skillfish Governor performance
mode at 500-2000 MHz, 85 °C thermal ceiling — a hot summer in Italy). Runs
use greedy decoding, `--gen-tokens 128`, the extended long-context story
prompt, and Q8_0 K / Q4_0 V cache. The q2 model is fully resident. The mixed
q2-q4 model uses SSD streaming with the automatic 5724-expert (4.72 GiB)
cache. Each row was measured in a separate process, with only one benchmark
process running at a time.

| Machine | Quant | Mode | Prompt | Prefill | Generation |
| --- | ---: | --- | ---: | ---: | ---: |
| BC-250 (40 CU) | q2 | Resident | 2048 ctx | 639.85 t/s | 81.85 t/s |
| BC-250 (40 CU) | q2 | Resident | 4096 ctx | 597.10 t/s | 79.74 t/s |
| BC-250 (40 CU) | q2 | Resident | 8192 ctx | 501.50 t/s | 74.72 t/s |
| BC-250 (40 CU) | q2 | Resident | 16384 ctx | 373.32 t/s | 65.04 t/s |
| BC-250 (40 CU) | q2 | Resident | 24576 ctx | 287.24 t/s | 56.17 t/s |
| BC-250 (40 CU) | q2 | Resident | 32768 ctx | 244.06 t/s | 51.26 t/s |
| BC-250 (40 CU) | q2-q4 | SSD streaming | 4096 ctx | 45.43 t/s | 14.52 t/s |
| BC-250 (40 CU) | q2-q4 | SSD streaming | 8192 ctx | 48.78 t/s | 13.98 t/s |

![BC-250 Q2 t/s](speed-bench/bc250_ts.svg)

### Mainline q36 vs PreonStar fork

Interleaved A/B/A/B comparison on the same BC-250, mainline q36 (upstream
`Ninnix/q36` at `9fb537a`) against this fork (`9caf3c5`), two reps per binary
per context size, averaged. Same run conditions as above (greedy decoding,
`--gen-tokens 128`, Q8_0 K / Q4_0 V cache). Benchmarked using the model
`Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf`
(https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf).

| Binary | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: |
| mainline (9fb537a) | 2048 ctx | 602 t/s | 81 t/s |
| PreonStar (9caf3c5) | 2048 ctx | 924 t/s | 85 t/s |
| mainline (9fb537a) | 4096 ctx | 378 t/s | 71 t/s |
| PreonStar (9caf3c5) | 4096 ctx | 538 t/s | 68 t/s |
| mainline (9fb537a) | 8192 ctx | 298 t/s | 66 t/s |
| PreonStar (9caf3c5) | 8192 ctx | 402 t/s | 79 t/s |
| mainline (9fb537a) | 16384 ctx | 184 t/s | 56 t/s |
| PreonStar (9caf3c5) | 16384 ctx | 293 t/s | 57 t/s |

PreonStar's prefill throughput is 53-59% higher than mainline at every
context size tested; generation throughput is close, with a modest edge for
PreonStar at 8k+ context.

### Long-context prefill: flash-attention

Attention is the only prefill cost that grows with context, so it decides how
fast long agent prompts start. `attn_prefill_fa.comp` serves 8 tokens × 6
heads (dense Qwen3.8-27B) or 4 tokens × 8 heads (Qwen3.6/3.8-35B-A3B) per
workgroup, dequantizes each K/V tile into LDS once, and keeps Q in registers.
It is 2.4-3.5x faster than the previous kernel at 4k-16k context. Measured on
one BC-250 at 8192 ctx, 5 interleaved reps per arm:

| Model | Before | Flash-attention | Delta |
| --- | ---: | ---: | ---: |
| Swift-Qwen3.8-27B-IQ3_XXS (dense) | 97.98 t/s | 107.28 t/s | +9.49% |
| Qwen3.6-35B-A3B IQ2_XXS (MoE) | 386.47 t/s | 491.38 t/s | +27.15% |

Decode is unchanged. The gain grows with context and is small at 1-2k. It is
on by default; `Q36_VK_ATTN_FA=0` restores the previous kernel. Numbers,
parity and quality checks: `karpathy/evidence/attn-fa-prefill.md`.

Metal benchmark numbers (M2 Pro) were dropped from this README since the
primary tested/tuned device is the BC-250; the Metal backend is still fully
supported (see Requirements above), just not benchmarked here.

Use `q36-bench` for reproducible prefill and decode measurements. Release
builds also have a conservative BC-250 performance gate under
`make benchmark-gate`; record results on the same board and power state when
comparing changes.

## Running Models Larger Than Available Memory

Metal and Vulkan SSD streaming keep non-routed weights resident and load
selected routed experts into a bounded backend cache. The planner reserves memory
for the configured context, prefill scratch, planned server sessions, and a staging
margin before assigning the remaining model budget to static weights and experts.
Explicit cache sizes are targets capped by the same budget. Creating a larger or
additional session reduces the expert cache when necessary; it fails cleanly if
static weights and the minimum working cache leave insufficient room. Startup
logs show the resolved context and model budgets.

```sh
./q36 --metal --ssd-streaming -p "Explain radix trees."
./q36 --vulkan --ssd-streaming --ssd-streaming-cache-experts 256 -p "Hello"
./q36 --metal --ssd-streaming --ssd-streaming-cache-experts 1GB -p "Hello"
```

`NGB` is a routed-expert byte budget. Q36 converts it to the number of expert
slots that fit the release GGUF. A plain integer requests an expert-slot count;
both forms are capped to leave room for the configured contexts.
Non-routed weights, KV cache, activations, and graph scratch need additional
memory. Startup prints the resolved slot count and actual cache allocation.

A built-in or `Q36_VK_STREAMING_EXPERT_HOTLIST` profile biases eviction
without reading expert weights at startup. It does not fill the cache with
arbitrary experts when no profile exists. Use `--ssd-streaming-cold` for an
empty cache, or `--ssd-streaming-preload-experts N` to request an explicit
weight preload. Metal stores mixed expert sizes in component-wise padded cache
slots, so the IQ2/Q2 and Q4 routed layers share one bounded cache and one
hotlist policy. Vulkan retains its original single-size-class behavior. The
`Q36_VK_STREAMING_EXPERT_HOTLIST` and
`Q36_VK_DISABLE_STREAMING_EXPERT_HOTLIST` names are retained for compatibility
and apply to both graph runtimes.

As in DS4, an explicit routed prefix can stay fully resident:

```sh
./q36 --ssd-streaming --ssd-streaming-full-layers 4 -p "Hello"
```

Full layers are charged at their actual byte size and the remaining budget
must still hold one layer of dynamic expert slots for prefill. The default is
zero because the dynamic-only cache is faster on the BC-250. Pass
`--ssd-streaming-full-layers 0` to disable an explicit setting.

### Metal SSD streaming

Metal uses the same bounded-cache design, backed by shared Metal buffers with
biased LRU eviction instead of macOS VM-paging hints. The automatic cache
size uses `recommendedMaxWorkingSetSize`.

## Native Agent

Q36 includes a native coding agent. Inference is controlled inside the agent
itself, without a socket or API boundary, so the transcript and live KV state
are one session. The tools and system prompt use Qwen3.6's native tagged tool
format directly. This provides a few advantages:

- Low latency for generated text, tool calls, and new sessions.
- Live progress during long prefills.
- No OpenAI, Anthropic, or Hermes conversion in the model loop.
- The transcript and KV state cannot drift apart.
- Built-in file, search, shell, process, and web tools tuned for the model.
- Saved sessions can be switched without prefill when their KV payload is
  present.

Start the agent in the current directory, another project, or one-shot mode:

```sh
./q36-agent
./q36-agent --chdir /path/to/project
./q36-agent --non-interactive -p "Inspect the tests and fix the failure."
```

Adaptive thinking closure starts after 50000 thinking tokens by default. For
budgets of at least 8000, its allowed `</think>` rank rises to the top 64 over
the next `min(N/2, 8192)` tokens, where `N` is `--thinking-budget`. Smaller
budgets keep the previous ranking schedule. The rank can rise further if
thinking continues. This is a soft budget; `--tokens` is the hard output limit.

With Qwen3.8-27B, `--thinking-budget` also selects the starting effort:
up to 8000 tokens is `low`, up to 16000 is `medium`, up to 24000 is `high`,
and larger budgets use `xhigh`. The default is `xhigh` with a 50000-token
closure target. `high` uses Qwen3.8's native `xhigh` instruction. Explicit
`--think-low`, `--think-medium`, `--think`, or `--think-xhigh` overrides the
starting mode without changing the budget. At an empty idle agent prompt,
Tab cycles `low → medium → high → xhigh` for Qwen3.8 and `off ↔ on` for
Qwen3.6. Tab leaves the closure budget unchanged; the footer shows both.

Agent user and system messages accept `<|think_on|>` and `<|think_off|>`.
The marker is removed before rendering and remains in effect for later turns.
Historical thinking stays in the append-only transcript until compaction.

Resident Metal and Vulkan both use Q8_0 keys with Q4_0 values and default to a
100000-token agent context. The backend's automatic resident GPU prefill width
resolves to 1024 tokens. Compact Metal attention scratch keeps this faster
chunk width practical at long contexts. CPU uses F16 KV. SSD-streamed model
weights also default to a 100000-token context with F16 KV.
Explicit `--ctx`, `-ctk`, and `-ctv` values override the preset.

`q36-agent --chdir` loads its model and runtime assets from the launch directory,
then changes to the requested project for agent tools. Built-in path tools also
expand `~` to the current user's home directory.

When a command such as `sudo` requests a terminal password, the interactive
agent opens a private prompt with input hidden. Type the password there;
Ctrl+C cancels the command. Password input is sent directly to the command,
without entering chat history, traces, or captured tool output. Other command
stdin reads receive EOF; `--non-interactive` cannot prompt for passwords.

Use `/hints on` for occasional short explanations of programming concepts behind
the current work, rendered as teal blockquotes. `/hints off` disables them. This
setting lasts for the current process and is reapplied after context compaction.

Both `q36` and `q36-agent` accept `--prefix-file FILE`. The file contains alternating
`USER:` and `ASSISTANT:` lines, starting with a user and ending with an assistant.
Turn content may span multiple lines. The CLI prefills this conversation before
the first interactive prompt; the agent keeps it through reset and compaction.
For example:

```text
USER: Our project uses C99 and has no external dependencies.
ASSISTANT: I will follow those constraints.
```

Sessions are stored in `~/.q36/kvcache`. Use `/save` to persist the current
session, `/list` to show saved sessions, and `/switch <sha>` to resume one.
The session ID remains stable across later saves. `/del <sha>` removes a saved
session. `/strip <sha>` keeps its transcript and title but removes the KV
payload; switching to a stripped session rebuilds the KV cache by prefilling
the saved text. `/compact` compacts the current context immediately.
Exiting during generation stops the worker before asking whether to save.
Sessions containing images cannot be saved yet; declining to exit after a save
failure returns to the current chat.

## Benchmarking

`q36-bench` measures instantaneous prefill and generation throughput at
context frontiers instead of reporting one whole-run average. It loads the
model once, walks a fixed token sequence to frontiers such as 2048, 4096, and
6144, and uses incremental prefill so each row measures only the newly added
token interval. After each frontier it saves the live KV state to memory,
generates a fixed greedy non-EOS probe, restores the snapshot, and continues
prefill.

```sh
./q36-bench --vulkan --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 32768 --gen-tokens 128
```

## Capability Evaluation

`q36-eval` is a real-model integration benchmark, not a leaderboard runner.
Its 92 embedded questions are a regression subset: 25 GPQA Diamond, 25
curated SuperGPQA, 25 AIME 2025, and 17 COMPSEC cases. It loads the GGUF,
renders Qwen3.6 chat prompts, streams sampled tokens in a TUI, grades the final
answers, and prints prompt-token, generated-token, and pass/fail results.

```sh
./q36-eval --trace /tmp/q36-eval.txt
```

The default run uses a 16000-token generation budget and thinking mode. The
context is sized from the largest selected prompt plus that budget, up to the
model's 262144-token native context. Press `p` to pause, `q` to stop and print
the report, Up/Down to select a question, and Enter to queue it next. `--plain`
disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to rerun the current answer extractor
and scorer on a saved trace without loading the model. A short deterministic
smoke run is:

```sh
./q36-eval --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

## CLI

One-shot prompt:

```sh
./q36 -p "Explain Redis streams in one paragraph."
```

Without `-p`, Q36 starts an interactive multi-turn chat:

```sh
./q36
q36>
```

The CLI keeps the rendered transcript and live graph KV checkpoint, so each
turn extends the previous conversation. Useful commands are `/help`, `/think`,
`/think-max`, `/nothink`, `/ctx N`, `/read FILE [PROMPT]`, and `/quit`. Ctrl+C interrupts
the current generation and returns to `q36>`.

Thinking mode is enabled by default. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path for greedy decoding. It uses `--mtp-margin` as a confidence gate and is
currently an experimental slight-speedup path.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./q36-server --vulkan --ctx 32768 \
  --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192

./q36-server --metal --ctx 32768 \
  --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

On an 8 GB Mac, reduce context and use the bounded Metal cache:

```sh
./q36-server --metal --ssd-streaming \
  --ssd-streaming-cache-experts 512 --ctx 4096
```

Without extra options the server keeps one mutable backend/KV checkpoint and
uses the original single graph worker. Stateless clients that resend a longer
version of the same prompt can reuse that prefix instead of pre-filling from
token zero.

`--batched-session N` opts into `N` independent resident sessions:

```sh
./q36-server --ctx 32768 --batched-session 4
```

Use `--mixed-prefill-quantum N` to tune how many prefill tokens a batched
session runs per scheduling turn while another session is generating. The
default is 128; smaller values favor decode latency, larger values favor
prefill throughput.

Each active request owns one slot until it finishes; excess requests wait for
an idle slot. Assignment prefers the resident live/token prefix with the
longest match. When disk KV caching is enabled, an unmatched idle slot is
persisted before reuse. Thinking state, tool continuations, RNG, logits,
recurrent state, and full-attention KV remain session-local.

One coordinator owns model execution. Decode-ready slots coalesce for up to
2 ms and advance in one model step. Prefills run round-robin in bounded
quanta: 2048 tokens while no generation is active and 128 while any generation
is active. This prevents one large prompt from blocking every active decoder.
MTP speculative decoding is disabled in batched mode.

Resident context memory is multiplied by `N`; startup prints both the
per-session estimate and the total. Choose `N` and `--ctx` so all session KV,
recurrent state, and graph scratch fit. Backend behavior is:

| Backend | Multi-session execution |
| --- | --- |
| Vulkan resident, 2-8 decode-ready rows | Native row-batched shared projections and FFN work with private positions, recurrent state, and typed KV when graph scratch can hold every row. F16/F16 and Q8_0/Q4_0 KV are supported; other KV pairs and unsupported shapes use ordered fallback. |
| Vulkan SSD streaming | Deterministic ordered fallback, preserving expert-cache ownership. |
| Metal resident, 2-8 decode-ready rows | Native row-batched graph execution where the kernel shape is supported; ordered exact fallback otherwise. |
| Metal SSD streaming | Deterministic ordered fallback, preserving bounded expert-cache ownership. |
| CPU, batches above 8, or forced `Q36_VK_SESSION_BATCH=0` | Deterministic ordered fallback. The compatibility environment variable also controls Metal session batching. |

Batch size one calls `q36_session_eval()` directly. If a batched step fails,
all members are invalidated so none can silently continue from a partially
advanced frontier. Ordered fallback preserves concurrency and scheduling
fairness, but does not provide the aggregate throughput gain of native GPU
batching.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/<loaded-model-id>`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `presence_penalty`, `frequency_penalty`, `seed`, `stream`,
`stream_options.include_usage`, `tools`, and `tool_choice`. Tool schemas and
calls use the native Qwen3 Coder tagged format, and generated calls are mapped
back to OpenAI tool calls.

Both Qwen3.6 and Qwen3.8 accept the fixed-template controls:

```json
"chat_template_kwargs": {
  "enable_thinking": false,
  "preserve_thinking": true
}
```

`preserve_thinking` defaults to `true`, retaining earlier assistant reasoning
verbatim so later prompts remain a prefix-cache match. Set it to `false` to
strip reasoning before the latest user query. System and user messages may also
contain `<|think_on|>` or `<|think_off|>`; Q36 removes the control marker before
rendering and applies it to subsequent turns.

When omitted by the client, Qwen uses `temperature=1`, `top_p=1`, no top-k cap,
and `min_p=0.05`. CLI, agent, and server use the loaded model's defaults. Eval
stays fixed at the Qwen sampling defaults for comparable runs. Explicit values
always win.

`/v1/responses` accepts string or message-array input, instructions, direct
tool schemas, function-call continuations, function-call outputs, sampling
controls, reasoning controls, and `max_output_tokens`. It returns native
Responses API message, reasoning, and function-call output items. With
`stream:true`, it emits Responses API SSE events through
`response.completed`.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code
style clients. It accepts `system`, `messages`, `tools`, `tool_choice`,
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`,
and thinking controls. Tool uses are returned as Anthropic `tool_use`
blocks.

Both APIs support SSE streaming. In thinking mode, reasoning is streamed in
the native API shape instead of being mixed into final text. OpenAI chat
streaming also streams tool calls as soon as the `<tool_call>` opening is
recognized: the tool header is sent first, then each completed native
parameter is forwarded as a `tool_calls[].function.arguments` delta while
generation continues. The Anthropic endpoint streams thinking and text live, then emits
structured `tool_use` blocks when the generated tool block is complete.

Pass `--cors` to add `Access-Control-Allow-Origin`, methods, and headers and
to answer browser `OPTIONS` preflight requests. CORS headers are disabled by
default.

### Tool call handling and canonicalization

Qwen3.6-35B-A3B emits tool calls in its native tagged format. Tool definitions
are provided in the system prompt inside `<tools>...</tools>`. A call has one
`<function=name>` block inside `<tool_call>`, with one
`<parameter=name>` block per argument:

```text
<tool_call>
<function=list_files>
<parameter=pattern>
*.c
</parameter>
<parameter=max_depth>
2
</parameter>
</function>
</tool_call>
```

Agent clients do not send that same text back on the next request: they send
normalized OpenAI/Anthropic JSON tool-call objects. **If the server
re-rendered those objects slightly differently, the rendered byte prefix
would no longer match the live KV checkpoint** and the next turn would have
to be rebuilt.

All markers are plain ASCII. Q36 keeps the exact replay and canonicalization
machinery inherited from `ds4`, because a sampled call and its next-turn API
rendering must still be byte-identical to avoid silent KV drift.

The first line of defense is exact replay. Every tool call gets an
unguessable API tool ID, and the server remembers `tool id -> exact sampled
<tool_call> block` in a bounded in-memory map backed by radix trees. When
the client later sends that tool ID back, the prompt renderer uses the exact
bytes the model sampled, not a freshly formatted approximation. This map
can also be saved inside KV cache files, so exact replay survives server
restarts for cached histories.

**Canonicalization is only the backup path**. If the exact sampled block is
missing, or exact replay is disabled with
`--disable-exact-tool-replay`, the server renders deterministic native Qwen
tags from the JSON tool object, following schema property order. After a
tool-call turn, it compares the live sampled token stream with the prompt
that the next client request will render. If needed, it rewrites the live
checkpoint, or falls back to an older disk KV snapshot and replays only the
suffix. This keeps the model continuation aligned with the stateless API
transcript.

During generation, the server also treats native Qwen syntax differently from
payload. When the model is emitting stable protocol structure — tool,
function, and parameter tags — sampling is forced to `temperature=0` so
the tool call stays parseable. This greedy mode does **not** apply to
argument values: string contents inside the arguments JSON, including file
contents and edit text, use the request's normal sampling settings. That
separation is important: deterministic decoding is helpful for syntax, but
can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`q36-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit
no higher than the `--ctx` value you started the server with:

```sh
./q36-server --ctx 32768 --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

On a BC-250 with 16 GB of unified memory, weights take ~10–11 GB at Q2,
which leaves roughly 2–4 GB for KV cache, scratch buffers, OS and your
client. The model's native context is 256K tokens, but the live context must
still fit in available memory. Disk KV checkpoints avoid repeated prefill and
preserve sessions across restarts; they do not enlarge the active context
window.

The `262144` output limit in the configs below matches the model's native
context ceiling. The server stops earlier when its configured context window
is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "q36": {
      "name": "q36.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "q36-local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 MoE (q36.c local)",
          "limit": {
            "context": 32768,
            "output": 262144
          }
        }
      }
    }
  },
  "agent": {
    "q36": {
      "description": "Qwen 3.6 MoE served by local q36-server",
      "model": "q36/qwen3.6-35b-a3b",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "q36": {
      "name": "q36.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "q36-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "qwen",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen 3.6 MoE (q36.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 262144,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "q36",
  "defaultModel": "qwen3.6-35b-a3b"
}
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like
this matches the local `~/bin/claude-q36` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${Q36_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${Q36_API_KEY:-q36-local}"
export ANTHROPIC_MODEL="qwen3.6-35b-a3b"

export ANTHROPIC_CUSTOM_MODEL_OPTION="qwen3.6-35b-a3b"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 3.6 MoE local q36"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="q36.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.6-35b-a3b"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-35b-a3b"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens,
before it starts doing useful work. Keep `--kv-disk-dir` enabled: after the
first expensive prefill, the disk KV cache lets later continuations or
restarted sessions reuse the saved prefix instead of processing the whole
prompt again.

## Thinking Modes

Qwen3.6-35B-A3B and Qwen3.8-27B have distinct non-thinking and thinking modes, controlled
natively by `<think>...</think>` blocks rendered into the prompt. The server
defaults to thinking mode.

Mapping of API thinking controls to prompt rendering:

- Anthropic `thinking: {"type":"enabled"}` → thinking on (default).
- Anthropic `thinking: {"type":"disabled"}` → thinking off.
- Qwen3.8-27B supports `low`, `medium`, and `xhigh` in its chat template.
  `high` and the default select `xhigh`; `minimal` selects `low`.
- Qwen3.6-35B-A3B has only thinking on or off. Positive efforts all enable
  thinking without a model-specific effort instruction.
- Numeric effort strings map `"0"` to thinking off, `"1"`–`"33"` to `low`,
  `"34"`–`"66"` to `medium`, and `"67"`–`"100"` to `xhigh`. On Qwen3.6,
  all positive values enable the same thinking mode. Only `"max"` selects
  QuarkStar Think Max.
- `thinking.budget_tokens` on chat or Anthropic messages starts adaptive
  `</think>` ranking at that token count. On Qwen3.8 it also selects `low`
  through 8000, `medium` through 16000, `high` through 24000, and `xhigh`
  above 24000. Explicit `none` and `max` efforts keep their mode.
- Explicit non-thinking: `thinking:{"type":"disabled"}`, `think:false`, or
  `chat_template_kwargs:{"enable_thinking":false}`. Both profiles use the
  `<think>\n\n</think>\n\n` non-thinking prefix.

## KV Cache Quantization

Q36 supports `f16`, `q8_0`, and `q4_0` KV rows. Select key and value types
independently with `-ctk` and `-ctv`:

```sh
./q36 -ctk q8_0 -ctv q4_0 -p "Explain radix trees."
./q36-server -ctk f16 -ctv f16
```

Resident Metal and Vulkan frontends default to Q8_0 keys and Q4_0 values. CPU
and SSD streaming default to F16 for both. Explicit flags always override
these defaults. KV quantization reduces context memory; it does not change
model weight quantization or extend Qwen3.6's 262144-token native context.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `q36-server` first tries the cheap exact
token-prefix check, then falls back to comparing rendered prompt bytes with
decoded checkpoint bytes. The live in-memory checkpoint covers the current
session; the disk KV cache makes useful prefixes survive session switches
and server restarts.

The default server has one live KV cache. With `--batched-session N`, each
resident slot has one and active slots are never evicted. When an unrelated
request reuses an idle slot, its old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache.

Enable it with:

```sh
./q36-server --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The Q36 payload still stores the exact token IDs and graph
state for that prefix. This matters for continued chats: the model may have
generated one token whose decoded text is later sent back by a client as two
canonical prompt tokens. A rendered byte-prefix hit can still reuse the
checkpoint and tokenize only the new suffix. The file is intentionally
written with ordinary `read`/`write` I/O, not `mmap`, so restoring cache
entries does not add more VM mappings to a process that already maps the
model.

Tool calls also keep a bounded exact-replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text.
The RAM map keeps up to 100000 IDs by default; tune it with
`--tool-memory-max-ids`. Use `--disable-exact-tool-replay` to disable this
and fall back to canonical JSON-to-Qwen rendering.

Each cache file (`<sha1>.kv`) holds a small header, the human-readable
rendered prefix text (both the lookup key source and the check against BPE
re-tokenization drift), the full Q36 session payload (checkpoint tokens,
next-token logits, and per-layer K/V or recurrent state), and an optional
tool-call exact-replay map so restarted servers can still render client tool
history byte-for-byte. See
[CONTRIBUTING.md](CONTRIBUTING.md#disk-kv-cache-file-format) for the exact
on-disk byte layout.

Checkpoints are written at four points — `cold` (first stable prefix),
`continued` (roughly every 10k tokens by default), `evict` (before an idle
slot is reused), and `shutdown` — with conservative defaults (512-token
minimum, 30000-token cold cap, 32-token tail trim, 2048-token alignment)
chosen to avoid retokenization misses at chunk boundaries. Tunable knobs:
`--kv-cache-min-tokens`, `--kv-cache-cold-max-tokens`,
`--kv-cache-continued-interval-tokens`, `--kv-cache-boundary-trim-tokens`,
`--kv-cache-boundary-align-tokens`, `--tool-memory-max-ids`,
`--disable-exact-tool-replay`.

The cache directory is disposable — stop the server and remove it if
behavior looks suspicious. Files use plain `read`/`write` I/O (not `mmap`)
and store the verbatim cached prompt text, so `hexdump` can inspect them
directly.

## Backends

Q36 is multi-runtime at the source and API level, but Metal and Vulkan are
separate native builds rather than one fat executable. Each build produces the
same command names—`q36`, `q36-server`, `q36-bench`, `q36-agent`, `q36-eval`,
and `q36_test`—linked to the selected graph runtime:

| Platform/runtime | Build | Explicit invocation |
| --- | --- | --- |
| Linux / generic Vulkan | `make` or `make vulkan-generic` | `./q36 --vulkan -p "Hello"` |
| Linux / BC-250 checked Vulkan | `make vulkan-bc250` | `./q36 --vulkan -p "Hello"` |
| macOS / Metal | `make metal` | `./q36 --metal -p "Hello"` |
| CPU reference | `make cpu` | `./q36 --cpu -p "Hello"` |

Building another runtime overwrites those local executable names. Use separate
worktrees or copy build artifacts if Metal, Vulkan, and CPU binaries must be
kept side by side. The `--backend metal|vulkan|cpu` spelling is equivalent to
the short backend switches and is accepted by the CLI, server, agent,
benchmark, and evaluation harness. A Metal-linked binary rejects `--vulkan`,
and a Vulkan-linked binary rejects `--metal`, so deployment mistakes fail
before model loading.

### Vulkan device and compatibility policy

The generic and BC-250 targets contain the same Vulkan backend and SPIR-V.
`vulkan-generic` queries device capabilities and uses portable fallbacks when
the subgroup or 16-bit features required by a tuned kernel are absent. A
BC-250 is recognized at runtime and keeps the current optimized path.
`vulkan-bc250` adds a vendor/device check so a board-specific release artifact
cannot silently run on different hardware.

The generic target removes the hard device identity restriction, not the need
to validate a new GPU. Before calling another Vulkan device supported, run the
isolated kernel suite and the short CPU/GPU parity gate, then record prefill
and decode throughput at the intended context sizes.

#### Prefill chunk safety clamp

On GFX1013 the prefill chunk is clamped (to 1024, or 256 when the model's GQA
ratio is not 8) after any `--prefill-chunk` override, because the override is
exactly the value that can push one attention dispatch past the kernel watchdog
and destroy the GPU context. The clamp also feeds the context-memory estimate,
so scratch is sized for the chunk that will actually run rather than the one
requested — scratch grows with chunk width, and sizing for an unclamped request
reserves memory that is never touched.

`Q36_VK_PREFILL_CHUNK_UNSAFE=1` restores the raw value. It is not recommended:
beyond the watchdog risk, a chunk wider than the clamp has measured *slower* on
this hardware as well as larger, so the clamp costs nothing in practice. Sweep
`--prefill-chunk` on your own board if you want to confirm the optimum for it.

#### Optional GPU fast paths

Two kernel variants trade bit-exactness for speed. Both are **off by default**,
both are available as flags on every front end (`q36`, `q36-server`, `q36-bench`,
`q36-eval`, `q36-agent`), and both also read an equivalent `Q36_VK_*` environment
variable.

| flag | what it changes | exactness |
| --- | --- | --- |
| `--f32-fast-wide` | Runs the f32 matvec with a 256-thread workgroup instead of 64. The narrow form dispatches a single wave for the router and shared-expert gates, which cannot cover memory latency. | **Not bit-exact, and known to fail in production use.** Four subgroup partials are combined through shared memory instead of one subgroup reducing alone, and this kernel computes the MoE router gate, so a rounding flip can change *which experts* a token selects. Treat it as a benchmarking switch only. Validate with the evaluation harness, not with a logits diff. |
| `--attn-span N` | Split-K span width in keys for decode attention (default 512). Narrower spans raise occupancy, since decode dispatches `n_head * spans` workgroups. | **Not bit-exact.** `attn_combine` reduces the per-span partials sequentially in f32, so a different span count regroups that sum. |

`--f32-fast-wide` is the more dangerous of the two and the smaller: isolated
per-kernel profiling puts it at about **half a percent of GPU time** on one
BC-250 (whole-run comparisons overstated it). Its danger is not hypothetical —
it perturbs the router softmax input, so a rounding flip can change which
experts a token selects; on one BC-250 it caused a degenerate failure in long
tool-calling chat (sampled distribution collapsed, generation ran to the token
budget). A short-context logits diff will not catch this. Enable it only for
measurement runs, never for a served endpoint.

Prefill attention uses the flash-attention kernel by default (see
[Speed](#long-context-prefill-flash-attention)). It is **not bit-exact** with
the previous `attn_prefill_qtile2` kernel: f32 sums are added in a different
order. On a teacher-forced next-token check at 8k it moved scores less than
changing `--prefill-chunk` from 256 to 128 does. Set `Q36_VK_ATTN_FA=0` to go
back for A/B comparisons.

`--attn-span` trades sequential `attn_combine` rescale cost against split-K
occupancy: narrower spans help at short context and can hurt at long context
(measured **+2.2% decode at ctx 2048, +1.2% at 8192, -1.9% at 16384** on one
BC-250 at span 128 vs. the default 512). Prefill is unaffected. Measure both
flags on your own board at the context length you actually run — benefit
depends on CU count, thermal headroom, and context.

### Metal device and compatibility policy

Like DS4, a normal local Metal process uses `MTLCreateSystemDefaultDevice`.
Apple Silicon exposes its unified GPU as one logical Metal device, so Q36
needs no device list for M1/M2/M3/M4+; one process owns one default device
and one model, with no layer split or multi-Mac mode.

The Metal binary targets macOS 11; newer facilities (macOS 15 residency sets,
`MTLMathMode`) are selected only after a runtime availability check, and host
code defaults to the Apple M1 instruction baseline so a binary built on a
newer Mac stays usable on M1 (`NATIVE_CPU_FLAG=-mcpu=native make metal` for a
local-only build). Routed MoE prefill uses DS4's expert-major batch-MM
dispatch where the shape is supported, falling back to the exact matvec path
otherwise (`Q36_METAL_MOE_MM=0` forces that fallback). Cache auto-sizing uses
`recommendedMaxWorkingSetSize`; all model buffers use unified storage, with
no discrete-GPU/eGPU path.

Metal shaders compile at runtime from `metal/*.metal` — run frontends from
the project tree and ship the `metal` directory with binary packages. A
frontend that changes working directory before model startup must set
absolute paths via `Q36_METAL_DENSE_SOURCE`, `Q36_METAL_MOE_SOURCE`,
`Q36_METAL_NORM_SOURCE`, `Q36_METAL_OPS_SOURCE`,
`Q36_METAL_RECURRENT_SOURCE`, `Q36_METAL_KV_SOURCE`, and
`Q36_METAL_ATTN_SOURCE`.

### CPU reference runtime

Do not treat the CPU path as the production target. The CLI and `q36-server`
support the CPU backend for reference/debug use and share the same KV
session and snapshot format as Metal and Vulkan, but normal inference should
use a GPU graph runtime.

## Steering

Q36 can edit attention and FFN outputs with the same projection used by DS4:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

The steering file is a flat `40 x 2048` native-endian `f32` matrix with one
normalized direction per Qwen layer. Positive scales remove the represented
direction; negative scales amplify it. FFN steering defaults to `1` when a
file is supplied without an explicit scale:

```sh
./q36 -p "Write tersely" \
  --dir-steering-file qwen-direction.f32 \
  --dir-steering-ffn 0.8
```

`--dir-steering-attn F` applies the same edit after attention outputs. With no
file, or with both scales set to zero, inference follows the normal path. CPU
and both GPU runtimes apply the same operation during prefill and decode; Metal
and Vulkan keep the matrix resident and project activations in place.

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get
a first answer:

```sh
./q36 --dump-tokens -p "..."
./q36 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./q36-server --trace /tmp/q36-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p`/`--prompt-file` string exactly as
  written, recognizes Qwen protocol specials (`<|im_start|>`, `<think>`,
  `<tool_call>`, `<tools>`, etc.), and exits before inference starts —
  useful for confirming a tool block tokenizes the way you expect.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, separating sampling choices from logit/model
  issues.
- `q36-server --trace` writes rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.

A GPU hang (most often the kernel watchdog firing) destroys the Vulkan
context; the backend latches this and refuses further GPU work rather than
returning uncomputed results — restart the process. See
[CONTRIBUTING.md](CONTRIBUTING.md#reporting-session-bugs) for the full
device-loss diagnosis flow.

## Logo

The QwarkStar logo is an AI-edited version of the DwarfStar logo, designed by
hand by Salvatore Sanfilippo, made more graphical with AI, and manually
reworked by Ben Gnomino. All credit to Salvatore.

## Testing, QA, and Regression Checks

Release testing (correctness/speed regression `make` targets, the release QA
checklist, test vectors, and extended `q36-eval` suites) lives in
[CONTRIBUTING.md](CONTRIBUTING.md) and [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) —
**read those before sending a pull request**.
