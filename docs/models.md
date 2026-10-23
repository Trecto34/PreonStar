# Models and weights

This implementation works with Qwen3.6-35B-A3B and dense Qwen3.8-27B GGUFs.
It is not a general GGUF loader: arbitrary GGUF files will not have the
tensor layout, quantization mix, metadata, or optional MTP state the engine
expects. The 2-bit quants provided here are verified high quality — they
behave well and call tools reliably under coding agents.

Start dense Qwen3.8-27B explicitly:

```sh
./p36-server \
  -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vulkan
```

## Vision

P36 accepts the matching Qwen3-VL projector as a disk-backed sidecar. It maps
the file without copying the model and streams one weight tensor at a time
through a reusable Vulkan buffer. The full 0.9 GiB projector is never retained
in RAM or VRAM, and streamed pages are released after each dispatch.

```sh
./p36 --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf
./p36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vision gguf/Qwen3.8-27B-mmproj-F16.gguf
```

At the interactive prompt, `/read photo.jpg` and `/read image.png` submit an
image turn, optionally with a prompt (`/read photo.jpg Describe this image`).
JPEG/PNG decoding is built in. Vision prompt execution currently requires
Vulkan. `p36-agent --vision FILE` exposes `view_image` for local files;
`p36-server --vision FILE` accepts OpenAI image data URIs, Responses image
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
./p36 -m gguf/Qwen3.6-35B-A3B-Layers34-39Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-Q8Rest-imatrix.gguf \
  --ssd-streaming -p "Hello"
```

The script downloads Qwen3.6 models from
`https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf` and dense Qwen3.8-27B
from `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`. It stores files under
`./gguf/`, resumes partial downloads with `curl -C -`, and updates
`./p36moe.gguf` to point at the selected model for older scripts.

For the reasoning-efficient Swift derivative, the practical tier for this
16 GB BC-250 is IQ3_XXS, from
`ukisai/Swift-Qwen3.8-27B-GGUF`:

```sh
./download_model.sh swift-iq3
./run_swift_iq3_xxs.sh -p "Explain why the sky is blue in two sentences." --nothink
```

`run_swift.sh` uses a 4096-token default context and binds the embedded MTP
draft head, but defaults to one draft position because the BC-250 benchmark
was faster without speculative verification. Set `P36_SWIFT_MTP_DRAFT=3` to
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
supported GGUF from `./gguf/`. Run `./p36 --help` and
`./p36-server --help` for the full flag list.

If you want to regenerate GGUF files, quantize community fine-tunes / abliterated models
(via `--strip-nextn`), or collect a new imatrix, see
[gguf-tools/README.md](../gguf-tools/README.md). Those tools are meant for offline
Qwen3.6 model-building work. The native quantizer accepts Q8, F16, or BF16
inputs; imatrix collection still uses the optional llama.cpp tooling.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF for Qwen 3.6 MoE. It can be used with the `q2-imatrix` and
`q2-q4-imatrix` main models, but must be enabled explicitly with `--mtp`. The
current MTP/speculative decoding path is still experimental: it is
correctness-gated and currently provides at most a slight speedup, not a
meaningful generation-speed win.
