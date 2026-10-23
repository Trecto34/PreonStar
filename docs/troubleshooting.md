# Troubleshooting

Failure modes, debug switches, and the output-steering hook -- the things you
reach for when a run misbehaves rather than when it is merely slow. For speed
problems start with [benchmarks.md](benchmarks.md); for device detection start
with [backends.md](backends.md).

## Steering

P36 can edit attention and FFN outputs with the same projection used by DS4:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

The steering file is a flat `40 x 2048` native-endian `f32` matrix with one
normalized direction per Qwen layer. Positive scales remove the represented
direction; negative scales amplify it. FFN steering defaults to `1` when a
file is supplied without an explicit scale:

```sh
./p36 -p "Write tersely" \
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
./p36 --dump-tokens -p "..."
./p36 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./p36-server --trace /tmp/p36-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p`/`--prompt-file` string exactly as
  written, recognizes Qwen protocol specials (`<|im_start|>`, `<think>`,
  `<tool_call>`, `<tools>`, etc.), and exits before inference starts —
  useful for confirming a tool block tokenizes the way you expect.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, separating sampling choices from logit/model
  issues.
- `p36-server --trace` writes rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.

A GPU hang (most often the kernel watchdog firing) destroys the Vulkan
context; the backend latches this and refuses further GPU work rather than
returning uncomputed results — restart the process. See
[CONTRIBUTING.md](../CONTRIBUTING.md#reporting-session-bugs) for the full
device-loss diagnosis flow.
