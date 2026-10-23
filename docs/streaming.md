# Running models larger than memory

Metal and Vulkan SSD streaming keep non-routed weights resident and load
selected routed experts into a bounded backend cache. The planner reserves memory
for the configured context, prefill scratch, planned server sessions, and a staging
margin before assigning the remaining model budget to static weights and experts.
Explicit cache sizes are targets capped by the same budget. Creating a larger or
additional session reduces the expert cache when necessary; it fails cleanly if
static weights and the minimum working cache leave insufficient room. Startup
logs show the resolved context and model budgets.

```sh
./p36 --metal --ssd-streaming -p "Explain radix trees."
./p36 --vulkan --ssd-streaming --ssd-streaming-cache-experts 256 -p "Hello"
./p36 --metal --ssd-streaming --ssd-streaming-cache-experts 1GB -p "Hello"
```

`NGB` is a routed-expert byte budget. P36 converts it to the number of expert
slots that fit the release GGUF. A plain integer requests an expert-slot count;
both forms are capped to leave room for the configured contexts.
Non-routed weights, KV cache, activations, and graph scratch need additional
memory. Startup prints the resolved slot count and actual cache allocation.

A built-in or `P36_VK_STREAMING_EXPERT_HOTLIST` profile biases eviction
without reading expert weights at startup. It does not fill the cache with
arbitrary experts when no profile exists. Use `--ssd-streaming-cold` for an
empty cache, or `--ssd-streaming-preload-experts N` to request an explicit
weight preload. Metal stores mixed expert sizes in component-wise padded cache
slots, so the IQ2/Q2 and Q4 routed layers share one bounded cache and one
hotlist policy. Vulkan retains its original single-size-class behavior. The
`P36_VK_STREAMING_EXPERT_HOTLIST` and
`P36_VK_DISABLE_STREAMING_EXPERT_HOTLIST` names are retained for compatibility
and apply to both graph runtimes.

As in DS4, an explicit routed prefix can stay fully resident:

```sh
./p36 --ssd-streaming --ssd-streaming-full-layers 4 -p "Hello"
```

Full layers are charged at their actual byte size and the remaining budget
must still hold one layer of dynamic expert slots for prefill. The default is
zero because the dynamic-only cache is faster on the BC-250. Pass
`--ssd-streaming-full-layers 0` to disable an explicit setting.

## Metal SSD streaming

Metal uses the same bounded-cache design, backed by shared Metal buffers with
biased LRU eviction instead of macOS VM-paging hints. The automatic cache
size uses `recommendedMaxWorkingSetSize`.
