# Which tensors p36 re-faults every token — 2026-10-23

**Verdict: the re-read working set is the routed-expert down-projection, touched
sparsely. Decode is limited by page-cache capacity (MemFree 170 MB), not by
kernel design.** Model: `MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf`.

## Method

`/proc/<pid>/pagemap` present-bit sampling over the whole model mapping, once a
second, while decoding (`-n 4000`). Pages that flip present→absent→present
between samples are the re-faulted set; their file offsets are then binned
against the GGUF tensor table. Tool: `karpathy/tools/pagemap_probe.py`,
`karpathy/tools/tensor_thrash.py`.

## Residency collapses while decoding

```
resident pages per sample: 1513694 1359461 1538454 986652 1032381 570864 569265 471895
                           of 3064917 pages (11.69 GiB)
evicted between samples: 2426608 pages      refaulted: 1384820 pages
```

The cache holds ~6.0 GiB at the start and is down to **1.8 GiB after 10 s**. The
kernel is evicting clean file pages because there is nowhere else to go:

```
MemFree 170 MB · used 11173 MB · buff/cache 4145 MB · Inactive(file) 3912888 kB
p36 VmRSS 4147156 kB · zram 1.4 GiB in use
```

So ~4 GB of an 11.7 GB model is cached, and every token re-reads whatever slice
it needs and loses. This is why both engines sit at the NVMe rate.

## What is being re-read

Top refaulted tensors (9 s of decode, pages × 4 KiB):

| MiB | pages | tensor |
|---:|---:|---|
| 108.1 | 27678 | `blk.38.ffn_down_exps.weight` |
| 107.0 | 27400 | `blk.39.ffn_down_exps.weight` |
| 98.0 | 25088 | `blk.21/27/28/29/30/31/35/36/37.ffn_down_exps.weight` |

Every single hot tensor is `ffn_down_exps` — the routed-expert down projection.
No attention, no norms, no embeddings. ~98 MiB per layer across 9 s ≈ 0.16
MiB/token/layer, ≈ 6.5 MiB/token over 41 layers, consistent with the measured
10.2 MiB/token total.

The access is **sparse**: 8 of 256 experts per layer per token. Each touch is a
cold 4 KiB fault scattered across a 98 MiB tensor, so there is no readahead to
exploit and no reuse before eviction.

`output.weight` is the most *evicted* tensor (397 MiB) but does not appear in the
refault list — touched once per token, then dropped. Not a problem.

## Why prefetching is not trivial here

The tile schedule is built **on the GPU** from the router selection precisely to
avoid a host readback (`p36_vk_moe_tiles_gpu`: "GPU tile schedule straight from the
router selection, no host read"). So the host does not know which 8 experts a
layer picked, and cannot `posix_fadvise(WILLNEED)` them without reintroducing the
stall the design removed. Options, in order of expected value:

1. **Free RAM for page cache.** p36 RSS is 4.1 GB on an 11.7 GB model; anything
   that drops the resident copy frees directly into `Inactive(file)`. Cheapest
   experiment: check whether any weight upload path duplicates data that is also
   read via the map.
2. **Prefetch with a bounded readback.** Read the selection for layer L, fadvise
   the 8 expert ranges for L+1 while L computes. Costs one sync per layer; the
   sync already exists in `p36_gpu_router_topk_tensor` for other callers.
3. **Expert-major packing at quantize time** so co-selected experts land
   contiguously — real fix, but needs a new model layout.

## Negative result: don't use --ssd-streaming here

`--ssd-streaming --ssd-streaming-cache-experts 8GB`: **3.18 t/s decode** vs 71.26
baseline (cache 796/6853 experts, 97.8% hit rate, 796 loads). The explicit cache
path is 22× slower on this model. Do not retry without a new streaming design.
