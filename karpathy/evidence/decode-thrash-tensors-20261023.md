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

## Falsification, same day: page-cache residency is NOT causal

Dropping the **entire** model page cache mid-decode (external `posix_fadvise
(POSIX_FADV_DONTNEED)`, no root needed) changes decode by nothing. Three
interleaved pairs, drop vs keep:

```
drop=73.34 t/s  keep=73.65
drop=73.76 t/s  keep=73.82
drop=73.49 t/s  keep=73.53
```

The residency collapse above is real, but it is a *symptom*: the engine imports
weight ranges through `VK_EXT_external_memory_host` (AGENT.md: "no copy, no
staging-buffer path"), so the pages it needs must be resident for the GPU to read
them, and they are fetched either way. Making them stay cached buys nothing.

Consequences for the three options in the previous section:

1. **Free RAM for page cache — dead.** p36 holds 6 MB of anonymous memory
   (`smaps_rollup`); its 4.1-5.9 GB RSS is the model's own file pages. There is no
   duplicate copy to release, by design.
2. **Prefetch the next layer's experts — dead.** The device tops out at
   **776 MiB/s sequential** and p36 already streams 730 MiB/s during decode, i.e.
   94% of the ceiling. Prefetch cannot add bandwidth, and cache state does not
   move the number.
3. **Expert-major packing — no upside left.** Packing changes layout, not bytes
   read, and bytes read is not what gates the run.

What the measurements leave: decode is gated by something that is neither ALU
(three kernel A/Bs neutral), nor page-cache state (this test), nor CPU saturation
(1.5 of 16 cores). The next measurement must be a GPU-side per-kernel timing
breakdown, which this engine does not currently produce (`gpu_ms=0.000` on every
op even with `P36_VK_PROF_OP=1`).
