# Decode is disk-read-bound on this host — both engines, 2026-10-23

**Verdict: the APEX-I decode gap is bytes-per-token, not arithmetic.** p36 is not
slow because of a kernel; it pulls ~30% more bytes per token than llama.cpp, and
both engines are limited by the same NVMe stream.

**Raw:** `raw/io-llamacpp-nanoplus.log`, `raw/io-p36-nanoplus.log`,
`raw/io-p36-huihui.log`

## Method

Steady-state decode only. Start the engine, let it load, then sample
`/proc/<pid>/io: read_bytes` every 2 s while it generates. `read_bytes` counts
bytes that actually came off the block device — page-cache hits do not appear.

Model: `MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf` (11.69 GiB,
qwen35moe, 41 blocks, 256 experts / 8 used).
Control: `Huihui-Qwen3.6-35B-A3B-...-IQ2XXS.gguf` (10.93 GiB).

## llama.cpp

Build `839a572` (2026-10-20). Note: origin is 67 commits ahead (`06230cb`,
2026-10-23); disk behaviour is not version-sensitive, so this build was used as is.

```
llama-bench -m NanoPlus -p 0 -n 1024 -t 16 -r 1
t=4s  481 MiB/s   t=6s 743   t=8s 783   t=10s 727   t=12s 785
t=14s 782 MiB/s   t=16s 760  t=18s 717  t=20s 183 (draining)
tg1024: 96.62 t/s
```

## p36

```
NanoPlus, 400 tokens: read_bytes 2980 MiB / 4 s  = 745 MiB/s   gen 72.15 t/s
Huihui,   400 tokens: read_bytes 3048 MiB / 4 s  = 762 MiB/s   gen 86.42 t/s
```

`P36_VK_PROF_OP=1` shows **zero** `*_stream` dispatches, so this is not the SSD
streaming path: it is the OS evicting pages out of the weight **mmap**. The only
`mlock` in the tree is gated behind streaming (`src/core/core_09_weights_bind.inc:892`);
plain mmap mode pins nothing.

## The arithmetic that matters

| engine | decode | disk rate | bytes/token |
|---|---:|---:|---:|
| llama.cpp | 96.6 t/s | ~760 MiB/s | **7.9 MiB** |
| p36 | 71.4 t/s | ~745 MiB/s | **10.3 MiB** |

Speed ratio 96.6/71.4 = **1.35**; bytes ratio 10.3/7.9 = **1.30**. Decode time is
`bytes_per_token / disk_rate` to within measurement error — i.e. decode is
**100% disk-bound on this box**, and the engine that reads less wins.

Corollary: any change to the MoE matvec ALU path is invisible here. Two such
changes were built and measured neutral the same day (see
`ab-moe-type-specialization-20261023.md`).

## Consequences

1. Do not rank decode experiments on this host without an I/O control — a
   kernel win will read as noise.
2. Real levers, in order: fewer bytes per token (residency/locality), then pinning
   the mapping so decode becomes compute-bound.
3. `RLIMIT_MEMLOCK` is **8 MiB** here. Any `mlock` of an 11 GiB model fails until
   that is raised (privileged `ulimit -l` or a unit with `LimitMEMLOCK=infinity`).
4. Upstream `q36` refuses these files outright
   (`tensor token_embd.weight has dim[1]=248144, expected 248320`), so there is no
   upstream baseline for the APEX set — recorded for the compatibility gate only.
