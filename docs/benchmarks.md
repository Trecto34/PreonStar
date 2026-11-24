# Benchmarks

How throughput is measured in this project, how to reproduce a number, and how
to add a row for your own hardware.

Two rules apply to everything below:

1. **A number without its configuration is noise.** Prefill and decode depend on
   context length, KV cache type, quantization, residency mode, and prefill chunk
   width. Every table row records all of them.
2. **A single run is not a measurement.** Run-to-run variance on a shared host is
   large -- 33% on a consumer APU is normal. Changes are accepted only on
   interleaved A/B runs, never on one-off numbers.

## What is measured

- **Prefill** -- prompt tokens per second, measured at context frontiers (2048,
  4096, 6144, ... up to the model's limit).
- **Decode** -- generated tokens per second at the same frontiers, after the
  prompt is resident.

Both are reported for a fixed token budget so rows are comparable across models
and machines.

## Harness

`p36-bench` drives one model process, slices a fixed prompt at each frontier, and
writes a CSV.

```sh
./p36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 \
  --gen-tokens 128 \
  -ctk q8_0 -ctv q4_0 \
  --csv speed-bench/bc250.csv
```

The bundled story prompt tokenizes to 34392 tokens, so 32768 is the last complete
2048-token frontier.

Flags that change what the numbers mean -- record them with the result:

| flag | meaning | default |
|---|---|---|
| `-ctk` / `-ctv` | KV cache type for keys / values: `f16`, `q8_0`, `q4_0` | resident GPU: `q8_0` / `q4_0`; CPU: `f16` |
| `--prefill-chunk` | GPU graph prefill width | auto (resident GPU resolves to 1024) |
| `--ssd-streaming` | stream experts from disk instead of full residency | off |
| `--think` / `--nothink` | thinking mode changes the generated token distribution | `--think` |
| `--backend` | `vulkan`, `cpu` | `vulkan` |

```sh
./p36-bench --help        # full list, including the measured cost of each tradeoff
```

Generation is always greedy, runs exactly `--gen-tokens` tokens, and skips EOS so
every row covers the same work.

Plot a CSV:

```sh
python3 speed-bench/plot_speed.py speed-bench/bc250.csv --title "BC-250 Q2 t/s"
```

## A/B protocol

Any change that claims a speedup must be measured against the binary it replaces,
alternating A and B so host drift hits both equally.

```sh
tests/bench_ab.sh ./p36-old ./p36-new gguf/<MODEL>.gguf 7 \
  > karpathy/evidence/raw/ab-<slug>.csv 2> ab-<slug>.summary
```

- **7 interleaved reps** minimum.
- **Gate: prefill median gain >= 1.5%** (`P36_AB_MIN_GAIN`) to keep a change.
- **Noise floors**: MoE prefill 3.4%, dense 27B prefill 0.7%, decode up to 9.4%.
  A result below the floor is not a result.

The rejection ledger in karpathy/AlreadyTried.md records ideas already measured
and dropped. Read it before benchmarking an idea that sounds new.

## Result table

Copy this block into a PR or a new file under `speed-bench/` and fill every
field. **Do not add a row with unknown hardware or an unstated KV cache type.**

```markdown
| Device | CPU / RAM | GPU CUs | Memory | Bus | Model | Params (active) | Quant | Residency | KV cache | Prompt | Prefill t/s | Decode t/s |
|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| <device> | <cpu> / <ram> | <n> | <gb> <type> | <bits> / <GB/s> | <model> | <total>B (<active>B) | <quant> | resident | q8_0 / q4_0 | 2048 | <n> | <n> |
```

### Reference results -- BC-250

One BC-250 (16 GB GDDR6 unified, 40 CUs, RADV), greedy decoding, Q8_0 K / Q4_0 V
cache, one process at a time. **The GPU governor must be running**
(`systemctl is-active cyan-skillfish-governor-smu`); stopped, the same build reads
about 20-28% slower and looks like a regression.

| Model | Context | Measurement | Result | Date / source |
|---|---:|---|---:|---|
| Sage-A1.5 NanoPlus-V2.1 (Qwen3.6-35B-A3B finetune) | 2048 | pp2048 / tg128 | **415.2 / 112.7 t/s** | 2026-11-24, `p36-bench`, 1 run |
| Inkling-2-Mini NanoPlus-V2.1 | 2048 | pp2048 / tg128 | **740.9 / 83.7 t/s** | 2026-11-24, `p36-bench`, 1 run |
| Inkling-2-Mini MiniPlus-V2.1 | 2048 | pp2048 / tg128 | **740.3 / 83.0 t/s** | 2026-11-24, `p36-bench`, 1 run |
| Axion-1.0-35B (Qwen3.6-35B-A3B finetune) | 1024 | decode, gen 64 | **111.78 t/s** (from ~103.9 at sprint start, bit-identical) | 2026-11-20, ROADMAP.md |
| Inkling-2-Mini NanoPlus-V2.1 | 1024 | pp1024 | 655.41 t/s | recorded run, `p36-bench` n=128 |
| | 1k / 2k | tg128 | 77.08 / 76.89 t/s | llama.cpp inkling2 fork: 47.06 t/s |
| Inkling-2-Mini MiniPlus-V2.1 | 1024 | pp1024 | 634.28 t/s | recorded run, `p36-bench` n=128 |
| | 1k / 2k | tg128 | 75.42 / 75.09 t/s | |
| | 8192 | 6,495-token needle | 582.61 pp / 66.53 tg t/s | |

Inkling improvements landed after those runs and were not re-tabulated: MoE expert
re-layout at upload, +4.7% MiniPlus decode, bit-identical (5c08a39); long-context
attention prefill +4% (MiniPlus 16k->32k prefill 516.9 -> 542.0 t/s,
bit-identical, 18648e8).

Older Qwen3.6-35B-A3B Q2 imatrix sweep (`speed-bench/bc250.csv`, `--gen-tokens 128`,
prefill / decode t/s): 2048 640 / 82, 8192 502 / 75, 32768 244 / 51.

### Rebuild before you measure

`make` links six binaries from shared objects, so a route change compiled into
the common objects still leaves stale executables behind. Always rebuild the
whole set and check the clock:

```sh
make -j"$(nproc)"
ls -l p36 p36-server p36-agent
```

A stale `p36-server` does not error. It keeps the old route and reads about 30%
slow, which is indistinguishable from a real regression.

**These are reference points, not guarantees.** They were taken on one board in
one thermal environment, and the board is the variable that moves most: a stock
24-CU part, a different governor, or a warm case will not reproduce them. Refresh
with the harness above before quoting them anywhere.

Streaming from disk trades throughput for the ability to run a larger model on the
same memory; expect roughly an order of magnitude less prefill and decode.

## Interpreting results

- Prefill falls with context because attention cost grows with sequence length; a
  flat curve means you are measuring something else.
- Decode falls more slowly and is dominated by memory bandwidth, not compute. Two
  devices with the same decode number are usually bandwidth-matched, not
  compute-matched.
- A change that helps prefill and hurts decode (or vice versa) is a tradeoff, not
  a win, unless the project's gate says otherwise.
