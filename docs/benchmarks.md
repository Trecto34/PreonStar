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
| `--backend` | `vulkan`, `metal`, `cpu` | platform default |

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

Measured on a BC-250 reference configuration (40 CUs unlocked, performance
governor, 500--2000 MHz, 85 C ceiling), greedy decoding, `--gen-tokens 128`,
Q8_0 K / Q4_0 V cache, one process at a time. Model: Qwen3.6-35B-A3B (35B total,
about 3B active), Q2 imatrix quant.

| Prompt | Prefill t/s | Decode t/s |
|---:|---:|---:|
| 2048 | 640 | 82 |
| 4096 | 597 | 80 |
| 8192 | 502 | 75 |
| 16384 | 373 | 65 |
| 24576 | 287 | 56 |
| 32768 | 244 | 51 |

Hardware: 16 GB GDDR6 unified memory, 256-bit bus (vendor spec, about 448 GB/s),
8-core Zen 2, Linux + RADV.

**These are reference points, not guarantees.** They were taken on one board in
one thermal environment, and the board is the variable that moves most: a stock
24-CU part, a different governor, or a warm case will not reproduce them. Refresh
with the harness above before quoting them anywhere.

Streaming from disk trades throughput for the ability to run a larger model on the
same memory; expect roughly an order of magnitude less prefill and decode:

| Prompt | Prefill t/s | Decode t/s |
|---:|---:|---:|
| 4096 | 45 | 15 |
| 8192 | 49 | 14 |

## Interpreting results

- Prefill falls with context because attention cost grows with sequence length; a
  flat curve means you are measuring something else.
- Decode falls more slowly and is dominated by memory bandwidth, not compute. Two
  devices with the same decode number are usually bandwidth-matched, not
  compute-matched.
- A change that helps prefill and hurts decode (or vice versa) is a tradeoff, not
  a win, unless the project's gate says otherwise.
