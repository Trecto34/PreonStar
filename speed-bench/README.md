## Benchmarking

Prefill and generation speed on the BC-250 (GPU governor `cyan-skillfish-governor-smu` running).
`bc250.csv` is the older Qwen3.6-35B-A3B Q2 imatrix sweep; current Axion and
Inkling-2-Mini results are in [docs/benchmarks.md](../docs/benchmarks.md).

Run `p36-bench` as:

```
./p36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  -ctk q8_0 -ctv q4_0 \
  --csv speed-bench/bc250.csv
```

The current tokenizer produces 34392 tokens from the story fixture, making
32768 the last complete 2048-token frontier.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/bc250.csv --title "BC-250 Q2 t/s"
```
