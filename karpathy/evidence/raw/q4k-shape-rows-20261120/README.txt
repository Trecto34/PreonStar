Arms: same binary, P36_VK_Q4K_DECODE_ROWS=1 (A, fixed 1-row) vs default auto (B, 4 rows when out_dim >= 4096 or (out_dim >= 2048 && in_dim <= 512), else 1 row).
Prompt: "Write a C function that reverses a singly linked list." -n 64 --ctx 1024 --nothink --temp 0.
Profile: P36_VK_PROF_KERNEL=1 P36_VK_PROF_SHAPE=1 P36_VK_PROF_SKIP=3000 (2 interleaved pairs: A1, B1, A2, B2).
Parity: --dump-logprobs (top-20 logits, 64 steps) byte-identical A vs B on Axion and Qwen3.6-35B-A3B (md5.txt).
A/B: P36_AB_GEN=64 tests/bench_ab.sh (5-rep interleaved on Axion-1.0-35B.APEX-I-NanoPlus-V2.1.gguf).
