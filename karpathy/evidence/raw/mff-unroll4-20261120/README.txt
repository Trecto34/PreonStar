Arms: same binary, P36_SHADER_DIR=/tmp/mff_head (A, 08d8f0d) vs /home/server/q36 (B, matmul_f32_fast.comp 4x unroll).
Prompt: "Write a C function that reverses a singly linked list." -n 64 --ctx 1024 --nothink --temp 0.
Profile: P36_VK_PROF_KERNEL=1 P36_VK_PROF_SHAPE=1 P36_VK_PROF_SKIP=3000 (2 interleaved pairs: A1, B1, A2, B2).
Parity: --dump-logprobs (top-20 logits, 64 steps) byte-identical A vs B on Axion and Qwen3.6-35B-A3B (md5.txt).
A/B: P36_AB_GEN=64 tests/bench_ab.sh (3-rep interleaved on Axion-1.0-35B.APEX-I-NanoPlus-V2.1.gguf).
