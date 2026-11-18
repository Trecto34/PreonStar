Arms: H = HEAD f0d857d (binary + shaders, P36_SHADER_DIR), Q1 = this change with
P36_VK_MOE_DOWN_ROWS=1 (also puts IQ3_XXS down back to 1 row, so its span is not
comparable), Q4 = this change, default.
Prompt "Write a C function that reverses a singly linked list." -n 64 --ctx 1024
--nothink --temp 0. Profile: P36_VK_PROF_KERNEL=1 P36_VK_PROF_SKIP=3000.
Parity: --dump-logprobs (top-20 logits, 64 steps) byte-identical H vs Q4 on
Axion and Ornith NanoPlus; md5.txt. q3k_equiv.py: old vs new byte extraction on
random blocks, all 16 lanes (python3 q3k_equiv.py).
A/B: P36_AB_GEN=64 tests/bench_ab.sh H/p36-bench ./p36-bench Axion 3.
