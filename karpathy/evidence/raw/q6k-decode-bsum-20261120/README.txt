Rejected experiment: specialized dense_q6k_decode.spv (P36_Q6K_ONLY, q8_K bsums, compact uint qword[TOK][4], wh[] loads).
Arms: same binary, P36_VK_DENSE_Q6K_DECODE=0 (A, generic dense_kquant_decode.spv) vs 1 (B).
Kernel (2x interleaved 63-tok windows on Axion): 85.180/85.040 ms -> 85.215/85.308 ms (flat).
A/B (5-rep interleaved, ctx 1024, gen 64): decode 108.39 -> 108.76 t/s (+0.34%, overlapping).
Parity: --dump-logprobs byte-identical on Axion and Qwen3.6 (md5.txt). Reverted.
