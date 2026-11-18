Arms: A = HEAD bb435e6 shaders, B = fp64 hoist. Same p36/p36-bench binary,
shaders picked with P36_SHADER_DIR. Prompt: "Write a C function that reverses
a singly linked list." -n 64 --ctx 1024 --nothink --temp 0.
Profile: P36_VK_PROF_KERNEL=1 P36_VK_PROF_SKIP=3000 P36_VK_PROF_TIMELINE=tl-*.txt
tl-A6/A8: HEAD.  tl-B7/B9: fp64 hoist (committed).
tl-B2: rejected LDS-row + w-prefetch variant (neutral, 8.14 -> 8.06 us).
A/B: P36_AB_GEN=64 tests/bench_ab.sh A/p36-bench ./p36-bench Axion 3
