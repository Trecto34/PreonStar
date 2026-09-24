# R2 — IQ2_S down sum-decode wide-load rewrite: **NEGATIVE** (bit-exact, zero gain)

2026-09-25, round-2 R2.  Model `gguf/Qwen3.8-35B-A3B-IQ2_M.gguf`.  Reopens round-1
**P1** (`## MoE IQ2_S down sum-decode — audited levers NEGATIVE, load shape CONFIRMED
as the lever, not built`), whose `reconsider_if` is "someone builds the coalesced
fetch and it measures".

## What was built

`vulkan/moe_down_q2k_sum_decode.comp`, `Q36_MOE_IQ2S` branch only.  The per-element
`wb_u8()` reads (33 `buffer_load_ushort` in the hot basic block, each lane a
different 2-byte address) are replaced by a staged LDS copy of the 82-byte
superblock: the 16 lanes of one `ix` half fetch its 41 words in three consecutive
rounds (`k = itid; k < 41; k += 16`), so each round covers one contiguous 32-byte
span and only lanes 0..8 are active in the third; `barrier()` around the refill; all
field reads (`d`, `scb`, `entry`, `signs`) then go through LDS.  Element slotting,
per-element fma order, the serial expert loop, `subgroupAdd(acc)` and the `tid == 0`
weighted `fma32` combine are untouched.  Diff: `wideload-rejected.patch`.

## ISA, before and after (`isa-old.asm.txt`, `isa-wideload.asm.txt`)

| | shipped | wide-load |
|---|---|---|
| `buffer_load_ushort` (the field reads) | 33 | **3** (staging only) |
| total `buffer_load_*` per workgroup-iteration | 52 | 22 |
| `ds_read_b32` / `ds_write_b32` | 0 / 0 | 33 / 3 |
| VALU in the whole shader | 478 | 476 |
| hot basic block, instructions | 549 | 529 |
| LDS per workgroup | 0 B (no `shared_size`) | 384 B (`shared_size: 384`) |

The global-load instruction count really did drop ~2.4x and the field reads really
did move to LDS.  It just does not matter — see below.

## Measurements

Kernel (`Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1`, ctx 512,
chunk 256, gen 128), same 4736 dispatches / 9699328 groups in both arms:

| arm | `gpu_ms` | share of IQ2_M decode |
|---|---|---|
| shipped | **316.383** | 21.5% |
| wide-load | **316.571** | 21.6% |

End to end (`tests/bench_ab.sh`, 7 interleaved reps, same binary, only
`vulkan/moe_down_q2k_sum_decode_iq2s.spv` swapped between runs, ctx 1024, chunk 256,
gen 128, IQ2_M; plain decode, no MTP):

| arm | prefill med (MAD) | decode med (MAD) |
|---|---|---|
| shipped | 617.06 (8.30) | **82.15 (0.150)** |
| wide-load | 615.01 (7.36) | **82.32 (0.290)** |

decode **+0.21%** with MADs that overlap, prefill **-0.33%** (the kernel is
decode-only: `identity = n_tok == 1u` in `q36_vulkan.c:9503`, so prefill should not
move and does not).  Verdict: **no effect**, reject.

Parity was clean: frontier dumps 512/520 on both IQ2_M and the guard, 2 reps per arm,
`max_abs_diff = 0`, top-1 same, top64 64/64 in every pairing; shipped-vs-shipped
re-run also 0 (`parity/cmp.txt`).  So the rewrite is bit-exact and simply useless.

## Why it is useless — the P1 attribution was wrong

P1 decomposed the kernel as `315.4 ms = 97.0 coalesced-load floor + 167.2 scattered
field loads + 51.2 ALU` and concluded that the 167.2 ms component was the lever.
Removing that component entirely (moved to LDS) changes the kernel by 0.06%.  The
component was miscounted: it is not the *loads* that stall, they were never the
critical path.

The kernel is **instruction-issue bound**:

* No overfetch to fix.  The loop already only moves 99.4 MB/token of unique bytes
  (12.72 GB per 128-token run) in 316 ms = **40.6 GB/s**, and the staged version
  moves the same bytes at the same rate.  The 131 GB/s floor is real (P1 probe D) but
  a load-only kernel has no dependent consumers, so it does not follow that the
  shipped kernel is waiting on memory.
* Instructions per dispatch: the hot basic block is 549 instructions and executes
  once per block *pair* (its two `ix` halves cover 512 weights), i.e. once per expert
  per row = 2048 rows x 8 experts = 16384 times per dispatch -> **9.0M
  warp-instructions** for 8.39M weights = **1.07 warp-instructions
  per weight**.  Against 40 CUs x 4 issue slots at ~1.5 GHz that is ~37.5 us of pure
  issue per dispatch against 66.8 us measured, i.e. ~56% of peak issue.
* Only ~89 of those 549 instructions are arithmetic (`v_cvt_f32_u32` 24,
  `v_mul_f32` 16, `v_mac_f32` 16, `v_add_f32` 8, `v_cndmask_b32` 16, `v_cmp_ne_i32`
  16, `v_fma_mix_f32` 1); ~348 are index/address math (`v_bfe_u32` 72,
  `v_and_b32` 72, `v_lshlrev_b32` 64, `v_add_nc_u32` 85, `v_lshl_add_u32` 25,
  `v_lshrrev_b32` 21, `v_mad_i32_i24` 8) for the 16 weights one lane owns per block.
  **34 instructions per weight per lane, 63% of the loop index arithmetic.**
  Full mix: `opmix-main-loop.txt`.

## What would actually move it

Not load width.  Either fewer instructions per weight (a mapping where one lane owns
a contiguous 32 bytes = 64 weights of a block instead of 16 interleaved ones, so the
`entry`/`signs`/`scale` index math amortizes 4x), or the P4-style integer inner
product, or more weights per lane via a wider row tile.  All of those are new work
with a real chance of the same surprise, so they are not attempted here.

## Not built

The same load restructure for the Q2_K branch (the guard's `moe_q2k_down_sum_decode`,
62 GB/s) and for `moe_iq2s_gate_up_decode` (~150 GB/s, which is *not* a sum-decode
kernel: it is already 3.7x faster per byte and its shape is different).  The premise
is refuted on the kernel where it was strongest, so porting it would be paying the
same cost for the same nothing.

## Gates

`./karpathy/compat_gate.sh` **PASS**; `./q36_test --vulkan-kernels` **OK**
(the R1 fix, commit `e77871d`).  Shipped blob restored byte-identically
(`md5 2603136d…`); the variant lives only in this directory.

`reconsider_if`: a variant that changes the **instruction count per weight** (index
arithmetic or the dequant extraction), not the load width; a mapping that gives one
lane 64 contiguous weights; or evidence that occupancy rather than issue is the
limiter (then the fix is more rows or waves per workgroup, not fewer loads).
