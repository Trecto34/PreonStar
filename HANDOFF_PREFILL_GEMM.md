# Handoff: prefill GEMM tuning on the BC-250

You are picking up **prefill** optimization for the q36 Vulkan engine. Decode work is finished and
its cheap wins are exhausted; prefill is the larger and entirely unexplored half.

Everything below was measured on this box unless marked otherwise. **Read the Measurement section
before running anything** — the box has a thermal behaviour that will hand you convincing wrong
answers if you benchmark naively.

---

## 1. Where things stand

Branch `feat/vulkan-opt-phase1`, worktree `/home/server/q36-opt`. Pristine baseline for A/B is the
untouched `10c6965` build at `/home/server/q36`.

Shipping result so far, ctx 2048, interleaved and thermally gated:

| config | decode t/s | prefill t/s |
|---|---:|---:|
| baseline `10c6965` | 80.31 | 605.98 |
| branch default (bit-exact) | 82.15 (+2.3%) | 605.22 |
| branch + opt-ins | 86.11 (+7.2%) | **708.85 (+17.0%)** |

Opt-ins are `Q36_VK_F32_FAST_WIDE=1` and `Q36_VK_ATTN_SPAN=128`. Both change numerics and are off
by default pending an eval — see `OPTIMIZATION_LOG.md`.

**The +17% prefill was accidental.** It came from widening `matmul_f32_fast`, which serves prefill
as well as decode. Nobody was aiming at prefill, and it returned the largest number of the session.
That is the reason for this handoff.

---

## 2. Why prefill is the right target

For agentic workloads prefill *is* the user-visible latency. Task time is
`prompt / prefill_tps + output / decode_tps`:

- 2048-token prompt, 128 out: prefill is **68%** of wall clock
- 8192-token prompt, 200 out: prefill is **~91%** (31 s reading vs 3 s writing)

Decode t/s is the number people quote. Prefill is the number people wait on.

The four largest kernels in the entire profile are all prefill, and **none has been examined**:

| kernel | ms (ctx 2048, 128 gen) |
|---|---:|
| `matmul_q8_0_mm_f16` | **2173** |
| `moe_gate_up_gemm` | **1881** |
| `attn_prefill_qtile2` | 1108 |
| `moe_down_gemm` | 1092 |

For contrast, the biggest *decode* row is 191 ms.

---

## 3. What is already closed — do not redo these

**Prefill chunk size is optimal at 1024.** Measured 512 -> 435, 1024 -> 494, 2048 -> 178 t/s at
ctx 4096. Do not spend time here.

**The chunk-2048 cliff is diagnosed and not worth fixing.** It is a host readback of the MoE expert
selection (`submit_wait_moe_gate_up_selected`, 50.6 s, 170 flushes vs 16), triggered because
`n_slot = tokens * 8 = 16384` exceeds `q36_vk_moe_tiles_gpu()`'s hard 8192-slot bound. Two fixes
were tried and reverted: raising `Q36_VK_PRIVATE_LIMIT` (refuted by its own control) and raising
the call-site guards 4096 -> 8192 (bit-exact but inert). Even with the stall removed, chunk 2048's
*GPU* time is 12-15% worse than 1024, so it cannot win. Full detail in `OPTIMIZATION_LOG.md`.

**Register prefetch in `matmul_q8_0_mm_f16` was tried and measured flat** —
`matmul_q8_0_mm_f16.comp:70`: *"the extra VGPRs cost more occupancy than the hidden latency
bought."* This is a strong hint that the kernel is **occupancy-limited by register pressure**, which
should shape every idea you have about enlarging tiles.

**LDS/shared-memory KV tiling regressed** — `attn_decode_fused.comp:16`: a 33 KB tile cut occupancy
to one workgroup per CU.

---

## 4. The target, and the constraints that bind it

### `matmul_q8_0_mm_f16` (2173 ms)

```
BM = 128, BN = 64, BK = 32, KK = BK/4 = 8
local_size_x = 256      -> 128x64 output tile, 8x4 register tile per thread
shared f16vec4 As[BM][KK+1], Bs[BN][KK+1]; shared float Ds[BM]
LDS = 9216 + 4608 + 512 = 14336 bytes
```

Read the header comment in full before touching it. The important parts:

- The f32 kernel is **instruction/latency-bound, not bandwidth-bound**, on this chip. The f16
  variant exists to halve FMA count and double per-barrier work. So this is a **latency and
  occupancy problem, not a bandwidth problem** — the opposite of everything in decode.
- `acc[8][4]` is 32 VGPRs of accumulator alone, before staging registers. Enlarging the register
  tile costs occupancy directly, and prefetch already failed for exactly that reason.

### HARD CONSTRAINT: batch-size invariance

From the header: *"Per-output k order is fixed regardless of `n_tok`, keeping the batch-size
invariance rule (warm == cold)."*

This means a prompt processed in one chunk must produce identical results to the same prompt
processed across several chunks, and a warm KV-cache continuation must match a cold one. **Any tile
or loop restructuring that makes the k-accumulation order depend on `n_tok` breaks this**, and the
frontier-logits gate below will NOT catch it — that gate uses a fixed chunk size.

To test invariance explicitly, dump frontier logits at several `--prefill-chunk` values (256, 512,
1024) and compare across them, not just against baseline.

### Other traps found the hard way

- `moe_gate_up_decode` has `shared float sh_gate[2]` with `acc += sh_gate[1]`, hardcoded for
  exactly two subgroups. Widening past 128 threads **silently drops subgroups 2 and 3** — wrong
  results, not slow ones. Assume similar hidden assumptions elsewhere; grep for fixed-size shared
  arrays before changing any `local_size`.
- `glslc` in the repo root is a shim over `glslangValidator`. It **accepts and ignores `-O`** and
  always passes `-Os`. This is deliberate and documented — do not "fix" it.
- The pooled `groups/dispatch` column in the profiler averages prefill and decode for any kernel
  used in both. It is only trustworthy for `*_decode` shaders. This artifact produced two wrong
  diagnoses in the previous session.

---

## 5. Measurement — read this or your results will be wrong

**The box thermally drifts, badly.** Four consecutive runs of an *unmodified* binary lost 13% of
decode throughput (84.3 -> 73.0 t/s). GPU idle floor moved from 44 C at session start to ~60 C
after a few hours of benchmarking. A naive A-then-B comparison can manufacture or mask a 10%+
effect.

**Protocol that works:**

1. **Interleave** variants (A,B,A,B), never run blocks.
2. **Gate on entry temperature** before every run and record it:
   ```bash
   H=/sys/class/drm/card0/device/hwmon/hwmon1
   for i in $(seq 1 30); do t=$(( $(cat $H/temp1_input)/1000 )); [ "$t" -le 61 ] && break; sleep 5; done
   ```
   Pick a threshold that is actually reachable — check the current idle floor first. A 54 C gate
   timed out every time and wasted 5 minutes per run while still giving mismatched temperatures.
3. **Discard any pair whose entry temperatures differ by more than ~2 C.** One result this session
   looked like a 5% win purely because that run started 5 C cooler.
4. **Swap `.spv` files rather than rebuilding** where possible, so both arms run on a byte-identical
   binary. Shaders load from `vulkan/` at runtime.
5. Never run two model processes at once (`SPEEDUP.md` forbids it; the instance lock is intentional).

**Three times this session a harness produced a plausible but meaningless result:** `cmp` reporting
"identical" on two empty files, a thermal gate that never triggered, and `set -- $cfg` failing to
split under zsh so every row ran the same config. Always print the thing you think you varied.

---

## 6. Parity gate

```bash
./q36-bench -m gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf --vulkan \
  --prompt-file tests/long_context_story_prompt.txt --prefill-chunk 1024 \
  -ctk q8_0 -ctv q4_0 --ctx-start 2048 --ctx-max 2064 --step-incr 1 --gen-tokens 0 \
  --dump-frontier-logits-dir <DIR>
```

17 files. Frontier `002048` gates prefill; `002049`-`002064` are single-token forward passes and
gate decode. `cmp` each against a baseline captured from the unmodified binary.

The gate is validated: two captures from the same binary are byte-identical, so a diff means a real
numerical change. **For prefill work, also run the chunk-invariance check in §4.**

Note `--gpu-cpu-parity` is **not** a usable gate here: on a 35B it is a large CPU inference run
(7.8 GB RSS, 662% CPU) and `AGENT.md` warns that path has exposed kernel VM failures.
`--vulkan-fusion-parity` and `--vulkan-queue-parity` are cheap and do pass.

---

## 7. Suggested starting points, ranked

1. **Get real occupancy numbers first.** Everything about this kernel hinges on VGPR pressure vs
   waves in flight, and nobody has measured it. Try `RADV_DEBUG=shaderstats` (or the ACO stats env
   for your Mesa build) to get VGPR/SGPR/LDS per shader. **Do this before proposing any tile
   change** — the previous session's failures all came from reasoning instead of measuring.
2. **`BN` 64 -> 32 with the same `BM`.** Smaller B tile cuts LDS from 14336 to ~9728 bytes and may
   raise workgroups-per-CU. Cheap to test if the thread mapping tolerates it — verify the mapping
   is derived from BM/BN and not hardcoded.
3. **`moe_gate_up_gemm` (1881 ms) may be the softer target.** It carries `sh_grid[512]` plus two
   `[BM][KK+1]` staging arrays, so its LDS budget is larger than `matmul_q8_0_mm_f16`'s and its
   `BM = 64` is already half. It has had no attention at all, whereas `matmul_q8_0_mm_f16` has
   clearly been tuned.
4. **`attn_prefill_qtile2` (1108 ms).** Uses `subgroupClusteredAdd(d, 4u)` and two large shared
   arrays. Note the GQA-8 requirement — the dense 27B misses this kernel entirely and falls back.

Accept a change only if the median improves **and** the candidate's minimum beats the baseline's
maximum, or the improvement exceeds 3%.

---

## 8. Build

```bash
make q36-bench VULKAN_CFLAGS=-DQ36_VULKAN_REQUIRE_BC250 -j8
```

`make vulkan-bc250` runs `$(MAKE) -B all` and force-rebuilds all 100 shaders every time; the
incremental form above is equivalent after the first build.

Model: `gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf` (arch `qwen35moe`, 40 layers,
GQA 8, 256 experts / 8 used). **Do not use the `q36moe.gguf` symlink** — it was repointed at the
dense 27B.

---

## 9. One thing outside the code

The board's idle temperature drifted 44 C -> 60 C over a single session, and identical code loses
13% across four consecutive runs. That is larger than any optimization landed so far. If the cooling
on this box can be improved, it is worth more than the work described here and costs no code. It was
out of scope for the previous session; it may not be for yours.
