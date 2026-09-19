# Handoff — PTQ1_0 (type 143) Track B / Phase B2

**Status: B2.1–B2.7 complete and verified on the shipped model.** All phases of
the B2 plan below are done. STOP until directed further (B2.7 says "then
STOP"; Phase B3 is not in this plan).

Branch `trackB-ptq1_0`, base `e9f794b`. The B2 work is **not committed**
(uncommitted: `Makefile`, `q36.c`, `q36_vulkan.c`, `q36_mmq_contract.h`,
`tests/q36_test.c`, `vulkan/dense_extra_decode_ptq1_0.comp`,
`vulkan/dense_extra_mmq_ptq1_0.comp`). Do not commit unless told to.

## Evidence files (this dir; mirrors the PQ2_0 numbering)

| File | Phase | Result |
|---|---|---|
| `00-env.txt` | env | BC-250 / RADV GFX1013, subgroup 64, glslc 2026.3 |
| `01-B2_4-dense-quant-model-rows.txt` | B2.4 | decode (n_tok 1) max_abs ≤ 9.7e-8 vs f32, tol 1e-5; MMQ rows also present |
| `02-B2_5-n_tok-1248-mmq.txt` | B2.5 | GPU MMQ vs `q36_contract_mmq_ptq1_dot` exact to 1 f16ulp (max_abs ~1e-7, tol 4.9e-4), tokens 2/4/8, 3 tensors (5120/17408/6144), 9 cases |
| `03-B2_6-shader-trace-and-kernel-profile.txt` | B2.6 | both new `.spv` build+dispatch; no 142/42 sibling; no CPU fallback; no NaN/Inf/reset; clean generation |
| `04-B2_7-baseline-matrix.txt` | B2.7 | resource + timing matrix vs PQ2_0 |

`--ptq1-0-layout-oracle` additionally passes: **28 probes x 8 rows (decode+mmq),
240 checks, 0 failures, worst abs err 0** (B2.2 with one-hot activations and
distinct signed per-128 scales).

## Ground truth (fetched, not guessed)

Publisher fork `github.com/PrismML-Eng/llama.cpp`, branch `prism-v7`. Two files
were cached to `/tmp/opencode/ggml-quants-prism.c` and
`/tmp/opencode/ggml-common-prism.h` (re-fetch if gone):

    curl -sL https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism-v7/ggml/src/ggml-quants.c -o /tmp/opencode/ggml-quants-prism.c
    curl -sL https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism-v7/ggml/src/ggml-common.h  -o /tmp/opencode/ggml-common-prism.h

`ggml-quants.c:2198-2285` is the canonical encode/decode
(`quantize_row_ptq1_0_ref`, `dequantize_row_ptq1_0`).

### Block geometry — 28 B / 128 weights, scale LAST

    typedef struct {
        uint8_t  qs[24];   // 5 trits/byte -> 120 values
        uint8_t  qh[ 2];   // 4 trits/byte ->   8 values
        uint16_t d;        // f16 scale @ bytes 26..27  (NOT d-first like PQ2_0)
    } q36_block_ptq1_0;    // QK_PTQ1_0 = 128, sizeof = 28

### Position -> packed source mapping (p in 0..127, verified exhaustively)

    p in   0.. 79 : qs byte = p % 16,          trit n = p / 16
    p in  80..119 : l=p-80,  qs byte = 16+(l%8), trit n = l/8
    p in 120..127 : l=p-120, qh byte = 24+(l%2), trit n = l/2

Per-trit decode (the u8/256 truncation is REQUIRED):

    uint qq = (uint(byte) * pow3[n]) & 0xffu;   // pow3 = {1,3,9,27,81}
    uint xi = (qq * 3u) >> 8u;                  // 0..2
    value  = (float(xi) - 1.0) * d;

## Integration point map (current line numbers)

| Site | File:line | What |
|---|---|---|
| byte constants | `q36.c:63-64` | `Q36_QK_PTQ1_0 128`, `Q36_PTQ1_0_BYTES 28` |
| type table | `q36.c:292` | `[143] = {"ptq1_0", 128, 28}` |
| tensor enum | `q36.c:317` | `Q36_TENSOR_PTQ1_0 = 143` |
| block struct | `q36.c:385` | `q36_block_ptq1_0` (28 B, d last) |
| dequant fwd-decl | `q36.c:1937` | `q36_dequantize_row_ptq1_0` |
| dequant dispatch | `q36.c:1997-1998` | case 143 |
| dequant impl | `q36.c:2100` | canonical loop, u8 truncation preserved |
| q8k-dot support | `q36.c:2595` | case 143 |
| cpu-matrix support | `q36.c:4379` | case 143 |
| quant-bits | `q36.c:4481` | 143 -> 2 |
| q8_scaled dispatch | `q36.c:6346` | case 143 -> iq path |
| VK tensor enum | `q36_vulkan.c:42` | `Q36_VK_TENSOR_PTQ1_0 = 143` |
| kernel handles | `q36_vulkan.c:247,251` | decode + mmq |
| kernel ctor | `q36_vulkan.c:3268,3272` | `4, 20` and `4, 24` bindings, `1u<<2` |
| block bytes/256 | `q36_vulkan.c:7786` | `case Q36_VK_TENSOR_PTQ1_0: return 56;` |
| host guard | `q36_vulkan.c:8203-8282` | `extra_type |= 143`; require int-dot + subgroup64 |
| decode dispatch | `q36_vulkan.c:8250-8282` | select 143 kernel |
| MMQ dispatch | `q36_vulkan.c:8302-8322` | select 143 kernel; grid x=(out_dim+3)/4, y=n_tok |
| shader build | `Makefile:150-151, 233-236` | two SPIR-V targets |
| contract | `q36_mmq_contract.h:213` `q36_contract_q2_geometry` (exists) / `:237` `q36_contract_mmq_ptq1_dot` |
| test type name | `tests/q36_test.c:3556` | `[143] = "PTQ1_0"` |
| layout oracle | `tests/q36_test.c:3244` | `test_ptq1_0_layout_oracle` |
| test registration | `tests/q36_test.c:8216` | `--ptq1-0-layout-oracle` |

The generic `dense_extra_decode`/`_mmq` shaders do NOT know 143 (their SPIR-V is
untouched). Two new dedicated files were added:

    vulkan/dense_extra_decode_ptq1_0.comp
    vulkan/dense_extra_mmq_ptq1_0.comp

## Deviations / notes (read before touching)

1. **MMQ reduction is LDS, not `subgroupAdd`.** `vulkan/dense_extra_mmq_ptq1_0.comp`
   originally used `int total = subgroupAdd(s);`. It passed the one-hot layout
   oracle yet produced wrong results on dense (all-lane-nonzero) activations
   on this device (reproduced as ~0.33 absolute error on real weights, exact
   same shader output every dispatch). Replaced with a 64-lane shared-memory
   tree reduce (`shared int sh_sum[64]`) across the local_size-64 workgroup.
   That made dense-quant MMQ bit-tight (see 02). Occupancy stays 40 (LDS 512 B).
   The decode shader keeps `subgroupAdd` and is correct.
2. **The PTQ1_0 MMQ contract is integer/f32, not f16-staged.**
   `q36_contract_mmq_ptq1_dot` mirrors the kernel exactly:
   `acc += wd * yd * float(sum over 128 of (trit-1)*int8(q8))` with an f32 fma
   on the final product. It agrees with a plain f32 dot to ~1e-7 (its f32
   rounding walk) and with the GPU to 1 f16ulp (02).
3. **Harness token counts moved 1/2/5/8 -> 1/2/4/8** in `tests/q36_test.c`
   `test_dense_quant_model_rows` per the B2.5 instruction.
4. The dense-quant diagnostic `f16-contract vs f32 reference` label is printed
   for PTQ1_0 as well; for PTQ1_0 it actually compares the *integer contract*
   against the f32 dot (~1e-7). Cosmetic mislabel only (the diagnostic is not
   pass/fail; the real gate is GPU-vs-contract at `mmq_tol_f16`).
5. `--ptq1-0-native-128` was NOT added. It is not in the B2 phased plan, and
   the layout oracle already falsifies per-128 scale independence using
   distinct signed scales per block and one-hot activations.
6. MMQ grid: x = (out_dim+3)/4 (ROWS=4), y = n_tok (one workgroup per token),
   versus PQ2_0's x=(out_dim+63)/64, y=(n_tok+127)/128. This was required --
   the first MMQ attempt used the PQ2_0 grid shape and failed.

## Phased plan status

| Phase | Status | Evidence |
|---|---|---|
| B2.1 plumbing | DONE | build ok; model loads 402 type-143 tensors |
| B2.2 independent layout oracle | DONE | 240 checks, 0 failures, worst err 0 |
| B2.3 decode shader | DONE | dedicated .comp; no generic-kernel changes |
| B2.4 B=1 decode validation | DONE | 01 (max_abs 6e-8..9.7e-8, tol 1e-5) |
| B2.5 MMQ/prefill n_tok 1/2/4/8 | DONE | 02 (1 f16ulp, 9 cases) |
| B2.6 smoke trace | DONE | 03 (no NaN/Inf/reset; both .spv dispatch) |
| B2.7 baseline matrix, then STOP | DONE | 04 (updated with controlled sweep), 05, 06 |

## Commands

    make -j"$(nproc)"                          # builds shaders + q36_test
    flock -w 900 /tmp/q36-gpu.lock ./q36_test --ptq1-0-layout-oracle
    flock -w 2400 /tmp/q36-gpu.lock ./q36_test --dense-quant-model-rows \
        --model gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
    flock -w 2400 /tmp/q36-gpu.lock bash -c \
        'Q36_VK_SHADER_TRACE=1 Q36_VK_PROF_KERNEL=1 ./q36 \
        -m gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf -p "The quick brown fox" 2>&1'

`gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf` is 5946.6 MB, 402 type-143 tensors,
in_dims {5120, 6144, 17408}. `verify_native_128_ptq1_0.py` (this dir)
independently proved the 28 B/block / native-128 layout in Phase B1.

## Traps (from the B2 plan, still true)

- `pgrep -f`/`pkill -f` match the calling shell here. Use `pgrep -x`.
- One GPU job at a time; the flock is contended, not shared.
- Rebuild everything with `make`; stale binaries otherwise.
- Don't edit `llama.cpp/`/`ds4/` checkouts (ignored references only).
- The whole-model CPU<->GPU parity issue is closed as non-blocking (GPU-first
  policy). Do not resume it for PTQ1_0 unless the brief says so.
- A test that reports OK may have compared nothing — assert coverage
  (the dense-quant test prints a coverage table; 143 shows 402/3/24/12).

## Files touched

`Makefile`, `q36.c`, `q36_vulkan.c`, `q36_mmq_contract.h`,
`tests/q36_test.c`, `vulkan/dense_extra_decode_ptq1_0.comp`,
`vulkan/dense_extra_mmq_ptq1_0.comp`, plus this dir's validation files.