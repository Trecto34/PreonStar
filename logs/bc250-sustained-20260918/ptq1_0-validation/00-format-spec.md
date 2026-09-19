# PTQ1_0 (type 143) — Track B recon, spec extracted from the publisher

Status: **recon only.** Track A (PQ2_0) gates are still running; nothing here is
implemented in `q36`. Written 2026-09-18 after reading the publisher's own
source, so the geometry below is not guessed from the file size.

## Source of truth

Fork `github.com/PrismML-Eng/llama.cpp`, branch **`prism-v7`** (the PRISM types
are *not* on `master` — `master`'s `ggml-quants.c` contains no `prism`/`pq2_0`/
`ptq1_0` symbol at all):

    ggml/include/ggml.h        :435  GGML_TYPE_PQ2_0 = 142
                               :436  GGML_TYPE_PTQ1_0 = 143  // Prism-private ternary, group 128
                               :483  GGML_FTYPE_MOSTLY_PTQ1_0 = 129
    ggml/src/ggml-common.h     :215  block_ptq1_0
    ggml/src/ggml-quants.c     :2205 quantize_row_ptq1_0_ref   (canonical rule)
                               :2255 dequantize_row_ptq1_0
                               :2287 quantize_ptq1_0          (row driver)

## Block geometry — 28 B per 128 weights

    typedef struct {
        uint8_t  qs[24];   // 5 trits per byte -> 120 values
        uint8_t  qh[ 2];   // 4 trits per byte ->   8 values
        ggml_half d;       // scale, f16
    } block_ptq1_0;        // QK_PTQ1_0 = 128, sizeof = 28

Two things to note against PQ2_0 (34 B, `d` first): the ternary block is
**28 B/128** (-17.6 % weight bytes) and the **scale is the last field**, so any
kernel ported from the PQ2_0 path must not reuse its `d`-first offset.

## Code mapping (base-3 trits, staged packing)

`ptq1_0_stages = {32, 16, 8}`, walked with `j += c` while `j + c <= 24`. Stage
`c=32` never runs (32 > 24), so the 120 `qs` values split as:

    values   0.. 79 : qs byte (v % 16),        trit n = v / 16        (bytes 0..15)
    values  80..119 : qs byte 16 + (l % 8),    trit n = l / 8,  l = v-80  (bytes 16..23)
    values 120..127 : qh byte (l % 2),         trit n = l / 2,  l = v-120

The three-way split above is hand-derived from the loop order, so it was checked
against a literal simulation of the C traversal (`j + c <= 24`, `for n in 0..4`,
`for m in 0..c`, then `for n in 0..3`, `for h in 0..1`): it emits exactly 128
values, finishes at `j == 24`, and matches the mapping element-for-element. That
simulation is reproducible with the loop transcribed verbatim; re-run it before
trusting the mapping in a kernel.

Encode: `xi = lround(x * id) + 1` in {0,1,2}; `q = base-3 sum` of the trits, most
significant first; byte = `(q*256 + 242) / 243` (ceiling division). `qh` does one
extra `q *= 3`, shifting the first value into the most significant trit.

Decode (per trit `n`, `pow3 = {1,3,9,27,81}`):

    xi = ((uint16_t)(byte * pow3[n]) * 3) >> 8;
    value = (xi - 1) * d;

This multiply-and-shift is upstream `TQ1_0`'s trick, reused. PTQ1_0 is *not* a
new arithmetic scheme: it is `TQ1_0`'s base-3 packing moved from its 48-byte
sub-block to a 128-weight block under one f16 scale.

## The shipped file

    gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf   5946.6 MB, 402 type-143 tensors
    in_dims present: 5120, 6144, 17408      (same set as PQ2_0)
    other types: 353 x type 0 (f32 1-D/norms), 96 x type 30
    quant payload at 28 B/128 over ~26.87e9 weights: 5.88 GB  -- matches the file

The publisher repo `prism-ml/Ternary-Bonsai-2-27B-gguf` ships the **F16 source**
alongside PQ2_0 and PTQ1_0, so the same range-fetch proof used by
`../pq2_0-validation/verify_native_128.py` can establish whether this file is
natively group-128. **Not yet run** — it is the first Track B gate, and it is
cheap (header + sampled rows, no GPU).

## Engine gap (measured, not assumed)

`grep` over `q36*.{c,h}` and `vulkan/*.comp`: **no reference to 143 or PTQ1_0**.
The only `143` hit outside the IQ lookup tables and `third_party/iris/png.h` is
`q36_eval.c:667 .choice[5] = "143"`, an unrelated eval option string.

Track A's type-142 plumbing is the template; the sites to mirror are

    q36.c:53-59        QK_Q2_0/PQ2_0 byte constants, blocks per 256
    q36.c:285          {name, qk, bytes} type table
    q36.c:308-309      Q36_TENSOR_* enum, block struct
    q36_vulkan.c:40-41 Q36_VK_TENSOR_* enum
    q36_vulkan.c:245-248, :3264-3267  kernel handles + Q36_VK_KERNEL(...)
    q36_vulkan.c:7779-7780            bytes per 256 weights (72 / 68)
    q36_vulkan.c:8251                 host feature guard (int-dot, subgroup64)
    vulkan/dense_extra_{decode,mmq}.comp + Makefile targets
    tests/q36_test.c   layout oracle + `--dense-quant-model-rows` type list

The fork carries its own PTQ1_0 Vulkan work on side branches (`fix/ptq1_0-vulkan-supports-op`,
`perf/ptq1_0-cuda-mmq-prism`); those are useful cross-checks but our kernels are
independent ports, as with PQ2_0.

## Open questions for Track B

1. Is the shipped PTQ1_0 file natively group-128 (F16-source proof)? Not run.
2. Does `q36`'s existing ternary ordering assumption hold for the staged 32/16/8
   split, or does the round-trip in `quantize_row_ptq1_0_ref` (ceiling division,
   multiply-shift decode) need to be modelled exactly in the test reference, the
   way the f16-MMQ contract was for PQ2_0?
3. Weight traffic per 256 weights: PTQ1_0 reads 2x28 = 56 B vs PQ2_0's 68 B and
   g64's 72 B, so the expected decode gain is larger than Track A's. Measure,
   do not extrapolate.
