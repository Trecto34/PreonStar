LEDGER_CHECK:
source=AlreadyTried.md
active=AlreadyTried.active.md
source_sha256=72ee2996e75961a63749c7f2044a25064c046d9179423dd26201d5e032999fa2
priority=8
entries=7

# Active Optimization Ledger (Priority 8)

> **Auditable source:** `AlreadyTried.md` (# Local Swift campaign ledger) (SHA-256: `72ee2996e75961a6...`)  
> **Filter:** Priority 8  
> **Mandatory Policy:** Do not repeat rejected approaches without a stated new hardware/runtime reason. Preserve MoE routing parity for Qwen3.6 reference.

## Summary of Prior Attempts

| ID | Abordagem | Alvo | Resultado | Motivo | Reconsiderar se |
|---|---|---|---|---|---|
| `AT-SWIFT-001` | **128-lane MMQ split** | `vulkan/dense_iq3_xxs_mmq.comp` | **REJECTED** | Regressed down-projection microbenchmark by ~2.7% (first build failed on GLSL-reserved 'half'); bit-exact but no end-to-end win. | Measured explanation and architectural fix for down-projection regression with alternative work/memory distribution. |
| `AT-SWIFT-002` | **XOR-swizzled IQ3_XXS MMQ staging** | `vulkan/dense_iq3_xxs_mmq.comp` | **REJECTED** | 128-token prefill MMQ kernel slowed from 940.28 ms to 1309.31 ms (+39.2% slower). Existing 17-element padded stride is faster on BC-250. | New staging architecture avoiding LDS bank conflicts without 16-element XOR stride overhead. |
| `AT-SWIFT-003` | **IQ3_XXS MMQ token tiles 64 or 256** | `vulkan/dense_iq3_xxs_mmq.comp` | **REJECTED** | Produced 161.07 tok/s (tile 64) and 166.87 tok/s (tile 256) at context 1024, versus 172.41 tok/s for default 128-token tile. | Workgroup dispatch divisor or tile dimensions changed in conjunction with different register tiling. |
| `AT-SWIFT-004` | **IQ3_XXS MMQ dynamic k loop (removing [[unroll]])** | `vulkan/dense_iq3_xxs_mmq.comp` | **REJECTED** | Removing [[unroll]] raised 128-token MMQ GPU time from ~940 to 960 ms (+2.1% slower). | Dynamic bounds proven strictly necessary and pipelined without compiler unroll loss. |
| `AT-SWIFT-005` | **Prefill chunk size > 256 on GFX1013** | `runtime prefill chunk` | **REJECTED** | Exceeds the 10 s amdgpu watchdog timer and destroys GPU context on BC-250 (GQA 6). Default 256 is fastest safe cap. | Mathematical proof of watchdog-safety or kernel execution time reduced sufficiently under 10s watchdog. |
| `AT-SWIFT-010` | **MoE matvec type specialization + Q4_K packed loads** | `vulkan/moe_matvec{,_fast}.comp (APEX-I decode)` | **REJECTED** | Both neutral: decode median 71.8 t/s before, 71.8-71.9 t/s after (3 reps). Not a kernel verdict - decode on this host is disk-read-bound (10.3 MiB/token against a 745 MiB/s NVMe stream), so the ALU path is off the critical path. | Weights become resident (pinning, or a box with RAM for model + working set), making decode compute-bound. |
| `AT-SWIFT-011` | **--ssd-streaming with an 8GB expert cache** | `runtime SSD streaming on APEX-I MoE` | **REJECTED** | Decode 3.18 t/s vs 71.26 t/s baseline (22x slower) despite a 97.8% cache hit rate; the staged-copy path dominates at these expert sizes. | The streaming copy is fused into the matvec, or a model's expert working set exceeds RAM by an order of magnitude. |

## Structured Records

### AT-SWIFT-001: 128-lane MMQ split
- **Priority:** 8
- **Target:** `vulkan/dense_iq3_xxs_mmq.comp`
- **Result:** REJECTED
- **Reason Rejected:** Regressed down-projection microbenchmark by ~2.7% (first build failed on GLSL-reserved 'half'); bit-exact but no end-to-end win.
- **Evidence:** Down-projection microbench: -2.7% regression. Build broke on GLSL keyword 'half'.
- **Reconsider If:** Measured explanation and architectural fix for down-projection regression with alternative work/memory distribution.
- **Source Reference:** AlreadyTried.md:18-25

### AT-SWIFT-002: XOR-swizzled IQ3_XXS MMQ staging
- **Priority:** 8
- **Target:** `vulkan/dense_iq3_xxs_mmq.comp`
- **Result:** REJECTED
- **Reason Rejected:** 128-token prefill MMQ kernel slowed from 940.28 ms to 1309.31 ms (+39.2% slower). Existing 17-element padded stride is faster on BC-250.
- **Evidence:** Kernel latency: 940.28 ms -> 1309.31 ms (+39.2% slower) at 128-token context.
- **Reconsider If:** New staging architecture avoiding LDS bank conflicts without 16-element XOR stride overhead.
- **Source Reference:** AlreadyTried.md:32-34

### AT-SWIFT-003: IQ3_XXS MMQ token tiles 64 or 256
- **Priority:** 8
- **Target:** `vulkan/dense_iq3_xxs_mmq.comp`
- **Result:** REJECTED
- **Reason Rejected:** Produced 161.07 tok/s (tile 64) and 166.87 tok/s (tile 256) at context 1024, versus 172.41 tok/s for default 128-token tile.
- **Evidence:** 161.07 & 166.87 tok/s vs 172.41 tok/s at context 1024.
- **Reconsider If:** Workgroup dispatch divisor or tile dimensions changed in conjunction with different register tiling.
- **Source Reference:** AlreadyTried.md:41-43

### AT-SWIFT-004: IQ3_XXS MMQ dynamic k loop (removing [[unroll]])
- **Priority:** 8
- **Target:** `vulkan/dense_iq3_xxs_mmq.comp`
- **Result:** REJECTED
- **Reason Rejected:** Removing [[unroll]] raised 128-token MMQ GPU time from ~940 to 960 ms (+2.1% slower).
- **Evidence:** 128-token MMQ GPU time: 940 -> 960 ms.
- **Reconsider If:** Dynamic bounds proven strictly necessary and pipelined without compiler unroll loss.
- **Source Reference:** AlreadyTried.md:44-45

### AT-SWIFT-005: Prefill chunk size > 256 on GFX1013
- **Priority:** 8
- **Target:** `runtime prefill chunk`
- **Result:** REJECTED
- **Reason Rejected:** Exceeds the 10 s amdgpu watchdog timer and destroys GPU context on BC-250 (GQA 6). Default 256 is fastest safe cap.
- **Evidence:** AMDGPU 10s watchdog timeout and GPU context reset.
- **Reconsider If:** Mathematical proof of watchdog-safety or kernel execution time reduced sufficiently under 10s watchdog.
- **Source Reference:** AlreadyTried.md:38-40

### AT-SWIFT-010: MoE matvec type specialization + Q4_K packed loads
- **Priority:** 8
- **Target:** `vulkan/moe_matvec{,_fast}.comp (APEX-I decode)`
- **Result:** REJECTED
- **Reason Rejected:** Both neutral: decode median 71.8 t/s before, 71.8-71.9 t/s after (3 reps). Not a kernel verdict - decode on this host is disk-read-bound (10.3 MiB/token against a 745 MiB/s NVMe stream), so the ALU path is off the critical path.
- **Evidence:** evidence/ab-moe-type-specialization-20261023.md; evidence/decode-disk-bound-20261023.md. Packed-load arm bit-identical (md5 4915110d over 64 greedy tokens).
- **Reconsider If:** Weights become resident (pinning, or a box with RAM for model + working set), making decode compute-bound.
- **Source Reference:** AlreadyTried.md:2291-2315

### AT-SWIFT-011: --ssd-streaming with an 8GB expert cache
- **Priority:** 8
- **Target:** `runtime SSD streaming on APEX-I MoE`
- **Result:** REJECTED
- **Reason Rejected:** Decode 3.18 t/s vs 71.26 t/s baseline (22x slower) despite a 97.8% cache hit rate; the staged-copy path dominates at these expert sizes.
- **Evidence:** evidence/decode-thrash-tensors-20261023.md; run log cache=796/6853 hits=36068/36864 loads=796 evictions=0.
- **Reconsider If:** The streaming copy is fused into the matvec, or a model's expert working set exceeds RAM by an order of magnitude.
- **Source Reference:** AlreadyTried.md:2317-2332

## Transferable Winning Patterns (from OPT-20/21/23)
- **Hoist Invariant Addressing:** Compute row/col indexing, stride, and scale group offsets outside `kb` block loops.
- **Vectorized Loads:** Replace scalar 32-bit loads with 128-bit `vec4` / `uvec4` memory loads.
- **Compiler Loop Unrolling:** Use `[[unroll]]` on inner fixed-iteration loops (e.g. 8-step `kk` loop) for pipeline scheduling.

