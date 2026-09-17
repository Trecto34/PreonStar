#!/usr/bin/env bash
# bench_ab.sh B-side wrapper: identical q36-bench binary, RADV_PERFTEST=cswave32.
#
# Why this exists
# ---------------
# q36_vk_force_wave32 (q36_vulkan.c:1569) forces Wave32 only for shaders whose
# path matches `dense_*_mmq.spv`, plus a four-entry named list (delta_net_cols,
# rope_qwen, rope_qwen_mrope, quantize_q8_0). Every DECODE kernel therefore
# still runs at the RADV default of subgroupSize 64, even though gfx1013 is a
# wave32-first ISA.
#
# A peer BC-250 cluster measured this driver flag at +1.3% prefill in a
# tq-patched llama.cpp fork, on top of forcing subgroup 32 into the FA prefill
# pipeline (+7.4% pp16384 / +8.5% pp24576). Our prefill MMQ path is plausibly
# already covered by the source-level force; our decode path is not covered at
# all, and decode is where 57.8% of the time sits in dense_iq3_xxs_decode_r4 at
# 269 GB/s against a 333 GB/s sibling. That is the surface this test probes.
#
# Fully reversible: it is an environment variable, nothing is rebuilt. The
# control arm is the unmodified binary, so an interleaved bench_ab.sh run
# isolates the flag.
set -euo pipefail
export RADV_PERFTEST=cswave32
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/q36-bench" "$@"
