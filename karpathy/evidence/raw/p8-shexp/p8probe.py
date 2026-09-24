#!/usr/bin/env python3
"""P8 premise probe: how much of IQ2_M decode is the shared-expert path, and is
that path dispatch-bound (foldable) or efficiency-bound (not foldable)?

Reads the Q36_VK_PROF=1 Q36_VK_PROF_SHAPE=1 profile of a ctx-512 / gen-128
q36-bench run.  The shape in the op name is `<in_dim>x<out_dim>` for `n_tok == 1`
(see q36_vulkan.c q36_vk_prof_iq3_shape), so:
  dense_iq2_s_decode_2048x512_n1 -> shared-expert gate and up (2048 -> 512)
  dense_iq2_s_decode_512x2048_n1 -> shared-expert down      (512 -> 2048)
The routed kernels are the comparison: moe_iq2s_gate_up_decode /
moe_iq2s_down_sum_decode over 8 experts, same IQ2_S format, same model.

IQ2_S is 2.5625 bpw = 0.3203 B/weight.
"""
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p36/out10/p8-shape.log"
NTOK = 128
N_LAYER = 40
BPW_BYTES = 0.3203
# prefill-only rows (n256 tiles and the GEMM pair); ignored for a decode share
PREFILL = re.compile(r"_n256|_gemm")

rows = {}
for ln in open(path):
    m = re.match(r"q36:   op (\S+)\s+dispatches=(\d+).*gpu_ms=([\d.]+)", ln)
    if m:
        rows[m.group(1)] = (int(m.group(2)), float(m.group(3)))

dec = {k: v for k, v in rows.items() if not PREFILL.search(k)}
total = sum(v[1] for v in dec.values())
print(f"op rows {len(rows)}, decode rows {len(dec)}, "
      f"decode gpu total {total:.3f} ms = {total / NTOK:.3f} ms/tok")

# kernels that serve both prefill and decode: scale by the P1 decode-only
# dispatch counts (128 decode steps) over this run's total dispatches.
MIXED = {"q8_k_quant": 20608, "add_rms_norm": 10240}
for k, share in MIXED.items():
    if k in rows:
        d, g = rows[k]
        print(f"  mixed {k:14} disp={d} gpu={g:.3f} -> decode est {g * share / d:.3f} ms")

print("\nshared-expert matvecs vs the routed kernels:")
def bw(name, in_dim, out_dim, experts, mult):
    d, g = rows[name]
    by = experts * mult * in_dim * out_dim * BPW_BYTES
    print(f"  {name:34} disp={d:6} ({d / NTOK:5.1f}/tok, {d / NTOK / N_LAYER:4.2f}/layer)"
          f"  {g / d * 1e3:6.2f} us/disp  {by / 1e6:5.3f} MB/disp"
          f"  {by / (g / d) * 1e3 / 1e9:6.1f} GB/s")  # g is ms, so x1e3 for B/s
    return g

g1 = bw("dense_iq2_s_decode_2048x512_n1", 2048, 512, 1, 1)
g2 = bw("dense_iq2_s_decode_512x2048_n1", 512, 2048, 1, 1)
bw("moe_iq2s_gate_up_decode", 2048, 512, 8, 2)
bw("moe_iq2s_down_sum_decode", 2048, 512, 8, 1)

sh = g1 + g2
print(f"\nshared-expert matvecs {sh:.3f} ms = {sh / NTOK:.4f} ms/tok "
      f"= {100 * sh / total:.2f}% of the decode-row total")
print(f"  + swiglu and one mid q8_k_quant per layer: "
      f"~{100 * (sh + total * 0.005) / total:.1f}% including them")
