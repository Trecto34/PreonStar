#!/usr/bin/env python3
"""Summarise one instrumented run: decode op-group shares, dispatch counts, WG buckets.

Caveat baked in: the bracketed timestamp intervals over-count (they run longer
than wall clock), so 'ms' is reported raw AND scaled to the measured wall time
of the same run.  Shares are the trustworthy output.
"""
import re
import sys

OP = re.compile(r"p36:\s+op (\S+)\s+dispatches=(\d+) flushes=(\d+) barriers=(\d+) groups=(\d+)"
                r" record_ms=([\d.]+) submit_wait_ms=([\d.]+) gpu_ms=([\d.]+) wall_ms=([\d.]+)")
WG = re.compile(r"p36:\s+wg (\S+)\s+n=(\d+) <5=(\d+) 5-20=(\d+) 20-40=(\d+) >40=(\d+)")

PREFILL_KEYS = ("_mmq", "_gemm", "_p_", "attn_prefill", "predequant", "embed_stage_copy")


def classify(name):
    return "prefill" if any(k in name for k in PREFILL_KEYS) else "decode"


def main(log, tag, ntok, wall_ms_tok):
    ops, wgs = {}, {}
    for line in open(log, errors="replace"):
        m = OP.search(line)
        if m:
            ops[m.group(1)] = dict(d=int(m.group(2)), g=int(m.group(5)), ms=float(m.group(8)))
            continue
        m = WG.search(line)
        if m:
            wgs[m.group(1)] = [int(m.group(2))] + [int(m.group(3 + i)) for i in range(4)]
    dec = {k: v for k, v in ops.items() if classify(k) == "decode"}
    pre = {k: v for k, v in ops.items() if classify(k) == "prefill"}
    tot = sum(v["ms"] for v in dec.values())
    scale = wall_ms_tok / (tot / ntok) if tot else 1.0
    print(f"## {tag}  (ctx 2048, {ntok} decode tokens, wall {wall_ms_tok:.2f} ms/token)")
    print(f"decode-classified groups: {len(dec)} ops, {sum(v['d'] for v in dec.values())} dispatches,"
          f" {tot:.1f} ms raw = {tot/ntok:.2f} ms/tok; over-count factor {1/scale:.2f}x")
    print(f"prefill-classified: {sum(v['d'] for v in pre.values())} dispatches,"
          f" {sum(v['ms'] for v in pre.values()):.0f} ms raw")
    print(f"\n{'op':34s}{'disp/tok':>9s}{'ms_raw/tok':>11s}{'ms_scaled':>10s}{'%dec':>7s}")
    for k, v in sorted(dec.items(), key=lambda kv: -kv[1]["ms"])[:20]:
        print(f"{k:34s}{v['d']/ntok:9.1f}{v['ms']/ntok:11.3f}{v['ms']/ntok*scale:10.3f}{100*v['ms']/tot:7.1f}")
    print("\nWG buckets over decode ops (dispatches, % of decode dispatches):")
    agg = [0] * 4
    for k, r in wgs.items():
        if classify(k) != "decode":
            continue
        for i in range(4):
            agg[i] += r[i + 1]
    t = sum(agg)
    print("  " + "  ".join(f"{l}={c} ({100*c/t:.1f}%)" for l, c in zip(("<5", "5-20", "20-40", ">40"), agg)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
