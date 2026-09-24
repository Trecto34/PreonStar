#!/usr/bin/env python3
"""P7 attribution: subtract a prefill-mostly run from the d3 profile.

The prof "decode" label leaks prefill work: both runs prefill 512 tokens, but a
gen-1 run also carries one decode step of its own, so the subtraction is only
good to ~20 ms per row (the audit's profiling caveat).  Rows whose subtraction
comes out negative are prefill-dominated and are reported as such.

Usage: p7attr.py <prefill-mostly.log> <d3on.log> <d3off.log> <decode_tokens>
"""
import re
import sys


def rows(path):
    out = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"op ([a-z0-9_]+) +dispatches=(\d+) flushes=\d+ groups=(\d+) "
                      r"record_ms=([\d.]+) submit_wait_ms=([\d.]+) gpu_ms=([\d.]+)", ln)
        if m:
            # last block per op = the prof label=decode block
            out.setdefault(m.group(1), (int(m.group(2)), float(m.group(6))))
            out[m.group(1)] = (int(m.group(2)), float(m.group(6)))
    return out


def main():
    pre_p, on_p, off_p, tok = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    pre, on, off = rows(pre_p), rows(on_p), rows(off_p)
    print(f"decode tokens = {tok:.0f};  prefill-only baseline = {pre_p}")
    print(f"{'op':26} {'d3on-pre/tok':>12} {'d3off-pre/tok':>13} {'on_disp':>8} {'off_disp':>8}")
    for k in sorted(set(pre) | set(on) | set(off)):
        a = pre.get(k, (0, 0.0))[1]
        bd, b = on.get(k, (0, 0.0))
        cd, c = off.get(k, (0, 0.0))
        t1, t2 = (b - a) / tok, (c - a) / tok
        if abs(t1) < 0.005 and abs(t2) < 0.005:
            continue
        print(f"{k:26} {t1:12.3f} {t2:13.3f} {bd:8d} {cd:8d}")
    for name, d in (("d3on", on), ("d3off", off)):
        tot = sum(v[1] for v in d.values())
        ptot = sum(v[1] for v in pre.values() if v[1] >= 0)
        print(f"{name}: decode-block kernel total {tot:.1f} ms over {tok:.0f} tok = "
              f"{tot/tok:.1f} ms/tok (prefill-mostly run total {ptot:.1f} ms)")


if __name__ == "__main__":
    main()
