#!/usr/bin/env python3
"""P9a bit-identity gate: the text generated after the prompt must be identical
with the GPU min-p pack ON and OFF (only the throughput lines may differ)."""
import os, re, sys

D = "/tmp/p36/out15"
PAIRS = [("swift", "", 0.8, 0.05), ("moe", "", 0.8, 0.05),
         ("moe", "-t1", 1.0, 0.05), ("swift", "-mp2", 0.8, 0.2)]

def body(path):
    t = open(path, "r", errors="replace").read()
    m = re.search(r"processing \d+ input tokens[^\n]*\n(.*?)\nq36: prefill:", t, re.S)
    if not m:
        return None
    return m.group(1)

def tps(path):
    t = open(path, "r", errors="replace").read()
    m = re.search(r"generation: ([0-9.]+) t/s", t)
    return float(m.group(1)) if m else float("nan")

bad = 0
for name, sfx, temp, mp in PAIRS:
    a = os.path.join(D, f"par-{name}-on{sfx}.txt")
    b = os.path.join(D, f"par-{name}-off{sfx}.txt")
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"{name:10s} MISSING")
        continue
    ba, bb = body(a), body(b)
    same = ba is not None and ba == bb
    if not same:
        bad += 1
    print(f"{name+sfx:12s} temp={temp} min_p={mp} "
          f"{'IDENTICAL' if same else 'DIFFER'} "
          f"bytes={len(ba or '')}/{len(bb or '')} "
          f"gen {tps(a):.2f}/{tps(b):.2f} t/s")
sys.exit(1 if bad else 0)
