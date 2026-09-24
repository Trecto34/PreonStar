#!/usr/bin/env python3
"""Median + MAD of generation t/s per arm from the out15 A/B logs."""
import glob, os, re, statistics as st

def tps(path):
    t = open(path, "r", errors="replace").read()
    m = re.search(r"generation: ([0-9.]+) t/s", t)
    return float(m.group(1)) if m else None

for model in ("swift", "moe"):
    for arm in ("off", "on"):
        v = []
        for rep in range(1, 8):
            p = f"/tmp/p36/out15/r{rep}-{model}-{arm}.txt"
            if os.path.exists(p):
                x = tps(p)
                if x:
                    v.append(x)
        if not v:
            print(f"{model:6s} {arm:3s} (no data)")
            continue
        med = st.median(v)
        mad = st.median([abs(x - med) for x in v])
        print(f"{model:6s} {arm:3s} n={len(v)} reps={['%.2f' % x for x in v]} "
              f"median={med:.3f} t/s MAD={mad:.3f} ({1000.0/med:.3f} ms/token)")
    off = [tps(f"/tmp/p36/out15/r{r}-{model}-off.txt") for r in range(1, 8)]
    on = [tps(f"/tmp/p36/out15/r{r}-{model}-on.txt") for r in range(1, 8)]
    off = [x for x in off if x]
    on = [x for x in on if x]
    if off and on:
        mo, mn = st.median(off), st.median(on)
        print(f"{model:6s} median delta ON vs OFF = {100.0*(mn-mo)/mo:+.2f}%  "
              f"(per-rep worst pairing "
              f"{min(100.0*(on[i]-off[i])/off[i] for i in range(min(len(off),len(on)))):+.2f}%)")
