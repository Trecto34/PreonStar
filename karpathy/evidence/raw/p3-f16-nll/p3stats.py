#!/usr/bin/env python3
"""Paired statistics for q36-bench frontier dumps (frontier_nll.py only prints means).

Usage: p3stats.py TOKENS_JSON DIR_REF DIR... [--dump]
Prints per-frontier signed dNLL (DIR - REF), count of sign, top-1 agreement and
mean |NLL| (not |mean NLL|) so a change that helps some frontiers and hurts
others cannot average out to "no effect".
"""
import json, math, os, re, sys

tokens = json.loads(open(sys.argv[1]).readline())
args = [a for a in sys.argv[2:] if not a.startswith("--")]
dump = "--dump" in sys.argv
dirs = args
ref = dirs[0]
load = {}
for d in dirs:
    rows = {}
    for name in sorted(os.listdir(d)):
        m = re.match(r"frontier_(\d+)\.logits\.json$", name)
        if not m:
            continue
        n = int(m.group(1))
        lg = json.load(open(os.path.join(d, name)))["logits"]
        mx = max(lg)
        lse = mx + math.log(sum(math.exp(x - mx) for x in lg))
        t = tokens[n]
        rows[n] = (lse - lg[t], max(range(len(lg)), key=lg.__getitem__) == t)
    load[d] = rows

base = load[ref]
for d in dirs:
    rows = load[d]
    ns = sorted(set(rows) & set(base))
    dlt = [rows[n][0] - base[n][0] for n in ns]
    mean = sum(dlt) / len(dlt)
    srt = sorted(dlt)
    med = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
    mad = sorted(abs(x - med) for x in dlt)[len(dlt) // 2]
    amean = sum(abs(x) for x in dlt) / len(dlt)
    up = sum(1 for x in dlt if x > 1e-6)
    dn = sum(1 for x in dlt if x < -1e-6)
    same = sum(1 for n in ns if rows[n][1] == base[n][1])
    top1 = sum(1 for n in ns if rows[n][1])
    print(f"{d}: n={len(ns)} mean_nll={sum(rows[n][0] for n in ns)/len(ns):.4f} top1={top1}/{len(ns)}")
    if d != ref:
        print(f"   vs {ref}: dNLL mean={mean:+.4f} med={med:+.4f} MAD={mad:.4f} mean|d|={amean:.4f}"
              f" worse={up} better={dn} top1_same={same}/{len(ns)}")
        if dump:
            for n, x in zip(ns, dlt):
                print(f"   frontier {n}: dNLL {x:+.3f}")
