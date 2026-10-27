#!/usr/bin/env python3
"""Parse p36-bench instrumented logs: per-op GPU ms, dispatch counts, WG buckets.

Decode numbers are derived as (192-token run) - (prefill-only run): same 2048
token prefill in both, so the difference is the 192-token decode tail.
"""
import re
import sys
import statistics

OP = re.compile(
    r"p36:\s+op (\S+)\s+dispatches=(\d+) flushes=(\d+) barriers=(\d+) groups=(\d+)"
    r" record_ms=([\d.]+) submit_wait_ms=([\d.]+) gpu_ms=([\d.]+) wall_ms=([\d.]+)")
WG = re.compile(r"p36:\s+wg (\S+)\s+n=(\d+) <5=(\d+) 5-20=(\d+) 20-40=(\d+) >40=(\d+)")


def parse(path):
    ops, wgs, dev = {}, {}, {}
    for line in open(path, errors="replace"):
        m = OP.search(line)
        if m:
            r = ops.setdefault(m.group(1), dict(d=0, b=0, g=0, ms=0.0))
            r["d"] += int(m.group(2)); r["b"] += int(m.group(4))
            r["g"] += int(m.group(5)); r["ms"] += float(m.group(8))
            continue
        m = WG.search(line)
        if m:
            r = wgs.setdefault(m.group(1), [0, 0, 0, 0, 0])
            r[0] += int(m.group(2))
            for i in range(4):
                r[i + 1] += int(m.group(3 + i))
            continue
        if "timestamps period_ns" in line:
            dev["ts"] = line.strip()
    return ops, wgs, dev


def diff(a, b, dec_ops, dec_hist):
    """b - a per op."""
    out = {}
    for name in set(a) | set(b):
        ra, rb = a.get(name), b.get(name)
        out[name] = dict(
            d=(rb["d"] if rb else 0) - (ra["d"] if ra else 0),
            b=(rb["b"] if rb else 0) - (ra["b"] if ra else 0),
            g=(rb["g"] if rb else 0) - (ra["g"] if ra else 0),
            ms=(rb["ms"] if rb else 0.0) - (ra["ms"] if ra else 0.0),
        )
    return out


def main(full_log, prefill_log, tag, ntok=192):
    full, fw, dev = parse(full_log)
    pre, pw, _ = parse(prefill_log)
    if dev:
        print(dev["ts"])
    dec = diff(pre, full, None, None)
    tot_full_ms = sum(v["ms"] for v in full.values())
    tot_pre_ms = sum(v["ms"] for v in pre.values())
    tot_dec_ms = sum(v["ms"] for v in dec.values())
    tot_full_d = sum(v["d"] for v in full.values())
    tot_pre_d = sum(v["d"] for v in pre.values())
    tot_dec_d = sum(v["d"] for v in dec.values())
    print(f"\n== {tag} ==")
    print(f"wall gpu_ms: prefill-run={tot_pre_ms:.1f} full={tot_full_ms:.1f} decode(diff)={tot_dec_ms:.1f}")
    print(f"dispatches:  prefill-run={tot_pre_d} full={tot_full_d} decode(diff)={tot_dec_d}")
    print(f"dispatches/token: prefill={tot_pre_d/2048:.2f} decode={tot_dec_d/ntok:.2f}")
    print(f"decode ms/token   = {tot_dec_ms/ntok:.3f}")
    print("\n per-op decode (diff), sorted by ms descending:")
    print(f"{'op':38s} {'disp':>7s} {'disp/tok':>9s} {'ms':>8s} {'ms/tok':>7s} {'%dec':>6s}")
    for name, v in sorted(dec.items(), key=lambda kv: -kv[1]["ms"])[:24]:
        if v["ms"] <= 0 and v["d"] <= 0:
            continue
        pct = 100.0 * v["ms"] / tot_dec_ms if tot_dec_ms else 0
        print(f"{name:38s} {v['d']:7d} {v['d']/ntok:9.2f} {v['ms']:8.2f} {v['ms']/ntok:7.3f} {pct:6.1f}")
    # WG histogram for decode = full - prefill
    print("\n decode WG histogram (diff, dispatches per bucket):")
    agg = [0, 0, 0, 0]
    rows = []
    for name in set(fw) | set(pw):
        a = pw.get(name, [0] * 5)
        b = fw.get(name, [0] * 5)
        d = [b[i] - a[i] for i in range(5)]
        if d[0] > 0:
            rows.append((name, d))
        for i in range(4):
            agg[i] += max(0, d[i + 1])
    for name, d in sorted(rows, key=lambda kv: -kv[1][0])[:15]:
        print(f"  {name:36s} n={d[0]:6d} <5={d[1]:5d} 5-20={d[2]:5d} 20-40={d[3]:5d} >40={d[4]:5d}")
    t = sum(agg)
    print(f"  {'TOTAL':36s} n={t:6d} <5={agg[0]:5d} 5-20={agg[1]:5d} 20-40={agg[2]:5d} >40={agg[3]:5d}")
    if t:
        print("  share: " + "  ".join(f"{lab}={100.0*c/t:.1f}%" for lab, c in
                                      zip(("<5", "5-20", "20-40", ">40"), agg)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "run")
