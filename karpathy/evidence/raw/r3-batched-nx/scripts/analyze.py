#!/usr/bin/env python3
"""R3 analysis: nx ON/OFF on --batched-session N (aggregate, per-stream,
decode-only), plus the per-batched-step cost from the server logs and the
kernel attribution from the profiled arms."""
import json, re, statistics, glob, os

OUT = "/tmp/r3/out"

def rows():
    r = []
    for line in open(f"{OUT}/results.jsonl"):
        r.append(json.loads(line))
    return r

def step_med(tag):
    t = open(f"{OUT}/srv-{tag}.log").read()
    per = {}
    for c, ms in re.findall(r"decode batch count=(\d+) elapsed=([0-9.]+) ms", t):
        per.setdefault(int(c), []).append(float(ms))
    return per

def main():
    arms = {}
    for r in rows():
        if r["label"].startswith("prof"):
            continue
        n, nx = r["streams"], 1 if "-nx1-" in r["label"] else 0
        # decode-only: wall_i - ttft_i, from per_stream_tps = tokens / wall_i
        dec = [r["max_tokens"] / (r["max_tokens"] / p - tt)
               for p, tt in zip(r["per_stream_tps"], r["ttft_s"])]
        arms.setdefault((n, nx), []).append(
            (r["aggregate_tps"], statistics.median(dec), sum(dec), sum(r["ttft_s"]) / len(r["ttft_s"])))
    print("N nx  agg_tps(rep1,rep2) med | per-stream gen (excl TTFT) med | decode-only agg | TTFT med")
    for n in (1, 2, 4, 8):
        for nx in (0, 1):
            v = arms.get((n, nx))
            if not v:
                continue
            agg = [x[0] for x in v]
            ps = [x[1] for x in v]
            da = [x[2] for x in v]
            tt = [x[3] for x in v]
            print(f"{n}  {nx}   {['%.3f' % a for a in agg]} med={statistics.median(agg):6.3f} | "
                  f"{statistics.median(ps):6.3f} | {statistics.median(da):6.3f} | {statistics.median(tt):6.3f}")
    print()
    print("per-batched-step cost (server 'decode batch count=N elapsed')")
    for tag in ("n1-nx0-r1", "n1-nx1-r1", "n2-nx0-r1", "n2-nx1-r1",
                "n4-nx0-r1", "n4-nx1-r1", "n8-nx0-r1", "n8-nx1-r1"):
        per = step_med(tag)
        s = " ".join(f"c{c}:n={len(v)},med={statistics.median(v):.0f}ms"
                     for c, v in sorted(per.items()) if len(v) > 1)
        print(f"{tag:12s} {s}")
    print()
    print("profiled arms: dense trunk kernels at n_tok 2..8 (gpu_ms over the run)")
    for tag in ("prof-n1-nx0", "prof-n1-nx1", "prof-n4-nx0", "prof-n4-nx1"):
        t = open(f"{OUT}/srv-{tag}.log").read()
        print(f"-- {tag}")
        for name, d, g in re.findall(
                r"vulkan/([a-z0-9_]+)\.spv\s+dispatches=(\d+) groups=\d+ gpu_ms=([0-9.]+)", t):
            if any(k in name for k in ("mmq", "decode", "pair", "top2", "dense_kquant")):
                print(f"   {name:32s} disp={d:>6s} gpu_ms={float(g):9.1f}")

if __name__ == "__main__":
    main()
