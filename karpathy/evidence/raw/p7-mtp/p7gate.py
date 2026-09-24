#!/usr/bin/env python3
"""P6 gate stats: gen tps and MTP accept counters per arm, median + MAD."""
import glob, os, re, statistics as st

def stats(v):
    m = st.median(v)
    return m, st.median([abs(x - m) for x in v])

def load(pat):
    rows = {}
    for p in sorted(glob.glob(pat)):
        arm = os.path.basename(p).split(".log")[0]
        arm = re.sub(r"^r\d+-", "", arm)
        tps = None; acc = None
        for ln in open(p):
            if ln.startswith("512,"):
                tps = float(ln.split(",")[4])
            m = re.search(r"MTP stats calls=(\d+) drafted=(\d+) accepted=(\d+) full=(\d+) accept=([\d.]+)%", ln)
            if m:
                acc = tuple(int(x) for x in m.group(1, 2, 3, 4)) + (float(m.group(5)),)
        rows.setdefault(arm, []).append((tps, acc))
    return rows

rows = load("/tmp/p36/out16/*.log")
print(f"{'arm':8} {'n':>2} {'gen_tps median':>14} {'MAD':>7}  values")
for arm in sorted(rows):
    v = [r[0] for r in rows[arm] if r[0] is not None]
    if not v: continue
    med, mad = stats(v)
    print(f"{arm:8} {len(v):>2} {med:>14.3f} {mad:>7.3f}  {['%.2f' % x for x in v]}")

print("\ndecode t/s medians: off -> on")
off = [r[0] for r in rows.get("d3off", []) if r[0] is not None]
on  = [r[0] for r in rows.get("d3on", []) if r[0] is not None]
if off and on:
    print(f"  d3off median {st.median(off):.3f}  d3on median {st.median(on):.3f}  "
          f"ratio {st.median(on)/st.median(off):.3f}x  ({(st.median(on)/st.median(off)-1)*100:+.1f}%)")
    print(f"  worst-case ratio min(on)/max(off) = {min(on)/max(off):.3f}x")

print("\nMTP accept counters (deterministic - identical every rep?)")
for arm in sorted(rows):
    accs = [r[1] for r in rows[arm] if r[1]]
    if not accs: continue
    uniq = set(accs)
    print(f"  {arm:6} n={len(accs)} distinct={len(uniq)}  {sorted(uniq)}")
