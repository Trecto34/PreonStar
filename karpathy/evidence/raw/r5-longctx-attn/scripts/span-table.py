#!/usr/bin/env python3
"""R5 span-sweep table (stage A 512 + stages B/C) at ctx 16384, markdown."""
import re, pathlib
GEN=64
def row(p):
    t=p.read_text(errors="replace")
    m=re.search(r"^(\d+),\d+,([\d.]+),\d+,([\d.]+),\d+$",t,re.M)
    if not m: return None
    ctx=int(m.group(1)); tps=float(m.group(3))
    a=re.search(r"op attn_decode_split\s+dispatches=(\d+) .*?gpu_ms=([\d.]+)",t)
    b=re.search(r"op attn_combine\s+dispatches=(\d+) .*?gpu_ms=([\d.]+)",t)
    if not a: return None
    sp=a[1]/GEN; L=a[0]/GEN
    spm=a[1] and float(a[1])  # noQA
    split=float(a[2])/int(a[0]); comb=(float(b[2])*int(a[0])/int(b[0])/GEN) if b else 0.0
    ms=re.search(r"span(\d+)",p.name); span=int(ms.group(1)) if ms else 512
    nh=24 if p.name.startswith("swift") else 16
    spans=-(-ctx//span)+1; wg=nh*spans
    return dict(model="swift" if p.name.startswith("swift") else "guard", span=span,
                wg=wg, split=split, comb=comb, tps=tps, nsk=1e6*split/(wg*span)/40*40,
                key=1e6*split/(wg*span))
rows=[]
for p in sorted(pathlib.Path('/tmp/r5/out').glob('*.log')):
    r=row(p)
    if r and r["span"]==512 and "-r1" in p.name: rows.append(r)
for p in sorted(pathlib.Path('/tmp/r5/outB').glob('*.log')):
    r=row(p)
    if r: rows.append(r)
for m in ("swift","guard"):
    grp=sorted([r for r in rows if r["model"]==m], key=lambda r:r["span"])
    base=next((r for r in grp if r["span"]==512), None)
    for r in grp:
        d = "" if not base or r is base else " %+5.1f%%" % (100.0*(r["split"]/base["split"]-1))
        print("|  %-5s | %5d | %5d | %8.4f%s | %8.4f | %5.2f | %7.4f |" %
              (r["model"], r["span"], r["wg"], r["split"], d, r["comb"], r["tps"], r["key"]))
