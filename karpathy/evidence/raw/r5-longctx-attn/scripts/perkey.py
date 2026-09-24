#!/usr/bin/env python3
"""Per-key cost of attn_decode_split: splits the split's ms/token into the
per-workgroup and per-key terms, to separate 'grid saturated' from
'per-element work'."""
import re, sys, pathlib
GEN=64
def parse(p):
    t=p.read_text(errors="replace")
    m=re.search(r"^(\d+),\d+,([\d.]+),\d+,([\d.]+),\d+$",t,re.M)
    if not m: return None
    d={"ctx":int(m.group(1)),"tps":float(m.group(3))}
    for op in ("attn_decode_split","attn_combine"):
        mm=re.search(r"op %s\s+dispatches=(\d+) .*?gpu_ms=([\d.]+)"%op,t)
        if mm: d[op]=(int(mm.group(1)),float(mm.group(2)))
    return d
rows=[]
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.log")):
    d=parse(p)
    if not d or "attn_decode_split" not in d: continue
    m=re.search(r"span(\d+)",p.name); span=int(m.group(1)) if m else 512
    nh=24 if p.name.startswith("swift") else 16
    L=d["attn_decode_split"][0]/GEN
    ms=d["attn_decode_split"][1]/d["attn_decode_split"][0]   # ms per call
    nk=d["ctx"]+GEN/2
    spans=-(-d["ctx"]//span)+1
    wg=nh*spans
    cu_us=40.0*ms*1e3/wg        # CU-time per workgroup (us)
    rows.append((p.name.replace(".log",""),span,d["ctx"],d["tps"],ms,wg,cu_us,
                 1e6*ms/(wg*span),   # ns per (workgroup,key)
                 1e6*ms/nk))         # ns per key per head
hdr="arm                 span   ctx    t/s  ms/call    wg  us/wg  ns/wgkey  ns/keyhead"
print(hdr); print("-"*len(hdr))
for r in rows:
    print("%-20s %5d %5d %6.2f %8.4f %5d %6.1f %9.4f %11.4f"%r)
