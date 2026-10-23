
import collections
rows = []
for line in open("/tmp/tl.txt"):
    if line.startswith("#"): continue
    p = line.split()
    if len(p) < 8: continue
    rows.append((p[2].split("/")[-1], int(p[4]), int(p[5]), int(p[6])))
rows.sort(key=lambda r: r[1])
n = len(rows)
span = (max(r[2] for r in rows) - min(r[1] for r in rows)) * 10
gpu = sum(r[3] for r in rows)

# sampling sync gaps: pairs top2 -> logits_minp_pack
sync = [ (b[1]-a[2])*10 for a,b in zip(rows, rows[1:]) if a[0].startswith("top2") and b[0].startswith("logits_minp_pack") ]
allgaps = [ (b[1]-a[2])*10 for a,b in zip(rows, rows[1:]) ]
print(f"tokens inferred from sampling syncs: {len(sync)}")
print(f"sync gaps: n={len(sync)} total {sum(sync)/1e6:.2f} ms  mean {sum(sync)/len(sync)/1000:.0f} us")
print(f"all gaps total {sum(allgaps)/1e6:.2f} ms over {len(allgaps)} transitions")
if len(sync):
    print(f"sampling sync share of all gap time: {sum(sync)/sum(allgaps)*100:.0f}%")
tok = len(sync)
print(f"per token: span {span/tok/1e6:.2f} ms  gpu {gpu/tok/1e6:.2f} ms  gap {sum(allgaps)/tok/1e6:.2f} ms")
