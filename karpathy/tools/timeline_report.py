
import collections
rows = []
for line in open(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "/tmp/tl.txt"):
    if line.startswith("#"): continue
    p = line.split()
    if len(p) < 8: continue
    seq, flush, path, groups, st, en, dur, host = int(p[0]), int(p[1]), p[2], int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7])
    rows.append((path, groups, st, en, dur, host))

rows.sort(key=lambda r: r[2])
print("dispatches:", len(rows))
gpu_ns = sum(r[4] for r in rows)
# timeline span by device clock
span = (max(r[3] for r in rows) - min(r[2] for r in rows)) * 10
print(f"device span {span/1e6:.1f} ms   sum of dispatch durations {gpu_ns/1e6:.1f} ms   busy {gpu_ns/span*100:.1f}%")

# gaps between consecutive dispatches
gaps = []
for a, b in zip(rows, rows[1:]):
    g = (b[2] - a[3]) * 10
    gaps.append((g, b[0], a[0]))
gaps.sort(reverse=True)
print(f"total gap {sum(g[0] for g in gaps)/1e6:.1f} ms  ({(1-gpu_ns/span)*100:.1f}% of span)")
print("largest gaps (us, after -> before):")
for g, nxt, prv in gaps[:6]:
    print(f"  {g/1000:8.1f} us  {prv.split('/')[-1]} -> {nxt.split('/')[-1]}")

per = collections.Counter()
for path, groups, st, en, dur, host in rows:
    per[path] += dur
print("\nGPU ms by kernel, top:")
for k, v in per.most_common(8):
    print(f"  {v/1e6:7.2f} ms  {v/gpu_ns*100:5.1f}%  {k.split('/')[-1]}")
