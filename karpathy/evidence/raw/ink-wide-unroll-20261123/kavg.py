#!/usr/bin/env python3
# usage: kavg.py timeline.txt [substr...] -> per-kernel mean us/dispatch, count; plus per-token GPU ms (top2 count = tokens)
import sys, collections
t = collections.defaultdict(lambda: [0, 0])
for l in open(sys.argv[1]):
    if l[0] == '#': continue
    p = l.split()
    k = p[2].split('/')[-1]; t[k][0] += int(p[6]); t[k][1] += 1
toks = t.get('top2.spv', [0, 1])[1] or 1
tot = sum(v[0] for v in t.values())
print(f"tokens={toks} gpu_ms/tok={tot/toks/1e6:.3f}")
for k, (ns, n) in sorted(t.items(), key=lambda kv: -kv[1][0]):
    if len(sys.argv) > 2 and not any(s in k for s in sys.argv[2:]): continue
    print(f"{ns/n/1e3:9.2f} us x{n//toks:4d}/tok  {ns/toks/1e6:6.3f} ms/tok  {k}")
