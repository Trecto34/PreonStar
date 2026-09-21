import re, sys

def ops(p):
    d = {}
    for l in open(p):
        m = re.match(r'q36:\s+op (\S+)\s+.*gpu_ms=([\d.]+)', l)
        if m:
            d[m.group(1)] = float(m.group(2))
        m = re.match(r'q36:\s+(\S+)\.spv\s+.*gpu_ms=([\d.]+)', l)
        if m:
            d['SPV:' + m.group(1)] = float(m.group(2))
    return d

a = ops('/tmp/qwen38-diff1.txt')
b = ops('/tmp/qwen38-diff65.txt')
d = {k: (b.get(k, 0) - a.get(k, 0)) / 64 for k in set(a) | set(b)}
tot = sum(v for k, v in d.items() if not k.startswith('SPV:'))
for k, v in sorted(d.items(), key=lambda x: -x[1]):
    if v > 0.005:
        print(f"{v:8.3f} ms/tok {100*v/tot:5.1f}%  {k}")
print(f"TOTAL GPU {tot:.2f} ms/tok")
for f in ('/tmp/qwen38-diff1.txt', '/tmp/qwen38-diff65.txt'):
    for l in open(f):
        if l.startswith('512,') or 'prefill_tps' in l:
            print(f.split('/')[-1], l.strip())
