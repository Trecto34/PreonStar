#!/usr/bin/env python3
"""Compare two q36-bench --dump-frontier-logits-dir outputs (ref dir, new dir).
Prints max_abs_diff, top-1 agreement and top-64 overlap per frontier file."""
import json, os, sys

ref_dir, new_dir = sys.argv[1], sys.argv[2]
for name in sorted(os.listdir(ref_dir)):
    if not name.endswith(".logits.json"):
        continue
    a = json.load(open(os.path.join(ref_dir, name)))["logits"]
    b = json.load(open(os.path.join(new_dir, name)))["logits"]
    assert len(a) == len(b), (len(a), len(b))
    i = max(range(len(a)), key=lambda k: abs(a[k] - b[k]))
    top = lambda v: sorted(range(len(v)), key=v.__getitem__, reverse=True)[:64]
    ta, tb = top(a), top(b)
    print(f"{name}: max_abs_diff={abs(a[i] - b[i]):.6g} at {i}  "
          f"top1 ref={ta[0]} ({a[ta[0]]:.6f}) new={tb[0]} ({b[tb[0]]:.6f}) same={ta[0] == tb[0]}  "
          f"top64 overlap={len(set(ta) & set(tb))}/64")
