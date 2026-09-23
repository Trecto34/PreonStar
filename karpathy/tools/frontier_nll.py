#!/usr/bin/env python3
"""Teacher-forced quality from q36-bench frontier dumps.

Usage: frontier_nll.py TOKENS_JSON DIR [DIR ...]
TOKENS_JSON is `q36 --dump-tokens` output for the same --prompt-file.  For each
frontier_NNNNNN.logits.json the true next token is tokens[NNNNNN]; prints mean
NLL, top-1 accuracy, and the mean |NLL delta| vs the first DIR."""
import json, math, os, re, sys

tokens = json.loads(open(sys.argv[1]).readline())  # first line is the id list
dirs = sys.argv[2:]
per = {}
for d in dirs:
    rows = {}
    for name in sorted(os.listdir(d)):
        m = re.match(r"frontier_(\d+)\.logits\.json$", name)
        if not m:
            continue
        n = int(m.group(1))
        lg = json.load(open(os.path.join(d, name)))["logits"]
        mx = max(lg)
        lse = mx + math.log(sum(math.exp(x - mx) for x in lg))
        t = tokens[n]
        rows[n] = (lse - lg[t], max(range(len(lg)), key=lg.__getitem__) == t)
    per[d] = rows
ref = per[dirs[0]]
for d in dirs:
    rows = per[d]
    nll = sum(v[0] for v in rows.values()) / len(rows)
    acc = sum(v[1] for v in rows.values())
    dv = sum(abs(rows[n][0] - ref[n][0]) for n in rows) / len(rows)
    print(f"{d}: frontiers={len(rows)} mean_nll={nll:.4f} top1={acc}/{len(rows)} mean|dNLL| vs ref={dv:.4f}")
