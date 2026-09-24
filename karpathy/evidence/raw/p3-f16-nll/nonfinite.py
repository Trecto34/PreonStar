import json, math, os, re, sys
bad = 0; files = 0; worst = 0.0
for d in sys.argv[1:]:
    for name in sorted(os.listdir(d)):
        if not name.endswith('.logits.json'): continue
        files += 1
        lg = json.load(open(os.path.join(d, name)))["logits"]
        for v in lg:
            if not math.isfinite(v):
                bad += 1; print("NONFINITE", d, name, v); break
        m = max(abs(x) for x in lg)
        worst = max(worst, m)
print(f"dirs={len(sys.argv)-1} files={files} nonfinite_values={bad} max|logit|={worst:.3f}")
