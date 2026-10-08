#!/usr/bin/env python3
"""Build fixtures/guard_lost_calib.jsonl: a calibration set DISJOINT from fixtures/guard_lost_sample.jsonl.

Same source, schema and balance as fetch_fixtures.py (qtype 'lost', state = floor, 50 per split x gold), but a different
seed, and every source row used by the evaluation sample is excluded. Anything fitted on this set (temperature, decision
threshold) is therefore never fitted on evaluation data. Uses the cached raw files (run fetch_fixtures.py first).
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_fixtures as ff  # noqa: E402
from common import CACHE  # noqa: E402

CALIB_SEED = 7_310_042
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cache = os.path.join(CACHE, "ekbasis-data-raw")
    used = {(json.loads(l)["split"], json.loads(l)["source_row"])
            for l in open(os.path.join(HERE, "fixtures", "guard_lost_sample.jsonl"), encoding="utf-8") if l.strip()}
    rng = random.Random(CALIB_SEED)
    out = []
    for split, name in (("known", "guard_known.jsonl"), ("held", "guard_held.jsonl")):
        rows = ff.load_rows(os.path.join(cache, name))
        for gold in (True, False):
            cands = [i for i, r in enumerate(rows)
                     if r["qtype"] == "lost" and r["gold"] is gold and (split, i) not in used]
            for i in sorted(rng.sample(cands, ff.PER_CELL)):
                r = rows[i]
                out.append({"id": f"guard-lost-calib-{split}-{i}", "split": split, "destructive": gold, "gold": gold,
                            "expected_label": "yes" if gold else "no", "expected_letter": "A" if gold else "B",
                            "qtype": r["qtype"], "world": r["world"], "steps": r["steps"], "kinds": r["kinds"],
                            "question": {"type": "noul", "instructions": r["question"]["instructions"]},
                            "state": r["floor"], "source_row": i})
    rng.shuffle(out)
    path = os.path.join(HERE, "fixtures", "guard_lost_calib.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ev = {(r["split"], r["source_row"]) for r in out}
    assert not (ev & used), "calibration set overlaps the evaluation sample"
    print(f"wrote {len(out)} rows to {path}; overlap with eval sample: 0")


if __name__ == "__main__":
    main()
