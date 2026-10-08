"""Download a balanced sample of the Ekbasis guard set and write it to fixtures/guard_lost_sample.jsonl.

Source: HF dataset caiovicentino1/ekbasis-data (MIT/Apache/CC-BY mix; see its README). Files used, from the tree API
https://huggingface.co/api/datasets/caiovicentino1/ekbasis-data/tree/main/raw :
  raw/guard_known.jsonl (8.7 MB, known command types), raw/guard_held.jsonl (12.5 MB LFS, held-out command types).
Row keys: ceiling, floor, gold, gold0, held_used, kinds, known0, qtype, question, steps, world (verified on the file).

Sample: qtype "lost" only, question "Do these commands permanently lose any uncommitted work ...?" (type noul).
100 destructive + 100 safe, 50 + 50 from each split, drawn with random.Random(SEED). Files over 50 MB are refused.

Gold -> answer label (how eval.py scores a row):
  - gold is a JSON bool. gold true = the question's "yes" = destructive (work is lost). gold false = "no" = safe.
  - decision_core.options_of makes noul options [("yes","yes"), ("no","no")] in that order, so the letter is
    expected_letter = "A" when gold is true (yes), "B" when gold is false (no).
  - The server answers noul with "value" = P(yes) >= 0.5. A row is correct when value == gold.
State: the row's `floor` text (state before, plus the commands). NOT `ceiling`: the ceiling text is the hindsight
prompt and states the answer ("Uncommitted work lost for good: yes|no"). Checked on guard_known: that phrase matches
gold on all 1,726 lost rows, so using ceiling would leak the label. `floor` never contains the phrase.

Usage:
  python3 fetch_fixtures.py                       # downloads into the cache (~/.cache/q36-ekbasis-eval)
  python3 fetch_fixtures.py --source-dir DIR      # offline: read DIR/guard_known.jsonl and DIR/guard_held.jsonl
"""
import argparse
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, hf_get, hf_tree  # noqa: E402

REPO = "caiovicentino1/ekbasis-data"
FILES = ("guard_known.jsonl", "guard_held.jsonl")
SEED = 20261008
PER_CELL = 50  # per (split, gold): 2 splits x 2 golds x 50 = 200 rows
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "guard_lost_sample.jsonl")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sample(sources, rng):
    out = []
    for split, path in sources:
        rows = load_rows(path)
        for gold in (True, False):
            cands = [i for i, r in enumerate(rows) if r["qtype"] == "lost" and r["gold"] is gold]
            for i in sorted(rng.sample(cands, PER_CELL)):
                r = rows[i]
                assert r["question"]["type"] == "noul"
                out.append({
                    "id": f"guard-lost-{split}-{i}",
                    "split": split,                       # known = seen command types, held = held-out types
                    "destructive": gold,
                    "gold": gold,                          # JSON bool, see module docstring
                    "expected_label": "yes" if gold else "no",
                    "expected_letter": "A" if gold else "B",
                    "qtype": r["qtype"],
                    "world": r["world"],
                    "steps": r["steps"],
                    "kinds": r["kinds"],
                    "question": {"type": "noul", "instructions": r["question"]["instructions"]},
                    "state": r["floor"],
                    "source_row": i,
                })
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(CACHE, "ekbasis-data-raw"))
    ap.add_argument("--source-dir", help="offline: directory holding guard_known.jsonl and guard_held.jsonl")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.source_dir:
        paths = {n: os.path.join(args.source_dir, n) for n in FILES}
    else:
        entries = hf_tree(REPO, dataset=True, sub="raw")
        paths = {n: hf_get(REPO, f"raw/{n}", entries[f"raw/{n}"], args.cache_dir, dataset=True) for n in FILES}
    sources = [("known", paths["guard_known.jsonl"]), ("held", paths["guard_held.jsonl"])]
    rows = build_sample(sources, random.Random(SEED))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_d = sum(r["destructive"] for r in rows)
    print(f"wrote {len(rows)} rows ({n_d} destructive, {len(rows) - n_d} safe) to {args.out}; "
          f"{os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
