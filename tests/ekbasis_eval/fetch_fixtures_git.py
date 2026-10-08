"""Sample the Git CHOICE questions of the Ekbasis git test sets and write fixtures/git_choice_sample.jsonl.

Source: HF dataset caiovicentino1/ekbasis-data (Apache-2.0 for the generated git rows; see its README). Files used, from
the tree API https://huggingface.co/api/datasets/caiovicentino1/ekbasis-data/tree/main/raw :
  raw/git3_test_known.jsonl, raw/git3_test_held.jsonl, raw/git2_test_known.jsonl, raw/git2_test_held.jsonl
(JSONL, read with the stdlib. data/git2/ and data/git3/ hold parquet files for the same configs; they need pyarrow,
which this harness does not use, and I did not compare them with the raw files.) Row keys, checked on the files:
ceiling, floor, gold, gold0, held_used, kinds, known0, qtype, question, steps, world.
The release's own request builder (release_eval.py, named in the client SPEC.md) was not found in the GitHub repo
or the HF model repo, so the mapping below rests on serve.py, decision_core.py, the C server and checks on the rows.

How a row becomes a decision request (quoted source lines):

1. README of the dataset, on the row fields:
   "`question` (the typed question as JSON: yes/no, or a choice with its options)"
   "...the answer before the actions, `gold0`)."
   Only the choice rows are used here: question.type == "choice" with qtype stash, content or branch. The git rows
   with question.type "noul" (lost, fails, exists, staged, clean, remote, in_progress) are the yes/no family, not
   sampled here.

2. State = `floor`, not `ceiling`. The ceiling is the hindsight prompt and states the answer. From a row of
   git3_test_known.jsonl (stash question):
   floor:   "...Stash entries: 0 ... Commands, in order:\n1. git checkout -- .\n2. git cherry-pick dev\n\nThe questions
             are about the state after all these commands."
   ceiling: "...Commands that were run:\n1. git checkout -- .\n2. git cherry-pick dev\n\nState after the commands:\n..."
   Checked on all 1,508 choice rows of the four files: "State after" never occurs in floor, and "State before" never
   occurs in ceiling. So `floor` (the state before the commands, plus the command list) does not reveal the answer.

3. Question -> /v1/systemone question. The row's question is already the server's shape:
   {"type": "choice", "instructions": ..., "criteria": {label: description, ...}} (criteria is a dict on all rows).
   Request body: {"state": floor, "questions": {"choice": question}}; one request per row, one question per request.
   Server side, decision_core.py (copied from Ekbasis-27B, MIT; tests/ekbasis_eval/decision_core.py):
     "    if isinstance(crit, dict):"
     "        keys = list(labels) if labels else list(crit.keys())"
     "        return [(k, _desc(crit.get(k), k)) for k in keys]"
   so the option order is the criteria dict's key order, and option i gets letter LABELS[i] (A, B, ...).
   q36_server.c at 608f304, sys_dict_options: "/* A criteria object: keys in document order, description = value
   (null shows the key). */"; for choice with no "options" member it uses the criteria, as the Python does.
   The rows carry no "options" member, so the criteria dict is the option list.
   Ekbasis-27B serve.py (HF model repo), _format, the answer the server returns:
     'return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}'
   where top = max(probs, key=probs.get) over the labels.
   No score questions occur in these four files (question.type is only "noul" or "choice"), so score is not mapped.

4. gold -> answer label. gold is a string equal to a criteria key on all 1,508 choice rows. It is the answer AFTER the
   commands, not gold0 (the answer before them, per the README). Checked by parsing the state text of each row:
     stash:   "Stash entries: N" in the after-state == gold (594 rows); the before-state == gold0 (594 rows).
     branch:  "Current branch: X" in the after-state == gold (302 rows); before-state == gold0 (302 rows).
     content: the file's "(value)" in "Files in the working directory (content): ..." after the commands == gold, or
              "(no such file)" when the file is not listed (612 rows, all match; gold0 matches the before-state too).
   Answer label = gold. expected_letter = LABELS[index of gold in the criteria key order]. A choice is correct when
   the server's "choice" equals gold.

Gold0 is not used for scoring. Rows whose gold is not a criteria key are refused (the script asserts it).

Sample: qtypes stash, content, branch x splits known, held: 6 cells of 66 rows = 396 rows, drawn with
random.Random(SEED) from the rows of both files of the split (the git3 and git2 files are pooled; duplicate requests
across them are kept once, the first seen). Files over 50 MB are refused (common.hf_get).

Usage:
  python3 fetch_fixtures_git.py                       # downloads into the cache (~/.cache/q36-ekbasis-eval)
  python3 fetch_fixtures_git.py --source-dir DIR      # offline: read DIR/git3_test_known.jsonl etc.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, hf_get, hf_tree  # noqa: E402
import decision_core as dc  # noqa: E402

REPO = "caiovicentino1/ekbasis-data"
FILES = {  # file name -> split
    "git3_test_known.jsonl": "known",
    "git3_test_held.jsonl": "held",
    "git2_test_known.jsonl": "known",
    "git2_test_held.jsonl": "held",
}
QTYPES = ("stash", "content", "branch")
PER_CELL = 66  # 3 qtypes x 2 splits x 66 = 396 rows (<= 400)
SEED = 20261008
QNAME = "choice"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "git_choice_sample.jsonl")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidates(sources):
    """(split, source name, row index, row) for every choice row of the three qtypes, first occurrence kept."""
    seen, out = set(), []
    for name, split, path in sources:
        for i, r in enumerate(load_rows(path)):
            q = r["question"]
            if q.get("type") != "choice" or r["qtype"] not in QTYPES:
                continue
            assert isinstance(q.get("criteria"), dict) and str(r["gold"]) in q["criteria"], (name, i)
            assert "State after" not in r["floor"], (name, i)  # the state must not contain the after-state
            key = json.dumps([r["floor"], q], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append((split, name, i, r))
    return out


def build_sample(sources, rng):
    pool = candidates(sources)
    out = []
    for split in ("known", "held"):
        for qtype in QTYPES:
            cell = [c for c in pool if c[0] == split and c[3]["qtype"] == qtype]
            assert len(cell) >= PER_CELL, (split, qtype, len(cell))
            for split_, name, i, r in sorted(rng.sample(cell, PER_CELL), key=lambda c: (c[1], c[2])):
                q = r["question"]
                labels = list(q["criteria"].keys())
                opts = dc.options_of(q)  # (label, description) in the order the server uses
                assert [lab for lab, _ in opts] == labels
                gold = str(r["gold"])
                out.append({
                    "id": f"git-{split}-{name[:-6]}-{i}",
                    "split": split,
                    "source": name,
                    "source_row": i,
                    "qtype": r["qtype"],
                    "gold": gold,                                  # answer after the commands; a criteria key
                    "expected_letter": dc.LABELS[labels.index(gold)],
                    "n_options": len(labels),
                    "request": {"state": r["floor"], "questions": {QNAME: q}},
                })
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join(CACHE, "ekbasis-data-raw"))
    ap.add_argument("--source-dir", help="offline: directory holding the four git*_test_*.jsonl files")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.source_dir:
        paths = {n: os.path.join(args.source_dir, n) for n in FILES}
    else:
        entries = hf_tree(REPO, dataset=True, sub="raw")
        paths = {n: hf_get(REPO, f"raw/{n}", entries[f"raw/{n}"], args.cache_dir, dataset=True) for n in FILES}
    sources = [(n, FILES[n], paths[n]) for n in FILES]
    rows = build_sample(sources, random.Random(SEED))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[(r["split"], r["qtype"])] = counts.get((r["split"], r["qtype"]), 0) + 1
    print(f"wrote {len(rows)} rows to {args.out}; {os.path.getsize(args.out)} bytes; cells {counts}")


if __name__ == "__main__":
    main()
