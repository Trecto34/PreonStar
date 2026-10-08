"""Regenerate the Python-reference goldens for POST /v1/systemone (q36_server.c unit tests).

usage: python3 -I gen_golden.py <dir containing decision_core.py> <output golden dir>

Requires decision_core.py from the Ekbasis serve sources (MIT). Only stdlib is used.
Each *.user.txt / *.followup.txt is decision_core.render_user / render_followup output; each
*.answers.txt is serve.py _format + json.dumps(ensure_ascii=False) for the probabilities in *.probs.txt.
"""
import json
import os
import sys

src_dir, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(src_dir))  # -I does not add the script directory either
import decision_core as dc  # noqa: E402

CASES = {
    "a": r'''{"state": {"name": "Ana \"Q\"\n", "tags": ["ação", "日本"], "n": 3, "x": 1.50,
 "z": null, "t": true, "obj": {"k": [1, 2.5e3]}}, "questions": {
 "color": {"type": "choice", "instructions": "Pick the colour: \t ok",
           "criteria": {"red": "warm / bright", "blue": null, "green": "cool\u0001x"}},
 "ok": {"type": "noul", "instructions": "Is it ok?", "criteria": {"true": "it is fine", "false": null}}}}''',
    "b": r'''{"state": "line1\nline2 é ⁣ end\t\"q\"\\", "questions": {
 "rate": {"type": "score", "instructions": "Rate it", "criteria": ["terrible", "ok", "great"]},
 "pick": {"type": "choice", "instructions": "Pick one", "options": ["x y", "z"]}}}''',
    "c": r'''{"state": [1, "a", {}, [], -0.0, 1e100], "read_once": true, "questions": {
 "ok2": {"type": "boolean", "instructions": "yes-ish?", "criteria": {"true": "yes-ish", "false": "no-ish"}},
 "how": {"type": "score", "instructions": "how much?", "criteria": {"0": "none", "1": null, "2": "all"}},
 "hm": {"type": "choice", "instructions": "hm", "criteria": {"p": "one", "q": 2, "r": false}}}}''',
}

# Probabilities per question (decimal literals; the C test reads the same numbers back with strtod).
PROBS = {
    "a": {"color": [0.2, 0.5, 0.3], "ok": [0.7, 0.3]},
    "b": {"rate": [0.1, 0.2, 0.7], "pick": [0.6, 0.4]},
    "c": {"ok2": [0.45, 0.55], "how": [0.05, 0.15, 0.8], "hm": [0.3, 0.3, 0.4]},
}

# serve.py _format, copied (serve.py imports torch at module level).
def fmt(q, probs):
    t = q.get("type")
    top = max(probs, key=probs.get)
    if t in ("noul", "boolean"):
        return {"type": t, "noul": probs["yes"], "probability": probs["yes"], "value": probs["yes"] >= 0.5,
                "confidence": max(probs.values())}
    if t == "score":
        return {"type": "score", "probabilities": probs, "score": int(top),
                "expected": sum(float(k) * v for k, v in probs.items()), "confidence": probs[top]}
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}


def write(name, text):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


os.makedirs(out_dir, exist_ok=True)
write("labels.txt", "".join(lab + "\n" for lab in dc.LABELS))
write("floatrepr.txt", "".join(f"{s}\t{repr(float(s))}\n" for s in [
    "0.1", "0.25", "1e-05", "1e-4", "0.0001", "1e16", "9999999999999998.0", "123456.789", "1.5e-07",
    "5e-324", "1.7976931348623157e308", "0.30000000000000004", "100000.0", "-0.0", "0.5", "1.0", "1e15",
    "0.9999999999999999", "2.5e-05", "3.0"]))

for cid, raw in CASES.items():
    write(f"{cid}.request.json", raw + "\n")
    body = json.loads(raw)
    state, questions = body["state"], body["questions"]
    answers, probs_lines = {}, []
    for qname, q in questions.items():
        opts = dc.options_of(q)
        write(f"{cid}.{qname}.user.txt", dc.render_user(state, q, opts))
        write(f"{cid}.{qname}.followup.txt", dc.render_followup(q, opts))
        probs = dict(zip([lab for lab, _ in opts], PROBS[cid][qname]))
        probs_lines.append(qname + " " + " ".join(repr(p) for p in PROBS[cid][qname]) + "\n")
        answers[qname] = fmt(q, probs)
    write(f"{cid}.probs.txt", "".join(probs_lines))
    write(f"{cid}.answers.txt", json.dumps(answers, ensure_ascii=False))

print("golden files written to", out_dir)
