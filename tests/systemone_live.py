#!/usr/bin/env python3
"""Live contract checks for POST /v1/systemone (alias /v1/evaluate) on a running q36-server.

Needs a loaded model and a running server; uses only the standard library.

  tests/systemone_live.py --server http://127.0.0.1:8000
  options: --max-readonce-diff 0.25   (largest allowed |p(read_once) - p(one prompt)| per option)

Checks: response shape and answer fields, probabilities sum to 1, noul value/confidence rules, the /v1/evaluate
alias, read_once on/off agreement, and the 422 error cases. Exits 1 on any failed check.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL:", msg)
    else:
        print("ok:  ", msg)


def post(server, path, body, raw=False):
    data = body if raw else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


QUESTIONS = {
    "colour": {"type": "choice", "instructions": "Which colour is the sky usually?",
               "criteria": {"red": "warm", "blue": "clear sky", "green": "grass"}},
    "rate": {"type": "score", "instructions": "How likely is rain? Rate 0 to 2.",
             "criteria": ["unlikely", "possible", "certain"]},
    "umbrella": {"type": "noul", "instructions": "Should I take an umbrella?",
                 "criteria": {"true": "rain is likely", "false": "dry day"}},
}
STATE = {"weather": "clouds all morning, forecast 80% rain", "place": "Lisbon"}


def check_answers(ans, qs):
    for name, q in qs.items():
        a = ans.get(name)
        check(isinstance(a, dict), f"answer present for {name}")
        if not isinstance(a, dict):
            continue
        t = q["type"]
        check(a.get("type") == t, f"{name}: type is {t}")
        if t == "noul":
            check(0.0 <= a["noul"] <= 1.0 and a["probability"] == a["noul"], f"{name}: noul equals probability in [0,1]")
            check(a["value"] == (a["noul"] >= 0.5), f"{name}: value is noul >= 0.5")
            check(abs(a["confidence"] - max(a["noul"], 1 - a["noul"])) < 1e-9, f"{name}: confidence is max(p, 1-p)")
        else:
            probs = a["probabilities"]
            check(abs(sum(probs.values()) - 1.0) < 1e-6, f"{name}: probabilities sum to 1")
            top = max(probs, key=probs.get)
            check(abs(a["confidence"] - probs[top]) < 1e-12, f"{name}: confidence is the top probability")
            if t == "choice":
                check(a["choice"] == top, f"{name}: choice is the top option")
            else:
                check(a["score"] == int(top) and abs(a["expected"] - sum(float(k) * v for k, v in probs.items())) < 1e-9,
                      f"{name}: score is the top label and expected is its weighted sum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--max-readonce-diff", type=float, default=0.25)
    a = ap.parse_args()
    s = a.server.rstrip("/")

    body = {"state": STATE, "questions": QUESTIONS}
    code, res = post(s, "/v1/systemone", body)
    check(code == 200, f"POST /v1/systemone -> 200 (got {code})")
    check(set(res) >= {"model", "answers", "usage"}, "response has model, answers, usage")
    check(res.get("usage", {}).get("output_tokens") == 0 and res.get("usage", {}).get("input_tokens", 0) > 0,
          "usage: output_tokens 0, input_tokens > 0")
    check(list(res.get("answers", {})) == list(QUESTIONS), "answers keep the question order")
    check_answers(res.get("answers", {}), QUESTIONS)
    one_prompt = res.get("answers", {})

    code2, res2 = post(s, "/v1/evaluate", body)
    check(code2 == 200 and res2.get("answers") == one_prompt, "/v1/evaluate gives the same answers as /v1/systemone")

    code3, ro = post(s, "/v1/systemone", {**body, "read_once": True})
    check(code3 == 200, f"read_once request -> 200 (got {code3})")
    if code3 == 200:
        check_answers(ro["answers"], QUESTIONS)
        worst = 0.0
        for name, q in QUESTIONS.items():
            if q["type"] == "noul":
                worst = max(worst, abs(one_prompt[name]["noul"] - ro["answers"][name]["noul"]))
            else:
                pa, pb = one_prompt[name]["probabilities"], ro["answers"][name]["probabilities"]
                worst = max(worst, max(abs(pa[k] - pb[k]) for k in pa))
        print(f"info: largest |p(read_once) - p(one prompt)| = {worst:.4f}")
        check(worst <= a.max_readonce_diff, f"read_once agrees with one-prompt answers within {a.max_readonce_diff}")

    one_q = {"state": STATE, "read_once": True, "questions": {"colour": QUESTIONS["colour"]}}
    code4, res4 = post(s, "/v1/systemone", one_q)
    check(code4 == 200 and list(res4.get("answers", {})) == ["colour"], "read_once with one question is ignored (200)")

    # Official serve.py behaviour: defaults are accepted, validation errors are 422 with {"error": "<message>"}.
    code_d, res_d = post(s, "/v1/systemone", {"questions": QUESTIONS})
    check(code_d == 200 and set(res_d.get("answers", {})) == set(QUESTIONS), "state missing -> defaults to '' (200)")
    code_d, res_d = post(s, "/v1/systemone", {"state": STATE})
    check(code_d == 200 and res_d.get("answers") == {}, "questions missing -> 200 with no answers")
    code_d, res_d = post(s, "/v1/systemone", {"state": STATE, "questions": {"q": {"type": "ranking", "criteria": ["a", "b"]}}})
    check(code_d == 200 and res_d["answers"]["q"].get("type") == "choice", "unknown type is a choice (200)")
    code_d, res_d = post(s, "/v1/systemone", one_q)
    check(isinstance(res_d.get("latency_s"), float) and res_d["latency_s"] > 0, "latency_s is returned")
    check(list(res_d) == ["model", "answers", "usage", "latency_s"], f"response keys in serve.py order (got {list(res_d)})")
    errors = [
        ({"state": STATE, "questions": {"q": {"type": "choice", "criteria": ["only one"]}}}, "one option"),
        ({"state": STATE, "questions": {"q": {"type": "choice", "criteria": [str(i) for i in range(161)]}}}, "161 options"),
        ({"state": STATE, "questions": {f"q{i}": {"type": "noul"} for i in range(65)}}, "65 questions"),
        ({"state": STATE, "images": ["x"], "questions": QUESTIONS}, "images"),
        ({"state": "a\u0000b", "questions": QUESTIONS}, "NUL in state"),
    ]
    for payload, what in errors:
        code_e, res_e = post(s, "/v1/systemone", payload)
        check(code_e == 422 and isinstance(res_e.get("error"), str), f"error case '{what}' -> 422 with string error (got {code_e})")
    code_b, res_b = post(s, "/v1/systemone", b"{not json", raw=True)
    check(code_b == 422 and isinstance(res_b.get("error"), str), f"malformed JSON -> 422 (got {code_b})")

    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("systemone live checks: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
