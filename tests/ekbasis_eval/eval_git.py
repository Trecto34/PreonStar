"""Evaluate a running server's POST /v1/systemone on the git CHOICE sample (stdlib only).

  python3 eval_git.py --url http://127.0.0.1:8123 [--fixtures fixtures/git_choice_sample.jsonl] [--out-dir DIR]
                      [--limit N] [--timeout S]

Each fixture row is sent as its own request, row.request = {"state": floor, "questions": {"choice": {...}}}, unchanged.
The mapping from the dataset row is documented in fetch_fixtures_git.py (state = floor; gold = answer after the
commands, a criteria key). Requests run one at a time (the server has one model and one GPU).

Scoring:
  - The answer is answers["choice"]. It has "choice" (the predicted label), "probabilities" (label -> probability over
    the criteria keys) and "confidence" (the server's, = probabilities[choice] = the largest probability).
  - Contract checks, counted and not scored: "choice" is the largest probability; the probability keys are exactly the
    criteria keys; the probabilities sum to 1 (tolerance 1e-3); "confidence" equals probabilities[choice].
  - Correct when choice == gold.
  - ECE: 15 equal-width bins over the confidence (eval.ece_table, the same routine as the guard eval).
  - Brier (gold-label term, as asked): mean over answered rows of (1 - P(gold))^2, where P(gold) = probabilities[gold].
    This is only the gold class term of the multi-class Brier score, not the sum over all labels.
  - Groups: overall, each qtype (stash, content, branch), split known, split held, and qtype x split.
  - Requests that fail (HTTP error, timeout, malformed answer) are counted and left out of accuracy/ECE/Brier.
  - Latency: client wall-clock seconds for HTTP 200 responses, p50 and p95.
  - No score questions are in the sample (the source has only noul and choice), so only choice is scored here.

Outputs in --out-dir (default: CACHE/runs/<UTC time>_git):
  responses.jsonl  one line per request: id, split, qtype, gold, request sha256, HTTP status, latency, the raw response
                   text exactly as received ("raw") and its parsed JSON ("response", null when not JSON).
  results.json     counts, groups, reliability table, latency, the server's model name, the run configuration.
  results.md       the same numbers as markdown tables; also printed to stdout.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import eval as ev  # noqa: E402  (ece_table is shared with the guard eval)

DEFAULT_FIXTURES = os.path.join(HERE, "fixtures", "git_choice_sample.jsonl")
QNAME = "choice"
BINS = ev.BINS
QTYPES = ("stash", "content", "branch")
TOL = 1e-3


def answer_of(response):
    """The choice answer dict, or None when the response does not have the documented shape."""
    try:
        a = response["answers"][QNAME]
        if a["type"] != "choice" or not isinstance(a["probabilities"], dict):
            return None
        float(a["confidence"])
        return a
    except (TypeError, KeyError, ValueError):
        return None


def contract_violations(a, criteria_keys):
    """Problems with one answer, as a list of strings (empty when it follows the documented contract)."""
    probs = {str(k): float(v) for k, v in a["probabilities"].items()}
    bad = []
    if set(probs) != set(criteria_keys):
        bad.append("probability keys differ from criteria keys")
    if abs(sum(probs.values()) - 1.0) > TOL:
        bad.append("probabilities do not sum to 1")
    if a["choice"] not in probs or probs[a["choice"]] < max(probs.values()) - 1e-12:
        bad.append("choice is not the largest probability")
    elif abs(float(a["confidence"]) - probs[a["choice"]]) > 1e-6:
        bad.append("confidence differs from probabilities[choice]")
    return bad


def group_metrics(items):
    """items: dicts with gold, pred (str), conf (float), p_gold (float). Metric dict for one group."""
    n = len(items)
    if n == 0:
        return {"n": 0}
    correct = [it["pred"] == it["gold"] for it in items]
    confs = [it["conf"] for it in items]
    ece, _ = ev.ece_table(confs, correct)
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "mean_confidence": sum(confs) / n,
        "ece": ece,
        "brier": sum((1.0 - it["p_gold"]) ** 2 for it in items) / n,
    }


def score(answered):
    """Groups and the reliability table for the whole set."""
    groups = {"overall": answered}
    for q in QTYPES:
        groups[q] = [x for x in answered if x["qtype"] == q]
    groups["known"] = [x for x in answered if x["split"] == "known"]
    groups["held"] = [x for x in answered if x["split"] == "held"]
    for split in ("known", "held"):
        for q in QTYPES:
            groups[f"{q} / {split}"] = [x for x in answered if x["qtype"] == q and x["split"] == split]
    metrics = {k: group_metrics(v) for k, v in groups.items()}
    _, table = ev.ece_table([x["conf"] for x in answered], [x["pred"] == x["gold"] for x in answered])
    return metrics, table


def run(args):
    with open(args.fixtures, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out = args.out_dir or os.path.join(
        c.CACHE, "runs", datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_git")
    os.makedirs(out, exist_ok=True)
    answered, latencies, failures, model_name, contract_bad = [], [], [], None, []
    with open(os.path.join(out, "responses.jsonl"), "w", encoding="utf-8") as log:
        for i, r in enumerate(rows, 1):
            body = r["request"]
            body_text = json.dumps(body)
            sent = datetime.datetime.now(datetime.timezone.utc).isoformat()
            rec = {"id": r["id"], "split": r["split"], "qtype": r["qtype"], "gold": r["gold"], "sent_at": sent,
                   "request_sha256": hashlib.sha256(body_text.encode("utf-8")).hexdigest()}
            try:
                status, raw, dt = c.post_systemone(args.url, body, timeout=args.timeout)
            except OSError as e:  # refused, timeout, reset
                status, raw, dt = None, None, None
                rec["error"] = f"{type(e).__name__}: {e}"
            rec.update(http_status=status, latency_s=dt, raw=raw)
            parsed = None
            if raw is not None:
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
            rec["response"] = parsed
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if status == 200 and dt is not None:
                latencies.append(dt)
            a = answer_of(parsed) if status == 200 else None
            if a is None:
                failures.append({"id": r["id"], "http_status": status, "error": rec.get("error")})
            else:
                model_name = model_name or (parsed.get("model") if isinstance(parsed, dict) else None)
                probs = {str(k): float(v) for k, v in a["probabilities"].items()}
                problems = contract_violations(a, list(r["request"]["questions"][QNAME]["criteria"].keys()))
                contract_bad += [{"id": r["id"], "problem": p} for p in problems]
                answered.append({"id": r["id"], "split": r["split"], "qtype": r["qtype"], "gold": r["gold"],
                                 "pred": str(a["choice"]), "conf": float(a["confidence"]),
                                 "p_gold": probs.get(r["gold"], 0.0), "contract_ok": not problems})
            if i % 25 == 0:
                print(f"{i}/{len(rows)} done", file=sys.stderr)
    metrics, table = score(answered)
    results = {
        "run": {"url": args.url, "fixtures": os.path.abspath(args.fixtures), "rows": len(rows),
                "timeout_s": args.timeout, "bins": BINS, "temperature": c.TEMPERATURE,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        "server_model": model_name,
        "counts": {"sent": len(rows), "answered": len(answered), "failed": len(failures),
                   "contract_violations": sum(1 for x in answered if not x["contract_ok"])},
        "contract_problems": contract_bad[:50],
        "metrics": metrics,
        "reliability": table,
        "latency_s": {"n": len(latencies), "p50": c.percentile(latencies, 50), "p95": c.percentile(latencies, 95)},
        "failures": failures[:50],
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
        f.write("\n")
    md = to_markdown(results)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}/responses.jsonl, results.json, results.md")


def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.4f}"


def to_markdown(res):
    m, lat, cnt = res["metrics"], res["latency_s"], res["counts"]
    lines = [f"Server model: `{res['server_model']}`; sent {cnt['sent']}, answered {cnt['answered']}, "
             f"failed {cnt['failed']}, contract violations {cnt['contract_violations']}.", "",
             "| Group | n | Accuracy | Mean confidence | ECE (15 bins) | Brier (gold term) |",
             "|---|---|---|---|---|---|"]
    for k in m:
        g = m[k]
        lines.append(f"| {k} | {g['n']} | {fmt(g.get('accuracy'), True)} | {fmt(g.get('mean_confidence'))} | "
                     f"{fmt(g.get('ece'))} | {fmt(g.get('brier'))} |")
    lines += ["", f"Latency (HTTP 200, n={lat['n']}): p50 {fmt(lat['p50'])} s, p95 {fmt(lat['p95'])} s.", "",
              "Reliability (confidence = probability of the predicted label, per bin):", "",
              "| Bin | n | Mean confidence | Accuracy |", "|---|---|---|---|"]
    for b in res["reliability"]:
        lines.append(f"| [{b['lo']:.2f}, {b['hi']:.2f}) | {b['n']} | {fmt(b['mean_confidence'])} | "
                     f"{fmt(b['accuracy'], True)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=c.DEFAULT_URL)
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    ap.add_argument("--out-dir")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    ap.add_argument("--timeout", type=float, default=600.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
