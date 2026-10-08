"""Evaluate a running q36-server POST /v1/systemone on the guard fixture sample (stdlib only).

  python3 eval.py --url http://127.0.0.1:8123 [--fixtures fixtures/guard_lost_sample.jsonl] [--out-dir DIR] [--limit N]

Each fixture row is sent as one request: state = row.state (the `floor` text), questions = {"lost": {"type": "noul",
"instructions": ...}}. Requests run one at a time (the server has one model and one GPU).

Scoring (the gold-to-label mapping is in the fetch_fixtures.py docstring):
  - The answer is answers["lost"] of the response. For noul it has "noul" (= P(yes)), "value" (= P(yes) >= 0.5),
    "confidence" (= max(P, 1-P)) and "probability" (= P(yes)).
  - Correct when value == gold (gold true = yes = destructive, work is lost).
  - Confidence for calibration is "confidence" from the server, in [0.5, 1] for a binary question.
  - ECE: 15 equal-width bins over [0, 1] on the confidence; ECE = sum_b (n_b / N) * |acc_b - conf_b|. Empty bins are
    skipped. Brier: mean (P(yes) - gold)^2 over answered rows.
  - Groups: overall, destructive (gold true), safe (gold false), known (split known), held (split held).
  - Requests that fail (HTTP error, timeout, malformed answer) are counted and left out of accuracy/ECE/Brier.
  - Latency: client wall-clock seconds for HTTP 200 responses, p50 and p95 (linear interpolation).

Outputs in --out-dir (default: CACHE/runs/<UTC time>):
  responses.jsonl  one line per request: the request sent (sha256 of its body), HTTP status, latency, the raw response
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

DEFAULT_FIXTURES = os.path.join(HERE, "fixtures", "guard_lost_sample.jsonl")
QNAME = "lost"
BINS = 15


def answer_of(response):
    """The noul answer dict, or None when the response does not have the documented shape."""
    try:
        a = response["answers"][QNAME]
        float(a["noul"]), bool(a["value"]), float(a["confidence"]), float(a["probability"])
        return a
    except (TypeError, KeyError, ValueError):
        return None


def ece_table(confs, correct, bins=BINS):
    """Reliability table over `bins` equal-width bins of confidence in [0, 1]."""
    buckets = [[] for _ in range(bins)]
    for conf, ok in zip(confs, correct):
        buckets[min(int(conf * bins), bins - 1)].append((conf, ok))
    n = len(confs)
    rows, ece = [], 0.0
    for b, items in enumerate(buckets):
        row = {"lo": round(b / bins, 4), "hi": round((b + 1) / bins, 4), "n": len(items)}
        if items:
            mc = sum(x for x, _ in items) / len(items)
            acc = sum(1 for _, ok in items if ok) / len(items)
            row.update(mean_confidence=mc, accuracy=acc)
            ece += len(items) / n * abs(acc - mc)
        else:
            row.update(mean_confidence=None, accuracy=None)
        rows.append(row)
    return ece, rows


def group_metrics(items):
    """items: list of dicts with p_yes, gold (bool), value (bool), conf. Returns the metric dict for one group."""
    n = len(items)
    if n == 0:
        return {"n": 0}
    correct = [it["value"] == it["gold"] for it in items]
    confs = [it["conf"] for it in items]
    ece, _ = ece_table(confs, correct)
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "mean_confidence": sum(confs) / n,
        "ece": ece,
        "brier": sum((it["p_yes"] - float(it["gold"])) ** 2 for it in items) / n,
        "predicted_yes_rate": sum(1 for it in items if it["value"]) / n,
    }


def score(answered):
    """answered: list of dicts (see group_metrics). Returns the groups and the reliability table."""
    groups = {
        "overall": answered,
        "destructive": [x for x in answered if x["gold"]],
        "safe": [x for x in answered if not x["gold"]],
        "known": [x for x in answered if x["split"] == "known"],
        "held": [x for x in answered if x["split"] == "held"],
    }
    metrics = {k: group_metrics(v) for k, v in groups.items()}
    _, table = ece_table([x["conf"] for x in answered], [x["value"] == x["gold"] for x in answered])
    return metrics, table


def run(args):
    with open(args.fixtures, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out = args.out_dir or os.path.join(c.CACHE, "runs", datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(out, exist_ok=True)
    answered, latencies, failures, model_name = [], [], [], None
    with open(os.path.join(out, "responses.jsonl"), "w", encoding="utf-8") as log:
        for i, r in enumerate(rows, 1):
            body = c.build_request(r["state"], [(QNAME, r["question"])])
            body_text = json.dumps(body)
            sent = datetime.datetime.now(datetime.timezone.utc).isoformat()
            rec = {"id": r["id"], "split": r["split"], "gold": r["gold"], "sent_at": sent,
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
                pyes = float(a["probability"])
                answered.append({"id": r["id"], "split": r["split"], "gold": bool(r["gold"]), "p_yes": pyes,
                                 "value": bool(a["value"]), "conf": float(a["confidence"]),
                                 "contract_ok": abs(float(a["noul"]) - pyes) < 1e-9 and
                                 bool(a["value"]) == (pyes >= 0.5) and
                                 abs(float(a["confidence"]) - max(pyes, 1 - pyes)) < 1e-6})
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
             "| Group | n | Accuracy | Mean confidence | ECE (15 bins) | Brier | P(yes) rate |",
             "|---|---|---|---|---|---|---|"]
    for k in ("overall", "destructive", "safe", "known", "held"):
        g = m[k]
        lines.append(f"| {k} | {g['n']} | {fmt(g.get('accuracy'), True)} | {fmt(g.get('mean_confidence'))} | "
                     f"{fmt(g.get('ece'))} | {fmt(g.get('brier'))} | {fmt(g.get('predicted_yes_rate'), True)} |")
    lines += ["", f"Latency (HTTP 200, n={lat['n']}): p50 {fmt(lat['p50'])} s, p95 {fmt(lat['p95'])} s.", "",
              "Reliability (confidence = max(P(yes), P(no)) per bin):", "",
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
