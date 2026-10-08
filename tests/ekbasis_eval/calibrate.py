#!/usr/bin/env python3
"""Offline calibration study for the guard 'lost' question (stdlib only, no server needed).

  python3 -I calibrate.py --fit calib/responses.jsonl --eval eval200/responses.jsonl [--out calibration.md]

The server answers noul with p = P(yes) = sigmoid(z) where z is the yes-minus-no logit gap at temperature 1, so the
temperature-scaled probability is sigmoid(z / T) and can be recomputed from the recorded p without touching the model.
Everything that is FITTED (one temperature T, one decision threshold) is fitted on the --fit file only, which is
disjoint from the --eval file (see make_calib_set.py). The --eval file is scored once with the frozen values. Results
are reported next to the untouched baseline (T=1, threshold 0.5); nothing here changes the server.
"""
import argparse, json, math

BINS = 15


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("http_status") != 200:
            continue
        a = r["response"]["answers"]
        a = a.get("lost") or next(iter(a.values()))
        p = min(max(a["noul"], 1e-12), 1 - 1e-12)
        rows.append({"gold": bool(r["gold"]), "p": p, "z": math.log(p / (1 - p)), "split": r.get("split")})
    return rows


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def metrics(rows, T=1.0, thr=0.5):
    n = len(rows)
    ps = [sigmoid(r["z"] / T) for r in rows]
    nll = -sum(math.log(max(p if r["gold"] else 1 - p, 1e-12)) for r, p in zip(rows, ps)) / n
    brier = sum((p - r["gold"]) ** 2 for r, p in zip(rows, ps)) / n
    pred = [p >= thr for p in ps]
    acc = sum(a == r["gold"] for a, r in zip(pred, rows)) / n
    pos = [a for a, r in zip(pred, rows) if r["gold"]]
    neg = [a for a, r in zip(pred, rows) if not r["gold"]]
    fn = sum(1 for a in pos if not a)
    fp = sum(1 for a in neg if a)
    # ECE on the confidence of the predicted class, 15 equal-width bins over [0.5, 1] (as eval.py: over [0, 1])
    bins = [[] for _ in range(BINS)]
    for p, a, r in zip(ps, pred, rows):
        conf = p if a else 1 - p
        bins[min(int(conf * BINS), BINS - 1)].append((conf, a == r["gold"]))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(k for _, k in b) / len(b)) for b in bins if b)
    return {"n": n, "acc": acc, "nll": nll, "brier": brier, "ece": ece, "fn": fn, "n_pos": len(pos), "fp": fp,
            "n_neg": len(neg), "recall": (len(pos) - fn) / max(len(pos), 1), "spec": (len(neg) - fp) / max(len(neg), 1)}


def fit_T(rows):
    lo, hi = 0.2, 20.0  # golden-section search on NLL (convex in 1/T)
    f = lambda T: metrics(rows, T)["nll"]  # noqa: E731
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2


def fit_threshold(rows, T, target_recall):
    """Largest threshold on P(yes) (at temperature T) that still reaches target_recall on the fit set."""
    best = 0.0
    for thr in [i / 1000 for i in range(1, 1000)]:
        if metrics(rows, T, thr)["recall"] >= target_recall:
            best = thr
    return best


def fmt(m):
    return (f"acc {m['acc']*100:.1f}%  recall {m['recall']*100:.1f}% ({m['n_pos']-m['fn']}/{m['n_pos']})  "
            f"spec {m['spec']*100:.1f}% ({m['n_neg']-m['fp']}/{m['n_neg']})  NLL {m['nll']:.3f}  "
            f"Brier {m['brier']:.4f}  ECE {m['ece']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    fit, ev = load(a.fit), load(a.eval)
    T = fit_T(fit)
    thr95 = fit_threshold(fit, T, 0.95)
    thr98 = fit_threshold(fit, T, 0.98)
    lines = [f"fit set n={len(fit)}, eval set n={len(ev)} (disjoint). Fitted on the fit set only: T={T:.3f}, "
             f"threshold for >=95% recall {thr95:.3f}, for >=98% recall {thr98:.3f}.", "",
             "| Setting | Set | Metrics |", "|---|---|---|"]
    for name, TT, thr in (("baseline T=1, thr 0.5", 1.0, 0.5), (f"T={T:.2f}, thr 0.5", T, 0.5),
                          (f"T={T:.2f}, thr {thr95:.2f}", T, thr95), (f"T={T:.2f}, thr {thr98:.2f}", T, thr98)):
        lines.append(f"| {name} | fit | {fmt(metrics(fit, TT, thr))} |")
        lines.append(f"| {name} | eval | {fmt(metrics(ev, TT, thr))} |")
    lines += ["", "Reading: T rescales confidence only (accuracy is unchanged at threshold 0.5); the threshold trades "
              "specificity for recall on destructive commands. Values come from a fit set of this size; treat them "
              "as indicative, not as a shipped setting. The server still answers at T=1 like the official one."]
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
