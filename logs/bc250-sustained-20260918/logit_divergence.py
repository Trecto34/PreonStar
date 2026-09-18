#!/usr/bin/env python3
"""Tier-1 quality signal for a quantization change, where bit-exact parity
cannot apply.

The campaign's contract for a *kernel* change is max_abs_diff = 0. A format
change (Q2_0-g64 -> PQ2_0/PTQ1_0) alters the weights themselves, so that test
is meaningless. This reports how far the output distribution moved instead:

  top-1 agreement   did the greedy token change?
  top-5 overlap     did the candidate set move?
  KL(base||cand)    distribution shift, in nats, over the full vocab
  max/mean |dlogit| raw scale of the move
  rms / p50 / p99 / p99.9 |dlogit|

KL is computed on softmax-NORMALIZED PROBABILITIES, not on raw logits, in the
direction KL(base || cand) = sum_i p_i * log(p_i / q_i).  softmax() subtracts
the row max before exp() and runs in Python float (IEEE double), so a term
underflows to zero only for a logit gap beyond ~745.  Terms where either p or q
underflowed are excluded from the sum and reported as `kl_skipped`; a nonzero
count means the KL is a lower bound and must not be quoted as exact.

Non-finite inputs are counted and reported rather than silently poisoning the
statistics: any nan/inf makes the comparison invalid.

Usage: logit_divergence.py BASE_DIR CAND_DIR
where each DIR holds frontier_*.logits.json from --dump-frontier-logits-dir.
"""
import glob, json, math, os, sys


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def pct(sorted_d, q):
    # nearest-rank, no interpolation: these are error magnitudes, not a fitted
    # distribution, so an actual observed value is the honest answer.
    if not sorted_d:
        return float("nan")
    i = min(len(sorted_d) - 1, max(0, int(math.ceil(q * len(sorted_d))) - 1))
    return sorted_d[i]


def compare(pa, pb, expect_n=None):
    ja, jb = json.load(open(pa)), json.load(open(pb))
    a, b = ja["logits"], jb["logits"]
    if len(a) != len(b):
        raise SystemExit(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    # The vocab each dump recorded is the authority on how many logits it should
    # hold; never take the observed length as self-evidently right.
    for j, p in ((ja, pa), (jb, pb)):
        v = j.get("vocab")
        if v is not None and v != len(j["logits"]):
            raise SystemExit(f"{p}: vocab={v} but {len(j['logits'])} logits")
    if expect_n is not None and n != expect_n:
        raise SystemExit(f"element count {n} != expected {expect_n}")
    nonfinite = sum(1 for v in a if not math.isfinite(v)) + \
                sum(1 for v in b if not math.isfinite(v))
    nans = sum(1 for v in a + b if math.isnan(v))
    infs = sum(1 for v in a + b if math.isinf(v))
    if nonfinite:
        raise SystemExit(f"{nonfinite} non-finite logits ({nans} nan, {infs} inf) "
                         "- the comparison is meaningless, fix the run first")
    ra = sorted(range(n), key=a.__getitem__, reverse=True)[:5]
    rb = sorted(range(n), key=b.__getitem__, reverse=True)[:5]
    d = [abs(x - y) for x, y in zip(a, b)]
    qa, qb = softmax(a), softmax(b)
    # KL(base || cand) on normalized probabilities. Count, do not hide, the
    # terms dropped to underflow: they make the sum a lower bound.
    kl, skipped = 0.0, 0
    for p, q in zip(qa, qb):
        if p > 0.0 and q > 0.0:
            kl += p * math.log(p / q)
        elif p > 0.0:
            skipped += 1
    ds = sorted(d)
    return {
        "n": n,
        "top1_same": ra[0] == rb[0],
        "top1_base": ra[0],
        "top1_cand": rb[0],
        "top5_overlap": len(set(ra) & set(rb)),
        "kl_nats": kl,
        "kl_skipped": skipped,
        "max_abs": max(d),
        "mean_abs": sum(d) / n,
        "rms": math.sqrt(sum(x * x for x in d) / n),
        "p50": pct(ds, 0.50),
        "p99": pct(ds, 0.99),
        "p999": pct(ds, 0.999),
        "nans": nans,
        "infs": infs,
    }


base_dir, cand_dir = sys.argv[1], sys.argv[2]
files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base_dir, "frontier_*.logits.json")))
if not files:
    raise SystemExit(f"no frontier logits in {base_dir}")

print(f"{'frontier':<22}{'top1':<7}{'top5':<6}{'KL(nats)':<12}{'max|d|':<11}{'mean|d|'}")
agree = 0
for f in files:
    cand = os.path.join(cand_dir, f)
    if not os.path.exists(cand):
        print(f"{f:<22}MISSING in candidate")
        continue
    r = compare(os.path.join(base_dir, f), cand)
    agree += r["top1_same"]
    tag = "same" if r["top1_same"] else f'{r["top1_base"]}->{r["top1_cand"]}'
    print(f'{f.replace("frontier_","").replace(".logits.json",""):<22}'
          f'{tag:<7}{r["top5_overlap"]}/5   {r["kl_nats"]:<12.3e}'
          f'{r["max_abs"]:<11.4f}{r["mean_abs"]:.5f}')
    print(f'{"":<22}rms={r["rms"]:.5f}  p50={r["p50"]:.5f}  p99={r["p99"]:.5f}  '
          f'p99.9={r["p999"]:.5f}  nan={r["nans"]}  inf={r["infs"]}  '
          f'kl_skipped={r["kl_skipped"]}  n={r["n"]}')
print(f"\ntop-1 agreement: {agree}/{len(files)} frontiers, vocab {r['n']}")
print("KL is KL(base||cand) over softmax-normalized probabilities (IEEE double, "
      "max-subtracted); kl_skipped>0 would make it a lower bound.")
print("Reference: a pure kernel change must read max|d| = 0 exactly. Any nonzero "
      "value here is the quantization itself, not a bug -- judge it against the "
      "tier-3 eval, not against zero.")
