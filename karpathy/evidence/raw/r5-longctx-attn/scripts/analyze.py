#!/usr/bin/env python3
"""R5 stage A/B analysis: decode attention cost vs context length.

Reads /tmp/r5/out/*.log (q36-bench profile output) and prints, per arm:
gen t/s, per-token split/combine ms, attention share of the decode token,
workgroup count, and the implied request bytes/s and unique (DRAM) bytes/s.

Accounting: the run generates GEN tokens and dispatch counts identify the rest.
  split dispatches  = GEN * L   (one per full-attn layer per decode token)
  combine dispatches= 2*chunks*L (prefill FA) + GEN*L (decode split)
  FA dispatches     = 2*chunks*L
so L = split_disp / GEN and the decode combine share is recovered from the
combine/split ratio: combine_ms_per_tok = combine_gpu_ms * split_disp/combine_disp / GEN.
"""
import re, sys, pathlib

GEN = 64
HEAD_DIM = 256
K_BYTES = 1        # Q8_0
V_BYTES = 0.5      # Q4_0


def parse(path):
    txt = path.read_text(errors="replace")
    d = {"file": path.name}
    m = re.search(r"^(\d+),(\d+),([\d.]+),(\d+),([\d.]+),(\d+)$", txt, re.M)
    if not m:
        return None
    d["ctx"], _, d["prefill_tps"], _, d["gen_tps"] = m.group(1), None, float(m.group(3)), None, float(m.group(5))
    for op in ("attn_decode_split", "attn_combine", "attn_prefill_fa_gqa6",
               "attn_prefill_fa_gqa8"):
        mm = re.search(r"op %s\s+dispatches=(\d+) .*?gpu_ms=([\d.]+)" % op, txt)
        if mm:
            d[op] = (int(mm.group(1)), float(mm.group(2)))
    mm = re.search(r"Vulkan kernel GPU time total_ms=([\d.]+)", txt)
    if mm:
        d["total_ms"] = float(mm.group(1))
    return d


def report(d, n_head, span):
    split = d.get("attn_decode_split")
    comb = d.get("attn_combine")
    if not split:
        return None
    L = split[0] / GEN
    tok_ms = 1000.0 / d["gen_tps"]
    split_tok = split[1] / GEN
    comb_tok = (comb[1] * split[0] / comb[0] / GEN) if comb else 0.0
    ctx = int(d["ctx"])
    spans = -(-ctx // span) + 1          # ceil + the growth span
    wg = n_head * spans
    ratio = 6 if n_head == 24 else 8
    n_keys = ctx + GEN / 2.0
    per_layer_mb = n_head * n_keys * HEAD_DIM * (K_BYTES + V_BYTES) / 1e6
    req_mb = L * per_layer_mb                       # every query head reads its own copy
    uniq_mb = req_mb / ratio                        # KV is shared by the ratio's q heads
    return dict(file=d["file"], ctx=ctx, gen_tps=d["gen_tps"], tok_ms=tok_ms,
                L=L, split_tok=split_tok, comb_tok=comb_tok,
                attn_pct=100.0 * (split_tok + comb_tok) / tok_ms,
                split_pct=100.0 * split_tok / tok_ms,
                comb_pct=100.0 * comb_tok / tok_ms,
                spans=spans, wg=wg,
                split_ms_per_call=split[1] / split[0],
                comb_ms_per_call=comb[1] / comb[0],
                req_gbs=req_mb * 1e-3 / (split_tok * 1e-3),
                uniq_gbs=uniq_mb * 1e-3 / (split_tok * 1e-3),
                total_ms=d.get("total_ms"))


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/r5/out")
    rows = []
    for p in sorted(root.glob("*.log")):
        d = parse(p)
        if not d:
            continue
        n_head = 24 if p.name.startswith("swift") else 16
        m = re.search(r"span(\d+)", p.name)
        span = int(m.group(1)) if m else 512
        r = report(d, n_head, span)
        if r:
            r["model"] = "swift" if p.name.startswith("swift") else "guard"
            r["span"] = span
            rows.append(r)
    hdr = ("arm                  span   ctx   t/s  tok_ms  L  split/tok comb/tok attn%"
           "  (split% comb%)  spans   wg  split_ms/call comb_ms/call  req GB/s  uniq GB/s")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-20s %5d %5d %5.2f %7.2f %2.0f %9.2f %8.2f %5.1f  (%4.1f %4.1f) %5d %4d %13.4f %12.4f %9.1f %10.1f"
              % (r["file"].replace(".log", ""), r["span"], r["ctx"], r["gen_tps"], r["tok_ms"], r["L"],
                 r["split_tok"], r["comb_tok"], r["attn_pct"], r["split_pct"], r["comb_pct"],
                 r["spans"], r["wg"], r["split_ms_per_call"], r["comb_ms_per_call"],
                 r["req_gbs"], r["uniq_gbs"]))


main()
