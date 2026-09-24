#!/usr/bin/env python3
"""N concurrent greedy streaming clients against a q36-server, measuring TTFT,
per-stream decode tok/s and aggregate tok/s.

Fixed prompts (one per stream index, identical across arms), fixed max_tokens,
temperature 0.  Requests are released together by a barrier so the server sees
them as one batch.
"""
import argparse, json, threading, time, urllib.request

FILLER = " ".join(["alpha beta gamma delta"] * 96)


def prompt(i):
    return (f"Stream {i}. Read the filler, then write a long numbered explanation "
            f"of request isolation with at least two hundred words. Do not quote it."
            f"\n{FILLER}")


def one(url, i, max_tokens, barrier, out):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt(i)}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1000 + i,
            "think": False, "stream": True}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    barrier.wait()
    t0 = time.monotonic()
    ttft = None
    deltas = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                d = line[5:].strip()
                if not d or d == b"[DONE]":
                    continue
                ev = json.loads(d)
                if ev.get("usage"):
                    usage = ev["usage"]
                ch = ev.get("choices") or []
                if not ch:
                    continue
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    deltas += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        out[i] = {"error": repr(exc)}
        return
    t1 = time.monotonic()
    out[i] = {"ttft": ttft, "tokens": (usage or {}).get("completion_tokens") or deltas,
              "wall": t1 - t0}


def med(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--streams", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    barrier = threading.Barrier(a.streams + 1)
    out = {}
    threads = [threading.Thread(target=one,
                                args=(a.url, i, a.max_tokens, barrier, out))
               for i in range(a.streams)]
    for t in threads:
        t.start()
    barrier.wait()
    t_start = time.monotonic()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start

    errs = [v for v in out.values() if "error" in v]
    if errs:
        print(json.dumps({"label": a.label, "streams": a.streams, "errors": errs}))
        return 1
    toks = [out[i]["tokens"] for i in range(a.streams)]
    ttfts = [out[i]["ttft"] for i in range(a.streams)]
    per = [out[i]["tokens"] / out[i]["wall"] for i in range(a.streams)]
    total = sum(toks)
    print(json.dumps({
        "label": a.label,
        "streams": a.streams,
        "max_tokens": a.max_tokens,
        "total_tokens": total,
        "wall_s": round(wall, 3),
        "aggregate_tps": round(total / wall, 3),
        "per_stream_tps_med": round(med(per), 3),
        "per_stream_tps": [round(x, 2) for x in per],
        "ttft_s_med": round(med(ttfts), 3),
        "ttft_s": [round(x, 3) for x in ttfts],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
