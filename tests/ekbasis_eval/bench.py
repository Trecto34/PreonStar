"""Latency benchmark for POST /v1/systemone on synthetic states (stdlib only; no GPU use by this script).

  python3 bench.py --url http://127.0.0.1:8123 [--sizes 128,256,512,1024] [--questions 1,3,5] [--reps 3]
                   [--token-mode auto|tokenizers|server] [--out-dir DIR]

Configs: every size x question count x read_once (false, true) = 24 configs by default, `reps` requests each.
Sizing: a state of N lines (synthetic text, "file" lines) is grown until the STATE TEXT has ~SIZE tokens. The prompt
also carries a fixed wrapper (system prompt, question, ChatML tags: about 110 tokens), which is measured once as the
prompt with an empty state and not counted in SIZE. The count comes from the tokenizers lib (tokenizer.json from the
model repo, the same prompt text as build_golden.py) when available (--token-mode auto|tokenizers), or from the
server's usage.input_tokens with a secant search (--token-mode server, a few probe requests per size, not timed).
The same state is used for every question count and read_once setting of a size. Reported input_tokens are the
server's usage.input_tokens, i.e. the full prompt, so they are about SIZE + wrapper + question text.

Measured per request: client wall-clock seconds (time.perf_counter around the POST), HTTP status, and the server's
usage.input_tokens. Per config: median, min and max latency over the reps (errors excluded and counted), the median
input_tokens, and the GTT / MemAvailable range seen during the config's requests.

Memory: a thread samples /sys/class/drm/card0/device/mem_info_gtt_used (bytes) and MemAvailable from /proc/meminfo
once per second for the whole run (samples.jsonl). Missing files are recorded as null.

Outputs in --out-dir (default CACHE/bench/<UTC time>): requests.jsonl (every probe and timed request, raw response
text included), samples.jsonl, results.json, results.md (also printed).
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402

GTT = "/sys/class/drm/card0/device/mem_info_gtt_used"
MEMINFO = "/proc/meminfo"


def state_text(lines):
    """Synthetic git-like state: `lines` numbered file lines (about 25 tokens each)."""
    return "".join(f"[{i:04d}] file_{i:04d}.txt rev {i * 7 % 997} owner team-{i % 13} "
                   f"status {'dirty' if i % 3 == 0 else 'clean'}\n" for i in range(lines))


def questions_for(n):
    return [(f"q{k}", {"type": "noul",
                       "instructions": f"Does file_{k:04d}.txt need a backup before the commands run?"})
            for k in range(n)]


def read_gtt():
    try:
        with open(GTT) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_mem_available_kb():
    try:
        with open(MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


class Sampler(threading.Thread):
    def __init__(self, path, interval=1.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop_evt = path, interval, threading.Event()
        self.samples = []  # (t_monotonic, gtt_bytes, mem_available_kb)
        self.f = open(path, "w", encoding="utf-8")

    def run(self):
        while not self.stop_evt.is_set():
            t = time.perf_counter()
            s = (t, read_gtt(), read_mem_available_kb())
            self.samples.append(s)
            self.f.write(json.dumps({"t": round(t, 3), "gtt_used_bytes": s[1], "mem_available_kb": s[2]}) + "\n")
            self.f.flush()
            self.stop_evt.wait(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join()
        self.f.close()

    def window(self, t0, t1):
        pts = [s for s in self.samples if t0 <= s[0] <= t1]
        gtt = [s[1] for s in pts if s[1] is not None]
        mem = [s[2] for s in pts if s[2] is not None]
        return {"gtt_used_max_mib": max(gtt) / 2 ** 20 if gtt else None,
                "gtt_used_min_mib": min(gtt) / 2 ** 20 if gtt else None,
                "mem_available_min_mib": min(mem) / 1024 if mem else None,
                "mem_available_max_mib": max(mem) / 1024 if mem else None, "samples": len(pts)}


def make_counter(mode):
    """Returns measure(lines) -> input tokens of the one-question prompt, and a label for the method."""
    if mode in ("auto", "tokenizers"):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(c.model_file("tokenizer.json"))
            q1 = questions_for(1)[0][1]
            return (lambda lines: len(tok.encode(c.chat_prompt(state_text(lines), q1),
                                                 add_special_tokens=False).ids)), "tokenizers"
        except Exception as e:  # noqa: BLE001 - fall back only in auto mode
            if mode == "tokenizers":
                raise
            print(f"tokenizers unavailable ({type(e).__name__}: {e}); calibrating with server usage", file=sys.stderr)

    def server_measure(lines):
        status, raw, dt = c.post_systemone(URL[0], c.build_request(state_text(lines), questions_for(1)),
                                           timeout=TIMEOUT[0])
        LOG.write(json.dumps({"kind": "probe", "lines": lines, "http_status": status, "latency_s": dt,
                              "raw": raw}) + "\n")
        LOG.flush()
        usage = json.loads(raw).get("usage", {}) if status == 200 else {}
        if "input_tokens" not in usage:
            raise SystemExit(f"probe with {lines} lines failed: HTTP {status} {raw[:300]}")
        return int(usage["input_tokens"])
    return server_measure, "server_usage"


def size_state(target, measure):
    """Secant search for the line count whose one-question prompt has about `target` tokens."""
    l0 = max(1, target // 25)
    t0 = measure(l0)
    l1 = max(1, round(l0 * target / t0))
    t1 = measure(l1) if l1 != l0 else t0
    for _ in range(4):
        if abs(t1 - target) <= max(2, 0.03 * target) or l1 == l0:
            break
        slope = (t1 - t0) / (l1 - l0)
        if slope <= 0:  # not monotone at this size; step by the average
            slope = t1 / l1
        l2 = max(1, round(l1 + (target - t1) / slope))
        l0, t0 = l1, t1
        l1, t1 = l2, measure(l2)
    return l1, t1


URL, TIMEOUT, LOG = [c.DEFAULT_URL], [600.0], None


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=c.DEFAULT_URL)
    ap.add_argument("--sizes", default="128,256,512,1024")
    ap.add_argument("--questions", default="1,3,5")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--token-mode", default="auto", choices=["auto", "tokenizers", "server"])
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--cold", action="store_true",
                    help="prefix every request's state with a unique line so none can reuse the server's live prefix "
                         "(otherwise a repeated 1-question request is a ~1 ms no-op sync); adds ~8 tokens")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    URL[0], TIMEOUT[0] = args.url, args.timeout
    sizes = [int(x) for x in args.sizes.split(",")]
    qcounts = [int(x) for x in args.questions.split(",")]
    out = args.out_dir or os.path.join(c.CACHE, "bench", datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(out, exist_ok=True)
    LOG = open(os.path.join(out, "requests.jsonl"), "w", encoding="utf-8")
    measure, method = make_counter(args.token_mode)
    base = measure(0)  # the prompt with an empty state: system, question and wrapper tokens
    sampler = Sampler(os.path.join(out, "samples.jsonl"), args.sample_interval)
    sampler.start()
    configs, sizing = [], {}
    try:
        for size in sizes:
            lines, counted = size_state(size, lambda n: measure(n) - base)
            sizing[size] = {"lines": lines, "state_tokens": counted, "prompt_tokens_1q": counted + base,
                            "wrapper_tokens": base, "method": method}
            state = state_text(lines)
            for nq in qcounts:
                for ro in (False, True):
                    body = c.build_request(state, questions_for(nq), ro)
                    rec = {"size": size, "questions": nq, "read_once": ro, "lat": [], "tokens": [], "errors": 0}
                    t_start = time.perf_counter()
                    for rep in range(args.reps):
                        try:
                            b = c.build_request(f"[run {time.time_ns()}]\n" + state, questions_for(nq), ro) if args.cold else body
                            status, raw, dt = c.post_systemone(args.url, b, timeout=args.timeout)
                        except OSError as e:  # refused, reset or timed out: recorded as an error, the run goes on
                            status, raw, dt = None, json.dumps({"error": f"{type(e).__name__}: {e}"}), None
                        usage = {}
                        if status == 200:
                            usage = json.loads(raw).get("usage", {})
                        LOG.write(json.dumps({"kind": "timed", "size": size, "questions": nq, "read_once": ro,
                                              "rep": rep, "http_status": status, "latency_s": dt,
                                              "usage": usage, "raw": raw}) + "\n")
                        LOG.flush()
                        if status == 200 and "input_tokens" in usage:
                            rec["lat"].append(dt)
                            rec["tokens"].append(int(usage["input_tokens"]))
                        else:
                            rec["errors"] += 1
                    rec["window"] = sampler.window(t_start, time.perf_counter())
                    configs.append(rec)
                    print(f"size {size} q {nq} read_once {ro}: "
                          f"median {statistics.median(rec['lat']) if rec['lat'] else float('nan'):.3f} s",
                          file=sys.stderr)
    finally:
        sampler.stop()
        LOG.close()
    results = {"run": {"url": args.url, "sizes": sizes, "questions": qcounts, "reps": args.reps,
                       "token_mode": method, "sample_interval_s": args.sample_interval,
                       "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
               "sizing": sizing, "configs": []}
    for r in configs:
        lat = r["lat"]
        results["configs"].append({
            "size": r["size"], "questions": r["questions"], "read_once": r["read_once"], "reps": args.reps,
            "ok": len(lat), "errors": r["errors"],
            "latency_s": {"median": statistics.median(lat) if lat else None, "min": min(lat) if lat else None,
                          "max": max(lat) if lat else None},
            "input_tokens_median": statistics.median(r["tokens"]) if r["tokens"] else None,
            "memory": r["window"]})
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
        f.write("\n")
    md = to_markdown(results)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}/requests.jsonl, samples.jsonl, results.json, results.md")


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def to_markdown(res):
    lines = [f"Sizing method: {res['run']['token_mode']}. Sizes (state text tokens): "
             + ", ".join(f"{s} -> {v['lines']} lines = {v['state_tokens']} tokens "
                         f"(wrapper {v['wrapper_tokens']})" for s, v in res["sizing"].items()), "",
             "| Size | Questions | read_once | ok/err | Latency median s | min s | max s | input_tokens median "
             "| GTT used max MiB | MemAvailable min MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in res["configs"]:
        lat, mem = cfg["latency_s"], cfg["memory"]
        lines.append(f"| {cfg['size']} | {cfg['questions']} | {str(cfg['read_once']).lower()} | "
                     f"{cfg['ok']}/{cfg['errors']} | {fmt(lat['median'])} | {fmt(lat['min'])} | {fmt(lat['max'])} | "
                     f"{fmt(cfg['input_tokens_median'], 0)} | {fmt(mem['gtt_used_max_mib'], 1)} | "
                     f"{fmt(mem['mem_available_min_mib'], 0)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
