#!/usr/bin/env python3
"""Lookup A/B client: temp-0 chat requests, streamed; prints decode t/s and saves text."""
import json, os, sys, time, urllib.request, pathlib
port, tag, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
src = pathlib.Path("/home/server/q36-wt/lookup-test/core_19.orig").read_text()
func = src[src.index("static int p36_session_eval_lookup("):]
func = func[:func.index("\n}\n") + 3]
tools_json = json.dumps({"files": [{"path": f"src/mod{i}.c", "lines": 120 + i, "owner": "core", "status": "modified"} for i in range(12)]}, indent=2)
cases = {
  "edit": f"Here is a C function:\n\n```c\n{func}```\n\nRewrite the whole function unchanged except: rename the variable `verify_n` to `n_verify` everywhere. Output only the full function in a code block.",
  "newcode": "Write a Python function that parses an INI file into a dict of dicts, with comments and type hints. Output only code.",
  "prose": "Write a 250-word story about a lighthouse keeper who finds a strange radio signal.",
  "tooljson": f"Here is a JSON tool result:\n\n{tools_json}\n\nReturn the same JSON, but set \"status\" to \"clean\" for every file whose lines value is even. Output only the JSON.",
}
import os
sel=os.environ.get('CASES')
for name, prompt in cases.items():
    if sel and name not in sel.split(','): continue
    body = {"model": "x", "stream": True, "temperature": 0, "max_tokens": 700,
            "think": os.environ.get("THINK","1") == "1", "reasoning_effort": "high" if os.environ.get("THINK","1") == "1" else "0", "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ch = json.loads(line[6:])
            if ch.get("usage"): usage = ch["usage"]
            for c in ch.get("choices", []):
                d = c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                if d:
                    if t_first is None: t_first = time.time()
                    n += 1; text.append(d)
    t1 = time.time()
    full = "".join(text)
    (outdir / f"{name}.txt").write_text(full)
    toks = usage.get("completion_tokens", 0)
    dec = (toks - 1) / (t1 - t_first) if t_first and toks > 1 else 0
    print(f"{tag} {name}: tokens={toks} chunks={n} decode_tps={dec:.2f} ttft={t_first - t0 if t_first else -1:.2f}s", flush=True)
