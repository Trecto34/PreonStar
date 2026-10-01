#!/usr/bin/env python3
"""Short local tool-loop benchmark: tokens and steady-state tok/s of a tiny agent task.
Usage: tests/bench_agent_loop.py AGENT MODEL [reps]   (run once per binary to compare)"""
import datetime, os, re, subprocess, sys, tempfile

agent, model = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PROMPT = "Create hello.txt containing the word hi, then read it back and tell me what it says."


def ts(line):
    return datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f").timestamp()


tot = []
for i in range(reps):
    d = tempfile.mkdtemp(prefix="q36bench-")
    tr = os.path.join(d, ".trace")
    where = ["--server", model] if model.startswith("http") else ["--vulkan", "-m", model, "--ctx", "8192"]
    subprocess.run([agent, *where, "--think", "--thinking-budget", "256",
                    "--temp", "0", "--seed", "1", "--chdir", d, "--non-interactive", "--trace", tr, "-p", PROMPT],
                   capture_output=True, text=True, timeout=400)
    lines = open(tr, errors="replace").read().splitlines()
    remote = model.startswith("http")
    if remote:   # first request is logged at [REQUEST]-less start: use user event to final step
        first = next(l for l in lines if '"user"' in l or "user=" in l)
        steps_l = [l for l in lines if "[STEP]" in l]
        last = steps_l[-1]
        gen = sum(int(a) + int(b) for a, b in re.findall(r"reasoning=(\d+) action=(\d+)", "\n".join(steps_l)))
        steps = len(steps_l)
        prompt0 = int(re.search(r"prompt=(\d+)", steps_l[0]).group(1))
    else:
        first = next(l for l in lines if "prefill tool_round=0" in l)
        last = [l for l in lines if "generation finished" in l][-1]
        gen = sum(int(m) for m in re.findall(r"generation finished .*?generated=(\d+)", "\n".join(lines)))
        steps = len([l for l in lines if "generation finished" in l])
        prompt0 = int(re.search(r"prompt=(\d+)", first).group(1))
    wall = ts(last) - ts(first)
    tot.append((wall, gen, steps, prompt0))
    print("rep %d: loop_wall=%.2fs generated=%d steps=%d initial_prompt=%d" % ((i,) + tot[-1]))
n = len(tot)
print("MEAN loop_wall=%.2fs generated=%.0f steps=%.1f initial_prompt=%.0f" % tuple(sum(t[k] for t in tot) / n for k in range(4)))
