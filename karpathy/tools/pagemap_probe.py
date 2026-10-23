
import subprocess, time, os, sys, json

MODEL = "/home/server/models/llm/MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf"
PAT = os.path.basename(MODEL)
log = open("/tmp/probe.log", "w")
p = subprocess.Popen(["./p36", "-m", MODEL, "-p", "hello", "-n", "4000",
                      "--ctx", "512", "--nothink", "--vulkan"],
                     cwd="/home/server/q36", stdout=log, stderr=log)

def model_region(pid):
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                if PAT in line:
                    a, b = line.split()[0].split("-")
                    return int(a, 16), int(b, 16)
    except FileNotFoundError:
        return None
    return None

reg = None
for _ in range(120):
    time.sleep(0.5)
    reg = model_region(p.pid)
    if reg: break
    if p.poll() is not None: break
if not reg:
    print("no mapping; alive=", p.poll() is None); p.kill(); sys.exit(1)
start, end = reg
PAGE = 4096
n = (end - start) // PAGE
print(f"mapping {end-start:#x} bytes, pages={n}")

def present_bits(pid):
    bits = bytearray(n)
    fd = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
    try:
        os.lseek(fd, (start // PAGE) * 8, os.SEEK_SET)
        buf = os.read(fd, n * 8)
    finally:
        os.close(fd)
    for i in range(len(buf)//8):
        if buf[i*8 + 7] & 0x80:
            bits[i] = 1
    return bits

time.sleep(8)
snaps = []
for k in range(10):
    if p.poll() is not None: break
    try:
        snaps.append(present_bits(p.pid))
    except (FileNotFoundError, ProcessLookupError):
        break
    time.sleep(1.0)

if p.poll() is None: p.kill()
p.wait()

if not snaps:
    print("no snapshots; alive=", p.poll()); print(open("/tmp/probe.log").read()[-400:]); sys.exit(1)

print("resident per sample:", [sum(s) for s in snaps], "of", n)
evicted = [i for i in range(n) if any(snaps[k][i] and not snaps[k+1][i] for k in range(len(snaps)-1))]
back    = [i for i in range(n) if any((not snaps[k][i]) and snaps[k+1][i] for k in range(len(snaps)-1))]
print("evicted between samples:", len(evicted))
print("refaulted between samples:", len(back))
json.dump({"model": MODEL, "start": start, "back": back, "evicted": evicted}, open("/tmp/thrash.json","w"))
