#!/usr/bin/env python3
"""Model-free checks for gguf-tools/qwen36-quantize: atomic output, flag validation.

Builds a tiny synthetic GGUF (one F32 norm tensor, copied verbatim by the
quantizer), so no model and no GPU are needed.

Usage: test_quantize_output.py [--golden OUT]   (--golden writes the quantized
bytes for the tiny input to OUT, for old/new binary bit-identity comparison)
"""
import os
import resource
import signal
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT = os.environ.get("P36_QUANT_BIN") or os.path.join(ROOT, "gguf-tools", "qwen36-quantize")


def gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def tiny_gguf(path: str) -> None:
    align = 32
    kv = gstr("general.architecture") + struct.pack("<I", 8) + gstr("qwen35moe")
    info = gstr("output_norm.weight") + struct.pack("<IQIQ", 1, 32, 0, 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + kv + info
    head += b"\0" * (-len(head) % align)
    data = struct.pack("<32f", *[float(i) for i in range(32)])
    with open(path, "wb") as f:
        f.write(head + data)


def run(*args):
    return subprocess.run((QUANT,) + args, capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="p36-quant-") as d:
        src = os.path.join(d, "in.gguf")
        out = os.path.join(d, "out.gguf")
        tiny_gguf(src)

        r = run("--in", src, "--out", out, "--allow-synthetic-imatrix")
        assert r.returncode == 0, r.stderr
        good = open(out, "rb").read()
        if len(sys.argv) == 3 and sys.argv[1] == "--golden":
            open(sys.argv[2], "wb").write(good)
            return 0
        assert sorted(os.listdir(d)) == ["in.gguf", "out.gguf"], f"temp file left: {os.listdir(d)}"

        # Failure after the output is opened (RLIMIT_FSIZE makes the write fail):
        # no partial file at the final path, and an existing --force target is
        # left untouched.
        def limited(*args):
            def pre():
                resource.setrlimit(resource.RLIMIT_FSIZE, (100, 100))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            return subprocess.run((QUANT,) + args, capture_output=True, text=True, preexec_fn=pre)

        out2 = os.path.join(d, "out2.gguf")
        r = limited("--in", src, "--out", out2, "--allow-synthetic-imatrix")
        assert r.returncode != 0, "write failure must fail the run"
        assert not os.path.exists(out2), "partial output left at final path"
        r = limited("--in", src, "--out", out, "--force", "--allow-synthetic-imatrix")
        assert r.returncode != 0
        assert open(out, "rb").read() == good, "failed run clobbered existing output"
        assert sorted(os.listdir(d)) == ["in.gguf", "out.gguf"], f"temp file left: {os.listdir(d)}"

        # --imatrix-strict without --imatrix used to be silently ignored.
        r = run("--in", src, "--out", os.path.join(d, "o3.gguf"),
                "--allow-synthetic-imatrix", "--imatrix-strict")
        assert r.returncode != 0 and "imatrix-strict" in r.stderr, r.stderr
    print("quantize output tests: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
