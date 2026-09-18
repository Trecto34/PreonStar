#!/usr/bin/env python3
"""Prove the shipped PQ2_0 GGUF is natively quantized at 128 granularity.

It must NOT be a paired repack of a Q2_0-g64 file. The original "repack"
finding was circular: the local Q2_0-g64 file was derived FROM this PQ2_0
file (one 128-block scale duplicated into two 64-block scales), so of course
adjacent g64 scales were equal.

Ground truth is the publisher's own F16 source and their canonical rule
(github.com/PrismML-Eng/llama.cpp, ggml-quants.c quantize_row_pq2_0_ref):

    d = max|x| over the 128-block      (stored as f16)
    q = clamp(round(x/d), -1, 2)       (2-bit code, code-1 decoded)

This script range-fetches selected F16 tensor rows and checks the stored
PQ2_0 scales/codes against that rule, and separately checks that adjacent
128-block scales in the PQ2_0 file are not degenerate.

Usage: python3 verify_native_128.py [n_tensors]
"""
import struct
import subprocess
import sys

PQ2 = "/home/server/q36-opt-27b/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
HF = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/"
F16 = HF + "Ternary-Bonsai-2-27B-F16.gguf"


def curl_range(url, start, n):
    last = start + n - 1
    out = subprocess.run(["curl", "-sL", "-r", f"{start}-{last}", url],
                         capture_output=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode()[:200])
    return out.stdout


def read_str(f):
    (l,) = struct.unpack("<Q", f.read(8))
    return f.read(l).decode("utf-8", "replace")


def read_val(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_val(f, et) for _ in range(n)]
    return f.read(sizes[t])


def parse_header(f):
    """Returns (tensor_infos dict, data_start)."""
    f.read(4)
    struct.unpack("<I", f.read(4))
    (nt,) = struct.unpack("<Q", f.read(8))
    (nkv,) = struct.unpack("<Q", f.read(8))
    align = 32
    for _ in range(nkv):
        k = read_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        v = read_val(f, t)
        if k == "general.alignment":
            align = v
    infos = {}
    for _ in range(nt):
        nm = read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
        (ty,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        infos[nm] = (dims, ty, off)
    pos = f.tell()
    return infos, pos + ((align - (pos % align)) % align)


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with open(PQ2, "rb") as f:
        pinfos, pdata = parse_header(f)
        quant = sorted((n for n, v in pinfos.items() if v[1] == 142),
                       key=lambda n: pinfos[n][0][0] * pinfos[n][0][1])
        # pick across distinct input widths
        chosen, widths = [], set()
        for n in quant:
            w = pinfos[n][0][0]
            if w in widths:
                continue
            widths.add(w)
            chosen.append(n)
            if len(chosen) >= n_sample:
                break
        # local PQ2_0 scale-variation over a wide sample
        total_pairs = differ = 0
        for n in quant[:20]:
            dims, ty, off = pinfos[n]
            in_dim, nrow = dims
            nb = in_dim // 128
            f.seek(pdata + off)
            row = f.read(nb * 34)
            scales = [struct.unpack("<e", row[b * 34:b * 34 + 2])[0] for b in range(nb)]
            for b in range(1, nb):
                total_pairs += 1
                if abs(scales[b] - scales[b - 1]) > 1e-9:
                    differ += 1
        print(f"PQ2_0 local: {differ}/{total_pairs} adjacent 128-block scale pairs differ")
        if total_pairs and differ == 0:
            print("REJECT: adjacent 128-block scales are degenerate")
            return 1

        # fetch F16 header once
        hdr = curl_range(F16, 0, 32 * 1024 * 1024)
        open("/tmp/opencode/f16hdr.bin", "wb").write(hdr)
        with open("/tmp/opencode/f16hdr.bin", "rb") as g:
            finfos, fdata = parse_header(g)

        grand_scale = grand_code = grand_blocks = 0
        for name in chosen:
            dims, ty, off = pinfos[name]
            in_dim = dims[0]
            f.seek(pdata + off)
            row = f.read((in_dim // 128) * 34)
            fdims, fty, foff = finfos[name]
            frow = curl_range(F16, fdata + foff, in_dim * 2)
            if len(frow) != in_dim * 2:
                print(f"{name}: short F16 fetch ({len(frow)} != {in_dim*2})")
                return 1
            vals = struct.unpack("<%de" % in_dim, frow)
            ms = mc = 0
            nb = in_dim // 128
            for b in range(nb):
                blk = vals[b * 128:(b + 1) * 128]
                d = max(abs(v) for v in blk)
                stored = struct.unpack("<e", row[b * 34:b * 34 + 2])[0]
                if abs(d - stored) > 1e-3 * max(1e-9, abs(d)):
                    ms += 1
                for j in range(128):
                    q = max(-1, min(2, int(round(blk[j] / d)))) if d > 0 else 0
                    if ((q + 1) & 3) != ((row[b * 34 + 2 + (j // 4)] >> ((j % 4) * 2)) & 3):
                        mc += 1
            print(f"{name:28s} in_dim={in_dim:6d} blocks={nb:4d} "
                  f"scale_mismatch={ms} code_mismatch={mc}")
            grand_scale += ms
            grand_code += mc
            grand_blocks += nb
        print(f"TOTAL blocks={grand_blocks} scale_mismatch={grand_scale} "
              f"code_mismatch={grand_code}")
        return 0 if grand_scale == 0 and grand_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
