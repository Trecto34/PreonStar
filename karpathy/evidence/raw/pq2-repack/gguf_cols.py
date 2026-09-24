#!/usr/bin/env python3
"""Byte-weighted histogram of PQ2_0 row widths (dims[0]) in a GGUF.

Why: probe regimes must match the *byte-dominant* production row widths, not
the widths that happen to be convenient.  prints, for type 142/143, each
dims[0] with tensor count and total bytes, sorted by bytes.

usage: gguf_cols.py FILE.gguf
"""
import struct
import sys

BLK = {142: (34, 128), 143: (28, 128), 0: (4, 1), 1: (2, 1), 30: (2, 1)}


def main(path):
    f = open(path, "rb")
    magic, ver, ntensor, nkv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF", magic
    for _ in range(nkv):
        (klen,) = struct.unpack("<Q", f.read(8))
        f.seek(klen, 1)
        (vtype,) = struct.unpack("<I", f.read(4))
        if vtype == 8:
            (slen,) = struct.unpack("<Q", f.read(8))
            f.seek(slen, 1)
        elif vtype == 9:
            (etype, alen) = struct.unpack("<IQ", f.read(12))
            for _ in range(alen):
                if etype == 8:
                    (slen,) = struct.unpack("<Q", f.read(8))
                    f.seek(slen, 1)
                else:
                    f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                            10: 8, 11: 8, 12: 8}.get(etype, 8), 1)
        else:
            f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}.get(vtype, 8), 1)

    hist = {}
    for _ in range(ntensor):
        (nlen,) = struct.unpack("<Q", f.read(8))
        name = f.read(nlen).decode()
        (ndim,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % ndim, f.read(8 * ndim))
        (ttype, _off) = struct.unpack("<IQ", f.read(12))
        if ttype not in BLK:
            continue
        bpb, wpb = BLK[ttype]
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = (nelem // wpb) * bpb
        cols = dims[0]
        rows = nelem // cols
        key = (ttype, cols)
        e = hist.setdefault(key, [0, 0, rows])
        e[0] += 1
        e[1] += nbytes

    tot = sum(v[1] for v in hist.values() if True) or 1
    print("%-8s %10s %8s %8s %10s %10s %8s" % ("type", "cols", "units", "rows", "MiB", "share", "rowB"))
    for (ttype, cols), (cnt, nb, rows) in sorted(hist.items(), key=lambda kv: -kv[1][1]):
        print("%-8s %10d %8d %8d %10.2f %9.2f%% %8d"
              % (ttype, cols, cols // 256, rows, nb / 2**20, 100.0 * nb / tot, cols // 128 * 34))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "model.gguf"))
