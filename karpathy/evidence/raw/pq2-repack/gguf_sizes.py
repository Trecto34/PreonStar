#!/usr/bin/env python3
"""Exact per-tensor byte accounting for a GGUF, by tensor type.

Why: the decode bandwidth everyone quotes (6.87 GB -> ~318 GB/s) assumes the
whole file is streamed per token.  Layers whose weights are gathered
(token_embd) or unused at draft=1 (the nextn/MTP head) are not.  This prints
the real denominator so kernel GB/s is not computed against the wrong bytes.

usage: gguf_sizes.py FILE.gguf
"""
import struct
import sys

# type id -> (bytes_per_block, weights_per_block); unquantised types use 1/1
BLK = {0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 30: (2, 1),
       41: (20, 32), 42: (18, 64), 142: (34, 128), 143: (28, 128)}
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 30: "BF16",
         41: "Q2_K", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0"}


def main(path):
    f = open(path, "rb")
    magic, ver, ntensor, nkv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF", magic
    for _ in range(nkv):
        (klen,) = struct.unpack("<Q", f.read(8))
        f.seek(klen, 1)
        (vtype,) = struct.unpack("<I", f.read(4))
        if vtype == 8:                                  # string
            (slen,) = struct.unpack("<Q", f.read(8))
            f.seek(slen, 1)
        elif vtype == 9:                                # array
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

    total = 0
    per_type = {}
    blk64 = 0                                       # blk.64.* = nextn/MTP head
    per_type_blk64 = {}
    for _ in range(ntensor):
        (nlen,) = struct.unpack("<Q", f.read(8))
        name = f.read(nlen).decode()
        (ndim,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<%dQ" % ndim, f.read(8 * ndim))
        (ttype, _off) = struct.unpack("<IQ", f.read(12))
        nelem = 1
        for d in dims:
            nelem *= d
        try:
            bpb, wpb = BLK[ttype]
        except KeyError:
            print("unknown type id", ttype, "on", name)
            return 1
        nbytes = (nelem // wpb) * bpb
        total += nbytes
        per_type[ttype] = per_type.get(ttype, 0) + nbytes
        if name.startswith("blk.64."):
            blk64 += nbytes
            per_type_blk64[ttype] = per_type_blk64.get(ttype, 0) + nbytes

    print("%s: %d tensors, %.2f GiB counted" % (path, ntensor, total / 2**30))
    for t, b in sorted(per_type.items(), key=lambda kv: -kv[1]):
        print("  %-8s %10.2f MiB" % (TYPES.get(t, t), b / 2**20))
    print("  blk.64.* (nextn/MTP head): %.2f MiB" % (blk64 / 2**20))
    print("streamed per token, excluding blk.64 head: %.3f GiB" % ((total - blk64) / 2**30))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "model.gguf"))
