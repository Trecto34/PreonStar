#!/usr/bin/env python3
"""Compare two attn_parity span dumps and report the span-vs-span difference.

Regression test for the D2 mechanism: a span change must stay at float32
reassociation magnitude (~1e-5 here) and not reveal an indexing/masking bug.
The harness itself checks each GPU output against its own fp64 CPU reference
(attn_parity prints maxabs_gpu_ref); this script adds the cross-span view and
exits nonzero if the difference exceeds a reassociation bound.
"""
import struct
import sys

NQ = 24 * 256
POS0S = [511, 512, 767, 768, 1023, 1024, 1279, 1280]
# Reassociation only: both spans match fp64 to ~1e-5, so their mutual
# difference must stay small.  1e-4 is far below any indexing/masking bug.
BOUND = 1e-4


def load(path):
    d = open(path, "rb").read()
    assert len(d) == len(POS0S) * NQ * 4, (path, len(d))
    return [struct.unpack_from("<%df" % NQ, d, i * NQ * 4) for i in range(len(POS0S))]


def main():
    if len(sys.argv) != 3:
        print("usage: attn_parity_cmp.py a.bin b.bin", file=sys.stderr)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    worst = 0.0
    for i, p in enumerate(POS0S):
        md, idx, nd = 0.0, -1, 0
        for j in range(NQ):
            e = abs(a[i][j] - b[i][j])
            if e > md:
                md, idx = e, j
            if a[i][j] != b[i][j]:
                nd += 1
        worst = max(worst, md)
        print("n_kv=%-5d max|A-B|=%.3e ndiff=%d/%d first_idx=%d head=%d dim=%d"
              % (p + 1, md, nd, NQ, idx, idx // 256, idx % 256))
    verdict = "reassociation (ok)" if worst <= BOUND else "OVER BOUND (FAIL)"
    print("worst=%.3e bound=%.1e %s" % (worst, BOUND, verdict))
    return 0 if worst <= BOUND else 1


if __name__ == "__main__":
    sys.exit(main())
