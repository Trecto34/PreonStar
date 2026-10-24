#!/usr/bin/env python3
"""Diff per-layer residual sweeps written by P36_VK_SWEEP_LAYERS=1.

Usage: layer_sweep_diff.py <batch_dir> <serial_dir> [threshold]
Each dir holds LNN_hidden.f32 / LNN_ffn.f32 with two n_tok=2 rows.
"""
import os
import sys

import numpy as np

EMBD = 5120
N_ROW = 2


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    batch, serial = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    first = None
    for layer in range(64):
        worst = 0.0
        detail = []
        for kind in ("hidden", "ffn"):
            a = np.fromfile(os.path.join(batch, f"L{layer:02d}_{kind}.f32"),
                            dtype=np.float32)
            b = np.fromfile(os.path.join(serial, f"L{layer:02d}_{kind}.f32"),
                            dtype=np.float32)
            if a.size != N_ROW * EMBD or b.size != N_ROW * EMBD:
                print(f"L{layer:02d} {kind}: bad size {a.size}/{b.size}")
                continue
            d = np.abs(a - b).reshape(N_ROW, EMBD)
            row_max = d.max(axis=1)
            worst = max(worst, float(row_max.max()))
            detail.append(f"{kind[0]}={row_max[0]:.2e}/{row_max[1]:.2e}")
        flag = ""
        if first is None and worst > threshold:
            first = layer
            flag = "  <-- FIRST"
        print(f"L{layer:02d} max={worst:.3e} {' '.join(detail)}{flag}")
    print(f"first_gt_{threshold:g}={first}")
    return 0 if first is None else 1


if __name__ == "__main__":
    sys.exit(main())
