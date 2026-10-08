"""make_targets.py – Build targets.npz and samples.npz from the example h5 file.

Run from C:\\Desktop\\RealPDE\\Track1:
    python experiments/benchmark/make_targets.py

Layout written to experiments/results/ref/:
    targets.npz  – key "target"  shape (N, 20, 32, 64, 3)
    samples.npz  – key "input"   shape (N, 20, 32, 64, 3)

Window strategy (non-overlapping, stride=20):
    window i: input  = frames [i*20 : i*20+20]
              target = frames [i*20+20 : i*20+40]
    Only windows where both input AND target fit within T=868 are kept.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
H5_PATH = os.path.join(REPO_ROOT, "example_data", "3750_0.h5")
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "results", "ref")

T_IN = 20
T_OUT = 20
SUB_S = 2   # subsample 64x128 → 32x64


def main() -> None:
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required.  Install it with:  pip install h5py")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading {H5_PATH} …")
    with h5py.File(H5_PATH, "r") as f:
        u_all = np.asarray(f["u"][:, ::SUB_S, ::SUB_S], dtype=np.float32)  # (T, 32, 64)
        v_all = np.asarray(f["v"][:, ::SUB_S, ::SUB_S], dtype=np.float32)

    T = u_all.shape[0]
    print(f"  u shape after subsampling: {u_all.shape}  (T={T})")

    inputs, targets = [], []
    i = 0
    while True:
        t0 = i * T_IN
        t1 = t0 + T_IN       # end of input window
        t2 = t1 + T_OUT      # end of target window
        if t2 > T:
            break
        # input window: shape (T_IN, H, W, 3)
        u_in = u_all[t0:t1]
        v_in = v_all[t0:t1]
        inp = np.stack([u_in, v_in, np.zeros_like(u_in)], axis=-1)

        # target window: shape (T_OUT, H, W, 3)
        u_tgt = u_all[t1:t2]
        v_tgt = v_all[t1:t2]
        tgt = np.stack([u_tgt, v_tgt, np.zeros_like(u_tgt)], axis=-1)

        inputs.append(inp)
        targets.append(tgt)
        i += 1

    inputs = np.stack(inputs, axis=0).astype(np.float32)    # (N, 20, 32, 64, 3)
    targets = np.stack(targets, axis=0).astype(np.float32)

    print(f"  Windows found: {len(inputs)}")
    print(f"  inputs  shape: {inputs.shape}")
    print(f"  targets shape: {targets.shape}")

    samples_path = os.path.join(OUT_DIR, "samples.npz")
    targets_path = os.path.join(OUT_DIR, "targets.npz")

    np.savez_compressed(samples_path, input=inputs)
    np.savez_compressed(targets_path, target=targets)

    print(f"Saved  {samples_path}")
    print(f"Saved  {targets_path}")


if __name__ == "__main__":
    main()
