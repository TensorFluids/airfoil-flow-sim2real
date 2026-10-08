"""diff_heatmap.py – Spatial "difference" heatmap between FNO and the ensemble.

For each channel (u, v):
  - MAE_fno(y,x)      = mean(|fno_pred - target|)       over (N, T)
  - MAE_ensemble(y,x) = mean(|ensemble_pred - target|)   over (N, T)
  - diff(y,x)         = MAE_fno - MAE_ensemble
        diff > 0  ->  ensemble has LOWER error here (improvement, blue)
        diff < 0  ->  ensemble has HIGHER error here (regression, red)

Run from C:\\Desktop\\RealPDE\\Track1:
    c:\Python\venv\heta\Scripts\python.exe experiments/benchmark/diff_heatmap.py

Reads:  experiments/results/ref/targets.npz
        experiments/results/fno_fp16/predictions.npz
        experiments/results/ensemble/predictions.npz
Writes: experiments/results/diff_heatmap_fno_vs_ensemble.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
REF_DIR = os.path.join(RESULTS_DIR, "ref")


def load_npz(path: str, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        return d[key].astype(np.float32)


def main() -> None:
    import matplotlib.pyplot as plt

    targets = load_npz(os.path.join(REF_DIR, "targets.npz"), "target")            # (N,T,H,W,3)
    fno_pred = load_npz(os.path.join(RESULTS_DIR, "fno_fp16", "predictions.npz"), "prediction")
    ens_pred = load_npz(os.path.join(RESULTS_DIR, "ensemble", "predictions.npz"), "prediction")

    channels = [(0, "u"), (1, "v")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (c, cname) in enumerate(channels):
        tgt_c = targets[..., c]
        mae_fno = np.mean(np.abs(fno_pred[..., c] - tgt_c), axis=(0, 1))   # (H, W)
        mae_ens = np.mean(np.abs(ens_pred[..., c] - tgt_c), axis=(0, 1))   # (H, W)
        diff = mae_fno - mae_ens                                          # >0 => ensemble better

        vmax_mae = max(mae_fno.max(), mae_ens.max())
        vmax_diff = np.abs(diff).max()

        im0 = axes[row, 0].imshow(mae_fno, origin="lower", aspect="auto", vmin=0, vmax=vmax_mae, cmap="hot")
        axes[row, 0].set_title(f"FNO MAE ({cname})")
        axes[row, 0].axis("off")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(mae_ens, origin="lower", aspect="auto", vmin=0, vmax=vmax_mae, cmap="hot")
        axes[row, 1].set_title(f"Ensemble MAE ({cname})")
        axes[row, 1].axis("off")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(
            diff, origin="lower", aspect="auto", vmin=-vmax_diff, vmax=vmax_diff, cmap="RdBu"
        )
        axes[row, 2].set_title(f"Diff: FNO-err − Ens-err ({cname})\nblue=ensemble better, red=worse")
        axes[row, 2].axis("off")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle("FNO vs Ensemble — Spatial Error Comparison (MAE over all samples/timesteps)")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "diff_heatmap_fno_vs_ensemble.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # Quick numeric summary
    for c, cname in channels:
        tgt_c = targets[..., c]
        mae_fno = np.mean(np.abs(fno_pred[..., c] - tgt_c))
        mae_ens = np.mean(np.abs(ens_pred[..., c] - tgt_c))
        print(f"channel {cname}: mean MAE fno={mae_fno:.6f}  ensemble={mae_ens:.6f}  "
              f"({'ensemble better' if mae_ens < mae_fno else 'fno better'} by {abs(mae_fno-mae_ens):.6f})")


if __name__ == "__main__":
    main()
