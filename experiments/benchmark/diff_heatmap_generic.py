"""diff_heatmap_generic.py – Spatial "difference" heatmap between any two models.

For each channel (u, v):
  - MAE_a(y,x) = mean(|pred_a - target|)  over (N, T)
  - MAE_b(y,x) = mean(|pred_b - target|)  over (N, T)
  - diff(y,x)  = MAE_a - MAE_b
        diff > 0  ->  model B has LOWER error here (improvement, blue)
        diff < 0  ->  model B has HIGHER error here (regression, red)

Run from C:\\Desktop\\RealPDE\\Track1:
    c:\\Python\\venv\\heta\\Scripts\\python.exe experiments\\benchmark\\diff_heatmap_generic.py MODEL_A MODEL_B

Reads:  experiments/results/ref/targets.npz
        experiments/results/<MODEL_A>/predictions.npz
        experiments/results/<MODEL_B>/predictions.npz
Writes: experiments/results/diff_heatmap_<MODEL_A>_vs_<MODEL_B>.png
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

    if len(sys.argv) == 3:
        model_a, model_b = sys.argv[1], sys.argv[2]
    else:
        model_a, model_b = "fno_fp16", "ensemble"

    targets = load_npz(os.path.join(REF_DIR, "targets.npz"), "target")            # (N,T,H,W,3)
    pred_a = load_npz(os.path.join(RESULTS_DIR, model_a, "predictions.npz"), "prediction")
    pred_b = load_npz(os.path.join(RESULTS_DIR, model_b, "predictions.npz"), "prediction")

    channels = [(0, "u"), (1, "v")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (c, cname) in enumerate(channels):
        tgt_c = targets[..., c]
        mae_a = np.mean(np.abs(pred_a[..., c] - tgt_c), axis=(0, 1))   # (H, W)
        mae_b = np.mean(np.abs(pred_b[..., c] - tgt_c), axis=(0, 1))   # (H, W)
        diff = mae_a - mae_b                                          # >0 => model_b better

        vmax_mae = max(mae_a.max(), mae_b.max())
        vmax_diff = np.abs(diff).max()

        im0 = axes[row, 0].imshow(mae_a, origin="lower", aspect="auto", vmin=0, vmax=vmax_mae, cmap="hot")
        axes[row, 0].set_title(f"{model_a} MAE ({cname})")
        axes[row, 0].axis("off")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(mae_b, origin="lower", aspect="auto", vmin=0, vmax=vmax_mae, cmap="hot")
        axes[row, 1].set_title(f"{model_b} MAE ({cname})")
        axes[row, 1].axis("off")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(
            diff, origin="lower", aspect="auto", vmin=-vmax_diff, vmax=vmax_diff, cmap="RdBu"
        )
        axes[row, 2].set_title(
            f"Diff: {model_a}-err - {model_b}-err ({cname})\nblue={model_b} better, red=worse"
        )
        axes[row, 2].axis("off")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_a} vs {model_b} - Spatial Error Comparison (MAE over all samples/timesteps)")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"diff_heatmap_{model_a}_vs_{model_b}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # Quick numeric summary
    for c, cname in channels:
        tgt_c = targets[..., c]
        mae_a = np.mean(np.abs(pred_a[..., c] - tgt_c))
        mae_b = np.mean(np.abs(pred_b[..., c] - tgt_c))
        better = model_b if mae_b < mae_a else model_a
        print(f"channel {cname}: mean MAE {model_a}={mae_a:.6f}  {model_b}={mae_b:.6f}  "
              f"({better} better by {abs(mae_a-mae_b):.6f})")


if __name__ == "__main__":
    main()
