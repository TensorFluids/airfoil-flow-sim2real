"""run_ensemble_benchmark.py – Run the trained StackedEnsemble (mixer on top of
frozen FNO+CNO+Transolver) on the same example data used for the baseline
benchmark, and save predictions.npz in the same format so it can be scored
and compared side-by-side via compare_models.py.

Run from C:\\Desktop\\RealPDE\\Track1:
    c:\Python\venv\heta\Scripts\python.exe experiments/benchmark/run_ensemble_benchmark.py

Reads:  experiments/results/ref/samples.npz          (key "input", RAW/physical space)
        checkpoints/ensemble/best_ensemble_mixer.pth (trained mixer weights)
Writes: experiments/results/ensemble/predictions.npz
        keys: "prediction" (N,20,32,64,3), "mean_t_neural_s" (scalar float32)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# ── make kit + experiments importable ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "_vendor"))
sys.path.insert(0, EXP_DIR)

import torch  # noqa: E402

from models.stacked_ensemble import StackedEnsemble  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "sim_real_ft")
FNO_CKPT = os.path.join(CKPT_DIR, "sim_real_fno_fp16.pth")
CNO_CKPT = os.path.join(CKPT_DIR, "sim_real_cno.pth")
TRANSOLVER_CKPT = os.path.join(CKPT_DIR, "sim_real_transolver.pth")
MIXER_CKPT = os.path.join(REPO_ROOT, "checkpoints", "ensemble", "best_ensemble_mixer.pth")

REF_DIR = os.path.join(REPO_ROOT, "experiments", "results", "ref")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

BATCH = 8


def main() -> None:
    samples_path = os.path.join(REF_DIR, "samples.npz")
    if not os.path.exists(samples_path):
        sys.exit(f"samples.npz not found at {samples_path}\nRun make_targets.py first.")
    if not os.path.exists(MIXER_CKPT):
        sys.exit(f"Trained mixer checkpoint not found at {MIXER_CKPT}\nRun train_ensemble.py first.")

    print(f"Loading inputs from {samples_path} …")
    with np.load(samples_path, allow_pickle=False) as d:
        inputs = d["input"].astype(np.float32)
    print(f"  inputs shape: {inputs.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building StackedEnsemble on {device} …")
    model = StackedEnsemble(
        fno_ckpt=FNO_CKPT, cno_ckpt=CNO_CKPT, transolver_ckpt=TRANSOLVER_CKPT,
        mixer_hidden=32, device=device,
    )

    ckpt = torch.load(MIXER_CKPT, map_location=device)
    model.mixer.load_state_dict(ckpt["mixer_state_dict"])
    model.eval()
    print(f"  Loaded trained mixer (epoch {ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss'):.6f})")

    N = inputs.shape[0]
    outs: list[np.ndarray] = []
    times: list[float] = []

    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(inputs[i : i + BATCH])).to(device)

            t_start = time.perf_counter()
            yb = model(xb)   # StackedEnsemble.forward handles normalise/denormalise internally
            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            yb_np = yb.float().cpu().numpy()
            n_samples = xb.shape[0]
            times.append((t_end - t_start) / n_samples)
            outs.append(yb_np)

    predictions = np.concatenate(outs, axis=0).astype(np.float32)
    mean_t = float(np.mean(times))

    out_dir = os.path.join(RESULTS_DIR, "ensemble")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(out_path, prediction=predictions, mean_t_neural_s=np.float32(mean_t))

    print(f"  predictions shape : {predictions.shape}")
    print(f"  mean_t_neural_s   : {mean_t:.6f} s")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
