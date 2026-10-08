"""run_benchmark.py – Run the 3 baseline models on the example data and save
predictions.npz for each one.

Run from C:\\Desktop\\RealPDE\\Track1:
    python experiments/benchmark/run_benchmark.py

Reads:  experiments/results/ref/samples.npz    (key "input")
Writes: experiments/results/<model>/predictions.npz
        keys: "prediction" (N,20,32,64,3), "mean_t_neural_s" (scalar float32)

Normalization (mirrors submission_example_fno.py / benchmark eval.py):
    x_norm = (x - MEAN_IN) / STD_IN
    y_phys = y_norm * STD_TGT + MEAN_TGT
    y_phys[..., 2] = 0  (p is unmeasured in real data)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ── make kit importable ────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "_vendor"))

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────────
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "sim_real_ft")
REF_DIR = os.path.join(REPO_ROOT, "experiments", "results", "ref")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

MODELS = {
    "fno_fp16": os.path.join(CKPT_DIR, "sim_real_fno_fp16.pth"),
    "cno": os.path.join(CKPT_DIR, "sim_real_cno.pth"),
    "transolver": os.path.join(CKPT_DIR, "sim_real_transolver.pth"),
}

# ── GaussianNormalizer stats (fitted on train_real 32x64) ─────────────────────
MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

BATCH = 8  # reduce if GPU OOM


def run_model(name: str, ckpt_path: str, inputs: np.ndarray) -> tuple[np.ndarray, float]:
    """Run one model; return (predictions, mean_t_neural_s)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {name} on {device} …")
    model, meta = load_baseline(ckpt_path, device=device)
    model.eval()
    print(f"  Loaded  ({meta})")

    mean_in = torch.from_numpy(MEAN_IN).to(device)
    std_in = torch.from_numpy(STD_IN).to(device)
    mean_tgt = torch.from_numpy(MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(STD_TGT).to(device)

    N = inputs.shape[0]
    outs: list[np.ndarray] = []
    times: list[float] = []

    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = torch.from_numpy(
                np.ascontiguousarray(inputs[i : i + BATCH])
            ).to(device)

            # z-score normalise inputs
            xb = (xb - mean_in) / std_in

            t_start = time.perf_counter()
            yb = model(xb)
            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            # de-normalise to physical space
            yb = yb * std_tgt + mean_tgt
            yb_np = yb.float().cpu().numpy()
            yb_np[..., 2] = 0.0  # p is unmeasured

            n_samples = xb.shape[0]
            times.append((t_end - t_start) / n_samples)
            outs.append(yb_np)

    predictions = np.concatenate(outs, axis=0).astype(np.float32)
    mean_t = float(np.mean(times))
    return predictions, mean_t


def main() -> None:
    samples_path = os.path.join(REF_DIR, "samples.npz")
    if not os.path.exists(samples_path):
        sys.exit(
            f"samples.npz not found at {samples_path}\n"
            "Run make_targets.py first."
        )

    print(f"Loading inputs from {samples_path} …")
    with np.load(samples_path, allow_pickle=False) as d:
        inputs = d["input"].astype(np.float32)
    print(f"  inputs shape: {inputs.shape}")

    for name, ckpt_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        out_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.npz")

        try:
            predictions, mean_t = run_model(name, ckpt_path, inputs)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            continue
        except Exception as exc:
            print(f"  ERROR running {name}: {exc}")
            import traceback; traceback.print_exc()
            continue

        np.savez_compressed(
            out_path,
            prediction=predictions,
            mean_t_neural_s=np.float32(mean_t),
        )
        print(f"  predictions shape : {predictions.shape}")
        print(f"  mean_t_neural_s   : {mean_t:.6f} s")
        print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
