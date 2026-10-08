"""Run trained FST-HMoE mixer on benchmark samples and save predictions.npz.

Run from Track1 root:
    c:\\Python\\venv\\heta\\Scripts\\python.exe experiments\\benchmark\\run_fst_hmoe_benchmark.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "_vendor"))
sys.path.insert(0, EXP_DIR)

import torch  # noqa: E402

from load_baseline import load_baseline  # noqa: E402
from models.fst_hmoe_mixer import FSTHMoEConfig, FSTHMoEMixer  # noqa: E402
from models.stacked_ensemble import _MEAN_IN, _STD_IN, _MEAN_TGT, _STD_TGT  # noqa: E402

CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "sim_real_ft")
FNO_CKPT = os.path.join(CKPT_DIR, "sim_real_fno_fp16.pth")
CNO_CKPT = os.path.join(CKPT_DIR, "sim_real_cno.pth")
TRANSOLVER_CKPT = os.path.join(CKPT_DIR, "sim_real_transolver.pth")
MIXER_CKPT = os.path.join(REPO_ROOT, "checkpoints", "fst_hmoe", "best_fst_hmoe.pth")

REF_DIR = os.path.join(REPO_ROOT, "experiments", "results", "ref")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
BATCH = 8


def main() -> None:
    samples_path = os.path.join(REF_DIR, "samples.npz")
    if not os.path.exists(samples_path):
        sys.exit(f"samples.npz not found at {samples_path}. Run make_targets.py first.")
    if not os.path.exists(MIXER_CKPT):
        sys.exit(f"FST checkpoint not found at {MIXER_CKPT}. Train first.")

    with np.load(samples_path, allow_pickle=False) as d:
        inputs = d["input"].astype(np.float32)  # raw/physical space
    print(f"inputs shape: {inputs.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mean_in = torch.tensor(_MEAN_IN, dtype=torch.float32, device=device)
    std_in = torch.tensor(_STD_IN, dtype=torch.float32, device=device)
    mean_tgt = torch.tensor(_MEAN_TGT, dtype=torch.float32, device=device)
    std_tgt = torch.tensor(_STD_TGT, dtype=torch.float32, device=device)

    n = inputs.shape[0]

    # Run each frozen base model fully to completion, one at a time, and free it
    # before loading the next. This keeps only ONE model resident on the GPU at
    # any moment instead of 3+mixer simultaneously (avoids VRAM/power spikes that
    # crashed the machine when everything was loaded at once).
    base_outputs: dict[str, np.ndarray] = {}
    for name, ckpt_path in (("fno", FNO_CKPT), ("cno", CNO_CKPT), ("transolver", TRANSOLVER_CKPT)):
        print(f"Loading {name} …")
        model, _ = load_baseline(ckpt_path, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n, BATCH):
                xb = torch.from_numpy(np.ascontiguousarray(inputs[i : i + BATCH])).to(device)
                xb_norm = (xb - mean_in) / std_in
                pred = model(xb_norm)
                outs.append(pred.float().cpu().numpy())
        base_outputs[name] = np.concatenate(outs, axis=0)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  done, freed {name} from GPU")

    print("Loading FST-HMoE mixer …")
    ckpt = torch.load(MIXER_CKPT, map_location=device)
    model_cfg = FSTHMoEConfig(**ckpt["model_config"])
    mixer = FSTHMoEMixer(model_cfg).to(device)
    mixer.load_state_dict(ckpt["model_state_dict"])
    mixer.eval()
    print(f"Loaded FST checkpoint from epoch {ckpt.get('epoch')}, best_val={ckpt.get('best_val_loss'):.6f}")

    outs = []
    times: list[float] = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            p_fno = torch.from_numpy(base_outputs["fno"][i : i + BATCH]).to(device)
            p_cno = torch.from_numpy(base_outputs["cno"][i : i + BATCH]).to(device)
            p_trans = torch.from_numpy(base_outputs["transolver"][i : i + BATCH]).to(device)
            stacked = torch.cat([p_fno, p_cno, p_trans], dim=-1)

            t0 = time.perf_counter()
            pred_norm = mixer(stacked)
            pred = pred_norm * std_tgt + mean_tgt
            pred[..., 2] = 0.0
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            outs.append(pred.float().cpu().numpy())
            times.append((t1 - t0) / p_fno.shape[0])

    del mixer
    if device == "cuda":
        torch.cuda.empty_cache()

    predictions = np.concatenate(outs, axis=0).astype(np.float32)
    mean_t = float(np.mean(times))

    out_dir = os.path.join(RESULTS_DIR, "fst_hmoe")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(out_path, prediction=predictions, mean_t_neural_s=np.float32(mean_t))

    print(f"predictions shape: {predictions.shape}")
    print(f"mean_t_neural_s: {mean_t:.6f} s")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

