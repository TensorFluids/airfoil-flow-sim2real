"""Training script for FNO + ConvLSTM hybrid residual model.

Only the ConvLSTM branch weights are trained; FNO is frozen.

KEY OPTIMIZATION: the frozen FNO forward pass is precomputed ONCE for every
window (train + val) before the epoch loop starts, instead of being re-run
every single epoch. FNO is by far the most expensive part of the forward pass
(50M params, FFT-based spectral convs), while the trainable ConvLSTM branch is
tiny (173K params). Caching gives a large speedup (5-10x) with zero change in
the math, since FNO's output never changes (it's frozen and deterministic).

Run from the finetune/ directory:
    python train_hybrid.py
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — make Track1 root importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACK1 = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VENDOR = os.path.join(_TRACK1, "_vendor")
for _p in (_TRACK1, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EXPERIMENTS = os.path.join(_TRACK1, "experiments")
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_config import Config
from dataset import RealPDEDataset
from models.convlstm_fno import FNOConvLSTMHybrid
from mlflow_tracking import (
    log_epoch_metrics,
    log_training_artifact,
    log_training_params,
    training_run,
)

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def tke_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalise error in temporal variance (proxy for TKE). Both args (B,T,H,W,C) normalised."""
    return F.mse_loss(pred.var(dim=1), target.var(dim=1))


def combined_loss(pred, target, alpha, beta):
    mse = F.mse_loss(pred, target)
    tke = tke_loss(pred, target)
    return alpha * mse + beta * tke, mse.item(), tke.item()


# ---------------------------------------------------------------------------
# Precompute frozen FNO predictions once (major speedup)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_fno_cache(model: FNOConvLSTMHybrid, dataset, device, batch_size=16):
    """Run the frozen FNO once over the whole dataset (already-normalised windows).

    Returns aligned (inputs_norm, targets_norm, fno_preds_norm) CPU tensors,
    so we never need to touch the FNO again during the training loop.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.fno.eval()

    inputs_all, targets_all, fno_all = [], [], []
    for inp_norm, tgt_norm in tqdm(loader, desc="  caching FNO", unit="batch", leave=False):
        inp_norm = inp_norm.to(device)
        fno_pred_norm = model.fno(inp_norm)   # FNO operates directly in normalised space
        inputs_all.append(inp_norm.cpu())
        targets_all.append(tgt_norm.cpu())
        fno_all.append(fno_pred_norm.cpu())

    return (torch.cat(inputs_all), torch.cat(targets_all), torch.cat(fno_all))


# ---------------------------------------------------------------------------
# Epoch runner (entirely in normalised space, using cached FNO predictions)
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, cfg, train: bool, device, grad_clip: float = 1.0):
    model.encoder.train(train)
    model.decoder.train(train)
    total_loss = total_mse = total_tke = 0.0
    n = 0

    context = torch.enable_grad() if train else torch.no_grad()
    phase = "train" if train else "val"
    with context:
        pbar = tqdm(loader, desc=f"  {phase}", leave=False, unit="batch")
        for inp_norm, tgt_norm, fno_pred_norm in pbar:
            inp_norm = inp_norm.to(device)
            tgt_norm = tgt_norm.to(device)
            fno_pred_norm = fno_pred_norm.to(device)

            residual_norm = model.convlstm_branch(inp_norm)   # only trainable part runs
            pred_norm = fno_pred_norm + residual_norm

            loss, mse_v, tke_v = combined_loss(pred_norm, tgt_norm, cfg.alpha, cfg.beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.decoder.parameters()),
                    grad_clip,
                )
                optimizer.step()

            bs = inp_norm.size(0)
            total_loss += loss.item() * bs
            total_mse  += mse_v * bs
            total_tke  += tke_v * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.5f}", mse=f"{mse_v:.5f}", tke=f"{tke_v:.5f}")

    return total_loss / n, total_mse / n, total_tke / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _train(cfg):
    device = torch.device(cfg.device)

    data_dir = os.path.join(_HERE, cfg.data_dir)
    fno_ckpt = os.path.join(_HERE, cfg.fno_ckpt)
    save_dir = os.path.join(_HERE, cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading datasets …")
    train_ds = RealPDEDataset(data_dir, split="train", val_fraction=cfg.val_fraction, seed=cfg.seed)
    val_ds   = RealPDEDataset(data_dir, split="val",   val_fraction=cfg.val_fraction, seed=cfg.seed)
    print(f"  train windows: {len(train_ds)}, val windows: {len(val_ds)}")
    log_training_params({"train_windows": len(train_ds), "validation_windows": len(val_ds)})

    print(f"Loading FNO from: {fno_ckpt}")
    model = FNOConvLSTMHybrid(
        fno_ckpt_path=fno_ckpt, hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_convlstm_layers, device=str(device),
    ).to(device)

    trainable = list(model.encoder.parameters()) + list(model.decoder.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters (ConvLSTM branch): {n_params:,}")

    # ---- Precompute frozen FNO predictions once --------------------------
    print("Precomputing frozen FNO predictions (one-time cost) …")
    tr_inp, tr_tgt, tr_fno = precompute_fno_cache(model, train_ds, device, batch_size=16)
    vl_inp, vl_tgt, vl_fno = precompute_fno_cache(model, val_ds,   device, batch_size=16)
    print(f"  cached train: {tr_inp.shape}   cached val: {vl_inp.shape}")

    train_cached = TensorDataset(tr_inp, tr_tgt, tr_fno)
    val_cached   = TensorDataset(vl_inp, vl_tgt, vl_fno)

    train_loader = DataLoader(train_cached, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_cached,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ---- Training loop with early stopping --------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    epochs_no_improve = 0
    ckpt_path = os.path.join(save_dir, "best_hybrid.pth")

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValMSE':>10}  {'ValTKE':>10}  {'LR':>10}")
    print("-" * 70)

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        tr_loss, tr_mse, tr_tke = _run_epoch(model, train_loader, optimizer, cfg, train=True,  device=device)
        val_loss, val_mse, val_tke = _run_epoch(model, val_loader, optimizer, cfg, train=False, device=device)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(tr=f"{tr_loss:.5f}", val=f"{val_loss:.5f}", tke=f"{val_tke:.5f}")
        tqdm.write(
            f"{epoch:>6}  {tr_loss:>10.6f}  {val_loss:>10.6f}"
            f"  {val_mse:>10.6f}  {val_tke:>10.6f}  {cur_lr:>10.2e}"
        )
        log_epoch_metrics(
            {
                "train_loss": tr_loss,
                "validation_loss": val_loss,
                "train_mse": tr_mse,
                "validation_mse": val_mse,
                "train_tke": tr_tke,
                "validation_tke": val_tke,
                "learning_rate": cur_lr,
            },
            epoch,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "convlstm_state_dict": {
                        "encoder": model.encoder.state_dict(),
                        "decoder": model.decoder.state_dict(),
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "hidden_dim": cfg.hidden_dim,
                        "n_convlstm_layers": cfg.n_convlstm_layers,
                        "lr": cfg.lr,
                        "batch_size": cfg.batch_size,
                        "alpha": cfg.alpha,
                        "beta": cfg.beta,
                    },
                },
                ckpt_path,
            )
            tqdm.write(f"         ^ saved best (val={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"\nEarly stopping: no val improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best checkpoint: {ckpt_path}  (val_loss={best_val_loss:.6f})")
    log_training_params({"best_epoch": best_epoch, "best_validation_loss": best_val_loss})
    return ckpt_path


def main():
    cfg = Config()
    with training_run(
        "fno-convlstm-hybrid",
        {
            "model": "fno_convlstm_residual",
            "base_model": "FNO",
            "hidden_dim": cfg.hidden_dim,
            "convlstm_layers": cfg.n_convlstm_layers,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.lr,
            "epochs": cfg.epochs,
            "validation_fraction": cfg.val_fraction,
            "seed": cfg.seed,
            "alpha_mse": cfg.alpha,
            "beta_tke": cfg.beta,
            "num_workers": cfg.num_workers,
            "device": cfg.device,
        },
        tags={"kind": "trained_model", "model_family": "fno_convlstm"},
    ):
        checkpoint_path = _train(cfg)
        log_training_artifact(checkpoint_path)


if __name__ == "__main__":
    main()
