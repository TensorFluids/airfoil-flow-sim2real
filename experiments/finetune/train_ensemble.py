"""Train the SpatialAttentionMixer on top of frozen FNO + CNO + Transolver.

KEY OPTIMIZATION: the 3 frozen base models (FNO + CNO + Transolver) are run
ONCE per window to produce a cached (B, T, H, W, 9) stack, instead of being
re-run every epoch. Only the ~35K-param mixer is ever trained, so caching the
much more expensive frozen forward passes gives a large speedup.

Run from Track1/ root:
    c:\Python\venv\heta\Scripts\python.exe experiments\finetune\train_ensemble.py
"""
from __future__ import annotations

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_TRACK1 = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VENDOR = os.path.join(_TRACK1, "_vendor")
_EXPDIR = os.path.join(_TRACK1, "experiments")
for _p in (_TRACK1, _VENDOR, _EXPDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from finetune.dataset import RealPDEDataset
from models.stacked_ensemble import StackedEnsemble

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR        = r"C:\Desktop\RealPDE\data\train_real"
FNO_CKPT        = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_fno_fp16.pth")
CNO_CKPT        = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_cno.pth")
TRANSOLVER_CKPT = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_transolver.pth")
SAVE_DIR        = os.path.join(_TRACK1, "checkpoints", "ensemble")

MIXER_HIDDEN  = 32
BATCH_SIZE    = 16      # cached base preds make bigger batches cheap; also smooths gradients
LR            = 1e-4    # lowered from 3e-4 — beta=2.0 TKE term was causing divergence
EPOCHS        = 100
VAL_FRACTION  = 0.15
SEED          = 42
ALPHA         = 1.0     # MSE weight
BETA          = 1.0     # TKE weight — lowered from 2.0; was destabilising training
GRAD_CLIP     = 1.0
PATIENCE      = 10      # early-stopping patience (epochs w/o val improvement)
CACHE_BATCH   = 8       # batch size while precomputing frozen base predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def tke_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.var(dim=1), target.var(dim=1))


def combined_loss(pred, target, alpha, beta):
    mse  = F.mse_loss(pred, target)
    tke  = tke_loss(pred, target)
    return alpha * mse + beta * tke, mse.item(), tke.item()


# ---------------------------------------------------------------------------
# Precompute frozen base-model predictions once (major speedup)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_base_cache(model: StackedEnsemble, dataset, device, batch_size=8):
    """Run the 3 frozen base models once over the whole dataset.

    dataset yields already-normalised (inp_norm, tgt_norm) pairs. Returns
    aligned (targets_norm, stacked_preds_norm) CPU tensors where
    stacked_preds_norm is (N, T, H, W, 9) = concat([fno, cno, transolver]).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.fno.eval(); model.cno.eval(); model.transolver.eval()

    targets_all, stacked_all = [], []
    for inp_norm, tgt_norm in tqdm(loader, desc="  caching bases", unit="batch", leave=False):
        inp_norm = inp_norm.to(device)
        p_fno        = model.fno(inp_norm)
        p_cno        = model.cno(inp_norm)
        p_transolver = model.transolver(inp_norm)
        stacked = torch.cat([p_fno, p_cno, p_transolver], dim=-1)  # (B,T,H,W,9)

        targets_all.append(tgt_norm.cpu())
        stacked_all.append(stacked.cpu())

    return torch.cat(targets_all), torch.cat(stacked_all)


# ---------------------------------------------------------------------------
# Epoch runner (entirely in normalised space, using cached base predictions)
# ---------------------------------------------------------------------------

def run_epoch(mixer, loader, optimizer, train: bool, device, grad_clip=GRAD_CLIP):
    mixer.train(train)
    total_loss = total_mse = total_tke = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    phase = "train" if train else "val"
    with ctx:
        pbar = tqdm(loader, desc=f"  {phase}", leave=False, unit="batch")
        for tgt_norm, stacked_norm in pbar:
            tgt_norm     = tgt_norm.to(device)
            stacked_norm = stacked_norm.to(device)

            pred_norm = mixer(stacked_norm)   # (B, T, H, W, 3)

            loss, mse_v, tke_v = combined_loss(pred_norm, tgt_norm, ALPHA, BETA)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mixer.parameters(), grad_clip)
                optimizer.step()

            bs = tgt_norm.size(0)
            total_loss += loss.item() * bs
            total_mse  += mse_v * bs
            total_tke  += tke_v * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.5f}", mse=f"{mse_v:.5f}", tke=f"{tke_v:.5f}")

    return total_loss / n, total_mse / n, total_tke / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    print("Loading datasets …")
    train_ds = RealPDEDataset(DATA_DIR, split="train", val_fraction=VAL_FRACTION, seed=SEED)
    val_ds   = RealPDEDataset(DATA_DIR, split="val",   val_fraction=VAL_FRACTION, seed=SEED)
    print(f"  train: {len(train_ds)} windows   val: {len(val_ds)} windows")

    print("Building StackedEnsemble …")
    model = StackedEnsemble(
        fno_ckpt=FNO_CKPT, cno_ckpt=CNO_CKPT, transolver_ckpt=TRANSOLVER_CKPT,
        mixer_hidden=MIXER_HIDDEN, device=str(device),
    )
    n_trainable = sum(p.numel() for p in model.mixer.parameters())
    print(f"  Trainable (mixer only): {n_trainable:,} params")

    # ---- Precompute frozen base-model predictions once --------------------
    print("Precomputing frozen FNO+CNO+Transolver predictions (one-time cost) …")
    tr_tgt, tr_stack = precompute_base_cache(model, train_ds, device, batch_size=CACHE_BATCH)
    vl_tgt, vl_stack = precompute_base_cache(model, val_ds,   device, batch_size=CACHE_BATCH)
    print(f"  cached train: {tr_stack.shape}   cached val: {vl_stack.shape}")

    train_cached = TensorDataset(tr_tgt, tr_stack)
    val_cached   = TensorDataset(vl_tgt, vl_stack)

    train_loader = DataLoader(train_cached, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_cached,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.mixer.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    ckpt_path = os.path.join(SAVE_DIR, "best_ensemble_mixer.pth")

    best_val = float("inf")
    epochs_no_improve = 0
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValMSE':>10}  {'ValTKE':>10}  {'LR':>10}")
    print("-" * 68)

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        tr_loss, tr_mse, tr_tke = run_epoch(model.mixer, train_loader, optimizer, train=True,  device=device)
        vl_loss, vl_mse, vl_tke = run_epoch(model.mixer, val_loader,   optimizer, train=False, device=device)
        scheduler.step(vl_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        epoch_bar.set_postfix(tr=f"{tr_loss:.5f}", val=f"{vl_loss:.5f}", tke=f"{vl_tke:.5f}")
        tqdm.write(f"{epoch:>6}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  {vl_mse:>10.6f}  {vl_tke:>10.6f}  {cur_lr:>10.2e}")

        if vl_loss < best_val - 1e-6:
            best_val = vl_loss
            epochs_no_improve = 0
            torch.save({
                "mixer_state_dict": model.mixer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "mixer_hidden": MIXER_HIDDEN,
            }, ckpt_path)
            tqdm.write(f"         ^ saved best (val={best_val:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                tqdm.write(f"\nEarly stopping: no val improvement for {PATIENCE} epochs.")
                break

    print(f"\nDone. Best checkpoint: {ckpt_path}  (val_loss={best_val:.6f})")


if __name__ == "__main__":
    main()
