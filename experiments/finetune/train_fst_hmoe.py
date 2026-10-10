"""Train an FST-style HMoE mixer over frozen FNO/CNO/Transolver predictions.

This adapts FSTformer concepts to RealPDE Track 1:
  - temporal attention + spatial attention
  - learned temporal/spatial fusion gate
  - HMoE with 6 experts (3 MLP + 3 temporal-conv)
  - Kernel-MSE robustness loss (+ optional TKE term)

The heavy frozen base forward passes are cached once before training.

Run from Track1 root:
    c:\\Python\\venv\\heta\\Scripts\\python.exe experiments\\finetune\\train_fst_hmoe.py
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACK1 = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VENDOR = os.path.join(_TRACK1, "_vendor")
_EXPDIR = os.path.join(_TRACK1, "experiments")
for _p in (_TRACK1, _VENDOR, _EXPDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune.dataset import RealPDEDataset  # noqa: E402
from load_baseline import load_baseline  # noqa: E402
from models.fst_hmoe_mixer import FSTHMoEConfig, FSTHMoEMixer  # noqa: E402
from mlflow_tracking import (  # noqa: E402
    log_epoch_metrics,
    log_training_artifact,
    log_training_params,
    training_run,
)


@dataclass
class TrainConfig:
    # Data / checkpoints
    data_dir: str = r"C:\Desktop\RealPDE\data\train_real"
    fno_ckpt: str = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_fno_fp16.pth")
    cno_ckpt: str = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_cno.pth")
    transolver_ckpt: str = os.path.join(_TRACK1, "checkpoints", "sim_real_ft", "sim_real_transolver.pth")
    save_dir: str = os.path.join(_TRACK1, "checkpoints", "fst_hmoe")
    checkpoint_name: str = "best_fst_hmoe.pth"
    # Optional checkpoint to continue from. The existing checkpoint has no
    # optimizer/scheduler state, so that older run resumes with fresh states.
    # resume_checkpoint: str | None = None
    resume_checkpoint: str | None = os.path.join(
    _TRACK1, "checkpoints", "fst_hmoe", "best_fst_hmoe.pth"
    )
    epochs: int = 4

    # Split
    val_fraction: float = 0.15
    seed: int = 42

    # FST-style mixer architecture
    d_model: int = 32
    num_layers: int = 1
    num_heads: int = 4
    window_size: int = 4
    n_experts: int = 6
    n_mlp_experts: int = 3
    n_tcn_experts: int = 3
    mlp_ratio: float = 2.0
    tcn_kernel: int = 3
    dropout: float = 0.0

    # Optimisation
    batch_size: int = 8
    cache_batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5

    # Loss
    kmse_weight: float = 1.0
    tke_weight: float = 0.35
    mse_weight: float = 0.0
    kmse_sigma: float = math.sqrt(2.0) / 2.0

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    cache_dtype: str = "float16"
    use_amp: bool = True


def kernel_mse_loss(pred: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    err2 = (pred - target) ** 2
    return (1.0 - torch.exp(-err2 / (2.0 * sigma * sigma))).mean()


def tke_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.var(dim=1), target.var(dim=1))


def combined_loss(pred: torch.Tensor, target: torch.Tensor, cfg: TrainConfig):
    kmse = kernel_mse_loss(pred, target, cfg.kmse_sigma)
    tke = tke_loss(pred, target)
    mse = F.mse_loss(pred, target)
    total = cfg.kmse_weight * kmse + cfg.tke_weight * tke + cfg.mse_weight * mse
    return total, kmse.item(), tke.item(), mse.item()


@torch.no_grad()
def _load_frozen_bases(cfg: TrainConfig, device: torch.device):
    fno, _ = load_baseline(cfg.fno_ckpt, device=str(device))
    cno, _ = load_baseline(cfg.cno_ckpt, device=str(device))
    transolver, _ = load_baseline(cfg.transolver_ckpt, device=str(device))
    for model in (fno, cno, transolver):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return fno, cno, transolver


@torch.no_grad()
def precompute_base_cache(
    fno,
    cno,
    transolver,
    dataset: RealPDEDataset,
    device: torch.device,
    batch_size: int = 8,
) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    targets_all: List[torch.Tensor] = []
    stacked_all: List[torch.Tensor] = []

    for batch_idx, (inp_norm, tgt_norm) in enumerate(tqdm(loader, desc="  caching bases", unit="batch", leave=False)):
        inp_norm = inp_norm.to(device)
        p_fno = fno(inp_norm)
        p_cno = cno(inp_norm)
        p_transolver = transolver(inp_norm)
        stacked = torch.cat([p_fno, p_cno, p_transolver], dim=-1)  # (B,T,H,W,9)

        # Store cache on CPU in fp16 to cut RAM pressure roughly in half.
        # Training converts back to fp32 on-device before computing losses.
        targets_all.append(tgt_norm.detach().cpu().to(dtype=torch.float16))
        stacked_all.append(stacked.detach().cpu().to(dtype=torch.float16))

        del inp_norm, tgt_norm, p_fno, p_cno, p_transolver, stacked
        if device.type == "cuda" and (batch_idx + 1) % 16 == 0:
            torch.cuda.empty_cache()

    targets = torch.cat(targets_all)
    stacked = torch.cat(stacked_all)
    return TensorDataset(targets, stacked)


def build_cached_datasets(cfg: TrainConfig, device: torch.device) -> Tuple[TensorDataset, TensorDataset]:
    print("Loading RealPDE datasets …")
    train_ds = RealPDEDataset(cfg.data_dir, split="train", val_fraction=cfg.val_fraction, seed=cfg.seed)
    val_ds = RealPDEDataset(cfg.data_dir, split="val", val_fraction=cfg.val_fraction, seed=cfg.seed)
    print(f"  train windows: {len(train_ds)}  val windows: {len(val_ds)}")
    log_training_params({"train_windows": len(train_ds), "validation_windows": len(val_ds)})

    print("Loading frozen base models (FNO/CNO/Transolver) …")
    fno, cno, transolver = _load_frozen_bases(cfg, device)

    print("Precomputing frozen base predictions (one-time cost) …")
    train_cached = precompute_base_cache(fno, cno, transolver, train_ds, device, batch_size=cfg.cache_batch_size)
    val_cached = precompute_base_cache(fno, cno, transolver, val_ds, device, batch_size=cfg.cache_batch_size)
    print(
        "  cached train: "
        f"{train_cached.tensors[1].shape}   cached val: {val_cached.tensors[1].shape}"
    )
    return train_cached, val_cached


def _run_epoch(
    model: FSTHMoEMixer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainConfig,
    train: bool,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train(train)
    total_loss = total_kmse = total_tke = total_mse = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    phase = "train" if train else "val"
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    with ctx:
        pbar = tqdm(loader, desc=f"  {phase}", leave=False, unit="batch")
        for tgt_norm, stacked_norm in pbar:
            tgt_norm = tgt_norm.to(device=device, dtype=torch.float32)
            stacked_norm = stacked_norm.to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_norm = model(stacked_norm)
                loss, kmse_v, tke_v, mse_v = combined_loss(pred_norm.float(), tgt_norm.float(), cfg)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

            bs = tgt_norm.size(0)
            total_loss += loss.item() * bs
            total_kmse += kmse_v * bs
            total_tke += tke_v * bs
            total_mse += mse_v * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.5f}", kmse=f"{kmse_v:.5f}", tke=f"{tke_v:.5f}")

    return {
        "loss": total_loss / n,
        "kmse": total_kmse / n,
        "tke": total_tke / n,
        "mse": total_mse / n,
    }


def _maybe_update_live_plot(history: Dict[str, List[float]], figure_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display
    except Exception:
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    ax[0].plot(epochs, history["train_loss"], label="train_loss", color="tab:blue")
    ax[0].plot(epochs, history["val_loss"], label="val_loss", color="tab:orange")
    ax[0].set_title("Total loss")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, history["val_mse"], label="val_mse", color="tab:green")
    ax[1].plot(epochs, history["val_tke"], label="val_tke", color="tab:red")
    ax[1].set_title("Validation components")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=120)
    clear_output(wait=True)
    display(fig)
    plt.close(fig)


def fit_with_cached(
    cfg: TrainConfig,
    train_cached: TensorDataset,
    val_cached: TensorDataset,
    live_plot: bool = False,
) -> Tuple[Dict[str, List[float]], str]:
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, cfg.checkpoint_name)
    live_curve_path = os.path.join(cfg.save_dir, "live_curves.png")

    device = torch.device(cfg.device)
    train_loader = DataLoader(
        train_cached,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_cached,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model_cfg = FSTHMoEConfig(
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        n_experts=cfg.n_experts,
        n_mlp_experts=cfg.n_mlp_experts,
        n_tcn_experts=cfg.n_tcn_experts,
        mlp_ratio=cfg.mlp_ratio,
        tcn_kernel=cfg.tcn_kernel,
        dropout=cfg.dropout,
    )
    model = FSTHMoEMixer(model_cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (FST-HMoE mixer): {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp and device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_kmse": [], "val_kmse": [],
        "train_tke": [], "val_tke": [],
        "train_mse": [], "val_mse": [],
        "lr": [],
    }

    best_val = float("inf")
    best_epoch = 0
    stale = 0

    start_epoch = 1
    if cfg.resume_checkpoint:
        resume_path = os.path.abspath(cfg.resume_checkpoint)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        saved_model_cfg = checkpoint.get("model_config")
        if saved_model_cfg is not None and saved_model_cfg != asdict(model_cfg):
            raise ValueError(
                "Resume checkpoint model_config does not match the current model configuration. "
                "Use the same architecture settings as the checkpoint."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history.update(checkpoint.get("history", {}))
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(
            f"Resuming weights/history from {resume_path} at epoch {start_epoch}; "
            f"best validation loss so far is {best_val:.6f}."
        )
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"Target epochs ({cfg.epochs}) must be greater than the checkpoint epoch "
                f"({start_epoch - 1})."
            )

    print(
        f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValKMSE':>10}  "
        f"{'ValMSE':>10}  {'ValTKE':>10}  {'LR':>10}"
    )
    print("-" * 86)

    epoch_bar = tqdm(range(start_epoch, cfg.epochs + 1), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        tr = _run_epoch(model, train_loader, optimizer, cfg, train=True, device=device, scaler=scaler)
        vl = _run_epoch(model, val_loader, optimizer=None, cfg=cfg, train=False, device=device)
        scheduler.step(vl["loss"])
        lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_kmse"].append(tr["kmse"])
        history["val_kmse"].append(vl["kmse"])
        history["train_tke"].append(tr["tke"])
        history["val_tke"].append(vl["tke"])
        history["train_mse"].append(tr["mse"])
        history["val_mse"].append(vl["mse"])
        history["lr"].append(lr)

        epoch_bar.set_postfix(tr=f"{tr['loss']:.5f}", val=f"{vl['loss']:.5f}", tke=f"{vl['tke']:.5f}")
        tqdm.write(
            f"{epoch:>6}  {tr['loss']:>10.6f}  {vl['loss']:>10.6f}  {vl['kmse']:>10.6f}  "
            f"{vl['mse']:>10.6f}  {vl['tke']:>10.6f}  {lr:>10.2e}"
        )
        log_epoch_metrics(
            {
                "train_loss": tr["loss"],
                "validation_loss": vl["loss"],
                "train_kmse": tr["kmse"],
                "validation_kmse": vl["kmse"],
                "train_tke": tr["tke"],
                "validation_tke": vl["tke"],
                "train_mse": tr["mse"],
                "validation_mse": vl["mse"],
                "learning_rate": lr,
            },
            epoch,
        )

        if live_plot:
            _maybe_update_live_plot(history, live_curve_path)

        if vl["loss"] < best_val - 1e-6:
            best_val = vl["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "history": history,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                ckpt_path,
            )
            tqdm.write(f"         ^ saved best (val={best_val:.6f})")
        else:
            stale += 1
            if stale >= cfg.early_stopping_patience:
                tqdm.write(
                    f"\nEarly stopping: no val improvement for {cfg.early_stopping_patience} epochs."
                )
                break

    print(f"\nDone. Best checkpoint: {ckpt_path}  (val_loss={best_val:.6f})")
    log_training_params({"best_epoch": best_epoch, "best_validation_loss": best_val})
    if live_plot:
        print(f"Live curve image: {live_curve_path}")
    return history, ckpt_path


def run_training(cfg: TrainConfig, live_plot: bool = False) -> Tuple[Dict[str, List[float]], str]:
    device = torch.device(cfg.device)
    train_cached, val_cached = build_cached_datasets(cfg, device=device)
    return fit_with_cached(cfg, train_cached, val_cached, live_plot=live_plot)


def main():
    cfg = TrainConfig()
    run_name = "fst-hmoe-mixer-resume" if cfg.resume_checkpoint else "fst-hmoe-mixer"
    with training_run(
        run_name,
        asdict(cfg),
        tags={
            "kind": "trained_model",
            "model_family": "fst_hmoe_ensemble",
            "continuation": str(bool(cfg.resume_checkpoint)).lower(),
        },
    ):
        _, checkpoint_path = run_training(cfg, live_plot=False)
        log_training_artifact(checkpoint_path)


if __name__ == "__main__":
    main()
