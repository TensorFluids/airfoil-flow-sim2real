"""Shared numerical and data helpers for the controlled stacked-mixer study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from finetune.dataset import RealPDEDataset, _MEAN_TGT, _STD_TGT
from scoring import (
    aggregate_sps, mvpe_rel_l2_per_sample, rel_l2_per_sample,
    score_error, score_sps, tke_rel_l2_per_sample,
)


CACHE_VERSION = 1


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_splits(names: list[str], seed: int, val_fraction: float, test_fraction: float) -> dict:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and leave training data.")
    names = sorted(set(names))
    random.Random(seed).shuffle(names)
    n_test = max(1, int(len(names) * test_fraction))
    n_val = max(1, int(len(names) * val_fraction))
    if n_test + n_val >= len(names):
        raise ValueError("Not enough trajectories for nonempty train/validation/test splits.")
    return {
        "train": sorted(names[n_test + n_val:]),
        "val": sorted(names[n_test:n_test + n_val]),
        "test": sorted(names[:n_test]),
    }


def build_datasets(data_dir: Path, splits: dict) -> dict:
    datasets = {}
    for split, names in splits.items():
        ds = RealPDEDataset(str(data_dir), split=split, file_names=names, normalize_targets=False)
        if {Path(path).name for path, _ in ds.samples} != set(names):
            raise ValueError(f"Some {split} trajectories contain no usable windows.")
        datasets[split] = ds
    return datasets


class CachedWindows(Dataset):
    """Memory-mapped arrays keep the full shared cache off the Python heap."""

    def __init__(self, root: Path, split: str):
        self.targets = np.load(root / f"{split}_targets.npy", mmap_mode="r", allow_pickle=False)
        self.stacked = np.load(root / f"{split}_stacked.npy", mmap_mode="r", allow_pickle=False)
        if self.targets.shape[:-1] != self.stacked.shape[:-1] or self.stacked.shape[-1] != 9:
            raise ValueError(f"Invalid cached shapes for {split}.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.stacked[index], dtype=np.float32, copy=True)),
        )


def prepare_cache(root: Path, datasets: dict, checkpoints: dict, identity: dict, device: str, batch_size: int) -> dict:
    ready_path = root / "ready.json"
    if ready_path.is_file():
        if json.loads(ready_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Cache identity mismatch; choose a new cache directory.")
        cached = {split: CachedWindows(root, split) for split in datasets}
        if any(len(cached[split]) != len(ds) for split, ds in datasets.items()):
            raise ValueError("Cache window counts do not match the saved split.")
        print(f"Reusing shared cache: {root}", flush=True)
        return cached

    from load_baseline import load_baseline

    root.mkdir(parents=True, exist_ok=True)
    # A missing ready.json means an interrupted cache must be rebuilt. Only
    # this fingerprint's generated arrays are overwritten, never source data.
    for split, ds in datasets.items():
        shape = tuple(ds[0][1].shape)
        np.lib.format.open_memmap(root / f"{split}_targets.npy", mode="w+", dtype=np.float32, shape=(len(ds), *shape)).flush()
        np.lib.format.open_memmap(root / f"{split}_stacked.npy", mode="w+", dtype=np.float16, shape=(len(ds), *shape[:-1], 9)).flush()

    for model_index, (name, checkpoint) in enumerate(checkpoints.items()):
        print(f"Caching frozen {name} on all three splits ...", flush=True)
        model, _ = load_baseline(str(checkpoint), device=device)
        model.eval()
        model.requires_grad_(False)
        with torch.no_grad():
            for split, ds in datasets.items():
                stacked = np.load(root / f"{split}_stacked.npy", mmap_mode="r+")
                targets = np.load(root / f"{split}_targets.npy", mmap_mode="r+")
                offset = 0
                for inputs, raw_targets in tqdm(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), desc=f"{name}/{split}"):
                    pred = model(inputs.to(device)).float().cpu().numpy()
                    if not np.isfinite(pred).all() or np.max(np.abs(pred)) > np.finfo(np.float16).max:
                        raise ValueError(f"Invalid frozen predictions from {name} on {split}.")
                    count = len(inputs)
                    stacked[offset:offset + count, ..., model_index * 3:(model_index + 1) * 3] = pred
                    if model_index == 0:
                        if not torch.isfinite(raw_targets).all():
                            raise ValueError(f"Non-finite targets on {split}.")
                        targets[offset:offset + count] = raw_targets.numpy()
                    offset += count
                stacked.flush()
                targets.flush()
                del stacked, targets
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_json(ready_path, identity)
    return {split: CachedWindows(root, split) for split in datasets}


def physical_tke(fields: torch.Tensor) -> torch.Tensor:
    """Scorer definition: half the sum of population temporal u/v variances."""
    return 0.5 * fields[..., :2].var(dim=1, unbiased=False).sum(dim=-1)


def loss_terms(pred_norm: torch.Tensor, target_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = pred_norm.new_tensor(_MEAN_TGT)
    std = pred_norm.new_tensor(_STD_TGT)
    field = F.mse_loss(pred_norm, (target_raw - mean) / std)
    pred_raw = pred_norm * std + mean
    tke = F.mse_loss(physical_tke(pred_raw), physical_tke(target_raw))
    return field, tke


def legacy_variance_loss(pred_norm: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    """Exact old penalty: normalized u/v/p, sample variance (correction=1)."""
    target_norm = (target_raw - pred_norm.new_tensor(_MEAN_TGT)) / pred_norm.new_tensor(_STD_TGT)
    return F.mse_loss(pred_norm.var(dim=1, unbiased=True), target_norm.var(dim=1, unbiased=True))


def gradient_diagnostics(model, train_cache, indices, device: str, scale: float) -> dict:
    """One reproducible training batch, no optimizer, no accumulated .grad."""
    model.eval()
    target, stacked = next(iter(DataLoader(Subset(train_cache, indices), batch_size=len(indices))))
    pred = model(stacked.to(device))
    target = target.to(device)
    field, tke = loss_terms(pred, target)
    legacy = legacy_variance_loss(pred, target)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    vectors = []
    for loss in (field, tke, legacy):
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                  for p, g in zip(parameters, grads)]).double())
    norms = [float(torch.linalg.vector_norm(g)) for g in vectors]
    if not np.isfinite(norms).all() or norms[0] <= 0:
        raise ValueError("Gradient diagnostic is degenerate; inspect before optimization.")
    return {
        "window_indices": indices, "source_split": "train",
        "field_gradient_norm": norms[0], "physical_tke_gradient_norm": norms[1],
        "legacy_variance_gradient_norm": norms[2],
        "scaled_tke_to_field_gradient_ratio": scale * norms[1] / norms[0],
        "legacy_to_field_gradient_ratio": norms[2] / norms[0],
        "field_tke_gradient_cosine": float(torch.dot(vectors[0], vectors[1]) / (norms[0] * norms[1])) if norms[1] else None,
        "field_legacy_gradient_cosine": float(torch.dot(vectors[0], vectors[2]) / (norms[0] * norms[2])) if norms[2] else None,
        "field_mse": float(field.detach()), "physical_tke_mse": float(tke.detach()),
        "legacy_variance_mse": float(legacy.detach()),
        "stage": "initialization_before_clipping_or_optimizer",
    }


def calibrate(model, train_cache, device: str, seed: int, windows: int, batch_size: int) -> dict:
    indices = np.random.default_rng(seed).choice(len(train_cache), size=min(windows, len(train_cache)), replace=False).tolist()
    sums = np.zeros(2, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for target, stacked in DataLoader(Subset(train_cache, indices), batch_size=batch_size, shuffle=False):
            field, tke = loss_terms(model(stacked.to(device)), target.to(device))
            sums += len(target) * np.array([field.item(), tke.item()])
    field_value, tke_value = sums / len(indices)
    if not np.isfinite(sums).all() or min(field_value, tke_value) <= 1e-20:
        raise ValueError("Loss calibration is degenerate; inspect the cached fields before training.")
    scale = float(field_value / tke_value)
    return {
        "source_split": "train",
        "window_indices": indices,
        "initial_field_mse": float(field_value),
        "initial_physical_tke_mse": float(tke_value),
        "tke_scale": scale,
        "scale_policy": "match_initial_training_loss_magnitudes_once_for_all_arms",
        "gradient_diagnostics": gradient_diagnostics(model, train_cache, indices[:batch_size], device, scale),
    }


def epoch_loader(dataset, batch_size: int, seed: int, epoch: int, train: bool) -> DataLoader:
    # An epoch's permutation is independent of previous epochs and resume state.
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, generator=generator, num_workers=0)


def run_epoch(model, loader, optimizer, device: str, weight: float, scale: float, grad_clip: float, legacy: bool = False) -> dict:
    model.train(optimizer is not None)
    totals = np.zeros(5, dtype=np.float64)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for targets, stacked in loader:
            pred = model(stacked.to(device))
            targets = targets.to(device)
            field, raw_tke = loss_terms(pred, targets)
            variance = legacy_variance_loss(pred, targets)
            penalty = variance if legacy else weight * scale * raw_tke
            loss = field + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; the ablation cannot be compared fairly.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            totals += len(targets) * np.array([loss.item(), field.item(), raw_tke.item(), variance.item(), penalty.item()])
            count += len(targets)
    total, field, raw_tke, variance, penalty = totals / count
    return {
        "loss": float(total), "field_mse": float(field), "tke_mse_raw": float(raw_tke),
        "tke_mse_scaled": float(scale * raw_tke), "tke_penalty": 0.0 if legacy else float(penalty),
        "legacy_variance_mse": float(variance), "auxiliary_penalty": float(penalty),
    }


def score_fields(pred: np.ndarray, target: np.ndarray) -> dict:
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid evaluation predictions.")
    rel_l2 = float(np.mean(rel_l2_per_sample(pred, target, 2)))
    tke = float(np.mean(tke_rel_l2_per_sample(pred, target, 2)))
    mvpe = float(np.mean(mvpe_rel_l2_per_sample(pred, target)))
    sps, coverage = aggregate_sps(pred, target, 2)
    return {
        "rel_l2_error": rel_l2, "tke_rel_l2_error": tke, "mvpe_rel_l2_error": mvpe,
        "rel_l2_score": score_error(rel_l2), "tke_score": score_error(tke),
        "mvpe_score": score_error(mvpe), "sps_score": score_sps(sps), "sps_coverage": coverage,
    }


def aggregate_trajectories(rows: dict) -> dict:
    windows = sum(row["windows"] for row in rows.values())
    elements = sum(row["scored_elements"] for row in rows.values())
    result = {}
    for error, score in (("rel_l2_error", "rel_l2_score"), ("tke_rel_l2_error", "tke_score"), ("mvpe_rel_l2_error", "mvpe_score")):
        result[error] = sum(row["metrics"][error] * row["windows"] for row in rows.values()) / windows
        result[score] = score_error(result[error])
    for key in ("sps_score", "sps_coverage"):
        result[key] = sum(row["metrics"][key] * row["scored_elements"] for row in rows.values()) / elements
    return result


def evaluate(model, cached_test, original_test, device: str, batch_size: int, output: Path) -> dict:
    """Score one trajectory at a time, preserving exact raw-target zero masks."""
    groups = {}
    for index, (path, start) in enumerate(original_test.samples):
        groups.setdefault(Path(path).name, []).append(index)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    model.eval()
    with torch.no_grad():
        for trajectory, indices in groups.items():
            predictions, targets = [], []
            for raw_target, stacked in DataLoader(Subset(cached_test, indices), batch_size=batch_size, shuffle=False):
                pred_norm = model(stacked.to(device))
                pred = pred_norm * pred_norm.new_tensor(_STD_TGT) + pred_norm.new_tensor(_MEAN_TGT)
                pred[..., 2] = 0.0
                predictions.append(pred.cpu().numpy())
                targets.append(raw_target.numpy())
            pred, target = np.concatenate(predictions), np.concatenate(targets)
            rows[trajectory] = {
                "windows": len(indices),
                "scored_elements": int(np.count_nonzero(target[..., :2])),
                "metrics": score_fields(pred, target),
            }
            np.savez_compressed(
                output / f"{Path(trajectory).stem}_predictions.npz", prediction=pred,
                start_frames=np.array([original_test.samples[i][1] for i in indices]),
            )
    result = {"aggregate": aggregate_trajectories(rows), "trajectories": rows}
    write_json(output / "scores.json", result)
    return result


def tradeoff_metrics(metrics: dict, control: dict, limits: dict) -> dict:
    changes = {}
    for key in ("rel_l2_error", "tke_rel_l2_error", "mvpe_rel_l2_error"):
        base = control[key]
        changes[f"{key}_change_percent"] = 100.0 * (metrics[key] - base) / base if base > 0 else None
    changes["sps_change_points"] = metrics["sps_score"] - control["sps_score"]
    if all(value is not None for value in limits.values()):
        comparable = all(changes[f"{key}_change_percent"] is not None for key in ("rel_l2_error", "tke_rel_l2_error", "mvpe_rel_l2_error"))
        changes["acceptable_tke_improvement"] = bool(
            comparable and changes["tke_rel_l2_error_change_percent"] < 0
            and changes["rel_l2_error_change_percent"] <= limits["max_l2_increase_pct"]
            and changes["mvpe_rel_l2_error_change_percent"] <= limits["max_mvpe_increase_pct"]
            and changes["sps_change_points"] >= -limits["max_sps_drop_points"]
        )
    return changes
