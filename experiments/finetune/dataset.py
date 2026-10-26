"""Dataset class for RealPDE Track 1 (train_real / train_sim).

Reads all .h5 files in a directory, sub-samples spatially, builds
non-overlapping input/target windows of length 20, and applies
GaussianNormalizer statistics.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError as e:
    raise ImportError("h5py is required for RealPDEDataset") from e

# ---------------------------------------------------------------------------
# Normalisation stats (fitted on train_real 32x64)
# ---------------------------------------------------------------------------
_MEAN_IN  = np.array([0.154960856,   -0.000513992854, 0.0], dtype=np.float32)
_STD_IN   = np.array([0.0968056545,   0.015960684,    1.0], dtype=np.float32)
_MEAN_TGT = np.array([0.154962569,   -0.000517793698, 0.0], dtype=np.float32)
_STD_TGT  = np.array([0.0968104079,   0.0159636438,   1.0], dtype=np.float32)

_DEFAULT_EXCLUDE = ["7575_0.h5"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_uv(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read (u, v) arrays from an h5 file, handling two possible layouts."""
    with h5py.File(h5_path, "r") as f:
        if "u" in f and "v" in f:
            u = np.asarray(f["u"], dtype=np.float32)
            v = np.asarray(f["v"], dtype=np.float32)
        elif "measured_data" in f:
            u = np.asarray(f["measured_data/u"], dtype=np.float32)
            v = np.asarray(f["measured_data/v"], dtype=np.float32)
        else:
            raise KeyError(f"Cannot find u/v in {h5_path}; top-level keys: {list(f.keys())}")
    return u, v  # both (T, H_full, W_full)


def _build_windows(T: int, window: int = 20) -> List[int]:
    """Return start indices i such that [i*w : i*w+2w] fits within T."""
    starts = []
    i = 0
    while i * window + 2 * window <= T:
        starts.append(i * window)
        i += 1
    return starts


def _get_T(h5_path: str) -> int:
    """Return the number of frames without reading the full u/v arrays into memory."""
    with h5py.File(h5_path, "r") as f:
        if "u" in f:
            return f["u"].shape[0]
        elif "measured_data" in f:
            return f["measured_data/u"].shape[0]
        raise KeyError(f"Cannot find u/v in {h5_path}; top-level keys: {list(f.keys())}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealPDEDataset(Dataset):
    """PyTorch Dataset for RealPDE Track 1 .h5 files.

    Args:
        data_dir:     directory containing *.h5 files
        sub_s:        spatial subsampling factor (default 2 → 32×64)
        split:        'train' or 'val'
        val_fraction: fraction of files held out for validation
        seed:         random seed for file-level train/val split
        exclude:      list of filenames (basename) to skip entirely
        file_names:   explicit trajectory list, bypassing the random split
        normalize_targets: False returns physical targets (including exact zero masks)
    """

    def __init__(
        self,
        data_dir: str,
        sub_s: int = 2,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        exclude: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.sub_s = sub_s
        self.split = split
        self.normalize_targets = normalize_targets

        if exclude is None:
            exclude = _DEFAULT_EXCLUDE
        exclude_set = set(exclude)

        # Collect all h5 files
        all_files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".h5") and f not in exclude_set
        )
        if not all_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

        # File-level train/val split (shuffle then slice)
        rng = random.Random(seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        if file_names is not None:
            if len(set(file_names)) != len(file_names):
                raise ValueError("Explicit file_names must not contain duplicates.")
            if not file_names or any(name not in all_files for name in file_names):
                raise ValueError("Explicit file_names must name existing, non-excluded trajectories.")
            chosen_files = list(file_names)
        elif split == "val":
            chosen_files = shuffled[:n_val]
        else:
            chosen_files = shuffled[n_val:]

        # Build (file_path, start_frame) index
        self.samples: List[Tuple[str, int]] = []
        for fname in chosen_files:
            fpath = os.path.join(data_dir, fname)
            try:
                T = _get_T(fpath)
            except Exception as exc:
                if file_names is not None:
                    raise RuntimeError(f"Cannot read explicitly selected trajectory {fname}") from exc
                print(f"[RealPDEDataset] Skipping {fname}: {exc}")
                continue
            for start in _build_windows(T, window=20):
                self.samples.append((fpath, start))

        if not self.samples:
            raise RuntimeError(
                f"No valid windows found in {data_dir} for split='{split}'"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath, start = self.samples[idx]
        sub = self.sub_s

        with h5py.File(fpath, "r") as f:
            if "u" in f:
                u_win = np.asarray(
                    f["u"][start : start + 40 : 1, ::sub, ::sub], dtype=np.float32
                )
                v_win = np.asarray(
                    f["v"][start : start + 40 : 1, ::sub, ::sub], dtype=np.float32
                )
            else:
                u_win = np.asarray(
                    f["measured_data/u"][start : start + 40 : 1, ::sub, ::sub],
                    dtype=np.float32,
                )
                v_win = np.asarray(
                    f["measured_data/v"][start : start + 40 : 1, ::sub, ::sub],
                    dtype=np.float32,
                )

        # Stack channels: (40, H, W, 3)
        p_win = np.zeros_like(u_win)
        data = np.stack([u_win, v_win, p_win], axis=-1)  # (40, H, W, 3)

        inp = data[:20]   # (20, H, W, 3)
        tgt = data[20:]   # (20, H, W, 3)

        # Normalise
        inp_norm = (inp - _MEAN_IN) / _STD_IN
        tgt_norm = (tgt - _MEAN_TGT) / _STD_TGT if self.normalize_targets else tgt

        return (
            torch.from_numpy(inp_norm).float(),  # (20, H, W, 3)
            torch.from_numpy(tgt_norm).float(),  # (20, H, W, 3)
        )
