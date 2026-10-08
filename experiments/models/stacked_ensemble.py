"""Stacked ensemble: FNO + CNO + Transolver with a spatial attention mixer.

Architecture:
- All 3 base models are frozen (no grad)
- Each produces (B, T, H, W, 3) predictions
- Stack along channel dim → (B, T, H, W, 9)
- Small Conv3d mixer learns spatially/temporally varying blend weights → (B, T, H, W, 3)
- The mixer uses a residual around a channel-attention path so it starts near a simple average

Submission size budget:
  sim_real_fno_fp16.pth   ~201 MB
  sim_real_cno.pth         ~32 MB
  sim_real_transolver.pth  ~50 MB
  mixer checkpoint          < 1 MB
  Total                   ~284 MB  → pack CNO+Transolver to fp16 to fit under 256 MB
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrap (works when imported from anywhere inside Track1/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACK1 = os.path.abspath(os.path.join(_HERE, "..", ".."))
_VENDOR = os.path.join(_TRACK1, "_vendor")
for _p in (_TRACK1, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Normalisation stats (from submission_example_fno.py / train_real split)
# ---------------------------------------------------------------------------
_MEAN_IN  = [0.154960856,   -0.000513992854, 0.0]
_STD_IN   = [0.0968056545,   0.015960684,    1.0]
_MEAN_TGT = [0.154962569,   -0.000517793698, 0.0]
_STD_TGT  = [0.0968104079,   0.0159636438,   1.0]

# ---------------------------------------------------------------------------
# Spatial-attention mixer
# ---------------------------------------------------------------------------

class SpatialAttentionMixer(nn.Module):
    """Small Conv3d network that blends 3 model predictions.

    Input : (B, T, H, W, 9)  — 3 models × 3 channels stacked
    Output: (B, T, H, W, 3)  — blended prediction

    Design:
      1. Reorder to (B, 9, T, H, W) for Conv3d
      2. Two conv layers with GELU to extract spatial-temporal context
      3. A 1×1×1 conv to produce 3 output channels (the blend)
      4. Residual: add the channel-mean of the 3 inputs so the mixer starts
         near a simple average even with near-zero initialisation
    """

    def __init__(self, in_channels: int = 9, hidden: int = 32, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden,      hidden, kernel_size=3, padding=1)
        self.head  = nn.Conv3d(hidden, out_channels, kernel_size=1)

        # Only zero-init the FINAL head layer so the mixer *output* starts
        # near a simple average (residual ~0), while conv1/conv2 keep their
        # normal PyTorch init. Zero-initing every layer (as done previously)
        # compounds two near-zero layers in series, leaving almost no
        # input-dependent signal and causing vanishing gradients that made
        # training barely move regardless of LR.
        nn.init.normal_(self.head.weight, std=1e-3)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, 9)
        Returns:
            (B, T, H, W, 3)
        """
        B, T, H, W, C = x.shape
        # (B, 9, T, H, W)
        z = x.permute(0, 4, 1, 2, 3).contiguous()

        # Simple average residual: mean over the 3 model blocks (each 3 ch)
        residual = z.reshape(B, 3, 3, T, H, W).mean(dim=1)  # (B, 3, T, H, W)

        # Mixer path
        z = F.gelu(self.conv1(z))
        z = F.gelu(self.conv2(z))
        z = self.head(z)                                     # (B, 3, T, H, W)

        out = z + residual                                   # residual connection
        return out.permute(0, 2, 3, 4, 1).contiguous()      # (B, T, H, W, 3)


# ---------------------------------------------------------------------------
# Full stacked ensemble model
# ---------------------------------------------------------------------------

class StackedEnsemble(nn.Module):
    """FNO + CNO + Transolver frozen base models + trainable SpatialAttentionMixer.

    All three base models operate in normalised space internally.
    This wrapper handles raw-space I/O: normalise → bases → mixer → denormalise.
    """

    def __init__(
        self,
        fno_ckpt:        str,
        cno_ckpt:        str,
        transolver_ckpt: str,
        mixer_hidden:    int = 32,
        device:          str = "cpu",
    ):
        super().__init__()

        from load_baseline import load_baseline  # noqa: E402

        # Load and freeze all three base models
        fno, _        = load_baseline(fno_ckpt,        device=device)
        cno, _        = load_baseline(cno_ckpt,        device=device)
        transolver, _ = load_baseline(transolver_ckpt, device=device)

        for model in (fno, cno, transolver):
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        self.fno        = fno
        self.cno        = cno
        self.transolver = transolver

        # Trainable mixer (~50K params)
        self.mixer = SpatialAttentionMixer(in_channels=9, hidden=mixer_hidden, out_channels=3)

        # Normalisation buffers
        self.register_buffer("mean_in",  torch.tensor(_MEAN_IN,  dtype=torch.float32))
        self.register_buffer("std_in",   torch.tensor(_STD_IN,   dtype=torch.float32))
        self.register_buffer("mean_tgt", torch.tensor(_MEAN_TGT, dtype=torch.float32))
        self.register_buffer("std_tgt",  torch.tensor(_STD_TGT,  dtype=torch.float32))

        self.to(device)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, H, W, C=3) raw (unnormalised) input
        Returns:
            (B, T_out, H, W, 3) raw (unnormalised) prediction
        """
        x_norm = (x - self.mean_in) / self.std_in   # normalise once

        with torch.no_grad():
            p_fno        = self.fno(x_norm)          # (B, T, H, W, 3)
            p_cno        = self.cno(x_norm)          # (B, T, H, W, 3)
            p_transolver = self.transolver(x_norm)   # (B, T, H, W, 3)

        # Stack along channel dim
        stacked = torch.cat([p_fno, p_cno, p_transolver], dim=-1)  # (B, T, H, W, 9)

        # Mixer (operates in normalised space)
        mixed_norm = self.mixer(stacked)             # (B, T, H, W, 3)

        # Denormalise
        out = mixed_norm * self.std_tgt + self.mean_tgt
        out[..., 2] = 0.0                            # zero pressure channel
        return out
