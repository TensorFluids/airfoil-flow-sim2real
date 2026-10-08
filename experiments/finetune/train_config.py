"""Hyperparameter configuration for hybrid FNO+ConvLSTM training."""
from __future__ import annotations

import os
import torch


class Config:
    # ---- Data ---------------------------------------------------------------
    _HERE = os.path.dirname(os.path.abspath(__file__))
    data_dir = r"C:\Desktop\RealPDE\data\train_real"
    fno_ckpt = os.path.normpath(os.path.join(_HERE, r"..\..\checkpoints\sim_real_ft\sim_real_fno_fp16.pth"))
    save_dir = os.path.normpath(os.path.join(_HERE, r"..\..\checkpoints\hybrid"))

    # ---- Architecture -------------------------------------------------------
    hidden_dim = 32
    n_convlstm_layers = 2

    # ---- Training -----------------------------------------------------------
    batch_size = 16    # cached FNO preds make bigger batches cheap; also smooths gradients
    lr = 1e-4          # lowered further — beta=2.0 TKE term was causing divergence at 3e-4
    epochs = 100       # early stopping will halt sooner if it plateaus/diverges
    val_fraction = 0.15
    seed = 42

    # ---- Loss weights -------------------------------------------------------
    alpha = 1.0   # MSE weight
    beta  = 1.0   # TKE loss weight — lowered from 2.0; was destabilising training

    # ---- Misc ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0   # required by competition rules
