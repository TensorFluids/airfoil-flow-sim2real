"""FST-style spatiotemporal HMoE mixer for RealPDE Track 1.

This is an adaptation of the FSTformer ideas to our setting:
  - Input is the stacked frozen-base predictions (FNO/CNO/Transolver): (B,T,H,W,9)
  - Temporal attention models dependencies across T
  - Spatial attention is local-window attention with a shortest-path-inspired bias
  - Temporal+spatial streams are fused by a learned gate alpha
  - HMoE block uses 6 experts (3 MLP + 3 temporal-conv experts)
  - Output predicts a residual on top of the simple mean of the 3 base models

The module operates in *normalised space* and returns normalised predictions
(B,T,H,W,3), so it plugs directly into our cached training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, D)
        b, t, h, w, d = x.shape
        z = x.permute(0, 2, 3, 1, 4).contiguous().view(b * h * w, t, d)
        z, _ = self.attn(z, z, z, need_weights=False)
        return z.view(b, h, w, t, d).permute(0, 3, 1, 2, 4).contiguous()


class WindowSpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window_size: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Learnable scalar for shortest-path-distance bias
        self.phi = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Local SPD proxy inside one window: Manhattan distance on a ws x ws grid
        ws = window_size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(ws, dtype=torch.float32),
                torch.arange(ws, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).view(ws * ws, 2)
        spd = (coords[:, None, :] - coords[None, :, :]).abs().sum(dim=-1)  # (Nw, Nw)
        self.register_buffer("spd_window", spd, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, D)
        b, t, h, w, d = x.shape
        ws = self.window_size
        if (h % ws) != 0 or (w % ws) != 0:
            raise ValueError(f"H/W must be divisible by window_size={ws}, got H={h}, W={w}")

        nwh = h // ws
        nww = w // ws
        nwindows = nwh * nww
        ntok = ws * ws

        # (B,T,H,W,D) -> (B*T*nwindows, ntok, D)
        z = x.view(b, t, nwh, ws, nww, ws, d)
        z = z.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(b * t * nwindows, ntok, d)

        qkv = self.qkv(z).view(b * t * nwindows, ntok, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, BTW, heads, ntok, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (BTW, heads, ntok, ntok)

        # SA_bias = phi(SPD(G)): here we use -|phi| * SPD so distant tokens are down-weighted
        bias = -torch.abs(self.phi) * self.spd_window.to(attn.dtype)
        attn = attn + bias[None, None, :, :]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (BTW, heads, ntok, head_dim)
        out = out.transpose(1, 2).contiguous().view(b * t * nwindows, ntok, d)
        out = self.proj(out)

        # Back to (B,T,H,W,D)
        out = out.view(b, t, nwh, nww, ws, ws, d)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(b, t, h, w, d)
        return out


class MLPExpert(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvExpert(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,H,W,D) -> conv along T for each spatial location
        b, t, h, w, d = x.shape
        z = x.permute(0, 2, 3, 4, 1).contiguous().view(b * h * w, d, t)
        z = F.gelu(self.conv1(z))
        z = self.dropout(z)
        z = self.conv2(z)
        return z.view(b, h, w, d, t).permute(0, 4, 1, 2, 3).contiguous()


class HMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 6,
        n_mlp: int = 3,
        n_tcn: int = 3,
        mlp_ratio: float = 2.0,
        tcn_kernel: int = 3,
        dropout: float = 0.0,
        divide_by_num_experts: bool = False,
    ):
        super().__init__()
        if n_mlp + n_tcn != n_experts:
            raise ValueError("n_mlp + n_tcn must equal n_experts")

        experts: List[nn.Module] = []
        for _ in range(n_mlp):
            experts.append(MLPExpert(d_model, mlp_ratio=mlp_ratio, dropout=dropout))
        for _ in range(n_tcn):
            experts.append(TemporalConvExpert(d_model, kernel_size=tcn_kernel, dropout=dropout))
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Linear(d_model, n_experts)
        self.divide_by_num_experts = divide_by_num_experts
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,H,W,D)
        gate_logits = self.gate(x)                      # (B,T,H,W,E)
        gate_probs = F.softmax(gate_logits, dim=-1)     # (B,T,H,W,E)

        expert_outs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outs, dim=-2)  # (B,T,H,W,E,D)

        out = (gate_probs.unsqueeze(-1) * expert_stack).sum(dim=-2)
        if self.divide_by_num_experts:
            out = out / float(self.n_experts)
        return out


@dataclass
class FSTHMoEConfig:
    in_channels: int = 9
    out_channels: int = 3
    d_model: int = 64
    num_layers: int = 3
    num_heads: int = 8
    window_size: int = 4
    n_experts: int = 6
    n_mlp_experts: int = 3
    n_tcn_experts: int = 3
    mlp_ratio: float = 2.0
    tcn_kernel: int = 3
    dropout: float = 0.0
    divide_by_num_experts: bool = False


class FSTHMoEBlock(nn.Module):
    def __init__(self, cfg: FSTHMoEConfig):
        super().__init__()
        self.temp_norm = nn.LayerNorm(cfg.d_model)
        self.spat_norm = nn.LayerNorm(cfg.d_model)
        self.moe_norm = nn.LayerNorm(cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)

        self.temp_attn = TemporalSelfAttention(cfg.d_model, cfg.num_heads, dropout=cfg.dropout)
        self.spat_attn = WindowSpatialAttention(
            cfg.d_model, cfg.num_heads, window_size=cfg.window_size, dropout=cfg.dropout
        )
        self.hmoe = HMoE(
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            n_mlp=cfg.n_mlp_experts,
            n_tcn=cfg.n_tcn_experts,
            mlp_ratio=cfg.mlp_ratio,
            tcn_kernel=cfg.tcn_kernel,
            dropout=cfg.dropout,
            divide_by_num_experts=cfg.divide_by_num_experts,
        )

        # alpha in fused = alpha * H_t + (1-alpha) * H_s
        self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x + self.temp_attn(self.temp_norm(x))
        s = x + self.spat_attn(self.spat_norm(x))

        alpha = torch.sigmoid(self.alpha_logit)
        fused = alpha * t + (1.0 - alpha) * s

        moe_out = self.hmoe(self.moe_norm(fused))
        out = self.out_norm(fused + moe_out)
        return out


class FSTHMoEMixer(nn.Module):
    """FST-style mixer for stacked base predictions.

    Input:
      stacked_norm: (B,T,H,W,9) where channels are concatenated [FNO(3), CNO(3), Transolver(3)]
    Output:
      pred_norm: (B,T,H,W,3) normalised prediction
    """

    def __init__(self, cfg: FSTHMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.in_channels, cfg.d_model)
        self.xy_proj = nn.Linear(2, cfg.d_model, bias=False)

        # s_in = z^- deg^- + z^+ deg^+
        self.z_in = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.z_out = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.s_embed = nn.Parameter(torch.zeros(cfg.d_model, dtype=torch.float32))

        self.blocks = nn.ModuleList([FSTHMoEBlock(cfg) for _ in range(cfg.num_layers)])
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_channels)

        # Start close to simple mean of frozen bases
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _grid_degree(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        deg = torch.full((h, w), 4.0, device=device, dtype=dtype)
        deg[0, :] -= 1.0
        deg[-1, :] -= 1.0
        deg[:, 0] -= 1.0
        deg[:, -1] -= 1.0
        return deg / 4.0  # normalise to [0.5, 1.0]

    @staticmethod
    def _xy_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=-1)  # (H,W,2)

    def forward(self, stacked_norm: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = stacked_norm.shape
        if c != self.cfg.in_channels:
            raise ValueError(f"Expected in_channels={self.cfg.in_channels}, got {c}")

        x = self.in_proj(stacked_norm)

        # Add static structural embeddings (s_in + xy)
        deg = self._grid_degree(h, w, x.device, x.dtype)                # (H,W)
        s_in = self.z_in * deg + self.z_out * deg                        # in/out same on 4-neighbour grid
        x = x + s_in[None, None, :, :, None] * self.s_embed[None, None, None, None, :]

        xy = self._xy_grid(h, w, x.device, x.dtype)                      # (H,W,2)
        x = x + self.xy_proj(xy)[None, None, :, :, :]

        for block in self.blocks:
            x = block(x)

        delta = self.out_proj(x)                                         # (B,T,H,W,3)

        # Residual around simple mean of the 3 frozen-base predictions
        base_mean = stacked_norm.view(b, t, h, w, 3, 3).mean(dim=-2)
        out = base_mean + delta
        out[..., 2] = 0.0
        return out

