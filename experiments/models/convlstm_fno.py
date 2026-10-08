"""FNO + ConvLSTM residual hybrid model.

Architecture:
- Frozen pretrained FNO provides base prediction
- ConvLSTM encoder learns temporal residuals that FNO misses (e.g., TKE)
- Final prediction = FNO_output + ConvLSTM_residual
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ConvLSTM cell
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Args:
        input_dim:   number of input channels
        hidden_dim:  number of hidden state channels
        kernel_size: convolution kernel size (default 3)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        # Gates i, f, o, g packed into one conv for efficiency
        self.gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x:  (B, C_in, H, W)
            h:  (B, hidden_dim, H, W)
            c:  (B, hidden_dim, H, W)
        Returns:
            h_new, c_new both (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)  # (B, C_in + hidden_dim, H, W)
        gates = self.gates(combined)          # (B, 4*hidden_dim, H, W)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size: int, height: int, width: int, device):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


# ---------------------------------------------------------------------------
# Stacked ConvLSTM encoder
# ---------------------------------------------------------------------------

class ConvLSTMEncoder(nn.Module):
    """Stack of ConvLSTM layers that encode a sequence into a final hidden state.

    Args:
        input_dim:   C of input frames
        hidden_dim:  hidden channels per layer
        n_layers:    number of stacked ConvLSTM layers
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            Final hidden state h of last layer: (B, hidden_dim, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialise hidden/cell states for each layer
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        for t in range(T):
            inp = x[:, t]  # (B, C, H, W)
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h, c = cell(inp, h, c)
                states[layer_idx] = (h, c)
                inp = h  # next layer receives current layer's hidden state

        # Return final hidden state of last layer
        return states[-1][0]  # (B, hidden_dim, H, W)


# ---------------------------------------------------------------------------
# Residual decoder
# ---------------------------------------------------------------------------

class ResidualDecoder(nn.Module):
    """Expand ConvLSTM hidden state to (B, T_out*C_out, H, W), then reshape.

    Args:
        hidden_dim: input channels (ConvLSTM hidden dim)
        T_out:      number of output time steps
        C_out:      number of output channels
    """

    def __init__(self, hidden_dim: int, T_out: int = 20, C_out: int = 3):
        super().__init__()
        self.T_out = T_out
        self.C_out = C_out
        mid = hidden_dim * 2
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, T_out * C_out, kernel_size=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim, H, W)
        Returns:
            (B, T_out, H, W, C_out)
        """
        out = self.net(h)                                      # (B, T_out*C_out, H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.T_out, self.C_out, H, W)       # (B, T_out, C_out, H, W)
        out = out.permute(0, 1, 3, 4, 2)                      # (B, T_out, H, W, C_out)
        return out.contiguous()


# ---------------------------------------------------------------------------
# FNO + ConvLSTM hybrid
# ---------------------------------------------------------------------------

# Normalization statistics (fitted on train_real 32x64, channel order [u,v,p])
_MEAN_IN  = [0.154960856,   -0.000513992854, 0.0]
_STD_IN   = [0.0968056545,   0.015960684,    1.0]
_MEAN_TGT = [0.154962569,   -0.000517793698, 0.0]
_STD_TGT  = [0.0968104079,   0.0159636438,   1.0]


class FNOConvLSTMHybrid(nn.Module):
    """Frozen FNO base + trainable ConvLSTM residual branch.

    The FNO runs in normalised space; ConvLSTM also receives the normalised
    input sequence and predicts a residual in normalised space. The combined
    output is then denormalised.

    Args:
        fno_ckpt_path: path to the fp16-packed FNO checkpoint
        hidden_dim:    ConvLSTM hidden channels (default 32)
        n_layers:      stacked ConvLSTM layers (default 2)
        T_in / T_out:  temporal context / prediction horizon (default 20)
        C_in / C_out:  number of channels (default 3)
        device:        torch device string
    """

    def __init__(
        self,
        fno_ckpt_path: str,
        hidden_dim: int = 32,
        n_layers: int = 2,
        T_in: int = 20,
        T_out: int = 20,
        C_in: int = 3,
        C_out: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        # ---- load frozen FNO ------------------------------------------------
        _HERE = os.path.dirname(os.path.abspath(__file__))
        _TRACK1 = os.path.abspath(os.path.join(_HERE, "..", ".."))
        if _TRACK1 not in sys.path:
            sys.path.insert(0, _TRACK1)
        _VENDOR = os.path.join(_TRACK1, "_vendor")
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)

        from load_baseline import load_baseline  # noqa: E402
        fno, _meta = load_baseline(fno_ckpt_path, device=device)
        fno.eval()
        for p in fno.parameters():
            p.requires_grad_(False)
        self.fno = fno

        # ---- ConvLSTM branch ------------------------------------------------
        self.encoder = ConvLSTMEncoder(C_in, hidden_dim, n_layers=n_layers)
        self.decoder = ResidualDecoder(hidden_dim, T_out=T_out, C_out=C_out)

        # Initialise branch weights near zero for stable warm-start
        self._init_near_zero()

        # ---- normalisation buffers ------------------------------------------
        self.register_buffer(
            "mean_in",  torch.tensor(_MEAN_IN,  dtype=torch.float32)
        )
        self.register_buffer(
            "std_in",   torch.tensor(_STD_IN,   dtype=torch.float32)
        )
        self.register_buffer(
            "mean_tgt", torch.tensor(_MEAN_TGT, dtype=torch.float32)
        )
        self.register_buffer(
            "std_tgt",  torch.tensor(_STD_TGT,  dtype=torch.float32)
        )

        # Move entire model (including buffers) to target device
        self.to(device)

    # ------------------------------------------------------------------
    def _init_near_zero(self, scale: float = 1e-3):
        """Zero-init only the FINAL decoder layer so the branch's *output*
        starts near zero (training begins from the pure-FNO baseline),
        while the ConvLSTM cells and earlier decoder conv layers keep their
        normal (Kaiming/uniform) PyTorch init.

        Zero-initialising the ConvLSTM's own weights (as done previously)
        makes every sigmoid/tanh gate saturate near a fixed point (~0.5/~0)
        regardless of the input, which starves the hidden state of any
        input-dependent signal and causes vanishing gradients throughout
        the whole branch -- training then barely moves even at high LR.
        Zero-initing only the last layer (a standard residual-branch trick)
        keeps full representational/gradient capacity internally while
        still starting the residual at ~0.
        """
        final_layer = self.decoder.net[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=scale)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    # ------------------------------------------------------------------
    def convlstm_branch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run ConvLSTM encoder + residual decoder on normalised input.

        Args:
            x_norm: (B, T_in, H, W, C) normalised input
        Returns:
            residual: (B, T_out, H, W, C) in normalised space
        """
        B, T, H, W, C = x_norm.shape
        # Rearrange to (B, T, C, H, W) for ConvLSTM
        x_seq = x_norm.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        h = self.encoder(x_seq)                              # (B, hidden_dim, H, W)
        residual = self.decoder(h)                           # (B, T_out, H, W, C)
        return residual

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, H, W, C) raw (unnormalised) input
        Returns:
            (B, T_out, H, W, C) raw (unnormalised) prediction
        """
        # Normalise
        x_norm = (x - self.mean_in) / self.std_in  # (B, T, H, W, C)

        # FNO (frozen, in normalised space)
        with torch.no_grad():
            fno_norm = self.fno(x_norm)             # (B, T_out, H, W, C)

        # ConvLSTM residual (in normalised space)
        residual_norm = self.convlstm_branch(x_norm)  # (B, T_out, H, W, C)

        # Combine in normalised space, then denormalise
        combined_norm = fno_norm + residual_norm
        out = combined_norm * self.std_tgt + self.mean_tgt
        return out
