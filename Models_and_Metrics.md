# RealPDE — Models & Metrics Reference

Applies to both **Track 1 (Sim2Real)** and **Track 2 (LTTTA)**.  
The 5 subscores are identical across tracks; only the interface differs.

---

## The Problem

Predict 2D airfoil flow fields (NACA4418 airfoil, PIV measurements) given a history window.

- **Input:** 20 consecutive flow frames → **predict** the next 20 frames  
- **Resolution:** 32×64 spatial grid (height × width), evaluated at 2× downsampled real PIV  
- **Channels:** `[u, v, p]` — horizontal velocity, vertical velocity, pressure  
  - `p` is always **zero** for real data (not measurable with PIV) — return zeros, not scored  
  - Only `u` and `v` are scored  
- **Sample shape:** `(N, 20, 32, 64, 3)`  
- The model sees the whole 20-frame block at once and outputs another 20-frame block — **not** a frame-by-frame RNN problem

---

## The 5 Metrics

All scores are 0–100, higher is better. Combined into a single unpublished `final_score`.

---

### 1. `rel_l2_score` — Data Fidelity

```
error = ||pred_uv - target_uv||₂ / ||target_uv||₂
score = 100 / (1 + 0.5 * error)
```

Relative L2 norm over the full `u,v` field across all 20 time steps.  
The most direct accuracy metric. FNO baseline scores **96.86** (~3% relative error).  
Already near-saturated for all baselines — hard to improve much further.

---

### 2. `tke_score` — Turbulent Kinetic Energy

```
TKE = 0.5 * (variance_over_time(u) + variance_over_time(v))
error = ||TKE_pred - TKE_target||₂ / ||TKE_target||₂
score = 100 / (1 + 0.5 * error)
```

Measures whether the model reproduces the **temporal fluctuation energy** at each spatial point.  
TKE is the variance of `u` and `v` over the 20 time steps — it captures how turbulent the flow is locally.  
A model that predicts the time-mean correctly but smooths out fluctuations will score well on Rel-L2 but poorly here.

FNO baseline scores **81.59** — the weakest metric across all baselines and the **biggest opportunity**.  
Better temporal dynamics modeling (temporal attention, ConvLSTM-style recurrence, higher Fourier time modes) directly improves this.

---

### 3. `mvpe_score` — Mean Velocity Profile Error

```
Probe grid: 9 points vertically × 4 positions horizontally downstream of airfoil
Average u,v over time at each probe point
error = ||mean_uv_pred - mean_uv_target||₂ / ||mean_uv_target||₂
score = 100 / (1 + 0.5 * error)
```

Checks whether the **time-averaged wake structure** is correct.  
Physically the most interpretable: does the model capture the correct steady-state flow pattern behind the airfoil?  
FNO baseline scores **97.09** — near-saturated, not a focus area.

---

### 4. `time_score` — Inference Speed

```
r = t_neural / 0.72896        (0.72896s = reference numerical solver time per sample)
score = 100 × 1 / (1 + √r)
```

Rewards models faster than the numerical CFD solver.  
On the eval server (GPU, CUDA 12.1), all models will be much faster than on CPU.  
CPU benchmark results (from local run):

| Model | Time/sample | time_score (CPU) |
|-------|------------|-----------------|
| FNO (fp16) | 0.23s | 63.88 |
| CNO | 0.73s | 50.03 |
| Transolver | 1.59s | 40.38 |

On GPU, all three will likely score 90+. Smaller/simpler models have an edge here.  
**Do not over-engineer for speed** — this is less decisive than TKE/SPS on GPU.

---

### 5. `sps_score` — Safe Prediction Score (Uncertainty Calibration)

```
sigma_global = 0.0563870   (frozen: mean of u,v std on train_real split)

For each scored element:
  pm      = error / (0.5 + error)           (squashed accuracy, in [0,1))
  inside  = lower <= target <= upper
  reward  = (1 - pm) × exp(−(upper−lower) / sigma_global)   if inside, else 0

branch  = mean reward over scored elements (targets inside PIV FoV and outside airfoil body)
weighted = 0.5 × branch_l2 + 0.3 × branch_tke + 0.2 × branch_mvpe
sps_score = 100 × weighted
```

Rewards predictions that are **accurate AND paired with tight, well-calibrated uncertainty intervals**.  
- Wider interval = lower reward (exp decay penalizes width)  
- Target outside interval = zero reward for that element  
- Narrow interval around wrong prediction = zero (no coverage)

If you don't return `lower`/`upper`, the scorer defaults to `±5% × |pred|` — this is what all 3 baselines use.  
FNO baseline scores only **27.04** with the default band. **This is the biggest untapped opportunity.**

Practical strategies to improve SPS:
- **Ensemble spread:** run FNO + CNO + Transolver, use their disagreement as uncertainty
- **MC Dropout:** enable dropout at test time, sample multiple forward passes
- **Direct interval prediction:** train a model to predict `lower` and `upper` alongside the mean
- **Tighter default:** even `±2% × |pred|` may score higher if your predictions are accurate enough

---

## Score Mapping Summary

| Metric | What it captures | Baseline best | Improvement potential |
|--------|-----------------|---------------|----------------------|
| rel_l2_score | Raw prediction accuracy | 96.86 (FNO) | Low — near-saturated |
| tke_score | Temporal fluctuation energy | 81.59 (FNO) | **High** — all models miss turbulence |
| mvpe_score | Time-mean wake structure | 97.09 (FNO) | Low — near-saturated |
| time_score | Inference speed vs CFD solver | 63.88 (FNO, CPU) | Medium — GPU will be ~90+ |
| sps_score | Uncertainty calibration | 27.04 (FNO) | **Very High** — all baselines use dumb default |

---

## The 3 Baseline Models

All trained with the RealPDEBench codebase. Checkpoints in `Track1/checkpoints/sim_real_ft/`.

---

### FNO — Fourier Neural Operator ⭐ Best Baseline

**Paper:** [Fourier Neural Operator for PDEs (Li et al., 2021)](https://arxiv.org/abs/2010.08895)

**Core idea:** Learn a filter in *frequency space* via 3D FFT, rather than in pixel space. Captures global spatial patterns from the first layer.

**Architecture:**
1. Append spatial grid coordinates `(x, y, t)` to input → lift to width=64 via linear layer
2. **4 layers:** `SpectralConv3d` (FFT → multiply learned modes → IFFT) + `Conv3d(1×1×1)` residual + `BatchNorm3d` + `GELU`
3. Project to output via two linear layers + reshape

**Key hyperparams:** `modes=(4,12,16)`, `width=64`, `n_layers=4`, padded with 6 zeros on each side before spectral ops

**Spectral modes explained:** Only the lowest 4×12×16 frequency components of the (T=20, H=32, W=64) volume are learned. This is a very smooth, global filter — great for capturing large-scale flow patterns, less so for fine turbulent details.

**Strengths:** Global receptive field from layer 1, captures space+time jointly, fast inference  
**Weaknesses:** Spectral bias toward smooth solutions (hurts TKE), fixed modes = fixed resolution assumption  
**Size:** 403MB fp32 / **201MB fp16** (needs packing to fit 256MB cap)  
**Local speed:** 0.23s/sample (CPU)

---

### CNO — Convolutional Neural Operator

**Core idea:** U-Net style encoder-decoder with anti-aliasing filters (from StyleGAN3) to prevent artifacts at resolution changes. Designed to be resolution-equivariant.

**Architecture:**
- **3 encoder stages:** `Conv3d` → anti-aliased downsample (LeakyReLU, BN)
- **Bottleneck**
- **3 decoder stages:** upsample → `Conv3d` + skip connections
- Each `CNOBlock3d`: Conv → BatchNorm → LeakyReLU(0.2) — the released checkpoints use plain LeakyReLU, NOT the filtered variant (which requires `scipy`)

**Key hyperparams:** `N_layers=3`, `channel_multiplier=32`, `C_in=C_out=3`

**Strengths:** Smallest model (32MB), local spatial structure, fast, resolution-flexible  
**Weaknesses:** Limited receptive field per layer (local convolutions), misses long-range correlations  
**Size:** **32MB** (fits in 256MB easily)  
**Local speed:** 0.73s/sample (CPU)  
**Best val loss:** 0.00861 @ iter 5000

---

### Transolver — Transformer for Physics Simulation

**Paper:** [Transolver (Wu et al., 2024)](https://arxiv.org/abs/2402.02366)

**Core idea:** Apply attention over space+time, but instead of attending over all 20×32×64=40,960 tokens (too expensive), learn to group spatial points into 16 "physics slices" via soft clustering, then apply attention within and across slices.

**Architecture:**
1. Flatten `(B, T×H×W, 3)` → embed with MLP using spatial coords as features
2. **3 blocks** of `Physics_Attention_Structured_Mesh_3D`:
   - Learn `slice_num=16` soft clusters (physics-based groupings of space-time points)
   - Multi-head attention (8 heads) within each cluster
   - Mix information across clusters
3. Final MLP to output

**Key hyperparams:** `n_layers=3`, `n_hidden=256`, `n_head=8`, `slice_num=16`, `H=64, W=32, D=20` (note: W=H_s, H=W_s, D=T in the code)

**Strengths:** Global receptive field via attention, can capture arbitrary long-range physics correlations, high capacity  
**Weaknesses:** Slowest inference, likely undertrained (5000 iters), attention is O(n²) in slice count  
**Size:** **50MB**  
**Local speed:** 1.59s/sample (CPU)  
**Best val loss:** 0.01224 @ iter 4400 (worse than CNO's 0.00861 — needs more training)

---

## Comparison Table

| | FNO | CNO | Transolver |
|--|-----|-----|-----------|
| **Inductive bias** | Global frequency filter | Local conv + anti-aliasing | Attention over physics clusters |
| **Receptive field** | Global from layer 1 | Grows with depth | Global via attention |
| **Best at** | Smooth global patterns, speed | Local spatial features, tiny size | Long-range correlations |
| **Weakest at** | Fine turbulent fluctuations | Global patterns | Speed, needs more training |
| **Size** | 403MB (201MB fp16) | **32MB** | 50MB |
| **Speed (CPU)** | **0.23s** | 0.73s | 1.59s |
| **rel_l2_score** | **96.86** | 95.85 | 93.49 |
| **tke_score** | **81.59** | 77.58 | 69.50 |
| **mvpe_score** | **97.09** | 96.10 | 93.65 |
| **time_score** | **63.88** | 50.03 | 40.38 |
| **sps_score** | **27.04** | 21.97 | 5.91 |

---

## Directions for Improvement

### High Impact
1. **SPS via ensemble uncertainty** — run all 3 baselines, use spread as `lower/upper` bounds. Zero extra training, potentially doubles SPS score.
2. **Better TKE** — the spectral bias of FNO smooths out fluctuations. Approaches:
   - More Fourier modes in the time dimension (currently only 4 out of 20)
   - ConvLSTM or temporal attention to model dynamics more explicitly
   - Training with a TKE-aware loss term

### Medium Impact
3. **Fine-tune FNO/CNO on `train_real`** — the baselines are already fine-tuned, but more epochs / better LR schedule may help
4. **Ensemble prediction** (FNO + CNO weighted average) — typically gains 1-3 points on Rel-L2

### Lower Impact (already near-saturated)
5. Rel-L2 and MVPE improvements — already at 96-97/100, diminishing returns

---

*Source: RealPDE NeurIPS 2026 Competition, Codabench page 17363/17385, starting kit v9, scoring.py, rpde_baselines source code.*
