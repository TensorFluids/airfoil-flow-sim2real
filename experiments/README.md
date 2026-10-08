# Track 1 Experiments

This is where all our custom work lives. The starting kit files one level up are
read-only references — never modify them.

## Structure

```
experiments/
├── models/         custom model architectures (CNN, U-Net, ConvLSTM, hybrids, etc.)
├── ensemble/       ensemble strategies combining FNO + CNO + Transolver + custom models
├── finetune/       fine-tuning scripts for the provided baselines on train_real
├── results/        saved prediction .npz files and local scores.json outputs
└── submissions/    ready-to-zip submission folders (each is a self-contained archive root)
```

## How to run an experiment

1. **Define your model** in `models/` or `ensemble/`
2. **Train / fine-tune** using data from `../../data/train_real/` and `../../data/train_sim/`
3. **Save predictions** to `results/<experiment_name>/predictions.npz`
4. **Score locally** using the kit's scoring.py:
   ```
   python ../scoring.py results/<experiment_name>/ results/<experiment_name>/
   ```
5. **Package for submission** into `submissions/<experiment_name>/`:
   - `submission.py` at root
   - checkpoint `.pth` at root
   - `load_baseline.py`, `rpde_baselines/`, `_vendor/` copied from the kit
   - Zip it: total extracted size must be < 256 MB

## Key paths

| Resource | Path |
|----------|------|
| Real training data | `../../data/train_real/` (exclude `7575_0.h5`!) |
| Sim training data | `../../data/train_sim/` |
| Baseline checkpoints | `../checkpoints/sim_real_ft/` |
| Example data | `../example_data/3750_0.h5` |
| Kit scoring script | `../scoring.py` |
| Kit load_baseline | `../load_baseline.py` |

## Rules reminder

- Training data: only `train_real/` and `train_sim/` from this repo
- No pretrained weights from outside the competition release
- Exclude `train_real/7575_0.h5` (known bad file)
- Submission size < 256 MB extracted
- No `scipy`, `h5py`, `matplotlib`, `pandas` in `submission.py`
- `DataLoader` must use `num_workers=0`
- See `../../RULES.md` for the full ruleset
