#!/usr/bin/env python3
"""
Test baseline model loading and inference.
"""

import sys
import os
import numpy as np
import torch
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_vendor'))
sys.path.insert(0, os.path.dirname(__file__))

from load_baseline import load_baseline, make_example_input

def test_model(model_name, ckpt_path):
    """Load a model and run inference on example data."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Load model
        print(f"Loading checkpoint: {ckpt_path}")
        model, meta = load_baseline(ckpt_path)
        print(f"[OK] Model loaded successfully")
        
        # Show metadata
        if meta:
            print(f"  Metadata keys: {list(meta.keys())}")
            for key, val in meta.items():
                print(f"    - {key}: {val}")
        
        # Create example input
        print(f"\nLoading example data...")
        x = make_example_input("example_data/3750_0.h5", sub_s=2)
        print(f"[OK] Example input loaded")
        print(f"  Input shape: {x.shape}  (N, T_in, H, W, C)")
        print(f"  Input dtype: {x.dtype}")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
        
        # Forward pass
        print(f"\nRunning forward pass...")
        with torch.no_grad():
            y = model(x)
        print(f"[OK] Forward pass successful")
        print(f"  Output shape: {y.shape}  (N, T_out, H, W, C)")
        print(f"  Output dtype: {y.dtype}")
        print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Compute statistics
        print(f"\nOutput statistics by channel:")
        y_np = y.cpu().numpy()
        channels = ['u', 'v', 'p']
        for c in range(3):
            ch_data = y_np[..., c]
            print(f"  {channels[c]:1s}: mean={ch_data.mean():.6f}, std={ch_data.std():.6f}, "
                  f"min={ch_data.min():.6f}, max={ch_data.max():.6f}")
        
        print(f"\n[PASS] {model_name} inference complete\n")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    os.chdir(os.path.dirname(__file__) or '.')
    
    print("\n" + "="*70)
    print("RealPDE Track 1 - Baseline Model Inference Test")
    print("="*70)
    
    # Test each model from sim_real_ft (the fine-tuned Track 1 models)
    models_to_test = [
        ("FNO (fp16-packed)", "checkpoints/sim_real_ft/sim_real_fno_fp16.pth"),
        ("CNO", "checkpoints/sim_real_ft/sim_real_cno.pth"),
        ("Transolver", "checkpoints/sim_real_ft/sim_real_transolver.pth"),
    ]
    
    results = {}
    for model_name, ckpt_path in models_to_test:
        if os.path.exists(ckpt_path):
            results[model_name] = test_model(model_name, ckpt_path)
        else:
            print(f"\n⚠ Checkpoint not found: {ckpt_path}")
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    for model_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10s} {model_name}")
    
    if all(results.values()):
        print("\n[SUCCESS] All baseline models loaded and ran inference successfully!")
    else:
        print("\n[WARNING] Some models failed; see errors above")


if __name__ == "__main__":
    main()
