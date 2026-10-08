#!/usr/bin/env python3
"""
Download baseline checkpoints and example data from Google Drive.

Usage:
    python download_data.py

Requirements:
    - gdown: pip install gdown
    - Internet access to Google Drive
"""

import os
import sys
import subprocess

# Google Drive file IDs (these would need to be obtained from the shared folder)
# For now, we provide instructions on how to download manually

INSTRUCTIONS = """
RealPDE Track 1 - Data Download Instructions
=============================================

The baseline checkpoints and example data must be downloaded from Google Drive.

OPTION 1: Using gdown (Recommended)
------------------------------------
1. Install gdown if not already installed:
   pip install gdown

2. Download the checkpoints folder (id: 1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ):
   gdown --folder "https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ" \\
         --output checkpoints/

3. Download the example data:
   - Find example_data/3750_0.h5 in the shared Google Drive folder
   - Download and place in: example_data/3750_0.h5

OPTION 2: Manual Download
--------------------------
1. Visit the Google Drive folder: https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ
2. Download these files to checkpoints/sim_real_ft/ (for Track 1):
   - sim_real_cno.pth     (32 MB)
   - sim_real_fno.pth     (403 MB, or sim_real_fno_fp16.pth at 201 MB)
   - sim_real_transolver.pth (50 MB)

3. Download example_data/3750_0.h5 to example_data/

FOLDER STRUCTURE:
-----------------
Track1/
  ├── checkpoints/
  │   ├── sim_pretrain/
  │   │   ├── sim_cno.pth
  │   │   ├── sim_fno.pth
  │   │   ├── sim_fno_fp16.pth
  │   │   └── sim_transolver.pth
  │   └── sim_real_ft/         (USE THESE FOR TRACK 1)
  │       ├── sim_real_cno.pth
  │       ├── sim_real_fno.pth
  │       ├── sim_real_fno_fp16.pth
  │       └── sim_real_transolver.pth
  ├── example_data/
  │   └── 3750_0.h5
  └── [rest of kit files]

NEXT STEPS:
-----------
1. Once downloaded, run:
   python smoke_test_kit.py

2. This verifies all imports work and models load correctly.

GOOGLE DRIVE:
-------------
Checkpoints & example data folder: 1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ
See README.md for more details on data specifications.
"""

def main():
    """Print download instructions and attempt gdown if available."""
    print(INSTRUCTIONS)
    
    try:
        import gdown
        print("\n✓ gdown is installed. You can use it to download the files.")
        print("\nExample command:")
        print('  gdown --folder "https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ" --output checkpoints/')
    except ImportError:
        print("\n⚠ gdown not found. Install with: pip install gdown")
        print("  Then use the commands above to download the files.")

if __name__ == "__main__":
    main()
