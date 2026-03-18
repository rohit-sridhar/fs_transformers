import os
import sys
import logging

import pandas as pd

from pathlib import Path

# Adjust path to import utils from parent directory
ANALYSIS_FILE_ROOT = Path(__file__).resolve().parent
sys.path.append(f"{ANALYSIS_FILE_ROOT}/..")

from args import parse_args
from utils import(
    make_loss_plots,
)

global args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    if args.analysis_type == "loss_plots":
        make_loss_plots(args.models_dir)

