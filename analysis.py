#!/home/rsridhar37/miniconda3/envs/fs_transformers/bin/python

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
    make_small,
    sample_classification,
)

global args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    if args.analysis_type == "loss_plots":
        make_loss_plots(args.models_dir)
    elif args.analysis_type == "make_small":
        make_small(args.data_file, args.seed)
    elif args.analysis_type == "sample_classification":
        sample_classification(
            args.dataset,
            args.data_file,
            args.sample_strategy,
            args.n,
            args.seed,
        )
    elif args.analysis_type == "count_label_chars":
        count_row_characters(args.dataset)

