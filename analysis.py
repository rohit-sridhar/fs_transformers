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
    ScorerStepByStep,
    get_split_file_name,
    misc_data_analysis,
    make_timeline_charts,
    make_model_cmp_charts,
    make_gradient_plots,
    make_loss_plots,
    preprocess,
    run_pca,
    split_dataset,
    make_small,
    read_parquet,
    X_COLS,
    BID_PRICE_COLS,
    BID_VOL_COLS,
    ASK_PRICE_COLS,
    ASK_VOL_COLS,
    TRADE_PRICE_COLS,
    TRADE_VOL_COLS,
)

DATA_COLS = [
    X_COLS,
    BID_PRICE_COLS,
    BID_VOL_COLS,
    ASK_PRICE_COLS,
    ASK_VOL_COLS,
    TRADE_PRICE_COLS,
    TRADE_VOL_COLS,
]

global args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    if args.analysis_type == "split_val":
        if args.data_file is None:
            raise ValueError("must pass data_file when using split_val")

        split_dataset(args.data_file, args.split_ratio, args.seed)
    elif args.analysis_type == "cmp_model_out":
        df = read_parquet(args.data_file)
        make_model_cmp_charts(df, args.chkpt_path)
    elif args.analysis_type == "get_basic_stats":
        df = read_parquet(args.data_file)
        
        if args.preprocess:
            df, _ = preprocess(df, pca=args.use_pca)
        
        for interval in args.intervals:
            make_timeline_charts(
                df,
                preprocess=args.preprocess,
                interval=interval,
                pca=args.use_pca
            )
    elif args.analysis_type == "pca":
        read_parquet(args.data_file)
        
        if args.preprocess:
            # don't use pca even if use_pca is passed here since
            # it is used later
            df, _ = preprocess(df)
        
        for col_list in DATA_COLS:
            run_pca(df, seed=args.seed, cols=col_list)

    elif args.analysis_type == "loss_plots":
        make_loss_plots(args.models_dir)
    elif args.analysis_type == "gradient_plots":
        make_gradient_plots(args.chkpt_path)
    elif args.analysis_type == "make_small":
        make_small(args.data_file, args.seed)

