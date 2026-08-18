#!/usr/bin/env python
# -*- coding: utf-8 -*-

##### Preprocess script.
## This script takes raw parquet data from a source and transforms
## it into pickle files. It makes data right handed/left handed 
## (unless already done) and center it around the wrist,
## thresholds then does train/test/val splits. Finally it saves
## the data as a pickle file.

import os
import sys
import random
import pandas as pd
import numpy as np

from itertools import product
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import (
    DATA_ROOT,
    FILE_PATHS,
    INTERPOLATE_VALS,
    pca,
    get_data_file_name,
    process_metadata,
    process_all_data,
    use_all_data,
    is_loocv,
    is_ntuples,
    is_pt_split,
    get_train_test_val_split,
    read_source_parquet_data,
    save_data,
)
from args import *

##### Check preprocess args
def check_preprocess_args(args):
    if len(args.participant_id) > 1 and args.participant_grp_name is None:
        raise ValueError("Must pass a participant group name when passing multiple participant ids.")
    
    if args.participant_grp_name == "loocv" and len(args.participant_id) != 1:
        raise ValueError("Must pass exactly 1 participant id for loocv")
    
    if args.participant_grp_name == "ntuples":
        if len(args.participant_id) == 0:
            raise ValueError("Error. Use the participant_id arg to pass size of tuples.")
        if not(args.participant_id[0].isdigit()):
            raise ValueError("Error. The first entry of participant id must be an int (size of tuples).")
        if int(args.participant_id[0]) < 2:
            raise ValueError("Error. For ntuples you must have > 2 participants per group.")
    
    if args.participant_grp_name == "pt_split" and len(args.participant_id) > 0:
        raise ValueError("For pt_split, all participants are randomly split. Do not pass pt ids.")
    
def preprocess():
    # if args.use_pca:
    #     # Get original data file name first
    #     file_prefix = get_data_file_name(args, pca_in_name=False)
    #     pipe = pca(file_prefix, ".pq.train", args, pipe=None)
    #     pipe = pca(file_prefix, ".pq.val", args, pipe=pipe)
    #     pipe = pca(file_prefix, ".pq.test", args, pipe=pipe)
    #     sys.exit(0)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    metadata = pd.read_csv(FILE_PATHS[args.dataset]["metadata"])
    metadata.set_index(["sequence_id"], inplace=True)

# old arg
# na_threshold=args.na_threshold,
    all_data = read_source_parquet_data(
        args.dataset,
        dropna=args.dropna,
        interpolate_val=args.interpolate_val,
        use_polar=args.use_polar,
        use_delta=args.use_delta,
        debug=args.debug,
        centering=args.centering,
        finger=args.finger,
        participant_ids=args.participant_id
    )
    all_data = all_data.set_index("sequence_id")
    
    # The process functions below convert and aggregate data.
    # The process_metadata function also considers cross_val,
    # pt grp, and dataset information 
    new_metadata = process_metadata(
        metadata,
        participant_id=args.participant_id,
        participant_grp_name=args.participant_grp_name,
        dataset=args.dataset,
        seed=args.seed,
        use_test=False,
    )
    all_data = process_all_data(all_data)

    full_data = new_metadata.merge(all_data, on="sequence_id", how="inner")
    data_file_name = get_data_file_name(args, pca_in_name=False)
    data_file_path = (DATA_ROOT / data_file_name).with_suffix(".pq")

    if use_all_data(args.train_ratio):
        save_data(data_file_path, all_data=full_data)
    
    elif is_ntuples(args.participant_grp_name):
        groups = set(new_metadata.grp.to_list())

        start = data_file_name.find("ntuples")
        end = data_file_name.find("_", start)

        for group in groups:
            group_metadata = new_metadata[new_metadata.grp == group]
            group_data = group_metadata.merge(all_data, on="sequence_id", how="inner").drop(["grp"], axis=1)
            train, val, test = get_train_test_val_split(group_data, train_ratio=args.train_ratio, split_once=args.split_once, seed=args.seed)

            new_data_file_name = data_file_name[:start] + group + data_file_name[end:]
            new_data_file_path = (DATA_ROOT / new_data_file_name).with_suffix(".pq")
            save_data(new_data_file_path, train=train, val=val, test=test)
    
    ##### loocv is now moving to train time handling (TODO, not yet)
    elif is_loocv(args.participant_grp_name):
        train, val, _ = get_train_test_val_split(full_data, train_ratio=args.train_ratio, split_once=True, seed=args.seed)

        # We call it the test set here, but in the context of loocv it is a val set.
        # Merge on all data here, bc for loocv full data specifically removes the test (cv) participant
        test_metadata = process_metadata(
            metadata,
            participant_id=args.participant_id,
            participant_grp_name=args.participant_grp_name,
            dataset=args.dataset,
            seed=args.seed,
            use_test=True,
        )
        test = test_metadata.merge(all_data, on="sequence_id", how="inner")

        save_data(data_file_path, train=train, val=val, test=test)
    elif is_pt_split(args.participant_grp_name):
        participants = list(set(metadata.participant_id.to_list()))

        pt_train, pt_test = train_test_split(participants, train_size=args.train_ratio, random_state=args.seed)
        pt_val = None

        if not args.split_once:
            pt_test, pt_val = train_test_split(pt_test, train_size=0.5, random_state=args.seed)
        
        train = full_data.loc[full_data.participant_id.isin(pt_train)]
        val = full_data.loc[full_data.participant_id.isin(pt_val)] if not args.split_once else None
        test = full_data.loc[full_data.participant_id.isin(pt_test)] 

        save_data(data_file_path, train=train, val=val, test=test)
    else:
        train, val, test = get_train_test_val_split(full_data, train_ratio=args.train_ratio, split_once=args.split_once, seed=args.seed)
        save_data(data_file_path, train=train, val=val, test=test)

if __name__ == "__main__":
    global args
    args = parse_args()
    print(args)
    check_preprocess_args(args)
    preprocess()

