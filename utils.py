#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import torch
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa

from collections import Counter
from pyarrow.parquet import ParquetFile
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.kernel_approximation import Nystroem

##### CONSTANTS ######

##### Directory information
ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"
ANALYSIS_ROOT = ROOT / "analysis"
DATA_ROOT = ROOT / "data"
GEN_DATA_ROOT = Path("/data/deep_learning/ISLR-ML/mputils/out")

# Regex for main dataset classifications (TODO move up later)
NAME_RE_PATTERNS=[r"([a-zA-Z]+\s+)+[a-zA-Z]+"]
ADDRESS_RE_PATTERNS=[r"[0-9]+\s+([a-zA-Z0-9\.]+\s+)*[a-zA-Z0-9\.]+"]
PHONE_RE_PATTERNS=[r"\+?([0-9]+-)+[0-9]+"]
URL_RE_PATTERNS=[r"(.+[/\.])+.+/?"]

##### File paths for base datasets
FILE_PATHS = {
    "supplemental_gen": {
        "metadata": GEN_DATA_ROOT / "metadata" / "supplemental_gen.csv",
        "landmarks": GEN_DATA_ROOT / "landmarks" / "supplemental_gen.parquet",
        "label_map": ROOT / "supplemental_character_to_prediction_index.json",
        "num_labels": 30,
    },
    "main_train": {
        "metadata": GEN_DATA_ROOT / "metadata" / "main_train.csv",
        "landmarks": GEN_DATA_ROOT / "landmarks" / "main_train.parquet",
        "label_map": ROOT / "character_to_prediction_index.json",
        "num_labels": 30,
    },
    "main_val": {
        "metadata": GEN_DATA_ROOT / "metadata" / "main_val.csv",
        "landmarks": GEN_DATA_ROOT / "landmarks" / "main_val.parquet",
        "label_map": ROOT / "character_to_prediction_index.json",
        "num_labels": 30,
    },
}
NEW_DATASETS = list(FILE_PATHS.keys())

COLS = [0,1,4,5,8,9,12,13,16,17,20]

X_SH_COLS = [f"x_hand_{col}" for col in COLS]
Y_SH_COLS = [f"y_hand_{col}" for col in COLS]

START_CHAR = "<"
END_CHAR = ">"

##### Model Constants
GB_BYTES = 1024**3
BYT5_NUM_SPECIAL_TOKENS = 3

HYPERPARAMS = [
    "Model:",
    "model_type",
    "optimizer",
    "loss_fn",
    "batch_size",
    "learning_rate",
    "hidden_size",
    "num_layers",
    "dropout",
    "preprocess",
    "use_pca",
    "data_file",
]

##### miscellaneous functions
def read_parquet(data_file):
    df = pd.read_parquet(data_file)
    return df

def rm_suffixes(p):
    while p.suffix:
        p = p.with_suffix("")
    return p

def get_new_file_with_ext(old_file, add_ext):
    old_file_name = rm_suffixes(old_file)
    new_path = (
        old_file.parent / f"{old_file_name.stem}_{add_ext}"
    ).with_suffix("".join(old_file.suffixes))
    return new_path

##### ANALYSIS FUNCTIONS ######

# gets the parameter to datapoint ratio for
# model trained
def get_param_and_var_count(lines):
    params = 0
    n = -1

    for ln in lines:
        pattern = r"[0-9]+$"

        if "trainable parameters:" in ln:
            match = re.search(pattern, ln)
            if match:
                params = int(match.group())
                if n != -1: # we're done in this case.
                    break
        
        elif "learnable data points:" in ln:
            match = re.search(pattern, ln)
            if match:
                n = int(match.group())
                if params != 0: # again, done in this case.
                    break
        
    return params, n

# Check log file for args and regex extract the args
def get_args_from_output(lines):
    for ln in lines:
        if "Namespace" in ln:
            match = re.search(r"Namespace.+$", ln)
            if match:
                all_hyperparams = match.group()
                all_hyperparams = all_hyperparams.replace(str(DATA_ROOT) + "/", "")
                hyperparams = ""
                n_str = "n0"

                for hyperparam in HYPERPARAMS[1:]:
                    match = re.search(f"{hyperparam}=[^,]+(,|$)", all_hyperparams)
                    if match:
                        if hyperparam == "data_file":
                            n_str_match = re.search("(ts|n)[0-9]+", match.group())
                            if n_str_match is not None:
                                n_str = n_str_match.group()
                        hyperparams += match.group() + " "
                hyperparams = HYPERPARAMS[0] + " " + hyperparams[:-2]

                return n_str, hyperparams

    return "n0", ""

##### count label chars
# pd apply function to count chars in row label
def count_row_characters(row, char_counts={}):
    for c in row.phrase:
        char_counts[c] += 1

# counts characters in labels across all metadata rows and prints counts
def count_label_characters(dataset):
    metadata_file = FILE_PATHS[dataset]["metadata"]
    metadata = pd.read_csv(metadata_file)

    char_counts = Counter()
    metadata.apply(
        count_row_characters,
        char_counts=char_counts,
        axis=1
    )
    for key in char_counts:
        print(f"{key} count: {char_counts[key]}")

##### classify data
# pd apply function to classify rows
def classify_row(row):
    classification = "NA"
    if re.fullmatch(NAME_RE_PATTERNS[0], row.phrase):
        classification = "name"
    elif re.fullmatch(ADDRESS_RE_PATTERNS[0], row.phrase):
        classification = "address"
    elif re.fullmatch(PHONE_RE_PATTERNS[0], row.phrase):
        classification = "phone"
    elif re.fullmatch(URL_RE_PATTERNS[0], row.phrase):
        classification = "url"
    
    return classification

# do (stratified or categorical) sampling and save data
def sample_and_save_data(metadata, df, data_file, n, seed, ext):
    train_seqs, _ = train_test_split(
        metadata.index.to_list(),
        train_size=n,
        stratify=metadata["classification"],
        random_state=seed,
    )

    new_file = get_new_file_with_ext(data_file, f"{ext}{n}")

    df = df.loc[train_seqs]
    df.to_parquet(new_file)

    print(metadata.loc[train_seqs])
    print(f"Created {new_file} with only sequences above")

# drop any indices in metadata not in df to prevent sampling discrepancies
def drop_metadata_seqs(metadata, df):
    md_ids = set(metadata.index.to_list())
    df_ids = set(df.index.to_list())

    metadata = metadata.drop(index=(md_ids - df_ids))
    return metadata

# classifies the data and provides summary
def sample_classification(dataset, data_file, sample_strategy, n, seed):
    metadata_file = FILE_PATHS[dataset]["metadata"]

    metadata = pd.read_csv(metadata_file)
    df = read_parquet(data_file)
    metadata = metadata.set_index("sequence_id")
    
    metadata = drop_metadata_seqs(metadata, df)
    metadata["classification"] = metadata.apply(classify_row, axis=1)
    print(metadata.groupby("classification").size())
    
    if sample_strategy == "stratified":
        sample_and_save_data(
            metadata,
            df, data_file,
            n, seed, "cls"
        )
    elif sample_strategy == "categorical":
        classes = set(metadata["classification"].to_list())
        for cls in classes:
            sample_and_save_data(
                metadata[metadata["classification"] == cls],
                df, data_file,
                n, seed, f"cls-{cls}"
            )

    # print(metadata[metadata["classification"] == "NA"])
    
##### make small
# make dataset small
def make_small(data_file, seed):
    random.seed(seed)
    df = read_parquet(data_file)

    seq_ids = df.index.to_list()
    random.shuffle(seq_ids)

    df_small = df.loc[seq_ids[:3]]
    new_file = get_new_file_with_ext(data_file, "smp3")
    
    df_small.to_parquet(new_file)

##### make loss plots
# gets the epoch and loss variables to make
# a simple matplotlib chart.
def get_loss_and_score_by_epoch(lines):
    epochs = []
    train_losses, val_losses = [], []
    train_scores, val_scores = [], []

    for ln in lines:
        if "Loss of epoch" in ln:
            epoch_match = re.search("epoch [0-9]+ -", ln)
            if epoch_match:
                epoch = re.search("[0-9]+", epoch_match.group())
                epochs.append(int(epoch.group()))
            
            loss_match = re.findall("[0-9]+\\.[0-9]+", ln)
            if len(loss_match) == 2:
                train_losses.append(float(loss_match[0]))
                val_losses.append(float(loss_match[1]))
        elif "Loss and score of epoch" in ln:
            epoch_match = re.search("epoch [0-9]+ -", ln)
            if epoch_match:
                epoch = re.search("[0-9]+", epoch_match.group())
                epochs.append(int(epoch.group()))
            
            loss_match = re.findall(r"-?[0-9]+\\.[0-9]+(?:e[+|-][0-9]+)?", ln)
            if len(loss_match) == 4:
                train_losses.append(float(loss_match[0]))
                val_losses.append(float(loss_match[1]))
                
                train_scores.append(float(loss_match[2]))
                val_scores.append(float(loss_match[3]))

    return epochs, train_losses, val_losses, train_scores, val_scores

# makes the loss plots by epoch
def make_loss_plots(model_dir):
    for log_file in model_dir.glob("**/log.txt"):
        with open(log_file, "r") as f:
            output = f.readlines()
        
        (epochs,
         train_losses,
         val_losses,
         train_scores,
         val_scores) = get_loss_and_score_by_epoch(output)
        train_scores_arr = np.array(train_scores)
        val_scores_arr = np.array(val_scores)
        
        params, num_vars = get_param_and_var_count(output)
        param_var_ratio = round(params / num_vars, 3)

        n_str, hyperparams = get_args_from_output(output)
        print(hyperparams)
        
        i = log_file.parts.index("models") + 1
        plt_filename = log_file.parts[-2]
        new_parts = ["loss_plots", *(log_file.parts[i:-2]), n_str, plt_filename]

        plt_file = ANALYSIS_ROOT.joinpath(*new_parts).with_suffix(".png")
        plt_file.parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 12.0))
        ax1.plot(epochs, train_losses, label="train")
        ax1.plot(epochs, val_losses, label="val")
        ax1.set_ylabel("losses")

        ax2.plot(epochs, train_scores, label="train")
        ax2.plot(epochs, val_scores, label="val")
        ax2.set_ylabel("scores")
        
        plt.legend()
        plt.xlabel(f"{hyperparams}", wrap=True)
        
        fig.suptitle(f"Params: {params:,} | Vars: {num_vars:,} | {param_var_ratio}")
        plt.savefig(plt_file, bbox_inches="tight")
        
        plt.clf()

###### PREPROCESSING FUNCTIONS ######
##### Boolean helpers
# checks if doing loocv
def is_pt_split(participant_grp_name):
    # Could be None if that is passed as an arg.
    return participant_grp_name == "pt-split"

# checks if doing loocv (during training)
def is_loocv(participant_grp_name):
    # If None, not doing loocv. else it should be the pt id.
    # return loocv_pt is not None
    return participant_grp_name is not None and "loocv" in participant_grp_name

# checks if doing loocv
def is_ntuples(participant_grp_name):
    # Could be None if that is passed as an arg.
    return participant_grp_name == "ntuples"

# Check is preprocess is pulling all data or splitting (train/test/val)
def use_all_data(train_ratio):
    return train_ratio == 0.0 or train_ratio == 1.0

# Return filename suffix for data file preprocessed by preprocess.py
###!! For now, get_data_file_name does not rely on use_pca for since PCA
###!! is done after preprocessing is complete (even though it is techni-
###!! cally a preprocessing step).
def get_data_file_name(args, pca_in_name=False):
# def get_data_file_name(args):
    # for now, return data file without .train/.test/.val ext if passed
    # if args.data_file is not None:
    #     data_file = os.path.basename(args.data_file)
    #     return data_file.replace(".pkl.train", "")

    name_params = ["data", f"{args.dataset}"]
    if args.na_threshold < 1.0:
        name_params.append(f"na-thr{args.na_threshold}")
    if args.dropna:
        name_params.append("drop-na")
    if args.interpolate_val is not None:
        name_params.append(f"lininterp{args.interpolate_val}")
    if not(use_all_data(args.train_ratio)):
         name_params.append(f"sd{args.seed}")
    
    # Multiple possibilities
    #  If 1 pt id use the pt id for the name if a group isn't given
    #  Use the grp name and pt id otherwise.
    #  If more than 1 pt id, then use the group only.
    #  If no pt ids, then no additional name
    if len(args.participant_id) == 1:
        pt_id = args.participant_id[0]
        if args.participant_grp_name is None:
            name_params.append(f"pt-{pt_id}")
        else:
            name_params.append(f"{args.participant_grp_name}-{pt_id}")
    elif len(args.participant_id) > 0 or is_pt_split(args.participant_grp_name):
        name_params.append(f"{args.participant_grp_name}")

    if args.use_polar:
        name_params.append("polar")

    if args.use_delta:
        name_params.append("delta")
    
    # if args.use_pca and pca_in_name:
    if pca_in_name:
        name_params.append(f"pca{args.n_components}")

    if args.make_left_handed:
        name_params.append("lh")
    else:
        name_params.append("rh")
    
    if args.data_file_name_ext is not None:
        name_params.append(args.data_file_name_ext)

    data_file_name = "_".join(name_params)
    return data_file_name

##### Batching/Splitting Dataset helpers
# Split into train/val sets by train_ratio. Then split against into traun/val/test sets
# using train_ratio again.
def get_train_test_val_split(data, train_ratio, split_once=False, seed=7248):
    if split_once:
        train, val = train_test_split(data, train_size=train_ratio, stratify=data["participant_id"], random_state=seed)
        return train, val, None
    else:
        train, test = train_test_split(data, train_size=train_ratio, stratify=data["participant_id"], random_state=seed)
        train, val = train_test_split(train, train_size=train_ratio, stratify=train["participant_id"], random_state=seed)
        return train, val, test

##### Label Manipulation helpers
# Loads label map (in reverse, if needed). The label map
# maps characters to indices and vice verse for label encoding
def load_label_map(dataset, reverse=False):
    with open(FILE_PATHS[dataset]["label_map"], "r") as f:
        label_map = json.load(f)
    
    if reverse:
        keys = list(label_map.keys())
        label_map = {label_map[key]:key for key in keys}

    return label_map

# Given a list of indices and a label map,
# this function returns the translated label
def get_text_from_idx(idx_label, label_map):
    label = ""
    for char_idx in idx_label:
        label += label_map[char_idx]
    return label

##### dataset conversion/reshape functions
# Convert raw data to numpy (or in the future, torch) arrays
# cross_val is only used when processing loocv data
def process_metadata(
    metadata,
    participant_id=[],
    participant_grp_name=None,
    dataset="supplemental_gen",
    seed=1248,
    cross_val=False,
):
    def add_groups(row):
        if row.participant_id in groups:
            row['grp'] = groups[row.participant_id]

        return row
    
    # groups variable is only used for n_tuples grouping
    groups = {}
    new_metadata = metadata.copy()
    if is_loocv(participant_grp_name):
        pt = participant_id[0]
        if cross_val:
            # make test set using this option
            new_metadata = metadata.loc[metadata["participant_id"].isin([pt])]
        else:
            # this is for main loocv dataset
            new_metadata = metadata.loc[~metadata["participant_id"].isin([pt])]

    elif is_ntuples(participant_grp_name):
        n = int(participant_id[0])
        participants = list(set(metadata.participant_id.to_list()))
        random.shuffle(participants)
        
        groups_save = {}
        for i in range(0, len(participants), n):
            ntup = tuple(participants[i:i+n])
            grp_name = f"grp.rnd{str(int(i / n))}.{n}"
            for pt in ntup:
                groups[pt] = grp_name
            groups_save['|'.join(ntup)] = grp_name
        
        with open(f"output/grp.rnd.{n}_{dataset}_sd{seed}.json", "w") as f:
            json.dump(groups_save, f, indent=4)

        new_metadata = new_metadata.apply(add_groups, axis=1)
        
    elif len(participant_id) > 0:
        new_metadata = metadata.loc[metadata["participant_id"].isin(participant_id)]
    
    return new_metadata[["phrase", "participant_id"]]

##### data processing functions
def df_to_bytes(df, eos_token_id=1):
    def phrase_to_bytes(row):
        row.phrase = list(row.phrase.encode("utf-8"))
        row.phrase = [enc + BYT5_NUM_SPECIAL_TOKENS for enc in row.phrase] + [eos_token_id]

        return row
    
    df = df.apply(phrase_to_bytes, axis=1)

    return df

# Convert all data so that frames are joined into a single list
def process_all_data(data):
    seq_ids = data.index.values.tolist()
    data["all_landmarks"] = data.values.tolist()
    
    new_data = data.copy()[["all_landmarks"]]
    new_data = new_data.groupby(["sequence_id"]).agg(lambda x: list(x))
    
    return new_data

##### data alteration functions
# Take the deltas.
def take_deltas(data):
    def take_delta(row):
        new_row = row.iloc[1:].reset_index() - row.iloc[:-1].reset_index()
        return new_row
        
    data = data.groupby("sequence_id").apply(take_delta)
    data = data.drop(["index", "sequence_id"], axis=1)
    
    # reset index twice (first for group by level then for numbering)
    data = data.reset_index(level="sequence_id").reset_index(drop=True)
    return data

# Center hands data around the 0 (wrist) data point.
def center_hands(data):
    data[X_SH_COLS] = data[X_SH_COLS].values - data["x_hand_0"].values[:,None]
    data[Y_SH_COLS] = data[Y_SH_COLS].values - data["y_hand_0"].values[:,None]

    data = data.drop(labels=["x_hand_0", "y_hand_0"], axis=1)
    return data

# Make sure that only any sequence with greater than threshold < len(landmarks) / len(sequence) is allowed
def threshold_data(data, threshold=1):
    def filter_rows(row):
        return len(row.all_landmarks) / len(row.phrase) >= threshold

    data = data[data.apply(filter_rows, axis=1)]
    return data

##### read and save data functions
# get nan percentage
def get_na_pct(seq, na_threshold=1.0):
    total_count = seq.x_right_0.shape[0]
    na_count = seq.x_right_0.isna().sum()
    na_pct = na_count / total_count
    return True if na_pct >= na_threshold else False

# threshold na sequences
def threshold_na_sequences(data, na_threshold=1.0):
    data_counts = data.groupby(level="sequence_id").apply(
        get_na_pct,
        na_threshold=na_threshold,
    )

    data_counts = pd.DataFrame(data_counts, columns=["remove_seq"])
    data = data.merge(data_counts, how="inner", on="sequence_id")
    
    data = data[~data.remove_seq].drop("remove_seq", axis=1)
    return data

# do interpolation on rows
def interpolate(row, interpolate_val=0): 
    X = row.x_right_0 == 0.0
    if X.all():
        return row

    X = X.to_list()
    start = X.index(False)
    end = len(X) - (1 + X[::-1].index(False))
    if start == end:
        return row
    
    row = row.reset_index(drop=True)
    row_nan = row.iloc[start:end+1].replace(0.0, np.nan)
    row_nan = row_nan.interpolate()

    if interpolate_val == 0:
        return pd.concat([row.iloc[:start], row_nan, row.iloc[end+1:]])

    fwd_row = row_nan[1:].reset_index(drop=True)
    bck_row = row_nan[:-1].reset_index(drop=True)
    row_sum = fwd_row + bck_row

    interp_row = (row_sum / 2)
    subset_row = row_nan.reset_index(drop=True)
    
    interp_row["order"] = list(range(1, interp_row.shape[0] * 2, 2))
    subset_row["order"] = list(range(0, subset_row.shape[0] * 2, 2))
    interp_row = interp_row.set_index("order")
    subset_row = subset_row.set_index("order")
    
    new_row = pd.concat([interp_row, subset_row])
    new_row = new_row.sort_index().reset_index().drop("order", axis=1)
    new_row = pd.concat([row.iloc[:start], new_row, row.iloc[end+1:]])

    return new_row

# read parquet file
###!! this function assumes the data is already right handed
###!! it also centers hands around the wrist point.
###!! For now, flip the y coordinates (1 - y) since they appear
###!! to be flipped.
def read_source_parquet_data(
    dataset,
    na_threshold=1.0,
    dropna=True,
    interpolate_val=None,
    use_polar=False,
    use_delta=False,
    debug=False,
):
    if debug:
        ## Use code below when developing or debugging
        pf = ParquetFile(FILE_PATHS[dataset]["landmarks"])
        first_ten_rows = next(pf.iter_batches(batch_size = 100000))
        all_data = pa.Table.from_batches([first_ten_rows]).to_pandas()
    else:
        ## Use below line for main code.
        all_data = pd.read_parquet(FILE_PATHS[dataset]["landmarks"])
    
    all_data = threshold_na_sequences(all_data, na_threshold=na_threshold)
    if dropna and interpolate is None:
        all_data = all_data.drop(columns=["frame"]).dropna(axis=0)
    else:
        ## Code for interpolation
        all_data = all_data.drop(columns=["frame"]).fillna(0.0)
        if interpolate is not None:
            all_data = all_data.groupby("sequence_id").apply(
                interpolate,
                interpolate_val=interpolate_val
            )
            all_data = all_data.reset_index().drop("level_1", axis=1).set_index("sequence_id")

        if dropna:
            all_data = all_data[~(all_data == 0).all(axis=1)]
 
    colmap = {col:col.replace("right","hand") for col in all_data.columns}
    all_data = all_data.rename(columns=colmap).reset_index()
    
    #!!! This line needs to be treated with caution. may not be needed if upstream
    #!!! mediapipe landmark generation is fixed. TODO
    all_data[Y_SH_COLS] = np.where(all_data[Y_SH_COLS] == 0, 0, 1 - all_data[Y_SH_COLS])

    all_data = center_hands(all_data).loc[:,["sequence_id"] + X_SH_COLS[1:] + Y_SH_COLS[1:]]
    all_data.sequence_id = all_data.sequence_id.astype(int)
    if use_polar:
        for x_col, y_col in zip(X_SH_COLS[1:], Y_SH_COLS[1:]):
            r_col = np.hypot(
                all_data[x_col],
                all_data[y_col],
            )
            theta_col = np.arctan2(
                all_data[y_col],
                all_data[x_col],
            )

            all_data[x_col] = r_col
            all_data[y_col] = theta_col / np.pi
    
    if use_delta:
        all_data = take_deltas(all_data)

    return all_data

# Saves train/test/val/all data into a pickle file
def save_data(data_file_name, train=None, val=None, test=None, all_data=None):
    if train is not None:
        train.to_parquet(data_file_name.with_suffix(".pq.train"))

    if val is not None:
        val.to_parquet(data_file_name.with_suffix(".pq.val"))

    if test is not None:
        test.to_parquet(data_file_name.with_suffix(".pq.test"))

    if all_data is not None:
        all_data.to_parquet(data_file_name.with_suffix(".pq.all"))

    print(f"Saved data file (root name; add ext train,test,val,all): {data_file_name}")

# ##### pca and friends
# # get landmark data as a list with sequence lengths
def get_landmark_data(df):
    data = df.all_landmarks.to_list()
    data = [np.stack(frames) for frames in data]
    seq_lens = [seq.shape[0] for seq in data]

    data = np.concatenate(data, axis=0)
    return data, seq_lens

# get pca features (transform data then use seq_lens to partition it into seqs again)
def get_pca_features(pipe, df, data, seq_lens):
    data = pipe.transform(data)
    new_data = []
    start = 0

    for length in seq_lens:
        end = start + length
        new_data.append(data[start:end].tolist())
        start = end

    df.all_landmarks = new_data
    return df

# run and save PCA data
def pca(file_prefix, file_suffix, args, pipe=None):
    full_prefix = Path(ROOT).joinpath("data", file_prefix)

    file = full_prefix.with_suffix(file_suffix)
    if not file.is_file():
        raise FileNotFoundError("Run without --use_pca/-pca flag first to create the original dataset.")
    df = pd.read_parquet(file)
    data, lens = get_landmark_data(df)

    if not pipe:
        pipe = Pipeline([
            ("pca_solver", PCA(n_components=args.n_components))
        ])
        pipe.fit(data)
    df = get_pca_features(pipe, df, data, lens)

    new_prefix = get_data_file_name(args, pca_in_name=True)
    pca_path = Path(ROOT).joinpath("data", new_prefix)
    pca_path = pca_path.with_suffix(file_suffix)

    df.to_parquet(pca_path)
    print(f"Saved PCA Transform: {pca_path}")
    return pipe

