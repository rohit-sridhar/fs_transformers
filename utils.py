import os
import re
import sys
import torch
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.kernel_approximation import Nystroem

##### Directory information
ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT.joinpath("models")
ANALYSIS_ROOT = ROOT.joinpath("analysis")
SPLITS_ROOT = ROOT.joinpath("datasets", "splits")

##### Helpful column constants
# BID_PRICE_COLS = [f"p{i}" for i in range(0,6)]
# ASK_PRICE_COLS = [f"p{i}" for i in range(6,12)]
# BID_VOL_COLS = [f"v{i}" for i in range(0,6)]
# ASK_VOL_COLS = [f"v{i}" for i in range(6,12)]
# TRADE_PRICE_COLS = [f"dp{i}" for i in range(0,4)]
# TRADE_VOL_COLS = [f"dv{i}" for i in range(0,4)]

# X_COLS = BID_PRICE_COLS + ASK_PRICE_COLS + BID_VOL_COLS + ASK_VOL_COLS + TRADE_PRICE_COLS + TRADE_VOL_COLS
# TARGET_COLS = [f"t{i}" for i in range(0,2)]
# PCA_COMPS = 21
# PCA_COLS = [f"pca{i}" for i in range(PCA_COMPS)]
# X_COLS_START = 2      # accounts for seq_ix, step_in_seq and needs_prediction

##### Model Constants

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
    # "step_size",
]

###### ANALYSIS FUNCTIONS ######

#################### miscellaneous functions ######
# read_pickle
def read_pickle(data_file):
    df = pd.read_pickle(data_file)
    return df

# read_parquet
# reads the parquet file in and sets index.
# def read_parquet(data_file):
#     df = pd.read_parquet(data_file).set_index("seq_ix")
#     return df

#################### Data Analysis functions ######
# In the ending, not much was necessary
# so we comment everything out.
def misc_data_analysis(df):
    pass 
    
    # # check a few random sequences
    # print("Check Seq 0 (first 200 rows, bid price)")
    # seq_0 = df[df.seq_ix == 0]
    # with pd.option_context('display.max_rows', 200):
    #     print(seq_0[BID_PRICE_COLS])

    # print the nan count by column (then by seq if necessary)
    # print("Check NaN count by column")
    # print(df.isna().sum())

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

#################### split dataset ######
# get the split file name, using the split ratio argument
# if split ratio < 0.01 or >= 1.0 the filename contains the
# number of data points. otherwise the name contains the
# ratio * 100
def get_split_file_name(split_name, split_ratio, len_data, seed):
    # logic will overwrite if sr > 0.01 and has > 2 precision
    if split_ratio < 0.01:
        float_str = f"n{int(split_ratio * len_data)}"
    elif split_ratio >= 1.0:
        float_str = f"n{int(split_ratio)}"
    else:
        float_str = f"ts{int(split_ratio * 100)}"
    split_file_name = f"{split_name}_{float_str}_{seed}"

    save_dir = ROOT.joinpath("datasets/splits/")
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir.joinpath(split_file_name).with_suffix(".parquet")

# splits the dataset into train and val
def split_dataset(data_file, split_ratio, seed):
    df = read_parquet(data_file)
    seq_ix = list(set(df.index.tolist()))
    
    random.seed(seed)
    random.shuffle(seq_ix)

    train_end = int(split_ratio * len(seq_ix)) if split_ratio < 1 else int(split_ratio)
    if train_end < 1:
        raise ValueError("Pass a larger split ratio. Current train size is 0.")
    
    train_seq_ix = seq_ix[:train_end]
    val_seq_ix = seq_ix[train_end:]
    
    train_df = df.loc[train_seq_ix]
    val_df = df.loc[val_seq_ix]
    
    train_file_loc = get_split_file_name("train", split_ratio, len(seq_ix), seed)
    val_file_loc = get_split_file_name("val", split_ratio, len(seq_ix), seed)
    
    train_df.to_parquet(train_file_loc)
    val_df.to_parquet(val_file_loc)
    
    print(f"Created train file: {train_file_loc} of len {len(train_seq_ix)}")
    print(f"Created val file: {val_file_loc} of len {len(val_seq_ix)}")

#################### make small ######
def make_small(data_file, seed):
    df = read_parquet(data_file)
    seq_ix = list(set(df.index.tolist()))
    
    random.seed(seed)
    random.shuffle(seq_ix)
    
    small_end = int(0.05 * len(seq_ix))
    if small_end < 1:
        raise ValueError("Dataset cannot be made smaller")
    
    small_seq_ix = seq_ix[:small_end]
    small_df = df.loc[small_seq_ix]
    
    small_file_name = "_".join([data_file.stem, f"small-{seed}"])
    small_file_loc = data_file.parent.joinpath(small_file_name).with_suffix(".parquet")

    small_df.to_parquet(small_file_loc)
    print(f"Created small file: {small_file_loc} of len {len(small_seq_ix)}")


#################### make model cmp charts ######
# makes model comparison charts using interval args.
# samples 5 datapoints for each column
def make_plot_df(df):
    every_fifty = np.arange(len(df)) // 50

    df_mean_fifty = df.reset_index().groupby(every_fifty).mean()
    df_mean_fifty = df_mean_fifty.set_index("seq_ix")
    df_mean_fifty.index = df_mean_fifty.index.astype(int)
    
    return df_mean_fifty

# add seq ix and step in seq back to a data frame
def add_seq_ix_and_step(add_df, df_orig, step_in_seq):
    add_df["seq_ix"] = df_orig.index.to_list()
    add_df["step_in_seq"] = step_in_seq
    add_df = add_df.set_index("seq_ix")

    return add_df

# compute metrics from target df and out tensor
def compute_metrics(loss_fn, target, out):
    out_tch = out[:,99:,:].reshape((-1, out.shape[-1]))
    out_np = out_tch.detach().cpu().numpy()

    y_np = target.loc[:,TARGET_COLS].to_numpy()
    y_np = y_np.reshape((-1, SEQ_LEN, len(TARGET_COLS)))
    y_np = y_np[:,99:,:].reshape((-1, y_np.shape[-1]))
    y_tch = torch.from_numpy(y_np).float()

    corr = weighted_pearson_correlation(y_np, out_np)
    with torch.no_grad():
        loss = loss_fn(out_tch, y_tch) / y_np.shape[0]
    print(f"correlation of plotted pts: {corr}")
    print(f"loss of plotted pts: {loss}")


def make_model_cmp_charts(df, chkpt_path, interval=50):
    seq_ix_list = list(set(df.index.to_list()))
    model_files = load_model(chkpt_path, cpu=True)

    model_date = chkpt_path.parts[-4]
    model_name = "_".join([chkpt_path.parts[-2], chkpt_path.parts[-3]])
    model_type = model_files["model"][0]

    model = model_files["model"][1]
    preprocessor = model_files["preprocessor"][1]
    loss_fn = model_files["loss_fn"][1]

    seq_ix_list.sort()
    seq_to_plot = seq_ix_list[0:10]
    step_in_seq = df.loc[seq_to_plot, ["step_in_seq"]].step_in_seq.to_list()
    df_sample = df.loc[seq_to_plot, X_COLS + TARGET_COLS]

    if preprocessor is not None:
        data = preprocessor.transform(df_sample.to_numpy())
        data = data[:, len(X_COLS):]
        df_sample = pd.DataFrame(data, columns=TARGET_COLS+PCA_COLS)
        sys.exit(1)
        # data_in = torch.reshape(data_in, (5, 1000, -1)).float()
        # target = data[:, len(X_COLS):len(X_COLS + TARGET_COLS)]
        
        target = pd.DataFrame(target, columns=TARGET_COLS)
        target = add_seq_ix_and_step(target, df_sample, step_in_seq)
    else:
        data_in = torch.from_numpy(df_sample.to_numpy()[:,:-len(TARGET_COLS)]).float()
        data_in = data_in.reshape((-1, SEQ_LEN, data_in.shape[-1]))

        target = df_sample.loc[:,TARGET_COLS]
        target = add_seq_ix_and_step(target, df_sample, step_in_seq)

    out, _ = model.forward(data_in, return_seq=True)
    # if model_type.startswith("ProbGRULSTMModel"):
    #     out = out[0]
    #     # variances = out[1][:,:,2:]
    #     # direction = torch.where(variances > 0, 1, -1)
    #     # out = out[0] + (torch.exp(variances) * direction)
    compute_metrics(loss_fn, target, out)
    out = out.reshape((-1, len(TARGET_COLS))).detach().cpu().numpy()

    out = pd.DataFrame(out, columns=TARGET_COLS)
    out = add_seq_ix_and_step(out, df_sample, step_in_seq)
    
    target = make_plot_df(target)
    out = make_plot_df(out)
    
    for column in target.columns:
        if column == "step_in_seq" or column == "need_prediction":
            continue

        for seq_ix in seq_to_plot:
            plt.plot(
                target.loc[seq_ix].step_in_seq,
                target.loc[seq_ix][column],
                label=f"{seq_ix} target",
            )
            plt.plot(
                out.loc[seq_ix].step_in_seq,
                out.loc[seq_ix][column],
                label=f"{seq_ix} out",
            )
        
            plt.xlabel("Seq Step")
            plt.ylabel(column)
            plt.title(f"{column} Seq Step (Sample {len(seq_to_plot)})")
            
            plt_name = ["analysis", "cmp_model_plots"]
            plt_name.extend([model_date, model_name, f"interval{interval}", f"{column}_plot", f"{seq_ix}"])
            
            plt_path = ROOT.joinpath(*plt_name).with_suffix(".png")
            plt_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plt_path)
            
            plt.clf()

#################### make timeline charts ######
# makes timeline charts using interval args.
# samples 5 datapoints for each column
def make_timeline_charts(df, preprocess=False, interval=50, pca=False):
    seq_ix_list = list(set(df.index.to_list()))

    seq_ix_list.sort()
    seq_to_plot = seq_ix_list[500:505]

    df_sample = df.loc[seq_to_plot]
    df_mean_fifty = make_plot_df(df_sample)

    for column in df_mean_fifty.columns:
        if column == "step_in_seq" or column == "need_prediction":
            continue

        for seq_ix in seq_to_plot:
            plt.plot(
                df_mean_fifty.loc[seq_ix].step_in_seq,
                df_mean_fifty.loc[seq_ix][column],
                label=str(seq_ix),
            )
        
        plt.xlabel("Seq Step")
        plt.ylabel(column)
        plt.title(f"{column} Seq Step (Sample {len(seq_to_plot)})")
        
        plt_name = ["analysis", "interval_plots"]
        if pca:
            plt_name.append("pca")
        plt_name.extend([f"interval{interval}", f"{column}_plot"])
        plt_name[-1] += "_pp" if preprocess else ""
        
        plt_path = ROOT.joinpath(*plt_name).with_suffix(".png")
        plt_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plt_path)
        
        plt.clf()

def get_args_from_output(lines):
    for ln in lines:
        if "Namespace" in ln:
            match = re.search(r"Namespace.+$", ln)
            if match:
                all_hyperparams = match.group()
                all_hyperparams = all_hyperparams.replace(str(SPLITS_ROOT) + "/", "")
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

#################### UNFINISHED ######
#################### make gradient plots ######
# note that wrapper functions are used for Python closure
# so that we can pass arguments.
def load_model(chkpt_path, cpu=False):
    if torch.cuda.is_available() and not cpu:
        model_files = torch.load(chkpt_path, weights_only=False, map_location=torch.device("cuda"))
    else:
        model_files = torch.load(chkpt_path, weights_only=False, map_location=torch.device("cpu"))
    
    return model_files

def get_args_from_log_file(log_file):
    with open(log_file, "r") as f:
        args = f.readline()
    return args

def get_data_file_from_args(args):
    df_match = re.search(r"data_file=PosixPath\(.[^\)]*[\)]{1}", args)
    pca_match = re.search(r"use_pca=.[^,]*,", args)

    data_file = df_match.group()[21:-2]
    use_pca = bool(pca_match.group()[8:-1])
    
    df = read_parquet(data_file)
    _, pipe = preprocess(df, pca=use_pca)
    return df
    

# from torch website
def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads.append((module_name, grad))
    return hook

def get_all_layers(model, hook_forward, hook_backward):
    """Register forward pass hook (which registers a backward hook) to model outputs

    Returns:
        - layers: a dict with keys as layer/module and values as layer/module names
                  e.g. layers[nn.Conv2d] = layer1.0.conv1
        - grads: a list of tuples with module name and tensor output gradient
                 e.g. grads[0] == (layer1.0.conv1, tensor.Torch(...))
    """
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

def mini_train(model, optimizer, loss_fn, grads):
    epochs = 10
    for epoch in range(epochs):
        # important to clear, because we append to
        # outputs everytime we do a forward pass
        grads.clear()
        
        optimizer.zero_grad()
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()

def make_gradient_plots(chkpt_path):
    model_files = load_model(chkpt_path)
    optimizer, model = model_files["optimizer"][1], model_files["model"][1]
    # if model_files["model"][0].startswith("ProbGRULSTM"):
    loss_fn = model_files["loss_fn"][1]

    args_str = get_args_from_log_file(chkpt_path.parent.joinpath("log.txt"))
    df = get_data_file_from_args(args_str)

    _, grads = get_all_layers(model, hook_forward, hook_backward)
    mini_train(model, df, optimizer, loss_fn, grads)
    
#################### make loss plots ######
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
            
            loss_match = re.findall("[0-9]+\.[0-9]+", ln)
            if len(loss_match) == 2:
                train_losses.append(float(loss_match[0]))
                val_losses.append(float(loss_match[1]))
        elif "Loss and score of epoch" in ln:
            epoch_match = re.search("epoch [0-9]+ -", ln)
            if epoch_match:
                epoch = re.search("[0-9]+", epoch_match.group())
                epochs.append(int(epoch.group()))
            
            loss_match = re.findall(r"-?[0-9]+\.[0-9]+(?:e[+|-][0-9]+)?", ln)
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

##### PCA Results #####
# pca on ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'dp0', 'dp1', 'dp2', 'dp3', 'dv0', 'dv1', 'dv2', 'dv3']
# variance ratios: [2.17503404e-01 1.09468576e-01 7.57392353e-02 6.28164566e-02
#  5.14800291e-02 4.41038505e-02 4.06376729e-02 4.03281762e-02
#  3.79259263e-02 3.29952990e-02 3.29014272e-02 3.13207802e-02
#  2.96195921e-02 2.70458917e-02 2.55823615e-02 2.35667024e-02
#  2.29956895e-02 2.15952423e-02 1.98581638e-02 1.73981819e-02
#  1.30439505e-02 6.00410772e-03 4.55362222e-03 3.41339648e-03
#  2.50729279e-03 1.49543049e-03 1.30988965e-03 1.06017397e-03
    #  7.53418304e-04 5.51940936e-04 3.03332923e-04 1.20785941e-04]

# pca on ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
# variance ratios: [0.67831032 0.2077182  0.05442041 0.02965447 0.02108016 0.00881643]

# pca on ['v0', 'v1', 'v2', 'v3', 'v4', 'v5']
# variance ratios: [0.30043638 0.21383082 0.16205762 0.11778383 0.11419606 0.09169528]

# pca on ['p6', 'p7', 'p8', 'p9', 'p10', 'p11']
# variance ratios: [0.56647158 0.30512466 0.06667396 0.02923335 0.02235182 0.01014464]

# pca on ['v6', 'v7', 'v8', 'v9', 'v10', 'v11']
# variance ratios: [0.32323046 0.16757107 0.14316814 0.13651361 0.12785393 0.10166279]

# pca on ['dp0', 'dp1', 'dp2', 'dp3']
# variance ratios: [0.55558456 0.28964016 0.08860935 0.06616592]

# pca on ['dv0', 'dv1', 'dv2', 'dv3']
# variance ratios: [0.42635404 0.33645795 0.15317949 0.08400852]

#################### PCA ######
# runs pca and prints the components
# def run_pca(df, seed=673, cols=X_COLS):
#     print(f"running pca on input cols")
#     X = df[cols]
# 
#     solver = Pipeline([
#         ("pca_step", PCA()),
#     ])
#     X_pca = solver.fit_transform(X)
#     
#     var_ratios = solver["pca_step"].explained_variance_ratio_
#     print(f"variance ratios: {var_ratios}")
#     print()

#################### Preprocesses the data ######
# preprocess by scaling the data and pca (if bool)
def preprocess(df, pca=False):
    data = df[X_COLS + TARGET_COLS].to_numpy()
    
    transformers = [
        ("z_norm", StandardScaler(), slice(len(X_COLS + TARGET_COLS))),
    ]

    if pca:
        transformers.append(
            ("pca", PCA(n_components=PCA_COMPS), slice(len(X_COLS))),
        )

    pipe = ColumnTransformer(
        transformers,
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    # pipe.set_output(transform="pandas")
    scaled_data = pipe.fit_transform(data)

    if pca:
        # set new pca col names. drop old X cols
        df = df.drop(columns=X_COLS)
        # df[pca_cols + TARGET_COLS] = scaled_data[pca_cols + TARGET_COLS]
        df[PCA_COLS] = scaled_data[:, slice(len(X_COLS + TARGET_COLS), None)]
        df[TARGET_COLS] = scaled_data[:, slice(len(X_COLS), len(X_COLS + TARGET_COLS))]
        
        # rearrange cols and put target at the end
        new_col_order = list(df.columns[:-(PCA_COMPS + len(TARGET_COLS))]) + \
             list(df.columns[-PCA_COMPS:]) + TARGET_COLS
        df = df[new_col_order]
    else:
        df[X_COLS + TARGET_COLS] = scaled_data

    return df, pipe
    
#################### Measure functions ######
def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Weighted Pearson Correlation Coefficient.

    This metric emphasizes performance on data points with larger target amplitudes
    (larger price movements) by using the absolute value of the target as a sample weight.

    Predictions are clipped to the range [-6, 6] before calculation to prevent
    outliers from dominating the metric.

    Args:
        y_true: Ground truth target values (numpy array).
        y_pred: Predicted values (numpy array).

    Returns:
        float: Weighted Pearson correlation coefficient.
    """
    # Clip predictions to valid range [-6, 6]
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)

    # Calculate weights based on target amplitude
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)

    # Calculate weighted means
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0

    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w

    # Calculate weighted deviations
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred

    # Calculate weighted covariance
    cov = np.sum(weights * dev_true * dev_pred) / sum_w

    # Calculate weighted variances
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w

    # Compute correlation
    if var_true <= 0 or var_pred <= 0:
        return 0.0

    corr = cov / (np.sqrt(var_true) * np.sqrt(var_pred))
    return float(corr)

# Modeling functions ######
@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # return dummy prediction
        return np.zeros(2)


class ScorerStepByStep:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)

        # Calc feature dimension: first 3 columns are seq_ix, step_in_seq & need_prediction
        # Total columns: 3 metadata + 32 features + 2 targets = 37
        # Features are cols [3:35]
        self.dim = 2
        self.features = self.dataset.columns[3:35]
        self.targets = self.dataset.columns[35:]

    def score(self, model: PredictionModel) -> dict:
        predictions = []
        targets = []

        prediction = None

        # Iterate over numpy array for speed
        for row in tqdm(self.dataset.values):
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            lob_data = row[3:35]  # 32 features
            labels = row[35:]     # 2 targets
            
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            prediction = model.predict(data_point)
            
            self.check_prediction(data_point, prediction)
            if prediction is not None:
                predictions.append(prediction)
                targets.append(labels)

        # report metrics
        return self.calc_metrics(np.array(predictions), np.array(targets))

    def check_prediction(self, data_point: DataPoint, prediction: np.ndarray):
        if not data_point.need_prediction:
            if prediction is not None:
                raise ValueError(f"Prediction is not needed for {data_point}")
            return

        if prediction is None:
            raise ValueError(f"Prediction is required for {data_point}")

        if prediction.shape[0] != self.dim:
            raise ValueError(
                f"Prediction has wrong shape: {prediction.shape[0]} != {self.dim}"
            )

    def calc_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        scores = {}
        for ix_target, target_name in enumerate(self.targets):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix_target], predictions[:, ix_target]
            )
        scores["weighted_pearson"] = np.mean(list(scores.values()))
        return scores
        
