import argparse
import sys
from pathlib import Path
from utils import (
    NEW_DATASETS
)

# Choices for analysis args
ANALYSIS_CHOICES = [
    "loss_plots",
    "make_small",
    "sample_classification",
    "count_label_chars",
]

# ranged float with open/closed interval options
def ranged_float(
    low=0.0,
    high=1.0,
    incl_low=False,
    incl_high=False
):
    def check_type(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"arg {x} not a float val")

        if (incl_low and x < low) or (not incl_low and x <= low):
            raise argparse.ArgumentTypeError(f"Not in range ({low}, {high})")

        if (incl_high and x > high) or (not incl_high and x >= high):
            raise argparse.ArgumentTypeError(f"Not in range ({low}, {high})")

        return x
    
    return check_type

# ranged int. all intervals are closed (since they are ints)
# Use -inf/+inf for unbounded sides.
def ranged_int(
    low=float("-inf"),
    high=float("inf"),
):
    def check_type(x):
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"arg {x} not a int val")

        if x < low or x > high:
            raise argparse.ArgumentTypeError(f"Not in range ({low}, {high})")

        return x
    
    return check_type

# check if path is valid and if it exists
def path_exists(pth):
    try:
        pth = Path(pth)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{pth} not a valid path.")

    if not pth.exists():
        print(pth)
        raise argparse.ArgumentTypeError("data path must exist.")

    return pth.resolve()

# check that conditions are met for an arg to be required
def arg_has_val(arg_name_opts, arg_val_opts):
    for i,arg in enumerate(sys.argv):
        if arg in arg_name_opts and i + 1 < len(sys.argv):
            return sys.argv[i + 1] in arg_val_opts
    return False

def is_script(script_names):
    for script_name in script_names:
        if script_name in sys.argv[0]:
            return True
    return False

def is_required(arg_name):
    if arg_name == "models_dir":
        return arg_has_val(["-at", "--analysis_type"], ["loss_plots"])
    elif arg_name == "chkpt_path":
        return is_script(["test.py"])
    elif arg_name == "data_file":
        return arg_has_val(["-at", "--analysis_type"], ["sample_classification", "make_small"]) \
                or is_script(["train.py", "test.py"])
    elif arg_name == "analysis_type":
        return is_script(["analysis.py"])
    elif arg_name == "dataset":
        return arg_has_val(["-at", "--analysis_type"], ["sample_classification", "count_label_chars"]) \
                or is_script(["preprocess.py"])
    elif arg_name == "model_sub_root" or arg_name == "model_type":
        return is_script(["train.py"])
    elif arg_name == "n":
        return arg_has_val(["-at", "--analysis_type"], ["sample_classification"]) \
                or is_script(["test.py"])
    elif arg_name == "sample_strategy":
        return arg_has_val(["-at", "--analysis_type"], ["sample_classification"])

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model Args
    parser.add_argument(
        "-oft",
        "--overfit",
        action="store_true",
        help="If true, trains move in eval mode to try and overfit to the dataset."
    )
    parser.add_argument(
        "-std",
        "--std_out",
        action="store_true",
        help="If set, logs to stdout instead of log file."
    )
    parser.add_argument(
        "-msr",
        "--model_sub_root",
        type=str,
        required=is_required("model_sub_root"),
        help="the model sub root to store model/logs in. not a full path (train)",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        choices=["ByT5"],
        required=is_required("model_type"),
        help="model type to train on (train)",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        choices=["Adam", "RMSprop", "Adafactor"],
        default="Adam",
        help="optimizer type for training",
    )
    parser.add_argument(
        "-se",
        "--save_every",
        type=ranged_int(low=1),
        default=100,
        help="save every n epochs (train)",
    )
    parser.add_argument(
        "-ep",
        "--num_epochs",
        type=ranged_int(low=1),
        default=500,
        help="number of epochs to train for (train LSTM/ConvLSTM)",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=ranged_int(low=1),
        default=1,
        help=(
            "batch size (number of seqs to be collated) per train step "
            "(train LSTM/ConvLSTM)"
        ),
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=ranged_float(low=0.0, incl_low=False),
        default=1e-5,
        help="learning rate for training (train LSTM/ConvLSTM)",
    )

    # testing Args
    parser.add_argument(
        "-tt",
        "--test_type",
        type=str,
        choices=["gen", "fwd", "grd"],
        default="gen",
        help="how to search for the output seq. pass 'fwd' for sanity checking",
    )
    parser.add_argument(
        "-cp",
        "--chkpt_path",
        type=path_exists,
        required=is_required("chkpt_path"),
        help="models dir for loss plot analysis (analysis)",
    )
    parser.add_argument(
        "-gpu",
        "--use_gpu",
        action="store_true",
        help="whether to use the GPU during test time (doesn't apply to trsining)",
    )

    # analysis args
    parser.add_argument(
        "-at",
        "--analysis_type",
        type=str,
        choices=ANALYSIS_CHOICES,
        required=is_required("analysis_type"),
        help="analysis type to run. (analysis)",
    )
    parser.add_argument(
        "-ss",
        "--sample_strategy",
        type=str,
        choices=["stratified", "categorical"],
        required=is_required("sample_strategy"),
        help=(
            "sampling strategy: if stratified, samples all categories (proportionally) "
            "and creates on dataframe. if cetegorical, samples n per category "
            "and creates separate data frames."
        ),
    )
    parser.add_argument(
        "-md",
        "--models_dir",
        type=path_exists,
        required=is_required("models_dir"),
        help="models dir for loss plot analysis (analysis)",
    )
    
    # preprocess args
    parser.add_argument(
        "-so",
        "--split_once",
        action="store_true",
        help="split into just train/val (preprocess)",
    )
    parser.add_argument(
        "-up",
        "--use_polar",
        action="store_true",
        help="use polar coordinates instead (preprocess)",
    )
    parser.add_argument(
        "-ud",
        "--use_delta",
        action="store_true",
        help="use deltas between rows (preprocess)",
    )
    parser.add_argument(
        "-dna",
        "--dropna",
        action="store_true",
        help=(
            "Drop NA values. If False, will replace with 0s. if interpolation "
            "is not None, interpolates first then drops na. (preprocess)"
        ),
    )
    parser.add_argument(
        "-mlh",
        "--make_left_handed",
        action="store_true",
        help="Make the data left handed instead of right handed (preprocess)",
    )
    parser.add_argument(
        "-ip",
        "--interpolate_val",
        type=ranged_int(0),
        default=None,
        help=(
            "Insert average between each non-zero frame if > 0. Only fills "
            "in nans if 0 is passed. (preprocess)"
        ),
    )
    parser.add_argument(
        "-nthr",
        "--na_threshold",
        type=ranged_float(0.0, 1.0, False, True),
        default=1.0,
        help="Filter out sequences with greater than na_threshold pct na frames (preprocess)",
    )
    parser.add_argument(
        "-tr",
        "--train_ratio",
        type=ranged_float(0.0, 1.0, True, True),
        default=0.9,
        help=(
            "Percentage of sequences for training. Whole set is split train_ratio/1-train_ratio "
            "for test/test. Train set is further split by train_ratio for train/val (preprocess)"
        ),
    )
    parser.add_argument(
        "-pt",
        "--participant_id",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Filters data to specific participant. If an empty list, uses "
            "all participants (preprocess)"
        ),
    )
    parser.add_argument(
        "-ptgrp",
        "--participant_grp_name",
        type=str,
        default=None,
        help="Name for participant group (preprocess)",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        required=is_required("dataset"),
        choices=NEW_DATASETS,
        help="Source dataset to preprocess (preprocess)",
    )
    parser.add_argument(
        "-ne",
        "--data_file_name_ext",
        type=str,
        default=None,
        help=(
            "data file name extension when outputting preprocessed pickle. this is not "
            "a file extension. (preprocess)"
        ),
    )

    # Shared
    parser.add_argument(
        "-pca",
        "--use_pca",
        action="store_true",
        help="whether or not to run pca. keeps 21 components (shared)",
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        action="store_true",
        help="run in debug mode (smaller datasets, verbose)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=ranged_int(low=0),
        required=is_required("n"),
        help=(
            "sample size for testing, sampling (analysis). if 0 (for testing) "
            "uses the entire dataset."
        ),
    )
    parser.add_argument(
        "-nc"
        "--n_components",
        type=ranged_int(1),
        default=10,
        help="n components for pca (shared)",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=1248,
        help="seed for randomness (shared)",
    )
    parser.add_argument(
        "-bp",
        "--bar_position",
        type=ranged_int(0),
        default=0,
        help="bar position for tqdm when multiprocessing (shared)",
    )
    parser.add_argument(
        "-df",
        "--data_file",
        type=path_exists,
        required=is_required("data_file"),
        help="data file to use when running a routine (shared)",
    )

    return parser.parse_args()
