import argparse
import sys
from pathlib import Path

# Choices for analysis args
ANALYSIS_CHOICES = [
    "split_val",
    "cmp_model_out",
    "get_basic_stats",
    "loss_plots",
    "gradient_plots",
    "make_small",
    "pca",
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
        return arg_has_val(["-at", "--analysis_type"], ["loss_plots", "gradients"])
    elif arg_name == "chkpt_path":
        return arg_has_val(["-at", "--analysis_type"], ["gradient_plots", "cmp_model_out"]) \
                or is_script(["test.py"])
    elif arg_name == "intervals":
        return arg_has_val(["-at", "--analysis_type"], ["get_basic_stats"])
    elif arg_name == "data_file":
        return arg_has_val(["-at", "--analysis_type"], ["cmp_model_out", "get_basic_stats", "split_val", "make_small"]) \
                or is_script(["train.py"])
    elif arg_name == "analysis_type":
        return is_script(["analysis.py"])
    elif arg_name == "model_sub_root" or arg_name == "model_type":
        return is_script(["train.py"])

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model Args
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
        choices=["LSTM", "GRU", "GRULSTM", "ProbGRULSTM1", "ProbGRULSTM2", "Transformer"],
        required=is_required("model_type"),
        help="model type to train on (train)",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        choices=["Adam", "RMSprop"],
        default="Adam",
        help="optimizer type for training",
    )
    parser.add_argument(
        "-lf",
        "--loss_fn",
        type=str,
        choices=["MSE", "MAE", "GNLL", "CEL"],
        default="MSE",
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
        help="batch size (number of seqs to be collated) per train step (train LSTM/ConvLSTM)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=ranged_float(low=0.0, incl_low=False),
        default=1e-5,
        help="learning rate for training (train LSTM/ConvLSTM)",
    )

    # GRU/LSTM/ConvLSTM Args
    # parser.add_argument(
    #     "-hs",
    #     "--hidden_size",
    #     type=ranged_int(low=1),
    #     default=128,
    #     help="num layers for lstm model (train LSTM/ConvLSTM)",
    # )
    # parser.add_argument(
    #     "-nl",
    #     "--num_layers",
    #     type=ranged_int(low=1),
    #     default=3,
    #     help="num layers for lstm model (train LSTM/ConvLSTM)",
    # )
    # parser.add_argument(
    #     "-dp",
    #     "--dropout",
    #     type=ranged_float(0.0, 1.0, True, False),
    #     default=0.0,
    #     help="use dropout with LSTM (train LSTM/ConvLSTM)",
    # )

    # ConvGRU Only Args
    # parser.add_argument(
    #     "-ks",
    #     "--kernel_size",
    #     type=ranged_int(low=2),
    #     default=5,
    #     help="kernel size for convolutions (train ConvLSTM)",
    # )
    # parser.add_argument(
    #     "-oc",
    #     "--out_channels",
    #     type=ranged_int(low=0),
    #     default=0,
    #     help="out channels for conv layer. set to 0 to use in_channels (train ConvLSTM)",
    # )
    # parser.add_argument(
    #     "-st",
    #     "--stride",
    #     type=ranged_int(low=1),
    #     default=1,
    #     help="out channels for conv layer. set to 0 to use in_channels (train ConvLSTM)",
    # )

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
        "--intervals",
        type=int,
        nargs="+",
        required=is_required("intervals"),
        help="time interval for plots (analysis)",
    )
    parser.add_argument(
        "-md",
        "--models_dir",
        type=path_exists,
        required=is_required("models_dir"),
        help="models dir for loss plot analysis (analysis)",
    )
    parser.add_argument(
        "-cp",
        "--chkpt_path",
        type=path_exists,
        required=is_required("chkpt_path"),
        help="models dir for loss plot analysis (analysis)",
    )
    parser.add_argument(
        "-sr",
        "--split_ratio",
        type=ranged_float(0.0, float("inf"), False, False),
        default=0.9,
        help="split ratio when doing train/test split. pass val in (0,1) for ratio and >= 1 for n samples (analysis)",
    )
    # Shared
    parser.add_argument(
        "-pca",
        "--use_pca",
        action="store_true",
        help="whether or not to run pca. keeps 21 components (shared)",
    )
    # parser.add_argument(
    #     "-pp",
    #     "--preprocess",
    #     action="store_true",
    #     help="whether or not to preprocess before analysis/training (shared)",
    # )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        # default=465, # Seed learns seq 52182 and 55278 perfectly with greedy decoding (sometimes, at least)
        default=248, # Seed learns seq 52182 and 55278 perfectly with greedy decoding (sometimes, at least)
        help="seed for randomness (shared)",
    )
    parser.add_argument(
        "-bp",
        "--bar_position",
        type=int,
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
