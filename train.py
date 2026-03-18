import os
import sys
import torch
import logging
import json
import inspect
import pandas as pd
import numpy as np

from datetime import datetime
from hashlib import blake2b
from pathlib import Path
from tqdm import tqdm

from torch import nn
from torch.optim import Adam, RMSprop, Adafactor
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer
from models import T5ForConditionalGenerationProjection

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Adjust path to import utils from parent directory
TRAIN_FILE_ROOT = Path(__file__).resolve().parent
sys.path.append(f"{TRAIN_FILE_ROOT}/..")

from utils import (
    read_pickle,
    df_to_bytes,
    MODELS_ROOT,
    BYT5_NUM_SPECIAL_TOKENS,
)
from args import parse_args
from models import (
    TransformerModel,
    DATA_PAD,
    # SUPP_PAD_IDX,
    INPUT_DIM,
    # SUPP_OUTPUT_DIM,
    IdxDataset,
    SimpleRandomSampler,
    collate_seq,
)

global args

######################################## Setup functions ########################################

# gets arg hash, used to name output files and folders
def get_arg_hash(args):
    args_dict = vars(args)
    args_dict = {key:str(args_dict[key]) for key in args_dict}

    args_str = json.dumps(args_dict).encode()
    args_hash = blake2b(args_str, digest_size=5).hexdigest()
    
    return args_hash

# set up the logger for training
def setup_logger(log_dir):
    log_file = log_dir.joinpath(f"log").with_suffix(".txt")

    filemode='w'
    if log_file.exists():
        filemode='a'

    if args.std_out:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d | %H:%M:%S'
        )
    else:
        logging.basicConfig(
            filename=log_file,
            filemode=filemode,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d | %H:%M:%S'
        )

# Sets up the sampler and loader and returns both.
# The sampler samples by seq_ix while the loader loads
# and collates sequences using the sampler.
# modified for ByT5
def get_sampler_and_loader(df, idx_list, device, generator, pad_token_id=0):
    X = df.loc[idx_list, ["all_landmarks"]]
    y = df.loc[idx_list, ["phrase"]]

    dataset = IdxDataset(X, y)
    idx_sampler = SimpleRandomSampler(idx_list, generator=generator)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=idx_sampler,
        collate_fn=lambda seq: collate_seq(
            seq,
            device=device,
            pad_token_id=pad_token_id
        ),
    )
    
    return idx_sampler, loader

# return the loss function base on args
# modified for ByT5
def get_loss_fn():
    if args.model_type == "ByT5":
        return None

    if args.loss_fn == "MSE":
        return nn.MSELoss(reduction="sum")
    elif args.loss_fn == "MAE":
        return nn.L1Loss(reduction="sum")
    elif args.loss_fn == "GNLL":
        if not args.model_type.startswith("ProbGRULSTM"):
            raise ValueError("Must pass ProbGRULSTM model type for GNLL loss")
        return (nn.L1Loss(reduction="sum"), nn.GaussianNLLLoss(reduction="sum"))
    elif args.loss_fn == "CEL":
        if not args.model_type.startswith("Transformer"):
            raise ValueError("Must use Transformer model with CrossEntropy Loss")
        return nn.CrossEntropyLoss(reduction="none")
        # return nn.CrossEntropyLoss(reduction="mean")
        # return nn.CrossEntropyLoss(reduction="sum", ignore_index=SUPP_PAD_IDX)

# returns optimizer based on optimizer type
def get_optimizer(model):
    if args.optimizer == "Adafactor":
        return Adafactor(model.parameters())
    elif args.optimizer == "Adam":
        return Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "RMSprop":
        return RMSprop(model.parameters(), lr=args.learning_rate, centered=True)

# Get model based on args model type
def get_model(device):
    # if args.model_type == "LSTM":
    #     model = LSTMModel(
    #         INPUT_DIM,
    #         args.hidden_size,
    #         len(TARGET_COLS),
    #         num_layers=args.num_layers,
    #         dropout=args.dropout,
    #     )
    # elif args.model_type == "GRU":
    #     model = GRUModel(
    #         INPUT_DIM,
    #         32, # args.hidden_size,
    #         4, # len(TARGET_COLS),
    #         2, # num_layers=args.num_layers,
    #         dropout=0.0, # dropout=args.dropout,
    #     )
    # elif args.model_type == "GRULSTM":
    #     model = GRULSTMModel(
    #         INPUT_DIM,
    #         args.hidden_size,
    #         len(TARGET_COLS),
    #         num_layers=args.num_layers,
    #         dropout=args.dropout,
    #     )
    # elif args.model_type == "ProbGRULSTM1":
    #     if args.loss_fn != "GNLL":
    #         raise ValueError("Must pass GNLL loss type for ProbGRULSTM1")
    #     
    #     model = ProbGRULSTMModel1(
    #         INPUT_DIM,
    #         args.hidden_size,
    #         len(TARGET_COLS),
    #         num_layers=args.num_layers,
    #         dropout=args.dropout,
    #     )
    # elif args.model_type == "ProbGRULSTM2":
    #     if args.loss_fn != "GNLL":
    #         raise ValueError("Must pass GNLL loss type for ProbGRULSTM2")
    #     
    #     model = ProbGRULSTMModel2(
    #         INPUT_DIM,
    #         args.hidden_size,
    #         len(TARGET_COLS),
    #         num_layers=args.num_layers,
    #         dropout=args.dropout,
    #     )
    # elif args.model_type == "Transformer":
    #     if args.loss_fn != "CEL":
    #         raise ValueError("Must use CEL with Transformer model type")
    #     model = TransformerModel(INPUT_DIM, SUPP_OUTPUT_DIM)
    if args.model_type == "ByT5":
        model_args = [INPUT_DIM]
        model = T5ForConditionalGenerationProjection.from_pretrained(
            "google/byt5-small",
            *model_args,
        )
        model.init_linear_projection()

        # print(torch.all(model.encoder.embed_tokens.weight == model.decoder.embed_tokens.weight))
        # print(torch.all(model.shared.weight == model.decoder.embed_tokens.weight))
        # sys.exit(1)

    model.to(device)
    model.train()

    return model

######################################## Training function and friends ########################################

# save model and various params and data about current state
# def save_model(model, optimizer, preprocessor, epoch, loss_fn, save_dir):
def save_model(model, optimizer, epoch, loss_fn, save_dir):
    chkpt_name = save_dir.joinpath(f"{epoch}_of_{args.num_epochs}").with_suffix(".chkpt")
    loss_tuple = (loss_fn.__class__.__name__, loss_fn)

    torch.save({
        "epoch": epoch,
        "loss_fn": loss_tuple,
        "model": (model.__class__.__name__, model),
        "optimizer": (optimizer.__class__.__name__, optimizer),
        # "preprocessor": (preprocessor.__class__.__name__, preprocessor),
    }, chkpt_name)

# get the batch score (if 
def get_batch_score(y_pred, y):
    y_pred = torch.reshape(y_pred, (-1, y_pred.shape[-1])).detach().cpu().numpy()
    y = torch.reshape(y, (-1, y.shape[-1])).detach().cpu().numpy()
    
    scores = []
    for tgt_ix in range(y_pred.shape[-1]):
        score = weighted_pearson_correlation(y[:, tgt_ix], y_pred[:,tgt_ix])
        scores.append(score)
    
    return scores

# get total computed loss for a batch
def get_batch_loss(loss_fn, out, y):
    if args.model_type.startswith("ProbGRULSTM"):
        t_out = out[0]
        stats_out = out[1]

        means = stats_out[:,:,:2]
        variances = torch.exp(stats_out[:,:,2:])

        loss = (0.55 * loss_fn[0](t_out, y)) + (0.45 * loss_fn[1](means, y, variances))
    else:
        loss = loss_fn(out, y)

    return loss

# def get_padding_mask(src, tgt, ans, device):
#     src_padding_mask = (src.sum(dim=-1) == DATA_PAD*src.shape[-1]).to(device)
#     tgt_padding_mask = (tgt == SUPP_PAD_IDX).to(device)
#     ans_padding_mask = (ans == SUPP_PAD_IDX).to(device)
# 
#     return src_padding_mask, tgt_padding_mask, ans_padding_mask

# def get_causal_mask(model, tgt, device):
#     tgt_mask = model.transformer.generate_square_subsequent_mask(
#         tgt.shape[1], device=device
#     )
#     return tgt_mask != 0.0

# get the validation loss (jsut for the training loop)
def get_val_loss_and_score(
    model,
    loader,
    device,
    pad_token_id=0
):
    len_loader = len(loader)
    total_scores = []
    total_tgts = []
    total_losses = []

    ####!!! Set bool below for small datasets !!!####
    # model.eval()
    ####!!! ||||||||||||||||||||||||||||||||| !!!####

    for seq in tqdm(
        loader,
        total=len_loader,
        desc="validation",
        leave=False,
        position=args.bar_position+2,
        disable=args.std_out,
    ):
        with torch.no_grad():
            inputs_embeds = seq[0]
            labels = seq[1]
            
            outputs = model(
                inputs_embeds=inputs_embeds,
                labels=labels,
            )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, axis=-1)
            tgts = (labels != pad_token_id)
            score = (labels == preds) * tgts

        total_losses.append(loss.item())
        total_tgts.append(tgts.sum().item())
        total_scores.append(score.sum().item() / tgts.sum().item())

        ################################################## OLD TRANSFORMER TRAINING
        #     tgt = seq[1][:,:-1]
        #     ans = seq[1][:,1:]
        #     
        #     first mask is the causal mask (for teacher forcing autoregressive training)
        #     tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.shape[1], device=device)
        #     tgt_mask = get_causal_mask(model, tgt, device)
        #     src_padding_mask, tgt_padding_mask, ans_padding_mask = get_padding_mask(
        #         src,
        #         tgt,
        #         ans,
        #         device
        #     )

        #     out = model.forward(
        #         src,
        #         tgt,
        #         tgt_mask=tgt_mask,
        #         src_padding_mask=src_padding_mask,
        #         tgt_padding_mask=tgt_padding_mask,
        #     )
        #     
        #     out_flat = out.reshape((-1, SUPP_OUTPUT_DIM))
        #     ans_flat = one_hot(ans, num_classes=SUPP_OUTPUT_DIM).float()
        #     ans_flat = ans_flat.reshape((-1, SUPP_OUTPUT_DIM))
        #     
        #     ans_padding_mask = ans_padding_mask.reshape((-1))
        #     loss = get_batch_loss(loss_fn, out_flat, ans_flat)
        #     loss = (loss * ~ans_padding_mask).sum() / (~ans_padding_mask).sum()
        # total_scores.append(get_batch_score(out, y))
        # total_tgts += ans.shape[0] * ans.shape[1]
        # total_loss += loss.item()

    model.train()
    val_loss = np.average(total_losses, weights=total_tgts, axis=0)
    val_score = np.average(total_scores, weights=total_tgts, axis=0)

    # return val loss and val score
    return val_loss, val_score

# main model training loop
# modified for ByT5
def train_model(
    model,
    sampler,
    train_loader,
    val_loader,
    save_dir,
    device,
    pad_token_id=0,
):
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()
    
    model.eval()
    for epoch in tqdm(
        list(range(1, args.num_epochs + 1)),
        desc="epochs",
        position=args.bar_position,
        disable=args.std_out,
    ):
        epoch_tgts = []
        epoch_losses = []
        epoch_scores = []
        
        len_loader = len(train_loader)
        for seq in tqdm(
            train_loader,
            total=len_loader,
            desc="sequences",
            leave=False,
            position=args.bar_position+1,
            disable=args.std_out,
        ):
            inputs_embeds = seq[0]
            labels = seq[1]

            inputs_attention_mask = (inputs_embeds != pad_token_id).all(dim=-1)
            decoder_attention_mask = (labels != pad_token_id)

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, axis=-1)
            tgts = (labels != pad_token_id)
            score = (labels == preds) * tgts

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            epoch_tgts.append(tgts.sum().item())
            epoch_scores.append(score.sum().item() / tgts.sum().item())

            ################################################## OLD TRANSFORMER CODE
            # src = seq[0]
            # tgt = seq[1][:,:-1]
            # ans = seq[1][:,1:]
            # 
            # # first mask is the causal mask (for teacher forcing autoregressive training)
            # tgt_mask = get_causal_mask(model, tgt, device)
            # src_padding_mask, tgt_padding_mask, ans_padding_mask = get_padding_mask(
            #     src,
            #     tgt,
            #     ans,
            #     device
            # )
            # 
            # optimizer.zero_grad()
            # out = model.forward(
            #     src,
            #     tgt,
            #     tgt_mask=tgt_mask,
            #     src_padding_mask=src_padding_mask,
            #     tgt_padding_mask=tgt_padding_mask,
            # )
            # 
            # out_flat = out.reshape((-1, SUPP_OUTPUT_DIM))
            # ans_flat = one_hot(ans, num_classes=SUPP_OUTPUT_DIM).float()
            # ans_flat = ans_flat.reshape((-1, SUPP_OUTPUT_DIM))

            # ans_padding_mask = ans_padding_mask.reshape((-1))
            # num_tgts = (~ans_padding_mask).sum()
            # loss = get_batch_loss(loss_fn, out_flat, ans_flat)
            # loss = (loss * ~ans_padding_mask).sum() / num_tgts
            # 
            # loss.backward()
            # optimizer.step()
            # 
            # out_idx = torch.argmax(out, dim=-1)
            # score = (out_idx == ans).reshape((-1))
            # score = (score * ~ans_padding_mask).sum() / num_tgts

            # epoch_tgts.append((~ans_padding_mask).sum().item())
            # epoch_scores.append(score.item())
            # epoch_losses.append(loss.item())
        
        # Hardcode logging epochs to every 2 (change as needed)
        if epoch % 2 == 0:
            ################################################## OLD TRANSFORMER CODE
            # train_score = np.average(epoch_scores, weights=epoch_tgts)
            # val_loss, val_score = get_val_loss_and_score(model, val_loader, loss_fn)
            # val_loss = get_val_loss_and_score(model, val_loader, loss_fn, device)
            # logging.info(f"Loss and score of epoch {epoch} - train: {train_loss} | val: {val_loss} | train: {train_score}") | val: {val_score}")

            val_loss, val_score = get_val_loss_and_score(model, val_loader, device, pad_token_id)
            train_loss = np.average(epoch_losses, weights=epoch_tgts)
            train_score = np.average(epoch_scores, weights=epoch_tgts)

            logging.info(f"Loss and score of epoch {epoch} - train: {train_loss} | val: {val_loss} | train: {train_score} | val: {val_score}")
            # logging.info(f"Preds: {preds} | labels: {labels}")
    
        if epoch % args.save_every == 0:
            save_model(model, optimizer, epoch, loss_fn, save_dir)
        
        sampler.shuffle_data()

######################################## Stats functions ########################################

# Counts parameters in the model and compares to the number of features
# per data point (1000 per sequence)
def count_parameters_and_data(model, num_rows, num_feats):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {trainable_params}")
    
    df_inp_count = num_rows * num_feats
    logging.info(f"Total learnable data points: {df_inp_count}")

# Gets num rows and num feats values. Removes target cols
# and the seq_ix/needs_pred cols from the calculation
def get_num_rows_feats(df):
    num_rows = df.shape[0]
    num_feats = df.shape[1]
    
    return num_rows, num_feats


# print(torch.version.cuda)
# print(f"Torch path: {os.path.dirname(torch.__file__)}")
if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    
    args_hash = get_arg_hash(args)
    print(f"args_hash: {args_hash}")
 
    ts_dir = datetime.now().strftime("%y%d%m_%H%M_%S")
    save_dir = MODELS_ROOT.joinpath(args.model_sub_root, ts_dir, args_hash)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(save_dir)
    logging.info(args)
     
    df = read_pickle(args.data_file)
    if args.model_type == "ByT5":
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        df = df_to_bytes(df, eos_token_id=tokenizer.eos_token_id)
    # num_rows, num_feats = get_num_rows_feats(df) # this output is used later
    
    idx_list = list(set(df.index.to_list()))
    train_idx, val_idx = train_test_split(
        idx_list,
        test_size=0.1,
        random_state=args.seed
    )

    ####!!! Print below for small datasets !!!####
    # print(f"Train Seqs: {train_idx}")
    ####!!! ||||||||||||||||||||||||||||||||| !!!####
    
    torch.manual_seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    device = torch.cuda.current_device()

    train_sampler, train_loader = get_sampler_and_loader(
        df,
        train_idx,
        device,
        gen, 
        pad_token_id=tokenizer.pad_token_id
    )
    _, val_loader = get_sampler_and_loader(
        df,
        val_idx,
        device,
        gen,
        pad_token_id=tokenizer.pad_token_id
    )

    # replace 20 with num_feats and 3 with num_rows later
    model = get_model(device)
    
    # count_parameters_and_data(model, 3, 20)
    train_model(
        model,
        train_sampler,
        train_loader,
        val_loader,
        save_dir,
        device,
        pad_token_id=tokenizer.pad_token_id,
    )

