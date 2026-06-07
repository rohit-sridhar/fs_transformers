#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from torch.optim import AdamW, RMSprop, Adafactor
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from transformers import AutoTokenizer
from models import T5ForConditionalGenerationProjection

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Adjust path to import utils from parent directory
TRAIN_FILE_ROOT = Path(__file__).resolve().parent
sys.path.append(f"{TRAIN_FILE_ROOT}/..")

from utils import (
    read_parquet,
    df_to_bytes,
    MODELS_ROOT,
    BYT5_NUM_SPECIAL_TOKENS,
    GB_BYTES,
)
from args import parse_args
from models import (
    DATA_PAD,
    INPUT_DIM,
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

# returns optimizer based on optimizer type
def get_optimizer(model):
    if args.optimizer == "Adafactor":
        return Adafactor(model.parameters())
    elif args.optimizer == "AdamW":
        return AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    elif args.optimizer == "RMSprop":
        return RMSprop(model.parameters(), lr=args.learning_rate, centered=True)

# Get model based on args model type
def get_model(device):
    if args.model_type == "ByT5":
        model_args = [INPUT_DIM]
        model = T5ForConditionalGenerationProjection.from_pretrained(
            "google/byt5-small",
            *model_args,
        )
        model.init_linear_projection()

    model.to(device)
    model.train()

    return model

######################################## Training function and friends ########################################

# save model and various params and data about current state
# def save_model(model, optimizer, preprocessor, epoch, loss_fn, save_dir):
# def save_model(model, optimizer, epoch, loss_fn, save_dir):
def save_model(model, optimizer, epoch, save_dir):
    chkpt_name = save_dir.joinpath(f"{epoch}_of_{args.num_epochs}").with_suffix(".chkpt")
    # loss_tuple = (loss_fn.__class__.__name__, loss_fn)

    torch.save({
        "epoch": epoch,
    #     "loss_fn": loss_tuple,
        "model": (model.__class__.__name__, model),
        "optimizer": (optimizer.__class__.__name__, optimizer),
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

    model.eval()
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

    ####!!! Keep in eval for overfitting !!!####
    if not args.overfit:
        model.train()
    ####!!! ||||||||||||||||||||||||||||||||| !!!####

    val_loss = np.average(total_losses, weights=total_tgts, axis=0)
    val_score = np.average(total_scores, weights=total_tgts, axis=0)

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
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=5e-4)
    # scheduler = StepLR(optimizer, 5, gamma=0.1)
    
    ####!!! Switch to eval below for overfitting !!!####
    if args.overfit:
        model.eval()
    ####!!! ||||||||||||||||||||||||||||||||| !!!####

    len_loader = len(train_loader)
    for epoch in tqdm(
        list(range(1, args.num_epochs + 1)),
        desc="epochs",
        position=args.bar_position,
        disable=args.std_out,
    ):
        epoch_tgts = []
        epoch_losses = []
        epoch_scores = []
         
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
            optimizer.zero_grad()

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
            
            epoch_losses.append(loss.item())
            epoch_tgts.append(tgts.sum().item())
            epoch_scores.append(score.sum().item() / tgts.sum().item())

        logging.info(f"Learning rate {scheduler.get_last_lr()}")
        scheduler.step()
        
        # Hardcode logging epochs to every 2 (change as needed)
        if epoch % 2 == 0:
            val_loss, val_score = get_val_loss_and_score(model, val_loader, device, pad_token_id)
            train_loss = np.average(epoch_losses, weights=epoch_tgts)
            train_score = np.average(epoch_scores, weights=epoch_tgts)

            current_mem = torch.cuda.memory_allocated() / GB_BYTES
            max_mem = torch.cuda.max_memory_allocated() / GB_BYTES

            logging.info(f"Loss and score of epoch {epoch} - train: {train_loss:.2f} | val: {val_loss:.2f} | train: {train_score:.2f} | val: {val_score:.2f}")
            logging.info(f"Memory usage: current {current_mem:.2f} | max: {max_mem:.2f}")
            # logging.info(f"Preds: {preds} | labels: {labels}")
    
        if epoch % args.save_every == 0:
            # save_model(model, optimizer, epoch, loss_fn, save_dir)
            save_model(model, optimizer, epoch, save_dir)
        
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
    logging.info(f"args_hash: {args_hash}")
     
    df = read_parquet(args.data_file)
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

    ####!!! Log below for overfitting !!!####
    if args.overfit:
        logging.info(f"Train Seqs: {train_idx}")
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

    model = get_model(device)
    memory_footprint = model.get_memory_footprint() / GB_BYTES
    logging.info(f"model memory: {memory_footprint:.2f}")
    train_model(
        model,
        train_sampler,
        train_loader,
        val_loader,
        save_dir,
        device,
        pad_token_id=tokenizer.pad_token_id,
    )

