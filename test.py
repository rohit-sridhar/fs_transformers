#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
import random
import evaluate
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import df_to_bytes, MODELS_ROOT
from args import parse_args
from pathlib import Path
from transformers import AutoTokenizer

global args

def get_output(
    out_seq,
    labels,
    tokenizer,
    print_str=False,
):
    if args.test_type in ["gen","grd"]:
        out_ids = out_seq.squeeze().detach()[1:-1]
    elif args.test_type == "fwd":
        out_ids = out_seq.squeeze().detach()[:-1]

    labels_ids = labels[0][:-1]

    out_decoded = tokenizer.decode(out_ids)
    labels_decoded = tokenizer.decode(labels_ids)

    if print_str:
        print(f"Pred: {out_ids}")
        print(f"Ref: {labels_ids}")
        print()
        print(f"Pred: {out_decoded}")
        print(f"Ref: {labels_decoded}")
        print()

    return out_decoded, labels_decoded

def greedy_decode(
    model,
    seq_id,
    inputs_embeds,
    labels,
    tokenizer,
    print_str=False,
):
    out_seq = labels[:,0:1]

    while True:
        out = model.forward(
            inputs_embeds=inputs_embeds,
            labels=out_seq,
        )
        
        next_char = out.logits[:,-1:,:].argmax(dim=-1)
        out_seq = torch.cat([out_seq, next_char], dim=-1)
        
        if next_char[0,0].item() == tokenizer.eos_token_id:
            break

    out_str, labels_str = get_output(out_seq, labels, tokenizer, print_str=print_str)
    return out_str, labels_str

def generate(
    model,
    seq_id,
    inputs_embeds,
    labels,
    tokenizer,
    print_str=False,
):
    inputs_embeds_proj = model.project_input(inputs_embeds)
    encoder_outputs = model.encoder(inputs_embeds=inputs_embeds_proj)
    kwargs={
        "encoder_outputs": encoder_outputs,
    }

    out_seq = model.generate(
        max_length=512,
        num_beams=5,
        **kwargs
    )

    out_str, labels_str = get_output(out_seq, labels, tokenizer, print_str=print_str)
    return out_str, labels_str

def sanity_check(
    model,
    seq_id,
    inputs_embeds,
    labels,
    tokenizer,
    print_str=False,
):
    out = model.forward(
        inputs_embeds=inputs_embeds,
        labels=labels,
    )

    out_seq = out.logits.argmax(dim=-1)
    out_str, labels_str = get_output(out_seq, labels, tokenizer, print_str=print_str)
    return out_str, labels_str

def evaluate_sequences(predictions, references):
    cer_metric = evaluate.load("cer")
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    print(f"Character Error Rate: {cer_score}")
    return cer_score

def make_results_json(pred_dict):
    def get_true_stem(p):
        p = Path(p)
        while p.suffix:
            p = Path(p.stem)
        return p.name
    
    data_file_name = get_true_stem(args.data_file)
    chkpt_name = get_true_stem(args.chkpt_path)
    pred_output_filename = Path("_".join([data_file_name, chkpt_name, f"n_{args.n}"]))
    pred_output_suffix = f"{args.data_file.suffixes[1]}.json"
    pred_output_file = Path(pred_output_filename).with_suffix(pred_output_suffix)

    pred_path_part_1 = args.chkpt_path.parts[-4]
    pred_path_part_2 = args.chkpt_path.parts[-3]
    pred_path_part_3 = args.chkpt_path.parts[-2]
    pred_output_dir = MODELS_ROOT/ pred_path_part_1 / pred_path_part_2 / pred_path_part_3
    pred_output_dir.mkdir(exist_ok=True, parents=True)

    pred_output_path = pred_output_dir / pred_output_file
    print(f"Results file: {pred_output_path}")
    with open(pred_output_path, "w") as f:
        json.dump(pred_dict, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_parquet(args.data_file)
    
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model_files = torch.load(args.chkpt_path, weights_only=False, map_location=device)
    model = model_files["model"][1]
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    df = df_to_bytes(df, eos_token_id=tokenizer.eos_token_id)
    
    seq_ids = df.index.to_list()
    sample_size = len(seq_ids) if args.n == 0 or args.n > len(seq_ids) else args.n
    random.shuffle(seq_ids)
    seq_ids = seq_ids[:sample_size]

    predictions = []
    references = []
    pred_dict = {}
    for i,seq_id in enumerate(tqdm(seq_ids, position=args.bar_position)):
        inputs_embeds = torch.tensor(np.vstack(df.loc[seq_id].all_landmarks))
        inputs_embeds = inputs_embeds[None,:].float().to(device)
        labels = torch.tensor(df.loc[seq_id].phrase)[None,:].to(device)
        
        if args.test_type == "grd":
            out_str, labels_str = greedy_decode(
                model,
                seq_id,
                inputs_embeds,
                labels,
                tokenizer,
                print_str=args.debug
            )
        elif args.test_type == "gen":
            out_str, labels_str = generate(
                model,
                seq_id,
                inputs_embeds,
                labels,
                tokenizer,
                print_str=args.debug
            )
        elif args.test_type == "fwd":
            out_str, labels_str = sanity_check(
                model,
                seq_id,
                inputs_embeds,
                labels,
                tokenizer,
                print_str=args.debug
            )

        predictions.append(out_str)
        references.append(labels_str)
        pred_dict[seq_id] = {
            'pred': out_str,
            'ref': labels_str,
            "correct": (out_str == labels_str)
        }

    cer_score = evaluate_sequences(predictions, references)
    pred_dict[-1] = {'final_cer': cer_score}
    if not args.debug:
        make_results_json(pred_dict)

