import torch
import sys
import pandas as pd

from utils import df_to_bytes
from args import parse_args
from transformers import AutoTokenizer

def greedy_decode(
    model,
    seq_id,
    inputs_embeds,
    labels,
    tokenizer
):
    print(seq_id)
    
    # out_seq = labels[:,0:1]
    # while True:
    #     kwargs={"inputs_embeds": inputs_embeds}
    #     out = model.generate(
    #         **kwargs
    #     )
    #     print(out)

    #     out = model.forward(
    #         inputs_embeds=inputs_embeds,
    #         labels=out_seq,
    #     )
    #     out_seq = out.logits.argmax(dim=-1)
    #     
    #     next_char = out.logits[:,-1:,:].argmax(dim=-1)
    #     out_seq = torch.cat([out_seq, next_char], dim=-1)
    #     
    #     if next_char[0,0].item() == tokenizer.eos_token_id:
    #         break

    out = model.forward(
        inputs_embeds=inputs_embeds,
        labels=labels,
    )
    out_seq = out.logits.argmax(dim=-1)
    
    print(f"Pred: {out_seq.squeeze()}")
    print(f"Ans: {labels[0]}")
    print()
    decoded = tokenizer.decode(out_seq.squeeze().detach())
    labels_decoded = tokenizer.decode(labels[0])
    print(f"Pred: {decoded}")
    print(f"Ans: {labels_decoded}")
    print()

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_pickle("./data/data_supplemental_gen_drop-na_lininterp0_sd1248_rh_smp3.pkl.train")
    
    model_files = torch.load(args.chkpt_path, weights_only=False)
    model=model_files["model"][1].to("cpu")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    df = df_to_bytes(df, eos_token_id=tokenizer.eos_token_id)
    print(df)
    
    seq_ids = df.index.to_list()
    inputs_embeds0 = torch.tensor(df.all_landmarks.to_list()[0])[None,:]
    inputs_embeds1 = torch.tensor(df.all_landmarks.to_list()[1])[None,:]
    inputs_embeds2 = torch.tensor(df.all_landmarks.to_list()[2])[None,:]
    
    labels0 = torch.tensor(df.phrase.to_list()[0])[None,:]
    labels1 = torch.tensor(df.phrase.to_list()[1])[None,:]
    labels2 = torch.tensor(df.phrase.to_list()[2])[None,:]
    
    greedy_decode(model, seq_ids[0], inputs_embeds0, labels0, tokenizer)
    greedy_decode(model, seq_ids[1], inputs_embeds1, labels1, tokenizer)
    greedy_decode(model, seq_ids[2], inputs_embeds2, labels2, tokenizer)

    # print(seq_ids[1])
    # print(out1.argmax(dim=-1))
    # print(tgt1)
    
    # out2 = model.forward(inp2, tgt2[:,0:1])
    # print(seq_ids[2])
    # print(out2.argmax(dim=-1))
    # print(tgt2)

