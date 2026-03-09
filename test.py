import torch
import pandas as pd

from args import parse_args

def greedy_decode(model, seq_id, inp, tgt):
    print(seq_id)
    print(tgt)
    out_seq = tgt[:,0:1]
    while True:
        out = model.forward(inp, out_seq)
        next_char = out[:,-1:,:].argmax(dim=-1)
        out_seq = torch.cat([out_seq, next_char], dim=-1)

        if next_char[0,0].item() == 28:
            break

    print(out_seq)
    print()

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_pickle("./datasets/data_supplemental_gen_drop-na_lininterp0_sd1248_rh_smp3.pkl.train")
    
    model_files = torch.load(args.chkpt_path, weights_only=False)
    model=model_files["model"][1].to("cpu")
    model.eval()
    
    seq_ids = df.index.to_list()
    inp0 = torch.tensor(df.all_landmarks.to_list()[0])[None,:]
    inp1 = torch.tensor(df.all_landmarks.to_list()[1])[None,:]
    inp2 = torch.tensor(df.all_landmarks.to_list()[2])[None,:]
    
    tgt0 = torch.tensor(df.phrase.to_list()[0])[None,:]
    tgt1 = torch.tensor(df.phrase.to_list()[1])[None,:]
    tgt2 = torch.tensor(df.phrase.to_list()[2])[None,:]

    # tgt0[:,-1] = tgt0[:,0]
    # tgt1[:,-1] = tgt1[:,0]
    # tgt2[:,-1] = tgt2[:,0]
    
    greedy_decode(model, seq_ids[1], inp1, tgt1)
    greedy_decode(model, seq_ids[2], inp2, tgt2)
    # print(seq_ids[1])
    # print(out1.argmax(dim=-1))
    # print(tgt1)

    # out2 = model.forward(inp2, tgt2[:,0:1])
    # print(seq_ids[2])
    # print(out2.argmax(dim=-1))
    # print(tgt2)
