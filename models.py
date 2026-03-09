import sys
import torch
import random
import math

import numpy as np
import torch.nn as nn

from pathlib import Path

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import tanh, leaky_relu
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
)

START_CHAR = "<"
END_CHAR = ">"
PAD_CHAR = "P"
DATA_PAD = -1.0

SUPP_START_IDX = 27
SUPP_END_IDX = 27
SUPP_PAD_IDX = 29

SUPP_INPUT_DIM = 20
SUPP_OUTPUT_DIM = 30

##### collate_fn
def collate_seq(seqs, generator=None, device=None):
    all_data = []
    all_labels = []
    
    for seq in seqs:
        data = torch.tensor(seq[0])
        labels = torch.tensor(seq[1])
        
        all_data.append(data)
        all_labels.append(labels)
        
    all_data = pad_sequence(all_data, batch_first=True, padding_value=DATA_PAD)
    all_labels = pad_sequence(all_labels, batch_first=True, padding_value=SUPP_PAD_IDX)

    all_data = all_data.to(device)
    all_labels = all_labels.to(device)
    
    return all_data, all_labels

##### Dataset and Sampler
class SimpleRandomSampler(Sampler):
    def __init__(self, data_source, generator=None):
        self.generator = generator
        self.n = len(data_source)
        self.data_source = data_source

        self.shuffle_data()
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        yield from self.data_source

    def shuffle_data(self):
        shuffle_idx = torch.randperm(self.n, generator=self.generator)
        self.data_source = torch.tensor(self.data_source)[shuffle_idx].tolist()

class IdxDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a data sample and its label at index idx
        sample = self.data.loc[idx].to_list()[0]
        label = self.labels.loc[idx].to_list()[0]

        return sample, label

##### Model Classes
##### Transformer Model (and associated classes)
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch_size, sequence length, embed dim]
            output: [batch_size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,
    ):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        self.src_projection = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(output_dim, d_model)
        
        self.src_positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_len=500)
        self.tgt_positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_len=500)
        
        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            dim_feedforward=d_model, # may change to be different later
            batch_first=True,
            dropout=0.0,
        )
        
        self.final_layer_1 = nn.Linear(d_model, output_dim)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        src = self.src_norm(self.src_projection(src) * math.sqrt(self.d_model))
        src = self.src_positional_encoding(src)

        tgt = self.tgt_norm(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        tgt = self.tgt_positional_encoding(tgt)
        
        out = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            src_is_causal=(src_mask is not None),
            tgt_is_causal=(tgt_mask is not None),
        )

        out = self.final_layer_1(out)
        
        return out

