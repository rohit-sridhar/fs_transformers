import sys
import torch
import random
import math
import copy

import numpy as np
import torch.nn as nn

from pathlib import Path
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
)

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

MAIN_START_IDX = 59
MAIN_END_IDX = 60
MAIN_PAD_IDX = 61

INPUT_DIM = 20
SUPP_OUTPUT_DIM = 30
MAIN_OUTPUT_DIM = 62

##### collate_fn
# modified for ByT5
def collate_seq(seqs, device=None, pad_token_id=0):
    all_data = []
    all_labels = []
    
    for seq in seqs:
        data = torch.tensor(seq[0])
        labels = torch.tensor(seq[1])
        
        all_data.append(data)
        all_labels.append(labels)
        
    all_data = pad_sequence(all_data, batch_first=True, padding_value=pad_token_id) # since pad token id is 0
    all_labels = pad_sequence(all_labels, batch_first=True, padding_value=pad_token_id)

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

##### Huggingface models
class T5ForConditionalGenerationProjection(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, input_dim):
        super().__init__(config)
        self.model_dim = config.d_model
        self.input_dim = input_dim
        self.pad_token_id = config.pad_token_id

        self.project_input = nn.Linear(input_dim, config.d_model)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def init_linear_projection(self):
        nn.init.xavier_uniform_(self.project_input.weight)
        nn.init.zeros_(self.project_input.bias)

    # @auto_docstring
    def forward(
        self,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            inputs_embeds = self.project_input(inputs_embeds)
            # if attention_mask is None:
            #     attention_mask = (inputs_embeds != self.pad_token_id).all(dim=-1)

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            # if decoder_attention_mask is None:
            #     decoder_attention_mask = (labels != self.pad_token_id)
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.config.scale_decoder_outputs:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            # loss_fct1 = nn.CrossEntropyLoss(ignore_index=-100)
            # loss_fct2 = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            # loss_fct3 = nn.CrossEntropyLoss(reduction='sum')
            # loss_fct4 = nn.CrossEntropyLoss(reduction='none')

            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # loss2 = loss_fct2(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # loss3 = loss_fct3(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # loss4 = loss_fct4(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # print(f"Loss Ignore Index -100: {loss1}")
            # print(f"Loss Ignore Index 0: {loss2}")
            # print(f"Loss Sum Reduce (No Pad): {loss3 / labels.view(-1).shape[0]}")
            # print(f"Loss Sum Reduce (Pad): {(loss4*pad_mask).sum() / pad_mask.sum()}")
            # print()

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

