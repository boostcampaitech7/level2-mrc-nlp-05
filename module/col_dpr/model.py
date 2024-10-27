import argparse
import random
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    BertTokenizerFast,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class ColbertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ColbertModel, self).__init__(config)

        # BertModel 사용

        self.dim = 128
        self.bert = BertModel(config)
        self.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        mask_symbol = [tokenizer.mask_token_id]
        mask_symbol.extend([tokenizer.encode(symbol, add_special_tokens=False)[0] for symbol in string.punctuation])
        self.register_buffer("mask_buffer", torch.tensor(mask_symbol))

    def forward(self, p_inputs, q_inputs, n_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**p_inputs)
        if n_inputs:  # negative input이 있을 경우
            N = self.doc(**n_inputs)
            return self.get_score(Q, D, N)

        return self.get_score(Q, D)

    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        D = self.linear(D)

        puntuation_mask = self.punctuation_mask(input_ids).unsqueeze(2)
        D = D * puntuation_mask
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def punctuation_mask(self, input_ids):
        mask = (input_ids.unsqueeze(-1) == self.mask_buffer).any(dim=-1)
        mask = (~mask).float()
        return mask

    def get_score(self, Q, D, eval=False, N=None):

        if eval:
            final_score = torch.tensor([])
            for D_batch in tqdm(D):
                D_batch = torch.Tensor(np.array(D_batch)).squeeze()  # (p_batch_size, p_sequence_length, hidden_size)
                p_sequence_output = D_batch.transpose(1, 2)  # (p_batch_size, hidden_size, p_sequence_length)
                q_sequence_output = Q.view(
                    Q.shape[0], 1, -1, self.dim
                )  # (q_batch_size, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(
                    q_sequence_output, p_sequence_output.to(Q.device)
                )  # (q_batch_size, p_batch_size, q_sequence_length, p_sequence_length)
                max_dot_prod_score = torch.max(dot_prod, dim=3).values
                # (q_batch_size, p_batch_size, q_sequnce_length)
                score = torch.sum(max_dot_prod_score, dim=2).to(final_score.device)  # (q_batch_size, p_batch_size)
                final_score = torch.cat([final_score, score], dim=1)
            print(final_score.size())  # (q_batch_size, num_whole_p)
            return final_score

        else:
            p_sequence_output = D.transpose(1, 2)  # (batch_size, hidden_size, p_sequence_length)
            if N:
                # N = (batch_size,n_sequence_length,hidden_size,negative_size)
                # change to (negative_size, batch_size, hidden_size, n_sequence_length)
                n_sequence_output = N.permute(3, 0, 2, 1)
                for i in range(n_sequence_output.size(0)):
                    p_sequence_output = torch.cat([p_sequence_output, n_sequence_output[i]], dim=0)
                # (batch_size + negative_size, hidden_size, p_sequence_length)

            q_sequence_output = Q.view(Q.shape[0], 1, -1, self.dim)  # (batch_size, 1, q_sequence_length, hidden_size)
            dot_prod = torch.matmul(
                q_sequence_output, p_sequence_output
            )  # (batch_size, batch_size + hard_negative_size, q_sequence_length, p_seqence_length)
            max_dot_prod_score = torch.max(dot_prod, dim=3)[
                0
            ]  # (batch_size, batch_size + hard_negative_size, q_sequnce_length)
            final_score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size, batch_size + hard_negative_size)

            return final_score
