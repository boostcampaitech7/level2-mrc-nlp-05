import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

from .arguments import ModelArguments, DataTrainingArguments
from .model_ret import Encoder

import wandb
from torch.optim import AdamW  # AdamW를 torch.optim에서 가져옴
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW  # AdamW를 torch.optim에서 가져옴
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup,
    get_linear_schedule_with_warmup, EarlyStoppingCallback, TrainingArguments
)  
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm, trange

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정


def train(training_dataloader, model_args, data_args, training_args):
        """
        Train passage and query encoders.
        """
        wandb.init(project="dense_retrieval_project")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_late,
            eps=training_args.apsilon
        )
        t_total = len(train_dataloader) // 1 * training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=t_total
        )

        best_loss = float('inf')  
        global_step = 0
        early_stopping_patience = 3
        patience_counter = 0
        best_q_encoder_path = os.path.join('/data/ephemeral/data/', "q_encoder_best.bin")

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache() 
        train_iterator = trange(training_args.num_train_epochs, desc="Epoch")

        for _ in train_iterator:
            epoch_loss = 0.0
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                        
                    p_encoder.train()
                    q_encoder.train()

                    targets = torch.zeros(training_args.per_device_train_batch_size).long() # positive example은 전부 첫 번째
                    targets = targets.to('cuda')

                    q_inputs = {
                        "input_ids": batch[0].squeeze().to('cuda'),
                        "attention_mask": batch[1].squeeze().to('cuda'),
                    }
                    
                    p_inputs = {
                        "input_ids": batch[2].view(training_args.per_device_train_batch_size * (data_args.num_neg + 1), -1).to('cuda'),
                        "attention_mask": batch[3].view(training_args.per_device_train_batch_size * (data_args.num_neg + 1), -1).to('cuda'),
 }
                    if 'bert' == model_args.model_name_or_path.split('/')[-1][:4]:
                        q_inputs["token_type_ids"] = batch[4].squeeze().to('cuda')
                        p_inputs["token_type_ids"] = batch[5].view(training_args.per_device_train_batch_size * (data_args.num_neg + 1), -1).to('cuda')
                   
                    del batch
                    torch.cuda.empty_cache()

                    p_outputs = p_encoder(**p_inputs).pooler_output
                    q_outputs = (**q_inputs).pooler_output

                    p_outputs = p_outputs.view(training_args.per_device_train_batch_size, data_args.num_neg + 1, -1)
                    q_outputs = q_outputs.view(training_args.per_device_train_batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(training_args.per_device_train_batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    wandb.log({"loss": loss.item(), "global_step": global_step})

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    q_encoder.zero_grad()
                    p_encoder.zero_grad()

                    global_step += 1
                    epoch_loss += loss.item()

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs
                epoch_loss /= len(train_dataloader)
                wandb.log({"epoch_loss": epoch_loss, "epoch": _ + 1})
        return p_encoder, q_encoder


def ret_train(cfg: DictConfig):
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))
    contexts = load_contexts(os.path.join(data_args.data_path, 'wikipedia_documents.json'))
    toknizer = AutoTokenizer.from_pretrained(model_args.dense_model_name_or_path, use_fast=True)
    training_dataloader = prepare_in_batch_negative(contexts, tokenizer, model_args, data_args, training_args)
    p_encoder = Encoder(model_args.model_name_or_path)
    q_encoder = Encoder(model_args.model_name_or_path)
    p_encoder, q_encoder = train(training_dataloader, model_args, data_args, training_args)
    build_dense_embedding(p_encoder, q_encoder, contexts, tokenizer)