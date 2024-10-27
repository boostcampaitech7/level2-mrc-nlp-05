import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


def set_columns(dataset):
    dataset = pd.DataFrame({"context": dataset["context"], "query": dataset["question"], "title": dataset["title"]})

    return dataset


def load_tokenizer(MODEL_NAME):
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_colbert(dataset, tokenizer, corpus):

    # for inference
    if corpus == "query":
        preprocessed_data = []
        for query in dataset:
            preprocessed_data.append("[Q] " + query)

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        return tokenized_query

    elif corpus == "doc":
        preprocessed_data = "[D] " + dataset
        tokenized_context = tokenizer(
            preprocessed_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return tokenized_context

    # for train
    else:
        preprocessed_query = []
        preprocessed_context = []
        for query, context in zip(dataset["query"], dataset["context"]):
            preprocessed_context.append("[D] " + context)
            preprocessed_query.append("[Q] " + query)
        tokenized_query = tokenizer(
            preprocessed_query, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context, tokenized_query
