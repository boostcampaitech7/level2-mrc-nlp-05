import json
import os
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from model import *
from tokenizer import *
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

BM25_USED = False


def main():

    set_seed(42)

    batch_size = 8
    data_path = "../../../data/train_dataset"

    lr = 2e-6
    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_steps=900,
    )

    MODEL_NAME = "klue/bert-base"
    tokenizer = load_tokenizer(MODEL_NAME)

    datasets = load_from_disk(data_path)
    train_dataset = pd.DataFrame(datasets["train"])
    train_dataset = train_dataset.reset_index(drop=True)
    train_dataset = set_columns(train_dataset)

    # 토크나이저
    train_context, train_query = tokenize_colbert(train_dataset, tokenizer, corpus="both")

    train_dataset = TensorDataset(
        train_context["input_ids"],
        train_context["attention_mask"],
        train_context["token_type_ids"],
        train_query["input_ids"],
        train_query["attention_mask"],
        train_query["token_type_ids"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    model.to(device)

    trained_model = train(args, train_dataset, model)
    torch.save(trained_model.state_dict(), "./colbert/colbert.pth")


def train(args, dataset, model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer 세팅
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training 시작
    global_step = 0

    model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss = 0
        steps = 0

        for step, batch in enumerate(epoch_iterator):
            steps += 1
            model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}

            q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

            # outputs with similiarity score
            outputs = model(p_inputs, q_inputs)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()
        final_loss = total_loss / len(dataset)
        print("total_loss :", final_loss)
        torch.save(model.state_dict(), f"./colbert/compare_colbert_pretrain_v3_finetune_{epoch+1}.pth")

    return model


if __name__ == "__main__":

    main()
