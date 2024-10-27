from typing import Callable, Dict, List, NoReturn, Tuple

import json
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from model import *
from tokenizer import *
from transformers import AutoTokenizer


def run_colbert_retrieval(datasets):
    test_dataset = datasets["validation"].flatten_indices().to_pandas()
    MODEL_NAME = "klue/bert-base"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    ret_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ret_tokenizer.add_special_tokens(special_tokens)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(ret_tokenizer.vocab_size + 2)

    model.to(device)

    model.load_state_dict(
        torch.load(
            "/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_6.pth"
        )
    )

    print("opening wiki passage...")
    with open("/data/ephemeral/home/LYJ/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(test_dataset["question"])
    mrc_ids = test_dataset["id"]
    length = len(test_dataset)

    batched_p_embs = []
    with torch.no_grad():
        model.eval

        q_seqs_val = tokenize_colbert(query, ret_tokenizer, corpus="query").to("cuda")
        q_emb = model.query(**q_seqs_val).to("cpu")
        print(q_emb.size())

        print(q_emb.size())

        print("Start passage embedding......")
        p_embs = []
        for step, p in enumerate(tqdm(context)):
            p = tokenize_colbert(p, ret_tokenizer, corpus="doc").to("cuda")
            p_emb = model.doc(**p).to("cpu").numpy()
            p_embs.append(p_emb)
            if (step + 1) % 200 == 0:
                batched_p_embs.append(p_embs)
                p_embs = []
        batched_p_embs.append(p_embs)

    dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    torch.save(rank, "/opt/ml/input/code/inferecne_colbert_rank.pth")
    print(rank.size())

    k = 100
    passages = []

    for idx in range(length):
        passage = ""
        for i in range(k):
            passage += context[rank[idx][i]]
            passage += " "
        passages.append(passage)

    df = pd.DataFrame({"question": query, "id": mrc_ids, "context": passages})
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )

    complete_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return complete_datasets


def main():
    epoch = 6
    MODEL_NAME = "klue/bert-base"
    data_path = "../../../data/train_dataset"
    datasets = load_from_disk(data_path)
    val_dataset = pd.DataFrame(datasets["validation"])
    val_dataset = val_dataset.reset_index(drop=True)
    val_dataset = set_columns(val_dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(
        torch.load(
            f"/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_{epoch}.pth"
        )
    )

    print("opening wiki passage...")
    with open("/data/ephemeral/home/LYJ/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(val_dataset["query"])
    ground_truth = list(val_dataset["context"])

    batched_p_embs = []
    with torch.no_grad():

        model.eval()

        # 토크나이저
        q_seqs_val = tokenize_colbert(query, tokenizer, corpus="query").to("cuda")
        q_emb = model.query(**q_seqs_val).to("cpu")

        print(q_emb.size())

        print("Start passage embedding......")
        p_embs = []
        for step, p in enumerate(tqdm(context)):
            p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
            p_emb = model.doc(**p).to("cpu").numpy()
            p_embs.append(p_emb)
            if (step + 1) % 200 == 0:
                batched_p_embs.append(p_embs)
                p_embs = []
        batched_p_embs.append(p_embs)

    print("passage tokenizing done!!!!")
    length = len(val_dataset["context"])

    dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)

    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    print(rank.size())
    torch.save(rank, f"./colbert/rank_epoch{epoch}.pth")

    k = 100
    score = 0

    for idx in range(length):
        print(dot_prod_scores[idx])
        print(rank[idx])
        print()
        for i in range(k):
            if ground_truth[idx] == context[rank[idx][i]]:
                score += 1

    print(f"{score} over {length} context found!!")
    print(f"final score is {score/length}")


if __name__ == "__main__":
    main()
