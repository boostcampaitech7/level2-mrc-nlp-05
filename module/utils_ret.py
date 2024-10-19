import json
import os
import pickle

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel


def load_contexts(context_path):
    """
    Load context data from file.
    """
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Loaded {len(contexts)} unique contexts.")
    return contexts


def build_dense_embedding(p_encoder, q_encoder, contexts, tokenizer):
    """
    Passage encoder로부터 문서 임베딩을 계산하고 반환합니다.
    """
    p_embedding = []
    p_encoder.eval()
    dense_embedding_path = os.path.join("/data/ephemeral/data/", 
    "dense_embedding.bin")
    q_encoder_path = os.path.join("/data/ephemeral/data/", "q_encoder.bin")

    for passage in tqdm(contexts, desc="Building dense embeddings"):
        passage_inputs = tokenizer(passage,
        return_tensors="pt",
        truncation=True,
        padding="max_length").to("cuda")

        with torch.no_grad():
            p_emb = p_encoder(**passage_inputs)  # pooled_output 사용
        p_embedding.append(p_emb.cpu().numpy())

    with open(dense_embedding_path, "wb") as file:
        pickle.dump(p_embedding, file)
    torch.save(q_encoder, q_encoder_path)
    print(f"Dense embedding and q_encoder saved to {dense_embedding_path} and {q_encoder_path}.")


def prepare_in_batch_negative(config, contexts, tokenizer, model_args, data_args, training_args) -> DataLoader:
    """
    Prepare in-batch negative samples for training.
    """
    dataset = load_from_disk(data_args.train_dataset_name)["train"]
    tokenized_contexts = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    q_input_ids, q_attention_mask, q_token_type_ids = [], [], []
    p_input_ids, p_attention_mask, p_token_type_ids = [], [], []
    bert = False
    if "bert" == model_args.dense_model_name_or_path.split("/")[-1][:4]:
        bert = True

    for i, context in enumerate(dataset["context"]):
        q_encodings = tokenizer(dataset["question"][i],
        truncation=True,
        padding="max_length",
        return_tensors="pt")
        q_input_ids.append(q_encodings["input_ids"].tolist())
        q_attention_mask.append(q_encodings["attention_mask"].tolist())

        neg_contexts = sample_negatives(dataset["question"][i],
            data_args.num_neg, 
            context, 
            contexts,
            bm25,
        )
        p_encodings = tokenizer([context] + neg_contexts,
        truncation=True,
        padding="max_length",
        return_tensors="pt")
        p_input_ids.append(p_encodings["input_ids"].tolist())
        p_attention_mask.append(p_encodings["attention_mask"].tolist())
        if bert:
            q_token_type_ids.append(q_encodings["token_type_ids"].tolist())
            p_token_type_ids.append(p_encodings["token_type_ids"].tolist())

    size = q_encodings["input_ids"].size(-1)
    dataset = create_tensor_dataset(
        bert,
        data_args.num_neg,
        size,
        q_input_ids,
        q_attention_mask,
        q_token_type_ids,
        p_input_ids,
        p_attention_mask,
        p_token_type_ids,
    )

    return DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)


def sample_negatives(question, num_neg, context, contexts, bm25):
    """
    Sample negative contexts using BM25.
    """
    tokenized_question = question.split()
    scores = bm25.get_scores(tokenized_question)
    ranked_idxs = np.argsort(scores)[::-1]  # 스코어가 높은 순으로 정렬
    neg_contexts = []

    for idx in ranked_idxs:
        if contexts[idx] != context:
            neg_contexts.append(contexts[idx])
        if len(neg_contexts) == num_neg:
            break
    return neg_contexts

def create_tensor_dataset(
    bert,
    num_neg,
    size,
    q_input_ids,
    q_attention_mask,
    q_token_type_ids,
    p_input_ids,
    p_attention_mask,
    p_token_type_ids,
):
    """
    Create tensor dataset for DataLoader.
    """
    if bert:
        return TensorDataset(
            torch.tensor(q_input_ids),
            torch.tensor(q_attention_mask),
            torch.tensor(q_token_type_ids),
            torch.tensor(p_input_ids).view(-1, num_neg + 1, size),
            torch.tensor(p_attention_mask).view(-1, num_neg + 1, size),
            torch.tensor(p_token_type_ids).view(-1, num_neg + 1, size),
        )

    else:
        return TensorDataset(
            torch.tensor(q_input_ids),
            torch.tensor(q_attention_mask),
            torch.tensor(p_input_ids).view(-1, num_neg + 1, size),
            torch.tensor(p_attention_mask).view(-1, num_neg + 1, size),
        )
