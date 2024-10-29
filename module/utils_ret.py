import json
import os
import pickle

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from datasets import load_from_disk, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def load_contexts(context_path):
    """
    Load context data from file.
    """
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Loaded {len(contexts)} unique contexts.")
    return contexts


def build_dense_embedding(p_encoder, q_encoder, contexts, tokenizer, training_args):
    """
    Passage encoder로부터 문서 임베딩을 계산하고 반환합니다.
    """
    p_embedding = []
    p_encoder.eval()
    dense_embedding_path = os.path.join("/data/ephemeral/data/", "dense_embedding.bin")
    q_encoder_path = os.path.join("/data/ephemeral/data/", "q_encoder.bin")
    
    for i in tqdm(range(0, len(contexts), training_args.per_device_train_batch_size), desc="Building dense embeddings"):
        batch_contexts = contexts[i:i + training_args.per_device_train_batch_size]  # 배치 크기로 슬라이싱
        batch_inputs = tokenizer(
            batch_contexts,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        ).to("cuda")

        with torch.no_grad():
            batch_embeddings = p_encoder(**batch_inputs)  # 배치 단위로 인코딩
        
        # numpy 배열로 변환 후 리스트에 추가
        p_embedding.extend(batch_embeddings.cpu().numpy())

    with open(dense_embedding_path, "wb") as file:
        pickle.dump(p_embedding, file)
    torch.save(q_encoder, q_encoder_path)
    print(f"Dense embedding and q_encoder saved to {dense_embedding_path} and {q_encoder_path}.")


def prepare_in_batch_negative(config, contexts, tokenizer, model_args, data_args, training_args) -> DataLoader:
    """
    Prepare in-batch negative samples for training.
    """

    dataset1 = load_from_disk('/data/ephemeral/data/train_dataset')["train"]
    tokenized_contexts = [tokenizer.convert_ids_to_tokens(tokenizer(context)['input_ids']) for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)

    q_input_ids, q_attention_mask, q_token_type_ids = [], [], []
    p_input_ids, p_attention_mask, p_token_type_ids = [], [], []
    bert = False
    if "bert" == model_args.dense_model_name_or_path.split("/")[-1][:4]:
        bert = True

    for i, data in enumerate(tqdm(dataset1, desc="Processing dataset")):
        q_inputs = tokenizer(
            data["question"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        q_input_ids.append(q_inputs["input_ids"].tolist())
        q_attention_mask.append(q_inputs["attention_mask"].tolist())

        neg_contexts = sample_negatives(
            tokenizer.convert_ids_to_tokens(tokenizer(data['question'])['input_ids']),
            data_args.num_neg, 
            data['context'], 
            contexts,
            bm25,
            data_args.top_k_retrieval
        )

        p_inputs = tokenizer(
            [data['context']] + neg_contexts,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        p_input_ids.append(p_inputs["input_ids"].tolist())
        p_attention_mask.append(p_inputs["attention_mask"].tolist())
        if bert:
            q_token_type_ids.append(q_inputs["token_type_ids"].tolist())
            p_token_type_ids.append(p_inputs["token_type_ids"].tolist())

    
    size = p_inputs["input_ids"].size(-1)
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

    return DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, drop_last=True)


def sample_negatives(question, num_neg, context, contexts, bm25, k):
    """
    Sample negative contexts using BM25.
    """
    
    scores = bm25.get_scores(question)
    sorted_result = np.argsort(scores)[::-1]
    doc_score = scores[sorted_result].tolist()[:k]
    doc_indices = sorted_result.tolist()[:k]
    neg_contexts = []
    for idx in doc_indices:
        if contexts[idx][:10] != context[:10]:
            neg_contexts.append(contexts[idx])
        if len(neg_contexts) == num_neg:
            break
    if len(neg_contexts)!=num_neg:
        print('fail')
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
            torch.tensor(p_input_ids),
            torch.tensor(p_attention_mask),
            torch.tensor(q_token_type_ids),
            torch.tensor(p_token_type_ids),
        )

    else:
        return TensorDataset(
            torch.tensor(q_input_ids),
            torch.tensor(q_attention_mask),
            torch.tensor(p_input_ids),
            torch.tensor(p_attention_mask),
        )

    