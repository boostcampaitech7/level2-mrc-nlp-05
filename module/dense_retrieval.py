from typing import List, NoReturn, Optional, Tuple, Union

import json
import os
import pickle
import random
import time
from contextlib import contextmanager

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, concatenate_datasets, load_from_disk
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW  # AdamW를 torch.optim에서 가져옴
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from .arguments import DataTrainingArguments, ModelArguments
from .utils_ret import load_contexts

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(self, model_args, data_args, training_args, context_path: Optional[str] = "wikipedia_documents.json"):
        """
        DenseRetrieval 클래스 초기화
        학습과 추론에 필요한 객체들을 초기화하며, in-batch negative 데이터를 준비합니다.
        """
        config = AutoConfig.from_pretrained(model_args.dense_model_name_or_path)
        self.data_path = data_args.train_dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.dense_model_name_or_path, use_fast=True)
        self.q_encoder = None
        self.p_embedding = None
        self.contexts = load_contexts(os.path.join(data_args.data_path, context_path))

    def get_dense_embedding(self) -> NoReturn:
        """
        Passage Embedding을 만들고 pickle로 저장하거나, 저장된 파일을 불러옵니다.
        q_encoder는 return 하여 다른 곳에서 사용할 수 있게 처리합니다.
        """
        dense_embedding_path = os.path.join(self.data_path, "dense_embedding.bin")
        q_encoder_path = os.path.join(self.data_path, "q_encoder.bin")
        p_encoder_path = os.path.join(self.data_path, "p_encoder.bin")

        if os.path.isfile(dense_embedding_path) and os.path.isfile(q_encoder_path):
            print("Loading saved dense embeddings and q_encoder...")
            with open(dense_embedding_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(q_encoder_path, "rb") as file:
                self.q_encoder = torch.load(file).to("cuda")
            self.q_encoder.eval()
            print("Loaded dense embedding and q_encoder from files.")
        else:
            print("You have to train encoders first.")
            exit()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 20
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
        """
        Query와 가장 유사한 K개의 Passage를 찾는 함수입니다.
        유사도 스코어와 인덱스를 반환합니다.
        Args:
            query_or_dataset (Union[str, Dataset]): 검색할 질문 또는 질문이 포함된 데이터셋
            topk (Optional[int]): 반환할 상위 문서 개수
        Returns:
            Tuple[List[float], List[str]] 또는 pd.DataFrame: 유사도 스코어와 해당 문서 리스트 또는 DataFrame
        """
        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야 합니다."

        if isinstance(query_or_dataset, str):
            query_inputs = self.tokenizer(
                query_or_dataset, return_tensors="pt", truncation=True, padding="max_length"
            ).to("cuda")
            query_vec = self.q_encoder(**query_inputs)
            passage_vecs = torch.tensor(self.p_embedding).to("cuda")

            with torch.no_grad():
                sim_scores = torch.matmul(query_vec, passage_vecs.T)

            topk_scores, topk_indices = torch.topk(sim_scores, k=topk)
            topk_passages = [self.contexts[idx] for idx in topk_indices.cpu().tolist()]

            del self.p_embedding
            del self.q_encoder

            return topk_scores.cpu().tolist(), topk_passages

        elif isinstance(query_or_dataset, Dataset):
            total = []
            queries = query_or_dataset["question"]
            query_inputs = self.tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(
                "cuda"
            )
            with torch.no_grad():
                q_outputs = self.q_encoder(**query_inputs)

            passage_vecs = torch.tensor(self.p_embedding).view(-1, self.q_encoder.config.hidden_size).to("cuda")

            with torch.no_grad():
                sim_scores = torch.matmul(q_outputs, passage_vecs.T)

            topk_scores, topk_indices = torch.topk(sim_scores, k=topk)
            for idx, query in enumerate(queries):
                topk_passages = [self.contexts[i] for i in topk_indices[idx].cpu().tolist()]

                joined_passages = " ".join(topk_passages)

                if "context" in query_or_dataset.column_names and "answers" in query_or_dataset.column_names:
                    tmp = {
                        "answers": query_or_dataset["answers"][idx],
                        "context": joined_passages,  # 상위 k개의 문서를 하나의 문자열로 결합하여 사용
                        "id": query_or_dataset["id"][idx],
                        "question": query,
                    }
                else:
                    tmp = {
                        "context": joined_passages,  # 상위 k개의 문서를 하나의 문자열로 결합하여 사용
                        "id": query_or_dataset["id"][idx],
                        "question": query,
                    }

                total.append(tmp)

            del self.p_embedding
            del self.q_encoder

            return pd.DataFrame(total)
