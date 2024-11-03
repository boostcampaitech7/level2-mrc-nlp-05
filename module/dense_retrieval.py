from typing import List, NoReturn, Optional, Tuple, Union

import json
import os
import pickle
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, BertTokenizerFast

from module.col_dpr.model import ColbertModel
from module.sparse_retrieval import SparseRetrieval


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def load_colbert_model(model_path: str, device: str) -> ColbertModel:
    """
    Args:
        model_path: 저장된 모델 경로
        device: 사용할 디바이스

    Returns:
        로드된 ColBERT 모델
    """
    # 모델 초기화
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("klue/bert-base")
    model = ColbertModel(config)
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    # state dict 로드
    state_dict = torch.load(model_path, map_location=device)

    # state dict 적용
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    return model.to(device)


class ColBERTRetrieval:
    def __init__(
        self,
        model_path: str,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,
    ) -> NoReturn:

        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size

        # Load contexts
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # Load model and tokenizer
        self.model = load_colbert_model(model_path, self.device)
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        self.max_length = 512

        # Initialize embeddings
        self.p_embedding = None

    def _tokenize(self, texts: Union[str, List[str]], padding=True) -> dict:
        """텍스트를 토크나이즈하는 내부 메서드"""
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length" if padding else False,
            truncation=True,
            return_tensors="pt",
        )

    def get_dense_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        pickle_name = f"colbert_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            print("Load passage embedding")
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
        else:
            print("Build passage embedding")
            passages_emb = []

            for i in tqdm(range(0, len(self.contexts), self.batch_size)):
                batch_texts = self.contexts[i : i + self.batch_size]
                inputs = self._tokenize(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    emb = self.model.doc(**inputs)
                    passages_emb.append(emb)

            self.p_embedding = passages_emb  # List of tensors
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding saved")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """단일 쿼리에 대해 유사도가 가장 높은 k개의 passage를 반환합니다."""
        with torch.no_grad():
            # Query encoding
            q_inputs = self._tokenize(query)
            q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
            Q = self.model.query(**q_inputs)

            # Calculate similarity scores using ColBERT's scoring method
            scores = self.model.get_score(Q, self.p_embedding, eval=True)
            scores = scores.squeeze()

            # Get top-k documents
            top_k_scores, top_k_indices = torch.topk(scores, k=k)

            return top_k_scores.tolist(), top_k_indices.tolist()

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """다수의 쿼리에 대해 유사도가 가장 높은 k개의 passage를 반환합니다."""
        all_scores = []
        all_indices = []

        # Process queries in batches
        for i in tqdm(range(0, len(queries), self.batch_size)):
            batch_queries = queries[i : i + self.batch_size]

            with torch.no_grad():
                # Query encoding
                q_inputs = self._tokenize(batch_queries)
                q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
                Q = self.model.query(**q_inputs)

                # Calculate similarity scores
                print(f"Q size: {Q.size()}====================")

                scores = self.model.get_score(Q, self.p_embedding, eval=True)

                # Get top-k documents for each query in batch
                top_k_scores, top_k_indices = torch.topk(scores, k=k)

                all_scores.extend(top_k_scores.tolist())
                all_indices.extend(top_k_indices.tolist())

        return all_scores, all_indices


class ReRankRetrieval:
    def __init__(
        self,
        sparse_retriever: SparseRetrieval,
        colbert_path: str,
        data_path="../data/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        alpha: float = 0.5,  # sparse와 dense 점수의 가중치
    ) -> NoReturn:
        """
        Sparse retrieval 결과를 받아 dense retrieval로 re-ranking 하는 클래스
        $$$negative sampe 구현을 해야함$$$$
        Arguments:
            sparse_retriever: 초기 retrieval을 위한 SparseRetrieval 인스턴스
            colbert_path: 학습된 ColBERT 모델 경로
            data_path: 데이터가 저장된 경로
            device: 연산 장치
            batch_size: 배치 크기
            alpha: sparse와 dense 점수를 결합할 때 sparse 점수의 가중치 (1-alpha가 dense 점수 가중치)
        """
        self.sparse_retriever = sparse_retriever
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
        self.ret_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.ret_tokenizer.add_special_tokens(special_tokens)
        model = ColbertModel.from_pretrained("klue/bert-base")
        model.resize_token_embeddings(self.ret_tokenizer.vocab_size + 2)

        model.to(device)
        # ColBERT 모델 로드
        self.colbert = model.load_state_dict(
            torch.load(
                "/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_6.pth"
            )
        )

        self.colbert.eval()

        self.max_length = 512

    def _tokenize(self, texts) -> dict:

        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

    def rerank_documents(self, query, docs, sparse_scores, topk: int = 5):

        with torch.no_grad():
            # Query encoding
            q_inputs = self._tokenize(query)
            q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
            Q = self.colbert.query(**q_inputs)

            # Document encoding and scoring
            dense_scores = []
            for i in range(0, len(docs), self.batch_size):
                batch_docs = docs[i : i + self.batch_size]
                d_inputs = self._tokenize(batch_docs)
                d_inputs = {k: v.to(self.device) for k, v in d_inputs.items()}
                D = self.colbert.doc(**d_inputs)

                # Calculate scores for batch
                batch_scores = self.colbert.get_score(Q, [D], eval=True)
                dense_scores.extend(batch_scores.cpu().squeeze().tolist())

            # Normalize scores
            sparse_scores = np.array(sparse_scores)
            dense_scores = np.array(dense_scores)

            # Min-Max normalization
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

            # Combine scores
            final_scores = self.alpha * sparse_scores + (1 - self.alpha) * dense_scores

            # Get top-k
            top_k_indices = np.argsort(final_scores)[::-1][:topk]
            top_k_scores = final_scores[top_k_indices]

            return top_k_scores.tolist(), top_k_indices.tolist()

    def retrieve(self, query_or_dataset, topk=5, candidate_k=None):  # sparse retrieval에서 가져올 후보 문서 수
        if candidate_k is None:
            candidate_k = topk * 3

        if isinstance(query_or_dataset, str):
            # Sparse retrieval로 후보 추출
            self.parse_retriever.get_dense_embedding()
            sparse_scores, candidate_docs_indices = self.sparse_retriever.get_relevant_doc(
                query_or_dataset, topk=candidate_k
            )
            candidate_docs = [self.contexts[idx] for idx in candidate_docs_indices]
            # Re-ranking
            final_scores, doc_indices = self.rerank_documents(
                query_or_dataset, candidate_docs, sparse_scores, topk=topk
            )

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {final_scores[i]:4f}")
                print(candidate_docs[doc_indices[i]])

            return (final_scores, [candidate_docs[idx] for idx in doc_indices])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("hybrid retrieval"):
                for idx, example in enumerate(tqdm(query_or_dataset, desc="Hybrid retrieval: ")):
                    query = example["question"]

                    # Sparse retrieval로 후보 추출
                    self.parse_retriever.get_dense_embedding()
                    sparse_scores, candidate_docs_indices = self.sparse_retriever.get_relevant_doc_bulk(
                        query, topk=candidate_k
                    )
                    candidate_docs = [self.contexts[idx] for idx in candidate_docs_indices]
                    # Re-ranking
                    final_scores, doc_indices = self.rerank_documents(query, candidate_docs, sparse_scores, topk=topk)

                    # 결과 저장
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "context": " ".join([candidate_docs[did] for did in doc_indices]),
                    }

                    if "context" in example.keys() and "answers" in example.keys():
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]

                    total.append(tmp)

            return pd.DataFrame(total)


class DenseRetrieval:
    def __init__(self, model_args, data_args, training_args, context_path: Optional[str] = "wikipedia_documents.json"):
        """
        DenseRetrieval 클래스 초기화
        학습과 추론에 필요한 객체들을 초기화하며, in-batch negative 데이터를 준비합니다.
        """
        config = AutoConfig.from_pretrained(model_args.dense_model_name_or_path)
        self.data_path = data_args.data_path
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
            passage_vecs = torch.tensor(self.p_embedding).view(-1, self.q_encoder.config.hidden_size).to("cuda")

            with torch.no_grad():
                sim_scores = torch.matmul(query_vec, passage_vecs.T)

            topk_scores, topk_indices = torch.topk(sim_scores, k=topk)

            topk_passages = [self.contexts[idx] for idx in topk_indices.squeeze().cpu().tolist()]

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
