import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import wandb
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW  # AdamW를 torch.optim에서 가져옴
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup  # 필요한 모듈 추가
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm, trange
from transformers import get_linear_schedule_with_warmup, EarlyStoppingCallback

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
        
class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

class DenseRetrieval:
    def __init__(self, model_args, data_args, tokenizer, context_path: Optional[str] = "wikipedia_documents.json"):
        """
        DenseRetrieval 클래스 초기화
        학습과 추론에 필요한 객체들을 초기화하며, in-batch negative 데이터를 준비합니다.
        """
        config = AutoConfig.from_pretrained('klue/roberta-large')#(model_args.model_name_or_path)
        self.dataset = load_from_disk('/data/ephemeral/data/train_dataset')["train"]
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        'klue/bert-base', use_fast=True)
        self.p_encoder = AutoModel.from_pretrained('klue/roberta-large', config=config).to('cuda')
        self.q_encoder = AutoModel.from_pretrained('klue/roberta-large', config=config).to('cuda')
        self.num_neg = 1
        self.contexts = self.load_contexts(os.path.join('/data/ephemeral/data/', context_path))
        self.train_dataloader = self.prepare_in_batch_negative(data_args)

        # Embedding 저장을 위한 변수
        self.p_embedding = None

    def load_contexts(self, context_path):
        """
        Load context data from file.
        """
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Loaded {len(contexts)} unique contexts.")
        return contexts

    def prepare_in_batch_negative(self, data_args) -> DataLoader:
        """
        Prepare in-batch negative samples for training.
        """
        q_input_ids, q_attention_mask = [], []
        p_input_ids, p_attention_mask = [], []
        corpus = np.array(self.contexts)

        for i, context in enumerate(self.dataset['context']):
            q_encodings = self.tokenizer(self.dataset['question'][i], truncation=True, padding='max_length', return_tensors="pt")
            q_input_ids.append(q_encodings['input_ids'].squeeze(0).tolist())
            q_attention_mask.append(q_encodings['attention_mask'].squeeze(0).tolist())

            neg_contexts = self.s_sample_negative(context, corpus)
            p_encodings = self.tokenizer([context] + neg_contexts.tolist(), truncation=True, padding='max_length', return_tensors="pt")
            p_input_ids.append(p_encodings['input_ids'].tolist())
            p_attention_mask.append(p_encodings['attention_mask'].tolist())

        # `token_type_ids`는 RoBERTa에서 사용하지 않으므로 제거
        dataset = self._create_tensor_dataset(q_input_ids, q_attention_mask, p_input_ids, p_attention_mask)
        return DataLoader(dataset, batch_size=3)

    def _sample_negatives(self, context, corpus):
        """
        Sample negative contexts.
        """
        while True:
            neg_idxs = np.random.randint(len(corpus), size=self.num_neg)
            neg_contexts = corpus[neg_idxs]
            if context not in neg_contexts:
                return neg_contexts

    def _create_tensor_dataset(self, q_input_ids, q_attention_mask, p_input_ids, p_attention_mask):
        """
        Create tensor dataset for DataLoader.
        """
        return TensorDataset(
            torch.tensor(q_input_ids), torch.tensor(q_attention_mask),
            torch.tensor(p_input_ids), torch.tensor(p_attention_mask)
        )

    def get_dense_embedding(self) -> NoReturn:
        """
        Passage Embedding을 만들고 pickle로 저장하거나, 저장된 파일을 불러옵니다.
        q_encoder는 return 하여 다른 곳에서 사용할 수 있게 처리합니다.
        """
        dense_embedding_path = os.path.join('/data/ephemeral/data/', "dense_embedding.bin")
        q_encoder_path = os.path.join('/data/ephemeral/data/', "q_encoder.bin")

        # 1. 이미 저장된 파일이 있으면 불러오기
        if os.path.isfile(dense_embedding_path) and os.path.isfile(q_encoder_path):
            print("Loading saved dense embeddings and q_encoder...")
            with open(dense_embedding_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(q_encoder_path, "rb") as file:
                self.q_encoder = torch.load(file).to('cuda')  # GPU로 옮김
            print("Loaded dense embedding and q_encoder from files.")
        
        # 2. 저장된 파일이 없으면 새로운 학습을 수행
        else:
            print("Training new q_encoder and p_encoder...")
            self.p_encoder, self.q_encoder = self.train()  # 학습 수행

            print("Building passage dense embeddings...")
            self.p_embedding = self.build_dense_embedding()

            # 저장
            with open(dense_embedding_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            torch.save(self.q_encoder, q_encoder_path)
            print(f"Dense embedding and q_encoder saved to {dense_embedding_path} and {q_encoder_path}.")

        # p_encoder 메모리에서 해제
        del self.p_encoder  # p_encoder 삭제
        torch.cuda.empty_cache()  # 메모리 정리

    def build_dense_embedding(self):
        """
        Passage encoder로부터 문서 임베딩을 계산하고 반환합니다.
        """
        p_embedding = []
        for passage in tqdm(self.contexts, desc="Building dense embeddings"):
            passage_inputs = self.tokenizer(passage, return_tensors="pt", truncation=True, padding="max_length").to('cuda')
            with torch.no_grad():
                p_emb = self.p_encoder(**passage_inputs).last_hidden_state.mean(dim=1)
            p_embedding.append(p_emb.cpu().numpy())

        return np.array(p_embedding)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10
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

        # 단일 Query가 입력된 경우
        if isinstance(query_or_dataset, str):
            query_inputs = self.tokenizer(query_or_dataset, return_tensors="pt", truncation=True, padding="max_length").to('cuda')
            query_vec = self.q_encoder(**query_inputs).last_hidden_state.mean(dim=1)

            passage_vecs = torch.tensor(self.p_embedding).squeeze(1).to('cuda')

            with torch.no_grad():
                sim_scores = torch.matmul(query_vec, passage_vecs.T).squeeze()

            topk_scores, topk_indices = torch.topk(sim_scores, k=topk)
            topk_passages = [self.contexts[idx] for idx in topk_indices.cpu().tolist()]

            return topk_scores.cpu().tolist(), topk_passages

        # 다수의 Query가 포함된 Dataset이 입력된 경우
        elif isinstance(query_or_dataset, Dataset):

            total = []
            queries = query_or_dataset["question"]

            for idx, query in enumerate(queries):
                query_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding="max_length").to('cuda')
                query_vec = self.q_encoder(**query_inputs).last_hidden_state.mean(dim=1)

                passage_vecs = torch.tensor(self.p_embedding).squeeze(1).to('cuda')

                with torch.no_grad():
                    sim_scores = torch.matmul(query_vec, passage_vecs.T).squeeze()

                topk_scores, topk_indices = torch.topk(sim_scores, k=topk)
                topk_passages = [self.contexts[i] for i in topk_indices.cpu().tolist()]

                # 상위 k개의 문서를 하나의 문자열로 결합
                joined_passages = " ".join(topk_passages)

                # 'train' 데이터셋의 경우
                if "context" in query_or_dataset.column_names and "answers" in query_or_dataset.column_names:
                    tmp = {
                        "context": joined_passages,  # 상위 k개의 문서를 하나의 문자열로 결합하여 사용
                        "id": query_or_dataset["id"][idx],
                        "question": query,
                        "answers": query_or_dataset["answers"][idx],
                    }

                # 'validation' 또는 'test' 데이터셋의 경우
                else:
                    tmp = {
                        "context": joined_passages,  # 상위 k개의 문서를 하나의 문자열로 결합하여 사용
                        "id": query_or_dataset["id"][idx],
                        "question": query,
                    }

                total.append(tmp)

            return pd.DataFrame(total)


    def train(self):
        """
        Train passage and query encoders.
        """
        wandb.init(project="dense_retrieval_project")
        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_scheduler(optimizer, len(self.train_dataloader)*10)

        best_loss = float('inf')  # Initialize best loss
        global_step = 0
        early_stopping_patience = 3
        patience_counter = 0  # Counter for early stopping
        best_q_encoder_path = os.path.join('/data/ephemeral/data/', "q_encoder_best.bin")

        # zero_grad() 및 empty_cache()를 학습 시작 전에 호출하여 상태를 초기화
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()  # GPU 메모리 정리

        for epoch in trange(10, desc="Epoch"):
            epoch_loss = 0.0
            for batch in tqdm(self.train_dataloader, desc="Training"):
                self.p_encoder.train()
                self.q_encoder.train()

                loss = self._compute_loss(batch)
                loss.backward()

                optimizer.step()
                scheduler.step()

                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                global_step += 1
                epoch_loss += loss.item()

                wandb.log({"loss": loss.item(), "global_step": global_step})

            epoch_loss /= len(self.train_dataloader)
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0  # Reset counter when improvement is seen
                torch.save(self.q_encoder, best_q_encoder_path)  # Save the best q_encoder
                print(f"Best model saved at epoch {epoch + 1} with loss {best_loss}")
            else:
                patience_counter += 1
                print(f"No improvement in epoch {epoch + 1}. Early stopping patience counter: {patience_counter}")

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # 학습이 끝나면 필요한 데이터만 남기고 나머지 삭제
        del self.train_dataloader  # 학습에 사용된 DataLoader 삭제
        del optimizer, scheduler  # Optimizer와 Scheduler 삭제

        torch.cuda.empty_cache()
        print("Training finished, all unnecessary memory released.")

        return self.p_encoder, self.q_encoder

    def _initialize_scheduler(self, optimizer, total_steps, num_warmup_steps=0):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer used for training.
            total_steps: Total number of training steps.
            num_warmup_steps: Number of warmup steps (default: 0).
        """
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=total_steps
        )
        return scheduler

    def _initialize_optimizer(self):
        """
        Initialize optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01}
        ]
        return AdamW(optimizer_grouped_parameters, lr=5e-5)

    def _compute_loss(self, batch):
        """
        배치 데이터로부터 손실을 계산하는 함수입니다.
        """
        batch_size = batch[0].size(0)
        targets = torch.zeros(batch_size).long().to('cuda')  # GPU로 옮김

        p_inputs = {"input_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to('cuda'),  # GPU로 옮김
                    "attention_mask": batch[3].view(batch_size * (self.num_neg + 1), -1).to('cuda')}  # GPU로 옮김
        q_inputs = {"input_ids": batch[0].to('cuda'),  # GPU로 옮김
                    "attention_mask": batch[1].to('cuda')}  # GPU로 옮김

        p_outputs = self.p_encoder(**p_inputs).last_hidden_state.mean(dim=1)
        q_outputs = self.q_encoder(**q_inputs).last_hidden_state.mean(dim=1)

        p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
        q_outputs = q_outputs.view(batch_size, 1, -1)

        sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
        sim_scores = F.log_softmax(sim_scores, dim=1)

        loss = F.nll_loss(sim_scores, targets)

        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, help="Dataset name")
    parser.add_argument("--model_name_or_path", metavar="bert-base-multilingual-cased", type=str, help="Model path")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, help="Path to context file")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_neg", type=int, default=2, help="Number of negative samples")

    args = parser.parse_args()

    # Load dataset, tokenizer, and models
    dataset = load_from_disk(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    p_encoder = AutoModel.from_pretrained(args.model_name_or_path)
    q_encoder = AutoModel.from_pretrained(args.model_name_or_path)

    retriever = DenseRetrieval(args, args, tokenizer)
    retriever.train(args)
    '''
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
        '''
