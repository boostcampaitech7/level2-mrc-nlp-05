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
from transformers import (AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup,
    get_linear_schedule_with_warmup, EarlyStoppingCallback, TrainingArguments
)  
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm, trange
from transformers import 
from .arguments import ModelArguments, DataTrainingArguments

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:
    def __init__(self, model_args, data_args, context_path: Optional[str] = "wikipedia_documents.json", training_args):
        """
        DenseRetrieval 클래스 초기화
        학습과 추론에 필요한 객체들을 초기화하며, in-batch negative 데이터를 준비합니다.
        """
        config = AutoConfig.from_pretrained(model_args.dense_model_name_or_path)#(model_args.model_name_or_path)
        self.dataset = load_from_disk(data_args.train_dataset_name)["train"]
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        model_args.dense_model_name_or_path, use_fast=True)
        self.p_encoder = AutoModel.from_pretrained(model_args.dense_model_name_or_path, config=config).to('cuda')
        self.q_encoder = AutoModel.from_pretrained(model_args.dense_model_name_or_path, config=config).to('cuda')
        self.num_neg = data_args.num_neg
        self.contexts = self.load_contexts(os.path.join(data_args.data_path, context_path))
        self.batch_size= training_args.batch_size
        self.num_train_epochs = training_args.num_train_epochs
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
        q_input_ids, q_attention_mask, q_token_type_ids = [], [], []
        p_input_ids, p_attention_mask, p_token_type_ids = [], [], []
        corpus = list(set([example for example in self.dataset['context']]))

        for i, context in enumerate(self.dataset['context']):
            q_encodings = self.tokenizer(self.dataset['question'][i], truncation=True, padding='max_length', return_tensors="pt")
            q_input_ids.append(q_encodings['input_ids'].squeeze(0).tolist())
            q_attention_mask.append(q_encodings['attention_mask'].tolist())
            q_token_type_ids.append(q_encodings['token_type_ids'].squeeze(0).tolist() if 'token_type_ids' in q_encodings else [0] * len(q_encodings['input_ids']))

            neg_contexts = self._sample_negatives(context, corpus)
            p_encodings = self.tokenizer([context] + neg_contexts, truncation=True, padding='max_length', return_tensors="pt")
            p_input_ids.append(p_encodings['input_ids'].view(-1, self.num_neg + 1, 512).tolist())
            p_attention_mask.append(p_encodings['attention_mask'].view(-1, self.num_neg + 1, 512).tolist())
            p_token_type_ids.append(p_encodings['token_type_ids'].view(-1, self.num_neg + 1, 512).tolist() if 'token_type_ids' in p_encodings else [[0] * 512] * (self.num_neg + 1))
        # `token_type_ids`는 RoBERTa에서 사용하지 않으므로 제거
        dataset = self._create_tensor_dataset(q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids)
        return DataLoader(dataset, batch_size=self.batch_size)

    def _sample_negatives(self, context, corpus):
        """
        Sample negative contexts.
        """
        while True:
            neg_idxs = np.random.randint(len(corpus), size=self.num_neg)
            neg_contexts = [corpus[idx] for idx in neg_idxs]
            if context not in neg_contexts:
                return neg_contexts

    def _create_tensor_dataset(self, q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids):
        """
        Create tensor dataset for DataLoader.
        """
        return TensorDataset(
            torch.tensor(q_input_ids), torch.tensor(q_attention_mask), torch.tensor(q_token_type_ids),
            torch.tensor(p_input_ids), torch.tensor(p_attention_mask), torch.tensor(p_token_type_ids)
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
                p_emb = self.p_encoder(**passage_inputs).pooler_output  # pooled_output 사용
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
            query_vec = self.q_encoder(**query_inputs).pooler_output

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
                query_vec = self.q_encoder(**query_inputs).pooler_output
                
                passage_vecs = torch.tensor(self.p_embedding).squeeze(1).to('cuda')

                with torch.no_grad():
                    sim_scores = torch.matmul(query_vec, passage_vecs.T).squeeze()
                topk_scores, topk_indices = torch.topk(sim_scores, k=topk)
                print(sim_scores.size())
                print(topk_indices)
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
            result = pd.DataFrame(total)
            print(result.head(10))
            return result


    def train(self):
        """
        Train passage and query encoders.
        """
        wandb.init(project="dense_retrieval_project")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=5e-5,
            eps=1e-8
        )
        t_total = len(self.train_dataloader) // 1 * self.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=t_total
        )

        best_loss = float('inf')  # Initialize best loss
        global_step = 0
        early_stopping_patience = 3
        patience_counter = 0  # Counter for early stopping
        best_q_encoder_path = os.path.join('/data/ephemeral/data/', "q_encoder_best.bin")

        # zero_grad() 및 empty_cache()를 학습 시작 전에 호출하여 상태를 초기화
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()  # GPU 메모리 정리
        '''
        for epoch in trange(1, desc="Epoch"):
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

        return self.p_encoder, self.q_encoder'''
        train_iterator = trange(int(self.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_loss = 0.0
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                        
                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(self.batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to('cuda')

                    q_inputs = {
                        "input_ids": batch[0].to('cuda'),
                        "attention_mask": batch[1].to('cuda'),
                        "token_type_ids": batch[2].to('cuda')  # token_type_ids 추가
                    }
                    
                    p_inputs = {
                        "input_ids": batch[3].view(self.batch_size * (self.num_neg + 1), -1).to('cuda'),
                        "attention_mask": batch[4].view(self.batch_size * (self.num_neg + 1), -1).to('cuda'),
                        "token_type_ids": batch[5].view(self.batch_size * (self.num_neg + 1), -1).to('cuda')  # token_type_ids 추가
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs).pooler_output
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs).pooler_output

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(self.batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(self.batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(self.batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    wandb.log({"loss": loss.item(), "global_step": global_step})

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1
                    epoch_loss += loss.item()

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs
                epoch_loss /= len(self.train_dataloader)
                wandb.log({"epoch_loss": epoch_loss, "epoch": _ + 1})
        return self.p_encoder, self.q_encoder

if __name__ == "__main__":
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))
    training_args.num_neg = 2
    training_args.batch_size = 8
    training_args.use_faiss = False

    retriever = DenseRetrieval(model_args=model_args, data_args=data_args, training_args=training_args)
    retriever.train()
    retriever.get_dense_embedding()
    #여기까지가 train + embedding 및 q_encoder 저장
    
    '''
    # Test Dense
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = retriever.tokenizer

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    
    if training_args.use_faiss:

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
