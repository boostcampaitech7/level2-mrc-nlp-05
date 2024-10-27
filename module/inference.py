from typing import Callable, Dict, List, NoReturn, Tuple

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, set_seed

from module.arguments import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from module.col_dpr.model import ColbertModel
from module.col_dpr.tokenizer import *
from module.dense_retrieval import ColBERTRetrieval, DenseRetrieval, ReRankRetrieval
from module.es_retrieval import ESRetrieval
from module.mrc import run_mrc
from module.sparse_retrieval import BM25andTfidfRetrieval, BM25Retrieval, SparseRetrieval
from module.trainer_qa import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def inference(cfg: DictConfig):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))
    customtraining_args = CustomTrainingArguments(**cfg.get("custom"))

    result_path = f"{model_args.model_name_or_path.split('/')[-1]}_{data_args.dataset_name.split('/')[-1]}_{datetime.now(timezone(timedelta(hours=9))).strftime('%m-%d-%H')}"
    training_args.output_dir = os.path.join(training_args.output_dir, result_path)
    project_name = f"{model_args.model_name_or_path.split('/')[-1]}_train_dataset_bm25Plus_max512_stride256"
    # project_name = f"roberta-large_train_preprocessed_fp16+gradient_accumulation_v2"
    model_path = os.path.join(model_args.saved_model_path, project_name)
    # training_args.do_train = True

    print(f"model file from {model_path}")
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    # print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    training_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        # datasets = run_sparse_retrieval(
        #     tokenizer.tokenize, datasets, training_args, data_args,
        # )
        if customtraining_args.retrieval_type == "dense":
            if customtraining_args.dense_retrieval_type == "colbert":
                datasets = run_colbert_retrieval(datasets)
            else:
                datasets = run_dense_retrieval(model_args=model_args, data_args=data_args, training_args=training_args)
        else:
            datasets = run_sparse_retrieval(
                tokenizer.tokenize, datasets, training_args, data_args, customtraining_args.sparse_retrieval_type
            )
    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, training_tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    sparse_retrieval_type: str,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.

    if sparse_retrieval_type == "elastic":
        retriever = ESRetrieval("wiki")
    else:
        if sparse_retrieval_type == "bm25":
            retriever = BM25Retrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
        elif sparse_retrieval_type == "tfidf":
            retriever = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
        retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_dense_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    retriever = DenseRetrieval(
        model_args=model_args, data_args=data_args, training_args=training_args, context_path=context_path
    )

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(datasets["validation"], topk=data_args.top_k_retrieval)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


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

    if os.path.exists("/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth"):
        rank = torch.load("/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth")
        print(rank.size())
    else:
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
        torch.save(rank, "/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth")
        print(rank.size())

    k = 50
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


def run_rerank_retrieval(
    datasets,
    data_args: DataTrainingArguments,
):
    df = ReRankRetrieval(datasets["validation"], topk=data_args.top_k_retrieval)
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets
