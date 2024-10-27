from datasets import concatenate_datasets, load_from_disk
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments

from .arguments import DataTrainingArguments, ModelArguments
from .dense_retrieval import DenseRetrieval
from .sparse_retrieval import BM25Retrieval, SparseRetrieval


def ret_evaluate(cfg: DictConfig):
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))

    dataset_dict = load_from_disk("/data/ephemeral/data/train_dataset")
    dataset1 = dataset_dict["train"].select(range(1000))
    dataset2 = dataset_dict["validation"]
    dataset_combined = concatenate_datasets([dataset1, dataset2])
    tokenizer = AutoTokenizer.from_pretrained(
        "klue/roberta-large",
        use_fast=True,
    )
    if data_args.which_retrieval == "dense":
        retrieval = DenseRetrieval(model_args, data_args, training_args)
        retrieval.get_dense_embedding()
    elif data_args.which_retrieval == "sparse":
        tokenize_fn = tokenizer.tokenize
        retrieval = SparseRetrieval(
            tokenize_fn,
            data_args.data_path,
        )
        retrieval.get_sparse_embedding()
    elif data_args.which_retrieval == "bm25":
        tokenize_fn = tokenizer.tokenize
        retrieval = BM25Retrieval(
            tokenize_fn,
            data_args.data_path,
        )
        retrieval.get_sparse_embedding()
    # else:

    top1_count = 0
    top10_count = 0
    top20_count = 0
    top40_count = 0
    top50_count = 0
    top100_count = 0

    topk_passages = retrieval.retrieve(dataset_combined, 50, True)

    for i, data in enumerate(tqdm(topk_passages, desc="Evaluating retrieval")):
        original_context = dataset_combined[i]["context"]
        if original_context == data[0]:
            top1_count += 1
        if original_context in data[0:10]:
            top10_count += 1
        if original_context in data[0:20]:
            top20_count += 1
        if original_context in data[0:40]:
            top40_count += 1
        if original_context in data[0:50]:
            top50_count += 1
        if original_context in data[:100]:
            top100_count += 1

    # 결과 출력 (f-string 사용)
    print(f"Top 1 Score: {top1_count / (i+1) * 100:.2f}%")
    print(f"Top 10 Score: {top10_count / (i+1) * 100:.2f}%")
    print(f"Top 20 Score: {top20_count / (i+1) * 100:.2f}%")
    print(f"Top 40 Score: {top40_count / (i+1) * 100:.2f}%")
    print(f"Top 50 Score: {top50_count / (i+1) * 100:.2f}%")
    print(f"Top 100 Score: {top100_count / (i+1) * 100:.2f}%")
