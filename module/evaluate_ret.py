from .dense_retrieval import DenseRetrieval
from .sparse_retrieval import SparseRetrieval
from .arguments import DataTrainingArguments, ModelArguments

from transformers import TrainingArguments
from datasets import load_from_disk, concatenate_datasets
from omegaconf import DictConfig
from tqdm import tqdm

def ret_evaluate(cfg: DictConfig):
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))

    dataset_dict = load_from_disk('/data/ephemeral/data/train_dataset')
    dataset1 = dataset_dict["train"].select(range(1000))
    dataset2 = dataset_dict["validation"]
    dataset_combined = concatenate_datasets([dataset1, dataset2])

    if data_args.which_retrieval == 'dense':
        retrieval = DenseRetrieval(model_args, data_args, training_args)
        retrieval.get_dense_embedding()
    elif data_args.which_retrieval == 'sparse':
        retrieval = SparseRetrieval(
            tokenize_fn,
            data_args.data_path,
        )
        retrieval.get_sparse_embedding()
    #else:
        
    
    top1_count=0
    top10_count=0
    topk_count=0 # 10 위로

    topk_passages = retrieval.retrieve(dataset_combined, data_args.top_k_retrieval, True)


    for i, data in enumerate(tqdm(topk_passages, desc="Evaluating retrieval")):
        original_context = dataset_combined[i]['context']
        if original_context == data[0]:
            top1_count+=1
        if original_context in data[0:10]:
            top10_count+=1
        if original_context in data[0:data_args.top_k_retrieval]:
            topk_count+=1
                
    
    # 결과 출력 (f-string 사용)
    print(f"Top 1 Score: {top1_count / (i+1) * 100:.2f}%")
    print(f"Top 10 Score: {top10_count / (i+1) * 100:.2f}%")
    print(f"Top {data_args.top_k_retrieval} Score: {topk_count / (i+1) * 100:.2f}%")

        
    
    