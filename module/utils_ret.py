

def load_contexts(context_path):
    """
    Load context data from file.
    """
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Loaded {len(contexts)} unique contexts.")
    return contexts

def build_dense_embedding(p_encoder, q_conder, contexts, tokenizer):
    """
    Passage encoder로부터 문서 임베딩을 계산하고 반환합니다.
    """
    p_embedding = []
    p_encoder.eval()
    dense_embedding_path = os.path.join('/data/ephemeral/data/', "dense_embedding.bin")
    q_encoder_path = os.path.join('/data/ephemeral/data/', "q_encoder.bin")

    for passage in tqdm(contexts, desc="Building dense embeddings"):
        passage_inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding="max_length").to('cuda')
        with torch.no_grad():
            p_emb = p_encoder(**passage_inputs).pooler_output  # pooled_output 사용
        p_embedding.append(p_emb.cpu().numpy())

    with open(dense_embedding_path, "wb") as file:
        pickle.dump(p_embedding, file)
    torch.save(q_encoder, q_encoder_path)
    print(f"Dense embedding and q_encoder saved to {dense_embedding_path} and {q_encoder_path}.")

def prepare_in_batch_negative(contexts, model_args, data_args, training_args) -> DataLoader:
    """
    Prepare in-batch negative samples for training.
    """
    dataset = load_from_disk(data_args.train_dataset_name)["train"]

    q_encoder = AutoModel.from_pretrained(model_args.dense_model_name_or_path, config=config).to('cuda')
    p_encoder = AutoModel.from_pretrained(model_args.dense_model_name_or_path, config=config).to('cuda')
    q_input_ids, q_attention_mask, q_token_type_ids = [], [], []
    p_input_ids, p_attention_mask, p_token_type_ids = [], [], []

    for i, context in enumerate(dataset['context']):
        q_encodings = tokenizer(dataset['question'][i], truncation=True, padding='max_length', return_tensors="pt")
        q_input_ids.append(q_encodings['input_ids'].tolist())
        q_attention_mask.append(q_encodings['attention_mask'].tolist())

        neg_contexts = sample_negatives(data_args.num_neg, context, contexts)
        p_encodings = tokenizer([context] + neg_contexts, truncation=True, padding='max_length', return_tensors="pt")
        p_input_ids.append(p_encodings['input_ids'].tolist())
        p_attention_mask.append(p_encodings['attention_mask'].tolist())
        if 'token_type_ids' in p_encodings:
            q_token_type_ids.append(q_encodings['token_type_ids'].tolist())
            p_token_type_ids.append(p_encodings['token_type_ids'].tolist())
    
    size=q_encodings['input_ids'].size(-1)
    dataset = create_tensor_dataset(data_args.num_neg, size, q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids)

    return DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)

def sample_negatives(num_neg, context, contexts):
    """
    Sample negative contexts.
    """
    while True:
        neg_idxs = np.random.randint(len(contexts), size=num_neg)
        neg_contexts = [contexts[idx] for idx in neg_idxs]
        if context not in neg_contexts:
            return neg_contexts

def create_tensor_dataset(num_neg, size, q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids):
    """
    Create tensor dataset for DataLoader.
    """
    if len(q_token_type_ids) == 0:
        return TensorDataset(
            torch.tensor(q_input_ids), torch.tensor(q_attention_mask),
            torch.tensor(p_input_ids).view(-1, num_neg + 1, size), torch.tensor(p_attention_mask).view(-1, num_neg + 1, size))
    else:
        return TensorDataset(
            torch.tensor(q_input_ids), torch.tensor(q_attention_mask), torch.tensor(q_token_type_ids),
            torch.tensor(p_input_ids).view(-1, num_neg + 1, size), torch.tensor(p_attention_mask).view(-1, num_neg + 1, size), torch.tensor(p_token_type_ids).vies(-1, num_neg+1, size))