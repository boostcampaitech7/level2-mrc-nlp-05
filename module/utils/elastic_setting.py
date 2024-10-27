import argparse
import json
import pickle
import re

import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm


def es_setting():
    es = Elasticsearch("http://localhost:9200/", verify_certs=True, timeout=30)
    print(f"Elasticsearch version {es.info()['version']['number']}connected.")
    print("====Elastic Search Information====")
    print(es.info())
    return es


def create_index(es, index_name):
    INDEX_SETTINGS = {
        "settings": {
            "analysis": {
                "filter": {"my_shingle": {"type": "shingle"}},
                "analyzer": {
                    "korean": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter": ["my_shingle"],
                    }
                },
                "similarity": {"my_similarity": {"type": "BM25", "b": 0.75, "k1": 1.2}},
            }
        },
        "mappings": {"properties": {"document_text": {"type": "text", "analyzer": "korean"}}},
    }
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    return es.indices.create(index=index_name, body=INDEX_SETTINGS)


def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”" ":%&《》〈〉''㈜·\-'+\s一-龥サマーン]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#", " ", text)

    return text


def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    data = list(dict.fromkeys(v["text"] for v in data.values()))
    data = [preprocess(d) for d in data]
    data = [{"document_text": d} for d in data]
    return data


def insert_data(es, idx, data):
    corpus = load_data(data)
    for i, d in enumerate(tqdm(corpus)):
        try:
            es.index(index=idx, body=d, id=i)
        except Exception as e:
            print(f"Unable to insert data {d} into {idx}")
    count = es.count(index=idx)["count"]
    print(f"======Insert {count} data into {idx}=========")


def es_search(es, index_name, query, topk):
    query = {"query": {"bool": {"must": [{"match": {"document_text": query}}]}}}
    res = es.search(index=index_name, body=query, size=topk)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="./setting.json")
    parser.add_argument("--data", type=str, default="../../../data/wikipedia_documents.json")
    parser.add_argument("--index", type=str, default="wiki")
    args = parser.parse_args()

    es = es_setting()
    create_index(es, args.index)
    insert_data(es, args.index, args.data)

    query = "현대적 인사조직관리의 시발점이 된 책은?"
    res = es_search(es, args.index, query, 10)
    print("========== RETRIEVE RESULTS ==========")
    print(res)
    print("\n=========== RETRIEVE SCORES ==========\n")
    for hit in res["hits"]["hits"]:
        print("Doc ID: %3r  Score: %5.2f" % (hit["_id"], hit["_score"]))
