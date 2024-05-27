from Logic.core.lsh.LSH import MinHashLSH
from ..core.preprocess.preprocess import set_empty_instead_none
import json


def read_docs_from_json():
    try:
        with open('../core/lsh/LSHFakeData.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("LSHFakeData.json not found, initializing an empty list or dict.")
        data = {}

    # try:
    #     with open('../core/lsh/LSHFakeData.json', 'r') as f:
    #         data.extend(json.load(f))
    # except FileNotFoundError:
    #     print("LSHFakeData.json not found, initializing an empty list or dict.")
    #     data = {}

    try:
        with open('../core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
            data.extend(json.load(f))
    except FileNotFoundError:
        print("IMDB_crawled.json not found, initializing an empty list or dict.")
        data = {}

    data = set_empty_instead_none(data)
    docs = list()
    for movie in data:
        if len(movie["summaries"]) > 0:
            docs.append(movie["summaries"][0])

    return docs


def run_LSH_test():
    documents = read_docs_from_json()
    print("step 1")
    lsh = MinHashLSH(documents, 100)
    print("step 2")
    buckets = lsh.perform_lsh()
    print("step 3")
    lsh.jaccard_similarity_test(buckets, documents)

