from LSH import MinHashLSH
import json


def read_docs_from_json():
    try:
        with open('LSHFakeData.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("LSHFakeData.json not found, initializing an empty list or dict.")
        data = {}
    try:
        with open('LSHFakeData.json', 'r') as f:
            data.extend(json.load(f))
    except FileNotFoundError:
        print("LSHFakeData.json not found, initializing an empty list or dict.")
        data = {}

    docs = list()
    for movie in data:
        docs.append(movie["summaries"][0])

    return docs


documents = read_docs_from_json()
lsh = MinHashLSH(documents, 100)
buckets = lsh.perform_lsh()
lsh.jaccard_similarity_test(buckets, documents)

