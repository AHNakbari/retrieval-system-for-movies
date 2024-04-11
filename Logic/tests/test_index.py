from Logic.core.indexer.index import Index
from Logic.core.preprocess.preprocess import Preprocessor
import json


def read_docs_from_json():
    try:
        with open('../core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("IMDB_crawled.json not found, initializing an empty list or dict.")
        data = {}

    return data


documents = read_docs_from_json()
print("start preprocessing")
preprocessor = Preprocessor(documents, "../core/preprocess/stopwords.txt")
preprocessed_docs = preprocessor.preprocess()

print("-" * 35)
print("start indexing")
indexer = Index(preprocessed_docs)

indexer.check_add_remove_is_correct()

indexes_test = {
    "documents": "tt0080684",
    "stars": "tim",
    "genres": "drama",
    "summaries": "good",
}

print("-" * 35)
for index_name in indexes_test.keys():
    indexer.store_index("../core/indexer", index_name)
    print("+" * 25)

print("-" * 35)
for index_name in indexes_test.keys():
    loaded = indexer.load_index("../core/indexer", index_name)
    indexer.check_if_index_loaded_correctly(index_name, loaded)
    print("+" * 25)

print("-" * 35)
for index_name in indexes_test.keys():
    indexer.check_if_indexing_is_good(index_name, indexes_test[index_name])
    print("+" * 25)

