import json
from Logic.core.utility.evaluation import Evaluation
from Logic.core.search import SearchEngine
from Logic.core.indexer.indexes_enum import Indexes

correct_list_for_spider_man = ['tt0145487', 'tt10872600', 'tt9362722', 'tt4633694', 'tt0316654']
correct_list_for_dark_knight = ['tt1877830', 'tt1345836', 'tt0468569']
correct_list_for_iron_man = ['tt0120737', 'tt0167261', 'tt0167260', 'tt0903624']
correct_list_for_harry_potter = ['tt0417741', 'tt0241527', 'tt0330373', 'tt0304141', 'tt0295297']

correct_list = [correct_list_for_spider_man, correct_list_for_dark_knight, correct_list_for_iron_man,
               correct_list_for_harry_potter]

try:
    with open('../core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("IMDB_crawled.json not found, initializing an empty list or dict.")
    data = {}

evaluation = Evaluation('MIR-Amirhossein')
search = SearchEngine()
weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
}


def run_test_evaluation():
    predicted_list_for_spider_man = search.search("spider man", "OkapiBM25", weights, True, 5)
    predicted_list_for_spider_man = [movie[0] for movie in predicted_list_for_spider_man]

    predicted_list_for_dark_knight = search.search("dark knight", "OkapiBM25", weights, True, 3)
    predicted_list_for_dark_knight = [movie[0] for movie in predicted_list_for_dark_knight]

    predicted_list_for_iron_man = search.search("iron man", "OkapiBM25", weights, True, 4)
    predicted_list_for_iron_man = [movie[0] for movie in predicted_list_for_iron_man]

    predicted_list_for_harry_potter = search.search("harry potter", "OkapiBM25", weights, True, 5)
    predicted_list_for_harry_potter = [movie[0] for movie in predicted_list_for_harry_potter]

    predicted_list = [predicted_list_for_spider_man, predicted_list_for_dark_knight, predicted_list_for_iron_man,
                      predicted_list_for_harry_potter]

    evaluation.calculate_evaluation(correct_list, predicted_list)
