import json
from Logic.core.utility.evaluation import Evaluation
from Logic.core.search import SearchEngine
from Logic.core.indexer.indexes_enum import Indexes


class TestEvaluation:
    def __init__(self):

        self.correct_list_for_spider_man = ['tt0145487', 'tt10872600', 'tt9362722', 'tt4633694', 'tt0316654']
        self.correct_list_for_dark_knight = ['tt1877830', 'tt1345836', 'tt0468569']
        self.correct_list_for_iron_man = ['tt0120737', 'tt0167261', 'tt0167260', 'tt0903624']
        self.correct_list_for_harry_potter = ['tt0417741', 'tt0241527', 'tt0330373', 'tt0304141', 'tt0295297']

        self.correct_list = [self.correct_list_for_spider_man, self.correct_list_for_dark_knight,
                             self.correct_list_for_iron_man, self.correct_list_for_harry_potter]

        try:
            with open('../core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print("IMDB_crawled.json not found, initializing an empty list or dict.")
            self.data = {}

        self.evaluation = Evaluation('MIR-Amirhossein')
        self.search = SearchEngine()
        self.weights = {
                Indexes.STARS: 1,
                Indexes.GENRES: 1,
                Indexes.SUMMARIES: 1
        }

        self.run_test_evaluation()

    def run_test_evaluation(self):
        predicted_list_for_spider_man = self.search.search("spider man", "OkapiBM25", self.weights, True, 5)
        predicted_list_for_spider_man = [movie[0] for movie in predicted_list_for_spider_man]
        # print(predicted_list_for_spider_man)

        predicted_list_for_dark_knight = self.search.search("dark knight", "OkapiBM25", self.weights, True, 3)
        predicted_list_for_dark_knight = [movie[0] for movie in predicted_list_for_dark_knight]
        # print(predicted_list_for_dark_knight)

        predicted_list_for_iron_man = self.search.search("iron man", "OkapiBM25", self.weights, True, 4)
        predicted_list_for_iron_man = [movie[0] for movie in predicted_list_for_iron_man]
        # print(predicted_list_for_iron_man)

        predicted_list_for_harry_potter = self.search.search("harry potter", "OkapiBM25", self.weights, True, 5)
        predicted_list_for_harry_potter = [movie[0] for movie in predicted_list_for_harry_potter]
        # print(predicted_list_for_harry_potter)

        predicted_list = [predicted_list_for_spider_man, predicted_list_for_dark_knight, predicted_list_for_iron_man,
                          predicted_list_for_harry_potter]

        self.evaluation.calculate_evaluation(self.correct_list, predicted_list)
