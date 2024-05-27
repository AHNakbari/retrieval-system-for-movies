from .graph import LinkGraph
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_id = movie['id']
            self.graph.add_node(movie_id)  # Add each movie as a node
            for star in movie['stars']:
                self.graph.add_edge(movie_id, star)  # Create edge from movie to star
                self.graph.add_edge(star, movie_id)  # Create edge from star to movie
                self.hubs.append(star)
                self.authorities.append(movie_id)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie_id = movie['id']
            if movie_id not in self.graph.adj_list:
                self.graph.add_node(movie_id)
            for star in movie['stars']:
                if star not in self.graph.adj_list:
                    self.graph.add_node(star)
                self.graph.add_edge(movie_id, star)
                self.graph.add_edge(star, movie_id)
                if star not in self.hubs:
                    self.hubs.append(star)
                if movie_id not in self.authorities:
                    self.authorities.append(movie_id)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = {node: 1.0 for node in self.authorities}  # Initialize authority scores
        h_s = {node: 1.0 for node in self.hubs}  # Initialize hub scores

        for _ in range(num_iteration):
            new_a_s = {node: sum(h_s[hub] for hub in self.graph.get_predecessors(node)) for node in self.authorities}
            new_h_s = {node: sum(a_s[auth] for auth in self.graph.get_successors(node)) for node in self.hubs}

            # Normalize scores
            norm = sum(new_a_s.values())
            a_s = {node: score / norm for node, score in new_a_s.items()}

            norm = sum(new_h_s.values())
            h_s = {node: score / norm for node, score in new_h_s.items()}

        sorted_authorities = sorted(a_s.items(), key=lambda x: x[1], reverse=True)[:max_result]
        sorted_hubs = sorted(h_s.items(), key=lambda x: x[1], reverse=True)[:max_result]

        return [node for node, _ in sorted_hubs], [node for node, _ in sorted_authorities]


def set_empty_instead_none(documents: list):

    for doc in documents:
        doc_id = doc.get('id')
        if not doc_id:
            continue

        if not doc.get('stars', []):
            doc['stars'] = []

        if not doc.get('genres', []):
            doc['genres'] = []

        if not doc.get('summaries', []):
            doc['summaries'] = []

    return documents


if __name__ == "__main__":
    try:
        with open('../crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("IMDB_crawled.json not found, initializing an empty list or dict.")
        data = {}

    data = set_empty_instead_none(data)

    # You can use this section to run and test the results of your link analyzer
    corpus = data
    root_set = data[0:1000]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')