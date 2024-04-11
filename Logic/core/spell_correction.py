import json
import string


def create_all_documents():
    try:
        with open('../core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("IMDB_crawled.json not found, initializing an empty list or dict.")
        data = {}

    all_documents = list()
    for movie in data:
        all_documents.extend(movie['stars'])
        all_documents.extend(movie['genres'])
        all_documents.extend(movie['summaries'])

    for idx, text in enumerate(all_documents):
        all_documents[idx] = text.translate(str.maketrans('', '', string.punctuation))

    return all_documents

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2) -> set:
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set([word[i:i + k] for i in range(len(word) - k + 1)])
        return shingles
    
    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        for document in all_documents:
            for word in document.split():
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                word_counter[word] = word_counter.get(word, 0) + 1
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        word_shingles = self.shingle_word(word)
        scores = [(other_word, self.jaccard_score(word_shingles, shingles))
                  for other_word, shingles in self.all_shingled_words.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        top5_candidates = [word for word, score in scores[:5]]
        return top5_candidates
    
    def spell_check(self, query, test: bool = False):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        test : bool
            just for print all nearest
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        corrected_query = []
        for word in query.split():
            if word in self.word_counter:
                corrected_query.append(word)
            else:
                nearest_words = self.find_nearest_words(word)
                if test:
                    print(nearest_words)
                corrected_query.append(nearest_words[0] if nearest_words else word)
        return ' '.join(corrected_query)
