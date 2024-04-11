class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        with open("../core/preprocess/stopwords.txt", 'r') as file:
            self.stopwords = set(file.read().splitlines())

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        words = query.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        query = self.remove_stop_words_from_query(query)
        query_words = set(query.split())
        doc_words = doc.split()
        indices = [i for i, word in enumerate(doc_words) if word.strip('.,!?').lower() in query_words]

        if not indices:
            return "", list(query_words)

        snippets = []
        for index in indices:
            start = max(index - self.number_of_words_on_each_side, 0)
            end = min(index + self.number_of_words_on_each_side + 1, len(doc_words))
            snippet = ' '.join(doc_words[start:end])
            snippets.append(snippet)

        final_snippet = " ... ".join(snippets)
        for word in query_words:
            final_snippet = final_snippet.replace(word, f"***{word}***")

        not_exist_words = [word for word in query_words if word.lower() not in doc.lower()]

        return final_snippet, not_exist_words

