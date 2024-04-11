import numpy as np


class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal, but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            df = len(self.index.get(term, {}))
            idf = np.log((self.N / (df + 1))) + 1
            self.idf[term] = idf
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        query_tfs = {}
        for term in query:
            if term in query_tfs:
                query_tfs[term] += 1
            else:
                query_tfs[term] = 1
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        query_tfs = self.get_query_tfs(query)
        list_of_documents = self.get_list_of_documents(query)
        doc_method, query_method = method.split('.')
        scores = {}
        for document_id in list_of_documents:
            score = self.get_vector_space_model_score(query, query_tfs, document_id, doc_method, query_method)
            if score > 0:
                scores[document_id] = score

        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_vector = []
        document_vector = []
        query_terms = list(set(query))
        for idx, term in enumerate(query_terms):
            if term not in self.index:
                continue
            query_vector = self.calculate_tf_idf(query_tfs[term], self.get_idf(term), query_method, query_vector)
            document_vector = self.calculate_tf_idf(self.index[term].get(document_id, 0), self.get_idf(term),
                                                    document_method, document_vector)

        query_vector = self.calculate_normal(query_vector, query_method)
        document_vector = self.calculate_normal(document_vector, document_method)

        return np.dot(np.array(query_vector), np.array(document_vector))

    def calculate_tf_idf(self, tf: int, idf: float, method: str, vector: list):
        if method[0] == 'n':
            vector.append(tf)
        elif method[0] == 'l':
            if tf == 0:
                vector.append(0)
            else:
                vector.append(1 + np.log(tf))

        if method[1] == 't':
            vector[-1] *= idf

        return vector

    def calculate_normal(self, vector: list, method: str):
        if method[2] == 'c':
            vector = np.array(vector)
            vector = vector / np.linalg.norm(vector)
        return vector

    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}
        list_of_documents = self.get_list_of_documents(query)
        for document_id in list_of_documents:
            scores[document_id] = self.get_okapi_bm25_score(query, document_id, average_document_field_length,
                                                            document_lengths)
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        k1 = 1.5
        b = 0.75
        score = 0

        for term in query:
            if term not in self.index:
                continue
            df = len(self.index[term])
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf = self.index[term].get(document_id, 0)
            doc_len = document_lengths.get(document_id, 0)
            len_norm = ((1 - b) + b * (doc_len / average_document_field_length))
            tf_component = tf * (k1 + 1) / (tf + k1 * len_norm)
            score += idf * tf_component
        return score
