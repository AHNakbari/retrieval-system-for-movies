import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents: list, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        words = document.split()
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        all_shingles = set()
        for doc in self.documents:
            shingles = self.shingle_document(doc)
            all_shingles = all_shingles.union(shingles)

        all_shingles = sorted(list(all_shingles))
        shingle_indices = {shingle: i for i, shingle in enumerate(all_shingles)}
        # print(all_shingles)
        matrix = np.zeros((len(all_shingles), len(self.documents)), dtype=int)

        for j, doc in enumerate(self.documents):
            shingles_in_doc = self.shingle_document(doc)
            for shingle in shingles_in_doc:
                i = shingle_indices[shingle]
                matrix[i, j] = 1

        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        char_matrix = self.build_characteristic_matrix()
        num_shingles, num_docs = char_matrix.shape
        signature_matrix = np.full((self.num_hashes, num_docs), np.inf)

        for hash_idx in range(self.num_hashes):
            hash_function = np.random.permutation(num_shingles)
            for doc_idx in range(num_docs):
                for shingle_idx in range(num_shingles):
                    if char_matrix[hash_function[shingle_idx], doc_idx] == 1:
                        if hash_function[shingle_idx] < signature_matrix[hash_idx, doc_idx]:
                            signature_matrix[hash_idx, doc_idx] = hash_function[shingle_idx]
                        break
        return signature_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        buckets = {}
        num_hashes, num_docs = signature.shape
        for band in range(bands):
            for doc_idx in range(num_docs):
                start = band * rows_per_band
                end = (band + 1) * rows_per_band
                if end > num_hashes:
                    end = num_hashes
                band_signature = signature[start:end, doc_idx]
                bucket_id = hash(tuple(band_signature))
                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(doc_idx)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signature_matrix = self.min_hash_signature()
        buckets = self.lsh_buckets(signature_matrix)
        return buckets

    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        jaccard_score = len(intersection) / len(union)
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
