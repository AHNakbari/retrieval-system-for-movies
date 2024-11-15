import time
import os
import json
import copy
from Logic.core.indexer.indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc.get('id')
            if doc_id is not None:
                current_index[doc_id] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each term's tf in each document.
            So the index type is: {star_name: {document_id: tf}}
        """
        stars_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc.get('id')
            if not doc_id:
                continue

            for star in doc.get('stars', []):
                if star not in stars_index:
                    stars_index[star] = {}

                if doc_id not in stars_index[star]:
                    stars_index[star][doc_id] = 1
                else:
                    stars_index[star][doc_id] += 1

        return stars_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        genres_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc.get('id')
            if not doc_id:
                continue

            for genre in doc.get('genres', []):
                if genre not in genres_index:
                    genres_index[genre] = {}

                if doc_id not in genres_index[genre]:
                    genres_index[genre][doc_id] = 1
                else:
                    genres_index[genre][doc_id] += 1


        return genres_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        summaries_index = {}

        for doc in self.preprocessed_documents:
            doc_id = doc.get('id')
            if not doc_id:
                continue

            for term in doc.get('summaries', []):
                if term not in summaries_index:
                    summaries_index[term] = {}

                if doc_id not in summaries_index[term]:
                    summaries_index[term][doc_id] = 1
                else:
                    summaries_index[term][doc_id] += 1

            # for summary_terms in doc.get('summaries', []):
            #     all_terms = [term for sublist in summary_terms for term in sublist]
            #     for term in set(all_terms):
            #         tf = all_terms.count(term)
            #
            #         if term not in summaries_index:
            #             summaries_index[term] = {}
            #
            #         summaries_index[term][doc_id] = tf

        return summaries_index

    def get_posting_list(self, word: str, index_type_str: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type_str: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """
        try:
            Indexes(index_type_str)
        except ValueError:
            return []

        specific_index = self.index[Indexes(index_type_str).value]

        if word in specific_index:
            return list(specific_index[word].keys())
        else:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        doc_id = document.get('id')
        if not doc_id:
            return

        self.index[Indexes.DOCUMENTS.value][doc_id] = document

        for star in document.get('stars', []):
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            if doc_id not in self.index[Indexes.STARS.value][star]:
                self.index[Indexes.STARS.value][star][doc_id] = 1
            else:
                self.index[Indexes.STARS.value][star][doc_id] += 1

        for genre in document.get('genres', []):
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            if doc_id not in self.index[Indexes.GENRES.value][genre]:
                self.index[Indexes.GENRES.value][genre][doc_id] = 1
            else:
                self.index[Indexes.GENRES.value][genre][doc_id] += 1

        for term in document.get('summaries', []):
            if term not in self.index[Indexes.SUMMARIES.value]:
                self.index[Indexes.SUMMARIES.value] = {}
            if doc_id not in self.index[Indexes.SUMMARIES.value][term]:
                self.index[Indexes.SUMMARIES.value][term][doc_id] = 1
            else:
                self.index[Indexes.SUMMARIES.value][term][doc_id] += 1

        # all_terms = [term for sublist in document.get('summaries', []) for term in sublist]
        # for term in set(all_terms):
        #     if term not in self.index[Indexes.SUMMARIES.value]:
        #         self.index[Indexes.SUMMARIES.value][term] = {}
        #     tf = all_terms.count(term)
        #     self.index[Indexes.SUMMARIES.value][term][doc_id] = tf

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        if document_id in self.index[Indexes.DOCUMENTS.value]:
            del self.index[Indexes.DOCUMENTS.value][document_id]

        for star in self.index[Indexes.STARS.value].keys():
            if document_id in self.index[Indexes.STARS.value][star]:
                del self.index[Indexes.STARS.value][star][document_id]

                if not self.index[Indexes.STARS.value][star]:
                    del self.index[Indexes.STARS.value][star]

        for genre in self.index[Indexes.GENRES.value].keys():
            if document_id in self.index[Indexes.GENRES.value][genre]:
                del self.index[Indexes.GENRES.value][genre][document_id]

                if not self.index[Indexes.GENRES.value][genre]:
                    del self.index[Indexes.GENRES.value][genre]

        for term in list(self.index[Indexes.SUMMARIES.value].keys()):
            if document_id in self.index[Indexes.SUMMARIES.value][term]:
                del self.index[Indexes.SUMMARIES.value][term][document_id]

                if not self.index[Indexes.SUMMARIES.value][term]:
                    del self.index[Indexes.SUMMARIES.value][term]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '10000',
            'stars': ['tim', 'henri'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['10000'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henri']).difference(set(index_before_add[Indexes.STARS.value]['henri']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('10000')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        filename = os.path.join(path, f"{index_name}_index.json")

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.index[Indexes(index_name).value], file, ensure_ascii=False, indent=4)

        print(f"Index '{index_name}' has been stored at {filename}")

    def load_index(self, path: str, index_name: str, store: bool = False):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        store : bool
            store the loaded index or just return it
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        file_path = os.path.join(path, f"{index_name}_index.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")

        if index_name not in [index.value for index in Indexes]:
            raise ValueError(f"Invalid index name derived from file name: {index_name}")

        with open(file_path, 'r', encoding='utf-8') as file:
            loaded_index = json.load(file)

        if store:
            self.index[Indexes(index_name).value] = loaded_index
            print(f"Index '{index_name}' has been loaded and stored.")
        else:
            print(f"Index '{index_name}' has been loaded.")

        return loaded_index

    def check_if_index_loaded_correctly(self, index_type_str: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type_str : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        # print(self.index[Indexes(index_type_str).value])
        # print("--------------------")
        # print(loaded_index)
        condition = self.index[Indexes(index_type_str).value] == loaded_index
        if condition:
            print("loaded indexes are correct")
        else:
            print("loaded indexes are not correct")

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word == field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time <= brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

