import fasttext
import re
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import string

import fasttext_data_loader

# import nltk
# nltk.download('stopwords')


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    if stopword_removal:
        stop_words = set(stopwords.words('english')).union(set(stopwords_domain))
        words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) >= minimum_length]
    return ' '.join(words)


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        # TODO: fix the path
        print("#")
        texts = [preprocess_text(text) for text in texts]
        print("$")
        pd.DataFrame(texts).to_csv("train_data.csv", index=False, header=False)
        self.model = fasttext.FastText.train_unsupervised(input="train_data.csv", model=self.method)

    def get_query_embedding(self, query, do_preprocess=True):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        preprocessed_query = preprocess_text(query)
        return self.model.get_sentence_vector(preprocessed_query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # result_vector = self.model.get_word_vector(word2) - self.model.get_word_vectors(word1)\
        #                 + self.model.get_word_vectors(word3)
        # words, _ = zip(*self.model.get_nearest_neighbors(result_vector, k=1))
        # return words[0]

        # Obtain word embeddings for the words in the analogy
        word1_vector = self.model.get_word_vector(word1)
        word2_vector = self.model.get_word_vector(word2)
        word3_vector = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        result_vector = word2_vector - word1_vector + word3_vector

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        words = self.model.get_words(include_freq=False)
        mapping = {word: self.model.get_word_vector(word) for word in words}

        # Exclude the input words from the possible results
        mapping.pop(word1)
        mapping.pop(word2)
        mapping.pop(word3)

        # Find the word whose vector is closest to the result vector
        min_distance = float('inf')
        result = None
        for word, vector in mapping.items():
            dist = distance.cosine(result_vector, vector)
            if dist < min_distance:
                min_distance = dist
                result = word

        return result

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        path : str
            path for the model to save and load
        save : bool
            Save or do not save the model
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    # path = '../indexer/index/'
    # ft_data_loader = fasttext_data_loader.FastTextDataLoader(path)
    #
    # X, y = ft_data_loader.create_train_data()

    # ft_model.prepare(X, mode="train", save=True)

    ft_model.prepare([], mode="load")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "king"
    word2 = "man"
    word3 = "queen"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
