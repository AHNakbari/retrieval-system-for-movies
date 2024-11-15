import nltk
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class Preprocessor:

    def __init__(self, documents: list, stopword_file_path: str):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        with open(stopword_file_path, 'r') as file:
            self.stopwords = set(file.read().splitlines())
        self.stemmer = PorterStemmer()

    def preprocess_pipline(self, text: str):
        normalized_text = self.normalize(text)
        no_links_text = self.remove_links(normalized_text)
        no_punctuations_text = self.remove_punctuations(no_links_text)
        tokenized_text = self.tokenize(no_punctuations_text)
        return self.remove_stopwords(list(tokenized_text))

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        self.documents = set_empty_instead_none(self.documents)

        preprocessed_docs = []

        for doc in self.documents:
            doc_id = doc.get('id')
            if not doc_id:
                continue

            stars_list = []
            for star in doc.get('stars', []):
                stars_list.extend(self.preprocess_pipline(star))
            doc['stars'] = stars_list

            genres_list = []
            for genre in doc.get('genres', []):
                genres_list.extend(self.preprocess_pipline(genre))
            doc['genres'] = genres_list

            summaries_list = []
            for summary in doc.get('summaries', []):
                summaries_list.extend(self.preprocess_pipline(summary))
                for title in doc.get('title', ''):
                    summaries_list.extend(self.preprocess_pipline(title))
            doc['summaries'] = summaries_list

            preprocessed_docs.append(doc)

        return preprocessed_docs

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        normalized_text = ' '.join(stemmed_tokens)
        return normalized_text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens: list):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        return filtered_tokens


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
