import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
import data_loader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.num_classes = len(np.unique(y))
        self.classes = np.unique(y)
        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, x.shape[1]))

        for idx, cl in enumerate(self.classes):
            x_c = x[y == cl]
            self.prior[idx] = x_c.shape[0] / float(x.shape[0])
            self.feature_probabilities[idx, :] = (np.sum(x_c, axis=0) + self.alpha) / (
                        np.sum(x_c) + self.alpha * x.shape[1])

        self.log_probs = np.log(self.feature_probabilities)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        return np.argmax(np.dot(x, self.log_probs.T) + np.log(self.prior), axis=1)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        pass


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader = data_loader.ReviewLoader("IMDB Dataset.csv")
    loader.load_data()
    x = loader.review_tokens
    y = loader.sentiments

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Convert text data to feature matrix
    cv = CountVectorizer(max_features=45000)
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)

    # Initialize and train Naive Bayes
    nb = NaiveBayes(cv, alpha=1)
    nb.fit(x_train_cv.toarray(), y_train)
    print(nb.prediction_report(x_test_cv.toarray(), y_test))

#              precision    recall  f1-score   support
#
#            0       0.83      0.88      0.85      4982
#            1       0.87      0.82      0.84      5018
#
#     accuracy                           0.85     10000
#    macro avg       0.85      0.85      0.85     10000
# weighted avg       0.85      0.85      0.85     10000

