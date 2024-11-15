import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd

from .basic_classifier import BasicClassifier
from Logic.core.classification import data_loader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.vectors = None
        self.labels = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.vectors = x
        self.labels = y

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
        distances = np.linalg.norm(self.vectors - x, axis=1)
        nearest_neighbor = np.argsort(distances)[:self.k]
        vote_counts = np.bincount(self.labels[nearest_neighbor])
        return np.argmax(vote_counts)

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
        y_pred = [self.predict(xi) for xi in tqdm(x, desc="Predicting")]
        return classification_report(y, y_pred)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    knn_model = KnnClassifier(5)
    loader = data_loader.ReviewLoader("IMDB Dataset.csv")
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    knn_model.fit(X_train, y_train)
    print(knn_model.prediction_report(X_test, y_test))

    # Generating embeddings: 100%|██████████| 50000/50000 [01:18<00:00, 639.26it/s]
    # Predicting: 100%|██████████| 10000/10000 [02:56<00:00, 56.64it/s]
    #               precision    recall  f1-score   support
    #
    #            0       0.80      0.85      0.82      4983
    #            1       0.84      0.79      0.81      5017
    #
    #     accuracy                           0.82     10000
    #    macro avg       0.82      0.82      0.82     10000
    # weighted avg       0.82      0.82      0.82     10000
