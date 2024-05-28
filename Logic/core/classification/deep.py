import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data_loader
from basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, train_loader, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        train_loader: DataLoader
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=False)
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}')

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        pass

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        f1 = f1_score(true_labels, predictions, average='macro')
        return total_loss / len(dataloader), predictions, true_labels, f1

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        self.test_loader = DataLoader(ReviewDataSet(X_test, y_test), batch_size=self.batch_size)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        test_loader = DataLoader(ReviewDataSet(x, y), batch_size=self.batch_size, shuffle=False)
        _, predictions, true_labels, _ = self._eval_epoch(test_loader)
        return classification_report(true_labels, predictions)


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    # loader = data_loader.ReviewLoader('IMDB Dataset.csv')
    # loader.load_data()
    # loader.get_embeddings()
    # X_train, X_test, y_train, y_test = loader.split_data()
    #
    # np.save("X_train.npy", X_train)
    # np.save("X_test.npy", X_test)
    # np.save("y_train.npy", y_train)
    # np.save("y_test.npy", y_test)

    X_train, X_test, y_train, y_test =\
        np.load("X_train.npy"),\
        np.load("X_test.npy"),\
        np.load("y_train.npy"),\
        np.load("y_test.npy")

    # Initialize classifier
    classifier = DeepModelClassifier(in_features=100, num_classes=2, batch_size=32, num_epochs=10)
    classifier.set_test_dataloader(X_test, y_test)

    # Create DataLoader for training
    train_loader = DataLoader(ReviewDataSet(X_train, y_train), batch_size=32, shuffle=True)
    classifier.fit(train_loader, y_train)

    # Evaluate and print the report
    print(classifier.prediction_report(X_test, y_test))

# Epoch 2/10:   0%|          | 0/1250 [00:00<?, ?it/s, loss=0.492]Epoch 1, Average Loss: 0.48010526366233824
# Epoch 3/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 2, Average Loss: 0.4511444072008133
# Epoch 4/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 3, Average Loss: 0.44965028653144834
# Epoch 5/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 4, Average Loss: 0.4456863194942474
# Epoch 6/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 5, Average Loss: 0.44486641683578493
# Epoch 7/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 6, Average Loss: 0.4438979401350021
# Epoch 8/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 7, Average Loss: 0.4447994443178177
# Epoch 9/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 8, Average Loss: 0.4408833240032196
# Epoch 10/10:   0%|          | 0/1250 [00:00<?, ?it/s]Epoch 9, Average Loss: 0.4423816330432892
# Epoch 10, Average Loss: 0.44072887208461764
#               precision    recall  f1-score   support
#
#            0       0.80      0.93      0.86      4961
#            1       0.92      0.77      0.84      5039
#
#     accuracy                           0.85     10000
#    macro avg       0.86      0.85      0.85     10000
# weighted avg       0.86      0.85      0.85     10000