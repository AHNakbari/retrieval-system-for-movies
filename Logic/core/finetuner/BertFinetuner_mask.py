import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres, problem_type="multi_label_classification")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        return data

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        data = self.load_dataset()
        genre_counts = {}
        for movie in data:
            for genre in movie['genres']:
                if genre in genre_counts:
                    genre_counts[genre] += 1
                else:
                    genre_counts[genre] = 1
        sorted_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)
        top_genres = sorted_genres[:self.top_n_genres]
        return top_genres

    def split_dataset(self, test_size=0.2, val_size=0.2):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        data = self.load_dataset()
        train_data, test_data = train_test_split(data, test_size=test_size)
        train_data, val_data = train_test_split(train_data, test_size=val_size)
        return train_data, val_data, test_data

    def create_dataset(self, data, top_genres):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        filtered_data = [movie for movie in data if movie['first_page_summary'] and any(genre in top_genres for genre in movie['genres'])]
        texts = [movie['first_page_summary'] if movie['first_page_summary'] is not None else '' for movie in filtered_data]
        genres = [[genre for genre in movie['genres'] if genre in top_genres] for movie in filtered_data]
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        mlb = MultiLabelBinarizer(classes=top_genres)
        labels = mlb.fit_transform(genres)
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, train_dataset, val_dataset, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        print("true label sample:", labels[0])
        preds = pred.predictions
        print("prediction label sample:", preds[0])
        probabilities = torch.sigmoid(torch.tensor(preds))
        predictions = (probabilities > 0.5).int().numpy()
        print("prediction sample after sigmoid and treshold:", predictions[0])
        precision = precision_score(labels, predictions, average='samples')
        recall = recall_score(labels, predictions, average='samples')
        f1 = f1_score(labels, predictions, average='samples')
        return {"Precision": precision,
                "Recall": recall,
                "F1-Score": f1}

    def evaluate_model(self, test_dataset):
        """
        Evaluate the fine-tuned model on the test set.
        """
        trainer = Trainer(
            model=self.model
        )
        results = trainer.evaluate(test_dataset)
        return results

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)