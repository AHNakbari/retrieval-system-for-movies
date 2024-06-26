from BertFinetuner_mask import BERTFinetuner

# Instantiate the class
bert_finetuner = BERTFinetuner('/kaggle/input/imdb-crawler/IMDB_crawled.json', top_n_genres=5)

# Load the dataset
data = bert_finetuner.load_dataset()

# Preprocess genre distribution
top_genres = bert_finetuner.preprocess_genre_distribution()

# Split the dataset
train_data, val_data, test_data = bert_finetuner.split_dataset()

# Create datasets
train_dataset = bert_finetuner.create_dataset(train_data, top_genres)
val_dataset = bert_finetuner.create_dataset(val_data, top_genres)
test_dataset = bert_finetuner.create_dataset(test_data, top_genres)

# Fine-tune BERT model
bert_finetuner.fine_tune_bert(train_dataset, val_dataset, epochs=6, batch_size=64)

# Compute metrics
result = bert_finetuner.evaluate_model(test_dataset)
print("Evaluation Results:", result)

# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')