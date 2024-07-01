import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
import random
from sklearn.preprocessing import LabelEncoder

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from .dimension_reduction import DimensionReduction
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
# from .clustering_metrics import ClusteringMetrics
from .clustering_utils import ClusteringUtils
from .clustering_metrics import purity_score





# Main Function: Clustering Tasks
if __name__ == '__main__':
    # wandb.login(key="e96c1f760eca2f02b2e8ce5987a5bb385677cbf8")
    # 0. Embedding Extraction
    model = FastText(method='skipgram')
    model.prepare([], mode="load", path="../word_embedding/FastText_model.bin")
    data_loader = FastTextDataLoader("../indexer/index/")
    X, y = data_loader.create_train_data()
    for i, movie in enumerate(y):
        if len(movie) != 0:
            y[i] = movie[0]
        else:
            y[i] = "drama"
    X = X[0:100]
    y = y[0:100]
    X_emb = np.array([model.get_query_embedding(text) for text in tqdm(X)])

    # 1. Dimension Reduction
    dimred = DimensionReduction()
    X = dimred.pca_reduce_dimension(X_emb, 50)
    dimred.wandb_plot_explained_variance_by_components(X_emb, "MIR_project", "final_test")
    x_tsne = dimred.convert_to_2d_tsne(X)
    dimred.wandb_plot_2d_tsne(X, "MIR_project", "final_test")

    # 2. Clustering
    # K-Means Clustering
    centeroids = []
    cluster_assignments = []
    cu = ClusteringUtils()
    k_values = [2 * i for i in range(1, 10)]
    for k in k_values:
        cu.visualize_kmeans_clustering_wandb(X, k, "MIR_project", "final_test")

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    cu.plot_kmeans_cluster_scores(X, y, k_values, "MIR_project", "final_test")

    ## Hierarchical Clustering
    linkages = ["average", "ward", "complete", "single"]
    for linkage in linkages:
        cu.wandb_plot_hierarchical_clustering_dendrogram(X, "MIR_project", linkage, "final_test")

    # 3. Evaluation
    cu.visualize_elbow_method_wcss(X, [2 * i for i in range(1, 20)], "MIR_project", "final_test")
    for k in range(2, 100, 4):
        centeroids, cluster_assignments = cu.cluster_kmeans(X, k)
        print(
            f"{k}:  ari: {adjusted_rand_score(y, cluster_assignments)} , purity: {purity_score(y, cluster_assignments)} , silhouette: {silhouette_score(X, cluster_assignments)}")
