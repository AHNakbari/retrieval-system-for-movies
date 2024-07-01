import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb


from typing import List, Tuple, Union, Any
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from .dimension_reduction import DimensionReduction
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from collections import Counter
from .clustering_metrics import purity_score

class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[object, object]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        centers = emb_vecs[np.random.choice(len(emb_vecs), n_clusters, replace=False)]
        labels = []
        for _ in range(max_iter):
            distances = np.linalg.norm(emb_vecs[:, np.newaxis] - centers, axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            centers = np.array([emb_vecs[labels == k].mean(axis=0) for k in range(n_clusters)])
        return centers.tolist(), labels.tolist()

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        word_count = Counter()
        for document in documents:
            words = document.split()
            word_count.update(words)
        words_frequency = word_count.most_common(top_n)
        return words_frequency

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[object, object, Union[int, Any]]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        emb_vecs = np.array(emb_vecs)
        centers = emb_vecs[np.random.choice(len(emb_vecs), n_clusters, replace=False)]
        labels = np.zeros(len(emb_vecs))
        distances = None
        for _ in range(100):
            distances = np.linalg.norm(emb_vecs[:, None] - centers, axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            centers = np.array([emb_vecs[labels == i].mean(axis=0) for i in range(n_clusters)])
        wcss = sum(np.min(distances, axis=1) ** 2)
        return centers.tolist(), labels.tolist(), wcss

    def cluster_hierarchical_single(self, emb_vecs: List) -> object:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        distances = linkage(emb_vecs, 'single', 'euclidean')
        max_distance = max(distances[:, 2]) / 2
        return fcluster(distances, max_distance, 'distance').tolist()

    def cluster_hierarchical_complete(self, emb_vecs: List) -> object:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        distances = linkage(emb_vecs, 'complete', 'euclidean')
        max_distance = max(distances[:, 2]) / 2
        return fcluster(distances, max_distance, 'distance').tolist()

    def cluster_hierarchical_average(self, emb_vecs: List) -> object:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        array_vectors = np.array(emb_vecs)
        distances = linkage(array_vectors, 'average', 'euclidean')
        max_distance = max(distances[:, 2]) / 2
        return fcluster(distances, max_distance, 'distance').tolist()

    def cluster_hierarchical_ward(self, emb_vecs: List) -> object:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        array_vectors = np.array(emb_vecs)
        distances = linkage(array_vectors, 'ward', 'euclidean')
        max_distance = max(distances[:, 2]) / 2
        return fcluster(distances, max_distance, 'distance').tolist()

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # wandb.init(project=project_name, name=run_name)
        centers, labels = self.cluster_kmeans(data, n_clusters)
        DR = DimensionReduction()
        data = DR.convert_to_2d_tsne(data)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.title(f"kmean clustering with {n_clusters} cluster")
        plt.show()
        # wandb.log({'kmean clustering': wandb.Image(plt)})
        # plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        # wandb.init(project=project_name, name=run_name)
        dendrogram(linkage(data, method=linkage_method))
        name = "Dendrogram (" + str(linkage_method) + ")"
        plt.title(name)
        plt.xlabel('Data')
        plt.ylabel('Distance')
        plt.show()
        # wandb.log({name: wandb.Image(plt)})


    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        k_list = []
        silhouette_scores = []
        purity_scores = []
        for k in k_values:
            k_list.append(k)
            # print(true_labels)
            centers, labels = self.cluster_kmeans(embeddings, k)
            silhouette_score_v = silhouette_score(embeddings, labels)
            # print(numeric_labels)
            # print(labels)
            purity_score_v = purity_score(true_labels, labels)
            silhouette_scores.append(silhouette_score_v)
            purity_scores.append(purity_score_v)

        plt.plot(k_list, silhouette_scores, color='red', label='Silhouette Score')
        plt.plot(k_list, purity_scores, color='blue', label='Purity Score')
        plt.title('Clustering Scores by Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.show()
        # if project_name and run_name:
        #     with wandb.init(project=project_name, name=run_name):
        #         wandb.log({"Cluster Scores": wandb.Image(plt)})
        #     plt.close()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # wandb.init(project=project_name, name=run_name)
        wcss_values = []
        for k in k_values:
            _, _, wcss = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(wcss)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
        plt.title('Elbow Method for Determining Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.grid(True)
        plt.show()
        # wandb.log({"Elbow Method WCSS Plot": wandb.Image(plt)})
        # plt.close()