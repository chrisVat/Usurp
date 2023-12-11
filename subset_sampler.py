# abstract class
from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.cluster import DBSCAN
import torchvision
from sklearn.cluster import AgglomerativeClustering
import itertools
import os
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin_min
# custom sampler passed to dataloader to get subset of dataset

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler

SAMPLER_TECHNIQUES = ["random", "mtds", "mds", "ltds", "ads", "topk", "prob", "hisu", "ce", "sub"]
__all__ = ["get_sampler", "SAMPLER_TECHNIQUES"]


def get_sampler(technique, dataset_len, subset_percentage, distance_path, generator=None):
    assert technique in SAMPLER_TECHNIQUES, f"Technique {technique} not supported. Choose from {SAMPLER_TECHNIQUES}"

    if technique == "random":
        return RandomSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "mtds":
        return MovingTargetDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "mds":
        return MovingDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "ltds":
        return LinearTargetDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "ads":
        return AccuracyDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    # Top K
    elif technique.lower() == "topk":
        return TopKDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    # Probability
    elif technique.lower() == "prob":
        return ProbabilityDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    # Hierarchical Subcluster
    elif technique.lower() == "hisu":
        return HierarchicalSubclusterSampler()
    elif technique.lower() == "ce":
        return ClusterEdgeSampler()
    elif technique.lower() == "sub":
        return KMeansSubclusterSampler()


class SubsetSampler(ABC):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        self.dataset_len = dataset_len
        self.subset_percentage = subset_percentage
        self.generator = generator
        self.subset_len = int(self.dataset_len * self.subset_percentage)
        self.distances = torch.from_numpy(np.load(distance_path))
        # new distance cleaning
        self.distances = (self.distances - torch.mean(self.distances)) / torch.std(self.distances)
        self.distances = (self.distances - self.distances.min()) / (self.distances.max() - self.distances.min())
    
    def __iter__(self):
        indice_list = self.get_indices()
        for ind in indice_list:
            yield ind

    def __len__(self):
        return self.subset_len

    def load_cluster_mapping(self):
        self.cluster_mapping = None

    # Select the coreset
    @abstractmethod
    def get_indices(self, indice_list):
        pass

    #
    @abstractmethod
    def feedback(self, feedback):
        pass


# Randomly select the coreset
class RandomSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)

    def get_indices(self):
        return torch.randperm(self.dataset_len, generator=self.generator)[:self.subset_len]

    def feedback(self, feedback):
        pass


# Keep changing the threshold of the distance
class MovingDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p = 0.9
        self.avg_loss = None
        self.ind = None
        self.dist_shift = 0.02

    def get_indices(self):
        # generate random value per each index 
        random_values = torch.rand(self.dataset_len)
        # get indices whose value is larger than random value
        ind = torch.where(self.distances < random_values)[0] 
        # shuffle results and take first subset_len
        self.ind = torch.randperm(ind.shape[0], generator=self.generator)[:self.subset_len]
        return self.ind

    # TODO: maintain table of losses per example and move based off distance from previous loss?
    def feedback(self, feedback):
        avg_loss = np.mean(feedback["losses"])
        if self.avg_loss is None:
            self.avg_loss = avg_loss
        else: 
            self.avg_loss = self.loss_p*self.avg_loss + (1-self.loss_p)*avg_loss
        self.distances[self.ind] += torch.where(avg_loss > self.avg_loss, -self.dist_shift, self.dist_shift)


class MovingTargetDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None
        self.distance_pref = 0.05
        self.dist_shift = 0.01
        self.noise = 0.3

        self.avg_loss_decrease = None
        self.avg_loss_decrease_p = 0.9
        
        self.dist_momentum = 0
        self.dist_momentum_p = 0.9
        
        # normalize distances from 0 to 1
        self.distances = self.distances/self.distances.max()

    def get_indices(self):
        dist_from_pref = torch.abs(self.distances - self.distance_pref) 
        dist_from_pref += (torch.randn(self.dataset_len) - 0.5) * self.noise 
        self.ind = torch.argsort(dist_from_pref)
        self.ind = self.ind[:self.subset_len]
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        avg_loss = np.mean(feedback["losses"])
        if self.avg_loss is None:
            self.avg_loss = avg_loss
        else: 
            self.avg_loss = self.loss_p*self.avg_loss + (1-self.loss_p)*avg_loss
        
        if self.avg_loss_decrease is None:
            self.avg_loss_decrease = avg_loss - self.avg_loss
        else:
            self.avg_loss_decrease = self.avg_loss_decrease_p*self.avg_loss_decrease + (1-self.avg_loss_decrease_p)*(avg_loss - self.avg_loss)
        
        if self.avg_loss_decrease == 0:
            percent_shift = 1
        else:
            percent_shift = ((avg_loss - self.avg_loss) / self.avg_loss_decrease) + 0.5
        
        # check if percent_shift is nan
        if percent_shift != percent_shift:
            percent_shift = 1
        
        dist_shift = percent_shift * self.dist_shift
        dist_shift = max(dist_shift, 0) # can't decrease anymore

        self.dist_momentum = self.dist_momentum_p*self.dist_momentum + (1-self.dist_momentum_p)*dist_shift
        # update distance using dist_momentum, BUT, scale down the shift if it too large. 
        # it is too large when the update puts the distance below 0 or above 1
        update_amount = self.dist_momentum 
        if update_amount >= 0:
            update_amount = update_amount * (1-(self.distance_pref+update_amount))
        else:
            update_amount = update_amount * (self.distance_pref+update_amount) 

        update_amount = min(update_amount, 4*self.dist_shift)
        update_amount = max(update_amount, 0)

        self.distance_pref += update_amount
        self.distance_pref = min(self.distance_pref, 1)
        self.distance_pref = max(self.distance_pref, 0)

        print("Distance pref: ", self.distance_pref)


class LinearTargetDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None
        self.distance_pref = 0.15
        self.dist_shift = 0.02
        self.noise = 0.2

        self.total_epochs = 400
        self.update_amount = (1-self.distance_pref)/self.total_epochs
        self.distances = self.distances/self.distances.max()

    def get_indices(self):
        dist_from_pref = torch.abs(self.distances - self.distance_pref) 
        dist_from_pref += (torch.randn(self.dataset_len) - 0.5) * self.noise 
        self.ind = torch.argsort(dist_from_pref)
        self.ind = self.ind[:self.subset_len]
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        self.distance_pref += self.update_amount
        print("Distance pref: ", self.distance_pref)


class AccuracyDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None
        self.distance_pref = 0.1
        self.noise = 0.4

        self.est_best_accuracy = 0.95

        self.total_epochs = 400
        self.update_amount = (1-self.distance_pref)/self.total_epochs
        # Old distance cleaning
        # self.distances = self.distances/s elf.distances.max()

    def get_indices(self):
        dist_from_pref = torch.abs(self.distances - self.distance_pref) 
        dist_from_pref += (torch.randn(self.dataset_len) - 0.5) * self.noise 
        self.ind = torch.argsort(dist_from_pref)
        self.ind = self.ind[:self.subset_len]
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        feedback_accuracy = np.mean(feedback["corrects"])
        self.distance_pref = feedback_accuracy / self.est_best_accuracy + self.noise / 4
        print("Distance pref: ", self.distance_pref)


# Top k closest to the medoid in every cluster will be selected
class TopKDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.top_k = self.dataset_len // self.subset_len            # In every cluster, top k closest will be selected

    def get_indices(self):
        return np.argsort(self.distances, axis=0)[:self.top_k].flatten()

    def feedback(self, feedback):
        pass


# Select data based on the probability
class ProbabilityDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.load_cluster_mapping()

    def get_indices(self):
        # Inverse the distances to use as weights (closer points have higher weights)
        weights = 1 / self.distances
        normalized_weights = weights / weights.sum()

        # Results storage
        selected_indices = []

        # For each cluster
        for i_cluster in np.unique(self.cluster_mapping):
            # Filter indices for this cluster
            indices = np.where(self.cluster_mapping == i_cluster)[0]
            cluster_weights = normalized_weights[indices]

            # Determine the sample size (rounded up)
            _percentage = 0.2           # The percentage we want to get from every cluster
            sample_size = int(np.ceil(len(indices) * _percentage))

            # Perform weighted random sampling
            selected = np.random.choice(indices, size=sample_size, replace=False, p=cluster_weights / cluster_weights.sum())
            selected_indices.extend(selected)

        return selected_indices

    def feedback(self):
        pass


# Hierarchical SubCluster Sampler (Too Time-consuming)
class HierarchicalSubclusterSampler():
    def __init__(self):
        print("Into Class")
        self.embeddings = np.load("C:\\Users\\11597\\Desktop\\Usurp\\embeddings\\cifar_10_trained\\train_embeddings.npy")
        self.num_clusters = 20
        self.num_points_per_cluster = 100
        self.indices = self.get_indices()     # Save the subset indices

    # Perform hierarchical clustering and select indices of a subset of data points from each cluster.
    def get_indices(self):
        """
        :param num_clusters: The number of clusters to form.
        :param num_points_per_cluster: The number of point indices to select from each cluster.
        """
        Z = linkage(self.embeddings, method='ward')         # Perform hierarchical clustering
        print("Finish Linkage")
        cluster_labels = fcluster(Z, self.num_clusters, criterion='maxclust')        # Assign cluster labels
        print(f"cluster_labels: {cluster_labels}")

        # Initialize a dictionary to hold indices of subsets from each cluster
        self.indices = np.array([])

        print("get indices")

        # Loop over each cluster and select a subset of data points
        for i_cluster_id in range(1, self.num_clusters + 1):
            print(f"i_cluster_id: {i_cluster_id}")
            # Find indices of data points in this cluster
            indices_in_cluster = np.where(cluster_labels == i_cluster_id)[0]

            # Select a subset of indices from this cluster
            if len(indices_in_cluster) > self.num_points_per_cluster:
                selected_indices = np.random.choice(indices_in_cluster, self.num_points_per_cluster, replace=False)
            else:
                selected_indices = indices_in_cluster

            # Store the indices
            self.indices = np.append(self.indices, selected_indices)

        print("Finish: Get Indices")

        return self.indices

    def __iter__(self):
        for i_indice in self.indices:
            yield i_indice

    def __len__(self):
        return len(self.indices)

    def feedback(self, feedback):
        pass


class ClusterEdgeSampler:
    def __init__(self):
        self.distance_matrix = np.load("C:\\Users\\11597\\Desktop\\Usurp\\embeddings\\cifar_10_trained\\skmedoids_per_class_all_distances.npy")
        self.labels = np.load("C:\\Users\\11597\\Desktop\\Usurp\\embeddings\\cifar_10_trained\\skmedoids_per_class_labels.npy")
        self.indices = self.get_indices()
        print(self.indices.shape)
        print(self.indices[0: 5])

    # Select data points that are near the edge of a cluster's radius
    def get_indices(self, num_samples_per_cluster=100, edge_percentage=0.1):
        """
        :param distance_matrix: A 2D numpy array where rows are data points and columns are distances to each cluster's medoid.
        :param num_samples_per_cluster: Number of samples to select per cluster.
        :param edge_percentage: Percentage to define the edge of the clusters.
        :return: the indices to select
        """
        num_clusters = self.distance_matrix.shape[1]
        edge_samples_indices = []

        for cluster_id in range(num_clusters):
            distances = self.distance_matrix[np.where(self.labels == cluster_id)[0]]        # Extract distances for the current cluster
            edge_threshold = np.quantile(distances, 1 - edge_percentage)    # Determine the edge threshold for this cluster
            edge_indices = np.where(distances >= edge_threshold)[0]     # Find indices of data points near the edge of the cluster

            # If there are more edge points than required samples, select randomly
            if len(edge_indices) > num_samples_per_cluster:
                selected_indices = np.random.choice(edge_indices, num_samples_per_cluster, replace=False)
            else:
                selected_indices = edge_indices

            # Append the new indices
            edge_samples_indices.extend(selected_indices)

        return np.array(edge_samples_indices)

    def __iter__(self):
        for i_indice in self.indices:
            yield i_indice

    def __len__(self):
        return len(self.indices)

    def feedback(self, feedback):
        pass


class KMeansSubclusterSampler:
    def __init__(self):
        self.embeddings = np.load("C:\\Users\\11597\\Desktop\\Usurp\\embeddings\\cifar_10_trained\\train_embeddings.npy")
        self.num_primary_clusters = 30      # Number of primary clusters.
        self.num_subclusters = 10           # Number of subclusters within each primary cluster.
        self.num_samples_per_subcluster = 30        # Number of samples to select from each subcluster.
        self.indices = self.get_indices()

    def get_indices(self):
        # Primary clustering
        primary_kmeans = KMeans(n_clusters=self.num_primary_clusters)
        cluster_labels = primary_kmeans.fit_predict(self.embeddings)      # cluster and label them

        subcluster_samples = []

        for i_cluster in range(self.num_primary_clusters):
            # Extract data for the primary cluster
            cluster_data_indices = np.where(cluster_labels == i_cluster)[0]
            cluster_data = self.embeddings[cluster_data_indices]

            # Subclustering
            if len(cluster_data) > self.num_subclusters:  # Check if subclustering is possible
                sub_kmeans = KMeans(n_clusters=self.num_subclusters)
                subcluster_labels = sub_kmeans.fit_predict(cluster_data)

                # Select data from each subcluster
                for j_subcluster in range(self.num_subclusters):
                    subcluster_center = sub_kmeans.cluster_centers_[j_subcluster]
                    subcluster_data_indices = cluster_data_indices[np.where(subcluster_labels == j_subcluster)[0]]
                    subcluster_data = self.embeddings[subcluster_data_indices]

                    # Calculate distances and find the closest points
                    indices_closest, _ = pairwise_distances_argmin_min([subcluster_center], subcluster_data)
                    closest_indices = subcluster_data_indices[indices_closest]

                    # Randomly select samples from the subcluster
                    if len(closest_indices) > self.num_samples_per_subcluster:
                        selected_indices = np.random.choice(closest_indices, self.num_samples_per_subcluster, replace=False)
                    else:
                        selected_indices = closest_indices
                    # Extend the selected data
                    subcluster_samples.extend(selected_indices)

        self.indices = np.array(subcluster_samples)

        return self.indices

    def __iter__(self):
        for i_indice in self.indices:
            yield i_indice

    def __len__(self):
        return len(self.indices)

    def feedback(self, feedback):
        pass
