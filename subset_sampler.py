# abstract class
from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.cluster import DBSCAN
import torchvision
from sklearn.cluster import AgglomerativeClustering
# custom sampler passed to dataloader to get subset of dataset

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler

SAMPLER_TECHNIQUES = ["random", "mtds", "mds", "ltds", "ads", "topk", "sa", "prob", "hisu"]
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
    elif technique.lower() == "topk":
        return TopKDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "sa":
        return SilhouetteAnalysisDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "prob":
        return ProbabilityDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "hisu":
        return HierarchicalSubClusterSampler(dataset_len, subset_percentage, distance_path, generator)


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

    def load_labels(self):
        cifar_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.labels = np.array(cifar_data.targets)

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


# Use a Silhouette Analysis score to value the distance
class SilhouetteAnalysisDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.distances = np.array(self.distances)       # Change the type for easier calculation
        self.load_cluster_mapping()
        self.cluster_mapping = np.array(self.cluster_mapping)
        self.n_clusters = len(np.unique(self.cluster_mapping))

    def silhouette_analysis(self):
        silhouette_vals = np.zeros(self.cluster_mapping.shape)

        # traverse all class
        for i_cluster in range(self.n_clusters):
            # Intra-cluster distance (distance to own medoid)
            intra_distance = [d for d, l in zip(self.distances, self.cluster_mapping) if l == i_cluster]

            # Inter-cluster distance (distance to nearest medoid of other cluster)
            # inter_distance = np.array([min(self.distances[self.cluster_mapping == j][self.cluster_mapping != i_cluster]) if len(self.distances[self.cluster_mapping == j][self.cluster_mapping != i_cluster]) > 0 else np.inf for j in range(self.n_clusters)])
            inter_distance = np.empty(self.n_clusters)
            for i in range(self.n_clusters):
                current_cluster_distances = self.distances[self.cluster_mapping == i]       # Get distances for the current cluster
                min_distances = []                      # Initialize a list to store minimum distances to other clusters
                for j in range(self.n_clusters):
                    if i == j:          # Skip if it's the same cluster
                        continue
                    # Get distances to other cluster
                    other_cluster_distances = self.distances[self.cluster_mapping == j]
                    # Calculate the minimum distance to the other cluster
                    min_distance = np.min(current_cluster_distances[other_cluster_distances != i])
                    min_distances.append(min_distance)
                # If there are no distances (i.e., the cluster is isolated)
                inter_distance[i] = np.inf if not min_distances else np.min(min_distances)
            inter_distance = inter_distance[np.newaxis].T  # Convert to column vector for broadcasting

            # Silhouette values for points in this cluster
            silhouette_vals[self.cluster_mapping == i_cluster] = (inter_distance - intra_distance) / np.maximum(intra_distance, inter_distance)

        # Handling case where a cluster has only one point
        silhouette_vals[np.isnan(silhouette_vals)] = 0

        return np.mean(silhouette_vals), silhouette_vals


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


# Get Sub-cluster
class HierarchicalSubClusterSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)

        self.n_sub_clusters = 5     # Number of sub-clusters within each primary cluster.
        self.subset_size = 0.2      # Fixed proportion of points to select from each sub-cluster.
        self.sub_cluster_models = {}
        self.subset_indices = {}
        self.load_cluster_mapping()

    # Fit Subcluster and extract subset indices
    def fit(self, X, cluster_medoids):
        """
        :param X: The data to be clustered.
        :param cluster_medoids: The medoids (central points) of the initial clusters.
        """
        # Perform hierarchical sub-clustering within each primary cluster
        self.subcluster_mapping = np.zeros_like(self.cluster_mapping)
        sub_cluster_label_start = 100

        # Traverse all the medoids
        for i_cluster_medoid in cluster_medoids:
            # Extract data points and their indices belonging to the current primary cluster
            cluster_indices = np.where(self.cluster_mapping == i_cluster_medoid)
            cluster_data = X[cluster_indices]

            # Apply hierarchical clustering to this subset
            hierarchical = AgglomerativeClustering(n_clusters=self.n_sub_clusters)
            sub_clusters = hierarchical.fit_predict(cluster_data)
            self.sub_cluster_models[i_cluster_medoid] = hierarchical

            # Assign unique sub-cluster labels
            self.subcluster_mapping[cluster_indices] = sub_clusters + sub_cluster_label_start

            # Increment the starting index for the next primary cluster
            sub_cluster_label_start += 100

            # Extract subsets
            for sub_cluster in np.unique(sub_clusters):
                sub_cluster_data_indices = cluster_indices[sub_clusters == sub_cluster]
                sub_cluster_data = cluster_data[sub_clusters == sub_cluster]

                # Determine the number of points to select
                if isinstance(self.subset_size, float):
                    n_points = int(np.ceil(len(sub_cluster_data) * self.subset_size))
                else:
                    n_points = min(len(sub_cluster_data), self.subset_size)

                # Select the closest n_points to the medoid
                distances = np.linalg.norm(sub_cluster_data - i_cluster_medoid, axis=1)
                closest_indices = np.argsort(distances)[:n_points]
                self.subset_indices[(i_cluster_medoid, sub_cluster)] = sub_cluster_data_indices[closest_indices]

    def get_sub_cluster_assignments(self):
        return self.subcluster_mapping     # An array of sub-cluster assignments.

    def get_indices(self):
        """
        Get the indices of the extracted subsets.

        :return: A dictionary of subset indices keyed by (medoid, sub-cluster) pairs.
        """
        return self.subset_indices


