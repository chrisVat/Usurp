# sklearn medoids
from abc import ABC, abstractmethod
from sklearn_extra.cluster import KMedoids
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.clarans import clarans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import kmedoids as km

DISTANCE_TECHNIQUES = ["fasterpam", "clarans", "kmedoids", "skmedoids"]
__all__ = ["get_distance_calculator", "DISTANCE_TECHNIQUES"]


def get_distance_calculator(technique, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
    assert technique in DISTANCE_TECHNIQUES, f"Technique {technique} not supported. Choose from {DISTANCE_TECHNIQUES}"

    if technique == "fasterpam":
        return FasterPam(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)
    elif technique == "clarans":
        return Clarans(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)
    elif technique == "kmedoids":
        return Kmedoids(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)
    elif technique == "skmedoids":
        return SKmedoids(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)


class Distance(ABC):
    def __init__(self, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
        self.raw_embeddings = np.load(embedding_path)
        self.reduced_embeddings = None
        self.num_clusters_per_class = 5
        self.num_labels = num_labels
        self.use_reduced_for_medoids = use_reduced_for_medoids
        self.use_reduced_for_dist = use_reduced_for_dist

    # Get the embeddings, which can be reduced or raw
    def get_embeddings_for_distance(self):
        if self.use_reduced_for_dist and self.use_reduced_for_medoids and self.reduced_embeddings is not None:
            return self.reduced_embeddings
        return self.raw_embeddings

    def get_embeddings_for_medoids(self):
        if self.use_reduced_for_medoids and self.reduced_embeddings is not None:
            return self.reduced_embeddings
        return self.raw_embeddings

    # Get the Embeddings after PCA
    def pca(self, embeddings):
        print("staring pca")
        print("embeddings shape: ", embeddings.shape)
        scaler = StandardScaler()
        embeddings_standardized = scaler.fit_transform(embeddings)
        pca = PCA(n_components=0.95)
        embeddings_pca = pca.fit_transform(embeddings_standardized)
        print("Completed PCA")
        print("embeddings_pca shape: ", embeddings_pca.shape)
        return embeddings_pca

    def set_raw_embeddings(self, embeddings):
        self.raw_embeddings = embeddings
        return

    def set_reduced_embeddings(self, embeddings):
        self.reduced_embeddings = embeddings
        return

    def set_labels(self, labels):
        self.labels = labels
        return

    # Get distances with designated medoids sets
    def get_distances_with_medoids(self, medoids):
        embeddings_for_dist = self.get_embeddings_for_distance()    # Embedding Shape: #data * #features
        distances = np.zeros(self.raw_embeddings.shape[0])

        for i, embedding in enumerate(embeddings_for_dist):
            distances[i] = np.min(np.linalg.norm(embedding - medoids, axis=1))      # The distance between embedding and its closest medoid

        return np.array(distances)

    # Get distances with newly calculated medoids sets
    def get_distances(self):
        embeddings_for_dist = self.get_embeddings_for_distance()
        medoids = self.get_medoids()
        distances = np.zeros(self.raw_embeddings.shape[0])
        for i, embedding in enumerate(embeddings_for_dist):
            distances[i] = np.min(np.linalg.norm(embedding - medoids, dim=1))
        return np.array(distances)

    # Different ways to get medoids implemented in the following class
    @abstractmethod
    def _get_medoids(self, data_idxs):
        pass

    # Get medoids but the implementation is from _get_medoids()
    def get_medoids(self):
        data_idxs = np.arange(len(self.reduced_embeddings))     # Either reduced ot raw embeddings share the same length
        medoids_idxs = self._get_medoids(data_idxs)             # Get the medoids
        return np.array(self.get_embeddings_for_distance()[medoids_idxs])

    # Get the medoid from the subset (like in subset by class)
    def get_embeddings_from_subset_idxs(self, data_subset_idxs, subset_medoid_idxs):
        data_idxs = data_subset_idxs[subset_medoid_idxs]
        return np.array(self.get_embeddings_for_distance()[data_idxs])

    # Separate the whole dataset by their labels
    def get_data_idxs_per_class(self):
        data_idxs_per_class = []
        for i in range(self.num_labels):
            data_idxs_per_class.append(np.where(self.labels == i)[0])
        return data_idxs_per_class

    # Get the medoid from each class
    def get_medoid_per_class(self):
        data_idxs_per_class = self.get_data_idxs_per_class()
        medoids = []

        for class_id, data_idxs in enumerate(data_idxs_per_class):
            new_medoid_idxs = self._get_medoids(data_idxs)      # Find the medoid in the subset of one class
            medoid_values = self.get_embeddings_from_subset_idxs(data_idxs, new_medoid_idxs)    # return a generator
            # print("medoid values for class",class_id,": ", medoid_values.shape, len(data_idxs))
            for medoid in medoid_values:
                medoids.append(medoid)

        return np.array(medoids)


# per class, fastest
class FasterPam(Distance):
    def __init__(self, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
        super().__init__(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)

    def _get_medoids(self, data_idxs):
        relevant_embeddings = self.get_embeddings_for_medoids()[data_idxs]
        diss = euclidean_distances(relevant_embeddings)
        fp = km.fasterpam(diss, self.num_clusters_per_class)
        subset_medoid_indxes = fp.medoids
        return fp.medoids
    
    def get_medoids(self):
        return self._get_medoids(np.arange(len(self.reduced_embeddings)))


# this runs like a potato
class Clarans(Distance):
    def __init__(self, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
        super().__init__(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)
    
    def _get_medoids(self, data_idxs):
        relevant_embeddings = self.get_embeddings_for_medoids()[data_idxs]
        num_clusters = int(self.num_labels*self.num_clusters_per_class * (len(data_idxs) / len(self.reduced_embeddings)))
        medoids = clarans(relevant_embeddings, num_clusters, numlocal=5, maxneighbor=4).process()
        return medoids.get_medoids()


# this runs ok
class Kmedoids(Distance):
    def __init__(self, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
        super().__init__(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)
    
    def _get_medoids(self, data_idxs):
        relevant_embeddings = self.get_embeddings_for_medoids()[data_idxs]
        num_clusters = int(self.num_labels*self.num_clusters_per_class * (len(data_idxs) / len(self.reduced_embeddings)))
        initial_index_medoids = np.random.randint(0, len(relevant_embeddings), num_clusters)
        medoids = kmedoids(relevant_embeddings, initial_index_medoids).process()
        return medoids.get_medoids()


class SKmedoids(Distance):
    def __init__(self, embedding_path, num_labels, use_reduced_for_medoids=True, use_reduced_for_dist=True):
        super().__init__(embedding_path, num_labels, use_reduced_for_medoids, use_reduced_for_dist)

    def _get_medoids(self, data_idxs):
        embeddings = self.get_embeddings_for_medoids()[data_idxs]
        num_clusters = int(self.num_labels*self.num_clusters_per_class * (len(data_idxs) / len(self.reduced_embeddings)))
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=0).fit(embeddings)
        return kmedoids.medoid_indices_

