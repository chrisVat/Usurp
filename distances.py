# sklearn medoids
from sklearn_extra.cluster import KMedoids


class DistanceCalculator():
    def __init__(self, embedding_paths, num_labels):
        self.embeddings = embeddings
        self.num_labels = num_labels

    def get_distances(self):
        medoids = get_medoids(self.embeddings, self.labels)
        distances = torch.zeros(self.embeddings.shape[0])
        for i, embedding in enumerate(self.embeddings):
            distances[i] = torch.min(torch.norm(embedding - medoids, dim=1))
        return distances
    
    def get_medoids(self, embeddings, labels):
        num_clusters = self.num_labels*5
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=0).fit(embeddings)
        return kmedoids.cluster_centers_
