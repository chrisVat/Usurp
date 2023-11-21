# abstract class
from abc import ABC, abstractmethod
import torch
import numpy as np
# custom sampler passed to dataloader to get subset of dataset

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler

SAMPLER_TECHNIQUES = ["random", "mtds", "mds"]
__all__ = ["get_sampler", "SAMPLER_TECHNIQUES"]


def get_sampler(technique, dataset_len, subset_percentage, generator=None):
    assert technique in SAMPLER_TECHNIQUES, f"Technique {technique} not supported. Choose from {SAMPLER_TECHNIQUES}"
    if technique == "random":
        return RandomSampler(dataset_len, subset_percentage, generator)
    elif technique == "mtds":
        return MovingTargetDistanceSampler(dataset_len, subset_percentage, generator)
    elif technique == "mds":
        return MovingDistanceSampler(dataset_len, subset_percentage, generator)


class SubsetSampler(ABC):
    def __init__(self, dataset_len, subset_percentage, generator=None):
        self.dataset_len = dataset_len
        self.subset_percentage = subset_percentage
        self.generator = generator
        self.subset_len = int(self.dataset_len * self.subset_percentage)
    
    def __iter__(self):
        indice_list = self.get_indices()
        for ind in indice_list:
            yield ind

    def __len__(self):
        return self.subset_len
        
    @abstractmethod
    def get_indices(self, indice_list):
        pass

    @abstractmethod
    def feedback(self, feedback):
        pass


class RandomSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, generator=None):
        super().__init__(dataset_len, subset_percentage, generator)

    def get_indices(self):
        return torch.randperm(self.dataset_len, generator=self.generator)[:self.subset_len]

    def feedback(self, feedback):
        pass


class MovingDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, generator=None):
        super().__init__(dataset_len, subset_percentage, generator)
        self.distances = torch.zeros(self.dataset_len)
        # assign uniform random values from 0 to 1 for now
        self.distances = torch.rand(self.dataset_len)
        self.loss_p=0.9
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
    def __init__(self, dataset_len, subset_percentage, generator=None):
        super().__init__(dataset_len, subset_percentage, generator)
        self.distances = torch.zeros(self.dataset_len)
        self.distances = torch.rand(self.dataset_len)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None
        self.distance_pref = 0.001
        self.dist_shift = 0.02
        self.noise = 0.2

    def get_indices(self):
        dist_from_pref = torch.abs(self.distances - self.distance_pref) 
        dist_from_pref += (torch.rand(self.dataset_len) - 0.5) * self.noise 
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
        
        if avg_loss < self.avg_loss:
            self.distance_pref += self.dist_shift
        else:
            self.distance_pref -= self.dist_shift

