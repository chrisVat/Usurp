# abstract class
from abc import ABC, abstractmethod
import torch
import numpy as np
# custom sampler passed to dataloader to get subset of dataset

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler

SAMPLER_TECHNIQUES = ["random", "mtds", "mds", "ltds", "ads"]
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
        
    @abstractmethod
    def get_indices(self, indice_list):
        pass

    @abstractmethod
    def feedback(self, feedback):
        pass


class RandomSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)

    def get_indices(self):
        return torch.randperm(self.dataset_len, generator=self.generator)[:self.subset_len]

    def feedback(self, feedback):
        pass


class MovingDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
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


