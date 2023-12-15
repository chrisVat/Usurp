# abstract class
from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity
import torch
import numpy as np
import random
# custom sampler passed to dataloader to get subset of dataset

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SequentialSampler

SAMPLER_TECHNIQUES = ["random", "mtds", "mds", "ltds", "ads", "sds", "srds", "lsds", "smds", "srbds", "bsmds", "bsds", "blsds", "bbsds", "busds", "bdps", "den", "dens", "bues", "forget", "forget_small", "forget_easy"]
__all__ = ["get_sampler", "SAMPLER_TECHNIQUES"]


def get_sampler(technique, dataset_len, subset_percentage, distance_path, labels, generator=None):
    # assert technique in SAMPLER_TECHNIQUES, f"Technique {technique} not supported. Choose from {SAMPLER_TECHNIQUES}"
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
    # largest
    elif technique == "sds":
        return StaticDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    # smallest
    elif technique == "lsds":
        return LargestStaticDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "srds":
        return StaticRandomDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "smds":
        return StaticMiddleDistanceSampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique == "srbds":
        return StaticRandomBalancedDistanceSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique.startswith("bsmds"):
        percentage=float(technique.split("_")[1])
        if len(technique.split("_")) > 1:
            percentage=float(technique.split("_")[1])
        return BalancedPercentileSampler(dataset_len, subset_percentage, distance_path, labels, percentage, generator)
    elif technique == "bsds":
        return BalancedStaticDistanceSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique == "blsds":
        return BalancedLargestStaticDistanceSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique == "bbsds":
        return BalancedBiasSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique == "busds":
        return BalancedUniformSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique.startswith("bdps"):
        percentage1=float(technique.split("_")[1])
        percentage2=float(technique.split("_")[2])
        return BalancedDoublePercentileSampler(dataset_len, subset_percentage, distance_path, labels, percentage1, percentage2, generator)
    elif technique.lower() == "den":
        return DensitySampler(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "dens":
        return DensitySamplerNoShuffle(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "bues":
        return BalancedUniformEvenSampler(dataset_len, subset_percentage, distance_path, labels, generator)
    elif technique.lower() == "forget":
        return ForgettingScore(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "forget_small":
        return ForgettingSmallScore(dataset_len, subset_percentage, distance_path, generator)
    elif technique.lower() == "forget_easy":
        return ForgettingEasyScore(dataset_len, subset_percentage, distance_path, generator)
    else:
        raise Exception("Sampler technique not supported")


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
        self.noise = 0.4

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
        self.noise = 0.4

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


class LargestStaticDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None        
        # argsort distances 
        self.ind = torch.argsort(self.distances)
        self.ind = self.ind[-self.subset_len:]
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class StaticMiddleDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None        
        # argsort distances 
        margin = int((1-self.subset_percentage)/2 * self.dataset_len)
        self.ind = torch.argsort(self.distances)
        self.ind = self.ind[margin:-margin]
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)

class BalancedPercentileSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, percent=0.5, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.percentage = percent

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))


        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            middle = int(len(sorted_dists) * self.percentage)
            desired_idxs = list(sorted_dists[int(middle-amount_per_class/2):int(middle+amount_per_class/2)])
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])

        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        # print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class BalancedDoublePercentileSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, percent1=0.3, percent2=0.7, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.percentage1 = percent1
        self.percentage2 = percent2
        self.percentages = [self.percentage1, self.percentage2]

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))


        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            for percentage in self.percentages:
                middle = int(len(sorted_dists) * percentage)
                desired_idxs = list(sorted_dists[int(middle-amount_per_class/4):int(middle+amount_per_class/4)])
                for idx in desired_idxs:
                    self.ind.append(data_idxs[idx])

        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        # print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class BalancedUniformSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, percent=0.5, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))


        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            desired_idxs = []
            # uniformly sample from the sorted distances
            while len(desired_idxs) < amount_per_class:
                rand_idx = random.randint(0, len(sorted_dists)-1)
                if rand_idx not in desired_idxs:
                    desired_idxs.append(rand_idx)
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])

        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        # print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class BalancedUniformEvenSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, percent=0.5, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))


        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            desired_idxs = []
            # uniformly sample from the sorted distances
            n = len(sorted_dists) / amount_per_class
            for i in range(amount_per_class):
                if int(i*n) not in desired_idxs:
                    desired_idxs.append(int(i*n))
            while len(desired_idxs) < amount_per_class:
                # normal distribution
                rand_idx = int(np.random.normal(0.5, 0.25) * len(sorted_dists))
                if rand_idx not in desired_idxs and rand_idx >= 0 and rand_idx < len(sorted_dists):
                    desired_idxs.append(rand_idx) 
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])

        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        # print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)

class BalancedBiasSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, percent=0.5, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))


        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            desired_idxs = []
            # normal sample with median at 0.33 from the sorted distances
            while len(desired_idxs) < amount_per_class:
                rand_idx = int(np.random.normal(0.33, 0.5) * len(sorted_dists))
                if rand_idx < 0 or rand_idx >= len(sorted_dists):
                    continue
                if rand_idx not in desired_idxs:
                    desired_idxs.append(rand_idx)
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])

        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        # print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class StaticDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None        
        # argsort distances 
        self.ind = torch.argsort(self.distances)
        self.ind = self.ind[:self.subset_len]
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)



class BalancedLargestStaticDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))

        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            desired_idxs = sorted_dists[-amount_per_class:]
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])
        
        
        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        print("Counts per class", counts_per_class)
        random.shuffle(self.ind)
            

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class BalancedStaticDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None     
        self.original_labels = labels

        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        print("Amount per class: ", amount_per_class)
        print("Subset percentage", self.subset_percentage)
        print("Dataset len", self.dataset_len)
        print("Subset len", self.subset_len)
        print("Unique labels", len(np.unique(self.original_labels)))

        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            related_dists = self.distances[data_idxs]
            sorted_dists = torch.argsort(related_dists)
            desired_idxs = sorted_dists[:amount_per_class]
            for idx in desired_idxs:
                self.ind.append(data_idxs[idx])
    
        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        print("Counts per class", counts_per_class)
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class StaticRandomDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None        
        # Random distance sampling
        self.ind = torch.randperm(self.dataset_len)[:self.subset_len]
        self.ind = self.ind.tolist()
        random.shuffle(self.ind)

    def get_indices(self):
        return self.ind
    

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class StaticRandomBalancedDistanceSampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", labels=None, generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.loss_p=0.9
        self.avg_loss = None
        self.ind = None 
        self.original_labels = labels

        # get indices per class
        self.data_idxs_per_class = []
        for i in range(len(np.unique(self.original_labels))):
            self.data_idxs_per_class.append(np.where(self.original_labels == i)[0])
        self.data_idxs_per_class = np.array(self.data_idxs_per_class)

        # randomly sample from each class 
        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            self.ind.append(torch.randperm(len(data_idxs), generator=self.generator)[:int(self.subset_len/len(np.unique(self.original_labels)))])
        self.ind = torch.cat(self.ind)
        random.shuffle(self.ind)


        amount_per_class = int(self.subset_len/len(np.unique(self.original_labels)))
        self.ind = []
        for data_idxs in self.data_idxs_per_class:
            selections = torch.randperm(len(data_idxs))[:amount_per_class]
            for selection in selections:
                self.ind.append(data_idxs[selection])
    
        counts_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for idx in self.ind:
            counts_per_class[self.original_labels[idx]] += 1
        print("Counts per class", counts_per_class)
        self.ind = self.ind.tolist()
        random.shuffle(self.ind)


    def get_indices(self):
        return self.ind

    # accuracy / loss per example
    def feedback(self, feedback):
        random.shuffle(self.ind)


class DensitySampler(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.densities = np.load("embeddings\\cifar_10_trained\\densities.npy")
        density_threshold = np.percentile(self.densities, (1 - self.subset_percentage)*100)        
        self.ind = np.where(self.densities > density_threshold)[0]       # Indices of high density samples

    def get_indices(self):
        return self.ind

    def feedback(self, feedback):
        random.shuffle(self.ind)


class DensitySamplerNoShuffle(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        self.densities = np.load("embeddings\\cifar_10_trained\\densities.npy")
        density_threshold = np.percentile(self.densities, (1 - self.subset_percentage)*100)        
        self.ind = np.where(self.densities > density_threshold)[0]       # Indices of high density samples

    def get_indices(self):
        return self.ind

    def feedback(self, feedback):
        pass

class ForgettingScore(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        forgetting_path = "forgetting/c10_forgetting_indices.npy"
        self.forgetting_indices = np.load(forgetting_path)
        self.ind = self.forgetting_indices[-int(self.subset_len):]
    
    def get_indices(self):
        return self.ind
    
    def feedback(self, feedback):
        random.shuffle(self.ind)


class ForgettingSmallScore(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        forgetting_path = "forgetting/c10_forgetting_indices.npy"
        self.first_nonzero = 23841 
        self.forgetting_indices = np.load(forgetting_path)
        self.ind = []
        if self.subset_len >= self.first_nonzero:
            self.ind = self.forgetting_indices[:self.subset_len]
        else: 
            possible_indices = [i for i in range(self.first_nonzero)]
            while len(self.ind) < self.subset_len:
                rand_idx = random.randint(0, len(possible_indices)-1)
                self.ind.append(possible_indices[rand_idx])
                possible_indices.pop(rand_idx)
    
    def get_indices(self):
        return self.ind
    
    def feedback(self, feedback):
        random.shuffle(self.ind)


class ForgettingEasyScore(SubsetSampler):
    def __init__(self, dataset_len, subset_percentage, distance_path="embeddings/cifar_10_trained/train.npy", generator=None):
        super().__init__(dataset_len, subset_percentage, distance_path, generator)
        print("This ran.")
        forgetting_path = "forgetting/c10_forgetting_indices.npy"
        self.first_nonzero = 23841 
        self.forgetting_indices = np.load(forgetting_path)
        total_scores = len(self.forgetting_indices)
        self.ind = self.forgetting_indices[self.first_nonzero:min(self.first_nonzero + self.subset_len, total_scores-1)]
        self.ind = self.ind.tolist()
        print("Subset len", self.subset_len)
        print("Added: ", len(self.ind))
        remaining = [i for i in range(self.first_nonzero)]
        while len(self.ind) < self.subset_len:
            # select and remove element from remaining
            rand_idx = random.randint(0, len(remaining)-1)
            self.ind.append(remaining[rand_idx])
            remaining.pop(rand_idx) 
    
    def get_indices(self):
        return self.ind
    
    def feedback(self, feedback):
        random.shuffle(self.ind)
