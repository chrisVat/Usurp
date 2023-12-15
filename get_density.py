import numpy as np
import torch
from distances import get_distance_calculator, DISTANCE_TECHNIQUES
import random
from sklearn.neighbors import KernelDensity

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True



def main():
    embeddings = np.load("embeddings\\cifar_10_trained\\train.npy")
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(embeddings)
    log_densities = kde.score_samples(embeddings)
    densities = np.exp(log_densities)  # converting log densities to actual densities
    # save densities
    np.save("embeddings\\cifar_10_trained\\densities.npy", densities)


if __name__ == "__main__":
    main()

