import numpy as np
import torchvision
import torch
from distances import get_distance_calculator, DISTANCE_TECHNIQUES
import argparse
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True



def main(args):
    cifar_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    labels = np.array(cifar_data.targets)
    num_unique_labels = len(np.unique(labels)) 

    dist_calc = get_distance_calculator(args.technique, args.embedding_path, num_unique_labels, \
        use_reduced_for_medoids=args.use_reduced_for_medoids, use_reduced_for_dist=args.use_reduced_for_dist)
    
    dist_calc.set_labels(labels)
    
    pca = dist_calc.pca(dist_calc.raw_embeddings)
    print("PCA: ", pca.shape)
    dist_calc.set_reduced_embeddings(pca) 
    
    if args.per_class:
        medoids = dist_calc.get_medoid_per_class()
    else:
        medoids = dist_calc.get_medoids()
    print("Medoids shape: ", medoids.shape)
    
    technique_name = args.technique + "_per_class" if args.per_class else args.technique

    np.save(args.save_dir + technique_name + "_medoids.npy", medoids)
    distances = dist_calc.get_distances_with_medoids(medoids)
    print("Distances: ", distances)
    np.save(args.save_dir + technique_name + "_distances.npy", distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Calculation")
    parser.add_argument("--technique", default="skmedoids", type=str, help="Clustering Technique")
    parser.add_argument("--per_class", default=True, type=bool, help="Cluster per class", choices=DISTANCE_TECHNIQUES)
    parser.add_argument("--embedding_path", default="embeddings/cifar_10_trained/train.npy", type=str, help="path to embeddings .npy file")
    parser.add_argument("--save_dir", default="embeddings/cifar_10_trained/", type=str, help="path to save medoids .npy file")
    parser.add_argument("--use_reduced_for_medoids", default=True, type=bool, help="Use PCA for medoid calculation")
    parser.add_argument("--use_reduced_for_dist", default=True, type=bool, help="Use PCA for distance calculation")
    args = parser.parse_args()   
    main(args)

