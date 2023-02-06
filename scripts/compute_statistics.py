"""
Compute means and standard deviations for each coordonate in the vertex coordinates for each tooth centroid based on the
training set.
"""
import numpy as np

if __name__ == "__main__":
    # Constants
    JAW = "upper"
    ids = np.load("../data/split/ids_train.npy")
    centroids_acc = []
    for id in ids:
        centroids = np.load(f"../data/processed/{id}/centroids_{JAW}.npy")
        centroids_acc.append(centroids)
    centroids_acc = np.array(centroids_acc)
    centroids_mean = np.mean(centroids_acc, axis=0)
    centroids_std = np.std(centroids_acc, axis=0)
    np.save(f"../data/statistics/centroids_mean_{JAW}.npy", centroids_mean)
    np.save(f"../data/statistics/centroids_std_{JAW}.npy", centroids_std)