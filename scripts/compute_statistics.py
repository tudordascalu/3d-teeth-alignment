"""
Compute means and standard deviations corresponding to (x, y, z) differences between each tooth-tooth centroid difference.
"""
import numpy as np

if __name__ == "__main__":
    # Constants
    JAW = "lower"
    ids = np.load("../data/split/ids_train.npy")
    distance_map_acc = []
    for id in ids:
        distance_map = np.load(f"../data/processed/{id}/distance_map_{JAW}.npy")
        distance_map_acc.append(distance_map)
    distance_map_mean = np.mean(distance_map_acc, axis=0)
    distance_map_std = np.std(distance_map_mean, axis=0)
    np.save(f"../data/statistics/distance_map_mean_{JAW}.npy", distance_map_mean)
    np.save(f"../data/statistics/distance_map_std_{JAW}.npy", distance_map_std)
