"""
Computes the distance between each tooth-tooth pair.
"""
import glob

import numpy as np

from scripts.utils.distance_mapper import DistanceMapper

if __name__ == "__main__":
    JAW = "lower"
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    distance_mapper = DistanceMapper()
    for id in ids:
        centroids = np.load(f"../data/final/{id}/centroids_{JAW}.npy")
        labels = np.load(f"../data/final/{id}/labels_{JAW}.npy")
        distance_map = distance_mapper(centroids, labels)
        break
