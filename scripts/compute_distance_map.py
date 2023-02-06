"""
Computes the distance between each tooth-tooth pair.
"""
import glob

import numpy as np
from tqdm import tqdm

from scripts.utils.distance_mapper import DistanceMapper

if __name__ == "__main__":
    JAW = "upper"
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    distance_mapper = DistanceMapper()
    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/final/{id}/centroids_{JAW}.npy")
        labels = np.load(f"../data/final/{id}/labels_{JAW}.npy")
        distance_map = distance_mapper(centroids, labels)
        np.save(f"../data/final/{id}/distance_map_{JAW}.npy", distance_map)