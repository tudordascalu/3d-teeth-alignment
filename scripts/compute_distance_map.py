"""
Computes the distance between each tooth-tooth pair. This should run following swap.
"""
import glob

import numpy as np
from tqdm import tqdm

from scripts.utils.distance_mapper import DistanceMapper

if __name__ == "__main__":
    DIR = "processed"  # "final" | "processed"
    JAW = "upper"
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    distance_mapper = DistanceMapper()
    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/{DIR}/{id}/centroids_{JAW}.npy")
        try:
            labels = np.load(f"../data/{DIR}/{id}/labels_{JAW}.npy")
        except:
            labels = np.arange(17)
        distance_map = distance_mapper(centroids, labels)
        np.save(f"../data/{DIR}/{id}/distance_map_{JAW}.npy", distance_map)
