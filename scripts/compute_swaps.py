"""
Perform swaps and slight nudges to the centroid within 1 standard deviation in X, Y, Z.
This should run following centroid computation.
"""
import glob
import os

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser
from scripts.utils.tooth_swapper import ToothSwapper

if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    print(args)
    jaw = args.jaw
    max_swaps = args.swaps
    n = args.n_samples
    n_teeth = args.teeth

    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    tooth_swapper = ToothSwapper(n_teeth=n_teeth, neighbor_range=2, max_swaps=max_swaps)
    for id in tqdm(ids, total=len(ids)):
        for i in range(n):
            centroids = np.load(f"../data/processed/{id}/centroids_augmented_{jaw}.npy")
            centroids, labels = tooth_swapper(centroids, np.arange(0, n_teeth))
            if not os.path.exists(f"../data/final/{id}"):
                os.mkdir(f"../data/final/{id}")
            np.save(f"../data/final/{id}/centroids_{jaw}_{i}.npy", centroids)
            np.save(f"../data/final/{id}/labels_{jaw}_{i}.npy", labels)
