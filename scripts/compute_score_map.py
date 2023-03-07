"""
Rate inter-tooth distances based on training set statistics (mean, std).
"""
import glob

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser
from scripts.utils.score_mapper import ScoreMapper

if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    print(args)
    # Load statistics
    distance_map_mean = np.load(f"../data/statistics/distance_map_mean_{args.jaw}.npy")
    distance_map_std = np.load(f"../data/statistics/distance_map_std_{args.jaw}.npy")
    score_mapper = ScoreMapper(args.teeth)
    # Compute ids
    # ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/final/*")))
    ids = np.load("../data/split/ids_test.npy")
    for id in tqdm(ids, total=len(ids)):
        for i in range(args.n_samples):
            centroids = np.load(f"../data/final/{id}/centroids_{args.jaw}_{i}.npy")
            distance_map = np.load(f"../data/final/{id}/distance_map_{args.jaw}_{i}.npy")
            score_map = score_mapper(distance_map, distance_map_mean, distance_map_std, None, centroids)
            np.save(f"../data/final/{id}/score_map_{args.jaw}_{i}.npy", score_map)
