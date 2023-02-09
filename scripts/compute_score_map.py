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
    jaw = args.jaw
    n_teeth = args.teeth
    mode = args.score_mode
    # Load statistics
    distance_map_mean = np.load(f"../data/statistics/distance_map_mean_{jaw}.npy")
    distance_map_std = np.load(f"../data/statistics/distance_map_std_{jaw}.npy")
    distance_map_cov = np.load(f"../data/statistics/distance_map_cov_{jaw}.npy")
    score_mapper = ScoreMapper(n_teeth)
    # Compute ids
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/final/*")))
    for id in tqdm(ids, total=len(ids)):
        distance_map = np.load(f"../data/final/{id}/distance_map_{jaw}.npy")
        score_map = score_mapper(distance_map, distance_map_mean, distance_map_std, distance_map_cov)
        np.save(f"../data/final/{id}/score_map_{jaw}.npy", score_map)
