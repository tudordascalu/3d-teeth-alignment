"""
Rate inter-tooth distances based on training set statistics (mean, std).
"""
import glob

import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from scripts.utils import arg_parser


class ScoreMapper:
    def __init__(self, n_teeth=17):
        """

        :param n_teeth: controls how many double teeth should be accounted for
        """
        self.n_teeth = n_teeth

    def __call__(self, distance_map, distance_map_mean, distance_map_cov):
        """

        :param distance_map: np.array of shape (17, 1) featuring labels for each instance
        :param distance_map_mean: np.array of shape (17, 17) featuring mean distances between tooth-tooth pairs
        :param distance_map_std: np.array of shape (17, 17) featuring stds for tooth-tooth pair distances
        :return: np.array of shape (17, 17) denoting tooth-tooth probabilities based on distances
        """
        score_map = np.zeros((self.n_teeth, self.n_teeth, 1))
        # Navigate through all instances [0, 17]
        for i in range(self.n_teeth):
            for j in range(self.n_teeth):
                distance = distance_map[i, j]
                distance_mean = distance_map_mean[i, j]
                distance_cov = distance_map_cov[i, j]
                try:
                    score = multivariate_normal.pdf(distance, mean=distance_mean, cov=distance_cov)
                except:
                    score = 0
                score_map[i, j] = score
        return score_map


if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = args.jaw
    n_teeth = args.teeth
    # Load statistics
    distance_map_mean = np.load(f"../data/statistics/distance_map_mean_{jaw}.npy")
    distance_map_cov = np.load(f"../data/statistics/distance_map_cov_{jaw}.npy")
    score_mapper = ScoreMapper(n_teeth)
    # Compute ids
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/final/*")))
    for id in tqdm(ids, total=len(ids)):
        distance_map = np.load(f"../data/final/{id}/distance_map_{jaw}.npy")
        score_map = score_mapper(distance_map, distance_map_mean, distance_map_cov)
        np.save(f"../data/final/{id}/score_map_{jaw}.npy", score_map)
