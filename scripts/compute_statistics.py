"""
Compute means and standard deviations corresponding to (x, y, z) differences between each tooth-tooth centroid difference.
"""
import numpy as np

from scripts.utils import arg_parser

if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = args.jaw
    ids = np.load("../data/split/ids_train.npy")
    distance_map_acc = []
    for id in ids:
        distance_map = np.load(f"../data/processed/{id}/distance_map_{jaw}.npy")
        distance_map_acc.append(distance_map)
    distance_map_mean = np.mean(distance_map_acc, axis=0)
    distance_map_std = np.std(distance_map_mean, axis=0)
    np.save(f"../data/statistics/distance_map_mean_{jaw}.npy", distance_map_mean)
    np.save(f"../data/statistics/distance_map_std_{jaw}.npy", distance_map_std)
