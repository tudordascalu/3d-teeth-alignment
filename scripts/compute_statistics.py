"""
Compute means and standard deviations corresponding to (x, y, z) differences between each tooth-tooth centroid difference.
"""
import numpy as np

from scripts.utils import arg_parser


class DistanceMapCov:
    def __init__(self):
        pass

    def __call__(self, distance_map_acc):
        """

            :param distance_map_acc: np.array of shape (n_patients, 17, 17, 3)
            :return: distance_map_cov of shape np.array((17, 17, 3, 3))
            """
        distance_map_cov = np.zeros((17, 17, 3, 3))
        for i in range(17):
            for j in range(17):
                tooth_tooth_distance_map = distance_map_acc[:, i, j, ...].T
                distance_map_cov[i, j] = np.cov(tooth_tooth_distance_map)
        return distance_map_cov


if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = args.jaw
    n_teeth = args.teeth
    ids = np.load("../data/split/ids_train.npy")
    distance_map_cov_mapper = DistanceMapCov()
    distance_map_acc = []
    for id in ids:
        distance_map = np.load(f"../data/processed/{id}/distance_map_{jaw}.npy")
        distance_map_acc.append(distance_map)
    distance_map_acc = np.array(distance_map_acc)
    distance_dict = {}
    distance_map_mean = np.zeros((n_teeth, n_teeth, 3))
    distance_map_std = np.zeros((n_teeth, n_teeth, 3))
    for i in range(17):
        for j in range(17):
            distances = distance_map_acc[:, i, j, ...]
            distances = distances[(distances != np.array([0, 0, 0])).all(axis=1)]
            if len(distances) > 0:
                distance_map_mean[i, j] = np.mean(distances, axis=0)
                distance_map_std[i, j] = np.std(distances, axis=0)
    # Uncomment the following in order to fit a custom distribution (e.g Laplace) to the data
    # distance_map_mean_std_mapper = DistanceMapMeanStdMapper(dist=laplace, n_teeth=n_teeth)
    # distance_map_mean, distance_map_std = distance_map_mean_std_mapper(distance_map_acc)
    # distance_map_cov = distance_map_cov_mapper(distance_map_acc)
    np.save(f"../data/statistics/distance_map_mean_{jaw}.npy", distance_map_mean)
    np.save(f"../data/statistics/distance_map_std_{jaw}.npy", distance_map_std)
    # np.save(f"../data/statistics/distance_map_cov_{jaw}.npy", distance_map_cov)
