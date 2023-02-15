"""
Perform swaps and slight nudges to the centroid within 1 standard deviation in X, Y, Z.
This should run following centroid computation.
"""
import glob
import os

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser


class ToothSwapper:
    def __init__(self, n_teeth=17, neighbor_range=2, max_swaps=2):
        """

        :param max_swaps: determines the maximum number of swaps performed for a give tooth
        """
        self.max_swaps = max_swaps
        self.n_teeth = n_teeth
        self.neighbor_range = neighbor_range

    def __call__(self, centroids, labels):
        """

        :param centroids: array of centroids
        :param labels: array of labels, where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        i_swapped_acc = []
        n_swaps = np.random.randint(0, self.max_swaps + 1)
        for _ in range(n_swaps):
            i_list = np.arange(0, self.n_teeth)
            i_list = np.delete(i_list, i_swapped_acc)
            i_1 = np.random.choice(i_list, size=1)[0]
            neighbors = np.arange(max(i_1 - self.neighbor_range, 0), min(i_1 + self.neighbor_range + 1, self.n_teeth))
            neighbors = np.delete(neighbors,
                                  np.concatenate((np.where(neighbors == i_1)[0], i_swapped_acc)).astype(np.int32))
            i_2 = np.random.choice(neighbors, size=1)[0]
            centroids, labels = self._swap(centroids, labels, i_1, i_2)
            i_swapped_acc.extend([i_1, i_2])
        return centroids, labels

    @staticmethod
    def _swap(centroids, labels, i_1, i_2):
        """
        :param centroids: array of centroids of shape (n, 3)
        :param labels: array of labels of shape (n,), where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        labels[i_1], labels[i_2] = labels[i_2], labels[i_1]
        centroids[[i_1, i_2]] = centroids[[i_2, i_1]]
        return centroids, labels


if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = args.jaw
    max_swaps = args.swaps
    n = args.n_samples
    n_teeth = args.teeth

    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    tooth_swapper = ToothSwapper(max_swaps)
    for id in tqdm(ids, total=len(ids)):
        for i in range(n):
            centroids = np.load(f"../data/processed/{id}/centroids_{jaw}.npy")
            centroids, labels = tooth_swapper(centroids, np.arange(0, n_teeth))
            if not os.path.exists(f"../data/final/{id}"):
                os.mkdir(f"../data/final/{id}")
            np.save(f"../data/final/{id}/centroids_{jaw}_{i}.npy", centroids)
            np.save(f"../data/final/{id}/labels_{jaw}_{i}.npy", labels)
