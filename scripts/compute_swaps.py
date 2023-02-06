"""
Perform swaps and slight nudges to the centroid within 1 standard deviation in X, Y, Z.
"""
import glob

import numpy as np


class ToothSwapper:
    def __init__(self, max_swaps=2):
        """

        :param max_swaps: determines the maximum number of swaps performed for a give tooth
        """
        self.max_swaps = max_swaps

    def __call__(self, centroids, labels):
        """

        :param centroids: array of centroids
        :param labels: array of labels, where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        for _ in np.arange(np.random.randint(0, self.max_swaps + 1)):
            centroids, labels = self._swap(centroids, labels)
        return centroids, labels

    @staticmethod
    def _swap(centroids, labels):
        """
        :param centroids: array of centroids of shape (n, 3)
        :param labels: array of labels of shape (n,), where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        i_1, i_2 = np.random.choice(np.unique(labels), size=2, replace=False)
        labels[i_1], labels[i_2] = labels[i_2], labels[i_1]
        centroids[[i_1, i_2]] = centroids[[i_2, i_1]]
        return centroids, labels


if __name__ == "__main__":
    # Constants
    JAW = "upper"
    MAX_SWAPS = 3
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    tooth_swapper = ToothSwapper(MAX_SWAPS)
    for id in ids:
        centroids = np.load(f"../data/processed/{id}/centroids_{JAW}.npy")
        centroids, labels = tooth_swapper(centroids, np.arange(0, 17))