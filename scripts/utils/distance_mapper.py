import numpy as np


class DistanceMapper:
    """
    Scope:
        Computes the distance map between teeth instances. The current implementation supports one double tooth.
    """

    def __init__(self, n_teeth=17):
        """
        :param n_teeth: number of teeth
        """
        self.n_teeth = n_teeth

    def __call__(self, centroids, tooth_labels):
        """
        Computes distances between centroids for identified teeeth instances. Leaves distance row of 0s for non-existend instances.

        :param centroids: np.array of shape (n_teeth, 3), featuring tooth centroids
        :return: np.array of shape (n_teeth, n_teeth, 3),  denoting the offsets in x, y, z direction between each
            tooth-tooth pair
        """
        distance_map = np.zeros((self.n_teeth, self.n_teeth, 3))
        # For each tooth, compute the distance to all other present teeth
        for tooth_label in tooth_labels:
            distance_map[tooth_label, tooth_labels] = centroids[tooth_label] - centroids[tooth_labels]
        # Set scores to 0s for missing teeth
        missing_teeth = np.where((centroids == np.array([0, 0, 0])).all(axis=1))[0]
        distance_map[missing_teeth] = np.zeros((self.n_teeth, 3))
        distance_map[:, missing_teeth] = np.zeros((self.n_teeth, len(missing_teeth), 3))
        return distance_map
