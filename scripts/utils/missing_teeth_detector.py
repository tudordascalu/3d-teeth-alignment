import numpy as np


class MissingTeethDetector:
    def __init__(self):
        pass

    def __call__(self, centroids, labels):
        i_missing_teeth = np.where((centroids == np.array([0, 0, 0])).all(axis=1))[0]
        return labels[i_missing_teeth]
