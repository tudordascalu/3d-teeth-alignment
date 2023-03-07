import numpy as np


class MissingTeethDetector:
    def __init__(self):
        pass

    def __call__(self, centroids, labels):
        return labels[np.where((centroids == np.array([0, 0, 0])).all(axis=1))[0]]


class MissingTeethRemover:
    def __init__(self):
        self.missing_teeth_detector = MissingTeethDetector()

    def __call__(self, centroids, labels):
        labels_missing = self.missing_teeth_detector(centroids, labels)
        return labels[~np.isin(labels, labels_missing)]
