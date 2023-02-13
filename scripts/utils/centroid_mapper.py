import numpy as np


class CentroidMapper:
    """
    Maps centroids given a set of labels.
    """

    def __init__(self, n_teeth=17):
        """
        :param n_teeth: number of teeth
        """
        self.n_teeth = n_teeth

    def __call__(self, vertices, tooth_labels):
        """
        Computes the centroid of each tooth instance.

        :param vertices: coordinates of all vertices
        :param tooth_labels: tooth labels for each vertex encoded from 0-16; 16 corresponds to double tooth;
        :return: np.array of shape (n_teeth, 3) featuring the centroid of each tooth;
         If tooth is missing, centroid coordinates are (0, 0, 0)
        """
        tooth_labels_unique = np.unique(tooth_labels)
        centroids = np.zeros((self.n_teeth, 3))
        for tooth_label in tooth_labels_unique:
            centroids[tooth_label] = np.mean(vertices[np.where(tooth_labels == tooth_label)], axis=0)
        return centroids
