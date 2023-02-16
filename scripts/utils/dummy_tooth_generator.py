from sklearn.decomposition import PCA

from scripts.utils.missing_teeth_detector import MissingTeethDetector
import numpy as np
import plotly.express as px


class DummyToothGenerator:
    def __init__(self, n_teeth=17, min_dist=5, max_dist=15, max_noise_dist=2):
        """

        :param axial_plane: np.array of shape (n_points, n_dim) including points belonging to the plane
        :param n_teeth:
        :param min_dist:
        :param max_dist:
        """
        self.n_teeth = n_teeth
        self.missing_teeth_detector = MissingTeethDetector()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_noise_dist = max_noise_dist
        self.axial_plane = AxialPlane()

    def __call__(self, centroids):
        # Find present teeth
        tooth_labels = np.arange(0, self.n_teeth)
        missing_tooth_labels = self.missing_teeth_detector(centroids, tooth_labels)
        tooth_labels = tooth_labels[~np.in1d(tooth_labels, missing_tooth_labels)]
        # Chose one random tooth that has neighbors both to the left and to the right
        i_tooth = np.random.choice(np.arange(1, len(tooth_labels) - 1), size=1)[0]
        # Find tooth and neighbor coordinates
        p = centroids[tooth_labels[i_tooth]]
        a = centroids[tooth_labels[i_tooth - 1]]
        b = centroids[tooth_labels[i_tooth + 1]]
        # Find axial plane
        self.axial_plane.fit(centroids[tooth_labels])
        # Project tooth and neighbor centroids on axial plane
        p = self.axial_plane.project(p)
        a = self.axial_plane.project(a)
        b = self.axial_plane.project(b)
        # Find the vector pointing in the direction perpendicular to the line formed by the neighbors
        ab = b - a
        ap = p - a
        proj_ap = np.dot(ap, ab) / np.dot(ab, ab) * ab
        intersection = a + proj_ap
        intp = (p - intersection) / np.linalg.norm(p - intersection)
        # Randomly chose how many mm away to place the tooth
        alpha = np.random.choice(np.arange(5, 10, step=.1), size=1)[0]
        # Check if tooth sits outside the line formed by the neighbors compared to the centroid
        if self.is_outer_tooth(p, self.axial_plane.plane[-1], a, b):
            dummy_tooth_centroid = p + alpha * intp
        else:
            dummy_tooth_centroid = p - alpha * intp
        # Randomly offset dummy_tooth_centroids by maximum of 1mm in the X, Y, Z axes
        offset = np.random.choice(np.arange(-self.max_noise_dist, self.max_noise_dist + .1, step=.1), size=3)
        # Set the centroid for the dummy tooth placeholder
        centroids[-1] = dummy_tooth_centroid + offset
        return centroids

    def is_outer_tooth(self, p, q, a, b):
        """
        This method assumes that the points p, q, a, b are coplanar.

        :param p: coordinates of tooth
        :param q: coordinates of centroid
        :param a: coordinates of neighbour 1
        :param b: coordinates of neigbhour 2
        :return: true if the tooth sits opposite from the centroid w.r.t the line formed by the neighbors, false otherwise
        """
        # Compute the direction of the line
        line_dir = b - a
        # Compute the cross product of the line direction and the vector from a to p (or q)
        p_cross = np.cross(line_dir, p - a)
        q_cross = np.cross(line_dir, q - a)
        # Check if the dot products of the cross products are negative
        if np.dot(p_cross, q_cross) < 0:
            return True
        else:
            return False


class AxialPlane:
    def __init__(self):
        self.plane = None

    def fit(self, centroids):
        centroids_mean = np.mean(centroids, axis=0)
        pca = PCA()
        pca.fit(centroids)
        pcs, variance = pca.components_, pca.explained_variance_
        pcs += centroids_mean
        self.plane = [pcs[0], pcs[1], centroids_mean]

    def project(self, p):
        a, b, c = self.plane
        # Calculate the normal vector of the plane
        n = np.cross(b - a, c - a)
        # Calculate the vector from point A to point P
        v = p - a
        # Calculate the projection of vector v onto the normal vector of the plane
        proj = v - np.dot(v, n) / np.dot(n, n) * n
        # Calculate the projected point Q by adding the projection of vector v to point A
        q = a + proj
        return q


if __name__ == "__main__":
    centroids = np.load("../../data/processed/0JN50XQR/centroids_lower.npy")
    # centroids = np.load("../../data/processed/0JN50XQR/centroids_augmented_lower.npy")
    # dummy_tooth_generator = DummyToothGenerator(n_teeth=17)
    # centroids = dummy_tooth_generator(centroids)
    # Visualization
    centroids = centroids[np.where((centroids != np.array([0, 0, 0])).all(axis=1))[0]]
    # a, b, c = dummy_tooth_generator.axial_plane.plane
    fig = px.scatter_3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2])
    # fig.add_trace(px.line_3d(x=[c[0], a[0]],
    #                          y=[c[1], a[1]],
    #                          z=[c[2], a[2]]).data[0])
    # fig.add_trace(px.line_3d(x=[c[0], b[0]],
    #                          y=[c[1], b[1]],
    #                          z=[c[2], b[2]]).data[0])
    fig.show()
