from scripts.utils.missing_teeth_detector import MissingTeethDetector
import numpy as np


class DummyToothGenerator:
    def __init__(self, n_teeth=17, min_dist=5, max_dist=15):
        self.n_teeth = n_teeth
        self.missing_teeth_detector = MissingTeethDetector()
        self.min_dist = min_dist
        self.max_dist = max_dist

    def __call__(self, centroids):
        # Find present teeth
        tooth_labels = np.arange(0, self.n_teeth)
        missing_tooth_labels = self.missing_teeth_detector(centroids, tooth_labels)
        tooth_labels = tooth_labels[~np.in1d(tooth_labels, missing_tooth_labels)]
        # Chose one random tooth that has neighbors both to the left and to the right
        # i_tooth = np.random.choice(np.arange(1, len(tooth_labels) - 1), size=1)[0]
        i_tooth = 4
        tooth = tooth_labels[i_tooth]
        neighbor_l = tooth_labels[i_tooth - 1]
        neighbor_r = tooth_labels[i_tooth + 1]
        print(f"tooth {tooth}, neighbor_l {neighbor_l}, neighbor_r {neighbor_r}")
        # Find the coordinates of the vectors corresponding to the centroids of the tooth, neighbor_l and neighbor_r
        tooth_centroid = centroids[tooth]
        neighbor_l_centroid = centroids[neighbor_l]
        neighbor_r_centroid = centroids[neighbor_r]
        # Find the vector unit vector pointing along the line defined by neighbor_r_centroid-neighbor_l_centroid
        u = (neighbor_r_centroid - neighbor_l_centroid) / np.linalg.norm((neighbor_r_centroid - neighbor_l_centroid))
        # Find the vector resulting from projecting the tooth_centroid vector onto the unit vector u
        v = np.dot(u, tooth_centroid) * u
        # Find the unit vector that is perpendicular to v passing through the tooth_centroid
        w = (tooth_centroid - v) / np.linalg.norm(tooth_centroid - v)
        # Generate dummy_tooth_centroid by moving the tooth_centroid by alpha mm in the direction of w
        # alpha = np.random.choice(np.arange(self.min_dist, self.max_dist, step=.1), size=1)[0]
        alpha = 5
        print(f"alpha  {alpha}")
        dummy_tooth_centroid = tooth_centroid + alpha * w
        centroids[-1] = dummy_tooth_centroid
        # # Visualize stuff
        dummy_centroid = centroids[16]
        centroids = centroids[tooth_labels]

        # fig = px.scatter_3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2])
        # fig.add_trace(px.scatter_3d(x=[dummy_centroid[0]],
        #                             y=[dummy_centroid[1]],
        #                             z=[dummy_centroid[2]]).data[0])
        # fig.add_trace(px.line_3d(x=[neighbor_l_centroid[0], neighbor_r_centroid[0]],
        #                          y=[neighbor_l_centroid[1], neighbor_r_centroid[1]],
        #                          z=[neighbor_l_centroid[2], neighbor_r_centroid[2]]).data[0])
        # fig.add_trace(px.line_3d(x=[0, neighbor_r_centroid[0]],
        #                          y=[0, neighbor_r_centroid[1]],
        #                          z=[0, neighbor_r_centroid[2]]).data[0])
        return centroids


if __name__ == "__main__":
    centroids = np.load("../../data/processed/0EAKT1CU/centroids_lower.npy")
    dummy_tooth_generator = DummyToothGenerator(n_teeth=3)
    centroids = dummy_tooth_generator(centroids)
