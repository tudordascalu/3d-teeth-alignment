"""
This script fits PCA to all centroids of a jaw and saves 3 points belonging to the axial plane.
"""
import glob
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from scripts.utils import arg_parser


class AxialPlaneDetector:
    def __init__(self):
        pass

    def __call__(self, centroids):
        centroids_mean = np.mean(centroids, axis=0)
        pca = PCA()
        pca.fit(centroids)
        pcs, variance = pca.components_, pca.explained_variance_
        return pcs[0], pcs[1], centroids_mean, variance


if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    print(args)
    jaw = args.jaw
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/processed/*")))
    centroids_acc = []
    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/processed/{id}/centroids_{jaw}.npy")
        centroids_acc.append(centroids)
    centroids_acc = np.concatenate(centroids_acc, axis=0)
    centroids_acc = centroids_acc[np.where((centroids_acc != np.array([0, 0, 0])).all(axis=1))[0]]
    axial_plane_detector = AxialPlaneDetector()
    pc1, pc2, mean, variance = axial_plane_detector(centroids_acc)
    np.save(f"../data/statistics/axial_plane_{jaw}.npy", np.array([pc1, pc2, mean]))
    # variance = np.sqrt(variance)
    # fig = px.scatter_3d(x=centroids_acc[:, 0], y=centroids_acc[:, 1], z=centroids_acc[:, 2])
    # fig.add_trace(px.line_3d(x=[mean[0], mean[0] + pc1[0] * variance[0]],
    #                          y=[mean[1], mean[1] + pc1[1] * variance[0]],
    #                          z=[mean[2], mean[2] + pc1[2] * variance[0]]).data[0])
    # fig.add_trace(px.line_3d(x=[mean[0], mean[0] + pc2[0] * variance[1]],
    #                          y=[mean[1], mean[1] + pc2[1] * variance[1]],
    #                          z=[mean[2], mean[2] + pc2[2] * variance[1]]).data[0])
    # fig.show()
