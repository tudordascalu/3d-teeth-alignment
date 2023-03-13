"""
This scripts navigates the test set and compares the distances compares the label of the centroids computed on the U-net
outputs with the labels of their closest counterparts in the gold standard anotation.
"""
import os

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser

if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    ids = np.load("../data/split/ids_test.npy")
    centroid_missing = np.zeros(3)

    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/processed/{id}/centroids_true_{args.jaw}.npy")
        centroids_pred = np.load(f"../data/processed/{id}/centroids_{args.jaw}.npy")
        if not os.path.exists(f"../data/processed/{id}/centroids_pred_{args.jaw}.npy"):
            np.save(f"../data/processed/{id}/centroids_pred_{args.jaw}.npy", centroids_pred)
        for label, centroid in enumerate(centroids):
            dif = np.linalg.norm(centroids_pred - centroid, axis=1)
            label_pred = np.argmin(dif)
            dist, centroid_pred = dif[label_pred], centroids_pred[label_pred]
            # Missing tooth
            if np.array_equal(centroid, centroid_missing):
                # Do nothing
                pass
            # Close enough to the actual centroid, but not missing
            elif dist <= args.max_distance and not np.array_equal(centroid_pred, centroid_missing):
                centroids[label] = centroid_pred
                centroids_pred[label_pred] = centroid_missing
            else:
                centroids[label] = centroid_missing
        np.save(f"../data/processed/{id}/centroids_{args.jaw}.npy", centroids)
