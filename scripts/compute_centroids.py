"""
The role of this script is to process label files and output centroids.
"""
import glob
import json
import os

import numpy as np
import trimesh
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from scripts.utils import arg_parser


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
        :param tooth_labels: tooth labels for each vertex encoded from 0-17; 0 corresponds to gums, 17 corresponds to double tooth;
        :return: np.array of shape (n_teeth, 3) featuring the centroid of each tooth;
         If tooth is missing, centroid coordinates are (0, 0, 0)
        """
        # Compute unique tooth labels
        tooth_labels_unique = np.unique(tooth_labels)
        tooth_labels_unique = np.delete(tooth_labels_unique, tooth_labels_unique == 0)
        centroids = np.zeros((self.n_teeth, 3))
        for tooth_label in tooth_labels_unique:
            centroids[tooth_label - 1] = np.mean(vertices[np.where(tooth_labels == tooth_label)], axis=0)
        return centroids


class ToothLabelEncoder:
    def __init__(self):
        pass

    @staticmethod
    def encoder(jaw):
        if jaw == "lower":
            tooth_labels = [0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]  # lower
        elif jaw == "upper":
            tooth_labels = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]  # upper
        else:
            raise ValueError("jaw must be either 'lower' or 'upper'")
        encoder = LabelEncoder()
        encoder.fit(tooth_labels)
        return encoder


if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = args.jaw

    encoder = ToothLabelEncoder.encoder(jaw)
    centroid_mapper = CentroidMapper(17)
    # Compute patient ids
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/raw/patient_labels/*")))
    # Iterate over each patient id.
    for id in tqdm(ids, total=len(ids)):
        # Try to load the patient's jaw mesh in .obj format. If it fails, try to load it in .stl format
        try:
            mesh = trimesh.load(f"../data/raw/patient_obj/{id}/{id}_{jaw}.obj", process=False)
        except:
            mesh = trimesh.load(f"../data/raw/patient_stl/{id}/{id}_{jaw}.stl", process=False)
        # Load the patient's labels, which are stored in a .json file
        with open(f"../data/raw/patient_labels/{id}/{id}_{jaw}.json", "r") as f:
            labels = json.load(f)
        # Convert the labels to numpy arrays.
        instance_labels = np.array(labels["instances"])
        tooth_labels = np.array(labels["labels"])
        tooth_labels = encoder.transform(tooth_labels)
        # Find dummy tooth and assign it label 17
        tooth_labels_unique = np.unique(tooth_labels)
        # Iterate over each unique label
        for tooth_label in tooth_labels_unique:
            tooth_instance_labels = instance_labels[tooth_labels == tooth_label]
            tooth_instance_labels_unique = np.unique(tooth_instance_labels)
            # If there are exactly two unique instance labels, it means there is a dummy tooth
            # Assign the label 17 to the dummy tooth
            if len(tooth_instance_labels_unique) == 2:
                tooth_labels[instance_labels == tooth_instance_labels_unique[-1]] = 17
        # Find the tooth centroids of the jaw mesh, given the vertices
        centroids = centroid_mapper(mesh.vertices, tooth_labels)
        if not os.path.exists(f"../data/processed/{id}"):
            os.mkdir(f"../data/processed/{id}")
        # Save the tooth centroids
        np.save(f"../data/processed/{id}/centroids_{jaw}.npy", centroids)
