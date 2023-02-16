"""
The role of this script is to process label files and output centroids.
"""
import glob
import json
import os

import numpy as np
import trimesh
from tqdm import tqdm

from scripts.utils import arg_parser
from scripts.utils.centroid_mapper import CentroidMapper
from scripts.utils.tooth_label_encoder import ToothLabelEncoder


class GumRemover:
    def __init__(self, gum_cls=0):
        self.gum_cls = gum_cls

    def __call__(self, tooth_labels, instance_labels, vertices):
        """

        :param tooth_labels: np.array of shape (n_vertices,)
        :param isntance_labels: np.array of shape (n_vertices,)
        :param vertices: np.array of shape (n_vertices, 3)
        :return: tooth_labels_processed, instance_labels_processed, vertices_processed
        """
        i_gum_list = tooth_labels == self.gum_cls
        tooth_labels_processed = np.delete(tooth_labels, i_gum_list)
        instance_labels_processed = np.delete(instance_labels, i_gum_list)
        vertices_processed = np.delete(vertices, i_gum_list, axis=0)
        return tooth_labels_processed, instance_labels_processed, vertices_processed


class DummyToothDetector:
    def __init__(self, dummy_cls=16):
        self.dummy_cls = dummy_cls

    def __call__(self, tooth_labels, instance_labels):
        """

        :param tooth_labels: np.array of shape (n_vertices,)
        :param instance_labels: np.array of shape (n_vertices,)
        :return: tooth_labels_processed
        """
        # Find dummy tooth and assign it label 17
        tooth_labels_unique = np.unique(tooth_labels)
        # Iterate over each unique label
        for tooth_label in tooth_labels_unique:
            tooth_instance_labels = instance_labels[tooth_labels == tooth_label]
            tooth_instance_labels_unique = np.unique(tooth_instance_labels)
            # If there are exactly two unique instance labels, it means there is a dummy tooth
            if len(tooth_instance_labels_unique) == 2:
                tooth_labels[instance_labels == tooth_instance_labels_unique[-1]] = self.dummy_cls
        return tooth_labels


if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    jaw = "lower"
    n_teeth = args.teeth
    encoder = ToothLabelEncoder.encoder(jaw)
    centroid_mapper = CentroidMapper(n_teeth)
    # Compute patient ids
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/raw/patient_labels/*")))
    ids_test = np.load("../data/split/ids_test.npy")
    # Iterate over each patient id.
    for id in tqdm(ids, total=len(ids)):
        # Try to load the patient's jaw mesh in .obj format. If it fails, try to load it in .stl format
        try:
            mesh = trimesh.load(f"../data/raw/patient_obj/{id}/{id}_{jaw}.obj", process=False)
        except:
            mesh = trimesh.load(f"../data/raw/patient_stl/{id}/{id}_{jaw}.stl", process=False)
        # Load the patient's labels, which are stored in a .json file
        if id in ids_test:
            with open(f"../data/raw/predicted_patient_labels/{id}_{jaw}.json", "r") as f:
                labels = json.load(f)
        else:
            with open(f"../data/raw/patient_labels/{id}/{id}_{jaw}.json", "r") as f:
                labels = json.load(f)
        # Convert the labels to numpy arrays.
        instance_labels = np.array(labels["instances"])
        tooth_labels = np.array(labels["labels"])
        # Remove gums
        gum_remover = GumRemover(0)
        tooth_labels, instance_labels, vertices = gum_remover(tooth_labels, instance_labels, mesh.vertices)
        # Encode labels
        tooth_labels = encoder.transform(tooth_labels)
        # Overwrite dummy tooth
        dummy_tooth_detector = DummyToothDetector(n_teeth - 1)
        tooth_labels = dummy_tooth_detector(tooth_labels, instance_labels)
        # Find the tooth centroids of the jaw mesh, given the vertices
        centroids = centroid_mapper(vertices, tooth_labels)
        if not os.path.exists(f"../data/processed/{id}"):
            os.mkdir(f"../data/processed/{id}")
        # Save the tooth centroids
        np.save(f"../data/processed/{id}/centroids_{jaw}.npy", centroids)
