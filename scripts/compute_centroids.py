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
from scripts.utils.centroid_mapper import CentroidMapper
from scripts.utils.tooth_label_encoder import ToothLabelEncoder

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
        # Remove gums
        instance_labels = np.delete(instance_labels, instance_labels == 0)
        tooth_labels = np.delete(tooth_labels, tooth_labels == 0)
        tooth_labels = encoder.transform(tooth_labels)
        # Find dummy tooth and assign it label 17
        tooth_labels_unique = np.unique(tooth_labels)
        # Iterate over each unique label
        for tooth_label in tooth_labels_unique:
            tooth_instance_labels = instance_labels[tooth_labels == tooth_label]
            tooth_instance_labels_unique = np.unique(tooth_instance_labels)
            # If there are exactly two unique instance labels, it means there is a dummy tooth
            # Assign the label 16 to the dummy tooth
            if len(tooth_instance_labels_unique) == 2:
                tooth_labels[instance_labels == tooth_instance_labels_unique[-1]] = n_teeth - 1
        # Find the tooth centroids of the jaw mesh, given the vertices
        centroids = centroid_mapper(mesh.vertices, tooth_labels)
        if not os.path.exists(f"../data/processed/{id}"):
            os.mkdir(f"../data/processed/{id}")
        # Save the tooth centroids
        np.save(f"../data/processed/{id}/centroids_{jaw}.npy", centroids)
