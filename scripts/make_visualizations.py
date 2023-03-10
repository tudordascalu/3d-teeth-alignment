"""
The script creates visualizations of how the alignment network handles cases of missing and double teeth.
"""
import json
import os

import numpy as np
import trimesh
from tqdm import tqdm

from scripts.utils.mesh_saver import MeshSaver
from scripts.utils.missing_teeth_detector import MissingTeethDetector
from scripts.utils.tooth_label_encoder import ToothLabelEncoder

if __name__ == "__main__":
    # Constants
    jaw = "upper"
    mode = "correct"  # "correct" | "error"
    ext = "ply"  # "ply" | "obj"
    samples = np.load(f"../output/correct/sample_correct_acc_{jaw}.npy")[
              10:]  # [("01MAVT6A", 9)] was displayed in model summary
    # Helpers
    colors = np.load(f"../data/assets/colors.npy")
    mesh_saver = MeshSaver(colors)
    encoder = ToothLabelEncoder.encoder(jaw)
    missing_teeth_detector = MissingTeethDetector()
    centroid_missing = np.array([0, 0, 0])
    for id, i in tqdm(samples, total=len(samples)):
        if not os.path.exists(f"../output/{mode}/{id}/"):
            os.mkdir(f"../output/correct/{id}/")
        # Load data
        mesh = trimesh.load(f"../data/raw/patient_obj/{id}/{id}_{jaw}.obj", process=False)
        with open(f"../data/raw/patient_labels/{id}/{id}_{jaw}.json", "r") as f:
            labels = json.load(f)
            # Convert the labels to numpy arrays.
            tooth_labels = np.array(labels["labels"])
        with open(f"../data/raw/patient_labels_pred/{id}/{jaw}.json", "r") as f:
            labels_pred = json.load(f)
            # Convert the labels to numpy arrays.
            tooth_labels_pred = np.array(labels_pred["labels"])
        # Prepare ground truth mesh
        mesh_saver(mesh, tooth_labels, f"../output/{mode}/{id}/gt_{jaw}.{ext}")
        # Navigate all augmented instances
        # Load old labels, new labels and new centroids (following augmentation)
        old_labels = encoder.inverse_transform(np.arange(0, 17))
        new_labels = encoder.inverse_transform(np.load(f"../data/final/{id}/labels_{jaw}_{i}.npy"))
        predicted_labels = encoder.inverse_transform(
            np.load(f"../output/aligner/{id}/labels_pred_{jaw}_{i}.npy"))
        new_centroids = np.load(f"../data/final/{id}/centroids_{jaw}_{i}.npy")
        # # Generate vertex labels following augmentation
        tooth_labels_gt = np.copy(tooth_labels)
        tooth_labels_candidate = np.copy(tooth_labels_pred)
        for old_label, new_label, new_centroid in zip(old_labels, new_labels, new_centroids):
            if np.array_equal(new_centroid, centroid_missing):
                tooth_labels_gt[tooth_labels == old_label] = 0
                tooth_labels_candidate[tooth_labels_pred == old_label] = 0
            else:
                tooth_labels_candidate[tooth_labels_pred == old_label] = new_label
        # Save candidate mesh
        mesh_saver(mesh, tooth_labels_candidate, f"../output/{mode}/{id}/candidate_{jaw}_{i}.{ext}")
        # Save ground truth mesh
        mesh_saver(mesh, tooth_labels_gt, f"../output/{mode}/{id}/gt_{jaw}_{i}.{ext}")
        # Generate vertex labels following alignment
        tooth_labels_aligned = np.copy(tooth_labels_candidate)
        for old_label, predicted_label in zip(old_labels, predicted_labels):
            # TODO fix problem here, something is fishy with the interpretation
            # if np.array_equal(new_centroid, centroid_missing):
            #     pass
            # tooth_labels_aligned[tooth_labels_candidate == old_label] = 0  # TODO: fix removal of unwanted teeth
            # else:
            tooth_labels_aligned[tooth_labels_candidate == old_label] = predicted_label
        # Save aligned mesh
        mesh_saver(mesh, tooth_labels_aligned, f"../output/{mode}/{id}/aligned_{jaw}_{i}.{ext}")
