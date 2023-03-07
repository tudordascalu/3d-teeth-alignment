import os

import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data import AlignmentDataset
from src.model.model import AlignmentNet
from src.utils.assignment_solver import AssignmentSolver

if __name__ == "__main__":
    with open("./src/config.yml", "r") as f:
        config = yaml.safe_load(f)
    # Data
    ids = np.load("data/split/ids_test.npy")
    # Define dataset
    dataset_args = dict(data_path=config["data_path"], jaw=config["jaw"], n_samples=config["n_samples"])
    dataset = AlignmentDataset(ids=ids, **dataset_args)
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_test = DataLoader(dataset, **loader_args)
    model = AlignmentNet.load_from_checkpoint("checkpoints/version_148/checkpoints/epoch=39-step=39.ckpt")
    assignment_solver = AssignmentSolver()

    y_acc = []
    y_pred_acc = []
    y_pred_default_acc = []
    for i, id in tqdm(enumerate(ids), total=len(ids)):
        # Create directory for patient
        if not os.path.exists(f"output/aligner/{id}"):
            os.mkdir(f"output/aligner/{id}")
        # Loop through all samples
        for j in range(config["n_samples"]):
            x, y = dataset[i + j]
            y = y.argmax(0).numpy()
            centroids = np.load(f"data/final/{id}/centroids_{config['jaw']}_{j}.npy")
            y_pred = model(x.unsqueeze(0)).clone().detach().numpy()
            y_pred, _ = assignment_solver(y_pred)
            y_pred = y_pred[0]
            print(f"y: {y}")
            print(f"y_pred: {y_pred}")
            np.save(f"output/aligner/{id}/aligned_labels_pred_{config['jaw']}_{j}.npy", y_pred)
            np.save(f"output/aligner/{id}/aligned_labels_{config['jaw']}_{j}.npy", y)
            np.save(f"output/aligner/{id}/centroids_{config['jaw']}_{j}.npy", centroids)