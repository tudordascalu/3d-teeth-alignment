import os

import numpy as np
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.data import AlignmentDataset, DatasetMode
from src.model.model import AlignmentNet
from src.utils.assignment_solver import AssignmentSolver

if __name__ == "__main__":
    with open("./src/config.yml", "r") as f:
        config = yaml.safe_load(f)
    # Data
    ids = np.load("data/split/ids_test.npy")
    # Define dataset
    dataset_args = dict(data_path=config["data_path"], jaw=config["jaw"], mode=DatasetMode.PREDICT)
    dataset = AlignmentDataset(ids=ids, n_samples=int(config["n_samples"]), **dataset_args)

    # Define loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_test = DataLoader(dataset, **loader_args)

    if config["jaw"] == "upper":
        model = AlignmentNet.load_from_checkpoint("checkpoints/version_148/checkpoints/epoch=39-step=39.ckpt")
    else:
        model = AlignmentNet.load_from_checkpoint("checkpoints/version_147/checkpoints/epoch=28-step=28.ckpt")
    assignment_solver = AssignmentSolver()

    trainer = Trainer()
    y_pred = trainer.predict(model, dataloaders=loader_test)
