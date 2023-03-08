import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.data.data import AlignmentDataset
from src.model.model import AlignmentNet

if __name__ == "__main__":
    with open("./src/config.yml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids_test = np.load("data/split/ids_test.npy")
    # Define dataset
    dataset_args = dict(data_path=config["data_path"], jaw=config["jaw"])
    dataset_test = AlignmentDataset(ids=ids_test, n_samples=int(config["n_samples"]), **dataset_args)
    # Define loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_test = DataLoader(dataset_test, **loader_args)
    if config["jaw"] == "upper":
        model = AlignmentNet.load_from_checkpoint("checkpoints/version_148/checkpoints/epoch=39-step=39.ckpt")
    else:
        model = AlignmentNet.load_from_checkpoint("checkpoints/version_147/checkpoints/epoch=28-step=28.ckpt")
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_loss",
                                                   mode="min")],
                        log_every_n_steps=100)
    trainer = Trainer(**trainer_args)
    trainer.test(model, dataloaders=loader_test)
