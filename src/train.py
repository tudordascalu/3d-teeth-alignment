import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from src.data.data import AlignmentDataset
from src.model.model import AlignmentNet

if __name__ == "__main__":
    with open("./src/config.yml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids_train, ids_val, ids_test = np.load("data/split/ids_train.npy"), \
                                   np.load("data/split/ids_val.npy"), \
                                   np.load("data/split/ids_test.npy")
    # Define dataset
    dataset_args = dict(data_path=config["data_path"], jaw=config["jaw"])
    dataset_train = AlignmentDataset(ids=ids_train, **dataset_args)
    dataset_val = AlignmentDataset(ids=ids_val, **dataset_args)
    dataset_test = AlignmentDataset(ids=ids_test, **dataset_args)
    # Define loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    loader_val = DataLoader(dataset_val, **loader_args)
    loader_test = DataLoader(dataset_test, **loader_args)
    # Define model
    model = AlignmentNet(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    trainer_args = dict(max_epochs=config["max_epochs"],
                        callbacks=[ModelCheckpoint(save_top_k=1,
                                                   monitor="val_loss",
                                                   mode="min")],
                        logger=logger,
                        log_every_n_steps=100)

    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=4, **trainer_args)

    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)
    trainer.test(ckpt_path="best", dataloaders=loader_test)
