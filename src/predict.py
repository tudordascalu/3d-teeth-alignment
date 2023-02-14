import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.data.data import AlignmentDataset
from src.model.model import AlignmentNet
from src.utils.assignment_solver import AssignmentSolver

if __name__ == "__main__":
    with open("./src/config.yml", "r") as f:
        config = yaml.safe_load(f)
    # Data
    ids = np.load("data/split/ids_test.npy")
    # Define dataset
    dataset_args = dict(data_path=config["data_path"], jaw="lower")
    dataset = AlignmentDataset(ids=ids, **dataset_args)
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_test = DataLoader(dataset, **loader_args)
    model = AlignmentNet.load_from_checkpoint("checkpoints/version_123/checkpoints/epoch=17-step=17.ckpt")
    assignment_solver = AssignmentSolver()

    y_acc = []
    y_pred_acc = []
    y_pred_default_acc = []
    for i, id in enumerate(ids):
        x, y = dataset[i]
        centroids = np.load(f"data/final/{id}/centroids_lower_0.npy")
        present_teeth = np.where((centroids != np.array([0, 0, 0])).all(axis=1))[0]
        y_pred = model(x.unsqueeze(0)).clone().detach().numpy()
        y_pred, _ = assignment_solver(y_pred)
        y_pred = y_pred[0]
        y = y.argmax(0).numpy()
        y_pred_default = np.arange(17)
        i_delete_acc = []
        for ii, tooth in enumerate(y):
            if tooth not in present_teeth:
                i_delete_acc.append(ii)
        y = np.delete(y, i_delete_acc)
        y_pred = np.delete(y_pred, i_delete_acc)
        y_pred_default = np.delete(y_pred_default, i_delete_acc)
        y_acc.extend(y)
        y_pred_acc.extend(y_pred)
        y_pred_default_acc.extend(y_pred_default)
        print(f"-- {id}")
        print(f"y {y}")
        print(f"y_pred {y_pred}")
