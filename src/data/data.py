import numpy as np
import torch
from torch.nn.functional import one_hot

import enum


class DatasetMode(enum.Enum):
    DEFAULT = "default"
    PREDICT = "predict"


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_path="data/alignment/", jaw="lower", n_samples=1, mode=DatasetMode.DEFAULT):
        """
        :param ids: np.array featuring patient ids
        :param n_samples: determines the number of samples included for each patient id
        :param mode: DatasetMode.PREDICT, we also return id and sample index
        """
        self.ids = ids
        self.data_path = data_path
        self.jaw = jaw
        self.n_samples = n_samples
        self.mode = mode

    def __getitem__(self, idx):
        i_id = idx // self.n_samples
        i_sample = idx % self.n_samples
        score_map = np.load(f"{self.data_path}{self.ids[i_id]}/score_map_{self.jaw}_{i_sample}.npy")
        score_map = torch.from_numpy(score_map).type(
            torch.float32).permute(2, 0, 1)
        distance_map = np.load(f"{self.data_path}{self.ids[i_id]}/distance_map_{self.jaw}_{i_sample}.npy")
        distance_map = torch.from_numpy(distance_map).type(
            torch.float32).permute(2, 0, 1)
        x = torch.concatenate([score_map, distance_map], axis=0)
        labels = torch.from_numpy(np.load(f"{self.data_path}{self.ids[i_id]}/labels_{self.jaw}_{i_sample}.npy")).type(
            torch.int64)
        labels = one_hot(labels, num_classes=17).squeeze(1).type(torch.float32)
        if self.mode == DatasetMode.PREDICT:
            return x, labels, self.ids[i_id], i_sample
        return x, labels

    def __len__(self):
        return len(self.ids) * self.n_samples
