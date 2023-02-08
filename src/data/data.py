import numpy as np
import torch
from torch.nn.functional import one_hot


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_path="data/alignment/", jaw="lower"):
        """
        :param ids: np.array featuring patient ids
        """
        self.ids = ids
        self.data_path = data_path
        self.jaw = jaw

    def __getitem__(self, idx):
        score_map = torch.from_numpy(np.load(f"{self.data_path}{self.ids[idx]}/score_map_{self.jaw}.npy")).type(
            torch.float32)
        labels = torch.from_numpy(np.load(f"{self.data_path}{self.ids[idx]}/labels_{self.jaw}.npy")).type(torch.int64)
        labels = one_hot(labels, num_classes=17).squeeze(1).type(torch.float32)
        return score_map, labels

    def __len__(self):
        return len(self.ids)
