import unittest

import numpy as np
import torch

from src.data.data import AlignmentDataset


class TestAlignmentDataset(unittest.TestCase):
    def setUp(self):
        self.ids = ["0JN50XQR", "0EJBIPTC", "0EAKT1CU", "0IU0UV8E"]
        self.n_samples = 3
        self.jaw = "lower"
        self.dataset = AlignmentDataset(ids=self.ids,
                                        data_path="../../resources/data/",
                                        jaw=self.jaw,
                                        n_samples=self.n_samples)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.ids) * self.n_samples)

    def test_getitem(self):
        input, labels = self.dataset[0]
        self.assertIsInstance(input, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(input.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(input.shape, torch.Size([4, 17, 17]))
        self.assertEqual(labels.shape, torch.Size([17, 17]))

    def test_getitem_2(self):
        i_id = 2
        i_sample = 1
        score_map = np.load(f"../../resources/data/0EAKT1CU/score_map_{self.jaw}_{i_sample}.npy")
        score_map = torch.from_numpy(score_map).type(
            torch.float32).permute(2, 0, 1)
        distance_map = np.load(f"../../resources/data/0EAKT1CU/distance_map_{self.jaw}_{i_sample}.npy")
        distance_map = torch.from_numpy(distance_map).type(
            torch.float32).permute(2, 0, 1)
        x_expected = torch.concatenate([score_map, distance_map], axis=0)
        x, y = self.dataset[i_id * self.n_samples + i_sample]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(x.shape, torch.Size([4, 17, 17]))
        self.assertEqual(y.shape, torch.Size([17, 17]))
        np.testing.assert_array_equal(x.numpy(), x_expected.numpy())


if __name__ == '__main__':
    unittest.main()
