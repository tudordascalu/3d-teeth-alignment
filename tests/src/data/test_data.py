import unittest
import torch

from src.data.data import AlignmentDataset


class TestAlignmentDataset(unittest.TestCase):
    def setUp(self):
        self.ids = ["0EAKT1CU", "0EJBIPTC", "0IU0UV8E"]
        self.dataset = AlignmentDataset(ids=self.ids,
                                        data_path="../../resources/data/",
                                        jaw="lower")

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.ids))

    def test_getitem(self):
        score_map, labels = self.dataset[0]
        self.assertIsInstance(score_map, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(score_map.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(score_map.shape, torch.Size([17, 17, 1]))
        self.assertEqual(labels.shape, torch.Size([17]))


if __name__ == '__main__':
    unittest.main()
