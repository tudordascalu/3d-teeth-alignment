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
        input, labels = self.dataset[0]
        self.assertIsInstance(input, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(input.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(input.shape, torch.Size([4, 17, 17]))
        self.assertEqual(labels.shape, torch.Size([17, 17]))


if __name__ == '__main__':
    unittest.main()
