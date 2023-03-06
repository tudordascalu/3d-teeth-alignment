import numpy as np
import unittest

from scripts.utils.tooth_swapper import ToothSwapper


class TestToothSwapper(unittest.TestCase):
    def test_init(self):
        swapper = ToothSwapper()
        self.assertEqual(swapper.max_swaps, 2)

        swapper = ToothSwapper(max_swaps=3)
        self.assertEqual(swapper.max_swaps, 3)

    def test_call(self):
        np.random.seed(13)
        swapper = ToothSwapper(n_teeth=5)
        centroids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        labels = np.array([0, 1, 2, 3, 4])
        centroids, labels = swapper(centroids, labels)
        self.assertTrue((centroids == np.array([[4, 5, 6], [1, 2, 3], [10, 11, 12], [7, 8, 9], [13, 14, 15]])).all())
        self.assertTrue((labels == np.array([1, 0, 3, 2, 4])).all())

    def test__swap(self):
        centroids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array([0, 1, 2])
        centroids, labels = ToothSwapper._swap(centroids, labels, 0, 2)
        self.assertTrue((centroids == np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])).all())
        self.assertTrue((labels == np.array([2, 1, 0])).all())


if __name__ == '__main__':
    unittest.main()
