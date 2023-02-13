import unittest
import numpy as np

from scripts.utils.centroid_mapper import CentroidMapper


class TestCentroidMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = CentroidMapper()

    def test_init(self):
        self.assertEqual(self.mapper.n_teeth, 17)

    def test_call(self):
        vertices = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        tooth_labels = np.array([1, 1, 5])
        centroids = self.mapper(vertices, tooth_labels)
        expected_centroids = np.zeros((17, 3))
        expected_centroids[1] = np.array([1.5, 2.5, 3.5])
        expected_centroids[5] = np.array([3, 4, 5])
        self.assertTrue(np.array_equal(centroids, expected_centroids))

        vertices = np.array([[1, 2, 3], [2, 3, 4]])
        tooth_labels = np.array([2, 2])
        centroids = self.mapper(vertices, tooth_labels)
        expected_centroids = np.zeros((17, 3))
        expected_centroids[2] = np.array([1.5, 2.5, 3.5])
        self.assertTrue(np.array_equal(centroids, expected_centroids))


if __name__ == '__main__':
    unittest.main()
