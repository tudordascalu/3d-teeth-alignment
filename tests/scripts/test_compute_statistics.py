import numpy as np
import unittest

from scripts.compute_statistics import DistanceMapCov


class TestComputeDistanceMapCov(unittest.TestCase):
    def setUp(self):
        self.distance_map_cov_mapper = DistanceMapCov()

    def test_distance_map_cov_shape(self):
        n_patients = 5
        distance_map_acc = np.zeros((n_patients, 17, 17, 3))
        cov = self.distance_map_cov_mapper(distance_map_acc)
        self.assertEqual(cov.shape, (17, 17, 3, 3))

    def test_distance_map_cov_values(self):
        n_patients = 2
        distance_map_acc = np.ones((n_patients, 17, 17, 3))
        cov = self.distance_map_cov_mapper(distance_map_acc)
        expected_cov = np.zeros((17, 17, 3, 3))
        for i in range(17):
            for j in range(17):
                expected_cov[i, j] = np.cov(np.ones((3, n_patients)))
        np.testing.assert_array_equal(cov, expected_cov)

    def test_distance_map_cov_different_values(self):
        n_patients = 3
        distance_map_acc = np.zeros((n_patients, 17, 17, 3))
        distance_map_acc[0, 0, 0] = np.array([1, 2, 3])
        distance_map_acc[1, 0, 0] = np.array([2, 3, 4])
        distance_map_acc[2, 0, 0] = np.array([3, 4, 5])
        cov = self.distance_map_cov_mapper(distance_map_acc)
        expected_cov = np.zeros((17, 17, 3, 3))
        expected_cov[0, 0] = np.cov(distance_map_acc[:, 0, 0, ...].T)
        np.testing.assert_array_equal(cov, expected_cov)


if __name__ == '__main__':
    unittest.main()
