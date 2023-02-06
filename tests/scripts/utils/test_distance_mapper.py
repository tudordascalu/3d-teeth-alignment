import numpy as np
import unittest

from scripts.utils.distance_mapper import DistanceMapper


class TestDistanceMapper(unittest.TestCase):
    def setUp(self):
        self.n_teeth = 17
        self.distance_mapper = DistanceMapper(n_teeth=self.n_teeth)
        self.centroids = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.tooth_labels = np.array([0, 1, 2])

    def test_n_teeth_attribute(self):
        self.assertEqual(self.distance_mapper.n_teeth, self.n_teeth)

    def test_call_method_return_shape(self):
        distance_map = self.distance_mapper(self.centroids, self.tooth_labels)
        self.assertEqual(distance_map.shape, (self.n_teeth, self.n_teeth, 3))

    def test_call_method_return_value(self):
        distance_map = self.distance_mapper(self.centroids, self.tooth_labels)
        expected_distance_map = np.zeros((self.n_teeth, self.n_teeth, 3))
        expected_distance_map[0, 1, :] = self.centroids[0] - self.centroids[1]
        expected_distance_map[0, 2, :] = self.centroids[0] - self.centroids[2]
        expected_distance_map[1, 0, :] = self.centroids[1] - self.centroids[0]
        expected_distance_map[1, 2, :] = self.centroids[1] - self.centroids[2]

        np.testing.assert_array_equal(distance_map[0, :], expected_distance_map[0, :])
        np.testing.assert_array_equal(distance_map[1, :], expected_distance_map[1, :])


if __name__ == '__main__':
    unittest.main()
