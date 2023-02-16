import numpy as np
import unittest
from numpy.testing import assert_allclose

from scripts.utils.dummy_tooth_generator import AxialPlane


class AxialPlaneTestCase(unittest.TestCase):
    def test_fit_and_project(self):
        # Test case 1: Planar points
        centroids = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0]])
        p = np.array([2, 2, 1])
        expected_q = np.array([2, 2, 0])
        axial_plane = AxialPlane()
        axial_plane.fit(centroids)
        q = axial_plane.project(p)
        assert_allclose(q, expected_q)

        # Test case 2: Non-planar points
        centroids = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 1]])
        p = np.array([2, 2, 1])
        expected_q = np.array([2, 2, 1])
        axial_plane = AxialPlane()
        axial_plane.fit(centroids)
        q = axial_plane.project(p)
        assert_allclose(q, expected_q)


if __name__ == '__main__':
    unittest.main()
