import unittest

import numpy as np

from scripts.utils.missing_teeth_detector import MissingTeethRemover, MissingTeethDetector


class TestMissingTeethDetector(unittest.TestCase):

    def test_no_missing_teeth(self):
        detector = MissingTeethDetector()
        centroids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([])
        self.assertTrue(np.array_equal(detector(centroids, labels), expected_labels))

    def test_one_missing_tooth(self):
        detector = MissingTeethDetector()
        centroids = np.array([[1, 2, 3], [0, 0, 0], [7, 8, 9]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([2])
        self.assertTrue(np.array_equal(detector(centroids, labels), expected_labels))

    def test_all_missing_teeth(self):
        detector = MissingTeethDetector()
        centroids = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(detector(centroids, labels), expected_labels))


class TestMissingTeethRemover(unittest.TestCase):

    def test_remove_missing_teeth(self):
        remover = MissingTeethRemover()
        centroids = np.array([[1, 2, 3], [0, 0, 0], [7, 8, 9]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([1, 3])
        self.assertTrue(np.array_equal(remover(centroids, labels), expected_labels))

    def test_no_missing_teeth(self):
        remover = MissingTeethRemover()
        centroids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(remover(centroids, labels), expected_labels))

    def test_all_missing_teeth(self):
        remover = MissingTeethRemover()
        centroids = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        labels = np.array([1, 2, 3])
        expected_labels = np.array([])
        self.assertTrue(np.array_equal(remover(centroids, labels), expected_labels))


if __name__ == '__main__':
    unittest.main()
