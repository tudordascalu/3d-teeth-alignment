import unittest
import numpy as np

from scripts.compute_centroids import GumRemover, DummyToothDetector


class TestGumRemover(unittest.TestCase):
    def test_remove_gum(self):
        gum_remover = GumRemover(gum_cls=3)
        tooth_labels = np.array([1, 2, 3, 3, 1])
        instance_labels = np.array([1, 1, 2, 2, 1])
        vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        tooth_labels_processed, instance_labels_processed, vertices_processed = gum_remover(tooth_labels,
                                                                                            instance_labels, vertices)
        np.testing.assert_array_equal(tooth_labels_processed, np.array([1, 2, 1]))
        np.testing.assert_array_equal(instance_labels_processed, np.array([1, 1, 1]))
        np.testing.assert_array_equal(vertices_processed, np.array([[1, 2, 3], [4, 5, 6], [13, 14, 15]]))


class TestDummyToothDetector(unittest.TestCase):
    def test_detect_dummy_tooth(self):
        # Test case where there is a dummy tooth present
        dummy_detector = DummyToothDetector(dummy_cls=17)
        tooth_labels = np.array([1, 2, 2, 2, 2, 4, 5, 6])
        instance_labels = np.array([1, 2, 2, 3, 3, 4, 5, 6])
        tooth_labels_processed = dummy_detector(tooth_labels, instance_labels)
        np.testing.assert_array_equal(tooth_labels_processed, np.array([1, 2, 2, 17, 17, 4, 5, 6]))

        # Test case where there is no dummy tooth present
        tooth_labels = np.array([11, 12, 13, 14, 15, 16])
        instance_labels = np.array([1, 2, 3, 4, 5, 6])
        tooth_labels_processed = dummy_detector(tooth_labels, instance_labels)
        np.testing.assert_array_equal(tooth_labels_processed, np.array([11, 12, 13, 14, 15, 16]))


if __name__ == '__main__':
    unittest.main()
