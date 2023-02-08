import unittest

import numpy as np

from src.utils.assignment_solver import AssignmentSolver


class TestAssignmentSolver(unittest.TestCase):
    def setUp(self):
        self.assignment_solver = AssignmentSolver()

    def test_assignment_solver_2d_shape(self):
        cost_matrix = np.array([[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]])
        labels, cost_matrix_processed = self.assignment_solver(cost_matrix)
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(cost_matrix_processed.shape, (3, 3))

    def test_assignment_solver_2d_value(self):
        cost_matrix = np.array([[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]])
        labels, cost_matrix_processed = self.assignment_solver(cost_matrix)
        np.testing.assert_array_equal(labels, np.array([2, 0, 1]))
        np.testing.assert_array_equal(cost_matrix_processed, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

    def test_assignment_solver_3d_shape(self):
        cost_matrix = np.array([[[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]],
                                [[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]]])
        labels, cost_matrix_processed = self.assignment_solver(cost_matrix)
        self.assertEqual(labels.shape, (2, 3))
        self.assertEqual(cost_matrix_processed.shape, (2, 3, 3))

    def test_assignment_solver_3d_value(self):
        cost_matrix = np.array([[[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]],
                                [[0.2, 0.4, 0.6], [0.6, 0.4, 0.2], [0.5, 0.4, 0.2]]])
        labels, cost_matrix_processed = self.assignment_solver(cost_matrix)
        np.testing.assert_array_equal(labels, np.array([[2, 0, 1], [2, 0, 1]]))
        np.testing.assert_array_equal(cost_matrix_processed, np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                                                       [[0, 0, 1], [1, 0, 0], [0, 1, 0]]]))


if __name__ == '__main__':
    unittest.main()
