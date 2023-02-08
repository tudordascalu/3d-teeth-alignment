import numpy as np
from scipy.optimize import linear_sum_assignment


class AssignmentSolver:
    def __init__(self):
        pass

    def __call__(self, cost_matrix):
        """

        :param cost_matrix: 3D or 2D numpy array; in 3D the first dimension denotes the number of cost matrices passed on
        :return: labels for each tooth and new cost matrix where non-assigned jobs per worker are zero
        """
        if len(cost_matrix.shape) == 2:
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            cost_matrix_processed = np.zeros(cost_matrix.shape)
            cost_matrix_processed[row_ind, col_ind] = 1
            return col_ind, cost_matrix_processed
        elif len(cost_matrix.shape) == 3:
            col_ind_acc = []
            cost_matrix_processed_acc = []
            for cost_m in cost_matrix:
                col_ind, cost_matrix_processed = self(cost_m)
                col_ind_acc.append(col_ind)
                cost_matrix_processed_acc.append(cost_matrix_processed)
            return np.array(col_ind_acc), np.array(cost_matrix_processed_acc)
        else:
            raise ValueError("The parameter cost_matrix should be either 2D or 3D.")
