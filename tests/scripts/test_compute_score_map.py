import numpy as np
import unittest
from scipy.stats import multivariate_normal

from scripts.compute_score_map import ScoreMapper


class TestScoreMapper(unittest.TestCase):
    def setUp(self):
        self.n_teeth = 17
        np.random.seed(42)
        self.distance_map = np.random.randn(self.n_teeth, self.n_teeth, 3)
        self.distance_map_mean = np.load("../resources/distance_map_mean_lower.npy")
        self.distance_map_cov = np.load("../resources/distance_map_cov_lower.npy")

    def test_n_teeth(self):
        mapper = ScoreMapper(n_teeth=self.n_teeth)
        self.assertEqual(mapper.n_teeth, self.n_teeth)

    def test_score_map_shape(self):
        mapper = ScoreMapper(n_teeth=self.n_teeth)
        score_map = mapper(self.distance_map, self.distance_map_mean, self.distance_map_cov)
        self.assertEqual(score_map.shape, (self.n_teeth, self.n_teeth, 1))

    def test_score_map_values(self):
        mapper = ScoreMapper(n_teeth=self.n_teeth)
        score_map = mapper(self.distance_map, self.distance_map_mean, self.distance_map_cov)
        expected_scores = []
        for i in range(self.n_teeth):
            for j in range(self.n_teeth):
                distance = self.distance_map[i, j]
                distance_mean = self.distance_map_mean[i, j]
                distance_cov = self.distance_map_cov[i, j]
                try:
                    expected_score = multivariate_normal.pdf(distance, mean=distance_mean, cov=distance_cov,
                                                             allow_singular=False)
                except:
                    expected_score = 0
                expected_scores.append(expected_score)
        expected_scores = np.array(expected_scores).reshape(self.n_teeth, self.n_teeth, 1)
        np.testing.assert_array_almost_equal(score_map, expected_scores, decimal=5)


if __name__ == '__main__':
    unittest.main()
