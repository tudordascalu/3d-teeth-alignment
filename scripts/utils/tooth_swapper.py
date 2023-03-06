import numpy as np


class ToothSwapper:
    def __init__(self, n_teeth=17, neighbor_range=2, max_swaps=2):
        """

        :param max_swaps: determines the maximum number of swaps performed for a give tooth
        """
        self.max_swaps = max_swaps
        self.n_teeth = n_teeth
        self.neighbor_range = neighbor_range

    def __call__(self, centroids, labels):
        """

        :param centroids: array of centroids
        :param labels: array of labels, where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        i_swapped_acc = []
        n_swaps = np.random.randint(0, self.max_swaps + 1)
        j = 0
        while j < n_swaps:
            try:
                i_list = np.arange(0, self.n_teeth)
                i_list = np.delete(i_list, i_swapped_acc)
                i_1 = np.random.choice(i_list, size=1)[0]
                neighbors = np.arange(max(i_1 - self.neighbor_range, 0),
                                      min(i_1 + self.neighbor_range + 1, self.n_teeth))
                neighbors = neighbors[~np.in1d(neighbors, np.concatenate((i_swapped_acc, [i_1])))]
                i_2 = np.random.choice(neighbors, size=1)[0]
                centroids, labels = self._swap(centroids, labels, i_1, i_2)
                i_swapped_acc.extend([i_1, i_2])
                j += 1
            except:
                print(f"One exception! i_1 {i_1}, i_swapped_acc {i_swapped_acc}")
        return centroids, labels

    @staticmethod
    def _swap(centroids, labels, i_1, i_2):
        """
        :param centroids: array of centroids of shape (n, 3)
        :param labels: array of labels of shape (n,), where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        labels[i_1], labels[i_2] = labels[i_2], labels[i_1]
        centroids[[i_1, i_2]] = centroids[[i_2, i_1]]
        return centroids, labels
