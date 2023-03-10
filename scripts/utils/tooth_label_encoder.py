import numpy as np


class ToothLabelEncoder:
    def __init__(self):
        pass

    @staticmethod
    def encoder(jaw):
        if jaw == "lower":
            # tooth_labels = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]  # lower
            tooth_labels = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48, 80]
        elif jaw == "upper":
            # tooth_labels = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]  # upper
            tooth_labels = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 80]
        else:
            raise ValueError("jaw must be either 'lower' or 'upper'")
        encoder = LabelEncoder()
        encoder.fit(tooth_labels)
        return encoder


class LabelEncoder:
    def __init__(self, labels=None, encoder=None):
        if labels is not None:
            self.fit(labels)
        elif encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = {}

    def fit(self, labels):
        self.encoder = {label: i for i, label in enumerate(labels)}

    def transform(self, labels):
        return np.array([self.encoder[label] for label in labels])

    def inverse_transform(self, encoded_labels):
        decoder = {i: label for label, i in self.encoder.items()}
        return np.array([decoder[encoded_label] for encoded_label in encoded_labels])
