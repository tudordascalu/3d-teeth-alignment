from scripts.utils.tooth_label_encoder import ToothLabelEncoder, LabelEncoder
import unittest
import numpy as np


class ToothLabelEncoderTestCase(unittest.TestCase):
    def test_encoder_lower_jaw(self):
        encoder = ToothLabelEncoder.encoder("lower")
        expected_encoder = {
            38: 0, 37: 1, 36: 2, 35: 3, 34: 4, 33: 5, 32: 6, 31: 7,
            41: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13, 47: 14, 48: 15
        }
        self.assertDictEqual(encoder.encoder, expected_encoder)

    def test_encoder_upper_jaw(self):
        encoder = ToothLabelEncoder.encoder("upper")
        expected_encoder = {
            18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
            21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
        }
        self.assertDictEqual(encoder.encoder, expected_encoder)

    def test_encoder_invalid_jaw(self):
        with self.assertRaises(ValueError) as context:
            ToothLabelEncoder.encoder("invalid")
        self.assertEqual(str(context.exception), "jaw must be either 'lower' or 'upper'")


class LabelEncoderTestCase(unittest.TestCase):
    def setUp(self):
        self.labels = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
        self.encoder = LabelEncoder(labels=self.labels)

    def test_fit(self):
        expected_encoder = {
            38: 0, 37: 1, 36: 2, 35: 3, 34: 4, 33: 5, 32: 6, 31: 7,
            41: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13, 47: 14, 48: 15
        }
        self.assertDictEqual(self.encoder.encoder, expected_encoder)

    def test_transform(self):
        labels = [34, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
        expected = np.array([4, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15])
        self.assertTrue(np.array_equal(self.encoder.transform(labels), expected))

    def test_inverse_transform(self):
        encoded_labels = np.array([3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        labels = self.encoder.inverse_transform(encoded_labels)
        expected = np.array([35, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48])
        np.testing.assert_array_equal(labels, expected)


if __name__ == '__main__':
    unittest.main()
