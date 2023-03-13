# Dental Casts Labeling

This project trains a neural network to assign optimal labels to a set of candidate instances corresponding to teeth in
dental casts. The network is implemented using PyTorch-Lightning.

## Preprocessing

1. Load RAW data in "data/raw" directory.
2. For each jaw, run the preprocessing scripts included in "scripts" directory in the following order:
    1. `compute_centroids.py`
    2. `order_test_centroids.py` (Note that this file expects true centroids for each test sample.)
    3. `compute_distance_map.py` (Note that this should be computed for both "data/processed", and "data/final"
       directories.)
    4. `compute_statistics.py`
    5. `compute_score_map.py`

## Train

1. Run the train script located at "src/train.py"

### Predict

3. Run the prediction script located at "src/predict.py"

### Test

2. Run the test script located at "src/test.py"
