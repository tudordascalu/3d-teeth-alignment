"""
Splits the patients into train, test based on images featuring double teeth.
"""
import glob

import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Patients to remove because of various reasons
    ids_remove = ["0169NHT6", "01KTRG9K", "0154T9CN"]
    # Here are all patients containing double teeth. We only have cases of double teeth on the upper jaw.
    ids_double_tooth = np.array(["0169NHT6", "Y48DURWV", "0154T9CN", "0140E7V2"])
    # Filter out double teeth
    ids = list(map(lambda x: x.split("/")[-1], glob.glob("../data/processed/*")))
    # Perform train-test split
    ids_train, ids_test = train_test_split(ids, train_size=497, random_state=42)
    ids_train, ids_val = train_test_split(ids_train, train_size=449, random_state=42)
    # Distribute double teeth patients
    ids_train = np.concatenate((ids_train, [ids_double_tooth[0]]))
    ids_val = np.concatenate((ids_train, [ids_double_tooth[1]]))
    ids_test = np.concatenate((ids_test, ids_double_tooth[2:]))
    np.save("../data/split/ids_train.npy", ids_train)
    np.save("../data/split/ids_val.npy", ids_val)
    np.save("../data/split/ids_test.npy", ids_test)
