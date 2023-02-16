import glob

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser
from scripts.utils.dummy_tooth_generator import DummyToothGenerator

if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    p_dummy = args.p_dummy
    p_missing = args.p_missing
    n_teeth = args.teeth
    max_missing = args.max_missing
    print(args)
    jaw = args.jaw
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/processed/*")))
    dummy_tooth_generator = DummyToothGenerator(n_teeth=n_teeth, min_dist=5, max_dist=15, max_noise_dist=2)
    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/processed/{id}/centroids_{jaw}.npy")
        # Add dummy tooth w.p "p_dummy"
        if np.random.rand() <= p_dummy:
            centroids = dummy_tooth_generator(centroids)
        # Remove tooth w.p "p_missing"
        if np.random.rand() <= p_missing:
            n_remove = np.random.randint(1, max_missing + 1)
            # Avoid removing dummy and wisdom teeth as they are scarce
            i_remove = np.random.choice(np.arange(1, n_teeth - 2), size=n_remove, replace=False)
            centroids[i_remove] = np.array([0, 0, 0])
        np.save(f"../data/processed/{id}/centroids_augmented_{jaw}.npy", centroids)
