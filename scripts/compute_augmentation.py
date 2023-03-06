import glob
import os

import numpy as np
from tqdm import tqdm

from scripts.utils import arg_parser
from scripts.utils.dummy_tooth_generator import DummyToothGenerator
from scripts.utils.tooth_swapper import ToothSwapper

if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/processed/*")))
    dummy_tooth_generator = DummyToothGenerator(n_teeth=args.teeth, min_dist=5, max_dist=15, max_noise_dist=2)
    tooth_swapper = ToothSwapper(n_teeth=args.teeth, neighbor_range=2, max_swaps=args.max_swaps)
    for id in tqdm(ids, total=len(ids)):
        if not os.path.exists(f"../data/final/{id}"):
            os.mkdir(f"../data/final/{id}")
        for i in range(args.n_samples):
            centroids = np.load(f"../data/processed/{id}/centroids_{args.jaw}.npy")
            # Add dummy tooth w.p "p_dummy"
            if np.random.rand() <= args.p_dummy:
                centroids = dummy_tooth_generator(centroids)
            # Remove tooth w.p "p_missing"
            if np.random.rand() <= args.p_missing:
                n_remove = np.random.randint(1, args.max_missing + 1)
                # Avoid removing dummy and wisdom teeth as they are scarce
                i_remove = np.random.choice(np.arange(1, args.teeth - 2), size=n_remove, replace=False)
                centroids[i_remove] = np.array([0, 0, 0])
            # Swap labels w.p "p_swap"
            labels = np.arange(0, args.teeth)
            if np.random.rand() <= args.p_swap:
                centroids, labels = tooth_swapper(centroids, labels)

            np.save(f"../data/final/{id}/labels_{args.jaw}_{i}.npy", labels)
            np.save(f"../data/final/{id}/centroids_{args.jaw}_{i}.npy", centroids)
