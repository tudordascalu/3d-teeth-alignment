"""
Computes the distance between each tooth-tooth pair. This should run following swap.
"""
import numpy as np
from tqdm import tqdm
from scripts.utils.distance_mapper import DistanceMapper
from scripts.utils import arg_parser

if __name__ == "__main__":
    # Parse args
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    print(args)

    # ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/{dir}/*")))
    ids = np.load("../data/split/ids_test.npy")
    distance_mapper = DistanceMapper()
    for id in tqdm(ids, total=len(ids)):
        if args.dir == "processed":
            centroids = np.load(f"../data/{args.dir}/{id}/centroids_{args.jaw}.npy")
            labels = np.arange(17)
            distance_map = distance_mapper(centroids, labels)
            np.save(f"../data/{args.dir}/{id}/distance_map_{args.jaw}.npy", distance_map)
            np.save(f"../data/{args.dir}/{id}/distance_map_{args.jaw}.npy", distance_map)
        elif args.dir == "final":
            for i in range(args.n_samples):
                centroids = np.load(f"../data/{args.dir}/{id}/centroids_{args.jaw}_{i}.npy")
                labels = np.load(f"../data/{args.dir}/{id}/labels_{args.jaw}_{i}.npy")
                distance_map = distance_mapper(centroids, labels)
                np.save(f"../data/{args.dir}/{id}/distance_map_{args.jaw}_{i}.npy", distance_map)
                np.save(f"../data/{args.dir}/{id}/distance_map_{args.jaw}_{i}.npy", distance_map)
