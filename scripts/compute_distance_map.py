"""
Computes the distance between each tooth-tooth pair. This should run following swap.
"""
import glob
import numpy as np
from tqdm import tqdm
from scripts.utils.distance_mapper import DistanceMapper
from scripts.utils import arg_parser

if __name__ == "__main__":
    parser = arg_parser.create_parser()
    args = parser.parse_args()
    dir = args.dir  # "final" | "processed"
    jaw = args.jaw
    ids = list(map(lambda x: x.split("/")[-1], glob.glob(f"../data/{dir}/*")))
    distance_mapper = DistanceMapper()
    for id in tqdm(ids, total=len(ids)):
        centroids = np.load(f"../data/{dir}/{id}/centroids_{jaw}.npy")
        try:
            labels = np.load(f"../data/{dir}/{id}/labels_{jaw}.npy")
        except:
            labels = np.arange(17)
        distance_map = distance_mapper(centroids, labels)
        np.save(f"../data/{dir}/{id}/distance_map_{jaw}.npy", distance_map)
