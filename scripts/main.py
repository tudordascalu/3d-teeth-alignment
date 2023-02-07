import subprocess

if __name__ == "__main__":
    # The scripts in the order that they should run
    scripts = ["compute_centroids.py -j lower", "compute_centroids.py -j upper",
               "compute_splits.py",
               "compute_swaps.py -j lower -s 3", "compute_swaps.py -j upper -s 3",
               "compute_distance_map.py -d processed -j lower", "compute_distance_map.py -d processed -j upper",
               "compute_distance_map.py -d final -j lower", "compute_distance_map.py -d final -j upper",
               "compute_statistics.py -j lower", "compute_statistics.py -j upper"
               ]
    for script in scripts:
        subprocess.run(["python", script])
