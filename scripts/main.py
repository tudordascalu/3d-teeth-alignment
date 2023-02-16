import subprocess

if __name__ == "__main__":
    jaw = "lower"
    # The scripts in the order that they should run
    scripts = [
        "compute_splits.py",
        f"compute_centroids.py -j {jaw}",
        f"compute_augmentation.py -j {jaw}",
        f"compute_swaps.py -j {jaw} -s 2",
        f"compute_distance_map.py -d processed -j {jaw}",
        f"compute_distance_map.py -d final -j {jaw}",
        f"compute_statistics.py -j {jaw}",
        f"compute_score_map.py -j {jaw}",
    ]
    for script in scripts:
        subprocess.run(["python", script])
