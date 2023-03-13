#!/bin/bash

jaw=$1

if [[ "$jaw" != "upper" && "$jaw" != "lower" ]]; then
  echo "Error: jaw should be either 'upper' or 'lower'"
  exit 1
fi

echo "Computing centroids.. $jaw"
python ./scripts/compute_centroids.py --jaw "$jaw"
python ../order_test_centroids --jaw "$jaw"
python scripts/compute_distance_map.py --jaw "$jaw" --dir processed
python scripts/compute_distance_map.py --jaw "$jaw" --dir final
python scripts/compute_statistics.py --jaw "$jaw"
python ../compute_score_map --jaw "$jaw"
