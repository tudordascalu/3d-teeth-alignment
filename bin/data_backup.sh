#!/bin/bash

# Make backup file if not exists
if [ ! -d "data/backup/" ]; then
  # Create the directory
  mkdir "data/backup/"
fi

directories=("final" "processed" "assets" "split")

# Loop through the list of strings
for dir in "${directories[@]}"; do
  zip -r "data/backup/$dir.zip" "data/$dir/"
done
