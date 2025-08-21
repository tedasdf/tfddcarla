#!/bin/bash

SRC_DIR="."

find "$SRC_DIR" -type f -name "*.json" | while read -r file; do
    # Get the folder path (without filename)
    dir_path="$(dirname "$file")"
    echo "$dir_path"
done | sort -u
