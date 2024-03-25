#!/bin/bash

SOURCE_DIR="/Users/Ted/Documents/GitHub/neuralhydrology/data/CAMELS_US"
DEST_DIR="/work/pi_kandread_umass_edu/neuralhydrology/data/CAMELS_US"

# Define folders to exclude
EXCLUDE=(
    ".git"
    ".gitignore"
    ".github*"
    ".DS_Store"
    "__pycache__"
    "runs"
    "test"
    "docs"
)
# Build the exclude options
EXCLUDE_OPTIONS=""
for ex in "${EXCLUDE[@]}"; do
    EXCLUDE_OPTIONS+=" --exclude $ex"
done

# rsync -avp --delete --dry-run $EXCLUDE_OPTIONS "$SOURCE_DIR/" unity:$DEST_DIR
rsync -avp $EXCLUDE_OPTIONS "$SOURCE_DIR/" unity:$DEST_DIR