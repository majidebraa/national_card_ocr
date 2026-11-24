#!/usr/bin/env bash
# Download the Kaggle "Iranian National ID Card Dataset" into data/raw (git-ignored).
# Needs Kaggle credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY.
set -euo pipefail
pip install --quiet kaggle
mkdir -p data/raw
kaggle datasets download -d kibordevents/iranian-national-id-card-dataset -p data/raw --unzip
echo "Done. Inspect the license and the metadata CSV before use. Never commit data/."
