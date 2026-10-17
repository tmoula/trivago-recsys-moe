#!/usr/bin/env bash
# Download the public datasets into $TRIVAGO_MOE_DATA_DIR (default: ./data).
# Requires a Kaggle API token: https://www.kaggle.com/docs/api#authentication
#   pip install kaggle  &&  put kaggle.json in ~/.kaggle/  (or set KAGGLE_USERNAME / KAGGLE_KEY)
#
# Usage: scripts/download_data.sh [trivago|yoochoose|all]   (default: trivago)
set -euo pipefail

DATA_DIR="${TRIVAGO_MOE_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA_DIR"
what="${1:-trivago}"

if [[ "$what" == "trivago" || "$what" == "all" ]]; then
  # ~780 MB zip: train.csv, test.csv, item_metadata.csv. Needed for the Trivago model.
  kaggle datasets download -d pranavmahajan725/trivagorecsyschallengedata2019 -p "$DATA_DIR" --unzip
fi

if [[ "$what" == "yoochoose" || "$what" == "all" ]]; then
  # ~1.5 GB unzipped yoochoose-clicks.dat. Only needed to re-run Phase 1 pre-training.
  kaggle datasets download -d chadgostopp/recsys-challenge-2015 -p "$DATA_DIR" --unzip
fi

ls -lh "$DATA_DIR"
