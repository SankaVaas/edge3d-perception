#!/usr/bin/env bash
# Download a TUM RGB-D sequence into ./data
# Usage: bash scripts/download_data.sh [sequence_name]
# Default: rgbd_dataset_freiburg1_xyz (~0.4 GB)
set -euo pipefail

SEQ="${1:-rgbd_dataset_freiburg1_xyz}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data"
# Sequence names look like rgbd_dataset_freiburg1_xyz -> folder "freiburg1"
FR="$(echo "${SEQ}" | sed -E 's/rgbd_dataset_(freiburg[0-9]+)_.*/\1/')"
URL="https://cvg.cit.tum.de/rgbd/dataset/${FR}/${SEQ}.tgz"

mkdir -p "${DEST}"
if [ -d "${DEST}/${SEQ}" ]; then
  echo "Already present: ${DEST}/${SEQ}"
  exit 0
fi

echo "Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${DEST}/${SEQ}.tgz" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${DEST}/${SEQ}.tgz" "${URL}"
else
  echo "Need curl or wget" >&2
  exit 1
fi

tar -xzf "${DEST}/${SEQ}.tgz" -C "${DEST}"
rm "${DEST}/${SEQ}.tgz"
echo "Done: ${DEST}/${SEQ}"
