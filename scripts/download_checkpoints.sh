#!/usr/bin/env bash
# Fetch the promptable-model checkpoints (and MedSAM2's config, which is not
# part of the sam2 package) into checkpoints/. Sizes: ~150 MB each.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints

fetch() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then echo "have $dest"; return; fi
    echo "downloading $dest"
    curl -sL --fail "$url" -o "$dest"
}

fetch "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
    checkpoints/sam2.1_hiera_tiny.pt
fetch "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt" \
    checkpoints/MedSAM2_latest.pt
fetch "https://raw.githubusercontent.com/bowang-lab/MedSAM2/main/sam2/configs/sam2.1_hiera_t512.yaml" \
    checkpoints/sam2.1_hiera_t512.yaml
ls -lh checkpoints/
