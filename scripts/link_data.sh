#!/usr/bin/env bash
# Link a local CAMUS download into the repo at data/camus.
#
# Usage: ./scripts/link_data.sh [path-to-camus]
#
# The path may be absolute or relative to the data/ directory.
# Default matches the layout on the development machine, where the dataset
# lives in ../Resources next to the repo.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-../../Resources}"

mkdir -p "$repo_root/data"
ln -sfn "$target" "$repo_root/data/camus"

echo "data/camus -> $target"
ls "$repo_root/data/camus/" >/dev/null || {
    echo "warning: link target does not resolve" >&2
    exit 1
}
