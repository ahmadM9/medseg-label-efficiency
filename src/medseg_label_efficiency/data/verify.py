"""Check an on-disk dataset copy for completeness and split consistency.

Run after downloading locally or rsyncing to a cluster:

    python -m medseg_label_efficiency.data.verify --data-root data/camus

The check itself lives in each dataset module (it is dataset-specific);
this is the command-line wrapper around the registry.
"""

import argparse
import sys

from medseg_label_efficiency.data.registry import get_dataset_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="camus")
    parser.add_argument("--data-root", default="data/camus")
    args = parser.parse_args()

    problems = get_dataset_module(args.dataset).verify(args.data_root)
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        sys.exit(1)
    print(f"OK — {args.dataset} copy at {args.data_root} is complete")


if __name__ == "__main__":
    main()
