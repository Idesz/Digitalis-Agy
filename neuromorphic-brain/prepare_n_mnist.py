"""Download official N-MNIST and save the project's deterministic 16x64 format."""

import argparse
import json
from pathlib import Path

from src.core.nmnist import download_and_preprocess_n_mnist


def main() -> None:
    default_data_dir = Path(__file__).parent / "data" / "n_mnist"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--train-samples", type=int, default=0, help="0 processes the full train split")
    parser.add_argument("--test-samples", type=int, default=0, help="0 processes the full test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_samples < 0 or args.test_samples < 0:
        parser.error("sample limits must be non-negative")
    result = download_and_preprocess_n_mnist(
        data_dir=args.data_dir,
        train_samples=args.train_samples or None,
        test_samples=args.test_samples or None,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
