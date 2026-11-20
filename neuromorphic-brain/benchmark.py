"""Command-line entry point for reproducible held-out task benchmarks."""

import argparse
import json
import os

from src.core.benchmark import (
    run_classification_benchmark,
    run_sequential_forgetting_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("xor", "parity", "classification"), default="xor")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-forgetting", action="store_true")
    args = parser.parse_args()

    if args.measure_forgetting:
        result = run_sequential_forgetting_benchmark(
            num_samples=args.samples,
            epochs_per_task=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        result = run_classification_benchmark(
            task=args.task,
            num_samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=os.path.join(os.path.dirname(__file__), "data", "n_mnist"),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
