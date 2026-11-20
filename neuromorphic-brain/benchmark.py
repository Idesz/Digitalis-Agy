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
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "n_mnist"),
    )
    parser.add_argument("--require-n-mnist", action="store_true")
    args = parser.parse_args()

    if args.require_n_mnist and (args.measure_forgetting or args.task != "classification"):
        parser.error("--require-n-mnist can only be used with --task classification")

    if args.measure_forgetting:
        result = run_sequential_forgetting_benchmark(
            num_samples=args.samples,
            time_steps=args.time_steps,
            input_dim=args.input_dim,
            epochs_per_task=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        result = run_classification_benchmark(
            task=args.task,
            num_samples=args.samples,
            time_steps=args.time_steps,
            input_dim=args.input_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            data_dir=args.data_dir,
            require_n_mnist=args.require_n_mnist,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
