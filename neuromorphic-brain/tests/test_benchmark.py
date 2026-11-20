import pytest
import torch

from src.core.benchmark import (
    run_classification_benchmark,
    run_sequential_forgetting_benchmark,
    split_dataset,
)
from src.core.dataset import NeuromorphicDataset


def test_dataset_split_is_reproducible_and_disjoint():
    dataset = NeuromorphicDataset(num_samples=40, task="xor", seed=7)

    train, validation, test = split_dataset(dataset, seed=11)
    repeated = split_dataset(dataset, seed=11)
    indices = [set(split.indices) for split in (train, validation, test)]

    assert [split.indices for split in (train, validation, test)] == [
        split.indices for split in repeated
    ]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert sum(map(len, (train, validation, test))) == len(dataset)


def test_classification_benchmark_reports_held_out_metrics():
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_classification_benchmark(
            task="xor",
            num_samples=40,
            time_steps=2,
            input_dim=16,
            batch_size=8,
            epochs=1,
            seed=5,
            device=torch.device("cpu"),
        )
    finally:
        torch.set_num_threads(original_num_threads)

    assert result["data_source"] == "Synthetic XOR"
    assert result["split_sizes"] == {"train": 28, "validation": 6, "test": 6}
    assert result["best_epoch"] == 1
    assert set(result["test"]) == {"loss", "accuracy", "samples"}
    assert result["test"]["samples"] == 6.0
    assert result["baseline"]["strategy"] == "majority_class"
    assert 0.0 <= result["baseline"]["test_accuracy"] <= 1.0


def test_classification_benchmark_requires_and_reports_processed_n_mnist(tmp_path):
    generator = torch.Generator().manual_seed(19)
    train_samples = (torch.rand(40, 16, 64, generator=generator) > 0.5).to(torch.uint8)
    test_samples = (torch.rand(12, 16, 64, generator=generator) > 0.5).to(torch.uint8)
    train_labels = torch.arange(40).remainder(10)
    test_labels = torch.full((12,), 9, dtype=torch.long)
    torch.save(
        {
            "format_version": 2,
            "train_samples": train_samples,
            "train_labels": train_labels,
            "test_samples": test_samples,
            "test_labels": test_labels,
        },
        tmp_path / "processed_nmnist.pt",
    )
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_classification_benchmark(
            task="classification",
            num_samples=40,
            time_steps=16,
            input_dim=64,
            batch_size=8,
            epochs=1,
            seed=19,
            data_dir=str(tmp_path),
            require_n_mnist=True,
            device=torch.device("cpu"),
        )
    finally:
        torch.set_num_threads(original_num_threads)

    assert result["data_source"] == "N-MNIST"
    assert result["input_shape"] == {"time_steps": 16, "input_dim": 64}
    assert result["split_sizes"] == {"train": 34, "validation": 6, "test": 12}
    assert result["split_sources"] == {
        "train": "N-MNIST train",
        "validation": "N-MNIST train",
        "test": "N-MNIST test",
    }
    assert result["test"]["samples"] == 12.0


def test_n_mnist_requirement_fails_instead_of_using_synthetic_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="N-MNIST was required"):
        run_classification_benchmark(
            task="classification",
            num_samples=40,
            time_steps=2,
            input_dim=16,
            epochs=1,
            data_dir=str(tmp_path),
            require_n_mnist=True,
            device=torch.device("cpu"),
        )


def test_sequential_benchmark_reports_forgetting_after_task_switch():
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_sequential_forgetting_benchmark(
            tasks=("xor", "parity"),
            num_samples=40,
            time_steps=2,
            input_dim=16,
            batch_size=8,
            epochs_per_task=1,
            seed=13,
            device=torch.device("cpu"),
        )
    finally:
        torch.set_num_threads(original_num_threads)

    assert [stage["after_task"] for stage in result["stages"]] == ["xor", "parity"]
    assert set(result["forgetting"]) == {"xor"}
    assert 0.0 <= result["forgetting"]["xor"] <= 1.0
    assert "validation" in result["stages"][-1]["metrics"]["xor"]
    assert "test" in result["stages"][-1]["metrics"]["xor"]
