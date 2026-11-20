"""Reproducible held-out benchmarks for the local classification tasks."""

import copy
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from src.core.dataset import NeuromorphicDataset
from src.core.pipeline import NeuromorphicCognitivePipeline
from src.routing.cdr_router import TaskRoutingContext


def split_dataset(
    dataset: NeuromorphicDataset,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    if not math.isfinite(train_ratio) or not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be finite and in (0, 1)")
    if not math.isfinite(validation_ratio) or not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be finite and in (0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train and validation ratios must sum to less than one")

    sample_count = len(dataset)
    train_count = int(sample_count * train_ratio)
    validation_count = int(sample_count * validation_ratio)
    test_count = sample_count - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("dataset must provide at least one sample for each split")

    return tuple(random_split(
        dataset,
        [train_count, validation_count, test_count],
        generator=torch.Generator().manual_seed(seed),
    ))


def evaluate_classifier(
    pipeline: NeuromorphicCognitivePipeline,
    loader: DataLoader,
) -> dict[str, float]:
    correct = 0
    sample_count = 0
    total_loss = 0.0
    pipeline.snn.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(pipeline.device)
            targets = targets.to(pipeline.device)
            _, readouts = pipeline.snn.forward_sequence(inputs)
            logits = readouts[-1]
            total_loss += float(F.cross_entropy(logits, targets, reduction="sum").item())
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            sample_count += targets.shape[0]
    if sample_count == 0:
        raise ValueError("evaluation loader must contain at least one sample")
    return {
        "loss": total_loss / sample_count,
        "accuracy": correct / sample_count,
        "samples": float(sample_count),
    }


def split_training_validation(
    dataset: NeuromorphicDataset,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    if not math.isfinite(validation_ratio) or not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be finite and in (0, 1)")
    sample_count = len(dataset)
    validation_count = max(1, int(sample_count * validation_ratio))
    train_count = sample_count - validation_count
    if train_count < 1:
        raise ValueError("dataset must provide at least one train and validation sample")
    return tuple(random_split(
        dataset,
        [train_count, validation_count],
        generator=torch.Generator().manual_seed(seed),
    ))


def majority_class_baseline(
    dataset: NeuromorphicDataset,
    train_subset: Subset,
    test_loader: DataLoader,
) -> dict[str, Any]:
    train_labels = dataset.labels[list(train_subset.indices)]
    class_counts = torch.bincount(train_labels, minlength=dataset.num_classes)
    majority_class = int(class_counts.argmax().item())
    correct = 0
    sample_count = 0
    for _, targets in test_loader:
        correct += int((targets == majority_class).sum().item())
        sample_count += targets.shape[0]
    if sample_count == 0:
        raise ValueError("test loader must contain at least one sample")
    return {
        "strategy": "majority_class",
        "class": majority_class,
        "test_accuracy": correct / sample_count,
        "samples": sample_count,
    }


def train_classifier_epoch(
    pipeline: NeuromorphicCognitivePipeline,
    loader: DataLoader,
    task: str,
) -> dict[str, float]:
    pipeline.snn.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for inputs, targets in loader:
        context = TaskRoutingContext(
            domains=(task,),
            entity_ids=tuple(
                f"class_{label}" for label in sorted(set(targets.tolist()))
            ),
        )
        stats = pipeline.process_batch(
            inputs,
            targets,
            train_step=True,
            routing_context=context,
        )
        count = targets.shape[0]
        total_loss += float(stats["loss"]) * count
        correct += round(float(stats["accuracy"]) * count)
        sample_count += count
    if sample_count == 0:
        raise ValueError("training loader must contain at least one sample")
    return {
        "loss": total_loss / sample_count,
        "accuracy": correct / sample_count,
        "samples": float(sample_count),
    }


def run_classification_benchmark(
    task: str = "xor",
    num_samples: Optional[int] = 256,
    time_steps: int = 8,
    input_dim: int = 64,
    batch_size: int = 16,
    epochs: int = 10,
    seed: int = 42,
    data_dir: Optional[str] = None,
    require_n_mnist: bool = False,
    device: Optional[torch.device] = None,
    test_samples: Optional[int] = None,
) -> dict[str, Any]:
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if num_samples is not None and (type(num_samples) is not int or num_samples < 10):
        raise ValueError("num_samples must be None or an integer of at least 10")
    if test_samples is not None and (type(test_samples) is not int or test_samples < 1):
        raise ValueError("test_samples must be None or a positive integer")
    if not isinstance(require_n_mnist, bool):
        raise ValueError("require_n_mnist must be a boolean")

    normalized_task = task.lower()
    if normalized_task not in {"xor", "parity", "classification"}:
        raise ValueError(f"Unsupported benchmark task: {task}")
    class_count = 2 if normalized_task in {"xor", "parity"} else 10
    dataset = NeuromorphicDataset(
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=class_count,
        data_dir=data_dir,
        task=normalized_task,
        seed=seed,
        split="train",
    )
    if require_n_mnist and dataset.data_source != "N-MNIST":
        raise FileNotFoundError(
            "N-MNIST was required but processed_nmnist.pt was not loaded from data_dir"
        )
    if dataset.data_source == "N-MNIST":
        train_subset, validation_subset = split_training_validation(dataset, seed=seed)
        test_dataset = NeuromorphicDataset(
            num_samples=test_samples,
            time_steps=time_steps,
            input_dim=input_dim,
            num_classes=class_count,
            data_dir=data_dir,
            task=normalized_task,
            seed=seed + 1,
            split="test",
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        split_sources = {
            "train": "N-MNIST train",
            "validation": "N-MNIST train",
            "test": "N-MNIST test",
        }
        test_count = len(test_dataset)
    else:
        if num_samples is None:
            raise ValueError("num_samples=None requires processed N-MNIST data")
        train_subset, validation_subset, test_subset = split_dataset(dataset, seed=seed)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        split_sources = {
            "train": dataset.data_source,
            "validation": dataset.data_source,
            "test": dataset.data_source,
        }
        test_count = len(test_subset)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        num_classes=class_count,
        device=device,
    )

    epoch_metrics = []
    best_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_classifier_epoch(pipeline, train_loader, normalized_task)
        validation_metrics = evaluate_classifier(pipeline, validation_loader)
        epoch_metrics.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        validation_accuracy = validation_metrics["accuracy"]
        validation_loss = validation_metrics["loss"]
        if (
            validation_accuracy > best_accuracy
            or (
                validation_accuracy == best_accuracy
                and validation_loss < best_validation_loss
            )
        ):
            best_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(pipeline.snn.state_dict())

    if best_state is None:
        raise RuntimeError("benchmark did not produce a validation checkpoint")
    pipeline.snn.load_state_dict(best_state)
    test_metrics = evaluate_classifier(pipeline, test_loader)
    baseline = majority_class_baseline(dataset, train_subset, test_loader)
    return {
        "task": normalized_task,
        "seed": seed,
        "data_source": dataset.data_source,
        "input_shape": {
            "time_steps": int(dataset.samples.shape[1]),
            "input_dim": int(dataset.samples.shape[2]),
        },
        "split_sizes": {
            "train": len(train_subset),
            "validation": len(validation_subset),
            "test": test_count,
        },
        "split_sources": split_sources,
        "epochs": epoch_metrics,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "baseline": baseline,
    }


def run_sequential_forgetting_benchmark(
    tasks: tuple[str, ...] = ("xor", "parity"),
    num_samples: int = 256,
    time_steps: int = 8,
    input_dim: int = 64,
    batch_size: int = 16,
    epochs_per_task: int = 5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    if len(tasks) < 2 or len(set(tasks)) != len(tasks):
        raise ValueError("at least two distinct tasks are required")
    if any(task not in {"xor", "parity"} for task in tasks):
        raise ValueError("forgetting benchmark supports xor and parity tasks")
    if type(epochs_per_task) is not int or epochs_per_task < 1:
        raise ValueError("epochs_per_task must be a positive integer")

    loaders = {}
    validation_loaders = {}
    test_loaders = {}
    split_sizes = {}
    for task_index, task in enumerate(tasks):
        dataset = NeuromorphicDataset(
            num_samples=num_samples,
            time_steps=time_steps,
            input_dim=input_dim,
            num_classes=2,
            task=task,
            seed=seed + task_index,
        )
        train_subset, validation_subset, test_subset = split_dataset(
            dataset,
            seed=seed + task_index,
        )
        loaders[task] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + task_index + 1),
        )
        validation_loaders[task] = DataLoader(
            validation_subset,
            batch_size=batch_size,
            shuffle=False,
        )
        test_loaders[task] = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        split_sizes[task] = {
            "train": len(train_subset),
            "validation": len(validation_subset),
            "test": len(test_subset),
        }

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        num_classes=2,
        device=device,
    )
    stages = []
    peak_accuracy = {}
    for task in tasks:
        for _ in range(epochs_per_task):
            train_classifier_epoch(pipeline, loaders[task], task)
        measured = {
            seen_task: {
                "validation": evaluate_classifier(pipeline, validation_loaders[seen_task]),
                "test": evaluate_classifier(pipeline, test_loaders[seen_task]),
            }
            for seen_task in tasks[: tasks.index(task) + 1]
        }
        for seen_task, metrics in measured.items():
            peak_accuracy[seen_task] = max(
                peak_accuracy.get(seen_task, 0.0),
                metrics["test"]["accuracy"],
            )
        stages.append({"after_task": task, "metrics": measured})

    final_metrics = stages[-1]["metrics"]
    forgetting = {
        task: max(0.0, peak_accuracy[task] - final_metrics[task]["test"]["accuracy"])
        for task in tasks[:-1]
    }
    return {
        "tasks": list(tasks),
        "seed": seed,
        "split_sizes": split_sizes,
        "epochs_per_task": epochs_per_task,
        "stages": stages,
        "forgetting": forgetting,
    }
