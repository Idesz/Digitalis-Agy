"""
Neuromorphic Dataset and Synthetic DVS Spike Generator.
Supports:
1. Synthetic event generator (Poisson / DVS-like spatio-temporal spike streams)
   ensuring complete out-of-the-box CPU/GPU testing without external download dependencies.
2. Loader for locally preprocessed N-MNIST spike tensors.
"""

from typing import Tuple, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader

from src.core.nmnist import N_MNIST_FORMAT_VERSION


class NeuromorphicDataset(Dataset):
    """
    Neuromorphic spiking dataset yielding spatio-temporal spike tensors:
    [T, num_channels, height, width] or flattened [T, input_dim].
    """

    def __init__(
        self,
        num_samples: Optional[int] = 100,
        time_steps: int = 16,
        input_dim: int = 64,
        num_classes: int = 10,
        firing_prob: float = 0.08,
        data_dir: Optional[str] = None,
        task: str = "classification",
        seed: int = 42,
        split: str = "train",
    ):
        super().__init__()
        if num_samples is not None and (type(num_samples) is not int or num_samples < 1):
            raise ValueError("num_samples must be a positive integer or None")
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.firing_prob = firing_prob
        self.data_dir = data_dir
        self.task = task.lower()
        self.seed = seed
        self.split = split
        if self.task not in {"classification", "parity", "xor"}:
            raise ValueError(f"Unsupported synthetic task: {task}")
        self.data_source = f"Synthetic {self.task.upper()}" if self.task in {"parity", "xor"} else "Synthetic DVS"

        self.samples, self.labels = self._generate_or_load()
        self.num_samples = len(self.samples)

    def _generate_or_load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the official train or test split from the versioned preprocessed file.
        if (
            self.task == "classification"
            and self.data_dir
            and os.path.exists(os.path.join(self.data_dir, "processed_nmnist.pt"))
        ):
            path = os.path.join(self.data_dir, "processed_nmnist.pt")
            data = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(data, dict) or data.get("format_version") != N_MNIST_FORMAT_VERSION:
                raise ValueError("Unsupported N-MNIST format; run prepare_n_mnist.py to create it")
            sample_key = f"{self.split}_samples"
            label_key = f"{self.split}_labels"
            if sample_key not in data or label_key not in data:
                raise ValueError(f"processed_nmnist.pt must contain the {self.split} split")
            samples = data[sample_key]
            labels = data[label_key]
            if not isinstance(samples, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError("N-MNIST samples and labels must be tensors")
            if (
                samples.ndim != 3
                or samples.shape[0] < 1
                or samples.shape[1] != 16
                or samples.shape[2] != self.input_dim
                or samples.dtype != torch.uint8
            ):
                raise ValueError(
                    f"N-MNIST samples must be uint8 tensors shaped [N, 16, {self.input_dim}]"
                )
            if labels.ndim != 1 or labels.shape[0] != samples.shape[0]:
                raise ValueError("N-MNIST labels must have shape [N] matching the samples")
            if labels.dtype != torch.long:
                raise ValueError("N-MNIST labels must use int64 dtype")
            if torch.any(samples > 1):
                raise ValueError("N-MNIST samples must contain binary 0/1 values")
            samples = samples.to(dtype=torch.float32)
            if torch.any((labels < 0) | (labels >= self.num_classes)):
                raise ValueError(f"N-MNIST labels must be in [0, {self.num_classes - 1}]")
            self.data_source = "N-MNIST"
            if self.num_samples is None:
                return samples, labels
            return samples[: self.num_samples], labels[: self.num_samples]

        # Using a dedicated local Generator so global RNG state is not perturbed
        if self.num_samples is None:
            raise ValueError("num_samples=None requires a processed N-MNIST dataset")
        gen = torch.Generator().manual_seed(self.seed)
        if self.task in {"parity", "xor"}:
            bit_count = 2 if self.task == "xor" else 8
            if self.input_dim < bit_count * 2:
                raise ValueError(f"input_dim must be at least {bit_count * 2} for the {self.task} task")
            bits = torch.randint(0, 2, (self.num_samples, bit_count), generator=gen)
            labels = bits.sum(dim=1).remainder(2)
            samples = (
                torch.rand(self.num_samples, self.time_steps, self.input_dim, generator=gen)
                < self.firing_prob * 0.3
            ).float()
            signal_rate = min(1.0, self.firing_prob * 8.0)
            signal = (
                torch.rand(self.num_samples, self.time_steps, bit_count, generator=gen)
                < signal_rate
            ).float()
            features = 2 * torch.arange(bit_count) + bits
            samples.scatter_(
                2,
                features.unsqueeze(1).expand(-1, self.time_steps, -1),
                signal,
            )
            return samples, labels

        # Generate synthetic DVS Poisson-like spike patterns with class-conditioned signatures
        labels = torch.randint(0, self.num_classes, (self.num_samples,), generator=gen)

        samples = torch.zeros(self.num_samples, self.time_steps, self.input_dim)
        band_width = max(1, self.input_dim // self.num_classes)
        for i in range(self.num_samples):
            cls = labels[i].item()
            # Active sub-band of neurons firing preferentially for this class
            start_idx = (cls * band_width) % self.input_dim
            end_idx = min(start_idx + band_width, self.input_dim)

            # Background noise
            noise = (torch.rand(self.time_steps, self.input_dim, generator=gen) < (self.firing_prob * 0.3)).float()
            # Signal
            signal_width = max(1, end_idx - start_idx)
            signal = (torch.rand(self.time_steps, signal_width, generator=gen) < (self.firing_prob * 3.0)).float()

            samples[i] = noise
            samples[i, :, start_idx:end_idx] = signal[:, : end_idx - start_idx]

        return samples, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


def get_neuromorphic_dataloader(
    batch_size: int = 16,
    num_samples: Optional[int] = 100,
    time_steps: int = 16,
    input_dim: int = 64,
    num_classes: int = 10,
    data_dir: Optional[str] = None,
    task: str = "classification",
    split: str = "train",
    seed: int = 42,
) -> DataLoader:
    dataset = NeuromorphicDataset(
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir=data_dir,
        task=task,
        split=split,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(seed),
    )
