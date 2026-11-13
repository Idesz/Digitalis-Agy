"""
Neuromorphic Dataset and Synthetic DVS Spike Generator.
Supports:
1. Synthetic event generator (Poisson / DVS-like spatio-temporal spike streams)
   ensuring complete out-of-the-box CPU/GPU testing without external download dependencies.
2. Standard N-MNIST binning loader from raw events if dataset files are supplied in data/n_mnist.
"""

from typing import Tuple, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader


class NeuromorphicDataset(Dataset):
    """
    Neuromorphic spiking dataset yielding spatio-temporal spike tensors:
    [T, num_channels, height, width] or flattened [T, input_dim].
    """

    def __init__(
        self,
        num_samples: int = 100,
        time_steps: int = 16,
        input_dim: int = 64,
        num_classes: int = 10,
        firing_prob: float = 0.08,
        data_dir: Optional[str] = None,
        task: str = "classification",
    ):
        super().__init__()
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.firing_prob = firing_prob
        self.data_dir = data_dir
        self.task = task.lower()
        if self.task not in {"classification", "parity", "xor"}:
            raise ValueError(f"Unsupported synthetic task: {task}")
        self.data_source = f"Synthetic {self.task.upper()}" if self.task in {"parity", "xor"} else "Synthetic DVS"

        self.samples, self.labels = self._generate_or_load()

    def _generate_or_load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # If an existing N-MNIST directory with preprocessed files is present, load it
        if (
            self.task == "classification"
            and self.data_dir
            and os.path.exists(os.path.join(self.data_dir, "processed_nmnist.pt"))
        ):
            data = torch.load(os.path.join(self.data_dir, "processed_nmnist.pt"))
            self.data_source = "N-MNIST"
            return data["samples"][: self.num_samples], data["labels"][: self.num_samples]

        # Using a dedicated local Generator so global RNG state is not perturbed
        gen = torch.Generator().manual_seed(42)
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
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


def get_neuromorphic_dataloader(
    batch_size: int = 16,
    num_samples: int = 100,
    time_steps: int = 16,
    input_dim: int = 64,
    num_classes: int = 10,
    data_dir: Optional[str] = None,
    task: str = "classification",
) -> DataLoader:
    dataset = NeuromorphicDataset(
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir=data_dir,
        task=task,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
