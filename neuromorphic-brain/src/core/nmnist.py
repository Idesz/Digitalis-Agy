"""Download and deterministically preprocess the official N-MNIST event dataset."""

import hashlib
import os
from pathlib import Path
from numbers import Integral
from typing import Any, Callable, Optional
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F


N_MNIST_FORMAT_VERSION = 2
N_MNIST_TIME_BINS = 16
N_MNIST_FEATURES = 64
N_MNIST_SENSOR_SIZE = (34, 34, 2)
N_MNIST_GRID_SIZE = (8, 8)
_NMNIST_BASE_URL = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
_NMNIST_SPLITS = {
    "train": {
        "url": _NMNIST_BASE_URL + "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",
        "filename": "train.zip",
        "folder": "Train",
        "md5": "20959b8e626244a1b502305a9e6e2031",
    },
    "test": {
        "url": _NMNIST_BASE_URL + "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",
        "filename": "test.zip",
        "folder": "Test",
        "md5": "69ca8762b2fe404d9b9bad1103e97832",
    },
}


def decode_nmnist_events(raw: bytes) -> np.ndarray:
    """Decode the official five-byte x/y/p/t N-MNIST event records."""
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size % 5:
        raise ValueError("N-MNIST event files must contain complete five-byte records")
    records = data.reshape(-1, 5).astype(np.uint32)
    if records.shape[0] == 0:
        return np.empty(0, dtype=[("x", "i8"), ("y", "i8"), ("t", "i8"), ("p", "i8")])

    x = records[:, 0]
    y = records[:, 1]
    polarity_and_time = records[:, 2]
    polarity = (polarity_and_time & 0x80) >> 7
    timestamps = (
        ((polarity_and_time & 0x7F) << 16)
        | (records[:, 3] << 8)
        | records[:, 4]
    ).astype(np.uint64)

    overflow = y == 240
    timestamps += np.cumsum(overflow, dtype=np.uint64) * (2**13)
    valid = ~overflow
    if np.any(x[valid] >= 34) or np.any(y[valid] >= 34):
        raise ValueError("N-MNIST event coordinates must be within the 34x34 sensor")

    events = np.empty(
        int(valid.sum()),
        dtype=[("x", "i8"), ("y", "i8"), ("t", "i8"), ("p", "i8")],
    )
    events["x"] = x[valid]
    events["y"] = y[valid]
    events["t"] = timestamps[valid]
    events["p"] = polarity[valid]
    return events


def frame_nmnist_events(events: np.ndarray) -> np.ndarray:
    """Accumulate all events into 16 equal-duration bins shaped [16, 2, 34, 34]."""
    frames = np.zeros((16, 2, 34, 34), dtype=np.uint32)
    if len(events) == 0:
        return frames
    timestamps = events["t"].astype(np.uint64)
    span = int(timestamps[-1] - timestamps[0]) + 1
    time_bins = np.minimum(
        ((timestamps - timestamps[0]) * N_MNIST_TIME_BINS) // span,
        N_MNIST_TIME_BINS - 1,
    )
    np.add.at(
        frames,
        (
            time_bins.astype(np.int64),
            events["p"].astype(np.int64),
            events["y"].astype(np.int64),
            events["x"].astype(np.int64),
        ),
        1,
    )
    return frames


def preprocess_events(
    events: Any,
    frame_transform: Callable[[Any], Any] = frame_nmnist_events,
) -> torch.Tensor:
    """Convert one x/y/t/p event sample to binary [16, 64] spike frames."""
    if len(events) == 0:
        frames = torch.zeros((16, 2, 34, 34), dtype=torch.float32)
    else:
        frames = torch.as_tensor(frame_transform(events))
    if tuple(frames.shape) != (N_MNIST_TIME_BINS, *N_MNIST_SENSOR_SIZE[::-1]):
        raise ValueError("N-MNIST frame transform must produce [16, 2, 34, 34]")
    if frames.is_complex():
        raise ValueError("N-MNIST frames must be real-valued event counts")
    frames = frames.to(dtype=torch.float32)
    if not torch.isfinite(frames).all() or torch.any(frames < 0):
        raise ValueError("N-MNIST frames must contain finite, non-negative event counts")

    merged_polarities = (frames > 0).any(dim=1, keepdim=True).to(dtype=torch.float32)
    pooled = F.adaptive_max_pool2d(merged_polarities, N_MNIST_GRID_SIZE)
    return pooled.reshape(N_MNIST_TIME_BINS, N_MNIST_FEATURES).to(dtype=torch.uint8)


def preprocess_split(
    dataset: Any,
    frame_transform: Callable[[Any], Any] = frame_nmnist_events,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single official split, selecting and ordering samples reproducibly."""
    sample_count = len(dataset)
    if sample_count < 1:
        raise ValueError("N-MNIST split must contain at least one sample")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    selected_count = min(sample_count, max_samples) if max_samples is not None else sample_count
    indices = torch.randperm(
        sample_count,
        generator=torch.Generator().manual_seed(seed),
    )[:selected_count]
    samples = torch.empty((selected_count, 16, 64), dtype=torch.uint8)
    labels = torch.empty(selected_count, dtype=torch.long)
    for output_index, source_index in enumerate(indices.tolist()):
        events, label = dataset[source_index]
        if isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label < 10:
            raise ValueError("N-MNIST labels must be integer class indices in [0, 9]")
        samples[output_index] = preprocess_events(events, frame_transform)
        labels[output_index] = label
        if progress is not None:
            progress(output_index + 1, selected_count)
    return samples, labels


def save_processed_splits(
    train: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    output_path: str,
    seed: int,
) -> dict[str, int]:
    """Persist already separated splits atomically in the loader's safe tensor format."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    payload: dict[str, Any] = {
        "format_version": N_MNIST_FORMAT_VERSION,
        "source": "Official N-MNIST archive",
        "preprocessing": {
            "sensor_size": list(N_MNIST_SENSOR_SIZE),
            "time_bins": N_MNIST_TIME_BINS,
            "feature_grid": list(N_MNIST_GRID_SIZE),
            "polarity_merge": "logical_or",
            "spatial_pooling": "adaptive_max",
            "train_seed": seed,
            "test_seed": seed + 1,
        },
        "train_samples": train[0],
        "train_labels": train[1],
        "test_samples": test[0],
        "test_labels": test[1],
    }
    _validate_processed_split(payload, "train")
    _validate_processed_split(payload, "test")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"train": len(train[1]), "test": len(test[1])}


def _validate_processed_split(payload: dict[str, Any], split: str) -> None:
    samples = payload[f"{split}_samples"]
    labels = payload[f"{split}_labels"]
    if (
        not isinstance(samples, torch.Tensor)
        or samples.dtype != torch.uint8
        or tuple(samples.shape[1:]) != (N_MNIST_TIME_BINS, N_MNIST_FEATURES)
        or samples.shape[0] < 1
        or torch.any(samples > 1)
    ):
        raise ValueError(f"{split} samples must be uint8 tensors shaped [N, 16, 64]")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.shape != (samples.shape[0],)
        or torch.any((labels < 0) | (labels > 9))
    ):
        raise ValueError(f"{split} labels must be int64 tensors shaped [N] in [0, 9]")


def download_and_preprocess_n_mnist(
    data_dir: str,
    output_path: Optional[str] = None,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Download official train/test archives and save the processed tensor format."""
    raw_dir = Path(data_dir) / "raw"
    train_dataset = _ensure_official_split(raw_dir, "train")
    test_dataset = _ensure_official_split(raw_dir, "test")

    def report_progress(name: str) -> Callable[[int, int], None]:
        def report(completed: int, total: int) -> None:
            if completed == 1 or completed % 1000 == 0 or completed == total:
                print(f"{name}: processed {completed}/{total} samples")
        return report

    train = preprocess_split(
        train_dataset,
        max_samples=train_samples,
        seed=seed,
        progress=report_progress("train"),
    )
    test = preprocess_split(
        test_dataset,
        max_samples=test_samples,
        seed=seed + 1,
        progress=report_progress("test"),
    )
    target = output_path or str(Path(data_dir) / "processed_nmnist.pt")
    counts = save_processed_splits(train, test, target, seed)
    return {"output_path": target, **counts}


class _NMNISTArchiveSplit:
    def __init__(self, split_root: Path):
        self.files = sorted(split_root.rglob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No N-MNIST .bin event files found under {split_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        path = self.files[index]
        try:
            label = int(path.parent.name)
        except ValueError as exc:
            raise ValueError(f"N-MNIST sample has no digit-class directory: {path}") from exc
        if not 0 <= label < 10:
            raise ValueError(f"N-MNIST sample has invalid digit label: {path}")
        return decode_nmnist_events(path.read_bytes()), label


def _ensure_official_split(raw_dir: Path, split: str) -> _NMNISTArchiveSplit:
    metadata = _NMNIST_SPLITS[split]
    split_root = raw_dir / metadata["folder"]
    expected_samples = 60000 if split == "train" else 10000
    sample_count = sum(1 for _ in split_root.rglob("*.bin"))
    if sample_count != expected_samples:
        archive_path = raw_dir / metadata["filename"]
        if not archive_path.is_file() or _md5(archive_path) != metadata["md5"]:
            raw_dir.mkdir(parents=True, exist_ok=True)
            temporary = archive_path.with_name(archive_path.name + ".download")
            try:
                previous_reported = 0

                def report_download(block_count: int, block_size: int, total_size: int) -> None:
                    nonlocal previous_reported
                    downloaded = block_count * block_size
                    if block_count == 0 or downloaded - previous_reported >= 50 * 1024 * 1024:
                        total = f"/{total_size / (1024 * 1024):.0f} MiB" if total_size > 0 else ""
                        print(f"Downloading N-MNIST {split}: {downloaded / (1024 * 1024):.0f} MiB{total}")
                        previous_reported = downloaded

                urlretrieve(metadata["url"], temporary, reporthook=report_download)
                if _md5(temporary) != metadata["md5"]:
                    raise ValueError(f"MD5 checksum mismatch for official N-MNIST {split} archive")
                temporary.replace(archive_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        _extract_zip_safely(archive_path, raw_dir)
    sample_count = sum(1 for _ in split_root.rglob("*.bin"))
    if sample_count != expected_samples:
        raise ValueError(
            f"Official N-MNIST {split} split must contain {expected_samples} samples, "
            f"found {sample_count}"
        )
    return _NMNISTArchiveSplit(split_root)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip_safely(archive_path: Path, extract_root: Path) -> None:
    root = extract_root.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative_path = Path(member.filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe path in N-MNIST archive: {member.filename}")
            destination = (root / relative_path).resolve()
            if os.path.commonpath((str(root), str(destination))) != str(root):
                raise ValueError(f"Unsafe path in N-MNIST archive: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
