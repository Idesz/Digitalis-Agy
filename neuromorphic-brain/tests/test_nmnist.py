import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.benchmark import evaluate_classifier
from src.core.dataset import NeuromorphicDataset, get_neuromorphic_dataloader
from src.core.nmnist import (
    decode_nmnist_events,
    frame_nmnist_events,
    preprocess_events,
    preprocess_split,
    save_processed_splits,
)


class FixtureEventSplit:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [index], self.labels[index]


def test_preprocess_events_merges_polarities_and_pools_to_binary_8_by_8():
    frames = torch.zeros(16, 2, 34, 34)
    frames[0, 0, 0, 0] = 3
    frames[1, 1, 33, 33] = 2

    processed = preprocess_events([1], lambda events: frames)

    assert processed.shape == (16, 64)
    assert processed.dtype == torch.uint8
    assert set(processed.unique().tolist()) <= {0, 1}
    assert processed[0, 0].item() == 1
    assert processed[1, 63].item() == 1
    assert processed.sum().item() == 2


def test_decode_nmnist_events_handles_polarity_and_timestamp_overflow():
    def event_record(x, y, polarity, timestamp):
        return bytes((
            x,
            y,
            ((timestamp >> 16) & 0x7F) | (polarity << 7),
            (timestamp >> 8) & 0xFF,
            timestamp & 0xFF,
        ))

    raw = (
        event_record(1, 2, 0, 100)
        + event_record(0, 240, 0, 0)
        + event_record(3, 4, 1, 200)
    )
    events = decode_nmnist_events(raw)
    frames = frame_nmnist_events(events)

    assert events["x"].tolist() == [1, 3]
    assert events["y"].tolist() == [2, 4]
    assert events["p"].tolist() == [0, 1]
    assert events["t"].tolist() == [100, 8392]
    assert frames.shape == (16, 2, 34, 34)
    assert frames[0, 0, 2, 1] == 1
    assert frames[15, 1, 4, 3] == 1


def test_preprocess_events_handles_empty_event_samples():
    processed = preprocess_events([], lambda _events: pytest.fail("empty samples have no frames"))

    assert processed.shape == (16, 64)
    assert processed.dtype == torch.uint8
    assert processed.sum().item() == 0


def test_preprocess_split_is_reproducible_capped_and_validates_labels():
    dataset = FixtureEventSplit(list(range(10)))
    frame_transform = lambda _events: torch.zeros(16, 2, 34, 34)

    samples, labels = preprocess_split(dataset, frame_transform, max_samples=4, seed=8)
    repeated_samples, repeated_labels = preprocess_split(
        dataset,
        frame_transform,
        max_samples=4,
        seed=8,
    )

    assert samples.shape == (4, 16, 64)
    assert labels.dtype == torch.long
    assert torch.all((labels >= 0) & (labels <= 9))
    assert torch.equal(samples, repeated_samples)
    assert torch.equal(labels, repeated_labels)


def test_preprocess_split_rejects_labels_outside_digit_classes():
    dataset = FixtureEventSplit([10])
    frame_transform = lambda _events: torch.zeros(16, 2, 34, 34)

    with pytest.raises(ValueError, match=r"integer class indices in \[0, 9\]"):
        preprocess_split(dataset, frame_transform)


def test_processed_train_and_test_splits_remain_separate_and_can_use_all_samples(tmp_path):
    train_samples = torch.zeros(12, 16, 64, dtype=torch.uint8)
    test_samples = torch.ones(5, 16, 64, dtype=torch.uint8)
    train_labels = torch.arange(12).remainder(10)
    test_labels = torch.full((5,), 9, dtype=torch.long)
    output = tmp_path / "processed_nmnist.pt"

    counts = save_processed_splits(
        (train_samples, train_labels),
        (test_samples, test_labels),
        str(output),
        seed=17,
    )
    train_loader = get_neuromorphic_dataloader(
        batch_size=4,
        num_samples=None,
        input_dim=64,
        data_dir=str(tmp_path),
        split="train",
    )
    test_dataset = NeuromorphicDataset(
        num_samples=None,
        input_dim=64,
        data_dir=str(tmp_path),
        split="test",
    )

    assert counts == {"train": 12, "test": 5}
    assert train_loader.dataset.data_source == "N-MNIST"
    assert len(train_loader.dataset) == 12
    assert len(test_dataset) == 5
    assert torch.equal(train_loader.dataset.samples, train_samples.float())
    assert torch.equal(test_dataset.samples, test_samples.float())
    assert torch.equal(test_dataset.labels, test_labels)


def test_test_metric_uses_only_the_supplied_held_out_loader():
    class FixedSNN:
        def eval(self):
            return self

        def forward_sequence(self, inputs):
            logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])[: inputs.shape[0]]
            return [], [logits]

    class FixedPipeline:
        device = torch.device("cpu")
        snn = FixedSNN()

    held_out = DataLoader(
        TensorDataset(
            torch.zeros(2, 1, 1),
            torch.tensor([1, 0]),
        ),
        batch_size=2,
    )
    metrics = evaluate_classifier(FixedPipeline(), held_out)

    assert metrics["samples"] == 2.0
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] < 0.1
