import pytest
import torch
from src.core.dataset import NeuromorphicDataset, get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline, _normalized_process_cpu_load
from src.dashboard.monitor import build_dashboard_telemetry


def test_neuromorphic_dataset_and_loader():
    num_samples = 20
    time_steps = 8
    input_dim = 32
    num_classes = 4
    batch_size = 5

    loader = get_neuromorphic_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    batches = list(loader)
    assert len(batches) == num_samples // batch_size
    sample_x, sample_y = batches[0]
    assert sample_x.shape == (batch_size, time_steps, input_dim)
    assert sample_y.shape == (batch_size,)
    assert torch.all((sample_x == 0.0) | (sample_x == 1.0))
    assert loader.dataset.data_source == "Synthetic DVS"


@pytest.mark.parametrize(("task", "bit_count"), [("xor", 2), ("parity", 8)])
def test_binary_synthetic_tasks_match_encoded_bits(task, bit_count):
    dataset = NeuromorphicDataset(num_samples=64, input_dim=64, task=task)
    encoded_bits = dataset.samples[:, :, : bit_count * 2].mean(dim=1)
    encoded_bits = encoded_bits.reshape(dataset.num_samples, bit_count, 2).argmax(dim=2)

    assert torch.equal(dataset.labels, encoded_bits.sum(dim=1).remainder(2))
    assert set(dataset.labels.tolist()) == {0, 1}
    assert dataset.data_source == f"Synthetic {task.upper()}"


def test_dataset_loads_local_preprocessed_nmnist_and_reports_source(tmp_path):
    samples = torch.zeros(3, 4, 8)
    labels = torch.tensor([0, 1, 2])
    torch.save({"samples": samples, "labels": labels}, tmp_path / "processed_nmnist.pt")

    dataset = NeuromorphicDataset(
        num_samples=2,
        time_steps=4,
        input_dim=8,
        num_classes=3,
        data_dir=str(tmp_path),
    )

    assert dataset.data_source == "N-MNIST"
    assert len(dataset) == 2
    assert torch.equal(dataset.samples, samples[:2])
    assert torch.equal(dataset.labels, labels[:2])


def test_full_cognitive_pipeline_cycle():
    device = torch.device("cpu")
    input_dim = 16
    hidden_dims = [32, 16]
    num_classes = 4

    pipeline = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        device=device,
        lr=1e-3,
    )

    inputs = (torch.rand(4, 6, input_dim) > 0.8).float()
    targets = torch.randint(0, num_classes, (4,))

    stats = pipeline.process_batch(inputs, targets, train_step=True)

    assert stats["step"] == 1
    assert stats["loss"] == pytest.approx(sum(stats["layer_losses"]) / len(stats["layer_losses"]))
    assert stats["spike_count"] >= 0
    assert stats["connection_count"] == 1216
    assert [len(layer) for layer in stats["layer_activity"]] == [16, 32, 16, num_classes]
    assert "accuracy" in stats
    assert "surprise" in stats
    assert "route" in stats
    assert stats["route"] in ["FAST_SYSTEM_1", "SLOW_SYSTEM_2"]
    assert "behavioral_mode" in stats
    assert len(stats["state_vector"]) == 4
    assert 0.0 <= stats["process_cpu_load"] <= 1.0
    assert stats["state_vector"][2] == pytest.approx(0.08 + 0.2 * stats["process_cpu_load"])

    telemetry = build_dashboard_telemetry(
        stats,
        pipeline,
        epoch=1,
        epoch_target=1,
        total_steps=1,
        data_source="Synthetic DVS",
    )
    assert telemetry["last_loss"] == stats["loss"]
    assert telemetry["last_spike_count"] == stats["spike_count"]
    assert telemetry["connection_count"] == stats["connection_count"]
    assert telemetry["process_cpu_load"] == stats["process_cpu_load"]
    assert telemetry["total_steps"] == 1
    assert telemetry["working_memory_items"] == 1
    assert telemetry["episodic_events_count"] == 1
    assert telemetry["data_source"] == "Synthetic DVS"

    # Run sleep consolidation
    c_stats = pipeline.sleep_and_consolidate()
    assert "processed_events" in c_stats
    assert c_stats["processed_events"] >= 1


@pytest.mark.parametrize(
    ("cpu_time_delta", "wall_time_delta", "expected"),
    [(0.25, 0.5, 0.5), (2.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
)
def test_process_cpu_load_is_normalized_and_bounded(cpu_time_delta, wall_time_delta, expected):
    assert _normalized_process_cpu_load(cpu_time_delta, wall_time_delta) == pytest.approx(expected)
