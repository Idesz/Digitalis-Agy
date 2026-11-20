import pytest
import torch
from src.core.dataset import NeuromorphicDataset, get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline, _normalized_process_cpu_load
from src.memory.episodic_memory import EpisodicEvent
from src.routing.cdr_router import SystemRoute, TaskRoutingContext
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
    samples = torch.zeros(3, 16, 64, dtype=torch.uint8)
    labels = torch.tensor([0, 1, 2])
    torch.save(
        {
            "format_version": 2,
            "train_samples": samples,
            "train_labels": labels,
            "test_samples": samples[:2],
            "test_labels": labels[:2],
        },
        tmp_path / "processed_nmnist.pt",
    )

    dataset = NeuromorphicDataset(
        num_samples=2,
        time_steps=16,
        input_dim=64,
        num_classes=3,
        data_dir=str(tmp_path),
    )

    assert dataset.data_source == "N-MNIST"
    assert len(dataset) == 2
    assert torch.equal(dataset.samples, samples[:2])
    assert torch.equal(dataset.labels, labels[:2])


def test_dataset_rejects_malformed_preprocessed_nmnist(tmp_path):
    torch.save(
        {
            "format_version": 2,
            "train_samples": torch.zeros(2, 16, 64, dtype=torch.uint8),
            "train_labels": torch.tensor([0]),
            "test_samples": torch.zeros(1, 16, 64, dtype=torch.uint8),
            "test_labels": torch.tensor([0]),
        },
        tmp_path / "processed_nmnist.pt",
    )

    with pytest.raises(ValueError, match=r"labels must have shape"):
        NeuromorphicDataset(
            num_samples=2,
            time_steps=16,
            input_dim=64,
            data_dir=str(tmp_path),
        )


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


def test_pipeline_records_only_observed_synaptic_coactivations(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=2,
        hidden_dims=[2, 2],
        num_classes=2,
        device=torch.device("cpu"),
    )
    layer_spikes = [
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
    ]
    readouts = [torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])]
    monkeypatch.setattr(pipeline.snn, "forward_sequence", lambda inputs: (layer_spikes, readouts))

    stats = pipeline.process_batch(
        inputs=torch.zeros(1, 3, 2),
        targets=torch.zeros(1, dtype=torch.long),
        train_step=False,
    )

    assert set(pipeline.synaptic_graph.edges) == {
        "layer_0:neuron_0->layer_1:neuron_0",
        "layer_0:neuron_1->layer_1:neuron_0",
    }
    assert pipeline.synaptic_graph.edges["layer_0:neuron_0->layer_1:neuron_0"].weight > (
        pipeline.synaptic_graph.initial_weight
    )
    assert stats["synaptic_graph_nodes"] == 3
    assert stats["synaptic_graph_edges"] == 2


def test_pipeline_prunes_inactive_synapses_periodically():
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=2,
        hidden_dims=[2, 2],
        num_classes=2,
        device=torch.device("cpu"),
    )
    pipeline.synaptic_graph.record_coactivation("old_source", "old_target", timestamp=0.0)
    pipeline.step = 99

    stats = pipeline.process_batch(
        inputs=torch.zeros(1, 2, 2),
        targets=torch.zeros(1, dtype=torch.long),
        train_step=False,
    )

    assert "old_source->old_target" not in pipeline.synaptic_graph.edges
    assert stats["pruned_synaptic_edges"] == 1


def test_synaptic_graph_remains_stable_for_fifty_training_epochs():
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4, 4],
        num_classes=2,
        device=torch.device("cpu"),
    )
    inputs = torch.ones(1, 2, 4)
    targets = torch.zeros(1, dtype=torch.long)

    try:
        for _ in range(50):
            pipeline.process_batch(inputs, targets, train_step=True)

        assert pipeline.step == 50
        assert all(torch.isfinite(layer.linear.weight).all() for layer in pipeline.snn.layers)
        assert all(
            torch.isfinite(torch.tensor(edge.weight))
            for edge in pipeline.synaptic_graph.edges.values()
        )
        assert all(
            len(node.activation_history) <= 100
            for node in pipeline.synaptic_graph.nodes.values()
        )
        max_edges = sum(
            source_layer.out_features * target_layer.out_features
            for source_layer, target_layer in zip(
                pipeline.snn.layers,
                pipeline.snn.layers[1:],
            )
        )
        assert len(pipeline.synaptic_graph.edges) <= max_edges
    finally:
        torch.set_num_threads(original_num_threads)


def test_cautious_homeostasis_forces_system_two(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4],
        num_classes=2,
        device=torch.device("cpu"),
    )
    pipeline.homeostasis.error_rate = 0.9
    monkeypatch.setattr(pipeline.router, "route", lambda **kwargs: (SystemRoute.FAST, 0.1, {}))
    inputs = (torch.rand(2, 2, 4) > 0.8).float()
    targets = torch.zeros(2, dtype=torch.long)

    stats = pipeline.process_batch(inputs, targets, train_step=False)

    assert stats["behavioral_mode"] == "CAUTIOUS"
    assert stats["route"] == "SLOW_SYSTEM_2"
    assert stats["rollout_skipped"] is True
    assert stats["rollout_skip_reason"] == "insufficient_observed_transitions"


def test_resource_conservation_skips_world_model_rollout(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4],
        num_classes=2,
        device=torch.device("cpu"),
    )
    pipeline.homeostasis.resource_stress = 0.99
    monkeypatch.setattr(pipeline.router, "route", lambda **kwargs: (SystemRoute.SLOW, 0.9, {}))
    monkeypatch.setattr(
        pipeline.world_model,
        "simulate_rollout",
        lambda *args: pytest.fail("Rollout should be skipped in conservation mode"),
    )
    inputs = (torch.rand(2, 2, 4) > 0.8).float()
    targets = torch.zeros(2, dtype=torch.long)

    stats = pipeline.process_batch(inputs, targets, train_step=False)

    assert stats["behavioral_mode"] == "CONSERVATION"
    assert stats["route"] == "SLOW_SYSTEM_2"
    assert stats["rollout_skipped"] is True
    assert stats["rollout_skip_reason"] == "resource_conservation"
    assert stats["simulated_rollout_val"] is None


def test_predictive_surprise_escalates_route(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4],
        num_classes=2,
        device=torch.device("cpu"),
        prediction_escalation_threshold=0.5,
    )
    observed_router_args = {}
    monkeypatch.setattr(
        pipeline.predictive,
        "gated_loss",
        lambda inputs: (torch.tensor(0.0), torch.tensor(0.9), False),
    )

    def fast_route(**kwargs):
        observed_router_args.update(kwargs)
        return SystemRoute.FAST, 0.1, {}

    monkeypatch.setattr(pipeline.router, "route", fast_route)
    inputs = (torch.rand(5, 4, 4) > 0.8).float()
    targets = torch.zeros(5, dtype=torch.long)
    transition_actions = torch.zeros(5, 3, 5)
    transition_actions[:, :, 0] = 1.0

    stats = pipeline.process_batch(
        inputs,
        targets,
        train_step=True,
        transition_actions=transition_actions,
    )

    assert observed_router_args["uncertainty"] == pytest.approx(0.9)
    assert stats["raw_uncertainty"] == pytest.approx(0.9)
    assert stats["calibrated_uncertainty"] == pytest.approx(0.9)
    assert stats["uncertainty_calibration_ready"] is False
    assert stats["prediction_escalated"] is True
    assert stats["route"] == "SLOW_SYSTEM_2"
    assert stats["world_model_transition_count"] == 15
    assert stats["selected_world_model_action"] in range(5)
    assert len(stats["planned_action_sequence"]) == 3
    assert stats["simulated_rollout_risk"] is not None


def test_pipeline_includes_representation_drift_in_uncertainty(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4],
        num_classes=2,
        device=torch.device("cpu"),
    )
    representations = [
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * 2),
        torch.tensor([[[0.0, 1.0, 0.0, 0.0]] * 2] * 2),
    ]

    def forward_sequence(_inputs):
        spikes = representations.pop(0)
        readouts = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        return [spikes], [readouts]

    monkeypatch.setattr(pipeline.snn, "forward_sequence", forward_sequence)
    monkeypatch.setattr(
        pipeline.predictive,
        "gated_loss",
        lambda inputs: (torch.tensor(0.0), torch.tensor(0.1), False),
    )
    inputs = torch.zeros(2, 2, 4)
    targets = torch.zeros(2, dtype=torch.long)

    first = pipeline.process_batch(inputs, targets, train_step=False)
    second = pipeline.process_batch(inputs, targets, train_step=False)

    assert first["representation_change"] == 0.0
    assert second["representation_change"] == pytest.approx(0.5)
    assert second["raw_uncertainty"] == pytest.approx(0.5)
    assert second["uncertainty_calibration_ready"] is False


def test_pipeline_routes_using_task_context_and_semantic_dependencies(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4, 2],
        num_classes=2,
        device=torch.device("cpu"),
    )
    pipeline.semantic_mem.add_concept("entity_a", {"domain": "math"})
    pipeline.semantic_mem.add_concept("entity_b", {"domain": "safety"})
    pipeline.semantic_mem.add_concept("action_3", {"action_index": 3})
    pipeline.semantic_mem.add_relation("entity_a", "entity_b", "depends_on")
    pipeline.semantic_mem.add_relation(
        "entity_a",
        "action_3",
        "suggests_action",
        weight=0.8,
    )
    event = EpisodicEvent(
        embedding=[1.0, 0.0],
        context={
            "concept": "entity_a",
            "related_to": "entity_b",
            "domain": "vision",
            "observed_action_index": 1,
            "selected_world_model_action": 4,
        },
    )
    monkeypatch.setattr(pipeline.episodic_mem, "search_similar", lambda *args, **kwargs: [(event, 0.9)])
    pipeline.world_model_transition_count = 10
    observed = {}

    def capture_route(**kwargs):
        observed.update(kwargs)
        return SystemRoute.SLOW, 0.9, {}

    monkeypatch.setattr(pipeline.router, "route", capture_route)
    original_plan_candidates = pipeline.planner.plan_candidates

    def capture_plan_candidates(initial_latent, horizon=3, action_priors=None):
        observed["action_priors"] = action_priors
        return original_plan_candidates(
            initial_latent,
            horizon=horizon,
            action_priors=action_priors,
        )

    monkeypatch.setattr(pipeline.planner, "plan_candidates", capture_plan_candidates)
    stats = pipeline.process_batch(
        inputs=torch.zeros(1, 2, 4),
        targets=torch.zeros(1, dtype=torch.long),
        train_step=False,
        transition_actions=torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]]),
        routing_context=TaskRoutingContext(
            domains=("finance",),
            entity_ids=("entity_a",),
        ),
    )

    assert observed["correlation_strength"] == pytest.approx(0.9)
    assert observed["domain_crossing"] == pytest.approx(0.75)
    assert observed["stakeholder_count"] == 2
    assert observed["action_priors"] == pytest.approx([0.0, 1.0, 0.0, 0.8 / 0.9, 0.0])
    assert stats["planning_action_priors"] == pytest.approx(observed["action_priors"])
    stored_event = pipeline.episodic_mem._fallback_store.all_events()[-1]
    assert stored_event.context["observed_action_index"] == 2


def test_fast_route_does_not_run_planner_when_model_is_ready(monkeypatch):
    pipeline = NeuromorphicCognitivePipeline(
        input_dim=4,
        hidden_dims=[4],
        num_classes=2,
        device=torch.device("cpu"),
    )
    pipeline.world_model_transition_count = 10
    monkeypatch.setattr(
        pipeline.predictive,
        "gated_loss",
        lambda inputs: (torch.tensor(0.0), torch.tensor(0.1), False),
    )
    monkeypatch.setattr(
        pipeline.router,
        "route",
        lambda **kwargs: (SystemRoute.FAST, 0.1, {}),
    )
    monkeypatch.setattr(
        pipeline.planner,
        "plan_candidates",
        lambda *args, **kwargs: pytest.fail("Planner should not run on the fast route"),
    )

    stats = pipeline.process_batch(
        inputs=torch.zeros(1, 2, 4),
        targets=torch.zeros(1, dtype=torch.long),
        train_step=False,
    )

    assert stats["route"] == "FAST_SYSTEM_1"
    assert stats["planning_action_priors"] is None


@pytest.mark.parametrize(
    ("cpu_time_delta", "wall_time_delta", "expected"),
    [(0.25, 0.5, 0.5), (2.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
)
def test_process_cpu_load_is_normalized_and_bounded(cpu_time_delta, wall_time_delta, expected):
    assert _normalized_process_cpu_load(cpu_time_delta, wall_time_delta) == pytest.approx(expected)
