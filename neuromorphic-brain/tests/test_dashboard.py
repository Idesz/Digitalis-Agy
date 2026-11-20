import pytest
import urllib.request
import json
import threading
import torch
import dashboard_app
from src.dashboard.monitor import (
    BrainStateMonitor,
    build_dashboard_telemetry,
    global_monitor,
    start_dashboard_server,
)


def test_brain_state_monitor_updates_and_ascii_render():
    monitor = BrainStateMonitor()
    monitor.update({
        "has_metrics": True,
        "curiosity": 0.85,
        "error_rate": 0.12,
        "resource_stress": 0.35,
        "process_cpu_load": 0.3,
        "uncertainty": 0.22,
        "behavioral_mode": "EXPLORATION",
        "last_route": "FAST_SYSTEM_1",
        "last_accuracy": 0.92,
        "last_surprise": 0.045,
        "last_loss": 0.1,
        "last_spike_count": 8,
        "connection_count": 100,
        "layer_activity": [[0.1], [0.2]],
        "output_classes": 2,
        "epoch": 1,
        "epoch_target": 1,
        "total_steps": 1,
        "step": 1,
        "data_source": "Synthetic DVS",
        "working_memory_items": 1,
        "episodic_events_count": 14,
        "semantic_concepts_count": 6,
        "semantic_relations_count": 2,
        "log": "Test cognitive event executed successfully.",
    })

    snap = monitor.get_snapshot()
    assert snap["curiosity"] == 0.85
    assert snap["behavioral_mode"] == "EXPLORATION"
    assert snap["last_accuracy"] == 0.92
    assert snap["has_metrics"] is True
    assert len(snap["recent_logs"]) == 1

    # Check ASCII render
    rendered = monitor.render_terminal_ascii()
    assert "NEUROMORPHIC BRAIN MONITOR" in rendered
    assert "EXPLORATION" in rendered
    assert "FAST_SYSTEM_1" in rendered


def test_tui_does_not_render_default_values_as_telemetry():
    rendered = BrainStateMonitor().render_terminal_ascii()

    assert "Waiting for telemetry data" in rendered
    assert "Curiosity:" not in rendered


@pytest.mark.parametrize(
    ("web", "tui", "no_tui", "simulate", "expected"),
    [
        (False, False, False, False, (True, True, True)),
        (True, False, False, False, (True, False, False)),
        (False, True, False, False, (False, True, True)),
        (True, False, False, True, (True, False, True)),
        (False, False, True, False, (True, False, True)),
    ],
)
def test_dashboard_cli_modes(web, tui, no_tui, simulate, expected):
    args = type(
        "Args",
        (),
        {"web": web, "tui": tui, "no_tui": no_tui, "simulate": simulate},
    )()

    assert dashboard_app._resolve_modes(args) == expected


def test_dashboard_cli_rejects_conflicting_terminal_flags():
    args = type(
        "Args",
        (),
        {"web": False, "tui": True, "no_tui": True, "simulate": False},
    )()

    with pytest.raises(ValueError):
        dashboard_app._resolve_modes(args)


def test_dashboard_server_endpoints():
    server = start_dashboard_server(host="127.0.0.1", port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # GET / (HTML)
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            html = response.read().decode("utf-8")
            assert "Neuromorphic Brain — Control Center" in html
            assert "Neural Activity — Aggregated, Illustrative Layout" in html
            assert 'id="view-lab"' in html
            assert 'id="prompt-help-dialog"' in html
            assert 'id="ranking-type"' in html
            assert "Math.random" not in html
            assert "requestAnimationFrame" not in html
            assert "BATCH ACCURACY" in html
            assert "Waiting for telemetry" in html
            assert 'id="ms-acc">—</div>' in html
            assert 'id="gv-curiosity">—</span>' in html
            assert 'id="mv-episodic">—</span>' in html

        # GET /api/state (JSON)
        req_api = urllib.request.Request(f"{base_url}/api/state")
        with urllib.request.urlopen(req_api) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert "curiosity" in data
            assert "behavioral_mode" in data
            assert data["step"] == 0
            assert data["last_route"] == "—"
            assert data["has_metrics"] is False

        # POST /api/update
        update_payload = json.dumps({"curiosity": 0.99, "log": "External telemetry update"}).encode("utf-8")
        req_post = urllib.request.Request(
            f"{base_url}/api/update",
            data=update_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_post) as response:
            assert response.status == 200
            res_json = json.loads(response.read().decode("utf-8"))
            assert res_json["status"] == "ok"

        with urllib.request.urlopen(f"{base_url}/api/state") as response:
            data = json.loads(response.read().decode("utf-8"))
            assert data["curiosity"] == 0.99
            assert data["has_metrics"] is False
            assert data["recent_logs"][0].endswith("External telemetry update")

        with urllib.request.urlopen(f"{base_url}/api/stream") as response:
            event = response.readline().decode("utf-8")
            assert event.startswith("data: ")
            assert json.loads(event[6:])["has_metrics"] is False

    finally:
        server.shutdown()
        server.server_close()


def test_catalog_and_leaderboard_endpoints_report_supported_items_and_missing_sources():
    server = start_dashboard_server(host="127.0.0.1", port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/catalog") as response:
            catalog = json.loads(response.read().decode("utf-8"))
        workflows = {item["id"]: item for item in catalog["workflows"]}
        assert workflows["xor"]["status"] == "ready"
        assert workflows["parity"]["prompt"] == "parity"
        assert workflows["classification"]["status"] == "ready"
        assert "N-MNIST" in workflows["sequence"]["dataset"]
        assert any(item["name"] == "MMLU" for item in catalog["benchmarks"])
        assert {item["id"] for item in catalog["ranking_types"]} == {
            "lmsys_arena_elo",
            "benchlm",
            "epoch_ai",
        }
        assert all(not item["source_configured"] for item in catalog["ranking_types"])

        with urllib.request.urlopen(f"{base_url}/api/leaderboard?type=benchlm") as response:
            ranking = json.loads(response.read().decode("utf-8"))
        assert ranking["status"] == "not_configured"
        assert ranking["name"] == "BenchLM"
        assert ranking["rows"] == []

        refresh_request = urllib.request.Request(
            f"{base_url}/api/leaderboard/refresh",
            data=json.dumps({"type": "benchlm"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(refresh_request)
        assert error.value.code == 409
        refresh_result = json.loads(error.value.read().decode("utf-8"))
        assert refresh_result["status"] == "not_configured"
        assert refresh_result["type"] == "benchlm"
    finally:
        server.shutdown()
        server.server_close()


def test_training_api_validates_requests_and_returns_plans(monkeypatch):
    server = start_dashboard_server(host="127.0.0.1", port=0)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        plan_request = urllib.request.Request(
            f"{base_url}/api/plan",
            data=json.dumps({
                "prompt": "XOR",
                "epochs": 3,
                "lr": 0.001,
                "batch_size": 4,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(plan_request) as response:
            plan = json.loads(response.read().decode("utf-8"))["plan"]
            assert "XOR tanulás" in plan
            assert "XOR szintetikus tüskesorozat" in plan
            assert "DECOLLE SNN, 2 réteg" in plan

        invalid_plan_request = urllib.request.Request(
            f"{base_url}/api/plan",
            data=json.dumps({"prompt": "XOR", "epochs": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(invalid_plan_request)
        assert error.value.code == 400

        for invalid_epochs in (2.9, True):
            invalid_epoch_request = urllib.request.Request(
                f"{base_url}/api/plan",
                data=json.dumps({"prompt": "XOR", "epochs": invalid_epochs}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(invalid_epoch_request)
            assert error.value.code == 400

        invalid_request = urllib.request.Request(
            f"{base_url}/api/train/start",
            data=json.dumps({"prompt": "test", "epochs": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(invalid_request)
        assert error.value.code == 400
        assert json.loads(error.value.read().decode("utf-8"))["status"] == "error"

        for unsupported_prompt in ("MMLU", "HumanEval", "GSM8K", "MATH benchmark", "MATH"):
            unsupported_request = urllib.request.Request(
                f"{base_url}/api/plan",
                data=json.dumps({"prompt": unsupported_prompt}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(unsupported_request)
            assert error.value.code == 400
            assert "jelenlegi tüskés osztályozóval" in json.loads(
                error.value.read().decode("utf-8")
            )["detail"]

        started = threading.Event()

        def wait_for_stop(prompt, epochs, lr, batch_size):
            global_monitor.update({"training_active": True, "training_status": "RUNNING"})
            started.set()
            global_monitor.training_stop_event.wait(timeout=2)
            global_monitor.update({"training_active": False, "training_status": "STOPPED"})

        monkeypatch.setattr(global_monitor, "_training_loop", wait_for_stop)
        start_request = urllib.request.Request(
            f"{base_url}/api/train/start",
            data=json.dumps({
                "prompt": "parity",
                "epochs": 1,
                "lr": 0.001,
                "batch_size": 4,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(start_request) as response:
            assert json.loads(response.read().decode("utf-8"))["ok"] is True
        assert started.wait(timeout=2)
        stop_request = urllib.request.Request(
            f"{base_url}/api/train/stop",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(stop_request) as response:
            assert json.loads(response.read().decode("utf-8"))["ok"] is True
        global_monitor.training_thread.join(timeout=2)
        assert global_monitor.get_snapshot()["training_status"] == "STOPPED"
    finally:
        server.shutdown()
        server.server_close()


def test_simulation_launcher_keeps_server_available_after_simulation(monkeypatch):
    observed_loader_options = {}
    server_state = {
        "shutdown": False,
        "closed": False,
        "available_after_completion": False,
    }

    class FakeServer:
        def shutdown(self):
            server_state["shutdown"] = True

        def server_close(self):
            server_state["closed"] = True

    class FakeLoader:
        dataset = type("Dataset", (), {"data_source": "N-MNIST", "task": "classification"})()

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([(None, torch.tensor([0]))])

    class FakeAgent:
        def __init__(self, **kwargs):
            self.step = 0
            self.working_mem = type("WorkingMemory", (), {"get": lambda self, key: {"active": True}})()
            self.episodic_mem = type("EpisodicMemory", (), {"num_events": lambda self: 3})()
            self.semantic_mem = type(
                "SemanticMemory",
                (),
                {
                    "num_concepts": lambda self: 2,
                    "num_relations": lambda self: 1,
                },
            )()

        def process_batch(self, inputs, targets, train_step, routing_context=None):
            self.step += 1
            return {
                "step": self.step,
                "loss": 0.4,
                "accuracy": 0.75,
                "surprise": 0.2,
                "state_vector": [0.6, 0.2, 0.3, 0.4],
                "behavioral_mode": "BALANCED",
                "route": "FAST_SYSTEM_1",
                "spike_count": 17,
                "connection_count": 18304,
                "process_cpu_load": 0.25,
                "layer_activity": [[0.1] * 64, [0.2] * 128, [0.3] * 64, [0.4] * 10],
            }

        def sleep_and_consolidate(self):
            return {"consolidated_concepts": 0}

    def create_loader(**kwargs):
        observed_loader_options.update(kwargs)
        return FakeLoader()

    def wait_until_demo_completes(seconds):
        if global_monitor.get_snapshot()["training_status"] == "COMPLETED":
            server_state["available_after_completion"] = not server_state["shutdown"]
            raise KeyboardInterrupt

    monkeypatch.setattr(dashboard_app, "start_dashboard_server", lambda **kwargs: FakeServer())
    monkeypatch.setattr(dashboard_app, "NeuromorphicCognitivePipeline", FakeAgent)
    monkeypatch.setattr(dashboard_app, "get_neuromorphic_dataloader", create_loader)
    monkeypatch.setattr(dashboard_app.time, "sleep", wait_until_demo_completes)

    dashboard_app.run_simulation(server_port=0, show_tui=False)

    snapshot = global_monitor.get_snapshot()
    assert observed_loader_options["data_dir"].endswith("data\\n_mnist")
    assert snapshot["training_status"] == "STOPPED"
    assert snapshot["training_active"] is False
    assert snapshot["epoch"] == 3
    assert snapshot["step"] == 3
    assert snapshot["data_source"] == "N-MNIST"
    assert snapshot["last_loss"] == 0.4
    assert snapshot["total_steps"] == 3
    assert snapshot["last_spike_count"] == 17
    assert snapshot["episodic_events_count"] == 3
    assert snapshot["semantic_relations_count"] == 1
    assert server_state == {
        "shutdown": True,
        "closed": True,
        "available_after_completion": True,
    }


def test_web_only_launcher_waits_without_claiming_metrics(monkeypatch):
    server_state = {"shutdown": False, "closed": False}

    class FakeServer:
        def shutdown(self):
            server_state["shutdown"] = True

        def server_close(self):
            server_state["closed"] = True

    monkeypatch.setattr(dashboard_app, "start_dashboard_server", lambda **kwargs: FakeServer())

    def interrupt_wait(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(dashboard_app.time, "sleep", interrupt_wait)
    dashboard_app.run_simulation(
        server_port=0,
        show_tui=False,
        show_web=True,
        simulate=False,
    )

    snapshot = global_monitor.get_snapshot()
    assert snapshot["training_status"] == "STOPPED"
    assert snapshot["training_active"] is False
    assert snapshot["has_metrics"] is False
    assert snapshot["data_source"] == "—"
    assert server_state == {"shutdown": True, "closed": True}


def test_monitor_requires_explicit_complete_telemetry_flag():
    monitor = BrainStateMonitor()

    assert monitor.get_snapshot()["has_metrics"] is False
    monitor.update({"log": "server started"})
    assert monitor.get_snapshot()["has_metrics"] is False
    monitor.update({"curiosity": 0.82})
    assert monitor.get_snapshot()["has_metrics"] is False
    with pytest.raises(ValueError, match="Incomplete telemetry sample"):
        monitor.update({"has_metrics": True})
    with pytest.raises(ValueError, match="has_metrics must be a boolean"):
        monitor.update({"has_metrics": "true"})
    assert monitor.get_snapshot()["has_metrics"] is False
    monitor.update({"has_metrics": False})
    assert monitor.get_snapshot()["has_metrics"] is False
    monitor.update({"output_classes": 2})
    assert monitor.get_snapshot()["output_classes"] == 2
    assert monitor.get_snapshot()["has_metrics"] is False


def test_monitor_accepts_complete_telemetry_sample():
    monitor = BrainStateMonitor()
    sample = {
        "epoch": 1,
        "epoch_target": 1,
        "total_steps": 1,
        "data_source": "Synthetic DVS",
        "step": 1,
        "curiosity": 0.5,
        "error_rate": 0.1,
        "resource_stress": 0.2,
        "process_cpu_load": 0.3,
        "uncertainty": 0.2,
        "behavioral_mode": "BALANCED",
        "last_route": "FAST_SYSTEM_1",
        "last_accuracy": 0.5,
        "last_surprise": 0.1,
        "last_loss": 1.0,
        "last_spike_count": 2,
        "connection_count": 20,
        "layer_activity": [[0.1], [0.2]],
        "output_classes": 2,
        "working_memory_items": 1,
        "episodic_events_count": 1,
        "semantic_concepts_count": 0,
        "semantic_relations_count": 0,
        "has_metrics": True,
    }

    monitor.update(sample)

    assert monitor.get_snapshot()["has_metrics"] is True


def test_training_loop_uses_learning_rate_and_reports_completion(monkeypatch):
    observed = {}
    loader_options = {}

    class FakeLoader:
        dataset = type("Dataset", (), {"data_source": "Synthetic PARITY"})()

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([(None, torch.tensor([0]))])

    class FakeAgent:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.working_mem = type("WorkingMemory", (), {"get": lambda self, key: {"predicted_class": 0}})()
            self.episodic_mem = type("EpisodicMemory", (), {"num_events": lambda self: 12})()
            self.semantic_mem = type(
                "SemanticMemory",
                (),
                {
                    "num_concepts": lambda self: 5,
                    "num_relations": lambda self: 3,
                },
            )()

        def process_batch(self, inputs, targets, train_step, routing_context=None):
            return {
                "step": 7,
                "loss": 0.25,
                "state_vector": [0.5, 0.1, 0.1, 0.2],
                "behavioral_mode": "BALANCED",
                "route": "FAST_SYSTEM_1",
                "accuracy": 0.75,
                "surprise": 0.1,
                "spike_count": 19,
                "connection_count": 18304,
                "process_cpu_load": 0.3,
                "layer_activity": [[0.1] * 64, [0.2] * 128, [0.3] * 64, [0.4] * 2],
            }

        def start_background_consolidation(self, interval_seconds):
            pass

        def stop_background_consolidation(self):
            pass

        def sleep_and_consolidate(self):
            return {"consolidated_concepts": 0}

    monkeypatch.setattr("src.core.pipeline.NeuromorphicCognitivePipeline", FakeAgent)
    def create_loader(**kwargs):
        loader_options.update(kwargs)
        return FakeLoader()

    monkeypatch.setattr("src.core.dataset.get_neuromorphic_dataloader", create_loader)

    monitor = BrainStateMonitor()
    monitor._training_loop("parity", epochs=1, lr=0.02, batch_size=4)

    assert observed["lr"] == 0.02
    assert observed["num_classes"] == 2
    assert loader_options["task"] == "parity"
    assert loader_options["num_classes"] == 2
    assert loader_options["data_dir"].endswith("data\\n_mnist")
    snapshot = monitor.get_snapshot()
    assert snapshot["training_active"] is False
    assert snapshot["training_status"] == "COMPLETED"
    assert snapshot["last_loss"] == 0.25
    assert snapshot["total_steps"] == 1
    assert snapshot["step"] == 7
    assert snapshot["last_spike_count"] == 19
    assert snapshot["connection_count"] == 18304
    assert snapshot["data_source"] == "Synthetic PARITY"
    assert snapshot["working_memory_items"] == 1
    assert snapshot["episodic_events_count"] == 12
    assert snapshot["semantic_concepts_count"] == 5
    assert snapshot["semantic_relations_count"] == 3


def test_dashboard_telemetry_maps_real_pipeline_metrics():
    class FakeAgent:
        working_mem = type("WorkingMemory", (), {"get": lambda self, key: {"active": True}})()
        episodic_mem = type("EpisodicMemory", (), {"num_events": lambda self: 8})()
        semantic_mem = type(
            "SemanticMemory",
            (),
            {
                "num_concepts": lambda self: 4,
                "num_relations": lambda self: 2,
            },
        )()

    stats = {
        "step": 5,
        "loss": 0.37,
        "accuracy": 0.75,
        "surprise": 0.12,
        "state_vector": [0.8, 0.25, 0.3, 0.4],
        "behavioral_mode": "CAUTIOUS",
        "route": "SLOW_SYSTEM_2",
        "spike_count": 23,
        "connection_count": 18304,
        "process_cpu_load": 0.45,
        "layer_activity": [[0.1], [0.2], [0.3], [0.4, 0.6]],
    }

    telemetry = build_dashboard_telemetry(
        stats,
        FakeAgent(),
        epoch=2,
        epoch_target=5,
        total_steps=25,
        data_source="Synthetic XOR",
    )

    assert telemetry["has_metrics"] is True
    assert telemetry["step"] == 5
    assert telemetry["total_steps"] == 25
    assert telemetry["process_cpu_load"] == 0.45
    assert telemetry["last_loss"] == 0.37
    assert telemetry["last_accuracy"] == 0.75
    assert telemetry["last_spike_count"] == 23
    assert telemetry["data_source"] == "Synthetic XOR"
    assert telemetry["output_classes"] == 2
    assert telemetry["layer_activity"] == stats["layer_activity"]
    assert telemetry["working_memory_items"] == 1
    assert telemetry["episodic_events_count"] == 8
    assert telemetry["semantic_concepts_count"] == 4
    assert telemetry["semantic_relations_count"] == 2


@pytest.mark.parametrize(
    ("prompt", "expected_task"),
    [("binary parity", "BinaryParity klasszifikáció"), ("XOR", "XOR tanulás")],
)
def test_training_plan_matches_supported_task(prompt, expected_task):
    plan = BrainStateMonitor().generate_plan(prompt, epochs=5, lr=0.001, batch_size=4)

    assert expected_task in plan
    assert "Becsült idő" not in plan
    assert "tényleges batch-eket mutatja" in plan


def test_classification_plan_reports_the_n_mnist_or_synthetic_fallback():
    plan = BrainStateMonitor().generate_plan(
        "classification",
        epochs=1,
        lr=0.001,
        batch_size=4,
    )

    assert "N-MNIST" in plan
    assert "Synthetic DVS" in plan


def test_training_loop_preserves_error_status(monkeypatch):
    def fail_to_create_agent(**kwargs):
        raise RuntimeError("model initialization failed")

    monkeypatch.setattr("src.core.pipeline.NeuromorphicCognitivePipeline", fail_to_create_agent)

    monitor = BrainStateMonitor()
    monitor._training_loop("test", epochs=1, lr=0.001, batch_size=4)

    snapshot = monitor.get_snapshot()
    assert snapshot["training_active"] is False
    assert snapshot["training_status"] == "ERROR"
    assert "model initialization failed" in snapshot["recent_logs"][0]


@pytest.mark.parametrize(
    ("prompt", "epochs", "lr", "batch_size"),
    [
        ("", 1, 0.001, 4),
        ("MMLU", 1, 0.001, 4),
        ("test", 0, 0.001, 4),
        ("test", True, 0.001, 4),
        ("test", 1, 0.0, 4),
        ("test", 1, True, 4),
        ("test", 1, float("nan"), 4),
        ("test", 1, float("inf"), 4),
        ("test", 1, 0.001, 0),
        ("test", 1, 0.001, True),
    ],
)
def test_start_training_rejects_invalid_parameters(prompt, epochs, lr, batch_size):
    with pytest.raises(ValueError):
        BrainStateMonitor().start_training(prompt, epochs, lr, batch_size)


def test_training_can_be_stopped_and_duplicate_start_is_rejected(monkeypatch):
    monitor = BrainStateMonitor()
    started = threading.Event()

    def wait_for_stop(prompt, epochs, lr, batch_size):
        monitor.update({"training_active": True, "training_status": "RUNNING"})
        started.set()
        monitor.training_stop_event.wait(timeout=2)
        monitor.update({"training_active": False, "training_status": "STOPPED"})

    monkeypatch.setattr(monitor, "_training_loop", wait_for_stop)

    assert monitor.start_training("test")[0] is True
    assert started.wait(timeout=2)
    assert monitor.start_training("duplicate")[0] is False
    assert monitor.stop_training()[0] is True
    monitor.training_thread.join(timeout=2)

    assert monitor.get_snapshot()["training_status"] == "STOPPED"
    assert monitor.stop_training()[0] is False
