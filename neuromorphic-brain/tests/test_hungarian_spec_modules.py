import pytest
import time
from src.core.synaptic_graph import SynapticGraphManager, Node, Edge
from src.memory.hierarchy_controller import MemoryHierarchyController, SleepConsolidationWorker
from src.homeostasis.homeostatic_controller import GlobalStateVector, HomeostaticController
from src.predictive.predictive_unit import PredictiveProcessingUnit
from src.routing.dual_process_router import DualProcessRouter, System1FastEngine, System2SlowPlanner
from src.world_model.jepa_sandbox import JEPASandboxWorldModel


def test_synaptic_graph_manager():
    sg = SynapticGraphManager(learning_rate=0.2, decay_factor=0.05, initial_weight=0.3)
    sg.add_node("n1")
    sg.add_node("n2")
    assert "n1" in sg.nodes
    assert "n2" in sg.nodes

    # Activate nodes together
    now = time.time()
    sg.trigger_activation("n1", timestamp=now)
    sg.trigger_activation("n2", timestamp=now)

    sg.update_local_learning("n1", "n2", time_window=0.5)
    edge_key = "n1->n2"
    assert edge_key in sg.edges
    assert sg.edges[edge_key].weight == 0.3

    # Subsequent activation strengthens weight
    sg.trigger_activation("n1", timestamp=now + 0.1)
    sg.trigger_activation("n2", timestamp=now + 0.1)
    sg.update_local_learning("n1", "n2", time_window=0.5)
    assert sg.edges[edge_key].weight > 0.3

    # Test edge pruning
    sg.edges[edge_key].last_activated = now - 20.0
    sg.prune_inactive_edges(threshold=0.5, max_idle_time=5.0)
    assert edge_key not in sg.edges


def test_memory_hierarchy_controller_and_sleep_worker():
    controller = MemoryHierarchyController()
    controller.write_working_memory("active_task", {"step": 1})
    assert controller.working_memory["active_task"] == {"step": 1}

    # Store episodic event
    controller.store_episodic_event(
        event_type="perception",
        payload={"subject": "conceptA", "predicate": "conceptB"},
        vector_embedding=[0.1, 0.2, 0.3],
    )
    assert len(controller.episodic_memory) == 1
    assert not controller.episodic_memory[0]["consolidated"]

    worker = SleepConsolidationWorker(controller, cycle_threshold=10)
    assert worker.should_trigger_consolidation(current_cycle_count=10, is_idle=False)
    assert worker.should_trigger_consolidation(current_cycle_count=1, is_idle=True)

    worker.run_sleep_consolidation(noise_threshold_age=3600.0)
    assert controller.episodic_memory[0]["consolidated"]
    assert "conceptA" in controller.semantic_graph
    assert "conceptB" in controller.semantic_graph["conceptA"]
    assert len(controller.working_memory) == 0


def test_homeostatic_controller():
    state = GlobalStateVector(curiosity=0.8, error_rate=0.05, energy_level=1.0, uncertainty=0.1)
    hc = HomeostaticController(state)

    hc.update_state(error_delta=0.3)
    assert state.error_rate == 0.35
    mod = hc.evaluate_and_modulate()
    assert mod["trigger_safety_protocol"] is True
    assert mod["recommended_system1_threshold"] == 0.95


def test_predictive_processing_unit():
    ppu = PredictiveProcessingUnit(default_threshold=0.15)
    current = {"feature_1": 1.0, "feature_2": 2.0}
    predicted = ppu.predict_next_state(current)
    assert predicted["feature_1"] > 1.0

    actual = {"feature_1": 1.0, "feature_2": 2.0}
    err = ppu.calculate_residual_error(actual, predicted)
    assert err > 0.0

    exceeds, msg = ppu.route_signal(residual_error=0.25, threshold=0.15)
    assert exceeds is True
    assert "ROUTED_TO_UPPER_LAYERS" in msg


def test_dual_process_router():
    router = DualProcessRouter(default_threshold=0.80)
    # Simple task -> System 1
    res_fast = router.process_input("simple routine job")
    assert res_fast["execution_path"] == "SYSTEM_1_FAST"

    # Complex task -> System 2
    res_slow = router.process_input("hard complex optimization problem")
    assert res_slow["execution_path"] == "SYSTEM_2_SLOW"


def test_jepa_sandbox_world_model():
    model = JEPASandboxWorldModel()
    raw = {"x": 10, "status": "nominal"}
    abstract = model.abstract_state_encoder(raw)
    assert "abstract_x" in abstract

    # Action candidate count validation [5, 10]
    with pytest.raises(ValueError):
        model.virtual_rollout(["act1", "act2"])  # Too few (<5)

    candidates = [f"candidate_action_{i}" for i in range(6)]
    candidates.append("unsafe_risk_action")
    outcomes = model.virtual_rollout(candidates, depth=5)
    assert len(outcomes) == 7

    best = model.evaluate_and_select_path(outcomes)
    assert "action" in best
    assert "net_score" in best
