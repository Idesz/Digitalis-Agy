import pytest
import time
import threading
import numpy as np
import sys
from types import SimpleNamespace
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory
from src.memory.sleep_consolidation import SleepConsolidator


def test_working_memory_ttl_and_operations():
    wm = WorkingMemory(force_fallback=True)
    wm.put("test_key", {"state": "active", "value": 42}, ttl=1)
    data = wm.get("test_key")
    assert data is not None
    assert data["value"] == 42

    # Verify key deletion
    wm.put("to_delete", "val")
    assert wm.get("to_delete") == "val"
    wm.delete("to_delete")
    assert wm.get("to_delete") is None

    # Test TTL expiration
    time.sleep(1.1)
    assert wm.get("test_key") is None


def test_working_memory_clear_only_removes_its_namespace(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.values = {"shared:key": "keep"}

        def ping(self):
            return True

        def set(self, name, value, ex=None):
            self.values[name] = value
            return True

        def get(self, name):
            return self.values.get(name)

        def delete(self, *names):
            for name in names:
                self.values.pop(name, None)

        def scan_iter(self, match):
            prefix = match.removesuffix("*")
            return iter([key for key in self.values if key.startswith(prefix)])

    client = FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=lambda **kwargs: client))
    memory = WorkingMemory()
    memory.put("active", {"step": 1})

    memory.clear()

    assert client.values == {"shared:key": "keep"}


@pytest.mark.parametrize("ttl", [0, -1, True, 1.5])
def test_working_memory_rejects_invalid_ttl(ttl):
    with pytest.raises(ValueError, match="ttl must be"):
        WorkingMemory(force_fallback=True).put("key", "value", ttl=ttl)


def test_episodic_memory_store_and_search():
    em = EpisodicMemory(vector_dim=4, force_fallback=True)
    e1 = EpisodicEvent(
        embedding=[1.0, 0.0, 0.0, 0.0],
        context={"concept": "digit_zero", "category": "vision"},
        surprise=0.1,
    )
    e2 = EpisodicEvent(
        embedding=[0.0, 1.0, 0.0, 0.0],
        context={"concept": "digit_one", "category": "vision"},
        surprise=0.8,
    )
    em.store_event(e1)
    em.store_event(e2)
    assert em.num_events() == 2

    # Search for vector close to e1
    hits = em.search_similar([0.9, 0.1, 0.0, 0.0], limit=1)
    assert len(hits) == 1
    assert hits[0][0].context["concept"] == "digit_zero"
    assert hits[0][1] > 0.8


def test_episodic_memory_validates_vectors_and_limits():
    memory = EpisodicMemory(vector_dim=2, force_fallback=True)
    with pytest.raises(ValueError, match="exactly 2"):
        memory.store_event(EpisodicEvent(embedding=[1.0], context={}))
    with pytest.raises(ValueError, match="finite"):
        memory.search_similar([float("nan"), 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        memory.retrieve_recent(limit=-1)
    assert memory.search_similar([1.0, 0.0], limit=0) == []


def test_episodic_memory_retrieves_and_deletes_qdrant_events():
    class FakeQdrant:
        def __init__(self):
            self.points = [
                SimpleNamespace(id=f"event_{timestamp}", payload={"timestamp": timestamp}, vector=[1.0, 0.0])
                for timestamp in (1.0, 3.0, 2.0)
            ]

        def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
            start = offset or 0
            page = self.points[start : start + 2]
            next_offset = start + len(page) if start + len(page) < len(self.points) else None
            return page, next_offset

        def count(self, collection_name, exact):
            return SimpleNamespace(count=len(self.points))

        def delete(self, collection_name, points_selector, wait):
            ids = set(points_selector.points)
            self.points = [point for point in self.points if point.id not in ids]

    memory = EpisodicMemory(vector_dim=2, force_fallback=True)
    client = FakeQdrant()
    memory._qdrant_client = client
    memory.is_qdrant_connected = True

    recent = memory.retrieve_recent(limit=2)
    assert [event.timestamp for event in recent] == [3.0, 2.0]
    assert memory.num_events() == 3

    memory.delete_events(["event_2.0"])
    assert memory.num_events() == 2
    assert "event_2.0" not in memory._fallback_store.events


def test_semantic_memory_knowledge_graph():
    sm = SemanticMemory()
    sm.add_concept("digit_zero", {"type": "number", "parity": "even"})
    sm.add_concept("digit_one", {"type": "number", "parity": "odd"})
    sm.add_relation("digit_zero", "digit_one", "successor", weight=1.0)

    assert sm.num_concepts() == 2
    assert sm.num_relations() == 1

    sub = sm.query_subgraph("digit_zero", depth=1)
    assert len(sub["nodes"]) == 2
    assert len(sub["edges"]) == 1
    assert sub["edges"][0]["attrs"]["relation"] == "successor"

    attrs = {"source": "caller"}
    sm.add_relation("digit_one", "digit_zero", "predecessor", attributes=attrs)
    assert attrs == {"source": "caller"}
    with pytest.raises(ValueError, match="depth must be non-negative"):
        sm.query_subgraph("digit_zero", depth=-1)


def test_semantic_memory_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "semantic" / "memory.json"
    memory = SemanticMemory(storage_path=str(checkpoint))
    memory.add_concept("concept_a", {"category": "math", "occurrences": 2})
    memory.add_relation("concept_a", "concept_b", "associated_with", weight=0.75)
    memory.save()

    restored = SemanticMemory.load(str(checkpoint))

    assert restored.get_concept("concept_a") == {"category": "math", "occurrences": 2}
    assert restored.query_subgraph("concept_a")["edges"][0]["attrs"]["weight"] == 0.75
    restored.graph.graph["worker_state"] = {"done": True}
    restored.save()
    assert SemanticMemory.load(str(checkpoint)).graph.graph["worker_state"] == {"done": True}


def test_sleep_consolidator_rejects_mismatched_identity_anchor():
    em = EpisodicMemory(vector_dim=4, force_fallback=True)
    with pytest.raises(ValueError, match="must have 4 values"):
        SleepConsolidator(em, SemanticMemory(), identity_anchor_vector=[1.0, 0.0])


def test_sleep_consolidation_and_identity_drift_protection():
    dim = 4
    anchor = [1.0, 0.0, 0.0, 0.0]
    em = EpisodicMemory(vector_dim=dim, force_fallback=True)
    sm = SemanticMemory()

    consolidator = SleepConsolidator(
        episodic_memory=em,
        semantic_memory=sm,
        identity_anchor_vector=anchor,
        max_drift_threshold=0.3,
    )

    # Event 1: aligned with anchor (cos_sim ~ 1.0 -> drift ~ 0.0)
    aligned_event = EpisodicEvent(
        embedding=[0.99, 0.05, 0.0, 0.0],
        context={"concept": "core_digit", "category": "math", "related_to": "foundation"},
        surprise=0.1,
    )
    # Event 2: orthogonal/divergent vector (drift ~ 1.0 > 0.3)
    drifting_event = EpisodicEvent(
        embedding=[0.0, 1.0, 0.0, 0.0],
        context={"concept": "rogue_concept", "category": "unknown"},
        surprise=0.9,
    )

    em.store_event(aligned_event)
    em.store_event(drifting_event)

    stats = consolidator.consolidate()
    assert stats["processed_events"] == 2
    assert stats["consolidated_concepts"] >= 1
    assert stats["rejected_for_drift"] >= 1
    assert sm.get_concept("core_digit") is not None
    assert sm.get_concept("rogue_concept") is None


def test_sleep_consolidation_processes_events_beyond_one_batch():
    em = EpisodicMemory(vector_dim=2, force_fallback=True)
    sm = SemanticMemory()
    consolidator = SleepConsolidator(em, sm)

    for index in range(7):
        em.store_event(EpisodicEvent(
            embedding=[1.0, 0.0],
            context={"concept": f"concept_{index}"},
        ))

    first_pass = consolidator.consolidate(batch_size=3)
    second_pass = consolidator.consolidate(batch_size=3)
    final_pass = consolidator.consolidate(batch_size=3)

    assert first_pass["processed_events"] == 3
    assert second_pass["processed_events"] == 3
    assert final_pass["processed_events"] == 1
    assert first_pass["redundant_events_removed"] == 2
    assert sm.num_concepts() == 3
    assert em.num_events() == 3


def test_dynamic_identity_anchor_rejects_drifting_events():
    em = EpisodicMemory(vector_dim=2, force_fallback=True)
    sm = SemanticMemory()
    consolidator = SleepConsolidator(em, sm, max_drift_threshold=0.35)
    em.store_event(EpisodicEvent(
        embedding=[1.0, 0.0],
        timestamp=time.time(),
        context={"concept": "anchor"},
    ))
    em.store_event(EpisodicEvent(
        embedding=[0.0, 1.0],
        timestamp=time.time() - 1,
        context={"concept": "drift"},
    ))

    stats = consolidator.consolidate()

    assert stats["rejected_for_drift"] == 1
    assert sm.get_concept("anchor") is not None
    assert sm.get_concept("drift") is None


def test_sleep_consolidation_clusters_similar_events_and_filters_irrelevant_ones():
    em = EpisodicMemory(vector_dim=3, force_fallback=True)
    sm = SemanticMemory()
    consolidator = SleepConsolidator(
        em,
        sm,
        max_drift_threshold=1.0,
        similarity_threshold=0.85,
        min_surprise=0.2,
    )
    for event in (
        EpisodicEvent(
            embedding=[1.0, 0.0, 0.0],
            context={"concept": "similar", "category": "vision", "related_to": "shape"},
            surprise=0.3,
        ),
        EpisodicEvent(
            embedding=[0.99, 0.1, 0.0],
            context={"concept": "similar", "category": "vision", "related_to": "shape"},
            surprise=0.7,
        ),
        EpisodicEvent(
            embedding=[0.0, 1.0, 0.0],
            context={"concept": "distinct", "category": "vision"},
            surprise=0.5,
        ),
        EpisodicEvent(
            embedding=[0.0, 0.0, 1.0],
            context={"concept": "irrelevant"},
            surprise=0.1,
        ),
    ):
        em.store_event(event)

    stats = consolidator.consolidate()

    assert stats["consolidated_concepts"] == 2
    assert stats["redundant_events_removed"] == 1
    assert stats["filtered_events"] == 1
    assert sm.get_concept("similar")["occurrences"] == 2
    assert sm.get_concept("similar")["source_concepts"] == ["similar"]
    assert em.num_events() == 2


def test_sleep_consolidation_persists_processed_events_and_dynamic_anchor(tmp_path):
    checkpoint = tmp_path / "semantic.json"
    memory = EpisodicMemory(vector_dim=2, force_fallback=True)
    semantic = SemanticMemory(storage_path=str(checkpoint))
    memory.store_event(EpisodicEvent(
        embedding=[1.0, 0.0],
        context={"concept": "anchor"},
    ))

    SleepConsolidator(memory, semantic).consolidate()
    restored_semantic = SemanticMemory.load(str(checkpoint))
    restored = SleepConsolidator(memory, restored_semantic)

    stats = restored.consolidate()
    assert stats["processed_events"] == 0
    assert restored.identity_anchor.tolist() == pytest.approx([1.0, 0.0])
    assert restored_semantic.get_concept("anchor")["occurrences"] == 1


def test_sleep_consolidator_background_worker_runs_and_stops():
    completed = threading.Event()
    memory = EpisodicMemory(vector_dim=2, force_fallback=True)
    semantic = SemanticMemory()
    memory.store_event(EpisodicEvent(
        embedding=[1.0, 0.0],
        context={"concept": "background"},
    ))

    class ObservedConsolidator(SleepConsolidator):
        def consolidate(self, batch_size=50):
            result = super().consolidate(batch_size)
            completed.set()
            return result

    consolidator = ObservedConsolidator(memory, semantic)
    consolidator.start_background(interval_seconds=0.01)
    assert completed.wait(timeout=2.0)
    consolidator.stop_background(timeout=2.0)

    assert semantic.get_concept("background") is not None
