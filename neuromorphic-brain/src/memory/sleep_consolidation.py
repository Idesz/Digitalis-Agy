"""
Sleep Consolidation Worker.
Abstracts episodic memory into the semantic knowledge graph during simulated sleep cycles.
Guards against identity drift using cosine similarity verification against core identity anchors.
"""

from collections import Counter
import logging
import threading
from typing import Dict, List, Optional
import math
import time
import numpy as np
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory


logger = logging.getLogger(__name__)


class SleepConsolidator:
    """
    Background consolidation process simulating NREM/REM sleep cycles.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        identity_anchor_vector: Optional[List[float]] = None,
        max_drift_threshold: float = 0.35,
        similarity_threshold: float = 0.85,
        min_surprise: float = 0.0,
        max_event_age_seconds: Optional[float] = None,
    ):
        if not math.isfinite(max_drift_threshold) or not 0.0 <= max_drift_threshold <= 2.0:
            raise ValueError("max_drift_threshold must be finite and in [0, 2]")
        if not math.isfinite(similarity_threshold) or not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be finite and in [0, 1]")
        if not math.isfinite(min_surprise) or not 0.0 <= min_surprise <= 1.0:
            raise ValueError("min_surprise must be finite and in [0, 1]")
        if max_event_age_seconds is not None and (
            not math.isfinite(max_event_age_seconds) or max_event_age_seconds <= 0
        ):
            raise ValueError("max_event_age_seconds must be finite and positive")
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.max_drift_threshold = max_drift_threshold
        self.similarity_threshold = similarity_threshold
        self.min_surprise = min_surprise
        self.max_event_age_seconds = max_event_age_seconds
        self.consolidated_event_ids: set[str] = set()
        self._consolidation_lock = threading.Lock()
        self._background_stop = threading.Event()
        self._background_thread: Optional[threading.Thread] = None
        self._background_error: Optional[Exception] = None

        # Core anchor vector defining agent baseline identity / knowledge centroid
        # If None, dynamic anchoring: initial anchor learned or zeroed until established
        self._dynamic_anchor = identity_anchor_vector is None
        if identity_anchor_vector is not None:
            vec = np.array(identity_anchor_vector, dtype=np.float32)
            if vec.ndim != 1 or vec.size != episodic_memory.vector_dim:
                raise ValueError(f"identity_anchor_vector must have {episodic_memory.vector_dim} values")
            if not np.all(np.isfinite(vec)):
                raise ValueError("identity_anchor_vector values must be finite")
            norm = np.linalg.norm(vec)
            self.identity_anchor = (vec / norm) if norm > 0 else None
        else:
            self.identity_anchor = None
        self._restore_state()

    def calculate_drift(self, candidate_vector: np.ndarray) -> float:
        """Computes cosine distance (1 - cosine_similarity) from identity anchor."""
        norm = np.linalg.norm(candidate_vector)
        if norm == 0:
            return 0.0
        unit_cand = candidate_vector / norm

        if self.identity_anchor is None:
            # Anchor initializes with the first stable observation
            self.identity_anchor = unit_cand.copy()
            return 0.0

        cos_sim = float(np.dot(self.identity_anchor, unit_cand))
        return max(0.0, 1.0 - cos_sim)

    def consolidate(self, batch_size: int = 50) -> Dict[str, int]:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        with self._consolidation_lock:
            if self._background_error is not None:
                raise RuntimeError("Background sleep consolidation failed") from self._background_error
            return self._consolidate(batch_size)

    def _consolidate(self, batch_size: int) -> Dict[str, int]:
        """
        Executes a consolidation pass:
        1. Retrieves un-consolidated recent episodic events.
        2. Identifies high-surprise or recurring concepts.
        3. Integrates them into the semantic graph while verifying no identity drift occurs.
        """
        recent_events = self.episodic.retrieve_recent(limit=self.episodic.num_events())
        unprocessed_events = [e for e in recent_events if e.event_id not in self.consolidated_event_ids]
        unprocessed_events = unprocessed_events[:batch_size]
        consolidated_concepts = 0
        consolidated_relations = 0
        rejected_for_drift = 0
        filtered_events = 0
        now = time.time()
        eligible_events = []
        filtered_event_ids = []

        for event in unprocessed_events:
            if (
                event.surprise < self.min_surprise
                or (
                    self.max_event_age_seconds is not None
                    and now - event.timestamp > self.max_event_age_seconds
                )
            ):
                filtered_events += 1
                filtered_event_ids.append(event.event_id)
                continue

            vec = np.array(event.embedding, dtype=np.float32)
            drift = self.calculate_drift(vec)

            # Reject memories that diverge too far from the fixed or learned anchor.
            if self.identity_anchor is not None and drift > self.max_drift_threshold:
                rejected_for_drift += 1
                self.consolidated_event_ids.add(event.event_id)
                continue

            eligible_events.append(event)

        redundant_event_ids = []
        processed_event_ids = []
        clusters = self._cluster_events(eligible_events)
        for cluster in clusters:
            concepts = [
                event.context.get("concept")
                for event in cluster
                if event.context.get("concept")
            ]
            if concepts:
                concept_name = Counter(concepts).most_common(1)[0][0]
                categories = [
                    event.context.get("category", "general")
                    for event in cluster
                    if event.context.get("concept")
                ]
                vectors = np.array([event.embedding for event in cluster], dtype=np.float32)
                centroid = vectors.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                last_consolidated = max(event.timestamp for event in cluster)
                max_surprise = max(event.surprise for event in cluster)
                self.semantic.add_concept(
                    concept_id=concept_name,
                    attributes={
                        "category": Counter(categories).most_common(1)[0][0],
                        "surprise": max_surprise,
                        "occurrences": len(cluster),
                        "source_concepts": sorted(set(concepts)),
                        "embedding_centroid": centroid.tolist(),
                        "last_consolidated": last_consolidated,
                    },
                )
                consolidated_concepts += 1

                related_counts = Counter(
                    event.context["related_to"]
                    for event in cluster
                    if event.context.get("related_to")
                )
                for related_to, count in related_counts.items():
                    self.semantic.add_relation(
                        source=concept_name,
                        target=related_to,
                        relation_type="associated_with",
                        weight=float(count / len(cluster)),
                        attributes={"support": count, "cluster_size": len(cluster)},
                    )
                    consolidated_relations += 1

                representative = max(cluster, key=lambda event: (event.surprise, event.timestamp))
                redundant_event_ids.extend(
                    event.event_id for event in cluster if event.event_id != representative.event_id
                )

                if self._dynamic_anchor and self.identity_anchor is not None:
                    for event in cluster:
                        vec = np.array(event.embedding, dtype=np.float32)
                        norm_v = np.linalg.norm(vec)
                        if norm_v > 0:
                            self.identity_anchor = 0.95 * self.identity_anchor + 0.05 * (vec / norm_v)
                            norm_a = np.linalg.norm(self.identity_anchor)
                            if norm_a > 0:
                                self.identity_anchor /= norm_a

            processed_event_ids.extend(event.event_id for event in cluster)

        self.episodic.delete_events([*redundant_event_ids, *filtered_event_ids])
        self.consolidated_event_ids.update(processed_event_ids)
        self.consolidated_event_ids.update(filtered_event_ids)
        self.semantic.graph.graph["sleep_consolidator_state"] = {
            "consolidated_event_ids": sorted(self.consolidated_event_ids),
            "identity_anchor": (
                self.identity_anchor.tolist() if self.identity_anchor is not None else None
            ),
        }
        if self.semantic.storage_path is not None:
            self.semantic.save()

        return {
            "processed_events": len(unprocessed_events),
            "consolidated_concepts": consolidated_concepts,
            "consolidated_relations": consolidated_relations,
            "rejected_for_drift": rejected_for_drift,
            "filtered_events": filtered_events,
            "redundant_events_removed": len(redundant_event_ids),
        }

    def start_background(self, interval_seconds: float = 3600.0) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        if self._background_thread is not None and self._background_thread.is_alive():
            raise RuntimeError("Background sleep consolidation is already running")
        if self._background_error is not None:
            raise RuntimeError("Background sleep consolidation previously failed") from self._background_error
        self._background_stop.clear()
        self._background_thread = threading.Thread(
            target=self._run_background,
            args=(interval_seconds,),
            name="sleep-consolidation",
            daemon=True,
        )
        self._background_thread.start()

    def stop_background(self, timeout: Optional[float] = None) -> None:
        thread = self._background_thread
        if thread is None:
            if self._background_error is not None:
                raise RuntimeError("Background sleep consolidation failed") from self._background_error
            return
        self._background_stop.set()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("Background sleep consolidation did not stop in time")
        self._background_thread = None
        if self._background_error is not None:
            raise RuntimeError("Background sleep consolidation failed") from self._background_error

    def _run_background(self, interval_seconds: float) -> None:
        while not self._background_stop.wait(interval_seconds):
            try:
                self.consolidate()
            except Exception as exc:
                logger.exception("Background sleep consolidation stopped after an error")
                self._background_error = exc
                self._background_stop.set()
                return

    def _restore_state(self) -> None:
        state = self.semantic.graph.graph.get("sleep_consolidator_state", {})
        if not isinstance(state, dict):
            raise ValueError("Invalid persisted sleep consolidation state")
        event_ids = state.get("consolidated_event_ids", [])
        if not isinstance(event_ids, list) or any(not isinstance(item, str) for item in event_ids):
            raise ValueError("Invalid persisted consolidated event IDs")
        self.consolidated_event_ids.update(event_ids)

        anchor = state.get("identity_anchor")
        if self._dynamic_anchor and self.identity_anchor is None and anchor is not None:
            vector = np.asarray(anchor, dtype=np.float32)
            if (
                vector.ndim != 1
                or vector.size != self.episodic.vector_dim
                or not np.all(np.isfinite(vector))
            ):
                raise ValueError("Invalid persisted identity anchor")
            norm = np.linalg.norm(vector)
            self.identity_anchor = vector / norm if norm > 0 else None

    def _cluster_events(self, events: List[EpisodicEvent]) -> List[List[EpisodicEvent]]:
        clusters: List[List[EpisodicEvent]] = []
        centroids: List[np.ndarray] = []
        for event in events:
            vector = np.array(event.embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            normalized = vector / norm if norm > 0 else vector
            best_index = None
            best_similarity = -1.0
            for index, centroid in enumerate(centroids):
                similarity = float(np.dot(normalized, centroid))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = index
            if best_index is None or best_similarity < self.similarity_threshold:
                clusters.append([event])
                centroids.append(normalized.copy())
            else:
                clusters[best_index].append(event)
                centroid = np.mean(
                    [np.array(item.embedding, dtype=np.float32) for item in clusters[best_index]],
                    axis=0,
                )
                centroid_norm = np.linalg.norm(centroid)
                centroids[best_index] = centroid / centroid_norm if centroid_norm > 0 else centroid
        return clusters
