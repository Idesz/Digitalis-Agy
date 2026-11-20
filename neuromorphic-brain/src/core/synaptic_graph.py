"""
Synaptic Graph Manager (Hebbian Local Learning & Continuous Pruning).
Corresponds directly to Pillar 1 from software_and_code_specification.md.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


@dataclass
class Node:
    """Represents a node (neuron or abstract concept) in the dynamic graph."""
    node_id: str
    activation_history: List[float] = field(default_factory=list)


@dataclass
class Edge:
    """Represents a dynamic synaptic connection between two nodes."""
    source_id: str
    target_id: str
    weight: float
    last_activated: float
    last_decayed: Optional[float] = None


class SynapticGraphManager:
    """
    Distributed directed synaptic graph manager executing Hebbian local learning and continuous pruning.
    """

    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.01, initial_weight: float = 0.2) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}  # key format: "source_id->target_id"
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.initial_weight = initial_weight

    def save_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        checkpoint = {
            "version": 1,
            "config": {
                "learning_rate": self.learning_rate,
                "decay_factor": self.decay_factor,
                "initial_weight": self.initial_weight,
            },
            "nodes": {
                node_id: {"activation_history": node.activation_history}
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_key: {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                    "last_activated": edge.last_activated,
                    "last_decayed": edge.last_decayed,
                }
                for edge_key, edge in self.edges.items()
            },
        }
        temporary_path.write_text(
            json.dumps(checkpoint, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "SynapticGraphManager":
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
            raise ValueError("Unsupported synaptic graph checkpoint version")

        try:
            manager = cls(**checkpoint["config"])
            nodes = checkpoint["nodes"]
            edges = checkpoint["edges"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid synaptic graph checkpoint structure") from exc

        if not isinstance(nodes, dict) or not isinstance(edges, dict):
            raise ValueError("Checkpoint nodes and edges must be objects")

        for node_id, node_data in nodes.items():
            if not isinstance(node_id, str) or not isinstance(node_data, dict):
                raise ValueError("Invalid synaptic graph node entry")
            history = node_data.get("activation_history")
            if not isinstance(history, list) or any(
                not isinstance(timestamp, (int, float)) for timestamp in history
            ):
                raise ValueError(f"Invalid activation history for node {node_id!r}")
            manager.nodes[node_id] = Node(node_id=node_id, activation_history=history)

        for edge_key, edge_data in edges.items():
            if not isinstance(edge_key, str) or not isinstance(edge_data, dict):
                raise ValueError("Invalid synaptic graph edge entry")
            try:
                edge = Edge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    weight=edge_data["weight"],
                    last_activated=edge_data["last_activated"],
                    last_decayed=edge_data.get("last_decayed"),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid edge data for {edge_key!r}") from exc
            if (
                edge_key != f"{edge.source_id}->{edge.target_id}"
                or edge.source_id not in manager.nodes
                or edge.target_id not in manager.nodes
            ):
                raise ValueError(f"Inconsistent synaptic graph edge {edge_key!r}")
            manager.edges[edge_key] = edge

        return manager

    def add_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id)

    def trigger_activation(self, node_id: str, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        if node_id in self.nodes:
            self.nodes[node_id].activation_history.append(timestamp)
            if len(self.nodes[node_id].activation_history) > 100:
                self.nodes[node_id].activation_history = self.nodes[node_id].activation_history[-100:]

    def record_coactivation(
        self,
        source_id: str,
        target_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time()

        self.add_node(source_id)
        self.add_node(target_id)
        self.trigger_activation(source_id, timestamp)
        self.trigger_activation(target_id, timestamp)

        edge_key = f"{source_id}->{target_id}"
        if edge_key not in self.edges:
            self.edges[edge_key] = Edge(
                source_id=source_id,
                target_id=target_id,
                weight=self.initial_weight,
                last_activated=timestamp,
            )
        else:
            edge = self.edges[edge_key]
            edge.weight += self.learning_rate * (1.0 - edge.weight)
            edge.last_activated = timestamp
            edge.last_decayed = None

    def update_local_learning(self, node_a: str, node_b: str, time_window: float = 0.5) -> None:
        if node_a not in self.nodes or node_b not in self.nodes:
            return

        now = time.time()
        history_a = [t for t in self.nodes[node_a].activation_history if now - t <= time_window]
        history_b = [t for t in self.nodes[node_b].activation_history if now - t <= time_window]

        if history_a and history_b:
            edge_key = f"{node_a}->{node_b}"
            if edge_key not in self.edges:
                self.edges[edge_key] = Edge(
                    source_id=node_a,
                    target_id=node_b,
                    weight=self.initial_weight,
                    last_activated=now,
                )
            else:
                current_weight = self.edges[edge_key].weight
                self.edges[edge_key].weight = current_weight + self.learning_rate * (1.0 - current_weight)
                self.edges[edge_key].last_activated = now
                self.edges[edge_key].last_decayed = None

    def prune_inactive_edges(self, threshold: float = 0.1, max_idle_time: float = 10.0) -> int:
        if max_idle_time <= 0:
            return 0
        now = time.time()
        keys_to_delete: List[str] = []

        for edge_key, edge in self.edges.items():
            idle_duration = now - edge.last_activated
            if idle_duration > max_idle_time:
                decay_start = edge.last_decayed or edge.last_activated + max_idle_time
                decay_duration = max(0.0, now - decay_start)
                edge.weight -= self.decay_factor * (decay_duration / max_idle_time)
                edge.last_decayed = now

            if edge.weight < threshold:
                keys_to_delete.append(edge_key)

        for key in keys_to_delete:
            del self.edges[key]
        return len(keys_to_delete)
