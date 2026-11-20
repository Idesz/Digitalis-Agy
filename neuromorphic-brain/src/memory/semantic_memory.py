"""
Semantic Memory Module.
Structured knowledge graph using NetworkX for persistent concept representation,
relational inference, and topological queries.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx


class SemanticMemory:
    """
    Graph-based semantic memory representing enduring factual relations.
    """

    def __init__(self, storage_path: Optional[str] = None):
        configured_path = storage_path or os.environ.get("SEMANTIC_MEMORY_PATH")
        self.storage_path = Path(configured_path) if configured_path else None
        self.graph = nx.DiGraph()
        if self.storage_path is not None and self.storage_path.exists():
            self._load(self.storage_path)

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path is not None else self.storage_path
        if target is None:
            raise ValueError("A semantic memory storage path is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = target.with_name(target.name + ".tmp")
        serialized = {
            "version": 1,
            "graph_attributes": dict(self.graph.graph),
            "nodes": [
                {"id": node_id, "attributes": attributes}
                for node_id, attributes in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target_id, "attributes": attributes}
                for source, target_id, attributes in self.graph.edges(data=True)
            ],
        }
        temporary_path.write_text(
            json.dumps(serialized, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(target)

    @classmethod
    def load(cls, path: str) -> "SemanticMemory":
        return cls(storage_path=path)

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported semantic memory checkpoint version")
        nodes = data.get("nodes")
        edges = data.get("edges")
        graph_attributes = data.get("graph_attributes", {})
        if (
            not isinstance(nodes, list)
            or not isinstance(edges, list)
            or not isinstance(graph_attributes, dict)
        ):
            raise ValueError("Semantic memory checkpoint nodes and edges must be lists")

        graph = nx.DiGraph()
        graph.graph.update(graph_attributes)
        for node in nodes:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("id"), str)
                or not isinstance(node.get("attributes"), dict)
            ):
                raise ValueError("Invalid semantic memory node entry")
            graph.add_node(node["id"], **node["attributes"])

        for edge in edges:
            if (
                not isinstance(edge, dict)
                or not isinstance(edge.get("source"), str)
                or not isinstance(edge.get("target"), str)
                or not isinstance(edge.get("attributes"), dict)
                or edge["source"] not in graph
                or edge["target"] not in graph
            ):
                raise ValueError("Invalid semantic memory edge entry")
            graph.add_edge(edge["source"], edge["target"], **edge["attributes"])
        self.graph = graph

    def add_concept(self, concept_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Adds or updates a semantic entity/concept node."""
        attrs = attributes or {}
        if self.graph.has_node(concept_id):
            self.graph.nodes[concept_id].update(attrs)
        else:
            self.graph.add_node(concept_id, **attrs)

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a directed semantic relationship edge."""
        if not relation_type:
            raise ValueError("relation_type must not be empty")
        if not self.graph.has_node(source):
            self.add_concept(source)
        if not self.graph.has_node(target):
            self.add_concept(target)

        edge_attrs = dict(attributes or {})
        edge_attrs.update({"relation": relation_type, "weight": weight})
        self.graph.add_edge(source, target, **edge_attrs)

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(concept_id):
            return dict(self.graph.nodes[concept_id])
        return None

    def query_subgraph(self, concept_id: str, depth: int = 1) -> Dict[str, Any]:
        """Extracts ego subgraph around a concept for associative reasoning."""
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if not self.graph.has_node(concept_id):
            return {"nodes": [], "edges": []}

        subgraph_nodes = nx.single_source_shortest_path_length(self.graph, concept_id, cutoff=depth).keys()
        sub = self.graph.subgraph(subgraph_nodes)

        nodes = [{"id": n, "attrs": sub.nodes[n]} for n in sub.nodes()]
        edges = [{"source": u, "target": v, "attrs": sub.edges[u, v]} for u, v in sub.edges()]
        return {"nodes": nodes, "edges": edges}

    def num_concepts(self) -> int:
        return self.graph.number_of_nodes()

    def num_relations(self) -> int:
        return self.graph.number_of_edges()
