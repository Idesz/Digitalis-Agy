"""
Episodic Memory Module.
Stores time-stamped experience embeddings and associative context using Qdrant
with seamless fallback to an in-memory vector store.
"""

from typing import Any, List, Optional
import os
import time
import uuid
import logging
import math
import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EpisodicEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    embedding: List[float]
    context: dict[str, Any] = Field(default_factory=dict)
    surprise: float = 0.0


class InMemoryVectorStore:
    """In-memory cosine vector store fallback when Qdrant server is unreachable."""

    def __init__(self):
        self.events: dict[str, EpisodicEvent] = {}

    def insert(self, event: EpisodicEvent) -> str:
        self.events[event.event_id] = event
        return event.event_id

    def search(self, query_vec: List[float], limit: int = 5) -> List[tuple[EpisodicEvent, float]]:
        if not self.events:
            return []
        q = np.array(query_vec, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        if norm_q > 0:
            q = q / norm_q

        scored: list[tuple[EpisodicEvent, float]] = []
        for event in self.events.values():
            e = np.array(event.embedding, dtype=np.float32)
            norm_e = np.linalg.norm(e)
            if norm_e > 0:
                e = e / norm_e
            sim = float(np.dot(q, e))
            scored.append((event, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def all_events(self) -> List[EpisodicEvent]:
        return list(self.events.values())


class EpisodicMemory:
    """
    Episodic Memory interface for temporal autobiographical experience logging.
    """

    def __init__(
        self,
        collection_name: str = "brain_episodes",
        host: Optional[str] = None,
        port: Optional[int] = None,
        vector_dim: int = 64,
        force_fallback: bool = False,
    ):
        if vector_dim < 1:
            raise ValueError("vector_dim must be positive")
        self.collection_name = collection_name
        self.host = host or os.environ.get("QDRANT_HOST", "localhost")
        self.port = port if port is not None else int(os.environ.get("QDRANT_PORT", 6333))
        self.vector_dim = vector_dim
        self.is_qdrant_connected = False
        self._qdrant_client = None
        self._fallback_store = InMemoryVectorStore()

        if not force_fallback:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                client = QdrantClient(host=self.host, port=self.port, timeout=0.5, check_compatibility=False)
                # Test connection
                client.get_collections()
                # Ensure collection exists
                collections = [c.name for c in client.get_collections().collections]
                if collection_name not in collections:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
                    )
                self._qdrant_client = client
                self.is_qdrant_connected = True
            except Exception:
                self.is_qdrant_connected = False
        else:
            self.is_qdrant_connected = False

    def store_event(self, event: EpisodicEvent) -> str:
        """Stores a new episodic event record."""
        self._validate_vector(event.embedding)
        # Always update local store
        self._fallback_store.insert(event)

        if self.is_qdrant_connected and self._qdrant_client is not None:
            try:
                from qdrant_client.models import PointStruct

                self._qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=event.event_id,
                            vector=event.embedding,
                            payload={
                                "timestamp": event.timestamp,
                                "context": event.context,
                                "surprise": event.surprise,
                            },
                        )
                    ],
                )
            except Exception:
                logger.exception("Failed to store episodic event in Qdrant; switching to in-memory storage")
                self.is_qdrant_connected = False
        return event.event_id

    def search_similar(self, query_vec: List[float], limit: int = 5) -> List[tuple[EpisodicEvent, float]]:
        """Returns most similar events and their cosine similarity score."""
        self._validate_vector(query_vec)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []

        if self.is_qdrant_connected and self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=limit,
                    with_vectors=True,
                )
                results = []
                for hit in hits:
                    retrieved_embedding = hit.vector if hit.vector is not None else query_vec
                    event = EpisodicEvent(
                        event_id=str(hit.id),
                        timestamp=hit.payload.get("timestamp", 0.0),
                        embedding=retrieved_embedding,
                        context=hit.payload.get("context", {}),
                        surprise=hit.payload.get("surprise", 0.0),
                    )
                    results.append((event, float(hit.score)))
                return results
            except Exception:
                logger.exception("Qdrant episodic search failed; switching to in-memory storage")
                self.is_qdrant_connected = False
        return self._fallback_store.search(query_vec, limit)

    def retrieve_recent(self, limit: int = 100) -> List[EpisodicEvent]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        events = self._fallback_store.all_events()
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]

    def num_events(self) -> int:
        """Returns the total number of events mirrored in the local store."""
        return len(self._fallback_store.events)

    def _validate_vector(self, vector: List[float]) -> None:
        if len(vector) != self.vector_dim:
            raise ValueError(f"Embedding must contain exactly {self.vector_dim} values")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding values must be finite")
