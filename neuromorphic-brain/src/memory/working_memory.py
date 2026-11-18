"""
Working Memory Module.
Provides high-speed contextual storage with TTL support.
Connects to Redis when available, or seamlessly falls back to a thread-safe in-memory cache.
"""

import os
import time
import json
import threading
from typing import Any, Optional


class InMemoryCache:
    """Thread-safe in-memory key-value cache with TTL expiration."""

    def __init__(self):
        self._store: dict[str, tuple[str, Optional[float]]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        expire_at = time.time() + ex if ex else None
        with self._lock:
            self._store[key] = (value, expire_at)
        return True

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._store:
                return None
            val, expire_at = self._store[key]
            if expire_at is not None and time.time() >= expire_at:
                del self._store[key]
                return None
            return val

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def flushall(self) -> None:
        with self._lock:
            self._store.clear()

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [key for key in self._store if key.startswith(prefix)]:
                del self._store[key]

    def ping(self) -> bool:
        return True


class WorkingMemory:
    """
    Working Memory manager for rapid state retrieval and contextual caching.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        force_fallback: bool = False,
        namespace: str = "digitalis_brain:",
    ):
        if not namespace:
            raise ValueError("namespace must not be empty")
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port if port is not None else int(os.environ.get("REDIS_PORT", 6379))
        self.namespace = namespace
        self.is_redis_connected = False
        self._client = None

        if not force_fallback:
            try:
                import redis
                client = redis.Redis(host=self.host, port=self.port, db=db, socket_connect_timeout=0.5)
                client.ping()
                self._client = client
                self.is_redis_connected = True
            except Exception:
                self._client = InMemoryCache()
                self.is_redis_connected = False
        else:
            self._client = InMemoryCache()
            self.is_redis_connected = False

    def put(self, key: str, data: Any, ttl: Optional[int] = 3600) -> bool:
        """Serializes and stores item in working memory."""
        self._validate_key(key)
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 1):
            raise ValueError("ttl must be a positive integer or None")
        serialized = json.dumps(data)
        namespaced_key = self._key(key)
        if self.is_redis_connected:
            return bool(self._client.set(name=namespaced_key, value=serialized, ex=ttl))
        return self._client.set(key=namespaced_key, value=serialized, ex=ttl)

    def get(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes item from working memory."""
        self._validate_key(key)
        raw = self._client.get(self._key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def delete(self, key: str) -> bool:
        self._validate_key(key)
        if self.is_redis_connected:
            return bool(self._client.delete(self._key(key)))
        return self._client.delete(self._key(key))

    def clear(self) -> None:
        if self.is_redis_connected:
            keys = list(self._client.scan_iter(match=f"{self.namespace}*"))
            if keys:
                self._client.delete(*keys)
            return
        self._client.delete_prefix(self.namespace)

    def _key(self, key: str) -> str:
        return f"{self.namespace}{key}"

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
