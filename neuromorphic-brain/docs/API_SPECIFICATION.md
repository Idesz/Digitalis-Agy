# API Specification: Event Bus, Modules & Interfaces

This document describes the data contracts, Python class interfaces, and event bus schema for the Neuromorphic Brain architecture.

---

## 1. Core Data Models (`src/core/` & `src/homeostasis/`)

### HomeostaticState & StateVector
```python
class HomeostaticState:
    curiosity: float         # [0.0, 1.0]
    error_rate: float        # [0.0, 1.0]
    resource_stress: float   # [0.0, 1.0], smoothed process CPU load in the dashboard
    uncertainty: float       # [0.0, 1.0]

    def update_observation(self, surprise: float, task_error: float, system_load: float, entropy: float, smoothing: float = 0.2) -> "HomeostaticState": ...
    def get_behavioral_mode(self) -> BehavioralMode: ...  # "EXPLORATION", "CAUTIOUS", "CONSERVATION", "BALANCED"
    @property
    def vector(self) -> list[float]: ...
```

### EpisodicEvent
```python
class EpisodicEvent(BaseModel):
    event_id: str
    timestamp: float
    embedding: list[float]
    context: dict[str, Any]
    surprise: float
```

---

## 2. Component Interfaces

### DECOLLE Layer (`src/core/decolle_layer.py`)
```python
class DECOLLELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_classes: int = 10,
        tau_m: float = 20.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        gamma: float = 10.0,
        trace_decay: float = 0.95,
        target_rate: float = 0.05,
        reg_rate_coef: float = 0.1,
    ): ...

    def forward_step(
        self,
        input_spikes: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Processes one time-step.
        Returns: (output_spikes, (next_potential, next_trace), local_readout_logits)
        """

    def compute_local_loss(
        self,
        readout_logits: torch.Tensor,
        targets: torch.Tensor,
        accumulated_spikes: torch.Tensor,
        time_steps: int,
    ) -> torch.Tensor:
        """
        Computes local loss including task cross-entropy and firing rate regularization.
        """
```

### Synthetic Dataset (`src/core/dataset.py`)
```python
def get_neuromorphic_dataloader(
    batch_size: int = 16,
    num_samples: int = 100,
    num_classes: int = 10,
    data_dir: str | None = None,
    task: str = "classification",
) -> DataLoader: ...
```

`task="classification"` loads `processed_nmnist.pt` from `data_dir` when present, otherwise it uses
the class-conditioned synthetic event stream. `task="xor"` and `task="parity"` always generate
binary spike-pattern tasks with matching two-class labels. The dataset reports its active source
(`N-MNIST`, `Synthetic DVS`, `Synthetic XOR`, or `Synthetic PARITY`).

### Memory Hierarchy (`src/memory/`)

```python
class WorkingMemory:
    def put(self, key: str, data: Any, ttl: int | None = 3600) -> bool: ...
    def get(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...

class EpisodicMemory:
    def store_event(self, event: EpisodicEvent) -> str: ...
    def search_similar(self, query_vec: list[float], limit: int = 5) -> list[tuple[EpisodicEvent, float]]: ...
    def retrieve_recent(self, limit: int = 100) -> list[EpisodicEvent]: ...
    def delete_events(self, event_ids: list[str]) -> None: ...

class SemanticMemory:
    def __init__(self, storage_path: str | None = None): ...
    def add_concept(self, concept_id: str, attributes: dict[str, Any] | None = None) -> None: ...
    def add_relation(self, source: str, target: str, relation_type: str, weight: float = 1.0, attributes: dict[str, Any] | None = None) -> None: ...
    def query_subgraph(self, concept_id: str, depth: int = 1) -> dict[str, Any]: ...
    def save(self, path: str | None = None) -> None: ...
    @classmethod
    def load(cls, path: str) -> "SemanticMemory": ...

class SleepConsolidator:
    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        similarity_threshold: float = 0.85,
        min_surprise: float = 0.0,
        max_event_age_seconds: float | None = None,
    ): ...
    def consolidate(self, batch_size: int = 50) -> dict[str, int]:
        """
        Clusters similar recent events, filters configured age/relevance limits,
        abstracts them into persistent semantic concepts, removes redundant events,
        and rejects events beyond the identity-drift threshold.
        """
```

### Predictive Coding (`src/predictive/residual_error.py`)
```python
class ResidualPredictiveModule(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (prediction, residual_error, latent)
        """

    def gated_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Calculates loss gated by residual error magnitude above surprise threshold.
        """
```

### Cognitive Decision Routing (`src/routing/cdr_router.py`)
```python
class SystemRoute(str, Enum):
    FAST = "FAST_SYSTEM_1"
    SLOW = "SLOW_SYSTEM_2"

class CDRRouter:
    def route(
        self,
        correlation_strength: float,
        domain_crossing: float,
        stakeholder_count: int,
        uncertainty: float
    ) -> tuple[SystemRoute, float, dict[str, float]]:
        """
        Calculates gating metric Phi(q) and routes to FAST or SLOW system.
        """

@dataclass(frozen=True)
class TaskRoutingContext:
    domains: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
```

Routing dimensions and the threshold are finite values in `[0, 1]`; stakeholder counts must be non-negative integers. Invalid values or all-zero/negative routing weights are rejected instead of silently clipped.
The pipeline accepts `TaskRoutingContext` per batch. It combines explicit task domains and entities with similar episodic event metadata and connected semantic-graph nodes/relations to estimate domain crossing and entity count; it no longer infers domain complexity from batch accuracy.

### World Model (`src/world_model/jepa_simulator.py`)
```python
class JEPASimulator(nn.Module):
    def predict_next_latent(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
    def simulate_rollout(self, initial_latent: torch.Tensor, action_sequence: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]: ...
    def simulate_candidates(self, initial_latent: torch.Tensor, candidate_action_sequences: torch.Tensor) -> tuple[list[list[torch.Tensor]], torch.Tensor]: ...
    def estimate_trajectory_risk(self, trajectory: list[torch.Tensor]) -> torch.Tensor: ...
```

`candidate_action_sequences` has shape `[batch, candidates, horizon, action_dim]`; the returned scores have shape `[batch, candidates]`.
`NeuromorphicCognitivePipeline.process_batch` accepts optional `transition_actions` of shape `[batch, time_steps - 1, action_dim]`. These must be observed actions corresponding to the transitions in the input sequence; class labels and neutral placeholders are not converted into actions. The world model learns from all observed batch transitions. Planning uses a risk-aware latent beam search over at least five distinct first actions and is skipped until ten transitions have been observed or when homeostasis selects conservation mode.

---

## 3. Communication Protocols
- Internal Module Communication: Python In-Process Direct Calls / Async Queues.
- Memory Layer: Redis (TCP port 6379) and Qdrant (gRPC/HTTP port 6333) with seamless fallback to In-Memory stores when external daemons are absent.

## 4. Dashboard Training API
- `GET /api/catalog` lists locally runnable workflows, external benchmarks, training paradigms, and ranking types.
- `POST /api/plan` accepts `prompt`, `epochs`, `lr`, and `batch_size`, then describes the task selected from the prompt.
- `POST /api/train/start` accepts the same fields, validates positive training parameters, and returns `{ "ok": bool, "message": str }`.
- Prompts containing `parity`/`paritás` or `xor` train the corresponding binary synthetic dataset; other supported prompts use general classification.
- MMLU, HumanEval, GSM8K, MATH, and LMSYS Arena are catalogued but are not executed by the current spiking classifier; unsupported benchmark prompts return HTTP 400 rather than silently running a different task.
- `lr` is passed to the DECOLLE optimizer. Training state is streamed through `/api/stream` with `RUNNING`, `COMPLETED`, `STOPPED`, or `ERROR` status.
- `POST /api/train/stop` requests cooperative cancellation.
- `GET /api/state` returns the latest snapshot. Its `has_metrics` flag is false until a complete measured snapshot is published. External `/api/update` clients must explicitly send `has_metrics: true` only with a complete sample; partial updates do not make defaults look measured.
- A complete external telemetry sample includes every measured state, training progress, activity, and memory-count field emitted by the local trainer. An incomplete update with `has_metrics: true` returns HTTP 400.
- `step` counts successfully processed batches and `total_steps` is `epochs * len(dataloader)`. The progress bar is based on those values, while `epoch` identifies the current epoch.
- `last_accuracy` is the accuracy on the current training batch, not a held-out test-set score. `last_loss` is the mean of the DECOLLE layer-local losses for that training batch.
- `process_cpu_load` is the process CPU-time delta divided by elapsed wall time between consecutive batch samples, bounded to `[0, 1]` (one logical core, capped at 100%). The interval includes time between batches. `resource_stress` is the homeostatic smoothed value derived from that measurement; it is not system-wide CPU or memory utilization.
- Loss, batch accuracy, homeostatic values, spike counts, and layer activity are taken from the processed pipeline batch. The canvas compresses actual input, hidden-spike, and output-probability means into an illustrative layout; it does not animate or generate synthetic activity.
- The displayed output-layer size is derived from the final layer's measured activity vector, so binary tasks show two outputs rather than a fixed default of ten.
- Working memory indicates whether the active pipeline currently holds its `current_perceptual_state` (active/empty), not a general item total. Episodic count is the active pipeline's local event mirror; semantic concepts and relations count the active session's in-memory graph.
- The displayed input source is `N-MNIST` only when `data/n_mnist/processed_nmnist.pt` is available and loaded. Otherwise classification uses `Synthetic DVS`; XOR and parity always use their labeled synthetic data.
- When the launcher's initial simulation finishes, the web server remains available for dashboard training controls until the process is stopped with `Ctrl+C`.
- `GET /api/leaderboard?type=<id>` returns ranking metadata and available rows; `POST /api/leaderboard/refresh` reports HTTP 409 until a verified official source is configured. At present all ranking URLs are unset, and even a configured source needs an adapter before refresh can work. No ranking values are fabricated.

See [DASHBOARD_TRAINING_GUIDE.md](DASHBOARD_TRAINING_GUIDE.md) for the UI, data provenance, metric semantics, workflow limitations, launch commands, and verification instructions in one place.
