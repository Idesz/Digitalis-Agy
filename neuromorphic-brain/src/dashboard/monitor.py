"""
Dashboard Monitor Server & Terminal Visualizer.

Provides:
1. Live web API & single-page HTML/JS cyberpunk dashboard:
   - 3-column layout: homeostatic gauges | neural-net canvas | metrics + logs
   - Activity-encoded 2-D illustration with cyan nodes & magenta edges
   - Live Chart.js loss/accuracy line graphs
   - Prompt → Plan → Approve → Train flow (no file loader)
   - SSE stream for real-time telemetry
2. Terminal text dashboard (TUI) with ASCII progress meters.
"""

from __future__ import annotations

import json
import math
import os
import re
import queue
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


WORKFLOW_CATALOG = [
    {
        "id": "xor",
        "name": "XOR",
        "prompt": "XOR",
        "status": "ready",
        "dataset": "Synthetic XOR",
        "description": "Kétbites XOR-feladat, két kimeneti osztállyal.",
        "prompt_help": "Írd be: XOR vagy XOR tanulás.",
    },
    {
        "id": "parity",
        "name": "Binary parity",
        "prompt": "parity",
        "status": "ready",
        "dataset": "Synthetic PARITY",
        "description": "Bináris paritás felismerése, két kimeneti osztállyal.",
        "prompt_help": "Írd be: parity vagy paritás.",
    },
    {
        "id": "classification",
        "name": "Általános osztályozás",
        "prompt": "classification",
        "status": "ready",
        "dataset": "N-MNIST, ha a feldolgozott fájl elérhető; különben Synthetic DVS",
        "description": "Felügyelt, címkézett, 10 osztályos osztályozás.",
        "prompt_help": "Írd be: classification vagy egy általános osztályozási célt.",
    },
    {
        "id": "sequence",
        "name": "Szekvencia-osztályozás",
        "prompt": "sequence",
        "status": "experimental",
        "dataset": "Általános osztályozás: N-MNIST, ha elérhető; különben Synthetic DVS",
        "description": "Kísérleti prompt-alias; jelenleg nem külön szekvencia-benchmark.",
        "prompt_help": "Írd be: sequence vagy szekvencia; jelenleg általános osztályozásra irányít.",
    },
]

EXTERNAL_BENCHMARK_CATALOG = [
    {
        "id": "mmlu",
        "name": "MMLU",
        "focus": "Többdoménes, feleletválasztós tudásmérés.",
        "status": "not_connected",
        "reason": "Külön nyelvimodell-adapter és MMLU adatkészlet szükséges.",
    },
    {
        "id": "humaneval",
        "name": "HumanEval",
        "focus": "Kódgenerálás és unit tesztekkel történő futtatás.",
        "status": "not_connected",
        "reason": "Kódoló nyelvimodell és izolált kódfuttató szükséges.",
    },
    {
        "id": "gsm8k_math",
        "name": "GSM8K & MATH",
        "focus": "Többlépcsős matematikai következtetés.",
        "status": "not_connected",
        "reason": "Szöveges feladatokat megoldó nyelvimodell-adapter és adatkészlet szükséges.",
    },
    {
        "id": "lmsys_arena",
        "name": "LMSYS Chatbot Arena",
        "focus": "Emberi preferencia alapján képzett Elo-rangsor.",
        "status": "external_ranking",
        "reason": "Külső rangsor, nem a helyi SNN-futtatóval elvégezhető teszt.",
    },
]

TRAINING_PARADIGMS = [
    {
        "name": "Felügyelt tanulás",
        "status": "supported",
        "description": "A jelenlegi XOR, paritás és osztályozási feladatok címkézett adatokon tanulnak.",
    },
    {
        "name": "Felügyelet nélküli / prediktív tanulás",
        "status": "partial",
        "description": "Prediktív kiegészítő veszteség működik; önálló, címkézetlen tanítási mód nincs kitéve a kezelőfelületen.",
    },
    {
        "name": "Megerősítéses tanulás / RLHF",
        "status": "not_connected",
        "description": "Jutalommodell és interaktív környezet még nincs implementálva.",
    },
]

RANKING_TYPES = [
    {
        "id": "lmsys_arena_elo",
        "name": "LMSYS Chatbot Arena",
        "metric": "Elo / emberi preferencia",
        "source_url": None,
    },
    {
        "id": "benchlm",
        "name": "BenchLM",
        "metric": "Aggregált benchmark-eredmények",
        "source_url": None,
    },
    {
        "id": "epoch_ai",
        "name": "Epoch AI",
        "metric": "Számítási kapacitás, FLOPs és paraméterszám",
        "source_url": None,
    },
]

METRIC_SAMPLE_FIELDS = {
    "epoch",
    "epoch_target",
    "total_steps",
    "data_source",
    "step",
    "curiosity",
    "error_rate",
    "resource_stress",
    "process_cpu_load",
    "uncertainty",
    "behavioral_mode",
    "last_route",
    "last_accuracy",
    "last_surprise",
    "last_loss",
    "last_spike_count",
    "connection_count",
    "layer_activity",
    "output_classes",
    "working_memory_items",
    "episodic_events_count",
    "semantic_concepts_count",
    "semantic_relations_count",
}


def get_dashboard_catalog() -> Dict[str, Any]:
    return {
        "workflows": WORKFLOW_CATALOG,
        "benchmarks": EXTERNAL_BENCHMARK_CATALOG,
        "paradigms": TRAINING_PARADIGMS,
        "ranking_types": [
            {
                **{key: value for key, value in item.items() if key != "source_url"},
                "source_configured": bool(item["source_url"]),
            }
            for item in RANKING_TYPES
        ],
    }


def _reject_unavailable_benchmark(prompt: str) -> None:
    low = prompt.lower()
    unsupported = (
        ("mmlu", "MMLU"),
        ("humaneval", "HumanEval"),
        ("human eval", "HumanEval"),
        ("gsm8k", "GSM8K"),
        ("math benchmark", "MATH"),
        ("math", "MATH"),
        ("lmsys", "LMSYS Chatbot Arena"),
        ("chatbot arena", "LMSYS Chatbot Arena"),
    )
    for keyword, name in unsupported:
        matched = bool(re.search(r"\bmath\b", low)) if keyword == "math" else keyword in low
        if matched:
            raise ValueError(
                f"A(z) {name} szerepel a katalógusban, de a jelenlegi tüskés osztályozóval "
                "nem futtatható; nyelvimodell-adapter és adatforrás szükséges."
            )


def _resolve_training_task(prompt: str) -> tuple[str, str, str, int]:
    low = prompt.lower()
    if "paritás" in low or "parity" in low:
        return "parity", "BinaryParity klasszifikáció", "Bináris paritás szintetikus tüskesorozat", 2
    if "xor" in low:
        return "xor", "XOR tanulás", "XOR szintetikus tüskesorozat", 2
    if "sequence" in low or "szekvencia" in low:
        return (
            "classification",
            "Szekvencia-osztályozás",
            "Általános osztályozási adat (N-MNIST, ha elérhető; különben Synthetic DVS)",
            10,
        )
    return (
        "classification",
        "Általános osztályozás",
        "N-MNIST, ha a feldolgozott fájl elérhető; különben Synthetic DVS",
        10,
    )


def _validate_training_parameters(prompt: str, epochs: int, lr: float, batch_size: int) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A training prompt is required")
    _reject_unavailable_benchmark(prompt)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Learning rate must be a positive finite number")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")


def build_dashboard_telemetry(
    stats: Dict[str, Any],
    agent: Any,
    epoch: int,
    epoch_target: int,
    total_steps: int,
    data_source: str,
) -> Dict[str, Any]:
    state_vector = stats["state_vector"]
    return {
        "has_metrics": True,
        "epoch": epoch,
        "epoch_target": epoch_target,
        "total_steps": total_steps,
        "data_source": data_source,
        "step": stats["step"],
        "training_active": True,
        "training_status": "RUNNING",
        "curiosity": float(state_vector[0]),
        "error_rate": float(state_vector[1]),
        "resource_stress": float(state_vector[2]),
        "process_cpu_load": float(stats["process_cpu_load"]),
        "uncertainty": float(state_vector[3]),
        "behavioral_mode": stats["behavioral_mode"],
        "last_route": stats["route"],
        "last_accuracy": float(stats["accuracy"]),
        "last_surprise": float(stats["surprise"]),
        "last_loss": float(stats["loss"]),
        "last_spike_count": stats["spike_count"],
        "connection_count": stats["connection_count"],
        "layer_activity": stats["layer_activity"],
        "output_classes": len(stats["layer_activity"][-1]),
        "working_memory_items": int(agent.working_mem.get("current_perceptual_state") is not None),
        "episodic_events_count": agent.episodic_mem.num_events(),
        "semantic_concepts_count": agent.semantic_mem.num_concepts(),
        "semantic_relations_count": agent.semantic_mem.num_relations(),
    }


# ---------------------------------------------------------------------------
# State monitor
# ---------------------------------------------------------------------------

class BrainStateMonitor:
    """Thread-safe state repository tracking agent state and telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sse_clients: list[queue.Queue] = []
        self._sse_lock = threading.Lock()

        # Training control
        self._training_lock = threading.Lock()
        self.training_thread: Optional[threading.Thread] = None
        self.training_stop_event = threading.Event()
        self.training_plan: Optional[str] = None

        self.state: Dict[str, Any] = {
            "curiosity": 0.5,
            "error_rate": 0.1,
            "resource_stress": 0.1,
            "process_cpu_load": 0.0,
            "uncertainty": 0.2,
            "behavioral_mode": "BALANCED",
            "last_route": "—",
            "last_accuracy": 0.0,
            "last_surprise": 0.0,
            "last_loss": 0.0,
            "last_spike_count": 0,
            "connection_count": 0,
            "layer_activity": [],
            "output_classes": 0,
            "epoch": 0,
            "epoch_target": 0,
            "total_steps": 0,
            "step": 0,
            "data_source": "—",
            "working_memory_items": 0,
            "episodic_events_count": 0,
            "semantic_concepts_count": 0,
            "semantic_relations_count": 0,
            "has_metrics": False,
            "training_active": False,
            "training_status": "IDLE",
            "recent_logs": [],
        }

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, Any]):
        if "has_metrics" in payload and not isinstance(payload["has_metrics"], bool):
            raise ValueError("has_metrics must be a boolean")
        if payload.get("has_metrics") is True:
            missing_fields = METRIC_SAMPLE_FIELDS.difference(payload)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Incomplete telemetry sample; missing: {missing}")

        with self._lock:
            for k, v in payload.items():
                if k == "log":
                    ts = time.strftime("%H:%M:%S")
                    self.state["recent_logs"].insert(0, f"[{ts}] {v}")
                    self.state["recent_logs"] = self.state["recent_logs"][:30]
                elif k in self.state:
                    self.state[k] = v
        self._broadcast_sse()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(
                json.dumps(
                    self.state,
                    default=lambda x: x.item() if hasattr(x, "item") else str(x),
                )
            )

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    def register_sse_client(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._sse_lock:
            self._sse_clients.append(q)
        return q

    def unregister_sse_client(self, q: queue.Queue):
        with self._sse_lock:
            try:
                self._sse_clients.remove(q)
            except ValueError:
                pass

    def _broadcast_sse(self):
        snap = self.get_snapshot()
        payload = f"data: {json.dumps(snap)}\n\n"
        dead = []
        with self._sse_lock:
            for q in self._sse_clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_clients.remove(q)

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(self, prompt: str, epochs: int = 50, lr: float = 0.001, batch_size: int = 16):
        _validate_training_parameters(prompt, epochs, lr, batch_size)

        with self._training_lock:
            if self.training_thread and self.training_thread.is_alive():
                return False, "Training already running"
            self.training_stop_event.clear()
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(prompt, epochs, lr, batch_size),
                daemon=True,
            )
            self.training_thread.start()
        return True, "Training started"

    def stop_training(self):
        if not self.training_thread or not self.training_thread.is_alive():
            return False, "No training is running"
        self.training_stop_event.set()
        return True, "Stop signal sent"

    def _training_loop(self, prompt: str, epochs: int, lr: float, batch_size: int):
        """Run actual pipeline training loop."""
        self.update({
            "epoch": 0,
            "epoch_target": epochs,
            "total_steps": 0,
            "step": 0,
            "data_source": "—",
            "last_loss": 0.0,
            "last_accuracy": 0.0,
            "last_surprise": 0.0,
            "last_spike_count": 0,
            "connection_count": 0,
            "layer_activity": [],
            "has_metrics": False,
            "training_active": True,
            "training_status": "RUNNING",
            "log": f"Training started: {prompt[:60]}",
        })
        final_status = "COMPLETED"
        try:
            # Import here to avoid circular imports at module load time
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.core.dataset import get_neuromorphic_dataloader
            from src.core.pipeline import NeuromorphicCognitivePipeline

            task, _, _, num_classes = _resolve_training_task(prompt)
            agent = NeuromorphicCognitivePipeline(
                input_dim=64,
                hidden_dims=[128, 64],
                num_classes=num_classes,
                lr=lr,
            )
            loader = get_neuromorphic_dataloader(
                batch_size=batch_size,
                num_samples=batch_size * 10,
                num_classes=num_classes,
                data_dir=os.path.join(project_root, "data", "n_mnist"),
                task=task,
            )
            total_steps = epochs * len(loader)
            self.update({
                "total_steps": total_steps,
                "log": f"Using {loader.dataset.data_source}; {total_steps} batches planned.",
            })

            for epoch in range(1, epochs + 1):
                if self.training_stop_event.is_set():
                    self.update({"log": "Training stopped by user."})
                    final_status = "STOPPED"
                    break
                for batch_x, batch_y in loader:
                    if self.training_stop_event.is_set():
                        break
                    stats = agent.process_batch(batch_x, batch_y, train_step=True)
                    telemetry = build_dashboard_telemetry(
                        stats,
                        agent,
                        epoch,
                        epochs,
                        total_steps,
                        loader.dataset.data_source,
                    )
                    telemetry["log"] = (
                        f"Epoch {epoch} | Batch accuracy {stats['accuracy']*100:.1f}% | Loss {stats['loss']:.4f}"
                    )
                    self.update(telemetry)
                    time.sleep(0.05)

                if self.training_stop_event.is_set():
                    final_status = "STOPPED"
                    self.update({"log": "Training stopped by user."})
                    break

                c_stats = agent.sleep_and_consolidate()
                self.update({
                    "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                    "semantic_relations_count": agent.semantic_mem.num_relations(),
                    "log": f"Sleep consolidation: {c_stats['consolidated_concepts']} concepts",
                })
        except Exception as exc:
            final_status = "ERROR"
            self.update({"log": f"[ERROR] Training failed: {exc}", "training_status": "ERROR"})
        finally:
            self.update({"training_active": False, "training_status": final_status})

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(self, prompt: str, epochs: int, lr: float, batch_size: int) -> str:
        """Heuristic plan generator based on the user's prompt."""
        _validate_training_parameters(prompt, epochs, lr, batch_size)
        _, task, dataset, _ = _resolve_training_task(prompt)

        lr_desc = "gyors konvergencia" if lr > 0.005 else ("finom hangolás" if lr < 0.0005 else "kiegyensúlyozott")
        plan_md = (
            f"**Feladat:** {task}\n"
            f"**Cél (prompt):** {prompt}\n\n"
            f"**Lépések:**\n"
            f"1. 📦 Adathalmaz: {dataset} — batch={batch_size}\n"
            f"2. 🧠 Modell: NeuromorphicCognitivePipeline (DECOLLE SNN, 2 réteg)\n"
            f"3. ⚙️ Tanítás: {epochs} epoch, lr={lr} ({lr_desc})\n"
            f"4. 💤 Sleep consolidation minden epoch után\n"
            f"5. 📊 Élő metrikák: loss, accuracy, surprise\n"
            f"6. ✅ A tanítás a megadott epochok teljesítésével ér véget\n\n"
            f"**Időtartam:** a gép és adatforrás függvénye; a futás közbeni előrehaladás a tényleges batch-eket mutatja."
        )
        return plan_md

    # ------------------------------------------------------------------
    # TUI
    # ------------------------------------------------------------------

    def render_terminal_ascii(self) -> str:
        snap = self.get_snapshot()

        if not snap["has_metrics"]:
            return "\n".join([
                "╔══════════════════════════════════════════════════════════════════════════╗",
                "║                  🧠 NEUROMORPHIC BRAIN MONITOR TUI                      ║",
                "╠══════════════════════════════════════════════════════════════════════════╣",
                f"║ Status: {snap['training_status']:<8} | Waiting for telemetry data...            ║",
                "╚══════════════════════════════════════════════════════════════════════════╝",
            ])

        def bar(val: float, length: int = 20) -> str:
            filled = int(round(val * length))
            return "█" * filled + "░" * (length - filled)

        lines = [
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║                  🧠 NEUROMORPHIC BRAIN MONITOR TUI                      ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║ Mode: {snap['behavioral_mode']:<14} | Route: {snap['last_route']:<24}        ║",
            f"║ Epoch: {snap['epoch']:<6} | Batch Acc: {snap['last_accuracy']*100:>5.1f}% | Local Loss: {snap['last_loss']:>7.4f} | Status: {snap['training_status']:<8}║",
            "╟──────────────────────────────────────────────────────────────────────────╢",
            f"║ Curiosity:       [{bar(snap['curiosity'])}] {snap['curiosity']:>4.2f}                   ║",
            f"║ Error Rate:      [{bar(snap['error_rate'])}] {snap['error_rate']:>4.2f}                   ║",
            f"║ CPU Stress (smoothed): [{bar(snap['resource_stress'])}] {snap['resource_stress']:>4.2f}       ║",
            f"║ Uncertainty:     [{bar(snap['uncertainty'])}] {snap['uncertainty']:>4.2f}                   ║",
            "╟──────────────────────────────────────────────────────────────────────────╢",
            f"║ Local episodic events: {snap['episodic_events_count']:<5} | Session concepts: {snap['semantic_concepts_count']:<5}             ║",
            "╟────────────────────────────── RECENT LOGS ───────────────────────────────╢",
        ]
        for log_line in (snap["recent_logs"][:4] or ["(No logs yet)"]):
            lines.append(f"║ {log_line[:72]:<72} ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

global_monitor = BrainStateMonitor()

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<title>🧠 Neuromorphic Brain — Control Center</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
/* ═══════════════════════════════════════════════
   RESET & ROOT
═══════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;
  --bg2:#0a0e14;
  --card:#111827;
  --card2:#161f2e;
  --border:#1f2937;
  --border2:#2d3748;
  --text:#c9d1d9;
  --text2:#8b949e;
  --cyan:#00f5ff;
  --cyan2:#00bcd4;
  --magenta:#ff4de6;
  --green:#3fb950;
  --yellow:#e3b341;
  --red:#f85149;
  --blue:#58a6ff;
  --purple:#a371f7;
  --font:'JetBrains Mono','Cascadia Code','Fira Code','Courier New',monospace;
  --glow-cyan:0 0 8px #00f5ff88,0 0 20px #00f5ff33;
  --glow-magenta:0 0 8px #ff4de688,0 0 20px #ff4de633;
  --glow-green:0 0 8px #3fb95088;
}
html,body{height:100%;background:var(--bg2)}
body{font-family:var(--font);color:var(--text);font-size:12px;overflow:hidden;display:flex;flex-direction:column}

/* ═══════════════════════════════════════════════
   WINDOW CHROME (macOS style)
═══════════════════════════════════════════════ */
#window-chrome{
  display:flex;align-items:center;gap:8px;
  padding:8px 14px;
  background:#1a1f2e;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
  user-select:none;
}
.traffic-light{width:12px;height:12px;border-radius:50%;cursor:pointer;transition:opacity .2s}
#tl-red{background:#ff5f57}
#tl-yellow{background:#febc2e}
#tl-green{background:#28c840}
.traffic-light:hover{opacity:.7}
#window-title{
  margin-left:8px;
  font-size:12px;
  color:var(--text2);
  flex:1;
  letter-spacing:.08em;
}
#status-pill{
  display:flex;align-items:center;gap:6px;
  padding:3px 10px;border-radius:12px;
  background:#0d2818;border:1px solid var(--green);
  font-size:10px;color:var(--green);letter-spacing:.1em;
}
#status-dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--green);
  animation:pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot{0%,100%{opacity:1;box-shadow:var(--glow-green)}50%{opacity:.4;box-shadow:none}}

/* Main view navigation and benchmark workspace */
[hidden]{display:none!important}
.view-page{flex:1;min-height:0}
#control-view{display:flex;flex-direction:column}
#view-nav,#lab-nav{
  display:flex;gap:6px;align-items:center;
  padding:6px 12px;background:#0a0e14;border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.view-tab,.lab-tab{
  padding:7px 12px;background:transparent;color:var(--text2);
  border:1px solid transparent;border-radius:4px;
  font:600 10px var(--font);letter-spacing:.08em;text-transform:uppercase;
  cursor:pointer;
}
.view-tab[aria-selected="true"],.lab-tab[aria-selected="true"]{
  color:var(--cyan);border-color:var(--cyan);background:#001a1a;
}
#lab-view{overflow-y:auto;padding:16px}
.lab-heading{margin:2px 0 14px}
.lab-heading h1{font-size:16px;color:var(--cyan);margin-bottom:5px}
.lab-heading p,.catalog-card p,.paradigm-card p{color:var(--text2);line-height:1.55}
.catalog-section{margin:16px 0 20px}
.catalog-section h2{font-size:12px;color:var(--text);margin-bottom:8px;letter-spacing:.08em}
.catalog-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.catalog-card{
  display:flex;flex-direction:column;gap:8px;padding:12px;
  background:var(--card);border:1px solid var(--border2);border-radius:6px;
}
.catalog-card h3{font-size:12px;color:var(--cyan)}
.catalog-card code{color:var(--yellow);white-space:pre-wrap;overflow-wrap:anywhere}
.catalog-status{font-size:9px;font-weight:700;letter-spacing:.08em;color:var(--green)}
.catalog-status.experimental,.catalog-status.partial{color:var(--yellow)}
.catalog-status.not_connected,.catalog-status.external_ranking{color:var(--text2)}
.catalog-card button,.ranking-action,.help-close{
  align-self:flex-start;padding:6px 10px;border:1px solid var(--cyan);
  border-radius:4px;background:#001a1a;color:var(--cyan);
  font:600 10px var(--font);cursor:pointer;
}
.catalog-card button:disabled{opacity:.55;cursor:not-allowed}
.catalog-note,.ranking-notice{
  padding:10px 12px;background:#111827;border:1px solid var(--border2);
  border-radius:5px;color:var(--text2);line-height:1.6;
}
#catalog-status,#ranking-status{margin:8px 0}
.ranking-toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}
.ranking-toolbar label{display:flex;align-items:center;gap:8px;color:var(--text2)}
#ranking-type{
  min-width:240px;padding:7px;background:#111827;color:var(--text);
  border:1px solid var(--border2);border-radius:4px;font:11px var(--font);
}
.ranking-action{background:#102038;border-color:var(--blue);color:var(--blue)}
.ranking-table-wrap{overflow-x:auto;margin-top:12px}
#ranking-table{width:100%;border-collapse:collapse;min-width:500px;background:var(--card)}
#ranking-table th,#ranking-table td{padding:9px 10px;border-bottom:1px solid var(--border);text-align:left}
#ranking-table th{color:var(--text2);font-size:9px;text-transform:uppercase;letter-spacing:.08em}
#ranking-table td{color:var(--text)}
.ranking-subheading{font-size:12px;color:var(--cyan);margin:18px 0 7px}
#prompt-help-dialog{
  width:min(760px,92vw);max-height:82vh;margin:auto;padding:0;
  color:var(--text);background:var(--card);border:1px solid var(--cyan);border-radius:8px;
}
#prompt-help-dialog::backdrop{background:#000b;backdrop-filter:blur(2px)}
.help-content{padding:18px;overflow-y:auto;max-height:82vh}
.help-content h2{font-size:14px;color:var(--cyan);margin-bottom:8px}
.help-content h3{font-size:11px;color:var(--yellow);margin:14px 0 6px}
.help-content p,.help-content li{color:var(--text2);line-height:1.6}
.help-content ul{padding-left:20px}
.help-close{margin-top:14px}

/* ═══════════════════════════════════════════════
   METRIC STRIP
═══════════════════════════════════════════════ */
#metric-strip{
  display:flex;align-items:center;gap:0;
  background:#0a0e14;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
  overflow:hidden;
}
.metric-cell{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:6px 4px;
  border-right:1px solid var(--border);
  min-width:80px;
}
.metric-cell:last-child{border-right:none}
.mc-label{font-size:9px;color:var(--text2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:2px}
.mc-value{font-size:16px;font-weight:700;letter-spacing:.04em;line-height:1}
.mc-cyan{color:var(--cyan);text-shadow:var(--glow-cyan)}
.mc-green{color:var(--green);text-shadow:var(--glow-green)}
.mc-magenta{color:var(--magenta);text-shadow:var(--glow-magenta)}
.mc-yellow{color:var(--yellow)}
.mc-blue{color:var(--blue)}
.mc-purple{color:var(--purple)}

/* ═══════════════════════════════════════════════
   MAIN GRID
═══════════════════════════════════════════════ */
#main-grid{
  flex:1;
  display:grid;
  grid-template-columns:260px 1fr 300px;
  gap:1px;
  background:var(--border);
  overflow:hidden;
  min-height:0;
}
.panel{
  background:var(--card);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.panel-header{
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;
  background:#0d1117;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.panel-header-icon{font-size:14px}
.panel-header-title{
  font-size:10px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--text2);font-weight:600;
}
.panel-body{flex:1;overflow-y:auto;padding:10px 12px;min-height:0}
.panel-body::-webkit-scrollbar{width:4px}
.panel-body::-webkit-scrollbar-track{background:transparent}
.panel-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* ═══════════════════════════════════════════════
   LEFT PANEL — HOMEOSTATIC GAUGES
═══════════════════════════════════════════════ */
.gauge-row{margin-bottom:14px}
.gauge-label{
  display:flex;justify-content:space-between;
  font-size:10px;color:var(--text2);letter-spacing:.08em;
  margin-bottom:4px;
}
.gauge-label span:last-child{color:var(--text);font-weight:700}
.gauge-track{
  height:5px;background:#1a2235;border-radius:3px;overflow:hidden;
  position:relative;
}
.gauge-fill{
  height:100%;border-radius:3px;width:0%;
  transition:width .5s cubic-bezier(.4,0,.2,1);
}
.gf-cyan{background:linear-gradient(90deg,var(--cyan2),var(--cyan));box-shadow:var(--glow-cyan)}
.gf-red{background:linear-gradient(90deg,#c0392b,var(--red))}
.gf-orange{background:linear-gradient(90deg,#c0680b,#f97316)}
.gf-purple{background:linear-gradient(90deg,#6b46c1,var(--purple))}

.section-sep{
  border:none;border-top:1px solid var(--border);
  margin:14px 0;
}
.memory-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid var(--border);
}
.memory-row:last-child{border-bottom:none}
.mr-label{font-size:10px;color:var(--text2)}
.mr-value{font-size:12px;font-weight:700;color:var(--cyan)}

.badge-row{margin-top:12px}
.badge{
  display:inline-block;padding:3px 10px;border-radius:4px;
  font-size:10px;letter-spacing:.1em;font-weight:700;
  border:1px solid;
}
.badge-green{color:var(--green);border-color:var(--green);background:#0d2818}
.badge-yellow{color:var(--yellow);border-color:var(--yellow);background:#1f1700}
.badge-blue{color:var(--blue);border-color:var(--blue);background:#0d1f33}
.badge-magenta{color:var(--magenta);border-color:var(--magenta);background:#1f0d1a}

/* ═══════════════════════════════════════════════
   CENTER PANEL — NEURAL NETWORK CANVAS
═══════════════════════════════════════════════ */
#nn-canvas-wrap{
  flex:1;display:flex;align-items:center;justify-content:center;
  min-height:0;position:relative;
  padding:8px;
}
#nn-canvas{
  width:100%;height:100%;
  display:block;
}
#nn-layer-bar{
  display:flex;gap:6px;padding:8px 12px;
  background:#0d1117;border-top:1px solid var(--border);
  flex-shrink:0;overflow-x:auto;
}
.layer-pill{
  flex-shrink:0;padding:4px 10px;border-radius:4px;
  border:1px solid var(--border2);background:#111827;
  font-size:10px;color:var(--text2);letter-spacing:.08em;
  white-space:nowrap;
}
.layer-pill span{color:var(--cyan);font-weight:700}

/* ═══════════════════════════════════════════════
   RIGHT PANEL — CHARTS + LOGS
═══════════════════════════════════════════════ */
.chart-wrap{position:relative;height:120px;margin-bottom:10px}
.chart-title{font-size:9px;color:var(--text2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px}

#logs-terminal{
  background:#060a0f;border:1px solid var(--border);
  border-radius:4px;padding:8px;
  font-size:10px;line-height:1.6;
  max-height:160px;overflow-y:auto;
  flex-shrink:0;
}
#logs-terminal::-webkit-scrollbar{width:3px}
#logs-terminal::-webkit-scrollbar-thumb{background:var(--border2)}
.log-line{color:#6b8099}
.log-line.new{color:var(--text);animation:log-fade 2s ease forwards}
@keyframes log-fade{from{color:var(--cyan)}to{color:#6b8099}}

/* ═══════════════════════════════════════════════
   BOTTOM CONTROL PANEL
═══════════════════════════════════════════════ */
#control-panel{
  background:#0a0e14;border-top:2px solid var(--border);
  padding:10px 14px;
  display:flex;flex-direction:column;gap:8px;
  flex-shrink:0;
}
#ctrl-top{display:flex;gap:8px;align-items:flex-start}
#prompt-input{
  flex:1;
  background:#111827;border:1px solid var(--border2);border-radius:4px;
  color:var(--text);font-family:var(--font);font-size:11px;
  padding:8px;resize:none;height:52px;
  outline:none;transition:border-color .2s;
}
#prompt-input:focus{border-color:var(--cyan);box-shadow:0 0 0 2px #00f5ff22}
#prompt-input::placeholder{color:var(--text2)}
#btn-prompt-help{
  flex-shrink:0;padding:8px 10px;border:1px solid var(--border2);
  border-radius:4px;background:#111827;color:var(--cyan);
  font:600 11px var(--font);cursor:pointer;
}
#btn-prompt-help:hover{border-color:var(--cyan)}

.ctrl-btn{
  padding:8px 14px;border-radius:4px;border:none;
  font-family:var(--font);font-size:11px;font-weight:700;
  letter-spacing:.06em;cursor:pointer;
  transition:all .15s ease;white-space:nowrap;
}
.ctrl-btn:active{transform:scale(.96)}
#btn-plan{background:var(--blue);color:#0d1117}
#btn-plan:hover{background:#79c0ff}
#btn-start{background:var(--green);color:#0d1117;display:none}
#btn-start:hover{background:#56d364}
#btn-stop{background:var(--red);color:#fff;display:none}
#btn-stop:hover{background:#ff7b72}

#ctrl-params{
  display:flex;gap:12px;align-items:center;flex-wrap:wrap;
}
.param-group{display:flex;align-items:center;gap:6px}
.param-label{font-size:10px;color:var(--text2);white-space:nowrap}
.param-slider{width:80px;accent-color:var(--cyan);cursor:pointer}
.param-val{font-size:11px;color:var(--cyan);min-width:36px;text-align:right}

/* Plan box */
#plan-box{
  display:none;
  background:#0d1a0d;border:1px solid #1f4a1f;border-radius:4px;
  padding:10px 12px;
  font-size:11px;line-height:1.65;
  color:#b7f0b7;
  max-height:120px;overflow-y:auto;
  position:relative;
}
#plan-box::-webkit-scrollbar{width:3px}
#plan-box::-webkit-scrollbar-thumb{background:#1f4a1f}
#plan-box-actions{display:flex;gap:8px;margin-top:8px}
#btn-approve{
  background:linear-gradient(135deg,#1a4a1a,#2d7a2d);
  color:var(--green);border:1px solid var(--green);
  border-radius:4px;padding:5px 14px;cursor:pointer;
  font-family:var(--font);font-size:11px;font-weight:700;
  letter-spacing:.05em;
  transition:all .15s;
}
#btn-approve:hover{background:linear-gradient(135deg,#2d7a2d,#3fb950);color:#fff}
#btn-discard{
  background:transparent;color:var(--text2);border:1px solid var(--border2);
  border-radius:4px;padding:5px 10px;cursor:pointer;
  font-family:var(--font);font-size:11px;
  transition:all .15s;
}
#btn-discard:hover{color:var(--red);border-color:var(--red)}

/* training progress bar */
#training-bar-wrap{display:none;align-items:center;gap:8px}
#training-bar-track{
  flex:1;height:4px;background:#1a2235;border-radius:2px;overflow:hidden;
}
#training-bar-fill{
  height:100%;width:0%;
  background:linear-gradient(90deg,var(--cyan2),var(--cyan));
  transition:width .6s ease;
  box-shadow:var(--glow-cyan);
}
#training-bar-label{font-size:10px;color:var(--cyan);min-width:80px;text-align:right}

/* ═══════════════════════════════════════════════
   SCROLLBAR GLOBAL
═══════════════════════════════════════════════ */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body>

<!-- ══ WINDOW CHROME ══════════════════════════════════════════════════ -->
<div id="window-chrome">
  <div class="traffic-light" id="tl-red" title="Close"></div>
  <div class="traffic-light" id="tl-yellow" title="Minimize"></div>
  <div class="traffic-light" id="tl-green" title="Maximize"></div>
  <span id="window-title">🧠 NEUROMORPHIC BRAIN — CONTROL CENTER &nbsp;·&nbsp; <span id="wt-host">localhost:8000</span></span>
  <div id="status-pill">
    <div id="status-dot"></div>
    <span id="status-text">ACTIVE</span>
  </div>
</div>

<nav id="view-nav" aria-label="Fő nézetek">
  <button class="view-tab" id="view-control" data-view="control" aria-selected="true">Control Center</button>
  <button class="view-tab" id="view-lab" data-view="lab" aria-selected="false">Tesztközpont</button>
</nav>

<div id="control-view" class="view-page">
<!-- ══ METRIC STRIP ═══════════════════════════════════════════════════ -->
<div id="metric-strip">
  <div class="metric-cell">
    <div class="mc-label">EPOCH</div>
    <div class="mc-value mc-cyan" id="ms-epoch">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">LOCAL LOSS / BATCH</div>
    <div class="mc-value mc-magenta" id="ms-loss">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">BATCH ACCURACY</div>
    <div class="mc-value mc-green" id="ms-acc">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">SURPRISE</div>
    <div class="mc-value mc-yellow" id="ms-surprise">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">MODE</div>
    <div class="mc-value mc-blue" id="ms-mode" style="font-size:11px">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">ROUTE</div>
    <div class="mc-value mc-purple" id="ms-route" style="font-size:10px">—</div>
  </div>
  <div class="metric-cell">
    <div class="mc-label">STATUS</div>
    <div class="mc-value mc-green" id="ms-status" style="font-size:11px">IDLE</div>
  </div>
</div>

<!-- ══ MAIN GRID ══════════════════════════════════════════════════════ -->
<div id="main-grid">

  <!-- LEFT: Homeostatic Panel -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-header-icon">⚡</span>
      <span class="panel-header-title">Homeostatic State</span>
    </div>
    <div class="panel-body">

      <div class="gauge-row">
        <div class="gauge-label"><span>CURIOSITY</span><span id="gv-curiosity">—</span></div>
        <div class="gauge-track"><div class="gauge-fill gf-cyan" id="gf-curiosity"></div></div>
      </div>
      <div class="gauge-row">
        <div class="gauge-label"><span>ERROR RATE</span><span id="gv-error">—</span></div>
        <div class="gauge-track"><div class="gauge-fill gf-red" id="gf-error"></div></div>
      </div>
      <div class="gauge-row">
        <div class="gauge-label"><span>PROCESS CPU STRESS (SMOOTHED)</span><span id="gv-stress">—</span></div>
        <div class="gauge-track"><div class="gauge-fill gf-orange" id="gf-stress"></div></div>
      </div>
      <div class="gauge-row">
        <div class="gauge-label"><span>UNCERTAINTY</span><span id="gv-uncertainty">—</span></div>
        <div class="gauge-track"><div class="gauge-fill gf-purple" id="gf-uncertainty"></div></div>
      </div>

      <hr class="section-sep">

      <div class="memory-row"><span class="mr-label">Local Episodic Events</span><span class="mr-value" id="mv-episodic">—</span></div>
      <div class="memory-row"><span class="mr-label">Session Semantic Concepts</span><span class="mr-value" id="mv-semantic">—</span></div>
      <div class="memory-row"><span class="mr-label">Session Semantic Relations</span><span class="mr-value" id="mv-relations">—</span></div>
      <div class="memory-row"><span class="mr-label">Current Percept</span><span class="mr-value" id="mv-working">—</span></div>

      <hr class="section-sep">

      <div class="badge-row">
        <span class="mr-label" style="display:block;margin-bottom:6px">Behavioral Mode</span>
        <span class="badge badge-green" id="badge-mode">BALANCED</span>
      </div>
      <div class="badge-row" style="margin-top:8px">
        <span class="mr-label" style="display:block;margin-bottom:6px">Decision Route</span>
        <span class="badge badge-blue" id="badge-route">FAST_SYSTEM_1</span>
      </div>

    </div>
  </div>

  <!-- CENTER: Neural Network Canvas -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-header-icon">🕸</span>
      <span class="panel-header-title">Neural Activity — Aggregated, Illustrative Layout</span>
    </div>
    <div id="nn-canvas-wrap">
      <canvas id="nn-canvas"></canvas>
    </div>
    <div id="nn-layer-bar">
      <div class="layer-pill">INPUT <span id="lp-in">64</span></div>
      <div class="layer-pill">DECOLLE-1 <span id="lp-h1">128</span></div>
      <div class="layer-pill">DECOLLE-2 <span id="lp-h2">64</span></div>
      <div class="layer-pill">OUTPUT <span id="lp-out">—</span></div>
      <div class="layer-pill">SPIKES / BATCH <span id="lp-spikes" style="color:var(--magenta)">—</span></div>
      <div class="layer-pill">CONNECTIONS <span id="lp-conns" style="color:var(--magenta)">—</span></div>
      <div class="layer-pill">DATA <span id="data-source" style="color:var(--magenta)">—</span></div>
    </div>
  </div>

  <!-- RIGHT: Charts + Logs -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-header-icon">📈</span>
      <span class="panel-header-title">Metrics &amp; Telemetry</span>
    </div>
    <div class="panel-body" style="display:flex;flex-direction:column;gap:8px">
      <div>
        <div class="chart-title">LOCAL LOSS PER BATCH</div>
        <div class="chart-wrap"><canvas id="loss-chart"></canvas></div>
      </div>
      <div>
        <div class="chart-title">BATCH ACCURACY</div>
        <div class="chart-wrap"><canvas id="acc-chart"></canvas></div>
      </div>
      <div class="chart-title" style="margin-top:4px">SYSTEM LOGS</div>
      <div id="logs-terminal">
        <div class="log-line">[ SYSTEM ] Dashboard initialised — waiting for data stream...</div>
      </div>
    </div>
  </div>

</div><!-- /main-grid -->

<!-- ══ CONTROL PANEL ══════════════════════════════════════════════════ -->
<div id="control-panel">

  <div id="ctrl-top">
    <textarea id="prompt-input" placeholder="Írd le a tanítási célt... (pl. 'Tanítsd a hálót bináris paritás felismerésére 50 elemen')"></textarea>
    <button id="btn-prompt-help" type="button" aria-label="Prompt súgó">? SÚGÓ</button>
    <button class="ctrl-btn" id="btn-plan">🔍 TERV</button>
    <button class="ctrl-btn" id="btn-stop">⏹ STOP</button>
  </div>

  <div id="plan-box">
    <div id="plan-content"></div>
    <div id="plan-box-actions">
      <button id="btn-approve">✅ Jóváhagyás — Tanítás indítása</button>
      <button id="btn-discard">✕ Elvetés</button>
    </div>
  </div>

  <div id="ctrl-params">
    <div class="param-group">
      <span class="param-label">EPOCHOK</span>
      <input type="range" class="param-slider" id="sl-epochs" min="5" max="200" value="50">
      <span class="param-val" id="sv-epochs">50</span>
    </div>
    <div class="param-group">
      <span class="param-label">LR</span>
      <input type="range" class="param-slider" id="sl-lr" min="1" max="100" value="34">
      <span class="param-val" id="sv-lr">0.001</span>
    </div>
    <div class="param-group">
      <span class="param-label">BATCH</span>
      <input type="range" class="param-slider" id="sl-batch" min="1" max="8" value="4">
      <span class="param-val" id="sv-batch">16</span>
    </div>
    <div id="training-bar-wrap">
      <div id="training-bar-track"><div id="training-bar-fill"></div></div>
      <span id="training-bar-label">— / —</span>
    </div>
    <button class="ctrl-btn" id="btn-start">▶ INDÍTÁS</button>
  </div>

</div>
</div>

<main id="lab-view" class="view-page" hidden>
  <div class="lab-heading">
    <h1>Tesztközpont</h1>
    <p>Helyben futtatható feladatok, külső benchmarkok és összehasonlító rangsorok.</p>
  </div>
  <nav id="lab-nav" aria-label="Tesztközpont nézetei">
    <button class="lab-tab" id="lab-tests-tab" data-lab-tab="tests" aria-selected="true">Tesztkatalógus</button>
    <button class="lab-tab" id="lab-ranking-tab" data-lab-tab="ranking" aria-selected="false">AI rangsor</button>
  </nav>

  <section id="tests-panel">
    <div id="catalog-status" class="catalog-note" role="status" aria-live="polite">Tesztkatalógus betöltése…</div>
    <section class="catalog-section">
      <h2>HELYBEN FUTTATHATÓ FELADATOK</h2>
      <div id="workflow-cards" class="catalog-grid"></div>
    </section>
    <section class="catalog-section">
      <h2>KÜLSŐ STANDARD BENCHMARKOK</h2>
      <div id="benchmark-cards" class="catalog-grid"></div>
    </section>
    <section class="catalog-section">
      <h2>TANÍTÁSI PARADIGMÁK</h2>
      <div id="paradigm-cards" class="catalog-grid"></div>
    </section>
  </section>

  <section id="ranking-panel" hidden>
    <div class="lab-heading">
      <h1>AI rangsor</h1>
      <p>Rangsorforrásonként jelenítjük meg az adott mérőszámot; eltérő rangsorok nem hasonlíthatók közvetlenül össze.</p>
    </div>
    <div class="ranking-toolbar">
      <label for="ranking-type">Rangsor típusa
        <select id="ranking-type"></select>
      </label>
      <button class="ranking-action" id="refresh-ranking" type="button">↻ Adatok frissítése</button>
    </div>
    <div id="ranking-status" class="ranking-notice" role="status" aria-live="polite">Rangsorforrás betöltése…</div>
    <h2 class="ranking-subheading" id="ranking-title">Aktuális rangsor</h2>
    <p class="catalog-note" id="ranking-metric"></p>
    <div class="ranking-table-wrap">
      <table id="ranking-table">
        <thead><tr><th>Hely</th><th>Modell</th><th>Mérőszám</th><th>Frissítve</th></tr></thead>
        <tbody id="ranking-rows"></tbody>
      </table>
    </div>
    <p class="catalog-note" id="ranking-empty">Nincs megjeleníthető rangsoradat.</p>
    <h2 class="ranking-subheading">Digitalis-Agy összehasonlítás</h2>
    <p id="digitalis-ranking-status" class="catalog-note">
      Még nincs azonos benchmarkon mért, összehasonlítható Digitalis-Agy eredmény.
    </p>
  </section>
</main>

<dialog id="prompt-help-dialog">
  <div class="help-content">
    <h2>Prompt- és tanítási súgó</h2>
    <p>A prompt jelenleg a feladattípust választja ki kulcsszavak alapján; nem általános nyelvi utasításértelmező.</p>
    <div id="prompt-help-content"></div>
    <button class="help-close" id="close-prompt-help" type="button">Bezárás</button>
  </div>
</dialog>

<!-- ══ JAVASCRIPT ═════════════════════════════════════════════════════ -->
<script>
// ─── State ───────────────────────────────────────────────────────────────
let state = {};
let lossHistory = [];
let accHistory  = [];
let epochTarget = 50;
let currentPlan = null;
let lastChartStep = -1;

// ─── Chart.js setup ──────────────────────────────────────────────────────
const CHART_OPTS = (color, label) => ({
  type:'line',
  data:{
    labels:[],
    datasets:[{
      label,
      data:[],
      borderColor:color,
      borderWidth:1.5,
      pointRadius:0,
      fill:true,
      backgroundColor: color+'18',
      tension:.35,
    }]
  },
  options:{
    responsive:true,maintainAspectRatio:false,
    animation:{duration:0},
    plugins:{legend:{display:false}},
    scales:{
      x:{display:false},
      y:{
        grid:{color:'#1f293788'},
        ticks:{color:'#8b949e',font:{family:"'JetBrains Mono',monospace",size:9},maxTicksLimit:4},
        border:{color:'#1f2937'},
      }
    }
  }
});

const lossChart = new Chart(document.getElementById('loss-chart'), CHART_OPTS('#ff4de6','Local loss/batch'));
const accChart  = new Chart(document.getElementById('acc-chart'),  CHART_OPTS('#3fb950','Batch accuracy'));

function pushChart(chart, val, maxLen=80){
  const d=chart.data.datasets[0].data;
  const l=chart.data.labels;
  d.push(val); l.push('');
  if(d.length>maxLen){d.shift();l.shift()}
  chart.update('none');
}

// ─── Neural Network Canvas ────────────────────────────────────────────────
const nnCanvas = document.getElementById('nn-canvas');
const ctx = nnCanvas.getContext('2d');

const LAYERS = [5, 8, 6, 3];  // hidden nodes are illustrative; output count follows telemetry
let nnActivity = Array.from({length:LAYERS.length}, (_,li)=>Array(LAYERS[li]).fill(0));

function resizeNN(){
  const wrap = document.getElementById('nn-canvas-wrap');
  nnCanvas.width  = wrap.clientWidth;
  nnCanvas.height = wrap.clientHeight;
  drawNN();
}

function getNodePos(li, ni){
  const W=nnCanvas.width, H=nnCanvas.height;
  const pad=40;
  const x = pad + (li/(LAYERS.length-1))*(W-pad*2);
  const count = LAYERS[li];
  const spacing = Math.min((H-pad*2)/(count+1), 48);
  const totalH = spacing*(count-1);
  const y = H/2 - totalH/2 + ni*spacing;
  return {x,y};
}

function drawNN(){
  const W=nnCanvas.width, H=nnCanvas.height;
  ctx.clearRect(0,0,W,H);

  // Background grid
  ctx.strokeStyle='#0f1824';ctx.lineWidth=.5;
  for(let gx=0;gx<W;gx+=30){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke()}
  for(let gy=0;gy<H;gy+=30){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke()}

  if(!state.has_metrics){
    ctx.fillStyle='#748094';
    ctx.font='12px monospace';
    ctx.textAlign='center';
    ctx.fillText('Waiting for telemetry',W/2,H/2);
    return;
  }

  // Edges
  for(let li=0;li<LAYERS.length-1;li++){
    for(let ni=0;ni<LAYERS[li];ni++){
      for(let nj=0;nj<LAYERS[li+1];nj++){
        const a=getNodePos(li,ni), b=getNodePos(li+1,nj);
        const activity = (nnActivity[li][ni]+nnActivity[li+1][nj])/2;
        const alpha = 0.07 + activity*0.5;
        ctx.beginPath();
        ctx.moveTo(a.x,a.y);
        ctx.lineTo(b.x,b.y);
        ctx.strokeStyle=`hsla(290,100%,70%,${alpha})`;
        ctx.lineWidth=activity>0.3?1.5:0.6;
        ctx.stroke();
      }
    }
  }

  // Nodes
  for(let li=0;li<LAYERS.length;li++){
    for(let ni=0;ni<LAYERS[li];ni++){
      const {x,y}=getNodePos(li,ni);
      const act=nnActivity[li][ni];

      // Glow
      if(act>0.2){
        const g=ctx.createRadialGradient(x,y,0,x,y,22*act);
        g.addColorStop(0,`rgba(0,245,255,${act*0.35})`);
        g.addColorStop(1,'rgba(0,245,255,0)');
        ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,22*act,0,Math.PI*2);ctx.fill();
      }

      // Node circle
      const r=5+act*4;
      const g2=ctx.createRadialGradient(x,y,0,x,y,r);
      const brightness=Math.floor(180+act*75);
      g2.addColorStop(0,`rgb(${brightness},255,255)`);
      g2.addColorStop(1,`rgb(0,${Math.floor(180+act*50)},${Math.floor(200+act*55)})`);
      ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=g2;ctx.fill();
      ctx.strokeStyle=`rgba(0,245,255,${0.4+act*0.5})`;ctx.lineWidth=1;ctx.stroke();
    }
  }

}
resizeNN();
window.addEventListener('resize',resizeNN);

// ─── UI State update from SSE / polling ──────────────────────────────────
function applyState(d){
  state=d;
  const hasMetrics=Boolean(d.has_metrics);

  // Metric strip
  document.getElementById('ms-epoch').textContent = hasMetrics?d.epoch:'—';
  document.getElementById('ms-loss').textContent  = hasMetrics?d.last_loss.toFixed(4):'—';
  document.getElementById('ms-acc').textContent   = hasMetrics?(d.last_accuracy*100).toFixed(1)+'%':'—';
  document.getElementById('ms-surprise').textContent = hasMetrics?d.last_surprise.toFixed(4):'—';
  document.getElementById('ms-mode').textContent  = hasMetrics?(d.behavioral_mode||'—'):'—';
  const routeShort=hasMetrics?(d.last_route||'—').replace('FAST_SYSTEM_1','FAST_S1').replace('SLOW_SYSTEM_2','SLOW_S2'):'—';
  document.getElementById('ms-route').textContent = routeShort;
  const sts=d.training_status||'IDLE';
  const msStatus=document.getElementById('ms-status');
  msStatus.textContent=sts;
  msStatus.className='mc-value '+(sts==='RUNNING'?'mc-cyan':sts==='ERROR'?'mc-magenta':'mc-green');

  // Status pill
  const pill=document.getElementById('status-pill');
  const dot=document.getElementById('status-dot');
  const stxt=document.getElementById('status-text');
  if(d.training_active){
    pill.style.borderColor='var(--cyan)';pill.style.background='#001a1a';
    dot.style.background='var(--cyan)';stxt.style.color='var(--cyan)';
    stxt.textContent='TRAINING';dot.style.animationDuration='.6s';
  } else {
    const isError=sts==='ERROR';
    pill.style.borderColor=isError?'var(--red)':'var(--green)';
    pill.style.background=isError?'#2a1010':'#0d2818';
    dot.style.background=isError?'var(--red)':'var(--green)';
    stxt.style.color=isError?'var(--red)':'var(--green)';
    stxt.textContent=sts;dot.style.animationDuration='1.4s';
  }

  // Gauges
  const setGauge=(id,val)=>{
    if(!hasMetrics){
      document.getElementById('gf-'+id).style.width='0%';
      document.getElementById('gv-'+id).textContent='—';
      return;
    }
    const pct=Math.min(100,Math.max(0,val*100)).toFixed(0)+'%';
    document.getElementById('gf-'+id).style.width=pct;
    document.getElementById('gv-'+id).textContent=(+val).toFixed(2);
  };
  setGauge('curiosity',d.curiosity);
  setGauge('error',d.error_rate);
  setGauge('stress',d.resource_stress);
  setGauge('uncertainty',d.uncertainty);

  // Memory
  document.getElementById('mv-episodic').textContent=hasMetrics?Number(d.episodic_events_count).toLocaleString():'—';
  document.getElementById('mv-semantic').textContent=hasMetrics?Number(d.semantic_concepts_count).toLocaleString():'—';
  document.getElementById('mv-relations').textContent=hasMetrics?Number(d.semantic_relations_count).toLocaleString():'—';
  document.getElementById('mv-working').textContent=hasMetrics?(d.working_memory_items?'ACTIVE':'EMPTY'):'—';

  // Badges
  const mode=hasMetrics?(d.behavioral_mode||'BALANCED'):'—';
  const modeEl=document.getElementById('badge-mode');
  modeEl.textContent=mode;
  modeEl.className='badge '+(mode==='EXPLORATORY'?'badge-cyan':mode==='CONSERVATIVE'?'badge-yellow':'badge-green');

  const route=hasMetrics?(d.last_route||'—'):'—';
  const routeEl=document.getElementById('badge-route');
  routeEl.textContent=route;
  routeEl.className='badge '+(route.includes('SLOW')?'badge-magenta':'badge-blue');

  // Charts
  if(d.step===0)lastChartStep=-1;
  if(d.training_active && hasMetrics && d.step!==lastChartStep){
    pushChart(lossChart, d.last_loss);
    pushChart(accChart,  d.last_accuracy);
    lastChartStep=d.step;
  }

  if(Array.isArray(d.layer_activity)){
    if(Number.isInteger(d.output_classes) && d.output_classes>0){
      LAYERS[LAYERS.length-1]=d.output_classes;
    }
    for(let li=0;li<LAYERS.length;li++){
      nnActivity[li]=compressActivity(d.layer_activity[li],LAYERS[li]);
    }
  }
  drawNN();
  document.getElementById('lp-out').textContent=hasMetrics?d.output_classes:'—';
  document.getElementById('lp-spikes').textContent=hasMetrics?Number(d.last_spike_count).toLocaleString():'—';
  document.getElementById('lp-conns').textContent=hasMetrics?Number(d.connection_count).toLocaleString():'—';
  document.getElementById('data-source').textContent=hasMetrics?(d.data_source||'—'):'—';

  // Logs
  if(d.recent_logs&&d.recent_logs.length>0){
    const box=document.getElementById('logs-terminal');
    const existing=box.querySelectorAll('.log-line');
    if(existing.length===0||existing[0].textContent!==d.recent_logs[0]){
      box.innerHTML=d.recent_logs.slice(0,20).map((l,i)=>
        `<div class="log-line${i===0?' new':''}">${escHtml(l)}</div>`
      ).join('');
    }
  }

  // Training bar
  const tbw=document.getElementById('training-bar-wrap');
  if(d.training_active && d.total_steps>0){
    tbw.style.display='flex';
    const pct=Math.min(100,((d.step||0)/d.total_steps)*100);
    document.getElementById('training-bar-fill').style.width=pct+'%';
    document.getElementById('training-bar-label').textContent=`Batch ${d.step||0} / ${d.total_steps} · Epoch ${d.epoch||0} / ${d.epoch_target}`;
  } else {
    tbw.style.display='none';
  }

  // Show/hide stop button
  document.getElementById('btn-stop').style.display=d.training_active?'block':'none';
}

function escHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function compressActivity(values,count){
  if(!Array.isArray(values)||values.length===0)return Array(count).fill(0);
  return Array.from({length:count},(_,index)=>{
    const start=Math.floor(index*values.length/count);
    const end=Math.max(start+1,Math.floor((index+1)*values.length/count));
    const bucket=values.slice(start,end).map(Number).filter(Number.isFinite);
    return bucket.length?bucket.reduce((sum,value)=>sum+value,0)/bucket.length:0;
  });
}

// ─── SSE connection ───────────────────────────────────────────────────────
function connectSSE(){
  const es=new EventSource('/api/stream');
  es.onmessage=e=>{
    try{applyState(JSON.parse(e.data))}catch(_){}
  };
  es.onerror=()=>{
    es.close();
    setTimeout(connectSSE, 2000);
    // Fallback: poll
    pollOnce();
  };
}

async function pollOnce(){
  try{
    const r=await fetch('/api/state');
    if(r.ok) applyState(await r.json());
  }catch(_){}
}

connectSSE();
// Also poll every 2 s as fallback
setInterval(pollOnce, 2000);

// ─── Sliders ─────────────────────────────────────────────────────────────
const slEpochs=document.getElementById('sl-epochs');
const slLr=document.getElementById('sl-lr');
const slBatch=document.getElementById('sl-batch');

const lrValues=[0.0001,0.0002,0.0003,0.0005,0.0007,0.001,0.002,0.003,0.005,0.007,0.01,0.02,0.03,0.05,0.07,0.1];
const batchMap=[4,8,12,16,24,32,48,64];

slEpochs.addEventListener('input',()=>{
  epochTarget=+slEpochs.value;
  document.getElementById('sv-epochs').textContent=epochTarget;
});
slLr.addEventListener('input',()=>{
  const idx=Math.min(Math.round((+slLr.value/100)*(lrValues.length-1)),lrValues.length-1);
  document.getElementById('sv-lr').textContent=lrValues[idx];
});
slBatch.addEventListener('input',()=>{
  const idx=Math.min(+slBatch.value-1,batchMap.length-1);
  document.getElementById('sv-batch').textContent=batchMap[idx];
});

function getLr(){
  const idx=Math.min(Math.round((+slLr.value/100)*(lrValues.length-1)),lrValues.length-1);
  return lrValues[idx];
}
function getBatch(){
  return batchMap[Math.min(+slBatch.value-1,batchMap.length-1)];
}

// ─── Plan flow ────────────────────────────────────────────────────────────
document.getElementById('btn-plan').addEventListener('click', async ()=>{
  const prompt=document.getElementById('prompt-input').value.trim();
  if(!prompt){
    document.getElementById('prompt-input').style.borderColor='var(--red)';
    setTimeout(()=>document.getElementById('prompt-input').style.borderColor='',1000);
    return;
  }
  document.getElementById('btn-plan').textContent='⏳ TERVEZ...';
  document.getElementById('btn-plan').disabled=true;
  try{
    const r=await fetch('/api/plan',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        prompt,
        epochs:epochTarget,
        lr:getLr(),
        batch_size:getBatch()
      })
    });
    const d=await r.json();
    if(!r.ok||!d.plan)throw new Error(d.detail||'A terv nem készíthető el.');
    currentPlan=d.plan;
    document.getElementById('plan-content').innerHTML=markdownSimple(d.plan);
    document.getElementById('plan-box').style.display='block';
  }catch(e){
    alert('Hiba a tervezés során: '+e);
  }finally{
    document.getElementById('btn-plan').textContent='🔍 TERV';
    document.getElementById('btn-plan').disabled=false;
  }
});

document.getElementById('btn-discard').addEventListener('click',()=>{
  document.getElementById('plan-box').style.display='none';
  currentPlan=null;
});

document.getElementById('btn-approve').addEventListener('click', async ()=>{
  document.getElementById('plan-box').style.display='none';
  await startTraining();
});

document.getElementById('btn-start').addEventListener('click', startTraining);

document.getElementById('btn-stop').addEventListener('click', async ()=>{
  try{
    const r=await fetch('/api/train/stop',{method:'POST'});
    const d=await r.json();
    if(!r.ok||!d.ok)throw new Error(d.message||d.detail||'A leállítás nem sikerült.');
  }catch(e){alert('Hiba a leállítás során: '+e.message);}
});

async function startTraining(){
  const prompt=document.getElementById('prompt-input').value.trim()||'Általános tanítás';
  epochTarget=+slEpochs.value;
  lastChartStep=-1;
  lossHistory=[];accHistory=[];
  lossChart.data.datasets[0].data=[];lossChart.data.labels=[];lossChart.update('none');
  accChart.data.datasets[0].data=[]; accChart.data.labels=[]; accChart.update('none');
  try{
    const r=await fetch('/api/train/start',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        prompt, epochs:epochTarget, lr:getLr(), batch_size:getBatch()
      })
    });
    const d=await r.json();
    if(!r.ok||!d.ok)throw new Error(d.message||d.detail||'A tanítás nem indítható el.');
  }catch(e){alert('Hiba a tanítás indítása során: '+e.message);}
}

function addCatalogCard(container, item, statusLabel){
  const card=document.createElement('article');
  card.className='catalog-card';
  const title=document.createElement('h3');
  title.textContent=item.name;
  const status=document.createElement('span');
  status.className='catalog-status '+item.status;
  status.textContent=statusLabel;
  const description=document.createElement('p');
  description.textContent=item.description||item.focus||'';
  card.append(title,status,description);
  if(item.dataset){
    const dataset=document.createElement('p');
    dataset.textContent='Adat: '+item.dataset;
    card.append(dataset);
  }
  if(item.reason){
    const reason=document.createElement('p');
    reason.textContent=item.reason;
    card.append(reason);
  }
  if(item.prompt){
    const prompt=document.createElement('p');
    prompt.textContent='Prompt: '+item.prompt;
    card.append(prompt);
    const button=document.createElement('button');
    button.type='button';
    button.textContent='Beállítás a vezérlőben';
    button.addEventListener('click',()=>{
      document.getElementById('prompt-input').value=item.prompt;
      setMainView('control');
      document.getElementById('prompt-input').focus();
    });
    card.append(button);
  }
  container.append(card);
}

function renderPromptHelp(catalog){
  const content=document.getElementById('prompt-help-content');
  content.replaceChildren();
  const workflowHeading=document.createElement('h3');
  workflowHeading.textContent='Helyi feladatok és promptok';
  content.append(workflowHeading);
  for(const workflow of catalog.workflows){
    const item=document.createElement('p');
    item.textContent=workflow.name+': '+workflow.prompt_help;
    content.append(item);
  }
  const settingsHeading=document.createElement('h3');
  settingsHeading.textContent='Beállítások';
  const settings=document.createElement('ul');
  for(const text of [
    'EPOCHOK: a tanítási menetek száma.',
    'LR: a tanulási ráta; a promptba nem kell beírni.',
    'BATCH: az egy frissítésben feldolgozott minták száma.',
    'A TERV gomb előnézetet készít; jóváhagyással indul a futás.',
  ]){
    const entry=document.createElement('li');
    entry.textContent=text;
    settings.append(entry);
  }
  content.append(settingsHeading,settings);
  const paradigmHeading=document.createElement('h3');
  paradigmHeading.textContent='Tanítási paradigmák';
  content.append(paradigmHeading);
  for(const paradigm of catalog.paradigms){
    const item=document.createElement('p');
    item.textContent=paradigm.name+' — '+paradigm.description;
    content.append(item);
  }
  const benchmarkHeading=document.createElement('h3');
  benchmarkHeading.textContent='Külső benchmarkok';
  content.append(benchmarkHeading);
  for(const benchmark of catalog.benchmarks){
    const item=document.createElement('p');
    item.textContent=benchmark.name+' — '+benchmark.reason;
    content.append(item);
  }
}

function setMainView(view){
  const showLab=view==='lab';
  document.getElementById('control-view').hidden=showLab;
  document.getElementById('lab-view').hidden=!showLab;
  document.getElementById('view-control').setAttribute('aria-selected',String(!showLab));
  document.getElementById('view-lab').setAttribute('aria-selected',String(showLab));
}

function setLabTab(tab){
  const showRanking=tab==='ranking';
  document.getElementById('tests-panel').hidden=showRanking;
  document.getElementById('ranking-panel').hidden=!showRanking;
  document.getElementById('lab-tests-tab').setAttribute('aria-selected',String(!showRanking));
  document.getElementById('lab-ranking-tab').setAttribute('aria-selected',String(showRanking));
}

function renderRankingRows(rows){
  const body=document.getElementById('ranking-rows');
  body.replaceChildren();
  for(const row of rows){
    const tr=document.createElement('tr');
    for(const value of [row.rank,row.model,row.score,row.updated_at]){
      const td=document.createElement('td');
      td.textContent=value==null?'—':String(value);
      tr.append(td);
    }
    body.append(tr);
  }
  document.getElementById('ranking-empty').hidden=rows.length>0;
}

async function loadLeaderboard(refresh){
  const type=document.getElementById('ranking-type').value;
  const selected=document.getElementById('ranking-type').selectedOptions[0];
  const sourceConfigured=selected&&selected.dataset.sourceConfigured==='true';
  const shouldRefresh=refresh&&sourceConfigured;
  const status=document.getElementById('ranking-status');
  const url=shouldRefresh?'/api/leaderboard/refresh':'/api/leaderboard?type='+encodeURIComponent(type);
  status.textContent='Rangsoradatok lekérése…';
  try{
    const requestOptions=shouldRefresh
      ?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type})}
      :{method:'GET'};
    const response=await fetch(url,requestOptions);
    const data=await response.json();
    if(data.type){
      document.getElementById('ranking-title').textContent=data.name+' — aktuális rangsor';
      document.getElementById('ranking-metric').textContent='Mérőszám: '+data.metric;
      document.getElementById('digitalis-ranking-status').textContent=data.message;
      renderRankingRows(data.rows||[]);
    }
    if(!response.ok){
      status.textContent=data.detail||'A rangsor frissítése nem sikerült.';
      return;
    }
    status.textContent=data.status==='not_configured'
      ?(data.source_status||'Még nincs hivatalos adatforrás beállítva.')
      :'A rangsor frissült.';
  }catch(error){
    status.textContent='Nem sikerült elérni a rangsor API-t: '+error.message;
  }
}

async function loadDashboardCatalog(){
  const status=document.getElementById('catalog-status');
  try{
    const response=await fetch('/api/catalog');
    const catalog=await response.json();
    if(!response.ok)throw new Error(catalog.detail||'A katalógus nem tölthető be.');
    const workflowCards=document.getElementById('workflow-cards');
    workflowCards.replaceChildren();
    for(const workflow of catalog.workflows){
      const statusLabel=workflow.status==='ready'?'HELYBEN FUTTATHATÓ':'KÍSÉRLETI / KORLÁTOZOTT';
      addCatalogCard(workflowCards,workflow,statusLabel);
    }
    const benchmarkCards=document.getElementById('benchmark-cards');
    benchmarkCards.replaceChildren();
    for(const benchmark of catalog.benchmarks){
      const statusLabel=benchmark.status==='external_ranking'?'KÜLSŐ RANGSOR':'NINCS CSATLAKOZTATVA';
      addCatalogCard(benchmarkCards,benchmark,statusLabel);
    }
    const paradigmCards=document.getElementById('paradigm-cards');
    paradigmCards.replaceChildren();
    for(const paradigm of catalog.paradigms){
      const statusLabel={
        supported:'TÁMOGATOTT',
        partial:'RÉSZLEGES',
        not_connected:'NINCS IMPLEMENTÁLVA'
      }[paradigm.status]||paradigm.status;
      addCatalogCard(paradigmCards,paradigm,statusLabel);
    }
    renderPromptHelp(catalog);
    const rankingType=document.getElementById('ranking-type');
    rankingType.replaceChildren();
    for(const ranking of catalog.ranking_types){
      const option=document.createElement('option');
      option.value=ranking.id;
      option.textContent=ranking.name;
      option.dataset.sourceConfigured=String(ranking.source_configured);
      rankingType.append(option);
    }
    status.textContent='A tesztkatalógus betöltve. A külső benchmarkok csak adapter és adatforrás után futtathatók.';
    await loadLeaderboard(true);
  }catch(error){
    status.textContent='Nem sikerült betölteni a tesztkatalógust: '+error.message;
  }
}

document.querySelectorAll('.view-tab').forEach(button=>{
  button.addEventListener('click',()=>setMainView(button.dataset.view));
});
document.querySelectorAll('.lab-tab').forEach(button=>{
  button.addEventListener('click',()=>setLabTab(button.dataset.labTab));
});
document.getElementById('ranking-type').addEventListener('change',()=>loadLeaderboard(false));
document.getElementById('refresh-ranking').addEventListener('click',()=>loadLeaderboard(true));
document.getElementById('btn-prompt-help').addEventListener('click',()=>{
  document.getElementById('prompt-help-dialog').showModal();
});
document.getElementById('close-prompt-help').addEventListener('click',()=>{
  document.getElementById('prompt-help-dialog').close();
});
document.getElementById('prompt-help-dialog').addEventListener('click',event=>{
  if(event.target===event.currentTarget)event.currentTarget.close();
});
loadDashboardCatalog();

// Very simple markdown-ish renderer for plan text
function markdownSimple(md){
  return escHtml(md)
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/^(\d+\.\s.+)$/gm,'<div style="margin:3px 0;padding-left:8px">$1</div>')
    .replace(/\n/g,'<br>');
}

// ─── Layer bar stats ──────────────────────────────────────────────────────
// ─── Window title ─────────────────────────────────────────────────────────
document.getElementById('wt-host').textContent=location.host;
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the cyberpunk dashboard and JSON/SSE API."""

    def log_message(self, format, *args):  # silence access logs
        return

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._html(HTML_TEMPLATE)

        elif path == "/api/state":
            self._json(global_monitor.get_snapshot())

        elif path == "/api/catalog":
            self._json(get_dashboard_catalog())

        elif path == "/api/leaderboard":
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ranking_id = params.get("type", [RANKING_TYPES[0]["id"]])[0]
            ranking = next((item for item in RANKING_TYPES if item["id"] == ranking_id), None)
            if ranking is None:
                self._json({"status": "error", "detail": "Unknown ranking type."}, code=400)
                return
            self._json({
                "type": ranking_id,
                "name": ranking["name"],
                "metric": ranking["metric"],
                "status": "ready" if ranking["source_url"] else "not_configured",
                "rows": [],
                "message": "Még nincs összehasonlítható Digitalis-Agy eredmény.",
                "source_status": (
                    "Még nincs hivatalos adatforrás beállítva."
                    if not ranking["source_url"]
                    else "A hivatalos adatforrás be van állítva."
                ),
            })

        elif path == "/api/stream":
            self._sse_stream()

        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = self._read_body()

        if path == "/api/update":
            try:
                global_monitor.update(json.loads(body))
                self._json({"status": "ok"})
            except Exception as exc:
                self._json({"status": "error", "detail": str(exc)}, code=400)

        elif path == "/api/plan":
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object")
                plan = global_monitor.generate_plan(
                    prompt=data.get("prompt", ""),
                    epochs=data.get("epochs", 50),
                    lr=data.get("lr", 0.001),
                    batch_size=data.get("batch_size", 16),
                )
                self._json({"plan": plan})
            except Exception as exc:
                self._json({"status": "error", "detail": str(exc)}, code=400)

        elif path == "/api/train/start":
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object")
                ok, msg = global_monitor.start_training(
                    prompt=data.get("prompt", ""),
                    epochs=data.get("epochs", 50),
                    lr=data.get("lr", 0.001),
                    batch_size=data.get("batch_size", 16),
                )
                self._json({"ok": ok, "message": msg})
            except Exception as exc:
                self._json({"status": "error", "detail": str(exc)}, code=400)

        elif path == "/api/train/stop":
            ok, msg = global_monitor.stop_training()
            self._json({"ok": ok, "message": msg})

        elif path == "/api/leaderboard/refresh":
            try:
                data = json.loads(body)
                ranking_id = data.get("type")
                ranking = next((item for item in RANKING_TYPES if item["id"] == ranking_id), None)
                if ranking is None:
                    self._json({"status": "error", "detail": "Unknown ranking type."}, code=400)
                elif not ranking["source_url"]:
                    self._json({
                        "status": "not_configured",
                        "type": ranking_id,
                        "name": ranking["name"],
                        "metric": ranking["metric"],
                        "rows": [],
                        "message": "Még nincs összehasonlítható Digitalis-Agy eredmény.",
                        "detail": (
                            "Ehhez a rangsorhoz még nincs hivatalos URL beállítva. "
                            "Frissítés előtt adj meg ellenőrzött forrást."
                        ),
                    }, code=409)
                else:
                    self._json({
                        "status": "not_implemented",
                        "detail": "The source adapter is not configured.",
                    }, code=501)
            except Exception as exc:
                self._json({"status": "error", "detail": str(exc)}, code=400)

        else:
            self.send_response(404)
            self.end_headers()

    # ── SSE ──────────────────────────────────────────────────────────────

    def _sse_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = global_monitor.register_sse_client()
        # Send initial snapshot immediately
        try:
            snap = json.dumps(global_monitor.get_snapshot())
            self.wfile.write(f"data: {snap}\n\n".encode())
            self.wfile.flush()
        except Exception:
            global_monitor.unregister_sse_client(q)
            return

        try:
            while True:
                try:
                    payload = q.get(timeout=15)
                    self.wfile.write(payload.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Send heartbeat keep-alive comment
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            global_monitor.unregister_sse_client(q)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _html(self, content: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _json(self, obj: Any, code: int = 200):
        body = json.dumps(
            obj, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        ).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8") if length else ""


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def start_dashboard_server(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    """Starts the dashboard HTTP server in a background daemon thread."""
    server = ThreadingHTTPServer((host, port), DashboardHTTPRequestHandler)
    server.daemon_threads = True
    server.allow_reuse_address = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
