"""
Interactive Dashboard and Monitoring Launcher.
Can run standalone or alongside the agent.
Supports:
1. --web: launches the live web dashboard on http://localhost:8000
2. --tui: renders a terminal-based live refreshing monitor
3. --simulate: runs a pipeline stream using local N-MNIST data or a synthetic fallback
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dashboard.monitor import build_dashboard_telemetry, global_monitor, start_dashboard_server
from src.core.dataset import get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline


def run_simulation(
    server_port: int,
    show_tui: bool = True,
    show_web: bool = True,
    simulate: bool = True,
):
    if not show_web and not show_tui:
        raise ValueError("At least one dashboard output must be enabled.")

    server = start_dashboard_server(host="0.0.0.0", port=server_port) if show_web else None
    if show_web:
        print(f"[*] Starting Neuromorphic Brain Web Dashboard on http://localhost:{server_port}")

    status = "COMPLETED" if simulate else "IDLE"
    try:
        if not simulate:
            global_monitor.update({
                "epoch": 0,
                "epoch_target": 0,
                "step": 0,
                "data_source": "—",
                "last_route": "—",
                "last_loss": 0.0,
                "last_accuracy": 0.0,
                "last_surprise": 0.0,
                "last_spike_count": 0,
                "connection_count": 0,
                "layer_activity": [],
                "working_memory_items": 0,
                "episodic_events_count": 0,
                "semantic_concepts_count": 0,
                "semantic_relations_count": 0,
                "has_metrics": False,
                "training_active": False,
                "training_status": "IDLE",
            })
            while True:
                if show_tui:
                    print("\033[H\033[J", end="")
                    print(global_monitor.render_terminal_ascii())
                    if show_web:
                        print(f"\n🌐 Web UI available at: http://localhost:{server_port} (Press Ctrl+C to stop)")
                time.sleep(1.0)

        print("[*] Instantiating Pipeline for Live Telemetry Stream...")
        agent = NeuromorphicCognitivePipeline(
            input_dim=64,
            hidden_dims=[128, 64],
            num_classes=10,
            semantic_memory_path=os.environ.get(
                "SEMANTIC_MEMORY_PATH",
                os.path.join(os.path.dirname(__file__), "data", "semantic_memory.json"),
            ),
        )
        loader = get_neuromorphic_dataloader(
            batch_size=16,
            num_samples=160,
            data_dir=os.path.join(os.path.dirname(__file__), "data", "n_mnist"),
        )
        global_monitor.update({
            "epoch": 0,
            "epoch_target": 3,
            "total_steps": 3 * len(loader),
            "step": 0,
            "last_loss": 0.0,
            "last_accuracy": 0.0,
            "last_surprise": 0.0,
            "last_spike_count": 0,
            "connection_count": 0,
            "layer_activity": [],
            "has_metrics": False,
            "working_memory_items": 0,
            "episodic_events_count": 0,
            "semantic_concepts_count": 0,
            "semantic_relations_count": 0,
            "last_route": "—",
            "data_source": "—",
            "training_active": True,
            "training_status": "RUNNING",
            "log": f"Training simulation started using {loader.dataset.data_source}.",
        })

        for epoch in range(1, 4):
            for batch_x, batch_y in loader:
                stats = agent.process_batch(batch_x, batch_y, train_step=True)
                telemetry = build_dashboard_telemetry(
                    stats,
                    agent,
                    epoch,
                    3,
                    3 * len(loader),
                    loader.dataset.data_source,
                )
                telemetry["log"] = (
                    f"Batch {stats['step']:02d}: Batch accuracy {stats['accuracy']*100:.1f}%, "
                    f"Loss {stats['loss']:.4f}, Route {stats['route']}, "
                    f"Surprise {stats['surprise']:.4f}"
                )
                global_monitor.update(telemetry)

                if show_tui:
                    # Clear screen and render TUI using ANSI codes (fast and non-blocking)
                    print("\033[H\033[J", end="")
                    print(global_monitor.render_terminal_ascii())
                    if show_web:
                        print(f"\n🌐 Web UI available at: http://localhost:{server_port} (Press Ctrl+C to stop)")

                time.sleep(0.4)

            # Consolidate at end of epoch
            c_stats = agent.sleep_and_consolidate()
            global_monitor.update({
                "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                "semantic_relations_count": agent.semantic_mem.num_relations(),
                "log": f"Sleep cycle completed: {c_stats['consolidated_concepts']} concepts abstracted.",
            })
            time.sleep(1.0)

        if show_web:
            global_monitor.update({"training_active": False, "training_status": "COMPLETED"})
            print(f"[*] Demo complete. Dashboard remains available at http://localhost:{server_port}")
            while True:
                if show_tui:
                    print("\033[H\033[J", end="")
                    print(global_monitor.render_terminal_ascii())
                    print(
                        f"\n🌐 Web UI available at: http://localhost:{server_port} "
                        "(Press Ctrl+C to stop)"
                    )
                time.sleep(1.0)

    except KeyboardInterrupt:
        status = "STOPPED"
        print("\nStopping dashboard monitor.")
    except Exception as exc:
        status = "ERROR"
        global_monitor.update({"log": f"[ERROR] Simulation failed: {exc}"})
        raise
    finally:
        global_monitor.update({"training_active": False, "training_status": status})
        if server:
            server.shutdown()
            server.server_close()


def _resolve_modes(args):
    if args.tui and args.no_tui:
        raise ValueError("--tui and --no-tui cannot be used together.")

    has_display_option = args.web or args.tui or args.no_tui
    show_web = args.web or not args.tui
    show_tui = args.tui or (not args.no_tui and not args.web)
    simulate = args.simulate or args.tui or not args.web

    if not has_display_option:
        show_web = True
        show_tui = True
    if not show_web and not show_tui:
        raise ValueError("Select --web, --tui, or both.")
    return show_web, show_tui, simulate


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Brain Dashboard Monitor")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--no-tui", action="store_true", help="Disable terminal ASCII dashboard rendering")
    parser.add_argument("--tui", action="store_true", help="Select the terminal monitor and live simulation")
    parser.add_argument("--web", action="store_true", help="Select the web dashboard")
    parser.add_argument("--simulate", action="store_true", help="Feed the dashboard from a training simulation")
    args = parser.parse_args()

    try:
        show_web, show_tui, simulate = _resolve_modes(args)
    except ValueError as exc:
        parser.error(str(exc))
    run_simulation(
        server_port=args.port,
        show_tui=show_tui,
        show_web=show_web,
        simulate=simulate,
    )


if __name__ == "__main__":
    main()
