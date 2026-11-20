"""
Train the neuromorphic cognitive pipeline and report held-out test metrics.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.benchmark import evaluate_classifier
from src.core.dataset import get_neuromorphic_dataloader
from src.core.pipeline import MIN_WORLD_MODEL_TRANSITIONS, NeuromorphicCognitivePipeline
from src.dashboard.monitor import global_monitor
from src.routing.cdr_router import TaskRoutingContext


class SampleWeightedMean:
    def __init__(self) -> None:
        self._weighted_total = 0.0
        self._sample_count = 0

    def add(self, value: float, sample_count: int) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("value must be a finite number")
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        self._weighted_total += float(value) * sample_count
        self._sample_count += sample_count

    @property
    def mean(self) -> float:
        if self._sample_count == 0:
            raise ValueError("cannot calculate a mean without samples")
        return self._weighted_total / self._sample_count


def world_model_training_status(transition_count: int) -> str:
    if type(transition_count) is not int or transition_count < 0:
        raise ValueError("transition_count must be a non-negative integer")
    if transition_count == 0:
        return "NOT TRAINED: no observed action transitions were supplied"
    if transition_count < MIN_WORLD_MODEL_TRANSITIONS:
        return (
            f"PARTIALLY TRAINED: {transition_count} observed transitions; "
            f"planning requires {MIN_WORLD_MODEL_TRANSITIONS}"
        )
    return (
        f"TRAINED: {transition_count} observed transitions; "
        "planning is eligible on slow routes"
    )


def main() -> None:
    default_data_dir = Path(__file__).parent / "data" / "n_mnist"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--samples", type=int, default=160, help="training sample cap")
    parser.add_argument("--test-samples", type=int, default=160, help="test sample cap")
    parser.add_argument("--all-train", action="store_true", help="use every processed training sample")
    parser.add_argument("--all-test", action="store_true", help="evaluate the full processed test split")
    parser.add_argument("--require-n-mnist", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples < 1 or args.test_samples < 1:
        parser.error("sample limits must be positive; use --all-train or --all-test for a full split")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    n_mnist_path = Path(args.data_dir) / "processed_nmnist.pt"
    if (args.all_train or args.all_test or args.require_n_mnist) and not n_mnist_path.is_file():
        raise FileNotFoundError(f"Processed N-MNIST file not found: {n_mnist_path}")

    print("=" * 70)
    print("  NEUROMORPHIC BRAIN EMULATION (DECOLLE HYBRID ARCHITECTURE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Computing Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    input_dim = 64
    time_steps = 16
    num_classes = 10
    train_limit = None if args.all_train else args.samples
    test_limit = None if args.all_test else args.test_samples
    train_loader = get_neuromorphic_dataloader(
        batch_size=args.batch_size,
        num_samples=train_limit,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir=args.data_dir,
        seed=args.seed,
        split="train",
    )
    data_source = train_loader.dataset.data_source
    if args.require_n_mnist and data_source != "N-MNIST":
        raise FileNotFoundError("N-MNIST was required but was not loaded")
    test_loader = get_neuromorphic_dataloader(
        batch_size=args.batch_size,
        num_samples=test_limit,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
        data_dir=args.data_dir,
        seed=args.seed + 1,
        split="test",
    )

    print(f"[*] Data source: {data_source}")
    print(
        f"[*] Samples used: train={len(train_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    print(f"[*] Initializing data loaders (steps: {time_steps})")
    print("[*] Instantiating 6-Pillar Neuromorphic Cognitive Agent...")
    agent = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=num_classes,
        device=device,
        lr=2e-3,
        semantic_memory_path=os.environ.get(
            "SEMANTIC_MEMORY_PATH",
            os.path.join("data", "semantic_memory.json"),
        ),
    )
    agent.start_background_consolidation(
        interval_seconds=float(os.environ.get("SLEEP_CONSOLIDATION_INTERVAL_SECONDS", "3600"))
    )
    print("    [Pillar 1] DECOLLE Local Plasticity LIF Network: INITIALIZED")
    print("    [Pillar 2] Hierarchical Memory & Sleep Engine:   INITIALIZED")
    print("    [Pillar 3] Homeostatic State Vector Monitor:     INITIALIZED")
    print("    [Pillar 4] Predictive Residual Error Module:     INITIALIZED")
    print("    [Pillar 5] Cognitive Decision Router (CDR):      INITIALIZED")
    print(
        "    [Pillar 6] JEPA Latent World Model:             "
        f"{world_model_training_status(agent.world_model_transition_count)}"
    )
    print("-" * 70)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs} Active Cycle")
        epoch_accuracy = SampleWeightedMean()
        epoch_surprise = SampleWeightedMean()
        routes_taken = {"FAST_SYSTEM_1": 0, "SLOW_SYSTEM_2": 0}
        rollouts_executed = 0
        rollout_skip_reasons = {}

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            stats = agent.process_batch(
                inputs,
                targets,
                train_step=True,
                routing_context=TaskRoutingContext(
                    domains=(train_loader.dataset.task,),
                    entity_ids=tuple(
                        f"class_{label}" for label in sorted(set(targets.tolist()))
                    ),
                ),
            )

            cur_v = stats["state_vector"]
            sample_count = targets.shape[0]
            global_monitor.update({
                "curiosity": cur_v[0],
                "error_rate": cur_v[1],
                "resource_stress": cur_v[2],
                "uncertainty": cur_v[3],
                "behavioral_mode": stats["behavioral_mode"],
                "last_route": stats["route"],
                "last_accuracy": stats["accuracy"],
                "last_surprise": stats["surprise"],
                "data_source": data_source,
                "episodic_events_count": len(agent.episodic_mem.retrieve_recent(1000)),
                "semantic_concepts_count": agent.semantic_mem.num_concepts(),
                "log": (
                    f"Epoch {epoch} Batch {batch_idx + 1}: "
                    f"Acc {stats['accuracy'] * 100:.1f}%, Route {stats['route']}"
                ),
            })

            epoch_accuracy.add(stats["accuracy"], sample_count)
            epoch_surprise.add(stats["surprise"], sample_count)
            routes_taken[stats["route"]] += 1
            if stats["rollout_skipped"]:
                skip_reason = stats["rollout_skip_reason"]
                if not isinstance(skip_reason, str) or not skip_reason:
                    raise RuntimeError("Skipped world-model rollout did not include a reason")
                rollout_skip_reasons[skip_reason] = rollout_skip_reasons.get(skip_reason, 0) + 1
            elif stats["planned_action_sequence"] is not None:
                rollouts_executed += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  [Batch {batch_idx + 1:02d}/{len(train_loader):02d}] "
                    f"Acc: {stats['accuracy'] * 100:.1f}% | "
                    f"Surprise: {stats['surprise']:.4f} | "
                    f"Route: {stats['route']} | "
                    f"Mode: {stats['behavioral_mode']} | "
                    f"Homeostasis [C:{cur_v[0]:.2f}, E:{cur_v[1]:.2f}, "
                    f"R:{cur_v[2]:.2f}, U:{cur_v[3]:.2f}]"
                )

        print(
            f"--- Epoch {epoch} Summary: Sample-weighted train accuracy: "
            f"{epoch_accuracy.mean * 100:.2f}%, Sample-weighted surprise: "
            f"{epoch_surprise.mean:.4f}"
        )
        print(
            f"--- Routes: System 1: {routes_taken['FAST_SYSTEM_1']}; "
            f"System 2 requested: {routes_taken['SLOW_SYSTEM_2']}; "
            f"world-model rollouts executed: {rollouts_executed}; "
            f"rollouts skipped: {rollout_skip_reasons}"
        )
        agent.homeostasis.step_decay()

    print(
        "[*] JEPA world-model training status: "
        f"{world_model_training_status(agent.world_model_transition_count)}"
    )
    test_metrics = evaluate_classifier(agent, test_loader)
    print(
        f"[*] Held-out test metrics ({data_source}; "
        f"{int(test_metrics['samples'])} samples): "
        f"loss={test_metrics['loss']:.4f}, accuracy={test_metrics['accuracy'] * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("  ENTERING SLEEP CONSOLIDATION PHASE (NREM/REM Simulation)")
    print("=" * 70)
    consolidation_stats = agent.sleep_and_consolidate()
    agent.stop_background_consolidation()
    print(f"[*] Episodic traces processed: {consolidation_stats['processed_events']}")
    print(f"[*] Concepts integrated into Semantic Graph: {consolidation_stats['consolidated_concepts']}")
    print(f"[*] Relational links established: {consolidation_stats['consolidated_relations']}")
    print(f"[*] Rogue traces rejected (Anti-Identity-Drift Guard): {consolidation_stats['rejected_for_drift']}")
    print(f"[*] Total Semantic Concepts in Graph: {agent.semantic_mem.num_concepts()}")
    print(f"[*] Total Semantic Relations in Graph: {agent.semantic_mem.num_relations()}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Cognitive cycle finished successfully in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
