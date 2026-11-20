"""
Orchestrator for the Complete Neuromorphic Cognitive Architecture.
Integrates:
1. DECOLLE Local Plasticity SNN
2. Multi-level Memory (Working, Episodic, Semantic, Sleep Consolidation)
3. Global State Vector & Homeostasis
4. Residual Error Predictive Processing
5. Cognitive Decision Routing (CDR System 1 / System 2)
6. JEPA World Model Simulation
"""

from typing import Dict, Any, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.decolle_layer import DECOLLENetwork
from src.core.synaptic_graph import SynapticGraphManager
from src.memory.working_memory import WorkingMemory
from src.memory.episodic_memory import EpisodicMemory, EpisodicEvent
from src.memory.semantic_memory import SemanticMemory
from src.memory.sleep_consolidation import SleepConsolidator
from src.homeostasis.state_vector import HomeostaticState, BehavioralMode
from src.predictive.residual_error import ResidualPredictiveModule
from src.routing.cdr_router import CDRRouter, SystemRoute, TaskRoutingContext
from src.world_model.jepa_simulator import JEPASimulator


SYNAPTIC_PRUNE_INTERVAL = 100


def _normalized_process_cpu_load(cpu_time_delta: float, wall_time_delta: float) -> float:
    if wall_time_delta <= 0:
        return 0.0
    return float(np.clip(cpu_time_delta / wall_time_delta, 0.0, 1.0))


class NeuromorphicCognitivePipeline:
    """
    End-to-end cognitive agent orchestrator executing perception, local learning,
    routing, memory consolidation, and world-model simulations.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] = None,
        num_classes: int = 10,
        device: torch.device = None,
        lr: float = 1e-3,
        prediction_escalation_threshold: float = 0.5,
        semantic_memory_path: Optional[str] = None,
    ):
        if not 0.0 <= prediction_escalation_threshold <= 1.0:
            raise ValueError("prediction_escalation_threshold must be in [0, 1]")
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prediction_escalation_threshold = prediction_escalation_threshold
        self.step = 0

        # Pillar 1: SNN with DECOLLE local plasticity
        all_layer_dims = [input_dim] + hidden_dims
        self.snn = DECOLLENetwork(layer_sizes=all_layer_dims, num_classes=num_classes).to(self.device)
        self.optimizers = self.snn.get_adamax_optimizers(lr=lr)
        self.synaptic_graph = SynapticGraphManager()

        # Pillar 2: Memory hierarchy
        self.working_mem = WorkingMemory()
        self.episodic_mem = EpisodicMemory(vector_dim=hidden_dims[-1])
        self.semantic_mem = SemanticMemory(storage_path=semantic_memory_path)
        self.sleep_worker = SleepConsolidator(
            episodic_memory=self.episodic_mem,
            semantic_memory=self.semantic_mem,
        )

        # Pillar 3: Homeostasis & State Vector
        self.homeostasis = HomeostaticState()

        # Pillar 4: Predictive Residual Coding
        self.predictive = ResidualPredictiveModule(input_dim=input_dim, latent_dim=hidden_dims[0]).to(self.device)
        self.pred_optimizer = torch.optim.Adam(self.predictive.parameters(), lr=1e-3)

        # Pillar 5: Cognitive Decision Router (CDR)
        self.router = CDRRouter(tau=0.5)

        # Pillar 6: JEPA World Model
        self.world_model = JEPASimulator(
            latent_dim=hidden_dims[-1],
            action_dim=num_classes,
            hidden_dim=128,
        ).to(self.device)
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        self._last_cpu_sample_time = time.process_time()
        self._last_wall_sample_time = time.perf_counter()

    def process_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        train_step: bool = True,
        routing_context: Optional[TaskRoutingContext] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete cognitive cycle over a batch of spatio-temporal inputs:
        inputs: [batch_size, time_steps, input_dim]
        targets: [batch_size]
        """
        if routing_context is not None and not isinstance(routing_context, TaskRoutingContext):
            raise TypeError("routing_context must be a TaskRoutingContext")
        batch_size, time_steps, _ = inputs.shape
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # 1. Predictive Coding & Surprise Evaluation
        # Evaluate mean input pattern over time
        mean_input = inputs.mean(dim=1)  # [batch_size, input_dim]
        pred_loss, mean_surprise, is_plastic = self.predictive.gated_loss(mean_input)

        if train_step and is_plastic:
            self.pred_optimizer.zero_grad()
            pred_loss.backward()
            self.pred_optimizer.step()

        # 2. DECOLLE SNN Local Learning
        layer_spikes, layer_readouts = self.snn.forward_sequence(inputs)
        self._record_synaptic_coactivations(layer_spikes)
        pruned_synaptic_edges = 0
        if (self.step + 1) % SYNAPTIC_PRUNE_INTERVAL == 0:
            pruned_synaptic_edges = self.synaptic_graph.prune_inactive_edges()

        layer_losses = []
        if train_step:
            for layer, spikes, readout, opt in zip(self.snn.layers, layer_spikes, layer_readouts, self.optimizers):
                opt.zero_grad()
                loss = layer.compute_local_loss(readout, targets, spikes, time_steps)
                loss.backward()
                opt.step()
                layer_losses.append(loss.item())

        # Top layer outputs
        final_readout = layer_readouts[-1]
        preds = final_readout.argmax(dim=-1)
        batch_accuracy = (preds == targets).float().mean().item()
        task_error = 1.0 - batch_accuracy
        loss = sum(layer_losses) / len(layer_losses) if layer_losses else float(pred_loss.item())
        layer_activity = [
            inputs.detach().mean(dim=(0, 1)).cpu().tolist(),
            *[spikes.detach().mean(dim=(0, 1)).cpu().tolist() for spikes in layer_spikes],
            torch.softmax(final_readout.detach(), dim=-1).mean(dim=0).cpu().tolist(),
        ]
        spike_count = sum(int(torch.count_nonzero(spikes).item()) for spikes in layer_spikes)
        connection_count = sum(
            layer.linear.weight.numel() + layer.random_readout.weight.numel()
            for layer in self.snn.layers
        )

        # Latent representation from top SNN layer (mean spike trace)
        top_spikes = layer_spikes[-1].mean(dim=1)  # [batch_size, hidden_dims[-1]]
        latent_repr = top_spikes[0].detach().cpu().numpy().tolist()

        if train_step and time_steps > 1:
            observed_latents = layer_spikes[-1][0].detach()
            current_latents = observed_latents[:-1]
            next_latents = observed_latents[1:]
            neutral_actions = current_latents.new_zeros(current_latents.shape[0], self.num_classes)
            predicted_latents = self.world_model.predict_next_latent(current_latents, neutral_actions)
            transition_loss = F.mse_loss(predicted_latents, next_latents)
            stability_targets = torch.exp(-torch.mean(
                (next_latents - current_latents).pow(2),
                dim=-1,
                keepdim=True,
            ))
            stability_loss = F.mse_loss(self.world_model.value_head(current_latents), stability_targets)
            self.wm_optimizer.zero_grad()
            (transition_loss + stability_loss).backward()
            self.wm_optimizer.step()

        # 3. CDR Decision Routing
        # Retrieve similar past memories to compute familiarity / correlation
        similar_episodes = self.episodic_mem.search_similar(latent_repr, limit=5)
        corr_strength = max((score for _, score in similar_episodes), default=0.0)
        domain_crossing, stakeholder_count = self._routing_context_metrics(
            similar_episodes,
            routing_context or TaskRoutingContext(),
        )
        predictive_uncertainty = max(
            float(self.homeostasis.uncertainty),
            float(mean_surprise.item()),
        )

        decision, complexity, metrics = self.router.route(
            correlation_strength=float(corr_strength),
            domain_crossing=domain_crossing,
            stakeholder_count=stakeholder_count,
            uncertainty=predictive_uncertainty,
        )

        # 4. Update homeostasis and apply its behavioral constraints to routing.
        cpu_sample_time = time.process_time()
        wall_sample_time = time.perf_counter()
        process_cpu_load = _normalized_process_cpu_load(
            cpu_sample_time - self._last_cpu_sample_time,
            wall_sample_time - self._last_wall_sample_time,
        )
        self._last_cpu_sample_time = cpu_sample_time
        self._last_wall_sample_time = wall_sample_time
        self.homeostasis.update_observation(
            surprise=float(mean_surprise.item()),
            task_error=task_error,
            system_load=process_cpu_load,
            entropy=max(float(complexity), float(mean_surprise.item())),
        )
        mode = self.homeostasis.get_behavioral_mode()
        prediction_escalated = float(mean_surprise.item()) > self.prediction_escalation_threshold
        if mode == BehavioralMode.CAUTIOUS or prediction_escalated:
            decision = SystemRoute.SLOW

        # 5. Conserve resources by skipping expensive rollouts under high load.
        simulated_rollout_val = None
        selected_world_model_action = None
        rollout_skipped = decision == SystemRoute.SLOW and mode == BehavioralMode.CONSERVATION
        if decision == SystemRoute.SLOW and not rollout_skipped:
            candidate_actions = torch.eye(self.num_classes, device=self.device)
            candidate_actions = candidate_actions[:, None, :].expand(-1, 3, -1)
            candidate_actions = candidate_actions.unsqueeze(0)
            _, candidate_values = self.world_model.simulate_candidates(
                top_spikes[:1].detach(),
                candidate_actions,
            )
            selected_world_model_action = int(candidate_values[0].argmax().item())
            simulated_rollout_val = float(candidate_values[0, selected_world_model_action].item())

        # 6. Working Memory & Episodic Logging
        self.working_mem.put(
            key="current_perceptual_state",
            data={"predicted_class": int(preds[0].item()), "complexity": complexity},
            ttl=60 if mode == BehavioralMode.CONSERVATION else 300,
        )

        # Store in episodic memory
        event = EpisodicEvent(
            embedding=latent_repr,
            context={
                "concept": f"class_{targets[0].item()}",
                "category": "digit",
                "domain": (
                    routing_context.domains[0]
                    if routing_context and routing_context.domains
                    else "classification"
                ),
                "route": decision.value,
                "related_to": f"parity_{targets[0].item() % 2}",
            },
            surprise=float(mean_surprise.item()),
        )
        self.episodic_mem.store_event(event)

        self.step += 1
        return {
            "step": self.step,
            "loss": loss,
            "accuracy": batch_accuracy,
            "pred_loss": pred_loss.item(),
            "surprise": mean_surprise.item(),
            "layer_activity": layer_activity,
            "spike_count": spike_count,
            "connection_count": connection_count,
            "synaptic_graph_nodes": len(self.synaptic_graph.nodes),
            "synaptic_graph_edges": len(self.synaptic_graph.edges),
            "pruned_synaptic_edges": pruned_synaptic_edges,
            "process_cpu_load": process_cpu_load,
            "layer_losses": layer_losses,
            "route": decision.value,
            "complexity": complexity,
            "behavioral_mode": mode.value,
            "state_vector": self.homeostasis.vector,
            "simulated_rollout_val": simulated_rollout_val,
            "rollout_skipped": rollout_skipped,
            "prediction_escalated": prediction_escalated,
            "selected_world_model_action": selected_world_model_action,
        }

    def sleep_and_consolidate(self) -> Dict[str, int]:
        """Runs offline sleep consolidation cycle."""
        return self.sleep_worker.consolidate()

    def start_background_consolidation(self, interval_seconds: float = 3600.0) -> None:
        self.sleep_worker.start_background(interval_seconds)

    def stop_background_consolidation(self, timeout: Optional[float] = None) -> None:
        self.sleep_worker.stop_background(timeout)

    def _record_synaptic_coactivations(self, layer_spikes: list[torch.Tensor]) -> None:
        for layer_index, (source_spikes, target_spikes) in enumerate(
            zip(layer_spikes, layer_spikes[1:])
        ):
            coactivations = (
                (source_spikes > 0).unsqueeze(-1)
                & (target_spikes > 0).unsqueeze(-2)
            ).nonzero(as_tuple=False)
            for _, _, source_neuron, target_neuron in coactivations.detach().cpu().tolist():
                self.synaptic_graph.record_coactivation(
                    source_id=f"layer_{layer_index}:neuron_{source_neuron}",
                    target_id=f"layer_{layer_index + 1}:neuron_{target_neuron}",
                )

    def _routing_context_metrics(
        self,
        similar_episodes: list[tuple[EpisodicEvent, float]],
        context: TaskRoutingContext,
    ) -> tuple[float, int]:
        domains = set(context.domains)
        entities = set(context.entity_ids)
        cross_domain_relations = set()

        for event, _ in similar_episodes:
            event_context = event.context
            event_domain = event_context.get("domain", event_context.get("category"))
            if isinstance(event_domain, str) and event_domain:
                domains.add(event_domain)
            for key in ("concept", "related_to"):
                entity = event_context.get(key)
                if isinstance(entity, str) and entity:
                    entities.add(entity)

        for entity in tuple(entities):
            concept = self.semantic_mem.get_concept(entity)
            if concept is None:
                continue
            concept_domain = concept.get("domain", concept.get("category"))
            if isinstance(concept_domain, str) and concept_domain:
                domains.add(concept_domain)
            subgraph = self.semantic_mem.query_subgraph(entity, depth=1)
            node_domains = {
                node["id"]: node["attrs"].get("domain", node["attrs"].get("category"))
                for node in subgraph["nodes"]
            }
            entities.update(node_domains)
            domains.update(
                domain for domain in node_domains.values()
                if isinstance(domain, str) and domain
            )
            for edge in subgraph["edges"]:
                source_domain = node_domains.get(edge["source"])
                target_domain = node_domains.get(edge["target"])
                if (
                    isinstance(source_domain, str)
                    and isinstance(target_domain, str)
                    and source_domain != target_domain
                ):
                    cross_domain_relations.add(
                        tuple(sorted((source_domain, target_domain)))
                    )

        crossing_units = max(len(domains) - 1, len(cross_domain_relations))
        domain_crossing = min(crossing_units / 4.0, 1.0)
        return domain_crossing, max(1, len(entities))
