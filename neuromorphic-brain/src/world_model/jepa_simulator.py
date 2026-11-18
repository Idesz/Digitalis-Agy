"""
World Model & Virtual Rollout Simulation (JEPA Architecture).
Simulates future environmental transitions and action evaluations in an abstract latent embedding space
rather than computationally expensive pixel/signal space.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPASimulator(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) simulator.
    Predicts next latent representation given current latent state and action/perturbation:
    s_{t+1} = T(s_t, a_t)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if latent_dim < 1 or action_dim < 1 or hidden_dim < 2:
            raise ValueError("latent_dim and action_dim must be positive and hidden_dim must be at least 2")
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Predictor network in latent space
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Evaluator / Value estimation head (predicts outcome expected reward or stability)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def predict_next_latent(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single transition step in latent space.
        latent_state: [batch_size, latent_dim]
        action: [batch_size, action_dim]
        Returns:
            next_latent: [batch_size, latent_dim] (residual connection for stability)
        """
        if latent_state.ndim != 2 or latent_state.shape[1] != self.latent_dim:
            raise ValueError(f"latent_state must have shape [batch_size, {self.latent_dim}]")
        if action.ndim != 2 or action.shape != (latent_state.shape[0], self.action_dim):
            raise ValueError(f"action must have shape [batch_size, {self.action_dim}]")
        if not torch.isfinite(latent_state).all() or not torch.isfinite(action).all():
            raise ValueError("latent_state and action must contain only finite values")

        combined = torch.cat([latent_state, action], dim=-1)
        delta = self.transition_net(combined)
        next_latent = latent_state + delta
        return next_latent

    def simulate_rollout(
        self,
        initial_latent: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Executes multi-step virtual rollout in latent space.
        initial_latent: [batch_size, latent_dim]
        action_sequence: [batch_size, horizon, action_dim]
        Returns:
            trajectory: list of predicted latent states across horizon
            total_value: accumulated evaluated value across trajectory [batch_size, 1]
        """
        if initial_latent.ndim != 2 or initial_latent.shape[1] != self.latent_dim:
            raise ValueError(f"initial_latent must have shape [batch_size, {self.latent_dim}]")
        if action_sequence.ndim != 3:
            raise ValueError("action_sequence must have shape [batch_size, horizon, action_dim]")
        if action_sequence.shape[1] < 1 or action_sequence.shape[2] != self.action_dim:
            raise ValueError(f"action_sequence must have positive horizon and action_dim {self.action_dim}")
        if action_sequence.shape[0] != initial_latent.shape[0]:
            raise ValueError("initial_latent and action_sequence batch sizes must match")
        if not torch.isfinite(action_sequence).all():
            raise ValueError("action_sequence must contain only finite values")

        batch_size, horizon, _ = action_sequence.shape
        current_state = initial_latent
        trajectory = []
        total_value = initial_latent.new_zeros(batch_size, 1)

        for step in range(horizon):
            action_t = action_sequence[:, step, :]
            current_state = self.predict_next_latent(current_state, action_t)
            trajectory.append(current_state)
            step_val = self.value_head(current_state)
            total_value = total_value + step_val

        return trajectory, total_value

    def simulate_candidates(
        self,
        initial_latent: torch.Tensor,
        candidate_action_sequences: torch.Tensor,
    ) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """
        Evaluates candidate actions shaped [batch, candidates, horizon, action_dim].
        Returns one trajectory per candidate and values shaped [batch, candidates].
        """
        if candidate_action_sequences.ndim != 4:
            raise ValueError(
                "candidate_action_sequences must have shape "
                "[batch_size, candidates, horizon, action_dim]"
            )
        if initial_latent.ndim != 2 or initial_latent.shape[1] != self.latent_dim:
            raise ValueError(f"initial_latent must have shape [batch_size, {self.latent_dim}]")
        if candidate_action_sequences.shape[0] != initial_latent.shape[0]:
            raise ValueError("initial_latent and candidate batch sizes must match")
        if candidate_action_sequences.shape[1] < 1:
            raise ValueError("at least one candidate action sequence is required")

        trajectories = []
        values = []
        for candidate_index in range(candidate_action_sequences.shape[1]):
            trajectory, total_value = self.simulate_rollout(
                initial_latent,
                candidate_action_sequences[:, candidate_index],
            )
            trajectories.append(trajectory)
            values.append(total_value)

        return trajectories, torch.cat(values, dim=1)
