"""Risk-aware beam search over discrete latent action rollouts."""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

import torch

from src.world_model.jepa_simulator import JEPASimulator


MAX_PLANNED_ACTIONS = 10
MEMORY_PRIOR_WEIGHT = 0.1


@dataclass(frozen=True)
class PlannedActionSequence:
    action_indices: tuple[int, ...]
    trajectory: tuple[torch.Tensor, ...]
    predicted_value: float
    predicted_risk: float
    score: float


class LatentBeamPlanner:
    """Searches a discrete action model while preserving one branch per first action."""

    def __init__(self, model: JEPASimulator, beam_width: int = 3, risk_weight: float = 0.5):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if not 0.0 <= risk_weight <= 1.0:
            raise ValueError("risk_weight must be in [0, 1]")
        if model.action_dim < 5:
            raise ValueError("the planner requires at least five discrete actions")
        self.model = model
        self.beam_width = beam_width
        self.risk_weight = risk_weight

    def plan_candidates(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 3,
        action_priors: Optional[Sequence[float]] = None,
    ) -> List[PlannedActionSequence]:
        if initial_latent.ndim != 2 or initial_latent.shape[0] != 1:
            raise ValueError("initial_latent must contain exactly one batch state")
        if initial_latent.shape[1] != self.model.latent_dim:
            raise ValueError(f"initial_latent must have shape [1, {self.model.latent_dim}]")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if action_priors is None:
            normalized_priors = [0.0] * self.model.action_dim
        else:
            if len(action_priors) != self.model.action_dim:
                raise ValueError(f"action_priors must contain {self.model.action_dim} values")
            normalized_priors = []
            for prior in action_priors:
                if (
                    isinstance(prior, bool)
                    or not isinstance(prior, (int, float))
                    or not math.isfinite(prior)
                    or not 0.0 <= prior <= 1.0
                ):
                    raise ValueError("action_priors values must be finite numbers in [0, 1]")
                normalized_priors.append(float(prior))

        candidates = []
        with torch.no_grad():
            candidate_count = min(self.model.action_dim, MAX_PLANNED_ACTIONS)
            for first_action in range(candidate_count):
                first_state = self._step(initial_latent, first_action)
                state, value, risk = first_state
                beams = [([first_action], [state], value, risk)]

                for _ in range(1, horizon):
                    expanded = []
                    for actions, trajectory, total_value, total_risk in beams:
                        current_state = trajectory[-1]
                        for action_index in range(candidate_count):
                            next_state, step_value, step_risk = self._step(current_state, action_index)
                            expanded.append((
                                actions + [action_index],
                                trajectory + [next_state],
                                total_value + step_value,
                                total_risk + step_risk,
                            ))
                    expanded.sort(
                        key=lambda candidate: self._score(candidate[2], candidate[3], horizon),
                        reverse=True,
                    )
                    beams = expanded[: self.beam_width]

                actions, trajectory, total_value, total_risk = max(
                    beams,
                    key=lambda candidate: self._score(candidate[2], candidate[3], horizon),
                )
                predicted_value = total_value / horizon
                predicted_risk = total_risk / horizon
                candidates.append(PlannedActionSequence(
                    action_indices=tuple(actions),
                    trajectory=tuple(trajectory),
                    predicted_value=float(predicted_value.item()),
                    predicted_risk=float(predicted_risk.item()),
                    score=(
                        self._score(total_value, total_risk, horizon)
                        + MEMORY_PRIOR_WEIGHT * normalized_priors[first_action]
                    ),
                ))

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def plan(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 3,
        action_priors: Optional[Sequence[float]] = None,
    ) -> PlannedActionSequence:
        return self.plan_candidates(
            initial_latent,
            horizon=horizon,
            action_priors=action_priors,
        )[0]

    def _step(self, state: torch.Tensor, action_index: int):
        action = state.new_zeros(state.shape[0], self.model.action_dim)
        action[:, action_index] = 1.0
        next_state = self.model.predict_next_latent(state, action)
        value = self.model.value_head(next_state).mean()
        risk = self.model.risk_head(next_state).mean()
        return next_state, value, risk

    def _score(self, total_value: torch.Tensor, total_risk: torch.Tensor, horizon: int) -> float:
        return float(((total_value - self.risk_weight * total_risk) / horizon).item())
