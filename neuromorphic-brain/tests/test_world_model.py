import pytest
import torch
from src.world_model.jepa_simulator import JEPASimulator
from src.world_model.planner import LatentBeamPlanner


def test_jepa_single_step():
    batch_size = 4
    latent_dim = 16
    action_dim = 4

    jepa = JEPASimulator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32)
    s_t = torch.randn(batch_size, latent_dim)
    a_t = torch.randn(batch_size, action_dim)

    s_next = jepa.predict_next_latent(s_t, a_t)
    assert s_next.shape == (batch_size, latent_dim)


def test_jepa_virtual_rollout():
    batch_size = 2
    latent_dim = 16
    action_dim = 4
    horizon = 5

    jepa = JEPASimulator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32)
    s_0 = torch.randn(batch_size, latent_dim)
    action_seq = torch.randn(batch_size, horizon, action_dim)

    trajectory, total_value = jepa.simulate_rollout(s_0, action_seq)

    assert len(trajectory) == horizon
    for state in trajectory:
        assert state.shape == (batch_size, latent_dim)
    assert total_value.shape == (batch_size, 1)

    loss = total_value.sum() + jepa.estimate_trajectory_risk(trajectory).sum()
    loss.backward()
    for param in jepa.parameters():
        assert param.grad is not None


def test_jepa_simulates_and_scores_multiple_candidates():
    batch_size = 2
    latent_dim = 4
    action_dim = 3
    candidates = 5
    horizon = 2
    model = JEPASimulator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=8)
    initial = torch.zeros(batch_size, latent_dim)
    actions = torch.randn(batch_size, candidates, horizon, action_dim)

    trajectories, values = model.simulate_candidates(initial, actions)

    assert len(trajectories) == candidates
    assert all(len(trajectory) == horizon for trajectory in trajectories)
    assert values.shape == (batch_size, candidates)
    assert torch.all((values >= 0.0) & (values <= horizon))
    risks = model.estimate_trajectory_risk(trajectories[0])
    assert risks.shape == (batch_size, 1)
    assert torch.all((risks >= 0.0) & (risks <= 1.0))


def test_latent_beam_planner_returns_distinct_first_actions_and_risk_scores():
    model = JEPASimulator(latent_dim=4, action_dim=5, hidden_dim=8)
    planner = LatentBeamPlanner(model, beam_width=2, risk_weight=0.25)

    candidates = planner.plan_candidates(torch.zeros(1, 4), horizon=3)

    assert len(candidates) == 5
    assert {candidate.action_indices[0] for candidate in candidates} == set(range(5))
    assert all(len(candidate.trajectory) == 3 for candidate in candidates)
    assert all(0.0 <= candidate.predicted_risk <= 1.0 for candidate in candidates)
    assert candidates[0].score >= candidates[-1].score


def test_latent_beam_planner_uses_memory_prior_for_first_action(monkeypatch):
    model = JEPASimulator(latent_dim=4, action_dim=5, hidden_dim=8)
    planner = LatentBeamPlanner(model, beam_width=1)
    monkeypatch.setattr(
        planner,
        "_step",
        lambda state, action: (state, torch.tensor(0.5), torch.tensor(0.0)),
    )

    candidates = planner.plan_candidates(
        torch.zeros(1, 4),
        horizon=1,
        action_priors=[0.0, 0.0, 1.0, 0.0, 0.0],
    )

    assert candidates[0].action_indices == (2,)
    assert candidates[0].score - candidates[1].score == pytest.approx(0.1)


@pytest.mark.parametrize(
    "priors",
    [
        [0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, float("nan")],
        [0.0, 0.0, 0.0, 0.0, 1.1],
        [0.0, 0.0, 0.0, 0.0, True],
    ],
)
def test_latent_beam_planner_rejects_invalid_memory_priors(priors):
    planner = LatentBeamPlanner(
        JEPASimulator(latent_dim=4, action_dim=5, hidden_dim=8),
    )

    with pytest.raises(ValueError, match="action_priors"):
        planner.plan_candidates(torch.zeros(1, 4), action_priors=priors)


@pytest.mark.parametrize(
    ("initial", "actions"),
    [
        (torch.zeros(1, 4), torch.zeros(2, 2, 3)),
        (torch.zeros(1, 5), torch.zeros(1, 3)),
        (torch.zeros(1, 4), torch.zeros(1, 2, 2)),
    ],
)
def test_jepa_rejects_invalid_rollout_shapes(initial, actions):
    model = JEPASimulator(latent_dim=4, action_dim=3, hidden_dim=8)
    with pytest.raises(ValueError):
        model.simulate_rollout(initial, actions)
