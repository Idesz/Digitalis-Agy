import pytest
import torch
from src.routing.cdr_router import CDRRouter, SystemRoute, TaskRoutingContext
from src.routing.uncertainty import RollingUncertaintyCalibrator, representation_drift


def test_task_routing_context_normalizes_and_validates_fields():
    context = TaskRoutingContext(domains=["math", "math"], entity_ids=("a", "b", "a"))

    assert context.domains == ("math",)
    assert context.entity_ids == ("a", "b")
    with pytest.raises(ValueError, match="domains"):
        TaskRoutingContext(domains="math")
    with pytest.raises(ValueError, match="entity_ids"):
        TaskRoutingContext(entity_ids=("entity", ""))


def test_cdr_routing_fast_path():
    router = CDRRouter(tau=0.5)
    # Familiar query: high correlation, low domain crossing, low stakeholders, low uncertainty
    decision, score, metrics = router.route(
        correlation_strength=0.95,
        domain_crossing=0.1,
        stakeholder_count=1,
        uncertainty=0.1,
    )
    assert decision == SystemRoute.FAST
    assert score < router.tau
    assert metrics["complexity_score"] == score


def test_cdr_routing_slow_path():
    router = CDRRouter(tau=0.5)
    # Complex query: unfamiliar, high domain crossing, many stakeholders, high uncertainty
    decision, score, metrics = router.route(
        correlation_strength=0.1,
        domain_crossing=0.9,
        stakeholder_count=8,
        uncertainty=0.85,
    )
    assert decision == SystemRoute.SLOW
    assert score >= router.tau


def test_cdr_weight_normalization():
    router = CDRRouter(
        tau=0.4,
        w_correlation=1.0,
        w_domain=1.0,
        w_stakeholders=1.0,
        w_uncertainty=1.0,
    )
    total = router.w_correlation + router.w_domain + router.w_stakeholders + router.w_uncertainty
    assert abs(total - 1.0) < 1e-5


@pytest.mark.parametrize(
    "options",
    [
        {"tau": float("nan")},
        {"tau": 1.1},
        {"w_correlation": -1.0},
        {"w_correlation": 0.0, "w_domain": 0.0, "w_stakeholders": 0.0, "w_uncertainty": 0.0},
    ],
)
def test_cdr_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        CDRRouter(**options)


@pytest.mark.parametrize(
    "overrides",
    [
        {"correlation_strength": float("nan")},
        {"domain_crossing": -0.1},
        {"uncertainty": 1.1},
        {"stakeholder_count": -1},
        {"stakeholder_count": 1.5},
    ],
)
def test_cdr_rejects_invalid_routing_dimensions(overrides):
    inputs = {
        "correlation_strength": 0.5,
        "domain_crossing": 0.2,
        "stakeholder_count": 1,
        "uncertainty": 0.1,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError):
        CDRRouter().route(**inputs)


def test_rolling_uncertainty_calibrator_uses_ties_and_bounded_history():
    calibrator = RollingUncertaintyCalibrator(window_size=3, min_samples=2)

    assert calibrator.update(0.2) == (0.2, False)
    assert calibrator.update(0.2) == (0.2, False)
    assert calibrator.update(0.2) == (0.5, True)
    assert calibrator.update(0.9) == (0.875, True)
    assert calibrator.update(0.0) == (0.125, True)
    assert calibrator.update(0.95) == (0.875, True)


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1, True])
def test_rolling_uncertainty_calibrator_rejects_invalid_scores(score):
    with pytest.raises(ValueError, match="score"):
        RollingUncertaintyCalibrator().update(score)


def test_representation_drift_measures_normalized_rate_change():
    previous = torch.tensor([1.0, 0.0, 0.0])

    assert representation_drift(None, previous) == 0.0
    assert representation_drift(previous, previous) == 0.0
    assert representation_drift(torch.ones(3), torch.full((3,), 0.5)) == 0.5
    assert representation_drift(previous, previous.flip(0)) == pytest.approx(2.0 / 3.0)
