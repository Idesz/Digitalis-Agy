import pytest

from train import SampleWeightedMean, world_model_training_status


def test_sample_weighted_mean_accounts_for_uneven_batches():
    mean = SampleWeightedMean()
    mean.add(0.25, 2)
    mean.add(1.0, 1)

    assert mean.mean == pytest.approx(0.5)


def test_sample_weighted_mean_requires_samples():
    mean = SampleWeightedMean()
    with pytest.raises(ValueError, match="without samples"):
        _ = mean.mean
    with pytest.raises(ValueError, match="positive integer"):
        mean.add(0.5, 0)


def test_world_model_status_reflects_observed_training_data():
    assert "NOT TRAINED" in world_model_training_status(0)
    assert "PARTIALLY TRAINED" in world_model_training_status(9)
    assert "planning is eligible" in world_model_training_status(10)

    with pytest.raises(ValueError, match="non-negative integer"):
        world_model_training_status(-1)
