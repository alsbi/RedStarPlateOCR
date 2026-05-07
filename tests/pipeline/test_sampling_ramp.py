"""Tests for P0: scheduled sampling ramp (0.0 → max_prob)."""

from redstar_plate_ocr.pipeline.process_epoch import compute_sampling_prob
from redstar_plate_ocr.pipeline.training_config import TrainingConfig


class TestComputeSamplingProb:
    """Tests for compute_sampling_prob helper."""

    def test_epoch_zero_returns_zero(self) -> None:
        prob = compute_sampling_prob(
            epoch=0,
            max_prob=0.5,
            ramp_epochs=5,
        )
        assert prob == 0.0

    def test_ramp_mid_returns_half(self) -> None:
        prob = compute_sampling_prob(
            epoch=2,
            max_prob=0.5,
            ramp_epochs=5,
        )
        assert abs(prob - 0.2) < 1e-6

    def test_at_ramp_end_returns_max_prob(self) -> None:
        prob = compute_sampling_prob(
            epoch=5,
            max_prob=0.5,
            ramp_epochs=5,
        )
        assert abs(prob - 0.5) < 1e-6

    def test_past_ramp_clamps_to_max(self) -> None:
        prob = compute_sampling_prob(
            epoch=100,
            max_prob=0.5,
            ramp_epochs=5,
        )
        assert prob == 0.5

    def test_zero_ramp_returns_max(self) -> None:
        prob = compute_sampling_prob(
            epoch=0,
            max_prob=0.5,
            ramp_epochs=0,
        )
        assert prob == 0.5

    def test_default_config_ramp_epochs(self) -> None:
        cfg = TrainingConfig()
        assert cfg.scheduled_sampling_ramp_epochs == 5

    def test_old_formula_epoch_1_of_200(self) -> None:
        """Old formula: min(0.3, 1/200) = 0.005.
        New formula: min(0.3, 1/5) = 0.2.
        The ramp gives much faster schedule.
        """
        prob = compute_sampling_prob(
            epoch=1,
            max_prob=0.3,
            ramp_epochs=5,
        )
        assert abs(prob - 0.06) < 1e-6
