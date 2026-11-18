"""Resource bounds and precision checks. Use pytest --durations=0 to inspect timings."""

import numpy as np
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--durations=0"]))

from models import LanchesterSquare, SalvoCombatModel, Ship


def test_large_sample_grid_respects_requested_size():
    model = LanchesterSquare(100, 80, 0.01, 0.01)
    model.DEFAULT_TIME_POINTS = 100_000
    result = model.analytical_solution()
    for field in ("time", "A", "B"):
        assert result[field].shape == (100_000,)
        assert np.isfinite(result[field]).all()


@pytest.mark.parametrize(
    "a, b, rate",
    [
        (1e6, 8e5, 0.01),
        (1.0000000001, 1, 0.01),
        (1e15, 1e15 * 0.999999999, 0.01),
        (100, 80, 1e-10),
        (100, 50, 1e6),
        (100, 99, 1e-6),
    ],
    ids=["large-input", "small-difference", "large-close-inputs", "long-horizon", "fast", "slow"],
)
def test_default_grid_bounds_and_precision(a, b, rate):
    model = LanchesterSquare(a, b, rate, rate)
    result = model.analytical_solution()
    assert result["winner"] == "A"
    assert result["remaining_strength"] > 0
    for field in ("time", "A", "B"):
        assert result[field].shape == (model.DEFAULT_TIME_POINTS,)
        assert np.isfinite(result[field]).all()


@pytest.mark.parametrize(
    "size, params, limit",
    [(200, (5, 0.3, 2), 3), (1, (1, 0.9, 100), 20), (1, (1, 0.99, 5), 10)],
    ids=["large-fleet", "long-lived", "low-activity"],
)
def test_round_log_growth_is_bounded(size, params, limit):
    side_a = [Ship(f"A{i}", *params) for i in range(size)]
    side_b = [Ship(f"B{i}", *params) for i in range(size)]
    model = SalvoCombatModel(side_a, side_b, random_seed=42)
    model.run_simulation(max_rounds=limit, quiet=True)
    assert 0 < len(model.battle_log) == model.round_number <= limit


def test_low_activity_ensemble_finishes_within_round_limit():
    model = SalvoCombatModel([Ship("A", 0.1, 0.95, 10)], [Ship("B", 0.1, 0.95, 10)], random_seed=42)
    analysis = model.run_monte_carlo_analysis(iterations=5)
    assert len(analysis["battle_durations"]) == 5
    assert all(0 < duration <= model.DEFAULT_MAX_ROUNDS for duration in analysis["battle_durations"])
