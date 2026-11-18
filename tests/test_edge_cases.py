"""Boundary and scale checks for the three public models."""

import numpy as np
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship


@pytest.mark.parametrize(
    "initial, rate, winner, remaining, end_time",
    [
        ((0, 100), 0.01, "B", 100, 0),
        ((100, 0), 0.01, "A", 100, 0),
        ((1e10, 1e9), 0.01, "A", 9e9, 1e11),
        ((100, 80), 1e-10, "A", 20, 8e11),
        ((100, 80), 1e6, "A", 20, 8e-5),
    ],
    ids=["empty-a", "empty-b", "large-initial-values", "small-rate", "large-rate"],
)
def test_linear_edge_cases(initial, rate, winner, remaining, end_time):
    result = LanchesterLinear(*initial, rate, rate).analytical_solution()
    assert result["winner"] == winner
    assert result["remaining_strength"] == pytest.approx(remaining)
    assert result["battle_end_time"] == pytest.approx(end_time)
    for side in ("A", "B"):
        assert np.isfinite(result[side]).all()
        assert result[side][-1] == pytest.approx(remaining if side == winner else 0)


@pytest.mark.parametrize(
    "a, b, alpha, beta, winner",
    [
        (100, 100, 0.01, 0.01, "Draw"),
        (100.001, 100, 0.01, 0.01, "A"),
        (100, 80, 0, 0.01, "B"),
        (100, 80, 0.01, 0, "A"),
        (100, 80, 0, 0, "Draw"),
        (0.001, 0.0005, 0.01, 0.01, "A"),
        (1000, 999.9, 0.01, 0.01, "A"),
        (100, 99.999, 1e-6, 1e-6, "A"),
        (1000, 100, 1e6, 1e6, "A"),
        (1e10, 1e10 - 1, 0.01, 0.01, "A"),
        (1e-10, 1e-10 - 1e-15, 0.01, 0.01, "A"),
    ],
    ids=["balanced", "near-balanced", "zero-alpha", "zero-beta", "static",
         "small-values", "near-unit-atanh", "slow", "fast", "large-close", "small-close"],
)
def test_square_edge_cases(a, b, alpha, beta, winner):
    result = LanchesterSquare(a, b, alpha, beta).analytical_solution()
    assert result["winner"] == winner
    assert np.isinf(result["battle_end_time"]) == (winner == "Draw")
    for side, initial in (("A", a), ("B", b)):
        values = result[side]
        assert np.isfinite(values).all()
        assert (values >= 0).all()
        assert values[0] == pytest.approx(initial, rel=1e-12, abs=0)
        if alpha == beta == 0:
            np.testing.assert_array_equal(values, initial)


@pytest.mark.parametrize(
    "a, b, size, limit",
    [
        ((0, 0.5, 3), (10, 0.3, 2), 1, 10),
        ((15, 0, 1), (5, 0.9, 10), 1, 10),
        ((10, 1.0, 5), (10, 0.3, 5), 1, 10),
        ((8, 0.3, 1), (8, 0.3, 1), 1, 10),
        ((1, 0.1, 1), (1, 0.1, 1), 100, 5),
    ],
    ids=["zero-offense", "zero-defense", "maximum-defense", "single-hit", "many-ships"],
)
def test_salvo_edge_cases(a, b, size, limit):
    side_a = [Ship(f"A{i}", *a) for i in range(size)]
    side_b = [Ship(f"B{i}", *b) for i in range(size)]
    model = SalvoCombatModel(side_a, side_b, random_seed=42)
    outcome = model.run_simulation(max_rounds=limit, quiet=True)
    stats = model.get_battle_statistics()
    assert 0 < stats["rounds"] <= limit
    assert stats["outcome"] == outcome
    assert stats["force_a_survivors"] == sum(ship.is_operational() for ship in side_a)
    assert stats["force_b_survivors"] == sum(ship.is_operational() for ship in side_b)


def test_monte_carlo_counts_all_iterations():
    model = SalvoCombatModel([Ship("A", 100, 0.95, 20)], [Ship("B", 100, 0.95, 20)], random_seed=42)
    analysis = model.run_monte_carlo_analysis(iterations=10)
    assert sum(analysis["outcome_counts"].values()) == analysis["iterations"] == 10
    assert len(analysis["battle_durations"]) == 10
