"""Checks on trajectories, boundary conditions and reported totals."""

import numpy as np
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

from models import LanchesterLinear, LanchesterSquare, Ship
from models.ode_solver_lanchester_square import LanchesterSquareODESolver


def test_square_law_differential_equation():
    model = LanchesterSquare(100, 80, 0.01, 0.01)
    result = model.analytical_solution(t_max=1)
    t, a, b = result["time"], result["A"], result["B"]
    np.testing.assert_allclose(np.gradient(a, t)[1:-1], -model.beta * b[1:-1], rtol=1e-8)
    np.testing.assert_allclose(np.gradient(b, t)[1:-1], -model.alpha * a[1:-1], rtol=1e-8)


def test_linear_law_differential_equation():
    model = LanchesterLinear(100, 80, 0.5, 0.6)
    result = model.analytical_solution(t_max=1)
    for side, rate in (("A", model.beta), ("B", model.alpha)):
        np.testing.assert_allclose(np.diff(result[side]) / np.diff(result["time"]), -rate)


@pytest.mark.parametrize("model", [LanchesterLinear, LanchesterSquare])
def test_boundary_conditions_and_conservation(model):
    result = model(100, 80, 0.01, 0.01).analytical_solution()
    assert result["time"][0] == 0
    assert result["time"][-1] >= result["battle_end_time"]
    assert result["winner"] == "A"
    for side, initial in (("A", 100), ("B", 80)):
        values = result[side]
        assert values[0] == pytest.approx(initial)
        assert (values >= 0).all()
        assert (np.diff(values) <= 1e-10).all()
        assert values[-1] + result[f"{side}_casualties"] == pytest.approx(initial)
    assert result["B"][-1] == 0
    assert result["A"][-1] == pytest.approx(result["remaining_strength"])


def test_analytical_vs_numerical_solutions():
    model = LanchesterSquare(50, 40, 0.01, 0.01)
    analytical = model.analytical_solution()
    # Compare the actual sampled paths, including the interior of the interval.
    numerical = LanchesterSquareODESolver(50, 40, 0.01, 0.01).solve(sample_times=analytical["time"])
    np.testing.assert_allclose(analytical["A"], numerical.force_a, atol=1e-7)
    np.testing.assert_allclose(analytical["B"], numerical.force_b, atol=1e-7)


def test_ship_health_after_partial_and_excess_damage():
    ship = Ship("A", 10, 0.5, 5)
    assert ship.is_operational()
    assert ship.take_damage(3) == 3
    assert ship.current_health == 2
    assert ship.is_operational()
    assert ship.take_damage(10) == 2
    assert ship.current_health == 0
    assert not ship.is_operational()
