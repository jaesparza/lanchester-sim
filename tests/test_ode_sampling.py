import math

import numpy as np
import pytest

from models.ode_solver_lanchester_square import LanchesterSquareODESolver
from models.ode_solver_lanchseter_linear import LanchesterLinearODESolver


@pytest.mark.parametrize("points", [2, 8, 301])
def test_linear_stops_between_samples(points):
    solver = LanchesterLinearODESolver(3, 1, 1, 1)
    times = np.linspace(0, 3, points)
    solution = solver.solve(sample_times=times)
    summary = solver.numerical_solution(t_max=3, num_points=points)
    assert solution.t_end == summary["battle_end_time"] == 1
    assert solution.remaining_strength == summary["remaining_strength"] == 2
    elapsed = np.minimum(times, 1)
    np.testing.assert_allclose(solution.force_a, 3 - elapsed, atol=1e-12)
    np.testing.assert_allclose(solution.force_b, 1 - elapsed, atol=1e-12)
    np.testing.assert_array_equal(summary["A"], solution.force_a)
    np.testing.assert_array_equal(summary["B"], solution.force_b)


@pytest.mark.parametrize("points", [2, 8, 301])
def test_square_decay_on_requested_grid(points):
    times = np.linspace(0, 3, points)
    solution = LanchesterSquareODESolver(1, 1, 1, 1).solve(sample_times=times)
    np.testing.assert_allclose(solution.force_a, np.exp(-times), rtol=1e-8)
    np.testing.assert_allclose(solution.force_b, np.exp(-times), rtol=1e-8)


def test_square_samples_before_and_after_finite_stop():
    # For initial state (5, 3), the sum decays from 8 and the difference grows from 2.
    times = np.array([0.4, 0.4, 0.67, math.log(2), 3])
    elapsed = np.minimum(times, math.log(2))
    half_sum = 4 * np.exp(-elapsed)
    half_difference = np.exp(elapsed)
    result = LanchesterSquareODESolver(5, 3, 1, 1).solve(sample_times=times)
    np.testing.assert_allclose(result.force_a, half_sum + half_difference, atol=1e-8)
    np.testing.assert_allclose(result.force_b, half_sum - half_difference, atol=1e-8)


def test_linear_empty_state_preserves_requested_samples():
    times = [0.5, 0.5, 3]
    result = LanchesterLinearODESolver(0, 0, 1, 1).solve(sample_times=times)
    np.testing.assert_array_equal(result.time, times)
    np.testing.assert_array_equal(result.force_a, np.zeros(3))
    np.testing.assert_array_equal(result.force_b, np.zeros(3))
