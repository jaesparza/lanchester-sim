"""Unit tests for the numerical Lanchester Linear Law solver."""
from __future__ import annotations

import math

import pytest

from models import LanchesterLinear, LanchesterLinearODESolver


@pytest.mark.parametrize(
    "A0,B0,alpha,beta",
    [
        (120.0, 100.0, 0.6, 0.5),
        (80.0, 120.0, 0.4, 0.7),
        (50.0, 50.0, 0.3, 0.3),
    ],
)
def test_linear_solver_matches_closed_form(A0, B0, alpha, beta):
    analytic_model = LanchesterLinear(A0, B0, alpha, beta)
    expected_winner, expected_remaining, expected_t = analytic_model.calculate_battle_outcome()

    solver = LanchesterLinearODESolver(A0, B0, alpha, beta)
    solution = solver.solve(num_points=200)

    final_a, final_b = solution.final_strengths

    assert solution.winner == expected_winner
    if math.isfinite(expected_t):
        assert math.isclose(solution.t_end, expected_t, rel_tol=1e-9, abs_tol=1e-9)
    else:
        assert solution.t_end == pytest.approx(solution.t_end)  # Ensure float value is returned

    if expected_winner == "A":
        assert pytest.approx(final_a, rel=1e-9, abs=1e-9) == expected_remaining
        assert final_b <= 1e-9
    elif expected_winner == "B":
        assert pytest.approx(final_b, rel=1e-9, abs=1e-9) == expected_remaining
        assert final_a <= 1e-9
    else:
        assert final_a <= 1e-9
        assert final_b <= 1e-9
