"""Unit tests for the numerical Lanchester Square Law solver."""
from __future__ import annotations

import math

import pytest

from models import LanchesterSquare, LanchesterSquareODESolver


@pytest.mark.parametrize(
    "A0,B0,alpha,beta",
    [
        (150.0, 110.0, 0.8, 0.6),
        (90.0, 140.0, 0.5, 0.7),
        (200.0, 150.0, 1.2, 0.9),
    ],
)
def test_square_solver_matches_invariant(A0, B0, alpha, beta):
    analytic_model = LanchesterSquare(A0, B0, alpha, beta)
    expected_winner, expected_remaining, invariant = analytic_model.calculate_battle_outcome()
    expected_time = analytic_model.calculate_battle_end_time(
        expected_winner, expected_remaining, invariant
    )

    solver = LanchesterSquareODESolver(A0, B0, alpha, beta)
    solution = solver.solve(num_points=600)

    final_a, final_b = solution.final_strengths

    assert solution.winner == expected_winner
    if math.isfinite(expected_time):
        assert math.isclose(solution.t_end, expected_time, rel_tol=2e-3, abs_tol=2e-3)
    else:
        assert math.isinf(solution.t_end)

    if expected_winner == "A":
        assert pytest.approx(final_a, rel=2e-3, abs=2e-3) == expected_remaining
        assert final_b <= 1e-3
    elif expected_winner == "B":
        assert pytest.approx(final_b, rel=2e-3, abs=2e-3) == expected_remaining
        assert final_a <= 1e-3
    else:
        assert final_a <= 1e-3
        assert final_b <= 1e-3
