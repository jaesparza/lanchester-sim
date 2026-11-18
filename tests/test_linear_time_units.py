"""Time-unit changes for the analytical and ODE linear implementations."""

import numpy as np
import pytest

from models import LanchesterLinear
from models.ode_solver_lanchseter_linear import LanchesterLinearODESolver


@pytest.mark.parametrize("scale", [1e10, 1, 1e-18])
@pytest.mark.parametrize(
    "a, b, alpha, beta, winner, end, final",
    [
        (2, 1, 1, 1, "A", 1, (1, 0)),
        (1, 2, 1, 1, "B", 1, (0, 1)),
        (4, 2, 1, 2, "Draw", 2, (0, 0)),
        (2, 4, 0, 2, "B", 1, (0, 4)),
    ],
)
def test_linear_time_units(a, b, alpha, beta, winner, end, final, scale):
    args = (a, b, alpha * scale, beta * scale)
    model = LanchesterLinear(*args)
    solver = LanchesterLinearODESolver(*args)
    horizon = 2 * end / scale

    for result in (model.analytical_solution(t_max=horizon),
                   model.simple_analytical_solution(t_max=horizon),
                   solver.numerical_solution(t_max=horizon)):
        assert result["winner"] == winner
        assert result["remaining_strength"] == pytest.approx(max(final))
        assert result["battle_end_time"] * scale == pytest.approx(end)
        np.testing.assert_allclose((result["A"][-1], result["B"][-1]), final, atol=1e-12)

    times = np.array([0, end / 2, end, 2 * end]) / scale
    solution = solver.solve(sample_times=times)
    assert solution.winner == winner
    assert solution.t_end * scale == pytest.approx(end)
    assert solution.remaining_strength == pytest.approx(max(final))
    elapsed = np.minimum(times * scale, end)
    np.testing.assert_allclose(solution.force_a, a - beta * elapsed, atol=1e-12)
    np.testing.assert_allclose(solution.force_b, b - alpha * elapsed, atol=1e-12)


def test_large_finite_end_time_has_a_finite_default_preview():
    rate = 6e-309
    model = LanchesterLinear(1, 1, rate, rate)
    solver = LanchesterLinearODESolver(1, 1, rate, rate)
    with np.errstate(over="raise", invalid="raise"):
        for result in (model.analytical_solution(), solver.numerical_solution()):
            assert result["battle_end_time"] == 1 / rate
            assert np.isfinite(result["time"]).all()
            assert result["time"][-1] == result["battle_end_time"]
