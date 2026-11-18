import numpy as np
import pytest

from models import LanchesterSquare
from models.ode_solver_lanchester_square import LanchesterSquareODESolver


@pytest.mark.parametrize("a0, b0, winner", [(2.5, 0, "A"), (0, 2.5, "B"), (0, 0, "Draw")])
@pytest.mark.parametrize("alpha, beta", [(0, 1), (1, 0), (0, 0), (1, 1)])
def test_square_empty_state_is_terminal(a0, b0, winner, alpha, beta):
    analytical = LanchesterSquare(a0, b0, alpha, beta)
    numerical = LanchesterSquareODESolver(a0, b0, alpha, beta)
    remaining = max(a0, b0)
    invariant = alpha * a0**2 - beta * b0**2
    times = np.array([0, 1, 3])

    for model in (analytical, numerical):
        outcome = model.calculate_battle_outcome()
        assert outcome == (winner, remaining, invariant)
        assert model.calculate_battle_end_time(*outcome) == 0

    a, b = analytical.generate_force_trajectories(winner, remaining, 0, times, invariant)
    np.testing.assert_array_equal(a, np.full(times.size, a0))
    np.testing.assert_array_equal(b, np.full(times.size, b0))

    samples = numerical.solve(sample_times=times)
    assert (samples.winner, samples.remaining_strength, samples.t_end) == (winner, remaining, 0)
    np.testing.assert_array_equal(samples.time, times)
    np.testing.assert_array_equal(samples.force_a, a)
    np.testing.assert_array_equal(samples.force_b, b)

    for solve in (analytical.analytical_solution, analytical.simple_analytical_solution,
                  numerical.numerical_solution):
        result = solve(t_max=3)
        assert result["winner"] == winner
        assert result["remaining_strength"] == remaining
        assert result["battle_end_time"] == result["A_casualties"] == result["B_casualties"] == 0
        assert result["invariant"] == invariant
        assert np.all(result["A"] == a0)
        assert np.all(result["B"] == b0)
