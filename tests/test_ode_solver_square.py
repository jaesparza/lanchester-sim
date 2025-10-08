"""
Unit tests for LanchesterSquareODESolver - numerical integration of the square law.

Mirrors coverage of the linear-law ODE tests: constructor validation, integration
behavior, consistency with analytical solutions, edge cases, and plotting.
"""

import unittest
from math import isfinite

import numpy as np

from models.ode_solver_lanchester_square import (
    LanchesterSquareODESolver,
    SquareODESolution,
)
from models import LanchesterSquare


class TestLanchesterSquareODESolver(unittest.TestCase):
    """Test cases for the square-law ODE solver."""

    def setUp(self):
        self.solver_a_wins = LanchesterSquareODESolver(A0=120, B0=80, alpha=0.02, beta=0.02)
        self.analytical_a_wins = LanchesterSquare(A0=120, B0=80, alpha=0.02, beta=0.02)

        self.solver_b_wins = LanchesterSquareODESolver(A0=80, B0=120, alpha=0.01, beta=0.02)
        self.analytical_b_wins = LanchesterSquare(A0=80, B0=120, alpha=0.01, beta=0.02)

        # Draw scenario: alpha*A0^2 == beta*B0^2
        self.solver_draw = LanchesterSquareODESolver(A0=100, B0=50, alpha=1.0, beta=4.0)
        self.analytical_draw = LanchesterSquare(A0=100, B0=50, alpha=1.0, beta=4.0)

        # Superior effectiveness case
        self.solver_superior = LanchesterSquareODESolver(A0=60, B0=110, alpha=0.06, beta=0.01)
        self.analytical_superior = LanchesterSquare(A0=60, B0=110, alpha=0.06, beta=0.01)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            LanchesterSquareODESolver(A0=-1, B0=50, alpha=0.01, beta=0.01)
        with self.assertRaises(ValueError):
            LanchesterSquareODESolver(A0=50, B0=-1, alpha=0.01, beta=0.01)
        with self.assertRaises(ValueError):
            LanchesterSquareODESolver(A0=50, B0=50, alpha=-0.01, beta=0.01)
        with self.assertRaises(ValueError):
            LanchesterSquareODESolver(A0=50, B0=50, alpha=0.01, beta=-0.01)

        solver_zero = LanchesterSquareODESolver(A0=0, B0=50, alpha=0, beta=0.02)
        self.assertEqual(solver_zero.A0, 0.0)
        self.assertEqual(solver_zero.alpha, 0.0)

    def test_rhs_function(self):
        solver = LanchesterSquareODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        y = np.array([50.0, 40.0])
        rhs = solver._rhs(0.0, y)
        expected = np.array([-0.6 * 40.0, -0.5 * 50.0])
        np.testing.assert_array_almost_equal(rhs, expected)

    def test_rk4_step(self):
        solver = LanchesterSquareODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        y0 = np.array([100.0, 80.0])
        dt = 0.1
        y1 = solver._rk4_step(0.0, y0, dt)

        # Expect close to first-order decay for small dt
        expected = np.array([
            100.0 - 0.6 * 80.0 * dt,
            80.0 - 0.5 * 100.0 * dt,
        ])
        np.testing.assert_array_almost_equal(y1, expected, decimal=1)

        y_same = solver._rk4_step(0.0, y0, 0.0)
        np.testing.assert_array_equal(y_same, y0)

    def test_prepare_time_array(self):
        solver = self.solver_a_wins
        winner, remaining, invariant = solver.calculate_battle_outcome()
        t_end = solver.calculate_battle_end_time(winner, remaining, invariant)
        times = solver._prepare_time_array(t_end=t_end, num_points=50)

        self.assertEqual(times[0], 0.0)
        self.assertGreaterEqual(times[-1], t_end)
        self.assertEqual(len(times), 50)

        # Infinite t_end uses preview window
        zero_solver = LanchesterSquareODESolver(A0=100, B0=80, alpha=0.0, beta=0.0)
        zero_times = zero_solver._prepare_time_array(t_end=float("inf"), num_points=10)
        self.assertAlmostEqual(zero_times[-1], zero_solver.DRAW_PREVIEW_WINDOW)

        # Custom sample times pass-through
        custom = np.array([0.0, 1.0, 3.0])
        custom_times = solver._prepare_time_array(t_end=t_end, num_points=10, sample_times=custom)
        np.testing.assert_array_equal(custom_times, custom)

        with self.assertRaises(ValueError):
            solver._prepare_time_array(t_end=t_end, num_points=10, sample_times=np.array([0.0, -1.0]))
        with self.assertRaises(ValueError):
            solver._prepare_time_array(t_end=t_end, num_points=10, sample_times=np.array([-1.0, 0.0]))

    def test_solve_basic_functionality(self):
        solver = self.solver_a_wins
        solution = solver.solve()

        self.assertIsInstance(solution, SquareODESolution)
        self.assertEqual(len(solution.time), len(solution.force_a))
        self.assertEqual(len(solution.time), len(solution.force_b))
        self.assertEqual(solution.time[0], 0.0)
        self.assertAlmostEqual(solution.force_a[0], solver.A0)
        self.assertAlmostEqual(solution.force_b[0], solver.B0)
        self.assertTrue(np.all(solution.force_a >= 0))
        self.assertTrue(np.all(solution.force_b >= 0))

    def test_solve_with_custom_samples(self):
        solver = self.solver_a_wins
        custom_times = np.linspace(0.0, 10.0, 21)
        solution = solver.solve(sample_times=custom_times)
        np.testing.assert_array_equal(solution.time, custom_times)
        self.assertLess(solution.force_a[1], solver.A0)

    def test_solve_with_late_start_samples(self):
        solver = self.solver_a_wins
        winner, remaining, invariant = solver.calculate_battle_outcome()
        analytical_t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

        start_time = analytical_t_end * 0.25
        custom_times = np.linspace(start_time, analytical_t_end * 0.75, 6)
        solution = solver.solve(sample_times=custom_times)

        np.testing.assert_array_equal(solution.time, custom_times)
        self.assertLess(solution.force_a[0], solver.A0)

    def test_t_end_matches_analytical_with_truncated_samples(self):
        solver = self.solver_a_wins
        winner, remaining, invariant = solver.calculate_battle_outcome()
        analytical_t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

        custom_times = np.linspace(0.0, analytical_t_end * 0.5, 5)
        solution = solver.solve(sample_times=custom_times)

        self.assertAlmostEqual(solution.t_end, analytical_t_end, places=6)

    def test_winner_determination(self):
        cases = [
            (self.solver_a_wins, 'A'),
            (self.solver_b_wins, 'B'),
            (self.solver_draw, 'Draw'),
            (self.solver_superior, 'A'),
        ]
        for solver, expected in cases:
            with self.subTest(expected=expected):
                solution = solver.solve()
                self.assertEqual(solution.winner, expected)
                self.assertGreaterEqual(solution.remaining_strength, 0.0)

    def test_final_strengths_property(self):
        solver = self.solver_b_wins
        solution = solver.solve()
        final_a, final_b = solution.final_strengths
        self.assertAlmostEqual(final_a, solution.force_a[-1])
        self.assertAlmostEqual(final_b, solution.force_b[-1])
        self.assertIsInstance(final_a, float)
        self.assertIsInstance(final_b, float)

    def test_calculate_battle_outcome_consistency(self):
        solvers = [self.solver_a_wins, self.solver_b_wins, self.solver_draw, self.solver_superior]
        for solver in solvers:
            with self.subTest(solver=solver):
                solution = solver.solve()
                winner, remaining, invariant = solver.calculate_battle_outcome()
                t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

                self.assertEqual(solution.winner, winner)
                self.assertAlmostEqual(solution.remaining_strength, remaining, places=1)
                if isfinite(solution.t_end) and isfinite(t_end):
                    self.assertAlmostEqual(solution.t_end, t_end, places=1)

    def test_generate_force_trajectories(self):
        solver = self.solver_superior
        t = np.linspace(0.0, 5.0, 11)
        force_a, force_b = solver.generate_force_trajectories(t)
        self.assertEqual(len(force_a), len(t))
        self.assertEqual(len(force_b), len(t))
        self.assertAlmostEqual(force_a[0], solver.A0)
        self.assertAlmostEqual(force_b[0], solver.B0)

        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([[0.0, 1.0], [2.0, 3.0]]))
        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([2.0, 1.0, 3.0]))

    def test_numerical_solution_format(self):
        solver = self.solver_a_wins
        result = solver.numerical_solution(num_points=200)

        required_keys = {
            'time', 'A', 'B', 'battle_end_time', 'winner',
            'remaining_strength', 'A_casualties', 'B_casualties', 'invariant'
        }
        self.assertTrue(required_keys.issubset(result.keys()))
        self.assertEqual(result['time'][0], 0.0)
        self.assertEqual(len(result['time']), len(result['A']))
        self.assertEqual(len(result['time']), len(result['B']))
        expected_invariant = solver.alpha * solver.A0**2 - solver.beta * solver.B0**2
        self.assertAlmostEqual(result['invariant'], expected_invariant, places=6)

    def test_invariant_preservation(self):
        solver = self.solver_a_wins
        solution = solver.solve(num_points=500)
        invariant_series = solver.alpha * solution.force_a**2 - solver.beta * solution.force_b**2
        self.assertTrue(np.allclose(invariant_series, invariant_series[0], atol=1e-1))

    def test_casualty_calculation(self):
        solver = self.solver_a_wins
        result = solver.numerical_solution()
        if result['winner'] == 'A':
            expected_a = solver.A0 - result['remaining_strength']
            expected_b = solver.B0
        elif result['winner'] == 'B':
            expected_a = solver.A0
            expected_b = solver.B0 - result['remaining_strength']
        else:
            if isfinite(result['battle_end_time']):
                expected_a = solver.A0
                expected_b = solver.B0
            else:
                expected_a = 0.0
                expected_b = 0.0
        self.assertAlmostEqual(result['A_casualties'], expected_a, places=3)
        self.assertAlmostEqual(result['B_casualties'], expected_b, places=3)

    def test_plot_battle_functionality(self):
        import matplotlib
        matplotlib.use('Agg')

        solver = self.solver_b_wins
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('matplotlib unavailable')

        fig, ax = plt.subplots()
        solver.plot_battle(ax=ax)
        plt.close(fig)

        solution = solver.numerical_solution()
        fig, ax = plt.subplots()
        solver.plot_battle(solution=solution, ax=ax)
        plt.close(fig)


class TestConsistencyWithAnalytical(unittest.TestCase):
    """Compare ODE solver outputs with analytical square-law results."""

    def setUp(self):
        self.cases = [
            (120, 80, 0.02, 0.02, 'A wins'),
            (80, 120, 0.01, 0.02, 'B wins'),
            (100, 50, 1.0, 4.0, 'draw'),
            (60, 110, 0.06, 0.01, 'superior effectiveness'),
            (90, 90, 0.015, 0.015, 'equal forces'),
        ]

    def test_winner_consistency(self):
        for A0, B0, alpha, beta, label in self.cases:
            with self.subTest(label=label):
                ode_solver = LanchesterSquareODESolver(A0, B0, alpha, beta)
                analytical = LanchesterSquare(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                analytical_winner, _, _ = analytical.calculate_battle_outcome()
                self.assertEqual(ode_solution.winner, analytical_winner)

    def test_remaining_strength_consistency(self):
        tolerance = 1.0
        for A0, B0, alpha, beta, label in self.cases:
            with self.subTest(label=label):
                ode_solver = LanchesterSquareODESolver(A0, B0, alpha, beta)
                analytical = LanchesterSquare(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                _, analytical_remaining, _ = analytical.calculate_battle_outcome()
                if ode_solution.winner != 'Draw':
                    self.assertAlmostEqual(ode_solution.remaining_strength, analytical_remaining, delta=tolerance)

    def test_battle_end_time_consistency(self):
        tolerance = 1.0
        for A0, B0, alpha, beta, label in self.cases:
            with self.subTest(label=label):
                ode_solver = LanchesterSquareODESolver(A0, B0, alpha, beta)
                analytical = LanchesterSquare(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                winner, remaining, invariant = analytical.calculate_battle_outcome()
                analytical_t_end = analytical.calculate_battle_end_time(winner, remaining, invariant)
                if np.isfinite(analytical_t_end):
                    self.assertAlmostEqual(ode_solution.t_end, analytical_t_end, delta=tolerance)

    def test_trajectory_accuracy(self):
        ode_solver = LanchesterSquareODESolver(120, 80, 0.02, 0.02)
        analytical = LanchesterSquare(120, 80, 0.02, 0.02)

        _, _, t_end = analytical.calculate_battle_outcome()
        t_sample = np.linspace(0.0, min(t_end * 0.8, 50.0), 25)

        ode_a, ode_b = ode_solver.generate_force_trajectories(t_sample)
        analytical_solution = analytical.analytical_solution(t_max=t_end)
        analytical_a = np.interp(t_sample, analytical_solution['time'], analytical_solution['A'])
        analytical_b = np.interp(t_sample, analytical_solution['time'], analytical_solution['B'])

        for i, t in enumerate(t_sample):
            self.assertAlmostEqual(ode_a[i], analytical_a[i], delta=1.0, msg=f'A mismatch at t={t:.2f}')
            self.assertAlmostEqual(ode_b[i], analytical_b[i], delta=1.0, msg=f'B mismatch at t={t:.2f}')


class TestSquareSolverEdgeCases(unittest.TestCase):
    """Edge-case and error-condition coverage."""

    def test_zero_initial_forces(self):
        solver = LanchesterSquareODESolver(0, 0, 0.01, 0.02)
        solution = solver.solve()
        self.assertEqual(solution.winner, 'Draw')
        self.assertTrue(np.all(solution.force_a == 0))
        self.assertTrue(np.all(solution.force_b == 0))

        solver_a_zero = LanchesterSquareODESolver(0, 50, 0.01, 0.02)
        solution_a_zero = solver_a_zero.solve()
        self.assertEqual(solution_a_zero.winner, 'B')
        self.assertAlmostEqual(solution_a_zero.remaining_strength, 50.0)

        solver_b_zero = LanchesterSquareODESolver(50, 0, 0.01, 0.02)
        solution_b_zero = solver_b_zero.solve()
        self.assertEqual(solution_b_zero.winner, 'A')
        self.assertAlmostEqual(solution_b_zero.remaining_strength, 50.0)

    def test_zero_effectiveness_coefficients(self):
        solver_zero_alpha = LanchesterSquareODESolver(100, 80, 0.0, 0.01)
        solution_zero_alpha = solver_zero_alpha.solve()
        self.assertEqual(solution_zero_alpha.winner, 'B')
        self.assertAlmostEqual(solution_zero_alpha.remaining_strength, 80.0)

        solver_zero_beta = LanchesterSquareODESolver(100, 80, 0.01, 0.0)
        solution_zero_beta = solver_zero_beta.solve()
        self.assertEqual(solution_zero_beta.winner, 'A')
        self.assertAlmostEqual(solution_zero_beta.remaining_strength, 100.0)

        solver_zero_both = LanchesterSquareODESolver(100, 80, 0.0, 0.0)
        solution_zero_both = solver_zero_both.solve()
        self.assertEqual(solution_zero_both.winner, 'Draw')
        self.assertTrue(np.all(solution_zero_both.force_a == 100.0))
        self.assertTrue(np.all(solution_zero_both.force_b == 80.0))

    def test_large_forces(self):
        solver = LanchesterSquareODESolver(1e6, 8e5, 0.005, 0.004)
        solution = solver.solve(num_points=2000)
        self.assertFalse(np.any(np.isnan(solution.force_a)))
        self.assertFalse(np.any(np.isnan(solution.force_b)))
        self.assertFalse(np.any(np.isinf(solution.force_a)))
        self.assertFalse(np.any(np.isinf(solution.force_b)))

    def test_monotonic_time_validation(self):
        solver = LanchesterSquareODESolver(120, 80, 0.02, 0.02)
        with self.assertRaises(ValueError):
            solver.solve(sample_times=np.array([0.0, 2.0, 1.0]))
        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([1.0, 0.5, 2.0]))

    def test_force_clamping(self):
        solver = LanchesterSquareODESolver(10, 8, 0.5, 0.6)
        solution = solver.solve(num_points=2000)
        self.assertTrue(np.all(solution.force_a >= -1e-9))
        self.assertTrue(np.all(solution.force_b >= -1e-9))

    def test_extreme_time_arrays(self):
        solver = LanchesterSquareODESolver(120, 80, 0.02, 0.02)
        custom_times = np.array([0.0, 0.1, 0.1, 5.0, 10.0])
        solution = solver.solve(sample_times=custom_times)
        np.testing.assert_array_equal(solution.time, custom_times)
        for i in range(1, len(custom_times)):
            if custom_times[i] == custom_times[i - 1]:
                self.assertAlmostEqual(solution.force_a[i], solution.force_a[i - 1], places=6)
                self.assertAlmostEqual(solution.force_b[i], solution.force_b[i - 1], places=6)


class TestSquareODESolverAdditionalCoverage(unittest.TestCase):
    """Additional tests to improve coverage of Square ODE Solver."""

    def test_num_points_less_than_two(self):
        """Test that num_points < 2 is corrected to 2."""
        solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)
        solution = solver.solve(num_points=1)  # Should be corrected to 2

        # Should have at least 2 points
        self.assertGreaterEqual(len(solution.time), 2)
        self.assertGreaterEqual(len(solution.force_a), 2)
        self.assertGreaterEqual(len(solution.force_b), 2)

    def test_instant_victory_a_when_b_no_combat_power(self):
        """Test instant victory calculation when B has no combat power."""
        # B0=0 and beta=0 means B has no combat capability
        solver = LanchesterSquareODESolver(100, 0, 0.5, 0.0)
        winner, remaining, invariant = solver.calculate_battle_outcome()
        t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

        self.assertEqual(winner, 'A')
        self.assertEqual(remaining, 100.0)
        self.assertEqual(t_end, 0.0)  # Instant victory

    def test_instant_victory_b_when_a_no_combat_power(self):
        """Test instant victory calculation when A has no combat power."""
        # A0=0 and alpha=0 means A has no combat capability
        solver = LanchesterSquareODESolver(0, 100, 0.0, 0.5)
        winner, remaining, invariant = solver.calculate_battle_outcome()
        t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

        self.assertEqual(winner, 'B')
        self.assertEqual(remaining, 100.0)
        self.assertEqual(t_end, 0.0)  # Instant victory

    def test_draw_with_exact_balance(self):
        """Test draw case with exact balance."""
        # Create exact draw: alpha*A0^2 = beta*B0^2
        solver = LanchesterSquareODESolver(100, 50, 1.0, 4.0)
        winner, remaining, invariant = solver.calculate_battle_outcome()

        # Should be a draw
        self.assertEqual(winner, 'Draw')
        # Invariant should be very close to zero
        self.assertAlmostEqual(invariant, 0.0, places=6)

    def test_draw_casualty_calculation_finite_time(self):
        """Test casualty calculation for draw with finite battle time."""
        # Exact draw where both forces are eliminated
        solver = LanchesterSquareODESolver(100, 50, 1.0, 4.0)
        result = solver.numerical_solution()

        self.assertEqual(result['winner'], 'Draw')

        # For finite time draw, both forces are eliminated
        if np.isfinite(result['battle_end_time']):
            self.assertAlmostEqual(result['A_casualties'], solver.A0, places=1)
            self.assertAlmostEqual(result['B_casualties'], solver.B0, places=1)

    def test_draw_casualty_calculation_infinite_time(self):
        """Test casualty calculation for draw with infinite time (no combat)."""
        # Both coefficients zero = no combat
        solver = LanchesterSquareODESolver(100, 80, 0.0, 0.0)
        result = solver.numerical_solution()

        self.assertEqual(result['winner'], 'Draw')
        self.assertEqual(result['battle_end_time'], float('inf'))

        # With no combat, no casualties
        self.assertEqual(result['A_casualties'], 0.0)
        self.assertEqual(result['B_casualties'], 0.0)

    def test_negative_force_a_raises_value_error(self):
        """Test that negative A0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LanchesterSquareODESolver(-10, 50, 0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_force_b_raises_value_error(self):
        """Test that negative B0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LanchesterSquareODESolver(50, -10, 0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_alpha_raises_value_error(self):
        """Test that negative alpha raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LanchesterSquareODESolver(50, 50, -0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_beta_raises_value_error(self):
        """Test that negative beta raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LanchesterSquareODESolver(50, 50, 0.5, -0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_degenerate_draw_fallback(self):
        """Test unexpected degenerate draw case fallback."""
        # Create scenario with one zero coefficient in draw conditions
        # This tests edge cases in the draw handling logic
        solver = LanchesterSquareODESolver(100, 100, 0.5, 0.0)
        winner, remaining, invariant = solver.calculate_battle_outcome()

        # A should win because B can't damage A
        self.assertEqual(winner, 'A')

    def test_plot_battle_with_default_solution(self):
        """Test plot_battle generates solution when none provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

            fig, ax = plt.subplots()
            solver.plot_battle(ax=ax)  # Should generate solution internally
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_autoshow(self):
        """Test plot_battle auto-shows when no axes provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from unittest import mock

            solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

            with mock.patch.object(plt, "show") as show_mock:
                solver.plot_battle()
                show_mock.assert_called_once()
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_no_autoshow_with_axes(self):
        """Test plot_battle doesn't auto-show when axes provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from unittest import mock

            solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

            fig, ax = plt.subplots()
            with mock.patch.object(plt, "show") as show_mock:
                solver.plot_battle(ax=ax)
                show_mock.assert_not_called()
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_multiple_battles_single_battle(self):
        """Test plot_multiple_battles with single battle (edge case)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

            # Single battle should work (axes wrapping edge case)
            LanchesterSquareODESolver.plot_multiple_battles([solver])
            plt.close('all')
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_multiple_battles_with_solutions_and_titles(self):
        """Test plot_multiple_battles with provided solutions and titles."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            solver1 = LanchesterSquareODESolver(100, 80, 0.5, 0.6)
            solver2 = LanchesterSquareODESolver(120, 90, 0.6, 0.5)

            sol1 = solver1.numerical_solution()
            sol2 = solver2.numerical_solution()

            LanchesterSquareODESolver.plot_multiple_battles(
                [solver1, solver2],
                solutions=[sol1, sol2],
                titles=["Battle 1", "Battle 2"]
            )
            plt.close('all')
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_battle_end_time_for_decisive_victory(self):
        """Test battle end time calculation for decisive victory."""
        # Clear victory scenario
        solver = LanchesterSquareODESolver(100, 60, 0.5, 0.3)
        winner, remaining, invariant = solver.calculate_battle_outcome()
        t_end = solver.calculate_battle_end_time(winner, remaining, invariant)

        # Should have finite positive end time
        self.assertGreater(t_end, 0)
        self.assertTrue(np.isfinite(t_end))

    def test_solution_with_very_short_time_span(self):
        """Test solution generation with very short time span."""
        solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

        # Very short time span
        solution = solver.solve(sample_times=np.array([0.0, 0.01, 0.02]))

        # Should have 3 time points
        self.assertEqual(len(solution.time), 3)

        # Forces should not have changed much
        self.assertAlmostEqual(solution.force_a[0], 100, places=0)
        self.assertAlmostEqual(solution.force_b[0], 80, places=0)

    def test_invariant_value_in_numerical_solution(self):
        """Test that numerical_solution returns correct invariant value."""
        solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)
        result = solver.numerical_solution()

        # Calculate expected invariant
        expected_invariant = solver.alpha * solver.A0**2 - solver.beta * solver.B0**2

        self.assertAlmostEqual(result['invariant'], expected_invariant, places=6)

    def test_solve_with_custom_num_points(self):
        """Test solve respects custom num_points parameter."""
        solver = LanchesterSquareODESolver(100, 80, 0.5, 0.6)

        # Request specific number of points
        solution = solver.solve(num_points=50)

        self.assertEqual(len(solution.time), 50)
        self.assertEqual(len(solution.force_a), 50)
        self.assertEqual(len(solution.force_b), 50)


if __name__ == '__main__':
    unittest.main()
