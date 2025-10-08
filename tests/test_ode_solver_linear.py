"""
Unit tests for LanchesterLinearODESolver - numerical integration implementation.

Tests numerical accuracy, edge cases, consistency with analytical solutions,
and proper handling of integration parameters.
"""

import unittest
import numpy as np
from models.ode_solver_lanchseter_linear import LanchesterLinearODESolver, LinearODESolution
from models import LanchesterLinear


class TestLanchesterLinearODESolver(unittest.TestCase):
    """Test cases for ODE-based Linear Law numerical solver."""

    def setUp(self):
        """Set up test fixtures with various scenarios."""
        # Standard test case: A has numerical advantage
        self.solver_a_wins = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        self.analytical_a_wins = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

        # B wins scenario
        self.solver_b_wins = LanchesterLinearODESolver(A0=60, B0=90, alpha=0.5, beta=0.4)
        self.analytical_b_wins = LanchesterLinear(A0=60, B0=90, alpha=0.5, beta=0.4)

        # Draw scenario - equal elimination times
        self.solver_draw = LanchesterLinearODESolver(A0=100, B0=50, alpha=1.0, beta=2.0)
        self.analytical_draw = LanchesterLinear(A0=100, B0=50, alpha=1.0, beta=2.0)

        # Superior effectiveness scenario
        self.solver_superior = LanchesterLinearODESolver(A0=50, B0=100, alpha=2.0, beta=0.1)
        self.analytical_superior = LanchesterLinear(A0=50, B0=100, alpha=2.0, beta=0.1)

    def test_constructor_validation(self):
        """Test that constructor validates input parameters correctly."""
        # Test negative initial forces
        with self.assertRaises(ValueError):
            LanchesterLinearODESolver(A0=-10, B0=50, alpha=0.5, beta=0.5)

        with self.assertRaises(ValueError):
            LanchesterLinearODESolver(A0=50, B0=-10, alpha=0.5, beta=0.5)

        # Test negative effectiveness coefficients
        with self.assertRaises(ValueError):
            LanchesterLinearODESolver(A0=50, B0=50, alpha=-0.5, beta=0.5)

        with self.assertRaises(ValueError):
            LanchesterLinearODESolver(A0=50, B0=50, alpha=0.5, beta=-0.5)

        # Test zero values (should be allowed)
        solver_zero = LanchesterLinearODESolver(A0=0, B0=50, alpha=0, beta=0.5)
        self.assertEqual(solver_zero.A0, 0)
        self.assertEqual(solver_zero.alpha, 0)

    def test_rhs_function(self):
        """Test the right-hand side function used by the integrator."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Both forces active
        y = np.array([50.0, 40.0])
        rhs = solver._rhs(0.0, y)
        expected = np.array([-0.6, -0.5])  # [-beta, -alpha]
        np.testing.assert_array_almost_equal(rhs, expected)

        # One force eliminated (should stop attrition)
        y_eliminated = np.array([50.0, 0.0])
        rhs_eliminated = solver._rhs(0.0, y_eliminated)
        expected_eliminated = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(rhs_eliminated, expected_eliminated)

        # Very small force (below tolerance)
        y_small = np.array([solver.ZERO_TOLERANCE / 2, 40.0])
        rhs_small = solver._rhs(0.0, y_small)
        expected_small = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(rhs_small, expected_small)

    def test_rk4_step(self):
        """Test Runge-Kutta 4th order integration step."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        y0 = np.array([100.0, 80.0])
        dt = 1.0
        y1 = solver._rk4_step(0.0, y0, dt)

        # Should approximate linear decay: A(1) ≈ 100 - 0.6*1, B(1) ≈ 80 - 0.5*1
        expected_approx = np.array([99.4, 79.5])
        np.testing.assert_array_almost_equal(y1, expected_approx, decimal=1)

        # Test zero step size
        y_same = solver._rk4_step(0.0, y0, 0.0)
        np.testing.assert_array_equal(y_same, y0)

    def test_elimination_time_calculation(self):
        """Test analytical elimination time calculation."""
        # A eliminates B faster
        solver1 = LanchesterLinearODESolver(A0=100, B0=60, alpha=2.0, beta=0.5)
        t_elim1 = solver1._compute_elimination_time()
        expected1 = min(60/2.0, 100/0.5)  # min(30, 200) = 30
        self.assertAlmostEqual(t_elim1, expected1, places=2)

        # B eliminates A faster
        solver2 = LanchesterLinearODESolver(A0=50, B0=100, alpha=0.1, beta=2.0)
        t_elim2 = solver2._compute_elimination_time()
        expected2 = min(100/0.1, 50/2.0)  # min(1000, 25) = 25
        self.assertAlmostEqual(t_elim2, expected2, places=2)

        # Zero effectiveness (infinite time)
        solver_zero = LanchesterLinearODESolver(A0=100, B0=80, alpha=0, beta=0)
        t_elim_zero = solver_zero._compute_elimination_time()
        self.assertTrue(np.isinf(t_elim_zero))

        # One-sided zero effectiveness
        solver_one_zero = LanchesterLinearODESolver(A0=100, B0=80, alpha=1.0, beta=0)
        t_elim_one_zero = solver_one_zero._compute_elimination_time()
        self.assertAlmostEqual(t_elim_one_zero, 80.0, places=2)  # B0/alpha

    def test_sample_times_preparation(self):
        """Test sample time array preparation for various scenarios."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Test explicit t_span
        times1 = solver._prepare_sample_times(t_span=(0, 10), num_points=11, sample_times=None)
        expected1 = np.linspace(0, 10, 11)
        np.testing.assert_array_almost_equal(times1, expected1)

        # Test explicit sample_times
        custom_times = np.array([0, 1, 2, 5, 10])
        times2 = solver._prepare_sample_times(t_span=None, num_points=100, sample_times=custom_times)
        np.testing.assert_array_equal(times2, custom_times)

        # Test auto time span based on elimination time
        times3 = solver._prepare_sample_times(t_span=None, num_points=10, sample_times=None)
        self.assertEqual(times3[0], 0.0)
        self.assertGreaterEqual(times3[-1], solver._compute_elimination_time())

        # Test validation errors
        with self.assertRaises(ValueError):
            solver._prepare_sample_times(t_span=(10, 5), num_points=10, sample_times=None)  # t1 < t0

        with self.assertRaises(ValueError):
            solver._prepare_sample_times(t_span=None, num_points=10, sample_times=np.array([5, 3, 8]))  # Not monotonic

        with self.assertRaises(ValueError):
            solver._prepare_sample_times(t_span=None, num_points=10, sample_times=np.array([]))  # Empty array

    def test_solve_basic_functionality(self):
        """Test basic solve functionality and return structure."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        solution = solver.solve()

        # Check solution structure
        self.assertIsInstance(solution, LinearODESolution)
        self.assertIsInstance(solution.time, np.ndarray)
        self.assertIsInstance(solution.force_a, np.ndarray)
        self.assertIsInstance(solution.force_b, np.ndarray)
        self.assertIsInstance(solution.winner, str)
        self.assertIsInstance(solution.t_end, float)
        self.assertIsInstance(solution.remaining_strength, float)

        # Check array shapes
        self.assertEqual(len(solution.time), len(solution.force_a))
        self.assertEqual(len(solution.time), len(solution.force_b))

        # Check initial conditions
        self.assertAlmostEqual(solution.force_a[0], 100, places=1)
        self.assertAlmostEqual(solution.force_b[0], 80, places=1)

        # Check time starts at zero
        self.assertEqual(solution.time[0], 0.0)

        # Check forces are non-negative
        self.assertTrue(np.all(solution.force_a >= 0))
        self.assertTrue(np.all(solution.force_b >= 0))

    def test_solve_with_parameters(self):
        """Test solve with various parameter combinations."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Test with explicit t_span
        solution1 = solver.solve(t_span=(0, 50), num_points=51)
        self.assertEqual(len(solution1.time), 51)
        self.assertAlmostEqual(solution1.time[-1], 50, places=1)

        # Test with custom sample times
        custom_times = np.linspace(0, 20, 21)
        solution2 = solver.solve(sample_times=custom_times)
        np.testing.assert_array_almost_equal(solution2.time, custom_times)

        # Test with different num_points
        solution3 = solver.solve(num_points=1000)
        self.assertEqual(len(solution3.time), 1000)

    def test_t_end_matches_analytical_with_truncated_span(self):
        """Ensure truncated integration windows still report the analytical end time."""
        solver = self.solver_a_wins
        _, _, analytical_t_end = solver.calculate_battle_outcome()

        truncated_end = analytical_t_end * 0.4
        solution = solver.solve(t_span=(0.0, truncated_end), num_points=5)

        self.assertAlmostEqual(solution.t_end, analytical_t_end, places=6)

    def test_t_end_matches_analytical_with_sparse_samples(self):
        """Ensure sparse sample grids that stop early still return correct t_end."""
        solver = self.solver_a_wins
        _, _, analytical_t_end = solver.calculate_battle_outcome()

        custom_times = np.linspace(0.0, analytical_t_end * 0.5, 6)
        solution = solver.solve(sample_times=custom_times)

        self.assertAlmostEqual(solution.t_end, analytical_t_end, places=6)

    def test_winner_determination(self):
        """Test that winner is correctly determined from numerical integration."""
        test_cases = [
            (self.solver_a_wins, 'A'),
            (self.solver_b_wins, 'B'),
            (self.solver_draw, 'Draw'),
            (self.solver_superior, 'A')
        ]

        for solver, expected_winner in test_cases:
            with self.subTest(solver=solver, expected=expected_winner):
                solution = solver.solve()
                self.assertEqual(solution.winner, expected_winner)

                # Check remaining strength logic
                if expected_winner == 'A':
                    self.assertGreater(solution.remaining_strength, 0)
                    # The remaining strength should be the survivor's strength, not necessarily final value
                    # due to discrete time steps in numerical integration
                    self.assertGreaterEqual(solution.remaining_strength, 0)
                elif expected_winner == 'B':
                    self.assertGreater(solution.remaining_strength, 0)
                    # Similar for B winning
                    self.assertGreaterEqual(solution.remaining_strength, 0)
                elif expected_winner == 'Draw':
                    # In a draw, remaining strength should be near zero
                    self.assertLessEqual(solution.remaining_strength, 1.0)

    def test_final_strengths_property(self):
        """Test the final_strengths property of LinearODESolution."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        solution = solver.solve()

        final_a, final_b = solution.final_strengths
        self.assertAlmostEqual(final_a, solution.force_a[-1], places=2)
        self.assertAlmostEqual(final_b, solution.force_b[-1], places=2)
        self.assertIsInstance(final_a, float)
        self.assertIsInstance(final_b, float)

    def test_calculate_battle_outcome_consistency(self):
        """Test that calculate_battle_outcome matches solve results."""
        solvers = [self.solver_a_wins, self.solver_b_wins, self.solver_draw, self.solver_superior]

        for solver in solvers:
            with self.subTest(solver=solver):
                # Get results from both methods
                solution = solver.solve()
                winner, remaining, t_end = solver.calculate_battle_outcome()

                # Check consistency
                self.assertEqual(solution.winner, winner)
                self.assertAlmostEqual(solution.remaining_strength, remaining, places=1)
                self.assertAlmostEqual(solution.t_end, t_end, places=1)

    def test_generate_force_trajectories(self):
        """Test force trajectory generation at custom time points."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Test with simple time array
        t = np.linspace(0, 10, 11)
        force_a, force_b = solver.generate_force_trajectories(t)

        self.assertEqual(len(force_a), len(t))
        self.assertEqual(len(force_b), len(t))
        self.assertAlmostEqual(force_a[0], 100, places=1)
        self.assertAlmostEqual(force_b[0], 80, places=1)

        # Test with single time point
        t_single = np.array([5.0])
        fa_single, fb_single = solver.generate_force_trajectories(t_single)
        self.assertEqual(len(fa_single), 1)
        self.assertEqual(len(fb_single), 1)

        # Test empty array
        t_empty = np.array([])
        fa_empty, fb_empty = solver.generate_force_trajectories(t_empty)
        self.assertEqual(len(fa_empty), 0)
        self.assertEqual(len(fb_empty), 0)

        # Test validation
        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([[1, 2], [3, 4]]))  # 2D array

        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([3, 1, 2]))  # Not monotonic

    def test_numerical_solution_format(self):
        """Test numerical_solution method returns expected format."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        result = solver.numerical_solution()

        # Check required keys
        required_keys = ['time', 'A', 'B', 'battle_end_time', 'winner',
                        'remaining_strength', 'A_casualties', 'B_casualties', 'linear_advantage']
        for key in required_keys:
            self.assertIn(key, result)

        # Check types and shapes
        self.assertIsInstance(result['time'], np.ndarray)
        self.assertIsInstance(result['A'], np.ndarray)
        self.assertIsInstance(result['B'], np.ndarray)
        self.assertEqual(len(result['time']), len(result['A']))
        self.assertEqual(len(result['time']), len(result['B']))

        # Check values
        self.assertEqual(result['time'][0], 0.0)
        self.assertAlmostEqual(result['A'][0], 100, places=1)
        self.assertAlmostEqual(result['B'][0], 80, places=1)

        # Check linear advantage calculation
        expected_advantage = solver.alpha * solver.A0 - solver.beta * solver.B0
        self.assertAlmostEqual(result['linear_advantage'], expected_advantage, places=2)

    def test_numerical_solution_with_parameters(self):
        """Test numerical_solution with different parameters."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Test with custom t_max
        result1 = solver.numerical_solution(t_max=50)
        self.assertAlmostEqual(result1['time'][-1], 50, places=1)

        # Test with custom num_points
        result2 = solver.numerical_solution(num_points=200)
        self.assertEqual(len(result2['time']), 200)

        # Test both parameters
        result3 = solver.numerical_solution(t_max=30, num_points=31)
        self.assertEqual(len(result3['time']), 31)
        self.assertAlmostEqual(result3['time'][-1], 30, places=1)

    def test_casualties_calculation(self):
        """Test casualty calculations in numerical_solution."""
        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)
        result = solver.numerical_solution()

        if result['winner'] == 'A':
            expected_a_casualties = solver.A0 - result['remaining_strength']
            expected_b_casualties = solver.B0
        elif result['winner'] == 'B':
            expected_a_casualties = solver.A0
            expected_b_casualties = solver.B0 - result['remaining_strength']
        else:  # Draw
            expected_a_casualties = solver.A0
            expected_b_casualties = solver.B0

        self.assertAlmostEqual(result['A_casualties'], expected_a_casualties, places=1)
        self.assertAlmostEqual(result['B_casualties'], expected_b_casualties, places=1)

    def test_plot_battle_functionality(self):
        """Test that plot_battle method runs without errors."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing

        solver = LanchesterLinearODESolver(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Test with default parameters (should not show plot in test)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            solver.plot_battle(ax=ax)  # Pass ax to avoid showing plot
            plt.close(fig)
        except ImportError:
            self.skipTest("Matplotlib not available for plotting test")

        # Test with custom solution
        solution = solver.numerical_solution()
        try:
            fig, ax = plt.subplots()
            solver.plot_battle(solution=solution, ax=ax)
            plt.close(fig)
        except ImportError:
            self.skipTest("Matplotlib not available for plotting test")


class TestConsistencyWithAnalytical(unittest.TestCase):
    """Test consistency between ODE solver and analytical Linear Law implementation."""

    def setUp(self):
        """Set up comparison test cases."""
        self.test_cases = [
            # (A0, B0, alpha, beta, description)
            (100, 80, 0.5, 0.6, "A_wins_standard"),
            (60, 90, 0.5, 0.4, "B_wins_standard"),
            (100, 50, 1.0, 2.0, "draw_equal_elimination"),
            (50, 100, 2.0, 0.1, "A_superior_effectiveness"),
            (100, 100, 0.3, 0.3, "equal_forces_equal_effectiveness"),
        ]

    def test_winner_consistency(self):
        """Test that ODE solver and analytical solver agree on winner."""
        for A0, B0, alpha, beta, description in self.test_cases:
            with self.subTest(case=description, A0=A0, B0=B0, alpha=alpha, beta=beta):
                ode_solver = LanchesterLinearODESolver(A0, B0, alpha, beta)
                analytical = LanchesterLinear(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                analytical_winner, _, _ = analytical.calculate_battle_outcome()

                self.assertEqual(ode_solution.winner, analytical_winner,
                               f"Winner mismatch in {description}")

    def test_remaining_strength_consistency(self):
        """Test that remaining strength calculations are consistent."""
        tolerance = 0.1  # Allow small numerical differences

        for A0, B0, alpha, beta, description in self.test_cases:
            with self.subTest(case=description):
                ode_solver = LanchesterLinearODESolver(A0, B0, alpha, beta)
                analytical = LanchesterLinear(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                analytical_winner, analytical_remaining, _ = analytical.calculate_battle_outcome()

                if analytical_winner != 'Draw':
                    self.assertAlmostEqual(
                        ode_solution.remaining_strength, analytical_remaining,
                        delta=tolerance,
                        msg=f"Remaining strength mismatch in {description}"
                    )

    def test_battle_end_time_consistency(self):
        """Test that battle end times are consistent."""
        tolerance = 0.1  # Allow small numerical differences

        for A0, B0, alpha, beta, description in self.test_cases:
            with self.subTest(case=description):
                ode_solver = LanchesterLinearODESolver(A0, B0, alpha, beta)
                analytical = LanchesterLinear(A0, B0, alpha, beta)

                ode_solution = ode_solver.solve()
                _, _, analytical_t_end = analytical.calculate_battle_outcome()

                if np.isfinite(analytical_t_end):
                    self.assertAlmostEqual(
                        ode_solution.t_end, analytical_t_end,
                        delta=tolerance,
                        msg=f"Battle end time mismatch in {description}"
                    )

    def test_trajectory_accuracy_during_battle(self):
        """Test that ODE trajectories match analytical trajectories during battle."""
        # Use a case with clear linear behavior
        ode_solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)
        analytical = LanchesterLinear(100, 80, 0.5, 0.6)

        # Get battle end time
        _, _, t_end = analytical.calculate_battle_outcome()

        # Sample during battle (before elimination)
        t_sample = np.linspace(0, t_end * 0.8, 20)  # 80% of battle duration

        ode_force_a, ode_force_b = ode_solver.generate_force_trajectories(t_sample)
        analytical_sol = analytical.analytical_solution()

        # Interpolate analytical solution at sample times
        analytical_a = np.interp(t_sample, analytical_sol['time'], analytical_sol['A'])
        analytical_b = np.interp(t_sample, analytical_sol['time'], analytical_sol['B'])

        # Check accuracy (allow 1% tolerance for numerical integration)
        tolerance = 1.0
        for i, t in enumerate(t_sample):
            self.assertAlmostEqual(
                ode_force_a[i], analytical_a[i], delta=tolerance,
                msg=f"Force A trajectory mismatch at t={t:.2f}"
            )
            self.assertAlmostEqual(
                ode_force_b[i], analytical_b[i], delta=tolerance,
                msg=f"Force B trajectory mismatch at t={t:.2f}"
            )


class TestEdgeCasesAndErrorConditions(unittest.TestCase):
    """Test edge cases and error conditions for ODE solver."""

    def test_zero_initial_forces(self):
        """Test behavior with zero initial forces."""
        # Both forces zero
        solver_both_zero = LanchesterLinearODESolver(0, 0, 0.5, 0.6)
        solution = solver_both_zero.solve()
        self.assertEqual(solution.winner, 'Draw')
        self.assertEqual(solution.remaining_strength, 0)
        self.assertTrue(np.all(solution.force_a == 0))
        self.assertTrue(np.all(solution.force_b == 0))

        # One force zero
        solver_a_zero = LanchesterLinearODESolver(0, 50, 0.5, 0.6)
        solution_a_zero = solver_a_zero.solve()
        self.assertEqual(solution_a_zero.winner, 'B')
        self.assertEqual(solution_a_zero.remaining_strength, 50)

        solver_b_zero = LanchesterLinearODESolver(50, 0, 0.5, 0.6)
        solution_b_zero = solver_b_zero.solve()
        self.assertEqual(solution_b_zero.winner, 'A')
        self.assertEqual(solution_b_zero.remaining_strength, 50)

    def test_zero_effectiveness_coefficients(self):
        """Test behavior with zero effectiveness coefficients."""
        # Zero alpha - A can't damage B
        solver_zero_alpha = LanchesterLinearODESolver(100, 80, 0, 0.5)
        solution = solver_zero_alpha.solve()
        self.assertEqual(solution.winner, 'B')
        self.assertEqual(solution.remaining_strength, 80)  # B survives intact

        # Zero beta - B can't damage A
        solver_zero_beta = LanchesterLinearODESolver(100, 80, 0.5, 0)
        solution = solver_zero_beta.solve()
        self.assertEqual(solution.winner, 'A')
        self.assertEqual(solution.remaining_strength, 100)  # A survives intact

        # Both zero - stalemate
        solver_zero_both = LanchesterLinearODESolver(100, 80, 0, 0)
        solution = solver_zero_both.solve()
        self.assertEqual(solution.winner, 'Draw')
        # Forces should remain constant
        self.assertTrue(np.all(solution.force_a == 100))
        self.assertTrue(np.all(solution.force_b == 80))

    def test_very_small_effectiveness(self):
        """Test with very small but non-zero effectiveness coefficients."""
        solver = LanchesterLinearODESolver(100, 80, 1e-6, 1e-6)
        solution = solver.solve()

        # Should still determine a winner based on numerical advantage
        self.assertIn(solution.winner, ['A', 'B', 'Draw', 'Ongoing'])
        self.assertGreaterEqual(solution.remaining_strength, 0)

    def test_very_large_forces(self):
        """Test numerical stability with very large initial forces."""
        solver = LanchesterLinearODESolver(1e6, 8e5, 0.5, 0.6)
        solution = solver.solve()

        # Should handle large numbers without overflow
        self.assertFalse(np.any(np.isinf(solution.force_a)))
        self.assertFalse(np.any(np.isinf(solution.force_b)))
        self.assertFalse(np.any(np.isnan(solution.force_a)))
        self.assertFalse(np.any(np.isnan(solution.force_b)))

        # Winner should still be determined correctly
        self.assertIn(solution.winner, ['A', 'B', 'Draw'])

    def test_integration_with_zero_step_sizes(self):
        """Test behavior when sample times include duplicate values."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Include duplicate time points
        sample_times = np.array([0, 1, 1, 2, 3, 3, 3, 4, 5])
        solution = solver.solve(sample_times=sample_times)

        # Should handle zero step sizes gracefully
        self.assertEqual(len(solution.time), len(sample_times))
        np.testing.assert_array_equal(solution.time, sample_times)

        # Forces should be consistent at duplicate time points
        for i in range(1, len(sample_times)):
            if sample_times[i] == sample_times[i-1]:
                self.assertAlmostEqual(solution.force_a[i], solution.force_a[i-1], places=6)
                self.assertAlmostEqual(solution.force_b[i], solution.force_b[i-1], places=6)

    def test_force_clamping(self):
        """Test that forces are properly clamped to non-negative values."""
        # Use parameters that might cause numerical negative values
        solver = LanchesterLinearODESolver(10, 8, 0.5, 0.6)

        # Solve with many points to potentially trigger numerical issues
        solution = solver.solve(num_points=10000)

        # All forces should be non-negative
        self.assertTrue(np.all(solution.force_a >= 0))
        self.assertTrue(np.all(solution.force_b >= 0))

    def test_monotonic_time_validation(self):
        """Test validation of monotonic time arrays."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Non-monotonic sample times should raise error
        with self.assertRaises(ValueError):
            solver.solve(sample_times=np.array([0, 2, 1, 3]))

        # Negative time values in sample_times
        with self.assertRaises(ValueError):
            solver.generate_force_trajectories(np.array([-1, 0, 1]))

    def test_extreme_time_spans(self):
        """Test behavior with extreme time spans."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Very long time span
        solution_long = solver.solve(t_span=(0, 1000))
        self.assertAlmostEqual(solution_long.time[-1], 1000, places=1)

        # Very short time span
        solution_short = solver.solve(t_span=(0, 0.001))
        self.assertAlmostEqual(solution_short.time[-1], 0.001, places=6)

        # Forces should still be reasonable
        self.assertTrue(np.all(solution_short.force_a >= 0))
        self.assertTrue(np.all(solution_short.force_b >= 0))


class TestLinearODESolverAdditionalCoverage(unittest.TestCase):
    """Additional tests to improve coverage of Linear ODE Solver."""

    def test_negative_force_a_raises_value_error(self):
        """Test that negative A0 raises ValueError (line 86)."""
        with self.assertRaises(ValueError) as context:
            LanchesterLinearODESolver(-10, 50, 0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_force_b_raises_value_error(self):
        """Test that negative B0 raises ValueError (line 100)."""
        with self.assertRaises(ValueError) as context:
            LanchesterLinearODESolver(50, -10, 0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_alpha_raises_value_error(self):
        """Test that negative alpha raises ValueError (line 106)."""
        with self.assertRaises(ValueError) as context:
            LanchesterLinearODESolver(50, 50, -0.5, 0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_negative_beta_raises_value_error(self):
        """Test that negative beta raises ValueError (line 110)."""
        with self.assertRaises(ValueError) as context:
            LanchesterLinearODESolver(50, 50, 0.5, -0.5)

        self.assertIn("non-negative", str(context.exception))

    def test_zero_tolerance_edge_case(self):
        """Test battle outcome with forces very close to zero (line 148)."""
        # Create scenario where one force is just above zero tolerance
        solver = LanchesterLinearODESolver(0.0001, 50, 0.5, 0.5)
        winner, remaining, advantage = solver.calculate_battle_outcome()

        # B should win since A is essentially zero
        self.assertEqual(winner, 'B')

    def test_non_monotonic_time_array_raises_error(self):
        """Test that non-monotonic time array raises ValueError (lines 174-175)."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Create non-monotonic time array
        bad_times = np.array([0.0, 1.0, 0.5, 2.0])

        with self.assertRaises(ValueError) as context:
            solver.solve(sample_times=bad_times)

        self.assertIn("monotonically increasing", str(context.exception))

    def test_negative_time_in_array_raises_error(self):
        """Test that negative time raises ValueError (lines 184-185)."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Create time array with negative value
        bad_times = np.array([-1.0, 0.0, 1.0, 2.0])

        with self.assertRaises(ValueError) as context:
            solver.solve(sample_times=bad_times)

        self.assertIn("non-negative", str(context.exception))

    def test_integration_step_with_zero_step_size(self):
        """Test integration handles zero step size (lines 216, 218, 220)."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Create time array with repeated values (zero step size)
        times_with_duplicates = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
        solution = solver.solve(sample_times=times_with_duplicates)

        # Should handle gracefully
        self.assertEqual(len(solution.time), len(times_with_duplicates))

    def test_empty_sample_times_array(self):
        """Test that empty sample_times raises appropriate error (line 246)."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Empty array should raise error
        with self.assertRaises((ValueError, IndexError)):
            solver.solve(sample_times=np.array([]))

    def test_single_point_sample_times(self):
        """Test solve with single point in sample_times."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Single point
        solution = solver.solve(sample_times=np.array([0.0]))

        self.assertEqual(len(solution.time), 1)
        self.assertAlmostEqual(solution.force_a[0], 100, places=6)
        self.assertAlmostEqual(solution.force_b[0], 80, places=6)

    def test_draw_casualty_calculation_finite_time(self):
        """Test casualty calculation for draw with finite time (lines 267-269)."""
        # Exact draw: alpha*A0 = beta*B0
        solver = LanchesterLinearODESolver(100, 100, 0.5, 0.5)
        result = solver.numerical_solution()

        self.assertEqual(result['winner'], 'Draw')

        # For finite time draw, check casualties
        if np.isfinite(result['battle_end_time']):
            # Both should have casualties
            self.assertGreater(result['A_casualties'], 0)
            self.assertGreater(result['B_casualties'], 0)

    def test_draw_casualty_calculation_infinite_time(self):
        """Test casualty calculation for draw with infinite time (line 271)."""
        # No combat: alpha=0 and beta=0
        solver = LanchesterLinearODESolver(100, 80, 0.0, 0.0)
        result = solver.numerical_solution()

        self.assertEqual(result['winner'], 'Draw')
        self.assertEqual(result['battle_end_time'], float('inf'))

        # With no combat, no casualties
        self.assertEqual(result['A_casualties'], 0.0)
        self.assertEqual(result['B_casualties'], 0.0)

    def test_plot_battle_with_default_solution(self):
        """Test plot_battle generates solution when none provided (line 284)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

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

            solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

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

            solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

            fig, ax = plt.subplots()
            with mock.patch.object(plt, "show") as show_mock:
                solver.plot_battle(ax=ax)
                show_mock.assert_not_called()
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_battle_outcome_with_zero_alpha(self):
        """Test battle outcome when alpha is zero."""
        solver = LanchesterLinearODESolver(100, 80, 0.0, 0.5)
        winner, remaining, advantage = solver.calculate_battle_outcome()

        # B should win since A can't damage B
        self.assertEqual(winner, 'B')

    def test_battle_outcome_with_zero_beta(self):
        """Test battle outcome when beta is zero."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.0)
        winner, remaining, advantage = solver.calculate_battle_outcome()

        # A should win since B can't damage A
        self.assertEqual(winner, 'A')

    def test_numerical_solution_invariant_value(self):
        """Test that numerical_solution returns correct linear advantage."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)
        result = solver.numerical_solution()

        # Calculate expected linear advantage: alpha*A0 - beta*B0
        expected_advantage = solver.alpha * solver.A0 - solver.beta * solver.B0

        self.assertAlmostEqual(result['linear_advantage'], expected_advantage, places=6)

    def test_solve_with_custom_num_points(self):
        """Test solve respects custom num_points parameter."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)

        # Request specific number of points
        solution = solver.solve(num_points=50)

        self.assertEqual(len(solution.time), 50)
        self.assertEqual(len(solution.force_a), 50)
        self.assertEqual(len(solution.force_b), 50)

    def test_very_short_battle(self):
        """Test battle with very short duration."""
        # Large effectiveness coefficients for quick battle
        solver = LanchesterLinearODESolver(10, 5, 5.0, 5.0)
        result = solver.numerical_solution()

        # Battle should end quickly
        self.assertLess(result['battle_end_time'], 10)
        self.assertGreater(result['battle_end_time'], 0)

    def test_num_points_less_than_two(self):
        """Test that num_points < 2 is corrected to 2."""
        solver = LanchesterLinearODESolver(100, 80, 0.5, 0.6)
        solution = solver.solve(num_points=1)  # Should be corrected to 2

        # Should have at least 2 points
        self.assertGreaterEqual(len(solution.time), 2)
        self.assertGreaterEqual(len(solution.force_a), 2)
        self.assertGreaterEqual(len(solution.force_b), 2)


if __name__ == '__main__':
    unittest.main()
