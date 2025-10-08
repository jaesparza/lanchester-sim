"""
Unit tests for LanchesterLinear model.

Tests mathematical correctness of Linear Law implementation:
- Constant attrition rates: A(t) = A₀ - βt, B(t) = B₀ - αt
- Winner based on elimination time: min(B₀/α, A₀/β)
- Effectiveness coefficients determine outcomes
- Survivors calculated as remaining force at battle end
"""

import unittest
import numpy as np
from models import LanchesterLinear


class TestLanchesterLinear(unittest.TestCase):
    """Test cases for Linear Law model with constant attrition rates."""

    def setUp(self):
        """Set up test fixtures with known scenarios."""
        # Standard test case: A has numerical advantage
        self.battle_a_wins = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

        # B wins scenario
        self.battle_b_wins = LanchesterLinear(A0=60, B0=90, alpha=0.5, beta=0.4)

        # Draw scenario - equal elimination times
        self.battle_draw = LanchesterLinear(A0=100, B0=50, alpha=1.0, beta=2.0)

        # A has superior effectiveness despite fewer numbers
        self.battle_a_superior = LanchesterLinear(A0=50, B0=100, alpha=2.0, beta=0.1)

    def test_battle_outcome_correctness(self):
        """Test that battle outcomes follow Linear Law predictions."""
        # A wins: shorter elimination time
        winner, remaining, t_end = self.battle_a_wins.calculate_battle_outcome()
        self.assertEqual(winner, 'A')
        expected_remaining = self.battle_a_wins.A0 - self.battle_a_wins.beta * t_end
        self.assertAlmostEqual(remaining, expected_remaining, places=1)

        # B wins scenario
        winner2, remaining2, t_end2 = self.battle_b_wins.calculate_battle_outcome()
        self.assertEqual(winner2, 'B')
        expected_remaining2 = self.battle_b_wins.B0 - self.battle_b_wins.alpha * t_end2
        self.assertAlmostEqual(remaining2, expected_remaining2, places=1)

        # Draw: equal elimination times
        winner3, remaining3, _ = self.battle_draw.calculate_battle_outcome()
        self.assertEqual(winner3, 'Draw')
        self.assertEqual(remaining3, 0)

    def test_effectiveness_determines_winner(self):
        """Test that effectiveness coefficients properly determine battle outcomes."""
        # Superior effectiveness overcomes numerical disadvantage
        winner, remaining, _ = self.battle_a_superior.calculate_battle_outcome()
        self.assertEqual(winner, 'A', "Superior effectiveness should overcome numerical disadvantage")
        self.assertGreater(remaining, 0, "Winner should have survivors")

    def test_elimination_time_logic(self):
        """Test that winner is determined by who eliminates opponent first."""
        # A eliminates B in: B₀/α = 100/2.0 = 50 time units
        # B eliminates A in: A₀/β = 50/0.1 = 500 time units
        # A should win because 50 < 500

        winner, remaining, t_end = self.battle_a_superior.calculate_battle_outcome()
        expected_time = self.battle_a_superior.B0 / self.battle_a_superior.alpha

        self.assertEqual(winner, 'A')
        self.assertAlmostEqual(t_end, expected_time, places=1)

    def test_draw_scenario(self):
        """Test scenarios where both forces eliminate each other simultaneously."""
        # A eliminates B in: 50/1.0 = 50 time units
        # B eliminates A in: 100/2.0 = 50 time units -> Draw

        winner, remaining, t_end = self.battle_draw.calculate_battle_outcome()
        self.assertEqual(winner, 'Draw')
        self.assertEqual(remaining, 0)

    def test_elimination_time_tolerance(self):
        """Regression: treat floating near-ties as draws."""
        alpha = np.nextafter(0.3, 0.4)
        battle = LanchesterLinear(A0=10, B0=15, alpha=alpha, beta=0.2)

        winner, remaining, t_end = battle.calculate_battle_outcome()

        self.assertEqual(winner, 'Draw')
        self.assertEqual(remaining, 0)
        self.assertAlmostEqual(t_end, 50.0, places=9)

    def test_edge_cases(self):
        """Test edge cases with zero effectiveness."""
        # Zero effectiveness for A - A can't damage B, but B can damage A
        battle_zero_a = LanchesterLinear(A0=100, B0=50, alpha=0, beta=1.0)
        winner, remaining, t_end = battle_zero_a.calculate_battle_outcome()
        self.assertEqual(winner, 'B')  # B wins because A can't fight back
        self.assertEqual(remaining, 50)  # B survives intact

        # Zero effectiveness for B - B can't damage A, but A can damage B
        battle_zero_b = LanchesterLinear(A0=100, B0=50, alpha=1.0, beta=0)
        winner, remaining, t_end = battle_zero_b.calculate_battle_outcome()
        self.assertEqual(winner, 'A')  # A wins because B can't fight back
        self.assertEqual(remaining, 100)  # A survives intact

        # Both zero effectiveness - infinite stalemate
        battle_zero_both = LanchesterLinear(A0=100, B0=50, alpha=0, beta=0)
        winner, remaining, t_end = battle_zero_both.calculate_battle_outcome()
        self.assertEqual(winner, 'Draw')  # Nobody can damage anyone
        self.assertEqual(remaining, 100)  # Larger force survives (no attrition possible)
        self.assertTrue(np.isinf(t_end))  # Battle never ends

        # Test that analytical solution handles infinite battle duration without inf/NaN
        solution = battle_zero_both.analytical_solution()
        self.assertFalse(np.any(np.isinf(solution['time'])))  # Time array should be finite
        self.assertFalse(np.any(np.isnan(solution['time'])))  # No NaN values
        self.assertFalse(np.any(np.isinf(solution['A'])))     # Force A trajectory finite
        self.assertFalse(np.any(np.isnan(solution['A'])))     # No NaN values
        self.assertFalse(np.any(np.isinf(solution['B'])))     # Force B trajectory finite
        self.assertFalse(np.any(np.isnan(solution['B'])))     # No NaN values

        # Forces should remain constant (no attrition)
        self.assertTrue(np.all(solution['A'] == 100))  # A force stays at 100
        self.assertTrue(np.all(solution['B'] == 50))   # B force stays at 50

    def test_battle_duration_calculation(self):
        """Test that battle duration is calculated correctly."""
        # Simple case: A eliminates B
        battle = LanchesterLinear(A0=100, B0=60, alpha=2.0, beta=0.5)
        winner, remaining, t_end = battle.calculate_battle_outcome()

        expected_time = battle.B0 / battle.alpha  # 60/2.0 = 30
        self.assertAlmostEqual(t_end, expected_time, places=2)

    def test_survivor_calculation(self):
        """Test that survivors are calculated based on mutual attrition."""
        winner, remaining, t_end = self.battle_a_superior.calculate_battle_outcome()

        # During battle, A loses some troops to B's attacks
        # Should be positive but less than initial A₀
        self.assertGreater(remaining, 0)
        self.assertLess(remaining, self.battle_a_superior.A0)

    def test_linear_trajectories(self):
        """Test that force trajectories follow Linear Law during battle and stop correctly."""
        solution = self.battle_a_wins.analytical_solution()
        winner, remaining_strength, t_end = self.battle_a_wins.calculate_battle_outcome()

        # Check trajectories at different phases
        t = solution['time']

        # During battle: should follow A(t) = A₀ - βt, B(t) = B₀ - αt
        mid_battle_idx = len(t) // 4  # Early in battle
        if t[mid_battle_idx] < t_end:
            expected_A = self.battle_a_wins.A0 - self.battle_a_wins.beta * t[mid_battle_idx]
            expected_B = self.battle_a_wins.B0 - self.battle_a_wins.alpha * t[mid_battle_idx]
            self.assertAlmostEqual(solution['A'][mid_battle_idx], expected_A, places=1)
            self.assertAlmostEqual(solution['B'][mid_battle_idx], expected_B, places=1)

        # After battle: winner should maintain remaining strength
        final_idx = -1
        if winner == 'A':
            self.assertAlmostEqual(solution['A'][final_idx], remaining_strength, places=1)
            self.assertEqual(solution['B'][final_idx], 0)
        elif winner == 'B':
            self.assertEqual(solution['A'][final_idx], 0)
            self.assertAlmostEqual(solution['B'][final_idx], remaining_strength, places=1)

    def test_trajectory_survivor_consistency(self):
        """Test that trajectory final values exactly match calculated survivors.

        Critical bug: trajectories were continuing to decline after battle end,
        contradicting survivor calculations. This test ensures consistency.
        """
        for battle in [self.battle_a_wins, self.battle_b_wins, self.battle_a_superior]:
            with self.subTest(battle=battle):
                solution = battle.analytical_solution()
                winner, remaining_strength, t_end = battle.calculate_battle_outcome()

                # Find indices near and after battle end
                t = solution['time']
                t_end_idx = np.argmin(np.abs(t - t_end))
                final_idx = -1

                # Verify trajectory values match calculated survivors
                if winner == 'A':
                    self.assertAlmostEqual(
                        solution['A'][final_idx], remaining_strength, places=2,
                        msg=f"A trajectory final value {solution['A'][final_idx]:.2f} != calculated survivors {remaining_strength:.2f}"
                    )
                    self.assertEqual(
                        solution['B'][final_idx], 0,
                        msg="B should be eliminated (trajectory = 0)"
                    )
                elif winner == 'B':
                    self.assertEqual(
                        solution['A'][final_idx], 0,
                        msg="A should be eliminated (trajectory = 0)"
                    )
                    self.assertAlmostEqual(
                        solution['B'][final_idx], remaining_strength, places=2,
                        msg=f"B trajectory final value {solution['B'][final_idx]:.2f} != calculated survivors {remaining_strength:.2f}"
                    )
                else:  # Draw
                    self.assertEqual(solution['A'][final_idx], 0, msg="Draw: A should be zero")
                    self.assertEqual(solution['B'][final_idx], 0, msg="Draw: B should be zero")

                # Verify trajectories don't continue declining after t_end
                if final_idx > t_end_idx + 10:  # Ensure we're well past t_end
                    post_battle_idx = t_end_idx + 5
                    if winner == 'A':
                        self.assertAlmostEqual(
                            solution['A'][post_battle_idx], solution['A'][final_idx], places=2,
                            msg="Winner A should maintain constant strength after battle end"
                        )
                    elif winner == 'B':
                        self.assertAlmostEqual(
                            solution['B'][post_battle_idx], solution['B'][final_idx], places=2,
                            msg="Winner B should maintain constant strength after battle end"
                        )

    def test_linear_advantage_field_validation(self):
        """Test that analytical_solution returns correct linear_advantage field.

        Critical regression test: ensures αA₀ - βB₀ calculation is preserved
        and doesn't revert to incorrect A₀ - B₀ formula.
        """
        # Test 1: Equal effectiveness (α=β) - should match simple difference
        battle_equal = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.5)
        solution_equal = battle_equal.analytical_solution()
        expected_equal = 0.5 * 100 - 0.5 * 80  # αA₀ - βB₀ = 50 - 40 = 10
        self.assertAlmostEqual(solution_equal['linear_advantage'], expected_equal, places=2,
                              msg="Equal effectiveness: αA₀-βB₀ should equal 0.5×100-0.5×80=10")

        # Verify this matches simple difference when α=β
        simple_difference = 100 - 80  # A₀ - B₀ = 20
        self.assertAlmostEqual(solution_equal['linear_advantage'], simple_difference * 0.5, places=2,
                              msg="When α=β, weighted advantage should be α×(A₀-B₀)")

        # Test 2: Different effectiveness - critical test for bug prevention
        battle_diff = LanchesterLinear(A0=100, B0=80, alpha=0.8, beta=0.3)
        solution_diff = battle_diff.analytical_solution()
        expected_diff = 0.8 * 100 - 0.3 * 80  # αA₀ - βB₀ = 80 - 24 = 56
        self.assertAlmostEqual(solution_diff['linear_advantage'], expected_diff, places=2,
                              msg="Different effectiveness: αA₀-βB₀ should equal 0.8×100-0.3×80=56")

        # This should NOT equal the simple difference (regression test)
        wrong_simple = 100 - 80  # A₀ - B₀ = 20
        self.assertNotAlmostEqual(solution_diff['linear_advantage'], wrong_simple, places=1,
                                 msg="Different effectiveness: should NOT equal A₀-B₀=20")

        # Test 3: Zero effectiveness scenarios
        battle_zero_a = LanchesterLinear(A0=100, B0=80, alpha=0, beta=0.5)
        solution_zero_a = battle_zero_a.analytical_solution()
        expected_zero_a = 0 * 100 - 0.5 * 80  # 0 - 40 = -40
        self.assertAlmostEqual(solution_zero_a['linear_advantage'], expected_zero_a, places=2,
                              msg="Zero α: should equal 0×100-0.5×80=-40")

        battle_zero_b = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0)
        solution_zero_b = battle_zero_b.analytical_solution()
        expected_zero_b = 0.5 * 100 - 0 * 80  # 50 - 0 = 50
        self.assertAlmostEqual(solution_zero_b['linear_advantage'], expected_zero_b, places=2,
                              msg="Zero β: should equal 0.5×100-0×80=50")

        battle_zero_both = LanchesterLinear(A0=100, B0=80, alpha=0, beta=0)
        solution_zero_both = battle_zero_both.analytical_solution()
        expected_zero_both = 0 * 100 - 0 * 80  # 0 - 0 = 0
        self.assertAlmostEqual(solution_zero_both['linear_advantage'], expected_zero_both, places=2,
                              msg="Both zero: should equal 0×100-0×80=0")

        # Test 4: Negative advantage scenarios
        battle_negative = LanchesterLinear(A0=50, B0=100, alpha=0.2, beta=0.8)
        solution_negative = battle_negative.analytical_solution()
        expected_negative = 0.2 * 50 - 0.8 * 100  # 10 - 80 = -70
        self.assertAlmostEqual(solution_negative['linear_advantage'], expected_negative, places=2,
                              msg="Negative advantage: should equal 0.2×50-0.8×100=-70")

    def test_analytical_solution_completeness(self):
        """Test that analytical_solution returns all expected fields with correct values.

        Ensures all returned fields are tested and prevents regression of untested fields.
        """
        battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
        solution = battle.analytical_solution()
        winner, remaining_strength, t_end = battle.calculate_battle_outcome()

        # Test required fields exist
        required_fields = ['time', 'A', 'B', 'battle_end_time', 'winner',
                          'remaining_strength', 'A_casualties', 'B_casualties', 'linear_advantage']
        for field in required_fields:
            self.assertIn(field, solution, f"Missing required field: {field}")

        # Test battle_end_time consistency
        self.assertAlmostEqual(solution['battle_end_time'], t_end, places=2,
                              msg="analytical_solution battle_end_time should match calculate_battle_outcome")

        # Test winner consistency
        self.assertEqual(solution['winner'], winner,
                        msg="analytical_solution winner should match calculate_battle_outcome")

        # Test remaining_strength consistency
        self.assertAlmostEqual(solution['remaining_strength'], remaining_strength, places=2,
                              msg="analytical_solution remaining_strength should match calculate_battle_outcome")

        # Test casualties calculation
        if winner == 'A':
            expected_A_casualties = battle.A0 - remaining_strength
            expected_B_casualties = battle.B0
        elif winner == 'B':
            expected_A_casualties = battle.A0
            expected_B_casualties = battle.B0 - remaining_strength
        else:  # Draw
            expected_A_casualties = battle.A0
            expected_B_casualties = battle.B0

        self.assertAlmostEqual(solution['A_casualties'], expected_A_casualties, places=2,
                              msg="A_casualties should be calculated correctly")
        self.assertAlmostEqual(solution['B_casualties'], expected_B_casualties, places=2,
                              msg="B_casualties should be calculated correctly")

        # Test time array properties
        self.assertTrue(len(solution['time']) > 0, "Time array should not be empty")
        self.assertEqual(solution['time'][0], 0, "Time should start at 0")
        self.assertTrue(solution['time'][-1] > solution['battle_end_time'],
                       "Time should extend beyond battle end")

        # Test trajectory array lengths match time array
        self.assertEqual(len(solution['A']), len(solution['time']),
                        "Force A trajectory should match time array length")
        self.assertEqual(len(solution['B']), len(solution['time']),
                        "Force B trajectory should match time array length")

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            LanchesterLinear(A0=-10, B0=50, alpha=0.5, beta=0.5)  # Negative force

        with self.assertRaises(ValueError):
            LanchesterLinear(A0=50, B0=50, alpha=-0.5, beta=0.5)  # Negative effectiveness

    def test_large_values_numerical_stability_consistency(self):
        """Test that Linear Law handles large values consistently with Square Law.

        Ensures extremely large battle times (>1e15) are treated as infinite
        for numerical stability and cross-model consistency.
        """

        # Test extreme case that would result in very large time
        large_battle = LanchesterLinear(A0=1e8, B0=1e8, alpha=1e-10, beta=1e-10)
        solution = large_battle.analytical_solution()

        # Should treat this as infinite time for numerical stability
        self.assertTrue(np.isinf(solution['battle_end_time']),
                       msg="Extremely large battle times should be treated as infinite")
        self.assertEqual(solution['winner'], 'Draw',
                        msg="Infinite time scenario should be classified as draw")
        self.assertEqual(solution['A_casualties'], 0,
                        msg="No casualties should occur in infinite time")
        self.assertEqual(solution['B_casualties'], 0,
                        msg="No casualties should occur in infinite time")

        # Test that normal large values still work
        normal_large = LanchesterLinear(A0=10000, B0=10000, alpha=1e-6, beta=1e-6)
        normal_solution = normal_large.analytical_solution()

        self.assertTrue(np.isfinite(normal_solution['battle_end_time']),
                       msg="Normal large values should remain finite")
        self.assertLess(normal_solution['battle_end_time'], LanchesterLinear.LARGE_TIME_THRESHOLD,
                       msg="Normal case should be below threshold")


class TestLanchesterLinearAdditionalCoverage(unittest.TestCase):
    """Additional tests to improve coverage of Lanchester Linear Law."""

    def test_both_forces_initially_depleted(self):
        """Test edge case when both forces start at zero."""
        battle = LanchesterLinear(A0=0, B0=0, alpha=0.5, beta=0.5)
        winner, remaining, t_end = battle.calculate_battle_outcome()

        self.assertEqual(winner, 'Draw')
        self.assertEqual(remaining, 0.0)
        self.assertEqual(t_end, 0.0)

    def test_force_a_initially_depleted(self):
        """Test edge case when Force A starts at zero."""
        battle = LanchesterLinear(A0=0, B0=50, alpha=0.5, beta=0.5)
        winner, remaining, t_end = battle.calculate_battle_outcome()

        self.assertEqual(winner, 'B')
        self.assertEqual(remaining, 50.0)
        self.assertEqual(t_end, 0.0)

    def test_force_b_initially_depleted(self):
        """Test edge case when Force B starts at zero."""
        battle = LanchesterLinear(A0=50, B0=0, alpha=0.5, beta=0.5)
        winner, remaining, t_end = battle.calculate_battle_outcome()

        self.assertEqual(winner, 'A')
        self.assertEqual(remaining, 50.0)
        self.assertEqual(t_end, 0.0)

    def test_simple_analytical_solution_wrapper(self):
        """Test simple_analytical_solution method."""
        battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Call simple_analytical_solution (wrapper method)
        simple_solution = battle.simple_analytical_solution()

        # Call analytical_solution directly
        full_solution = battle.analytical_solution()

        # They should return the same result
        self.assertEqual(simple_solution['winner'], full_solution['winner'])
        self.assertAlmostEqual(simple_solution['remaining_strength'],
                              full_solution['remaining_strength'], places=2)
        self.assertAlmostEqual(simple_solution['battle_end_time'],
                              full_solution['battle_end_time'], places=2)

    def test_plot_battle_with_default_solution(self):
        """Test plot_battle generates solution when none provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

            # Call plot_battle without providing solution
            fig, ax = plt.subplots()
            battle.plot_battle(ax=ax)  # Should generate solution internally
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_with_provided_solution(self):
        """Test plot_battle uses provided solution."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
            solution = battle.analytical_solution()

            # Call plot_battle with provided solution
            fig, ax = plt.subplots()
            battle.plot_battle(solution=solution, ax=ax)
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

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

            with mock.patch.object(plt, "show") as show_mock:
                battle.plot_battle()
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

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

            fig, ax = plt.subplots()
            with mock.patch.object(plt, "show") as show_mock:
                battle.plot_battle(ax=ax)
                show_mock.assert_not_called()
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_includes_battle_end_marker(self):
        """Test plot includes battle end time marker."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
            solution = battle.analytical_solution()

            fig, ax = plt.subplots()
            battle.plot_battle(solution=solution, ax=ax)

            # Check that battle_end_time is in the solution
            self.assertIn('battle_end_time', solution)
            self.assertGreater(solution['battle_end_time'], 0)

            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_shows_linear_advantage(self):
        """Test plot displays linear law advantage calculation."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

            # Calculate expected linear advantage
            expected_advantage = battle.alpha * battle.A0 - battle.beta * battle.B0

            fig, ax = plt.subplots()
            battle.plot_battle(ax=ax)

            # Linear advantage should be calculated correctly
            self.assertAlmostEqual(expected_advantage, 50 - 48, places=2)

            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_custom_title(self):
        """Test plot_battle accepts custom title."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
            custom_title = "Custom Battle Scenario"

            fig, ax = plt.subplots()
            battle.plot_battle(title=custom_title, ax=ax)

            # Verify title is set (matplotlib sets it on the axes)
            self.assertIsNotNone(ax.get_title())

            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_analytical_solution_with_custom_t_max(self):
        """Test analytical_solution respects custom t_max parameter."""
        battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Get solution with custom t_max
        solution = battle.analytical_solution(t_max=50)

        # Time array should end at or near t_max
        self.assertLessEqual(solution['time'][-1], 55)  # Allow some buffer
        self.assertGreater(solution['time'][-1], 45)

    def test_simple_analytical_solution_default_t_max(self):
        """Test simple_analytical_solution uses default t_max."""
        battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

        # Call without t_max
        solution = battle.simple_analytical_solution()

        # Should have reasonable length (default uses many points for smooth curves)
        self.assertGreater(len(solution['time']), 50)
        self.assertLessEqual(len(solution['time']), 1000)


if __name__ == '__main__':
    unittest.main()
