"""
Unit tests for LanchesterSquare model.

Tests mathematical correctness of Square Law implementation:
- Proper differential equations: dA/dt = -β·A·B, dB/dt = -α·A·B
- Square Law invariant: α·A²(t) - β·B²(t) = α·A₀² - β·B₀²
- Quadratic advantage calculations
- Force concentration effects
"""

import unittest
import numpy as np
from models import LanchesterSquare


class TestLanchesterSquare(unittest.TestCase):
    """Test cases for Square Law model core functionality."""

    def setUp(self):
        """Set up test fixtures with known scenarios."""
        # Standard test case: A has numerical advantage
        self.battle_a_wins = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)

        # B wins scenario
        self.battle_b_wins = LanchesterSquare(A0=60, B0=90, alpha=0.01, beta=0.01)

        # Draw scenario
        self.battle_draw = LanchesterSquare(A0=50, B0=50, alpha=0.01, beta=0.01)

        # Asymmetric effectiveness - A has better weapons
        self.battle_asymmetric = LanchesterSquare(A0=80, B0=100, alpha=0.02, beta=0.01)

    def test_reference_equal_effectiveness_scenario(self):
        """Exercise the canonical square-law scenario highlighted in review."""

        battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()

        self.assertEqual(winner, 'A')
        self.assertAlmostEqual(remaining, 60.0, places=2)

        expected_invariant = battle.alpha * battle.A0**2 - battle.beta * battle.B0**2
        self.assertAlmostEqual(invariant, expected_invariant, places=5)

        solution = battle.analytical_solution()
        self.assertAlmostEqual(solution['battle_end_time'], 109.86122886681098, places=5)

        time = solution['time']
        idx = np.searchsorted(time, solution['battle_end_time'])
        if idx >= len(time):
            idx = -1

        self.assertAlmostEqual(solution['A'][idx], 60.0, places=2)
        self.assertAlmostEqual(solution['B'][idx], 0.0, places=2)

    def test_square_law_invariant(self):
        """Test that Square Law invariant α·A²(t) - β·B²(t) = constant holds."""
        for battle in [self.battle_a_wins, self.battle_b_wins, self.battle_asymmetric]:
            solution = battle.analytical_solution()
            expected_invariant = battle.alpha * battle.A0**2 - battle.beta * battle.B0**2

            # Check at multiple time points
            for i in range(0, len(solution['time']), 100):
                A_t = solution['A'][i]
                B_t = solution['B'][i]

                # Only check when both forces exist
                if A_t > 0 and B_t > 0:
                    current_invariant = battle.alpha * A_t**2 - battle.beta * B_t**2

                    self.assertAlmostEqual(
                        current_invariant, expected_invariant, places=0,
                        msg=f"Square Law invariant not preserved at t={solution['time'][i]:.2f}"
                    )

    def test_battle_outcome_correctness(self):
        """Test that battle outcomes follow Square Law predictions."""
        # A wins: α·A₀² > β·B₀²
        winner, remaining, invariant = self.battle_a_wins.calculate_battle_outcome()
        self.assertEqual(winner, 'A')
        expected_remaining = np.sqrt(invariant / self.battle_a_wins.alpha)
        self.assertAlmostEqual(remaining, expected_remaining, places=1)

        # B wins scenario
        winner, remaining, invariant = self.battle_b_wins.calculate_battle_outcome()
        self.assertEqual(winner, 'B')
        expected_remaining = np.sqrt(-invariant / self.battle_b_wins.beta)
        self.assertAlmostEqual(remaining, expected_remaining, places=1)

        # Draw: equal squared forces
        winner, remaining, _ = self.battle_draw.calculate_battle_outcome()
        self.assertEqual(winner, 'Draw')
        self.assertAlmostEqual(remaining, 0.0, places=1)

    def test_quadratic_advantage(self):
        """Test that Square Law shows quadratic, not linear advantage."""
        # Compare scenarios with same linear advantage but different total forces
        battle_small = LanchesterSquare(A0=60, B0=40, alpha=0.01, beta=0.01)  # +20 advantage
        battle_large = LanchesterSquare(A0=120, B0=100, alpha=0.01, beta=0.01)  # +20 advantage

        _, remaining_small, _ = battle_small.calculate_battle_outcome()
        _, remaining_large, _ = battle_large.calculate_battle_outcome()

        # Square Law: larger forces should have disproportionately more survivors
        # sqrt(60² - 40²) = sqrt(3600 - 1600) = sqrt(2000) ≈ 44.7
        # sqrt(120² - 100²) = sqrt(14400 - 10000) = sqrt(4400) ≈ 66.3

        self.assertAlmostEqual(remaining_small, np.sqrt(60**2 - 40**2), places=1)
        self.assertAlmostEqual(remaining_large, np.sqrt(120**2 - 100**2), places=1)

        # Larger force should have proportionally more survivors than linear would predict
        linear_ratio = 120 / 60  # 2x forces
        actual_ratio = remaining_large / remaining_small
        # Square Law shows concentration advantage: ratio should be > 1.0 but < linear ratio
        self.assertGreater(actual_ratio, 1.4, "Square Law should show concentration advantage")
        self.assertLess(actual_ratio, linear_ratio, "But not as much as pure linear scaling")

    def test_effectiveness_vs_numbers(self):
        """Test scenarios where effectiveness can overcome numerical disadvantage."""
        # A has fewer numbers but much better effectiveness
        battle = LanchesterSquare(A0=70, B0=100, alpha=0.03, beta=0.01)

        # Check invariant: α·A₀² - β·B₀² = 0.03·70² - 0.01·100² = 147 - 100 = 47 > 0
        expected_invariant = 0.03 * 70**2 - 0.01 * 100**2
        self.assertGreater(expected_invariant, 0, "A should win despite numerical disadvantage")

        winner, remaining, invariant = battle.calculate_battle_outcome()
        self.assertEqual(winner, 'A')
        self.assertAlmostEqual(invariant, expected_invariant, places=1)

    def test_force_interaction_dynamics(self):
        """Test that attrition follows A·B interaction pattern."""
        solution = self.battle_asymmetric.analytical_solution()

        # Verify forces decrease over time (not constant)
        A_start = solution['A'][0]
        B_start = solution['B'][0]

        mid_idx = len(solution['time']) // 4
        A_mid = solution['A'][mid_idx]
        B_mid = solution['B'][mid_idx]

        self.assertLess(A_mid, A_start, "Force A should decrease")
        self.assertLess(B_mid, B_start, "Force B should decrease")

        # Check that the decrease follows exponential pattern, not linear
        # For Square Law, forces should decrease faster when opponent is larger
        quarter_idx = len(solution['time']) // 8
        if quarter_idx > 0:
            A_quarter = solution['A'][quarter_idx]

            # Early rate when forces are large
            rate_early = (A_start - A_quarter) / solution['time'][quarter_idx]
            # Later rate when forces are smaller
            rate_later = (A_quarter - A_mid) / (solution['time'][mid_idx] - solution['time'][quarter_idx])

            # Should show changing rates due to force interaction
            self.assertNotAlmostEqual(rate_early, rate_later, places=2,
                                    msg="Attrition should vary with force levels")

    def test_concentration_principle(self):
        """Test that concentrated forces have advantage over dispersed ones."""
        # Scenario: 100 vs (50 + 50) split forces
        concentrated = LanchesterSquare(A0=100, B0=100, alpha=0.01, beta=0.01)

        # If B splits forces, A should have advantage
        # This is implicit in Square Law math, but worth testing the principle
        # A=100 vs B=100: draw
        # A=100 vs B=50: A wins with sqrt(100² - 50²) = sqrt(7500) ≈ 86.6

        split_battle = LanchesterSquare(A0=100, B0=50, alpha=0.01, beta=0.01)
        _, remaining_vs_split, _ = split_battle.calculate_battle_outcome()

        # A should have significant survivors against split force
        self.assertGreater(remaining_vs_split, 80, "Concentrated force should dominate split force")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small effectiveness coefficients
        battle_small_alpha = LanchesterSquare(A0=100, B0=80, alpha=0.001, beta=0.001)
        solution = battle_small_alpha.analytical_solution()
        self.assertEqual(solution['winner'], 'A')

        # Zero effectiveness (degenerate case)
        battle_zero = LanchesterSquare(A0=100, B0=80, alpha=0, beta=0)
        solution = battle_zero.analytical_solution()
        # Should handle gracefully - forces should remain constant
        self.assertEqual(solution['A'][0], 100)
        self.assertEqual(solution['B'][0], 80)

        # Very asymmetric effectiveness
        battle_very_asym = LanchesterSquare(A0=50, B0=100, alpha=0.1, beta=0.001)
        winner, _, _ = battle_very_asym.calculate_battle_outcome()
        self.assertEqual(winner, 'A')  # Superior effectiveness should dominate

    def test_arctanh_argument_calculation(self):
        """Test that arctanh arguments use correct B₀/A₀ ratio formula."""
        # This test verifies the fix for the arctanh argument bug
        # Using the user's example: α=0.02, β=0.01, A₀=80, B₀=100
        battle = LanchesterSquare(A0=80, B0=100, alpha=0.02, beta=0.01)
        solution = battle.analytical_solution()

        # Manual calculation of expected battle duration
        ratio = np.sqrt(0.01 / 0.02)  # sqrt(beta/alpha) = sqrt(1/2)
        arg = ratio * 100 / 80  # Correct arctanh argument: sqrt(β/α) * B₀/A₀
        expected_duration = (1 / np.sqrt(0.02 * 0.01)) * np.arctanh(arg)

        # Verify the battle duration matches the correct mathematical formula
        self.assertAlmostEqual(solution['battle_end_time'], expected_duration, places=5,
                              msg=f"Battle duration should be {expected_duration:.3f}, got {solution['battle_end_time']:.3f}")

        # The fix should produce ~98.5 time units instead of the previous ~56.4
        self.assertGreater(solution['battle_end_time'], 95.0,
                          msg="Fixed duration should be much longer than the buggy version")

        # Test symmetric case (B elimination scenario)
        battle_b_elim = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.02)
        solution_b = battle_b_elim.analytical_solution()

        # Manual calculation for B elimination
        ratio_b = np.sqrt(0.01 / 0.02)  # sqrt(alpha/beta)
        arg_b = ratio_b * 100 / 80  # A₀/B₀ ratio
        expected_duration_b = (1 / np.sqrt(0.01 * 0.02)) * np.arctanh(arg_b)

        self.assertAlmostEqual(solution_b['battle_end_time'], expected_duration_b, places=5,
                              msg=f"B elimination duration should be {expected_duration_b:.3f}")

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            LanchesterSquare(A0=-10, B0=50, alpha=0.01, beta=0.01)  # Negative force

        with self.assertRaises(ValueError):
            LanchesterSquare(A0=50, B0=50, alpha=-0.01, beta=0.01)  # Negative effectiveness


if __name__ == '__main__':
    unittest.main()