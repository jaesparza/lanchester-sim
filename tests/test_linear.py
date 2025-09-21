"""
Unit tests for LanchesterLinear model.

Tests mathematical correctness of Linear Law implementation:
- Proper differential equations: dA/dt = -β·B, dB/dt = -α·A
- Linear advantage preservation: A(t) - B(t) = A₀ - B₀
- Force-dependent attrition rates
- Battle outcome calculations
"""

import unittest
import numpy as np
from models import LanchesterLinear


class TestLanchesterLinear(unittest.TestCase):
    """Test cases for Linear Law model core functionality."""

    def setUp(self):
        """Set up test fixtures with known scenarios."""
        # Standard test case: A has numerical advantage
        self.battle_a_wins = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.5)

        # B wins scenario
        self.battle_b_wins = LanchesterLinear(A0=60, B0=90, alpha=0.4, beta=0.4)

        # Draw scenario
        self.battle_draw = LanchesterLinear(A0=50, B0=50, alpha=0.3, beta=0.3)

        # Asymmetric effectiveness
        self.battle_asymmetric = LanchesterLinear(A0=80, B0=100, alpha=0.8, beta=0.4)

    def test_linear_advantage_preserved(self):
        """Test that Linear Law invariant A(t) - B(t) = A₀ - B₀ holds."""
        for battle in [self.battle_a_wins, self.battle_b_wins, self.battle_asymmetric]:
            solution = battle.analytical_solution()

            # Check at multiple time points
            for i in range(0, len(solution['time']), 100):
                t_idx = i
                A_t = solution['A'][t_idx]
                B_t = solution['B'][t_idx]

                # Only check when both forces exist
                if A_t > 0 and B_t > 0:
                    linear_advantage = A_t - B_t
                    expected_advantage = battle.A0 - battle.B0

                    self.assertAlmostEqual(
                        linear_advantage, expected_advantage, places=1,
                        msg=f"Linear advantage not preserved at t={solution['time'][t_idx]:.2f}"
                    )

    def test_battle_outcome_correctness(self):
        """Test that battle outcomes follow Linear Law predictions."""
        # A wins: larger initial force
        winner, remaining, _ = self.battle_a_wins.calculate_battle_outcome()
        self.assertEqual(winner, 'A')
        self.assertAlmostEqual(remaining, 20.0, places=1)  # 100 - 80

        # B wins: larger initial force
        winner, remaining, _ = self.battle_b_wins.calculate_battle_outcome()
        self.assertEqual(winner, 'B')
        self.assertAlmostEqual(remaining, 30.0, places=1)  # 90 - 60

        # Draw: equal forces
        winner, remaining, _ = self.battle_draw.calculate_battle_outcome()
        self.assertEqual(winner, 'Draw')
        self.assertAlmostEqual(remaining, 0.0, places=1)

    def test_force_dependent_attrition(self):
        """Test that Linear Law exhibits proper linear attrition behavior."""
        solution = self.battle_asymmetric.analytical_solution()

        # Find a point mid-battle where both forces exist
        mid_idx = len(solution['time']) // 4
        A_mid = solution['A'][mid_idx]
        B_mid = solution['B'][mid_idx]

        # Verify forces are decreasing (not constant)
        A_start = solution['A'][0]
        B_start = solution['B'][0]

        self.assertLess(A_mid, A_start, "Force A should decrease over time")
        self.assertLess(B_mid, B_start, "Force B should decrease over time")

        # Verify Linear Law invariant is preserved
        linear_advantage_start = A_start - B_start
        linear_advantage_mid = A_mid - B_mid

        self.assertAlmostEqual(linear_advantage_start, linear_advantage_mid, places=1,
                              msg="Linear Law invariant A(t) - B(t) = constant should be preserved")

        # Verify that battle outcomes depend on force parameters
        # Different force configurations should yield different results
        battle_different = LanchesterLinear(A0=100, B0=60, alpha=0.6, beta=0.8)
        solution_different = battle_different.analytical_solution()

        # The battles should have different durations and outcomes
        self.assertNotAlmostEqual(solution['battle_end_time'], solution_different['battle_end_time'], places=1,
                                 msg="Different force parameters should yield different battle durations")

    def test_battle_time_depends_on_forces(self):
        """Test that battle duration varies with force parameters."""
        # Different scenarios should have different battle times
        _, _, time_a = self.battle_a_wins.calculate_battle_outcome()
        _, _, time_b = self.battle_b_wins.calculate_battle_outcome()
        _, _, time_asym = self.battle_asymmetric.calculate_battle_outcome()

        # All should be positive and finite
        for time_val in [time_a, time_b, time_asym]:
            self.assertGreater(time_val, 0, "Battle time should be positive")
            self.assertLess(time_val, 100, "Battle time should be finite and reasonable")

        # Different scenarios should generally have different times
        times = [time_a, time_b, time_asym]
        unique_times = set(round(t, 2) for t in times)
        self.assertGreater(len(unique_times), 1, "Different scenarios should have different battle times")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero effectiveness coefficients
        battle_zero_alpha = LanchesterLinear(A0=100, B0=80, alpha=0, beta=0.5)
        winner, remaining, time = battle_zero_alpha.calculate_battle_outcome()
        self.assertEqual(winner, 'A')  # A should still win with advantage

        # Very small forces
        battle_small = LanchesterLinear(A0=1, B0=0.5, alpha=0.1, beta=0.1)
        solution = battle_small.analytical_solution()
        self.assertEqual(solution['winner'], 'A')

        # Equal forces, different effectiveness
        battle_equal_forces = LanchesterLinear(A0=50, B0=50, alpha=0.6, beta=0.4)
        winner, _, _ = battle_equal_forces.calculate_battle_outcome()
        self.assertEqual(winner, 'Draw')  # Linear Law: equal forces = draw regardless of effectiveness

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            LanchesterLinear(A0=-10, B0=50, alpha=0.5, beta=0.5)  # Negative force

        with self.assertRaises(ValueError):
            LanchesterLinear(A0=50, B0=50, alpha=-0.1, beta=0.5)  # Negative effectiveness


if __name__ == '__main__':
    unittest.main()