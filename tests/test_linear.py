"""
Unit tests for LanchesterLinear model - CORRECTED VERSION.

Tests mathematical correctness of Linear Law implementation:
- Effectiveness coefficients determine winner (not just raw numbers)
- Proper differential equations: dA/dt = -β·B, dB/dt = -α·A
- Winner based on elimination time: min(B₀/α, A₀/β)
- Survivors calculated based on mutual attrition during battle
"""

import unittest
import numpy as np
from models import LanchesterLinear


class TestLanchesterLinearCorrected(unittest.TestCase):
    """Test cases for Linear Law model with correct effectiveness-based logic."""

    def setUp(self):
        """Set up test fixtures with effectiveness-aware scenarios."""
        # Equal effectiveness - numbers matter
        self.battle_equal_eff = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.5)

        # A has superior effectiveness despite fewer numbers
        self.battle_a_superior = LanchesterLinear(A0=50, B0=100, alpha=2.0, beta=0.1)

        # B has superior effectiveness
        self.battle_b_superior = LanchesterLinear(A0=200, B0=50, alpha=0.1, beta=2.0)

        # Draw scenario - equal elimination times
        self.battle_draw = LanchesterLinear(A0=100, B0=50, alpha=1.0, beta=2.0)

    def test_effectiveness_determines_winner(self):
        """Test that effectiveness coefficients properly determine battle outcomes."""
        # Test 1: Superior effectiveness overcomes numerical disadvantage
        winner, remaining, _ = self.battle_a_superior.calculate_battle_outcome()
        self.assertEqual(winner, 'A', "Superior effectiveness should overcome numerical disadvantage")
        self.assertGreater(remaining, 0, "Winner should have survivors")

        # Test 2: Equal effectiveness, numbers matter
        winner2, remaining2, _ = self.battle_equal_eff.calculate_battle_outcome()
        self.assertEqual(winner2, 'A', "Equal effectiveness: larger force should win")

        # Test 3: B has superior effectiveness
        winner3, remaining3, _ = self.battle_b_superior.calculate_battle_outcome()
        self.assertEqual(winner3, 'B', "High effectiveness should beat large but ineffective force")

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

    def test_edge_cases(self):
        """Test edge cases with zero effectiveness."""
        # Zero effectiveness for A - A can't damage B
        battle_zero_a = LanchesterLinear(A0=100, B0=50, alpha=0, beta=1.0)
        winner, remaining, t_end = battle_zero_a.calculate_battle_outcome()
        self.assertEqual(winner, 'A')  # A can't be damaged
        self.assertEqual(remaining, 100)

        # Zero effectiveness for B - B can't damage A
        battle_zero_b = LanchesterLinear(A0=100, B0=50, alpha=1.0, beta=0)
        winner, remaining, t_end = battle_zero_b.calculate_battle_outcome()
        self.assertEqual(winner, 'B')  # B can't be damaged
        self.assertEqual(remaining, 50)

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

    def test_trajectory_reflects_effectiveness(self):
        """Test that force trajectories reflect effectiveness coefficients."""
        solution = self.battle_a_superior.analytical_solution()

        # Forces should decrease over time (not stay constant)
        self.assertLess(solution['A'][-2], solution['A'][0])  # A decreases

        # Battle should end with A winning
        self.assertEqual(solution['winner'], 'A')
        self.assertGreater(solution['remaining_strength'], 0)

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            LanchesterLinear(A0=-10, B0=50, alpha=0.5, beta=0.5)  # Negative force

        with self.assertRaises(ValueError):
            LanchesterLinear(A0=50, B0=50, alpha=-0.5, beta=0.5)  # Negative effectiveness


if __name__ == '__main__':
    unittest.main()