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

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            LanchesterLinear(A0=-10, B0=50, alpha=0.5, beta=0.5)  # Negative force

        with self.assertRaises(ValueError):
            LanchesterLinear(A0=50, B0=50, alpha=-0.5, beta=0.5)  # Negative effectiveness


if __name__ == '__main__':
    unittest.main()