"""
Unit tests for SalvoCombatModel.

Tests core functionality of discrete round combat simulation:
- Ship damage and destruction mechanics
- Force effectiveness calculations
- Battle outcome determination
- Salvo allocation and damage distribution
"""

import unittest
from models import SalvoCombatModel, Ship


class TestSalvoCombatModel(unittest.TestCase):
    """Test cases for Salvo Combat Model core functionality."""

    def setUp(self):
        """Set up test fixtures with known ship configurations."""
        # Standard force A
        self.force_a = [
            Ship(name="Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
            Ship(name="Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5)
        ]

        # Standard force B
        self.force_b = [
            Ship(name="Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
            Ship(name="Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4)
        ]

        # Create test model
        self.model = SalvoCombatModel(force_a=self.force_a, force_b=self.force_b)

        # Single ship forces for precise testing
        self.single_a = [Ship(name="Solo A", offensive_power=10, defensive_power=0.2, staying_power=5)]
        self.single_b = [Ship(name="Solo B", offensive_power=8, defensive_power=0.3, staying_power=4)]
        self.single_model = SalvoCombatModel(force_a=self.single_a, force_b=self.single_b)

    def test_ship_creation(self):
        """Test Ship class initialization and properties."""
        ship = Ship(name="Test Ship", offensive_power=15, defensive_power=0.5, staying_power=10)

        self.assertEqual(ship.name, "Test Ship")
        self.assertEqual(ship.offensive_power, 15)
        self.assertEqual(ship.defensive_power, 0.5)
        self.assertEqual(ship.staying_power, 10)
        self.assertEqual(ship.current_health, 10)  # Should start at full health
        self.assertTrue(ship.is_operational())

    def test_ship_damage_mechanics(self):
        """Test ship damage and destruction mechanics."""
        ship = Ship(name="Target", offensive_power=10, defensive_power=0.0, staying_power=3)

        # Ship should start active
        self.assertTrue(ship.is_operational())
        self.assertEqual(ship.current_health, 3)

        # Apply damage
        ship.take_damage(1)
        self.assertTrue(ship.is_operational())
        self.assertEqual(ship.current_health, 2)

        # Apply more damage
        ship.take_damage(1)
        self.assertTrue(ship.is_operational())
        self.assertEqual(ship.current_health, 1)

        # Final damage should destroy ship
        ship.take_damage(1)
        self.assertFalse(ship.is_operational())
        self.assertEqual(ship.current_health, 0)

        # Additional damage shouldn't reduce health below zero
        ship.take_damage(5)
        self.assertEqual(ship.current_health, 0)

    def test_force_effectiveness_calculation(self):
        """Test accurate calculation of force combat statistics."""
        effectiveness_a = self.model.calculate_force_effectiveness(self.force_a)

        # Verify calculated values
        expected_offensive = 8 + 12  # Sum of offensive powers
        expected_staying = 3 + 5    # Sum of staying powers
        expected_avg_defensive = (0.3 + 0.4) / 2  # Average defensive probability

        self.assertEqual(effectiveness_a['total_offensive'], expected_offensive)
        self.assertEqual(effectiveness_a['total_staying_power'], expected_staying)
        self.assertAlmostEqual(effectiveness_a['average_defensive'], expected_avg_defensive, places=2)

    def test_salvo_effectiveness_calculation(self):
        """Test salvo damage calculation and allocation."""
        # Test with single ship for predictable results
        attacking_force = [Ship(name="Attacker", offensive_power=10, defensive_power=0.0, staying_power=3)]
        defending_force = [Ship(name="Defender", offensive_power=5, defensive_power=0.0, staying_power=2)]

        total_damage, damage_per_ship = self.model.calculate_salvo_effectiveness(attacking_force, defending_force)

        # With no defensive probability, total damage should equal offensive power
        self.assertEqual(total_damage, 10)
        self.assertEqual(len(damage_per_ship), 1)
        self.assertEqual(damage_per_ship[0], 10)  # All damage to single defender

    def test_battle_progression(self):
        """Test that battles progress logically through rounds."""
        # Run a short simulation
        outcome = self.single_model.run_simulation(max_rounds=5, quiet=True)
        result = self.single_model.get_battle_statistics()

        # Battle should end or be in progress
        self.assertIn(result['outcome'], ['Force A Victory', 'Force B Victory', 'Ongoing'])
        self.assertGreaterEqual(result['rounds'], 1)
        self.assertLessEqual(result['rounds'], 5)

        # Survivor counts should be consistent
        total_survivors = result['force_a_survivors'] + result['force_b_survivors']
        if result['outcome'] != 'Ongoing':
            self.assertGreaterEqual(total_survivors, 0)
            if result['outcome'] == 'Force A Victory':
                self.assertGreater(result['force_a_survivors'], 0)
                self.assertEqual(result['force_b_survivors'], 0)
            elif result['outcome'] == 'Force B Victory':
                self.assertEqual(result['force_a_survivors'], 0)
                self.assertGreater(result['force_b_survivors'], 0)

    def test_simple_simulation_defensive_similarity(self):
        """Test simple_simulation behavior with similar defensive capabilities."""
        # Create forces with similar defensive probabilities
        similar_force_a = [Ship(name="A1", offensive_power=10, defensive_power=0.3, staying_power=4)]
        similar_force_b = [Ship(name="B1", offensive_power=8, defensive_power=0.35, staying_power=3)]

        model = SalvoCombatModel(force_a=similar_force_a, force_b=similar_force_b)
        result = model.simple_simulation(quiet=True)

        # Should use simplified method
        self.assertEqual(result['method'], 'simplified')
        self.assertIn('defensive_similarity', result)

        # Should have reasonable outcome
        self.assertIn(result['outcome'], ['Force A Victory', 'Force B Victory', 'Mutual Annihilation'])

    def test_simple_simulation_fallback_to_full(self):
        """Test simple_simulation falls back to full simulation when needed."""
        # Create forces with very different defensive capabilities
        different_force_a = [Ship(name="A1", offensive_power=10, defensive_power=0.1, staying_power=4)]
        different_force_b = [Ship(name="B1", offensive_power=8, defensive_power=0.8, staying_power=3)]

        model = SalvoCombatModel(force_a=different_force_a, force_b=different_force_b)
        result = model.simple_simulation(quiet=True)

        # Should use full simulation method
        self.assertEqual(result['method'], 'full_simulation')
        self.assertIn('defensive_similarity', result)

    def test_reference_seeded_battle(self):
        """Exercise deterministic reference run highlighted during review."""

        force_a = [
            Ship(name="Destroyer", offensive_power=10, defensive_power=0.3, staying_power=5),
            Ship(name="Frigate", offensive_power=8, defensive_power=0.2, staying_power=4),
        ]
        force_b = [
            Ship(name="Cruiser", offensive_power=9, defensive_power=0.25, staying_power=4),
            Ship(name="Corvette", offensive_power=7, defensive_power=0.35, staying_power=3),
        ]

        model = SalvoCombatModel(force_a=force_a, force_b=force_b, random_seed=42)
        outcome = model.run_simulation(quiet=True)
        stats = model.get_battle_statistics()

        self.assertEqual(outcome, 'Force A Victory')
        self.assertEqual(stats['rounds'], 1)
        self.assertEqual(stats['force_a_survivors'], 2)
        self.assertEqual(stats['force_b_survivors'], 0)

    def test_battle_statistics(self):
        """Test battle statistics calculation."""
        # Run a battle and get statistics
        self.model.run_simulation(max_rounds=3, quiet=True)
        stats = self.model.get_battle_statistics()

        # Verify required fields exist
        required_fields = ['outcome', 'rounds', 'force_a_survivors', 'force_b_survivors']
        for field in required_fields:
            self.assertIn(field, stats)

        # Verify logical consistency
        if stats['outcome'] == 'Force A Victory':
            self.assertGreater(stats['force_a_survivors'], 0)
            self.assertEqual(stats['force_b_survivors'], 0)
        elif stats['outcome'] == 'Force B Victory':
            self.assertEqual(stats['force_a_survivors'], 0)
            self.assertGreater(stats['force_b_survivors'], 0)

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        # Invalid ship parameters
        with self.assertRaises(ValueError):
            Ship(name="Bad Ship", offensive_power=-5, defensive_power=0.3, staying_power=2)

        with self.assertRaises(ValueError):
            Ship(name="Bad Ship", offensive_power=5, defensive_power=1.5, staying_power=2)  # > 1.0

        with self.assertRaises(ValueError):
            Ship(name="Bad Ship", offensive_power=5, defensive_power=0.3, staying_power=0)  # Zero staying power

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single ship vs single ship
        lone_a = [Ship(name="Lone A", offensive_power=5, defensive_power=0.2, staying_power=1)]
        lone_b = [Ship(name="Lone B", offensive_power=3, defensive_power=0.1, staying_power=1)]

        lone_model = SalvoCombatModel(force_a=lone_a, force_b=lone_b)
        outcome = lone_model.run_simulation(max_rounds=10, quiet=True)
        result = lone_model.get_battle_statistics()

        # Should resolve quickly
        self.assertLessEqual(result['rounds'], 5)
        self.assertIn(result['outcome'], ['Force A Victory', 'Force B Victory'])

        # Maximum defensive probability
        max_def_ship = Ship(name="Fortress", offensive_power=1, defensive_power=1.0, staying_power=1)
        self.assertEqual(max_def_ship.defensive_power, 1.0)

        # Zero offensive power
        zero_off_ship = Ship(name="Pacifist", offensive_power=0, defensive_power=0.5, staying_power=3)
        self.assertEqual(zero_off_ship.offensive_power, 0)

    def test_monte_carlo_consistency(self):
        """Test that Monte Carlo results are statistically reasonable."""
        # Note: This is a basic consistency check, not full statistical validation
        results = self.model.run_monte_carlo_analysis(iterations=10, quiet=True)

        # Should have reasonable structure
        self.assertIn('outcome_probabilities', results)
        self.assertIn('battle_durations', results)
        self.assertEqual(len(results['battle_durations']), 10)

        # Should have expected fields
        required_fields = ['outcome_probabilities', 'average_battle_duration', 'iterations']
        for field in required_fields:
            self.assertIn(field, results)

        # Outcome probabilities should be reasonable
        probabilities = results['outcome_probabilities']
        total_probability = sum(probabilities.values())
        self.assertAlmostEqual(total_probability, 100.0, places=1)  # Should sum to 100%

    def test_phantom_round_prevention_regression(self):
        """Regression test for phantom round bug prevention.

        Previously, execute_round() incremented round_number before checking
        if forces were operational, causing phantom rounds to be counted
        even when no combat occurred. This wasted max_rounds budget.

        The fix: Check operational forces BEFORE incrementing round_number.
        """
        # Test case 1: Empty forces should not increment round number
        force_a = []
        force_b = [Ship("Ship B", offensive_power=1, defensive_power=0.0, staying_power=1)]

        simulation = SalvoCombatModel(force_a, force_b, random_seed=42)

        initial_round = simulation.round_number
        result = simulation.execute_round()

        self.assertFalse(result, "Battle should not continue with empty force A")
        self.assertEqual(simulation.round_number, initial_round,
                        "Round number should not increment when no combat occurs")
        self.assertEqual(len(simulation.battle_log), 0,
                        "No round should be logged when no combat occurs")

        # Test case 2: Both forces eliminated should not increment round number
        force_a = [Ship("Dead Ship A", offensive_power=0, defensive_power=0.0, staying_power=1)]
        force_b = [Ship("Dead Ship B", offensive_power=0, defensive_power=0.0, staying_power=1)]

        # Manually eliminate both ships
        force_a[0].take_damage(10)
        force_b[0].take_damage(10)

        simulation2 = SalvoCombatModel(force_a, force_b, random_seed=42)

        initial_round = simulation2.round_number
        result = simulation2.execute_round()

        self.assertFalse(result, "Battle should not continue with both forces eliminated")
        self.assertEqual(simulation2.round_number, initial_round,
                        "Round number should not increment when both forces eliminated")

        # Test case 3: Sequential calls should not keep incrementing
        for i in range(3):
            round_before = simulation2.round_number
            result = simulation2.execute_round()
            round_after = simulation2.round_number

            self.assertFalse(result, f"Battle should not continue on attempt {i+1}")
            self.assertEqual(round_before, round_after,
                           f"Round number should not change on attempt {i+1}")

        # Test case 4: Normal combat should increment properly
        healthy_a = [Ship("Healthy A", offensive_power=1, defensive_power=0.0, staying_power=2)]
        healthy_b = [Ship("Healthy B", offensive_power=1, defensive_power=0.0, staying_power=2)]

        simulation3 = SalvoCombatModel(healthy_a, healthy_b, random_seed=42)

        initial_round = simulation3.round_number
        result = simulation3.execute_round()

        # This should actually increment since combat occurs
        if result:  # If battle continues
            self.assertEqual(simulation3.round_number, initial_round + 1,
                           "Round number should increment when actual combat occurs")
            self.assertEqual(len(simulation3.battle_log), 1,
                           "One round should be logged when combat occurs")

    def test_empty_force_handling_fix_regression(self):
        """Regression test for empty force handling fix.

        Previously, empty force lists would return "Mutual Annihilation" which
        is misleading when there was no battle to begin with.

        The fix properly classifies empty initial forces with appropriate outcomes.
        """

        # Test case 1: Both forces empty → No Battle
        empty_model = SalvoCombatModel([], [])
        result = empty_model.run_simulation(quiet=True)
        stats = empty_model.get_battle_statistics()

        self.assertEqual(result, "No Battle - Both Forces Empty",
                        msg="Empty forces should result in 'No Battle', not 'Mutual Annihilation'")
        self.assertEqual(stats['rounds'], 0,
                        msg="No battle should mean 0 rounds")
        self.assertEqual(stats['force_a_survivors'], 0)
        self.assertEqual(stats['force_b_survivors'], 0)

        # Test case 2: Force A empty → Force B wins
        normal_ships = [Ship('Winner', offensive_power=10, defensive_power=0.3, staying_power=2)]
        empty_vs_normal = SalvoCombatModel([], normal_ships)
        result = empty_vs_normal.run_simulation(quiet=True)

        self.assertEqual(result, "Force B Victory - Force A Empty",
                        msg="Force A empty should result in Force B victory")
        self.assertEqual(empty_vs_normal.round_number, 0,
                        msg="Should not execute any combat rounds")

        # Test case 3: Force B empty → Force A wins
        normal_vs_empty = SalvoCombatModel(normal_ships, [])
        result = normal_vs_empty.run_simulation(quiet=True)

        self.assertEqual(result, "Force A Victory - Force B Empty",
                        msg="Force B empty should result in Force A victory")
        self.assertEqual(normal_vs_empty.round_number, 0,
                        msg="Should not execute any combat rounds")

        # Test case 4: Verify normal battle still works correctly
        normal_a = [Ship('Ship A', offensive_power=10, defensive_power=0.3, staying_power=1)]
        normal_b = [Ship('Ship B', offensive_power=8, defensive_power=0.2, staying_power=1)]
        normal_battle = SalvoCombatModel(normal_a, normal_b, random_seed=42)
        result = normal_battle.run_simulation(quiet=True)

        # Should be a proper battle outcome, not empty force outcome
        self.assertNotIn("Empty", result,
                        msg="Normal forces should not trigger empty force outcomes")
        self.assertIn("Victory", result,
                     msg="Normal battle should result in victory")
        self.assertGreater(normal_battle.round_number, 0,
                          msg="Normal battle should execute at least one round")

    def test_simplified_path_low_attrition_fallback_regression(self):
        """Regression test for simplified path false mutual annihilation.

        Previously, simple_simulation would declare "Mutual Annihilation" for
        equal offensive power cases regardless of whether meaningful attrition
        was possible. This led to incorrect results where forces with very low
        offensive power would be declared mutually annihilated when they could
        actually survive indefinitely.

        The fix detects low attrition scenarios and falls back to full simulation.
        """

        # Test case 1: Very low offensive power - should fall back to full simulation
        low_power_a = [Ship("Low A", offensive_power=0.5, defensive_power=0.3, staying_power=2)]
        low_power_b = [Ship("Low B", offensive_power=0.5, defensive_power=0.3, staying_power=2)]

        model_low = SalvoCombatModel(low_power_a, low_power_b, random_seed=42)
        result_low = model_low.simple_simulation(quiet=True)

        # Should fall back to full simulation due to low effective damage
        self.assertEqual(result_low['method'], 'full_simulation',
                        msg="Low attrition case should fall back to full simulation")
        self.assertIn('low_attrition', result_low.get('reason', ''),
                     msg="Should indicate low attrition as fallback reason")

        # Should not result in mutual annihilation with zero survivors
        self.assertNotEqual(result_low['outcome'], 'Mutual Annihilation',
                           msg="Low attrition should not result in mutual annihilation")

        # With these parameters, both forces should survive
        self.assertGreater(result_low['force_a_survivors'] + result_low['force_b_survivors'], 0,
                          msg="Low attrition case should have survivors")

        # Test case 2: Higher offensive power - should use simplified path
        high_power_a = [Ship("High A", offensive_power=5, defensive_power=0.3, staying_power=2)]
        high_power_b = [Ship("High B", offensive_power=5, defensive_power=0.3, staying_power=2)]

        model_high = SalvoCombatModel(high_power_a, high_power_b, random_seed=42)
        result_high = model_high.simple_simulation(quiet=True)

        # Should use simplified method for meaningful attrition
        self.assertEqual(result_high.get('method', 'simplified'), 'simplified',
                        msg="High damage case should use simplified method")

        # Test case 3: Boundary case - zero defensive power
        boundary_a = [Ship("Boundary A", offensive_power=0.5, defensive_power=0.0, staying_power=3)]
        boundary_b = [Ship("Boundary B", offensive_power=0.5, defensive_power=0.0, staying_power=3)]

        model_boundary = SalvoCombatModel(boundary_a, boundary_b, random_seed=42)
        result_boundary = model_boundary.simple_simulation(quiet=True)

        # With effective_damage = 0.5 and equal forces, should fall back for equal power low attrition
        self.assertEqual(result_boundary['method'], 'full_simulation',
                        msg="Equal power boundary case should fall back to full simulation")

        # Test case 4: Verify the thresholds work correctly
        # effective_damage_a = 0.5 * (1 - 0.3) = 0.35 < 1.0 (triggers equal power fallback)
        # effective_damage_a = 0.1 * (1 - 0.3) = 0.07 < 0.1 (triggers general fallback)

        very_low_a = [Ship("Very Low A", offensive_power=0.1, defensive_power=0.3, staying_power=2)]
        very_low_b = [Ship("Very Low B", offensive_power=0.1, defensive_power=0.3, staying_power=2)]

        model_very_low = SalvoCombatModel(very_low_a, very_low_b, random_seed=42)
        result_very_low = model_very_low.simple_simulation(quiet=True)

        # Should trigger the general low attrition fallback (effective_damage < 0.1)
        self.assertEqual(result_very_low['method'], 'full_simulation',
                        msg="Very low damage should fall back to full simulation")
        self.assertEqual(result_very_low['reason'], 'low_attrition_fallback',
                        msg="Should use general low attrition fallback")

        # Test case 5: Ensure high damage cases still work with simplified path
        strong_a = [Ship("Strong A", offensive_power=10, defensive_power=0.2, staying_power=3)]
        strong_b = [Ship("Strong B", offensive_power=8, defensive_power=0.2, staying_power=3)]

        model_strong = SalvoCombatModel(strong_a, strong_b, random_seed=42)
        result_strong = model_strong.simple_simulation(quiet=True)

        # Should use simplified method and produce clear winner
        self.assertEqual(result_strong.get('method', 'simplified'), 'simplified',
                        msg="High damage unequal forces should use simplified method")
        self.assertEqual(result_strong['outcome'], 'Force A Victory',
                        msg="Stronger force A should win in simplified calculation")


if __name__ == '__main__':
    unittest.main()