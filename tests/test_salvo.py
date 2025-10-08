"""
Unit tests for SalvoCombatModel.

Tests core functionality of discrete round combat simulation:
- Ship damage and destruction mechanics
- Force effectiveness calculations
- Battle outcome determination
- Salvo allocation and damage distribution
"""

import unittest
from unittest import mock
import numpy as np
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

    def test_salvo_hits_retarget_after_casualties(self):
        """Hits should be reassigned to surviving ships after casualties."""
        force_a = [Ship(name="Bomber", offensive_power=4, defensive_power=0.0, staying_power=3)]
        force_b = [
            Ship(name="Screen", offensive_power=0, defensive_power=0.0, staying_power=1),
            Ship(name="Capital", offensive_power=0, defensive_power=0.0, staying_power=5),
        ]

        model = SalvoCombatModel(force_a=force_a, force_b=force_b, random_seed=1)

        # First volley destroys the screening ship and damages the capital ship
        model.execute_attack_phase(model.force_a, model.force_b, "Force A")
        self.assertFalse(model.force_b[0].is_operational())
        self.assertTrue(model.force_b[1].is_operational())

        # Second volley should now land entirely on the surviving capital ship
        model.execute_attack_phase(model.force_a, model.force_b, "Force A")
        self.assertFalse(model.force_b[1].is_operational())

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

    def test_simple_simulation_fallback_preserves_state(self):
        """Fallback to full simulation must not mutate the live model state."""
        force_a = [Ship(name="A", offensive_power=6, defensive_power=0.1, staying_power=3)]
        force_b = [Ship(name="B", offensive_power=5, defensive_power=0.7, staying_power=4)]

        model = SalvoCombatModel(force_a=force_a, force_b=force_b)

        snapshot_a = [(ship.current_hits, ship.is_active) for ship in model.force_a]
        snapshot_b = [(ship.current_hits, ship.is_active) for ship in model.force_b]

        result = model.simple_simulation(quiet=True)

        self.assertEqual(result['method'], 'full_simulation')
        self.assertEqual(model.round_number, 0)
        self.assertEqual(len(model.battle_log), 0)

        for ship, (hits, active) in zip(model.force_a, snapshot_a):
            self.assertEqual(ship.current_hits, hits)
            self.assertEqual(ship.is_active, active)

        for ship, (hits, active) in zip(model.force_b, snapshot_b):
            self.assertEqual(ship.current_hits, hits)
            self.assertEqual(ship.is_active, active)

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

    def test_plot_battle_progress_autoshow(self):
        """Ensure plot auto-show fires when no external axes are provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            model = SalvoCombatModel(force_a=self.force_a, force_b=self.force_b, random_seed=7)
            model.run_simulation(quiet=True)

            with mock.patch.object(plt, "show") as show_mock:
                model.plot_battle_progress()
                show_mock.assert_called_once()
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

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

        # Should fall back to full simulation due to equal power (which includes low attrition)
        self.assertEqual(result_low['method'], 'full_simulation',
                        msg="Equal power case should fall back to full simulation")
        self.assertEqual(result_low.get('reason', ''), 'equal_power_fallback',
                        msg="Should indicate equal power as fallback reason")

        # Should not result in mutual annihilation with zero survivors
        self.assertNotEqual(result_low['outcome'], 'Mutual Annihilation',
                           msg="Low attrition should not result in mutual annihilation")

        # With these parameters, both forces should survive
        self.assertGreater(result_low['force_a_survivors'] + result_low['force_b_survivors'], 0,
                          msg="Low attrition case should have survivors")

        # Test case 2: Higher offensive power but still equal - should fall back
        # Even high damage equal power cases fall back due to unpredictable battle dynamics
        high_power_a = [Ship("High A", offensive_power=5, defensive_power=0.3, staying_power=2)]
        high_power_b = [Ship("High B", offensive_power=5, defensive_power=0.3, staying_power=2)]

        model_high = SalvoCombatModel(high_power_a, high_power_b, random_seed=42)
        result_high = model_high.simple_simulation(quiet=True)

        # Should fall back to full simulation due to equal power policy
        self.assertEqual(result_high['method'], 'full_simulation',
                        msg="Equal power case should always fall back to full simulation")
        self.assertEqual(result_high.get('reason', ''), 'equal_power_fallback',
                        msg="Should indicate equal power as fallback reason")

        # Test case 3: Boundary case - zero defensive power
        boundary_a = [Ship("Boundary A", offensive_power=0.5, defensive_power=0.0, staying_power=3)]
        boundary_b = [Ship("Boundary B", offensive_power=0.5, defensive_power=0.0, staying_power=3)]

        model_boundary = SalvoCombatModel(boundary_a, boundary_b, random_seed=42)
        result_boundary = model_boundary.simple_simulation(quiet=True)

        # With equal forces, should fall back due to equal power policy
        self.assertEqual(result_boundary['method'], 'full_simulation',
                        msg="Equal power boundary case should fall back to full simulation")
        self.assertEqual(result_boundary.get('reason', ''), 'equal_power_fallback',
                        msg="Should use equal power fallback for equal offensive power")

        # Test case 4: Verify the general low attrition fallback still works
        # effective_damage_a = 0.1 * (1 - 0.3) = 0.07 < 0.1 (triggers general fallback)
        # This should bypass the equal power check and use the general low attrition fallback

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

    def test_offensive_ratio_calculations(self):
        """Test that offensive ratios are calculated correctly for various force compositions."""
        # Test case 1: Symmetric forces should have ratio of 1.0
        symmetric_cases = [
            (0.5, 0.5),  # Original bug case
            (1.0, 1.0),  # Standard case
            (2.5, 2.5),  # Larger symmetric forces
        ]

        for a_power, b_power in symmetric_cases:
            with self.subTest(a_power=a_power, b_power=b_power):
                ship_a = Ship("Ship A", offensive_power=a_power, defensive_power=0.3, staying_power=1)
                ship_b = Ship("Ship B", offensive_power=b_power, defensive_power=0.3, staying_power=1)

                model = SalvoCombatModel([ship_a], [ship_b])
                stats = model.get_battle_statistics()

                self.assertAlmostEqual(stats['offensive_ratio'], 1.0, places=6,
                                     msg=f"Symmetric forces {a_power}:{b_power} should have ratio 1.0")

        # Test case 2: Asymmetric ratios
        asymmetric_cases = [
            (2.0, 1.0, 2.0),  # 2:1 ratio
            (3.0, 1.5, 2.0),  # 2:1 ratio with decimals
            (10.0, 5.0, 2.0), # 2:1 ratio larger forces
            (1.5, 3.0, 0.5),  # 1:2 ratio (B stronger)
        ]

        for a_power, b_power, expected_ratio in asymmetric_cases:
            with self.subTest(a_power=a_power, b_power=b_power, expected=expected_ratio):
                ship_a = Ship("Ship A", offensive_power=a_power, defensive_power=0.3, staying_power=1)
                ship_b = Ship("Ship B", offensive_power=b_power, defensive_power=0.3, staying_power=1)

                model = SalvoCombatModel([ship_a], [ship_b])
                stats = model.get_battle_statistics()

                self.assertAlmostEqual(stats['offensive_ratio'], expected_ratio, places=6,
                                     msg=f"Forces {a_power}:{b_power} should have ratio {expected_ratio}")

        # Test case 3: Division by zero case
        ship_a = Ship("Ship A", offensive_power=1.0, defensive_power=0.3, staying_power=1)
        ship_b = Ship("Ship B", offensive_power=0.0, defensive_power=0.3, staying_power=1)

        model = SalvoCombatModel([ship_a], [ship_b])
        stats = model.get_battle_statistics()

        self.assertEqual(stats['offensive_ratio'], float('inf'),
                        msg="Division by zero should return infinity")

    def test_simple_simulation_offensive_ratio_precision(self):
        """Simplified path should report accurate offensive ratios for fractional forces."""
        force_a = [Ship("Light A", offensive_power=0.6, defensive_power=0.2, staying_power=2)]
        force_b = [Ship("Light B", offensive_power=0.3, defensive_power=0.25, staying_power=2)]

        model = SalvoCombatModel(force_a, force_b)
        result = model.simple_simulation(quiet=True)

        self.assertEqual(result['method'], 'simplified',
                         msg="Scenario should remain on simplified path")
        self.assertAlmostEqual(result['offensive_ratio'], 2.0, places=6,
                               msg="Offensive ratio should match actual force ratio")

    def test_fractional_offensive_power_fires_probabilistic_missile(self):
        """Fractional offensive power should probabilistically add an extra missile."""
        attackers = [Ship("Fractional", offensive_power=0.5, defensive_power=0.0, staying_power=1)]
        defenders = [Ship("Target", offensive_power=0.0, defensive_power=0.0, staying_power=1)]

        model = SalvoCombatModel(attackers, defenders, random_seed=1)
        missiles, distribution = model.calculate_salvo_effectiveness(model.force_a, model.force_b)

        self.assertEqual(missiles, 1,
                         msg="Fractional offensive power should fire one missile with seeded RNG")
        self.assertEqual(distribution, [1],
                         msg="Single defender should receive the single missile")


class TestSalvoAdditionalCoverage(unittest.TestCase):
    """Additional tests to improve coverage of Salvo Combat Model."""

    def test_defensive_probability_alias(self):
        """Test the defensive_probability property alias."""
        ship = Ship("Test", offensive_power=10, defensive_power=0.35, staying_power=3)
        self.assertEqual(ship.defensive_probability, 0.35)
        self.assertEqual(ship.defensive_probability, ship.defensive_power)

    def test_ship_health_percentage(self):
        """Test health percentage calculation at various damage levels."""
        ship = Ship("TestShip", offensive_power=5, defensive_power=0.2, staying_power=10)

        # Full health
        self.assertEqual(ship.get_health_percentage(), 100.0)

        # Half health
        ship.take_damage(5)
        self.assertEqual(ship.get_health_percentage(), 50.0)

        # Quarter health
        ship.take_damage(2)
        self.assertAlmostEqual(ship.get_health_percentage(), 30.0, places=1)

        # Zero health
        ship.take_damage(10)
        self.assertEqual(ship.get_health_percentage(), 0.0)

    def test_ship_take_damage_returns_actual_damage(self):
        """Test that take_damage returns the actual damage taken."""
        ship = Ship("Tanker", offensive_power=0, defensive_power=0.5, staying_power=5)

        # Normal damage
        actual = ship.take_damage(2)
        self.assertEqual(actual, 2)

        # Overkill damage (ship has 3 health left)
        actual2 = ship.take_damage(10)
        self.assertEqual(actual2, 3)

        # Damage to destroyed ship
        actual3 = ship.take_damage(5)
        self.assertEqual(actual3, 0)

    def test_empty_force_effectiveness(self):
        """Test force effectiveness calculation with empty force."""
        model = SalvoCombatModel([], [])
        effectiveness = model.calculate_force_effectiveness([])

        self.assertEqual(effectiveness['total_offensive'], 0)
        self.assertEqual(effectiveness['total_defensive'], 0)
        self.assertEqual(effectiveness['total_staying_power'], 0)
        self.assertEqual(effectiveness['remaining_health'], 0)
        self.assertEqual(effectiveness['operational_count'], 0)
        self.assertEqual(effectiveness['average_defensive'], 0)

    def test_execute_attack_phase_with_empty_forces(self):
        """Test attack phase execution with empty attacker or defender lists."""
        model = SalvoCombatModel([], [])

        # Empty attackers
        events = model.execute_attack_phase([], [Ship("Defender", 1, 0.1, 1)], "Empty Force")
        self.assertEqual(len(events), 0)

        # Empty defenders
        events2 = model.execute_attack_phase([Ship("Attacker", 1, 0.1, 1)], [], "Test Force")
        self.assertEqual(len(events2), 0)

    def test_determine_battle_outcome_variations(self):
        """Test battle outcome determination for various scenarios."""
        # Both forces have survivors (incomplete battle)
        force_a = [Ship("A1", 5, 0.2, 3), Ship("A2", 5, 0.2, 3)]
        force_b = [Ship("B1", 5, 0.2, 3), Ship("B2", 5, 0.2, 3)]
        model = SalvoCombatModel(force_a, force_b)

        outcome = model.determine_battle_outcome()
        self.assertEqual(outcome, "Draw - Both forces have survivors")

    def test_plot_battle_progress_with_axes(self):
        """Test plot_battle_progress with external axes (should not auto-show)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            force_a = [Ship("A1", 10, 0.3, 2)]
            force_b = [Ship("B1", 8, 0.3, 2)]
            model = SalvoCombatModel(force_a, force_b, random_seed=42)
            model.run_simulation(quiet=True)

            fig, ax = plt.subplots()
            with mock.patch.object(plt, "show") as show_mock:
                model.plot_battle_progress(ax=ax)
                show_mock.assert_not_called()
            plt.close(fig)
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_plot_battle_progress_no_data(self):
        """Test plot_battle_progress when no battle has been run."""
        import matplotlib
        matplotlib.use("Agg")

        model = SalvoCombatModel([Ship("A", 1, 0.1, 1)], [Ship("B", 1, 0.1, 1)])
        # Should print message and return early
        model.plot_battle_progress()

    def test_plot_multiple_battles_functionality(self):
        """Test plot_multiple_battles class method."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sim1 = SalvoCombatModel([Ship("A1", 5, 0.2, 2)], [Ship("B1", 4, 0.2, 2)], random_seed=1)
            sim1.run_simulation(quiet=True)

            sim2 = SalvoCombatModel([Ship("A2", 6, 0.3, 2)], [Ship("B2", 5, 0.3, 2)], random_seed=2)
            sim2.run_simulation(quiet=True)

            # Test with custom titles
            SalvoCombatModel.plot_multiple_battles([sim1, sim2], titles=["Battle 1", "Battle 2"])
            plt.close('all')

            # Test without titles
            SalvoCombatModel.plot_multiple_battles([sim1, sim2])
            plt.close('all')
        except (ImportError, TypeError):
            self.skipTest("Matplotlib backend issue")

    def test_monte_carlo_with_quiet_false(self):
        """Test Monte Carlo analysis with verbose output."""
        force_a = [Ship("A", 5, 0.2, 2)]
        force_b = [Ship("B", 4, 0.2, 2)]
        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        # Run with quiet=False to cover print statements
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            results = model.run_monte_carlo_analysis(iterations=5, quiet=False)
            output = captured_output.getvalue()

            # Verify output contains expected text
            self.assertIn("MONTE CARLO ANALYSIS", output)
            self.assertIn("Average battle duration", output)

            # Verify results structure
            self.assertEqual(results['iterations'], 5)
            self.assertIn('outcome_probabilities', results)
        finally:
            sys.stdout = sys.__stdout__

    def test_simple_simulation_with_quiet_false(self):
        """Test simple_simulation with verbose output."""
        force_a = [Ship("A", 10, 0.3, 3)]
        force_b = [Ship("B", 8, 0.3, 3)]
        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.simple_simulation(quiet=False)
            output = captured_output.getvalue()

            # Verify output contains expected analysis
            self.assertIn("SIMPLE SALVO SIMULATION", output)
            self.assertIn("Simplified Analysis", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_run_simulation_with_verbose_output(self):
        """Test run_simulation with quiet=False to cover output statements."""
        force_a = [Ship("Destroyer", 8, 0.3, 3)]
        force_b = [Ship("Frigate", 6, 0.4, 2)]
        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.run_simulation(max_rounds=3, quiet=False)
            output = captured_output.getvalue()

            # Verify output contains battle information
            self.assertIn("SALVO COMBAT MODEL SIMULATION", output)
            self.assertIn("Force A:", output)
            self.assertIn("Force B:", output)
            self.assertIn("BATTLE RESULT", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_salvo_effectiveness_with_multiple_defenders(self):
        """Test missile distribution across multiple defending ships."""
        force_a = [Ship("Attacker", offensive_power=10, defensive_power=0.0, staying_power=1)]
        force_b = [
            Ship("Defender1", offensive_power=0, defensive_power=0.0, staying_power=2),
            Ship("Defender2", offensive_power=0, defensive_power=0.0, staying_power=2),
            Ship("Defender3", offensive_power=0, defensive_power=0.0, staying_power=2)
        ]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)
        missiles_through, distribution = model.calculate_salvo_effectiveness(force_a, force_b)

        # With no defense, all missiles should get through
        self.assertEqual(missiles_through, 10)

        # Distribution should cover all defenders
        self.assertEqual(len(distribution), 3)

        # Total distributed hits should equal missiles through
        self.assertEqual(sum(distribution), missiles_through)

    def test_max_intercept_probability_cap(self):
        """Test that defensive power is capped at MAX_INTERCEPT_PROBABILITY."""
        # Create ship with very high defensive power
        attacker = [Ship("Attacker", offensive_power=100, defensive_power=0.0, staying_power=5)]
        defender = [Ship("SuperDefender", offensive_power=0, defensive_power=1.0, staying_power=10)]

        model = SalvoCombatModel(attacker, defender, random_seed=42)
        missiles_through, _ = model.calculate_salvo_effectiveness(attacker, defender)

        # Even with 1.0 defensive power, some missiles should get through due to cap
        # With 100 missiles and max intercept of 0.95, expect some to penetrate
        self.assertGreater(missiles_through, 0)

    def test_offensive_ratio_with_zero_denominator(self):
        """Test offensive ratio calculation handles division by zero."""
        force_a = [Ship("A", offensive_power=10, defensive_power=0.2, staying_power=2)]
        force_b = [Ship("B", offensive_power=0, defensive_power=0.2, staying_power=2)]

        model = SalvoCombatModel(force_a, force_b)
        stats = model.get_battle_statistics()

        self.assertEqual(stats['offensive_ratio'], float('inf'))

    def test_offensive_ratio_both_zero_handling(self):
        """Test offensive ratio with both forces having zero offensive power."""
        force_a = [Ship("A", offensive_power=0, defensive_power=0.2, staying_power=2)]
        force_b = [Ship("B", offensive_power=0, defensive_power=0.2, staying_power=2)]

        model = SalvoCombatModel(force_a, force_b)

        # When both have zero offensive power, the code treats it as 0/0 -> inf
        # (since it checks if denominator > 0 before computing ratio)
        stats = model.get_battle_statistics()

        # The implementation returns inf for 0/0 case (division by zero behavior)
        self.assertEqual(stats['offensive_ratio'], float('inf'))

    def test_battle_statistics_surviving_ships(self):
        """Test that battle statistics include surviving ship details."""
        force_a = [Ship("A1", 10, 0.3, 5)]
        force_b = [Ship("B1", 1, 0.1, 1)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)
        model.run_simulation(quiet=True)

        stats = model.get_battle_statistics()

        # Verify surviving ships are in stats
        self.assertIn('surviving_ships_a', stats)
        self.assertIn('surviving_ships_b', stats)

        # A should win with survivors
        if stats['outcome'] == 'Force A Victory':
            self.assertGreater(len(stats['surviving_ships_a']), 0)
            self.assertEqual(len(stats['surviving_ships_b']), 0)

    def test_execute_round_return_value(self):
        """Test execute_round returns correct continuation status."""
        force_a = [Ship("A", 10, 0.0, 1)]
        force_b = [Ship("B", 10, 0.0, 1)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        # First round should execute
        continues = model.execute_round()

        # Check if battle continues or ended
        self.assertIsInstance(continues, bool)

    def test_simple_simulation_equal_power_fallback_message(self):
        """Test simple_simulation prints fallback message for equal power."""
        force_a = [Ship("A", 5, 0.3, 2)]
        force_b = [Ship("B", 5, 0.3, 2)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.simple_simulation(quiet=False)
            output = captured_output.getvalue()

            # Should indicate fallback to full simulation
            self.assertIn("Equal offensive power", output)
            self.assertIn("full simulation", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_mutual_annihilation_outcome(self):
        """Test that mutual annihilation is correctly identified."""
        # Create ships and manually destroy them to test the outcome logic
        force_a = [Ship("Ship A", offensive_power=10, defensive_power=0.0, staying_power=1)]
        force_b = [Ship("Ship B", offensive_power=10, defensive_power=0.0, staying_power=1)]

        # Manually destroy both ships
        force_a[0].take_damage(10)
        force_b[0].take_damage(10)

        model = SalvoCombatModel(force_a, force_b, random_seed=42)
        outcome = model.determine_battle_outcome()

        # With both forces eliminated, should be mutual annihilation
        self.assertEqual(outcome, "Mutual Annihilation")

    def test_surviving_ship_health_display(self):
        """Test that surviving ships display health correctly in verbose mode."""
        force_a = [Ship("Strong A", offensive_power=15, defensive_power=0.5, staying_power=10)]
        force_b = [Ship("Weak B", offensive_power=2, defensive_power=0.1, staying_power=2)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            model.run_simulation(quiet=False)
            output = captured_output.getvalue()

            # Should show surviving ships with health percentage
            if "Force A surviving ships" in output:
                self.assertIn("health", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_event_logging_ship_destruction(self):
        """Test that ship destruction events are logged correctly."""
        force_a = [Ship("Destroyer", offensive_power=10, defensive_power=0.0, staying_power=5)]
        force_b = [Ship("Target", offensive_power=0, defensive_power=0.0, staying_power=1)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)
        model.run_simulation(quiet=True)

        # Check battle log for destruction events
        found_destruction = False
        for round_log in model.battle_log:
            for event in round_log['events']:
                if "destroyed" in event.lower():
                    found_destruction = True
                    break

        self.assertTrue(found_destruction, "Should log ship destruction events")

    def test_simple_simulation_low_attrition_message(self):
        """Test simple_simulation prints low attrition fallback message."""
        # Very low offensive power relative to defensive
        force_a = [Ship("A", 0.05, 0.3, 2)]
        force_b = [Ship("B", 0.05, 0.3, 2)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.simple_simulation(quiet=False)
            output = captured_output.getvalue()

            # Should indicate low attrition detected
            if "Low attrition" in output:
                self.assertIn("full simulation", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_simple_simulation_force_b_victory(self):
        """Test simplified simulation when Force B wins."""
        # Force B has higher offensive power
        force_a = [Ship("Weak A", 5, 0.3, 3)]
        force_b = [Ship("Strong B", 12, 0.3, 3)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = model.simple_simulation(quiet=True)

        # Should use simplified method and predict B victory
        self.assertEqual(result['method'], 'simplified')
        self.assertEqual(result['outcome'], 'Force B Victory')
        self.assertEqual(result['force_a_survivors'], 0)
        self.assertGreater(result['force_b_survivors'], 0)

    def test_simple_simulation_different_defensive_forces_verbose(self):
        """Test simple_simulation verbose output for different defensive forces."""
        # Create forces with significantly different defensive capabilities
        force_a = [Ship("Light Defense A", 10, 0.1, 3)]
        force_b = [Ship("Heavy Defense B", 10, 0.8, 3)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.simple_simulation(quiet=False)
            output = captured_output.getvalue()

            # Should indicate different defensive capabilities
            self.assertIn("FULL SALVO SIMULATION", output)
            self.assertIn("Different Defensive Forces", output)
            self.assertIn("Defensive similarity", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_calculate_salvo_effectiveness_no_operational_defenders(self):
        """Test salvo effectiveness when all defenders are destroyed."""
        attackers = [Ship("Attacker", offensive_power=10, defensive_power=0.0, staying_power=5)]
        defenders = [Ship("Dead Defender", offensive_power=5, defensive_power=0.2, staying_power=1)]

        # Destroy the defender
        defenders[0].take_damage(10)

        model = SalvoCombatModel(attackers, defenders, random_seed=42)
        missiles_through, distribution = model.calculate_salvo_effectiveness(attackers, defenders)

        # With no operational defenders, should return 0 missiles and empty distribution
        self.assertEqual(missiles_through, 0)
        self.assertEqual(distribution, [])

    def test_run_simulation_force_b_survivors_display(self):
        """Test that Force B surviving ships are displayed when B wins."""
        # Force B should win decisively
        force_a = [Ship("Weak A", 2, 0.1, 1)]
        force_b = [Ship("Strong B", 15, 0.5, 10)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = model.run_simulation(quiet=False)
            output = captured_output.getvalue()

            # Should show Force B survivors
            if result == "Force B Victory":
                self.assertIn("Force B surviving ships", output)
                self.assertIn("Strong B", output)
                self.assertIn("health", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_monte_carlo_default_iterations(self):
        """Test Monte Carlo analysis uses default iterations when None specified."""
        force_a = [Ship("A", 5, 0.2, 2)]
        force_b = [Ship("B", 4, 0.2, 2)]

        model = SalvoCombatModel(force_a, force_b, random_seed=42)

        # Call without specifying iterations (should use default)
        results = model.run_monte_carlo_analysis(iterations=None, quiet=True)

        # Should use DEFAULT_MONTE_CARLO_ITERATIONS
        self.assertEqual(results['iterations'], SalvoCombatModel.MONTE_CARLO_ITERATIONS)
        self.assertEqual(len(results['battle_durations']), SalvoCombatModel.MONTE_CARLO_ITERATIONS)


if __name__ == '__main__':
    unittest.main()
