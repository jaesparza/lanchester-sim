#!/usr/bin/env python3
"""
Test script to identify subtle bugs and logic errors that might not be caught by edge case testing.
This focuses on mathematical correctness, consistency issues, and logic errors.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship

def test_linear_law_invariant():
    """Test if Linear Law maintains its mathematical invariant correctly."""
    print("="*60)
    print("TESTING LINEAR LAW MATHEMATICAL INVARIANT")
    print("="*60)

    issues = []

    # The Linear Law invariant should be: αA₀ - βB₀ = α*A(t) - β*B(t) at battle end
    # But during active combat, forces follow: A(t) = A₀ - βt, B(t) = B₀ - αt

    battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution = battle.analytical_solution()

    # Check the reported linear_advantage
    expected_advantage = battle.alpha * battle.A0 - battle.beta * battle.B0
    if abs(solution['linear_advantage'] - expected_advantage) > 1e-10:
        issues.append(f"Linear advantage calculation error: got {solution['linear_advantage']}, expected {expected_advantage}")

    # Check that during battle, the trajectories follow the correct equations
    t = solution['time']
    A_t = solution['A']
    B_t = solution['B']
    t_end = solution['battle_end_time']

    # Find a point during active combat
    mid_idx = len(t) // 4
    if t[mid_idx] < t_end:
        expected_A = battle.A0 - battle.beta * t[mid_idx]
        expected_B = battle.B0 - battle.alpha * t[mid_idx]

        if abs(A_t[mid_idx] - expected_A) > 0.1:
            issues.append(f"Linear trajectory A error at t={t[mid_idx]:.2f}: got {A_t[mid_idx]:.2f}, expected {expected_A:.2f}")
        if abs(B_t[mid_idx] - expected_B) > 0.1:
            issues.append(f"Linear trajectory B error at t={t[mid_idx]:.2f}: got {B_t[mid_idx]:.2f}, expected {expected_B:.2f}")

    print(f"Linear Law invariant check: {len(issues)} issues found")
    return issues

def test_square_law_invariant():
    """Test if Square Law maintains its mathematical invariant correctly."""
    print("\n" + "="*60)
    print("TESTING SQUARE LAW MATHEMATICAL INVARIANT")
    print("="*60)

    issues = []

    # The Square Law invariant is: α*A²(t) - β*B²(t) = α*A₀² - β*B₀²
    battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution = battle.analytical_solution()

    expected_invariant = battle.alpha * battle.A0**2 - battle.beta * battle.B0**2
    if abs(solution['invariant'] - expected_invariant) > 1e-10:
        issues.append(f"Square Law invariant calculation error: got {solution['invariant']}, expected {expected_invariant}")

    # Check invariant preservation during battle
    t = solution['time']
    A_t = solution['A']
    B_t = solution['B']
    t_end = solution['battle_end_time']

    # Check at multiple points during battle
    for i in range(0, len(t), len(t)//10):
        if t[i] < t_end and A_t[i] > 0 and B_t[i] > 0:
            current_invariant = battle.alpha * A_t[i]**2 - battle.beta * B_t[i]**2
            error = abs(current_invariant - expected_invariant)
            if error > abs(expected_invariant) * 0.01:  # 1% tolerance
                issues.append(f"Square Law invariant violated at t={t[i]:.2f}: error={error:.2f}")
                break

    print(f"Square Law invariant check: {len(issues)} issues found")
    return issues

def test_battle_end_consistency():
    """Test consistency between battle end calculations and trajectory endpoints."""
    print("\n" + "="*60)
    print("TESTING BATTLE END CONSISTENCY")
    print("="*60)

    issues = []

    # Test Linear Law
    battle_linear = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    winner, remaining, t_end = battle_linear.calculate_battle_outcome()
    solution = battle_linear.analytical_solution()

    # Find the index closest to battle end time
    t = solution['time']
    t_end_idx = np.argmin(np.abs(t - t_end))

    # Check that the trajectory values at t_end match the calculated outcome
    if winner == 'A':
        if abs(solution['A'][t_end_idx] - remaining) > 1.0:
            issues.append(f"Linear Law: A trajectory at t_end ({solution['A'][t_end_idx]:.2f}) doesn't match calculated remaining ({remaining:.2f})")
        if solution['B'][t_end_idx] > 0.1:
            issues.append(f"Linear Law: B should be eliminated but trajectory shows {solution['B'][t_end_idx]:.2f}")
    elif winner == 'B':
        if abs(solution['B'][t_end_idx] - remaining) > 1.0:
            issues.append(f"Linear Law: B trajectory at t_end ({solution['B'][t_end_idx]:.2f}) doesn't match calculated remaining ({remaining:.2f})")
        if solution['A'][t_end_idx] > 0.1:
            issues.append(f"Linear Law: A should be eliminated but trajectory shows {solution['A'][t_end_idx]:.2f}")

    # Test Square Law
    battle_square = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    winner, remaining, invariant = battle_square.calculate_battle_outcome()
    t_end = battle_square.calculate_battle_end_time(winner, remaining, invariant)
    solution = battle_square.analytical_solution()

    if not np.isinf(t_end):  # Skip infinite battle times
        t = solution['time']
        t_end_idx = np.argmin(np.abs(t - t_end))

        if winner == 'A':
            if abs(solution['A'][t_end_idx] - remaining) > 1.0:
                issues.append(f"Square Law: A trajectory at t_end ({solution['A'][t_end_idx]:.2f}) doesn't match calculated remaining ({remaining:.2f})")
            if solution['B'][t_end_idx] > 0.1:
                issues.append(f"Square Law: B should be eliminated but trajectory shows {solution['B'][t_end_idx]:.2f}")
        elif winner == 'B':
            if abs(solution['B'][t_end_idx] - remaining) > 1.0:
                issues.append(f"Square Law: B trajectory at t_end ({solution['B'][t_end_idx]:.2f}) doesn't match calculated remaining ({remaining:.2f})")
            if solution['A'][t_end_idx] > 0.1:
                issues.append(f"Square Law: A should be eliminated but trajectory shows {solution['A'][t_end_idx]:.2f}")

    print(f"Battle end consistency check: {len(issues)} issues found")
    return issues

def test_salvo_combat_logic():
    """Test Salvo Combat Model for logical consistency issues."""
    print("\n" + "="*60)
    print("TESTING SALVO COMBAT MODEL LOGIC")
    print("="*60)

    issues = []

    # Test 1: Check that defensive probability calculation makes sense
    force_a = [Ship("Test A", offensive_power=10, defensive_power=0.5, staying_power=3)]
    force_b = [Ship("Test B", offensive_power=10, defensive_power=0.3, staying_power=3)]
    sim = SalvoCombatModel(force_a, force_b, random_seed=42)

    # Test salvo effectiveness calculation
    missiles_through, hits_dist = sim.calculate_salvo_effectiveness(force_a, force_b)

    # With 10 missiles and 0.3 defensive power, we should intercept some but not all
    if missiles_through < 0 or missiles_through > 10:
        issues.append(f"Salvo effectiveness out of bounds: {missiles_through} missiles through from 10 fired")

    # Test 2: Check that damage application is consistent
    initial_health = force_b[0].current_health
    damage_applied = force_b[0].take_damage(2)
    final_health = force_b[0].current_health

    if initial_health - final_health != damage_applied:
        issues.append(f"Damage application inconsistent: health dropped by {initial_health - final_health}, but damage_applied={damage_applied}")

    # Test 3: Check that battle statistics are self-consistent
    force_a = [Ship("Ship A1", offensive_power=8, defensive_power=0.3, staying_power=3),
               Ship("Ship A2", offensive_power=12, defensive_power=0.4, staying_power=5)]
    force_b = [Ship("Ship B1", offensive_power=6, defensive_power=0.4, staying_power=2),
               Ship("Ship B2", offensive_power=10, defensive_power=0.35, staying_power=4)]

    sim = SalvoCombatModel(force_a, force_b, random_seed=42)
    result = sim.run_simulation(quiet=True)
    stats = sim.get_battle_statistics()

    # Check that survivor counts make sense
    if stats['force_a_survivors'] < 0 or stats['force_b_survivors'] < 0:
        issues.append(f"Negative survivor count: A={stats['force_a_survivors']}, B={stats['force_b_survivors']}")

    if stats['force_a_survivors'] > len(force_a) or stats['force_b_survivors'] > len(force_b):
        issues.append(f"Survivor count exceeds initial force size")

    # Check that outcome matches survivor counts
    if "Force A Victory" in stats['outcome'] and stats['force_a_survivors'] == 0:
        issues.append(f"Force A declared winner but has 0 survivors")
    if "Force B Victory" in stats['outcome'] and stats['force_b_survivors'] == 0:
        issues.append(f"Force B declared winner but has 0 survivors")

    # Test 4: Check round counting logic
    # Reset and run again to check round counting
    force_a = [Ship("Ship A1", offensive_power=1, defensive_power=0.8, staying_power=10),
               Ship("Ship A2", offensive_power=1, defensive_power=0.8, staying_power=10)]
    force_b = [Ship("Ship B1", offensive_power=1, defensive_power=0.8, staying_power=10),
               Ship("Ship B2", offensive_power=1, defensive_power=0.8, staying_power=10)]

    sim = SalvoCombatModel(force_a, force_b, random_seed=42)
    result = sim.run_simulation(quiet=True, max_rounds=5)  # Force early termination

    if sim.round_number > 5:
        issues.append(f"Battle exceeded max_rounds limit: {sim.round_number} > 5")

    print(f"Salvo Combat logic check: {len(issues)} issues found")
    return issues

def test_cross_model_consistency():
    """Test consistency between models for overlapping scenarios."""
    print("\n" + "="*60)
    print("TESTING CROSS-MODEL CONSISTENCY")
    print("="*60)

    issues = []

    # When effectiveness coefficients are equal and small, Linear and Square laws
    # should give similar results for small time scales
    A0, B0 = 100, 80
    alpha = beta = 0.001  # Small effectiveness

    linear_battle = LanchesterLinear(A0, B0, alpha, beta)
    square_battle = LanchesterSquare(A0, B0, alpha, beta)

    linear_solution = linear_battle.analytical_solution()
    square_solution = square_battle.analytical_solution()

    # Both should predict A wins
    if linear_solution['winner'] != square_solution['winner']:
        issues.append(f"Winner disagreement: Linear predicts {linear_solution['winner']}, Square predicts {square_solution['winner']}")

    # For equal effectiveness, Square Law should give more survivors than Linear Law
    if square_solution['remaining_strength'] <= linear_solution['remaining_strength']:
        issues.append(f"Square Law should give more survivors than Linear Law with equal effectiveness")

    print(f"Cross-model consistency check: {len(issues)} issues found")
    return issues

def test_input_validation_edge_cases():
    """Test input validation for edge cases that might not be caught."""
    print("\n" + "="*60)
    print("TESTING INPUT VALIDATION EDGE CASES")
    print("="*60)

    issues = []

    # Test 1: Very small but positive values
    try:
        battle = LanchesterLinear(A0=1e-15, B0=1e-15, alpha=1e-15, beta=1e-15)
        # This should work but might cause numerical issues
        solution = battle.analytical_solution()
    except Exception as e:
        issues.append(f"Ultra-small positive values failed: {e}")

    # Test 2: Ship validation edge cases
    try:
        ship = Ship("Test", offensive_power=1e-10, defensive_power=0.999999, staying_power=1000000)
        # This should work
    except Exception as e:
        issues.append(f"Extreme but valid ship parameters failed: {e}")

    # Test 3: Defensive power exactly at boundary
    try:
        ship1 = Ship("Boundary1", offensive_power=5, defensive_power=0.0, staying_power=3)
        ship2 = Ship("Boundary2", offensive_power=5, defensive_power=1.0, staying_power=3)
        # Both should work
    except Exception as e:
        issues.append(f"Boundary defensive power values failed: {e}")

    # Test 4: Check if defensive power > 1.0 is properly rejected
    defensive_error_caught = False
    try:
        ship = Ship("Invalid", offensive_power=5, defensive_power=1.1, staying_power=3)
    except ValueError:
        defensive_error_caught = True
    except Exception as e:
        issues.append(f"Wrong exception type for defensive_power > 1.0: {e}")

    if not defensive_error_caught:
        issues.append("Defensive power > 1.0 was not rejected")

    print(f"Input validation edge cases: {len(issues)} issues found")
    return issues

def run_subtle_bug_analysis():
    """Run all subtle bug tests and compile results."""
    print("LANCHESTER SIMULATION CODEBASE - SUBTLE BUG ANALYSIS")
    print("="*80)

    all_issues = []

    all_issues.extend(test_linear_law_invariant())
    all_issues.extend(test_square_law_invariant())
    all_issues.extend(test_battle_end_consistency())
    all_issues.extend(test_salvo_combat_logic())
    all_issues.extend(test_cross_model_consistency())
    all_issues.extend(test_input_validation_edge_cases())

    print("\n" + "="*80)
    print("SUBTLE BUG ANALYSIS SUMMARY")
    print("="*80)

    if all_issues:
        print("\nISSUES DETECTED:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i:2d}. {issue}")
    else:
        print("\n✓ No subtle bugs or logic errors detected")

    print(f"\nTOTAL ISSUES FOUND: {len(all_issues)}")
    return all_issues

if __name__ == "__main__":
    issues = run_subtle_bug_analysis()
    sys.exit(len(issues))