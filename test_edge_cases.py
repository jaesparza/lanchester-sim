#!/usr/bin/env python3
"""
Test script to explore edge cases and potential bugs in the lanchester-sim codebase.
This script tests extreme parameter values, boundary conditions, and numerical stability.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship

def test_linear_edge_cases():
    """Test Linear Law with edge cases that might expose bugs."""
    print("="*60)
    print("TESTING LINEAR LAW EDGE CASES")
    print("="*60)

    edge_cases = []

    # Test 1: Zero initial forces
    print("\n1. Testing zero initial forces:")
    try:
        battle = LanchesterLinear(A0=0, B0=100, alpha=0.01, beta=0.01)
        winner, remaining, t_end = battle.calculate_battle_outcome()
        print(f"  A0=0, B0=100: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        print(f"  Trajectories computed successfully")
        edge_cases.append(("Zero A0", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Zero A0", f"FAIL: {e}"))

    try:
        battle = LanchesterLinear(A0=100, B0=0, alpha=0.01, beta=0.01)
        winner, remaining, t_end = battle.calculate_battle_outcome()
        print(f"  A0=100, B0=0: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        print(f"  Trajectories computed successfully")
        edge_cases.append(("Zero B0", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Zero B0", f"FAIL: {e}"))

    # Test 2: Very large forces
    print("\n2. Testing very large forces:")
    try:
        battle = LanchesterLinear(A0=1e10, B0=1e9, alpha=0.01, beta=0.01)
        winner, remaining, t_end = battle.calculate_battle_outcome()
        print(f"  Large forces: Winner={winner}, Remaining={remaining:.2e}, Time={t_end:.2e}")
        solution = battle.analytical_solution()
        if np.any(np.isnan(solution['A'])) or np.any(np.isnan(solution['B'])):
            raise ValueError("NaN values in trajectories")
        edge_cases.append(("Large forces", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Large forces", f"FAIL: {e}"))

    # Test 3: Very small effectiveness coefficients
    print("\n3. Testing very small effectiveness:")
    try:
        battle = LanchesterLinear(A0=100, B0=80, alpha=1e-10, beta=1e-10)
        winner, remaining, t_end = battle.calculate_battle_outcome()
        print(f"  Tiny alpha/beta: Winner={winner}, Remaining={remaining}, Time={t_end:.2e}")
        solution = battle.analytical_solution()
        edge_cases.append(("Tiny effectiveness", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Tiny effectiveness", f"FAIL: {e}"))

    # Test 4: Very large effectiveness coefficients
    print("\n4. Testing very large effectiveness:")
    try:
        battle = LanchesterLinear(A0=100, B0=80, alpha=1e6, beta=1e6)
        winner, remaining, t_end = battle.calculate_battle_outcome()
        print(f"  Large alpha/beta: Winner={winner}, Remaining={remaining}, Time={t_end:.2e}")
        solution = battle.analytical_solution()
        edge_cases.append(("Large effectiveness", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Large effectiveness", f"FAIL: {e}"))

    return edge_cases

def test_square_edge_cases():
    """Test Square Law with edge cases that might expose bugs."""
    print("\n"+"="*60)
    print("TESTING SQUARE LAW EDGE CASES")
    print("="*60)

    edge_cases = []

    # Test 1: Exact draws (should have infinite battle time)
    print("\n1. Testing exact draws:")
    try:
        battle = LanchesterSquare(A0=100, B0=100, alpha=0.01, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Exact draw: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        print(f"  Solution battle_end_time: {solution['battle_end_time']}")
        edge_cases.append(("Exact draw", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Exact draw", f"FAIL: {e}"))

    # Test 2: Near-exact draws (should have finite time but may have numerical issues)
    print("\n2. Testing near-exact draws:")
    try:
        battle = LanchesterSquare(A0=100.001, B0=100, alpha=0.01, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Near draw: Winner={winner}, Remaining={remaining:.6f}, Time={t_end:.2e}")
        solution = battle.analytical_solution()
        edge_cases.append(("Near draw", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Near draw", f"FAIL: {e}"))

    # Test 3: Zero effectiveness coefficients
    print("\n3. Testing zero effectiveness:")
    try:
        battle = LanchesterSquare(A0=100, B0=80, alpha=0, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Alpha=0: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        edge_cases.append(("Alpha zero", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Alpha zero", f"FAIL: {e}"))

    try:
        battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Beta=0: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        edge_cases.append(("Beta zero", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Beta zero", f"FAIL: {e}"))

    # Test 4: Both effectiveness coefficients zero
    print("\n4. Testing both effectiveness zero:")
    try:
        battle = LanchesterSquare(A0=100, B0=80, alpha=0, beta=0)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Both zero: Winner={winner}, Remaining={remaining}, Time={t_end}")
        solution = battle.analytical_solution()
        edge_cases.append(("Both effectiveness zero", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Both effectiveness zero", f"FAIL: {e}"))

    # Test 5: Very small forces
    print("\n5. Testing very small forces:")
    try:
        battle = LanchesterSquare(A0=0.001, B0=0.0005, alpha=0.01, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Tiny forces: Winner={winner}, Remaining={remaining:.6f}, Time={t_end}")
        solution = battle.analytical_solution()
        edge_cases.append(("Tiny forces", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Tiny forces", f"FAIL: {e}"))

    # Test 6: Extreme arctanh arguments (test boundary cases)
    print("\n6. Testing extreme arctanh arguments:")
    try:
        # This should create an arctanh argument very close to ±1
        battle = LanchesterSquare(A0=1000, B0=999.9, alpha=0.01, beta=0.01)
        winner, remaining, invariant = battle.calculate_battle_outcome()
        t_end = battle.calculate_battle_end_time(winner, remaining, invariant)
        print(f"  Extreme arctanh: Winner={winner}, Remaining={remaining:.6f}, Time={t_end}")
        solution = battle.analytical_solution()
        edge_cases.append(("Extreme arctanh", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Extreme arctanh", f"FAIL: {e}"))

    return edge_cases

def test_salvo_edge_cases():
    """Test Salvo Combat Model with edge cases."""
    print("\n"+"="*60)
    print("TESTING SALVO COMBAT MODEL EDGE CASES")
    print("="*60)

    edge_cases = []

    # Test 1: Ships with zero offensive power
    print("\n1. Testing zero offensive power:")
    try:
        force_a = [Ship("Pacifist", offensive_power=0, defensive_power=0.5, staying_power=3)]
        force_b = [Ship("Attacker", offensive_power=10, defensive_power=0.3, staying_power=2)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True)
        print(f"  Zero offensive: Result={result}")
        edge_cases.append(("Zero offensive", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Zero offensive", f"FAIL: {e}"))

    # Test 2: Ships with zero defensive power
    print("\n2. Testing zero defensive power:")
    try:
        force_a = [Ship("Glass Cannon", offensive_power=15, defensive_power=0, staying_power=1)]
        force_b = [Ship("Tank", offensive_power=5, defensive_power=0.9, staying_power=10)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True)
        print(f"  Zero defensive: Result={result}")
        edge_cases.append(("Zero defensive", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Zero defensive", f"FAIL: {e}"))

    # Test 3: Ships with maximum defensive power (1.0)
    print("\n3. Testing maximum defensive power:")
    try:
        force_a = [Ship("Invulnerable", offensive_power=10, defensive_power=1.0, staying_power=5)]
        force_b = [Ship("Normal", offensive_power=10, defensive_power=0.3, staying_power=5)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True, max_rounds=10)  # Limit rounds
        print(f"  Max defensive: Result={result}")
        edge_cases.append(("Max defensive", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Max defensive", f"FAIL: {e}"))

    # Test 4: Ships with staying power = 1 (destroyed in one hit)
    print("\n4. Testing staying power = 1:")
    try:
        force_a = [Ship("Fragile A", offensive_power=8, defensive_power=0.3, staying_power=1)]
        force_b = [Ship("Fragile B", offensive_power=8, defensive_power=0.3, staying_power=1)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True)
        print(f"  Staying power 1: Result={result}")
        edge_cases.append(("Staying power 1", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Staying power 1", f"FAIL: {e}"))

    # Test 5: Very large fleets
    print("\n5. Testing very large fleets:")
    try:
        force_a = [Ship(f"Ship_A_{i}", offensive_power=1, defensive_power=0.1, staying_power=1) for i in range(100)]
        force_b = [Ship(f"Ship_B_{i}", offensive_power=1, defensive_power=0.1, staying_power=1) for i in range(100)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True, max_rounds=5)  # Limit rounds
        print(f"  Large fleets: Result={result}")
        edge_cases.append(("Large fleets", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Large fleets", f"FAIL: {e}"))

    # Test 6: Monte Carlo with extreme parameters
    print("\n6. Testing Monte Carlo with edge case ships:")
    try:
        force_a = [Ship("Extreme A", offensive_power=100, defensive_power=0.95, staying_power=20)]
        force_b = [Ship("Extreme B", offensive_power=100, defensive_power=0.95, staying_power=20)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        analysis = sim.run_monte_carlo_analysis(iterations=10, quiet=True)
        print(f"  Monte Carlo extreme: Completed {analysis['iterations']} iterations")
        edge_cases.append(("Monte Carlo extreme", "PASS"))
    except Exception as e:
        print(f"  ERROR: {e}")
        edge_cases.append(("Monte Carlo extreme", f"FAIL: {e}"))

    return edge_cases

def test_numerical_stability():
    """Test numerical stability across models."""
    print("\n"+"="*60)
    print("TESTING NUMERICAL STABILITY")
    print("="*60)

    stability_issues = []

    # Test 1: Very long time scales in Square Law
    print("\n1. Testing long time scales:")
    try:
        battle = LanchesterSquare(A0=100, B0=99.999, alpha=1e-6, beta=1e-6)
        solution = battle.analytical_solution()

        # Check for NaN/Inf values
        if np.any(np.isnan(solution['A'])) or np.any(np.isnan(solution['B'])):
            stability_issues.append("NaN values in long time scale trajectories")
        if np.any(np.isinf(solution['A'])) or np.any(np.isinf(solution['B'])):
            stability_issues.append("Inf values in long time scale trajectories")

        print(f"  Long time scale test passed")
    except Exception as e:
        stability_issues.append(f"Long time scale error: {e}")
        print(f"  ERROR: {e}")

    # Test 2: Very short time scales
    print("\n2. Testing short time scales:")
    try:
        battle = LanchesterSquare(A0=1000, B0=100, alpha=1e6, beta=1e6)
        solution = battle.analytical_solution()

        if np.any(np.isnan(solution['A'])) or np.any(np.isnan(solution['B'])):
            stability_issues.append("NaN values in short time scale trajectories")
        if np.any(np.isinf(solution['A'])) or np.any(np.isinf(solution['B'])):
            stability_issues.append("Inf values in short time scale trajectories")

        print(f"  Short time scale test passed")
    except Exception as e:
        stability_issues.append(f"Short time scale error: {e}")
        print(f"  ERROR: {e}")

    # Test 3: Precision loss in calculations
    print("\n3. Testing precision in near-equal scenarios:")
    try:
        # Test scenarios that might lose precision
        battle1 = LanchesterSquare(A0=1e10, B0=1e10 - 1, alpha=0.01, beta=0.01)
        solution1 = battle1.analytical_solution()

        battle2 = LanchesterSquare(A0=1e-10, B0=1e-10 - 1e-15, alpha=0.01, beta=0.01)
        solution2 = battle2.analytical_solution()

        print(f"  Precision tests passed")
    except Exception as e:
        stability_issues.append(f"Precision error: {e}")
        print(f"  ERROR: {e}")

    return stability_issues

def run_all_edge_case_tests():
    """Run all edge case tests and compile results."""
    print("LANCHESTER SIMULATION CODEBASE - COMPREHENSIVE EDGE CASE TESTING")
    print("="*80)

    linear_cases = test_linear_edge_cases()
    square_cases = test_square_edge_cases()
    salvo_cases = test_salvo_edge_cases()
    stability_issues = test_numerical_stability()

    print("\n" + "="*80)
    print("EDGE CASE TEST SUMMARY")
    print("="*80)

    print(f"\nLinear Law Edge Cases:")
    for case_name, result in linear_cases:
        status = "✓ PASS" if result == "PASS" else "✗ FAIL"
        print(f"  {status} {case_name}")
        if result != "PASS":
            print(f"    {result}")

    print(f"\nSquare Law Edge Cases:")
    for case_name, result in square_cases:
        status = "✓ PASS" if result == "PASS" else "✗ FAIL"
        print(f"  {status} {case_name}")
        if result != "PASS":
            print(f"    {result}")

    print(f"\nSalvo Combat Edge Cases:")
    for case_name, result in salvo_cases:
        status = "✓ PASS" if result == "PASS" else "✗ FAIL"
        print(f"  {status} {case_name}")
        if result != "PASS":
            print(f"    {result}")

    print(f"\nNumerical Stability Issues:")
    if stability_issues:
        for issue in stability_issues:
            print(f"  ✗ {issue}")
    else:
        print(f"  ✓ No numerical stability issues detected")

    # Count total issues
    total_failures = (
        len([r for n, r in linear_cases if r != "PASS"]) +
        len([r for n, r in square_cases if r != "PASS"]) +
        len([r for n, r in salvo_cases if r != "PASS"]) +
        len(stability_issues)
    )

    print(f"\nTOTAL ISSUES DETECTED: {total_failures}")
    return total_failures

if __name__ == "__main__":
    total_issues = run_all_edge_case_tests()
    sys.exit(total_issues)