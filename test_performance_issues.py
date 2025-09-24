#!/usr/bin/env python3
"""
Test script to identify performance issues and potential problems with large inputs.
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship

def test_large_input_performance():
    """Test performance with large input values."""
    print("="*60)
    print("TESTING PERFORMANCE WITH LARGE INPUTS")
    print("="*60)

    performance_issues = []

    # Test 1: Large force sizes
    print("\n1. Testing large force sizes...")
    start_time = time.time()
    try:
        battle = LanchesterSquare(A0=1e6, B0=8e5, alpha=0.01, beta=0.01)
        solution = battle.analytical_solution()
        elapsed = time.time() - start_time
        print(f"   Large forces completed in {elapsed:.3f}s")
        if elapsed > 5.0:  # Arbitrary threshold
            performance_issues.append(f"Large force calculation took {elapsed:.3f}s")
    except Exception as e:
        performance_issues.append(f"Large force calculation failed: {e}")

    # Test 2: Many time points
    print("\n2. Testing many time points...")
    start_time = time.time()
    try:
        battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
        # Increase DEFAULT_TIME_POINTS temporarily
        old_points = battle.DEFAULT_TIME_POINTS
        battle.DEFAULT_TIME_POINTS = 100000
        solution = battle.analytical_solution()
        battle.DEFAULT_TIME_POINTS = old_points
        elapsed = time.time() - start_time
        print(f"   Many time points completed in {elapsed:.3f}s")
        if elapsed > 5.0:
            performance_issues.append(f"Many time points calculation took {elapsed:.3f}s")
    except Exception as e:
        performance_issues.append(f"Many time points calculation failed: {e}")

    # Test 3: Large Salvo fleets
    print("\n3. Testing large Salvo fleets...")
    start_time = time.time()
    try:
        force_a = [Ship(f"A_{i}", offensive_power=5, defensive_power=0.3, staying_power=2) for i in range(200)]
        force_b = [Ship(f"B_{i}", offensive_power=5, defensive_power=0.3, staying_power=2) for i in range(200)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True, max_rounds=3)  # Limit rounds
        elapsed = time.time() - start_time
        print(f"   Large Salvo fleets completed in {elapsed:.3f}s")
        if elapsed > 10.0:
            performance_issues.append(f"Large Salvo fleet simulation took {elapsed:.3f}s")
    except Exception as e:
        performance_issues.append(f"Large Salvo fleet simulation failed: {e}")

    return performance_issues

def test_memory_usage_issues():
    """Test for potential memory issues."""
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)

    memory_issues = []

    # Test 1: Check if arrays grow unboundedly
    print("\n1. Testing array size bounds...")
    try:
        battle = LanchesterSquare(A0=100, B0=80, alpha=1e-10, beta=1e-10)  # Very slow battle
        solution = battle.analytical_solution()

        # Check that arrays aren't excessively large
        time_array_size = len(solution['time'])
        print(f"   Time array size: {time_array_size}")
        if time_array_size > 100000:  # Arbitrary large threshold
            memory_issues.append(f"Time array grew to {time_array_size} points")

        # Check memory usage of arrays (rough estimate)
        estimated_memory = time_array_size * 8 * 3 / (1024*1024)  # 3 arrays, 8 bytes per float, convert to MB
        print(f"   Estimated memory usage: {estimated_memory:.2f} MB")
        if estimated_memory > 100:  # 100 MB threshold
            memory_issues.append(f"Estimated memory usage: {estimated_memory:.2f} MB")

    except Exception as e:
        memory_issues.append(f"Array size test failed: {e}")

    # Test 2: Battle log growth in Salvo model
    print("\n2. Testing battle log growth...")
    try:
        # Create a battle that might run for many rounds
        force_a = [Ship("Tough A", offensive_power=1, defensive_power=0.9, staying_power=100)]
        force_b = [Ship("Tough B", offensive_power=1, defensive_power=0.9, staying_power=100)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)
        result = sim.run_simulation(quiet=True, max_rounds=20)

        log_entries = len(sim.battle_log)
        print(f"   Battle log entries: {log_entries}")

        # Rough memory estimate for battle log
        if log_entries > 0:
            events_per_round = len(sim.battle_log[0]['events']) if sim.battle_log else 0
            estimated_log_memory = log_entries * events_per_round * 100 / (1024*1024)  # Rough estimate
            print(f"   Estimated log memory: {estimated_log_memory:.3f} MB")

        if log_entries > 1000:
            memory_issues.append(f"Battle log grew to {log_entries} entries")

    except Exception as e:
        memory_issues.append(f"Battle log test failed: {e}")

    return memory_issues

def test_numerical_precision_loss():
    """Test for numerical precision loss in edge cases."""
    print("\n" + "="*60)
    print("TESTING NUMERICAL PRECISION")
    print("="*60)

    precision_issues = []

    # Test 1: Very close values
    print("\n1. Testing very close initial forces...")
    try:
        battle = LanchesterSquare(A0=1.0000000001, B0=1.0, alpha=0.01, beta=0.01)
        solution = battle.analytical_solution()

        # Check if the small difference is preserved
        if solution['winner'] == 'Draw':
            precision_issues.append("Very small force difference lost to floating-point precision")
        else:
            print(f"   Small difference preserved: {solution['winner']} wins with {solution['remaining_strength']:.10f}")

    except Exception as e:
        precision_issues.append(f"Close values test failed: {e}")

    # Test 2: Large numbers with small differences
    print("\n2. Testing large numbers with small relative differences...")
    try:
        battle = LanchesterSquare(A0=1e15, B0=1e15 * 0.999999999, alpha=0.01, beta=0.01)
        solution = battle.analytical_solution()

        # The small relative difference should still determine the winner
        if solution['winner'] == 'Draw':
            precision_issues.append("Small relative difference in large numbers lost to precision")
        else:
            print(f"   Large number precision preserved: {solution['winner']} wins")

    except Exception as e:
        precision_issues.append(f"Large numbers precision test failed: {e}")

    # Test 3: Extreme time scales
    print("\n3. Testing extreme time scales...")
    try:
        # Very fast battle
        battle_fast = LanchesterSquare(A0=100, B0=50, alpha=1e6, beta=1e6)
        solution_fast = battle_fast.analytical_solution()

        if np.any(np.isnan(solution_fast['A'])) or np.any(np.isnan(solution_fast['B'])):
            precision_issues.append("NaN values in fast battle trajectories")

        if np.any(np.isinf(solution_fast['A'])) or np.any(np.isinf(solution_fast['B'])):
            precision_issues.append("Inf values in fast battle trajectories")

        print(f"   Fast battle time scale: {solution_fast['battle_end_time']:.2e}")

        # Very slow battle
        battle_slow = LanchesterSquare(A0=100, B0=99, alpha=1e-6, beta=1e-6)
        solution_slow = battle_slow.analytical_solution()

        if np.any(np.isnan(solution_slow['A'])) or np.any(np.isnan(solution_slow['B'])):
            precision_issues.append("NaN values in slow battle trajectories")

        print(f"   Slow battle time scale: {solution_slow['battle_end_time']:.2e}")

    except Exception as e:
        precision_issues.append(f"Extreme time scales test failed: {e}")

    return precision_issues

def test_potential_infinite_loops():
    """Test for potential infinite loops or very long computations."""
    print("\n" + "="*60)
    print("TESTING POTENTIAL INFINITE LOOPS")
    print("="*60)

    loop_issues = []

    # Test 1: Salvo combat with very high defensive power
    print("\n1. Testing high defensive power scenarios...")
    try:
        force_a = [Ship("Super Defense A", offensive_power=1, defensive_power=0.99, staying_power=5)]
        force_b = [Ship("Super Defense B", offensive_power=1, defensive_power=0.99, staying_power=5)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)

        start_time = time.time()
        result = sim.run_simulation(quiet=True, max_rounds=10)  # Force limit
        elapsed = time.time() - start_time

        print(f"   High defensive power battle completed in {elapsed:.3f}s")
        if elapsed > 5.0:
            loop_issues.append(f"High defensive power battle took {elapsed:.3f}s")

    except Exception as e:
        loop_issues.append(f"High defensive power test failed: {e}")

    # Test 2: Monte Carlo with problematic scenarios
    print("\n2. Testing Monte Carlo with edge cases...")
    try:
        force_a = [Ship("Edge A", offensive_power=0.1, defensive_power=0.95, staying_power=10)]
        force_b = [Ship("Edge B", offensive_power=0.1, defensive_power=0.95, staying_power=10)]
        sim = SalvoCombatModel(force_a, force_b, random_seed=42)

        start_time = time.time()
        analysis = sim.run_monte_carlo_analysis(iterations=5, quiet=True)  # Small number
        elapsed = time.time() - start_time

        print(f"   Monte Carlo edge cases completed in {elapsed:.3f}s")
        if elapsed > 10.0:
            loop_issues.append(f"Monte Carlo edge cases took {elapsed:.3f}s")

    except Exception as e:
        loop_issues.append(f"Monte Carlo edge cases failed: {e}")

    return loop_issues

def run_performance_analysis():
    """Run all performance tests and compile results."""
    print("LANCHESTER SIMULATION CODEBASE - PERFORMANCE ANALYSIS")
    print("="*80)

    all_issues = []

    all_issues.extend(test_large_input_performance())
    all_issues.extend(test_memory_usage_issues())
    all_issues.extend(test_numerical_precision_loss())
    all_issues.extend(test_potential_infinite_loops())

    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)

    if all_issues:
        print("\nPERFORMANCE ISSUES DETECTED:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i:2d}. {issue}")
    else:
        print("\n✓ No significant performance issues detected")

    print(f"\nTOTAL PERFORMANCE ISSUES: {len(all_issues)}")
    return all_issues

if __name__ == "__main__":
    issues = run_performance_analysis()
    sys.exit(len(issues))