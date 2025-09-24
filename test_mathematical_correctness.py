#!/usr/bin/env python3
"""
Test script to verify mathematical correctness of the combat models.
This focuses on differential equations, boundary conditions, and analytical solutions.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship

def test_square_law_differential_equation():
    """Test that Square Law trajectories satisfy the correct differential equations."""
    print("="*70)
    print("TESTING SQUARE LAW DIFFERENTIAL EQUATION CORRECTNESS")
    print("="*70)

    math_issues = []

    # Square Law should follow: dA/dt = -β*A*B, dB/dt = -α*A*B
    battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution = battle.analytical_solution()

    t = solution['time']
    A_t = solution['A']
    B_t = solution['B']
    t_end = solution['battle_end_time']

    # Check derivatives numerically during active combat
    dt = t[1] - t[0]
    for i in range(1, len(t) - 1):
        if t[i] < t_end and A_t[i] > 1 and B_t[i] > 1:  # During active combat
            # Numerical derivatives
            dA_dt = (A_t[i+1] - A_t[i-1]) / (2 * dt)
            dB_dt = (B_t[i+1] - B_t[i-1]) / (2 * dt)

            # Expected derivatives from Square Law
            expected_dA_dt = -battle.beta * A_t[i] * B_t[i]
            expected_dB_dt = -battle.alpha * A_t[i] * B_t[i]

            # Check relative error (allow some numerical error)
            if abs(expected_dA_dt) > 1e-6:  # Avoid division by tiny numbers
                rel_error_A = abs(dA_dt - expected_dA_dt) / abs(expected_dA_dt)
                if rel_error_A > 0.1:  # 10% tolerance
                    math_issues.append(f"Square Law dA/dt error at t={t[i]:.2f}: got {dA_dt:.6f}, expected {expected_dA_dt:.6f}")
                    break

            if abs(expected_dB_dt) > 1e-6:
                rel_error_B = abs(dB_dt - expected_dB_dt) / abs(expected_dB_dt)
                if rel_error_B > 0.1:
                    math_issues.append(f"Square Law dB/dt error at t={t[i]:.2f}: got {dB_dt:.6f}, expected {expected_dB_dt:.6f}")
                    break

    print(f"Differential equation check: {len(math_issues)} issues found")
    return math_issues

def test_linear_law_differential_equation():
    """Test that Linear Law trajectories satisfy the correct differential equations."""
    print("\n" + "="*70)
    print("TESTING LINEAR LAW DIFFERENTIAL EQUATION CORRECTNESS")
    print("="*70)

    math_issues = []

    # Linear Law should follow: dA/dt = -β*B, dB/dt = -α*A
    # But this is actually incorrect! The Linear Law has CONSTANT attrition rates:
    # dA/dt = -β, dB/dt = -α (independent of current force levels)

    battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution = battle.analytical_solution()

    t = solution['time']
    A_t = solution['A']
    B_t = solution['B']
    t_end = solution['battle_end_time']

    # Check derivatives numerically during active combat
    dt = t[1] - t[0]
    for i in range(1, len(t) - 1):
        if t[i] < t_end and A_t[i] > 1 and B_t[i] > 1:  # During active combat
            # Numerical derivatives
            dA_dt = (A_t[i+1] - A_t[i-1]) / (2 * dt)
            dB_dt = (B_t[i+1] - B_t[i-1]) / (2 * dt)

            # Expected derivatives from Linear Law (CONSTANT rates)
            expected_dA_dt = -battle.beta
            expected_dB_dt = -battle.alpha

            # Check error
            error_A = abs(dA_dt - expected_dA_dt)
            error_B = abs(dB_dt - expected_dB_dt)

            if error_A > 0.01:  # Small absolute tolerance
                math_issues.append(f"Linear Law dA/dt error at t={t[i]:.2f}: got {dA_dt:.6f}, expected {expected_dA_dt:.6f}")
                break

            if error_B > 0.01:
                math_issues.append(f"Linear Law dB/dt error at t={t[i]:.2f}: got {dB_dt:.6f}, expected {expected_dB_dt:.6f}")
                break

    print(f"Differential equation check: {len(math_issues)} issues found")
    return math_issues

def test_boundary_conditions():
    """Test boundary conditions for both models."""
    print("\n" + "="*70)
    print("TESTING BOUNDARY CONDITIONS")
    print("="*70)

    boundary_issues = []

    # Test 1: Initial conditions
    print("1. Testing initial conditions...")

    # Linear Law
    battle_linear = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution_linear = battle_linear.analytical_solution()

    if abs(solution_linear['A'][0] - 100) > 1e-10:
        boundary_issues.append(f"Linear Law initial A condition: got {solution_linear['A'][0]}, expected 100")
    if abs(solution_linear['B'][0] - 80) > 1e-10:
        boundary_issues.append(f"Linear Law initial B condition: got {solution_linear['B'][0]}, expected 80")

    # Square Law
    battle_square = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution_square = battle_square.analytical_solution()

    if abs(solution_square['A'][0] - 100) > 1e-10:
        boundary_issues.append(f"Square Law initial A condition: got {solution_square['A'][0]}, expected 100")
    if abs(solution_square['B'][0] - 80) > 1e-10:
        boundary_issues.append(f"Square Law initial B condition: got {solution_square['B'][0]}, expected 80")

    # Test 2: Non-negativity constraint
    print("2. Testing non-negativity constraint...")

    # Check that forces never go negative
    if np.any(solution_linear['A'] < 0):
        boundary_issues.append("Linear Law: Force A goes negative")
    if np.any(solution_linear['B'] < 0):
        boundary_issues.append("Linear Law: Force B goes negative")
    if np.any(solution_square['A'] < 0):
        boundary_issues.append("Square Law: Force A goes negative")
    if np.any(solution_square['B'] < 0):
        boundary_issues.append("Square Law: Force B goes negative")

    # Test 3: Battle end conditions
    print("3. Testing battle end conditions...")

    # At battle end, the losing force should be exactly zero
    t_linear = solution_linear['time']
    t_end_linear = solution_linear['battle_end_time']

    if not np.isinf(t_end_linear):
        # Find index closest to battle end
        end_idx_linear = np.argmin(np.abs(t_linear - t_end_linear))
        winner_linear = solution_linear['winner']

        if winner_linear == 'A' and solution_linear['B'][end_idx_linear] > 0.1:
            boundary_issues.append(f"Linear Law: B not eliminated at battle end: {solution_linear['B'][end_idx_linear]:.6f}")
        elif winner_linear == 'B' and solution_linear['A'][end_idx_linear] > 0.1:
            boundary_issues.append(f"Linear Law: A not eliminated at battle end: {solution_linear['A'][end_idx_linear]:.6f}")

    print(f"Boundary conditions check: {len(boundary_issues)} issues found")
    return boundary_issues

def test_analytical_vs_numerical_solutions():
    """Compare analytical solutions with numerical integration."""
    print("\n" + "="*70)
    print("TESTING ANALYTICAL VS NUMERICAL SOLUTIONS")
    print("="*70)

    analytical_issues = []

    # For Square Law, we can numerically integrate the ODEs and compare
    def square_law_ode(t, y, alpha, beta):
        A, B = y
        if A <= 0:
            A = 0
        if B <= 0:
            B = 0
        dA_dt = -beta * A * B
        dB_dt = -alpha * A * B
        return [dA_dt, dB_dt]

    try:
        from scipy.integrate import solve_ivp

        # Test case
        A0, B0 = 50, 40
        alpha, beta = 0.01, 0.01

        battle = LanchesterSquare(A0=A0, B0=B0, alpha=alpha, beta=beta)
        analytical = battle.analytical_solution()

        # Numerical integration
        t_span = (0, analytical['battle_end_time'])
        t_eval = analytical['time'][analytical['time'] <= analytical['battle_end_time']]

        if len(t_eval) > 0:
            sol = solve_ivp(lambda t, y: square_law_ode(t, y, alpha, beta),
                           t_span, [A0, B0], t_eval=t_eval, rtol=1e-8)

            # Compare results
            for i, t in enumerate(sol.t):
                if i < len(analytical['A']) and analytical['time'][i] <= analytical['battle_end_time']:
                    analytical_A = analytical['A'][i]
                    analytical_B = analytical['B'][i]
                    numerical_A = sol.y[0][i]
                    numerical_B = sol.y[1][i]

                    error_A = abs(analytical_A - numerical_A)
                    error_B = abs(analytical_B - numerical_B)

                    if error_A > 0.1 or error_B > 0.1:
                        analytical_issues.append(f"Large discrepancy at t={t:.2f}: A error={error_A:.3f}, B error={error_B:.3f}")
                        break

        print("   Analytical vs numerical comparison completed")

    except ImportError:
        print("   Skipping numerical integration test (scipy not available)")
    except Exception as e:
        analytical_issues.append(f"Numerical integration comparison failed: {e}")

    print(f"Analytical solution check: {len(analytical_issues)} issues found")
    return analytical_issues

def test_physical_realism():
    """Test if model behaviors are physically realistic."""
    print("\n" + "="*70)
    print("TESTING PHYSICAL REALISM")
    print("="*70)

    realism_issues = []

    # Test 1: Conservation of total casualties
    print("1. Testing casualty conservation...")

    battle_linear = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution_linear = battle_linear.analytical_solution()

    total_initial = battle_linear.A0 + battle_linear.B0
    total_casualties = solution_linear['A_casualties'] + solution_linear['B_casualties']
    total_survivors = solution_linear['remaining_strength']

    if abs(total_casualties + total_survivors - total_initial) > 1e-6:
        realism_issues.append(f"Linear Law casualty conservation error: {total_casualties} + {total_survivors} ≠ {total_initial}")

    # Same test for Square Law
    battle_square = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution_square = battle_square.analytical_solution()

    total_initial = battle_square.A0 + battle_square.B0
    total_casualties = solution_square['A_casualties'] + solution_square['B_casualties']
    total_survivors = solution_square['remaining_strength']

    if abs(total_casualties + total_survivors - total_initial) > 1e-6:
        realism_issues.append(f"Square Law casualty conservation error: {total_casualties} + {total_survivors} ≠ {total_initial}")

    # Test 2: Monotonic decrease (forces should only decrease)
    print("2. Testing monotonic force decrease...")

    for i in range(1, len(solution_linear['A'])):
        if solution_linear['time'][i] <= solution_linear['battle_end_time']:
            if solution_linear['A'][i] > solution_linear['A'][i-1] + 1e-10:
                realism_issues.append(f"Linear Law: Force A increased during battle at t={solution_linear['time'][i]:.2f}")
                break
            if solution_linear['B'][i] > solution_linear['B'][i-1] + 1e-10:
                realism_issues.append(f"Linear Law: Force B increased during battle at t={solution_linear['time'][i]:.2f}")
                break

    # Test 3: Salvo model ship health logic
    print("3. Testing Salvo model ship health logic...")

    ship = Ship("Test Ship", offensive_power=10, defensive_power=0.5, staying_power=5)

    # Ship should be operational initially
    if not ship.is_operational():
        realism_issues.append("Ship not operational when at full health")

    # Take some damage
    damage_taken = ship.take_damage(3)
    if damage_taken != 3:
        realism_issues.append(f"Damage application error: took {damage_taken}, expected 3")

    # Should still be operational
    if not ship.is_operational():
        realism_issues.append("Ship not operational after taking non-fatal damage")

    # Take fatal damage
    damage_taken = ship.take_damage(5)  # More than remaining health
    remaining_health = 5 - 3  # 2 remaining health
    if damage_taken != remaining_health:
        realism_issues.append(f"Excess damage not capped: took {damage_taken}, should be {remaining_health}")

    # Should not be operational now
    if ship.is_operational():
        realism_issues.append("Ship still operational after fatal damage")

    print(f"Physical realism check: {len(realism_issues)} issues found")
    return realism_issues



def run_mathematical_correctness_analysis():
    """Run all mathematical correctness tests."""
    print("LANCHESTER SIMULATION CODEBASE - MATHEMATICAL CORRECTNESS ANALYSIS")
    print("="*80)

    all_issues = []

    all_issues.extend(test_square_law_differential_equation())
    all_issues.extend(test_linear_law_differential_equation())
    all_issues.extend(test_boundary_conditions())
    all_issues.extend(test_analytical_vs_numerical_solutions())
    all_issues.extend(test_physical_realism())

    print("\n" + "="*80)
    print("MATHEMATICAL CORRECTNESS SUMMARY")
    print("="*80)

    if all_issues:
        print("\nMATHEMATICAL ISSUES DETECTED:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i:2d}. {issue}")
    else:
        print("\n✓ No mathematical correctness issues detected")

    print(f"\nTOTAL MATHEMATICAL ISSUES: {len(all_issues)}")
    return all_issues

if __name__ == "__main__":
    issues = run_mathematical_correctness_analysis()
    sys.exit(len(issues))