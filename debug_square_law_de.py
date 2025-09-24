#!/usr/bin/env python3
"""
Debug the Square Law differential equation issue detected in mathematical correctness testing.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import LanchesterSquare

def debug_square_law_derivatives():
    """Debug the Square Law differential equation implementation."""
    print("="*70)
    print("DEBUGGING SQUARE LAW DIFFERENTIAL EQUATION")
    print("="*70)

    battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution = battle.analytical_solution()

    t = solution['time']
    A_t = solution['A']
    B_t = solution['B']
    t_end = solution['battle_end_time']

    print(f"Battle parameters:")
    print(f"  A0={battle.A0}, B0={battle.B0}")
    print(f"  alpha={battle.alpha}, beta={battle.beta}")
    print(f"  Battle end time: {t_end:.6f}")

    # Check the first few time steps in detail
    dt = t[1] - t[0]
    print(f"\nTime step dt = {dt:.6f}")

    print(f"\nDetailed analysis at early time points:")
    print(f"{'i':>3} {'t':>8} {'A(t)':>12} {'B(t)':>12} {'dA/dt_num':>12} {'dA/dt_exp':>12} {'dB/dt_num':>12} {'dB/dt_exp':>12}")
    print("-" * 90)

    for i in range(1, min(10, len(t) - 1)):
        if t[i] < t_end and A_t[i] > 1 and B_t[i] > 1:
            # Numerical derivatives (central difference)
            dA_dt_num = (A_t[i+1] - A_t[i-1]) / (2 * dt)
            dB_dt_num = (B_t[i+1] - B_t[i-1]) / (2 * dt)

            # Expected derivatives from Square Law: dA/dt = -β*A*B, dB/dt = -α*A*B
            dA_dt_exp = -battle.beta * A_t[i] * B_t[i]
            dB_dt_exp = -battle.alpha * A_t[i] * B_t[i]

            print(f"{i:3d} {t[i]:8.6f} {A_t[i]:12.6f} {B_t[i]:12.6f} {dA_dt_num:12.6f} {dA_dt_exp:12.6f} {dB_dt_num:12.6f} {dB_dt_exp:12.6f}")

            # Check if there's a large discrepancy
            if abs(dA_dt_exp) > 1e-6:
                rel_error_A = abs(dA_dt_num - dA_dt_exp) / abs(dA_dt_exp)
                if rel_error_A > 0.1:
                    print(f"    *** Large dA/dt error: {rel_error_A*100:.1f}% ***")

            if abs(dB_dt_exp) > 1e-6:
                rel_error_B = abs(dB_dt_num - dB_dt_exp) / abs(dB_dt_exp)
                if rel_error_B > 0.1:
                    print(f"    *** Large dB/dt error: {rel_error_B*100:.1f}% ***")

    # Let's also check what the analytical solution is using
    print(f"\nAnalytical solution method analysis:")
    print(f"The Square Law trajectories should follow:")
    print(f"  dA/dt = -β*A*B = -{battle.beta} * A(t) * B(t)")
    print(f"  dB/dt = -α*A*B = -{battle.alpha} * A(t) * B(t)")

    # Check if the hyperbolic solution is correct
    print(f"\nVerifying hyperbolic solution formulation:")
    gamma = np.sqrt(battle.alpha * battle.beta)
    print(f"gamma = sqrt(α*β) = sqrt({battle.alpha}*{battle.beta}) = {gamma}")

    # The correct hyperbolic solutions should be:
    # A(t) = A₀*cosh(γt) - √(β/α)*B₀*sinh(γt)
    # B(t) = B₀*cosh(γt) - √(α/β)*A₀*sinh(γt)

    print(f"\nExpected hyperbolic form:")
    print(f"  A(t) = {battle.A0}*cosh({gamma}*t) - {np.sqrt(battle.beta/battle.alpha)}*{battle.B0}*sinh({gamma}*t)")
    print(f"  B(t) = {battle.B0}*cosh({gamma}*t) - {np.sqrt(battle.alpha/battle.beta)}*{battle.A0}*sinh({gamma}*t)")

    # Test the hyperbolic solution derivatives analytically
    test_time = 0.13  # The problematic time from the error
    print(f"\nAnalytical verification at t = {test_time}:")

    cosh_term = np.cosh(gamma * test_time)
    sinh_term = np.sinh(gamma * test_time)

    A_exact = battle.A0 * cosh_term - np.sqrt(battle.beta / battle.alpha) * battle.B0 * sinh_term
    B_exact = battle.B0 * cosh_term - np.sqrt(battle.alpha / battle.beta) * battle.A0 * sinh_term

    print(f"  A({test_time}) = {A_exact:.6f}")
    print(f"  B({test_time}) = {B_exact:.6f}")

    # Analytical derivatives of the hyperbolic solution
    dA_dt_analytical = gamma * (battle.A0 * sinh_term - np.sqrt(battle.beta / battle.alpha) * battle.B0 * cosh_term)
    dB_dt_analytical = gamma * (battle.B0 * sinh_term - np.sqrt(battle.alpha / battle.beta) * battle.A0 * cosh_term)

    print(f"  dA/dt_analytical = {dA_dt_analytical:.6f}")
    print(f"  dB/dt_analytical = {dB_dt_analytical:.6f}")

    # Expected from differential equation
    dA_dt_from_de = -battle.beta * A_exact * B_exact
    dB_dt_from_de = -battle.alpha * A_exact * B_exact

    print(f"  dA/dt_from_DE = {dA_dt_from_de:.6f}")
    print(f"  dB/dt_from_DE = {dB_dt_from_de:.6f}")

    print(f"\nComparison:")
    print(f"  dA/dt analytical vs DE: {dA_dt_analytical:.6f} vs {dA_dt_from_de:.6f}")
    print(f"  dB/dt analytical vs DE: {dB_dt_analytical:.6f} vs {dB_dt_from_de:.6f}")

    if abs(dA_dt_analytical - dA_dt_from_de) > 1e-10:
        print(f"  *** MISMATCH in dA/dt: difference = {abs(dA_dt_analytical - dA_dt_from_de):.2e} ***")
    else:
        print(f"  ✓ dA/dt matches (difference = {abs(dA_dt_analytical - dA_dt_from_de):.2e})")

    if abs(dB_dt_analytical - dB_dt_from_de) > 1e-10:
        print(f"  *** MISMATCH in dB/dt: difference = {abs(dB_dt_analytical - dB_dt_from_de):.2e} ***")
    else:
        print(f"  ✓ dB/dt matches (difference = {abs(dB_dt_analytical - dB_dt_from_de):.2e})")

    # Let's check what our numerical derivative calculation is actually getting
    test_idx = np.argmin(np.abs(t - test_time))
    print(f"\nNumerical derivative calculation at index {test_idx} (t={t[test_idx]:.6f}):")
    if test_idx > 0 and test_idx < len(t) - 1:
        dA_dt_numerical = (A_t[test_idx+1] - A_t[test_idx-1]) / (2 * dt)
        dB_dt_numerical = (B_t[test_idx+1] - B_t[test_idx-1]) / (2 * dt)
        print(f"  Numerical dA/dt = {dA_dt_numerical:.6f}")
        print(f"  Numerical dB/dt = {dB_dt_numerical:.6f}")

        print(f"  A values: A[{test_idx-1}]={A_t[test_idx-1]:.6f}, A[{test_idx}]={A_t[test_idx]:.6f}, A[{test_idx+1}]={A_t[test_idx+1]:.6f}")
        print(f"  B values: B[{test_idx-1}]={B_t[test_idx-1]:.6f}, B[{test_idx}]={B_t[test_idx]:.6f}, B[{test_idx+1}]={B_t[test_idx+1]:.6f}")

if __name__ == "__main__":
    debug_square_law_derivatives()