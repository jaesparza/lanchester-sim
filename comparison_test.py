#!/usr/bin/env python3
"""
Compare original jaesparza implementation vs current restored implementation
of Lanchester Linear Law side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current implementation
sys.path.insert(0, os.getcwd())
from models import LanchesterLinear as CurrentLinear

# Original implementation (from August 2025)
class OriginalLanchesterLinear:
    """
    Implementation of Lanchester's Linear Law for ancient-style combat.

    The Linear Law assumes sequential combat where forces engage one-on-one.
    Combat effectiveness is proportional to force size.
    """

    def __init__(self, A0, B0, alpha, beta):
        """
        Initialize the combat scenario.

        Parameters:
        A0 (float): Initial strength of force A
        B0 (float): Initial strength of force B
        alpha (float): Effectiveness coefficient of A against B
        beta (float): Effectiveness coefficient of B against A
        """
        self.A0 = A0
        self.B0 = B0
        self.alpha = alpha
        self.beta = beta

    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Linear Law.
        Forces decrease linearly until one is eliminated.

        Returns:
        dict: Contains time arrays, force strengths, battle end time, and winner
        """
        # Calculate when each force would be eliminated
        t_A_eliminated = self.A0 / self.beta if self.beta > 0 else np.inf
        t_B_eliminated = self.B0 / self.alpha if self.alpha > 0 else np.inf

        # Battle ends when first force is eliminated
        t_end = min(t_A_eliminated, t_B_eliminated)

        # Determine winner
        if t_A_eliminated < t_B_eliminated:
            winner = 'B'
            remaining_strength = self.B0 - self.alpha * t_end
        elif t_B_eliminated < t_A_eliminated:
            winner = 'A'
            remaining_strength = self.A0 - self.beta * t_end
        else:
            winner = 'Draw'
            remaining_strength = 0

        # Create time array
        if t_max is None:
            t_max = min(t_end * 1.2, t_end + 1)  # Show a bit beyond battle end

        t = np.linspace(0, t_max, 1000)

        # Calculate force strengths over time
        A_t = np.maximum(0, self.A0 - self.beta * t)
        B_t = np.maximum(0, self.B0 - self.alpha * t)

        return {
            'time': t,
            'A': A_t,
            'B': B_t,
            'battle_end_time': t_end,
            'winner': winner,
            'remaining_strength': remaining_strength,
            'A_casualties': self.A0 - (self.A0 - self.beta * t_end if winner != 'A' else remaining_strength),
            'B_casualties': self.B0 - (self.B0 - self.alpha * t_end if winner != 'B' else remaining_strength)
        }


def run_comparison():
    """Run side-by-side comparison of original vs current implementation."""

    # Test scenario
    A0, B0 = 100, 80
    alpha, beta = 0.5, 0.6

    print("="*60)
    print("LANCHESTER LINEAR LAW IMPLEMENTATION COMPARISON")
    print("="*60)
    print(f"Test scenario: A0={A0}, B0={B0}, alpha={alpha}, beta={beta}")
    print()

    # Original implementation
    original = OriginalLanchesterLinear(A0, B0, alpha, beta)
    original_solution = original.analytical_solution()

    print("ORIGINAL IMPLEMENTATION (jaesparza, August 2025):")
    print(f"Winner: {original_solution['winner']}")
    print(f"Battle duration: {original_solution['battle_end_time']:.2f} time units")
    print(f"Survivors: {original_solution['remaining_strength']:.1f}")
    print(f"A casualties: {original_solution['A_casualties']:.1f}")
    print(f"B casualties: {original_solution['B_casualties']:.1f}")
    print()

    # Current implementation
    current = CurrentLinear(A0, B0, alpha, beta)
    current_solution = current.analytical_solution()

    print("CURRENT IMPLEMENTATION (restored):")
    print(f"Winner: {current_solution['winner']}")
    print(f"Battle duration: {current_solution['battle_end_time']:.2f} time units")
    print(f"Survivors: {current_solution['remaining_strength']:.1f}")
    print(f"A casualties: {current_solution['A_casualties']:.1f}")
    print(f"B casualties: {current_solution['B_casualties']:.1f}")
    print()

    # Check for differences
    print("COMPARISON:")
    winner_match = original_solution['winner'] == current_solution['winner']
    time_match = abs(original_solution['battle_end_time'] - current_solution['battle_end_time']) < 0.01
    survivor_match = abs(original_solution['remaining_strength'] - current_solution['remaining_strength']) < 0.1

    print(f"Winner match: {'✅' if winner_match else '❌'} ({original_solution['winner']} vs {current_solution['winner']})")
    print(f"Battle time match: {'✅' if time_match else '❌'} ({original_solution['battle_end_time']:.2f} vs {current_solution['battle_end_time']:.2f})")
    print(f"Survivors match: {'✅' if survivor_match else '❌'} ({original_solution['remaining_strength']:.1f} vs {current_solution['remaining_strength']:.1f})")
    print(f"Overall: {'✅ IMPLEMENTATIONS MATCH' if all([winner_match, time_match, survivor_match]) else '❌ IMPLEMENTATIONS DIFFER'}")
    print()

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original implementation plot
    ax1.plot(original_solution['time'], original_solution['A'], 'b-', linewidth=2, label=f'Force A (initial: {A0})')
    ax1.plot(original_solution['time'], original_solution['B'], 'r-', linewidth=2, label=f'Force B (initial: {B0})')
    ax1.axvline(x=original_solution['battle_end_time'], color='gray', linestyle='--', alpha=0.7,
                label=f"Battle ends: t={original_solution['battle_end_time']:.2f}")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force Strength')
    ax1.set_title(f"ORIGINAL (jaesparza Aug 2025)\\nWinner: {original_solution['winner']}, Survivors: {original_solution['remaining_strength']:.1f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(original_solution['time']))
    ax1.set_ylim(0, max(A0, B0) * 1.1)

    # Current implementation plot
    ax2.plot(current_solution['time'], current_solution['A'], 'b-', linewidth=2, label=f'Force A (initial: {A0})')
    ax2.plot(current_solution['time'], current_solution['B'], 'r-', linewidth=2, label=f'Force B (initial: {B0})')
    ax2.axvline(x=current_solution['battle_end_time'], color='gray', linestyle='--', alpha=0.7,
                label=f"Battle ends: t={current_solution['battle_end_time']:.2f}")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Force Strength')
    ax2.set_title(f"CURRENT (restored)\\nWinner: {current_solution['winner']}, Survivors: {current_solution['remaining_strength']:.1f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(current_solution['time']))
    ax2.set_ylim(0, max(A0, B0) * 1.1)

    plt.tight_layout()
    plt.suptitle(f'Lanchester Linear Law Comparison\\nA0={A0}, B0={B0}, α={alpha}, β={beta}', y=1.02)
    plt.show()

    # Check trajectory consistency at key points
    print("TRAJECTORY ANALYSIS:")
    print("Time points where both have values:")
    min_time_len = min(len(original_solution['time']), len(current_solution['time']))
    for i in range(0, min_time_len, min_time_len//5):
        t_orig = original_solution['time'][i]
        t_curr = current_solution['time'][i]
        A_orig = original_solution['A'][i]
        A_curr = current_solution['A'][i]
        B_orig = original_solution['B'][i]
        B_curr = current_solution['B'][i]

        print(f"t≈{t_orig:.1f}: Original(A={A_orig:.1f},B={B_orig:.1f}) vs Current(A={A_curr:.1f},B={B_curr:.1f})")


if __name__ == "__main__":
    run_comparison()