#!/usr/bin/env python3
"""
Example runner for all Lanchester simulation models.

This script runs all the example cases from the models to verify they work correctly.
It addresses import path issues that occur when running model files directly.
"""

import sys
import os

# Add project root to Python path to enable model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("LANCHESTER SIMULATION MODEL EXAMPLES")
    print("=" * 60)
    print()

    try:
        # Test imports as specified in CLAUDE.md
        print("Testing imports...")
        from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship
        print("✅ All model imports successful")
        print()

        # Run Linear Law examples
        print("=" * 40)
        print("LANCHESTER LINEAR LAW EXAMPLES")
        print("=" * 40)

        print("Example 1: Numerical Advantage - Force A Superior")
        battle1 = LanchesterLinear(A0=100, B0=60, alpha=0.01, beta=0.01)
        solution1 = battle1.simple_analytical_solution()

        print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
        print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
        print(f"Linear Law advantage: αA₀ - βB₀ = {battle1.alpha}×{battle1.A0} - {battle1.beta}×{battle1.B0} = {battle1.alpha * battle1.A0 - battle1.beta * battle1.B0:.2f}")
        print()

        print("Example 2: Superior Effectiveness vs. Numbers")
        battle2 = LanchesterLinear(A0=80, B0=120, alpha=0.02, beta=0.01)
        solution2 = battle2.analytical_solution()

        print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
        print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
        print(f"Linear Law advantage: αA₀ - βB₀ = {battle2.alpha}×{battle2.A0} - {battle2.beta}×{battle2.B0} = {battle2.alpha * battle2.A0 - battle2.beta * battle2.B0:.2f}")
        print()

        # Run Square Law examples
        print("=" * 40)
        print("LANCHESTER SQUARE LAW EXAMPLES")
        print("=" * 40)

        print("Example 1: Equal Effectiveness - Size Matters")
        square1 = LanchesterSquare(A0=100, B0=60, alpha=0.01, beta=0.01)
        square_solution1 = square1.simple_analytical_solution()

        print(f"Battle ends at t = {square_solution1['battle_end_time']:.2f}")
        print(f"Winner: {square_solution1['winner']} with {square_solution1['remaining_strength']:.1f} units remaining")
        print(f"Square Law prediction: sqrt({square1.A0}² - {square1.B0}²) = {square_solution1['remaining_strength']:.1f}")
        print()

        print("Example 2: Superior Effectiveness vs. Numbers")
        square2 = LanchesterSquare(A0=80, B0=120, alpha=0.02, beta=0.01)
        square_solution2 = square2.analytical_solution()

        print(f"Battle ends at t = {square_solution2['battle_end_time']:.2f}")
        print(f"Winner: {square_solution2['winner']} with {square_solution2['remaining_strength']:.1f} units remaining")
        print(f"Invariant: {square_solution2['invariant']:.0f} ({'A wins' if square_solution2['invariant'] > 0 else 'B wins' if square_solution2['invariant'] < 0 else 'Draw'})")
        print()

        # Run Salvo Model examples
        print("=" * 40)
        print("SALVO COMBAT MODEL EXAMPLES")
        print("=" * 40)

        print("Example 1: Balanced Naval Forces")
        force_a1 = [
            Ship("Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
            Ship("Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5)
        ]
        force_b1 = [
            Ship("Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
            Ship("Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4)
        ]

        sim1 = SalvoCombatModel(force_a1, force_b1, random_seed=42)
        result1 = sim1.run_simulation(quiet=True)
        print(f"Result: {result1}")

        # Also get battle statistics for more detail
        stats = sim1.get_battle_statistics()
        print(f"Battle duration: {stats['rounds']} rounds")
        print()

        # Model comparison
        print("=" * 40)
        print("MODEL COMPARISON")
        print("=" * 40)

        print(f"Initial forces: A={battle1.A0}, B={battle1.B0}")
        print(f"Linear Law winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} survivors")
        print(f"Square Law winner: {square_solution1['winner']} with {square_solution1['remaining_strength']:.1f} survivors")
        print(f"Square Law advantage: {square_solution1['remaining_strength'] - solution1['remaining_strength']:.1f} more survivors")
        print()

        print("✅ All model examples completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()