#!/usr/bin/env python3
"""
Example runner for all Lanchester simulation models.

This script runs all the example cases from the models to verify they work correctly.
It addresses import path issues that occur when running model files directly.
"""

import math
import sys
import os

# Add project root to Python path to enable model imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    print("=" * 60)
    print("LANCHESTER SIMULATION MODEL EXAMPLES")
    print("=" * 60)
    print()

    try:
        # Test imports as specified in CLAUDE.md
        print("Testing imports...")
        from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship
        from models.ode_solver_lanchseter_linear import LanchesterLinearODESolver
        from models.ode_solver_lanchester_square import LanchesterSquareODESolver
        print("✅ All model imports successful")
        print()

        def float_close(a, b, tol=1e-9):
            """Helper to compare floats with tolerance and handle infinities."""
            if a is None or b is None:
                return a is None and b is None
            if math.isinf(a) and math.isinf(b):
                return True
            return abs(a - b) <= tol

        def verify_consistency(model_label, scenario_label, analytical_solution, ode_solution, extra_fields=None):
            """Ensure analytical and ODE implementations agree on battle outcome."""
            extra_fields = extra_fields or []
            mismatches = []

            if analytical_solution.get('winner') != ode_solution.get('winner'):
                mismatches.append(
                    f"winner mismatch (analytical={analytical_solution.get('winner')}, "
                    f"ode={ode_solution.get('winner')})"
                )

            numeric_fields = ['remaining_strength', 'battle_end_time', 'A_casualties', 'B_casualties']
            numeric_fields.extend(extra_fields)

            for field in numeric_fields:
                if field in analytical_solution and field in ode_solution:
                    if not float_close(analytical_solution[field], ode_solution[field]):
                        mismatches.append(
                            f"{field} mismatch (analytical={analytical_solution[field]:.6f}, "
                            f"ode={ode_solution[field]:.6f})"
                        )

            if mismatches:
                details = "; ".join(mismatches)
                raise AssertionError(
                    f"{model_label} [{scenario_label}] inconsistency detected: {details}"
                )

            print(f"   ✅ {model_label} ({scenario_label}) analytical vs ODE results match")

        def verify_linear_implementations(label, battle):
            """Compare Linear Law analytical implementation with ODE solver."""
            analytical_solution = battle.analytical_solution()
            solver = LanchesterLinearODESolver(battle.A0, battle.B0, battle.alpha, battle.beta)
            ode_solution = solver.numerical_solution(
                t_max=analytical_solution['time'][-1],
                num_points=len(analytical_solution['time'])
            )
            verify_consistency("Linear Law", label, analytical_solution, ode_solution, extra_fields=['linear_advantage'])

        def verify_square_implementations(label, battle):
            """Compare Square Law analytical implementation with ODE solver."""
            analytical_solution = battle.analytical_solution()
            solver = LanchesterSquareODESolver(battle.A0, battle.B0, battle.alpha, battle.beta)
            ode_solution = solver.numerical_solution(
                t_max=analytical_solution['time'][-1],
                num_points=len(analytical_solution['time'])
            )
            verify_consistency("Square Law", label, analytical_solution, ode_solution, extra_fields=['invariant'])

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

        # Implementation verification (analytical vs ODE versions)
        print("=" * 40)
        print("IMPLEMENTATION CONSISTENCY CHECKS")
        print("=" * 40)

        verify_linear_implementations("Numerical Advantage - Force A Superior", battle1)
        verify_linear_implementations("Superior Effectiveness vs. Numbers", battle2)

        verify_square_implementations("Equal Effectiveness - Size Matters", square1)
        verify_square_implementations("Superior Effectiveness vs. Numbers", square2)

        square_draw = LanchesterSquare(A0=100, B0=100, alpha=0.01, beta=0.01)
        verify_square_implementations("Equal Forces Draw", square_draw)
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
