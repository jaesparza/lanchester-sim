#!/usr/bin/env python3
"""
Simple examples demonstrating the Lanchester simulation models.

Run this file directly to see all models in action:
    python examples.py
"""

from lanchester_sim import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship
import numpy as np


def linear_example():
    """Example using Lanchester Linear Law"""
    print("="*50)
    print("LANCHESTER LINEAR LAW EXAMPLE")
    print("="*50)

    battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution = battle.simple_analytical_solution()

    print(f"Winner: {solution['winner']}")
    print(f"Battle duration: {solution['battle_end_time']:.2f} time units")
    print(f"Survivors: {solution['remaining_strength']:.1f}")
    print(f"Linear Law insight: {battle.A0} - {battle.B0} = {battle.A0 - battle.B0}")
    print()


def square_example():
    """Example using Lanchester Square Law"""
    print("="*50)
    print("LANCHESTER SQUARE LAW EXAMPLE")
    print("="*50)

    battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution = battle.simple_analytical_solution()

    print(f"Winner: {solution['winner']}")
    print(f"Battle duration: {solution['battle_end_time']:.2f} time units")
    print(f"Survivors: {solution['remaining_strength']:.1f}")
    print(f"Square Law insight: sqrt({battle.A0}² - {battle.B0}²) = {np.sqrt(battle.A0**2 - battle.B0**2):.1f}")
    print()


def salvo_example():
    """Example using Salvo Combat Model"""
    print("="*50)
    print("SALVO COMBAT MODEL EXAMPLE")
    print("="*50)

    # Create forces
    force_a = [
        Ship("Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
        Ship("Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5)
    ]

    force_b = [
        Ship("Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
        Ship("Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4)
    ]

    simulation = SalvoCombatModel(force_a, force_b, random_seed=42)
    result = simulation.run_simulation()

    print(f"Result: {result}")
    print()


def comparison_example():
    """Compare all three models"""
    print("="*50)
    print("MODEL COMPARISON")
    print("="*50)

    # Same initial forces for comparison
    A0, B0 = 100, 80

    # Linear Law
    linear = LanchesterLinear(A0, B0, alpha=0.5, beta=0.5)
    linear_result = linear.simple_analytical_solution()

    # Square Law
    square = LanchesterSquare(A0, B0, alpha=0.01, beta=0.01)
    square_result = square.simple_analytical_solution()

    print(f"Initial forces: A={A0}, B={B0}")
    print(f"Linear Law winner: {linear_result['winner']} with {linear_result['remaining_strength']:.1f} survivors")
    print(f"Square Law winner: {square_result['winner']} with {square_result['remaining_strength']:.1f} survivors")
    print(f"Square Law advantage: {square_result['remaining_strength'] - linear_result['remaining_strength']:.1f} more survivors")
    print()

    print("Key differences:")
    print("- Linear Law: suitable for hand-to-hand combat, guerrilla warfare")
    print("- Square Law: suitable for modern ranged combat with concentration effects")
    print("- Salvo Model: discrete rounds, individual ships, defensive capabilities")


if __name__ == "__main__":
    linear_example()
    square_example()
    salvo_example()
    comparison_example()