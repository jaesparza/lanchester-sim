#!/usr/bin/env python3
"""
Simple examples demonstrating the Lanchester simulation models.

Run this file directly to see all models in action:
    python examples.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports from models package
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship
import numpy as np
import matplotlib.pyplot as plt


def linear_example():
    """Example using Lanchester Linear Law"""
    print("="*50)
    print("LANCHESTER LINEAR LAW EXAMPLE")
    print("="*50)

    battle = LanchesterLinear(A0=100, B0=80, alpha=0.01, beta=0.01)
    solution = battle.simple_analytical_solution()

    print(f"Winner: {solution['winner']}")
    print(f"Battle duration: {solution['battle_end_time']:.2f} time units")
    print(f"Survivors: {solution['remaining_strength']:.1f}")
    print(f"Linear Law insight: α{battle.A0} - β{battle.B0} = {battle.alpha * battle.A0 - battle.beta * battle.B0:.2f}")
    print()

    return battle, solution


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

    return battle, solution


def salvo_example():
    """Example using Salvo Combat Model with multiple battle scenarios"""
    print("="*50)
    print("SALVO COMBAT MODEL EXAMPLES")
    print("="*50)

    scenarios = []

    # Scenario 1: Balanced Fleet Engagement
    print("Scenario 1: Balanced Fleet Engagement")
    force_a1 = [
        Ship("Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
        Ship("Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5)
    ]
    force_b1 = [
        Ship("Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
        Ship("Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4)
    ]
    sim1 = SalvoCombatModel(force_a1, force_b1, random_seed=42)
    result1 = sim1.run_simulation()
    print(f"  Result: {result1}")
    scenarios.append(("Balanced Fleet", sim1))

    # Scenario 2: Quality vs Quantity (Quantity wins)
    print("\nScenario 2: Quality vs Quantity")
    force_a2 = [
        Ship("Battleship Titan", offensive_power=12, defensive_power=0.5, staying_power=6)
    ]
    force_b2 = [
        Ship("Corvette 1", offensive_power=6, defensive_power=0.3, staying_power=2),
        Ship("Corvette 2", offensive_power=6, defensive_power=0.3, staying_power=2),
        Ship("Corvette 3", offensive_power=6, defensive_power=0.3, staying_power=2),
        Ship("Corvette 4", offensive_power=6, defensive_power=0.3, staying_power=2),
        Ship("Corvette 5", offensive_power=6, defensive_power=0.3, staying_power=2)
    ]
    sim2 = SalvoCombatModel(force_a2, force_b2, random_seed=42)
    result2 = sim2.run_simulation()
    print(f"  Result: {result2}")
    scenarios.append(("Quality vs Quantity", sim2))

    # Scenario 3: Extended Battle with Multiple Rounds
    print("\nScenario 3: Extended Multi-Round Battle")
    force_a3 = [
        Ship("Light Cruiser", offensive_power=5, defensive_power=0.6, staying_power=3),
        Ship("Support Frigate", offensive_power=4, defensive_power=0.5, staying_power=2)
    ]
    force_b3 = [
        Ship("Attack Corvette", offensive_power=6, defensive_power=0.4, staying_power=3),
        Ship("Defense Patrol", offensive_power=3, defensive_power=0.7, staying_power=4)
    ]
    sim3 = SalvoCombatModel(force_a3, force_b3, random_seed=42)
    result3 = sim3.run_simulation()
    print(f"  Result: {result3}")
    scenarios.append(("Extended Battle", sim3))

    print()
    return scenarios


def comparison_example():
    """Compare all three models"""
    print("="*50)
    print("MODEL COMPARISON")
    print("="*50)

    # Same initial forces for comparison
    A0, B0 = 100, 80

    # Linear Law
    linear = LanchesterLinear(A0, B0, alpha=0.01, beta=0.01)
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

    return linear, linear_result, square, square_result


def plot_all_battles():
    """Plot all battle examples in a single canvas with multiple subplots."""
    print("="*50)
    print("BATTLE VISUALIZATIONS")
    print("="*50)

    # Run all examples and collect results
    linear_battle, linear_solution = linear_example()
    square_battle, square_solution = square_example()
    salvo_scenarios = salvo_example()
    comp_linear, comp_linear_result, comp_square, comp_square_result = comparison_example()

    # Create a figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Lanchester and Salvo Combat Models - All Examples', fontsize=16)

    # Plot 1: Linear Law Example
    linear_battle.plot_battle(solution=linear_solution,
                             title="Linear Law Example",
                             ax=axes[0, 0])

    # Plot 2: Square Law Example
    square_battle.plot_battle(solution=square_solution,
                             title="Square Law Example",
                             ax=axes[0, 1])

    # Plot 3: Salvo Combat Models - Bar Chart Comparison
    axes[0, 2].set_title("Salvo Combat Scenarios\nInitial vs Survivors")

    # Create bar chart showing initial vs surviving forces
    scenario_labels = ['Balanced', 'Quality vs\nQuantity', 'Extended']
    x_pos = np.arange(len(scenario_labels))
    width = 0.2

    # Collect data
    initial_a = []
    survivors_a = []
    initial_b = []
    survivors_b = []

    for scenario_name, simulation in salvo_scenarios:
        stats = simulation.get_battle_statistics()
        initial_a.append(len(simulation.force_a))
        survivors_a.append(stats['force_a_survivors'])
        initial_b.append(len(simulation.force_b))
        survivors_b.append(stats['force_b_survivors'])

    # Plot bars
    axes[0, 2].bar(x_pos - 1.5*width, initial_a, width, label='Force A Initial',
                   color='lightblue', edgecolor='blue', linewidth=1.5)
    axes[0, 2].bar(x_pos - 0.5*width, survivors_a, width, label='Force A Survivors',
                   color='darkblue', edgecolor='blue', linewidth=1.5)
    axes[0, 2].bar(x_pos + 0.5*width, initial_b, width, label='Force B Initial',
                   color='lightcoral', edgecolor='red', linewidth=1.5)
    axes[0, 2].bar(x_pos + 1.5*width, survivors_b, width, label='Force B Survivors',
                   color='darkred', edgecolor='red', linewidth=1.5)

    axes[0, 2].set_xlabel('Scenario')
    axes[0, 2].set_ylabel('Number of Ships')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(scenario_labels, fontsize=9)
    axes[0, 2].legend(fontsize=7, loc='upper right', ncol=2)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_ylim(bottom=0)

    # Plot 4: Direct Comparison - Normalized Time Scale (0-100% completion)
    # Normalize time to battle completion percentage for fair comparison
    linear_time_norm = 100 * comp_linear_result['time'] / comp_linear_result['battle_end_time']
    square_time_norm = 100 * comp_square_result['time'] / comp_square_result['battle_end_time']

    axes[1, 0].plot(linear_time_norm, comp_linear_result['A'], 'b-', linewidth=2, label=f'Linear: Force A')
    axes[1, 0].plot(linear_time_norm, comp_linear_result['B'], 'r-', linewidth=2, label=f'Linear: Force B')
    axes[1, 0].plot(square_time_norm, comp_square_result['A'], 'b--', linewidth=2, alpha=0.7, label=f'Square: Force A')
    axes[1, 0].plot(square_time_norm, comp_square_result['B'], 'r--', linewidth=2, alpha=0.7, label=f'Square: Force B')
    axes[1, 0].set_xlabel('Battle Completion (%)')
    axes[1, 0].set_ylabel('Force Strength')
    axes[1, 0].set_title('Model Comparison\n(Normalized Time: Linear=8000, Square=110 units)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(0, max(comp_linear.A0, comp_linear.B0) * 1.1)

    # Plot 5: Force Advantage Over Time (normalized time scale)
    linear_advantage = comp_linear_result['A'] - comp_linear_result['B']
    square_advantage = comp_square_result['A'] - comp_square_result['B']
    axes[1, 1].plot(linear_time_norm, linear_advantage, 'g-', linewidth=2, label='Linear Law Advantage')
    axes[1, 1].plot(square_time_norm, square_advantage, 'purple', linestyle='--', linewidth=2, label='Square Law Advantage')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Battle Completion (%)')
    axes[1, 1].set_ylabel('Force Advantage (A - B)')
    axes[1, 1].set_title('Force Advantage Evolution\n(Positive = A Winning)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 100)

    # Plot 6: Model Comparison Summary
    axes[1, 2].text(0.1, 0.8, "Model Comparison Summary:", fontsize=12, weight='bold')
    axes[1, 2].text(0.1, 0.7, f"Same initial forces: A={comp_linear.A0}, B={comp_linear.B0}", fontsize=10)
    axes[1, 2].text(0.1, 0.6, f"Linear: {comp_linear_result['winner']} wins, {comp_linear_result['remaining_strength']:.0f} survivors", fontsize=10)
    axes[1, 2].text(0.1, 0.55, f"  Battle time: {comp_linear_result['battle_end_time']:.0f} units", fontsize=9)
    axes[1, 2].text(0.1, 0.5, f"Square: {comp_square_result['winner']} wins, {comp_square_result['remaining_strength']:.0f} survivors", fontsize=10)
    axes[1, 2].text(0.1, 0.45, f"  Battle time: {comp_square_result['battle_end_time']:.0f} units", fontsize=9)
    axes[1, 2].text(0.1, 0.35, "Key Differences:", fontsize=11, weight='bold')
    axes[1, 2].text(0.1, 0.25, "• Linear: hand-to-hand combat", fontsize=9)
    axes[1, 2].text(0.1, 0.2, "• Square: modern ranged combat", fontsize=9)
    axes[1, 2].text(0.1, 0.15, "• Salvo: 3 scenarios demonstrate", fontsize=9)
    axes[1, 2].text(0.1, 0.1, "  different tactical situations", fontsize=9)
    axes[1, 2].text(0.1, 0.05, "Time scales normalized for comparison", fontsize=8, style='italic')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print("All battle visualizations complete!")


if __name__ == "__main__":
    plot_all_battles()