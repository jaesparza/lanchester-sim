#!/usr/bin/env python3
"""
Extended Salvo Combat Model Examples

This module demonstrates 8 different battle scenarios showcasing various
configurations and tactical situations in salvo combat. Each scenario
explores different aspects of force composition, defensive capabilities,
and engagement dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import SalvoCombatModel, Ship
import matplotlib.pyplot as plt
import numpy as np


def create_battle_scenarios():
    """Create 8 diverse battle scenarios for demonstration."""

    scenarios = []

    # Scenario 1: Classic Destroyer Duel
    force_a1 = [
        Ship("HMS Striker", offensive_power=12, defensive_power=0.35, staying_power=4),
        Ship("HMS Thunder", offensive_power=10, defensive_power=0.40, staying_power=3)
    ]
    force_b1 = [
        Ship("USS Raptor", offensive_power=11, defensive_power=0.30, staying_power=4),
        Ship("USS Storm", offensive_power=13, defensive_power=0.25, staying_power=3)
    ]
    scenarios.append(("Classic Destroyer Duel", force_a1, force_b1))

    # Scenario 2: Battleship vs Cruiser Squadron
    force_a2 = [
        Ship("Titan Battleship", offensive_power=25, defensive_power=0.60, staying_power=8)
    ]
    force_b2 = [
        Ship("Light Cruiser Alpha", offensive_power=8, defensive_power=0.35, staying_power=3),
        Ship("Light Cruiser Beta", offensive_power=8, defensive_power=0.35, staying_power=3),
        Ship("Heavy Cruiser Gamma", offensive_power=15, defensive_power=0.45, staying_power=5)
    ]
    scenarios.append(("Battleship vs Cruiser Squadron", force_a2, force_b2))

    # Scenario 3: Swarm vs Elite
    force_a3 = [
        Ship("Elite Guardian", offensive_power=20, defensive_power=0.70, staying_power=6),
        Ship("Elite Vanguard", offensive_power=18, defensive_power=0.65, staying_power=5)
    ]
    force_b3 = [
        Ship("Swarm Unit 1", offensive_power=6, defensive_power=0.20, staying_power=2),
        Ship("Swarm Unit 2", offensive_power=6, defensive_power=0.20, staying_power=2),
        Ship("Swarm Unit 3", offensive_power=6, defensive_power=0.20, staying_power=2),
        Ship("Swarm Unit 4", offensive_power=6, defensive_power=0.20, staying_power=2),
        Ship("Swarm Unit 5", offensive_power=6, defensive_power=0.20, staying_power=2)
    ]
    scenarios.append(("Elite Units vs Swarm Tactics", force_a3, force_b3))

    # Scenario 4: Missile Boat Encounter
    force_a4 = [
        Ship("Missile Corvette A1", offensive_power=15, defensive_power=0.25, staying_power=2),
        Ship("Missile Corvette A2", offensive_power=15, defensive_power=0.25, staying_power=2)
    ]
    force_b4 = [
        Ship("Defense Frigate B1", offensive_power=8, defensive_power=0.55, staying_power=4),
        Ship("Defense Frigate B2", offensive_power=8, defensive_power=0.55, staying_power=4)
    ]
    scenarios.append(("Glass Cannons vs Defensive Ships", force_a4, force_b4))

    # Scenario 5: Asymmetric Engagement
    force_a5 = [
        Ship("Heavy Assault Ship", offensive_power=22, defensive_power=0.50, staying_power=7)
    ]
    force_b5 = [
        Ship("Fast Attack Craft 1", offensive_power=10, defensive_power=0.30, staying_power=2),
        Ship("Fast Attack Craft 2", offensive_power=10, defensive_power=0.30, staying_power=2)
    ]
    scenarios.append(("Heavy Hitter vs Fast Attackers", force_a5, force_b5))

    # Scenario 6: Balanced Fleet Action
    force_a6 = [
        Ship("Command Cruiser", offensive_power=14, defensive_power=0.45, staying_power=5),
        Ship("Support Destroyer", offensive_power=10, defensive_power=0.35, staying_power=3),
        Ship("Escort Frigate", offensive_power=7, defensive_power=0.40, staying_power=3)
    ]
    force_b6 = [
        Ship("Battle Cruiser", offensive_power=16, defensive_power=0.40, staying_power=5),
        Ship("Strike Destroyer", offensive_power=12, defensive_power=0.30, staying_power=3),
        Ship("Patrol Corvette", offensive_power=8, defensive_power=0.35, staying_power=2)
    ]
    scenarios.append(("Balanced Fleet Engagement", force_a6, force_b6))

    # Scenario 7: Defensive Specialists
    force_a7 = [
        Ship("Shield Cruiser Alpha", offensive_power=9, defensive_power=0.75, staying_power=6),
        Ship("Shield Cruiser Beta", offensive_power=9, defensive_power=0.75, staying_power=6)
    ]
    force_b7 = [
        Ship("Assault Ship Gamma", offensive_power=18, defensive_power=0.25, staying_power=4),
        Ship("Assault Ship Delta", offensive_power=18, defensive_power=0.25, staying_power=4)
    ]
    scenarios.append(("Shield Wall vs Assault Force", force_a7, force_b7))

    # Scenario 8: Close Match
    force_a8 = [
        Ship("Veteran Destroyer", offensive_power=13, defensive_power=0.42, staying_power=4),
        Ship("Veteran Frigate", offensive_power=9, defensive_power=0.38, staying_power=3)
    ]
    force_b8 = [
        Ship("Elite Corvette 1", offensive_power=11, defensive_power=0.40, staying_power=3),
        Ship("Elite Corvette 2", offensive_power=11, defensive_power=0.40, staying_power=3)
    ]
    scenarios.append(("Evenly Matched Forces", force_a8, force_b8))

    return scenarios


def print_force_summary(name, force):
    """Print a summary of force composition."""
    total_offensive = sum(ship.offensive_power for ship in force)
    avg_defensive = sum(ship.defensive_power for ship in force) / len(force)
    total_staying = sum(ship.staying_power for ship in force)

    print(f"\n{name}:")
    for ship in force:
        print(f"  - {ship.name}: OP={ship.offensive_power}, DP={ship.defensive_power:.2f}, SP={ship.staying_power}")
    print(f"  Total Offensive: {total_offensive}, Avg Defensive: {avg_defensive:.3f}, Total Hull: {total_staying}")


def run_extended_salvo_examples():
    """Run all 8 salvo combat scenarios and display results."""

    print("="*80)
    print("EXTENDED SALVO COMBAT MODEL EXAMPLES")
    print("="*80)

    scenarios = create_battle_scenarios()
    results = []

    # Run all scenarios
    for i, (title, force_a, force_b) in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {title.upper()}")
        print('='*60)

        print_force_summary("Force Alpha", force_a)
        print_force_summary("Force Bravo", force_b)

        # Create battle model with fixed seed for reproducible results
        battle = SalvoCombatModel(force_a, force_b, random_seed=42 + i)

        # Run simulation
        print(f"\n=== BATTLE SIMULATION ===")
        outcome = battle.run_simulation(quiet=False)
        stats = battle.get_battle_statistics()

        # Store results for plotting
        results.append({
            'title': title,
            'force_a': force_a.copy(),
            'force_b': force_b.copy(),
            'outcome': outcome,
            'stats': stats,
            'battle': battle
        })

        print(f"\n=== BATTLE RESULT ===")
        print(f"Winner: {outcome}")
        print(f"Battle Duration: {stats['rounds']} round(s)")
        print(f"Force Alpha Survivors: {stats['force_a_survivors']}")
        print(f"Force Bravo Survivors: {stats['force_b_survivors']}")
        print(f"Offensive Ratio (A/B): {stats['offensive_ratio']:.2f}")

        # Display surviving ships
        if stats['surviving_ships_a']:
            print("Force Alpha survivors:")
            for ship in stats['surviving_ships_a']:
                health_pct = ((ship.staying_power - ship.current_hits) / ship.staying_power) * 100
                print(f"  - {ship.name}: {health_pct:.1f}% health")

        if stats['surviving_ships_b']:
            print("Force Bravo survivors:")
            for ship in stats['surviving_ships_b']:
                health_pct = ((ship.staying_power - ship.current_hits) / ship.staying_power) * 100
                print(f"  - {ship.name}: {health_pct:.1f}% health")

    # Create visualization
    plot_battle_results(results)

    return results


def plot_battle_results(results):
    """Create a 4x2 grid plot showing all 8 battle scenarios."""

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Extended Salvo Combat Examples\n8 Battle Scenarios', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    for i, result in enumerate(results):
        ax = axes_flat[i]
        stats = result['stats']
        title = result['title']

        # Calculate force compositions for visualization
        force_a_total_op = sum(ship.offensive_power for ship in result['force_a'])
        force_b_total_op = sum(ship.offensive_power for ship in result['force_b'])
        force_a_ships = len(result['force_a'])
        force_b_ships = len(result['force_b'])

        # Create bar chart showing initial vs final forces
        categories = ['Initial\nShips A', 'Surviving\nShips A', 'Initial\nShips B', 'Surviving\nShips B']
        values = [force_a_ships, stats['force_a_survivors'], force_b_ships, stats['force_b_survivors']]
        colors = ['lightblue', 'blue', 'lightcoral', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold')

        # Customize subplot
        ax.set_title(f"{i+1}. {title}\n{result['outcome']}", fontweight='bold', fontsize=11)
        ax.set_ylabel('Number of Ships')
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis='y', alpha=0.3)

        # Add additional information as text
        info_text = f"Rounds: {stats['rounds']}\n"
        info_text += f"A/B Ratio: {stats['offensive_ratio']:.2f}\n"
        info_text += f"A Power: {force_a_total_op}\n"
        info_text += f"B Power: {force_b_total_op}"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
               facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()


def print_summary_analysis(results):
    """Print a summary analysis of all battle outcomes."""

    print("\n" + "="*80)
    print("BATTLE OUTCOMES SUMMARY")
    print("="*80)

    force_a_wins = sum(1 for r in results if 'Force A' in r['outcome'])
    force_b_wins = sum(1 for r in results if 'Force B' in r['outcome'])
    draws = len(results) - force_a_wins - force_b_wins

    print(f"Total Battles: {len(results)}")
    print(f"Force Alpha Victories: {force_a_wins}")
    print(f"Force Bravo Victories: {force_b_wins}")
    print(f"Draws: {draws}")

    print(f"\nBattle Analysis:")
    for i, result in enumerate(results, 1):
        stats = result['stats']
        force_a_power = sum(ship.offensive_power for ship in result['force_a'])
        force_b_power = sum(ship.offensive_power for ship in result['force_b'])

        print(f"{i:2d}. {result['title']:<30} | "
              f"Winner: {result['outcome']:<18} | "
              f"Rounds: {stats['rounds']:2d} | "
              f"Ratio: {stats['offensive_ratio']:5.2f} | "
              f"Powers: {force_a_power:2d} vs {force_b_power:2d}")

    # Calculate average battle duration
    avg_rounds = sum(r['stats']['rounds'] for r in results) / len(results)
    print(f"\nAverage Battle Duration: {avg_rounds:.1f} rounds")

    # Find most decisive and closest battles
    casualty_rates = []
    for result in results:
        total_initial = len(result['force_a']) + len(result['force_b'])
        total_survivors = result['stats']['force_a_survivors'] + result['stats']['force_b_survivors']
        casualty_rate = (total_initial - total_survivors) / total_initial
        casualty_rates.append((result['title'], casualty_rate))

    most_decisive = max(casualty_rates, key=lambda x: x[1])
    least_decisive = min(casualty_rates, key=lambda x: x[1])

    print(f"\nMost Decisive Battle: {most_decisive[0]} ({most_decisive[1]:.1%} casualties)")
    print(f"Least Decisive Battle: {least_decisive[0]} ({least_decisive[1]:.1%} casualties)")


if __name__ == "__main__":
    print("Running Extended Salvo Combat Examples...")
    results = run_extended_salvo_examples()
    print_summary_analysis(results)
    print("\nAll scenarios completed!")