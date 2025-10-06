#!/usr/bin/env python3
"""
Extended Salvo Combat Model Examples

This module demonstrates the Salvo Combat Model through diverse tactical scenarios.
The examples illustrate how ship characteristics (offensive power, defensive power,
staying power) interact to produce different combat outcomes.

Key Insights:
- Quality vs Quantity: A single powerful ship may lose to multiple weaker units
- Defense Matters: High defensive power can neutralize superior firepower
- Concentration of Fire: Smaller fleets often suffer from focusing limitations
- Glass Cannons: High offense/low defense ships are vulnerable to attrition
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

    # Scenario 1: Close Match - Alpha Advantage
    force_a1 = [
        Ship("HMS Striker", offensive_power=12, defensive_power=0.35, staying_power=4),
        Ship("HMS Thunder", offensive_power=10, defensive_power=0.40, staying_power=3)
    ]
    force_b1 = [
        Ship("USS Raptor", offensive_power=11, defensive_power=0.30, staying_power=4),
        Ship("USS Storm", offensive_power=11, defensive_power=0.25, staying_power=3)
    ]
    scenarios.append(("Close Match - Alpha Advantage", force_a1, force_b1))

    # Scenario 2: Quantity Overwhelms Quality - Bravo Wins
    force_a2 = [
        Ship("Titan Battleship", offensive_power=18, defensive_power=0.45, staying_power=5)
    ]
    force_b2 = [
        Ship("Light Cruiser Alpha", offensive_power=12, defensive_power=0.40, staying_power=4),
        Ship("Light Cruiser Beta", offensive_power=12, defensive_power=0.40, staying_power=4),
        Ship("Heavy Cruiser Gamma", offensive_power=16, defensive_power=0.45, staying_power=6)
    ]
    scenarios.append(("Quantity Overwhelms Quality", force_a2, force_b2))

    # Scenario 3: Swarm Overwhelms Elite - Bravo Wins
    force_a3 = [
        Ship("Elite Guardian", offensive_power=12, defensive_power=0.45, staying_power=3),
        Ship("Elite Vanguard", offensive_power=12, defensive_power=0.45, staying_power=3)
    ]
    force_b3 = [
        Ship("Swarm Unit 1", offensive_power=11, defensive_power=0.35, staying_power=3),
        Ship("Swarm Unit 2", offensive_power=11, defensive_power=0.35, staying_power=3),
        Ship("Swarm Unit 3", offensive_power=11, defensive_power=0.35, staying_power=3),
        Ship("Swarm Unit 4", offensive_power=11, defensive_power=0.35, staying_power=3),
        Ship("Swarm Unit 5", offensive_power=11, defensive_power=0.35, staying_power=3)
    ]
    scenarios.append(("Swarm Overwhelms Elite", force_a3, force_b3))

    # Scenario 4: Defense Beats Offense - Bravo Wins
    force_a4 = [
        Ship("Glass Cannon A1", offensive_power=16, defensive_power=0.15, staying_power=2),
        Ship("Glass Cannon A2", offensive_power=16, defensive_power=0.15, staying_power=2)
    ]
    force_b4 = [
        Ship("Defense Frigate B1", offensive_power=14, defensive_power=0.65, staying_power=6),
        Ship("Defense Frigate B2", offensive_power=14, defensive_power=0.65, staying_power=6)
    ]
    scenarios.append(("Defense Beats Offense", force_a4, force_b4))

    # Scenario 5: Alpha Quality Wins
    force_a5 = [
        Ship("Heavy Battlecruiser", offensive_power=28, defensive_power=0.55, staying_power=8)
    ]
    force_b5 = [
        Ship("Fast Attack Craft 1", offensive_power=12, defensive_power=0.30, staying_power=2),
        Ship("Fast Attack Craft 2", offensive_power=12, defensive_power=0.30, staying_power=2)
    ]
    scenarios.append(("Alpha Quality Wins", force_a5, force_b5))

    # Scenario 6: Bravo Fleet Superiority
    force_a6 = [
        Ship("Command Cruiser", offensive_power=10, defensive_power=0.35, staying_power=3),
        Ship("Support Destroyer", offensive_power=8, defensive_power=0.30, staying_power=2),
        Ship("Escort Frigate", offensive_power=6, defensive_power=0.30, staying_power=2)
    ]
    force_b6 = [
        Ship("Battle Cruiser", offensive_power=17, defensive_power=0.50, staying_power=7),
        Ship("Strike Destroyer", offensive_power=14, defensive_power=0.40, staying_power=5),
        Ship("Patrol Corvette", offensive_power=10, defensive_power=0.45, staying_power=4)
    ]
    scenarios.append(("Bravo Fleet Superiority", force_a6, force_b6))

    # Scenario 7: Alpha Defense Dominates
    force_a7 = [
        Ship("Shield Cruiser Alpha", offensive_power=11, defensive_power=0.75, staying_power=6),
        Ship("Shield Cruiser Beta", offensive_power=11, defensive_power=0.75, staying_power=6)
    ]
    force_b7 = [
        Ship("Assault Ship Gamma", offensive_power=20, defensive_power=0.20, staying_power=3),
        Ship("Assault Ship Delta", offensive_power=20, defensive_power=0.20, staying_power=3)
    ]
    scenarios.append(("Alpha Defense Dominates", force_a7, force_b7))

    # Scenario 8: Bravo Slight Edge
    force_a8 = [
        Ship("Veteran Destroyer", offensive_power=10, defensive_power=0.30, staying_power=3),
        Ship("Veteran Frigate", offensive_power=7, defensive_power=0.25, staying_power=2)
    ]
    force_b8 = [
        Ship("Elite Corvette 1", offensive_power=15, defensive_power=0.55, staying_power=5),
        Ship("Elite Corvette 2", offensive_power=15, defensive_power=0.55, staying_power=5)
    ]
    scenarios.append(("Bravo Slight Edge", force_a8, force_b8))

    return scenarios


def calculate_force_stats(force):
    """Calculate aggregate statistics for a force."""
    total_offensive = sum(ship.offensive_power for ship in force)
    avg_defensive = sum(ship.defensive_power for ship in force) / len(force) if force else 0
    total_staying = sum(ship.staying_power for ship in force)
    return total_offensive, avg_defensive, total_staying


def run_extended_salvo_examples():
    """Run all salvo combat scenarios and display results as tables."""

    print("="*100)
    print(" " * 30 + "EXTENDED SALVO COMBAT MODEL EXAMPLES")
    print("="*100)
    print()
    print("These scenarios demonstrate how ship characteristics interact in salvo combat:")
    print("  • Offensive Power (OP): Number of missiles launched")
    print("  • Defensive Power (DP): Probability of intercepting incoming missiles (0.0-1.0)")
    print("  • Staying Power (SP): Hull points before destruction")
    print()

    scenarios = create_battle_scenarios()
    results = []

    # Run all scenarios quietly (using varied seeds to avoid bias)
    seeds = [42, 137, 88, 201, 56, 333, 99, 175]  # Different seeds for variety
    for i, (title, force_a, force_b) in enumerate(scenarios, 1):
        battle = SalvoCombatModel(force_a, force_b, random_seed=seeds[i-1])
        outcome = battle.run_simulation(quiet=True)
        stats = battle.get_battle_statistics()

        a_off, a_def, a_stay = calculate_force_stats(force_a)
        b_off, b_def, b_stay = calculate_force_stats(force_b)

        results.append({
            'title': title,
            'force_a': force_a,
            'force_b': force_b,
            'outcome': outcome,
            'stats': stats,
            'a_off': a_off,
            'a_def': a_def,
            'a_stay': a_stay,
            'b_off': b_off,
            'b_def': b_def,
            'b_stay': b_stay
        })

    # Print results table
    print_results_table(results)
    print()
    print_ascii_comparison_chart(results)
    print()
    print_tactical_analysis(results)
    print()

    # Create matplotlib visualizations
    plot_battle_analysis(results)

    return results


def print_results_table(results):
    """Print results in a formatted table."""

    print("="*100)
    print(" " * 40 + "BATTLE RESULTS")
    print("="*100)
    print(f"{'#':<3} {'Scenario':<30} {'Winner':<12} {'Rds':<4} {'A Ships':<8} {'B Ships':<8} {'A Pwr':<6} {'B Pwr':<6}")
    print("-"*100)

    for i, r in enumerate(results, 1):
        winner = "Alpha" if "Force A" in r['outcome'] else ("Bravo" if "Force B" in r['outcome'] else "Draw")
        a_ships_str = f"{len(r['force_a'])}→{r['stats']['force_a_survivors']}"
        b_ships_str = f"{len(r['force_b'])}→{r['stats']['force_b_survivors']}"

        print(f"{i:<3} {r['title']:<30} {winner:<12} {r['stats']['rounds']:<4} "
              f"{a_ships_str:<8} {b_ships_str:<8} {r['a_off']:<6.0f} {r['b_off']:<6.0f}")

    print("="*100)


def print_ascii_comparison_chart(results):
    """Print ASCII visualization comparing offense vs defense for each scenario."""

    print("="*100)
    print(" " * 32 + "FORCE COMPOSITION VISUALIZATION")
    print("="*100)
    print()
    print("Chart shows relative Offensive Power (OP) and Defensive Power (DP) for each force")
    print("Offense: ═══  Defense: ▓▓▓")
    print()

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")

        # Normalize offensive power for visualization (max 50 chars)
        max_off = max(r['a_off'], r['b_off'])
        if max_off > 0:
            a_off_bars = int((r['a_off'] / max_off) * 50)
            b_off_bars = int((r['b_off'] / max_off) * 50)
        else:
            a_off_bars = b_off_bars = 0

        # Defensive power is 0-1, so scale to 50 chars
        a_def_bars = int(r['a_def'] * 50)
        b_def_bars = int(r['b_def'] * 50)

        # Print Force A
        winner_a = " ✓" if "Force A" in r['outcome'] else ""
        print(f"   Alpha{winner_a:3}  OP:{'═'*a_off_bars:<50} {r['a_off']:.0f}")
        print(f"           DP:{'▓'*a_def_bars:<50} {r['a_def']:.2f}")

        # Print Force B
        winner_b = " ✓" if "Force B" in r['outcome'] else ""
        print(f"   Bravo{winner_b:3}  OP:{'═'*b_off_bars:<50} {r['b_off']:.0f}")
        print(f"           DP:{'▓'*b_def_bars:<50} {r['b_def']:.2f}")
        print()

    print("="*100)


def plot_battle_analysis(results):
    """Create matplotlib visualizations of battle outcomes and force compositions."""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Extended Salvo Combat Analysis', fontsize=16, fontweight='bold')

    # Create grid layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Offensive vs Defensive Power Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    plot_offense_defense_scatter(ax1, results)

    # Plot 2: Battle Outcomes (Winners)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_battle_outcomes(ax2, results)

    # Plot 3: Force Composition Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    plot_force_composition(ax3, results)

    # Plot 4: Casualties and Survival Rates
    ax4 = fig.add_subplot(gs[1, 1])
    plot_survival_rates(ax4, results)

    print("Displaying matplotlib visualizations...")
    plt.tight_layout()
    plt.show()


def plot_offense_defense_scatter(ax, results):
    """Scatter plot showing offense vs defense trade-offs."""

    # Collect data points
    a_offense = [r['a_off'] for r in results]
    a_defense = [r['a_def'] for r in results]
    b_offense = [r['b_off'] for r in results]
    b_defense = [r['b_def'] for r in results]

    # Plot Force A (winners marked differently)
    for i, r in enumerate(results):
        marker = 'o' if 'Force A' in r['outcome'] else 'x'
        color = 'blue' if 'Force A' in r['outcome'] else 'lightblue'
        ax.scatter(a_offense[i], a_defense[i], marker=marker, s=150,
                  color=color, edgecolor='darkblue', linewidth=2, label='Alpha' if i == 0 else '', zorder=3)

    # Plot Force B
    for i, r in enumerate(results):
        marker = 'o' if 'Force B' in r['outcome'] else 'x'
        color = 'red' if 'Force B' in r['outcome'] else 'lightcoral'
        ax.scatter(b_offense[i], b_defense[i], marker=marker, s=150,
                  color=color, edgecolor='darkred', linewidth=2, label='Bravo' if i == 0 else '', zorder=3)

    ax.set_xlabel('Offensive Power', fontsize=11, fontweight='bold')
    ax.set_ylabel('Defensive Power', fontsize=11, fontweight='bold')
    ax.set_title('Offense vs Defense Trade-offs\n(○ = Winner, × = Loser)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(['Alpha wins', 'Alpha loses', 'Bravo wins', 'Bravo loses'], loc='upper right', fontsize=9)
    ax.set_xlim(0, max(max(a_offense), max(b_offense)) * 1.1)
    ax.set_ylim(0, 1.0)


def plot_battle_outcomes(ax, results):
    """Bar chart showing battle outcomes by scenario."""

    scenarios = [f"{i+1}" for i in range(len(results))]
    a_wins = [1 if 'Force A' in r['outcome'] else 0 for r in results]
    b_wins = [1 if 'Force B' in r['outcome'] else 0 for r in results]

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, a_wins, width, label='Alpha Victory', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, b_wins, width, label='Bravo Victory', color='indianred', edgecolor='black')

    ax.set_xlabel('Scenario Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Victory (1 = Win, 0 = Loss)', fontsize=11, fontweight='bold')
    ax.set_title('Battle Outcomes by Scenario', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def plot_force_composition(ax, results):
    """Grouped bar chart comparing offensive power between forces."""

    scenarios = [f"S{i+1}" for i in range(len(results))]
    a_power = [r['a_off'] for r in results]
    b_power = [r['b_off'] for r in results]

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, a_power, width, label='Alpha Offense', color='cornflowerblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, b_power, width, label='Bravo Offense', color='salmon', edgecolor='black')

    # Mark winners with a star
    for i, r in enumerate(results):
        if 'Force A' in r['outcome']:
            ax.text(i - width/2, a_power[i] + 1, '★', ha='center', fontsize=16, color='gold')
        elif 'Force B' in r['outcome']:
            ax.text(i + width/2, b_power[i] + 1, '★', ha='center', fontsize=16, color='gold')

    ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Offensive Power', fontsize=11, fontweight='bold')
    ax.set_title('Offensive Power Comparison\n(★ = Winner)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def plot_survival_rates(ax, results):
    """Stacked bar chart showing survival and casualty rates."""

    scenarios = [f"S{i+1}" for i in range(len(results))]

    # Calculate casualties and survivors for each side
    data = []
    for r in results:
        a_initial = len(r['force_a'])
        b_initial = len(r['force_b'])
        a_survivors = r['stats']['force_a_survivors']
        b_survivors = r['stats']['force_b_survivors']
        a_casualties = a_initial - a_survivors
        b_casualties = b_initial - b_survivors

        data.append({
            'a_surv': a_survivors,
            'a_cas': a_casualties,
            'b_surv': b_survivors,
            'b_cas': b_casualties,
            'total': a_initial + b_initial
        })

    # Create stacked bar chart showing composition
    x = np.arange(len(scenarios))
    width = 0.4

    # Calculate percentages
    a_surv_pct = [d['a_surv'] / d['total'] * 100 for d in data]
    a_cas_pct = [d['a_cas'] / d['total'] * 100 for d in data]
    b_surv_pct = [d['b_surv'] / d['total'] * 100 for d in data]
    b_cas_pct = [d['b_cas'] / d['total'] * 100 for d in data]

    # Alpha side (left bar)
    p1 = ax.bar(x - width/2, a_surv_pct, width, label='Alpha Survivors', color='lightblue', edgecolor='black')
    p2 = ax.bar(x - width/2, a_cas_pct, width, bottom=a_surv_pct, label='Alpha Casualties',
                color='darkblue', edgecolor='black', alpha=0.7)

    # Bravo side (right bar)
    p3 = ax.bar(x + width/2, b_surv_pct, width, label='Bravo Survivors', color='lightcoral', edgecolor='black')
    p4 = ax.bar(x + width/2, b_cas_pct, width, bottom=b_surv_pct, label='Bravo Casualties',
                color='darkred', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax.set_ylabel('Force Composition (%)', fontsize=11, fontweight='bold')
    ax.set_title('Casualties vs Survivors\n(% of Total Ships)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def print_tactical_analysis(results):
    """Print tactical insights from the scenarios."""

    print("="*100)
    print(" " * 38 + "TACTICAL INSIGHTS")
    print("="*100)
    print()

    # Scenario-specific explanations
    explanations = [
        ("Close Match - Alpha Advantage",
         "Slightly better defense (0.38 vs 0.28 avg) gives Alpha the edge despite similar firepower."),

        ("Quantity Overwhelms Quality",
         "Three cruisers (40 combined firepower) overwhelm a single battleship (18 offense). Multiple ships\n"
         "     can focus fire and distribute damage, negating the battleship's 0.45 defense."),

        ("Swarm Overwhelms Elite",
         "Five swarm units (55 total firepower) defeat two elite ships (24 firepower).\n"
         "     Concentrated salvo attacks overwhelm even 0.45 defensive systems through sheer volume."),

        ("Defense Beats Offense",
         "High defensive power (0.65) intercepts 65% of glass cannon missiles (0.15 DP only intercepts 15%).\n"
         "     Defensive frigates survive initial salvos and win through sustained counter-battery fire."),

        ("Alpha Quality Wins",
         "Superior single-ship stats (28 offense, 0.55 defense, 8 hull) defeat quantity.\n"
         "     Battlecruiser can absorb damage while systematically eliminating attackers."),

        ("Bravo Fleet Superiority",
         "Bravo has advantage in all categories: firepower (41 vs 24), defense (0.45 vs 0.32),\n"
         "     and total hull strength (16 vs 7). Comprehensive superiority ensures victory."),

        ("Alpha Defense Dominates",
         "Extreme defensive power (0.75) intercepts 75% of incoming missiles. Despite facing\n"
         "     nearly twice the firepower (40 vs 22), Alpha's shields prove nearly impenetrable."),

        ("Bravo Slight Edge",
         "Bravo has significantly better offense (30 vs 17) and defense (0.55 vs 0.28).\n"
         "     Superior stats in all categories produce a decisive victory despite only 2v2 ships.")
    ]

    for i, (title, explanation) in enumerate(explanations, 1):
        print(f"{i}. {title}")
        print(f"   {explanation}")
        print()

    # Overall statistics
    a_wins = sum(1 for r in results if 'Force A' in r['outcome'])
    b_wins = sum(1 for r in results if 'Force B' in r['outcome'])
    avg_rounds = sum(r['stats']['rounds'] for r in results) / len(results)

    print("-"*100)
    print(f"Overall Statistics:")
    print(f"  Force Alpha victories: {a_wins}/{len(results)}")
    print(f"  Force Bravo victories: {b_wins}/{len(results)}")
    print(f"  Average battle duration: {avg_rounds:.1f} rounds")
    print("="*100)


if __name__ == "__main__":
    results = run_extended_salvo_examples()