#!/usr/bin/env python3
"""
Extended Lanchester Models Examples

This module demonstrates the Linear and Square Law models through diverse scenarios.
The examples illustrate how force size, effectiveness coefficients, and combat type
(aimed vs unaimed fire) produce different outcomes.

Key Insights:
- Linear Law: Attrition warfare where size matters (guerrilla, melee combat)
- Square Law: Concentration of force is decisive (modern ranged combat)
- Same forces can have vastly different outcomes under different combat models
- Effectiveness coefficients can overcome numerical disadvantages
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import LanchesterLinear, LanchesterSquare
import matplotlib.pyplot as plt
import numpy as np


def create_battle_scenarios():
    """Create 8 diverse Lanchester battle scenarios."""

    scenarios = []

    # Scenario 1: Equal Numbers, Equal Effectiveness - Draw
    scenarios.append(("Perfect Balance - Draw", 100, 100, 0.01, 0.01, 0.01, 0.01))

    # Scenario 2: Slight Numerical Advantage
    scenarios.append(("Slight Numbers Advantage", 110, 100, 0.01, 0.01, 0.01, 0.01))

    # Scenario 3: Quality vs Quantity (Linear - Quantity Wins)
    scenarios.append(("Linear: Quantity Beats Quality", 80, 120, 0.015, 0.010, 0.010, 0.010))

    # Scenario 4: Quality vs Quantity (Square - Quality Wins)
    scenarios.append(("Square: Quality Beats Quantity", 80, 120, 0.015, 0.010, 0.015, 0.010))

    # Scenario 5: Overwhelming Force (Both Models)
    scenarios.append(("Overwhelming Numerical Superiority", 150, 50, 0.01, 0.01, 0.01, 0.01))

    # Scenario 6: Superior Effectiveness (Linear)
    scenarios.append(("Linear: Better Effectiveness", 90, 100, 0.020, 0.010, 0.010, 0.010))

    # Scenario 7: Superior Effectiveness (Square)
    scenarios.append(("Square: Better Effectiveness", 90, 100, 0.010, 0.010, 0.020, 0.010))

    # Scenario 8: Close Match with Different Models
    scenarios.append(("Close Match - Model Comparison", 100, 90, 0.012, 0.010, 0.012, 0.010))

    return scenarios


def run_extended_lanchester_examples():
    """Run all Lanchester scenarios and display results as tables."""

    print("="*110)
    print(" " * 35 + "EXTENDED LANCHESTER MODELS EXAMPLES")
    print("="*110)
    print()
    print("These scenarios demonstrate how force size and effectiveness interact in different combat models:")
    print("  • Linear Law: Suitable for unaimed fire, guerrilla warfare, ancient combat")
    print("  • Square Law: Suitable for aimed fire, modern combat, concentration effects")
    print("  • Parameters: A0, B0 (initial forces), α, β (effectiveness coefficients)")
    print()

    scenarios = create_battle_scenarios()
    results = []

    # Run all scenarios
    for title, A0, B0, alpha_lin, beta_lin, alpha_sq, beta_sq in scenarios:
        # Run Linear Law
        linear = LanchesterLinear(A0, B0, alpha_lin, beta_lin)
        linear_sol = linear.simple_analytical_solution()

        # Run Square Law
        square = LanchesterSquare(A0, B0, alpha_sq, beta_sq)
        square_sol = square.simple_analytical_solution()

        results.append({
            'title': title,
            'A0': A0,
            'B0': B0,
            'alpha_lin': alpha_lin,
            'beta_lin': beta_lin,
            'alpha_sq': alpha_sq,
            'beta_sq': beta_sq,
            'linear': linear,
            'linear_sol': linear_sol,
            'square': square,
            'square_sol': square_sol
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

    print("="*110)
    print(" " * 45 + "BATTLE RESULTS")
    print("="*110)
    print(f"{'#':<3} {'Scenario':<35} {'Forces':<10} {'Linear':<20} {'Square':<20}")
    print("-"*110)

    for i, r in enumerate(results, 1):
        forces = f"{r['A0']}v{r['B0']}"

        lin_winner = r['linear_sol']['winner']
        lin_surv = r['linear_sol']['remaining_strength']
        lin_str = f"{lin_winner} ({lin_surv:.0f} left)"

        sq_winner = r['square_sol']['winner']
        sq_surv = r['square_sol']['remaining_strength']
        sq_str = f"{sq_winner} ({sq_surv:.0f} left)"

        print(f"{i:<3} {r['title']:<35} {forces:<10} {lin_str:<20} {sq_str:<20}")

    print("="*110)


def print_ascii_comparison_chart(results):
    """Print ASCII visualization comparing Linear vs Square Law outcomes."""

    print("="*110)
    print(" " * 38 + "LINEAR VS SQUARE LAW COMPARISON")
    print("="*110)
    print()
    print("Chart shows survivors for each model. Bar length = number of survivors")
    print("Linear: ═══  Square: ▓▓▓")
    print()

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")

        # Get survivors
        lin_winner = r['linear_sol']['winner']
        lin_surv = int(r['linear_sol']['remaining_strength'])
        sq_winner = r['square_sol']['winner']
        sq_surv = int(r['square_sol']['remaining_strength'])

        # Normalize for visualization (max 60 chars)
        max_surv = max(lin_surv, sq_surv, 1)
        lin_bars = int((lin_surv / max_surv) * 60)
        sq_bars = int((sq_surv / max_surv) * 60)

        # Print Linear result
        lin_marker = " ✓" if lin_winner == 'A' else (" ✗" if lin_winner == 'B' else "")
        print(f"   Linear {lin_winner:>4}{lin_marker:2}  {'═'*lin_bars:<60} {lin_surv}")

        # Print Square result
        sq_marker = " ✓" if sq_winner == 'A' else (" ✗" if sq_winner == 'B' else "")
        print(f"   Square {sq_winner:>4}{sq_marker:2}  {'▓'*sq_bars:<60} {sq_surv}")
        print()

    print("="*110)


def print_tactical_analysis(results):
    """Print tactical insights from the scenarios."""

    print("="*110)
    print(" " * 43 + "TACTICAL INSIGHTS")
    print("="*110)
    print()

    # Scenario-specific explanations
    explanations = [
        ("Perfect Balance - Draw",
         "Equal forces with equal effectiveness result in mutual annihilation in both models.\n"
         "     This represents the baseline case where no side has any advantage."),

        ("Slight Numbers Advantage",
         "A 10% numerical advantage (110 vs 100) produces identical outcomes in both models\n"
         "     when effectiveness is equal. Linear: 10 survivors, Square: ~46 survivors."),

        ("Linear: Quantity Beats Quality",
         "In Linear Law, having more troops (120 vs 80) overwhelms better effectiveness (α=0.015).\n"
         "     The invariant αA₀ - βB₀ = 1.2 - 1.2 = 0 results in draw despite quality edge."),

        ("Square: Quality Beats Quantity",
         "In Square Law, better effectiveness (α=0.015 vs β=0.010) overcomes numerical disadvantage.\n"
         "     The invariant α²A₀² - β²B₀² matters more: quality multiplies force effectiveness."),

        ("Overwhelming Numerical Superiority",
         "3:1 numerical advantage (150 vs 50) guarantees victory in both models.\n"
         "     Linear: 100 survivors (difference). Square: ~141 survivors (√(150²-50²))."),

        ("Linear: Better Effectiveness",
         "In Linear Law, 2× effectiveness (α=0.020) nearly offsets 10% numerical disadvantage.\n"
         "     Combat power = α×A₀ = 1.8 vs β×B₀ = 1.0, so A wins decisively."),

        ("Square: Better Effectiveness",
         "In Square Law, 2× effectiveness (α=0.020) creates massive advantage through squaring.\n"
         "     Effective power = α×A₀² vs β×B₀²: quality advantage compounds geometrically."),

        ("Close Match - Model Comparison",
         "With 100 vs 90 and slight quality edge (α=0.012 vs β=0.010), both models predict\n"
         "     A victory but Square Law shows higher survivors due to concentration effects.")
    ]

    for i, (title, explanation) in enumerate(explanations, 1):
        print(f"{i}. {title}")
        print(f"   {explanation}")
        print()

    # Overall statistics
    linear_a_wins = sum(1 for r in results if r['linear_sol']['winner'] == 'A')
    square_a_wins = sum(1 for r in results if r['square_sol']['winner'] == 'A')

    print("-"*110)
    print(f"Overall Statistics:")
    print(f"  Linear Law - Force A victories: {linear_a_wins}/{len(results)}")
    print(f"  Square Law - Force A victories: {square_a_wins}/{len(results)}")
    print(f"  Key Difference: Square Law amplifies advantages through concentration effects")
    print("="*110)


def plot_battle_analysis(results):
    """Create matplotlib visualizations of Lanchester battle outcomes."""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Extended Lanchester Models Analysis', fontsize=16, fontweight='bold')

    # Create grid layout: 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Survivors Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_survivors_comparison(ax1, results)

    # Plot 2: Model Difference (Square - Linear)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_model_differences(ax2, results)

    # Plot 3: Force Ratios and Outcomes
    ax3 = fig.add_subplot(gs[1, 0])
    plot_force_ratios(ax3, results)

    # Plot 4: Sample Trajectory Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    plot_sample_trajectories(ax4, results)

    print("Displaying matplotlib visualizations...")
    plt.tight_layout()
    plt.show()


def plot_survivors_comparison(ax, results):
    """Bar chart comparing survivors between Linear and Square Law."""

    scenarios = [f"S{i+1}" for i in range(len(results))]
    x = np.arange(len(scenarios))
    width = 0.35

    linear_surv = [r['linear_sol']['remaining_strength'] for r in results]
    square_surv = [r['square_sol']['remaining_strength'] for r in results]

    bars1 = ax.bar(x - width/2, linear_surv, width, label='Linear Law',
                   color='steelblue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, square_surv, width, label='Square Law',
                   color='coral', edgecolor='black', linewidth=1)

    ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax.set_ylabel('Surviving Forces', fontsize=11, fontweight='bold')
    ax.set_title('Winner Survivors: Linear vs Square Law', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def plot_model_differences(ax, results):
    """Plot the difference in survivors between Square and Linear Law."""

    scenarios = [f"S{i+1}" for i in range(len(results))]
    x = np.arange(len(scenarios))

    differences = []
    colors = []

    for r in results:
        lin_surv = r['linear_sol']['remaining_strength']
        sq_surv = r['square_sol']['remaining_strength']
        diff = sq_surv - lin_surv
        differences.append(diff)
        colors.append('green' if diff > 0 else 'red')

    bars = ax.bar(x, differences, color=colors, edgecolor='black', linewidth=1, alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax.set_ylabel('Survivor Difference (Square - Linear)', fontsize=11, fontweight='bold')
    ax.set_title('Square Law Advantage\n(Positive = More Survivors)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')


def plot_force_ratios(ax, results):
    """Scatter plot showing initial force ratio vs outcome."""

    force_ratios = [r['A0'] / r['B0'] for r in results]
    linear_ratios = []
    square_ratios = []

    for r in results:
        lin_sol = r['linear_sol']
        sq_sol = r['square_sol']

        # Calculate outcome ratio (positive = A wins, negative = B wins)
        if lin_sol['winner'] == 'A':
            lin_ratio = lin_sol['remaining_strength'] / r['A0']
        elif lin_sol['winner'] == 'B':
            lin_ratio = -lin_sol['remaining_strength'] / r['B0']
        else:
            lin_ratio = 0

        if sq_sol['winner'] == 'A':
            sq_ratio = sq_sol['remaining_strength'] / r['A0']
        elif sq_sol['winner'] == 'B':
            sq_ratio = -sq_sol['remaining_strength'] / r['B0']
        else:
            sq_ratio = 0

        linear_ratios.append(lin_ratio)
        square_ratios.append(sq_ratio)

    ax.scatter(force_ratios, linear_ratios, s=120, marker='o', color='steelblue',
               edgecolor='darkblue', linewidth=2, label='Linear Law', zorder=3)
    ax.scatter(force_ratios, square_ratios, s=120, marker='s', color='coral',
               edgecolor='darkred', linewidth=2, label='Square Law', zorder=3)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Initial Force Ratio (A/B)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Outcome Ratio\n(+A wins, -B wins)', fontsize=11, fontweight='bold')
    ax.set_title('Force Ratio vs Battle Outcome', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_sample_trajectories(ax, results):
    """Plot trajectories for one interesting scenario with normalized time."""

    # Choose scenario 4 (Quality vs Quantity - different outcomes)
    r = results[3]

    linear_sol = r['linear_sol']
    square_sol = r['square_sol']

    # Normalize time to battle completion percentage (0-100%)
    lin_t_end = linear_sol['battle_end_time']
    sq_t_end = square_sol['battle_end_time']

    # Handle infinite battle times (draws)
    if np.isinf(lin_t_end):
        lin_time_norm = linear_sol['time'] / np.max(linear_sol['time']) * 100
    else:
        lin_time_norm = (linear_sol['time'] / lin_t_end) * 100

    if np.isinf(sq_t_end):
        sq_time_norm = square_sol['time'] / np.max(square_sol['time']) * 100
    else:
        sq_time_norm = (square_sol['time'] / sq_t_end) * 100

    # Plot Linear Law trajectories
    ax.plot(lin_time_norm, linear_sol['A'], 'b-', linewidth=2.5,
            label=f"Linear: Force A", alpha=0.8)
    ax.plot(lin_time_norm, linear_sol['B'], 'b--', linewidth=2.5,
            label=f"Linear: Force B", alpha=0.8)

    # Plot Square Law trajectories
    ax.plot(sq_time_norm, square_sol['A'], 'r-', linewidth=2.5,
            label=f"Square: Force A", alpha=0.8)
    ax.plot(sq_time_norm, square_sol['B'], 'r--', linewidth=2.5,
            label=f"Square: Force B", alpha=0.8)

    ax.set_xlabel('Battle Progress (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Force Strength', fontsize=11, fontweight='bold')
    ax.set_title(f'Sample Trajectories: {r["title"]}\n'
                 f'(A₀={r["A0"]}, B₀={r["B0"]}) - Normalized Time',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

    # Add annotation about actual battle times
    time_text = f"Linear: {lin_t_end:.0f} time units\nSquare: {sq_t_end:.0f} time units"
    ax.text(0.02, 0.02, time_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


if __name__ == "__main__":
    results = run_extended_lanchester_examples()
