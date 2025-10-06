#!/usr/bin/env python3
"""
Extended Multi-Phase Combat Examples

This module demonstrates how to model combat in discrete phases/turns using
Lanchester models. Phases can represent:
- Daily battles with nightly reinforcements
- Changing combat conditions (weather, terrain, fatigue)
- Strategic reserve deployment
- Attrition campaigns over extended periods

Key Concepts:
- Each phase runs for a fixed duration or until completion
- Forces carry over between phases
- Reinforcements can arrive between phases
- Effectiveness coefficients can change per phase
- Both Linear and Square Law models supported
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import LanchesterLinear, LanchesterSquare
import matplotlib.pyplot as plt
import numpy as np


def run_phase_simulation(initial_A, initial_B, phases_config, model_type='square'):
    """
    Run a multi-phase Lanchester simulation.

    Parameters:
    initial_A: Initial strength of Force A
    initial_B: Initial strength of Force B
    phases_config: List of phase dictionaries with keys:
        - duration: Time length of phase (None = until winner determined)
        - alpha: Effectiveness of A against B
        - beta: Effectiveness of B against A
        - A_reinforcements: Forces added to A at end of phase (default 0)
        - B_reinforcements: Forces added to B at end of phase (default 0)
        - description: Text description of phase
    model_type: 'linear' or 'square'

    Returns:
    List of phase result dictionaries
    """
    current_A = initial_A
    current_B = initial_B

    phase_results = []
    cumulative_time = 0

    print(f"\n{'='*90}")
    print(f"MULTI-PHASE COMBAT SIMULATION - {model_type.upper()} LAW")
    print(f"{'='*90}")
    print(f"Initial Forces: A = {initial_A:.0f}, B = {initial_B:.0f}\n")

    for phase_num, phase in enumerate(phases_config, 1):
        print(f"{'─'*90}")
        print(f"PHASE {phase_num}: {phase.get('description', 'Combat Phase')}")
        print(f"{'─'*90}")
        print(f"Starting forces: A = {current_A:.1f}, B = {current_B:.1f}")
        print(f"Effectiveness: α = {phase['alpha']}, β = {phase['beta']}")

        # Create appropriate model
        if model_type == 'square':
            battle = LanchesterSquare(current_A, current_B, phase['alpha'], phase['beta'])
        else:
            battle = LanchesterLinear(current_A, current_B, phase['alpha'], phase['beta'])

        # Determine phase duration
        duration = phase.get('duration')
        if duration is None:
            # Run until battle concludes
            solution = battle.analytical_solution()
            actual_duration = solution['battle_end_time']
            if np.isinf(actual_duration):
                # Draw - use default time window
                actual_duration = 100
        else:
            solution = battle.analytical_solution(t_max=duration)
            actual_duration = duration

        # Get forces at end of phase duration
        if np.isinf(actual_duration):
            end_idx = -1
        else:
            # Find the index closest to phase end
            time_array = solution['time']
            end_idx = np.argmin(np.abs(time_array - actual_duration))

        end_A = solution['A'][end_idx]
        end_B = solution['B'][end_idx]

        print(f"Phase duration: {actual_duration:.1f} time units")
        print(f"Combat results: A = {end_A:.1f}, B = {end_B:.1f}")

        # Calculate casualties
        casualties_A = current_A - end_A
        casualties_B = current_B - end_B
        print(f"Casualties: A lost {casualties_A:.1f}, B lost {casualties_B:.1f}")

        # Add reinforcements
        reinforcements_A = phase.get('A_reinforcements', 0)
        reinforcements_B = phase.get('B_reinforcements', 0)

        if reinforcements_A > 0:
            print(f"→ Force A receives {reinforcements_A} reinforcements")
            end_A += reinforcements_A

        if reinforcements_B > 0:
            print(f"→ Force B receives {reinforcements_B} reinforcements")
            end_B += reinforcements_B

        print(f"Phase end forces: A = {end_A:.1f}, B = {end_B:.1f}")

        # Store phase results
        phase_results.append({
            'phase': phase_num,
            'description': phase.get('description', f'Phase {phase_num}'),
            'start_A': current_A,
            'start_B': current_B,
            'end_A': max(0, end_A),
            'end_B': max(0, end_B),
            'casualties_A': casualties_A,
            'casualties_B': casualties_B,
            'reinforcements_A': reinforcements_A,
            'reinforcements_B': reinforcements_B,
            'duration': actual_duration,
            'cumulative_time_start': cumulative_time,
            'cumulative_time_end': cumulative_time + actual_duration,
            'solution': solution,
            'alpha': phase['alpha'],
            'beta': phase['beta']
        })

        cumulative_time += actual_duration

        # Update forces for next phase
        current_A = max(0, end_A)
        current_B = max(0, end_B)

        # Check if battle is over
        if current_A == 0 or current_B == 0:
            winner = 'A' if current_A > 0 else 'B'
            survivors = current_A if current_A > 0 else current_B
            print(f"\n{'='*90}")
            print(f"BATTLE CONCLUDED IN PHASE {phase_num}")
            print(f"Winner: Force {winner} with {survivors:.1f} survivors")
            print(f"Total battle duration: {cumulative_time:.1f} time units")
            print(f"{'='*90}")
            break
    else:
        # Battle not concluded
        print(f"\n{'='*90}")
        print(f"BATTLE ONGOING AFTER {len(phases_config)} PHASES")
        print(f"Current forces: A = {current_A:.1f}, B = {current_B:.1f}")
        print(f"Total time elapsed: {cumulative_time:.1f} time units")
        print(f"{'='*90}")

    return phase_results


def create_example_scenarios():
    """Create diverse multi-phase combat scenarios."""

    scenarios = []

    # Scenario 1: Daily Battles with Nightly Reinforcements
    scenarios.append({
        'name': 'Daily Battles with Reinforcements',
        'description': 'Three-day battle where both sides receive reinforcements each night',
        'initial_A': 100,
        'initial_B': 100,
        'model': 'square',
        'phases': [
            {'duration': 50, 'alpha': 0.01, 'beta': 0.01, 'A_reinforcements': 20, 'B_reinforcements': 15,
             'description': 'Day 1 - Initial Engagement'},
            {'duration': 50, 'alpha': 0.01, 'beta': 0.01, 'A_reinforcements': 25, 'B_reinforcements': 30,
             'description': 'Day 2 - Continued Combat'},
            {'duration': 50, 'alpha': 0.01, 'beta': 0.01, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Day 3 - Final Push (No Reinforcements)'}
        ]
    })

    # Scenario 2: Changing Weather Conditions
    scenarios.append({
        'name': 'Weather Impact on Combat',
        'description': 'Effectiveness changes due to weather affecting visibility and mobility',
        'initial_A': 120,
        'initial_B': 100,
        'model': 'square',
        'phases': [
            {'duration': 30, 'alpha': 0.012, 'beta': 0.010, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Clear Weather - A has advantage'},
            {'duration': 30, 'alpha': 0.008, 'beta': 0.008, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Heavy Rain - Reduced effectiveness both sides'},
            {'duration': 40, 'alpha': 0.010, 'beta': 0.012, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Fog - B gains advantage with better equipment'}
        ]
    })

    # Scenario 3: Strategic Reserve Deployment
    scenarios.append({
        'name': 'Strategic Reserve Commitment',
        'description': 'B commits reserves after initial losses, turning the tide',
        'initial_A': 100,
        'initial_B': 80,
        'model': 'linear',
        'phases': [
            {'duration': 60, 'alpha': 0.015, 'beta': 0.010, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Initial Phase - A has quality advantage'},
            {'duration': None, 'alpha': 0.015, 'beta': 0.010, 'A_reinforcements': 0, 'B_reinforcements': 60,
             'description': 'B Commits Strategic Reserve'},
        ]
    })

    # Scenario 4: Attrition Campaign with Fatigue
    scenarios.append({
        'name': 'Extended Attrition with Fatigue',
        'description': 'Long campaign where effectiveness degrades due to fatigue',
        'initial_A': 150,
        'initial_B': 150,
        'model': 'square',
        'phases': [
            {'duration': 40, 'alpha': 0.012, 'beta': 0.012, 'A_reinforcements': 10, 'B_reinforcements': 10,
             'description': 'Week 1 - Fresh forces, equal effectiveness'},
            {'duration': 40, 'alpha': 0.010, 'beta': 0.010, 'A_reinforcements': 8, 'B_reinforcements': 8,
             'description': 'Week 2 - Fatigue setting in'},
            {'duration': 40, 'alpha': 0.008, 'beta': 0.008, 'A_reinforcements': 5, 'B_reinforcements': 5,
             'description': 'Week 3 - Significant fatigue, fewer reinforcements'},
            {'duration': 50, 'alpha': 0.006, 'beta': 0.006, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Week 4 - Exhausted forces, no reinforcements'}
        ]
    })

    # Scenario 5: Phased Withdrawal and Counterattack
    scenarios.append({
        'name': 'Tactical Withdrawal and Counterattack',
        'description': 'A withdraws to defensive position, then counterattacks with reinforcements',
        'initial_A': 90,
        'initial_B': 120,
        'model': 'square',
        'phases': [
            {'duration': 25, 'alpha': 0.008, 'beta': 0.012, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Phase 1 - A Outnumbered, Fighting Withdrawal'},
            {'duration': 25, 'alpha': 0.015, 'beta': 0.008, 'A_reinforcements': 40, 'B_reinforcements': 0,
             'description': 'Phase 2 - A in Defensive Position + Reserves Arrive'},
            {'duration': None, 'alpha': 0.015, 'beta': 0.010, 'A_reinforcements': 0, 'B_reinforcements': 0,
             'description': 'Phase 3 - A Counterattacks'}
        ]
    })

    return scenarios


def plot_phase_results(phase_results, scenario_name, model_type):
    """Create visualizations for multi-phase combat results."""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Multi-Phase Combat: {scenario_name}\n({model_type.upper()} Law)',
                 fontsize=14, fontweight='bold')

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Force Strength Over Phases
    ax1 = fig.add_subplot(gs[0, 0])
    plot_force_progression(ax1, phase_results)

    # Plot 2: Casualties and Reinforcements
    ax2 = fig.add_subplot(gs[0, 1])
    plot_casualties_reinforcements(ax2, phase_results)

    # Plot 3: Continuous Trajectory
    ax3 = fig.add_subplot(gs[1, :])
    plot_continuous_trajectory(ax3, phase_results)

    plt.tight_layout()
    plt.show()


def plot_force_progression(ax, phase_results):
    """Bar chart showing force levels at start and end of each phase."""

    phases = [f"P{r['phase']}" for r in phase_results]
    x = np.arange(len(phases))
    width = 0.35

    start_A = [r['start_A'] for r in phase_results]
    end_A = [r['end_A'] for r in phase_results]
    start_B = [r['start_B'] for r in phase_results]
    end_B = [r['end_B'] for r in phase_results]

    ax.bar(x - width/2 - width, start_A, width, label='A Start', color='lightblue', edgecolor='blue')
    ax.bar(x - width/2, end_A, width, label='A End', color='darkblue', edgecolor='blue')
    ax.bar(x + width/2, start_B, width, label='B Start', color='lightcoral', edgecolor='red')
    ax.bar(x + width/2 + width, end_B, width, label='B End', color='darkred', edgecolor='red')

    ax.set_xlabel('Phase', fontsize=11, fontweight='bold')
    ax.set_ylabel('Force Strength', fontsize=11, fontweight='bold')
    ax.set_title('Force Levels by Phase', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')


def plot_casualties_reinforcements(ax, phase_results):
    """Stacked bar chart showing casualties and reinforcements per phase."""

    phases = [f"P{r['phase']}\n{r['description'][:15]}..." for r in phase_results]
    x = np.arange(len(phases))
    width = 0.35

    casualties_A = [r['casualties_A'] for r in phase_results]
    casualties_B = [r['casualties_B'] for r in phase_results]
    reinforcements_A = [r['reinforcements_A'] for r in phase_results]
    reinforcements_B = [r['reinforcements_B'] for r in phase_results]

    # Plot casualties as negative, reinforcements as positive
    ax.bar(x - width/2, [-c for c in casualties_A], width, label='A Casualties',
           color='lightblue', edgecolor='blue', alpha=0.7)
    ax.bar(x - width/2, reinforcements_A, width, bottom=[-c for c in casualties_A],
           label='A Reinforcements', color='darkblue', edgecolor='blue')

    ax.bar(x + width/2, [-c for c in casualties_B], width, label='B Casualties',
           color='lightcoral', edgecolor='red', alpha=0.7)
    ax.bar(x + width/2, reinforcements_B, width, bottom=[-c for c in casualties_B],
           label='B Reinforcements', color='darkred', edgecolor='red')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Phase', fontsize=11, fontweight='bold')
    ax.set_ylabel('Forces Lost (-) / Gained (+)', fontsize=11, fontweight='bold')
    ax.set_title('Casualties and Reinforcements', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=8)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')


def plot_continuous_trajectory(ax, phase_results):
    """Plot continuous force trajectories across all phases."""

    # Collect trajectory data with time offsets
    all_time_A = []
    all_forces_A = []
    all_time_B = []
    all_forces_B = []

    for r in phase_results:
        time_offset = r['cumulative_time_start']
        solution = r['solution']

        # Adjust times to cumulative scale
        adjusted_time = solution['time'] + time_offset

        # Limit to phase duration
        phase_mask = adjusted_time <= r['cumulative_time_end']

        all_time_A.extend(adjusted_time[phase_mask])
        all_forces_A.extend(solution['A'][phase_mask])
        all_time_B.extend(adjusted_time[phase_mask])
        all_forces_B.extend(solution['B'][phase_mask])

        # Add phase boundary marker
        if r['phase'] < len(phase_results):
            ax.axvline(r['cumulative_time_end'], color='gray', linestyle='--',
                      alpha=0.4, linewidth=1)

    ax.plot(all_time_A, all_forces_A, 'b-', linewidth=2.5, label='Force A', alpha=0.8)
    ax.plot(all_time_B, all_forces_B, 'r-', linewidth=2.5, label='Force B', alpha=0.8)

    # Annotate phases
    for r in phase_results:
        mid_time = (r['cumulative_time_start'] + r['cumulative_time_end']) / 2
        ax.text(mid_time, ax.get_ylim()[1] * 0.95, f"P{r['phase']}",
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlabel('Cumulative Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Force Strength', fontsize=11, fontweight='bold')
    ax.set_title('Continuous Combat Trajectory Across All Phases', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def print_summary_table(all_scenarios_results):
    """Print summary table comparing all scenarios."""

    print("\n" + "="*100)
    print(" " * 35 + "MULTI-PHASE SCENARIOS SUMMARY")
    print("="*100)
    print(f"{'Scenario':<40} {'Phases':<8} {'Winner':<8} {'Survivors':<12} {'Total Time':<12}")
    print("-"*100)

    for scenario_name, results in all_scenarios_results.items():
        num_phases = len(results)
        last_phase = results[-1]

        if last_phase['end_A'] > 0 and last_phase['end_B'] == 0:
            winner = 'A'
            survivors = last_phase['end_A']
        elif last_phase['end_B'] > 0 and last_phase['end_A'] == 0:
            winner = 'B'
            survivors = last_phase['end_B']
        else:
            winner = 'Ongoing'
            survivors = 0

        total_time = last_phase['cumulative_time_end']

        print(f"{scenario_name:<40} {num_phases:<8} {winner:<8} {survivors:<12.1f} {total_time:<12.1f}")

    print("="*100)


if __name__ == "__main__":
    scenarios = create_example_scenarios()
    all_results = {}

    print("="*100)
    print(" " * 30 + "MULTI-PHASE COMBAT EXAMPLES")
    print("="*100)
    print("\nThis module demonstrates combat simulation across multiple phases with:")
    print("  • Reinforcements arriving between phases")
    print("  • Changing effectiveness due to conditions (weather, fatigue, terrain)")
    print("  • Strategic decisions (withdrawal, reserve commitment)")
    print("  • Extended campaigns with cumulative effects")
    print()

    # Run each scenario
    for scenario in scenarios:
        print(f"\n{'#'*100}")
        print(f"# SCENARIO: {scenario['name']}")
        print(f"# {scenario['description']}")
        print(f"{'#'*100}")

        results = run_phase_simulation(
            scenario['initial_A'],
            scenario['initial_B'],
            scenario['phases'],
            scenario['model']
        )

        all_results[scenario['name']] = results

        # Generate plots for this scenario
        plot_phase_results(results, scenario['name'], scenario['model'])

    # Print summary comparison
    print_summary_table(all_results)
