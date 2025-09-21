import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class Ship:
    """Represents a ship in the combat simulation"""
    name: str
    offensive_power: float  # Number of missiles/projectiles per salvo
    defensive_power: float  # Probability of intercepting incoming missiles (0-1)
    staying_power: int      # Number of hits required to sink the ship
    current_hits: int = 0   # Current damage taken
    is_active: bool = True  # Whether ship can still fight
    
    def is_operational(self) -> bool:
        """Check if ship is still operational"""
        return self.current_hits < self.staying_power and self.is_active
    
    def take_damage(self, hits: int) -> int:
        """Apply damage to ship and return actual hits taken"""
        if not self.is_operational():
            return 0
        
        actual_hits = min(hits, self.staying_power - self.current_hits)
        self.current_hits += actual_hits
        
        if self.current_hits >= self.staying_power:
            self.is_active = False
            
        return actual_hits
    
    def get_health_percentage(self) -> float:
        """Get remaining health as percentage"""
        return max(0, (self.staying_power - self.current_hits) / self.staying_power * 100)

class SalvoCombatModel:
    """Salvo Combat Model implementation"""

    # Constants for simulation parameters
    MAX_INTERCEPT_PROBABILITY = 0.95    # Maximum missile interception probability cap. Prevents 100% defense scenarios.
    DEFAULT_MAX_ROUNDS = 50            # Default maximum battle rounds to prevent infinite loops
    MONTE_CARLO_ITERATIONS = 100       # Default Monte Carlo simulation count for statistical analysis
    HEALTH_PRECISION = 1               # Decimal places for health percentage display
    DEFENSIVE_EFFECTIVENESS_FACTOR = 1.0  # Multiplier for defensive calculations (allows tuning)
    FORCE_EFFECTIVENESS_TOLERANCE = 1e-10  # Tolerance for comparing force effectiveness
    HIT_DISTRIBUTION_RANDOMNESS = 0.2   # Factor for randomness in hit distribution (0-1)
    
    def __init__(self, force_a: List[Ship], force_b: List[Ship], random_seed: Optional[int] = None):
        self.force_a = force_a
        self.force_b = force_b
        self.round_number = 0
        self.battle_log = []
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def calculate_operational_forces(self) -> Tuple[List[Ship], List[Ship]]:
        """
        Calculate currently operational forces for both sides.

        Returns:
        tuple: (operational_force_a, operational_force_b)
        """
        operational_a = [ship for ship in self.force_a if ship.is_operational()]
        operational_b = [ship for ship in self.force_b if ship.is_operational()]
        return operational_a, operational_b

    def calculate_force_effectiveness(self, force: List[Ship]) -> dict:
        """
        Calculate comprehensive force effectiveness metrics.

        Parameters:
        force (List[Ship]): List of ships to analyze

        Returns:
        dict: Force effectiveness metrics
        """
        operational_ships = [ship for ship in force if ship.is_operational()]

        if not operational_ships:
            return {
                'total_offensive': 0,
                'total_defensive': 0,
                'total_staying_power': 0,
                'remaining_health': 0,
                'operational_count': 0,
                'average_defensive': 0
            }

        return {
            'total_offensive': sum(ship.offensive_power for ship in operational_ships),
            'total_defensive': sum(ship.defensive_power for ship in operational_ships),
            'total_staying_power': sum(ship.staying_power for ship in operational_ships),
            'remaining_health': sum(ship.get_health_percentage() for ship in operational_ships),
            'operational_count': len(operational_ships),
            'average_defensive': sum(ship.defensive_power for ship in operational_ships) / len(operational_ships)
        }

    def execute_attack_phase(self, attackers: List[Ship], defenders: List[Ship], attacking_force_name: str) -> List[str]:
        """
        Execute attack phase and return event log.

        Parameters:
        attackers (List[Ship]): Attacking ships
        defenders (List[Ship]): Defending ships
        attacking_force_name (str): Name of attacking force for logging

        Returns:
        List[str]: Event log for this attack phase
        """
        events = []

        if not attackers or not defenders:
            return events

        # Calculate salvo effectiveness
        missiles_through, hits_dist = self.calculate_salvo_effectiveness(attackers, defenders)
        total_missiles = sum(ship.offensive_power for ship in attackers)

        events.append(f"{attacking_force_name} fires {total_missiles:.1f} missiles, {missiles_through} penetrate defenses")

        # Apply damage to defenders
        for i, ship in enumerate(defenders):
            if i < len(hits_dist) and hits_dist[i] > 0:
                actual_hits = ship.take_damage(hits_dist[i])
                if actual_hits > 0:
                    events.append(f"{ship.name} takes {actual_hits} hit(s), health: {ship.get_health_percentage():.1f}%")
                    if not ship.is_operational():
                        events.append(f"{ship.name} is destroyed!")

        return events

    def determine_battle_outcome(self) -> str:
        """
        Determine and return battle outcome.

        Returns:
        str: Battle outcome description
        """
        operational_a, operational_b = self.calculate_operational_forces()

        if operational_a and operational_b:
            return "Draw - Both forces have survivors"
        elif operational_a:
            return "Force A Victory"
        elif operational_b:
            return "Force B Victory"
        else:
            return "Mutual Annihilation"

    def get_battle_statistics(self) -> dict:
        """
        Calculate comprehensive battle statistics.

        Returns:
        dict: Battle statistics including effectiveness metrics
        """
        operational_a, operational_b = self.calculate_operational_forces()
        effectiveness_a = self.calculate_force_effectiveness(self.force_a)
        effectiveness_b = self.calculate_force_effectiveness(self.force_b)

        return {
            'rounds': self.round_number,
            'outcome': self.determine_battle_outcome(),
            'force_a_survivors': len(operational_a),
            'force_b_survivors': len(operational_b),
            'force_a_effectiveness': effectiveness_a,
            'force_b_effectiveness': effectiveness_b,
            'total_a_offensive': effectiveness_a['total_offensive'],
            'total_b_offensive': effectiveness_b['total_offensive'],
            'offensive_ratio': effectiveness_a['total_offensive'] / max(effectiveness_b['total_offensive'], 1),
            'surviving_ships_a': operational_a,
            'surviving_ships_b': operational_b
        }

    def simple_simulation(self, max_rounds: int = None, quiet: bool = False) -> dict:
        """
        Simplified simulation for equal defensive capabilities.

        Optimizes for cases where forces have similar defensive characteristics
        by using statistical approximations rather than round-by-round simulation.
        For forces with significantly different defensive capabilities, falls back
        to full simulation.

        Parameters:
        max_rounds (int): Maximum rounds to simulate
        quiet (bool): If True, suppress output

        Returns:
        dict: Simplified battle statistics
        """
        if max_rounds is None:
            max_rounds = self.DEFAULT_MAX_ROUNDS

        # Calculate initial force effectiveness
        effectiveness_a = self.calculate_force_effectiveness(self.force_a)
        effectiveness_b = self.calculate_force_effectiveness(self.force_b)

        # Check if forces have similar defensive characteristics
        avg_def_a = effectiveness_a['average_defensive']
        avg_def_b = effectiveness_b['average_defensive']

        defensive_similarity = abs(avg_def_a - avg_def_b)

        # If defensive capabilities are similar (within tolerance), use simplified approach
        if defensive_similarity < self.FORCE_EFFECTIVENESS_TOLERANCE * 10:  # 10x tolerance for defensive similarity
            if not quiet:
                print("=== SIMPLE SALVO SIMULATION (Equal Defensive Forces) ===\n")

            # Simplified calculation based on offensive power ratio
            total_a_offensive = effectiveness_a['total_offensive']
            total_b_offensive = effectiveness_b['total_offensive']
            total_a_staying = effectiveness_a['total_staying_power']
            total_b_staying = effectiveness_b['total_staying_power']

            # Estimate battle duration based on average attrition
            avg_defensive = (avg_def_a + avg_def_b) / 2
            estimated_rounds = min(max_rounds,
                                 max(total_a_staying / max(total_b_offensive * (1 - avg_defensive), 1),
                                     total_b_staying / max(total_a_offensive * (1 - avg_defensive), 1)))

            # Determine winner based on total effectiveness
            if total_a_offensive > total_b_offensive:
                winner = "Force A Victory"
                # Estimate survivors based on offensive advantage
                advantage_ratio = total_a_offensive / max(total_b_offensive, 1)
                estimated_survivors_a = max(1, int(len(self.force_a) * (advantage_ratio - 1) / advantage_ratio))
                estimated_survivors_b = 0
            elif total_b_offensive > total_a_offensive:
                winner = "Force B Victory"
                advantage_ratio = total_b_offensive / max(total_a_offensive, 1)
                estimated_survivors_a = 0
                estimated_survivors_b = max(1, int(len(self.force_b) * (advantage_ratio - 1) / advantage_ratio))
            else:
                winner = "Mutual Annihilation"
                estimated_survivors_a = 0
                estimated_survivors_b = 0

            if not quiet:
                print(f"Simplified Analysis:")
                print(f"Force A Total Offensive: {total_a_offensive:.1f}")
                print(f"Force B Total Offensive: {total_b_offensive:.1f}")
                print(f"Average Defensive: {avg_defensive:.2f}")
                print(f"Estimated Rounds: {estimated_rounds:.1f}")
                print(f"Winner: {winner}")
                print(f"Estimated Force A survivors: {estimated_survivors_a}")
                print(f"Estimated Force B survivors: {estimated_survivors_b}")

            return {
                'outcome': winner,
                'rounds': int(estimated_rounds),
                'force_a_survivors': estimated_survivors_a,
                'force_b_survivors': estimated_survivors_b,
                'total_a_offensive': total_a_offensive,
                'total_b_offensive': total_b_offensive,
                'offensive_ratio': total_a_offensive / max(total_b_offensive, 1),
                'method': 'simplified',
                'defensive_similarity': defensive_similarity
            }
        else:
            # Forces have different defensive capabilities, use full simulation
            if not quiet:
                print("=== FULL SALVO SIMULATION (Different Defensive Forces) ===\n")
                print(f"Defensive similarity: {defensive_similarity:.3f} (threshold: {self.FORCE_EFFECTIVENESS_TOLERANCE * 10:.3f})")
                print("Using full simulation due to defensive capability differences.\n")

            # Run full simulation but potentially quieter
            original_quiet = not quiet
            result = self.run_simulation(max_rounds) if original_quiet else "Full simulation completed"

            stats = self.get_battle_statistics()
            stats['method'] = 'full_simulation'
            stats['defensive_similarity'] = defensive_similarity
            return stats
    
    def calculate_salvo_effectiveness(self, attacking_force: List[Ship], defending_force: List[Ship]) -> Tuple[int, List[int]]:
        """Calculate the effectiveness of a salvo from attacking force to defending force"""
        # Calculate total offensive power of attacking force
        total_offensive = sum(ship.offensive_power for ship in attacking_force if ship.is_operational())
        
        # Calculate total defensive power of defending force
        operational_defenders = [ship for ship in defending_force if ship.is_operational()]
        if not operational_defenders:
            return 0, []
        
        total_defensive = sum(ship.defensive_power for ship in operational_defenders)
        
        # Calculate missiles getting through defense
        # Using probabilistic model: each missile has chance of being intercepted
        missiles_fired = int(total_offensive)
        missiles_through = 0
        
        for _ in range(missiles_fired):
            # Each missile faces the combined defensive probability
            intercept_prob = min(self.MAX_INTERCEPT_PROBABILITY,
                               total_defensive * self.DEFENSIVE_EFFECTIVENESS_FACTOR / len(operational_defenders))
            if random.random() > intercept_prob:
                missiles_through += 1
        
        # Distribute hits among operational defending ships
        hits_distribution = []
        if missiles_through > 0 and operational_defenders:
            # Simple distribution: hits distributed roughly equally with some randomness
            base_hits_per_ship = missiles_through // len(operational_defenders)
            remaining_hits = missiles_through % len(operational_defenders)
            
            for i, ship in enumerate(operational_defenders):
                hits = base_hits_per_ship
                if i < remaining_hits:
                    hits += 1
                hits_distribution.append(hits)
        else:
            hits_distribution = [0] * len(operational_defenders)
        
        return missiles_through, hits_distribution
    
    def execute_round(self) -> bool:
        """Execute one round of combat. Returns True if battle continues, False if over"""
        self.round_number += 1

        # Calculate operational forces using helper method
        a_operational, b_operational = self.calculate_operational_forces()

        # Initialize round log
        round_log = {
            'round': self.round_number,
            'force_a_active': len(a_operational),
            'force_b_active': len(b_operational),
            'events': []
        }

        # Check if battle is already over
        if not a_operational or not b_operational:
            return False

        # Execute Force A attack phase using helper method
        attack_events = self.execute_attack_phase(a_operational, b_operational, "Force A")
        round_log['events'].extend(attack_events)

        # Execute Force B attack phase (if any ships remain) using helper method
        a_operational, b_operational = self.calculate_operational_forces()  # Recalculate after A's attack
        if b_operational:
            attack_events = self.execute_attack_phase(b_operational, a_operational, "Force B")
            round_log['events'].extend(attack_events)

        self.battle_log.append(round_log)

        # Check if battle continues using helper method
        a_operational, b_operational = self.calculate_operational_forces()
        return len(a_operational) > 0 and len(b_operational) > 0
    
    def run_simulation(self, max_rounds: int = None) -> str:
        """
        Run the complete simulation with enhanced battle outcome analysis.

        Parameters:
        max_rounds (int): Maximum rounds to simulate (uses DEFAULT_MAX_ROUNDS if None)

        Returns:
        str: Battle outcome description
        """
        if max_rounds is None:
            max_rounds = self.DEFAULT_MAX_ROUNDS

        # Display initial force composition and effectiveness
        print(f"=== SALVO COMBAT MODEL SIMULATION ===\n")
        print("Initial Forces:")
        print("Force A:")
        for ship in self.force_a:
            print(f"  - {ship.name}: OP={ship.offensive_power}, DP={ship.defensive_power:.2f}, SP={ship.staying_power}")

        print("\nForce B:")
        for ship in self.force_b:
            print(f"  - {ship.name}: OP={ship.offensive_power}, DP={ship.defensive_power:.2f}, SP={ship.staying_power}")

        # Display initial effectiveness metrics
        initial_stats = self.get_battle_statistics()
        print(f"\nInitial Force Effectiveness:")
        print(f"Force A: {initial_stats['total_a_offensive']:.1f} offensive power")
        print(f"Force B: {initial_stats['total_b_offensive']:.1f} offensive power")
        print(f"Offensive Ratio (A/B): {initial_stats['offensive_ratio']:.2f}")
        print("\n" + "="*50)

        # Execute battle rounds
        while self.execute_round() and self.round_number < max_rounds:
            # Print round summary
            round_log = self.battle_log[-1]
            print(f"\nRound {round_log['round']}:")
            for event in round_log['events']:
                print(f"  {event}")

        # Get comprehensive battle statistics using helper method
        final_stats = self.get_battle_statistics()
        result = final_stats['outcome']

        print("\n" + "="*50)
        print("BATTLE RESULT:")
        print(f"Winner: {result}")
        print(f"Rounds: {final_stats['rounds']}")
        print(f"Force A survivors: {final_stats['force_a_survivors']}")
        print(f"Force B survivors: {final_stats['force_b_survivors']}")

        # Display surviving ships with enhanced information
        if final_stats['surviving_ships_a']:
            print("Force A surviving ships:")
            for ship in final_stats['surviving_ships_a']:
                print(f"  - {ship.name}: {ship.get_health_percentage():.1f}% health")

        if final_stats['surviving_ships_b']:
            print("Force B surviving ships:")
            for ship in final_stats['surviving_ships_b']:
                print(f"  - {ship.name}: {ship.get_health_percentage():.1f}% health")

        return result
    
    def plot_battle_progress(self, title="Salvo Combat Model", ax=None):
        """
        Plot the battle dynamics over rounds with enhanced Salvo Model insights.

        Parameters:
        title (str): Plot title
        ax (matplotlib.axes): Axes to plot on. If None, creates new figure.
        """
        if not self.battle_log:
            print("No battle data to plot. Run simulation first.")
            return

        rounds = [log['round'] for log in self.battle_log]
        force_a_counts = [log['force_a_active'] for log in self.battle_log]
        force_b_counts = [log['force_b_active'] for log in self.battle_log]

        # Add initial state (round 0)
        initial_a = len([s for s in self.force_a if s.staying_power > 0])
        initial_b = len([s for s in self.force_b if s.staying_power > 0])
        rounds = [0] + rounds
        force_a_counts = [initial_a] + force_a_counts
        force_b_counts = [initial_b] + force_b_counts

        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        ax.plot(rounds, force_a_counts, 'b-', linewidth=2, label=f'Force A (initial: {initial_a})')
        ax.plot(rounds, force_b_counts, 'r-', linewidth=2, label=f'Force B (initial: {initial_b})')

        # Mark battle end
        if rounds:
            ax.axvline(x=max(rounds), color='gray', linestyle='--', alpha=0.7,
                      label=f"Battle ends: Round {max(rounds)}")

        ax.set_xlabel('Round')
        ax.set_ylabel('Active Ships')
        ax.set_title(f"{title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(rounds) if rounds else 1)
        ax.set_ylim(0, max(initial_a, initial_b) * 1.1)

        # Enhanced winner annotation with Salvo Model insights
        stats = self.get_battle_statistics()
        winner = stats['outcome']

        # Calculate initial offensive power for comparison
        initial_a_offensive = sum(ship.offensive_power for ship in self.force_a)
        initial_b_offensive = sum(ship.offensive_power for ship in self.force_b)

        info_text = f"Winner: {winner.split(' - ')[0] if ' - ' in winner else winner}\nSalvo Advantage: OP({initial_a_offensive:.0f}) vs OP({initial_b_offensive:.0f})"
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral'))

        if ax is None:
            plt.tight_layout()
            plt.show()
    
    @classmethod
    def plot_multiple_battles(cls, simulations, titles=None):
        """
        Plot multiple battle scenarios in parallel subplots.
        
        Parameters:
        simulations (list): List of SalvoCombatModel instances (already run)
        titles (list): List of titles for each subplot (optional)
        """
        n_battles = len(simulations)
        _, axes = plt.subplots(1, n_battles, figsize=(6*n_battles, 6))
        
        # Handle case where there's only one battle
        if n_battles == 1:
            axes = [axes]
        
        for i, simulation in enumerate(simulations):
            title = titles[i] if titles else f"Battle {i+1}"
            simulation.plot_battle_progress(title=title, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

    def run_monte_carlo_analysis(self, iterations: int = None, quiet: bool = True) -> dict:
        """
        Run Monte Carlo analysis to determine outcome probabilities.

        Parameters:
        iterations (int): Number of iterations (uses MONTE_CARLO_ITERATIONS if None)
        quiet (bool): If True, suppress individual simulation output

        Returns:
        dict: Statistical analysis of battle outcomes across multiple runs
        """
        if iterations is None:
            iterations = self.MONTE_CARLO_ITERATIONS

        results = {
            "Force A Victory": 0,
            "Force B Victory": 0,
            "Draw - Both forces have survivors": 0,
            "Mutual Annihilation": 0
        }

        battle_durations = []
        offensive_ratios = []

        if not quiet:
            print(f"=== MONTE CARLO ANALYSIS ({iterations} iterations) ===\n")

        for i in range(iterations):
            # Create fresh copies of forces for each simulation
            test_force_a = []
            for ship in self.force_a:
                test_force_a.append(Ship(
                    name=ship.name,
                    offensive_power=ship.offensive_power,
                    defensive_power=ship.defensive_power,
                    staying_power=ship.staying_power
                ))

            test_force_b = []
            for ship in self.force_b:
                test_force_b.append(Ship(
                    name=ship.name,
                    offensive_power=ship.offensive_power,
                    defensive_power=ship.defensive_power,
                    staying_power=ship.staying_power
                ))

            # Run simulation with unique seed
            test_sim = SalvoCombatModel(test_force_a, test_force_b, random_seed=i)

            # Execute battle quietly
            while test_sim.execute_round() and test_sim.round_number < self.DEFAULT_MAX_ROUNDS:
                pass

            # Get results
            stats = test_sim.get_battle_statistics()
            outcome = stats['outcome']
            results[outcome] += 1
            battle_durations.append(stats['rounds'])
            offensive_ratios.append(stats['offensive_ratio'])

        # Calculate statistics
        avg_duration = np.mean(battle_durations)
        std_duration = np.std(battle_durations)
        avg_offensive_ratio = np.mean(offensive_ratios)

        analysis = {
            'iterations': iterations,
            'outcome_probabilities': {outcome: (count / iterations) * 100
                                    for outcome, count in results.items()},
            'outcome_counts': results,
            'average_battle_duration': avg_duration,
            'std_battle_duration': std_duration,
            'average_offensive_ratio': avg_offensive_ratio,
            'battle_durations': battle_durations
        }

        if not quiet:
            print("Monte Carlo Results:")
            for outcome, probability in analysis['outcome_probabilities'].items():
                print(f"{outcome}: {probability:.1f}% ({results[outcome]}/{iterations})")
            print(f"\nAverage battle duration: {avg_duration:.1f} ± {std_duration:.1f} rounds")
            print(f"Average offensive ratio (A/B): {avg_offensive_ratio:.2f}")

        return analysis

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Balanced Naval Forces
    print("Example 1: Balanced Naval Forces")
    force_a1 = [
        Ship("Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
        Ship("Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5),
        Ship("Battleship Gamma", offensive_power=20, defensive_power=0.5, staying_power=8)
    ]
    
    force_b1 = [
        Ship("Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
        Ship("Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4),
        Ship("Cruiser Foxtrot", offensive_power=15, defensive_power=0.45, staying_power=6),
        Ship("Carrier Golf", offensive_power=25, defensive_power=0.3, staying_power=7)
    ]
    
    simulation1 = SalvoCombatModel(force_a1, force_b1, random_seed=42)
    result1 = simulation1.run_simulation()
    print(f"Battle duration: {simulation1.round_number} rounds")
    print(f"Result: {result1}")
    print()
    
    # Example 2: Superior Defensive Force
    print("Example 2: Superior Defensive Force")
    force_a2 = [
        Ship("Heavy Cruiser Alpha", offensive_power=15, defensive_power=0.6, staying_power=6),
        Ship("Battlecruiser Beta", offensive_power=18, defensive_power=0.7, staying_power=7)
    ]
    
    force_b2 = [
        Ship("Assault Ship Delta", offensive_power=25, defensive_power=0.2, staying_power=4),
        Ship("Missile Boat Echo", offensive_power=12, defensive_power=0.1, staying_power=2),
        Ship("Light Cruiser Foxtrot", offensive_power=10, defensive_power=0.3, staying_power=3)
    ]
    
    simulation2 = SalvoCombatModel(force_a2, force_b2, random_seed=123)
    result2 = simulation2.run_simulation()
    print(f"Battle duration: {simulation2.round_number} rounds")
    print(f"Result: {result2}")
    print()
    
    # Example 3: Overwhelming Force
    print("Example 3: Overwhelming Force")
    force_a3 = [
        Ship("Light Frigate Alpha", offensive_power=6, defensive_power=0.3, staying_power=2),
        Ship("Corvette Beta", offensive_power=4, defensive_power=0.25, staying_power=1)
    ]
    
    force_b3 = [
        Ship("Dreadnought Delta", offensive_power=35, defensive_power=0.8, staying_power=12),
        Ship("Super Carrier Echo", offensive_power=40, defensive_power=0.6, staying_power=10)
    ]
    
    simulation3 = SalvoCombatModel(force_a3, force_b3, random_seed=456)
    result3 = simulation3.run_simulation()
    print(f"Battle duration: {simulation3.round_number} rounds")
    print(f"Result: {result3}")
    print()
    
    # Plot all three examples using the new method
    try:
        SalvoCombatModel.plot_multiple_battles(
            simulations=[simulation1, simulation2, simulation3],
            titles=["Example 1: Balanced Naval Forces", 
                   "Example 2: Superior Defensive Force", 
                   "Example 3: Overwhelming Force"]
        )
    except:
        print("Matplotlib not available for plotting")
        
    # Single battle plot example
    try:
        simulation1.plot_battle_progress("Detailed Battle Analysis - Example 1")
    except:
        print("Matplotlib not available for single plot")
    
    print(f"\n=== COMPARATIVE STATISTICS ===")
    print(f"Example 1 - Initial Force A offensive: {sum(s.offensive_power for s in force_a1)}")
    print(f"Example 1 - Initial Force B offensive: {sum(s.offensive_power for s in force_b1)}")
    print(f"Example 2 - Initial Force A offensive: {sum(s.offensive_power for s in force_a2)}")
    print(f"Example 2 - Initial Force B offensive: {sum(s.offensive_power for s in force_b2)}")
    print(f"Example 3 - Initial Force A offensive: {sum(s.offensive_power for s in force_a3)}")
    print(f"Example 3 - Initial Force B offensive: {sum(s.offensive_power for s in force_b3)}")
    
    # Enhanced Monte Carlo analysis using new method
    print(f"\n=== ENHANCED MONTE CARLO ANALYSIS - Example 1 ===")
    monte_carlo_results = simulation1.run_monte_carlo_analysis(iterations=50, quiet=False)

    # Comparison with Lanchester Laws
    print("\n=== COMPARISON: Salvo Model vs Lanchester Laws ===")
    print("="*60)
    print("Salvo Model accounts for:")
    print("- Individual ship characteristics (staying power)")
    print("- Defensive capabilities (missile interception)")
    print("- Discrete rounds rather than continuous time")
    print("- Probabilistic combat resolution")
    print("- Ship-level damage tracking")

    # Calculate theoretical Lanchester predictions for comparison
    force_a_total_offensive = sum(s.offensive_power for s in force_a1)
    force_b_total_offensive = sum(s.offensive_power for s in force_b1)

    print(f"\nForce Comparison:")
    print(f"Force A: {len(force_a1)} ships, {force_a_total_offensive:.0f} total offensive power")
    print(f"Force B: {len(force_b1)} ships, {force_b_total_offensive:.0f} total offensive power")

    # Linear Law prediction (simple difference)
    linear_advantage = force_a_total_offensive - force_b_total_offensive
    print(f"Linear Law prediction: {linear_advantage:.0f} advantage to {'A' if linear_advantage > 0 else 'B'}")

    # Square Law prediction (quadratic difference)
    square_advantage = force_a_total_offensive**2 - force_b_total_offensive**2
    print(f"Square Law prediction: {square_advantage:.0f} quadratic advantage to {'A' if square_advantage > 0 else 'B'}")

    print(f"Salvo Model actual result: {result1}")
    print(f"Salvo Model probabilistic outcome: Force A wins {monte_carlo_results['outcome_probabilities']['Force A Victory']:.1f}% of the time")

    # Test simple simulation method
    print(f"\n=== SIMPLE SIMULATION TEST ===")
    simple_result = simulation1.simple_simulation(quiet=False)