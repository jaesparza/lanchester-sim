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
    
    def __init__(self, force_a: List[Ship], force_b: List[Ship], random_seed: Optional[int] = None):
        self.force_a = force_a
        self.force_b = force_b
        self.round_number = 0
        self.battle_log = []
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
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
            intercept_prob = min(0.95, total_defensive / len(operational_defenders))  # Cap at 95%
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
        round_log = {
            'round': self.round_number,
            'force_a_active': len([s for s in self.force_a if s.is_operational()]),
            'force_b_active': len([s for s in self.force_b if s.is_operational()]),
            'events': []
        }
        
        # Check if battle is already over
        a_operational = [s for s in self.force_a if s.is_operational()]
        b_operational = [s for s in self.force_b if s.is_operational()]
        
        if not a_operational or not b_operational:
            return False
        
        # Force A attacks Force B
        missiles_through, hits_dist = self.calculate_salvo_effectiveness(a_operational, b_operational)
        round_log['events'].append(f"Force A fires {sum(s.offensive_power for s in a_operational):.1f} missiles, {missiles_through} penetrate defenses")
        
        operational_b = [s for s in self.force_b if s.is_operational()]
        for i, ship in enumerate(operational_b):
            if i < len(hits_dist) and hits_dist[i] > 0:
                actual_hits = ship.take_damage(hits_dist[i])
                if actual_hits > 0:
                    round_log['events'].append(f"{ship.name} takes {actual_hits} hit(s), health: {ship.get_health_percentage():.1f}%")
                    if not ship.is_operational():
                        round_log['events'].append(f"{ship.name} is destroyed!")
        
        # Force B attacks Force A (if any ships remain)
        b_operational = [s for s in self.force_b if s.is_operational()]
        if b_operational:
            missiles_through, hits_dist = self.calculate_salvo_effectiveness(b_operational, a_operational)
            round_log['events'].append(f"Force B fires {sum(s.offensive_power for s in b_operational):.1f} missiles, {missiles_through} penetrate defenses")
            
            operational_a = [s for s in self.force_a if s.is_operational()]
            for i, ship in enumerate(operational_a):
                if i < len(hits_dist) and hits_dist[i] > 0:
                    actual_hits = ship.take_damage(hits_dist[i])
                    if actual_hits > 0:
                        round_log['events'].append(f"{ship.name} takes {actual_hits} hit(s), health: {ship.get_health_percentage():.1f}%")
                        if not ship.is_operational():
                            round_log['events'].append(f"{ship.name} is destroyed!")
        
        self.battle_log.append(round_log)
        
        # Check if battle continues
        a_operational = [s for s in self.force_a if s.is_operational()]
        b_operational = [s for s in self.force_b if s.is_operational()]
        
        return len(a_operational) > 0 and len(b_operational) > 0
    
    def run_simulation(self, max_rounds: int = 50) -> str:
        """Run the complete simulation"""
        print(f"=== SALVO COMBAT MODEL SIMULATION ===\n")
        print("Initial Forces:")
        print("Force A:")
        for ship in self.force_a:
            print(f"  - {ship.name}: OP={ship.offensive_power}, DP={ship.defensive_power:.2f}, SP={ship.staying_power}")
        print("\nForce B:")
        for ship in self.force_b:
            print(f"  - {ship.name}: OP={ship.offensive_power}, DP={ship.defensive_power:.2f}, SP={ship.staying_power}")
        print("\n" + "="*50)
        
        while self.execute_round() and self.round_number < max_rounds:
            # Print round summary
            round_log = self.battle_log[-1]
            print(f"\nRound {round_log['round']}:")
            for event in round_log['events']:
                print(f"  {event}")
        
        # Determine winner
        a_survivors = [s for s in self.force_a if s.is_operational()]
        b_survivors = [s for s in self.force_b if s.is_operational()]
        
        print("\n" + "="*50)
        print("BATTLE RESULT:")
        
        if a_survivors and b_survivors:
            result = "Draw - Both forces have survivors"
        elif a_survivors:
            result = "Force A Victory"
        elif b_survivors:
            result = "Force B Victory"
        else:
            result = "Mutual Annihilation"
        
        print(f"Winner: {result}")
        print(f"Rounds: {self.round_number}")
        print(f"Force A survivors: {len(a_survivors)}")
        print(f"Force B survivors: {len(b_survivors)}")
        
        if a_survivors:
            print("Force A surviving ships:")
            for ship in a_survivors:
                print(f"  - {ship.name}: {ship.get_health_percentage():.1f}% health")
        
        if b_survivors:
            print("Force B surviving ships:")
            for ship in b_survivors:
                print(f"  - {ship.name}: {ship.get_health_percentage():.1f}% health")
        
        return result
    
    def plot_battle_progress(self, title="Salvo Combat Model", ax=None):
        """
        Plot the battle dynamics over rounds.
        
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
        
        # Add winner annotation
        a_survivors = len([s for s in self.force_a if s.is_operational()])
        b_survivors = len([s for s in self.force_b if s.is_operational()])
        
        if a_survivors > 0 and b_survivors == 0:
            winner = "Force A"
        elif b_survivors > 0 and a_survivors == 0:
            winner = "Force B"
        elif a_survivors == 0 and b_survivors == 0:
            winner = "Mutual Annihilation"
        else:
            winner = "Draw"
        
        ax.text(0.02, 0.98, f"Winner: {winner}",
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
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
        fig, axes = plt.subplots(1, n_battles, figsize=(6*n_battles, 6))
        
        # Handle case where there's only one battle
        if n_battles == 1:
            axes = [axes]
        
        for i, simulation in enumerate(simulations):
            title = titles[i] if titles else f"Battle {i+1}"
            simulation.plot_battle_progress(title=title, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

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
    
    # Quick Monte Carlo for first example
    print(f"\n=== MONTE CARLO ANALYSIS - Example 1 (50 simulations) ===")
    results = {"Force A Victory": 0, "Force B Victory": 0, "Draw - Both forces have survivors": 0, "Mutual Annihilation": 0}
    
    for i in range(50):
        # Reset forces for Example 1
        test_force_a = [
            Ship("Destroyer Alpha", offensive_power=8, defensive_power=0.3, staying_power=3),
            Ship("Cruiser Beta", offensive_power=12, defensive_power=0.4, staying_power=5),
            Ship("Battleship Gamma", offensive_power=20, defensive_power=0.5, staying_power=8)
        ]
        
        test_force_b = [
            Ship("Frigate Delta", offensive_power=6, defensive_power=0.4, staying_power=2),
            Ship("Destroyer Echo", offensive_power=10, defensive_power=0.35, staying_power=4),
            Ship("Cruiser Foxtrot", offensive_power=15, defensive_power=0.45, staying_power=6),
            Ship("Carrier Golf", offensive_power=25, defensive_power=0.3, staying_power=7)
        ]
        
        test_sim = SalvoCombatModel(test_force_a, test_force_b, random_seed=i)
        # Run quietly
        while test_sim.execute_round() and test_sim.round_number < 50:
            pass
        
        a_survivors = [s for s in test_sim.force_a if s.is_operational()]
        b_survivors = [s for s in test_sim.force_b if s.is_operational()]
        
        if a_survivors and b_survivors:
            result = "Draw - Both forces have survivors"
        elif a_survivors:
            result = "Force A Victory"
        elif b_survivors:
            result = "Force B Victory"
        else:
            result = "Mutual Annihilation"
        
        results[result] += 1
    
    for outcome, count in results.items():
        print(f"{outcome}: {count*2}% ({count}/50)")