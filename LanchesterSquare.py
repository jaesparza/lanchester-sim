import numpy as np
import matplotlib.pyplot as plt

class LanchesterSquare:
    """
    Implementation of Lanchester's Square Law for modern combat.
    
    The Square Law assumes ranged combat where each unit can engage multiple enemies.
    Combat effectiveness is proportional to the square of force size.
    """
    
    def __init__(self, A0, B0, alpha, beta):
        """
        Initialize the combat scenario.
        
        Parameters:
        A0 (float): Initial strength of force A
        B0 (float): Initial strength of force B  
        alpha (float): Effectiveness coefficient of A against B
        beta (float): Effectiveness coefficient of B against A
        """
        if A0 < 0 or B0 < 0:
            raise ValueError("Initial strengths must be non-negative.")
        if alpha < 0 or beta < 0:
            raise ValueError("Effectiveness coefficients must be non-negative.")
        self.A0 = A0
        self.B0 = B0
        self.alpha = alpha
        self.beta = beta
    
    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Square Law.
        
        Uses the Square Law relationship: α*A²(t) - β*B²(t) = α*A₀² - β*B₀²
        
        Returns:
        dict: Contains time arrays, force strengths, battle end time, and winner
        """
        # Calculate the invariant quantity
        invariant = self.alpha * self.A0**2 - self.beta * self.B0**2
        
        # Determine winner based on invariant
        if invariant > 0:
            winner = 'A'
            # A wins when B = 0, so A_final = sqrt(invariant/alpha)
            remaining_strength = np.sqrt(invariant / self.alpha)
            # Time when B is eliminated
            if self.beta > 0:
                t_end = (1/np.sqrt(self.alpha * self.beta)) * np.arctanh(np.sqrt(self.beta/self.alpha) * self.B0/np.sqrt(invariant/self.alpha + self.B0**2))
            else:
                t_end = self.B0 / (self.alpha * self.A0)  # Degenerate case
        elif invariant < 0:
            winner = 'B'
            # B wins when A = 0, so B_final = sqrt(-invariant/beta)
            remaining_strength = np.sqrt(-invariant / self.beta)
            # Time when A is eliminated
            if self.alpha > 0:
                t_end = (1/np.sqrt(self.alpha * self.beta)) * np.arctanh(np.sqrt(self.alpha/self.beta) * self.A0/np.sqrt(-invariant/self.beta + self.A0**2))
            else:
                t_end = self.A0 / (self.beta * self.B0)  # Degenerate case
        else:
            winner = 'Draw'
            remaining_strength = 0
            # Both eliminated simultaneously - use approximation for mutual annihilation
            t_end = max(self.A0/np.sqrt(self.alpha * self.B0), self.B0/np.sqrt(self.beta * self.A0))
        
        # For numerical stability, use a simpler approach for time calculation
        if np.isnan(t_end) or np.isinf(t_end):
            # Use approximate method: integrate until one force is nearly eliminated
            if winner == 'A':
                t_end = self.B0 / (np.sqrt(self.alpha) * self.A0)
            elif winner == 'B':
                t_end = self.A0 / (np.sqrt(self.beta) * self.B0)
            else:
                t_end = min(self.A0 / (np.sqrt(self.beta) * self.B0), self.B0 / (np.sqrt(self.alpha) * self.A0))
        
        # Create time array
        if t_max is None:
            t_max = max(t_end * 1.2, t_end + 0.5)
        
        t = np.linspace(0, t_max, 1000)
        
        # Calculate force strengths using the Square Law relationship
        # This is more complex than Linear Law, so we'll use a simpler approximation
        # for visualization purposes
        A_t = np.zeros_like(t)
        B_t = np.zeros_like(t)
        
        for i, time in enumerate(t):
            if time >= t_end:
                if winner == 'A':
                    A_t[i] = remaining_strength
                    B_t[i] = 0
                elif winner == 'B':
                    A_t[i] = 0
                    B_t[i] = remaining_strength
                else:
                    A_t[i] = 0
                    B_t[i] = 0
            else:
                # Approximate solution using exponential decay
                decay_factor_A = self.beta * self.B0 * time / (1 + self.beta * self.B0 * time)
                decay_factor_B = self.alpha * self.A0 * time / (1 + self.alpha * self.A0 * time)
                
                A_t[i] = max(0, self.A0 * (1 - decay_factor_A))
                B_t[i] = max(0, self.B0 * (1 - decay_factor_B))
        
        # Calculate casualties
        if winner == 'A':
            A_casualties = self.A0 - remaining_strength
            B_casualties = self.B0
        elif winner == 'B':
            A_casualties = self.A0
            B_casualties = self.B0 - remaining_strength
        else:
            A_casualties = self.A0
            B_casualties = self.B0
        
        return {
            'time': t,
            'A': A_t,
            'B': B_t,
            'battle_end_time': t_end,
            'winner': winner,
            'remaining_strength': remaining_strength,
            'A_casualties': A_casualties,
            'B_casualties': B_casualties,
            'invariant': invariant
        }
    
    def simple_analytical_solution(self, t_max=None):
        """
        Simplified analytical solution using the key Square Law insight.
        
        For equal effectiveness (alpha = beta), the winner is determined by
        initial force sizes, and A_final = sqrt(A₀² - B₀²) if A wins.
        """
        # Calculate the Square Law invariant for equal effectiveness case
        if abs(self.alpha - self.beta) < 1e-10:  # Approximately equal
            effectiveness = self.alpha  # or self.beta, they're the same
            
            if self.A0**2 > self.B0**2:
                winner = 'A'
                remaining_strength = np.sqrt(self.A0**2 - self.B0**2)
                # Approximate battle duration
                t_end = 1.0 / (effectiveness * np.sqrt(self.A0 * self.B0))
            elif self.B0**2 > self.A0**2:
                winner = 'B'
                remaining_strength = np.sqrt(self.B0**2 - self.A0**2)
                t_end = 1.0 / (effectiveness * np.sqrt(self.A0 * self.B0))
            else:
                winner = 'Draw'
                remaining_strength = 0
                t_end = 1.0 / (effectiveness * np.sqrt(self.A0 * self.B0))
        else:
            # Use general case (fallback to original method)
            return self.analytical_solution(t_max)
        
        # Create time array
        if t_max is None:
            t_max = max(t_end * 1.5, 2.0)
        
        t = np.linspace(0, t_max, 1000)
        
        # Generate approximate force trajectories
        # Square Law: forces decrease in a curved fashion, not linearly
        A_t = np.zeros_like(t)
        B_t = np.zeros_like(t)
        
        for i, time in enumerate(t):
            if time >= t_end:
                if winner == 'A':
                    A_t[i] = remaining_strength
                    B_t[i] = 0
                elif winner == 'B':
                    A_t[i] = 0
                    B_t[i] = remaining_strength
                else:
                    A_t[i] = 0
                    B_t[i] = 0
            else:
                # Square Law approximation: faster initial casualties, then slowing
                progress = time / t_end
                
                if winner == 'A':
                    # B decreases faster (gets eliminated)
                    B_t[i] = self.B0 * (1 - progress**0.7)  # Curved decrease
                    A_t[i] = np.sqrt(max(0, self.A0**2 - (self.B0**2 - B_t[i]**2)))
                elif winner == 'B':
                    # A decreases faster (gets eliminated)
                    A_t[i] = self.A0 * (1 - progress**0.7)  # Curved decrease  
                    B_t[i] = np.sqrt(max(0, self.B0**2 - (self.A0**2 - A_t[i]**2)))
                else:
                    # Both decrease at same rate
                    A_t[i] = self.A0 * (1 - progress)
                    B_t[i] = self.B0 * (1 - progress)
                
                A_t[i] = max(0, A_t[i])
                B_t[i] = max(0, B_t[i])
        
        return {
            'time': t,
            'A': A_t,
            'B': B_t,
            'battle_end_time': t_end,
            'winner': winner,
            'remaining_strength': remaining_strength,
            'A_casualties': self.A0 - (remaining_strength if winner == 'A' else 0),
            'B_casualties': self.B0 - (remaining_strength if winner == 'B' else 0),
            'invariant': self.alpha * self.A0**2 - self.beta * self.B0**2
        }
    
    def plot_battle(self, solution=None, title="Lanchester Square Law", ax=None):
        """
        Plot the battle dynamics over time.
        
        Parameters:
        solution (dict): Solution dictionary from analytical_solution()
        title (str): Plot title
        ax (matplotlib.axes): Axes to plot on. If None, creates new figure.
        """
        if solution is None:
            solution = self.simple_analytical_solution()
        
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        ax.plot(solution['time'], solution['A'], 'b-', linewidth=2, label=f'Force A (initial: {self.A0})')
        ax.plot(solution['time'], solution['B'], 'r-', linewidth=2, label=f'Force B (initial: {self.B0})')
        
        # Mark battle end
        if 'battle_end_time' in solution:
            ax.axvline(x=solution['battle_end_time'], color='gray', linestyle='--', alpha=0.7, 
                       label=f"Battle ends: t={solution['battle_end_time']:.2f}")
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Force Strength')
        ax.set_title(f"{title}\nα={self.alpha}, β={self.beta}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(solution['time']))
        ax.set_ylim(0, max(self.A0, self.B0) * 1.1)
        
        # Add winner annotation and Square Law insight
        info_text = f"Winner: {solution['winner']}\nSquare Law Advantage: {self.A0**2:.0f} vs {self.B0**2:.0f}"
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    @classmethod
    def plot_multiple_battles(cls, battles, solutions=None, titles=None):
        """
        Plot multiple battle scenarios in parallel subplots.
        
        Parameters:
        battles (list): List of LanchesterSquare instances
        solutions (list): List of solution dictionaries (optional)
        titles (list): List of titles for each subplot (optional)
        """
        n_battles = len(battles)
        fig, axes = plt.subplots(1, n_battles, figsize=(6*n_battles, 6))
        
        # Handle case where there's only one battle
        if n_battles == 1:
            axes = [axes]
        
        for i, battle in enumerate(battles):
            solution = solutions[i] if solutions else None
            title = titles[i] if titles else f"Battle {i+1}"
            
            battle.plot_battle(solution=solution, title=title, ax=axes[i])
        
        plt.tight_layout()
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Equal effectiveness, size advantage
    print("Example 1: Equal Effectiveness - Size Matters")
    battle1 = LanchesterSquare(A0=100, B0=60, alpha=0.01, beta=0.01)
    solution1 = battle1.simple_analytical_solution()
    
    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print(f"Square Law prediction: sqrt({battle1.A0}² - {battle1.B0}²) = sqrt({battle1.A0**2} - {battle1.B0**2}) = {np.sqrt(battle1.A0**2 - battle1.B0**2):.1f}")
    print()
    
    # Example 2: Effectiveness vs. numbers
    print("Example 2: Superior Effectiveness vs. Numbers")
    battle2 = LanchesterSquare(A0=80, B0=120, alpha=0.02, beta=0.01)
    solution2 = battle2.analytical_solution()
    
    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print(f"A's effective strength: α×A₀² = {battle2.alpha}×{battle2.A0}² = {battle2.alpha * battle2.A0**2:.0f}")
    print(f"B's effective strength: β×B₀² = {battle2.beta}×{battle2.B0}² = {battle2.beta * battle2.B0**2:.0f}")
    print(f"Invariant: {solution2['invariant']:.0f} ({'A wins' if solution2['invariant'] > 0 else 'B wins' if solution2['invariant'] < 0 else 'Draw'})")
    print()
    
    # Plot both examples using the new method
    LanchesterSquare.plot_multiple_battles(
        battles=[battle1, battle2],
        solutions=[solution1, solution2],
        titles=["Example 1: Equal Effectiveness", "Example 2: Superior Effectiveness"]
    )
    
    # Comparison with Linear Law
    print("\nComparison: Square Law vs Linear Law")
    print("="*50)
    print(f"Scenario: A0={battle1.A0}, B0={battle1.B0}, equal effectiveness")
    print(f"Linear Law winner remainder: {battle1.A0 - battle1.B0} units")
    print(f"Square Law winner remainder: {solution1['remaining_strength']:.1f} units")
    print(f"Square Law advantage: {solution1['remaining_strength'] - (battle1.A0 - battle1.B0):.1f} more survivors")