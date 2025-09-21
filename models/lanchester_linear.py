import numpy as np
import matplotlib.pyplot as plt

class LanchesterLinear:
    """
    Implementation of Lanchester's Linear Law for ancient-style combat.

    The Linear Law assumes sequential combat where forces engage one-on-one.
    Combat effectiveness is proportional to force size.
    """

    # Constants for numerical calculations
    EFFECTIVENESS_TOLERANCE = 1e-10  # Numerical tolerance for determining if alpha ≈ beta (equal effectiveness). Avoids floating-point comparison issues.
    TIME_EXTENSION_FACTOR = 1.2      # Extends time arrays 20% beyond battle end for better visualization of final states
    TIME_MINIMUM_EXTENSION = 1.0     # Minimum time extension for very short battles to ensure readable plots
    DEFAULT_TIME_POINTS = 1000       # Number of time points for trajectory calculations. Balances smoothness with performance.
    SIMPLE_TIME_EXTENSION = 1.5      # 50% time extension for simple analytical solution (needs more padding for linear trajectories)
    SIMPLE_MINIMUM_TIME = 2.0        # Minimum visualization time for simple solution to show force dynamics clearly
    
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

    def calculate_battle_outcome(self):
        """
        Calculate battle outcome based on Linear Law dynamics.

        Returns:
        tuple: (winner, remaining_strength, t_end)
        """
        # Calculate when each force would be eliminated
        t_A_eliminated = self.A0 / self.beta if self.beta > 0 else np.inf
        t_B_eliminated = self.B0 / self.alpha if self.alpha > 0 else np.inf

        # Battle ends when first force is eliminated
        t_end = min(t_A_eliminated, t_B_eliminated)

        # Determine winner and remaining strength
        if t_A_eliminated < t_B_eliminated:
            winner = 'B'
            remaining_strength = self.B0 - self.alpha * t_end
        elif t_B_eliminated < t_A_eliminated:
            winner = 'A'
            remaining_strength = self.A0 - self.beta * t_end
        else:
            winner = 'Draw'
            remaining_strength = 0

        return winner, remaining_strength, t_end

    def generate_force_trajectories(self, t):
        """
        Generate force strength trajectories over time using Linear Law.

        Parameters:
        t (array): Time array

        Returns:
        tuple: (A_t, B_t) arrays of force strengths
        """
        # Linear Law dynamics: dA/dt = -β, dB/dt = -α (constant attrition rates)
        A_t = np.maximum(0, self.A0 - self.beta * t)
        B_t = np.maximum(0, self.B0 - self.alpha * t)

        return A_t, B_t
    
    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Linear Law.

        Uses the Linear Law relationship where forces decrease at constant rates:
        dA/dt = -β (force A loses β units per time)
        dB/dt = -α (force B loses α units per time)

        Returns:
        dict: Contains time arrays, force strengths, battle end time, and winner
        """
        # Calculate battle outcome using helper method
        winner, remaining_strength, t_end = self.calculate_battle_outcome()

        # Create time array
        if t_max is None:
            t_max = max(t_end * self.TIME_EXTENSION_FACTOR, t_end + self.TIME_MINIMUM_EXTENSION)

        t = np.linspace(0, t_max, self.DEFAULT_TIME_POINTS)

        # Generate force trajectories using helper method
        A_t, B_t = self.generate_force_trajectories(t)

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
            'linear_advantage': self.A0 - self.B0  # Linear Law insight: advantage = initial difference
        }

    def simple_analytical_solution(self, t_max=None):
        """
        Simplified analytical solution using the key Linear Law insight.

        For equal effectiveness (alpha = beta), the winner is determined by
        initial force sizes, and A_final = A₀ - B₀ if A wins (simple difference).
        """
        # Calculate battle outcome using helper method
        winner, remaining_strength, t_end = self.calculate_battle_outcome()

        # For Linear Law with equal effectiveness, use simplified calculation
        if abs(self.alpha - self.beta) < self.EFFECTIVENESS_TOLERANCE:  # Approximately equal
            effectiveness = self.alpha  # or self.beta, they're the same

            if self.A0 > self.B0:
                winner = 'A'
                remaining_strength = self.A0 - self.B0
                # Approximate battle duration: time for smaller force to be eliminated
                t_end = self.B0 / effectiveness
            elif self.B0 > self.A0:
                winner = 'B'
                remaining_strength = self.B0 - self.A0
                t_end = self.A0 / effectiveness
            else:
                winner = 'Draw'
                remaining_strength = 0
                t_end = self.A0 / effectiveness  # Both eliminated simultaneously
        # If effectiveness differs significantly, fall back to general solution
        else:
            return self.analytical_solution(t_max)

        # Create time array
        if t_max is None:
            t_max = max(t_end * self.SIMPLE_TIME_EXTENSION, self.SIMPLE_MINIMUM_TIME)

        t = np.linspace(0, t_max, self.DEFAULT_TIME_POINTS)

        # Generate force trajectories using helper method
        A_t, B_t = self.generate_force_trajectories(t)

        return {
            'time': t,
            'A': A_t,
            'B': B_t,
            'battle_end_time': t_end,
            'winner': winner,
            'remaining_strength': remaining_strength,
            'A_casualties': self.A0 - (remaining_strength if winner == 'A' else 0),
            'B_casualties': self.B0 - (remaining_strength if winner == 'B' else 0),
            'linear_advantage': self.A0 - self.B0
        }
    

    
    def plot_battle(self, solution=None, title="Lanchester Linear Law", ax=None):
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

        # Add winner annotation and Linear Law insight
        info_text = f"Winner: {solution['winner']}\nLinear Law Advantage: {self.A0:.0f} - {self.B0:.0f} = {self.A0 - self.B0:.0f}"
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))

        if ax is None:
            plt.tight_layout()
            plt.show()
    
    @classmethod
    def plot_multiple_battles(cls, battles, solutions=None, titles=None):
        """
        Plot multiple battle scenarios in parallel subplots.
        
        Parameters:
        battles (list): List of LanchesterLinear instances
        solutions (list): List of solution dictionaries (optional)
        titles (list): List of titles for each subplot (optional)
        """
        n_battles = len(battles)
        _, axes = plt.subplots(1, n_battles, figsize=(6*n_battles, 6))
        
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
    battle1 = LanchesterLinear(A0=100, B0=60, alpha=0.5, beta=0.5)
    solution1 = battle1.simple_analytical_solution()

    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print(f"Linear Law prediction: {battle1.A0} - {battle1.B0} = {battle1.A0 - battle1.B0:.1f}")
    print()

    # Example 2: Effectiveness vs. numbers
    print("Example 2: Superior Effectiveness vs. Numbers")
    battle2 = LanchesterLinear(A0=80, B0=120, alpha=0.8, beta=0.4)
    solution2 = battle2.analytical_solution()

    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print(f"A's effectiveness: α = {battle2.alpha}")
    print(f"B's effectiveness: β = {battle2.beta}")
    print(f"Linear advantage: {solution2['linear_advantage']:.0f} ({'A favored' if solution2['linear_advantage'] > 0 else 'B favored' if solution2['linear_advantage'] < 0 else 'Equal'})")
    print()

    # Plot both examples using the new method
    LanchesterLinear.plot_multiple_battles(
        battles=[battle1, battle2],
        solutions=[solution1, solution2],
        titles=["Example 1: Equal Effectiveness", "Example 2: Superior Effectiveness"]
    )

    # Comparison with Square Law
    print("\nComparison: Linear Law vs Square Law")
    print("="*50)
    print(f"Scenario: A0={battle1.A0}, B0={battle1.B0}, equal effectiveness")
    print(f"Linear Law winner remainder: {solution1['remaining_strength']:.1f} units")
    print(f"Square Law winner remainder: {np.sqrt(battle1.A0**2 - battle1.B0**2):.1f} units")
    print(f"Square Law advantage: {np.sqrt(battle1.A0**2 - battle1.B0**2) - solution1['remaining_strength']:.1f} more survivors")