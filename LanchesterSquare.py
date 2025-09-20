import numpy as np
import matplotlib.pyplot as plt

class LanchesterSquare:
    """
    Implementation of Lanchester's Square Law for modern combat.

    The Square Law assumes ranged combat where each unit can engage multiple enemies.
    Combat effectiveness is proportional to the square of force size.
    """

    # Constants for numerical calculations
    EFFECTIVENESS_TOLERANCE = 1e-10  # Numerical tolerance for determining if alpha ≈ beta (equal effectiveness). Avoids floating-point comparison issues.
    TIME_EXTENSION_FACTOR = 1.2      # Extends time arrays 20% beyond battle end for better visualization of final states
    TIME_MINIMUM_EXTENSION = 0.5     # Minimum time extension for very short battles to ensure readable plots
    DEFAULT_TIME_POINTS = 1000       # Number of time points for trajectory calculations. Balances smoothness with performance.
    SIMPLE_TIME_EXTENSION = 1.5      # 50% time extension for simple analytical solution (needs more padding for curved trajectories)
    SIMPLE_MINIMUM_TIME = 2.0        # Minimum visualization time for simple solution to show force dynamics clearly
    CURVE_EXPONENT = 0.7             # Exponent for curved force decrease in simple solution. Creates realistic non-linear casualty patterns where initial losses are slower, then accelerate.
    
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
        Calculate battle outcome based on Square Law invariant.

        Returns:
        tuple: (winner, remaining_strength, invariant)
        """
        invariant = self.alpha * self.A0**2 - self.beta * self.B0**2

        if invariant > 0:
            winner = 'A'
            remaining_strength = np.sqrt(invariant / self.alpha)
        elif invariant < 0:
            winner = 'B'
            remaining_strength = np.sqrt(-invariant / self.beta)
        else:
            winner = 'Draw'
            remaining_strength = 0

        return winner, remaining_strength, invariant

    def calculate_battle_end_time(self, winner, remaining_strength, invariant):
        """
        Calculate when the battle ends based on winner and invariant.

        Parameters:
        winner (str): 'A', 'B', or 'Draw'
        remaining_strength (float): Strength of winning force
        invariant (float): Square Law invariant

        Returns:
        float: Battle end time
        """
        if winner == 'A':
            # Time when B is eliminated
            if self.beta > 0:
                # Break down complex arctanh calculation for numerical stability
                ratio = np.sqrt(self.beta / self.alpha)
                denominator = np.sqrt(invariant / self.alpha + self.B0**2)
                arg = ratio * self.B0 / denominator

                # Check for valid arctanh domain [-1, 1]
                if abs(arg) >= 1.0:
                    t_end = self.B0 / (np.sqrt(self.alpha) * self.A0)  # Fallback approximation
                else:
                    t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                t_end = self.B0 / (np.sqrt(self.alpha) * self.A0)  # Degenerate case
        elif winner == 'B':
            # Time when A is eliminated
            if self.alpha > 0:
                # Break down complex arctanh calculation for numerical stability
                ratio = np.sqrt(self.alpha / self.beta)
                denominator = np.sqrt(-invariant / self.beta + self.A0**2)
                arg = ratio * self.A0 / denominator

                # Check for valid arctanh domain [-1, 1]
                if abs(arg) >= 1.0:
                    t_end = self.A0 / (np.sqrt(self.beta) * self.B0)  # Fallback approximation
                else:
                    t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                t_end = self.A0 / (np.sqrt(self.beta) * self.B0)  # Degenerate case
        else:
            # Both eliminated simultaneously - use approximation for mutual annihilation
            t_end = max(self.A0/np.sqrt(self.alpha * self.B0), self.B0/np.sqrt(self.beta * self.A0))

        return t_end

    def generate_force_trajectories(self, winner, remaining_strength, t_end, t, invariant):
        """
        Generate force strength trajectories over time.

        Parameters:
        winner (str): Battle winner
        remaining_strength (float): Final strength of winner
        t_end (float): Battle end time
        t (array): Time array
        invariant (float): Square Law invariant for proper dynamics

        Returns:
        tuple: (A_t, B_t) arrays of force strengths
        """
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
                # Square Law dynamics: dA/dt = -β*A*B, dB/dt = -α*A*B
                # Use Square Law approximation based on invariant conservation
                time_ratio = time / t_end

                if winner == 'A':
                    # B decreases faster, maintaining Square Law relationship
                    B_remaining = self.B0 * (1 - time_ratio**2)  # Quadratic decay for losing force
                    # Use invariant: α*A² - β*B² = constant
                    A_squared = (invariant + self.beta * B_remaining**2) / self.alpha
                    A_t[i] = max(0, np.sqrt(max(0, A_squared)))
                    B_t[i] = max(0, B_remaining)
                elif winner == 'B':
                    # A decreases faster, maintaining Square Law relationship
                    A_remaining = self.A0 * (1 - time_ratio**2)  # Quadratic decay for losing force
                    # Use invariant: α*A² - β*B² = constant
                    B_squared = (self.alpha * A_remaining**2 - invariant) / self.beta
                    A_t[i] = max(0, A_remaining)
                    B_t[i] = max(0, np.sqrt(max(0, B_squared)))
                else:
                    # Draw case: both decrease at same rate
                    decay = 1 - time_ratio**2
                    A_t[i] = max(0, self.A0 * decay)
                    B_t[i] = max(0, self.B0 * decay)

        return A_t, B_t

    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Square Law.

        Uses the Square Law relationship: α*A²(t) - β*B²(t) = α*A₀² - β*B₀²

        Returns:
        dict: Contains time arrays, force strengths, battle end time, and winner
        """
        # Calculate battle outcome using helper method
        winner, remaining_strength, invariant = self.calculate_battle_outcome()

        # Calculate battle end time using helper method
        t_end = self.calculate_battle_end_time(winner, remaining_strength, invariant)

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
            t_max = max(t_end * self.TIME_EXTENSION_FACTOR, t_end + self.TIME_MINIMUM_EXTENSION)

        t = np.linspace(0, t_max, self.DEFAULT_TIME_POINTS)

        # Generate force trajectories using helper method
        A_t, B_t = self.generate_force_trajectories(winner, remaining_strength, t_end, t, invariant)
        
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
        if abs(self.alpha - self.beta) < self.EFFECTIVENESS_TOLERANCE:  # Approximately equal
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

            invariant = self.alpha * self.A0**2 - self.beta * self.B0**2
        else:
            # Use general case (fallback to original method)
            return self.analytical_solution(t_max)

        # Create time array
        if t_max is None:
            t_max = max(t_end * self.SIMPLE_TIME_EXTENSION, self.SIMPLE_MINIMUM_TIME)

        t = np.linspace(0, t_max, self.DEFAULT_TIME_POINTS)

        # Use a simplified trajectory generation for equal effectiveness case
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
                    B_t[i] = self.B0 * (1 - progress**self.CURVE_EXPONENT)  # Curved decrease
                    A_t[i] = np.sqrt(max(0, self.A0**2 - (self.B0**2 - B_t[i]**2)))
                elif winner == 'B':
                    # A decreases faster (gets eliminated)
                    A_t[i] = self.A0 * (1 - progress**self.CURVE_EXPONENT)  # Curved decrease
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