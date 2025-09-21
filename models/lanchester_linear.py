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

        Linear Law: dA/dt = -β·B, dB/dt = -α·A
        Winner determined by who can eliminate opponent first given effectiveness rates.

        Returns:
        tuple: (winner, remaining_strength, t_end)
        """
        # Handle degenerate cases
        if self.alpha == 0 and self.beta == 0:
            return 'Draw', 0, float('inf')
        elif self.alpha == 0:
            return 'A', self.A0, float('inf')  # A can't be damaged
        elif self.beta == 0:
            return 'B', self.B0, float('inf')  # B can't be damaged

        # Calculate time for each force to eliminate the other if unopposed
        # Time for A to eliminate B: B₀ / α (A attacks B at rate α)
        # Time for B to eliminate A: A₀ / β (B attacks A at rate β)
        time_A_eliminates_B = self.B0 / self.alpha
        time_B_eliminates_A = self.A0 / self.beta

        if time_A_eliminates_B < time_B_eliminates_A:
            # A eliminates B first
            winner = 'A'
            t_end = time_A_eliminates_B
            # Calculate A's remaining strength
            # Since A eliminates B in time t_end, and A started with advantage,
            # A must survive. Use ratio-based approximation.
            survival_ratio = 1 - (t_end / time_B_eliminates_A)
            remaining_strength = max(1, self.A0 * survival_ratio)  # At least 1 survivor
        elif time_B_eliminates_A < time_A_eliminates_B:
            # B eliminates A first
            winner = 'B'
            t_end = time_B_eliminates_A
            # Calculate B's remaining strength
            # Since B eliminates A in time t_end, and B started with advantage,
            # B must survive. Use ratio-based approximation.
            survival_ratio = 1 - (t_end / time_A_eliminates_B)
            remaining_strength = max(1, self.B0 * survival_ratio)  # At least 1 survivor
        else:
            # Simultaneous elimination
            winner = 'Draw'
            remaining_strength = 0
            t_end = time_A_eliminates_B  # Same as time_B_eliminates_A

        return winner, remaining_strength, t_end

    def generate_force_trajectories(self, t):
        """
        Generate force strength trajectories over time using Linear Law.

        Linear Law: dA/dt = -β·B(t), dB/dt = -α·A(t)
        Forces decrease based on mutual attrition with effectiveness coefficients.

        Parameters:
        t (array): Time array

        Returns:
        tuple: (A_t, B_t) arrays of force strengths
        """
        # Handle edge cases
        if self.alpha == 0 and self.beta == 0:
            return np.full_like(t, self.A0), np.full_like(t, self.B0)

        # Get battle parameters
        winner, remaining_strength, t_end = self.calculate_battle_outcome()

        # Initialize arrays
        A_t = np.zeros_like(t)
        B_t = np.zeros_like(t)

        for i, time_val in enumerate(t):
            if time_val >= t_end:
                # Battle has ended
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
                # During battle - forces decrease based on differential equations
                # Simplified linear approximation of the differential system

                if winner == 'A':
                    # B is eliminated at t_end, A survives
                    progress = time_val / t_end
                    B_t[i] = self.B0 * (1 - progress)  # B decreases linearly to zero
                    # A decreases based on being attacked by remaining B forces
                    A_losses = self.beta * self.B0 * progress / 2  # Approximate integral
                    A_t[i] = max(0, self.A0 - A_losses)
                elif winner == 'B':
                    # A is eliminated at t_end, B survives
                    progress = time_val / t_end
                    A_t[i] = self.A0 * (1 - progress)  # A decreases linearly to zero
                    # B decreases based on being attacked by remaining A forces
                    B_losses = self.alpha * self.A0 * progress / 2  # Approximate integral
                    B_t[i] = max(0, self.B0 - B_losses)
                else:
                    # Draw: both forces decrease at rates determined by effectiveness
                    progress = time_val / t_end
                    A_t[i] = self.A0 * (1 - progress)
                    B_t[i] = self.B0 * (1 - progress)

        # Ensure non-negative values
        A_t = np.maximum(0, A_t)
        B_t = np.maximum(0, B_t)

        return A_t, B_t

    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Linear Law.

        Uses the Linear Law principle: A(t) - B(t) = A₀ - B₀ (linear advantage preserved)

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
            'effectiveness_ratio': self.alpha / self.beta if self.beta > 0 else float('inf')  # Linear Law: effectiveness matters
        }

    def simple_analytical_solution(self, t_max=None):
        """
        Simplified analytical solution using the key Linear Law insight.

        Returns the same result as analytical_solution since the Linear Law
        is simpler than the Square Law.
        """
        return self.analytical_solution(t_max)

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


if __name__ == "__main__":
    # Example 1: Force A has numerical advantage
    print("Example 1: Numerical Advantage - Force A Superior")
    battle1 = LanchesterLinear(A0=100, B0=60, alpha=0.01, beta=0.01)
    solution1 = battle1.simple_analytical_solution()

    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print(f"Linear Law advantage: A₀ - B₀ = {battle1.A0} - {battle1.B0} = {battle1.A0 - battle1.B0}")
    print()

    # Example 2: Effectiveness vs. numbers
    print("Example 2: Superior Effectiveness vs. Numbers")
    battle2 = LanchesterLinear(A0=80, B0=120, alpha=0.02, beta=0.01)
    solution2 = battle2.analytical_solution()

    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print(f"Linear Law advantage: A₀ - B₀ = {battle2.A0} - {battle2.B0} = {battle2.A0 - battle2.B0}")
    print(f"A's effectiveness: α = {battle2.alpha}")
    print(f"B's effectiveness: β = {battle2.beta}")
    print()

    # Plot both examples
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    battle1.plot_battle(solution=solution1, title="Example 1: Numerical Advantage", ax=plt.gca())

    plt.subplot(1, 2, 2)
    battle2.plot_battle(solution=solution2, title="Example 2: Superior Effectiveness", ax=plt.gca())

    plt.tight_layout()
    plt.show()