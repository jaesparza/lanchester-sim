import math
import numpy as np
import matplotlib.pyplot as plt

class LanchesterSquare:
    """
    Implementation of Lanchester's Square Law for modern combat.

    The Square Law assumes ranged combat where each unit can engage multiple enemies.
    Combat effectiveness is proportional to the square of force size.
    """

    # Constants for numerical calculations
    EFFECTIVENESS_TOLERANCE = 1e-10  # Relative tolerance for determining if alpha ≈ beta (equal effectiveness). Avoids floating-point comparison issues.
    TIME_EXTENSION_FACTOR = 1.2      # Extends time arrays 20% beyond battle end for better visualization of final states
    TIME_MINIMUM_EXTENSION = 0.5     # Minimum time extension for very short battles to ensure readable plots
    DEFAULT_TIME_POINTS = 1000       # Number of time points for trajectory calculations. Balances smoothness with performance.
    SIMPLE_TIME_EXTENSION = 1.5      # 50% time extension for simple analytical solution (needs more padding for curved trajectories)
    SIMPLE_MINIMUM_TIME = 2.0        # Minimum visualization time for simple solution to show force dynamics clearly
    SIMPLE_DRAW_PREVIEW = 5.0        # Preview window for exact draw cases where battle time is infinite
    CURVE_EXPONENT = 0.7             # Exponent for curved force decrease in simple solution. Creates realistic non-linear casualty patterns where initial losses are slower, then accelerate.
    LARGE_TIME_THRESHOLD = 1e15      # Times above this are treated as effectively infinite for numerical stability
    ARCTANH_CLIP = 1.0 - 1e-12       # Clamp argument to keep it within open interval (-1, 1)
    
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

        Handles degenerate cases (α=0 or β=0) before invariant calculation.

        Returns:
        tuple: (winner, remaining_strength, invariant)
        """
        # Handle degenerate cases first (one-sided combat)
        if self.alpha == 0 and self.beta > 0:
            # Only B can inflict casualties → B wins
            winner = 'B'
            remaining_strength = self.B0  # B takes no casualties
            invariant = -self.beta * self.B0**2  # Negative since B wins
        elif self.beta == 0 and self.alpha > 0:
            # Only A can inflict casualties → A wins
            winner = 'A'
            remaining_strength = self.A0  # A takes no casualties
            invariant = self.alpha * self.A0**2   # Positive since A wins
        elif self.alpha == 0 and self.beta == 0:
            # No combat effectiveness → Draw (stalemate)
            winner = 'Draw'
            remaining_strength = max(self.A0, self.B0)  # Convention: report larger force
            invariant = 0  # No effective combat power
        else:
            # Normal case: use Square Law invariant
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
            if self.beta > 0 and self.alpha > 0:
                # Correct arctanh formula: (1/√(αβ)) * arctanh(√(β/α) * B₀/A₀)
                ratio = np.sqrt(self.beta / self.alpha)
                arg = ratio * self.B0 / self.A0

                # Check for valid arctanh domain [-1, 1]
                # Numerical noise near the ±1 boundary can push arctanh outside its domain.
                # Clamp the argument into the open interval (-1, 1) so the analytic form stays valid.
                arg = np.clip(arg, -self.ARCTANH_CLIP, self.ARCTANH_CLIP)
                t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                # Degenerate case: β=0 (A wins because B can't damage A)
                # Use limiting integration: t = B₀/(α * A₀)
                if self.alpha > 0 and self.A0 > 0 and self.B0 > 0:
                    t_end = self.B0 / (self.alpha * self.A0)
                else:
                    t_end = 0.0  # Instant victory if B0=0 or A has no combat power
        elif winner == 'B':
            # Time when A is eliminated
            if self.alpha > 0 and self.beta > 0:
                # Correct arctanh formula: (1/√(αβ)) * arctanh(√(α/β) * A₀/B₀)
                ratio = np.sqrt(self.alpha / self.beta)
                arg = ratio * self.A0 / self.B0

                # Check for valid arctanh domain [-1, 1]
                arg = np.clip(arg, -self.ARCTANH_CLIP, self.ARCTANH_CLIP)
                t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                # Degenerate case: α=0 (B wins because A can't damage B)
                # Use limiting integration: t = A₀/(β * B₀)
                if self.beta > 0 and self.B0 > 0 and self.A0 > 0:
                    t_end = self.A0 / (self.beta * self.B0)
                else:
                    t_end = 0.0  # Instant victory if A0=0 or B has no combat power
        else:
            # Draw case: both eliminated simultaneously
            if self.alpha > 0 and self.beta > 0 and self.A0 > 0 and self.B0 > 0:
                # Check if this is an exact draw (invariant ≈ 0)
                # For exact draws, the mathematical solution decays exponentially and never reaches zero
                invariant_tolerance = 1e-10
                if abs(invariant) < invariant_tolerance:
                    t_end = float('inf')  # Exact draw: infinite battle time
                else:
                    # Near-draw case: use averaged elimination times as approximation
                    time_A_eliminates_B = self.B0 / (self.alpha * self.A0)
                    time_B_eliminates_A = self.A0 / (self.beta * self.B0)
                    t_end = (time_A_eliminates_B + time_B_eliminates_A) / 2
            else:
                # Handle degenerate draw case (α=β=0)
                if self.alpha == 0 and self.beta == 0:
                    t_end = float('inf')  # No combat effectiveness - battle never ends
                else:
                    t_end = 1.0  # Fallback for unexpected degenerate case

        # Treat extremely large finite times as infinite for numerical stability
        if np.isfinite(t_end) and t_end > self.LARGE_TIME_THRESHOLD:
            t_end = np.inf

        return t_end

    def _stable_hyperbolic_solution(self, t):
        """Compute exact Square Law trajectories with improved numerical stability.

        Uses log-space formulation for large gamma*t to prevent overflow.
        """
        ld = np.longdouble
        t_ld = t.astype(ld, copy=False)
        alpha_ld = ld(self.alpha)
        beta_ld = ld(self.beta)
        A0_ld = ld(self.A0)
        B0_ld = ld(self.B0)

        gamma = np.sqrt(alpha_ld * beta_ld)

        # Threshold for switching to limiting behavior to prevent overflow and unphysical values
        # Mathematical battle end occurs around gamma*t ≈ 1.1, so threshold should be much higher
        # to allow natural hyperbolic decay without visible discontinuities
        OVERFLOW_THRESHOLD = 1.5

        # Initialize output arrays
        A_exact = np.zeros_like(t, dtype=float)
        B_exact = np.zeros_like(t, dtype=float)

        # Split computation based on gamma*t magnitude
        gamma_t = gamma * t_ld
        safe_mask = gamma_t <= OVERFLOW_THRESHOLD
        overflow_mask = gamma_t > OVERFLOW_THRESHOLD

        # Always use safe computation for t=0 to ensure correct initial values
        t_zero_mask = t_ld == 0
        safe_mask = safe_mask | t_zero_mask
        overflow_mask = overflow_mask & (~t_zero_mask)

        # Safe computation for moderate gamma*t values
        if np.any(safe_mask):
            t_safe = t_ld[safe_mask]
            exp_neg = np.exp(-gamma * t_safe)
            exp_neg = np.clip(exp_neg, np.finfo(np.longdouble).tiny, None)
            exp_pos = 1.0 / exp_neg
            exp_neg_sq = exp_neg * exp_neg

            ratio_ab = np.sqrt(beta_ld / alpha_ld)
            ratio_ba = np.sqrt(alpha_ld / beta_ld)

            k1 = 0.5 * (A0_ld - ratio_ab * B0_ld)
            k2 = 0.5 * (A0_ld + ratio_ab * B0_ld)
            m1 = 0.5 * (B0_ld - ratio_ba * A0_ld)
            m2 = 0.5 * (B0_ld + ratio_ba * A0_ld)

            A_safe_ld = exp_pos * (k1 + k2 * exp_neg_sq)
            B_safe_ld = exp_pos * (m1 + m2 * exp_neg_sq)

            A_exact[safe_mask] = np.asarray(A_safe_ld, dtype=float)
            B_exact[safe_mask] = np.asarray(B_safe_ld, dtype=float)

        # Smooth transition for large gamma*t values to prevent abrupt steps
        if np.any(overflow_mask):
            # Determine final values based on invariant
            invariant = alpha_ld * A0_ld**2 - beta_ld * B0_ld**2

            if invariant > 0:
                # A should win
                final_A = float(np.sqrt(invariant / alpha_ld))
                final_B = 0.0
            elif invariant < 0:
                # B should win
                final_A = 0.0
                final_B = float(np.sqrt(-invariant / beta_ld))
            else:
                # Exact draw
                final_A = 0.0
                final_B = 0.0

            # Create smooth transition to final values to avoid abrupt steps
            gamma_t_overflow = gamma * t_ld[overflow_mask]

            # Find the boundary values at the threshold
            if np.any(safe_mask):
                # Get the last safe values as boundary conditions
                boundary_idx = np.sum(safe_mask) - 1  # Last safe index
                boundary_A = A_exact[safe_mask][boundary_idx] if boundary_idx >= 0 else final_A
                boundary_B = B_exact[safe_mask][boundary_idx] if boundary_idx >= 0 else final_B
            else:
                boundary_A = final_A
                boundary_B = final_B

            # Smooth exponential transition from boundary to final values
            decay_rate = 5.0  # Controls how quickly we approach final values
            transition_progress = np.minimum(1.0, (gamma_t_overflow - OVERFLOW_THRESHOLD) * decay_rate)

            A_exact[overflow_mask] = boundary_A * (1 - transition_progress) + final_A * transition_progress
            B_exact[overflow_mask] = boundary_B * (1 - transition_progress) + final_B * transition_progress

        return A_exact, B_exact

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

        # Handle zero effectiveness edge case
        if self.alpha == 0 and self.beta == 0:
            # No combat effectiveness - forces remain constant
            A_t = np.full_like(t, self.A0)
            B_t = np.full_like(t, self.B0)
            return A_t, B_t

        exact_A = exact_B = None
        if self.alpha > 0 and self.beta > 0:
            if np.isinf(t_end):
                mask = np.ones_like(t, dtype=bool)
            else:
                mask = t < t_end

            if np.any(mask):
                exact_vals_A = np.zeros_like(t)
                exact_vals_B = np.zeros_like(t)
                A_subset, B_subset = self._stable_hyperbolic_solution(t[mask])
                exact_vals_A[mask] = A_subset
                exact_vals_B[mask] = B_subset
                exact_A, exact_B = exact_vals_A, exact_vals_B
            else:
                exact_A = np.zeros_like(t)
                exact_B = np.zeros_like(t)

        for i, time in enumerate(t):
            # Post-battle: maintain final values for times at or after battle end
            if not np.isinf(t_end) and time >= t_end:
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
                # Square Law dynamics: dA/dt = -β*B, dB/dt = -α*A
                if self.alpha > 0 and self.beta > 0:
                    # Use numerically stable hyperbolic closed form solutions
                    A_exact = exact_A[i]
                    B_exact = exact_B[i]

                    # Use exact hyperbolic solution, clamped to physical bounds
                    if winner == 'A':
                        A_t[i] = max(0, min(A_exact, self.A0))
                        B_t[i] = max(0, B_exact)
                    elif winner == 'B':
                        A_t[i] = max(0, A_exact)
                        B_t[i] = max(0, min(B_exact, self.B0))
                    else:
                        # Draw case - use natural solution
                        A_t[i] = max(0, A_exact)
                        B_t[i] = max(0, B_exact)
                else:
                    # Degenerate cases: use proper limiting solutions
                    if self.alpha == 0 and self.beta > 0:
                        # Only B can inflict casualties: dA/dt = -β*B, dB/dt = 0
                        # Solution: A(t) = A₀ - β*B₀*t, B(t) = B₀
                        A_t[i] = max(0, self.A0 - self.beta * self.B0 * time)
                        B_t[i] = self.B0
                    elif self.beta == 0 and self.alpha > 0:
                        # Only A can inflict casualties: dA/dt = 0, dB/dt = -α*A
                        # Solution: A(t) = A₀, B(t) = B₀ - α*A₀*t
                        A_t[i] = self.A0
                        B_t[i] = max(0, self.B0 - self.alpha * self.A0 * time)
                    else:
                        # Both effectiveness coefficients are zero: no combat
                        A_t[i] = self.A0
                        B_t[i] = self.B0

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

        # Check for numerical issues and recalculate if needed
        # Note: infinite t_end is valid for exact draws, so don't recalculate
        if np.isnan(t_end) or t_end <= 0:
            # Recalculate using the corrected method
            t_end = self.calculate_battle_end_time(winner, remaining_strength, invariant)

        # Create time array
        if t_max is None:
            if np.isinf(t_end):
                # For exact draws, use a reasonable finite preview window
                t_max = 5.0  # Show first 5 time units of exponential decay
            else:
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
            # Draw case: check if battle actually ends
            if np.isinf(t_end):
                # Infinite battle time means no casualties occur
                # (zero effectiveness or exact draws with exponential decay)
                A_casualties = 0
                B_casualties = 0
            else:
                # Finite draw: both forces eliminated
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
        if math.isclose(self.alpha, self.beta, rel_tol=self.EFFECTIVENESS_TOLERANCE, abs_tol=0.0):  # Approximately equal
            # Guard against degenerate zero-effectiveness scenarios that would
            # otherwise create divide-by-zero issues in the hyperbolic forms.
            if self.alpha == 0 or self.beta == 0:
                return self.analytical_solution(t_max)

            # Use same outcome calculation as full analytical solution
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

            # Use the same battle end time calculation as the full analytical solution
            t_end = self.calculate_battle_end_time(winner, remaining_strength, invariant)

            # Exact draws (or other degenerate results) return infinite or
            # non-positive battle times. Fall back to the full solution which
            # already handles these cases with finite preview windows.
            if not np.isfinite(t_end) or t_end <= 0:
                return self.analytical_solution(t_max)
        else:
            # Use general case (fallback to original method)
            return self.analytical_solution(t_max)

        # Create time array
        if t_max is None:
            if np.isinf(t_end):
                t_max = max(self.SIMPLE_DRAW_PREVIEW, self.SIMPLE_MINIMUM_TIME)
            else:
                t_max = max(t_end * self.SIMPLE_TIME_EXTENSION, self.SIMPLE_MINIMUM_TIME)

        t = np.linspace(0, t_max, self.DEFAULT_TIME_POINTS)

        # Use exact cosh/sinh solutions even for the "simple" case
        A_t = np.zeros_like(t)
        B_t = np.zeros_like(t)

        exact_A = exact_B = None
        if self.alpha > 0 and self.beta > 0:
            if np.isinf(t_end):
                mask = np.ones_like(t, dtype=bool)
            else:
                mask = t < t_end

            if np.any(mask):
                exact_vals_A = np.zeros_like(t)
                exact_vals_B = np.zeros_like(t)
                A_subset, B_subset = self._stable_hyperbolic_solution(t[mask])
                exact_vals_A[mask] = A_subset
                exact_vals_B[mask] = B_subset
                exact_A, exact_B = exact_vals_A, exact_vals_B
            else:
                exact_A = np.zeros_like(t)
                exact_B = np.zeros_like(t)

        for i, time in enumerate(t):
            if time >= t_end:
                # Post-battle: maintain final values without discontinuous jump
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
                # Use numerically stable hyperbolic solutions up to and including t_end
                A_exact = exact_A[i]
                B_exact = exact_B[i]

                A_t[i] = max(0, A_exact)
                B_t[i] = max(0, B_exact)

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

        # Remember if we need to auto-show the plot
        auto_show = ax is None

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

        # Add winner annotation and correct Square Law advantage calculation
        # Show actual effective combat power: α×A₀² vs β×B₀²
        alpha_advantage = self.alpha * self.A0**2
        beta_advantage = self.beta * self.B0**2
        info_text = f"Winner: {solution['winner']}\nSquare Law Advantage: α×A₀²={alpha_advantage:.0f} vs β×B₀²={beta_advantage:.0f}"
        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

        if auto_show:
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
