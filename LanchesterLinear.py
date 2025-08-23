import numpy as np
import matplotlib.pyplot as plt

class LanchesterLinear:
    """
    Implementation of Lanchester's Linear Law for ancient-style combat.
    
    The Linear Law assumes sequential combat where forces engage one-on-one.
    Combat effectiveness is proportional to force size.
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
        self.A0 = A0
        self.B0 = B0
        self.alpha = alpha
        self.beta = beta
    
    def analytical_solution(self, t_max=None):
        """
        Analytical solution for the Linear Law.
        Forces decrease linearly until one is eliminated.
        
        Returns:
        dict: Contains time arrays, force strengths, battle end time, and winner
        """
        # Calculate when each force would be eliminated
        t_A_eliminated = self.A0 / self.beta if self.beta > 0 else np.inf
        t_B_eliminated = self.B0 / self.alpha if self.alpha > 0 else np.inf
        
        # Battle ends when first force is eliminated
        t_end = min(t_A_eliminated, t_B_eliminated)
        
        # Determine winner
        if t_A_eliminated < t_B_eliminated:
            winner = 'B'
            remaining_strength = self.B0 - self.alpha * t_end
        elif t_B_eliminated < t_A_eliminated:
            winner = 'A' 
            remaining_strength = self.A0 - self.beta * t_end
        else:
            winner = 'Draw'
            remaining_strength = 0
        
        # Create time array
        if t_max is None:
            t_max = min(t_end * 1.2, t_end + 1)  # Show a bit beyond battle end
        
        t = np.linspace(0, t_max, 1000)
        
        # Calculate force strengths over time
        A_t = np.maximum(0, self.A0 - self.beta * t)
        B_t = np.maximum(0, self.B0 - self.alpha * t)
        
        return {
            'time': t,
            'A': A_t,
            'B': B_t,
            'battle_end_time': t_end,
            'winner': winner,
            'remaining_strength': remaining_strength,
            'A_casualties': self.A0 - (self.A0 - self.beta * t_end if winner != 'A' else remaining_strength),
            'B_casualties': self.B0 - (self.B0 - self.alpha * t_end if winner != 'B' else remaining_strength)
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
            solution = self.analytical_solution()
        
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
        
        # Add winner annotation
        if 'winner' in solution and solution['winner'] != 'Ongoing':
            ax.text(0.02, 0.98, f"Winner: {solution['winner']}", 
                    transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
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
    # Example 1: Balanced forces
    print("Example 1: Balanced Forces")
    battle1 = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
    solution1 = battle1.analytical_solution()
    
    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print()
    
    # Example 2: Superior effectiveness
    print("Example 2: Superior Effectiveness")
    battle2 = LanchesterLinear(A0=100, B0=120, alpha=0.8, beta=0.4)
    solution2 = battle2.analytical_solution()
    
    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print()
    
    # Plot both examples using the new method
    LanchesterLinear.plot_multiple_battles(
        battles=[battle1, battle2],
        solutions=[solution1, solution2],
        titles=["Example 1: Balanced Forces", "Example 2: Superior Effectiveness"]
    )