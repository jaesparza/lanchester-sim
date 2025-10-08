"""Numerical integration routines for Lanchester's Square Law."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["SquareODESolution", "LanchesterSquareODESolver"]


@dataclass
class SquareODESolution:
    """Container holding the numerical solution for the square law."""

    time: np.ndarray
    force_a: np.ndarray
    force_b: np.ndarray
    winner: str
    t_end: float
    remaining_strength: float

    @property
    def final_strengths(self) -> Tuple[float, float]:
        """Return the final strengths of both forces as a tuple."""
        return float(self.force_a[-1]), float(self.force_b[-1])


class LanchesterSquareODESolver:
    """Numerically integrate the Lanchester square-law attrition equations."""

    EFFECTIVENESS_TOLERANCE = 1e-10
    FORCE_ZERO_TOLERANCE = 1e-12
    ZERO_TOLERANCE = 1e-9
    TIME_EXTENSION_FACTOR = 1.2
    TIME_MINIMUM_EXTENSION = 0.5
    DEFAULT_TIME_POINTS = 1000
    DRAW_PREVIEW_WINDOW = 5.0
    LARGE_TIME_THRESHOLD = 1e15
    ARCTANH_CLIP = 1.0 - 1e-12
    PLOT_GRID_ALPHA = 0.3
    PLOT_TEXT_Y_POSITION = 0.98
    PLOT_Y_AXIS_PADDING = 1.1

    def __init__(self, A0: float, B0: float, alpha: float, beta: float) -> None:
        if A0 < 0 or B0 < 0:
            raise ValueError("Initial strengths must be non-negative.")
        if alpha < 0 or beta < 0:
            raise ValueError("Effectiveness coefficients must be non-negative.")

        self.A0 = float(A0)
        self.B0 = float(B0)
        self.alpha = float(alpha)
        self.beta = float(beta)

    # ------------------------------------------------------------------
    # Core ODE integration helpers
    # ------------------------------------------------------------------
    def _rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        """Square-law attrition rates used by the integrator."""
        a, b = y
        return np.array([-self.beta * b, -self.alpha * a], dtype=float)

    def _rk4_step(self, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """Advance the state using a classical fourth-order Runge-Kutta step."""
        k1 = self._rhs(t, y)
        k2 = self._rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self._rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self._rhs(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _prepare_time_array(
        self,
        t_end: float,
        num_points: int,
        sample_times: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if sample_times is not None:
            times = np.asarray(sample_times, dtype=float)
            if times.ndim != 1 or times.size == 0:
                raise ValueError("sample_times must be a one-dimensional array with at least one entry")
            if np.any(times < 0):
                raise ValueError("sample_times must be non-negative")
            if np.any(np.diff(times) < 0):
                raise ValueError("sample_times must be monotonically increasing")
            return times

        if num_points < 2:
            num_points = 2

        if np.isfinite(t_end):
            t_max = max(t_end * self.TIME_EXTENSION_FACTOR, t_end + self.TIME_MINIMUM_EXTENSION)
        else:
            t_max = self.DRAW_PREVIEW_WINDOW

        return np.linspace(0.0, t_max, num_points)

    # ------------------------------------------------------------------
    # Outcome and timing calculations (mirroring the analytical model)
    # ------------------------------------------------------------------
    def calculate_battle_outcome(self) -> Tuple[str, float, float]:
        """Determine winner, remaining strength, and invariant."""
        if self.alpha == 0 and self.beta == 0:
            winner = "Draw"
            remaining_strength = max(self.A0, self.B0)
            invariant = 0.0
        else:
            invariant = self.alpha * self.A0**2 - self.beta * self.B0**2

            if invariant > 0:
                winner = "A"
                remaining_strength = np.sqrt(invariant / self.alpha)
            elif invariant < 0:
                winner = "B"
                remaining_strength = np.sqrt(-invariant / self.beta)
            else:
                winner = "Draw"
                remaining_strength = 0.0

        return winner, remaining_strength, invariant

    def calculate_battle_end_time(
        self,
        winner: str,
        remaining_strength: float,
        invariant: float,
    ) -> float:
        """Estimate the battle end time based on analytical expressions."""
        if winner == "A":
            if self.beta > 0 and self.alpha > 0:
                ratio = np.sqrt(self.beta / self.alpha)
                arg = ratio * self.B0 / self.A0
                arg = np.clip(arg, -self.ARCTANH_CLIP, self.ARCTANH_CLIP)
                t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                if self.alpha > 0 and self.A0 > 0 and self.B0 > 0:
                    t_end = self.B0 / (self.alpha * self.A0)
                else:
                    t_end = 0.0
        elif winner == "B":
            if self.alpha > 0 and self.beta > 0:
                ratio = np.sqrt(self.alpha / self.beta)
                arg = ratio * self.A0 / self.B0
                arg = np.clip(arg, -self.ARCTANH_CLIP, self.ARCTANH_CLIP)
                t_end = (1 / np.sqrt(self.alpha * self.beta)) * np.arctanh(arg)
            else:
                if self.beta > 0 and self.B0 > 0 and self.A0 > 0:
                    t_end = self.A0 / (self.beta * self.B0)
                else:
                    t_end = 0.0
        else:
            if self.alpha > 0 and self.beta > 0 and self.A0 > 0 and self.B0 > 0:
                invariant_tolerance = 1e-10
                if abs(invariant) < invariant_tolerance:
                    t_end = float("inf")
                else:
                    time_A_eliminates_B = self.B0 / (self.alpha * self.A0)
                    time_B_eliminates_A = self.A0 / (self.beta * self.B0)
                    t_end = (time_A_eliminates_B + time_B_eliminates_A) / 2
            else:
                if self.alpha == 0 and self.beta == 0:
                    t_end = float("inf")
                else:
                    t_end = 1.0

        if np.isfinite(t_end) and t_end > self.LARGE_TIME_THRESHOLD:
            t_end = float("inf")

        return t_end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(
        self,
        num_points: int = DEFAULT_TIME_POINTS,
        sample_times: Optional[Sequence[float]] = None,
    ) -> SquareODESolution:
        """Integrate the square-law ODEs and return sampled trajectories."""
        winner, remaining_strength, invariant = self.calculate_battle_outcome()
        analytical_t_end = self.calculate_battle_end_time(winner, remaining_strength, invariant)
        t_end = analytical_t_end

        times = self._prepare_time_array(t_end, num_points, sample_times)
        integration_times = times
        prepend_zero = False
        if integration_times[0] > 0.0:
            integration_times = np.concatenate(([0.0], integration_times))
            prepend_zero = True

        forces = np.zeros((integration_times.size, 2), dtype=float)
        forces[0] = np.array([self.A0, self.B0], dtype=float)

        current_state = forces[0].copy()
        for idx in range(1, integration_times.size):
            dt = integration_times[idx] - integration_times[idx - 1]
            if dt < 0:
                raise ValueError("time array must be monotonically increasing")
            if dt == 0:
                forces[idx] = current_state
                continue

            if np.isfinite(t_end) and integration_times[idx - 1] >= t_end:
                current_state = self._post_battle_state(winner, remaining_strength)
            elif np.isfinite(t_end) and integration_times[idx] >= t_end:
                # Step up to t_end only once to avoid overshoot, then clamp
                remaining_dt = t_end - integration_times[idx - 1]
                if remaining_dt > 0:
                    temp_state = self._rk4_step(integration_times[idx - 1], current_state, remaining_dt)
                    temp_state = np.maximum(temp_state, 0.0)
                else:
                    temp_state = current_state
                current_state = self._post_battle_state(winner, remaining_strength)
            else:
                current_state = self._rk4_step(integration_times[idx - 1], current_state, dt)
                current_state = np.maximum(current_state, 0.0)

            forces[idx] = current_state

        force_a_full = np.clip(forces[:, 0], 0.0, None)
        force_b_full = np.clip(forces[:, 1], 0.0, None)

        if prepend_zero:
            force_a = force_a_full[1:]
            force_b = force_b_full[1:]
        else:
            force_a = force_a_full
            force_b = force_b_full

        # Determine numerical end time based on trajectories (fallback to analytical if necessary)
        elimination_mask = (force_a_full <= self.ZERO_TOLERANCE) | (force_b_full <= self.ZERO_TOLERANCE)
        elimination_indices = np.where(elimination_mask)[0]
        if elimination_indices.size:
            numerical_end = float(integration_times[elimination_indices[0]])
        else:
            numerical_end = float(integration_times[-1])

        return SquareODESolution(times, force_a, force_b, winner, analytical_t_end, remaining_strength)

    def _post_battle_state(self, winner: str, remaining_strength: float) -> np.ndarray:
        if winner == "A":
            return np.array([remaining_strength, 0.0], dtype=float)
        if winner == "B":
            return np.array([0.0, remaining_strength], dtype=float)
        return np.array([0.0, 0.0], dtype=float)

    def generate_force_trajectories(self, t: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return force trajectories sampled at the requested time points."""
        solution = self.solve(sample_times=t)
        return solution.force_a, solution.force_b

    def numerical_solution(
        self,
        t_max: Optional[float] = None,
        num_points: Optional[int] = None,
    ) -> dict:
        """Compute a dictionary matching the analytical solver output format."""
        winner, remaining_strength, invariant = self.calculate_battle_outcome()
        t_end = self.calculate_battle_end_time(winner, remaining_strength, invariant)

        if num_points is None:
            num_points = self.DEFAULT_TIME_POINTS

        if t_max is None:
            if np.isfinite(t_end):
                t_max = max(t_end * self.TIME_EXTENSION_FACTOR, t_end + self.TIME_MINIMUM_EXTENSION)
            else:
                t_max = self.DRAW_PREVIEW_WINDOW

        time = np.linspace(0.0, t_max, num_points)
        force_a, force_b = self.generate_force_trajectories(time)

        if winner == "A":
            A_casualties = self.A0 - remaining_strength
            B_casualties = self.B0
        elif winner == "B":
            A_casualties = self.A0
            B_casualties = self.B0 - remaining_strength
        else:
            if np.isfinite(t_end):
                A_casualties = self.A0
                B_casualties = self.B0
            else:
                A_casualties = 0.0
                B_casualties = 0.0

        return {
            "time": time,
            "A": force_a,
            "B": force_b,
            "battle_end_time": t_end,
            "winner": winner,
            "remaining_strength": remaining_strength,
            "A_casualties": A_casualties,
            "B_casualties": B_casualties,
            "invariant": invariant,
        }

    def plot_battle(self, solution=None, title: str = "Lanchester Square Law (ODE)", ax=None):
        """Plot the battle dynamics using the numerical trajectories."""
        if solution is None:
            solution = self.numerical_solution()

        should_show = ax is None
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        ax.plot(solution["time"], solution["A"], "b-", linewidth=2, label=f"Force A (initial: {self.A0})")
        ax.plot(solution["time"], solution["B"], "r-", linewidth=2, label=f"Force B (initial: {self.B0})")

        if "battle_end_time" in solution and np.isfinite(solution["battle_end_time"]):
            ax.axvline(
                x=solution["battle_end_time"],
                color="gray",
                linestyle="--",
                alpha=0.7,
                label=f"Battle ends: t={solution['battle_end_time']:.2f}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Force Strength")
        ax.set_title(f"{title}\nα={self.alpha}, β={self.beta}")
        ax.legend()
        ax.grid(True, alpha=self.PLOT_GRID_ALPHA)
        ax.set_xlim(0, max(solution["time"]))
        ax.set_ylim(0, max(self.A0, self.B0) * self.PLOT_Y_AXIS_PADDING)

        alpha_advantage = self.alpha * self.A0**2
        beta_advantage = self.beta * self.B0**2
        info_text = (
            f"Winner: {solution['winner']}\n"
            f"Square Law Advantage: α×A₀²={alpha_advantage:.0f} vs β×B₀²={beta_advantage:.0f}"
        )
        ax.text(
            0.02,
            self.PLOT_TEXT_Y_POSITION,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue"),
        )

        if should_show:
            plt.tight_layout()
            plt.show()

    @classmethod
    def plot_multiple_battles(cls, battles, solutions=None, titles=None):
        """Plot multiple battle scenarios in parallel subplots."""
        n_battles = len(battles)
        fig, axes = plt.subplots(1, n_battles, figsize=(6 * n_battles, 6))

        if n_battles == 1:
            axes = [axes]

        for i, battle in enumerate(battles):
            solution = solutions[i] if solutions else None
            title = titles[i] if titles else f"Battle {i + 1}"
            battle.plot_battle(solution=solution, title=title, ax=axes[i])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Example 1: Equal Effectiveness - Size Matters")
    battle1 = LanchesterSquareODESolver(A0=100, B0=60, alpha=0.01, beta=0.01)
    solution1 = battle1.numerical_solution()

    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print(
        "Square Law prediction: "
        f"sqrt({battle1.A0}² - {battle1.B0}²) = sqrt({battle1.A0**2} - {battle1.B0**2}) = "
        f"{np.sqrt(max(battle1.A0**2 - battle1.B0**2, 0)):.1f}"
    )
    print()

    print("Example 2: Superior Effectiveness vs. Numbers")
    battle2 = LanchesterSquareODESolver(A0=80, B0=120, alpha=0.02, beta=0.01)
    solution2 = battle2.numerical_solution()

    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print(f"A's effective strength: α×A₀² = {battle2.alpha}×{battle2.A0}² = {battle2.alpha * battle2.A0**2:.0f}")
    print(f"B's effective strength: β×B₀² = {battle2.beta}×{battle2.B0}² = {battle2.beta * battle2.B0**2:.0f}")
    print(
        "Invariant: "
        f"{solution2['invariant']:.0f} "
        f"({'A wins' if solution2['invariant'] > 0 else 'B wins' if solution2['invariant'] < 0 else 'Draw'})"
    )
    print()

    LanchesterSquareODESolver.plot_multiple_battles(
        battles=[battle1, battle2],
        solutions=[solution1, solution2],
        titles=["Example 1: Equal Effectiveness", "Example 2: Superior Effectiveness"],
    )
