"""Numerical integration routines for Lanchester's Linear Law."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import math

import numpy as np

__all__ = ["LinearODESolution", "LanchesterLinearODESolver"]


@dataclass
class LinearODESolution:
    """Container holding the numerical solution for the linear law."""

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


class LanchesterLinearODESolver:
    """Numerically integrate the Lanchester linear law system."""

    FORCE_ZERO_TOLERANCE = 1e-12
    ZERO_TOLERANCE = 1e-9
    ELIMINATION_TIME_REL_TOL = 1e-12
    TIME_EXTENSION_FACTOR = 1.2
    TIME_MINIMUM_EXTENSION = 1.0
    DEFAULT_TIME_POINTS = 1000
    STATIC_PREVIEW_WINDOW = 10.0
    MIN_DURATION = 1e-6
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

    def _rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        """Linear-law attrition rates used by the integrator."""
        a, b = y
        active = (a > self.ZERO_TOLERANCE) and (b > self.ZERO_TOLERANCE)
        da_dt = -self.beta if active and self.beta > 0 else 0.0
        db_dt = -self.alpha if active and self.alpha > 0 else 0.0
        return np.array([da_dt, db_dt], dtype=float)

    def _rk4_step(self, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """Advance the state using a classical fourth-order Runge-Kutta step."""
        k1 = self._rhs(t, y)
        k2 = self._rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self._rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self._rhs(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _compute_elimination_time(self) -> float:
        """Return the first time at which either force would be eliminated analytically."""
        times = []
        if self.beta > 0:
            times.append(self.A0 / self.beta)
        if self.alpha > 0:
            times.append(self.B0 / self.alpha)

        if not times:
            return np.inf

        positive_times = [t for t in times if t > 0]
        if not positive_times:
            return 0.0
        return min(positive_times)

    def _prepare_sample_times(
        self,
        t_span: Optional[Tuple[float, float]],
        num_points: int,
        sample_times: Optional[np.ndarray],
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

        if t_span is not None:
            if len(t_span) != 2:
                raise ValueError("t_span must be a tuple of (t0, t1)")
            t0, t1 = float(t_span[0]), float(t_span[1])
            if t1 <= t0:
                raise ValueError("t_span must have t1 > t0")
        else:
            elimination_candidate = self._compute_elimination_time()
            if np.isfinite(elimination_candidate):
                t0, t1 = 0.0, max(elimination_candidate, self.MIN_DURATION)
            else:
                t0, t1 = 0.0, self.STATIC_PREVIEW_WINDOW

        return np.linspace(t0, t1, num_points)

    def solve(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
        sample_times: Optional[np.ndarray] = None,
    ) -> LinearODESolution:
        """Integrate the linear-law ODEs and return sampled trajectories."""
        initial_state = np.array([self.A0, self.B0], dtype=float)

        if (sample_times is None and t_span is None
                and np.all(initial_state <= self.FORCE_ZERO_TOLERANCE)):
            zeros = np.zeros(1, dtype=float)
            time = np.zeros(1, dtype=float)
            return LinearODESolution(time, zeros, zeros, "Draw", 0.0, 0.0)

        times = self._prepare_sample_times(t_span, num_points, sample_times)
        winner, remaining, t_end = self.calculate_battle_outcome()
        terminal_state = np.array([
            remaining if winner == "A" else 0.0,
            remaining if winner == "B" else 0.0,
        ])
        forces = np.empty((times.size, 2), dtype=float)
        state = initial_state
        previous = 0.0
        for idx, time in enumerate(times):
            if time >= t_end:
                forces[idx:] = terminal_state
                break
            if time != previous:
                state = self._rk4_step(previous, state, time - previous)
            forces[idx] = state
            previous = time

        return LinearODESolution(times, forces[:, 0], forces[:, 1], winner, t_end, remaining)

    def calculate_battle_outcome(self) -> Tuple[str, float, float]:
        """Replicate the analytical outcome logic used by the linear model."""
        a_initially_depleted = abs(self.A0) <= self.FORCE_ZERO_TOLERANCE
        b_initially_depleted = abs(self.B0) <= self.FORCE_ZERO_TOLERANCE

        if a_initially_depleted and b_initially_depleted:
            return "Draw", 0.0, 0.0
        if a_initially_depleted:
            return "B", self.B0, 0.0
        if b_initially_depleted:
            return "A", self.A0, 0.0

        t_A_eliminated = self.A0 / self.beta if self.beta > 0 else np.inf
        t_B_eliminated = self.B0 / self.alpha if self.alpha > 0 else np.inf

        # A relative tolerance keeps the comparison independent of the time unit.
        times_equal = math.isclose(t_A_eliminated, t_B_eliminated,
                                   rel_tol=self.ELIMINATION_TIME_REL_TOL)

        if times_equal:
            t_end = max(t_A_eliminated, t_B_eliminated)
            winner = "Draw"
            remaining_strength = max(self.A0, self.B0) if np.isinf(t_end) else 0.0
        elif t_A_eliminated < t_B_eliminated:
            t_end = t_A_eliminated
            winner = "B"
            remaining_strength = max(0.0, self.B0 - self.alpha * t_end)
        else:
            t_end = t_B_eliminated
            winner = "A"
            remaining_strength = max(0.0, self.A0 - self.beta * t_end)

        return winner, remaining_strength, t_end

    def generate_force_trajectories(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return force trajectories sampled at the requested time points."""
        time_array = np.asarray(t, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time array must be one-dimensional")
        if time_array.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if np.any(time_array < 0):
            raise ValueError("time array must be non-negative")
        if np.any(np.diff(time_array) < 0):
            raise ValueError("time array must be monotonically increasing")

        solution = self.solve(sample_times=time_array)
        return solution.force_a, solution.force_b

    def numerical_solution(self, t_max: Optional[float] = None, num_points: Optional[int] = None):
        """Compute a dictionary matching the analytical solver output format."""
        winner, remaining_strength, t_end = self.calculate_battle_outcome()

        if num_points is None:
            num_points = self.DEFAULT_TIME_POINTS

        if t_max is None:
            if np.isinf(t_end):
                t_max = self.STATIC_PREVIEW_WINDOW
            elif t_end > np.finfo(float).max / self.TIME_EXTENSION_FACTOR:
                t_max = t_end
            else:
                t_max = max(t_end * self.TIME_EXTENSION_FACTOR, t_end + self.TIME_MINIMUM_EXTENSION)

        time = np.linspace(0.0, t_max, num_points)
        force_a, force_b = self.generate_force_trajectories(time)

        if winner == "A":
            A_casualties = self.A0 - remaining_strength
            B_casualties = self.B0
        elif winner == "B":
            A_casualties = self.A0
            B_casualties = self.B0 - remaining_strength
        else:
            if np.isinf(t_end):
                A_casualties = 0.0
                B_casualties = 0.0
            else:
                A_casualties = self.A0
                B_casualties = self.B0

        return {
            "time": time,
            "A": force_a,
            "B": force_b,
            "battle_end_time": t_end,
            "winner": winner,
            "remaining_strength": remaining_strength,
            "A_casualties": A_casualties,
            "B_casualties": B_casualties,
            "linear_advantage": self.alpha * self.A0 - self.beta * self.B0,
        }

    def plot_battle(self, solution=None, title: str = "Lanchester Linear Law (ODE)", ax=None):
        """Plot the battle dynamics using the numerical trajectories."""
        if solution is None:
            solution = self.numerical_solution()

        should_show = ax is None
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        ax.plot(solution["time"], solution["A"], "b-", linewidth=2, label=f"Force A (initial: {self.A0})")
        ax.plot(solution["time"], solution["B"], "r-", linewidth=2, label=f"Force B (initial: {self.B0})")

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

        linear_advantage = self.alpha * self.A0 - self.beta * self.B0
        info_text = (
            "Winner: "
            f"{solution['winner']}\nLinear Law Advantage: α{self.A0:.0f} - β{self.B0:.0f} = {linear_advantage:.2f}"
        )
        ax.text(
            0.02,
            self.PLOT_TEXT_Y_POSITION,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen"),
        )

        if should_show:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    print("Example 1: Numerical Advantage - Force A Superior")
    battle1 = LanchesterLinearODESolver(A0=100, B0=60, alpha=0.01, beta=0.01)
    solution1 = battle1.numerical_solution()

    print(f"Battle ends at t = {solution1['battle_end_time']:.2f}")
    print(f"Winner: {solution1['winner']} with {solution1['remaining_strength']:.1f} units remaining")
    print(f"Force A casualties: {solution1['A_casualties']:.1f}")
    print(f"Force B casualties: {solution1['B_casualties']:.1f}")
    print(
        "Linear Law advantage: "
        f"αA₀ - βB₀ = {battle1.alpha}×{battle1.A0} - {battle1.beta}×{battle1.B0} = {battle1.alpha * battle1.A0 - battle1.beta * battle1.B0:.2f}"
    )
    print()

    print("Example 2: Superior Effectiveness vs. Numbers")
    battle2 = LanchesterLinearODESolver(A0=80, B0=120, alpha=0.02, beta=0.01)
    solution2 = battle2.numerical_solution()

    print(f"Battle ends at t = {solution2['battle_end_time']:.2f}")
    print(f"Winner: {solution2['winner']} with {solution2['remaining_strength']:.1f} units remaining")
    print(
        "Linear Law advantage: "
        f"αA₀ - βB₀ = {battle2.alpha}×{battle2.A0} - {battle2.beta}×{battle2.B0} = {battle2.alpha * battle2.A0 - battle2.beta * battle2.B0:.2f}"
    )
    print(f"A's effectiveness: α = {battle2.alpha}")
    print(f"B's effectiveness: β = {battle2.beta}")
    print()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    battle1.plot_battle(solution=solution1, title="Example 1: Numerical Advantage", ax=plt.gca())

    plt.subplot(1, 2, 2)
    battle2.plot_battle(solution=solution2, title="Example 2: Superior Effectiveness", ax=plt.gca())

    plt.tight_layout()
    plt.show()
