"""Numerical integration routines for Lanchester's Linear Law."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
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
    ELIMINATION_TIME_ABS_TOL = 1e-9
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

        if initial_state[0] <= self.FORCE_ZERO_TOLERANCE and initial_state[1] <= self.FORCE_ZERO_TOLERANCE:
            zeros = np.zeros(1, dtype=float)
            time = np.zeros(1, dtype=float)
            return LinearODESolution(time, zeros, zeros, "Draw", 0.0, 0.0)

        times = self._prepare_sample_times(t_span, num_points, sample_times)
        forces = np.zeros((times.size, 2), dtype=float)
        forces[0] = initial_state

        current_state = initial_state.copy()

        for idx in range(1, times.size):
            t_prev = times[idx - 1]
            dt = times[idx] - t_prev

            if dt < 0:
                raise ValueError("sample_times must be monotonically increasing")
            if dt == 0:
                forces[idx] = current_state
                continue

            current_state = self._rk4_step(t_prev, current_state, dt)
            current_state = np.maximum(current_state, 0.0)
            forces[idx] = current_state

        force_a = np.clip(forces[:, 0], 0.0, None)
        force_b = np.clip(forces[:, 1], 0.0, None)

        elimination_mask = (force_a <= self.ZERO_TOLERANCE) | (force_b <= self.ZERO_TOLERANCE)
        elimination_indices = np.where(elimination_mask)[0]
        if elimination_indices.size:
            t_end = float(times[elimination_indices[0]])
        else:
            t_end = float(times[-1])

        final_a = float(force_a[-1])
        final_b = float(force_b[-1])

        if final_a <= self.ZERO_TOLERANCE and final_b <= self.ZERO_TOLERANCE:
            winner = "Draw"
            remaining = 0.0 if np.isfinite(t_end) else max(self.A0, self.B0)
        elif final_a <= self.ZERO_TOLERANCE:
            winner = "B"
            remaining = final_b
        elif final_b <= self.ZERO_TOLERANCE:
            winner = "A"
            remaining = final_a
        else:
            if self.alpha <= self.ZERO_TOLERANCE and self.beta <= self.ZERO_TOLERANCE:
                winner = "Draw"
            else:
                winner = "Ongoing"
            remaining = max(final_a, final_b)

        return LinearODESolution(times, force_a, force_b, winner, t_end, remaining)

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

        times_equal = np.isclose(
            t_A_eliminated,
            t_B_eliminated,
            rtol=self.ELIMINATION_TIME_REL_TOL,
            atol=self.ELIMINATION_TIME_ABS_TOL,
        )

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

        if np.isfinite(t_end) and t_end > 1e15:
            t_end = np.inf

        return winner, remaining_strength, t_end

    def generate_force_trajectories(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return force trajectories sampled at the requested time points."""
        time_array = np.asarray(t, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time array must be one-dimensional")
        if time_array.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        if np.any(np.diff(time_array) < 0):
            raise ValueError("time array must be monotonically increasing")

        solution = self.solve(sample_times=time_array)
        force_a = solution.force_a
        force_b = solution.force_b

        if force_a.size < time_array.size:
            pad_length = time_array.size - force_a.size
            pad_a = np.full(pad_length, force_a[-1])
            pad_b = np.full(pad_length, force_b[-1])
            force_a = np.concatenate([force_a, pad_a])
            force_b = np.concatenate([force_b, pad_b])

        return force_a, force_b

    def numerical_solution(self, t_max: Optional[float] = None, num_points: Optional[int] = None):
        """Compute a dictionary matching the analytical solver output format."""
        winner, remaining_strength, t_end = self.calculate_battle_outcome()

        if num_points is None:
            num_points = self.DEFAULT_TIME_POINTS

        if t_max is None:
            if np.isinf(t_end):
                t_max = self.STATIC_PREVIEW_WINDOW
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
