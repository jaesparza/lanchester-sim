"""Numerical solver for Lanchester's Square Law."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._ode_solver_utils import solve_ivp

__all__ = ["SquareODESolution", "LanchesterSquareODESolver"]


@dataclass
class SquareODESolution:
    """Container for the numerical solution of the square law."""

    time: np.ndarray
    force_a: np.ndarray
    force_b: np.ndarray
    winner: str
    t_end: float
    remaining_strength: float

    @property
    def final_strengths(self) -> Tuple[float, float]:
        return float(self.force_a[-1]), float(self.force_b[-1])


class LanchesterSquareODESolver:
    """Numerically integrate Lanchester's Square Law."""

    ZERO_TOLERANCE = 1e-9

    def __init__(self, A0: float, B0: float, alpha: float, beta: float):
        if A0 < 0 or B0 < 0:
            raise ValueError("Initial strengths must be non-negative.")
        if alpha < 0 or beta < 0:
            raise ValueError("Effectiveness coefficients must be non-negative.")

        self.A0 = float(A0)
        self.B0 = float(B0)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _rhs(self, _t: float, y: np.ndarray) -> Tuple[float, float]:
        a, b = y
        a = max(a, 0.0)
        b = max(b, 0.0)
        return (-self.beta * b, -self.alpha * a)

    def _force_zero_event(self, _t: float, y: np.ndarray) -> float:
        return min(y[0], y[1])

    _force_zero_event.terminal = True  # type: ignore[attr-defined]
    _force_zero_event.direction = -1  # type: ignore[attr-defined]

    def _estimate_time_horizon(self) -> float:
        if self.alpha <= 0 and self.beta <= 0:
            return 1.0

        defensive_rate = max(self.beta * self.B0, self.alpha * self.A0, 1e-6)
        horizon = max(self.A0, self.B0) / defensive_rate
        return max(1.0, 5.0 * horizon)

    def solve(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
    ) -> SquareODESolution:
        if t_span is None:
            t_span = (0.0, self._estimate_time_horizon())

        if t_span[1] <= t_span[0]:
            raise ValueError("t_span must have t1 > t0")

        y0 = np.array([self.A0, self.B0], dtype=float)

        if np.allclose(y0, 0.0, atol=self.ZERO_TOLERANCE):
            time = np.linspace(t_span[0], t_span[0], 1)
            zeros = np.zeros_like(time)
            return SquareODESolution(time, zeros, zeros, "Draw", 0.0, 0.0)

        result = solve_ivp(
            fun=self._rhs,
            t_span=t_span,
            y0=y0,
            events=self._force_zero_event,
            dense_output=True,
            max_step=0.05 * t_span[1] if t_span[1] > 0 else np.inf,
        )

        if result.status == 1 and result.t_events[0].size:
            t_end = float(result.t_events[0][0])
        else:
            t_end = float(result.t[-1])

        if num_points <= 2:
            sample_times = np.array(result.t)
        else:
            sample_times = np.linspace(t_span[0], t_end, num_points)

        sol = result.sol(sample_times)
        force_a = np.clip(sol[0], 0.0, None)
        force_b = np.clip(sol[1], 0.0, None)

        final_a = float(force_a[-1])
        final_b = float(force_b[-1])

        if final_a <= self.ZERO_TOLERANCE and final_b <= self.ZERO_TOLERANCE:
            winner = "Draw"
            remaining = 0.0
        elif final_a <= self.ZERO_TOLERANCE:
            winner = "B"
            remaining = final_b
        elif final_b <= self.ZERO_TOLERANCE:
            winner = "A"
            remaining = final_a
        else:
            winner = "Ongoing"
            remaining = max(final_a, final_b)
            if self.alpha <= self.ZERO_TOLERANCE and self.beta <= self.ZERO_TOLERANCE:
                winner = "Draw"

        return SquareODESolution(sample_times, force_a, force_b, winner, t_end, remaining)
