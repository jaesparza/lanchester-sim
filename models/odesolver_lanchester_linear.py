"""Numerical solver for Lanchester's Linear Law without SciPy dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .lanchester_linear import LanchesterLinear

__all__ = ["LinearODESolution", "LanchesterLinearODESolver"]


@dataclass
class LinearODESolution:
    """Container for the numerical solution of the linear law."""

    time: np.ndarray
    force_a: np.ndarray
    force_b: np.ndarray
    winner: str
    t_end: float
    remaining_strength: float

    @property
    def final_strengths(self) -> Tuple[float, float]:
        """Return the final strengths of both forces."""

        return float(self.force_a[-1]), float(self.force_b[-1])


class LanchesterLinearODESolver:
    """Numerically integrate Lanchester's Linear Law."""

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

    def _estimate_time_horizon(self) -> float:
        if self.alpha <= 0 and self.beta <= 0:
            return 1.0

        horizons = []
        if self.beta > 0:
            horizons.append(self.A0 / self.beta)
        if self.alpha > 0:
            horizons.append(self.B0 / self.alpha)
        horizon = max(horizons) if horizons else 1.0
        return max(1.0, 1.5 * horizon)

    def solve(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
    ) -> LinearODESolution:
        if num_points <= 0:
            raise ValueError("num_points must be positive")

        if t_span is None:
            t_span = (0.0, self._estimate_time_horizon())

        if t_span[1] <= t_span[0]:
            raise ValueError("t_span must have t1 > t0")

        analytic = LanchesterLinear(self.A0, self.B0, self.alpha, self.beta)
        winner, remaining_strength, analytic_t_end = analytic.calculate_battle_outcome()

        if np.isfinite(analytic_t_end):
            t_end = float(analytic_t_end)
        else:
            # Infinite battles occur only when neither side can inflict casualties.
            t_end = float(t_span[1])

        sample_end = min(float(t_span[1]), t_end)
        if num_points == 1:
            sample_times = np.array([t_span[0]])
        else:
            sample_times = np.linspace(t_span[0], sample_end, num_points)

        elapsed = sample_times - t_span[0]
        force_a = np.clip(self.A0 - self.beta * elapsed, 0.0, None)
        force_b = np.clip(self.B0 - self.alpha * elapsed, 0.0, None)

        final_a = float(force_a[-1])
        final_b = float(force_b[-1])

        if final_a <= self.ZERO_TOLERANCE and final_b <= self.ZERO_TOLERANCE:
            winner_name = "Draw"
            remaining = 0.0
        elif final_a <= self.ZERO_TOLERANCE:
            winner_name = "B"
            remaining = final_b
        elif final_b <= self.ZERO_TOLERANCE:
            winner_name = "A"
            remaining = final_a
        else:
            if np.isfinite(analytic_t_end) and analytic_t_end <= sample_end + self.ZERO_TOLERANCE:
                winner_name = winner
                remaining = remaining_strength
            elif self.alpha <= self.ZERO_TOLERANCE and self.beta <= self.ZERO_TOLERANCE:
                winner_name = "Draw"
                remaining = max(final_a, final_b)
            else:
                winner_name = "Ongoing"
                remaining = max(final_a, final_b)

        return LinearODESolution(sample_times, force_a, force_b, winner_name, t_end, remaining)
