"""Continuous approximation of the Salvo combat model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .salvo import Ship
from ._ode_solver_utils import solve_ivp

__all__ = ["SalvoODESolution", "SalvoODESolver"]


@dataclass
class SalvoODESolution:
    """Container for the numerical solution of the continuous Salvo model."""

    time: np.ndarray
    staying_power_a: np.ndarray
    staying_power_b: np.ndarray
    winner: str
    t_end: float
    remaining_strength: float

    @property
    def final_strengths(self) -> Tuple[float, float]:
        return float(self.staying_power_a[-1]), float(self.staying_power_b[-1])


class SalvoODESolver:
    """Continuous approximation of the Salvo combat dynamics."""

    ZERO_TOLERANCE = 1e-6

    def __init__(self, force_a: Iterable[Ship], force_b: Iterable[Ship]):
        self.force_a = tuple(force_a)
        self.force_b = tuple(force_b)

        self.total_offense_a = sum(ship.offensive_power for ship in self.force_a)
        self.total_offense_b = sum(ship.offensive_power for ship in self.force_b)
        self.avg_defense_a = np.mean([ship.defensive_power for ship in self.force_a]) if self.force_a else 0.0
        self.avg_defense_b = np.mean([ship.defensive_power for ship in self.force_b]) if self.force_b else 0.0
        self.total_staying_a = float(sum(ship.staying_power for ship in self.force_a))
        self.total_staying_b = float(sum(ship.staying_power for ship in self.force_b))

    def _effective_offense(self, remaining: float, total_offense: float, total_staying: float) -> float:
        if total_staying <= 0:
            return 0.0
        ratio = np.clip(remaining / total_staying, 0.0, 1.0)
        return total_offense * ratio

    def _effective_defense(self, remaining: float, avg_defense: float, total_staying: float) -> float:
        if total_staying <= 0:
            return 0.0
        ratio = np.clip(remaining / total_staying, 0.0, 1.0)
        return float(np.clip(avg_defense * np.sqrt(ratio), 0.0, 0.95))

    def _rhs(self, _t: float, y: np.ndarray) -> Tuple[float, float]:
        a, b = y
        a = max(a, 0.0)
        b = max(b, 0.0)

        eff_offense_a = self._effective_offense(a, self.total_offense_a, self.total_staying_a)
        eff_offense_b = self._effective_offense(b, self.total_offense_b, self.total_staying_b)
        eff_defense_a = self._effective_defense(a, self.avg_defense_a, self.total_staying_a)
        eff_defense_b = self._effective_defense(b, self.avg_defense_b, self.total_staying_b)

        damage_to_a = eff_offense_b * (1.0 - eff_defense_a)
        damage_to_b = eff_offense_a * (1.0 - eff_defense_b)

        return (-damage_to_a, -damage_to_b)

    def _force_zero_event(self, _t: float, y: np.ndarray) -> float:
        return min(y[0], y[1])

    _force_zero_event.terminal = True  # type: ignore[attr-defined]
    _force_zero_event.direction = -1  # type: ignore[attr-defined]

    def _estimate_time_horizon(self) -> float:
        base_rate = max(
            self.total_offense_a * max(1.0 - self.avg_defense_b, 0.05),
            self.total_offense_b * max(1.0 - self.avg_defense_a, 0.05),
            1e-3,
        )
        total_staying = max(self.total_staying_a, self.total_staying_b, 1.0)
        return max(1.0, 3.0 * total_staying / base_rate)

    def solve(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
    ) -> SalvoODESolution:
        if t_span is None:
            t_span = (0.0, self._estimate_time_horizon())

        if t_span[1] <= t_span[0]:
            raise ValueError("t_span must have t1 > t0")

        y0 = np.array([self.total_staying_a, self.total_staying_b], dtype=float)

        if np.allclose(y0, 0.0, atol=self.ZERO_TOLERANCE):
            time = np.linspace(t_span[0], t_span[0], 1)
            zeros = np.zeros_like(time)
            return SalvoODESolution(time, zeros, zeros, "Draw", 0.0, 0.0)

        result = solve_ivp(
            fun=self._rhs,
            t_span=t_span,
            y0=y0,
            events=self._force_zero_event,
            dense_output=True,
            max_step=0.1 * t_span[1] if t_span[1] > 0 else np.inf,
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
        staying_a = np.clip(sol[0], 0.0, None)
        staying_b = np.clip(sol[1], 0.0, None)

        final_a = float(staying_a[-1])
        final_b = float(staying_b[-1])

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
            if self.total_offense_a <= self.ZERO_TOLERANCE and self.total_offense_b <= self.ZERO_TOLERANCE:
                winner = "Draw"

        return SalvoODESolution(sample_times, staying_a, staying_b, winner, t_end, remaining)
