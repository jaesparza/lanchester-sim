"""Fallback utilities for solving simple ODE systems without SciPy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - SciPy is optional
    from scipy.integrate import solve_ivp as _scipy_solve_ivp  # type: ignore
except Exception:  # pragma: no cover - SciPy not available
    _scipy_solve_ivp = None


Number = Union[float, np.floating]
Vector = Sequence[Number]
EventFunction = Callable[[float, np.ndarray], float]


@dataclass
class SimpleIVPResult:
    """Container mimicking :func:`scipy.integrate.solve_ivp` results."""

    t: np.ndarray
    y: np.ndarray
    status: int
    t_events: List[np.ndarray]
    sol: Optional[Callable[[Union[float, np.ndarray]], np.ndarray]]


class _SimpleDenseOutput:
    """Linear interpolant used when SciPy is unavailable."""

    def __init__(self, t: np.ndarray, y: np.ndarray) -> None:
        self._t = t
        self._y = y

    def __call__(self, t_eval: Union[float, np.ndarray]) -> np.ndarray:
        query = np.atleast_1d(t_eval)
        query = np.clip(query, self._t[0], self._t[-1])
        values = np.vstack([
            np.interp(query, self._t, self._y[i], left=self._y[i, 0], right=self._y[i, -1])
            for i in range(self._y.shape[0])
        ])
        if np.isscalar(t_eval):
            return values[:, 0]
        return values


def _prepare_events(events: Optional[Union[EventFunction, Sequence[EventFunction]]]) -> List[EventFunction]:
    if events is None:
        return []
    if isinstance(events, (list, tuple)):
        return [event for event in events if event is not None]
    return [events]


def _compute_step(
    remaining: float, y: np.ndarray, dy: np.ndarray, max_step: float
) -> float:
    if remaining <= 0:
        return 0.0

    step = remaining / 1000.0
    step = max(step, 1e-3)

    if np.isfinite(max_step):
        step = min(step, max_step)

    rate = float(np.max(np.abs(dy)))
    if rate > 0.0:
        scale = float(np.max(np.abs(y)))
        scale = max(scale, 1.0)
        adaptive = 0.3 * scale / rate
        step = min(step, adaptive)

    return max(step, 1e-6)


def _basic_solve_ivp(
    fun: Callable[[float, np.ndarray], Sequence[float]],
    t_span: Tuple[float, float],
    y0: Vector,
    events: Optional[Union[EventFunction, Sequence[EventFunction]]],
    dense_output: bool,
    max_step: float,
) -> SimpleIVPResult:
    t0, tf = t_span
    y = np.asarray(y0, dtype=float)
    times: List[float] = [t0]
    values: List[np.ndarray] = [y.copy()]

    event_functions = _prepare_events(events)
    event_values_prev = None
    t_events: List[np.ndarray] = [np.array([], dtype=float) for _ in event_functions]

    if event_functions:
        event_values_prev = np.array([
            float(event(t0, y.copy())) for event in event_functions
        ])

    t = t0
    status = 0
    max_iterations = 500000
    iteration = 0

    while t < tf and iteration < max_iterations:
        dy = np.asarray(fun(t, y.copy()), dtype=float)
        remaining = tf - t
        step = _compute_step(remaining, y, dy, max_step)
        step = min(step, remaining)
        if step <= 0.0:
            break

        k1 = dy
        k2 = np.asarray(fun(t + 0.5 * step, y + 0.5 * step * k1), dtype=float)
        k3 = np.asarray(fun(t + 0.5 * step, y + 0.5 * step * k2), dtype=float)
        k4 = np.asarray(fun(t + step, y + step * k3), dtype=float)
        y_new = y + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_new = t + step

        if event_functions and event_values_prev is not None:
            event_values_new = np.array([
                float(event(t_new, y_new.copy())) for event in event_functions
            ])

            triggered_index = None
            for idx, (prev, curr) in enumerate(zip(event_values_prev, event_values_new)):
                if prev > 0.0 and curr <= 0.0:
                    theta = prev / (prev - curr) if prev != curr else 1.0
                    theta = float(np.clip(theta, 0.0, 1.0))
                    t_event = t + theta * step
                    y_event = y + theta * (y_new - y)
                    times.append(t_event)
                    values.append(np.clip(y_event, 0.0, None))
                    t_events[idx] = np.array([t_event], dtype=float)
                    status = 1
                    triggered_index = idx
                    break

            if status == 1:
                break

            event_values_prev = event_values_new

        times.append(t_new)
        y = np.clip(y_new, 0.0, None)
        values.append(y)
        t = t_new
        iteration += 1

        if np.max(np.abs(dy)) < 1e-10 and np.max(np.abs(y_new - y)) < 1e-12:
            break

    if iteration >= max_iterations:
        status = -1

    t_array = np.asarray(times, dtype=float)
    y_array = np.vstack(values).T
    sol = _SimpleDenseOutput(t_array, y_array) if dense_output else None

    if not event_functions:
        t_events = []

    return SimpleIVPResult(t_array, y_array, status, t_events, sol)


def solve_ivp(
    fun: Callable[[float, np.ndarray], Sequence[float]],
    t_span: Tuple[float, float],
    y0: Vector,
    events: Optional[Union[EventFunction, Sequence[EventFunction]]] = None,
    dense_output: bool = False,
    max_step: float = np.inf,
) -> SimpleIVPResult:
    """Solve an initial value problem, falling back to a lightweight RK4 integrator."""

    if _scipy_solve_ivp is not None:  # pragma: no cover - exercised only when SciPy is available
        return _scipy_solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            events=events,
            dense_output=dense_output,
            max_step=None if np.isinf(max_step) else max_step,
        )

    return _basic_solve_ivp(fun, t_span, y0, events, dense_output, max_step)
