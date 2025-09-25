"""Unit tests for the numerical Salvo model solver."""
from __future__ import annotations

import math

from models import SalvoCombatModel, SalvoODESolver, Ship


def test_salvo_solver_tracks_simulation_expectation():
    force_a_solver = [
        Ship("A1", offensive_power=8, defensive_power=0.2, staying_power=5),
        Ship("A2", offensive_power=6, defensive_power=0.25, staying_power=6),
    ]
    force_b_solver = [
        Ship("B1", offensive_power=7, defensive_power=0.15, staying_power=5),
        Ship("B2", offensive_power=5, defensive_power=0.3, staying_power=6),
    ]

    solver = SalvoODESolver(force_a_solver, force_b_solver)
    solution = solver.solve(num_points=400)

    # Run discrete simulation with the same configuration for comparison.
    force_a_sim = [
        Ship("A1", offensive_power=8, defensive_power=0.2, staying_power=5),
        Ship("A2", offensive_power=6, defensive_power=0.25, staying_power=6),
    ]
    force_b_sim = [
        Ship("B1", offensive_power=7, defensive_power=0.15, staying_power=5),
        Ship("B2", offensive_power=5, defensive_power=0.3, staying_power=6),
    ]

    simulation = SalvoCombatModel(force_a_sim, force_b_sim, random_seed=42)
    simulation.run_simulation(max_rounds=10, quiet=True)
    stats = simulation.get_battle_statistics()

    remaining_a_expected = sum(ship.current_health for ship in stats["surviving_ships_a"])
    remaining_b_expected = sum(ship.current_health for ship in stats["surviving_ships_b"])

    final_a, final_b = solution.final_strengths

    assert math.isclose(final_a, remaining_a_expected, rel_tol=0.35, abs_tol=5.0)
    assert math.isclose(final_b, remaining_b_expected, rel_tol=0.35, abs_tol=2.5)
