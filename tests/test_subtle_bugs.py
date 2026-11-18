"""Consistency checks across calculations, final states, and validation paths."""

import numpy as np
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship


def test_linear_law_invariant():
    model = LanchesterLinear(100, 80, 0.5, 0.6)
    result = model.analytical_solution()
    expected = model.alpha * model.A0 - model.beta * model.B0
    assert result["linear_advantage"] == pytest.approx(expected)
    np.testing.assert_allclose(model.alpha * result["A"] - model.beta * result["B"], expected)
    elapsed = np.minimum(result["time"], result["battle_end_time"])
    np.testing.assert_allclose(result["A"], model.A0 - model.beta * elapsed, atol=1e-12)
    np.testing.assert_allclose(result["B"], model.B0 - model.alpha * elapsed, atol=1e-12)


def test_square_law_invariant():
    model = LanchesterSquare(100, 80, 0.01, 0.01)
    result = model.analytical_solution()
    expected = model.alpha * model.A0**2 - model.beta * model.B0**2
    assert result["invariant"] == pytest.approx(expected)
    np.testing.assert_allclose(model.alpha * result["A"]**2 - model.beta * result["B"]**2, expected)


@pytest.mark.parametrize("model", [LanchesterLinear, LanchesterSquare])
def test_battle_end_consistency(model):
    battle = model(100, 80, 0.01, 0.01)
    winner, remaining, third = battle.calculate_battle_outcome()
    end = battle.calculate_battle_end_time(winner, remaining, third) if model is LanchesterSquare else third
    solution = battle.analytical_solution(t_max=end)
    assert solution["time"][-1] == solution["battle_end_time"] == end
    assert solution["winner"] == winner == "A"
    assert solution["A"][-1] == pytest.approx(remaining)
    assert solution["B"][-1] == pytest.approx(0)


def test_salvo_result_bounds_and_damage_accounting():
    side_a = [Ship("A", 10, 0.5, 3)]
    side_b = [Ship("B", 10, 0.3, 3)]
    model = SalvoCombatModel(side_a, side_b, random_seed=42)
    through, _ = model.calculate_salvo_effectiveness(side_a, side_b)
    assert 0 <= through <= 10
    health = side_b[0].current_health
    damage = side_b[0].take_damage(2)
    assert health - side_b[0].current_health == damage == 2


def test_salvo_outcome_matches_survivors():
    side_a = [Ship("A1", 8, 0.3, 3), Ship("A2", 12, 0.4, 5)]
    side_b = [Ship("B1", 6, 0.4, 2), Ship("B2", 10, 0.35, 4)]
    model = SalvoCombatModel(side_a, side_b, random_seed=42)
    outcome = model.run_simulation(quiet=True)
    stats = model.get_battle_statistics()
    a = sum(ship.is_operational() for ship in side_a)
    b = sum(ship.is_operational() for ship in side_b)
    assert (stats["force_a_survivors"], stats["force_b_survivors"]) == (a, b)
    expected = {
        (True, True): "Draw - Both forces have survivors",
        (True, False): "Force A Victory",
        (False, True): "Force B Victory",
        (False, False): "Mutual Annihilation",
    }
    assert outcome == stats["outcome"] == expected[bool(a), bool(b)]


def test_cross_model_consistency():
    linear = LanchesterLinear(100, 80, 0.001, 0.001).analytical_solution()
    square = LanchesterSquare(100, 80, 0.001, 0.001).analytical_solution()
    assert linear["winner"] == square["winner"] == "A"
    assert square["remaining_strength"] > linear["remaining_strength"]


def test_small_positive_linear_inputs_produce_finite_values():
    result = LanchesterLinear(1e-15, 1e-15, 1e-15, 1e-15).analytical_solution()
    assert np.isfinite(result["A"]).all() and np.isfinite(result["B"]).all()


@pytest.mark.parametrize("defense", [0, 0.999999, 1])
def test_ship_accepts_defense_bounds(defense):
    ship = Ship("A", 1e-10, defense, 1_000_000)
    assert ship.current_health == 1_000_000
    assert ship.defensive_probability == defense


def test_ship_rejects_defense_above_one():
    with pytest.raises(ValueError):
        Ship("A", 5, 1.1, 3)
