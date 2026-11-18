import math

import pytest

from examples.extendedPhasesCombat import run_phase_simulation


@pytest.mark.parametrize(
    "model, initial, phase, duration, expected",
    [
        ("square", (1, 1), dict(alpha=0.01, beta=0.01), 100, (math.exp(-1), math.exp(-1))),
        ("linear", (3, 1), dict(alpha=1, beta=1, duration=None), 1, (2, 0)),
        ("linear", (3, 2), dict(alpha=1, beta=1, duration=0.5), 0.5, (2.5, 1.5)),
        ("square", (5, 3), dict(alpha=1, beta=1, duration=None), math.log(2), (4, 0)),
    ],
)
def test_phase_endpoint_and_handoff(model, initial, phase, duration, expected):
    phase["B_reinforcements"] = 1
    following = dict(alpha=0, beta=0, duration=2)
    first, second = run_phase_simulation(*initial, [phase, following], model_type=model)

    assert first["duration"] == pytest.approx(duration)
    assert first["solution"]["time"][-1] == first["duration"]
    assert second["cumulative_time_start"] == first["cumulative_time_end"] == first["duration"]
    for side, value in zip(("A", "B"), expected):
        assert first["solution"][side][-1] == pytest.approx(value)
        assert first[f"end_{side}"] == pytest.approx(value + first[f"reinforcements_{side}"])
        assert second[f"start_{side}"] == first[f"end_{side}"]
