from subprocess import CompletedProcess

import pytest

from tests import test_runner


@pytest.mark.parametrize(
    ("status", "output"),
    [
        (0, "1 passed"),
        (1, "1 failed"),
        (2, "1 error during collection"),
        (3, "INTERNALERROR"),
        (4, "usage error"),
        (5, "no tests ran"),
    ],
)
def test_runner_preserves_pytest_status(status, output, monkeypatch, capsys):
    result = CompletedProcess(["pytest"], status, stdout=output, stderr="")
    monkeypatch.setattr(test_runner, "run_all_tests", lambda: result)
    assert test_runner.main() == status
    report = capsys.readouterr().out
    assert output in report
    assert ("All tests passed!" in report) == (status == 0)
