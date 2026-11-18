# Improvement backlog

The Python package in `models/` exposes the linear, square, and salvo models. Two additional modules provide numerical ODE solvers. The scripts in `examples/` exercise these implementations, while `prototypes/PositionPlotting/` contains standalone UI experiments. This backlog prioritizes problems in the current package and its contributor workflow. Longer-term proposals live in `ProjectIdeas.md` and `Roadmap.md`.

**Baseline at `3cf728c`.** On Python 3.13.1 with Matplotlib's Agg backend, pytest reported 251 passed and 23 warnings. Nineteen warnings came from tests returning values and four from attempts to display figures. Both `python examples/examples.py` and `python examples/run_examples.py` completed. The latter ran five analytical-versus-ODE consistency checks.

## Simulation correctness

The items below are ordered by impact on the reported trajectories and results.

- [x] **C1. Preserve exponential decay in the analytical square-law model for exact draws.** With unit initial values and unit rates, both curves reached zero at `t=2`. **Done when** both analytical entry points match `exp(-t)` throughout the requested interval, including after `t=1.5`, and the balanced case with unequal initial values follows the same decay at its own rate. [Square model](../models/lanchester_square.py) · [Tests](../tests/test_square.py)

- [x] **C2. Carry the state at the recorded phase boundary.** Each phase now samples through its recorded duration and carries the final values, plus any reinforcements, into the next phase. Regression tests cover fixed durations, finite model end times, and the 100-unit window for an infinite end time. [Phase example](../examples/extendedPhasesCombat.py) · [Boundary tests](../tests/test_phase_boundaries.py)

- [x] **C3. Make ODE accuracy independent of the requested output spacing.** The square solver bounds its internal RK4 steps and evaluates balanced decay directly. The linear solver stops at the event even when it falls between samples. Both linear APIs agree on the event time and state. The consistency runner checks both trajectories against reference curves on sparse and dense grids. [Sampling tests](../tests/test_ode_sampling.py) · [Consistency runner](../examples/run_examples.py)

- [ ] **C4. Preserve initially empty sides in both square implementations.** For initial values (2, 0) and rates (0, 1), the analytical implementation reports B as the winner and the ODE solver reports a draw. **Done when** both implementations retain (2, 0), report A at time zero, and pass the corresponding case with sides reversed. [Square model](../models/lanchester_square.py) · [Square solver](../models/ode_solver_lanchester_square.py)

- [ ] **C5. Preserve linear outcomes under a change of time units.** Multiplying both unit rates by `1e10` changed the result for (2, 1) to a draw in both implementations. **Done when** rescaling both rates preserves the winner and remaining value and scales the end time inversely. [Linear model](../models/lanchester_linear.py) · [Linear solver](../models/ode_solver_lanchseter_linear.py)

- [ ] **C6. Honor the salvo round limit before advancing state.** A zero-round run executed one round. **Done when** a zero-round budget preserves initial state and history, and a run at its limit cannot execute another round. [Round loop](../models/salvo.py)

- [x] **C7. Count empty inputs consistently in Monte Carlo runs.** The ensemble counts the outcome returned by each single run, including all three empty-input combinations. Regression checks cover counts, percentages, and zero-round durations. [Simulation and ensemble](../models/salvo.py) · [Tests](../tests/test_salvo.py)

## First

- [ ] **1. Make diagnostic findings fail the test suite.** Four test modules collect issues and return lists or other values. For example, an exception in `test_linear_edge_cases` becomes a printable `FAIL` entry, which pytest ignores. Turn the correctness checks into assertions and move informational measurements into helpers or standalone diagnostics. **Done when** a deliberately wrong result fails the corresponding test by name and the suite passes with `-W error::pytest.PytestReturnNotNoneWarning`. [Edge cases](../tests/test_edge_cases.py) · [Mathematical checks](../tests/test_mathematical_correctness.py) · [Performance checks](../tests/test_performance_issues.py) · [Subtle bugs](../tests/test_subtle_bugs.py)

- [ ] **2. Repair the first-run instructions.** The README tells users at the repository root to run `python examples.py`. The script lives in `examples/`, alongside a separate consistency runner. Give each command a one-line purpose and exercise the documented commands in CI with a headless backend. **Done when** someone can follow the README from a fresh checkout and run both examples successfully. [README](../README.md) · [Basic example](../examples/examples.py) · [Consistency runner](../examples/run_examples.py) · [CI workflow](../.github/workflows/tests.yml)

- [ ] **3. Align the model descriptions with the shipped equations.** The linear implementation uses constant rates, while its README entry describes rates proportional to the opposing population. The square entry adds a population factor absent from its implementation. The TODO also cites a skipped test name missing from the suite. **Done when** each README description agrees with its implementation and tests, the roadmap uses the same definitions, and the TODO accurately records any remaining open question. [README](../README.md) · [Linear implementation](../models/lanchester_linear.py) · [Square implementation](../models/lanchester_square.py) · [Linear tests](../tests/test_linear.py) · [Existing TODO](TODO.md)

- [ ] **4. Preserve pytest's exit status in the local runner.** The wrapper infers success by searching stdout for counts. Giving it a pytest result with exit code 5 and output `no tests ran` produced wrapper exit code 0. CI invokes pytest directly, so this affects the local entry point. **Done when** the wrapper returns the subprocess exit code for success, failed tests, collection errors, and empty collection, and only announces success for code 0. [Test runner](../tests/test_runner.py)

## Next

- [ ] **5. Exercise the checked-in JSON fixtures.** The square suite reads its fixture, while the linear and salvo files have no test consumers. Review their expected values and distinguish documentation records from named cases before connecting them to pytest. **Done when** each applicable case is identifiable in test output and changing an expected value in either file causes a named failure. [Square fixture consumer](../tests/test_square.py) · [Linear fixtures](../tests/test_vectors_linear.json) · [Salvo fixtures](../tests/test_vectors_salvo.json)

- [ ] **6. Isolate random state between model instances.** Constructing a seeded `SalvoCombatModel` resets both Python's and NumPy's process-wide random streams. This changes subsequent draws in the caller's notebook and can make tests depend on execution order. **Done when** creating and running a seeded instance preserves the caller's random state, two interleaved seeded instances match their isolated runs, and the existing seeded regression still passes. [Model constructor and clone handling](../models/salvo.py) · [Seeded regression](../tests/test_salvo.py)

- [ ] **7. Make plotting regressions visible in headless tests.** Several plotting tests catch `TypeError` around the entire plotting call and skip on that exception. A broken method signature can therefore look like an unavailable backend. Set the backend before importing plotting code, narrow the skip condition, and mock `show()` where display behavior is under test. **Done when** a plotting `TypeError` fails its test and the headless suite runs without non-interactive display warnings. [Linear plot tests](../tests/test_linear.py) · [Square plot tests](../tests/test_square.py) · [Salvo plot tests](../tests/test_salvo.py) · [Square ODE tests](../tests/test_ode_solver_square.py)

## Later

- [ ] **8. Verify what users get in a wheel.** Package discovery currently includes the top-level `tests` package, and CI only installs the project in editable mode. Limit discovery to the intended package and smoke-test the built wheel from outside the checkout. **Done when** the wheel contains `models`, omits the top-level `tests` package, and supports the four public README imports in a clean environment. [Package configuration](../setup.py) · [Public imports](../models/__init__.py) · [CI workflow](../.github/workflows/tests.yml)

- [ ] **9. Correct the linear ODE module name without breaking imports.** The filename spells Lanchester as `lanchseter`, and the examples and tests import that spelling. Add the correctly spelled module, keep the existing path as a compatibility alias, and migrate in-repo callers. **Done when** both paths resolve to the same solver and solution classes and the consistency runner passes with the corrected import. [Current module](../models/ode_solver_lanchseter_linear.py) · [Solver tests](../tests/test_ode_solver_linear.py) · [Consistency runner](../examples/run_examples.py)
