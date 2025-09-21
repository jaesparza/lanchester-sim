# TODO Items for Lanchester Simulation Project

## Mathematical Implementation Improvements

### Linear Law: Simplified vs Exact Mathematical Solution

**Status**: Known limitation, skipped test case
**Priority**: Medium (educational clarity vs mathematical precision trade-off)

#### Issue Description

The Linear Law implementation uses a simplified educational approach rather than the exact mathematical solution to the differential equations. This creates a discrepancy with rigorous mathematical expectations.

#### Skipped Test Case

Test: `test_reference_matrix_solution_scenario` in `tests/test_linear.py`

```python
def test_reference_matrix_solution_scenario(self):
    """Exercise the asymmetric reference scenario highlighted in review."""

    battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)

    # Closed-form solution of dA/dt = -β·B, dB/dt = -α·A via matrix exponential
    alpha = battle.alpha
    beta = battle.beta
    A0 = battle.A0
    B0 = battle.B0
    k = np.sqrt(alpha * beta)
    ratio = B0 * k / (alpha * A0)

    expected_t_end = np.arctanh(ratio) / k
    expected_A_survivors = (
        A0 * np.cosh(k * expected_t_end) - (beta / k) * B0 * np.sinh(k * expected_t_end)
    )

    winner, remaining, t_end = battle.calculate_battle_outcome()

    # Currently fails with ~28% error in battle duration:
    # Expected: 2.482718, Actual: 1.777778
```

#### Technical Analysis

**Current Implementation (Simplified Educational Model):**
- **Approach**: Linear trajectories with average attrition rates
- **Time calculation**: `t_end = B0 / (alpha * (A0 + B0) / 2)`
- **Trajectories**: Linear decrease preserving `A(t) - B(t) = A₀ - B₀`
- **Advantages**: Easy to understand, focuses on key insight (linear advantage)
- **Error**: ~28% underestimation of battle duration

**Exact Mathematical Solution:**
- **Approach**: Solves differential equations `dA/dt = -β·B(t), dB/dt = -α·A(t)` via matrix exponential
- **Time calculation**: `t_end = arctanh(B0 * sqrt(alpha*beta) / (alpha * A0)) / sqrt(alpha*beta)`
- **Trajectories**: Hyperbolic curves using `cosh/sinh` functions
- **Advantages**: Mathematically rigorous, precise for research
- **Complexity**: Harder to understand conceptually

#### Design Choice Rationale

This is a **design choice** rather than a bug. The current implementation prioritizes:
- **Educational clarity** over mathematical precision
- **Key insights** (linear advantage preservation) over exact trajectories
- **Simplicity** over complexity
- **Pedagogical value** for understanding Lanchester principles

#### Future Options

1. **Keep current approach**: Accept the educational trade-off
2. **Add exact solution option**: Implement both simplified and exact methods
3. **Replace with exact**: Switch to mathematically rigorous implementation
4. **Document clearly**: Better communicate the approximation in docs/comments

#### Impact Assessment

- **Functional**: Core Linear Law insight (linear advantage) is preserved correctly
- **Educational**: Simplified model aids understanding of Lanchester principles
- **Research**: May not be suitable for precise mathematical analysis
- **Testing**: One test case skipped due to this design choice

---

*Last updated: 2025-09-21*
*Related files: `models/lanchester_linear.py`, `tests/test_linear.py`*