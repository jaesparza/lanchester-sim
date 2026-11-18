# Linear-law reference and open work

The linear model uses constant loss rates. While both sides are positive its exact trajectories are `A(t) = A₀ − βt` and `B(t) = B₀ − αt`. The linear tests check elimination times and the weighted difference `αA − βB`. The coupled equations and their weighted difference of squares belong to the square model. [Linear tests](../tests/test_linear.py) · [Square tests](../tests/test_square.py)

An open numerical issue affects very short durations. Rescaling both rates by the same factor can turn distinct elimination times into a tie under the current absolute tolerance. Backlog item C5 tracks this issue in both linear implementations. [Improvement backlog](Backlog.md#simulation-correctness)
