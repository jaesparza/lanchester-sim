**Generic salvo-model summary**

Model intent.
The salvo model estimates the probability that a grouped firing of $N$ projectiles (shells or missiles) produces at least the required effect on a target. The core math assumes independent per-projectile outcomes unless explicitly modelled otherwise.

Basic independent-hit formula:

```
P(≥1 hit) = 1 - (1-p)^N
```

Multiple-hit (kill requires r hits):

```
P(≥r hits) = \sum_{k=r}^{N} \binom{N}{k} p^k (1-p)^{N-k}
```

Layered interception. Let there be $K$ defensive layers with layer $k$ interception probability $q_k$. Per-projectile survival through defenses:

```
s = \prod_{k=1}^{K} (1-q_k)
```

Adjusted per-projectile hit probability:

```
p' = p_{\text{weapon}} \cdot s
```

Use $p'$ in the formulas above.

Expected kills (simple):

```
E[\text{kills}] = N \cdot p'
```

When independence fails. If hits are correlated use one of:

* Beta-Binomial approximation (overdispersion). PMF:

```
P(X=k) = \binom{N}{k}\frac{B(k+\alpha,\;N-k+\beta)}{B(\alpha,\beta)}
```

with $B$ the Beta function and $\alpha,\beta$ fitted to observed variance.

* Copula or explicit joint failure modes to capture common-mode losses (ECM, decoys, seeker jamming).

* Directly model pairwise correlation with covariance corrections if data permits.

Time-of-arrival and reengagement. If salvo arrivals are spread, allow defenders to re-target between arrivals. Represent time explicitly and compute dynamic interception probabilities $q_k(t)$. Consider using an order-statistic or event sequence Monte Carlo.

Missile-specific adaptations

* let

```
p_{\text{weapon}} = p_{\text{terminal}}(R,\text{ECM},\text{maneuver})
```

be range and state dependent.

* model terminal seekers, salvo-seeking interference, decoys and salvo spacing effects.
* include midcourse attrition from AD systems and SM-2/Patriot style layered effects via $q_k$.

Limitations of the simple salvo formula

1. Overestimates effect if hits are positively correlated.
2. Ignores timing and reengagement windows.
3. Treats target as a single homogeneous sink; not valid for distributed or multi-component targets without aggregation rules.
4. Ignores varying damage per hit. Use damage-state Markov models if partial damage matters.

Recommended extensions (implementation roadmap)

1. Add layered interception: compute $p'$ per projectile as above.
2. Replace independent binomial with Beta-Binomial for overdispersion. Fit $\alpha,\beta$ from data or expert judgement.
3. Introduce time axis and represent salvo arrival times. Compute time-dependent interception and target state updates.
4. Build a Monte Carlo engagement engine. Sample weapon flight, interception events, seeker failures, and target response. Track hits and damage states.
5. Add correlated failure models: common-mode ECM events, sensor blackout windows, or decoy clouds. Implement via copula or grouped random variables.
6. Replace binary hit→kill with conditional damage model: per-hit damage distribution and cumulative damage threshold for kill.
7. Validate against historical engagement data or high-fidelity simulators. Calibrate $p$, $q_k$, and correlation parameters.

Practical tips

* Use the simple formulas for back-of-envelope saturation analysis.
* Use Monte Carlo for operational estimates and sensitivity analysis.
* Report uncertainty bands, not just point estimates.
* Document assumptions for $p$, $q_k$, timing, and correlation. These dominate results.
