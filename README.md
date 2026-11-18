<div align="center">
<img src="images/logo.png" alt="logo" />
  <br>
  <b>Lanchester combat</b> • <b>Naval salvo exchanges</b> • <b>Missile engagements</b>
  <br>
</div>

Scripts for simulating and analyzing military engagements using classical combat models. This repository implements three foundational approaches to quantitative battle analysis, each suited for different combat scenarios and providing unique strategic insights.

## Combat Models

In the two Lanchester models, `A₀` and `B₀` are the starting values. The coefficient `α` governs the loss of B and `β` governs the loss of A. The equations below describe the interval while both values are positive.

### Lanchester Linear Law

Each side loses a constant amount per unit time. The equations are `dA/dt = −β` and `dB/dt = −α`, so the trajectories are straight lines. With positive rates the first side to reach zero sets the stopping time, `min(A₀/β, B₀/α)`.

### Lanchester Square Law

Each side's rate of loss scales with the current value of the other side. The equations are `dA/dt = −βB` and `dB/dt = −αA`. With positive rates and equal weighted starting values (`αA₀² = βB₀²`), both sides decay exponentially toward zero and the model reports an infinite end time.

### **Salvo Combat Model**
  *Naval and missile warfare*
  - Discrete combat rounds with offensive/defensive systems
  - Individual ships with staying power and defensive capabilities
  - Accounts for missile intercepts and multi-hit requirements
  - Best for: Naval battles, missile exchanges, modern precision warfare

## Installation

```bash
# Clone the repository
git clone https://github.com/jaesparza/lanchester-sim.git
cd lanchester-sim

# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

## Quick Start

```python
from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship

# Linear Law (guerrilla warfare)
battle = LanchesterLinear(A0=100, B0=80, alpha=0.5, beta=0.6)
result = battle.simple_analytical_solution()
print(f"Winner: {result['winner']} with {result['remaining_strength']:.1f} survivors")

# Square Law (modern combat)
battle = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
result = battle.simple_analytical_solution()
print(f"Winner: {result['winner']} with {result['remaining_strength']:.1f} survivors")

# Salvo Model (naval/missile combat)
force_a = [Ship("Destroyer", offensive_power=10, defensive_power=0.3, staying_power=5)]
force_b = [Ship("Frigate", offensive_power=8, defensive_power=0.4, staying_power=3)]
simulation = SalvoCombatModel(force_a, force_b, random_seed=42)
result = simulation.run_simulation()
```

## Examples
Run the examples file to see all models in action:

```bash
python examples.py
```

## Key Features
- **Monte Carlo analysis** for statistical outcomes
- **Simple and full simulation modes**
- **Plotting capabilities** with model-specific insights
- **Comparative analysis** between different models

## Model Comparison

| Model | Best For | Key Insight |
|-------|----------|-------------|
| Linear Law | Constant loss rates | The weighted difference `αA − βB` is constant. |
| Square Law | Loss rates proportional to the opposing value | The weighted difference of squares `αA² − βB²` is constant. |
| Salvo Model | Naval/missile combat | Discrete rounds, defensive systems |

For prioritized improvements and their acceptance criteria, see the [improvement backlog](docs/Backlog.md).

## Resources

* [Lanchester's laws](https://en.wikipedia.org/wiki/Lanchester's_laws)
* [Salvo combat model](https://en.wikipedia.org/wiki/Salvo_combat_model)
* [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka–Volterra_equations)
* N. K Jaiswal. 1997, Springer. Military Operations Research: Quantitative Decision Making.
* Helmbold, Robert L. (14 February 1961a). Lanchester Parameters for Some Battles of the Last Two Hundred Years. CORG Staff Paper CORG-SP-122.
* Helmbold, Robert L. (1961b). "Lanchester's Equations, Historical Battles, and War Games". Proceedings of the Eighth Military Operations Research Society Symposium, 18–20 October 1961.
* Alan Washburn, 2000. Lanchester Systems [link](https://faculty.nps.edu/awashburn/Files/Notes/Lanchester.pdf)
* Combat science: the emergence of Operational Research in World War II Erik P. Rau [rau2005]
* The influence of the numerical strength of engaged forces on their casualties. ByM. Osipov Originally Published in the Tzarist Russian Journal MILITARY COLLECTION June-October 1915
Translation of September 1991 by Dr. Robert L. Heimbold and Dr. Allan S. Rehm OFFICE, SPECIAL ASSISTANT FOR MODEL VALIDATION [link](https://web.archive.org/web/20211104093037/https://apps.dtic.mil/dtic/tr/fulltext/u2/a241534.pdf)
