# Combat simulation models

Scripts for simulating and analyzing military engagements using classical combat models. This repository implements three foundational approaches to quantitative battle analysis, each suited for different combat scenarios and providing unique strategic insights.

 ## Combat Models

  ### **Lanchester Linear Law**
  *Guerrilla warfare and close combat*
  - Models individual duels where combatants engage one-on-one
  - Linear attrition: `dA/dt = -βB`, `dB/dt = -αA`
  - Winner determined by initial advantage: `αA₀ - βB₀`
  - Best for: Hand-to-hand combat, guerrilla tactics, small unit actions

  ### **Lanchester Square Law**
  *Modern ranged combat with concentration effects*
  - Models ranged warfare where all units can engage all enemy units
  - Square law attrition: `dA/dt = -βB·A`, `dB/dt = -αA·B`
  - Quadratic advantage from concentration: Winner ∝ `√(A₀² - B₀²)`
  - Best for: Artillery duels, air combat, modern battlefieldengagements

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
| Linear Law | Hand-to-hand, guerrilla warfare | Winner = αA₀ - βB₀ |
| Square Law | Modern ranged combat | Winner = √(A₀² - B₀²) |
| Salvo Model | Naval/missile combat | Discrete rounds, defensive systems |

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