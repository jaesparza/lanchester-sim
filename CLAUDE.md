# Claude Instructions for Lanchester Simulation Project

## Git Workflow
- **DO NOT push or commit anything without explicit user approval**
- Always ask before running `git commit` or `git push`
- User will say "commit" or "push" when ready

## Project Overview
Educational Python package for combat simulation models with comprehensive examples:

### Core Models (in `models/`)
- **Lanchester Linear Law** (`lanchester_linear.py`) - Unaimed fire, guerrilla warfare
- **Lanchester Square Law** (`lanchester_square.py`) - Aimed fire, modern ranged combat
- **Salvo Combat Model** (`salvo.py`) - Discrete round-based naval/missile combat
- **ODE Solvers** - Numerical integration versions of Linear and Square Law

### Example Files (in `examples/`)
- **examples.py** - Basic demonstrations of all three models with visualizations
- **extendedLanchester.py** - 8 scenarios comparing Linear vs Square Law
- **extendedSalvo.py** - 8 balanced Salvo scenarios (4 Alpha wins, 4 Bravo wins)
- **extendedPhasesCombat.py** - Multi-phase/turn-based combat with reinforcements

## Code Style
- Well-documented constants at class level
- Helper methods to eliminate duplication
- Comprehensive docstrings with type hints
- Simple, educational implementations over complex optimizations
- ASCII visualizations for terminal output alongside matplotlib plots

## Testing
- Test functionality before suggesting commits
- Verify imports work: `from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship`
- Run example files to verify models work correctly
- Check that matplotlib visualizations display properly

## Examples Guidelines
- Include both ASCII visualizations (terminal) and matplotlib plots (detailed analysis)
- Provide tactical/strategic explanations for each scenario
- Balance outcomes when showing competitive scenarios (not all one-sided)
- Use tables for compact data presentation
- Normalize time scales when comparing different models (0-100% battle progress)
- Include varied seeds for Salvo model to demonstrate different outcomes

## Structure Preferences
- Keep simple - this is a tool, not a complex framework
- Avoid over-engineering
- Maintain educational value and readability
- Examples should be self-contained and runnable