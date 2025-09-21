# Claude Instructions for Lanchester Simulation Project

## Git Workflow
- **DO NOT push or commit anything without explicit user approval**
- Always ask before running `git commit` or `git push`
- User will say "commit" or "push" when ready

## Project Overview
Simple Python package for combat simulation models:
- Lanchester Linear Law (`models/linear.py`)
- Lanchester Square Law (`models/square.py`)
- Salvo Combat Model (`models/salvo.py`)

## Code Style
- Well-documented constants at class level
- Helper methods to eliminate duplication
- Comprehensive docstrings with type hints
- Simple, educational implementations over complex optimizations

## Testing
- Test functionality before suggesting commits
- Verify imports work: `from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship`
- Run examples.py to verify all models work

## Structure Preferences
- Keep simple - this is a tool, not a complex framework
- Avoid over-engineering
- Maintain educational value and readability