"""
Lanchester Combat Simulation Models

Simple implementations of:
- Lanchester Linear Law
- Lanchester Square Law
- Salvo Combat Model
"""

from .linear import LanchesterLinear
from .square import LanchesterSquare
from .salvo import SalvoCombatModel, Ship

__version__ = "0.1.0"
__all__ = ["LanchesterLinear", "LanchesterSquare", "SalvoCombatModel", "Ship"]