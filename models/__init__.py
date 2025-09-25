"""
Lanchester Combat Simulation Models

Simple implementations of:
- Lanchester Linear Law
- Lanchester Square Law
- Salvo Combat Model
"""

from .lanchester_linear import LanchesterLinear
from .lanchester_square import LanchesterSquare
from .odesolver_lanchester_linear import LanchesterLinearODESolver, LinearODESolution
from .odesolver_lanchester_square import LanchesterSquareODESolver, SquareODESolution
from .odesolver_salvo import SalvoODESolver, SalvoODESolution
from .salvo import SalvoCombatModel, Ship

__version__ = "0.1.0"
__all__ = [
    "LanchesterLinear",
    "LanchesterSquare",
    "SalvoCombatModel",
    "Ship",
    "LanchesterLinearODESolver",
    "LinearODESolution",
    "LanchesterSquareODESolver",
    "SquareODESolution",
    "SalvoODESolver",
    "SalvoODESolution",
]