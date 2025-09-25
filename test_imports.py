#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly.
This addresses the import test mentioned in CLAUDE.md.
"""

import sys
import os

# Add the current directory to Python path so models package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models import LanchesterLinear, LanchesterSquare, SalvoCombatModel, Ship
    print("✅ All imports successful!")

    # Test basic functionality
    print("\nTesting basic functionality:")

    # Test LanchesterLinear
    linear = LanchesterLinear(A0=100, B0=80, alpha=0.01, beta=0.01)
    linear_result = linear.calculate_battle_outcome()
    print(f"  Linear: {linear_result[0]} wins with {linear_result[1]:.1f} survivors")

    # Test LanchesterSquare
    square = LanchesterSquare(A0=100, B0=80, alpha=0.01, beta=0.01)
    square_result = square.calculate_battle_outcome()
    print(f"  Square: {square_result[0]} wins with {square_result[1]:.1f} survivors")

    # Test SalvoCombatModel
    force_a = [Ship("Test A", offensive_power=10, defensive_power=0.3, staying_power=3)]
    force_b = [Ship("Test B", offensive_power=8, defensive_power=0.2, staying_power=2)]
    salvo = SalvoCombatModel(force_a, force_b, random_seed=42)
    salvo_result = salvo.simple_simulation(quiet=True)
    winner = salvo_result.get('winner', 'Unknown')
    print(f"  Salvo: {winner}")

    print("\n✅ All models working correctly!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    sys.exit(1)