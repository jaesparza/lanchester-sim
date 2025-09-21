"""
Test runner for Lanchester simulation models.

Runs all unit tests and provides summary of results.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_linear import TestLanchesterLinear
from tests.test_square import TestLanchesterSquare
from tests.test_salvo import TestSalvoCombatModel


def run_all_tests():
    """Run all model tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLanchesterLinear))
    suite.addTests(loader.loadTestsFromTestCase(TestLanchesterSquare))
    suite.addTests(loader.loadTestsFromTestCase(TestSalvoCombatModel))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result


def main():
    """Main test runner entry point."""
    print("=" * 60)
    print("LANCHESTER SIMULATION MODEL TESTS")
    print("=" * 60)
    print()

    result = run_all_tests()

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            # Extract the most relevant error line safely
            lines = traceback.split('\\n')
            if len(lines) >= 2:
                error_line = lines[-2].strip()
            else:
                error_line = "Unknown error"
            print(f"- {test}: {error_line}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())