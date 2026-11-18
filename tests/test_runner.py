"""
Test runner for Lanchester simulation models.

Runs all unit tests and provides summary of results.
Uses pytest for comprehensive test discovery (finds both unittest-style and pytest-style tests).
"""

import subprocess
import sys
import os


def run_all_tests():
    """
    Run all model tests using pytest.

    Returns:
        subprocess.CompletedProcess: Result of pytest execution
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Run pytest with verbose output and capture the results
    result = subprocess.run(
        [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )

    return result


def main():
    """Main test runner entry point."""
    print("=" * 60)
    print("LANCHESTER SIMULATION MODEL TESTS")
    print("=" * 60)
    print("Using pytest for comprehensive test discovery")
    print("=" * 60)
    print()

    result = run_all_tests()

    # Print the full pytest output
    print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\nPytest exited with status {result.returncode}.")
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
