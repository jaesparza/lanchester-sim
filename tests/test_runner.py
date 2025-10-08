"""
Test runner for Lanchester simulation models.

Runs all unit tests and provides summary of results.
Uses pytest for comprehensive test discovery (finds both unittest-style and pytest-style tests).
"""

import subprocess
import sys
import os
import re


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


def parse_pytest_output(output):
    """
    Parse pytest output to extract test statistics.

    Args:
        output (str): The stdout from pytest

    Returns:
        dict: Parsed statistics (passed, failed, errors, total)
    """
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0
    }

    # Look for the summary line like "240 passed, 23 warnings in 2.41s"
    summary_pattern = r'(\d+)\s+passed'
    failed_pattern = r'(\d+)\s+failed'
    error_pattern = r'(\d+)\s+error'
    skipped_pattern = r'(\d+)\s+skipped'

    if match := re.search(summary_pattern, output):
        stats['passed'] = int(match.group(1))

    if match := re.search(failed_pattern, output):
        stats['failed'] = int(match.group(1))

    if match := re.search(error_pattern, output):
        stats['errors'] = int(match.group(1))

    if match := re.search(skipped_pattern, output):
        stats['skipped'] = int(match.group(1))

    stats['total'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']

    return stats


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

    # Parse and display summary
    stats = parse_pytest_output(result.stdout)

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Errors: {stats['errors']}")
    if stats['skipped'] > 0:
        print(f"Skipped: {stats['skipped']}")

    success = stats['failed'] == 0 and stats['errors'] == 0
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())