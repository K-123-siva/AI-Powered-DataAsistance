"""
Test Runner Script
Runs all tests and generates a report
"""

import subprocess
import sys
import os
from datetime import datetime


def run_tests():
    """Run all tests with pytest"""
    
    print("=" * 70)
    print("🧪 AI Data Science Assistant - Test Suite")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if pytest is installed
    try:
        import pytest
        print("✅ pytest is installed")
    except ImportError:
        print("❌ pytest is not installed")
        print("Installing pytest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    print()
    print("-" * 70)
    print("Running Tests...")
    print("-" * 70)
    print()
    
    # Run pytest with various options
    exit_code = pytest.main([
        "test_fast_app.py",
        "-v",                    # Verbose
        "--tb=short",           # Short traceback
        "--color=yes",          # Colored output
        "-ra",                  # Show summary of all test outcomes
        "--durations=10",       # Show 10 slowest tests
    ])
    
    print()
    print("-" * 70)
    print("Test Summary")
    print("-" * 70)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Some tests failed (exit code: {exit_code})")
    
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return exit_code


def run_specific_tests(test_class=None):
    """Run specific test class"""
    
    if test_class:
        print(f"Running tests for: {test_class}")
        exit_code = pytest.main([
            f"test_fast_app.py::{test_class}",
            "-v",
            "--tb=short"
        ])
    else:
        exit_code = run_tests()
    
    return exit_code


def run_coverage():
    """Run tests with coverage report"""
    
    print("Running tests with coverage...")
    
    try:
        import coverage
        print("✅ coverage is installed")
    except ImportError:
        print("❌ coverage is not installed")
        print("Installing coverage...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])
    
    # Run with coverage
    exit_code = subprocess.run([
        sys.executable, "-m", "pytest",
        "test_fast_app.py",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ]).returncode
    
    if exit_code == 0:
        print("\n✅ Coverage report generated in htmlcov/index.html")
    
    return exit_code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for AI Data Science Assistant")
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    
    args = parser.parse_args()
    
    if args.coverage:
        exit_code = run_coverage()
    elif args.test_class:
        exit_code = run_specific_tests(args.test_class)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
