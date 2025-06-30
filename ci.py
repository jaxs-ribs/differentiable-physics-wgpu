#!/usr/bin/env python3
"""Simple CI test dispatcher for the physics engine.

Basic Usage:
    python ci.py              # Run unit + integration tests (default)
    python ci.py --unit       # Run only unit tests
    python ci.py --integration # Run only integration tests  
    python ci.py --benchmarks # Run benchmarks
    python ci.py --debug      # Run debugging tests
    python ci.py --all        # Run everything including benchmarks
    python ci.py --quick      # Run minimal test set for quick feedback

Examples:
    python ci.py                    # Standard test suite before commit
    python ci.py --quick            # Quick smoke test during development
    python ci.py --unit --debug     # Run unit tests + debugging tools
    python ci.py --all              # Full test suite before merge
    python ci.py --integration -v   # Verbose integration tests only
    
Common Workflows:
    During development:  python ci.py --quick
    Before commit:       python ci.py
    Before merge:        python ci.py --all
    Debug an issue:      python ci.py --debug
    Performance check:   python ci.py --benchmarks
"""
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add paths for physics_core and tinygrad
physics_core_path = os.path.dirname(os.path.abspath(__file__))
tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")

# Add to Python path
if physics_core_path not in sys.path:
    sys.path.insert(0, physics_core_path)
if os.path.exists(tinygrad_path) and tinygrad_path not in sys.path:
    sys.path.insert(0, tinygrad_path)

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")

def run_pytest(test_path, args=""):
    """Run pytest on a directory or file."""
    # Use 'python' in conda, 'python3' otherwise
    python_cmd = "python" if os.environ.get("CONDA_DEFAULT_ENV") else "python3"
    cmd = f"{python_cmd} -m pytest {args} {test_path}"
    print(f"Running: {cmd}")
    
    # Set up environment with proper paths
    env = os.environ.copy()
    physics_core_path = os.path.dirname(os.path.abspath(__file__))
    tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")
    
    # Add to PYTHONPATH
    python_path = env.get('PYTHONPATH', '')
    paths_to_add = [physics_core_path, tinygrad_path]
    for path in paths_to_add:
        if path not in python_path:
            python_path = f"{path}:{python_path}" if python_path else path
    env['PYTHONPATH'] = python_path
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, env=env)
    elapsed = time.time() - start_time
    
    return result.returncode == 0, elapsed

def run_tests(test_type, test_path, quick_mode=False):
    """Run a set of tests and report results."""
    print_header(f"Running {test_type} Tests")
    
    # Use appropriate pytest args
    args = "-v -x"  # verbose, stop on first failure
    if os.environ.get('CI') == 'true':
        args += " --tb=short"  # shorter traceback in CI
    
    if quick_mode and test_type == "Integration":
        # In quick mode, skip slow tests
        args += " -k 'not slow'"
    
    success, elapsed = run_pytest(test_path, args)
    
    if success:
        print_success(f"{test_type} tests passed ({elapsed:.1f}s)")
    else:
        print_error(f"{test_type} tests failed ({elapsed:.1f}s)")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Physics Engine CI Test Runner")
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--benchmarks', action='store_true', help='Run benchmark tests')
    parser.add_argument('--debug', action='store_true', help='Run debugging tests')
    parser.add_argument('--all', action='store_true', help='Run all tests including benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run minimal test set')
    
    args = parser.parse_args()
    
    # Determine what to run
    run_unit = True
    run_integration = True
    run_benchmarks = args.benchmarks or args.all
    run_debug = args.debug
    
    if args.unit:
        run_integration = False
        run_benchmarks = False
        # Keep debug if explicitly requested
        run_debug = args.debug
    elif args.integration:
        run_unit = False
        run_benchmarks = False
        run_debug = False
    elif args.debug:
        # If only debug flag, don't run other tests by default
        run_unit = False
        run_integration = False
        run_benchmarks = False
        run_debug = True
    
    # Print header
    print_header("Physics Engine Test Suite")
    
    if args.quick:
        print(f"{YELLOW}Running in QUICK mode - minimal tests only{RESET}")
    
    # Check if we're in the right directory
    if not os.path.exists("physics") or not os.path.exists("tests"):
        print(f"{RED}Error: Must run from physics_core directory{RESET}")
        print(f"Current directory: {os.getcwd()}")
        return 1
    
    if os.environ.get('CI') == 'true':
        print(f"{GREEN}CI environment detected{RESET}")
    
    # Track results
    all_passed = True
    start_time = time.time()
    
    # Run tests
    if run_unit:
        if not run_tests("Unit", "tests/unit", args.quick):
            all_passed = False
    
    if run_integration:
        if not run_tests("Integration", "tests/integration", args.quick):
            all_passed = False
    
    if run_benchmarks:
        if not run_tests("Benchmark", "tests/benchmarks", args.quick):
            all_passed = False
    
    if run_debug:
        print_header("Running Debug Tests")
        print(f"{YELLOW}Warning: Debug tests are diagnostic tools, not regression tests{RESET}")
        print("They may fail or behave unexpectedly - this is normal\n")
        
        # Run debug tests but don't fail CI on their results
        run_tests("Debug", "tests/debugging", args.quick)
        print(f"\n{BLUE}Debug tests completed (results not counted in CI pass/fail){RESET}")
    
    # Summary
    total_time = time.time() - start_time
    print_header("Test Summary")
    print(f"Total time: {total_time:.1f}s")
    
    if all_passed:
        print_success("All tests passed!")
        return 0
    else:
        print_error("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())