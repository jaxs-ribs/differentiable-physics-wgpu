#!/usr/bin/env python3
"""Simple CI test runner for XPBD physics engine."""
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add paths for physics_core and tinygrad
physics_core_path = os.path.dirname(os.path.abspath(__file__))
tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")

if physics_core_path not in sys.path:
    sys.path.insert(0, physics_core_path)
if os.path.exists(tinygrad_path) and tinygrad_path not in sys.path:
    sys.path.insert(0, tinygrad_path)

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def run_pytest(test_path, args=""):
    """Run pytest on a directory or file."""
    python_cmd = "python" if os.environ.get("CONDA_DEFAULT_ENV") else "python3"
    cmd = f"{python_cmd} -m pytest {args} {test_path}"
    
    # Set up environment
    env = os.environ.copy()
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


def run_tests(test_name, test_path, quick_mode=False):
    """Run a set of tests and report results."""
    print(f"\n{BLUE}Running {test_name} Tests{RESET}")
    
    args = "-v"
    if quick_mode:
        args += " -x"  # Stop on first failure in quick mode
    
    success, elapsed = run_pytest(test_path, args)
    
    if success:
        print(f"{GREEN}✓ {test_name} tests passed ({elapsed:.1f}s){RESET}")
    else:
        print(f"{RED}✗ {test_name} tests failed ({elapsed:.1f}s){RESET}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="XPBD Physics Engine Test Runner")
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--quick', action='store_true', help='Stop on first failure')
    
    args = parser.parse_args()
    
    print(f"{BLUE}XPBD Physics Engine Test Suite{RESET}")
    
    if args.quick:
        print(f"{YELLOW}Quick mode: stopping on first failure{RESET}")
    
    # Check directory
    if not os.path.exists("physics") or not os.path.exists("tests"):
        print(f"{RED}Error: Must run from physics_core directory{RESET}")
        return 1
    
    all_passed = True
    start_time = time.time()
    
    # Run XPBD unit tests
    if not run_tests("XPBD Unit", "tests/unit/xpbd", args.quick):
        all_passed = False
    
    # Run basic XPBD integration tests (unless unit-only)
    if not args.unit:
        if not run_tests("XPBD Basic", "tests/test_xpbd_basic.py", args.quick):
            all_passed = False
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{BLUE}Summary{RESET}")
    print(f"Total time: {total_time:.1f}s")
    
    if all_passed:
        print(f"{GREEN}✓ All tests passed!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed!{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())