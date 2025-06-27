#!/usr/bin/env python3
"""
Run all tests in the physics engine codebase without requiring pytest
"""
import sys
import subprocess
import os
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{BLUE}Running: {description}{RESET}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{GREEN}✓ {description} passed{RESET}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"{RED}✗ {description} failed{RESET}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
    except Exception as e:
        print(f"{RED}✗ {description} failed with exception: {e}{RESET}")
        return False

def main():
    """Run all tests"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}      COMPREHENSIVE TEST SUITE FOR PHYSICS ENGINE{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Change to physics_core directory
    os.chdir(Path(__file__).parent.parent)
    
    tests = [
        # Main CI tests
        (["python3", "tests/run_ci.py"], "Main CI Test Suite"),
        
        # Custom ops tests
        (["python3", "tests/unit/custom_ops/test_c_library.py"], "C Library Tests"),
        (["python3", "tests/unit/custom_ops/test_integration.py"], "Custom Ops Integration Tests"),
        
        # Integration tests - these might need modifications to run without pytest
        (["python3", "tests/integration/test_simulation_stability.py"], "Simulation Stability Tests"),
        (["python3", "tests/integration/test_energy_conservation.py"], "Energy Conservation Tests"),
        (["python3", "tests/integration/test_fuzzing_stability.py"], "Fuzzing Stability Tests"),
        
        # Debugging tests
        (["python3", "tests/debugging/test_position_corruption.py"], "Position Corruption Tests"),
        (["python3", "tests/debugging/test_nan_propagation.py"], "NaN Propagation Tests"),
        (["python3", "tests/debugging/test_empty_contacts.py"], "Empty Contacts Tests"),
        (["python3", "tests/debugging/test_jit_early_return.py"], "JIT Early Return Tests"),
        
        # Benchmark test
        (["python3", "tests/benchmarks/test_physics_step_performance.py"], "Physics Step Performance"),
        
        # Demo scripts
        (["python3", "custom_ops/examples/basic_demo.py"], "Custom Ops Basic Demo"),
        (["python3", "custom_ops/examples/benchmark.py"], "Custom Ops Benchmark"),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for cmd, description in tests:
        # Check if file exists
        file_path = Path(cmd[1])
        if not file_path.exists():
            print(f"\n{RED}✗ {description} - File not found: {file_path}{RESET}")
            failed += 1
            continue
            
        # Check if it's a pytest file
        with open(file_path, 'r') as f:
            content = f.read()
            if 'import pytest' in content and 'test_' in file_path.name:
                print(f"\n{BLUE}⚠ {description} - Skipping (requires pytest){RESET}")
                skipped += 1
                continue
        
        if run_command(cmd, description):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}                     TEST SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"{BLUE}Skipped: {skipped}{RESET}")
    print(f"Total: {passed + failed + skipped}")
    
    if failed == 0:
        print(f"\n{GREEN}ALL TESTS PASSED!{RESET}")
        return 0
    else:
        print(f"\n{RED}SOME TESTS FAILED!{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())