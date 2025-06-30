#!/usr/bin/env python3
"""Common test setup for all physics tests."""

import sys
import os

def setup_test_paths():
    """Set up Python paths for tests to find physics modules."""
    # Get absolute path to this file
    test_setup_file = os.path.abspath(__file__)
    
    # Navigate up to physics_core directory
    # test_setup.py is in physics_core/tests/
    tests_dir = os.path.dirname(test_setup_file)
    physics_core_path = os.path.dirname(tests_dir)
    
    # Add physics_core to path
    if physics_core_path not in sys.path:
        sys.path.insert(0, physics_core_path)
    
    # Add tinygrad to path
    tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")
    if os.path.exists(tinygrad_path) and tinygrad_path not in sys.path:
        sys.path.insert(0, tinygrad_path)
    
    return physics_core_path, tinygrad_path

# Auto-setup when imported
physics_core_path, tinygrad_path = setup_test_paths()