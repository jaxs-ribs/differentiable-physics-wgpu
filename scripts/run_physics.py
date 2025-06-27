#!/usr/bin/env python3
"""Physics engine simulation runner.

This is the main entry point for running physics simulations.
The modular physics engine code is in the physics/ directory.

Usage:
  python3 run_physics.py --steps 200
  python3 run_physics.py --steps 500 --output artifacts/my_simulation.npy
"""
import sys
import os

# Add parent directory to path so we can import physics modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add tinygrad to path
tinygrad_path = os.path.join(parent_dir, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

from physics.main import main

if __name__ == "__main__":
  main()