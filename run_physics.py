#!/usr/bin/env python3
"""Physics engine simulation runner.

This is the main entry point for running physics simulations.
The modular physics engine code is in the physics/ directory.

Usage:
  python3 run_physics.py --steps 200
  python3 run_physics.py --steps 500 --output artifacts/my_simulation.npy
"""
from physics.main import main

if __name__ == "__main__":
  main()