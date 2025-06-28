#!/usr/bin/env python3
"""Trace through impulse calculation step by step."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

def test_impulse_calc():
    """Manual impulse calculation."""
    print("\n=== Manual Impulse Calculation ===")
    
    # Setup: ground (A) and ball (B)
    # Ball moving down at -3 m/s
    # Normal points from B to A (downward: [0, -1, 0])
    
    vel_a = np.array([0., 0., 0.])  # Ground velocity
    vel_b = np.array([0., -3., 0.])  # Ball velocity
    normal = np.array([0., -1., 0.])  # From ball to ground
    inv_mass_a = 1e-8  # Ground (very heavy)
    inv_mass_b = 1.0   # Ball (1 kg)
    e = 0.1
    
    print("Setup:")
    print(f"  Ground (A): vel={vel_a}, inv_mass={inv_mass_a}")
    print(f"  Ball (B): vel={vel_b}, inv_mass={inv_mass_b}")
    print(f"  Normal (B→A): {normal}")
    print(f"  Restitution: {e}")
    
    # Calculate relative velocity
    v_rel = vel_a - vel_b  # This is the relative velocity of A w.r.t B
    print(f"\nRelative velocity (A-B): {v_rel}")
    
    v_rel_normal = np.dot(v_rel, normal)
    print(f"v_rel·n: {v_rel_normal}")
    
    # Check if approaching
    if v_rel_normal < 0:
        print("Bodies are approaching (correct)")
    else:
        print("Bodies are separating (wrong!)")
    
    # Calculate impulse magnitude
    numerator = (1 + e) * (-v_rel_normal)
    denominator = inv_mass_a + inv_mass_b
    j = numerator / denominator
    
    print(f"\nImpulse calculation:")
    print(f"  numerator = (1 + {e}) * (-{v_rel_normal}) = {numerator}")
    print(f"  denominator = {inv_mass_a} + {inv_mass_b} = {denominator}")
    print(f"  j = {j}")
    
    # Apply impulse
    impulse_on_a = j * normal
    impulse_on_b = -j * normal
    
    print(f"\nImpulses:")
    print(f"  On A: {impulse_on_a}")
    print(f"  On B: {impulse_on_b}")
    
    # Calculate velocity changes
    delta_vel_a = impulse_on_a * inv_mass_a
    delta_vel_b = impulse_on_b * inv_mass_b
    
    vel_a_after = vel_a + delta_vel_a
    vel_b_after = vel_b + delta_vel_b
    
    print(f"\nVelocity changes:")
    print(f"  Ground: {vel_a} → {vel_a_after}")
    print(f"  Ball: {vel_b} → {vel_b_after}")
    
    print(f"\nExpected ball velocity: {-3 * e} = {-3 * e}")
    print(f"Actual ball velocity: {vel_b_after[1]}")
    
    # Now let's check what the solver code does
    print("\n\n=== What the solver does ===")
    # In solver.py line 84: v_rel = v_contact_a - v_contact_b
    # But wait, pair_indices has (0, 1), so:
    # - indices_a = 0 (ground)
    # - indices_b = 1 (ball)
    
    print("In the solver:")
    print("  indices_a = 0 (ground)")
    print("  indices_b = 1 (ball)")
    print("  v_rel = vel_a - vel_b = ground_vel - ball_vel")
    print(f"  v_rel = {vel_a} - {vel_b} = {vel_a - vel_b}")
    
    v_rel_solver = vel_a - vel_b
    v_rel_normal_solver = np.dot(v_rel_solver, normal)
    print(f"  v_rel·n = {v_rel_solver} · {normal} = {v_rel_normal_solver}")
    
    if v_rel_normal_solver < 0:
        print("  Solver sees: approaching (will apply impulse)")
    else:
        print("  Solver sees: separating (will skip)")

if __name__ == "__main__":
    test_impulse_calc()