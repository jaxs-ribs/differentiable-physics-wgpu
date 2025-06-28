#!/usr/bin/env python3
"""Debug impulse calculation in detail."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_impulse_calculation():
    """Debug the impulse calculation logic."""
    print("\n=== Debugging Impulse Calculation ===")
    
    # Scenario: Ball falling onto ground
    # Ground (box) is body A, Ball (sphere) is body B
    
    # Velocities
    vel_a = np.array([0., 0., 0.])  # Ground is static
    vel_b = np.array([0., -10., 0.]) # Ball falling down
    
    # For sphere-box collision, normal should point from B to A
    # Since ball is above ground, normal points upward
    normal = np.array([0., 1., 0.])
    
    # Calculate relative velocity
    v_rel = vel_a - vel_b
    print(f"Velocities:")
    print(f"  A (ground): {vel_a}")
    print(f"  B (ball): {vel_b}")
    print(f"  v_rel = v_a - v_b = {v_rel}")
    
    # Project onto normal
    v_rel_normal = np.dot(v_rel, normal)
    print(f"\nNormal: {normal} (points from B to A)")
    print(f"v_rel_normal = v_rel · n = {v_rel_normal}")
    
    # Check if approaching
    if v_rel_normal < 0:
        print("Bodies are approaching (v_rel_normal < 0)")
    else:
        print("Bodies are separating (v_rel_normal >= 0)")
    
    # Calculate impulse
    restitution = 0.8
    mass_a = 1e8  # Very large (static)
    mass_b = 1.0
    
    inv_mass_a = 1.0 / mass_a  # ~0
    inv_mass_b = 1.0 / mass_b  # 1
    
    numerator = -(1.0 + restitution) * v_rel_normal
    denominator = inv_mass_a + inv_mass_b
    
    j_magnitude = numerator / denominator
    
    print(f"\nImpulse calculation:")
    print(f"  Restitution: {restitution}")
    print(f"  Numerator: -(1 + e) * v_rel_normal = -(1 + {restitution}) * {v_rel_normal} = {numerator}")
    print(f"  Denominator: 1/m_a + 1/m_b = {inv_mass_a:.6f} + {inv_mass_b} = {denominator}")
    print(f"  j_magnitude = {j_magnitude}")
    
    # Apply impulse
    impulse_a = j_magnitude * normal
    impulse_b = -impulse_a
    
    delta_vel_a = impulse_a * inv_mass_a
    delta_vel_b = impulse_b * inv_mass_b
    
    print(f"\nImpulses:")
    print(f"  On A: {impulse_a}")
    print(f"  On B: {impulse_b}")
    
    print(f"\nVelocity changes:")
    print(f"  A: {delta_vel_a}")
    print(f"  B: {delta_vel_b}")
    
    new_vel_a = vel_a + delta_vel_a
    new_vel_b = vel_b + delta_vel_b
    
    print(f"\nNew velocities:")
    print(f"  A: {new_vel_a}")
    print(f"  B: {new_vel_b}")
    
    print(f"\nExpected ball velocity: {restitution * 10} m/s upward")
    print(f"Actual ball velocity: {new_vel_b[1]} m/s")

if __name__ == "__main__":
    debug_impulse_calculation()