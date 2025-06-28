#!/usr/bin/env python3
"""Debug impulse sign and magnitude."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_impulse_sign():
    """Debug the impulse sign issue."""
    print("\n=== Debugging Impulse Sign ===")
    
    # Scenario: Ball falling onto ground
    # Ground (box) is body A, Ball (sphere) is body B
    
    # Velocities
    vel_a = np.array([0., 0., 0.])  # Ground is static
    vel_b = np.array([0., -10., 0.]) # Ball falling down
    
    # Normal points from B to A (from sphere to box)
    # For ball above ground, this points downward
    normal = np.array([0., -1., 0.])
    
    # Calculate relative velocity
    v_rel = vel_a - vel_b
    print(f"Velocities:")
    print(f"  A (ground): {vel_a}")
    print(f"  B (ball): {vel_b}")
    print(f"  v_rel = v_a - v_b = {v_rel}")
    
    # Project onto normal
    v_rel_normal = np.dot(v_rel, normal)
    print(f"\nNormal: {normal} (points from B to A, downward)")
    print(f"v_rel_normal = v_rel · n = {v_rel_normal}")
    
    # Check if approaching
    if v_rel_normal < 0:
        print("Bodies are approaching (v_rel_normal < 0) ✓")
    else:
        print("Bodies are separating (v_rel_normal >= 0)")
    
    # Calculate impulse
    restitution = 0.8
    mass_a = 1e8  # Very large (static)
    mass_b = 1.0
    
    inv_mass_a = 1.0 / mass_a  # ~0
    inv_mass_b = 1.0 / mass_b  # 1
    
    # Standard formula
    numerator = -(1.0 + restitution) * v_rel_normal
    denominator = inv_mass_a + inv_mass_b
    j_magnitude = numerator / denominator
    
    print(f"\nImpulse calculation:")
    print(f"  Restitution: {restitution}")
    print(f"  Numerator: -(1 + e) * v_rel_normal = -(1 + {restitution}) * {v_rel_normal} = {numerator}")
    print(f"  Denominator: {denominator}")
    print(f"  j_magnitude = {j_magnitude}")
    
    # The issue: impulse_vectors_a uses the normal direction
    impulse_a = j_magnitude * normal
    impulse_b = -impulse_a
    
    print(f"\nImpulse vectors:")
    print(f"  impulse_a = j * n = {j_magnitude} * {normal} = {impulse_a}")
    print(f"  impulse_b = -impulse_a = {impulse_b}")
    
    # Velocity changes
    delta_vel_a = impulse_a * inv_mass_a
    delta_vel_b = impulse_b * inv_mass_b
    
    print(f"\nVelocity changes:")
    print(f"  delta_vel_a = {delta_vel_a}")
    print(f"  delta_vel_b = {delta_vel_b}")
    
    # New velocities
    new_vel_a = vel_a + delta_vel_a
    new_vel_b = vel_b + delta_vel_b
    
    print(f"\nNew velocities:")
    print(f"  A: {new_vel_a}")
    print(f"  B: {new_vel_b}")
    
    print(f"\nAnalysis:")
    print(f"  Ball was moving at {vel_b[1]} m/s downward")
    print(f"  Ball is now moving at {new_vel_b[1]} m/s")
    print(f"  Expected: {restitution * 10} m/s upward")
    
    # The problem: impulse magnitude is positive, normal is downward
    # So impulse on A is downward, impulse on B is upward
    # This accelerates the ball even more!

if __name__ == "__main__":
    debug_impulse_sign()