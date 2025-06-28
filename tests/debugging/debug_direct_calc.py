#!/usr/bin/env python3
"""Debug the impulse calculation directly."""

import numpy as np

def debug_impulse_calculation():
    """Calculate impulse step by step."""
    print("\n=== Direct Impulse Calculation ===")
    
    # Scenario: Ball at -5 m/s hits static ground
    # Normal points from B (ball) to A (ground), which is downward
    normal = np.array([0, -1, 0])
    
    # Velocities
    vel_a = np.array([0, 0, 0])  # Ground static
    vel_b = np.array([0, -5, 0])  # Ball falling
    
    # Relative velocity
    v_rel = vel_a - vel_b  # [0, 5, 0]
    v_rel_normal = np.dot(v_rel, normal)  # 5 * (-1) = -5
    
    print(f"Velocities:")
    print(f"  A (ground): {vel_a}")
    print(f"  B (ball): {vel_b}")
    print(f"  Normal (B→A): {normal}")
    print(f"  v_rel = v_a - v_b = {v_rel}")
    print(f"  v_rel_normal = {v_rel_normal}")
    
    # Check if approaching
    if v_rel_normal < 0:
        print("  -> Approaching (v_rel_normal < 0) ✓")
    else:
        print("  -> Separating")
        return
    
    # Impulse calculation
    restitution = 0.1
    mass_a = 1e8  # Very large
    mass_b = 1.0
    
    # Current code formula
    numerator = (1.0 + restitution) * (-v_rel_normal)  # (1.1) * 5 = 5.5
    denominator = 1/mass_a + 1/mass_b  # ~1.0
    j_magnitude = numerator / denominator  # ~5.5
    
    print(f"\nImpulse calculation (current code):")
    print(f"  numerator = (1 + e) * (-v_rel_normal) = {numerator}")
    print(f"  j_magnitude = {j_magnitude}")
    
    # Apply impulse
    impulse_a = j_magnitude * normal  # 5.5 * [0,-1,0] = [0,-5.5,0]
    impulse_b = -impulse_a  # [0,5.5,0]
    
    delta_vel_a = impulse_a / mass_a  # ~[0,0,0]
    delta_vel_b = impulse_b / mass_b  # [0,5.5,0]
    
    new_vel_b = vel_b + delta_vel_b  # [0,-5,0] + [0,5.5,0] = [0,0.5,0]
    
    print(f"\nResults:")
    print(f"  impulse_a: {impulse_a}")
    print(f"  impulse_b: {impulse_b}")
    print(f"  delta_vel_b: {delta_vel_b}")
    print(f"  new_vel_b: {new_vel_b}")
    
    print(f"\nExpected: ball should go from -5 m/s to +0.5 m/s")
    print(f"Actual: ball goes from -5 m/s to {new_vel_b[1]} m/s")
    
    # The correct formula should be:
    print(f"\n=== Correct Calculation ===")
    # For collisions, impulse magnitude is positive when bodies approach
    # j = (1 + e) * |v_rel·n| / (1/m_a + 1/m_b)
    correct_j = (1 + restitution) * abs(v_rel_normal) / denominator
    print(f"Correct j_magnitude: {correct_j}")
    
    # But the impulse on A should be OPPOSITE to the normal
    # Because we want to push A away from B
    correct_impulse_a = -correct_j * normal  # -5.5 * [0,-1,0] = [0,5.5,0]
    correct_impulse_b = -correct_impulse_a  # [0,-5.5,0]
    
    correct_delta_b = correct_impulse_b / mass_b
    correct_new_vel_b = vel_b + correct_delta_b
    
    print(f"Correct impulse_b: {correct_impulse_b}")
    print(f"Correct new_vel_b: {correct_new_vel_b}")

if __name__ == "__main__":
    debug_impulse_calculation()