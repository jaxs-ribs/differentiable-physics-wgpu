#!/usr/bin/env python3
"""Calculate effective restitution from observed behavior."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_effective_restitution():
    """Calculate what restitution would produce observed behavior."""
    
    # Observed data from test
    v_before = -10.85
    v_after = 13.01
    
    print(f"Observed:")
    print(f"  Velocity before: {v_before}")
    print(f"  Velocity after: {v_after}")
    
    # For a ball bouncing off a static surface:
    # v_after = -e * v_before
    # So: e = -v_after / v_before
    
    e_effective = -v_after / v_before
    
    print(f"\nEffective restitution: {e_effective:.3f}")
    
    # Check different theories
    print(f"\nTheory checks:")
    
    # Theory 1: e = 0.1 (expected)
    v_expected_01 = -0.1 * v_before
    print(f"  If e=0.1: v_after should be {v_expected_01:.3f}")
    
    # Theory 2: e = 1.2
    v_expected_12 = -1.2 * v_before
    print(f"  If e=1.2: v_after should be {v_expected_12:.3f}")
    
    # Theory 3: Double application with e=0.1
    # First bounce: -10.85 → +1.085
    # Second bounce: +1.085 → -0.1085 (but this doesn't make sense)
    
    # Theory 4: Formula error
    # What if j = (1+e)*v instead of (1+e)*v_rel/(1/m1+1/m2)?
    # With infinite mass ground: Δv = 2*j = 2*(1+e)*v = 2*1.1*10.85 = 23.87
    # Final v = -10.85 + 23.87 = 13.02
    print(f"\n  Theory: Double impulse with e=0.1")
    print(f"  Δv = 2*(1+0.1)*10.85 = {2*1.1*10.85:.3f}")
    print(f"  Final v = -10.85 + {2*1.1*10.85:.3f} = {-10.85 + 2*1.1*10.85:.3f}")
    print(f"  This matches the observed {v_after}!")

if __name__ == "__main__":
    calculate_effective_restitution()