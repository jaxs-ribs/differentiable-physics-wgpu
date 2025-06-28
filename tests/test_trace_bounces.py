#!/usr/bin/env python3
"""Track multiple bounces to find energy gain."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JIT'] = '0'

from physics.engine import TensorPhysicsEngine
from physics.types import create_body_array, ShapeType, BodySchema

def test_bounces():
    """Track energy through multiple bounces."""
    print("\n=== Tracking Multiple Bounces ===")
    
    bodies = []
    
    # Ground
    bodies.append(create_body_array(
        position=np.array([0., 0., 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1e8,
        inertia=np.eye(3, dtype=np.float32) * 1e8,
        shape_type=ShapeType.BOX,
        shape_params=np.array([10., 0.5, 10.], dtype=np.float32)
    ))
    
    # Ball
    bodies.append(create_body_array(
        position=np.array([0., 2.0, 0.], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1., 0., 0., 0.], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32) * 0.1,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0., 0.], dtype=np.float32)
    ))
    
    engine = TensorPhysicsEngine(np.stack(bodies), dt=0.001)
    
    bounces = []
    max_height = 2.0
    prev_vy = 0
    in_bounce = False
    
    print("\nSimulating...")
    for step in range(10000):
        state = engine.get_state()
        y = state[1, BodySchema.POS_Y]
        vy = state[1, BodySchema.VEL_Y]
        
        # Track max height
        if vy > 0 and prev_vy <= 0:  # Just bounced
            in_bounce = True
            bounce_velocity = vy
            
        if in_bounce and vy < 0 and prev_vy >= 0:  # Reached peak
            bounces.append({
                'step': step,
                'height': y,
                'bounce_velocity': bounce_velocity
            })
            print(f"Bounce {len(bounces)}: height={y:.4f}, bounce_vel={bounce_velocity:.4f}")
            
            if len(bounces) >= 5:
                break
            in_bounce = False
            
        prev_vy = vy
        engine.step()
    
    # Analyze bounces
    print("\nBounce Analysis:")
    print("Bounce | Height  | Ratio to Previous")
    print("-------|---------|------------------")
    for i, bounce in enumerate(bounces):
        if i == 0:
            print(f"   {i+1}   | {bounce['height']:.4f} | -")
        else:
            ratio = bounce['height'] / bounces[i-1]['height']
            print(f"   {i+1}   | {bounce['height']:.4f} | {ratio:.4f}")
    
    # Check energy conservation
    print("\nEnergy Analysis:")
    for i in range(1, len(bounces)):
        if bounces[i]['height'] > bounces[i-1]['height']:
            increase = (bounces[i]['height'] / bounces[i-1]['height'] - 1) * 100
            print(f"  Energy INCREASED by {increase:.1f}% from bounce {i} to {i+1}")

if __name__ == "__main__":
    test_bounces()