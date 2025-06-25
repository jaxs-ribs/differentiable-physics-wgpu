#!/usr/bin/env python3
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import sys

@dataclass
class Body:
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    orientation: np.ndarray   # quaternion [w, x, y, z]
    angular_vel: np.ndarray   # [wx, wy, wz]
    mass: float
    inertia: np.ndarray      # 3x3 tensor
    shape_type: int          # 0=sphere, 1=capsule, 2=box
    shape_params: np.ndarray # radius for sphere, [radius, height] for capsule, [hx,hy,hz] for box

class PhysicsEngine:
    def __init__(self, dt: float = 0.016, gravity: np.ndarray = np.array([0, -9.81, 0])):
        self.dt = dt
        self.gravity = gravity
        self.bodies = []
        
    def add_body(self, body: Body):
        self.bodies.append(body)
        
    def step(self):
        # Semi-implicit Euler integration
        for body in self.bodies:
            # Apply gravity
            body.velocity += self.gravity * self.dt
            
            # Update position
            body.position += body.velocity * self.dt
            
            # Update orientation (simplified - no angular dynamics yet)
            # TODO: proper quaternion integration
            
        # Simple collision detection and response
        self._detect_and_resolve_collisions()
            
    def _detect_and_resolve_collisions(self):
        # For each pair of bodies
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                body_a = self.bodies[i]
                body_b = self.bodies[j]
                
                # Compute SDF distance
                distance, normal = self._compute_sdf_distance(body_a, body_b)
                
                if distance < 0:  # Collision detected
                    # Soft penalty method
                    penalty_force = -distance * 1000.0  # Stiffness constant
                    impulse = normal * penalty_force * self.dt
                    
                    # Apply equal and opposite impulses
                    body_a.velocity += impulse / body_a.mass
                    body_b.velocity -= impulse / body_b.mass
                    
    def _compute_sdf_distance(self, body_a: Body, body_b: Body) -> Tuple[float, np.ndarray]:
        # Simplified sphere-sphere distance for now
        if body_a.shape_type == 0 and body_b.shape_type == 0:  # Both spheres
            center_dist = np.linalg.norm(body_b.position - body_a.position)
            radius_sum = body_a.shape_params[0] + body_b.shape_params[0]
            distance = center_dist - radius_sum
            
            if center_dist > 1e-6:
                normal = (body_b.position - body_a.position) / center_dist
            else:
                normal = np.array([0, 1, 0])  # Arbitrary up direction
                
            return distance, normal
        else:
            # Placeholder for other shape combinations
            return 1.0, np.array([0, 1, 0])
            
    def get_state(self) -> np.ndarray:
        # Pack all bodies into a single array
        # Format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, quat_y, quat_z, ...]
        state = []
        for body in self.bodies:
            state.extend(body.position)
            state.extend(body.velocity)
            state.extend(body.orientation)
            state.extend(body.angular_vel)
            state.append(body.mass)
            state.append(body.shape_type)
            state.extend(body.shape_params)
        return np.array(state, dtype=np.float32)

def create_test_scene() -> PhysicsEngine:
    engine = PhysicsEngine()
    
    # Create a falling sphere
    sphere = Body(
        position=np.array([0.0, 5.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        angular_vel=np.array([0.0, 0.0, 0.0]),
        mass=1.0,
        inertia=np.eye(3) * 0.4,  # Sphere inertia
        shape_type=0,  # Sphere
        shape_params=np.array([0.5, 0.0, 0.0])  # radius = 0.5
    )
    engine.add_body(sphere)
    
    # Create a ground sphere (large, stationary)
    ground = Body(
        position=np.array([0.0, -100.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel=np.array([0.0, 0.0, 0.0]),
        mass=1e6,  # Very heavy
        inertia=np.eye(3) * 1e6,
        shape_type=0,  # Sphere
        shape_params=np.array([100.0, 0.0, 0.0])  # radius = 100
    )
    engine.add_body(ground)
    
    return engine

def main():
    parser = argparse.ArgumentParser(description='Physics engine reference implementation')
    parser.add_argument('--dump', type=str, help='Dump state after one step to file')
    parser.add_argument('--steps', type=int, default=1, help='Number of steps to run')
    args = parser.parse_args()
    
    # Create test scene
    engine = create_test_scene()
    
    # Run simulation
    for i in range(args.steps):
        engine.step()
        
    # Dump state if requested
    if args.dump:
        state = engine.get_state()
        np.save(args.dump, state)
        print(f"Saved state to {args.dump}, shape: {state.shape}")
    else:
        # Print final positions
        for i, body in enumerate(engine.bodies):
            print(f"Body {i}: pos={body.position}, vel={body.velocity}")

if __name__ == "__main__":
    main()