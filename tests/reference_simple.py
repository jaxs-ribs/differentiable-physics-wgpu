#!/usr/bin/env python3
"""Simplified reference implementation for testing"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from enum import IntEnum

class ShapeType(IntEnum):
    SPHERE = 0
    CAPSULE = 1  
    BOX = 2

@dataclass
class Body:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_vel: np.ndarray
    mass: float
    inertia: np.ndarray
    shape_type: ShapeType
    shape_params: np.ndarray
    
    def get_world_inertia_inv(self) -> np.ndarray:
        R = self._quaternion_to_matrix(self.orientation)
        I_local_inv = np.linalg.inv(self.inertia)
        return R @ I_local_inv @ R.T
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

class PhysicsEngine:
    def __init__(self, dt=0.016):
        self.dt = dt
        self.bodies = []
        self.restitution = 0.5
        
    def add_body(self, body):
        self.bodies.append(body)
        
    def step(self):
        # Simple O(n^2) collision detection for testing
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                self._check_collision(self.bodies[i], self.bodies[j])
                
    def _check_collision(self, body_a, body_b):
        # Only sphere-sphere for now
        if body_a.shape_type == ShapeType.SPHERE and body_b.shape_type == ShapeType.SPHERE:
            center_vec = body_b.position - body_a.position
            center_dist = np.linalg.norm(center_vec)
            radius_sum = body_a.shape_params[0] + body_b.shape_params[0]
            
            distance = center_dist - radius_sum
            
            if distance < 0:  # Collision
                normal = center_vec / center_dist if center_dist > 1e-6 else np.array([0,1,0])
                # Contact point is on the surface of sphere A in the direction of sphere B
                # For penetrating spheres, this gives the point of deepest penetration
                contact_point = body_a.position + normal * body_a.shape_params[0]
                
                # Calculate relative velocity
                r_a = contact_point - body_a.position
                r_b = contact_point - body_b.position
                
                print(f"Debug: contact_point={contact_point}")
                print(f"Debug: body_a.pos={body_a.position}, body_b.pos={body_b.position}")
                print(f"Debug: r_a={r_a}, r_b={r_b}")
                print(f"Debug: normal={normal}")
                
                v_rel = (body_b.velocity + np.cross(body_b.angular_vel, r_b) - 
                        body_a.velocity - np.cross(body_a.angular_vel, r_a))
                
                v_rel_n = np.dot(v_rel, normal)
                
                if v_rel_n < 0:  # Approaching
                    # Calculate impulse
                    inv_mass_a = 1.0 / body_a.mass if body_a.mass < 1e6 else 0.0
                    inv_mass_b = 1.0 / body_b.mass if body_b.mass < 1e6 else 0.0
                    
                    I_inv_a = body_a.get_world_inertia_inv() if body_a.mass < 1e6 else np.zeros((3,3))
                    I_inv_b = body_b.get_world_inertia_inv() if body_b.mass < 1e6 else np.zeros((3,3))
                    
                    r_a_cross_n = np.cross(r_a, normal)
                    r_b_cross_n = np.cross(r_b, normal)
                    
                    denominator = (inv_mass_a + inv_mass_b + 
                                  np.dot(r_a_cross_n, I_inv_a @ r_a_cross_n) +
                                  np.dot(r_b_cross_n, I_inv_b @ r_b_cross_n))
                    
                    if denominator > 1e-6:
                        j = -(1.0 + self.restitution) * v_rel_n / denominator
                        impulse = j * normal
                        
                        # Apply impulse
                        body_a.velocity -= impulse * inv_mass_a
                        body_b.velocity += impulse * inv_mass_b
                        
                        body_a.angular_vel -= I_inv_a @ np.cross(r_a, impulse)
                        body_b.angular_vel += I_inv_b @ np.cross(r_b, impulse)
                        
                        print(f"Collision! impulse={j:.3f}, ang_vel_a={body_a.angular_vel}, ang_vel_b={body_b.angular_vel}")