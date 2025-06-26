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
        """Compute signed distance between two bodies and contact normal.
        Returns (distance, normal) where normal points from A to B."""
        
        # Get shape types
        type_a = body_a.shape_type
        type_b = body_b.shape_type
        
        # Sort to reduce cases (always call with smaller type first)
        if type_a > type_b:
            distance, normal = self._compute_sdf_distance(body_b, body_a)
            return distance, -normal
            
        # Sphere (0) - Sphere (0)
        if type_a == 0 and type_b == 0:
            return self._sdf_sphere_sphere(body_a, body_b)
            
        # Sphere (0) - Capsule (1)
        elif type_a == 0 and type_b == 1:
            return self._sdf_sphere_capsule(body_a, body_b)
            
        # Sphere (0) - Box (2)
        elif type_a == 0 and type_b == 2:
            return self._sdf_sphere_box(body_a, body_b)
            
        # Capsule (1) - Capsule (1)
        elif type_a == 1 and type_b == 1:
            return self._sdf_capsule_capsule(body_a, body_b)
            
        # Capsule (1) - Box (2)
        elif type_a == 1 and type_b == 2:
            return self._sdf_capsule_box(body_a, body_b)
            
        # Box (2) - Box (2)
        elif type_a == 2 and type_b == 2:
            return self._sdf_box_box(body_a, body_b)
            
        else:
            # Should never reach here
            return 1.0, np.array([0, 1, 0])
    
    # Helper functions for quaternion operations
    def _quaternion_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q."""
        # q = [w, x, y, z]
        w, x, y, z = q
        # Convert vector to quaternion form [0, v]
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # q * v * q^-1
        # For unit quaternions, q^-1 = q* (conjugate)
        q_conj = np.array([w, -x, -y, -z])
        
        # Quaternion multiplication: q * v
        qv = self._quaternion_multiply(q, v_quat)
        # (q * v) * q^-1
        result = self._quaternion_multiply(qv, q_conj)
        
        return result[1:4]  # Extract vector part
        
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
        
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Get inverse (conjugate) of unit quaternion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
        
    # SDF implementations for each shape pair
    def _sdf_sphere_sphere(self, body_a: Body, body_b: Body) -> Tuple[float, np.ndarray]:
        """SDF between two spheres."""
        center_dist = np.linalg.norm(body_b.position - body_a.position)
        radius_sum = body_a.shape_params[0] + body_b.shape_params[0]
        distance = center_dist - radius_sum
        
        if center_dist > 1e-6:
            normal = (body_b.position - body_a.position) / center_dist
        else:
            normal = np.array([0, 1, 0])  # Arbitrary up direction
            
        return distance, normal
        
    def _sdf_sphere_capsule(self, sphere: Body, capsule: Body) -> Tuple[float, np.ndarray]:
        """SDF between sphere and capsule."""
        sphere_center = sphere.position
        sphere_radius = sphere.shape_params[0]
        
        capsule_radius = capsule.shape_params[0]
        capsule_height = capsule.shape_params[1]
        
        # Transform sphere center to capsule's local space
        local_sphere = sphere_center - capsule.position
        local_sphere = self._quaternion_rotate(self._quaternion_inverse(capsule.orientation), local_sphere)
        
        # Capsule is aligned with Y axis in local space
        # Clamp to line segment
        half_height = capsule_height * 0.5
        clamped_y = np.clip(local_sphere[1], -half_height, half_height)
        
        # Closest point on capsule line segment
        closest_on_line = np.array([0, clamped_y, 0])
        
        # Distance from sphere center to closest point on line
        to_sphere = local_sphere - closest_on_line
        dist_to_line = np.linalg.norm(to_sphere)
        
        # Signed distance
        distance = dist_to_line - (sphere_radius + capsule_radius)
        
        # Normal in local space
        if dist_to_line > 1e-6:
            normal_local = to_sphere / dist_to_line
        else:
            normal_local = np.array([1, 0, 0])  # Arbitrary
            
        # Transform normal back to world space
        normal = self._quaternion_rotate(capsule.orientation, normal_local)
        
        return distance, normal
        
    def _sdf_sphere_box(self, sphere: Body, box: Body) -> Tuple[float, np.ndarray]:
        """SDF between sphere and box."""
        sphere_center = sphere.position
        sphere_radius = sphere.shape_params[0]
        
        box_half_extents = box.shape_params  # [hx, hy, hz]
        
        # Transform sphere center to box's local space
        local_sphere = sphere_center - box.position
        local_sphere = self._quaternion_rotate(self._quaternion_inverse(box.orientation), local_sphere)
        
        # Find closest point on box to sphere center
        closest = np.clip(local_sphere, -box_half_extents, box_half_extents)
        
        # Vector from closest point to sphere center
        to_sphere = local_sphere - closest
        dist = np.linalg.norm(to_sphere)
        
        # Signed distance
        distance = dist - sphere_radius
        
        # Normal in local space
        if dist > 1e-6:
            normal_local = to_sphere / dist
        else:
            # Sphere center is on box surface or inside
            # Find which face is closest
            distances_to_faces = np.zeros(6)
            face_normals = []
            
            # Check all 6 faces
            for axis in range(3):
                # Positive face
                distances_to_faces[axis*2] = box_half_extents[axis] - local_sphere[axis]
                normal = np.zeros(3)
                normal[axis] = 1.0
                face_normals.append(normal)
                
                # Negative face  
                distances_to_faces[axis*2 + 1] = box_half_extents[axis] + local_sphere[axis]
                normal = np.zeros(3)
                normal[axis] = -1.0
                face_normals.append(normal)
            
            # Use normal of closest face
            closest_face = np.argmin(distances_to_faces)
            normal_local = face_normals[closest_face]
            
        # Transform normal back to world space
        normal = self._quaternion_rotate(box.orientation, normal_local)
        
        return distance, normal
        
    def _sdf_capsule_capsule(self, cap_a: Body, cap_b: Body) -> Tuple[float, np.ndarray]:
        """SDF between two capsules."""
        # Get capsule parameters
        radius_a = cap_a.shape_params[0]
        height_a = cap_a.shape_params[1]
        radius_b = cap_b.shape_params[0]
        height_b = cap_b.shape_params[1]
        
        # Get capsule line segments in world space
        # Capsules are aligned with local Y axis
        y_axis_a = self._quaternion_rotate(cap_a.orientation, np.array([0, 1, 0]))
        y_axis_b = self._quaternion_rotate(cap_b.orientation, np.array([0, 1, 0]))
        
        # Line segment endpoints
        half_a = height_a * 0.5
        half_b = height_b * 0.5
        
        a1 = cap_a.position - y_axis_a * half_a
        a2 = cap_a.position + y_axis_a * half_a
        b1 = cap_b.position - y_axis_b * half_b
        b2 = cap_b.position + y_axis_b * half_b
        
        # Find closest points between line segments
        # Using vector formulation
        d1 = a2 - a1  # Direction vector of segment A
        d2 = b2 - b1  # Direction vector of segment B
        r = a1 - b1   # Vector between start points
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        
        denom = a * c - b * b
        
        # Check if lines are parallel
        if abs(denom) < 1e-6:
            # Parallel lines, use arbitrary point
            s = 0
            t = e / c if c > 1e-6 else 0
        else:
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
            
        # Clamp parameters to [0, 1]
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)
        
        # Recompute t for clamped s
        t = (e + b * s) / c if c > 1e-6 else 0
        t = np.clip(t, 0, 1)
        
        # Recompute s for clamped t
        s = (b * t - d) / a if a > 1e-6 else 0
        s = np.clip(s, 0, 1)
        
        # Closest points
        closest_a = a1 + s * d1
        closest_b = b1 + t * d2
        
        # Distance and normal
        vec = closest_b - closest_a
        dist = np.linalg.norm(vec)
        
        distance = dist - (radius_a + radius_b)
        
        if dist > 1e-6:
            normal = vec / dist
        else:
            normal = np.array([0, 1, 0])
            
        return distance, normal
        
    def _sdf_capsule_box(self, capsule: Body, box: Body) -> Tuple[float, np.ndarray]:
        """SDF between capsule and box."""
        # This is complex - we need to find closest point between line segment and box
        capsule_radius = capsule.shape_params[0]
        capsule_height = capsule.shape_params[1]
        box_half_extents = box.shape_params
        
        # Transform capsule to box's local space
        local_cap_pos = capsule.position - box.position
        local_cap_pos = self._quaternion_rotate(self._quaternion_inverse(box.orientation), local_cap_pos)
        
        # Get capsule axis in box's local space
        cap_y_world = self._quaternion_rotate(capsule.orientation, np.array([0, 1, 0]))
        cap_y_local = self._quaternion_rotate(self._quaternion_inverse(box.orientation), cap_y_world)
        
        # Capsule line segment in box's local space
        half_height = capsule_height * 0.5
        seg_start = local_cap_pos - cap_y_local * half_height
        seg_end = local_cap_pos + cap_y_local * half_height
        
        # Find closest point on line segment to box
        # This is done by checking multiple sample points
        min_dist = float('inf')
        closest_on_seg = seg_start
        
        # Sample along the line segment
        for i in range(11):  # 11 samples including endpoints
            t = i / 10.0
            point = seg_start + t * (seg_end - seg_start)
            
            # Closest point on box to this sample
            closest_on_box = np.clip(point, -box_half_extents, box_half_extents)
            dist = np.linalg.norm(point - closest_on_box)
            
            if dist < min_dist:
                min_dist = dist
                closest_on_seg = point
                
        # Now refine by checking box vertices against line segment
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    vertex = box_half_extents * np.array([sx, sy, sz])
                    
                    # Project vertex onto line segment
                    seg_vec = seg_end - seg_start
                    seg_len_sq = np.dot(seg_vec, seg_vec)
                    
                    if seg_len_sq > 1e-6:
                        t = np.dot(vertex - seg_start, seg_vec) / seg_len_sq
                        t = np.clip(t, 0, 1)
                        point_on_seg = seg_start + t * seg_vec
                    else:
                        point_on_seg = seg_start
                        
                    dist = np.linalg.norm(vertex - point_on_seg)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_on_seg = point_on_seg
                        
        # Final closest point on box
        closest_on_box = np.clip(closest_on_seg, -box_half_extents, box_half_extents)
        
        # Normal in local space
        vec = closest_on_seg - closest_on_box
        dist = np.linalg.norm(vec)
        
        distance = dist - capsule_radius
        
        if dist > 1e-6:
            normal_local = vec / dist
        else:
            # Capsule center line is inside or on box surface
            # Find which face is closest to the capsule center
            cap_center_local = local_cap_pos
            distances_to_faces = np.zeros(6)
            face_normals = []
            
            # Check all 6 faces
            for axis in range(3):
                # Positive face
                distances_to_faces[axis*2] = box_half_extents[axis] - cap_center_local[axis]
                normal = np.zeros(3)
                normal[axis] = 1.0
                face_normals.append(normal)
                
                # Negative face  
                distances_to_faces[axis*2 + 1] = box_half_extents[axis] + cap_center_local[axis]
                normal = np.zeros(3)
                normal[axis] = -1.0
                face_normals.append(normal)
            
            # Use normal of closest face
            closest_face = np.argmin(distances_to_faces)
            normal_local = face_normals[closest_face]
            
        # Transform normal back to world space
        normal = self._quaternion_rotate(box.orientation, normal_local)
        
        return distance, normal
        
    def _sdf_box_box(self, box_a: Body, box_b: Body) -> Tuple[float, np.ndarray]:
        """SDF between two boxes using SAT (Separating Axis Theorem)."""
        # Get half extents
        half_a = box_a.shape_params
        half_b = box_b.shape_params
        
        # Get box axes in world space
        axes_a = []
        axes_b = []
        for i in range(3):
            axis = np.zeros(3)
            axis[i] = 1.0
            axes_a.append(self._quaternion_rotate(box_a.orientation, axis))
            axes_b.append(self._quaternion_rotate(box_b.orientation, axis))
            
        # Vector between centers
        center_vec = box_b.position - box_a.position
        
        # Test all 15 potential separating axes
        min_penetration = float('inf')
        best_axis = np.array([0, 1, 0])
        
        # Test box A's face normals
        for i in range(3):
            axis = axes_a[i]
            penetration, sep_axis = self._test_separation_axis(
                axis, center_vec, half_a, half_b, axes_a, axes_b
            )
            if penetration < min_penetration:
                min_penetration = penetration
                best_axis = sep_axis
                
        # Test box B's face normals
        for i in range(3):
            axis = axes_b[i]
            penetration, sep_axis = self._test_separation_axis(
                axis, center_vec, half_a, half_b, axes_a, axes_b
            )
            if penetration < min_penetration:
                min_penetration = penetration
                best_axis = sep_axis
                
        # Test cross products of edges
        for i in range(3):
            for j in range(3):
                axis = np.cross(axes_a[i], axes_b[j])
                length = np.linalg.norm(axis)
                if length > 1e-6:
                    axis = axis / length
                    penetration, sep_axis = self._test_separation_axis(
                        axis, center_vec, half_a, half_b, axes_a, axes_b
                    )
                    if penetration < min_penetration:
                        min_penetration = penetration
                        best_axis = sep_axis
                        
        # The penetration depth is negative of the minimum separation
        distance = -min_penetration
        normal = best_axis
        
        # Ensure normal points from A to B
        if np.dot(normal, center_vec) < 0:
            normal = -normal
            
        return distance, normal
        
    def _test_separation_axis(self, axis: np.ndarray, center_vec: np.ndarray,
                              half_a: np.ndarray, half_b: np.ndarray,
                              axes_a: list, axes_b: list) -> Tuple[float, np.ndarray]:
        """Test a potential separating axis and return penetration depth."""
        # Project box A onto axis
        radius_a = 0
        for i in range(3):
            radius_a += half_a[i] * abs(np.dot(axis, axes_a[i]))
            
        # Project box B onto axis
        radius_b = 0
        for i in range(3):
            radius_b += half_b[i] * abs(np.dot(axis, axes_b[i]))
            
        # Distance between centers along axis
        center_dist = abs(np.dot(center_vec, axis))
        
        # Penetration depth (negative means separated)
        penetration = radius_a + radius_b - center_dist
        
        return penetration, axis
            
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