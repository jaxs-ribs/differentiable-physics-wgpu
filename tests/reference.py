#!/usr/bin/env python3
"""
Physics Engine Reference Implementation

A clean, modular physics engine following SOLID principles and Clean Code practices.
This serves as the golden reference for GPU physics implementations.
"""

import numpy as np
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Protocol
from enum import IntEnum
import abc


# ============================================================================
# Domain Models
# ============================================================================

class ShapeType(IntEnum):
    """Enumeration of supported collision shapes."""
    SPHERE = 0
    CAPSULE = 1
    BOX = 2


@dataclass
class Body:
    """Represents a rigid body in the physics simulation."""
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    orientation: np.ndarray   # quaternion [w, x, y, z]
    angular_vel: np.ndarray   # [wx, wy, wz]
    mass: float
    inertia: np.ndarray      # 3x3 tensor
    shape_type: ShapeType
    shape_params: np.ndarray  # Shape-specific parameters
    
    def __post_init__(self):
        """Validate body parameters."""
        assert len(self.position) == 3, "Position must be 3D"
        assert len(self.velocity) == 3, "Velocity must be 3D"
        assert len(self.orientation) == 4, "Orientation must be quaternion"
        assert self.mass > 0, "Mass must be positive"


# ============================================================================
# Math Utilities (Single Responsibility)
# ============================================================================

class Quaternion:
    """Utility class for quaternion operations."""
    
    @staticmethod
    def rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        w, x, y, z = quaternion
        v_quat = np.array([0, vector[0], vector[1], vector[2]])
        q_conj = np.array([w, -x, -y, -z])
        
        # q * v * q^-1
        qv = Quaternion.multiply(quaternion, v_quat)
        result = Quaternion.multiply(qv, q_conj)
        
        return result[1:4]  # Extract vector part
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(quaternion: np.ndarray) -> np.ndarray:
        """Get conjugate of unit quaternion."""
        return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


# ============================================================================
# Collision Detection Strategy Pattern
# ============================================================================

class CollisionDetector(Protocol):
    """Protocol for collision detection strategies."""
    
    def compute_distance(self, body_a: Body, body_b: Body) -> Tuple[float, np.ndarray]:
        """Compute signed distance and normal between two bodies."""
        ...


class SphereSphereDetector:
    """Collision detection for sphere-sphere pairs."""
    
    def compute_distance(self, sphere_a: Body, sphere_b: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between two spheres."""
        center_vector = sphere_b.position - sphere_a.position
        center_distance = np.linalg.norm(center_vector)
        radius_sum = sphere_a.shape_params[0] + sphere_b.shape_params[0]
        
        distance = center_distance - radius_sum
        normal = self._compute_normal(center_vector, center_distance)
        
        return distance, normal
    
    def _compute_normal(self, center_vector: np.ndarray, center_distance: float) -> np.ndarray:
        """Compute collision normal, handling edge cases."""
        if center_distance > 1e-6:
            return center_vector / center_distance
        return np.array([0, 1, 0])  # Arbitrary up direction for coincident centers


class SphereCapsuleDetector:
    """Collision detection for sphere-capsule pairs."""
    
    def compute_distance(self, sphere: Body, capsule: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between sphere and capsule."""
        # Transform to capsule's local space
        local_sphere_pos = self._to_local_space(
            sphere.position, capsule.position, capsule.orientation
        )
        
        # Find closest point on capsule's center line
        closest_point = self._closest_point_on_capsule_line(
            local_sphere_pos, capsule.shape_params[1]
        )
        
        # Compute distance and normal
        to_sphere = local_sphere_pos - closest_point
        dist_to_line = np.linalg.norm(to_sphere)
        
        distance = dist_to_line - (sphere.shape_params[0] + capsule.shape_params[0])
        normal = self._compute_normal(to_sphere, dist_to_line, capsule.orientation)
        
        return distance, normal
    
    def _to_local_space(self, world_pos: np.ndarray, body_pos: np.ndarray, 
                       body_orientation: np.ndarray) -> np.ndarray:
        """Transform world position to body's local space."""
        relative_pos = world_pos - body_pos
        return Quaternion.rotate(Quaternion.conjugate(body_orientation), relative_pos)
    
    def _closest_point_on_capsule_line(self, local_pos: np.ndarray, 
                                      capsule_height: float) -> np.ndarray:
        """Find closest point on capsule's center line segment."""
        half_height = capsule_height * 0.5
        clamped_y = np.clip(local_pos[1], -half_height, half_height)
        return np.array([0, clamped_y, 0])
    
    def _compute_normal(self, to_sphere: np.ndarray, distance: float,
                       capsule_orientation: np.ndarray) -> np.ndarray:
        """Compute collision normal in world space."""
        if distance > 1e-6:
            normal_local = to_sphere / distance
        else:
            normal_local = np.array([1, 0, 0])  # Arbitrary
        
        return Quaternion.rotate(capsule_orientation, normal_local)


class SphereBoxDetector:
    """Collision detection for sphere-box pairs."""
    
    def compute_distance(self, sphere: Body, box: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between sphere and box."""
        # Transform to box's local space
        local_sphere_pos = self._to_local_space(
            sphere.position, box.position, box.orientation
        )
        
        # Find closest point on box
        closest_on_box = np.clip(local_sphere_pos, -box.shape_params, box.shape_params)
        
        # Compute distance and normal
        to_sphere = local_sphere_pos - closest_on_box
        dist = np.linalg.norm(to_sphere)
        
        distance = dist - sphere.shape_params[0]
        normal = self._compute_normal(
            to_sphere, dist, local_sphere_pos, box.shape_params, box.orientation
        )
        
        return distance, normal
    
    def _to_local_space(self, world_pos: np.ndarray, body_pos: np.ndarray,
                       body_orientation: np.ndarray) -> np.ndarray:
        """Transform world position to body's local space."""
        relative_pos = world_pos - body_pos
        return Quaternion.rotate(Quaternion.conjugate(body_orientation), relative_pos)
    
    def _compute_normal(self, to_sphere: np.ndarray, dist: float,
                       local_sphere_pos: np.ndarray, half_extents: np.ndarray,
                       box_orientation: np.ndarray) -> np.ndarray:
        """Compute collision normal in world space."""
        if dist > 1e-6:
            normal_local = to_sphere / dist
        else:
            # Sphere inside box - find closest face
            normal_local = self._find_closest_face_normal(local_sphere_pos, half_extents)
        
        return Quaternion.rotate(box_orientation, normal_local)
    
    def _find_closest_face_normal(self, local_pos: np.ndarray, 
                                 half_extents: np.ndarray) -> np.ndarray:
        """Find normal of closest box face."""
        distances = []
        normals = []
        
        for axis in range(3):
            # Positive face
            distances.append(half_extents[axis] - local_pos[axis])
            normal = np.zeros(3)
            normal[axis] = 1.0
            normals.append(normal)
            
            # Negative face
            distances.append(half_extents[axis] + local_pos[axis])
            normal = np.zeros(3)
            normal[axis] = -1.0
            normals.append(normal)
        
        closest_face = np.argmin(distances)
        return normals[closest_face]


class CapsuleCapsuleDetector:
    """Collision detection for capsule-capsule pairs."""
    
    def compute_distance(self, cap_a: Body, cap_b: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between two capsules."""
        # Get line segments in world space
        seg_a = self._get_capsule_segment(cap_a)
        seg_b = self._get_capsule_segment(cap_b)
        
        # Find closest points between segments
        closest_a, closest_b = self._closest_points_between_segments(seg_a, seg_b)
        
        # Compute distance and normal
        vec = closest_b - closest_a
        dist = np.linalg.norm(vec)
        
        distance = dist - (cap_a.shape_params[0] + cap_b.shape_params[0])
        normal = vec / dist if dist > 1e-6 else np.array([0, 1, 0])
        
        return distance, normal
    
    def _get_capsule_segment(self, capsule: Body) -> Tuple[np.ndarray, np.ndarray]:
        """Get capsule's line segment endpoints in world space."""
        y_axis = Quaternion.rotate(capsule.orientation, np.array([0, 1, 0]))
        half_height = capsule.shape_params[1] * 0.5
        
        p1 = capsule.position - y_axis * half_height
        p2 = capsule.position + y_axis * half_height
        
        return p1, p2
    
    def _closest_points_between_segments(self, seg_a: Tuple[np.ndarray, np.ndarray],
                                       seg_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Find closest points between two line segments."""
        a1, a2 = seg_a
        b1, b2 = seg_b
        
        d1 = a2 - a1  # Direction vector of segment A
        d2 = b2 - b1  # Direction vector of segment B
        r = a1 - b1   # Vector between start points
        
        # Compute parameters
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        
        denom = a * c - b * b
        
        # Handle parallel lines
        if abs(denom) < 1e-6:
            s = 0
            t = e / c if c > 1e-6 else 0
        else:
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
        
        # Clamp parameters
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)
        
        # Recompute for clamped values
        t = np.clip((e + b * s) / c if c > 1e-6 else 0, 0, 1)
        s = np.clip((b * t - d) / a if a > 1e-6 else 0, 0, 1)
        
        return a1 + s * d1, b1 + t * d2


class CapsuleBoxDetector:
    """Collision detection for capsule-box pairs."""
    
    def compute_distance(self, capsule: Body, box: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between capsule and box."""
        # Transform capsule to box's local space
        local_segment = self._transform_segment_to_local(capsule, box)
        
        # Find closest points
        closest_on_seg = self._find_closest_point_on_segment_to_box(
            local_segment, box.shape_params
        )
        closest_on_box = np.clip(closest_on_seg, -box.shape_params, box.shape_params)
        
        # Compute distance and normal
        vec = closest_on_seg - closest_on_box
        dist = np.linalg.norm(vec)
        
        distance = dist - capsule.shape_params[0]
        normal = self._compute_normal(
            vec, dist, self._to_local_space(capsule.position, box.position, box.orientation),
            box.shape_params, box.orientation
        )
        
        return distance, normal
    
    def _transform_segment_to_local(self, capsule: Body, box: Body) -> Tuple[np.ndarray, np.ndarray]:
        """Transform capsule segment to box's local space."""
        # Get capsule axis in world space
        cap_y_world = Quaternion.rotate(capsule.orientation, np.array([0, 1, 0]))
        
        # Transform to box's local space
        local_pos = self._to_local_space(capsule.position, box.position, box.orientation)
        local_axis = Quaternion.rotate(Quaternion.conjugate(box.orientation), cap_y_world)
        
        half_height = capsule.shape_params[1] * 0.5
        return (local_pos - local_axis * half_height, local_pos + local_axis * half_height)
    
    def _to_local_space(self, world_pos: np.ndarray, body_pos: np.ndarray,
                       body_orientation: np.ndarray) -> np.ndarray:
        """Transform world position to body's local space."""
        relative_pos = world_pos - body_pos
        return Quaternion.rotate(Quaternion.conjugate(body_orientation), relative_pos)
    
    def _find_closest_point_on_segment_to_box(self, segment: Tuple[np.ndarray, np.ndarray],
                                            half_extents: np.ndarray) -> np.ndarray:
        """Find closest point on line segment to box."""
        seg_start, seg_end = segment
        
        # Sample along segment and check box vertices
        closest_point = seg_start
        min_dist = float('inf')
        
        # Sample along segment
        for t in np.linspace(0, 1, 11):
            point = seg_start + t * (seg_end - seg_start)
            closest_on_box = np.clip(point, -half_extents, half_extents)
            dist = np.linalg.norm(point - closest_on_box)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        
        # Check box vertices against segment
        for signs in [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                     (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]:
            vertex = half_extents * np.array(signs)
            point_on_seg = self._project_point_to_segment(vertex, segment)
            dist = np.linalg.norm(vertex - point_on_seg)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point_on_seg
        
        return closest_point
    
    def _project_point_to_segment(self, point: np.ndarray,
                                 segment: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Project point onto line segment."""
        seg_start, seg_end = segment
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq > 1e-6:
            t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0, 1)
            return seg_start + t * seg_vec
        return seg_start
    
    def _compute_normal(self, vec: np.ndarray, dist: float, local_cap_pos: np.ndarray,
                       half_extents: np.ndarray, box_orientation: np.ndarray) -> np.ndarray:
        """Compute collision normal in world space."""
        if dist > 1e-6:
            normal_local = vec / dist
        else:
            # Find closest face
            normal_local = self._find_closest_face_normal(local_cap_pos, half_extents)
        
        return Quaternion.rotate(box_orientation, normal_local)
    
    def _find_closest_face_normal(self, local_pos: np.ndarray,
                                 half_extents: np.ndarray) -> np.ndarray:
        """Find normal of closest box face."""
        distances = []
        normals = []
        
        for axis in range(3):
            # Positive face
            distances.append(half_extents[axis] - local_pos[axis])
            normal = np.zeros(3)
            normal[axis] = 1.0
            normals.append(normal)
            
            # Negative face
            distances.append(half_extents[axis] + local_pos[axis])
            normal = np.zeros(3)
            normal[axis] = -1.0
            normals.append(normal)
        
        closest_face = np.argmin(distances)
        return normals[closest_face]


class BoxBoxDetector:
    """Collision detection for box-box pairs using SAT."""
    
    def compute_distance(self, box_a: Body, box_b: Body) -> Tuple[float, np.ndarray]:
        """Compute SDF between two boxes using Separating Axis Theorem."""
        # Get box properties
        axes_a = self._get_box_axes(box_a.orientation)
        axes_b = self._get_box_axes(box_b.orientation)
        center_vec = box_b.position - box_a.position
        
        # Test all 15 potential separating axes
        min_penetration = float('inf')
        best_axis = np.array([0, 1, 0])
        
        # Test face normals
        for axes, half_extents in [(axes_a, box_a.shape_params), (axes_b, box_b.shape_params)]:
            for axis in axes:
                penetration = self._test_separation_axis(
                    axis, center_vec, box_a.shape_params, box_b.shape_params, axes_a, axes_b
                )
                if penetration < min_penetration:
                    min_penetration = penetration
                    best_axis = axis
        
        # Test edge cross products
        for axis_a in axes_a:
            for axis_b in axes_b:
                axis = np.cross(axis_a, axis_b)
                length = np.linalg.norm(axis)
                if length > 1e-6:
                    axis = axis / length
                    penetration = self._test_separation_axis(
                        axis, center_vec, box_a.shape_params, box_b.shape_params, axes_a, axes_b
                    )
                    if penetration < min_penetration:
                        min_penetration = penetration
                        best_axis = axis
        
        # Compute final distance and normal
        distance = -min_penetration
        normal = best_axis if np.dot(best_axis, center_vec) > 0 else -best_axis
        
        return distance, normal
    
    def _get_box_axes(self, orientation: np.ndarray) -> List[np.ndarray]:
        """Get box's local axes in world space."""
        return [
            Quaternion.rotate(orientation, np.array([1, 0, 0])),
            Quaternion.rotate(orientation, np.array([0, 1, 0])),
            Quaternion.rotate(orientation, np.array([0, 0, 1]))
        ]
    
    def _test_separation_axis(self, axis: np.ndarray, center_vec: np.ndarray,
                            half_a: np.ndarray, half_b: np.ndarray,
                            axes_a: List[np.ndarray], axes_b: List[np.ndarray]) -> float:
        """Test a potential separating axis and return penetration depth."""
        # Project boxes onto axis
        radius_a = sum(half_a[i] * abs(np.dot(axis, axes_a[i])) for i in range(3))
        radius_b = sum(half_b[i] * abs(np.dot(axis, axes_b[i])) for i in range(3))
        
        # Distance between centers along axis
        center_dist = abs(np.dot(center_vec, axis))
        
        # Penetration depth (negative means separated)
        return radius_a + radius_b - center_dist


# ============================================================================
# Collision Detection Factory (Factory Pattern)
# ============================================================================

class CollisionDetectorFactory:
    """Factory for creating appropriate collision detectors."""
    
    def __init__(self):
        self._detectors = {
            (ShapeType.SPHERE, ShapeType.SPHERE): SphereSphereDetector(),
            (ShapeType.SPHERE, ShapeType.CAPSULE): SphereCapsuleDetector(),
            (ShapeType.SPHERE, ShapeType.BOX): SphereBoxDetector(),
            (ShapeType.CAPSULE, ShapeType.CAPSULE): CapsuleCapsuleDetector(),
            (ShapeType.CAPSULE, ShapeType.BOX): CapsuleBoxDetector(),
            (ShapeType.BOX, ShapeType.BOX): BoxBoxDetector(),
        }
    
    def get_detector(self, type_a: ShapeType, type_b: ShapeType) -> CollisionDetector:
        """Get appropriate collision detector for shape pair."""
        # Always use smaller type first for consistency
        if type_a > type_b:
            type_a, type_b = type_b, type_a
        
        key = (type_a, type_b)
        if key in self._detectors:
            return self._detectors[key]
        
        raise ValueError(f"No detector for shape pair: {type_a}, {type_b}")


# ============================================================================
# Physics Components (Single Responsibility)
# ============================================================================

class Integrator:
    """Handles physics integration (motion update)."""
    
    def __init__(self, dt: float = 0.016):
        self.dt = dt
    
    def integrate(self, body: Body, acceleration: np.ndarray):
        """Update body position and velocity using semi-implicit Euler."""
        body.velocity += acceleration * self.dt
        body.position += body.velocity * self.dt
        # TODO: Integrate angular velocity and orientation


class CollisionResolver:
    """Handles collision response using penalty method."""
    
    def __init__(self, stiffness: float = 1000.0):
        self.stiffness = stiffness
    
    def resolve(self, body_a: Body, body_b: Body, distance: float, 
               normal: np.ndarray, dt: float):
        """Resolve collision between two bodies."""
        if distance >= 0:
            return  # No collision
        
        # Penalty force proportional to penetration
        penalty_force = -distance * self.stiffness
        impulse = normal * penalty_force * dt
        
        # Apply equal and opposite impulses
        body_a.velocity += impulse / body_a.mass
        body_b.velocity -= impulse / body_b.mass


# ============================================================================
# Main Physics Engine (Facade Pattern)
# ============================================================================

class PhysicsEngine:
    """Main physics engine that orchestrates all components."""
    
    def __init__(self, dt: float = 0.016, gravity: np.ndarray = np.array([0, -9.81, 0])):
        self.dt = dt
        self.gravity = gravity
        self.bodies: List[Body] = []
        
        # Initialize components
        self.integrator = Integrator(dt)
        self.collision_resolver = CollisionResolver()
        self.detector_factory = CollisionDetectorFactory()
    
    def add_body(self, body: Body):
        """Add a body to the simulation."""
        self.bodies.append(body)
    
    def step(self):
        """Perform one physics simulation step."""
        self._apply_forces()
        self._detect_and_resolve_collisions()
    
    def _apply_forces(self):
        """Apply forces and integrate motion."""
        for body in self.bodies:
            # For now, only gravity
            self.integrator.integrate(body, self.gravity)
    
    def _detect_and_resolve_collisions(self):
        """Detect and resolve all collisions."""
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                self._process_collision_pair(self.bodies[i], self.bodies[j])
    
    def _process_collision_pair(self, body_a: Body, body_b: Body):
        """Process potential collision between two bodies."""
        distance, normal = self._compute_sdf_distance(body_a, body_b)
        self.collision_resolver.resolve(body_a, body_b, distance, normal, self.dt)
    
    def _compute_sdf_distance(self, body_a: Body, body_b: Body) -> Tuple[float, np.ndarray]:
        """Compute signed distance between two bodies."""
        # Handle shape ordering
        if body_a.shape_type > body_b.shape_type:
            distance, normal = self._compute_sdf_distance(body_b, body_a)
            return distance, -normal
        
        # Get appropriate detector
        detector = self.detector_factory.get_detector(body_a.shape_type, body_b.shape_type)
        return detector.compute_distance(body_a, body_b)
    
    def get_state(self) -> np.ndarray:
        """Pack all bodies into a single array for serialization."""
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


# ============================================================================
# Scene Creation and Main Entry Point
# ============================================================================

def create_test_scene() -> PhysicsEngine:
    """Create a simple test scene with falling sphere."""
    engine = PhysicsEngine()
    
    # Falling sphere
    sphere = Body(
        position=np.array([0.0, 5.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        angular_vel=np.array([0.0, 0.0, 0.0]),
        mass=1.0,
        inertia=np.eye(3) * 0.4,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([0.5, 0.0, 0.0])  # radius = 0.5
    )
    engine.add_body(sphere)
    
    # Ground sphere (large, stationary)
    ground = Body(
        position=np.array([0.0, -100.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel=np.array([0.0, 0.0, 0.0]),
        mass=1e6,  # Very heavy
        inertia=np.eye(3) * 1e6,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([100.0, 0.0, 0.0])  # radius = 100
    )
    engine.add_body(ground)
    
    return engine


def main():
    """Main entry point for the physics engine."""
    parser = argparse.ArgumentParser(description='Physics engine reference implementation')
    parser.add_argument('--dump', type=str, help='Dump state after one step to file')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    args = parser.parse_args()
    
    # Create and run simulation
    engine = create_test_scene()
    
    for _ in range(args.steps):
        engine.step()
    
    # Output results
    if args.dump:
        state = engine.get_state()
        np.save(args.dump, state)
        print(f"Saved state to {args.dump}, shape: {state.shape}")
    else:
        for i, body in enumerate(engine.bodies):
            print(f"Body {i}: pos={body.position}, vel={body.velocity}")


if __name__ == "__main__":
    main()