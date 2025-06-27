"""Unit tests for broadphase collision detection.

Tests the differentiable all-pairs broadphase implementation.
"""
import numpy as np
import pytest
from tinygrad import Tensor
from physics.types import BodySchema, ShapeType, create_body_array
from physics.broadphase_tensor import differentiable_broadphase

class TestDifferentiableBroadphase:
  """Test the differentiable broadphase collision detection."""
  
  def test_all_pairs_generation(self):
    """Verify correct number and structure of pairs for N bodies."""
    # Create 5 bodies
    bodies_list = []
    for i in range(5):
      body = create_body_array(
        position=np.array([i * 10, 0, 0], dtype=np.float32),  # far apart
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1.0, 0, 0], dtype=np.float32)
      )
      bodies_list.append(body)
    
    bodies = Tensor(np.stack(bodies_list))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # Check shape: should be (5*4/2, 2) = (10, 2)
    assert pair_indices.shape == (10, 2)
    assert collision_mask.shape == (10,)
    
    # Check all unique pairs are present
    pairs_np = pair_indices.numpy()
    expected_pairs = []
    for i in range(5):
      for j in range(i + 1, 5):
        expected_pairs.append((i, j))
    
    # Convert to sets for comparison
    actual_pairs = set(map(tuple, pairs_np))
    expected_pairs = set(expected_pairs)
    assert actual_pairs == expected_pairs
  
  def test_aabb_collision_mask_no_collision(self):
    """Test that far-apart bodies don't collide."""
    bodies_list = []
    
    # Create 3 spheres far apart
    for i in range(3):
      body = create_body_array(
        position=np.array([i * 100, 0, 0], dtype=np.float32),  # 100 units apart
        velocity=np.zeros(3, dtype=np.float32),
        orientation=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_vel=np.zeros(3, dtype=np.float32),
        mass=1.0,
        inertia=np.eye(3, dtype=np.float32),
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([1.0, 0, 0], dtype=np.float32)  # radius 1
      )
      bodies_list.append(body)
    
    bodies = Tensor(np.stack(bodies_list))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # No collisions expected
    assert not np.any(collision_mask.numpy())
  
  def test_aabb_collision_mask_guaranteed_collision(self):
    """Test that overlapping bodies are detected."""
    bodies_list = []
    
    # Two overlapping spheres
    body1 = create_body_array(
      position=np.array([0, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([2.0, 0, 0], dtype=np.float32)  # radius 2
    )
    bodies_list.append(body1)
    
    body2 = create_body_array(
      position=np.array([3, 0, 0], dtype=np.float32),  # centers 3 units apart
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([2.0, 0, 0], dtype=np.float32)  # radius 2
    )
    bodies_list.append(body2)
    
    # Add a third sphere far away
    body3 = create_body_array(
      position=np.array([100, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([1.0, 0, 0], dtype=np.float32)
    )
    bodies_list.append(body3)
    
    bodies = Tensor(np.stack(bodies_list))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # Check collisions
    pairs_np = pair_indices.numpy()
    mask_np = collision_mask.numpy()
    
    # Find which pair is (0,1)
    collision_found = False
    for i, pair in enumerate(pairs_np):
      if (pair[0] == 0 and pair[1] == 1) or (pair[0] == 1 and pair[1] == 0):
        assert mask_np[i], "Bodies 0 and 1 should be colliding"
        collision_found = True
      elif 2 in pair:
        assert not mask_np[i], f"Body 2 should not collide with body {pair[0] if pair[1] == 2 else pair[1]}"
    
    assert collision_found, "Collision between bodies 0 and 1 not found in pairs"
  
  def test_box_collision_detection(self):
    """Test AABB collision detection with boxes."""
    bodies_list = []
    
    # Two overlapping boxes
    box1 = create_body_array(
      position=np.array([0, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.BOX,
      shape_params=np.array([1.0, 1.0, 1.0], dtype=np.float32)  # half-extents
    )
    bodies_list.append(box1)
    
    # Box that overlaps slightly on X axis
    box2 = create_body_array(
      position=np.array([1.5, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.BOX,
      shape_params=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    bodies_list.append(box2)
    
    bodies = Tensor(np.stack(bodies_list))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # Should detect collision
    assert np.any(collision_mask.numpy()), "Overlapping boxes should be detected"
  
  def test_rotated_box_collision(self):
    """Test AABB generation for rotated boxes."""
    bodies_list = []
    
    # Box rotated 45 degrees about Z axis
    angle = np.pi / 4
    quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)], dtype=np.float32)
    
    rotated_box = create_body_array(
      position=np.array([0, 0, 0], dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=quat,
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.BOX,
      shape_params=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    bodies_list.append(rotated_box)
    
    # Sphere that should collide with the rotated box's AABB
    sphere = create_body_array(
      position=np.array([1.3, 1.3, 0], dtype=np.float32),  # near corner of rotated box
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([0.5, 0, 0], dtype=np.float32)
    )
    bodies_list.append(sphere)
    
    bodies = Tensor(np.stack(bodies_list))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # The rotated box's AABB should extend to about ±1.414 on X and Y
    assert np.any(collision_mask.numpy()), "Sphere should collide with rotated box AABB"
  
  def test_empty_scene(self):
    """Test broadphase with no bodies."""
    bodies = Tensor(np.zeros((0, BodySchema.NUM_PROPERTIES), dtype=np.float32))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    assert pair_indices.shape == (0, 2)
    assert collision_mask.shape == (0,)
  
  def test_single_body(self):
    """Test broadphase with single body (no pairs)."""
    body = create_body_array(
      position=np.zeros(3, dtype=np.float32),
      velocity=np.zeros(3, dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.zeros(3, dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32),
      shape_type=ShapeType.SPHERE,
      shape_params=np.array([1.0, 0, 0], dtype=np.float32)
    )
    
    bodies = Tensor(body.reshape(1, -1))
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    assert pair_indices.shape == (0, 2)
    assert collision_mask.shape == (0,)