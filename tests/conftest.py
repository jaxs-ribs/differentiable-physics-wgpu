"""Test fixtures for physics engine tests.

Provides reusable scenes and test data for unit and integration tests.
"""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType, create_body_array

@pytest.fixture
def two_body_scene():
  """Create a physics engine with two bodies positioned for interaction.
  
  Returns:
    TensorPhysicsEngine with:
    - Body 0: Sphere at origin, radius 1, moving right
    - Body 1: Box at (3, 0, 0), stationary
  """
  bodies_list = []
  
  # Sphere moving right
  sphere = create_body_array(
    position=np.array([0, 0, 0], dtype=np.float32),
    velocity=np.array([2, 0, 0], dtype=np.float32),  # moving right
    orientation=np.array([1, 0, 0, 0], dtype=np.float32),
    angular_vel=np.array([0, 0, 0], dtype=np.float32),
    mass=1.0,
    inertia=np.eye(3, dtype=np.float32) * 2/5,  # sphere inertia
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([1.0, 0, 0], dtype=np.float32)
  )
  bodies_list.append(sphere)
  
  # Stationary box
  box = create_body_array(
    position=np.array([3, 0, 0], dtype=np.float32),
    velocity=np.array([0, 0, 0], dtype=np.float32),
    orientation=np.array([1, 0, 0, 0], dtype=np.float32),
    angular_vel=np.array([0, 0, 0], dtype=np.float32),
    mass=2.0,
    inertia=np.eye(3, dtype=np.float32) * 2/3,  # box inertia
    shape_type=ShapeType.BOX,
    shape_params=np.array([1.0, 1.0, 1.0], dtype=np.float32)
  )
  bodies_list.append(box)
  
  bodies = np.stack(bodies_list)
  engine = TensorPhysicsEngine(bodies, gravity=np.array([0, -9.81, 0], dtype=np.float32))
  return engine

@pytest.fixture
def multi_body_stack_scene():
  """Create a physics engine with a stack of boxes to test stability.
  
  Returns:
    TensorPhysicsEngine with ground + 5 boxes stacked vertically
  """
  bodies_list = []
  
  # Add ground plane
  ground = create_body_array(
    position=np.array([0, -2, 0], dtype=np.float32),
    velocity=np.array([0, 0, 0], dtype=np.float32),
    orientation=np.array([1, 0, 0, 0], dtype=np.float32),
    angular_vel=np.array([0, 0, 0], dtype=np.float32),
    mass=1e8,  # Static
    inertia=np.eye(3, dtype=np.float32) * 1e8,
    shape_type=ShapeType.BOX,
    shape_params=np.array([10.0, 0.5, 10.0], dtype=np.float32)
  )
  bodies_list.append(ground)
  
  # Stack of boxes
  box_size = 1.0
  for i in range(5):
    y_pos = i * (box_size * 2 + 0.1) + 0.5  # Start above ground
    box = create_body_array(
      position=np.array([0, y_pos, 0], dtype=np.float32),
      velocity=np.array([0, 0, 0], dtype=np.float32),
      orientation=np.array([1, 0, 0, 0], dtype=np.float32),
      angular_vel=np.array([0, 0, 0], dtype=np.float32),
      mass=1.0,
      inertia=np.eye(3, dtype=np.float32) * 1/3,
      shape_type=ShapeType.BOX,
      shape_params=np.array([box_size, box_size, box_size], dtype=np.float32)
    )
    bodies_list.append(box)
  
  bodies = np.stack(bodies_list)
  # Use lower restitution to help with stability
  engine = TensorPhysicsEngine(bodies, gravity=np.array([0, -9.81, 0], dtype=np.float32), restitution=0.1)
  return engine

@pytest.fixture
def random_bodies_scene():
  """Create a physics engine with many randomly positioned bodies.
  
  Returns:
    TensorPhysicsEngine with 20 mixed spheres and boxes
  """
  np.random.seed(42)  # for reproducible tests
  bodies_list = []
  
  for i in range(20):
    # Random position in a 10x10x10 volume
    position = np.random.uniform(-5, 5, 3).astype(np.float32)
    
    # Random velocity
    velocity = np.random.uniform(-2, 2, 3).astype(np.float32)
    
    # Random rotation (normalized quaternion)
    quat = np.random.randn(4).astype(np.float32)
    quat = quat / np.linalg.norm(quat)
    
    # Alternate between spheres and boxes
    if i % 2 == 0:
      # Sphere
      radius = np.random.uniform(0.5, 1.5)
      body = create_body_array(
        position=position,
        velocity=velocity,
        orientation=quat,
        angular_vel=np.array([0, 0, 0], dtype=np.float32),
        mass=radius**3 * 4/3 * np.pi,  # proportional to volume
        inertia=np.eye(3, dtype=np.float32) * 2/5 * radius**2,
        shape_type=ShapeType.SPHERE,
        shape_params=np.array([radius, 0, 0], dtype=np.float32)
      )
    else:
      # Box
      half_extents = np.random.uniform(0.5, 1.5, 3).astype(np.float32)
      volume = 8 * np.prod(half_extents)
      body = create_body_array(
        position=position,
        velocity=velocity,
        orientation=quat,
        angular_vel=np.array([0, 0, 0], dtype=np.float32),
        mass=volume,
        inertia=np.diag(half_extents**2 * volume / 3).astype(np.float32),
        shape_type=ShapeType.BOX,
        shape_params=half_extents
      )
    
    bodies_list.append(body)
  
  bodies = np.stack(bodies_list)
  engine = TensorPhysicsEngine(bodies, gravity=np.array([0, -9.81, 0], dtype=np.float32))
  return engine