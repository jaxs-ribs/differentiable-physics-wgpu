"""Test fixtures for physics engine tests.

Provides reusable scenes and test data for unit and integration tests.
"""
import pytest
import numpy as np
from physics.engine import TensorPhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

@pytest.fixture
def two_body_scene():
  """Create a physics engine with two bodies positioned for interaction.
  
  Returns:
    TensorPhysicsEngine with:
    - Body 0: Sphere at origin, radius 1, moving right
    - Body 1: Box at (3, 0, 0), stationary
  """
  builder = SceneBuilder()
  
  # Sphere at origin, moving right
  builder.add_body(
    position=[0, 0, 0],
    velocity=[2, 0, 0],  # Moving right
    mass=1.0,
    shape_type=ShapeType.SPHERE,
    shape_params=[1.0, 0, 0]  # Radius = 1
  )
  
  # Stationary box
  builder.add_body(
    position=[3, 0, 0],
    mass=2.0,
    shape_type=ShapeType.BOX,
    shape_params=[1.0, 1.0, 1.0]  # Unit cube
  )
  
  soa_data = builder.build()
  engine = TensorPhysicsEngine(
    x=soa_data['x'],
    q=soa_data['q'],
    v=soa_data['v'],
    omega=soa_data['omega'],
    inv_mass=soa_data['inv_mass'],
    inv_inertia=soa_data['inv_inertia'],
    shape_type=soa_data['shape_type'],
    shape_params=soa_data['shape_params'],
    gravity=np.array([0, -9.81, 0], dtype=np.float32)
  )
  return engine

@pytest.fixture
def multi_body_stack_scene():
  """Create a physics engine with a stack of boxes to test stability.
  
  Returns:
    TensorPhysicsEngine with ground + 5 boxes stacked vertically
  """
  builder = SceneBuilder()
  
  # Ground plane
  builder.add_body(
    position=[0, -2, 0],
    mass=1e8,  # Static
    shape_type=ShapeType.BOX,
    shape_params=[10.0, 0.5, 10.0]  # Large flat ground
  )
  
  # Stack of boxes
  box_size = 1.0
  for i in range(5):
    y_pos = i * (box_size * 2 + 0.1) + 0.5  # Start above ground
    builder.add_body(
      position=[0, y_pos, 0],
      mass=1.0,
      shape_type=ShapeType.BOX,
      shape_params=[box_size, box_size, box_size]
    )
  
  soa_data = builder.build()
  engine = TensorPhysicsEngine(
    x=soa_data['x'],
    q=soa_data['q'],
    v=soa_data['v'],
    omega=soa_data['omega'],
    inv_mass=soa_data['inv_mass'],
    inv_inertia=soa_data['inv_inertia'],
    shape_type=soa_data['shape_type'],
    shape_params=soa_data['shape_params'],
    gravity=np.array([0, -9.81, 0], dtype=np.float32),
    restitution=0.1
  )
  return engine

@pytest.fixture
def random_bodies_scene():
  """Create a physics engine with many randomly positioned bodies.
  
  Returns:
    TensorPhysicsEngine with 20 mixed spheres and boxes
  """
  np.random.seed(42)  # for reproducible tests
  builder = SceneBuilder()
  
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
      mass = radius**3 * 4/3 * np.pi  # proportional to volume
      builder.add_body(
        position=position,
        velocity=velocity,
        orientation=quat,
        mass=mass,
        shape_type=ShapeType.SPHERE,
        shape_params=[radius, 0, 0]
      )
    else:
      # Box
      half_extents = np.random.uniform(0.5, 1.5, 3).astype(np.float32)
      volume = 8 * np.prod(half_extents)
      builder.add_body(
        position=position,
        velocity=velocity,
        orientation=quat,
        mass=volume,
        shape_type=ShapeType.BOX,
        shape_params=half_extents
      )
  
  soa_data = builder.build()
  engine = TensorPhysicsEngine(
    x=soa_data['x'],
    q=soa_data['q'],
    v=soa_data['v'],
    omega=soa_data['omega'],
    inv_mass=soa_data['inv_mass'],
    inv_inertia=soa_data['inv_inertia'],
    shape_type=soa_data['shape_type'],
    shape_params=soa_data['shape_params'],
    gravity=np.array([0, -9.81, 0], dtype=np.float32)
  )
  return engine