"""Performance benchmarks for the physics engine.

Measures execution time of key operations to track performance regressions
and optimization opportunities.
"""
import time
import numpy as np
import pytest
from tinygrad import Tensor
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType, create_body_array

class TestPhysicsPerformance:
  """Benchmark tests for physics engine performance."""
  
  def create_large_scene(self, n_bodies: int) -> TensorPhysicsEngine:
    """Create a scene with many bodies for benchmarking."""
    np.random.seed(42)
    bodies_list = []
    
    for i in range(n_bodies):
      # Random position in a large volume
      position = np.random.uniform(-20, 20, 3).astype(np.float32)
      
      # Small random velocity
      velocity = np.random.uniform(-1, 1, 3).astype(np.float32)
      
      # Random shape type
      if i % 3 == 0:
        # Sphere
        radius = np.random.uniform(0.5, 1.0)
        body = create_body_array(
          position=position,
          velocity=velocity,
          orientation=np.array([1, 0, 0, 0], dtype=np.float32),
          angular_vel=np.zeros(3, dtype=np.float32),
          mass=1.0,
          inertia=np.eye(3, dtype=np.float32) * 0.4,
          shape_type=ShapeType.SPHERE,
          shape_params=np.array([radius, 0, 0], dtype=np.float32)
        )
      else:
        # Box
        half_extents = np.random.uniform(0.5, 1.0, 3).astype(np.float32)
        body = create_body_array(
          position=position,
          velocity=velocity,
          orientation=np.array([1, 0, 0, 0], dtype=np.float32),
          angular_vel=np.zeros(3, dtype=np.float32),
          mass=1.0,
          inertia=np.eye(3, dtype=np.float32) * 0.33,
          shape_type=ShapeType.BOX,
          shape_params=half_extents
        )
      
      bodies_list.append(body)
    
    bodies = np.stack(bodies_list)
    return TensorPhysicsEngine(bodies, gravity=np.array([0, -9.81, 0], dtype=np.float32))
  
  @pytest.mark.benchmark
  def test_physics_step_performance_small(self):
    """Benchmark physics step with 10 bodies."""
    engine = self.create_large_scene(10)
    
    # Warmup
    for _ in range(5):
      engine.step(0.016)
    
    # Benchmark
    start_time = time.perf_counter()
    n_steps = 100
    
    for _ in range(n_steps):
      engine.step(0.016)
    
    elapsed = time.perf_counter() - start_time
    avg_step_time = elapsed / n_steps * 1000  # Convert to ms
    
    print(f"\n10 bodies: {avg_step_time:.3f} ms/step ({1000/avg_step_time:.1f} Hz)")
    assert avg_step_time < 10.0, f"Step time too slow: {avg_step_time:.3f} ms"
  
  @pytest.mark.benchmark
  def test_physics_step_performance_medium(self):
    """Benchmark physics step with 50 bodies."""
    engine = self.create_large_scene(50)
    
    # Warmup
    for _ in range(5):
      engine.step(0.016)
    
    # Benchmark
    start_time = time.perf_counter()
    n_steps = 50
    
    for _ in range(n_steps):
      engine.step(0.016)
    
    elapsed = time.perf_counter() - start_time
    avg_step_time = elapsed / n_steps * 1000
    
    print(f"50 bodies: {avg_step_time:.3f} ms/step ({1000/avg_step_time:.1f} Hz)")
    assert avg_step_time < 50.0, f"Step time too slow: {avg_step_time:.3f} ms"
  
  @pytest.mark.benchmark
  def test_physics_step_performance_large(self):
    """Benchmark physics step with 200 bodies."""
    engine = self.create_large_scene(200)
    
    # Warmup
    for _ in range(5):
      engine.step(0.016)
    
    # Benchmark
    start_time = time.perf_counter()
    n_steps = 20
    
    for _ in range(n_steps):
      engine.step(0.016)
    
    elapsed = time.perf_counter() - start_time
    avg_step_time = elapsed / n_steps * 1000
    
    print(f"200 bodies: {avg_step_time:.3f} ms/step ({1000/avg_step_time:.1f} Hz)")
    # More lenient for large scenes
    assert avg_step_time < 200.0, f"Step time too slow: {avg_step_time:.3f} ms"
  
  @pytest.mark.benchmark
  def test_differentiable_broadphase_performance(self):
    """Benchmark the differentiable broadphase specifically."""
    from physics.broadphase_tensor import differentiable_broadphase
    
    # Test with different body counts
    body_counts = [10, 50, 100, 200]
    
    for n_bodies in body_counts:
      # Create random bodies
      bodies_np = np.random.randn(n_bodies, BodySchema.NUM_PROPERTIES).astype(np.float32)
      bodies = Tensor(bodies_np)
      
      # Warmup
      for _ in range(3):
        differentiable_broadphase(bodies)
      
      # Benchmark
      start_time = time.perf_counter()
      n_iterations = 20
      
      for _ in range(n_iterations):
        pair_indices, collision_mask = differentiable_broadphase(bodies)
      
      elapsed = time.perf_counter() - start_time
      avg_time = elapsed / n_iterations * 1000
      
      n_pairs = n_bodies * (n_bodies - 1) // 2
      print(f"Broadphase {n_bodies} bodies ({n_pairs} pairs): {avg_time:.3f} ms")
  
  @pytest.mark.benchmark
  def test_narrowphase_performance(self):
    """Benchmark narrowphase collision detection."""
    from physics.narrowphase import narrowphase
    
    # Create a scene where many bodies are colliding
    bodies_list = []
    
    # Create a grid of spheres that are touching
    grid_size = 5
    for i in range(grid_size):
      for j in range(grid_size):
        body = create_body_array(
          position=np.array([i * 2, 0, j * 2], dtype=np.float32),
          velocity=np.zeros(3, dtype=np.float32),
          orientation=np.array([1, 0, 0, 0], dtype=np.float32),
          angular_vel=np.zeros(3, dtype=np.float32),
          mass=1.0,
          inertia=np.eye(3, dtype=np.float32) * 0.4,
          shape_type=ShapeType.SPHERE,
          shape_params=np.array([1.1, 0, 0], dtype=np.float32)  # Slightly overlapping
        )
        bodies_list.append(body)
    
    bodies = Tensor(np.stack(bodies_list))
    
    # All adjacent pairs should collide
    from physics.broadphase_tensor import differentiable_broadphase
    pair_indices, collision_mask = differentiable_broadphase(bodies)
    
    # Benchmark narrowphase
    start_time = time.perf_counter()
    n_iterations = 50
    
    for _ in range(n_iterations):
      contacts = narrowphase(bodies, pair_indices, collision_mask)
    
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / n_iterations * 1000
    
    n_potential_collisions = int(collision_mask.numpy().sum())
    print(f"Narrowphase {len(bodies_list)} bodies, {n_potential_collisions} potential collisions: {avg_time:.3f} ms")