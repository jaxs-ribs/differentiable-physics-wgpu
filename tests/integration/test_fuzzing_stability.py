"""Property-based fuzzing tests for physics engine stability.

Uses the hypothesis library to generate random physics scenes and verify
that fundamental physical laws and stability properties hold across a
wide variety of configurations.
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from tinygrad import Tensor
from physics.engine import TensorPhysicsEngine
from physics.types import BodySchema, ShapeType, create_body_array

# Strategies for generating random but valid physics data
position_strategy = st.tuples(
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
)

velocity_strategy = st.tuples(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)

# Generate valid quaternions (must be normalized)
@st.composite
def quaternion_strategy(draw):
    """Generate a valid normalized quaternion."""
    q = draw(st.tuples(
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False)
    ))
    # Normalize
    norm = np.sqrt(sum(x*x for x in q))
    assume(norm > 1e-6)  # Avoid division by zero
    return tuple(x/norm for x in q)

@st.composite
def body_strategy(draw):
    """Generate a random but valid physics body."""
    # Choose shape type
    shape_type = draw(st.sampled_from([ShapeType.SPHERE, ShapeType.BOX]))
    
    # Generate shape parameters based on type
    if shape_type == ShapeType.SPHERE:
        radius = draw(st.floats(min_value=0.1, max_value=3.0, allow_nan=False))
        shape_params = (radius, 0.0, 0.0)
        mass = 4/3 * np.pi * radius**3  # Proportional to volume
        inertia = np.eye(3, dtype=np.float32) * (2/5 * mass * radius**2)
    else:  # BOX
        half_extents = draw(st.tuples(
            st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
            st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
            st.floats(min_value=0.1, max_value=3.0, allow_nan=False)
        ))
        shape_params = half_extents
        volume = 8 * np.prod(half_extents)
        mass = volume
        inertia = np.diag([
            mass/12 * (4*half_extents[1]**2 + 4*half_extents[2]**2),
            mass/12 * (4*half_extents[0]**2 + 4*half_extents[2]**2),
            mass/12 * (4*half_extents[0]**2 + 4*half_extents[1]**2)
        ]).astype(np.float32)
    
    # Generate other properties
    position = np.array(draw(position_strategy), dtype=np.float32)
    velocity = np.array(draw(velocity_strategy), dtype=np.float32)
    orientation = np.array(draw(quaternion_strategy()), dtype=np.float32)
    angular_vel = np.array(draw(velocity_strategy), dtype=np.float32) * 0.1  # Keep angular vel reasonable
    
    return create_body_array(
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_vel=angular_vel,
        mass=float(mass),
        inertia=inertia,
        shape_type=shape_type,
        shape_params=np.array(shape_params, dtype=np.float32)
    )

@st.composite
def scene_strategy(draw):
    """Generate a random physics scene with multiple bodies."""
    n_bodies = draw(st.integers(min_value=2, max_value=20))
    bodies = [draw(body_strategy()) for _ in range(n_bodies)]
    return np.stack(bodies)

class TestFuzzingStability:
    """Property-based tests that verify stability across random scenes."""
    
    def check_finite_values(self, bodies: Tensor) -> bool:
        """Check that all values in the state are finite."""
        bodies_np = bodies.numpy()
        return np.all(np.isfinite(bodies_np))
    
    def check_reasonable_velocities(self, bodies: Tensor, max_vel: float = 100.0) -> bool:
        """Check that velocities haven't exploded."""
        bodies_np = bodies.numpy()
        velocities = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        speeds = np.linalg.norm(velocities, axis=1)
        return np.all(speeds < max_vel)
    
    def check_reasonable_positions(self, bodies: Tensor, max_dist: float = 1000.0) -> bool:
        """Check that positions are within reasonable bounds."""
        bodies_np = bodies.numpy()
        positions = bodies_np[:, BodySchema.POS_X:BodySchema.POS_Z+1]
        distances = np.linalg.norm(positions, axis=1)
        return np.all(distances < max_dist)
    
    def calculate_total_momentum(self, bodies: Tensor) -> np.ndarray:
        """Calculate total linear momentum of the system."""
        bodies_np = bodies.numpy()
        velocities = bodies_np[:, BodySchema.VEL_X:BodySchema.VEL_Z+1]
        inv_masses = bodies_np[:, BodySchema.INV_MASS]
        masses = np.where(inv_masses > 0, 1.0 / inv_masses, 0.0)
        return np.sum(velocities * masses[:, np.newaxis], axis=0)
    
    @given(scene_strategy())
    @settings(max_examples=100, deadline=5000)  # Run 100 random scenes
    def test_fuzz_simulation_stability(self, bodies_np):
        """Test that random scenes remain stable and obey physical laws."""
        # Create engine
        engine = TensorPhysicsEngine(
            bodies_np, 
            gravity=np.array([0, -9.81, 0], dtype=np.float32),
            use_differentiable=True  # Test the differentiable broadphase
        )
        
        # Initial checks
        assert self.check_finite_values(engine.bodies), "Initial state has non-finite values"
        
        # Calculate initial momentum (for zero gravity test)
        engine.gravity = Tensor(np.zeros(3, dtype=np.float32))
        initial_momentum = self.calculate_total_momentum(engine.bodies)
        
        # Run simulation
        for step in range(50):
            engine.step(0.01)
            
            # Check stability invariants
            assert self.check_finite_values(engine.bodies), f"Non-finite values at step {step}"
            assert self.check_reasonable_velocities(engine.bodies), f"Velocity explosion at step {step}"
            assert self.check_reasonable_positions(engine.bodies), f"Position explosion at step {step}"
        
        # Check momentum conservation (should be approximately conserved in zero gravity)
        final_momentum = self.calculate_total_momentum(engine.bodies)
        momentum_change = np.linalg.norm(final_momentum - initial_momentum)
        initial_momentum_mag = np.linalg.norm(initial_momentum)
        
        # Allow 5% change due to numerical errors
        if initial_momentum_mag > 0.1:  # Only check if there's significant initial momentum
            relative_change = momentum_change / initial_momentum_mag
            assert relative_change < 0.05, f"Momentum not conserved: {relative_change*100:.1f}% change"
    
    @given(
        n_bodies=st.integers(min_value=50, max_value=100),
        density=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=20, deadline=10000)
    def test_fuzz_dense_scenes(self, n_bodies, density):
        """Test stability with many bodies in close proximity."""
        # Create bodies in a dense configuration
        bodies_list = []
        grid_size = int(np.ceil(n_bodies**(1/3)))
        
        spacing = 2.0 / density  # Closer spacing = higher density
        
        body_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if body_idx >= n_bodies:
                        break
                    
                    position = np.array([i * spacing, j * spacing, k * spacing], dtype=np.float32)
                    
                    # Alternate between spheres and boxes
                    if body_idx % 2 == 0:
                        body = create_body_array(
                            position=position,
                            velocity=np.zeros(3, dtype=np.float32),
                            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
                            angular_vel=np.zeros(3, dtype=np.float32),
                            mass=1.0,
                            inertia=np.eye(3, dtype=np.float32) * 0.4,
                            shape_type=ShapeType.SPHERE,
                            shape_params=np.array([0.5, 0, 0], dtype=np.float32)
                        )
                    else:
                        body = create_body_array(
                            position=position,
                            velocity=np.zeros(3, dtype=np.float32),
                            orientation=np.array([1, 0, 0, 0], dtype=np.float32),
                            angular_vel=np.zeros(3, dtype=np.float32),
                            mass=1.0,
                            inertia=np.eye(3, dtype=np.float32) * 0.33,
                            shape_type=ShapeType.BOX,
                            shape_params=np.array([0.4, 0.4, 0.4], dtype=np.float32)
                        )
                    
                    bodies_list.append(body)
                    body_idx += 1
        
        bodies_np = np.stack(bodies_list[:n_bodies])
        engine = TensorPhysicsEngine(bodies_np, gravity=np.array([0, -9.81, 0], dtype=np.float32))
        
        # Run simulation - dense scenes are more challenging
        max_steps = 30
        for step in range(max_steps):
            engine.step(0.01)
            
            # Check stability
            assert self.check_finite_values(engine.bodies), f"Non-finite values at step {step}"
            assert self.check_reasonable_velocities(engine.bodies, max_vel=50.0), f"Velocity explosion at step {step}"
    
    @given(
        timestep=st.floats(min_value=1e-6, max_value=0.1, allow_nan=False),
        n_steps=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20, deadline=5000)
    def test_fuzz_timestep_stability(self, timestep, n_steps):
        """Test stability across different timesteps."""
        # Create a simple test scene
        bodies_list = []
        for i in range(5):
            body = create_body_array(
                position=np.array([i * 3, 0, 0], dtype=np.float32),
                velocity=np.random.uniform(-2, 2, 3).astype(np.float32),
                orientation=np.array([1, 0, 0, 0], dtype=np.float32),
                angular_vel=np.zeros(3, dtype=np.float32),
                mass=1.0,
                inertia=np.eye(3, dtype=np.float32),
                shape_type=ShapeType.SPHERE,
                shape_params=np.array([1.0, 0, 0], dtype=np.float32)
            )
            bodies_list.append(body)
        
        bodies_np = np.stack(bodies_list)
        engine = TensorPhysicsEngine(bodies_np, gravity=np.array([0, -9.81, 0], dtype=np.float32))
        
        # Run with given timestep
        for step in range(n_steps):
            engine.step(timestep)
            
            # Stability checks
            assert self.check_finite_values(engine.bodies), f"Non-finite values with dt={timestep} at step {step}"
            assert self.check_reasonable_positions(engine.bodies), f"Position explosion with dt={timestep}"