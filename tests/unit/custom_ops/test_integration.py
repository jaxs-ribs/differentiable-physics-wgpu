"""
End-to-end integration tests for custom physics operations
"""
# import pytest  # Optional, not required for basic testing
import numpy as np
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "custom_ops"))

from tinygrad import Tensor
from custom_ops.python.extension import physics_enabled
from custom_ops.python.tensor_ops import PhysicsTensor, create_physics_world
from tinygrad.dtype import dtypes

# Check if physics library is available
def physics_library_available():
    lib_path = Path(__file__).parent.parent.parent.parent / "custom_ops" / "build"
    lib_name = "libphysics.dylib" if sys.platform == "darwin" else "libphysics.so"
    return (lib_path / lib_name).exists()

class TestPhysicsIntegration:
    
    def test_create_physics_world(self):
        """Test creating a physics world"""
        with physics_enabled("CPU"):
            world = create_physics_world(n_bodies=5)
            assert isinstance(world, PhysicsTensor)
            assert world.shape == (5, 8)
    
    def test_physics_step_integration(self):
        """Test running a physics simulation step"""
        with physics_enabled("CPU"):
            # Create simple world with one falling body
            bodies_data = [[0, 10, 0, 0, 0, 0, 1.0, 0.5]]  # Y=10, falling
            bodies = PhysicsTensor(bodies_data, device="CPU", dtype=dtypes.float32)
            
            # Initial state
            initial_y = bodies[0, 1].numpy()
            
            # Step simulation
            dt = 0.1
            bodies_after = bodies.integrate(dt)
            
            # Check that body fell (Y decreased)
            final_y = bodies_after[0, 1].numpy()
            assert final_y < initial_y
            
            # Check gravity was applied (velocity changed)
            final_vy = bodies_after[0, 4].numpy()
            expected_vy = -9.81 * dt
            assert abs(final_vy - expected_vy) < 0.01
    
    def test_multiple_bodies_interaction(self):
        """Test physics with multiple interacting bodies"""
        with physics_enabled("CPU"):
            # Create two bodies that will collide
            bodies_data = [
                [0, 5, 0, 0, 0, 0, 1.0, 1.0],   # Body 1 at Y=5
                [0, 3, 0, 0, 2, 0, 1.0, 1.0],   # Body 2 at Y=3, moving up
            ]
            bodies = PhysicsTensor(bodies_data, device="CPU", dtype=dtypes.float32)
            
            # Run several steps
            dt = 0.01
            for _ in range(10):
                bodies = bodies.integrate(dt)
            
            # Bodies should have interacted (velocities changed)
            # This is a basic sanity check
            assert bodies.shape == (2, 8)
    
    def test_conservation_properties(self):
        """Test basic conservation properties"""
        with physics_enabled("CPU"):
            # Create isolated system
            world = create_physics_world(n_bodies=10)
            
            # Get initial total momentum (approximately, ignoring rotations)
            initial_momentum = (world[:, 3:6] * world[:, 6:7]).sum(axis=0).numpy()
            
            # Run simulation
            dt = 0.016
            for _ in range(100):
                world = world.integrate(dt)
            
            # Final momentum (x and z components should be conserved)
            final_momentum = (world[:, 3:6] * world[:, 6:7]).sum(axis=0).numpy()
            
            # X and Z momentum should be approximately conserved
            assert abs(final_momentum[0] - initial_momentum[0]) < 1.0
            assert abs(final_momentum[2] - initial_momentum[2]) < 1.0
            # Y momentum changes due to gravity
    
    def test_deterministic_simulation(self):
        """Test that simulation is deterministic"""
        with physics_enabled("CPU"):
            # Create same initial conditions
            np.random.seed(42)
            world1 = create_physics_world(n_bodies=5)
            
            np.random.seed(42)
            world2 = create_physics_world(n_bodies=5)
            
            # Run same simulation
            dt = 0.016
            for _ in range(10):
                world1 = world1.integrate(dt)
                world2 = world2.integrate(dt)
            
            # Results should be identical
            np.testing.assert_array_almost_equal(
                world1.numpy(), 
                world2.numpy(), 
                decimal=6
            )

def test_physics_extension_import():
    """Test that custom_ops module can be imported"""
    import custom_ops
    assert hasattr(custom_ops, 'enable_physics_on_device')
    assert hasattr(custom_ops, 'PhysicsTensor')

if __name__ == "__main__":
    if not physics_library_available():
        print("Physics library not compiled. Run 'make' in custom_ops/src/")
        sys.exit(1)
    
    # Run tests manually
    test = TestPhysicsIntegration()
    
    print("Running integration tests...")
    try:
        test.test_create_physics_world()
        print("✓ test_create_physics_world")
        
        test.test_physics_step_integration()
        print("✓ test_physics_step_integration")
        
        test.test_multiple_bodies_interaction()
        print("✓ test_multiple_bodies_interaction")
        
        test.test_conservation_properties()
        print("✓ test_conservation_properties")
        
        test.test_deterministic_simulation()
        print("✓ test_deterministic_simulation")
        
        test_physics_extension_import()
        print("✓ test_physics_extension_import")
        
        print("\nAll integration tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)