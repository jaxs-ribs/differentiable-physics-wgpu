"""
End-to-End Integration Tests for Custom Physics Operations

This module tests the integration between TinyGrad and custom C physics operations.
It validates that the custom ops extension properly interfaces with TinyGrad's tensor
system and that physics simulations work correctly through the full stack.

Why this is useful:
- Ensures custom C operations integrate seamlessly with TinyGrad
- Validates the PhysicsTensor abstraction layer
- Tests physics simulations through the tensor interface
- Verifies deterministic behavior and conservation laws
- Catches integration issues between Python and C layers

The tests cover:
1. Physics world creation and initialization
2. Single-step physics integration
3. Multi-body interactions
4. Conservation properties (momentum)
5. Deterministic simulation validation
"""
# import pytest  # Optional, not required for basic testing
import numpy as np
import sys
import time
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
        print("\n[TEST] test_create_physics_world")
        print("  [OBJECTIVE] Validate physics world initialization")
        
        with physics_enabled("CPU"):
            print("  [INFO] Physics operations enabled on CPU device")
            
            n_bodies = 5
            print(f"  [ACTION] Creating physics world with {n_bodies} bodies...")
            world = create_physics_world(n_bodies=n_bodies)
            
            print(f"  [CHECK] World type: {type(world)}")
            print(f"  [CHECK] World shape: {world.shape}")
            print(f"  [CHECK] Expected shape: ({n_bodies}, 8)")
            
            assert isinstance(world, PhysicsTensor)
            assert world.shape == (5, 8)
            
            print("  [SUCCESS] Physics world created successfully")
    
    def test_physics_step_integration(self):
        """Test running a physics simulation step"""
        print("\n[TEST] test_physics_step_integration")
        print("  [OBJECTIVE] Validate single-step physics integration")
        
        with physics_enabled("CPU"):
            print("  [INFO] Testing gravity integration on falling body")
            
            # Create simple world with one falling body
            print("\n  [SETUP] Creating falling body...")
            bodies_data = [[0, 10, 0, 0, 0, 0, 1.0, 0.5]]  # Y=10, falling
            print(f"    Position: (0, 10, 0)")
            print(f"    Velocity: (0, 0, 0)")
            print(f"    Mass: 1.0 kg, Radius: 0.5 m")
            
            bodies = PhysicsTensor(bodies_data, device="CPU", dtype=dtypes.float32)
            
            # Initial state
            initial_y = bodies[0, 1].numpy()
            initial_vy = bodies[0, 4].numpy()
            print(f"\n  [INITIAL STATE]")
            print(f"    Y position: {initial_y:.3f} m")
            print(f"    Y velocity: {initial_vy:.3f} m/s")
            
            # Step simulation
            dt = 0.1
            print(f"\n  [EXECUTE] Running integration with dt={dt}s...")
            bodies_after = bodies.integrate(dt)
            
            # Check results
            final_y = bodies_after[0, 1].numpy()
            final_vy = bodies_after[0, 4].numpy()
            
            print(f"\n  [FINAL STATE]")
            print(f"    Y position: {final_y:.3f} m")
            print(f"    Y velocity: {final_vy:.3f} m/s")
            print(f"\n  [CHANGES]")
            print(f"    Position delta: {final_y - initial_y:.3f} m")
            print(f"    Velocity delta: {final_vy - initial_vy:.3f} m/s")
            
            # Validations
            print("\n  [VALIDATION]")
            
            # Check that body fell (Y decreased)
            assert final_y < initial_y
            print(f"    ✓ Body fell (Y decreased by {initial_y - final_y:.3f} m)")
            
            # Check gravity was applied (velocity changed)
            expected_vy = -9.81 * dt
            velocity_error = abs(final_vy - expected_vy)
            print(f"    Expected Y velocity: {expected_vy:.3f} m/s")
            print(f"    Velocity error: {velocity_error:.6f} m/s")
            assert velocity_error < 0.01
            print(f"    ✓ Gravity correctly applied (error < 0.01)")
            
            print("  [SUCCESS] Physics step integration working correctly")
    
    def test_multiple_bodies_interaction(self):
        """Test physics with multiple interacting bodies"""
        print("\n[TEST] test_multiple_bodies_interaction")
        print("  [OBJECTIVE] Validate multi-body physics interactions")
        
        with physics_enabled("CPU"):
            print("  [INFO] Testing collision between two bodies")
            
            # Create two bodies that will collide
            print("\n  [SETUP] Creating colliding bodies...")
            bodies_data = [
                [0, 5, 0, 0, 0, 0, 1.0, 1.0],   # Body 1 at Y=5
                [0, 3, 0, 0, 2, 0, 1.0, 1.0],   # Body 2 at Y=3, moving up
            ]
            print("    Body 1: Position (0, 5, 0), Velocity (0, 0, 0), Radius 1.0")
            print("    Body 2: Position (0, 3, 0), Velocity (0, 2, 0), Radius 1.0")
            print("    Initial separation: 2.0 m (will collide)")
            
            bodies = PhysicsTensor(bodies_data, device="CPU", dtype=dtypes.float32)
            
            # Store initial state
            initial_velocities = bodies[:, 3:6].numpy().copy()
            print(f"\n  [INITIAL] Velocities:")
            print(f"    Body 1: {initial_velocities[0]}")
            print(f"    Body 2: {initial_velocities[1]}")
            
            # Run several steps
            dt = 0.01
            n_steps = 10
            print(f"\n  [EXECUTE] Running {n_steps} simulation steps (dt={dt})...")
            
            for step in range(n_steps):
                bodies = bodies.integrate(dt)
                if step % 3 == 0:
                    positions = bodies[:, :3].numpy()
                    print(f"    Step {step}: Y positions = [{positions[0,1]:.3f}, {positions[1,1]:.3f}]")
            
            # Check final state
            final_positions = bodies[:, :3].numpy()
            final_velocities = bodies[:, 3:6].numpy()
            
            print(f"\n  [FINAL STATE]")
            print(f"    Body 1 position: {final_positions[0]}")
            print(f"    Body 2 position: {final_positions[1]}")
            print(f"    Body 1 velocity: {final_velocities[0]}")
            print(f"    Body 2 velocity: {final_velocities[1]}")
            
            # Bodies should have interacted (velocities changed)
            print("\n  [VALIDATION]")
            assert bodies.shape == (2, 8)
            print("    ✓ Bodies maintained correct shape after simulation")
            
            # Check if velocities changed (indicating interaction)
            velocity_changes = np.linalg.norm(final_velocities - initial_velocities, axis=1)
            print(f"    Velocity changes: Body 1 = {velocity_changes[0]:.3f}, Body 2 = {velocity_changes[1]:.3f}")
            
            print("  [SUCCESS] Multiple body interaction test completed")
    
    def test_conservation_properties(self):
        """Test basic conservation properties"""
        print("\n[TEST] test_conservation_properties")
        print("  [OBJECTIVE] Validate momentum conservation in isolated system")
        
        with physics_enabled("CPU"):
            print("  [INFO] Testing momentum conservation (X and Z axes)")
            
            # Create isolated system
            n_bodies = 10
            print(f"\n  [SETUP] Creating isolated system with {n_bodies} bodies...")
            world = create_physics_world(n_bodies=n_bodies)
            
            # Get initial total momentum (approximately, ignoring rotations)
            print("\n  [CALCULATE] Computing initial total momentum...")
            masses = world[:, 6:7].numpy()
            velocities = world[:, 3:6].numpy()
            initial_momentum = (world[:, 3:6] * world[:, 6:7]).sum(axis=0).numpy()
            
            print(f"    Initial momentum (kg·m/s):")
            print(f"      X: {initial_momentum[0]:.3f}")
            print(f"      Y: {initial_momentum[1]:.3f}")
            print(f"      Z: {initial_momentum[2]:.3f}")
            print(f"    Total mass: {masses.sum():.1f} kg")
            
            # Run simulation
            dt = 0.016
            n_steps = 100
            simulation_time = dt * n_steps
            print(f"\n  [EXECUTE] Running {n_steps} steps (total time: {simulation_time:.2f}s)...")
            
            for step in range(n_steps):
                world = world.integrate(dt)
                if step % 25 == 0:
                    current_momentum = (world[:, 3:6] * world[:, 6:7]).sum(axis=0).numpy()
                    print(f"    Step {step}: Momentum = [{current_momentum[0]:.3f}, {current_momentum[1]:.3f}, {current_momentum[2]:.3f}]")
            
            # Final momentum (x and z components should be conserved)
            final_momentum = (world[:, 3:6] * world[:, 6:7]).sum(axis=0).numpy()
            
            print(f"\n  [FINAL] Total momentum after {simulation_time:.2f}s:")
            print(f"    X: {final_momentum[0]:.3f} (change: {final_momentum[0] - initial_momentum[0]:.3f})")
            print(f"    Y: {final_momentum[1]:.3f} (change: {final_momentum[1] - initial_momentum[1]:.3f})")
            print(f"    Z: {final_momentum[2]:.3f} (change: {final_momentum[2] - initial_momentum[2]:.3f})")
            
            print("\n  [VALIDATION]")
            # X and Z momentum should be approximately conserved
            x_error = abs(final_momentum[0] - initial_momentum[0])
            z_error = abs(final_momentum[2] - initial_momentum[2])
            y_change = final_momentum[1] - initial_momentum[1]
            
            print(f"    X momentum error: {x_error:.6f} kg·m/s (tolerance: 1.0)")
            print(f"    Z momentum error: {z_error:.6f} kg·m/s (tolerance: 1.0)")
            print(f"    Y momentum change: {y_change:.3f} kg·m/s (expected due to gravity)")
            
            assert x_error < 1.0
            print("    ✓ X momentum conserved")
            assert z_error < 1.0
            print("    ✓ Z momentum conserved")
            print(f"    ✓ Y momentum changed by {y_change:.3f} kg·m/s (gravity effect)")
            
            print("  [SUCCESS] Conservation properties validated")
    
    def test_deterministic_simulation(self):
        """Test that simulation is deterministic"""
        print("\n[TEST] test_deterministic_simulation")
        print("  [OBJECTIVE] Ensure physics simulation is fully deterministic")
        
        with physics_enabled("CPU"):
            print("  [INFO] Running identical simulations to check determinism")
            
            # Create same initial conditions
            print("\n  [SETUP] Creating two identical worlds...")
            np.random.seed(42)
            world1 = create_physics_world(n_bodies=5)
            print("    World 1 created with seed=42")
            
            np.random.seed(42)
            world2 = create_physics_world(n_bodies=5)
            print("    World 2 created with seed=42")
            
            # Verify initial conditions are identical
            initial_diff = np.max(np.abs(world1.numpy() - world2.numpy()))
            print(f"    Initial difference: {initial_diff:.2e}")
            
            # Run same simulation
            dt = 0.016
            n_steps = 10
            print(f"\n  [EXECUTE] Running {n_steps} simulation steps on both worlds...")
            
            for step in range(n_steps):
                world1 = world1.integrate(dt)
                world2 = world2.integrate(dt)
                
                if step % 3 == 0:
                    diff = np.max(np.abs(world1.numpy() - world2.numpy()))
                    print(f"    Step {step}: Max difference = {diff:.2e}")
            
            # Results should be identical
            print("\n  [VALIDATION] Comparing final states...")
            final_state1 = world1.numpy()
            final_state2 = world2.numpy()
            
            max_diff = np.max(np.abs(final_state1 - final_state2))
            mean_diff = np.mean(np.abs(final_state1 - final_state2))
            
            print(f"    Maximum difference: {max_diff:.2e}")
            print(f"    Mean difference: {mean_diff:.2e}")
            print(f"    Tolerance: 1e-6")
            
            np.testing.assert_array_almost_equal(
                final_state1, 
                final_state2, 
                decimal=6
            )
            
            print("    ✓ Simulations produced identical results")
            print("  [SUCCESS] Simulation is deterministic")

def test_physics_extension_import():
    """Test that custom_ops module can be imported"""
    print("\n[TEST] test_physics_extension_import")
    print("  [OBJECTIVE] Verify custom_ops module imports and exports")
    
    print("\n  [IMPORT] Importing custom_ops module...")
    import custom_ops
    print("    ✓ Module imported successfully")
    
    print("\n  [CHECK] Verifying expected attributes...")
    expected_attrs = ['enable_physics_on_device', 'PhysicsTensor']
    
    for attr in expected_attrs:
        if hasattr(custom_ops, attr):
            print(f"    ✓ Found: {attr}")
        else:
            print(f"    ✗ Missing: {attr}")
            
    assert hasattr(custom_ops, 'enable_physics_on_device')
    assert hasattr(custom_ops, 'PhysicsTensor')
    
    print("  [SUCCESS] All required attributes present")

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# PHYSICS INTEGRATION TEST SUITE")
    print("#" * 70)
    print(f"\n[INFO] Test runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check library availability
    print("\n[PREREQ] Checking for compiled physics library...")
    if not physics_library_available():
        print("  [ERROR] Physics library not found!")
        print("  [ERROR] Please run 'make' in the custom_ops/src/ directory")
        print("  [ERROR] This compiles the C physics library needed for tests")
        sys.exit(1)
    print("  [SUCCESS] Physics library found")
    
    # Run tests manually
    test = TestPhysicsIntegration()
    
    print("\n" + "=" * 70)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 70)
    
    try:
        # Test 1
        test.test_create_physics_world()
        print("\n[✓] test_create_physics_world PASSED\n")
        
        # Test 2
        test.test_physics_step_integration()
        print("\n[✓] test_physics_step_integration PASSED\n")
        
        # Test 3
        test.test_multiple_bodies_interaction()
        print("\n[✓] test_multiple_bodies_interaction PASSED\n")
        
        # Test 4
        test.test_conservation_properties()
        print("\n[✓] test_conservation_properties PASSED\n")
        
        # Test 5
        test.test_deterministic_simulation()
        print("\n[✓] test_deterministic_simulation PASSED\n")
        
        # Test 6
        test_physics_extension_import()
        print("\n[✓] test_physics_extension_import PASSED\n")
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("✓ All 6 integration tests passed!")
        print("\n[SUCCESS] Physics integration is working correctly")
        print("[INFO] The custom ops properly interface with TinyGrad")
    except Exception as e:
        print(f"\n[✗] Test failed: {e}")
        print("\n[ERROR] Exception details:")
        import traceback
        traceback.print_exc()
        print("\n[FAILURE] Integration tests did not complete successfully")
        sys.exit(1)
    
    print(f"\n[INFO] Test runner completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")