"""
Test the physics C library directly using ctypes
"""
import ctypes
import numpy as np
from pathlib import Path
import sys

def test_physics_library():
    """Test the physics library functions directly"""
    
    # Load library
    lib_path = Path(__file__).parent.parent.parent.parent / "custom_ops" / "build"
    lib_name = "libphysics.dylib" if sys.platform == "darwin" else "libphysics.so"
    lib_file = lib_path / lib_name
    
    if not lib_file.exists():
        print(f"ERROR: Physics library not found at {lib_file}")
        print("Please run 'make' first")
        return False
    
    lib = ctypes.CDLL(str(lib_file))
    
    # Define function signatures
    lib.physics_integrate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.physics_integrate.restype = None
    
    lib.physics_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.physics_step.restype = None
    
    print("Physics Library Tests")
    print("=" * 50)
    
    # Test 1: Basic integration
    print("\nTest 1: Basic Integration")
    bodies = np.array([
        # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
    ], dtype=np.float32)
    
    output = np.zeros_like(bodies)
    dt = 0.1
    
    lib.physics_integrate(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(1),
        ctypes.c_float(dt),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    
    print(f"Initial Y position: {bodies[0, 1]:.3f}")
    print(f"Initial Y velocity: {bodies[0, 4]:.3f}")
    print(f"After {dt}s:")
    print(f"Final Y position: {output[0, 1]:.3f}")
    print(f"Final Y velocity: {output[0, 4]:.3f}")
    print(f"Expected Y velocity: {-9.81 * dt:.3f}")
    
    # Verify gravity was applied
    expected_vy = -9.81 * dt
    assert abs(output[0, 4] - expected_vy) < 0.001, f"Gravity not applied correctly"
    print("✓ Gravity integration working")
    
    # Test 2: Collision detection
    print("\nTest 2: Collision Detection")
    bodies = np.array([
        # Two spheres that overlap
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # radius 1 at origin
        [1.5, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0],  # radius 1 at x=1.5, moving left
    ], dtype=np.float32)
    
    output = np.zeros_like(bodies)
    
    lib.physics_step(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(2),
        ctypes.c_float(0.01),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    
    print(f"Body 1 initial X velocity: {bodies[0, 3]:.3f}")
    print(f"Body 2 initial X velocity: {bodies[1, 3]:.3f}")
    print(f"Body 1 final X velocity: {output[0, 3]:.3f}")
    print(f"Body 2 final X velocity: {output[1, 3]:.3f}")
    
    # Verify collision response (velocities should change)
    assert output[0, 3] < 0, "Body 1 should move left after collision"
    assert output[1, 3] > bodies[1, 3], "Body 2 should slow down after collision"
    print("✓ Collision detection and response working")
    
    # Test 3: Multiple bodies
    print("\nTest 3: Multiple Bodies")
    n_bodies = 10
    bodies = np.random.randn(n_bodies, 8).astype(np.float32)
    bodies[:, 6] = 1.0  # mass
    bodies[:, 7] = 0.5  # radius
    
    output = np.zeros_like(bodies)
    
    lib.physics_step(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(n_bodies),
        ctypes.c_float(0.016),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    
    # Verify all bodies were processed
    assert not np.array_equal(bodies, output), "Bodies should have moved"
    print(f"✓ Processed {n_bodies} bodies successfully")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    return True

if __name__ == "__main__":
    success = test_physics_library()
    sys.exit(0 if success else 1)