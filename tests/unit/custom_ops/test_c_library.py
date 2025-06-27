"""
Direct C Library Testing Module

This module tests the physics C library directly through ctypes FFI (Foreign Function Interface).
It validates the low-level C implementation without going through the TinyGrad tensor abstraction.

Why this is useful:
- Ensures the C library is correctly compiled and linked
- Validates core physics algorithms at the lowest level
- Helps debug issues by isolating C code from Python/TinyGrad layers
- Verifies memory layout and data structure compatibility
- Tests performance-critical code paths directly

The tests cover:
1. Basic integration (gravity application)
2. Collision detection between spheres
3. Multi-body physics processing
"""
import ctypes
import numpy as np
from pathlib import Path
import sys

def test_physics_library():
    """Test the physics library functions directly"""
    print("\n" + "="*70)
    print("PHYSICS C LIBRARY DIRECT TESTING")
    print("="*70)
    print("\n[INFO] This test validates the C physics library through ctypes FFI.")
    print("[INFO] It bypasses TinyGrad to test the raw C implementation.")
    print("")
    
    # Load library
    print("[PHASE 1] Loading physics library...")
    lib_path = Path(__file__).parent.parent.parent.parent / "custom_ops" / "build"
    lib_name = "libphysics.dylib" if sys.platform == "darwin" else "libphysics.so"
    lib_file = lib_path / lib_name
    
    if not lib_file.exists():
        print(f"  [ERROR] Physics library not found at {lib_file}")
        print("  [ERROR] Please run 'make' in the custom_ops/src directory first")
        return False
    
    print(f"  [FOUND] Library at: {lib_file}")
    print(f"  [INFO] Library size: {lib_file.stat().st_size:,} bytes")
    
    print("\n  [ACTION] Loading library with ctypes.CDLL...")
    lib = ctypes.CDLL(str(lib_file))
    print("  [SUCCESS] Library loaded successfully")
    
    # Define function signatures
    print("\n[PHASE 2] Configuring function signatures...")
    print("  [INFO] Setting up ctypes argument and return types")
    
    print("\n  [FUNCTION 1] physics_integrate")
    print("    - Args: float* bodies, int32 n_bodies, float dt, float* output")
    print("    - Returns: void")
    lib.physics_integrate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.physics_integrate.restype = None
    
    print("\n  [FUNCTION 2] physics_step")
    print("    - Args: float* bodies, int32 n_bodies, float dt, float* output")
    print("    - Returns: void")
    lib.physics_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.physics_step.restype = None
    print("  [SUCCESS] Function signatures configured")
    
    print("\n" + "="*70)
    print("TEST SUITE EXECUTION")
    print("="*70)
    
    # Test 1: Basic integration
    print("\n[TEST 1] Basic Integration - Gravity Application")
    print("-" * 50)
    print("[OBJECTIVE] Verify that gravity is correctly applied to a falling body")
    print("\n[SETUP] Creating test body...")
    bodies = np.array([
        # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
    ], dtype=np.float32)
    print("  Body format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius]")
    print(f"  Initial state: {bodies[0]}")
    
    output = np.zeros_like(bodies)
    dt = 0.1
    print(f"  Time step: {dt}s")
    print(f"  Gravity constant: -9.81 m/s²")
    
    print("\n[EXECUTE] Calling physics_integrate...")
    lib.physics_integrate(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(1),
        ctypes.c_float(dt),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    print("  [SUCCESS] Integration completed")
    
    print("\n[RESULTS]")
    print(f"  Initial Y position: {bodies[0, 1]:.3f} m")
    print(f"  Initial Y velocity: {bodies[0, 4]:.3f} m/s")
    print(f"\n  After {dt}s of gravity:")
    print(f"  Final Y position: {output[0, 1]:.3f} m")
    print(f"  Final Y velocity: {output[0, 4]:.3f} m/s")
    print(f"\n  Expected Y velocity: {-9.81 * dt:.3f} m/s (gravity * dt)")
    print(f"  Position change: {output[0, 1] - bodies[0, 1]:.3f} m")
    
    # Verify gravity was applied
    print("\n[VALIDATION]")
    expected_vy = -9.81 * dt
    error = abs(output[0, 4] - expected_vy)
    print(f"  Velocity error: {error:.6f} m/s")
    print(f"  Tolerance: 0.001 m/s")
    
    assert error < 0.001, f"Gravity not applied correctly (error: {error})"  
    print("  ✓ Gravity integration working correctly")
    print("  ✓ Test 1 PASSED")
    
    # Test 2: Collision detection
    print("\n\n[TEST 2] Collision Detection - Sphere-Sphere Impact")
    print("-" * 50)
    print("[OBJECTIVE] Verify collision detection and response between two spheres")
    print("\n[SETUP] Creating colliding spheres...")
    bodies = np.array([
        # Two spheres that overlap
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # radius 1 at origin
        [1.5, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0],  # radius 1 at x=1.5, moving left
    ], dtype=np.float32)
    print("  Sphere 1: Position (0,0,0), Radius 1.0, Stationary")
    print("  Sphere 2: Position (1.5,0,0), Radius 1.0, Velocity (-1,0,0)")
    print("  Distance between centers: 1.5")
    print("  Sum of radii: 2.0")
    print("  Overlap: 0.5 units (collision!)")
    
    output = np.zeros_like(bodies)
    
    print("\n[EXECUTE] Calling physics_step (full simulation step)...")
    lib.physics_step(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(2),
        ctypes.c_float(0.01),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    print("  [SUCCESS] Physics step completed")
    
    print("\n[RESULTS]")
    print("  Initial velocities:")
    print(f"    Body 1 X velocity: {bodies[0, 3]:.3f} m/s")
    print(f"    Body 2 X velocity: {bodies[1, 3]:.3f} m/s")
    print("\n  After collision:")
    print(f"    Body 1 X velocity: {output[0, 3]:.3f} m/s")
    print(f"    Body 2 X velocity: {output[1, 3]:.3f} m/s")
    print("\n  Velocity changes:")
    print(f"    Body 1: {output[0, 3] - bodies[0, 3]:.3f} m/s")
    print(f"    Body 2: {output[1, 3] - bodies[1, 3]:.3f} m/s")
    
    # Verify collision response (velocities should change)
    print("\n[VALIDATION]")
    print("  Checking collision response...")
    
    if output[0, 3] < 0:
        print("    ✓ Body 1 moving left after collision (momentum transfer)")
    else:
        print("    ✗ Body 1 velocity incorrect")
        
    if output[1, 3] > bodies[1, 3]:
        print("    ✓ Body 2 slowed down after collision (momentum conservation)")
    else:
        print("    ✗ Body 2 velocity incorrect")
    
    assert output[0, 3] < 0, "Body 1 should move left after collision"
    assert output[1, 3] > bodies[1, 3], "Body 2 should slow down after collision"
    print("  ✓ Collision detection and response working correctly")
    print("  ✓ Test 2 PASSED")
    
    # Test 3: Multiple bodies
    print("\n\n[TEST 3] Multiple Bodies - Scalability Test")
    print("-" * 50)
    print("[OBJECTIVE] Verify the library can handle multiple bodies simultaneously")
    
    n_bodies = 10
    print(f"\n[SETUP] Creating {n_bodies} random bodies...")
    bodies = np.random.randn(n_bodies, 8).astype(np.float32)
    bodies[:, 6] = 1.0  # mass
    bodies[:, 7] = 0.5  # radius
    print(f"  Generated {n_bodies} bodies with:")
    print("    - Random positions and velocities (Gaussian distribution)")
    print("    - Uniform mass: 1.0 kg")
    print("    - Uniform radius: 0.5 m")
    print(f"  Total memory: {bodies.nbytes} bytes")
    
    output = np.zeros_like(bodies)
    
    print("\n[EXECUTE] Running physics simulation...")
    print(f"  Processing {n_bodies} bodies")
    print(f"  Time step: 0.016s (60 FPS)")
    print(f"  Potential collision pairs: {n_bodies * (n_bodies - 1) // 2}")
    
    start_time = time.time()
    lib.physics_step(
        bodies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(n_bodies),
        ctypes.c_float(0.016),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    elapsed = time.time() - start_time
    print(f"  [SUCCESS] Simulation completed in {elapsed*1000:.2f} ms")
    
    # Verify all bodies were processed
    print("\n[VALIDATION]")
    print("  Checking that bodies were updated...")
    
    # Calculate how many bodies changed
    changed_bodies = np.sum(~np.isclose(bodies, output).all(axis=1))
    print(f"  Bodies with changed state: {changed_bodies}/{n_bodies}")
    
    # Check specific changes
    position_changes = np.linalg.norm(output[:, :3] - bodies[:, :3], axis=1)
    velocity_changes = np.linalg.norm(output[:, 3:6] - bodies[:, 3:6], axis=1)
    
    print(f"  Average position change: {np.mean(position_changes):.6f} m")
    print(f"  Average velocity change: {np.mean(velocity_changes):.6f} m/s")
    
    assert not np.array_equal(bodies, output), "Bodies should have moved"
    print(f"  ✓ Processed {n_bodies} bodies successfully")
    print("  ✓ Test 3 PASSED")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ Test 1: Basic Integration - PASSED")
    print("✓ Test 2: Collision Detection - PASSED")
    print("✓ Test 3: Multiple Bodies - PASSED")
    print("\n[SUCCESS] All C library tests passed!")
    print("[INFO] The physics library is correctly compiled and functional.")
    return True

if __name__ == "__main__":
    import time
    print("\n" + "#" * 70)
    print("# PHYSICS C LIBRARY TEST RUNNER")
    print("#" * 70)
    print(f"\n[START] Test execution began at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_physics_library()
    
    print(f"\n[END] Test execution completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(0 if success else 1)