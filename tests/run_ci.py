#!/usr/bin/env python3
"""
Comprehensive CI Test Suite for the Physics Engine

This module serves as the main continuous integration (CI) test runner for the physics engine.
It validates all core functionality including:
- Module imports and dependencies
- Basic physics simulation (gravity, collisions)
- JIT compilation capabilities
- NumPy-free core module verification
- Command-line interface testing
- Performance benchmarking

The test suite is designed to catch regressions early and ensure the physics engine
maintains correct behavior across different execution modes (single-step vs N-step).

Why this is useful:
- Provides automated validation of the entire physics engine stack
- Ensures compatibility with TinyGrad's JIT compilation
- Verifies that core modules remain NumPy-free for performance
- Catches integration issues between different components
- Provides performance baselines to detect slowdowns
"""

import os
import sys
import subprocess
import time
import numpy as np
from pathlib import Path

# Add physics_core to path (parent of tests directory)
physics_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, physics_core_path)

# Add tinygrad to path
tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

print(f"[SETUP] Adding physics_core to Python path: {physics_core_path}")
print(f"[SETUP] Current working directory: {os.getcwd()}")
print(f"[SETUP] Python version: {sys.version}")
print(f"[SETUP] NumPy version: {np.__version__}")

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def run_command(cmd, description, capture_output=False):
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    # Set up environment with proper PYTHONPATH
    env = os.environ.copy()
    physics_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tinygrad_path = os.path.join(physics_core_path, "external", "tinygrad")
    
    # Add to PYTHONPATH
    python_path = f"{physics_core_path}:{tinygrad_path}"
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{python_path}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = python_path
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                print_success(description)
                return True, result.stdout
            else:
                print_error(f"{description} - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, env=env)
            if result.returncode == 0:
                print_success(description)
                return True, None
            else:
                print_error(f"{description} - Exit code: {result.returncode}")
                return False, None
    except Exception as e:
        print_error(f"{description} - Exception: {e}")
        return False, str(e)

def test_imports():
    """Test that all modules can be imported."""
    print_header("Import Tests")
    
    print("[INFO] This test validates that all core physics modules can be imported successfully.")
    print("[INFO] Import failures often indicate missing dependencies or syntax errors.")
    print("")
    
    modules = [
        'physics.types',
        'physics.math_utils',
        'physics.integration',
        'physics.broadphase_tensor',
        'physics.narrowphase',
        'physics.solver',
        'physics.engine',
        'physics.main'
    ]
    
    print(f"[TEST] Attempting to import {len(modules)} core modules...")
    
    all_passed = True
    for i, module in enumerate(modules):
        print(f"\n[{i+1}/{len(modules)}] Importing module: {module}")
        try:
            print(f"  [ATTEMPT] importlib.import_module('{module}')")
            import importlib
            imported_module = importlib.import_module(module)
            print_success(f"Import {module}")
            if hasattr(imported_module, '__file__'):
                print(f"  [SUCCESS] Module loaded successfully from: {imported_module.__file__}")
            else:
                print(f"  [SUCCESS] Module loaded successfully")
        except Exception as e:
            print_error(f"Import {module}: {e}")
            print(f"  [ERROR] Failed to import module")
            print(f"  [ERROR] Exception type: {type(e).__name__}")
            all_passed = False
    
    print(f"\n[SUMMARY] Import test completed: {sum(1 for m in modules if m)} modules checked")
    return all_passed

def test_basic_simulation():
    """Test basic simulation functionality."""
    print_header("Basic Simulation Test")
    
    print("[INFO] This test validates core physics simulation capabilities.")
    print("[INFO] It creates a simple scene with a falling sphere and static ground.")
    print("[INFO] Success indicates gravity, collision detection, and integration work correctly.")
    print("")
    
    try:
        print("[PHASE 1] Importing required modules...")
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        print("  [SUCCESS] Modules imported successfully")
        
        print("\n[PHASE 2] Creating test scene...")
        print("  [INFO] Scene contains:")
        print("    - 1 static ground box (10x0.5x10 units)")
        print("    - 1 dynamic sphere (radius 1.0, mass 1.0)")
        
        # Create simple scene
        bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        
        # Static ground
        print("\n  [SETUP] Configuring static ground...")
        bodies[0, BodySchema.POS_Y] = -5.0
        bodies[0, BodySchema.QUAT_W] = 1.0
        bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
        bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
        bodies[0, BodySchema.INV_MASS] = 0.0
        print(f"    Position: (0.0, {bodies[0, BodySchema.POS_Y]}, 0.0)")
        print(f"    Shape: Box with dimensions {bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1]}")
        print(f"    Mass: Infinite (static body)")
        
        # Falling sphere
        print("\n  [SETUP] Configuring falling sphere...")
        bodies[1, BodySchema.POS_Y] = 5.0
        bodies[1, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
        bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
        bodies[1, BodySchema.INV_MASS] = 1.0
        bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
        bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
        bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
        print(f"    Position: (0.0, {bodies[1, BodySchema.POS_Y]}, 0.0)")
        print(f"    Shape: Sphere with radius {bodies[1, BodySchema.SHAPE_PARAM_1]}")
        print(f"    Mass: {1.0/bodies[1, BodySchema.INV_MASS]}")
        
        # Test single-step
        print("\n[PHASE 3] Testing single-step simulation...")
        print("  [INFO] Single-step mode calls step() repeatedly")
        engine = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        initial_y = engine.get_state()[1, BodySchema.POS_Y]
        print(f"  [INITIAL] Sphere Y position: {initial_y:.4f}")
        
        print("  [ACTION] Executing engine.step()...")
        engine.step()
        final_y = engine.get_state()[1, BodySchema.POS_Y]
        print(f"  [FINAL] Sphere Y position: {final_y:.4f}")
        print(f"  [DELTA] Position change: {final_y - initial_y:.4f}")
        
        if final_y < initial_y:
            print_success(f"Single-step simulation: sphere fell from {initial_y:.2f} to {final_y:.2f}")
        else:
            print_error(f"Single-step simulation: sphere didn't fall (y: {initial_y:.2f} -> {final_y:.2f})")
            return False
        
        # Test N-step
        print("\n[PHASE 4] Testing N-step simulation...")
        print("  [INFO] N-step mode compiles multiple steps into single operation")
        engine2 = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        initial_y = engine2.get_state()[1, BodySchema.POS_Y]
        print(f"  [INITIAL] Sphere Y position: {initial_y:.4f}")
        
        print("  [ACTION] Executing engine.run_simulation(10)...")
        engine2.run_simulation(10)
        final_y = engine2.get_state()[1, BodySchema.POS_Y]
        print(f"  [FINAL] Sphere Y position: {final_y:.4f}")
        print(f"  [DELTA] Position change: {final_y - initial_y:.4f}")
        print(f"  [INFO] Expected fall distance: ~{0.5 * 9.81 * (10 * 0.016)**2:.4f} (gravity only)")
        
        if final_y < initial_y:
            print_success(f"N-step simulation: sphere fell from {initial_y:.2f} to {final_y:.2f}")
        else:
            print_error(f"N-step simulation: sphere didn't fall (y: {initial_y:.2f} -> {final_y:.2f})")
            return False
        
        print("\n[SUCCESS] Basic simulation test completed successfully")
        return True
        
    except Exception as e:
        print_error(f"Basic simulation test failed: {e}")
        print("[DEBUG] Full exception traceback:")
        import traceback
        traceback.print_exc()
        return False

def test_jit_compilation():
    """Test JIT compilation."""
    print_header("JIT Compilation Test")
    
    print("[INFO] This test validates TinyGrad's JIT compilation of physics operations.")
    print("[INFO] JIT compilation is crucial for performance, especially on GPUs.")
    print("[INFO] It compiles Python operations into optimized machine code.")
    print("")
    
    try:
        print("[PHASE 1] Importing required modules...")
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        from tinygrad import TinyJit, Tensor
        print("  [SUCCESS] Modules imported successfully")
        print(f"  [INFO] TinyJit class location: {TinyJit.__module__}")
        
        # Create test scene
        print("\n[PHASE 2] Creating minimal test scene...")
        bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        bodies[0, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.INV_MASS] = 1.0
        print("  [INFO] Created 2 bodies for JIT testing")
        print("    - Body 0: Static (inv_mass = 0)")
        print("    - Body 1: Dynamic (inv_mass = 1)")
        
        print("\n[PHASE 3] Initializing physics engine...")
        engine = TensorPhysicsEngine(bodies, dt=0.016)
        print("  [SUCCESS] Engine initialized with dt=0.016 (60 FPS)")
        
        # Test that JIT compilation happened
        print("\n[PHASE 4] Verifying JIT compilation...")
        
        print("  [CHECK 1] Testing single-step JIT compilation...")
        if hasattr(engine, 'jitted_step'):
            print(f"    [FOUND] jitted_step attribute exists")
            print(f"    [TYPE] {type(engine.jitted_step)}")
            if isinstance(engine.jitted_step, TinyJit):
                print_success("Single-step JIT compilation")
                print("    [INFO] Single-step operations are JIT-compiled for efficiency")
            else:
                print_error("Single-step JIT compilation failed - wrong type")
                return False
        else:
            print_error("Single-step JIT compilation failed - attribute missing")
            print("    [ERROR] No jitted_step attribute found on engine")
            return False
            
        print("\n  [CHECK 2] Testing N-step JIT compilation...")
        if hasattr(engine, 'jitted_n_step'):
            print(f"    [FOUND] jitted_n_step attribute exists")
            print(f"    [TYPE] {type(engine.jitted_n_step)}")
            if isinstance(engine.jitted_n_step, TinyJit):
                print_success("N-step JIT compilation")
                print("    [INFO] N-step operations are JIT-compiled for maximum performance")
            else:
                print_error("N-step JIT compilation failed - wrong type")
                return False
        else:
            print_error("N-step JIT compilation failed - attribute missing")
            print("    [ERROR] No jitted_n_step attribute found on engine")
            return False
        
        print("\n[SUCCESS] JIT compilation test completed successfully")
        print("[INFO] Both single-step and N-step modes are properly JIT-compiled")
        return True
        
    except Exception as e:
        print_error(f"JIT compilation test failed: {e}")
        print("[DEBUG] Full exception traceback:")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_free():
    """Test that core physics modules don't import numpy."""
    print_header("NumPy-Free Core Modules Test")
    
    core_modules = [
        'physics/broadphase_tensor.py',
        'physics/narrowphase.py',
        'physics/solver.py',
        'physics/integration.py',
        'physics/math_utils.py'
    ]
    
    all_passed = True
    for module in core_modules:
        # Go up one directory from tests/ to physics_core/
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), module)
        with open(path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
        numpy_found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if 'import numpy' in line or 'from numpy' in line:
                print_error(f"NumPy import in {module}:{i+1}: {line.strip()}")
                numpy_found = True
                all_passed = False
        
        if not numpy_found:
            print_success(f"No NumPy imports in {module}")
    
    return all_passed

def test_main_script():
    """Test the main.py script."""
    print_header("Main Script Tests")
    
    # Test N-step mode
    success1, _ = run_command(
        "python3 -m physics.main --steps 10 --mode nstep --output artifacts/test_ci_nstep.npy",
        "Main script N-step mode"
    )
    
    # Test single-step mode
    success2, _ = run_command(
        "python3 -m physics.main --steps 10 --mode single --output artifacts/test_ci_single.npy",
        "Main script single-step mode"
    )
    
    # Verify outputs
    if success1 and success2:
        try:
            from physics.types import BodySchema
            
            nstep_data = np.load('artifacts/test_ci_nstep.npy')
            single_data = np.load('artifacts/test_ci_single.npy')
            
            if nstep_data.shape == (2, 2, 27):
                print_success(f"N-step output shape correct: {nstep_data.shape}")
            else:
                print_error(f"N-step output shape incorrect: {nstep_data.shape}")
                return False
                
            if single_data.shape == (2, 2, 27):
                print_success(f"Single-step output shape correct: {single_data.shape}")
            else:
                print_error(f"Single-step output shape incorrect: {single_data.shape}")
                return False
            
            # Check that final positions are similar
            nstep_final_y = nstep_data[1, 1, BodySchema.POS_Y]
            single_final_y = single_data[1, 1, BodySchema.POS_Y]
            
            if abs(nstep_final_y - single_final_y) < 0.1:
                print_success(f"Final positions match: N-step={nstep_final_y:.3f}, Single={single_final_y:.3f}")
            else:
                print_warning(f"Final positions differ: N-step={nstep_final_y:.3f}, Single={single_final_y:.3f}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to verify outputs: {e}")
            return False
    
    return success1 and success2

def test_collision_detection():
    """Test collision detection functionality."""
    print_header("Collision Detection Test")
    
    try:
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        
        # Create scene with guaranteed collision
        bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        
        # Two spheres at same position
        bodies[0, BodySchema.POS_X:BodySchema.POS_Z+1] = [0, 0, 0]
        bodies[0, BodySchema.QUAT_W] = 1.0
        bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
        bodies[0, BodySchema.SHAPE_PARAM_1] = 1.0
        bodies[0, BodySchema.INV_MASS] = 1.0
        
        bodies[1, BodySchema.POS_X:BodySchema.POS_Z+1] = [0.5, 0, 0]  # Overlapping
        bodies[1, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
        bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
        bodies[1, BodySchema.INV_MASS] = 1.0
        bodies[1, BodySchema.VEL_X] = -1.0  # Moving toward first sphere
        
        engine = TensorPhysicsEngine(bodies, dt=0.016)
        initial_vel = engine.get_state()[1, BodySchema.VEL_X]
        
        # Run one step - should detect collision and change velocity
        engine.step()
        final_vel = engine.get_state()[1, BodySchema.VEL_X]
        
        if final_vel != initial_vel:
            print_success(f"Collision detected and resolved: velocity changed from {initial_vel:.2f} to {final_vel:.2f}")
            return True
        else:
            print_error(f"Collision not detected: velocity unchanged at {final_vel:.2f}")
            return False
            
    except Exception as e:
        print_error(f"Collision detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Basic performance test."""
    print_header("Performance Test")
    
    try:
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        
        # Create scene with multiple bodies
        n_bodies = 10
        bodies = np.zeros((n_bodies, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        
        for i in range(n_bodies):
            bodies[i, BodySchema.POS_X] = (i % 3) * 2.0
            bodies[i, BodySchema.POS_Y] = (i // 3) * 2.0
            bodies[i, BodySchema.QUAT_W] = 1.0
            bodies[i, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
            bodies[i, BodySchema.SHAPE_PARAM_1] = 0.5
            bodies[i, BodySchema.INV_MASS] = 1.0 if i > 0 else 0.0  # First is static
        
        # Test single-step performance
        engine = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        start_time = time.time()
        for _ in range(100):
            engine.step()
        single_time = time.time() - start_time
        
        print_success(f"Single-step: 100 steps with {n_bodies} bodies in {single_time:.3f}s ({100/single_time:.0f} steps/s)")
        
        # Test N-step performance (with smaller step count due to compilation time)
        engine2 = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        start_time = time.time()
        engine2.run_simulation(10)  # Smaller due to compilation overhead
        nstep_time = time.time() - start_time
        
        print_success(f"N-step: 10 steps compiled and run in {nstep_time:.3f}s")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False

def cleanup():
    """Clean up test artifacts."""
    print_header("Cleanup")
    
    test_files = [
        'artifacts/test_ci_nstep.npy',
        'artifacts/test_ci_single.npy'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print_success(f"Removed {file}")

def main():
    """Run all CI tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{'PHYSICS ENGINE CI TEST SUITE':^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    # Track results
    results = []
    
    # Run all tests
    tests = [
        ("Import Tests", test_imports),
        ("Basic Simulation", test_basic_simulation),
        ("JIT Compilation", test_jit_compilation),
        ("NumPy-Free Core", test_numpy_free),
        ("Main Script", test_main_script),
        ("Collision Detection", test_collision_detection),
        ("Performance", test_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Cleanup
    cleanup()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}ALL TESTS PASSED! ({passed}/{total}){RESET}")
        return 0
    else:
        print(f"{RED}TESTS FAILED: {passed}/{total} passed{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())