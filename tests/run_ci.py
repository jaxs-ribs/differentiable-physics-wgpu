#!/usr/bin/env python3
"""Comprehensive CI script for the physics engine."""

import os
import sys
import subprocess
import time
import numpy as np
from pathlib import Path

# Add physics_core to path (parent of tests directory)
physics_core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, physics_core_path)

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
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print_success(description)
                return True, result.stdout
            else:
                print_error(f"{description} - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
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
    
    all_passed = True
    for module in modules:
        try:
            exec(f"import {module}")
            print_success(f"Import {module}")
        except Exception as e:
            print_error(f"Import {module}: {e}")
            all_passed = False
    
    return all_passed

def test_basic_simulation():
    """Test basic simulation functionality."""
    print_header("Basic Simulation Test")
    
    try:
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        
        # Create simple scene
        bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        
        # Static ground
        bodies[0, BodySchema.POS_Y] = -5.0
        bodies[0, BodySchema.QUAT_W] = 1.0
        bodies[0, BodySchema.SHAPE_TYPE] = ShapeType.BOX
        bodies[0, BodySchema.SHAPE_PARAM_1:BodySchema.SHAPE_PARAM_3+1] = [10.0, 0.5, 10.0]
        bodies[0, BodySchema.INV_MASS] = 0.0
        
        # Falling sphere
        bodies[1, BodySchema.POS_Y] = 5.0
        bodies[1, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.SHAPE_TYPE] = ShapeType.SPHERE
        bodies[1, BodySchema.SHAPE_PARAM_1] = 1.0
        bodies[1, BodySchema.INV_MASS] = 1.0
        bodies[1, BodySchema.INV_INERTIA_XX] = 2.5
        bodies[1, BodySchema.INV_INERTIA_YY] = 2.5
        bodies[1, BodySchema.INV_INERTIA_ZZ] = 2.5
        
        # Test single-step
        engine = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        initial_y = engine.get_state()[1, BodySchema.POS_Y]
        engine.step()
        final_y = engine.get_state()[1, BodySchema.POS_Y]
        
        if final_y < initial_y:
            print_success(f"Single-step simulation: sphere fell from {initial_y:.2f} to {final_y:.2f}")
        else:
            print_error(f"Single-step simulation: sphere didn't fall (y: {initial_y:.2f} -> {final_y:.2f})")
            return False
        
        # Test N-step
        engine2 = TensorPhysicsEngine(bodies.copy(), dt=0.016)
        initial_y = engine2.get_state()[1, BodySchema.POS_Y]
        engine2.run_simulation(10)
        final_y = engine2.get_state()[1, BodySchema.POS_Y]
        
        if final_y < initial_y:
            print_success(f"N-step simulation: sphere fell from {initial_y:.2f} to {final_y:.2f}")
        else:
            print_error(f"N-step simulation: sphere didn't fall (y: {initial_y:.2f} -> {final_y:.2f})")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Basic simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jit_compilation():
    """Test JIT compilation."""
    print_header("JIT Compilation Test")
    
    try:
        from physics.engine import TensorPhysicsEngine
        from physics.types import BodySchema, ShapeType
        from tinygrad import TinyJit, Tensor
        
        # Create test scene
        bodies = np.zeros((2, BodySchema.NUM_PROPERTIES), dtype=np.float32)
        bodies[0, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.QUAT_W] = 1.0
        bodies[1, BodySchema.INV_MASS] = 1.0
        
        engine = TensorPhysicsEngine(bodies, dt=0.016)
        
        # Test that JIT compilation happened
        if hasattr(engine, 'jitted_step') and isinstance(engine.jitted_step, TinyJit):
            print_success("Single-step JIT compilation")
        else:
            print_error("Single-step JIT compilation failed")
            return False
            
        if hasattr(engine, 'jitted_n_step') and isinstance(engine.jitted_n_step, TinyJit):
            print_success("N-step JIT compilation")
        else:
            print_error("N-step JIT compilation failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"JIT compilation test failed: {e}")
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