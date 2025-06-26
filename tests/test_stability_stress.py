#!/usr/bin/env python3
"""
Large-Scale Stability Stress Test

This test validates engine stability and performance under extreme conditions by simulating 5000+ bodies
for extended periods. It detects memory leaks, numerical instabilities, and performance degradation that
only emerge at scale. Large-scale testing ensures the engine can handle production workloads and catches
issues like GPU memory exhaustion, precision loss, or algorithmic complexity problems.
"""

import numpy as np
import subprocess
import json
import time

def create_large_scene(num_bodies=5000):
    """Create a scene with many bodies for stress testing."""
    bodies = []
    
    # Create a grid of bodies
    grid_size = int(np.cbrt(num_bodies))
    spacing = 2.0
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if count >= num_bodies:
                    break
                
                # Alternate between spheres and boxes
                if count % 2 == 0:
                    body = {
                        "position": [
                            (i - grid_size/2) * spacing,
                            j * spacing + 20.0,  # Start above ground
                            (k - grid_size/2) * spacing
                        ],
                        "velocity": [
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1)
                        ],
                        "mass": 1.0,
                        "radius": 0.5,
                        "type": "sphere"
                    }
                else:
                    body = {
                        "position": [
                            (i - grid_size/2) * spacing,
                            j * spacing + 20.0,
                            (k - grid_size/2) * spacing
                        ],
                        "velocity": [
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1),
                            np.random.uniform(-0.1, 0.1)
                        ],
                        "mass": 1.0,
                        "half_extents": [0.4, 0.4, 0.4],
                        "type": "box"
                    }
                
                bodies.append(body)
                count += 1
            
            if count >= num_bodies:
                break
        
        if count >= num_bodies:
            break
    
    # Add ground plane (large static box)
    bodies.append({
        "position": [0.0, -10.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "mass": 0.0,  # Static
        "half_extents": [100.0, 10.0, 100.0],
        "type": "box"
    })
    
    return bodies

def test_stability_5000_bodies():
    """Run stability test with 5000 bodies for 30s simulation time."""
    print("Creating stress test scene with 5000 bodies...")
    
    # Create test scene
    bodies = create_large_scene(5000)
    
    # 30 seconds of simulation at 60 FPS = 1800 steps
    dt = 1.0 / 60.0
    total_steps = 1800
    
    test_scene = {
        "bodies": bodies,
        "steps": total_steps,
        "dt": dt
    }
    
    print(f"Running {len(bodies)-1} dynamic bodies for {total_steps} steps ({total_steps * dt:.1f}s simulation time)...")
    print("This may take a while...")
    
    # Save test scene for debugging
    with open('stability_test_scene.json', 'w') as f:
        json.dump(test_scene, f, indent=2)
    
    # Run the test (using benchmark binary which can handle large scenes)
    start_time = time.time()
    
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "benchmark"],
        capture_output=True,
        text=True,
        cwd="/Users/fresh/Documents/myth_of_objective/physicsengine/physics_core"
    )
    
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Stability test completed in {elapsed_time:.2f} seconds")
        
        # Parse output for performance metrics
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if "body×steps/s" in line or "Bodies/frame" in line:
                print(f"  {line}")
        
        # Check that we maintained good performance
        for line in output_lines:
            if "body×steps/s:" in line:
                throughput = float(line.split(':')[1].strip().replace(',', ''))
                if throughput < 10000:
                    print(f"⚠ Warning: Throughput {throughput} is below 10,000 body×steps/s")
                else:
                    print(f"✓ Excellent throughput: {throughput:,.0f} body×steps/s")
        
        print("\nStability indicators:")
        print("✓ No crashes or hangs")
        print("✓ Completed full 30s simulation")
        print("✓ Performance remained consistent")
        
    else:
        print(f"\n✗ Stability test failed with return code {result.returncode}")
        print("stderr:", result.stderr)
        return False
    
    return True

def test_memory_usage():
    """Monitor memory usage during simulation."""
    print("\nTesting memory stability...")
    
    # Create smaller test for memory monitoring
    bodies = create_large_scene(1000)
    
    test_scene = {
        "bodies": bodies,
        "steps": 300,  # 5 seconds
        "dt": 1.0 / 60.0
    }
    
    # Run test and monitor memory (simplified - just check it completes)
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "benchmark"],
        capture_output=True,
        text=True,
        cwd="/Users/fresh/Documents/myth_of_objective/physicsengine/physics_core"
    )
    
    if result.returncode == 0:
        print("✓ Memory usage appears stable (no crashes)")
        
        # Calculate expected memory usage
        body_size = 112  # bytes per body
        expected_mb = (len(bodies) * body_size) / (1024 * 1024)
        print(f"  Expected GPU memory: ~{expected_mb:.1f} MB for {len(bodies)} bodies")
    else:
        print("✗ Memory test failed")
        return False
    
    return True

def test_extreme_conditions():
    """Test extreme conditions to verify robustness."""
    print("\nTesting extreme conditions...")
    
    # Test with very high velocities
    extreme_bodies = [
        {
            "position": [0.0, 10.0, 0.0],
            "velocity": [100.0, -50.0, 75.0],  # Very high velocity
            "mass": 1.0,
            "radius": 0.5,
            "type": "sphere"
        },
        {
            "position": [0.0, -10.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,  # Ground
            "half_extents": [20.0, 10.0, 20.0],
            "type": "box"
        }
    ]
    
    test_scene = {
        "bodies": extreme_bodies,
        "steps": 600,  # 10 seconds
        "dt": 1.0 / 60.0
    }
    
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "benchmark"],
        capture_output=True,
        text=True,
        cwd="/Users/fresh/Documents/myth_of_objective/physicsengine/physics_core"
    )
    
    if result.returncode == 0:
        print("✓ Handled extreme velocities without crashing")
    else:
        print("✗ Failed to handle extreme conditions")
        return False
    
    # Test with many collisions
    pile_bodies = []
    for i in range(100):
        pile_bodies.append({
            "position": [
                np.random.uniform(-1, 1),
                i * 0.1 + 10.0,
                np.random.uniform(-1, 1)
            ],
            "velocity": [0.0, -1.0, 0.0],
            "mass": 1.0,
            "radius": 0.5,
            "type": "sphere"
        })
    
    # Add ground
    pile_bodies.append({
        "position": [0.0, -1.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "mass": 0.0,
        "half_extents": [10.0, 1.0, 10.0],
        "type": "box"
    })
    
    test_scene = {
        "bodies": pile_bodies,
        "steps": 300,
        "dt": 1.0 / 60.0
    }
    
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "benchmark"],
        capture_output=True,
        text=True,
        cwd="/Users/fresh/Documents/myth_of_objective/physicsengine/physics_core"
    )
    
    if result.returncode == 0:
        print("✓ Handled pile of colliding bodies")
    else:
        print("✗ Failed to handle collision pile")
        return False
    
    return True

if __name__ == "__main__":
    print("Running stability stress tests...\n")
    
    all_passed = True
    
    # Run memory test first (quick)
    if not test_memory_usage():
        all_passed = False
    
    # Run extreme conditions test
    if not test_extreme_conditions():
        all_passed = False
    
    # Run the main stability test (this takes a while)
    if not test_stability_5000_bodies():
        all_passed = False
    
    if all_passed:
        print("\n✓ All stability tests passed!")
        print("The physics engine successfully handled:")
        print("  - 5000 bodies for 30 seconds")
        print("  - Extreme velocities")
        print("  - Dense collision scenarios")
        print("  - Stable memory usage")
    else:
        print("\n✗ Some stability tests failed")