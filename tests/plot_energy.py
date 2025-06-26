#!/usr/bin/env python3
"""
Energy Conservation Plotter

This script tracks and visualizes the total energy of the physics system over time.
It's the primary tool for diagnosing simulation instability and validating energy conservation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from reference import Body, PhysicsEngine, ShapeType

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_scene(scene_path):
    """Load a scene configuration from JSON file."""
    with open(scene_path, 'r') as f:
        return json.load(f)


def setup_engine_from_scene(scene_config):
    """Create and populate a PhysicsEngine from scene configuration."""
    # Extract simulation parameters
    dt = scene_config.get('dt', 0.01)
    gravity = scene_config.get('gravity', [0, -9.81, 0])
    
    # Create engine
    engine = PhysicsEngine(dt=dt, gravity=np.array(gravity))
    
    # Add bodies from scene
    for body_config in scene_config.get('bodies', []):
        # Map string shape type to enum
        shape_type_str = body_config.get('shape_type', 'sphere').lower()
        if shape_type_str == 'sphere':
            shape_type = ShapeType.SPHERE
        elif shape_type_str == 'box':
            shape_type = ShapeType.BOX
        elif shape_type_str == 'capsule':
            shape_type = ShapeType.CAPSULE
        else:
            shape_type = ShapeType.SPHERE
        
        # Create body
        body = Body(
            position=np.array(body_config.get('position', [0, 0, 0]), dtype=float),
            velocity=np.array(body_config.get('velocity', [0, 0, 0]), dtype=float),
            orientation=np.array(body_config.get('orientation', [1, 0, 0, 0]), dtype=float),
            angular_vel=np.array(body_config.get('angular_velocity', [0, 0, 0]), dtype=float),
            mass=body_config.get('mass', 1.0),
            inertia=np.eye(3) * body_config.get('inertia_scale', 1.0),
            shape_type=shape_type,
            shape_params=np.array(body_config.get('shape_params', [1, 0, 0]), dtype=float)
        )
        
        engine.add_body(body)
    
    return engine


def calculate_total_energy(engine):
    """
    Calculate the total energy of the system.
    
    Total Energy = Kinetic Energy + Potential Energy
    Kinetic Energy = Linear KE + Rotational KE
    Linear KE = 0.5 * m * v^2
    Rotational KE = 0.5 * w^T * I * w
    Potential Energy = m * g * h
    """
    total_energy = 0.0
    
    for body in engine.bodies:
        # Linear kinetic energy
        linear_ke = 0.5 * body.mass * np.dot(body.velocity, body.velocity)
        
        # Rotational kinetic energy
        # KE_rot = 0.5 * w^T * I * w
        I_world = body._quaternion_to_matrix(body.orientation) @ body.inertia @ body._quaternion_to_matrix(body.orientation).T
        rotational_ke = 0.5 * np.dot(body.angular_vel, I_world @ body.angular_vel)
        
        # Potential energy (assuming gravity in -Y direction)
        # PE = m * g * h, where h is the Y position
        g_magnitude = np.linalg.norm(engine.gravity)
        potential_energy = body.mass * g_magnitude * body.position[1]
        
        # Sum up
        body_energy = linear_ke + rotational_ke + potential_energy
        total_energy += body_energy
    
    return total_energy


def plot_energy_history(time_steps, energy_history, output_path="energy_drift.png"):
    """Create and save the energy drift plot."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation - matplotlib not available")
        initial_energy = energy_history[0]
        final_energy = energy_history[-1]
        drift_percent = abs((final_energy - initial_energy) / initial_energy) * 100
        return drift_percent
    
    plt.figure(figsize=(10, 6))
    
    # Plot energy over time
    plt.plot(time_steps, energy_history, 'b-', linewidth=2, label='Total Energy')
    
    # Calculate and display drift
    initial_energy = energy_history[0]
    final_energy = energy_history[-1]
    drift_percent = abs((final_energy - initial_energy) / initial_energy) * 100
    
    # Add reference line at initial energy
    plt.axhline(y=initial_energy, color='r', linestyle='--', alpha=0.5, 
                label=f'Initial Energy: {initial_energy:.2f} J')
    
    # Styling
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Total Energy (J)', fontsize=12)
    plt.title(f'System Energy Conservation Over Time\nDrift: {drift_percent:.2f}%', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text
    stats_text = f'Initial: {initial_energy:.2f} J\nFinal: {final_energy:.2f} J\nDrift: {drift_percent:.2f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Energy plot saved to: {output_path}")
    
    return drift_percent


def main():
    parser = argparse.ArgumentParser(description="Plot energy conservation of physics simulation")
    parser.add_argument('--scene', type=str, default='tests/scenes/stack.json',
                        help='Path to scene configuration file')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of simulation steps to run')
    parser.add_argument('--output', type=str, default='energy_drift.png',
                        help='Output file path for the plot')
    
    args = parser.parse_args()
    
    # Check if scene file exists
    scene_path = Path(args.scene)
    if not scene_path.exists():
        # Try some default locations
        alt_paths = [
            Path('scenes/stack.json'),
            Path('../scenes/stack.json'),
            Path('tests/scenes/stack.json')
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                scene_path = alt_path
                print(f"Using scene file: {scene_path}")
                break
        else:
            # Create a default scene if none exists
            print(f"Scene file not found. Creating default stack scene...")
            default_scene = {
                "dt": 0.01,
                "gravity": [0, -9.81, 0],
                "bodies": [
                    {
                        "shape_type": "box",
                        "position": [0, 0, 0],
                        "velocity": [0, 0, 0],
                        "orientation": [1, 0, 0, 0],
                        "angular_velocity": [0, 0, 0],
                        "mass": 1000.0,
                        "inertia_scale": 1000.0,
                        "shape_params": [10, 0.5, 10]
                    },
                    {
                        "shape_type": "box",
                        "position": [0, 2, 0],
                        "velocity": [0, 0, 0],
                        "orientation": [1, 0, 0, 0],
                        "angular_velocity": [0, 0, 0],
                        "mass": 1.0,
                        "inertia_scale": 1.0,
                        "shape_params": [1, 1, 1]
                    },
                    {
                        "shape_type": "box",
                        "position": [0, 4.5, 0],
                        "velocity": [0, 0, 0],
                        "orientation": [1, 0, 0, 0],
                        "angular_velocity": [0, 0, 0],
                        "mass": 1.0,
                        "inertia_scale": 1.0,
                        "shape_params": [1, 1, 1]
                    }
                ]
            }
            
            # Save default scene
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scene_path, 'w') as f:
                json.dump(default_scene, f, indent=2)
            
            scene_config = default_scene
    else:
        scene_config = load_scene(scene_path)
    
    # Setup engine
    print(f"Loading scene with {len(scene_config.get('bodies', []))} bodies...")
    engine = setup_engine_from_scene(scene_config)
    
    # Run simulation and track energy
    print(f"Running simulation for {args.steps} steps...")
    time_steps = []
    energy_history = []
    
    for step in range(args.steps):
        # Calculate and record energy
        total_energy = calculate_total_energy(engine)
        time_steps.append(step * engine.dt)
        energy_history.append(total_energy)
        
        # Step simulation
        engine.step()
        
        # Progress indicator
        if step % 100 == 0:
            print(f"  Step {step}/{args.steps} - Energy: {total_energy:.2f} J")
    
    # Plot results
    print("\nGenerating energy plot...")
    drift_percent = plot_energy_history(time_steps, energy_history, args.output)
    
    # Summary
    print(f"\nSimulation complete!")
    print(f"  Initial energy: {energy_history[0]:.2f} J")
    print(f"  Final energy: {energy_history[-1]:.2f} J")
    print(f"  Energy drift: {drift_percent:.2f}%")
    
    if drift_percent > 5.0:
        print("  ⚠️  Warning: High energy drift detected! Check simulation stability.")
    else:
        print("  ✅ Energy conservation within acceptable limits.")


if __name__ == "__main__":
    main()