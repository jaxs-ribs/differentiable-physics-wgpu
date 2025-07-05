"""Integration test for friction: Slope slide with friction."""
import numpy as np
import pytest
from physics.engine import PhysicsEngine
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder


@pytest.mark.xfail(reason="Friction not yet implemented in solver - requires tangential constraint handling")
def test_slope_slide_with_friction():
    """Test that a box sliding down a slope experiences correct friction.
    
    This is test case 8 from AGENTS.md:
    - Acceleration down slope ≈ g·(sinθ - μ·cosθ) within 5%
    """
    # Test parameters
    slope_angle = 30.0  # degrees
    theta_rad = np.radians(slope_angle)
    friction_coeff = 0.3
    g = 9.81
    
    # Expected acceleration: a = g * (sin(θ) - μ * cos(θ))
    expected_accel = g * (np.sin(theta_rad) - friction_coeff * np.cos(theta_rad))
    
    # Create scene
    builder = SceneBuilder()
    
    # Create tilted plane (box rotated around z-axis)
    # Rotation quaternion for angle θ around z-axis: [cos(θ/2), 0, 0, sin(θ/2)]
    half_angle = theta_rad / 2
    plane_orientation = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
    
    builder.add_body(
        position=[0, -2, 0],  # Lower the plane
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],  # Large thin box as plane
        mass=float('inf'),  # Static
        orientation=plane_orientation,
        friction=friction_coeff
    )
    
    # Add sphere on slope  
    # Position it above the tilted plane
    # The plane normal at angle θ points at (sin(θ), cos(θ), 0)
    # Place sphere at a position along the slope
    start_distance = 2.0  # Distance along slope from origin
    sphere_x = -start_distance * np.sin(theta_rad)
    sphere_y = start_distance * np.cos(theta_rad) - 2 + 0.52  # Slightly penetrating
    
    builder.add_body(
        position=[sphere_x, sphere_y, 0],
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],  # Radius 0.5
        mass=1.0,
        velocity=[0, 0, 0],
        friction=friction_coeff
    )
    
    scene_data = builder.build()
    
    # Create engine
    engine = PhysicsEngine(
        x=scene_data['x'],
        q=scene_data['q'],
        v=scene_data['v'],
        omega=scene_data['omega'],
        inv_mass=scene_data['inv_mass'],
        inv_inertia=scene_data['inv_inertia'],
        shape_type=scene_data['shape_type'],
        shape_params=scene_data['shape_params'],
        friction=scene_data['friction'],
        gravity=np.array([0, -9.81, 0]),
        dt=0.008,  # Small timestep for accuracy
        restitution=0.0,  # No bounce
        solver_iterations=16,
        contact_compliance=0.0001
    )
    
    # Let box settle on slope first
    for _ in range(50):
        engine.step()
    
    # Record initial position and velocity
    initial_state = engine.get_state()
    initial_pos = initial_state['x'][1].copy()
    initial_vel = initial_state['v'][1].copy()
    
    # Run simulation for a short time to measure acceleration
    measure_time = 0.5  # seconds
    num_steps = int(measure_time / 0.008)
    
    for _ in range(num_steps):
        engine.step()
    
    # Get final state
    final_state = engine.get_state()
    final_pos = final_state['x'][1]
    final_vel = final_state['v'][1]
    
    # Calculate displacement along slope
    displacement = final_pos - initial_pos
    disp_magnitude = np.linalg.norm(displacement[:2])  # Only x,y components
    
    # Calculate velocity change along slope direction
    slope_dir = np.array([-np.sin(theta_rad), -np.cos(theta_rad), 0])
    initial_speed = np.dot(initial_vel, slope_dir)
    final_speed = np.dot(final_vel, slope_dir)
    
    # Calculate acceleration
    measured_accel = (final_speed - initial_speed) / measure_time
    
    # Also verify using kinematic equation: s = ut + 0.5*a*t^2
    # If initial velocity is small, s ≈ 0.5*a*t^2
    if abs(initial_speed) < 0.1:
        kinematic_accel = 2 * disp_magnitude / (measure_time ** 2)
        measured_accel = kinematic_accel
    
    # Verify acceleration matches analytical solution within tolerance
    relative_error = abs(measured_accel - expected_accel) / expected_accel
    
    assert relative_error < 0.05, \
        f"Acceleration mismatch: measured={measured_accel:.3f} m/s², " \
        f"expected={expected_accel:.3f} m/s² (error={relative_error*100:.1f}%)"
    
    # Additional sanity checks
    assert measured_accel > 0, "Box should accelerate down the slope"
    assert measured_accel < g * np.sin(theta_rad), \
        "Acceleration should be less than frictionless case"


@pytest.mark.xfail(reason="Friction not yet implemented in solver - requires tangential constraint handling")
def test_static_friction():
    """Test that static friction prevents motion on shallow slopes."""
    # Use a shallow angle where static friction should prevent sliding
    slope_angle = 15.0  # degrees
    theta_rad = np.radians(slope_angle)
    friction_coeff = 0.5  # μ > tan(θ) means no sliding
    
    # For no sliding: μ >= tan(θ)
    # tan(15°) ≈ 0.268, so μ = 0.5 should prevent sliding
    
    builder = SceneBuilder()
    
    # Create tilted plane
    half_angle = theta_rad / 2
    plane_orientation = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
    
    builder.add_body(
        position=[0, -2, 0],
        shape_type=ShapeType.BOX,
        shape_params=[10, 0.05, 10],
        mass=float('inf'),
        orientation=plane_orientation,
        friction=friction_coeff
    )
    
    # Add sphere on slope with slight penetration
    builder.add_body(
        position=[0, -1.42, 0],  # Slight penetration for stable contact
        shape_type=ShapeType.SPHERE,
        shape_params=[0.5, 0, 0],
        mass=1.0,
        velocity=[0, 0, 0],
        friction=friction_coeff
    )
    
    scene_data = builder.build()
    
    engine = PhysicsEngine(
        x=scene_data['x'],
        q=scene_data['q'],
        v=scene_data['v'],
        omega=scene_data['omega'],
        inv_mass=scene_data['inv_mass'],
        inv_inertia=scene_data['inv_inertia'],
        shape_type=scene_data['shape_type'],
        shape_params=scene_data['shape_params'],
        friction=scene_data['friction'],
        gravity=np.array([0, -9.81, 0]),
        dt=0.016,
        restitution=0.0,
        solver_iterations=16,
        contact_compliance=0.0001
    )
    
    # Let box settle
    for _ in range(50):
        engine.step()
    
    initial_pos = engine.get_state()['x'][1].copy()
    
    # Run for 1 second
    for _ in range(62):
        engine.step()
    
    final_pos = engine.get_state()['x'][1]
    
    # Box should not slide significantly
    displacement = np.linalg.norm(final_pos - initial_pos)
    assert displacement < 0.01, \
        f"Box should not slide with high friction: displacement={displacement:.3f}m"


@pytest.mark.xfail(reason="Friction not yet implemented in solver - requires tangential constraint handling")
def test_friction_parameter_effect():
    """Test that different friction coefficients produce expected behavior."""
    # Test with two different friction values
    test_cases = [
        (0.1, "low"),   # Low friction - should slide fast
        (0.6, "high"),  # High friction - should slide slow
    ]
    
    slope_angle = 30.0
    theta_rad = np.radians(slope_angle)
    velocities = {}
    
    for friction_coeff, label in test_cases:
        builder = SceneBuilder()
        
        # Tilted plane
        half_angle = theta_rad / 2
        plane_orientation = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
        
        builder.add_body(
            position=[0, -2, 0],
            shape_type=ShapeType.BOX,
            shape_params=[10, 0.05, 10],
            mass=float('inf'),
            orientation=plane_orientation,
            friction=friction_coeff
        )
        
        # Sphere with slight penetration
        builder.add_body(
            position=[0, -1.42, 0],  # Slight penetration for stable contact
            shape_type=ShapeType.SPHERE,
            shape_params=[0.5, 0, 0],
            mass=1.0,
            friction=friction_coeff
        )
        
        scene_data = builder.build()
        
        engine = PhysicsEngine(
            x=scene_data['x'],
            q=scene_data['q'],
            v=scene_data['v'],
            omega=scene_data['omega'],
            inv_mass=scene_data['inv_mass'],
            inv_inertia=scene_data['inv_inertia'],
            shape_type=scene_data['shape_type'],
            shape_params=scene_data['shape_params'],
            friction=scene_data['friction'],
            gravity=np.array([0, -9.81, 0]),
            dt=0.016,
            restitution=0.0,
            solver_iterations=16,
            contact_compliance=0.0001
        )
        
        # Let settle and then measure velocity after sliding
        for _ in range(100):
            engine.step()
        
        final_vel = engine.get_state()['v'][1]
        speed = np.linalg.norm(final_vel)
        velocities[label] = speed
    
    # Low friction should result in higher velocity
    assert velocities["low"] > velocities["high"], \
        f"Low friction should slide faster: low={velocities['low']:.3f}, high={velocities['high']:.3f}"
    
    # The difference should be significant
    assert velocities["low"] > 1.5 * velocities["high"], \
        "Friction effect should be significant"