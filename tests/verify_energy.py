#!/usr/bin/env python3
"""Verify energy conservation with sphere collisions"""
import numpy as np
from reference import PhysicsEngine, Body, ShapeType

# Create scene with spheres only
engine = PhysicsEngine(dt=0.01, gravity=np.array([0, -9.81, 0]))

# Ground sphere
ground = Body(
    position=np.array([0., 0., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1e6,  # Very heavy
    inertia=np.eye(3) * 1e6,
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([5., 0., 0.])  # Large ground
)

# Falling sphere
ball = Body(
    position=np.array([0., 7., 0.]),
    velocity=np.array([0., 0., 0.]),
    orientation=np.array([1., 0., 0., 0.]),
    angular_vel=np.zeros(3),
    mass=1.0,
    inertia=np.eye(3) * 0.4,  # Sphere inertia
    shape_type=ShapeType.SPHERE,
    shape_params=np.array([1., 0., 0.])
)

engine.add_body(ground)
engine.add_body(ball)

# Calculate total energy
def calc_energy():
    total = 0.0
    for body in engine.bodies:
        # KE
        ke = 0.5 * body.mass * np.dot(body.velocity, body.velocity)
        # PE (only count if mass is reasonable)
        if body.mass < 1e5:
            pe = body.mass * 9.81 * body.position[1]
            total += ke + pe
    return total

initial_energy = calc_energy()
print(f"Initial energy: {initial_energy:.2f} J")
print(f"Ball starts at y={ball.position[1]}, ground top at y={ground.position[1] + ground.shape_params[0]}")

# Run simulation
energies = [initial_energy]
for i in range(500):
    engine.step()
    energies.append(calc_energy())
    
    if i % 100 == 0:
        print(f"Step {i:3d}: Ball y={ball.position[1]:.3f}, Energy={energies[-1]:.2f} J")

# Check energy conservation
final_energy = energies[-1]
drift = abs(final_energy - initial_energy) / initial_energy * 100

print(f"\nFinal energy: {final_energy:.2f} J")
print(f"Energy drift: {drift:.2f}%")

if drift < 5:
    print("✅ Good energy conservation!")
elif drift < 10:
    print("⚠️  Moderate energy drift")
else:
    print("❌ High energy drift - check collision resolution")