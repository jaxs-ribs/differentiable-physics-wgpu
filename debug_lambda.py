import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.engine import _physics_step_static
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.xpbd.solver import solve_constraints
from physics.xpbd.integration import predict_state

# Test with gravity to ensure normal force
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere on ground with initial penetration
builder.add_body(
    position=[0, -0.4, 0],  # Significant overlap
    velocity=[5, 0, 0],
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0,
    friction=0.5
)

scene_data = builder.build()

# Create tensors
x = Tensor(scene_data['x'].astype(np.float32))
q = Tensor(scene_data['q'].astype(np.float32))
v = Tensor(scene_data['v'].astype(np.float32))
omega = Tensor(scene_data['omega'].astype(np.float32))
inv_mass = Tensor(scene_data['inv_mass'].astype(np.float32))
inv_inertia = Tensor(scene_data['inv_inertia'].astype(np.float32))
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))
friction = Tensor(scene_data['friction'].astype(np.float32))
gravity = Tensor(np.array([0, -9.81, 0], dtype=np.float32))
dt = 0.016

print("Initial state:")
print(f"  Sphere at: {x.numpy()[1]}")
print(f"  Velocity: {v.numpy()[1]}")

# One physics step
x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)

print("\nAfter prediction:")
print(f"  Predicted position: {x_pred.numpy()[1]}")

candidate_pairs = uniform_spatial_hash(x_pred, shape_type, shape_params)
contacts = generate_contacts(x_pred, q_pred, candidate_pairs, shape_type, shape_params, friction, 0.001)

if 'ids_a' in contacts and contacts['ids_a'].shape[0] > 0:
    print("\nContacts detected:")
    for i in range(contacts['ids_a'].shape[0]):
        if contacts['ids_a'].numpy()[i] != -1:
            print(f"  Contact {i}: penetration = {contacts['p'].numpy()[i]:.4f}")

x_proj, q_proj, lambda_acc = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=8)

print("\nAfter position solve:")
print(f"  Corrected position: {x_proj.numpy()[1]}")
print(f"  Lambda values: {lambda_acc.numpy()}")

# Calculate expected normal impulse
# For a 1kg mass under gravity, normal force = mg = 1 * 9.81 = 9.81 N
# Impulse over dt = F * dt = 9.81 * 0.016 = 0.157
print(f"\nExpected normal impulse (mg*dt): {1.0 * 9.81 * dt:.3f}")

# The friction force should be μ * N = 0.5 * lambda
if lambda_acc.shape[0] > 0:
    for i in range(lambda_acc.shape[0]):
        if lambda_acc.numpy()[i] > 0:
            print(f"Contact {i}: lambda = {lambda_acc.numpy()[i]:.4f}, max friction = {0.5 * lambda_acc.numpy()[i]:.4f}")