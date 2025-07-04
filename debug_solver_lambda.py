import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.engine import PhysicsEngine

# Simple horizontal ground test
builder = SceneBuilder()

# Ground plane  
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere resting on ground with slight penetration
builder.add_body(
    position=[0, 0.52, 0],  # Penetration of 0.03
    velocity=[5, 0, 0],
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0,
    friction=0.5
)

scene_data = builder.build()

# Manually step through one iteration
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.xpbd.solver import solve_constraints
from physics.xpbd.integration import predict_state

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
print(f"  Sphere: pos={x.numpy()[1]}, vel={v.numpy()[1]}")

# Predict
x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
print(f"\nPredicted: pos={x_pred.numpy()[1]}, vel={v_new.numpy()[1]}")

# Get contacts
candidate_pairs = uniform_spatial_hash(x_pred, shape_type, shape_params)
contacts = generate_contacts(x_pred, q_pred, candidate_pairs, shape_type, shape_params, friction, 0.001)

print(f"\nContacts found: {contacts['ids_a'].shape[0]}")
valid_contacts = (contacts['ids_a'].numpy() != -1).sum()
print(f"Valid contacts: {valid_contacts}")

if valid_contacts > 0:
    idx = np.where(contacts['ids_a'].numpy() != -1)[0][0]
    print(f"  Contact {idx}: penetration={contacts['p'].numpy()[idx]:.4f}")

# Solve constraints
x_proj, q_proj, lambda_acc = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=8)

print(f"\nAfter constraint solve:")
print(f"  Position: {x_proj.numpy()[1]}")
print(f"  Lambda_acc: {lambda_acc.numpy()}")

# Calculate what lambda should be
# For a static contact, lambda = penetration / (gen_inv_mass * dt^2)
# With compliance alpha, lambda = penetration / (gen_inv_mass * dt^2 + alpha)
if valid_contacts > 0:
    pen = contacts['p'].numpy()[idx]
    compliance = 0.001
    gen_inv_mass = 1.0 + 0.0  # sphere inv_mass + ground inv_mass
    expected_lambda = pen / (gen_inv_mass * dt * dt + compliance)
    print(f"\nExpected lambda (approx): {expected_lambda:.4f}")
    print(f"Actual lambda: {lambda_acc.numpy()[idx]:.4f}")