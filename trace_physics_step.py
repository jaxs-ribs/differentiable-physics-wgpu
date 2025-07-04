import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.xpbd.solver import solve_constraints
from physics.xpbd.velocity_solver import solve_velocities
from physics.xpbd.velocity_update import reconcile_velocities
from physics.xpbd.integration import predict_state

# Simple test - sphere on ground
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere with small penetration
builder.add_body(
    position=[0, 0.54, 0],
    velocity=[0, 0, 0],
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

print("=== Initial State ===")
print(f"Sphere position: {x.numpy()[1]}")
print(f"Sphere velocity: {v.numpy()[1]}")

# Step 1: Prediction
x_old, q_old = x, q
x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)
print(f"\n=== After Prediction ===")
print(f"Predicted position: {x_pred.numpy()[1]}")
print(f"Predicted velocity: {v_new.numpy()[1]}")

# Step 2: Broadphase
candidate_pairs = uniform_spatial_hash(x_pred, shape_type, shape_params)
print(f"\n=== Broadphase ===")
print(f"Candidate pairs: {candidate_pairs.numpy()}")

# Step 3: Narrowphase
contacts = generate_contacts(x_pred, q_pred, candidate_pairs, shape_type, shape_params, friction, 0.0001)
print(f"\n=== Narrowphase ===")
valid_contacts = (contacts['ids_a'].numpy() != -1).sum()
print(f"Valid contacts: {valid_contacts}")
if valid_contacts > 0:
    idx = np.where(contacts['ids_a'].numpy() != -1)[0][0]
    print(f"Contact {idx}:")
    print(f"  Bodies: {contacts['ids_a'].numpy()[idx]} <-> {contacts['ids_b'].numpy()[idx]}")
    print(f"  Penetration: {contacts['p'].numpy()[idx]:.4f}")
    print(f"  Normal: {contacts['normal'].numpy()[idx]}")
    print(f"  Compliance: {contacts['compliance'].numpy()[idx]}")

# Step 4: Position solve
x_proj, q_proj, lambda_acc = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=16)
print(f"\n=== Position Solve ===")
print(f"Corrected position: {x_proj.numpy()[1]}")
print(f"Lambda (all): {lambda_acc.numpy()}")
if valid_contacts > 0:
    print(f"Lambda for contact {idx}: {lambda_acc.numpy()[idx]:.6f}")

# Step 5: Velocity reconciliation
v_reconciled, omega_reconciled = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_new, omega_new, dt)
print(f"\n=== Velocity Reconciliation ===")
print(f"Reconciled velocity: {v_reconciled.numpy()[1]}")

# Step 6: Velocity solve
v_final, omega_final = solve_velocities(v_reconciled, omega_reconciled, contacts, inv_mass, inv_inertia, dt, lambda_acc, 0.0)
print(f"\n=== Velocity Solve ===")
print(f"Final velocity: {v_final.numpy()[1]}")

print(f"\n=== Summary ===")
print(f"Initial Y: {x.numpy()[1][1]:.4f}")
print(f"Final Y: {x_proj.numpy()[1][1]:.4f}")
print(f"Y change: {x_proj.numpy()[1][1] - x.numpy()[1][1]:.4f}")
print(f"Final Y velocity: {v_final.numpy()[1][1]:.4f}")