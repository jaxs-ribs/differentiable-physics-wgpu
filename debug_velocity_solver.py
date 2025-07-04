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
from physics.xpbd.velocity_solver import solve_velocities
from physics.xpbd.velocity_update import reconcile_velocities
from physics.xpbd.integration import predict_state

# Simple test case
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere sliding on ground
builder.add_body(
    position=[0, -0.44, 0],  # Slightly overlapping
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
print(f"  Sphere position: {x.numpy()[1]}")
print(f"  Sphere velocity: {v.numpy()[1]}")

# Perform one physics step manually to debug
x_old, q_old = x, q
x_pred, q_pred, v_new, omega_new = predict_state(x, q, v, omega, inv_mass, inv_inertia, gravity, dt)

print("\nAfter prediction:")
print(f"  Predicted position: {x_pred.numpy()[1]}")
print(f"  Predicted velocity: {v_new.numpy()[1]}")

candidate_pairs = uniform_spatial_hash(x_pred, shape_type, shape_params)
contacts = generate_contacts(x_pred, q_pred, candidate_pairs, shape_type, shape_params, friction, 0.001)

print("\nContacts:")
if 'ids_a' in contacts and contacts['ids_a'].shape[0] > 0:
    valid_contacts = contacts['ids_a'].numpy() != -1
    num_valid = valid_contacts.sum()
    print(f"  Valid contacts: {num_valid}")
    if num_valid > 0:
        # Find first valid contact
        idx = np.where(valid_contacts)[0][0]
        print(f"  Contact {idx}:")
        print(f"    Bodies: {contacts['ids_a'].numpy()[idx]} <-> {contacts['ids_b'].numpy()[idx]}")
        print(f"    Normal: {contacts['normal'].numpy()[idx]}")
        print(f"    Penetration: {contacts['p'].numpy()[idx]}")
        print(f"    Friction: {contacts['friction'].numpy()[idx]}")

x_proj, q_proj, lambda_acc = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=8)

print("\nAfter position solve:")
print(f"  Position: {x_proj.numpy()[1]}")
print(f"  Lambda (impulse): {lambda_acc.numpy() if lambda_acc.shape[0] > 0 else 'No contacts'}")

v_reconciled, omega_reconciled = reconcile_velocities(x_proj, q_proj, x_old, q_old, v_new, omega_new, dt)

print("\nAfter velocity reconciliation:")
print(f"  Velocity: {v_reconciled.numpy()[1]}")

# Debug velocity solver in detail
if 'ids_a' in contacts and contacts['ids_a'].shape[0] > 0:
    # Extract contact data
    ids_a = contacts['ids_a']
    ids_b = contacts['ids_b']
    normals = contacts['normal']
    friction_values = contacts.get('friction', Tensor.zeros((ids_a.shape[0],)))
    
    valid_mask = (ids_a != -1) & (ids_b != -1)
    
    # Get velocities at contact points
    v_a = v_reconciled.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    v_b = v_reconciled.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    v_rel = v_a - v_b
    
    # Normal velocity
    v_n = (v_rel * normals).sum(axis=-1)
    
    # Tangential velocity
    v_t = v_rel - v_n.unsqueeze(-1) * normals
    v_t_mag = (v_t * v_t).sum(axis=-1).sqrt() + 1e-8
    
    print("\nVelocity solver debug:")
    for i in range(min(4, ids_a.shape[0])):
        if valid_mask.numpy()[i]:
            print(f"  Contact {i}:")
            print(f"    Relative velocity: {v_rel.numpy()[i]}")
            print(f"    Normal velocity: {v_n.numpy()[i]:.3f}")
            print(f"    Tangential velocity magnitude: {v_t_mag.numpy()[i]:.3f}")
            print(f"    Lambda (normal impulse): {lambda_acc.numpy()[i]:.3f}")
            print(f"    Max friction force: {(friction_values.numpy()[i] * abs(lambda_acc.numpy()[i])):.3f}")

v_final, omega_final = solve_velocities(v_reconciled, omega_reconciled, contacts, inv_mass, inv_inertia, dt, lambda_acc, 0.0)

print("\nAfter velocity solve:")
print(f"  Final velocity: {v_final.numpy()[1]}")