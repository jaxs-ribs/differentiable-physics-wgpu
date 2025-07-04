import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from physics.engine import PhysicsEngine, _physics_step_static
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from tinygrad import Tensor

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

# Sphere on ground
builder.add_body(
    position=[0, -0.45, 0],  # Should be in contact (radius 0.5, ground at -0.95)
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
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))
friction = Tensor(scene_data['friction'].astype(np.float32))

print("Initial positions:")
print(f"  Ground: {x.numpy()[0]}")
print(f"  Sphere: {x.numpy()[1]}")
print(f"  Distance between centers: {np.linalg.norm(x.numpy()[1] - x.numpy()[0]):.3f}")
print(f"  Expected contact distance: {0.5 + 0.05:.3f} (sphere radius + half box thickness)")

# Check contacts
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)
contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction, 0.001)

print("\nBroadphase candidates:")
print(f"  Pairs: {candidate_pairs.numpy()}")

print("\nContacts:")
if 'ids_a' in contacts and contacts['ids_a'].shape[0] > 0:
    print(f"  Number of contacts: {contacts['ids_a'].shape[0]}")
    print(f"  ids_a: {contacts['ids_a'].numpy()}")
    print(f"  ids_b: {contacts['ids_b'].numpy()}")
    print(f"  normals: {contacts['normal'].numpy()}")
    print(f"  penetrations: {contacts['p'].numpy()}")
    print(f"  friction: {contacts['friction'].numpy()}")
else:
    print("  No contacts detected!")

# Run one step and check contacts again
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

print("\n=== After 1 step ===")
engine.step()
state = engine.get_state()
print(f"Sphere position: {state['x'][1]}")
print(f"Sphere velocity: {state['v'][1]}")