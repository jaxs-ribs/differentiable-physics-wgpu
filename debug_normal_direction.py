import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts

# Simple test - sphere on ground
builder = SceneBuilder()

# Ground plane (body 0)
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere (body 1)
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
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))
friction = Tensor(scene_data['friction'].astype(np.float32))

print("Bodies:")
print(f"  Body 0: {shape_type.numpy()[0]} (BOX={ShapeType.BOX}) at {x.numpy()[0]}")
print(f"  Body 1: {shape_type.numpy()[1]} (SPHERE={ShapeType.SPHERE}) at {x.numpy()[1]}")

# Get broadphase pairs
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)
print(f"\nCandidate pairs: {candidate_pairs.numpy()}")

# Generate contacts
contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction, 0.001)

print(f"\nContacts:")
for i in range(contacts['ids_a'].shape[0]):
    id_a = contacts['ids_a'].numpy()[i]
    id_b = contacts['ids_b'].numpy()[i]
    if id_a != -1:
        print(f"  Contact {i}:")
        print(f"    Body A: {id_a}, Body B: {id_b}")
        print(f"    Normal: {contacts['normal'].numpy()[i]} (should point from A to B)")
        print(f"    Penetration: {contacts['p'].numpy()[i]}")
        
        # For sphere-plane, normal should point from plane to sphere (upward)
        # So if plane is A and sphere is B, normal should be [0, 1, 0]
        # If sphere is A and plane is B, normal should be [0, -1, 0]