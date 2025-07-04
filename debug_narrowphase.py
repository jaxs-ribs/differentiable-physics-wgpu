import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts

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
    position=[0, -0.44, 0],
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
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))
friction = Tensor(scene_data['friction'].astype(np.float32))

# Get broadphase pairs
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)

print("Candidate pairs from broadphase:")
print(candidate_pairs.numpy())

# Generate contacts
contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction, 0.001)

print("\nNarrowphase contact generation:")
print(f"Number of contact slots: {contacts['ids_a'].shape[0]}")

# Check each contact slot
for i in range(contacts['ids_a'].shape[0]):
    id_a = contacts['ids_a'].numpy()[i]
    id_b = contacts['ids_b'].numpy()[i]
    if id_a != -1 or id_b != -1:
        print(f"\nContact {i}:")
        print(f"  Bodies: {id_a} <-> {id_b}")
        print(f"  Normal: {contacts['normal'].numpy()[i]}")
        print(f"  Penetration: {contacts['p'].numpy()[i]}")
        print(f"  Friction: {contacts['friction'].numpy()[i]}")

# Let's manually check the sphere-plane test
print("\n=== Manual sphere-plane check ===")
sphere_pos = x.numpy()[1]
plane_pos = x.numpy()[0]
sphere_radius = shape_params.numpy()[1][0]
plane_half_thickness = shape_params.numpy()[0][1]

print(f"Sphere center Y: {sphere_pos[1]}")
print(f"Sphere bottom Y: {sphere_pos[1] - sphere_radius}")
print(f"Plane center Y: {plane_pos[1]}")
print(f"Plane top Y: {plane_pos[1] + plane_half_thickness}")
print(f"Expected penetration: {(plane_pos[1] + plane_half_thickness) - (sphere_pos[1] - sphere_radius)}")